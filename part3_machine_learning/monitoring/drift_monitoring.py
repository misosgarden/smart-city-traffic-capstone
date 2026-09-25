"""Monitor feature-distribution drift for the traffic prediction system."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

PART3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PART3_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "part2_python"
    / "data"
    / "processed"
    / "traffic_features.csv"
)

REPORTS_DIR = PART3_DIR / "reports"
RESULTS_PATH = REPORTS_DIR / "drift_monitoring_results.csv"
SUMMARY_PATH = REPORTS_DIR / "drift_monitoring_summary.txt"
LOG_PATH = PART3_DIR / "part3.log"

MONITORED_FEATURES = [
    "traffic_volume",
    "temp",
    "rain_1h",
    "clouds_all",
    "hour",
]

PSI_ALERT_THRESHOLD = 0.20
SAMPLE_SIZE = 5000
RANDOM_STATE = 42

REQUIRED_COLUMNS = set(MONITORED_FEATURES) | {"date_time"}


class DriftMonitoringError(RuntimeError):
    """Raised when drift monitoring cannot be completed safely."""


def configure_logging() -> None:
    """Configure console and append-only file logging."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_PATH,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_monitoring_data(path: Path) -> pd.DataFrame:
    """Load and validate the feature-engineered traffic dataset."""
    try:
        data = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DriftMonitoringError(
            f"Monitoring dataset was not found: {path}"
        ) from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DriftMonitoringError(
            f"Unable to read monitoring dataset {path}: {exc}"
        ) from exc

    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise DriftMonitoringError(
            f"Monitoring dataset is missing required columns: "
            f"{missing_columns}"
        )

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce",
    )

    invalid_dates = int(data["date_time"].isna().sum())
    if invalid_dates:
        raise DriftMonitoringError(
            f"Monitoring dataset contains {invalid_dates} invalid dates"
        )

    for feature in MONITORED_FEATURES:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

    missing_values = data[MONITORED_FEATURES].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise DriftMonitoringError(
            "Monitoring features contain missing or non-numeric values: "
            f"{missing_values.to_dict()}"
        )

    data = data.sort_values("date_time").reset_index(drop=True)

    LOGGER.info(
        "Loaded %d observations from %s",
        len(data),
        path,
    )

    return data


def create_reference_data(data: pd.DataFrame) -> pd.DataFrame:
    """Use the first 80 percent of observations as historical reference data."""
    split_index = int(len(data) * 0.80)

    if split_index < SAMPLE_SIZE:
        raise DriftMonitoringError(
            "The dataset is too small for the requested monitoring sample"
        )

    reference_data = data.iloc[:split_index].copy()

    LOGGER.info(
        "Created historical reference dataset with %d observations",
        len(reference_data),
    )

    return reference_data


def create_normal_sample(
    reference_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a reproducible sample representing normal operating conditions."""
    normal_sample = reference_data.sample(
        n=min(SAMPLE_SIZE, len(reference_data)),
        replace=False,
        random_state=RANDOM_STATE,
    ).copy()

    LOGGER.info(
        "Created normal monitoring scenario with %d observations",
        len(normal_sample),
    )

    return normal_sample


def create_shifted_sample(
    normal_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Create transparent synthetic shifts for alert testing."""
    shifted_sample = normal_sample.copy()

    shifted_sample["traffic_volume"] = (
        shifted_sample["traffic_volume"] * 1.35 + 500
    ).clip(lower=0)

    shifted_sample["temp"] = shifted_sample["temp"] + 12

    shifted_sample["clouds_all"] = (
        shifted_sample["clouds_all"] + 30
    ).clip(lower=0, upper=100)

    shifted_sample["rain_1h"] = (
        shifted_sample["rain_1h"] + 1
    ).clip(lower=0)

    LOGGER.info(
        "Created deliberately shifted monitoring scenario for alert testing"
    )

    return shifted_sample


def create_bin_edges(
    reference_values: np.ndarray,
    bins: int = 10,
) -> np.ndarray:
    """Create stable bin boundaries from the reference distribution."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(
        np.quantile(reference_values, quantiles)
    )

    if len(edges) < 3:
        minimum = float(np.min(reference_values))
        maximum = float(np.max(reference_values))

        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5

        edges = np.linspace(minimum, maximum, bins + 1)

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def calculate_psi(
    reference_values: pd.Series,
    current_values: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate the Population Stability Index for one feature."""
    reference_array = reference_values.to_numpy(dtype=float)
    current_array = current_values.to_numpy(dtype=float)

    edges = create_bin_edges(reference_array, bins=bins)

    reference_counts, _ = np.histogram(
        reference_array,
        bins=edges,
    )
    current_counts, _ = np.histogram(
        current_array,
        bins=edges,
    )

    reference_proportions = (
        reference_counts / reference_counts.sum()
    )
    current_proportions = (
        current_counts / current_counts.sum()
    )

    epsilon = 0.0001

    reference_proportions = np.where(
        reference_proportions == 0,
        epsilon,
        reference_proportions,
    )

    current_proportions = np.where(
        current_proportions == 0,
        epsilon,
        current_proportions,
    )

    psi_values = (
        current_proportions - reference_proportions
    ) * np.log(
        current_proportions / reference_proportions
    )

    return float(np.sum(psi_values))


def interpret_psi(psi_value: float) -> str:
    """Translate a PSI value into an interpretable drift level."""
    if psi_value < 0.10:
        return "No material drift"

    if psi_value < PSI_ALERT_THRESHOLD:
        return "Moderate change"

    return "Material drift"


def evaluate_scenario(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Evaluate all monitored features for one scenario."""
    records = []

    for feature in MONITORED_FEATURES:
        display_feature = feature
        reference_values = reference_data[feature]
        current_values = current_data[feature]

        if feature == "rain_1h":
            display_feature = "rain_occurrence"
            reference_values = (
                reference_data["rain_1h"] > 0
            ).astype(int)
            current_values = (
                current_data["rain_1h"] > 0
            ).astype(int)

        psi_value = calculate_psi(
            reference_values,
            current_values,
        )

        feature_status = (
            "ALERT"
            if psi_value >= PSI_ALERT_THRESHOLD
            else "PASS"
        )

        records.append(
            {
                "Scenario": scenario_name,
                "Feature": display_feature,
                "Reference_Mean": round(
                    float(reference_values.mean()),
                    4,
                ),
                "Current_Mean": round(
                    float(current_values.mean()),
                    4,
                ),
                "PSI": round(psi_value, 4),
                "Drift_Level": interpret_psi(psi_value),
                "Feature_Status": feature_status,
                "Threshold": PSI_ALERT_THRESHOLD,
            }
        )

    results = pd.DataFrame(records)

    alert_detected = results["Feature_Status"].eq("ALERT").any()

    overall_status = (
        "ALERT / Requires investigation"
        if alert_detected
        else "PASS / Normal"
    )

    results["Overall_Status"] = overall_status

    log_method = (
        LOGGER.warning
        if alert_detected
        else LOGGER.info
    )

    log_method(
        "%s completed with status: %s",
        scenario_name,
        overall_status,
    )

    return results


def save_results(results: pd.DataFrame) -> None:
    """Save detailed results and an interpretable text summary."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        results.to_csv(
            RESULTS_PATH,
            index=False,
        )

        summary_lines = [
            "Smart City Traffic Model Drift Monitoring Summary",
            "",
            "Method: Population Stability Index (PSI)",
            (
                "Rainfall method: PSI applied to rainfall occurrence, "
                "defined as rain_1h greater than zero"
            ),
            f"Alert threshold: PSI >= {PSI_ALERT_THRESHOLD:.2f}",
            "",
            "Interpretation:",
            "PSI < 0.10: No material drift",
            "PSI from 0.10 to below 0.20: Moderate change",
            "PSI >= 0.20: Material drift",
            "",
            (
                "The historical reference dataset contains the first "
                "80% of observations in chronological order."
            ),
            (
                "Each simulated monitoring scenario contains up to "
                f"{SAMPLE_SIZE:,} observations."
            ),
            (
                "The normal scenario is sampled from the historical "
                "reference distribution."
            ),
            (
                "The shifted scenario contains deliberately modified "
                "traffic volume, temperature, rainfall and cloud "
                "coverage values to test whether the monitoring "
                "process can generate an alert."
            ),
            (
                "Rainfall occurrence is monitored instead of raw "
                "rainfall depth because rain_1h is highly "
                "zero-inflated. Quantile-based PSI on the raw values "
                "may not reliably detect a change from dry to rainy "
                "conditions."
            ),
            (
                "The overall status is ALERT / Requires investigation "
                "when at least one monitored feature has a PSI value "
                f"greater than or equal to {PSI_ALERT_THRESHOLD:.2f}."
            ),
            "",
        ]

        for scenario_name, scenario_results in results.groupby(
            "Scenario",
            sort=False,
        ):
            overall_status = scenario_results[
                "Overall_Status"
            ].iloc[0]

            summary_lines.append(
                f"{scenario_name}: {overall_status}"
            )

            for row in scenario_results.itertuples():
                summary_lines.append(
                    f"  {row.Feature}: PSI={row.PSI:.4f}, "
                    f"{row.Drift_Level}, {row.Feature_Status}"
                )

            summary_lines.append("")

        summary_lines.extend(
            [
                "Operational response:",
                (
                    "A PASS result indicates that the monitored "
                    "feature distributions remain within the "
                    "defined threshold."
                ),
                (
                    "An ALERT result requires investigation before "
                    "model retraining, replacement or continued "
                    "deployment is considered."
                ),
                (
                    "A monitoring alert should not trigger automatic "
                    "retraining. Data quality, model performance and "
                    "the cause of the distribution change should be "
                    "reviewed first."
                ),
                (
                    "This demonstration uses simulated monitoring "
                    "scenarios and does not represent a live "
                    "production monitoring service."
                ),
            ]
        )

        SUMMARY_PATH.write_text(
            "\n".join(summary_lines),
            encoding="utf-8",
        )

    except OSError as exc:
        raise DriftMonitoringError(
            f"Unable to save monitoring outputs: {exc}"
        ) from exc

    LOGGER.info(
        "Saved detailed drift results to %s",
        RESULTS_PATH,
    )

    LOGGER.info(
        "Saved drift summary to %s",
        SUMMARY_PATH,
    )


def run_monitoring() -> pd.DataFrame:
    """Run normal and shifted drift-monitoring scenarios."""
    data = load_monitoring_data(DATA_PATH)
    reference_data = create_reference_data(data)

    normal_sample = create_normal_sample(reference_data)
    shifted_sample = create_shifted_sample(normal_sample)

    normal_results = evaluate_scenario(
        reference_data,
        normal_sample,
        "Normal simulated sample",
    )

    shifted_results = evaluate_scenario(
        reference_data,
        shifted_sample,
        "Shifted alert-test sample",
    )

    combined_results = pd.concat(
        [normal_results, shifted_results],
        ignore_index=True,
    )

    save_results(combined_results)

    return combined_results


def main() -> int:
    """Run drift monitoring from the command line."""
    configure_logging()
    LOGGER.info("Drift-monitoring workflow started")

    try:
        results = run_monitoring()
    except DriftMonitoringError:
        LOGGER.error(
            "Drift monitoring failed",
            exc_info=True,
        )
        return 1
    except Exception:
        LOGGER.error(
            "Unexpected drift-monitoring failure",
            exc_info=True,
        )
        return 1

    print(
        results[
            [
                "Scenario",
                "Feature",
                "PSI",
                "Drift_Level",
                "Feature_Status",
                "Overall_Status",
            ]
        ].to_string(index=False)
    )

    LOGGER.info("Drift-monitoring workflow completed successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())