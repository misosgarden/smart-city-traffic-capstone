"""Generate and interpret three reproducible traffic visualisations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "processed" / "traffic_features.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figures"
DEFAULT_INTERPRETATIONS = BASE_DIR / "visualisation_interpretations.txt"
DEFAULT_LOG = BASE_DIR / "pipeline.log"

REQUIRED_COLUMNS = {
    "hour",
    "is_weekend",
    "traffic_volume",
    "congestion_quartile",
}


class VisualisationError(RuntimeError):
    """Raised when the visualisation workflow cannot complete safely."""


def configure_logging(log_path: Path, debug: bool = False) -> None:
    """Configure console and append-only file handlers at the entry point."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    LOGGER.setLevel(level)
    LOGGER.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False

    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_feature_data(input_path: Path) -> pd.DataFrame:
    """Load and validate the engineered feature dataset."""
    try:
        data = pd.read_csv(input_path)
    except FileNotFoundError as exc:
        raise VisualisationError(f"Feature dataset not found: {input_path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise VisualisationError(
            f"Unable to read feature dataset {input_path}: {exc}"
        ) from exc

    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise VisualisationError(
            f"Feature dataset is missing required columns: {missing}"
        )

    LOGGER.info(
        "Loaded feature dataset from %s with %d rows and %d columns",
        input_path,
        data.shape[0],
        data.shape[1],
    )
    return data


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close a Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("Saved visualisation to %s", path)


def plot_hourly_patterns(data: pd.DataFrame, output_dir: Path) -> str:
    """Compare hourly average traffic on weekdays and weekends."""
    hourly = (
        data.groupby(["hour", "is_weekend"], as_index=False)["traffic_volume"]
        .mean()
        .pivot(index="hour", columns="is_weekend", values="traffic_volume")
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hourly.index, hourly[0], marker="o", linewidth=2.2, label="Weekday")
    ax.plot(hourly.index, hourly[1], marker="o", linewidth=2.2, label="Weekend")
    ax.set_title("Average Traffic Volume by Hour and Day Type", weight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average traffic volume")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "hourly_traffic_weekday_weekend.png")

    weekday_morning_hour = int(hourly.loc[5:11, 0].idxmax())
    weekday_morning_value = float(hourly.loc[5:11, 0].max())
    weekday_evening_hour = int(hourly.loc[12:20, 0].idxmax())
    weekday_evening_value = float(hourly.loc[12:20, 0].max())
    weekend_plateau_value = float(hourly.loc[12:16, 1].mean())

    morning_display = round(weekday_morning_value, -1)
    evening_display = round(weekday_evening_value, -1)
    weekend_display = round(weekend_plateau_value / 50) * 50

    return (
        f"Hourly patterns: weekday traffic is bimodal, with a morning peak around "
        f"{weekday_morning_hour:02d}:00 (~{morning_display:,.0f}) and a taller evening "
        f"peak at {weekday_evening_hour:02d}:00 (~{evening_display:,.0f}), reflecting "
        "the standard commute. Weekend traffic has no sharp peak, instead rising to a "
        f"broad plateau of around {weekend_display:,.0f} from midday to late afternoon. "
        "This means traffic-management interventions timed for weekday rush hours "
        "(signal retiming, incident response staffing) would be poorly matched to "
        "weekend demand, which needs a different, flatter response window."
    )


def plot_traffic_distribution(data: pd.DataFrame, output_dir: Path) -> str:
    """Show the distribution of hourly traffic volume."""
    traffic = data["traffic_volume"]
    mean_value = float(traffic.mean())
    median_value = float(traffic.median())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(traffic, bins=35, color="#2589BD", edgecolor="white", alpha=0.9)
    ax.axvline(mean_value, color="#C0392B", linestyle="--", linewidth=2,
               label=f"Mean: {mean_value:,.0f}")
    ax.axvline(median_value, color="#1B4F72", linestyle=":", linewidth=2.2,
               label=f"Median: {median_value:,.0f}")
    ax.set_title("Distribution of Hourly Traffic Volume", weight="bold")
    ax.set_xlabel("Traffic volume")
    ax.set_ylabel("Number of hourly records")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "traffic_volume_distribution.png")

    skew_value = float(traffic.skew())
    counts, bin_edges = np.histogram(traffic, bins=35)
    upper_bins = bin_edges[:-1] > median_value
    upper_counts = counts.copy()
    upper_counts[~upper_bins] = 0
    peak_idx = int(upper_counts.argmax())
    second_peak_low = float(bin_edges[peak_idx])
    second_peak_high = float(bin_edges[peak_idx + 1])

    return (
        f"Traffic distribution: the distribution is bimodal, with a large cluster of "
        f"low-traffic overnight readings and a second, broader concentration between "
        f"{second_peak_low:,.0f} and {second_peak_high:,.0f} vehicles (daytime commute "
        f"hours). The mean ({mean_value:,.0f}) sits close to the median "
        f"({median_value:,.0f}), with only a slight negative skew ({skew_value:.3f}), "
        "so the shape is better described as bimodal than simply left-skewed. This "
        "mirrors the weekday/weekend pattern in the hourly chart: traffic volume on "
        "this corridor is fundamentally a function of time of day rather than a "
        "smoothly varying quantity."
    )


def derive_weather_category(data: pd.DataFrame) -> pd.Series:
    """Recover weather labels from the one-hot-encoded weather columns."""
    weather_columns = sorted(
        column for column in data.columns if column.startswith("weather_main_")
    )
    if not weather_columns:
        raise VisualisationError("No one-hot-encoded weather columns were found")

    encoded = data[weather_columns]
    invalid_rows = int((encoded.sum(axis=1) != 1).sum())
    if invalid_rows:
        raise VisualisationError(
            f"Found {invalid_rows} rows without exactly one weather category"
        )

    labels = encoded.idxmax(axis=1).str.replace("weather_main_", "", regex=False)
    return labels


def plot_weather_comparison(data: pd.DataFrame, output_dir: Path) -> str:
    """Compare average traffic volume across main weather categories."""
    visual_data = data[["traffic_volume"]].copy()
    visual_data["weather_main"] = derive_weather_category(data)
    summary = (
        visual_data.groupby("weather_main")["traffic_volume"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(summary.index, summary["mean"], color="#2589BD")
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in summary["mean"]], padding=3)
    ax.set_title("Average Traffic Volume by Weather Condition", weight="bold")
    ax.set_xlabel("Average traffic volume")
    ax.set_ylabel("Weather condition")
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, summary["mean"].max() * 1.15)
    fig.tight_layout()
    save_figure(fig, output_dir / "average_traffic_by_weather.png")

    highest = summary["mean"].idxmax()
    lowest = summary["mean"].idxmin()
    difference = float(summary.loc[highest, "mean"] - summary.loc[lowest, "mean"])
    lowest_count = int(summary.loc[lowest, "count"])
    return (
        f"Weather comparison: {highest} has the highest average traffic and {lowest} "
        f"has the lowest, a difference of about {difference:,.0f} vehicles. The "
        f"{lowest} result is based on only {lowest_count} records, so it should be "
        "interpreted cautiously and weather should not be treated as a sole predictor."
    )


def write_interpretations(interpretations: list[str], output_path: Path) -> None:
    """Save concise stakeholder-focused interpretations for the figures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "Part 2 Visualisation Interpretations\n\n" + "\n\n".join(
        f"{index}. {interpretation}"
        for index, interpretation in enumerate(interpretations, start=1)
    )
    try:
        output_path.write_text(content + "\n", encoding="utf-8")
    except OSError as exc:
        raise VisualisationError(
            f"Unable to save interpretations to {output_path}: {exc}"
        ) from exc
    LOGGER.info("Saved visualisation interpretations to %s", output_path)


def create_visualisations(
    input_path: Path,
    output_dir: Path,
    interpretations_path: Path,
) -> list[str]:
    """Create all three figures and their written interpretations."""
    data = load_feature_data(input_path)
    interpretations = [
        plot_hourly_patterns(data, output_dir),
        plot_traffic_distribution(data, output_dir),
        plot_weather_comparison(data, output_dir),
    ]
    write_interpretations(interpretations, interpretations_path)
    return interpretations


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Create three traffic visualisations and interpretations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--interpretations", type=Path, default=DEFAULT_INTERPRETATIONS
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    """Run the visualisation workflow from the command line."""
    args = build_parser().parse_args()
    configure_logging(args.log, args.debug)
    LOGGER.info("Visualisation workflow invoked with arguments: %s", vars(args))

    try:
        create_visualisations(
            args.input,
            args.output_dir,
            args.interpretations,
        )
    except VisualisationError:
        LOGGER.error("Visualisation workflow failed", exc_info=True)
        return 1
    except Exception:
        LOGGER.error("Unexpected visualisation failure", exc_info=True)
        return 1

    LOGGER.info("Visualisation workflow completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
