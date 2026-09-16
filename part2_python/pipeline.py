"""Clean and validate the Metro Interstate Traffic Volume dataset.

This module forms Task 1 of the Part 2 capstone pipeline. It validates the
input schema before processing, removes duplicate records, standardises text
fields, parses timestamps, and imputes identified sensor outliers using
month-specific medians.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "processed" / "traffic_cleaned.csv"
DEFAULT_LOG = Path(__file__).resolve().parent / "pipeline.log"

EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

NUMERIC_COLUMNS = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "traffic_volume",
]


class PipelineError(Exception):
    """Raised when the cleaning pipeline cannot complete safely."""


def configure_logging(log_path: Path, debug: bool = False) -> None:
    """Configure console and file handlers for the entry-point script."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV and convert expected read failures to PipelineError."""
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise PipelineError(f"Input file was not found: {csv_path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise PipelineError(f"Input file could not be parsed: {csv_path}") from exc

    LOGGER.info(
        "Loaded raw dataset from %s with %d rows and %d columns",
        csv_path,
        data.shape[0],
        data.shape[1],
    )
    return data


def validate_schema(data: pd.DataFrame) -> None:
    """Validate the column schema before any data-processing operations."""
    missing = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    unexpected = sorted(set(data.columns) - set(EXPECTED_COLUMNS))

    if missing:
        raise PipelineError(f"Dataset is missing expected columns: {missing}")
    if unexpected:
        LOGGER.warning("Dataset contains unexpected columns: %s", unexpected)

    LOGGER.info("Schema validation passed for %d expected columns", len(EXPECTED_COLUMNS))


def standardise_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Standardise missing and inconsistent categorical values."""
    cleaned = data.copy()

    missing_holidays = int(cleaned["holiday"].isna().sum())
    if missing_holidays:
        cleaned["holiday"] = cleaned["holiday"].fillna("No Holiday")
        LOGGER.warning(
            "Modified %d rows by replacing blank holiday values with 'No Holiday'",
            missing_holidays,
        )

    category_rules = {
        "holiday": lambda values: values.astype(str).str.strip(),
        "weather_main": lambda values: values.astype(str).str.strip().str.title(),
        "weather_description": lambda values: values.astype(str).str.strip().str.lower(),
    }
    for column, rule in category_rules.items():
        before = cleaned[column].copy()
        cleaned[column] = rule(cleaned[column])
        changed = int(before.fillna("").ne(cleaned[column].fillna("")).sum())
        if changed:
            LOGGER.warning(
                "Modified %d rows while standardising categorical column %s",
                changed,
                column,
            )
        else:
            LOGGER.info("Categorical column %s required no standardisation", column)

    return cleaned


def parse_and_validate_types(data: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and validate numeric fields."""
    cleaned = data.copy()
    cleaned["date_time"] = pd.to_datetime(cleaned["date_time"], errors="coerce")
    invalid_dates = int(cleaned["date_time"].isna().sum())
    if invalid_dates:
        cleaned = cleaned.dropna(subset=["date_time"]).copy()
        LOGGER.warning("Dropped %d rows with invalid date_time values", invalid_dates)
    else:
        LOGGER.info("Parsed date_time successfully with no invalid timestamps")

    for column in NUMERIC_COLUMNS:
        original_missing = cleaned[column].isna()
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        newly_invalid = int((cleaned[column].isna() & ~original_missing).sum())
        if newly_invalid:
            LOGGER.warning(
                "Converted %d invalid values to missing in numeric column %s",
                newly_invalid,
                column,
            )

    missing_required = cleaned[["temp", "rain_1h", "traffic_volume"]].isna().any(axis=1)
    missing_count = int(missing_required.sum())
    if missing_count:
        cleaned = cleaned.loc[~missing_required].copy()
        LOGGER.warning(
            "Dropped %d rows missing required numeric sensor values",
            missing_count,
        )

    LOGGER.info("Data-type validation completed")
    return cleaned


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        cleaned = data.drop_duplicates().copy()
        LOGGER.warning("Dropped %d exact duplicate rows", duplicate_count)
        return cleaned

    LOGGER.info("No exact duplicate rows were found")
    return data.copy()


def impute_by_month(
    data: pd.DataFrame,
    column: str,
    outlier_mask: pd.Series,
    reason: str,
) -> pd.DataFrame:
    """Replace flagged values with valid medians calculated for each month."""
    cleaned = data.copy()
    affected = int(outlier_mask.sum())
    if not affected:
        LOGGER.info("No outliers detected in %s for rule: %s", column, reason)
        return cleaned

    valid_values = cleaned.loc[~outlier_mask, column]
    global_median = float(valid_values.median())
    affected_months = sorted(cleaned.loc[outlier_mask, "date_time"].dt.month.unique())

    for month in affected_months:
        month_outliers = outlier_mask & cleaned["date_time"].dt.month.eq(month)
        valid_month_values = cleaned.loc[
            ~outlier_mask & cleaned["date_time"].dt.month.eq(month), column
        ]
        replacement = (
            float(valid_month_values.median())
            if not valid_month_values.empty
            else global_median
        )
        cleaned.loc[month_outliers, column] = replacement
        LOGGER.debug(
            "Imputation value for %s in month %d was %.4f",
            column,
            month,
            replacement,
        )

    LOGGER.warning(
        "Imputed %d rows in %s using monthly medians because %s",
        affected,
        column,
        reason,
    )
    return cleaned


def handle_sensor_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Detect and impute known impossible or implausible sensor readings."""
    cleaned = data.copy()

    invalid_temperature = cleaned["temp"].le(0)
    cleaned = impute_by_month(
        cleaned,
        "temp",
        invalid_temperature,
        "temperature was 0 Kelvin or lower",
    )

    extreme_rainfall = cleaned["rain_1h"].gt(9_000)
    cleaned = impute_by_month(
        cleaned,
        "rain_1h",
        extreme_rainfall,
        "hourly rainfall exceeded 9,000 mm",
    )

    return cleaned


def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Run all validation and cleaning stages and save the cleaned dataset."""
    data = load_raw_data(input_path)
    validate_schema(data)
    data = standardise_categories(data)
    data = parse_and_validate_types(data)
    data = remove_duplicates(data)
    data = handle_sensor_outliers(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(output_path, index=False)
    except OSError as exc:
        raise PipelineError(f"Cleaned dataset could not be saved: {output_path}") from exc

    LOGGER.info(
        "Saved cleaned dataset to %s with %d rows and %d columns",
        output_path,
        data.shape[0],
        data.shape[1],
    )
    return data


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the pipeline entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Configure logging and run the pipeline with graceful error handling."""
    args = parse_arguments()
    configure_logging(args.log_file, debug=args.debug)
    try:
        run_pipeline(args.input, args.output)
    except PipelineError:
        LOGGER.error("Pipeline could not complete", exc_info=True)
        return 1
    except Exception:
        LOGGER.error("Unexpected pipeline failure", exc_info=True)
        return 1
    LOGGER.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
