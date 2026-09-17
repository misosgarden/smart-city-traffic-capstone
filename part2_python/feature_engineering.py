"""Create reproducible model-ready features from the cleaned traffic dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "processed" / "traffic_cleaned.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "processed" / "traffic_features.csv"
DEFAULT_LOG = Path(__file__).resolve().parent / "pipeline.log"

REQUIRED_COLUMNS = {
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "weather_main",
    "date_time",
    "traffic_volume",
}


class FeatureEngineeringError(RuntimeError):
    """Raised when feature engineering cannot complete safely."""


def configure_logging(log_path: Path, debug: bool = False) -> None:
    """Configure console and append-only file logging at the entry point."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_cleaned_data(input_path: Path) -> pd.DataFrame:
    """Load the cleaned data with clear error handling."""
    try:
        data = pd.read_csv(input_path)
    except FileNotFoundError as exc:
        raise FeatureEngineeringError(f"Cleaned dataset not found: {input_path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise FeatureEngineeringError(
            f"Unable to read cleaned dataset {input_path}: {exc}"
        ) from exc

    LOGGER.info(
        "Loaded cleaned dataset from %s with %d rows and %d columns",
        input_path,
        data.shape[0],
        data.shape[1],
    )
    return data


def validate_input(data: pd.DataFrame) -> None:
    """Check the required schema before creating any features."""
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise FeatureEngineeringError(
            f"Feature engineering input is missing required columns: {missing}"
        )
    LOGGER.info("Feature engineering schema validation passed")


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, weekend and cyclical time features."""
    featured = data.copy()
    featured["date_time"] = pd.to_datetime(featured["date_time"], errors="coerce")
    invalid_dates = int(featured["date_time"].isna().sum())
    if invalid_dates:
        raise FeatureEngineeringError(
            f"Found {invalid_dates} invalid date_time values in cleaned data"
        )

    featured["hour"] = featured["date_time"].dt.hour
    featured["day_of_week"] = featured["date_time"].dt.dayofweek
    featured["month"] = featured["date_time"].dt.month
    featured["is_weekend"] = featured["day_of_week"].isin([5, 6]).astype(int)

    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)
    featured["month_sin"] = np.sin(2 * np.pi * (featured["month"] - 1) / 12)
    featured["month_cos"] = np.cos(2 * np.pi * (featured["month"] - 1) / 12)

    LOGGER.info(
        "Added time and cyclical features; dataset shape is %s", featured.shape
    )
    return featured


def add_indicator_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable binary indicators derived from existing fields."""
    featured = data.copy()
    featured["is_holiday"] = featured["holiday"].ne("No Holiday").astype(int)
    featured["is_rainy"] = (featured["rain_1h"] > 0).astype(int)
    featured["is_snowy"] = (featured["snow_1h"] > 0).astype(int)
    featured["is_peak_hour"] = featured["hour"].isin([7, 8, 9, 15, 16, 17, 18]).astype(int)

    LOGGER.info(
        "Added holiday, precipitation and peak-hour indicators; dataset shape is %s",
        featured.shape,
    )
    return featured


def add_congestion_quartiles(data: pd.DataFrame) -> pd.DataFrame:
    """Create an ordered congestion category using traffic-volume quartiles."""
    featured = data.copy()
    labels = ["Low", "Medieum", "High", "Severe"]
    try:
        featured["congestion_quartile"] = pd.qcut(
            featured["traffic_volume"], q=4, labels=labels
        )
    except ValueError as exc:
        raise FeatureEngineeringError(
            "Unable to create four unique traffic-volume quartiles"
        ) from exc

    thresholds = featured["traffic_volume"].quantile([0.25, 0.50, 0.75]).to_dict()
    LOGGER.debug(
        "Congestion quartile thresholds: Q1=%.2f, Q2=%.2f, Q3=%.2f",
        thresholds[0.25],
        thresholds[0.50],
        thresholds[0.75],
    )
    LOGGER.info(
        "Added congestion quartile category; dataset shape is %s", featured.shape
    )
    return featured


def add_scaled_features(data: pd.DataFrame) -> pd.DataFrame:
    """Standardise temperature and traffic volume while retaining original values."""
    featured = data.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(featured[["temp", "traffic_volume"]])
    featured[["temp_scaled", "traffic_volume_scaled"]] = scaled

    LOGGER.debug(
        "Scaling parameters: temp mean=%.4f scale=%.4f; traffic_volume mean=%.4f scale=%.4f",
        scaler.mean_[0],
        scaler.scale_[0],
        scaler.mean_[1],
        scaler.scale_[1],
    )
    LOGGER.info(
        "Standardised temp and traffic_volume; dataset shape is %s", featured.shape
    )
    return featured


def encode_weather(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the main weather category."""
    encoded = pd.get_dummies(
        data,
        columns=["weather_main"],
        prefix="weather_main",
        dtype=int,
    )
    weather_columns = sorted(
        column for column in encoded.columns if column.startswith("weather_main_")
    )
    LOGGER.debug("Created weather indicator columns: %s", weather_columns)
    LOGGER.info(
        "One-hot encoded weather_main into %d columns; dataset shape is %s",
        len(weather_columns),
        encoded.shape,
    )
    return encoded


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Run all feature-engineering stages in a reproducible order."""
    validate_input(data)
    featured = add_time_features(data)
    featured = add_indicator_features(featured)
    featured = add_congestion_quartiles(featured)
    featured = add_scaled_features(featured)
    featured = encode_weather(featured)
    return featured


def run_feature_engineering(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Load cleaned data, create features and save the resulting dataset."""
    cleaned = load_cleaned_data(input_path)
    featured = engineer_features(cleaned)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        featured.to_csv(output_path, index=False)
    except OSError as exc:
        raise FeatureEngineeringError(
            f"Unable to save feature dataset to {output_path}: {exc}"
        ) from exc

    LOGGER.info(
        "Saved feature dataset to %s with %d rows and %d columns",
        output_path,
        featured.shape[0],
        featured.shape[1],
    )
    return featured


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Create engineered features from the cleaned traffic dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    """Run feature engineering from the command line."""
    args = build_parser().parse_args()
    configure_logging(args.log, args.debug)
    LOGGER.info("Feature engineering invoked with arguments: %s", vars(args))

    try:
        run_feature_engineering(args.input, args.output)
    except FeatureEngineeringError:
        LOGGER.error("Feature engineering failed", exc_info=True)
        return 1
    except Exception:
        LOGGER.error("Unexpected feature-engineering failure", exc_info=True)
        return 1

    LOGGER.info("Feature engineering completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
