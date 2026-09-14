"""Create the SQLite database used for the Part 1 traffic analysis."""

from pathlib import Path
import logging
import sqlite3

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
DATABASE_PATH = Path(__file__).resolve().parent / "traffic.db"
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


def create_database(csv_path: Path = CSV_PATH, database_path: Path = DATABASE_PATH) -> None:
    """Load the supplied CSV into a reproducible SQLite database."""
    try:
        traffic = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        LOGGER.error("Unable to load %s", csv_path, exc_info=True)
        raise SystemExit(1) from exc

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(traffic.columns))
    if missing_columns:
        LOGGER.error("Dataset is missing expected columns: %s", missing_columns)
        raise SystemExit(1)

    with sqlite3.connect(database_path) as connection:
        traffic.to_sql("traffic", connection, if_exists="replace", index=False)
        row_count = connection.execute("SELECT COUNT(*) FROM traffic").fetchone()[0]

    LOGGER.info(
        "Created %s with %d rows and %d columns",
        database_path,
        row_count,
        len(traffic.columns),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    create_database()
