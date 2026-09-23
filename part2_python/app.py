"""Interactive command-line application for exploring traffic patterns."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "traffic_features.csv"
LOG_PATH = BASE_DIR / "pipeline.log"

class AppError(Exception):
    """Raised when the traffic application cannot continue safely."""


def configure_logging() -> None:
    """Send application messages to both the terminal and pipeline.log."""
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_PATH,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def load_traffic_data() -> pd.DataFrame:
    """Load and validate the engineered traffic dataset."""
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError as exc:
        raise AppError(
            f"Traffic feature dataset not found: {DATA_PATH}"
        ) from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise AppError(
            f"Unable to read the traffic feature dataset: {exc}"
        ) from exc

    required_columns = {
        "date_time",
        "hour",
        "is_weekend",
        "traffic_volume",
        "congestion_quartile",
    }

    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise AppError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce",
    )

    invalid_dates = int(data["date_time"].isna().sum())
    if invalid_dates:
        raise AppError(
            f"Dataset contains {invalid_dates} invalid date_time values"
        )

    LOGGER.info(
        "Loaded traffic application dataset with %d rows and %d columns",
        data.shape[0],
        data.shape[1],
    )

    return data

def look_up_date(data: pd.DataFrame) -> None:
    """Display hourly traffic information for a user-selected date."""
    date_text = input("Enter a date in YYYY-MM-DD format: ").strip()
    LOGGER.info("Date lookup requested with date=%s", date_text)

    try:
        selected_date = pd.to_datetime(
            date_text,
            format="%Y-%m-%d",
            errors="raise",
        )
    except ValueError:
        LOGGER.error("Invalid date entered: %s", date_text)
        print("ERROR: Enter the date in YYYY-MM-DD format.")
        return

    matching_rows = data[
        data["date_time"].dt.date == selected_date.date()
    ]

    if matching_rows.empty:
        LOGGER.warning("No traffic records found for date=%s", date_text)
        print(f"No traffic records were found for {date_text}.")
        return

    hourly_summary = (
        matching_rows.groupby("hour")["traffic_volume"]
        .mean()
        .round(0)
        .astype(int)
        .reset_index(name="average_traffic_volume")
    )

    peak_row = hourly_summary.loc[
        hourly_summary["average_traffic_volume"].idxmax()
    ]

    print(f"\nTraffic summary for {date_text}")
    print(hourly_summary.to_string(index=False))
    print(
        f"\nBusiest recorded hour: {int(peak_row['hour']):02d}:00 "
        f"with average traffic volume of "
        f"{int(peak_row['average_traffic_volume']):,} vehicles."
    )

    LOGGER.info(
        "Date lookup completed for %s with %d matching records",
        date_text,
        len(matching_rows),
    )

def show_busiest_periods(data: pd.DataFrame) -> None:
    """Display the observations with the highest traffic volumes."""
    number_text = input(
        "How many high-traffic observations should be displayed? "
    ).strip()

    try:
        number = int(number_text)
    except ValueError:
        LOGGER.error(
            "Invalid number entered for busiest-period query: %s",
            number_text,
        )
        print("ERROR: Enter a whole number.")
        return

    if number < 1 or number > 20:
        LOGGER.error(
            "Busiest-period query requested an invalid count: %d",
            number,
        )
        print("ERROR: Enter a number between 1 and 20.")
        return

    LOGGER.info(
        "Busiest-period query requested with count=%d",
        number,
    )

    busiest = (
        data.nlargest(number, "traffic_volume")[
            [
                "date_time",
                "traffic_volume",
                "congestion_quartile",
            ]
        ]
        .copy()
    )

    busiest["date_time"] = busiest["date_time"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    print(f"\nTop {number} highest-traffic observations")
    print(busiest.to_string(index=False))

    LOGGER.info(
        "Busiest-period query completed with %d results",
        len(busiest),
    )

def compare_day_types(data: pd.DataFrame) -> None:
    """Compare average traffic volume on weekdays and weekends."""
    LOGGER.info("Weekday and weekend comparison requested")

    comparison_data = data.copy()
    comparison_data["day_type"] = comparison_data[
        "is_weekend"
    ].map(
        {
            0: "Weekday",
            1: "Weekend",
        }
    )

    summary = (
        comparison_data.groupby("day_type")["traffic_volume"]
        .agg(["mean", "median", "count"])
        .round(
            {
                "mean": 0,
                "median": 0,
            }
        )
        .astype(
            {
                "mean": int,
                "median": int,
                "count": int,
            }
        )
        .reset_index()
    )

    weekday_average = int(
        summary.loc[
            summary["day_type"] == "Weekday",
            "mean",
        ].iloc[0]
    )

    weekend_average = int(
        summary.loc[
            summary["day_type"] == "Weekend",
            "mean",
        ].iloc[0]
    )

    difference = weekday_average - weekend_average

    print("\nWeekday and weekend traffic comparison")
    print(summary.to_string(index=False))
    print(
        f"\nAverage weekday traffic is {difference:,} vehicles "
        "higher than average weekend traffic."
    )

    LOGGER.info(
        "Weekday and weekend comparison completed; "
        "average difference=%d",
        difference,
    )

def show_quiet_travel_hours(data: pd.DataFrame) -> None:
    """Recommend lower-traffic hours for weekdays and weekends."""
    LOGGER.info("Lower-traffic travel-hour query requested")

    hourly = (
        data.groupby(["is_weekend", "hour"])["traffic_volume"]
        .mean()
        .reset_index(name="average_traffic_volume")
    )

    hourly["day_type"] = hourly["is_weekend"].map(
        {
            0: "Weekday",
            1: "Weekend",
        }
    )

    recommendations = []

    for day_type in ["Weekday", "Weekend"]:
        day_summary = hourly[
            hourly["day_type"] == day_type
        ]

        quietest = day_summary.nsmallest(
            3,
            "average_traffic_volume",
        ).copy()

        quietest["average_traffic_volume"] = (
            quietest["average_traffic_volume"]
            .round(0)
            .astype(int)
        )

        quietest["recommended_time"] = quietest["hour"].map(
            lambda hour: f"{int(hour):02d}:00"
        )

        recommendations.append(
            quietest[
                [
                    "day_type",
                    "recommended_time",
                    "average_traffic_volume",
                ]
            ]
        )

    recommendation_table = pd.concat(
        recommendations,
        ignore_index=True,
    )

    print("\nRecommended lower-traffic travel hours")
    print(recommendation_table.to_string(index=False))
    print(
        "\nThese recommendations are based on historical average "
        "traffic volumes and do not account for real-time incidents."
    )

    LOGGER.info(
        "Lower-traffic travel-hour query completed with %d recommendations",
        len(recommendation_table),
    )
def display_menu() -> None:
    """Display the available traffic analysis commands."""
    print(
        "\nSmart City Traffic Analysis App\n"
        "1. Look up traffic by date\n"
        "2. Show busiest traffic periods\n"
        "3. Compare weekday and weekend traffic\n"
        "4. Show lower-traffic travel hours\n"
        "5. Exit"
    )


def run_app(data: pd.DataFrame) -> None:
    """Run the interactive menu until the user chooses to exit."""
    while True:
        display_menu()
        choice = input("Select an option (1-5): ").strip()
        LOGGER.info("Menu command selected: %s", choice)

        if choice == "1":
            look_up_date(data)
        elif choice == "2":
            show_busiest_periods(data)
        elif choice == "3":
            compare_day_types(data)
        elif choice == "4":
            show_quiet_travel_hours(data)
        elif choice == "5":
            LOGGER.info("User selected Exit")
            print("Exiting the traffic analysis app.")
            break
        else:
            LOGGER.error("Invalid menu selection: %s", choice)
            print("ERROR: Select a number from 1 to 5.")


def main() -> int:
    """Load the dataset and start the traffic analysis application."""
    configure_logging()
    LOGGER.info("Traffic analysis application started")

    try:
        data = load_traffic_data()
        run_app(data)
    except AppError as exc:
        LOGGER.error("Application failed: %s", exc)
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Application stopped by the user")
        print("\nApplication stopped.")
        return 0

    LOGGER.info("Traffic analysis application closed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())