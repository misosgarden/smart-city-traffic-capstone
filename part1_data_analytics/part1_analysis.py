"""Calculate and export the Part 1 statistical and probability results."""

from pathlib import Path
import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the capstone CSV and parse its timestamp."""
    traffic = pd.read_csv(csv_path)
    traffic["date_time"] = pd.to_datetime(traffic["date_time"], errors="raise")
    LOGGER.info("Loaded %d rows and %d columns", *traffic.shape)
    return traffic


def descriptive_statistics(traffic: pd.DataFrame) -> pd.DataFrame:
    """Return the required traffic-volume descriptive statistics."""
    volume = traffic["traffic_volume"]
    values = {
        "Mean": volume.mean(),
        "Median": volume.median(),
        "Standard deviation": volume.std(ddof=1),
        "Variance": volume.var(ddof=1),
        "Range": volume.max() - volume.min(),
        "Minimum": volume.min(),
        "Maximum": volume.max(),
    }
    return pd.DataFrame(values.items(), columns=["Statistic", "Value"])


def probability_analysis(traffic: pd.DataFrame) -> pd.DataFrame:
    """Calculate required congestion and weather probabilities."""
    congestion = traffic["traffic_volume"] > 5_500
    clear = traffic["weather_main"].eq("Clear")
    cloudy = traffic["weather_main"].eq("Clouds")
    high_temperature = traffic["temp"] > 292

    clear_congestion = int((clear & congestion).sum())
    clear_not_congestion = int((clear & ~congestion).sum())
    cloudy_congestion = int((cloudy & congestion).sum())
    cloudy_not_congestion = int((cloudy & ~congestion).sum())

    p_congestion = congestion.mean()
    p_clear = clear.mean()
    p_joint = (congestion & clear).mean()
    p_product = p_congestion * p_clear
    odds_ratio = (
        (clear_congestion / clear_not_congestion)
        / (cloudy_congestion / cloudy_not_congestion)
    )

    values = {
        "P(Congestion)": p_congestion,
        "P(Clear weather)": p_clear,
        "P(Congestion AND clear weather)": p_joint,
        "P(Clear weather | Congestion)": clear_congestion / congestion.sum(),
        "P(High temperature | Congestion)": (
            (high_temperature & congestion).sum() / congestion.sum()
        ),
        "P(Congestion) x P(Clear weather)": p_product,
        "Joint minus independence product": p_joint - p_product,
        "Odds ratio, clear versus cloudy": odds_ratio,
    }
    return pd.DataFrame(values.items(), columns=["Measure", "Value"])


def annual_analysis(traffic: pd.DataFrame) -> pd.DataFrame:
    """Summarise annual traffic while retaining data-coverage information."""
    selected = traffic.assign(year=traffic["date_time"].dt.year)
    selected = selected[selected["year"].between(2012, 2017)]
    annual = (
        selected.groupby("year")
        .agg(
            recorded_hours=("traffic_volume", "size"),
            total_traffic_volume=("traffic_volume", "sum"),
            average_hourly_volume=("traffic_volume", "mean"),
            first_record=("date_time", "min"),
            last_record=("date_time", "max"),
        )
        .reset_index()
    )
    annual["absolute_change"] = annual["total_traffic_volume"].diff()
    annual["percentage_change"] = annual["total_traffic_volume"].pct_change() * 100
    return annual


def holiday_analysis(traffic: pd.DataFrame) -> pd.DataFrame:
    """Compare unique holiday timestamps without double-counting weather rows."""
    selected = traffic.assign(year=traffic["date_time"].dt.year)
    selected = selected[
        selected["year"].between(2015, 2017)
        & selected["holiday"].isin(["New Years Day", "Labor Day"])
    ]
    selected = selected.drop_duplicates(
        subset=["date_time", "holiday", "temp", "traffic_volume"]
    )
    return (
        selected.groupby(["year", "holiday"], as_index=False)
        .agg(
            unique_observations=("date_time", "size"),
            average_temperature_kelvin=("temp", "mean"),
            average_traffic_volume=("traffic_volume", "mean"),
        )
        .assign(
            average_temperature_celsius=lambda frame: (
                frame["average_temperature_kelvin"] - 273.15
            )
        )
    )


def main() -> None:
    """Run the complete Part 1 analysis and save reproducible outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traffic = load_data()

    results = {
        "descriptive_statistics.csv": descriptive_statistics(traffic),
        "probability_analysis.csv": probability_analysis(traffic),
        "annual_analysis.csv": annual_analysis(traffic),
        "holiday_analysis.csv": holiday_analysis(traffic),
    }
    correlation = traffic[["temp", "traffic_volume"]].corr().iloc[0, 1]
    results["correlation_analysis.csv"] = pd.DataFrame(
        {
            "Variable 1": ["Temperature (K)"],
            "Variable 2": ["Traffic volume"],
            "Pearson correlation": [correlation],
        }
    )

    for filename, result in results.items():
        destination = OUTPUT_DIR / filename
        result.to_csv(destination, index=False)
        LOGGER.info("Saved %s", destination)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
