# Part 1 Data Analytics

This folder contains the reproducible SQL, statistical and probability analysis for the Metro Interstate Traffic Volume dataset. The Power BI dashboard and written insights report will also be stored here.

## Run the analysis

From the project root:

```bash
python part1_data_analytics/create_database.py
python part1_data_analytics/part1_analysis.py
```

Open `part1_data_analytics/traffic.db` with SQLite and run the statements in `traffic_analysis.sql`.

## Definitions

- Congestion: traffic volume greater than 5,500 vehicles per hour.
- Clear weather: `weather_main` equals `Clear`.
- Cloudy weather: `weather_main` equals `Clouds`.
- High temperature: temperature greater than 292 K.
- Odds ratio: congestion odds in clear weather divided by congestion odds in cloudy weather.

Annual totals are reported as requested, but they must be interpreted alongside recorded-hour counts because 2012, 2014 and 2015 have incomplete temporal coverage.
