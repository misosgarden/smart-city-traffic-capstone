# Part 1 Data Analytics

This folder contains the SQL, Excel, Power BI and written deliverables for Part 1 of the Smart City Traffic Intelligence capstone project.

## Files

- `traffic.db`: SQLite database containing the Metro Interstate Traffic Volume dataset.
- `traffic_analysis.sql`: SQL queries used to verify the database, analyse annual traffic trends and compare temperatures and traffic around New Year's Day and Labor Day.
- `Part_1_Data_Analytics_Excel_Analysis.xlsx`: Excel workbook containing the descriptive statistics, correlation analysis, probability calculations, SQL results and interpretations.
- `traffic_intelligence_dashboard.pbix`: Interactive Power BI dashboard covering daily trends, hourly patterns, weather conditions, temperature and traffic, KPI cards and slicers.
- `Part_1_Data_Analytics_Insights_Report_Final.docx`: Final 1 to 2 page insights report summarising the main findings and implications for the smart-city mobility team.

The original traffic CSV is stored separately in `data/raw/Metro_Interstate_Traffic_Volume.csv`.

`traffic.db` was created by importing `data/raw/Metro_Interstate_Traffic_Volume.csv` directly into SQLite using DB Browser for SQLite's **Import → Table from CSV File** feature, with the table named `traffic`.

## Running the SQL analysis

1. Open `traffic.db` in DB Browser for SQLite or another SQLite application.
2. Open the Execute SQL area.
3. Open or paste the contents of `traffic_analysis.sql`.
4. Run the statements in order.

The SQL script performs the following tasks:

1. Verifies that the `traffic` table contains 48,204 rows and the expected nine columns.
2. Calculates total and average hourly traffic volume for 2012 to 2017.
3. Calculates the absolute and percentage change between consecutive years.
4. Compares temperature and traffic for New Year's Day and Labor Day between 2015 and 2017.
5. Checks the first and last timestamp recorded for each year so that incomplete years are interpreted correctly.

The SQL results and written interpretations are presented in the `SQL Analysis` worksheet of the Excel workbook and reinforced in the final insights report.

## Excel analysis

The Excel workbook calculates the following directly from the raw dataset using Excel formulas:

- mean, median, standard deviation, variance and range of traffic volume
- Pearson correlation between temperature and traffic volume
- probability of congestion
- probability of clear weather
- probability of congestion and clear weather
- conditional probabilities for clear weather and high temperature given congestion
- independence comparison
- congestion odds ratio for clear weather versus cloudy weather

Each result includes a short interpretation explaining what the value means.

## Definitions

- Congestion: traffic volume greater than 5,500 vehicles per hour.
- Clear weather: `weather_main` equals `Clear`.
- Cloudy weather: `weather_main` equals `Clouds`.
- High temperature: temperature greater than 292 K.
- Odds ratio: congestion odds in clear weather divided by congestion odds in cloudy weather.

Annual totals must be interpreted alongside the number and date range of recorded observations because 2012, 2014 and 2015 have incomplete temporal coverage.
