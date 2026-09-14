-- NUS Capstone Part 1
-- SQL analysis of the Metro Interstate Traffic Volume dataset

-- 1.1 Verify that the CSV was loaded correctly.
SELECT COUNT(*) AS row_count
FROM traffic;

PRAGMA table_info(traffic);

SELECT *
FROM traffic
LIMIT 5;

-- 1.2 Compare annual traffic totals from 2012 to 2017.
-- Row counts are included because 2012, 2014 and 2015 have incomplete coverage.
WITH annual AS (
    SELECT
        CAST(strftime('%Y', date_time) AS INTEGER) AS year,
        COUNT(*) AS recorded_hours,
        SUM(traffic_volume) AS total_traffic_volume,
        AVG(traffic_volume) AS average_hourly_volume
    FROM traffic
    WHERE CAST(strftime('%Y', date_time) AS INTEGER) BETWEEN 2012 AND 2017
    GROUP BY year
),
annual_changes AS (
    SELECT
        year,
        recorded_hours,
        total_traffic_volume,
        average_hourly_volume,
        LAG(total_traffic_volume) OVER (ORDER BY year) AS previous_total
    FROM annual
)
SELECT
    year,
    recorded_hours,
    total_traffic_volume,
    ROUND(average_hourly_volume, 2) AS average_hourly_volume,
    total_traffic_volume - previous_total AS absolute_change,
    ROUND(
        100.0 * (total_traffic_volume - previous_total) / previous_total,
        2
    ) AS percentage_change
FROM annual_changes
ORDER BY year;

-- 1.3 Compare temperatures on New Years Day and Labor Day, 2015 to 2017.
-- Multiple records can share a timestamp because weather conditions are duplicated.
-- Grouping by timestamp prevents those repeated weather descriptions from overweighting a day.
WITH unique_holiday_times AS (
    SELECT DISTINCT
        date_time,
        holiday,
        temp,
        traffic_volume
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
)
SELECT
    CAST(strftime('%Y', date_time) AS INTEGER) AS year,
    holiday,
    COUNT(*) AS unique_observations,
    ROUND(AVG(temp), 2) AS average_temperature_kelvin,
    ROUND(AVG(temp - 273.15), 2) AS average_temperature_celsius,
    ROUND(AVG(traffic_volume), 2) AS average_traffic_volume
FROM unique_holiday_times
WHERE CAST(strftime('%Y', date_time) AS INTEGER) BETWEEN 2015 AND 2017
GROUP BY year, holiday
ORDER BY holiday, year;

-- Supporting coverage check for interpreting annual totals.
SELECT
    CAST(strftime('%Y', date_time) AS INTEGER) AS year,
    COUNT(*) AS recorded_hours,
    MIN(date_time) AS first_record,
    MAX(date_time) AS last_record
FROM traffic
GROUP BY year
ORDER BY year;
