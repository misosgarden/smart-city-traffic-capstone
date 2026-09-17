# Part 2 Python Data Pipeline and Application

This folder contains the Python work completed for Part 2 of the Smart City Traffic Intelligence capstone project. It includes a reproducible data-cleaning pipeline, feature engineering, traffic visualisations and a command-line application for exploring the processed dataset.

## Files and folders

- `pipeline.py`: Validates and cleans the raw traffic dataset.
- `feature_engineering.py`: Creates additional variables for analysis and modelling.
- `visualisations.py`: Generates the traffic visualisations and their written interpretations.
- `app.py`: Runs the interactive traffic analysis application.
- `pipeline.log`: Records processing stages, application commands, warnings and errors.
- `visualisation_interpretations.txt`: Contains written interpretations of the three visualisations.
- `data/processed/traffic_cleaned.csv`: Cleaned dataset produced by `pipeline.py`.
- `data/processed/traffic_features.csv`: Feature-engineered dataset produced by `feature_engineering.py`.
- `figures/`: Contains the three generated visualisations.

The original traffic dataset is stored separately at:

`data/raw/Metro_Interstate_Traffic_Volume.csv`

## Requirements

The workflow uses Python 3 with:

- pandas
- NumPy
- Matplotlib
- scikit-learn

The required packages are listed in the repository-level `requirements.txt` file. For reference, they can be installed from the repository root using:

```bash
python3 -m pip install -r requirements.txt
```

## Reproducing the workflow

The commands below assume that the terminal is open in the `part2_python` folder. They are included for reproducibility and as a reminder for my future self.

### 1. Prepare the cleaned dataset

```bash
python3 pipeline.py
```

The pipeline checks the expected schema, parses `date_time`, standardises categorical text and removes exact duplicates. It also identifies impossible temperature values and extreme rainfall values, which are replaced using month-specific medians.

The original dataset contains 48,204 rows. After removing 17 exact duplicate rows, the cleaned dataset contains 48,187 rows.

The result is saved as:

`data/processed/traffic_cleaned.csv`

### 2. Create the engineered features

```bash
python3 feature_engineering.py
```

The feature-engineering script creates:

- hour, day-of-week and month variables
- cyclical hour and month features
- a weekend indicator
- holiday, rainfall, snowfall and peak-hour indicators
- a data-driven congestion category with the labels `Low`, `Medium`, `High` and `Severe`
- standardised temperature and traffic-volume features
- one-hot encoded weather-condition variables

The congestion categories are based on traffic-volume quartiles rather than fixed thresholds.

The result is saved as:

`data/processed/traffic_features.csv`

### 3. Generate the visualisations

```bash
python3 visualisations.py
```

This produces:

- `figures/hourly_traffic_weekday_weekend.png`
- `figures/traffic_volume_distribution.png`
- `figures/average_traffic_by_weather.png`
- `visualisation_interpretations.txt`

The figures compare hourly weekday and weekend traffic, show the distribution of hourly traffic volume and compare average traffic across recorded weather conditions.

### 4. Explore the results using the application

```bash
python3 app.py
```

The application provides four analytical queries through a simple numbered menu:

1. Look up traffic by date
2. Show the busiest traffic periods
3. Compare weekday and weekend traffic
4. Show lower-traffic travel hours

The application checks user inputs and displays clear error messages for invalid menu selections, dates and numerical values. Application commands and supplied inputs are recorded in `pipeline.log`.

## Main findings

Traffic follows clear time-based patterns. Weekday traffic has pronounced morning and late-afternoon peaks, while weekend traffic increases more gradually and remains lower during the main commuting periods.

The traffic-volume distribution reflects a mixture of low overnight traffic and higher daytime or commuting traffic. Average traffic also differs across the recorded weather categories.

These descriptive comparisons should not be interpreted as proof that weather directly causes changes in traffic. Time of day, commuting behaviour and differences in the number of observations within each weather category may also influence the results.

## Logging and error handling

The scripts check for problems such as missing files, unreadable data, missing required columns and invalid values. Important processing stages, warnings, errors, generated outputs and application commands are recorded in `pipeline.log`.

This log provides a record of how the processed datasets and analytical outputs were produced.
