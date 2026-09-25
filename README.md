# Smart City Traffic Intelligence
NUS AI, ML and Data Science Capstone Project

This repository contains an end-to-end analysis of the Metro Interstate Traffic Volume dataset. The project progresses from historical data analysis to a reproducible Python pipeline, machine-learning models, explainability, experiment tracking, deployment simulation, drift monitoring and practical travel-timing recommendations.

The dataset contains hourly westbound I-94 traffic observations near Minneapolis and Saint Paul together with weather and US federal holiday information. Because it represents one monitored road corridor, findings should not be generalised to an entire city or used as the sole basis for safety-critical decisions.

## Project parts

| Part | Purpose | Main outputs |
| --- | --- | --- |
| `part1_data_analytics/` | SQL, descriptive statistics, probability and dashboard analysis | SQLite database, SQL queries, Excel analysis and insights report |
| `part2_python/` | Reproducible data cleaning, feature engineering, visualisation and command-line analysis | Python scripts, processed datasets, figures, interpretations and pipeline log |
| `part3_machine_learning/` | Predictive modelling, unsupervised learning, explainability, MLflow, recommendations, deployment and monitoring | Notebooks, saved models, reports, API, monitoring outputs and final report |

## Important scope statement

No real accident dataset was supplied. The classification task therefore uses a documented proxy label. A record is labelled `high_risk` when High or Severe congestion occurs together with severe or low-visibility weather. This label demonstrates a classification workflow only. It does not indicate that an accident occurred and must not be presented as a real accident-prediction model.

## Main findings

- Recurring hourly and weekly travel patterns were stronger indicators of traffic demand than weather alone.
- In the supervised-regression comparison, the Random Forest Regressor achieved the strongest performance, with MAE 244.04 vehicles per hour and R-squared 0.9538.
- The proxy-risk Random Forest achieved ROC AUC 0.9970, but this result must be interpreted cautiously because the target is rule-based and partly constructed from weather variables also available to the model.
- K-means produced five overlapping traffic-condition groups. The three largest groups represented low-demand normal weather, high-demand normal weather and adverse-weather moderate demand.
- Association rules linked weekend nights with low congestion at 88.61% confidence and lift 3.54.
- SHAP analysis of a separate comparable Random Forest identified cyclical hour variables as the strongest contributors to predicted volume.
- The travel-timing system recommends transparent lower-traffic windows from historical averages and falls back to broader day-type evidence when a weather-specific sample has fewer than 30 observations.
- The deployment simulation returned successful health and prediction responses, while PSI monitoring correctly produced `PASS / Normal` for a normal sample and `ALERT / Requires investigation` for a deliberately shifted sample.

## Model naming

Three Random Forest artifacts serve different purposes:

| Artifact | Role |
| --- | --- |
| `best_traffic_regressor.joblib` | Random Forest selected as the strongest traffic-volume model in the supervised-learning comparison |
| `shap_random_forest.joblib` | Separate comparable Random Forest trained on the chronological split for TreeSHAP analysis |
| `random_forest_shap_v1` | Version label for the SHAP Random Forest recorded in MLflow and served by the FastAPI deployment simulation |

The MLflow tag `selected_candidate` applies only to the comparison between the neural network and the Random Forest used for SHAP. It does not replace the supervised Random Forest as the selected model from Task 1.

## Setup

Python 3.13 is required. All saved model artifacts in this repository were created with scikit-learn 1.9.1 under Python 3.13.7; using an older Python/scikit-learn combination to load them will raise a version-incompatibility error.

```bash
git clone https://github.com/misosgarden/smart-city-traffic-capstone.git
cd smart-city-traffic-capstone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Run order

### Part 1

Open the supplied Excel workbook and Power BI dashboard for the analytical outputs. The SQLite database can be opened with a SQLite client and queried using `part1_data_analytics/traffic_analysis.sql`.

### Part 2

Run the pipeline in this order from the repository root:

```bash
python part2_python/pipeline.py
python part2_python/feature_engineering.py
python part2_python/visualisations.py
python part2_python/app.py
```

### Part 3

Run the notebooks in numerical order:

1. `01_supervised_models.ipynb`
2. `02_unsupervised_learning.ipynb`
3. `03_deep_learning_explainability.ipynb`
4. `04_mlflow_tracking.ipynb`
5. `05_recommendation_system.ipynb`

The notebooks reuse `part2_python/data/processed/traffic_features.csv`. Detailed Part 3 commands are provided in `part3_machine_learning/README.md`.

## Logging

Project modules use `logging.getLogger(__name__)`. Logs include a timestamp, level, module name and message.

- `DEBUG`: intermediate calculations used for troubleshooting
- `INFO`: expected milestones, including data loading, model completion and saved artifacts
- `WARNING`: recoverable issues, including imputation, fallback logic and monitoring alerts
- `ERROR`: failures that prevent the requested operation from completing

Part 2 writes its execution trail to `part2_python/pipeline.log`. Part 3 scripts append operational messages to `part3_machine_learning/part3.log`. Printed output is reserved for results intended for the end user.

## Responsible use

This repository is an educational proof of concept. The models use historical observations from one westbound road station, include incomplete temporal coverage and have limited evidence for rare weather categories. Predictions, proxy-risk classifications and travel recommendations should support human review rather than automatically initiate traffic-control or safety interventions.

See `part3_machine_learning/reports/bias_fairness_governance_report.md` for the full responsible-AI assessment.
