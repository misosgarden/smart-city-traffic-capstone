# Part 3 Machine Learning and AI

This folder develops the cleaned Part 2 data into an intelligent mobility proof of concept. It covers supervised and unsupervised learning, deep learning, SHAP explainability, MLflow experiment tracking, historical travel-timing recommendations, FastAPI deployment and PSI drift monitoring.

## Accident-data and proxy-label statement

No real accident dataset was sourced or supplied. The classification task uses a documented proxy label solely to demonstrate the required workflow. A record is labelled high risk when High or Severe congestion occurs together with severe or low-visibility weather. The output represents elevated-risk conditions under this rule, not the occurrence or probability of an actual accident.

The very high proxy-classification ROC AUC must be interpreted cautiously because weather contributes to both the proxy rule and the model inputs. Performance on this constructed target does not establish real-world accident-prediction ability.

## Project structure

```text
part3_machine_learning/
├── notebooks/
│   ├── 01_supervised_models.ipynb
│   ├── 02_unsupervised_learning.ipynb
│   ├── 03_deep_learning_explainability.ipynb
│   ├── 04_mlflow_tracking.ipynb
│   └── 05_recommendation_system.ipynb
├── models/
│   ├── best_risk_classifier.joblib
│   ├── best_traffic_regressor.joblib
│   ├── kmeans_traffic_clusters.joblib
│   ├── traffic_neural_network.keras
│   ├── neural_network_x_scaler.joblib
│   ├── neural_network_y_scaler.joblib
│   └── shap_random_forest.joblib
├── deployment/
│   └── app.py
├── monitoring/
│   └── drift_monitoring.py
├── recommendation_system/
│   └── travel_timing_recommendations.csv
├── mlflow/
│   ├── mlflow.db
│   └── artifacts/
├── reports/
│   ├── association_rules.csv
│   ├── bias_fairness_governance_report.md
│   ├── classification_model_results.csv
│   ├── classification_roc_curves.png
│   ├── deep_learning_model_comparison.csv
│   ├── drift_monitoring_results.csv
│   ├── drift_monitoring_summary.txt
│   ├── final_capstone_report.docx
│   ├── kmeans_cluster_profiles.csv
│   ├── kmeans_cluster_traffic.png
│   ├── neural_network_results.csv
│   ├── neural_network_training_loss.png
│   ├── regression_model_results.csv
│   ├── shap_feature_importance.csv
│   ├── shap_feature_summary.png
│   ├── top_association_rules.csv
│   └── traffic_cluster_assignments.csv
├── part3.log
└── README.md
```
## Environment

All saved model artifacts were created under Python 3.13.7. The `.joblib` artifacts were serialized using scikit-learn 1.9.1 and should be loaded using the same version because scikit-learn does not guarantee cross-version compatibility for serialized models. Run the notebooks and deployment API under Python 3.13 and install the dependencies from the repository-level `requirements.txt`.

## Notebook sequence

Run the notebooks in numerical order. Each notebook expects the processed dataset at:

```text
part2_python/data/processed/traffic_features.csv
```

1. **Supervised learning** compares Logistic Regression and Random Forest classification models using the documented proxy target, then compares Linear Regression and Random Forest Regression for hourly traffic volume.
2. **Unsupervised learning** applies standardised K-means clustering and mines association rules connecting time, day type, weather and congestion.
3. **Deep learning and explainability** trains a feed-forward neural network and applies TreeSHAP to a separate comparable Random Forest trained on the same 23 features and chronological split.
4. **MLflow tracking** records the neural network and the SHAP Random Forest as versioned runs in a local SQLite-backed experiment.
5. **Recommendation system** converts historical hourly patterns into plain-language travel-window guidance.

## Key model results

| Task | Model | Main result |
| --- | --- | --- |
| Proxy classification | Random Forest | Accuracy 0.9847, F1 0.9358, ROC AUC 0.9970 |
| Traffic regression | Random Forest Regressor | MAE 244.04, R-squared 0.9538 |
| Deep learning | Feed-forward neural network | MAE 493.06, R-squared 0.8845 |
| SHAP comparison model | Random Forest for SHAP | MAE 295.55, R-squared 0.9277 |

## Model naming and selection

The `best_traffic_regressor.joblib` artifact is the selected model from the supervised-regression comparison and has the strongest reported regression metrics in the project. Its result was produced with a reproducible random 80:20 split, whereas the neural-network and SHAP comparison used a chronological split. The values therefore describe their respective evaluations and should not be treated as a perfectly like-for-like cross-task benchmark.

The separate `shap_random_forest.joblib` was trained for efficient TreeSHAP explanation using the same features and chronological split as the neural network. Within the narrower MLflow comparison, this model is versioned as `random_forest_shap_v1` and tagged `selected_candidate` because it outperformed the neural network. That tag applies only within that two-model experiment.

The FastAPI mock-up serves `shap_random_forest.joblib`. This demonstrates model serving for the version recorded in MLflow and keeps the deployment evidence aligned with the explainability workflow. It does not claim that the SHAP model replaced the supervised Random Forest selected in Task 1.

## Travel-timing recommendation design

The recommendation engine uses transparent historical grouped averages rather than direct Random Forest predictions. This choice avoids presenting historical evidence as a live forecast.

- Weather-specific patterns require at least 30 observations.
- If the threshold is not met, the system falls back to the broader weekday or weekend pattern and reports the fallback.
- The default search range is 06:00 to 22:00. This manually selected practical boundary prevents very low overnight periods from dominating the recommendations. Users can change the boundaries when calling the function.
- The threshold of 30 is a pragmatic reliability rule, not an operationally validated standard.

## Run the API

From the repository root:

```bash
python -m uvicorn part3_machine_learning.deployment.app:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the interactive interface.

Example health check:

```bash
curl http://127.0.0.1:8000/health
```

Example prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date_time": "2018-09-28T17:00:00",
    "temp": 285,
    "rain_1h": 0,
    "snow_1h": 0,
    "clouds_all": 40,
    "is_holiday": false,
    "weather_main": "Clouds"
  }'
```

The tested request returned HTTP 200 and a prediction of 5,865 vehicles per hour from model version `random_forest_shap_v1`.

## Run drift monitoring

```bash
python part3_machine_learning/monitoring/drift_monitoring.py
```

The script compares a chronological historical reference with two simulated samples using the Population Stability Index. The alert threshold is `PSI >= 0.20`.

- The normal simulated sample should produce `PASS / Normal`.
- The deliberately shifted alert-test sample should produce `ALERT / Requires investigation`.
- Rainfall is monitored as occurrence, defined by `rain_1h > 0`, because raw rainfall is highly zero-inflated and quantile-based PSI may otherwise conceal a change from dry to rainy conditions.
- An alert requires investigation. It must not trigger automatic retraining without data-quality checks and current performance evaluation.

The script overwrites the current detailed results and summary files and appends operational events to `part3.log`. Re-running it is safe for this demonstration, although previous report files will be replaced by the latest reproducible run.

## MLflow

The experiment is named `smart_city_traffic_demand` and uses a local SQLite backend. To inspect it from the repository root, run:

```bash
mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow/mlflow.db
```

Then open the local URL shown in the terminal. Local storage is suitable for this demonstration but does not provide shared access, production governance or a remote model registry.

## Logging

Scripts define module-level loggers using `logging.getLogger(__name__)`. Expected milestones use `INFO`, recoverable issues and alerts use `WARNING`, and blocking failures use `ERROR`. Part 3 scripts append messages to `part3.log`; notebook outputs provide additional execution evidence.

## Limitations and responsible use

The dataset covers one westbound road station and includes incomplete years and rare weather categories. Overall metrics may hide weaker performance in particular time periods or conditions. SHAP values describe model associations and are not causal explanations. The travel guidance is based on historical averages, not live traffic data, and does not account for incidents, roadworks or special events.

The API and monitoring components are deployment simulations. Real use would require external validation, subgroup error analysis, authentication, access controls, secure logging, live data-quality checks, prediction-error monitoring, formal approval criteria and continued human oversight.
