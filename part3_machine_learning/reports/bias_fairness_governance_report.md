# Bias, Fairness, Governance and Sustainability Report

## 1. Purpose and scope

This report evaluates the responsible use of the machine-learning components developed for the Smart City Traffic Intelligence project. It considers data coverage, potential bias, fairness limitations, the accident-risk proxy, governance requirements and environmental sustainability.

The project uses the Metro Interstate Traffic Volume dataset to analyse and predict hourly traffic volume on westbound Interstate 94. It also includes clustering, deep learning, model explainability, travel-timing recommendations, deployment simulation and drift monitoring.

The system is an educational proof of concept. It is not a live traffic-control system and should not be used as the sole basis for operational or safety-critical decisions.

## 2. Data coverage and representation

The dataset represents traffic recorded at one westbound traffic station on Interstate 94 in the Minneapolis and Saint Paul area. It does not represent an entire city, all road directions, rural roads, public transport or road networks in other geographic locations.

Traffic behaviour may differ according to road design, population density, local commuting patterns, public transport availability and regional weather. Consequently, the models should not be assumed to generalise to other roads or cities without external validation.

The dataset also has uneven temporal coverage. Some years are incomplete, and observations are not distributed equally across all weather conditions, holidays and times. Rare conditions such as squalls have very few observations. Model performance during uncommon conditions is therefore less certain than performance during frequently observed conditions.

The cleaned dataset contains 48,187 observations after 17 exact duplicates were removed. Invalid temperature and extreme rainfall values were handled during preprocessing. Although these steps improve data quality, imputation can reduce the visibility of genuine extreme conditions and introduces assumptions into the dataset.

## 3. Accident-risk proxy

The supplied dataset does not contain actual accident records. The classification task therefore uses a documented proxy combining high or severe congestion with severe or low-visibility weather conditions.

This proxy identifies potentially elevated traffic-risk conditions. It does not predict whether an accident occurred and must not be described as an accident-prediction model.

The classification model’s very high ROC AUC of approximately 0.997 should be interpreted cautiously because the proxy target was partly constructed from weather conditions that were also supplied to the model as predictor features. This creates a leakage-like relationship in which the model can partially reconstruct the rule used to define the target, so the reported performance does not demonstrate an equivalent ability to predict real accidents.

Congestion and adverse weather may be associated with hazardous driving conditions, but accidents can also be affected by road design, vehicle condition, driver behaviour, construction, visibility, enforcement and unexpected incidents. These factors are not fully represented in the dataset.

A false positive could cause unnecessary warnings or interventions. A false negative could result in elevated-risk conditions not being flagged. Any real safety application would require verified accident data, additional explanatory variables and evaluation with transport and road-safety specialists.

## 4. Fairness considerations

The dataset does not include personal demographic characteristics such as age, sex, disability, ethnicity or socioeconomic status. Therefore, conventional demographic fairness measures cannot be calculated.

The absence of demographic variables does not mean that the system is automatically fair. Traffic-management decisions can still affect communities differently. For example, rerouting or signal changes may transfer congestion, noise, emissions or road-safety risks from one neighbourhood to another.

Recommendations may also be less useful for people who cannot change their travel time, including shift workers, carers and individuals dependent on fixed public-transport schedules. Travel-timing recommendations should therefore be presented as optional information rather than instructions that assume all users have equal flexibility.

Model errors may be distributed unevenly across conditions and time periods. Predictions may be less reliable during rare weather events, holidays and periods with limited historical coverage because these observations contribute less evidence during training. Error may also differ between peak and off-peak periods, weekdays and weekends, and earlier and later years if traffic behaviour changes over time. Overall performance metrics may conceal these subgroup differences.

Future evaluation should examine model errors across:

- weekday and weekend observations;
- peak and off-peak periods;
- common and rare weather conditions;
- holidays and non-holidays;
- seasons and years;
- low, medium, high and severe congestion conditions.

## 5. Model performance and explainability

The project compares multiple machine-learning approaches rather than relying on a single model. The selected Random Forest traffic model achieved a lower mean absolute error and a higher R-squared value than the neural-network baseline on the same chronologically held-out test data.

SHAP analysis was used to explain the contribution of model features. Explainability can support review and debugging, but it does not prove that a model is unbiased or causal. A feature may be influential because it is correlated with traffic patterns rather than because it directly causes them.

Performance metrics reported for the complete test set may conceal weaker performance for rare conditions. Subgroup performance should be examined before any operational use.

## 6. Drift monitoring

The deployment workflow includes feature-distribution monitoring using the Population Stability Index. PSI values below 0.10 indicate no material drift, values from 0.10 to below 0.20 indicate moderate change, and values of 0.20 or above indicate material drift.

The monitoring demonstration evaluates the distribution of the traffic-volume target alongside the model inputs of temperature, rainfall occurrence, cloud coverage and hour. Rainfall occurrence is defined as whether `rain_1h` is greater than zero. A normal simulated sample produces `PASS / Normal`, while a deliberately shifted test sample produces `ALERT / Requires investigation`. 

During development, initial monitoring of raw hourly rainfall produced a PSI of 0.0009 for the deliberately shifted scenario, despite a substantial increase in average rainfall. Examination of the bin-level counts showed that approximately 92.9% of the reference observations recorded no rainfall. Quantile-based binning therefore placed most dry and light-rain observations within the same dominant bin. The monitoring method was revised to assess rainfall occurrence, defined as `rain_1h > 0`, rather than raw rainfall depth. This produced a PSI of 10.9336 for the shifted scenario and correctly triggered an alert. A production system should consider monitoring both rainfall occurrence and nonzero rainfall intensity.

An alert should trigger investigation rather than automatic retraining. The response should include checking data quality, identifying changes in traffic or weather patterns, evaluating current model error and determining whether retraining is justified.

## 7. Governance and human oversight

A production implementation should have a named model owner and a documented review process. Model versions, training data, parameters, evaluation metrics and approval decisions should be recorded.

The following controls are recommended:

1. Validate incoming data and reject invalid or incomplete inputs.
2. Monitor feature distributions and prediction errors.
3. Review model performance at scheduled intervals.
4. Investigate alerts before retraining or replacing the model.
5. Require human approval before operational traffic interventions.
6. Retain model versions and an audit trail of changes.
7. Document the intended use, limitations and prohibited uses.
8. Provide a method for users and affected stakeholders to raise concerns.
9. Reassess geographic validity before applying the model elsewhere.
10. Consult transport and road-safety specialists for safety-related applications.

The FastAPI application is a deployment simulation. It should not be publicly exposed without authentication, access controls, secure logging, rate limiting and infrastructure monitoring.

## 8. Privacy and security

The current dataset contains aggregated traffic and weather observations rather than direct personal identifiers. This reduces personal privacy risk.

A future system could introduce privacy concerns if it incorporated vehicle identifiers, number plates, mobile-location data or individual travel histories. Such data would require data minimisation, access controls, retention limits, security testing and compliance with applicable data-protection requirements.

Saved models should be loaded only from trusted sources because serialised model files can present security risks if they have been modified by an untrusted party.

## 9. Sustainability

Training and repeatedly tuning complex models consumes computing resources and energy. The project therefore compares model performance with operational complexity.

The Random Forest model performed better than the neural-network baseline for the traffic-volume prediction task. Using the better-performing tree-based model for this deployment simulation avoids unnecessary neural-network retraining and reduces computational demand within this project.

The following sustainability measures are recommended:

- reuse validated model artifacts when retraining is unnecessary;
- retrain only when monitoring and performance evidence justify it;
- prefer the simplest model that meets performance requirements;
- limit unnecessary hyperparameter searches;
- record experiments to avoid duplicating unsuccessful work;
- schedule computationally intensive work efficiently;
- remove redundant artifacts while retaining required audit evidence.

Sustainability should be balanced with accuracy, safety, transparency and service reliability.

## 10. Limitations and conclusion

The project demonstrates a reproducible machine-learning workflow, but it remains limited by a single-location dataset, incomplete temporal coverage, rare weather categories and the absence of verified accident data.

The accident-risk classification is a proxy for elevated-risk conditions and must not be interpreted as actual accident prediction. Traffic recommendations and predictions should support human decision-making rather than replace professional judgement.

Responsible future use would require broader and more recent data, external validation, subgroup performance analysis, security controls, continued drift monitoring and formal human oversight.