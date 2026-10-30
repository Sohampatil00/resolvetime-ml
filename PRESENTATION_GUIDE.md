# ResolveTime: what to show and what to say

This guide matches the implementation and the saved run in `results/metrics.json` with training timestamp `2026-10-30T18:02:52.654311+00:00`. If you retrain, use the new numbers under **Model results**. The UI reads evaluation stored with the trained model.

## 1. Introduce the problem — 20 seconds

Show the main ResolveTime screen. Say:

> Our project estimates how many hours an IT support ticket may take to resolve. This is supervised learning because historical tickets have known resolution times. It is a regression problem because the output is a continuous number of hours.

Do not describe it as classification or claim a percentage accuracy.

## 2. Explain the dataset — 30 seconds

Open **How the ML works** and point to the dataset and target. Say:

> We use a real company's ServiceNow records from the UCI Incident Management Process Enriched Event Log. It has 141,712 event rows for 24,918 incidents. Several updates belong to one ticket, so we keep one initial snapshot per ticket. After cleaning, this run uses 23,357 resolved initial tickets.

The cleaning report excludes 5 incidents without an initial snapshot and 1,556 initial tickets without a known resolution. Missing outcomes are not labeled zero hours.

Say:

> The features, called X, are category, subcategory, impact, urgency, contact method, and time information. The target, called y, is final observed resolution time minus opening time, converted to hours.

Categories are anonymous codes. Do not claim that `Category 55` means “network” or another real problem type.

## 3. Show one prediction — 45 seconds

Close **How the ML works**. Enter this historical example from the raw dataset:

| Field | Value |
| --- | --- |
| Category | Category 55 |
| Subcategory | Subcategory 170 |
| Impact | 2 - Medium |
| Urgency | 2 - Medium |
| Additional ticket details → Reported through | Phone |
| Additional ticket details → Reported date | 29 February 2016 |
| Additional ticket details → Reported time | 01:16 |

Click **Predict resolution time**. Read the actual result from your saved model; this guide does not invent prediction values. Say:

> During inference, the saved preprocessing pipeline converts the ticket into numerical features. The saved model predicts the duration in hours. This example demonstrates the interface; it is a historical training input, so it is not an independent accuracy test.

Change urgency to another available level and predict again. Say:

> This shows how predictions respond to changed features. Higher urgency does not guarantee a faster estimate, and the prediction may stay the same. The model learns associations, not cause and effect.

The actual resolution timestamp is never supplied as a predictor. The estimate includes nights and weekends and is not an SLA promise.

## 4. Explain the pipeline — 60 seconds

Open **How the ML works** and follow these stages:

| Stage | Implementation | Simple explanation |
| --- | --- | --- |
| Cleaning | Parse dates, select initial snapshots, exclude unknown outcomes and invalid durations | Create valid examples without mixing later ticket updates into the inputs. |
| Missing values | Represent missing categories as `__unknown__` | Keep unknown information explicitly unknown. |
| Encoding | One-hot encoding with infrequent-category grouping | Convert category labels into indicator columns. Category 55 is not 55 times Category 1. |
| Scaling | StandardScaler on engineered numeric columns | Use comparable scales, particularly important for k-NN distance. |
| Feature engineering | Cyclic hour/weekday/month, weekend, impact–urgency combination | Derive useful predictors. Cyclic encoding keeps 23:00 and 00:00 close together. |
| Feature selection | Allowlist opening-time fields; remove constant encoded columns | Exclude future information and features with no variation. |
| Target transformation | `log1p(hours)` for learned models; invert predictions | Reduce the influence of very long durations while still evaluating in hours. |
| Hyperparameter tuning | GridSearchCV with expanding temporal folds | Compare model settings using earlier tickets to predict later ones. |

The numerical median imputer is a guard: valid opening timestamps already produce complete numerical time features. Do not claim we artificially inserted missing values.

## 5. Explain the syllabus algorithms — 40 seconds

| Algorithm | What to say | Syllabus |
| --- | --- | --- |
| Multiple Linear Regression | Learns a linear relationship between multiple features and the transformed target. | Assignment 2 |
| k-NN regression | Finds nearby tickets in the processed feature space and combines their transformed durations. | Assignment 5 |
| Gradient Boosting | Builds an ensemble of trees sequentially, with later trees improving the prediction. | Assignment 9 |

We use scikit-learn's histogram-based Gradient Boosting implementation. It is a boosting ensemble. Reinforcement learning is not used.

Since learned models operate on the log-transformed target, selected k-NN predictions invert the average of transformed neighbor durations, rather than simply averaging raw hours.

The **median baseline** always predicts the median training duration. It is a reference for assessing whether learned models improve on a simple guess.

## 6. Show the actual results — 60 seconds

Open **Model results**. Show the validation table. Say:

> We tune settings using three expanding cross-validation folds inside training data. We select the model with the lowest mean absolute error on a separate validation set. We refit it using eligible pre-test tickets and evaluate on a later test set.

Saved validation results:

| Model | Validation MAE, hours |
| --- | ---: |
| k-NN Regression | 126.36 |
| Median baseline | 126.38 |
| Gradient Boosting | 137.19 |
| Linear Regression | 2,566.95 |

The selected configuration is **15 neighbors with uniform weights**. Its validation advantage over the baseline is only about 0.02 hours. Do not describe this as a strong improvement. Linear Regression performed especially poorly on this later validation population.

Point to the test metrics:

| Metric | Saved result | Meaning |
| --- | ---: | --- |
| MAE | 147.69 hours | Average absolute prediction error. Lower is better. |
| RMSE | 269.76 hours | Penalizes large misses more strongly. Lower is better. |
| R² | -0.260 | Worse squared-error performance than predicting the test target's mean for every test ticket. Higher is better. |

The separate median baseline has test MAE **144.95 hours**, lower than selected k-NN. R²'s mathematical reference is the test mean; our explicitly trained baseline is the training median. They are different references.

Say honestly:

> k-NN won validation but did not beat the median baseline on test MAE. This version demonstrates the complete ML workflow, but it is not yet an accurate operational predictor. Negative R² and large errors show that improvement is needed.

MAE is not a per-ticket prediction, confidence interval, or guaranteed error bound. Do not claim an accuracy percentage, operational reliability, or that Gradient Boosting won.

## 7. Explain the split correctly — 20 seconds

| Population | Tickets |
| --- | ---: |
| Initial training | 22,968 |
| Validation | 61 |
| Test | 154 |
| Final pre-test refit | 23,176 |

Boundaries are at 60% and 80% of **unique opening days**, not ticket rows. Arrivals are uneven and outcomes unknown at cutoffs are excluded, making the groups very unequal. Validation starts 15 September 2016; test starts 22 December 2016. The small validation/test populations limit confidence in the comparisons.

Final refit is not a fourth disjoint holdout. It reuses eligible training/validation tickets and earlier tickets whose outcomes became known by the test cutoff. Do not add its count to the other counts.

Say:

> Chronological splitting imitates predicting future tickets. We prevent data leakage by keeping each incident together, excluding future resolution details from predictors, and fitting preprocessing only on each fold's training portion.

## 8. State limitations and future work — 20 seconds

Say:

> Records come from one historical company, category meanings are hidden, and unresolved tickets are excluded. Next steps are to investigate the uneven split and poor generalization, collect richer opening-time information, and validate on another organization. Improvements must be measured on an independent future holdout.

Streamlit is the local interface. Python, pandas, NumPy, and scikit-learn implement the ML workflow; joblib saves the fitted pipeline. GitHub publication has not been performed automatically.

## Quick viva answers

| Question | Answer |
| --- | --- |
| Why regression? | The target is a continuous duration in hours. |
| Why supervised learning? | Historical examples have known targets. |
| What is preprocessing? | Cleaning and converting data before fitting a model. |
| Why one-hot encoding? | Category numbers are labels, not ordered quantities. |
| Why scaling? | k-NN uses distances, which depend on feature scales. |
| What is a hyperparameter? | A setting chosen before fitting, such as neighbor count. |
| What is cross-validation? | Repeated training/validation comparisons; our folds advance through time. |
| What is an ensemble? | Multiple models combined into a prediction; Gradient Boosting combines trees. |
| What is data leakage? | Using information unavailable at prediction time, or allowing evaluation data to influence fitting. |
| Training versus inference? | Training learns from examples; inference applies the fitted pipeline. |
| Why not accuracy or F1? | Those evaluate classes; our output is a number of hours. |
| Is R² percentage accuracy? | No. It measures squared-error performance relative to a constant-mean reference and can be negative. |
| Can urgency guarantee the result? | No. Predictive associations are not causal guarantees. |
| Why are category names missing? | The publisher anonymized them; we do not invent meanings. |
| Which model is best? | k-NN won validation narrowly; the median baseline had lower test MAE. |

## Code to show only if asked

1. `resolvetime/data.py`: `prepare_incidents`, `make_features`, `temporal_holdouts`, `rolling_folds`.
2. `resolvetime/modeling.py`: `build_estimator`, `PARAMETER_GRIDS`, `evaluate`, `predict_tickets`.
3. `train.py`: tuning, validation selection, final refit, test evaluation.
4. `app.py`: load the model and predict one ticket.

In the main demonstration, show the input, prediction, model comparison, and actual limitations before explaining code.
