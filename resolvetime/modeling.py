"""Trainable preprocessing, syllabus models, and shared prediction validation."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURE_COLUMNS, NUMERIC, SCHEMA_VERSION, UNKNOWN, make_features


def build_estimator(name):
    if name == "Median baseline":
        return DummyRegressor(strategy="median")
    models = {
        "Linear Regression": LinearRegression(),
        "k-NN Regression": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=1),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, l2_regularization=5.0,
            early_stopping=False, random_state=42,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    preprocessing = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                     max_categories=32, sparse_output=False), CATEGORICAL),
        ("numbers", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
    ])
    pipeline = Pipeline([("preprocess", preprocessing), ("selection", VarianceThreshold()),
                         ("model", models[name])])
    # Long resolution times are retained. Log scaling reduces their influence;
    # model selection and reporting still measure errors in original hours.
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


PARAMETER_GRIDS = {
    "Linear Regression": {"regressor__model__fit_intercept": [True, False]},
    "k-NN Regression": {"regressor__model__n_neighbors": [5, 15],
                        "regressor__model__weights": ["distance", "uniform"]},
    "Gradient Boosting": {"regressor__model__max_leaf_nodes": [7, 15],
                           "regressor__model__learning_rate": [0.05, 0.1]},
}


def checked_predictions(estimator, features):
    prediction = np.asarray(estimator.predict(features), dtype=float).reshape(-1)
    if len(prediction) != len(features) or not np.isfinite(prediction).all():
        raise ValueError("The model returned invalid predictions. Check the data and retrain.")
    return np.maximum(prediction, 0.0)


def nonnegative_mae(actual, prediction):
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions during model selection.")
    return mean_absolute_error(actual, np.maximum(prediction, 0.0))


def evaluate(estimator, features, actual):
    prediction = checked_predictions(estimator, features)
    return {
        "mae_hours": float(mean_absolute_error(actual, prediction)),
        "rmse_hours": float(np.sqrt(mean_squared_error(actual, prediction))),
        "median_absolute_error_hours": float(np.median(np.abs(np.asarray(actual) - prediction))),
        "r2": float(r2_score(actual, prediction)), "tickets": int(len(actual)),
    }, prediction


def predict_tickets(artifact, raw):
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("The saved model uses a different input schema. Train the model again.")
    features = make_features(raw)
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("The saved feature definitions do not match this app.")
    prediction = checked_predictions(artifact["model"], features)
    unknown = pd.Series(False, index=raw.index)
    for column, levels in artifact["category_levels"].items():
        unknown |= ~features[column].isin(levels) | features[column].eq(UNKNOWN)
    result = raw.copy()
    result["predicted_resolution_hours"] = prediction
    result["input_note"] = np.where(unknown, "Contains missing or unseen categories; interpret cautiously", "")
    return result
