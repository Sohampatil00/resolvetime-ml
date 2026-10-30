"""Train syllabus models using incident-grouped, chronological evaluation."""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from resolvetime.data import (
    CATEGORICAL, FEATURE_COLUMNS, SCHEMA_VERSION, UNKNOWN,
    make_features, prepare_incidents, rolling_folds, temporal_holdouts,
)
from resolvetime.modeling import PARAMETER_GRIDS, build_estimator, evaluate, nonnegative_mae

ROOT = Path(__file__).resolve().parent


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/incident_event_log.csv")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing trained model and its results.")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Dataset not found. Run python download_data.py first, or use --data /path/to/incident_event_log.csv.")
    model_path = args.output / "models/resolution_model.joblib"
    if model_path.exists() and not args.overwrite:
        parser.error("A model already exists. Pass --overwrite to retrain and replace it.")
    started = time.perf_counter()
    print("Reading event log and preparing opening-time snapshots...", flush=True)
    events = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    prepared, audit = prepare_incidents(events)
    splits, split_summary = temporal_holdouts(prepared)
    train, validation, test, development = (splits[name] for name in ("train", "validation", "test", "development"))
    folds = rolling_folds(train)
    print(f"Usable resolved initial tickets: {len(prepared):,}", flush=True)
    print(f"Chronological splits: train={len(train):,}, validation={len(validation):,}, test={len(test):,}", flush=True)
    x_train, x_validation = make_features(train), make_features(validation)
    y_train, y_validation = train.resolution_hours, validation.resolution_hours
    scoring = make_scorer(nonnegative_mae, greater_is_better=False)
    candidates, validation_rows = {}, []

    baseline = build_estimator("Median baseline").fit(x_train, y_train)
    baseline_metrics, _ = evaluate(baseline, x_validation, y_validation)
    candidates["Median baseline"] = baseline
    validation_rows.append({"model": "Median baseline", "cv_mae_hours": None,
                            **baseline_metrics, "best_parameters": {}})

    for name, grid in PARAMETER_GRIDS.items():
        print(f"Tuning {name} on {len(folds)} expanding temporal folds...", flush=True)
        # Keep jobs bounded for a laptop. Every fold fits its own encoder,
        # scaling, constant-feature removal, and regression model.
        search = GridSearchCV(build_estimator(name), grid, scoring=scoring, cv=folds,
                              n_jobs=1, refit=True, error_score="raise")
        search.fit(x_train, y_train)
        metrics, _ = evaluate(search.best_estimator_, x_validation, y_validation)
        candidates[name] = search.best_estimator_
        validation_rows.append({"model": name, "cv_mae_hours": float(-search.best_score_),
                                **metrics, "best_parameters": search.best_params_})
        print(f"  Validation MAE: {metrics['mae_hours']:.2f} hours", flush=True)

    # Validation alone chooses the winner. The baseline is allowed to win;
    # the interface must not claim ML improvement if measurements disagree.
    winner = min(validation_rows, key=lambda row: row["mae_hours"])["model"]
    print(f"Selected on validation MAE: {winner}. Refitting using outcomes known before test...", flush=True)
    x_development, x_test = make_features(development), make_features(test)
    final_model = clone(candidates[winner]).fit(x_development, development.resolution_hours)
    test_metrics, test_prediction = evaluate(final_model, x_test, test.resolution_hours)
    final_baseline = build_estimator("Median baseline").fit(x_development, development.resolution_hours)
    baseline_test_metrics, _ = evaluate(final_baseline, x_test, test.resolution_hours)
    missing = {column: int((x_development[column] == UNKNOWN).sum()) for column in CATEGORICAL[:-1]}
    levels = {column: sorted(x_development[column].unique().tolist()) for column in CATEGORICAL[:-1]}
    allowed_subcategories = {
        str(category): sorted(group.subcategory.unique().tolist())
        for category, group in development.groupby("category", sort=True)
    }
    stamp = datetime.now(timezone.utc).isoformat()
    source_digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    result = {
        "trained_utc": stamp, "selected_model": winner, "selection_metric": "validation MAE in hours",
        "dataset_sha256": source_digest, "dataset_audit": audit, "split": split_summary,
        "cv_folds": len(folds), "validation_comparison": validation_rows,
        "test": test_metrics, "test_median_baseline": baseline_test_metrics,
        "missing_inputs_in_final_training": missing,
        "training_seconds": round(time.perf_counter() - started, 2),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "limitations": [
            "Anonymized records from one company's historical incident process; not validated on other organizations.",
            "Only incidents with sys_mod_count=0 and a known nonnegative resolution duration are modeled.",
            "Unresolved incidents are excluded; metrics may underrepresent very long unresolved cases.",
            "Snapshots for training and validation must have outcomes known before their respective next time cutoff.",
            "Opening snapshots are the first captured system version; ticket opening times may precede system recording.",
            "Categories are anonymized codes. Their real-world problem names cannot be recovered from this dataset.",
            "Forecasts are elapsed clock hours, including nights and weekends, not business hours or SLA guarantees.",
            "The source does not document the timezone. Enter all input timestamps in a consistent timezone.",
            "The final selected model and median baseline are evaluated on test; test is not used for model selection.",
        ],
    }
    artifact = {"schema_version": SCHEMA_VERSION, "feature_columns": FEATURE_COLUMNS,
                "model": final_model, "model_name": winner, "trained_utc": stamp,
                "category_levels": levels, "subcategory_by_category": allowed_subcategories,
                "evaluation": result}
    # Create outputs only after training and evaluation succeed.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    results_dir = args.output / "results"
    save_json(results_dir / "metrics.json", result)
    pd.DataFrame(validation_rows).drop(columns=["best_parameters"]).to_csv(
        results_dir / "validation_comparison.csv", index=False)
    predictions = test[["number", "opened_at", "resolved_at", "resolution_hours"]].copy()
    predictions["predicted_resolution_hours"] = test_prediction
    predictions["absolute_error_hours"] = np.abs(predictions.resolution_hours - test_prediction)
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    category_summary = development.groupby("category").agg(
        tickets=("number", "count"), median_resolution_hours=("resolution_hours", "median")
    ).reset_index().sort_values("tickets", ascending=False)
    category_summary.to_csv(results_dir / "historical_categories.csv", index=False)
    print(f"Test MAE: {test_metrics['mae_hours']:.2f} hours; median baseline: {baseline_test_metrics['mae_hours']:.2f} hours", flush=True)
    print(f"Saved model: {model_path}\nNext: python -m streamlit run app.py --server.port 8502", flush=True)


if __name__ == "__main__":
    main()
