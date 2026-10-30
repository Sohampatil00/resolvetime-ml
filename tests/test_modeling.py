"""Check inference uses the same feature schema as a persisted trained pipeline."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from resolvetime.data import CATEGORICAL, FEATURE_COLUMNS, SCHEMA_VERSION, make_features
from resolvetime.modeling import build_estimator, predict_tickets


class PredictionTests(unittest.TestCase):
    def artifact(self):
        raw = pd.DataFrame([
            {"opened_at": f"2016-02-{day:02} 09:00", "category": "Category 1",
             "subcategory": "Subcategory 1", "contact_type": "Phone",
             "impact": "2 - Medium", "urgency": "2 - Medium"}
            for day in range(1, 21)
        ])
        features = make_features(raw)
        model = build_estimator("Linear Regression").fit(features, np.arange(1, 21, dtype=float))
        return {"schema_version": SCHEMA_VERSION, "feature_columns": FEATURE_COLUMNS,
                "model": model, "category_levels": {
                    column: features[column].unique().tolist() for column in CATEGORICAL[:-1]
                }}, raw

    def test_saved_pipeline_handles_unseen_categories_and_ignores_outcomes(self):
        artifact, raw = self.artifact()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "model.joblib"
            joblib.dump(artifact, file)
            loaded = joblib.load(file)
        raw = raw.head(1).copy()
        raw["category"] = "New category"
        prediction = predict_tickets(loaded, raw)
        raw["resolved_at"] = "2099-01-01"
        second = predict_tickets(loaded, raw)
        np.testing.assert_allclose(prediction.predicted_resolution_hours, second.predicted_resolution_hours)
        self.assertGreaterEqual(prediction.iloc[0].predicted_resolution_hours, 0)
        self.assertIn("unseen", prediction.iloc[0].input_note)

    def test_rejects_incompatible_artifact(self):
        artifact, raw = self.artifact()
        artifact["schema_version"] = "old"
        with self.assertRaisesRegex(ValueError, "schema"):
            predict_tickets(artifact, raw)


if __name__ == "__main__":
    unittest.main()
