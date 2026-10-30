"""Regression checks for event snapshots, target construction, and leakage."""

import unittest

import numpy as np
import pandas as pd

from resolvetime.data import (
    FEATURE_COLUMNS, UNKNOWN, make_features, parse_times, prepare_incidents,
    rolling_folds, temporal_holdouts,
)


def event(number="INC1", **changes):
    row = {"number": number, "sys_mod_count": "0", "sys_updated_at": "01/02/2016 09:00",
           "opened_at": "01/02/2016 09:00", "resolved_at": "?", "category": "Category 1",
           "subcategory": "?", "contact_type": "Phone", "impact": "2 - Medium", "urgency": "2 - Medium"}
    row.update(changes)
    return row


def timeline():
    rows = []
    for day in range(40):
        for number in range(4):
            opening = pd.Timestamp("2016-01-01") + pd.Timedelta(days=day, hours=number)
            rows.append({"number": f"INC{day:03}{number}", "opened_at": opening,
                         "resolved_at": opening + pd.Timedelta(hours=1), "resolution_hours": 1.0})
    return pd.DataFrame(rows)


class IncidentPreparationTests(unittest.TestCase):
    def test_initial_missing_field_is_not_filled_from_later_event(self):
        events = pd.DataFrame([
            event(),
            event(sys_mod_count="1", sys_updated_at="01/02/2016 11:00", subcategory="Later detail",
                  urgency="1 - High", resolved_at="01/02/2016 12:00"),
        ])
        prepared, _ = prepare_incidents(events)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.iloc[0].subcategory, UNKNOWN)
        self.assertEqual(prepared.iloc[0].urgency, "2 - Medium")
        self.assertEqual(prepared.iloc[0].resolution_hours, 3.0)

    def test_reopening_uses_latest_resolution_and_excludes_unknown_or_negative(self):
        events = pd.DataFrame([
            event(), event(sys_mod_count="1", resolved_at="01/02/2016 10:00"),
            event(sys_mod_count="2", resolved_at="02/02/2016 09:00"),
            event(number="UNRESOLVED"),
            event(number="NEGATIVE", resolved_at="31/01/2016 09:00"),
            event(number="NO_INITIAL", sys_mod_count="2", resolved_at="02/02/2016 09:00"),
        ])
        prepared, audit = prepare_incidents(events)
        self.assertEqual(prepared.number.tolist(), ["INC1"])
        self.assertEqual(prepared.iloc[0].resolution_hours, 24.0)
        self.assertEqual(audit["unknown_resolution"], 1)
        self.assertEqual(audit["negative_or_invalid_durations"], 1)
        self.assertEqual(audit["incidents_without_initial_snapshot"], 1)

    def test_date_conventions_and_invalid_prediction_input(self):
        parsed = parse_times(pd.Series(["01/02/2016 09:00", "2016-02-01 09:00:00"]))
        self.assertEqual(parsed.iloc[0], parsed.iloc[1])
        with self.assertRaisesRegex(ValueError, "opened_at"):
            make_features(pd.DataFrame([event(opened_at="not a date")]))

    def test_future_columns_cannot_change_features(self):
        raw = pd.DataFrame([event()])
        original = make_features(raw)
        raw["resolved_at"] = "2099-01-01"
        raw["made_sla"] = "true"
        raw["close_code"] = "Known answer"
        raw["number"] = "DIFFERENT"
        pd.testing.assert_frame_equal(original, make_features(raw))
        self.assertEqual(original.columns.tolist(), FEATURE_COLUMNS)


class TemporalSplitTests(unittest.TestCase):
    def test_labels_must_be_known_at_training_and_validation_cutoffs(self):
        frame = timeline()
        # Old ticket whose outcome is learned in the future.
        frame.loc[0, "resolved_at"] = pd.Timestamp("2016-03-01")
        frame.loc[0, "resolution_hours"] = 1440
        splits, info = temporal_holdouts(frame)
        self.assertNotIn(frame.iloc[0].number, splits["train"].number.tolist())
        self.assertNotIn(frame.iloc[0].number, splits["development"].number.tolist())
        self.assertTrue((splits["train"].resolved_at < pd.Timestamp(info["validation_start"])).all())
        self.assertTrue((splits["development"].resolved_at < pd.Timestamp(info["test_start"])).all())
        groups = [set(splits[name].number) for name in ("train", "validation", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_cv_is_expanding_and_does_not_use_future_labels(self):
        frame = timeline()
        frame.loc[0, "resolved_at"] = pd.Timestamp("2016-03-01")
        folds = rolling_folds(frame)
        self.assertGreaterEqual(len(folds), 2)
        for train, validation in folds:
            self.assertNotIn(0, train)
            self.assertEqual(len(np.intersect1d(train, validation)), 0)
            cutoff = frame.iloc[validation].opened_at.min().normalize()
            self.assertTrue((frame.iloc[train].opened_at < cutoff).all())
            self.assertTrue((frame.iloc[train].resolved_at < cutoff).all())


if __name__ == "__main__":
    unittest.main()
