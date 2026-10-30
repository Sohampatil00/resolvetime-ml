"""Convert event logs into one opening snapshot per incident, without future inputs."""

import numpy as np
import pandas as pd

SCHEMA_VERSION = "opening_snapshot_v1"
UNKNOWN = "__unknown__"
INPUT_COLUMNS = ["opened_at", "category", "subcategory", "contact_type", "impact", "urgency"]
CATEGORICAL = ["category", "subcategory", "contact_type", "impact", "urgency", "impact_urgency"]
NUMERIC = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos", "weekend"]
FEATURE_COLUMNS = CATEGORICAL + NUMERIC


def clean_strings(series):
    result = series.astype("string").str.strip()
    missing = result.isna() | result.str.lower().isin(["", "?", "nan", "none", "null", "nat"])
    return result.mask(missing, UNKNOWN).astype(str)


def parse_times(series):
    """UCI dates use day/month/year; ISO timestamps are also accepted."""
    text = series.astype("string").str.strip()
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for mask, dayfirst in ((iso, False), (~iso, True)):
        if not mask.any():
            continue
        parsed = pd.to_datetime(text.loc[mask], format="mixed", dayfirst=dayfirst, errors="coerce")
        # This dataset does not document a timezone. Reject zoned/mixed input
        # instead of silently treating one timezone as another.
        if not pd.api.types.is_datetime64_any_dtype(parsed.dtype) or isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise ValueError("Use timezone-free dates, for example 2026-10-30 09:00:00.")
        result.loc[mask] = parsed.astype("datetime64[ns]")
    return result


def require_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))


def make_features(frame):
    """Explicit allowlist: closing details and identifiers cannot enter a model."""
    require_columns(frame, INPUT_COLUMNS)
    opened = parse_times(frame["opened_at"])
    invalid = opened.isna()
    if invalid.any():
        positions = (np.flatnonzero(invalid.to_numpy())[:5] + 1).tolist()
        raise ValueError(f"Missing or invalid opened_at in data row(s): {positions}.")
    result = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL[:-1]:
        result[column] = clean_strings(frame[column])
    result["impact_urgency"] = result["impact"] + "|" + result["urgency"]
    hour = opened.dt.hour + opened.dt.minute / 60
    for name, values, period in (("hour", hour, 24), ("weekday", opened.dt.dayofweek, 7),
                                 ("month", opened.dt.month - 1, 12)):
        result[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        result[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    result["weekend"] = (opened.dt.dayofweek >= 5).astype(float)
    return result[FEATURE_COLUMNS]


def prepare_incidents(events):
    """Use only genuine initial snapshots (sys_mod_count == 0).

    Outcomes are the latest valid resolved_at per incident. Reopened incidents
    therefore use final observed resolution, not their first temporary fix.
    Unknown outcomes are excluded and counted, never filled with zero hours.
    """
    require_columns(events, ["number", "sys_mod_count", "sys_updated_at", "resolved_at", *INPUT_COLUMNS])
    log = events.copy()
    log["number"] = clean_strings(log["number"])
    log = log.loc[log["number"] != UNKNOWN].copy()
    log["_order"] = np.arange(len(log))
    log["_opened"] = parse_times(log["opened_at"])
    log["_updated"] = parse_times(log["sys_updated_at"])
    log["_resolved"] = parse_times(log["resolved_at"])
    outcomes = log.groupby("number")["_resolved"].max()
    initial = log.loc[pd.to_numeric(log["sys_mod_count"], errors="coerce").eq(0)].copy()
    initial = initial.sort_values(["_updated", "_order"], kind="stable", na_position="last")
    # Do not use groupby.first(): it can fill an initial missing field from a
    # later row and accidentally expose information learned after opening.
    initial = initial.drop_duplicates("number", keep="first").copy()
    initial["resolved_at"] = initial["number"].map(outcomes)
    initial["opened_at"] = initial["_opened"]
    initial["resolution_hours"] = (
        initial["resolved_at"] - initial["opened_at"]
    ).dt.total_seconds() / 3600
    valid_opening = initial["opened_at"].notna()
    known_outcome = initial["resolved_at"].notna()
    valid_duration = initial["resolution_hours"].ge(0) & np.isfinite(initial["resolution_hours"])
    valid = valid_opening & known_outcome & valid_duration
    audit = {
        "event_rows": int(len(events)),
        "unique_incidents": int(log["number"].nunique()),
        "incidents_without_initial_snapshot": int(log["number"].nunique() - len(initial)),
        "initial_snapshots": int(len(initial)),
        "invalid_opening_dates": int((~valid_opening).sum()),
        "unknown_resolution": int((~known_outcome).sum()),
        "negative_or_invalid_durations": int((valid_opening & known_outcome & ~valid_duration).sum()),
        "usable_resolved_incidents": int(valid.sum()),
        "excluded_rows_without_incident_id": int((clean_strings(events["number"]) == UNKNOWN).sum()),
    }
    columns = ["number", *INPUT_COLUMNS, "resolved_at", "resolution_hours"]
    prepared = initial.loc[valid, columns].sort_values(["opened_at", "number"]).reset_index(drop=True)
    for column in INPUT_COLUMNS[1:]:
        prepared[column] = clean_strings(prepared[column])
    return prepared, audit


def temporal_holdouts(frame):
    """60/20/20 by unique opening day; purge labels unavailable at each cutoff."""
    days = pd.DatetimeIndex(frame["opened_at"].dt.normalize().unique()).sort_values()
    if len(days) < 10 or len(frame) < 50:
        raise ValueError("Training needs at least 50 resolved initial tickets spanning 10 opening days.")
    validation_start = days[int(len(days) * 0.6)]
    test_start = days[int(len(days) * 0.8)]
    train = frame.loc[(frame.opened_at < validation_start) & (frame.resolved_at < validation_start)].copy()
    validation = frame.loc[(frame.opened_at >= validation_start) & (frame.opened_at < test_start)
                           & (frame.resolved_at < test_start)].copy()
    test = frame.loc[frame.opened_at >= test_start].copy()
    development = frame.loc[(frame.opened_at < test_start) & (frame.resolved_at < test_start)].copy()
    if min(len(train), len(validation), len(test)) < 10:
        raise ValueError("Too few outcomes known at chronological cutoffs; need at least 10 per split.")
    summary = {
        "validation_start": validation_start.isoformat(), "test_start": test_start.isoformat(),
        "train": len(train), "validation": len(validation), "test": len(test),
        "final_training": len(development),
        "train_labels_unavailable_at_cutoff": int(((frame.opened_at < validation_start)
                                                  & (frame.resolved_at >= validation_start)).sum()),
        "validation_labels_unavailable_at_cutoff": int(((frame.opened_at >= validation_start)
            & (frame.opened_at < test_start) & (frame.resolved_at >= test_start)).sum()),
        "pretest_labels_unavailable_at_cutoff": int(((frame.opened_at < test_start)
                                                   & (frame.resolved_at >= test_start)).sum()),
    }
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True),
            "test": test.reset_index(drop=True), "development": development.reset_index(drop=True)}, summary


def rolling_folds(frame, n_splits=3):
    """Expanding training windows with labels known before each validation block."""
    days = pd.DatetimeIndex(frame.opened_at.dt.normalize().unique()).sort_values()
    blocks = np.array_split(np.arange(len(days)), n_splits + 1)
    folds = []
    for block in blocks[1:]:
        if not len(block):
            continue
        start = days[block[0]]
        end = days[block[-1]] + pd.Timedelta(days=1)
        train = np.flatnonzero(((frame.opened_at < start) & (frame.resolved_at < start)).to_numpy())
        valid = np.flatnonzero(((frame.opened_at >= start) & (frame.opened_at < end)
                               & (frame.resolved_at < end)).to_numpy())
        if len(train) >= 20 and len(valid) >= 5:
            folds.append((train, valid))
    if len(folds) < 2:
        raise ValueError("Not enough completed incidents for two leakage-safe temporal validation folds.")
    return folds
