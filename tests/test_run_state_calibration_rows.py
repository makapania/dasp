"""The run record stores which samples a Bayesian run excluded (review of R004).

Reloading a file clears its exclusions while the data fingerprint still matches, so
the resume gate needs the run's own exclusions and Analysis Subset to compare.
Labels are stored as type-tagged keys so any label type survives the JSON round trip.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from spectral_predict.io import rename_duplicate_ids
from spectral_predict.run_state import (
    LABEL_NORMALIZATION,
    RunMetadata,
    calibration_rows_record,
    canonical_label,
)


def _meta(**kwargs) -> dict:
    base = {
        "run_id": "r1",
        "storage_path": "x.sqlite3",
        "storage_url": "",
        "label": None,
        "dataset_fingerprint": None,
        "model_names": [],
        "n_trials_per_model": None,
        "started_iso": "2026-10-02T00:00:00",
    }
    base.update(kwargs)
    return base


def test_canonical_labels_keep_types_apart_and_survive_json():
    labels = [5, np.int64(5), 5.0, "5", ("a", 1), np.float64(2.5), None]
    keys = [canonical_label(v) for v in labels]
    assert keys[0] == keys[1]  # numpy and Python ints are the same label
    assert len({keys[0], keys[2], keys[3]}) == 3  # 5, 5.0 and "5" differ
    assert json.loads(json.dumps(keys)) == keys
    assert canonical_label(np.float64(2.5)) == canonical_label(2.5)


def test_record_stores_every_label_type():
    rows = calibration_rows_record([np.int64(5), 2.0, ("a", 1)], None)
    assert rows["active"] is None
    assert set(rows["excluded"]) == {
        canonical_label(5),
        canonical_label(2.0),
        canonical_label(("a", 1)),
    }
    assert calibration_rows_record([], ["b", "a"]) == {
        "excluded": [],
        "active": [canonical_label("a"), canonical_label("b")],
    }


def test_metadata_round_trip_keeps_calibration_rows_and_normalization():
    rows = calibration_rows_record(["S3"], None)
    meta = RunMetadata.from_dict(
        _meta(calibration_rows=rows, label_normalization=LABEL_NORMALIZATION)
    )
    again = RunMetadata.from_dict(json.loads(json.dumps(meta.to_dict())))
    assert again.calibration_rows == rows
    assert again.label_normalization == LABEL_NORMALIZATION


def test_old_and_malformed_records_mean_unchecked():
    old = RunMetadata.from_dict(_meta())
    assert old.calibration_rows is None and old.label_normalization is None
    bad = RunMetadata.from_dict(_meta(calibration_rows={"excluded": "S3"}))
    assert bad.calibration_rows is None


def test_repeated_missing_ids_get_unique_names():
    new, n, _ = rename_duplicate_ids(pd.Index(["A", np.nan, "B", np.nan, "A"]))
    assert list(new) == ["A", "nan", "B", "nan.1", "A.1"]
    assert new.is_unique and n == 3
    new, _, _ = rename_duplicate_ids(pd.Index(["nan", np.nan, "x", "x"]))
    assert new.is_unique
