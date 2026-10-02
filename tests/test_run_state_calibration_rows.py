"""The run record stores which samples a Bayesian run excluded (review of R004).

Reloading a file clears its exclusions while the data fingerprint still matches, so
the resume gate needs the run's own exclusions and Analysis Subset to compare.
"""

from __future__ import annotations

import numpy as np

from spectral_predict.run_state import RunMetadata, calibration_rows_record


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


def test_record_normalises_numpy_labels_and_keeps_empty_exclusions():
    rows = calibration_rows_record([np.int64(5), 2], None)
    assert rows == {"excluded": [2, 5], "active": None}
    assert all(type(v) is int for v in rows["excluded"])
    assert calibration_rows_record([], ["b", "a"]) == {"excluded": [], "active": ["a", "b"]}


def test_record_refuses_labels_it_cannot_round_trip():
    assert calibration_rows_record([("a", 1)], None) is None
    assert calibration_rows_record([], [1.5]) is None


def test_metadata_round_trip_keeps_calibration_rows():
    rows = {"excluded": ["S3"], "active": None}
    meta = RunMetadata.from_dict(_meta(calibration_rows=rows))
    assert RunMetadata.from_dict(meta.to_dict()).calibration_rows == rows


def test_old_and_malformed_records_mean_unchecked():
    assert RunMetadata.from_dict(_meta()).calibration_rows is None
    bad = RunMetadata.from_dict(_meta(calibration_rows={"excluded": "S3"}))
    assert bad.calibration_rows is None
