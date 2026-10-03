"""The run record identifies the calibration rows a Bayesian run used (review of R004).

Reloading a file clears its exclusions while the data fingerprint still matches, so
the record stores the run's sample selection (type-tagged label keys, used to offer
restoring it) and a digest of the exact calibration and holdout rows (the final
check on resume).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from spectral_predict.io import rename_duplicate_ids
from spectral_predict.run_state import (
    CALIBRATION_IDENTITY_VERSION,
    CALIBRATION_RECORD_VERSION,
    LABEL_NORMALIZATION,
    RunMetadata,
    UnsupportedLabelError,
    calibration_identity,
    calibration_rows_record,
    canonical_label,
    valid_calibration_identity,
    valid_calibration_rows,
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


def _frame(labels, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(len(labels), 5)), index=labels, columns=[1, 2, 3, 4, 5])
    return X, pd.Series(rng.normal(size=len(labels)), index=X.index)


def test_canonical_labels_keep_types_apart_and_survive_json():
    labels = [5, np.int64(5), 5.0, "5", ("a", 1), np.float64(2.5), float("nan"), True]
    keys = [canonical_label(v) for v in labels]
    assert keys[0] == keys[1]  # numpy and Python ints are the same label
    assert len({keys[0], keys[2], keys[3], keys[7]}) == 4  # 5, 5.0, "5", True differ
    assert json.loads(json.dumps(keys)) == keys
    assert canonical_label(np.float64(2.5)) == canonical_label(2.5)


@pytest.mark.parametrize("label", [None, dt.date(2026, 1, 1), frozenset({1}), object()])
def test_unsupported_labels_are_refused(label):
    with pytest.raises(UnsupportedLabelError):
        canonical_label(label)
    with pytest.raises(UnsupportedLabelError):
        calibration_rows_record([label], None)


def test_record_stores_selection_and_holdout():
    rows = calibration_rows_record([np.int64(5), 2.0], None, holdout=[("a", 1)])
    assert valid_calibration_rows(rows)
    assert rows["version"] == CALIBRATION_RECORD_VERSION and rows["active"] is None
    assert set(rows["excluded"]) == {canonical_label(5), canonical_label(2.0)}
    assert rows["holdout"] == [canonical_label(("a", 1))]


def test_identity_changes_with_labels_order_values_and_targets():
    X, y = _frame(["a", "b", "c"])
    base = calibration_identity(X, y, None, None)
    assert valid_calibration_identity(base)
    assert calibration_identity(X.copy(), y.copy(), None, None) == base
    swapped = X.rename(index={"a": "b", "b": "a"})
    assert calibration_identity(swapped, y.set_axis(swapped.index), None, None) != base
    order = ["c", "b", "a"]
    assert calibration_identity(X.loc[order], y.loc[order], None, None) != base
    edited = X.copy()
    edited.iloc[0, 0] += 1e-9
    assert calibration_identity(edited, y, None, None) != base
    assert calibration_identity(X, y + 1, None, None) != base
    assert calibration_identity(X, y, X.iloc[:1], y.iloc[:1]) != base


def test_metadata_round_trip_keeps_the_new_fields():
    X, y = _frame(["a", "b"])
    fields = {
        "calibration_rows": calibration_rows_record(["a"], None),
        "label_normalization": LABEL_NORMALIZATION,
        "calibration_identity": calibration_identity(X, y, None, None),
    }
    meta = RunMetadata.from_dict(_meta(**fields))
    again = RunMetadata.from_dict(json.loads(json.dumps(meta.to_dict())))
    for name, value in fields.items():
        assert getattr(again, name) == value


def test_legacy_and_unrecognised_records():
    old = RunMetadata.from_dict(_meta())
    assert old.calibration_rows is None and old.label_normalization is None
    assert old.calibration_identity is None
    # An intermediate-format record is kept as stored; the gate can't verify it.
    odd = RunMetadata.from_dict(_meta(calibration_rows={"excluded": [2], "active": None}))
    assert odd.calibration_rows is not None and not valid_calibration_rows(odd.calibration_rows)


def test_repeated_missing_ids_get_unique_names():
    new, n, _ = rename_duplicate_ids(pd.Index(["A", np.nan, "B", np.nan, "A"]))
    assert list(new) == ["A", "nan", "B", "nan.1", "A.1"]
    assert new.is_unique and n == 3
    new, _, _ = rename_duplicate_ids(pd.Index(["nan", np.nan, "x", "x"]))
    assert new.is_unique


def test_identity_hashes_large_integer_targets_losslessly():
    X, _ = _frame(["a", "b", "c"])
    big = 2**53
    y1 = pd.Series([0, big, big + 1], index=X.index)
    y2 = pd.Series([0, big + 1, big], index=X.index)
    assert calibration_identity(X, y1, None, None) != calibration_identity(X, y2, None, None)


def test_identity_is_container_independent_for_numeric_targets():
    X, _ = _frame(["a", "b", "c"])
    for values in ([1, 2, 3], [1.5, 2.0, float("nan")]):
        numeric = pd.Series(values, index=X.index)
        as_object = pd.Series(
            [np.float64(v) if isinstance(v, float) else np.int64(v) for v in values],
            index=X.index,
            dtype=object,
        )
        assert calibration_identity(X, numeric, None, None) == calibration_identity(
            X, as_object, None, None
        )
    strings = pd.Series(["1", "2", "3"], index=X.index)
    ints = pd.Series([1, 2, 3], index=X.index)
    assert calibration_identity(X, strings, None, None) != calibration_identity(X, ints, None, None)


def test_identity_labels_are_length_prefixed():
    X1, y = _frame(["a\x00s:b", "c"])
    X2 = X1.set_axis(["a", "b\x00s:c"])
    assert calibration_identity(X1, y, None, None) != calibration_identity(
        X2, y.set_axis(X2.index), None, None
    )


def test_strict_schema_and_version():
    assert not valid_calibration_rows({"version": 1, "excluded": [], "holdout": []})
    assert not valid_calibration_rows(
        {"version": 1, "excluded": [], "active": None, "holdout": [["S0"]]}
    )
    X, y = _frame(["a"])
    identity = calibration_identity(X, y, None, None)
    assert identity["version"] == CALIBRATION_IDENTITY_VERSION
    assert not valid_calibration_identity(dict(identity, version=1))
    assert not valid_calibration_identity({k: v for k, v in identity.items() if k != "n_holdout"})


def test_identity_keeps_unsigned_integers_above_int64_apart():
    X, _ = _frame(["a", "b"])
    top = np.uint64(2**64 - 1)
    y_big = pd.Series(np.array([top, 1], dtype=np.uint64), index=X.index)
    y_neg = pd.Series(np.array([-1, 1], dtype=np.int64), index=X.index)
    assert calibration_identity(X, y_big, None, None) != calibration_identity(X, y_neg, None, None)
    small = pd.Series(np.array([3, 1], dtype=np.uint64), index=X.index)
    signed = pd.Series(np.array([3, 1], dtype=np.int64), index=X.index)
    assert calibration_identity(X, small, None, None) == calibration_identity(X, signed, None, None)


def test_identity_object_and_uint64_containers_agree_above_int64():
    X, _ = _frame(["a", "b"])
    top = 2**64 - 1
    as_object = pd.Series([top, 1], index=X.index, dtype=object)
    as_uint = pd.Series(np.array([top, 1], dtype=np.uint64), index=X.index)
    assert calibration_identity(X, as_object, None, None) == calibration_identity(
        X, as_uint, None, None
    )


def test_combined_reader_renames_repeated_missing_ids(tmp_path):
    from spectral_predict.io import read_combined_csv

    rng = np.random.default_rng(0)
    ids = ["A", "", "B", "", "C", "D", "E", "F"]
    spectra = rng.normal(size=(len(ids), 120))
    df = pd.DataFrame(spectra, columns=[str(1000 + 2 * i) for i in range(120)])
    df.insert(0, "protein", rng.uniform(0, 10, len(ids)))
    df.insert(0, "sample_id", ids)
    path = tmp_path / "blank_ids.csv"
    df.to_csv(path, index=False)

    X, y, _, metadata = read_combined_csv(path, y_col="protein", specimen_id_col="sample_id")
    assert X.index.is_unique and len(X) == len(ids)
    assert y.index.equals(X.index)
