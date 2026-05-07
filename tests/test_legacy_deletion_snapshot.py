"""Before/after snapshot tests for the legacy Bayesian path deletion.

Pins ``run_unified_bayesian`` outputs on three representative configurations
with fixed RNG seeds. Generated with the legacy path present; asserted
byte-identical after the legacy path is deleted. This file is removed
once the deletion PR merges.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from spectral_predict.io import read_asd_dir
from spectral_predict.unified_bayesian import run_unified_bayesian

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
EXAMPLE_DIR = Path(__file__).parent.parent / "example"

REGRESSION_TARGET = "%Collagen"
CLASSIFICATION_TARGET = "CollagenCat"


def _load_joined_dataframe():
    """Mirrors tools/bench_baseline_compare.py:54-67 — canonical loader."""
    spectra, _meta = read_asd_dir(str(EXAMPLE_DIR))
    ref = pd.read_csv(EXAMPLE_DIR / "BoneCollagen.csv")
    spectra = spectra.sort_index()
    ref = ref.copy()
    ref.index = (
        ref["File Number"].astype(str)
        .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        .str.replace(" ", "", regex=False)
    )
    joined = spectra.join(ref, how="inner")
    feature_cols = list(spectra.columns)
    X = joined[feature_cols].to_numpy(dtype=float)
    wl = np.asarray([float(c) for c in feature_cols], dtype=float)
    return joined, X, wl


def _load_bone_collagen_xy():
    joined, X, wl = _load_joined_dataframe()
    y = joined[REGRESSION_TARGET].to_numpy(dtype=float)
    return X, y, wl


def _load_bone_collagen_classification():
    joined, X, wl = _load_joined_dataframe()
    le = LabelEncoder()
    y = le.fit_transform(joined[CLASSIFICATION_TARGET].astype(str))
    return X, y, wl


def _coerce_scalar(value):
    """Coerce arbitrary values to JSON-serializable forms with stable rounding."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "__nan__"
        if np.isinf(value):
            return "__pos_inf__" if value > 0 else "__neg_inf__"
        return round(float(value), 10)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (tuple, list)):
        return [_coerce_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_scalar(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if value is None:
        return None
    return str(value)


def _serialize_results(df: pd.DataFrame, study, top_n: int = 5) -> dict:
    """Capture per-trial ordered records + leaderboard surface + study best."""
    df_in_trial_order = df.reset_index(drop=True)
    df_sorted = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    top = df_sorted.head(top_n)
    trial_records = [
        {
            "number": int(t.number),
            "state": str(t.state),
            "value": _coerce_scalar(t.value) if t.value is not None else None,
            "params": {k: _coerce_scalar(v) for k, v in sorted(t.params.items())},
            "fingerprint": str(t.user_attrs.get("fingerprint", "")) or None,
            "duplicate_of": t.user_attrs.get("duplicate_of_trial"),
        }
        for t in sorted(study.trials, key=lambda t: t.number)
    ]
    return {
        "n_rows": len(df_in_trial_order),
        "columns": sorted(df_in_trial_order.columns.tolist()),
        "top_n": top_n,
        "all_rows_in_trial_order": [
            {col: _coerce_scalar(row[col]) for col in sorted(df_in_trial_order.columns)}
            for _, row in df_in_trial_order.iterrows()
        ],
        "top_rows_after_sort": [
            {col: _coerce_scalar(row[col]) for col in sorted(df_sorted.columns)}
            for _, row in top.iterrows()
        ],
        "best_value": _coerce_scalar(study.best_value),
        "best_params": {
            k: _coerce_scalar(v) for k, v in sorted(study.best_params.items())
        },
        "trial_count": len(trial_records),
        "trials": trial_records,
    }


def _assert_matches_snapshot(payload: dict, snapshot_name: str):
    """Compare payload against the committed JSON fixture."""
    snapshot_path = SNAPSHOT_DIR / snapshot_name
    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        pytest.fail(
            f"Snapshot {snapshot_name} did not exist; created from current run. "
            f"Re-run to verify it matches."
        )
    expected = json.loads(snapshot_path.read_text())
    assert payload == expected, (
        f"Snapshot {snapshot_name} drifted. "
        f"If intentional, delete the file and re-run to regenerate."
    )


@pytest.mark.integration
class TestUnifiedBayesianDeletionSnapshot:
    """Snapshot the public surface of run_unified_bayesian for three configs."""

    def test_pls_regression_snapshot(self):
        X, y, wl = _load_bone_collagen_xy()
        df, study = run_unified_bayesian(
            X=X, y=y, wavelengths=wl,
            model_name="PLS", task_type="regression",
            n_trials=20, random_state=42,
            cv_strategy="kfold", cv_folds=5,
        )
        _assert_matches_snapshot(
            _serialize_results(df, study), "unified_bayesian_pls_regression.json"
        )

    def test_lgbm_regression_snapshot(self):
        X, y, wl = _load_bone_collagen_xy()
        df, study = run_unified_bayesian(
            X=X, y=y, wavelengths=wl,
            model_name="LightGBM", task_type="regression",
            n_trials=15, random_state=42,
            cv_strategy="kfold", cv_folds=5,
        )
        _assert_matches_snapshot(
            _serialize_results(df, study), "unified_bayesian_lgbm_regression.json"
        )

    def test_plsda_classification_snapshot(self):
        X, y, wl = _load_bone_collagen_classification()
        df, study = run_unified_bayesian(
            X=X, y=y, wavelengths=wl,
            model_name="PLS-DA", task_type="classification",
            n_trials=15, random_state=42,
            cv_strategy="kfold", cv_folds=5,
        )
        _assert_matches_snapshot(
            _serialize_results(df, study), "unified_bayesian_plsda_classification.json"
        )
