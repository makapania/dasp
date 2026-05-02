"""T-19 Auto mode regression tests.

Auto mode is a third option (alongside off + class_weight) that calls
:func:`detect_class_imbalance` at fit time and applies class_weight only if
the severity threshold is exceeded. The point is automatic handling — the
user shouldn't need to think about whether class_weight is appropriate;
the system decides per fold from the actual class ratios.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from sklearn.datasets import make_classification

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.imbalance import resolve_auto_imbalance


def _imbalanced_data(seed: int = 42, n_samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=seed,
    )


def _balanced_data(seed: int = 42, n_samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=seed,
    )


# ----------------------------------------------------------------------
# resolve_auto_imbalance helper contract
# ----------------------------------------------------------------------

def test_resolve_auto_returns_class_weight_for_severe_imbalance():
    _, y = _imbalanced_data()
    resolved, info = resolve_auto_imbalance(y)
    assert resolved == "class_weight"
    assert info["is_imbalanced"] is True
    assert info["imbalance_ratio"] >= 3.0


def test_resolve_auto_returns_none_for_balanced():
    _, y = _balanced_data()
    resolved, info = resolve_auto_imbalance(y)
    assert resolved is None
    assert info["is_imbalanced"] is False
    assert info["imbalance_ratio"] < 3.0


def test_resolve_auto_returns_none_for_regression_task():
    _, y = _imbalanced_data()
    resolved, info = resolve_auto_imbalance(y, task_type="regression")
    assert resolved is None
    assert info == {}


def test_resolve_auto_respects_custom_threshold():
    """Mild imbalance (2:1) should resolve to None at default threshold (3.0)
    but to class_weight at a stricter threshold."""
    _, y = make_classification(
        n_samples=120,
        n_features=10,
        n_classes=2,
        weights=[0.66, 0.34],
        random_state=42,
    )
    assert resolve_auto_imbalance(y, threshold=3.0)[0] is None
    assert resolve_auto_imbalance(y, threshold=1.5)[0] == "class_weight"


# ----------------------------------------------------------------------
# Code generator emission for Auto mode
# ----------------------------------------------------------------------

def _generate_auto(model_name: str, params: dict | None = None) -> str:
    config = {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": "classification",
        "params": params or {"n_estimators": 30, "max_depth": 3},
        "cv_folds": 3,
        "imbalance_method": "auto",
    }
    return CodeGenerator(config, ExportOptions(include_data=False)).generate_script()


def _execute_auto(model_name: str, X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    config = {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": "classification",
        "params": params or {"n_estimators": 30, "max_depth": 3},
        "cv_folds": 3,
        "imbalance_method": "auto",
    }
    opts = ExportOptions(include_data=True, data_X=X.copy(), data_y=y.copy())
    script = CodeGenerator(config, opts).generate_script()
    g: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(script, g)
    g["_stdout"] = buf.getvalue()
    return g


def test_auto_mode_emits_runtime_resolution_block():
    """Generated script must include the auto-resolution conditional that
    mutates IMBALANCE_METHOD based on detected class ratio."""
    script = _generate_auto("RandomForest")
    assert 'IMBALANCE_METHOD = "auto"' in script
    assert 'if IMBALANCE_METHOD == "auto":' in script
    assert "Auto imbalance" in script


def test_auto_mode_bakes_class_weight_for_supported_libs():
    """Auto mode is treated like class_weight at codegen for libraries that
    accept the kwarg natively (RandomForest, LightGBM, sklearn LR/SVC)."""
    script = _generate_auto("RandomForest")
    assert "'class_weight': 'balanced'" in script


def test_auto_mode_does_not_inject_class_weight_for_xgboost():
    """XGBoost ignores class_weight in __init__; sample_weight at fit is the
    canonical balanced-loss path. Auto inherits that dispatch."""
    script = _generate_auto("XGBoost")
    assert "'class_weight': 'balanced'" not in script
    assert "fit_kwargs['sample_weight'] = compute_sample_weight('balanced'" in script


def test_auto_mode_uses_auto_class_weights_for_catboost():
    """Auto mode + CatBoost: same canonical dispatch as class_weight + CatBoost."""
    script = _generate_auto("CatBoost", params={"iterations": 30, "depth": 3})
    assert "'auto_class_weights': 'Balanced'" in script
    assert "'class_weight': 'balanced'" not in script


def test_auto_mode_does_not_inject_class_weight_for_mlp():
    """MLP rejects class_weight kwarg with TypeError; same exclusion as
    explicit class_weight mode."""
    script = _generate_auto("MLP", params={"hidden_layer_sizes": (10,), "max_iter": 50})
    assert "'class_weight': 'balanced'" not in script


# ----------------------------------------------------------------------
# End-to-end: imbalanced data triggers correction; balanced does not
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_name,params",
    [
        ("XGBoost", {"n_estimators": 30, "max_depth": 3}),
        ("CatBoost", {"iterations": 30, "depth": 3}),
        ("LightGBM", {"n_estimators": 30, "max_depth": 3}),
        ("RandomForest", {"n_estimators": 30, "max_depth": 3}),
    ],
)
def test_auto_mode_imbalanced_data_applies_class_weight(model_name: str, params: dict) -> None:
    X, y = _imbalanced_data()
    g = _execute_auto(model_name, X, y, params)
    assert "applying class_weight" in g["_stdout"]
    assert "accuracy" in g
    assert isinstance(g["accuracy"], float)


@pytest.mark.parametrize(
    "model_name,params",
    [
        ("XGBoost", {"n_estimators": 30, "max_depth": 3}),
        ("CatBoost", {"iterations": 30, "depth": 3}),
        ("LightGBM", {"n_estimators": 30, "max_depth": 3}),
        ("RandomForest", {"n_estimators": 30, "max_depth": 3}),
    ],
)
def test_auto_mode_balanced_data_skips_correction(model_name: str, params: dict) -> None:
    X, y = _balanced_data()
    g = _execute_auto(model_name, X, y, params)
    assert "no correction" in g["_stdout"]
    assert "accuracy" in g
