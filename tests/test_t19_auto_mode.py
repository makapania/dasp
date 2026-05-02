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


def test_auto_mode_emits_runtime_conditional_class_weight_for_supported_libs():
    """Post-fix-of-fixes (DeepSeek Q2): Auto mode injects class_weight via
    runtime conditional, NOT baked into the params literal. Required so the
    runtime resolution can correctly skip injection on balanced data."""
    script = _generate_auto("RandomForest")
    assert "model_params['class_weight'] = 'balanced'" in script
    # Critical: NOT baked into the literal — that was the Q2 trap.
    assert "'class_weight': 'balanced'" not in script


def test_auto_mode_does_not_inject_class_weight_for_xgboost():
    """XGBoost ignores class_weight in __init__; sample_weight at fit is the
    canonical balanced-loss path. Auto inherits that dispatch."""
    script = _generate_auto("XGBoost")
    assert "'class_weight': 'balanced'" not in script
    assert "model_params['class_weight']" not in script  # also not as runtime conditional
    assert "fit_kwargs['sample_weight'] = compute_sample_weight('balanced'" in script


def test_auto_mode_uses_auto_class_weights_for_catboost():
    """Auto mode + CatBoost: runtime conditional injection of auto_class_weights
    instead of baked literal (DeepSeek Q2 fix)."""
    script = _generate_auto("CatBoost", params={"iterations": 30, "depth": 3})
    assert "model_params['auto_class_weights'] = 'Balanced'" in script
    assert "'class_weight': 'balanced'" not in script
    # Not in literal either — the trap.
    assert "'auto_class_weights': 'Balanced'" not in script


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


# ----------------------------------------------------------------------
# Q2 trap (DeepSeek HIGH): mild imbalance (2:1, below threshold) under Auto
# must NOT leave a baked balanced kwarg on the model
# ----------------------------------------------------------------------

def _mild_imbalance_data(seed: int = 42, n_samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """2:1 ratio — below the 3:1 Auto threshold."""
    return make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.66, 0.34],
        random_state=seed,
    )


def _model_kwargs(model) -> dict:
    """Extract get_params() from the inner classifier (unwraps Pipelines)."""
    if hasattr(model, "steps"):
        steps = dict(model.steps)
        inner = steps.get("model") or steps.get("lr") or list(model.steps)[-1][1]
    else:
        inner = model
    return inner.get_params() if hasattr(inner, "get_params") else {}


@pytest.mark.parametrize(
    "model_name,params,balanced_kwarg",
    [
        ("RandomForest", {"n_estimators": 30, "max_depth": 3}, "class_weight"),
        ("LightGBM", {"n_estimators": 30, "max_depth": 3}, "class_weight"),
        ("SVC", {"C": 1.0}, "class_weight"),
        ("CatBoost", {"iterations": 30, "depth": 3}, "auto_class_weights"),
        ("PLS", {"n_components": 3}, "class_weight"),
    ],
)
def test_auto_mode_mild_imbalance_does_not_bake_balanced_kwarg(
    model_name: str, params: dict, balanced_kwarg: str
) -> None:
    """DeepSeek HIGH (Q2): pre-fix, the codegen baked class_weight='balanced'
    into the constructor literal under Auto mode regardless of resolution
    outcome. On mild imbalance (2:1, below the 3:1 threshold) the runtime
    resolution would print 'no correction' but the model would still train
    with non-uniform weights. Fix-of-fixes makes the kwarg injection a
    runtime conditional gated on IMBALANCE_METHOD == 'class_weight'.
    """
    X, y = _mild_imbalance_data()
    g = _execute_auto(model_name, X, y, params)
    assert "no correction" in g["_stdout"], (
        f"Setup error: {model_name} on 2:1 data should resolve auto → no correction"
    )
    kwargs = _model_kwargs(g["model"])
    val = kwargs.get(balanced_kwarg, "<absent>")
    # Either absent entirely or explicitly None — both indicate no balanced
    # weighting was applied. The trap was the value being 'balanced' / 'Balanced'.
    assert val in (None, "<absent>"), (
        f"{model_name} under Auto + mild imbalance must NOT have "
        f"{balanced_kwarg} set; got {val!r}. This is the Q2 trap — model "
        f"silently weights the loss while the run prints 'no correction'."
    )


@pytest.mark.parametrize(
    "model_name,params,balanced_kwarg,expected_value",
    [
        ("RandomForest", {"n_estimators": 30, "max_depth": 3}, "class_weight", "balanced"),
        ("LightGBM", {"n_estimators": 30, "max_depth": 3}, "class_weight", "balanced"),
        ("SVC", {"C": 1.0}, "class_weight", "balanced"),
        ("CatBoost", {"iterations": 30, "depth": 3}, "auto_class_weights", "Balanced"),
        ("PLS", {"n_components": 3}, "class_weight", "balanced"),
    ],
)
def test_auto_mode_severe_imbalance_applies_balanced_kwarg(
    model_name: str, params: dict, balanced_kwarg: str, expected_value: str
) -> None:
    """Symmetric to the Q2 trap test: on severe imbalance (5:1, above
    threshold) the runtime resolution mutates IMBALANCE_METHOD to
    'class_weight' and the conditional kwarg injection fires."""
    X, y = _imbalanced_data()
    g = _execute_auto(model_name, X, y, params)
    assert "applying class_weight" in g["_stdout"]
    kwargs = _model_kwargs(g["model"])
    assert kwargs.get(balanced_kwarg) == expected_value, (
        f"{model_name} under Auto + severe imbalance should have "
        f"{balanced_kwarg}={expected_value!r}; got {kwargs.get(balanced_kwarg)!r}"
    )


def test_auto_mode_mlp_severe_imbalance_emits_both_resolution_and_warning():
    """Codex LOW (post-Q2 review): the MLP-specific 'model will train
    unweighted' note is emitted from code_generator.py inside the auto
    resolution block, but no end-to-end test exercises Auto+MLP+severe.
    A future indentation/refactor break would slip through. Pin both the
    standard 'applying class_weight' resolution message AND the MLP-specific
    follow-up warning so MLP users see why no balancing is happening."""
    X, y = _imbalanced_data()
    g = _execute_auto("MLP", X, y, params={"hidden_layer_sizes": (10,), "max_iter": 50})
    assert "applying class_weight" in g["_stdout"], (
        "Auto resolution itself should fire on imbalanced data regardless of model"
    )
    assert "MLP does not support class_weight" in g["_stdout"], (
        "MLP-specific warning should appear when Auto resolves to class_weight "
        "but the model can't accept it"
    )
    assert "train unweighted" in g["_stdout"]
    # And confirm the model itself has no class_weight set (mirrors runtime fallback).
    kwargs = _model_kwargs(g["model"])
    assert kwargs.get("class_weight", None) is None
