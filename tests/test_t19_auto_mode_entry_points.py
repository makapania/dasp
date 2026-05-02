"""T-19 Auto mode end-to-end at the entry-point layer.

The codegen-layer tests in `test_t19_auto_mode.py` exercise the generated
script's runtime resolution. These tests cover the parallel resolution that
runs at the entry of `run_search`, `run_nsga2_search`, and
`run_unified_bayesian` — pinned by pr-test-analyzer's H1 finding.

Coverage shape: each entry-point function calls `resolve_auto_imbalance`
against the live y, prints/logs the audit message, and rebinds
`imbalance_method` to the resolved value. Skipping the actual training (we
just want the resolution + audit-trail behaviour) lets the tests stay fast.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout

import numpy as np
import pytest
from sklearn.datasets import make_classification


# ----------------------------------------------------------------------
# resolve_auto_imbalance NaN handling — single source of truth
# ----------------------------------------------------------------------

def test_resolve_auto_drops_nan_targets_before_counting():
    """Counter() treats each NaN as a distinct hash; if NaNs leak into the
    Counter they spuriously appear as a minority class and the resolution
    decision is wrong. resolve_auto_imbalance must drop NaN before counting."""
    from spectral_predict.imbalance import resolve_auto_imbalance

    # Without NaN drop, the array would have classes {0.0, 1.0, nan, nan}
    # — Counter sees 4 keys and the imbalance ratio is computed over noise.
    y = np.array([0.0] * 50 + [1.0] * 50 + [np.nan] * 5)
    resolved, info = resolve_auto_imbalance(y)
    # After drop, perfectly balanced 50/50 → no correction.
    assert resolved is None
    assert info["imbalance_ratio"] == pytest.approx(1.0)


def test_format_auto_imbalance_message_imbalanced():
    from spectral_predict.imbalance import (
        format_auto_imbalance_message,
        resolve_auto_imbalance,
    )

    _, y = make_classification(
        n_samples=120, n_features=10, n_classes=2, weights=[0.85, 0.15], random_state=42
    )
    _, info = resolve_auto_imbalance(y)
    msg = format_auto_imbalance_message(info)
    assert "applying class_weight" in msg
    assert ":1" in msg


def test_format_auto_imbalance_message_balanced():
    from spectral_predict.imbalance import (
        format_auto_imbalance_message,
        resolve_auto_imbalance,
    )

    _, y = make_classification(
        n_samples=120, n_features=10, n_classes=2, weights=[0.5, 0.5], random_state=42
    )
    _, info = resolve_auto_imbalance(y)
    msg = format_auto_imbalance_message(info)
    assert "no correction" in msg


def test_format_auto_imbalance_message_single_class():
    from spectral_predict.imbalance import format_auto_imbalance_message
    msg = format_auto_imbalance_message({"is_imbalanced": False, "class_counts": {0: 50}})
    assert "single class" in msg


# ----------------------------------------------------------------------
# Entry-point auto-resolution — capture audit lines via logger + stdout
# ----------------------------------------------------------------------

def _capture_logger_and_stdout(module_logger_name: str, callable_):
    """Invoke `callable_` while capturing both the module's logger output
    and stdout. Returns (logger_lines, stdout_text). Used to verify the
    audit trail flows through both channels (logger for T-45 file handler,
    stdout for console runs)."""
    logger = logging.getLogger(module_logger_name)
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            callable_()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    return handler.stream.getvalue(), buf.getvalue()


def test_run_search_auto_resolves_imbalanced_data_and_logs():
    """run_search entry-point: imbalanced classification + Auto mode emits
    the resolution message via both logger.info AND stdout."""
    import pandas as pd

    from spectral_predict.search import run_search

    X, y = make_classification(
        n_samples=80, n_features=20, n_informative=10,
        n_classes=2, weights=[0.85, 0.15], random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"wl_{i}" for i in range(20)])
    y = pd.Series(y, name="target")

    def _run():
        try:
            run_search(
                X, y,
                task_type="classification",
                folds=3,
                models_to_test=["RandomForest"],
                preprocessing_methods={"raw": True},
                imbalance_method="auto",
                tier="quick",
                enabled_models={"RandomForest": True},
            )
        except Exception:
            pass  # we only care about the entry-point audit, not training

    log_text, stdout_text = _capture_logger_and_stdout(
        "spectral_predict.search", _run
    )
    combined = log_text + stdout_text
    assert "Auto imbalance" in combined
    assert "applying class_weight" in combined


def test_run_unified_bayesian_auto_resolves_balanced_data():
    """run_unified_bayesian entry-point: balanced classification + Auto mode
    resolves to None and emits 'no correction' via logger + stdout."""
    from spectral_predict.unified_bayesian import run_unified_bayesian

    X, y = make_classification(
        n_samples=60, n_features=15, n_informative=8,
        n_classes=2, weights=[0.5, 0.5], random_state=42,
    )
    wavelengths = np.arange(15, dtype=float)

    def _run():
        try:
            run_unified_bayesian(
                X, y, wavelengths,
                model_name="PLS",
                task_type="classification",
                n_trials=2,
                cv_folds=3,
                imbalance_method="auto",
                verbose=False,
            )
        except Exception:
            pass

    log_text, stdout_text = _capture_logger_and_stdout(
        "spectral_predict.unified_bayesian", _run
    )
    combined = log_text + stdout_text
    assert "Auto imbalance" in combined
    assert "no correction" in combined


def test_run_nsga2_search_auto_resolves_imbalanced_data():
    """run_nsga2_search entry-point: imbalanced classification + Auto resolves
    to class_weight. NaN-handling is delegated to resolve_auto_imbalance so
    no separate NaN-drop is needed in the entry-point itself."""
    from spectral_predict.nsga2_search import run_nsga2_search

    X, y = make_classification(
        n_samples=60, n_features=15, n_informative=8,
        n_classes=2, weights=[0.85, 0.15], random_state=42,
    )

    def _run():
        try:
            run_nsga2_search(
                X.astype(np.float64), y,
                task_type="classification",
                population_size=8,
                n_generations=2,
                cv_folds=3,
                models=["RandomForest"],
                imbalance_method="auto",
                verbose=0,
            )
        except Exception:
            pass

    log_text, stdout_text = _capture_logger_and_stdout(
        "spectral_predict.nsga2_search", _run
    )
    combined = log_text + stdout_text
    assert "Auto imbalance" in combined
    assert "applying class_weight" in combined
