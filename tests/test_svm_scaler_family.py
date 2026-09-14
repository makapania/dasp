"""T-51 step 1: classification SVM must be StandardScaler-wrapped.

The registered classification family is ``'SVM'`` (``model_registry.py``), but the
scale-sensitive sets listed only ``'SVC'``, a string no model name ever matched.
Every classification SVM was therefore fit on unscaled spectra, in grid search,
Bayesian search and the validation rebuild, while exported scripts
(``code_generator._needs_standard_scaler``) did scale it.

The spy tests record the X each ``SVC.fit`` / ``SVR.fit`` receives. Behind a
fitted StandardScaler every training column has mean 0 and std 1. The raw
fixture spans twelve orders of magnitude in column scale, so an unscaled fit
cannot pass by accident.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from spectral_predict.search import SCALE_SENSITIVE_MODELS, _rebuild_model_from_row


def _scaled_spectra(n_samples: int = 60, n_features: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Two-class data whose columns differ in scale by 1e12."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], n_samples // 2)
    X = rng.normal(size=(n_samples, n_features)) + 0.8 * y[:, None]
    X *= 10.0 ** np.linspace(-6, 6, n_features)
    return X, y


def _spy_fit(
    monkeypatch: pytest.MonkeyPatch, estimator_cls: type, exclude_callers: frozenset[str] = frozenset()
) -> list[np.ndarray]:
    """Record the X passed to ``estimator_cls.fit``, skipping fits made from ``exclude_callers``."""
    seen: list[np.ndarray] = []
    original = estimator_cls.fit

    def fit(self, X, y, *args, **kwargs):
        if not any(frame.function in exclude_callers for frame in inspect.stack(0)):
            seen.append(np.asarray(X, dtype=float))
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(estimator_cls, "fit", fit)
    return seen


def _is_standardized(X: np.ndarray) -> bool:
    return bool(
        np.allclose(X.mean(axis=0), 0.0, atol=1e-6) and np.allclose(X.std(axis=0), 1.0, atol=1e-6)
    )


def test_registered_svm_family_is_scale_sensitive() -> None:
    assert "SVM" in SCALE_SENSITIVE_MODELS
    assert "SVR" in SCALE_SENSITIVE_MODELS


@pytest.mark.parametrize(
    ("model_name", "task_type", "estimator_cls"),
    [("SVM", "classification", SVC), ("SVR", "regression", SVR)],
)
def test_rebuild_wraps_svm_family_in_scaler(model_name, task_type, estimator_cls) -> None:
    row = pd.Series({"Model": model_name, "Params": "{}", "LVs": None})
    model = _rebuild_model_from_row(row, task_type)
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["model"], estimator_cls)


def test_rebuild_skips_per_model_scaler_under_autoscale() -> None:
    """T-36: autoscale already scales upstream; no second scaler."""
    row = pd.Series({"Model": "SVM", "Params": "{}", "LVs": None})
    model = _rebuild_model_from_row(row, "classification", autoscale=True)
    assert isinstance(model, SVC)


def test_grid_single_config_fits_svm_on_scaled_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from spectral_predict.search import _run_single_config

    X, y = _scaled_spectra()
    seen = _spy_fit(monkeypatch, SVC)
    _run_single_config(
        X=X,
        y=y,
        wavelengths=np.arange(X.shape[1], dtype=float),
        model=SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        model_name="SVM",
        params={"C": 1.0, "kernel": "rbf"},
        preprocess_cfg={"name": "raw", "deriv": 0, "window": np.nan, "polyorder": np.nan},
        cv_splitter=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        task_type="classification",
        is_binary_classification=True,
    )
    assert seen, "SVC.fit was never called"
    assert all(_is_standardized(x) for x in seen)


def test_bayesian_fits_svm_on_scaled_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from spectral_predict.unified_bayesian import run_unified_bayesian

    X, y = _scaled_spectra()
    # compute_importances fits a bare proxy estimator for variable selection. It is
    # unscaled for every scale-sensitive family, not just SVM, and is out of scope
    # here (recorded in SESSION_LOG 2026-09-13). Only the CV and refit fits are checked.
    seen = _spy_fit(monkeypatch, SVC, exclude_callers=frozenset({"compute_importances"}))
    run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=np.linspace(1000, 2500, X.shape[1]),
        model_name="SVM",
        task_type="classification",
        n_trials=3,
        cv_folds=3,
        random_state=42,
        verbose=False,
        enable_sqlite_persistence="never",
    )
    assert seen, "SVC.fit was never called"
    assert all(_is_standardized(x) for x in seen)


def test_nsga2_svm_family_fits_on_scaled_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """NSGA-II encodes classification SVM as 'SVR'; guard that it stays scaled."""
    from spectral_predict.nsga2_search import run_nsga2_search

    X, y = _scaled_spectra()
    seen = _spy_fit(monkeypatch, SVC)
    run_nsga2_search(
        X=X,
        y=y,
        task_type="classification",
        population_size=6,
        n_generations=1,
        cv_folds=3,
        min_wavelengths=10,
        random_state=42,
        verbose=0,
        models=["SVR"],
    )
    assert seen, "SVC.fit was never called"
    assert all(_is_standardized(x) for x in seen)


# Deliberately re-blessed by the T-51 step-1 scaler fix. Before it, the rebuilt
# classification SVM had no scaler and scored UNSCALED_ACCURACY on this fixture.
SCALED_ACCURACY = 1.0
UNSCALED_ACCURACY = 0.8833333333333332


def test_svm_classification_baseline_snapshot() -> None:
    X, y = _scaled_spectra()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    row = pd.Series({"Model": "SVM", "Params": "{}", "LVs": None})
    scaled = cross_val_score(_rebuild_model_from_row(row, "classification"), X, y, cv=cv).mean()
    unscaled_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    unscaled = cross_val_score(unscaled_model, X, y, cv=cv).mean()
    assert scaled == pytest.approx(SCALED_ACCURACY, abs=1e-9)
    assert unscaled == pytest.approx(UNSCALED_ACCURACY, abs=1e-9)
    assert scaled != pytest.approx(unscaled, abs=1e-9)
