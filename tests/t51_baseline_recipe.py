"""Deterministic inputs for the T-51 default-path baselines.

Shared by ``tests/test_t51_extra_axes_mechanism.py`` and the one-off capture that
produced ``tests/fixtures/t51_default_path_baseline.json``. Changing anything here
invalidates that fixture.
"""
from __future__ import annotations

from typing import Any

import numpy as np

TRACE_TRIALS = 30


def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(51)
    n_samples, n_features = 45, 80
    wavelengths = np.linspace(1000.0, 2500.0, n_features)
    y = rng.uniform(0.0, 10.0, n_samples)
    X = np.outer(y, np.sin(np.linspace(0.0, 3.0 * np.pi, n_features)))
    X += rng.normal(0.0, 0.3, (n_samples, n_features))
    return X, y, wavelengths


def classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, wavelengths = regression_data()
    return X, (y > np.median(y)).astype(int), wavelengths


def one_class_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, wavelengths = regression_data()
    return X, np.where(y > 3.0, 1, 0), wavelengths


# (model_name, task_type, data_fn name, extra kwargs) for pinned default study names.
NAME_CASES: list[tuple[str, str, str, dict[str, Any]]] = [
    ("PLS", "regression", "regression_data", {}),
    ("Ridge", "regression", "regression_data", {}),
    ("SVM", "classification", "classification_data", {}),
    ("PCA-SIMCA", "one_class", "one_class_data", {"inlier_class_label": 1}),
]

COMMON_KWARGS: dict[str, Any] = {
    "cv_folds": 3,
    "random_state": 42,
    "verbose": False,
    "enable_sqlite_persistence": "never",
}


def study_base_name(study_name: str) -> str:
    """Strip the ``_env1_<digest>`` suffix, leaving ``unified_bayesian_<model>_<confighash>``."""
    head, sep, _ = study_name.rpartition("_env")
    return head if sep else study_name
