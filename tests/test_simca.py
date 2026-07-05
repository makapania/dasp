"""Tests for spectral_predict.simca — T-31 multi-class class modeling.

A2: MultiClassClassModel core (per-class DD-SIMCA membership + decision matrix +
predict summary labels + min_class_samples unmodelable handling).

These tests define the A2 API contract. scaling="none" is used explicitly
throughout (per_class/global scaling modes arrive in A3, which also flips the
constructor default to "per_class").
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_predict.simca import MultiClassClassModel


def _blobs(sizes, n_features=20, seed=0, sep=10.0):
    """K well-separated isotropic Gaussian blobs; class k centered on axis k."""
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    for k, n in enumerate(sizes):
        center = np.zeros(n_features)
        center[k % n_features] = sep * (k + 1)
        Xs.append(rng.normal(size=(n, n_features)) + center)
        ys.append(np.full(n, k))
    return np.vstack(Xs), np.concatenate(ys)


def _center(k, n_features=20, sep=10.0):
    c = np.zeros(n_features)
    c[k % n_features] = sep * (k + 1)
    return c


class TestMultiClassCoreA2:
    def test_decision_matrix_shapes_and_bounds(self):
        X, y = _blobs([40, 40, 40])
        m = MultiClassClassModel(
            engine="pca-simca", alpha=0.05, n_components=3, scaling="none"
        ).fit(X, y)
        P, A = m.decision_matrix(X)
        assert P.shape == (len(X), 3)
        assert A.shape == (len(X), 3)
        assert A.dtype == bool
        finite = np.isfinite(P)
        assert np.all(P[finite] >= 0.0) and np.all(P[finite] <= 1.0)
        assert list(m.classes_) == [0, 1, 2]

    def test_predict_single_and_novel(self):
        X, y = _blobs([50, 50, 50])
        m = MultiClassClassModel(n_components=3, scaling="none").fit(X, y)
        # the exact class-0 center is the most-inlier point -> labeled 0
        x_in = _center(0).reshape(1, -1)
        assert m.predict(x_in)[0] == 0
        # a far-away sample belongs to no class -> "novel"
        x_far = np.full((1, 20), 500.0)
        assert m.predict(x_far)[0] == "novel"

    def test_predict_multiple_membership(self):
        # two classes sharing one center -> a central point is accepted by both
        rng = np.random.RandomState(2)
        X = np.vstack([rng.normal(size=(60, 20)), rng.normal(size=(60, 20))])
        y = np.array([0] * 60 + [1] * 60)
        m = MultiClassClassModel(n_components=3, scaling="none", alpha=0.05).fit(X, y)
        x_center = np.zeros((1, 20))
        P, A = m.decision_matrix(x_center)
        assert A[0].sum() >= 2
        assert m.predict(x_center)[0] == "multiple"

    def test_unmodelable_class_preserved(self):
        # class 0 (n=5) is below min_class_samples=10 -> unmodelable, column kept
        X, y = _blobs([5, 30, 100])
        m = MultiClassClassModel(
            n_components=3, scaling="none", min_class_samples=10
        ).fit(X, y)
        assert 0 in m.unmodelable_
        assert list(m.classes_) == [0, 1, 2]  # preserved, not dropped
        P, A = m.decision_matrix(X)
        assert P.shape[1] == 3
        assert np.all(np.isnan(P[:, 0]))  # unmodelable column is NaN
        assert not A[:, 0].any()          # never accept into an unmodelable class

    def test_false_rejection_calibration(self):
        # §9.5 (empirically honest): a WELL-SAMPLED class (n=100) calibrates near
        # alpha=0.05; a small-but-modelable class (n=30) over-rejects because the
        # DD-SIMCA method-of-moments chi^2 fit is high-variance at small n (spec
        # 5.1) — bounded but not tight; the size-5 class is flagged unmodelable.
        # (Verified empirically: n=100->~0.05-0.07, n=30->~0.08-0.12, n=15->~0.39.)
        X, y = _blobs([5, 30, 100], seed=3)
        m = MultiClassClassModel(
            n_components=3, scaling="none", min_class_samples=10, alpha=0.05
        ).fit(X, y)
        assert 0 in m.unmodelable_
        rng = np.random.RandomState(99)
        rates = {}
        for k in (1, 2):  # modeled classes only
            X_test = rng.normal(size=(500, 20)) + _center(k)
            _, A = m.decision_matrix(X_test)
            col = list(m.classes_).index(k)
            rates[k] = 1.0 - A[:, col].mean()
        # well-sampled class calibrates near alpha
        assert 0.02 <= rates[2] <= 0.10, f"n=100 class reject={rates[2]:.3f}"
        # small-but-modelable class over-rejects but stays bounded (not catastrophic)
        assert rates[1] <= 0.15, f"n=30 class reject={rates[1]:.3f}"
        # and small-n over-rejects at least as much as well-sampled n
        assert rates[1] >= rates[2] - 0.02, f"{rates[1]:.3f} vs {rates[2]:.3f}"
