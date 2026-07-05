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

from spectral_predict.contamination import PCASIMCA
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


class TestScalingModesA3:
    """A3: scaling modes (none / per_class / global), all fit train-only inside fit.

    per_class is the SIMCA-textbook default (column-autoscale within each class);
    none reproduces bare PCASIMCA exactly (the functional-equivalence anchor, §8/§9.3);
    global fits one scaler across all classes for cross-engine comparability.
    """

    def test_default_scaling_is_per_class(self):
        # A3 flips the constructor default from "none" (A2) to "per_class".
        assert MultiClassClassModel().scaling == "per_class"

    def test_functional_equivalence_scaling_none(self):
        # A single-class model with scaling="none" is identical (accept/reject and
        # p-value) to a bare PCASIMCA fit on the same rows.
        rng = np.random.RandomState(5)
        X = rng.normal(size=(60, 20)) + _center(0)
        y = np.zeros(60, dtype=int)
        m = MultiClassClassModel(
            engine="pca-simca", alpha=0.05, n_components=3, scaling="none"
        ).fit(X, y)
        ref = PCASIMCA(n_components=3, alpha=0.05).fit(X)
        X_test = rng.normal(size=(200, 20)) + _center(0)
        P, A = m.decision_matrix(X_test)
        np.testing.assert_array_equal(P[:, 0], ref.p_joint(X_test))
        np.testing.assert_array_equal(A[:, 0], ref.predict(X_test) == 1)

    def test_per_class_scaler_fit_on_class_rows_only(self):
        # per_class: one StandardScaler per class, fit on that class's rows only.
        X, y = _blobs([40, 40, 40])
        m = MultiClassClassModel(n_components=3, scaling="per_class").fit(X, y)
        for c in m.classes_:
            assert c in m.scalers_
            np.testing.assert_allclose(m.scalers_[c].mean_, X[y == c].mean(axis=0))

    def test_global_scaler_fit_on_all_rows(self):
        # global: one StandardScaler fit on all training rows across classes.
        X, y = _blobs([40, 40, 40])
        m = MultiClassClassModel(n_components=3, scaling="global").fit(X, y)
        np.testing.assert_allclose(m.global_scaler_.mean_, X.mean(axis=0))

    def test_per_class_scaling_changes_decisions(self):
        # On features with wildly different scales, per_class autoscaling changes
        # the decision vs no scaling — proving the scaler is actually applied.
        rng = np.random.RandomState(6)
        scale = np.geomspace(1.0, 1000.0, 20)
        X = rng.normal(size=(80, 20)) * scale + _center(0)
        y = np.zeros(80, dtype=int)
        X_test = rng.normal(size=(200, 20)) * scale + _center(0)
        none = MultiClassClassModel(n_components=3, scaling="none").fit(X, y)
        perc = MultiClassClassModel(n_components=3, scaling="per_class").fit(X, y)
        _, a_none = none.decision_matrix(X_test)
        _, a_perc = perc.decision_matrix(X_test)
        assert not np.array_equal(a_none[:, 0], a_perc[:, 0])


class TestNonSIMCAEnginesA4:
    """A4: pluggable per-class engines (OCSVM/IsolationForest/LOF/EllipticEnvelope),
    each calibrated to a per-class empirical p-value so accept/reject is a real
    level-alpha test (spec 5.3). p = (1 + #{null <= s}) / (m + 1) over a cross-fit
    null of the engine's "higher = more normal" score.
    """

    @pytest.mark.parametrize(
        "engine", ["ocsvm", "isolation-forest", "lof", "elliptic-envelope"]
    )
    def test_calibrated_false_rejection(self, engine):
        # Held-out inliers from the training distribution are accepted at ~1-alpha.
        rng = np.random.RandomState(10)
        X_train = rng.normal(size=(150, 12))
        y_train = np.zeros(150, dtype=int)
        X_test = rng.normal(size=(2000, 12))
        alpha = 0.05
        m = MultiClassClassModel(
            engine=engine, alpha=alpha, scaling="none", min_class_samples=10
        ).fit(X_train, y_train)
        _, A = m.decision_matrix(X_test)
        accept = A[:, 0].mean()
        assert (1 - alpha) - 0.10 <= accept <= (1 - alpha) + 0.10, f"{engine}: {accept:.3f}"

    def test_pvalues_in_unit_interval(self):
        rng = np.random.RandomState(12)
        X = rng.normal(size=(120, 12))
        m = MultiClassClassModel(engine="ocsvm", scaling="none").fit(
            X, np.zeros(120, dtype=int)
        )
        P, _ = m.decision_matrix(rng.normal(size=(300, 12)))
        finite = np.isfinite(P)
        assert np.all(P[finite] >= 0.0) and np.all(P[finite] <= 1.0)

    def test_isolationforest_direction_not_inverted(self):
        # The sign bug GPT-5.5 caught: a far outlier must get a LOWER p-value
        # (more anomalous) than a central inlier. If score_samples were negated,
        # this inverts and the assertion fails.
        rng = np.random.RandomState(11)
        X_train = rng.normal(size=(150, 12))
        m = MultiClassClassModel(
            engine="isolation-forest", alpha=0.05, scaling="none"
        ).fit(X_train, np.zeros(150, dtype=int))
        P_in, _ = m.decision_matrix(np.zeros((1, 12)))          # center = most normal
        P_out, _ = m.decision_matrix(np.full((1, 12), 20.0))    # far = most abnormal
        assert P_out[0, 0] < P_in[0, 0]


class TestNestedCVA5:
    """A5: per-class n_components tuning ("per_class_cv", the new default) via
    one-vs-rest CV, and a nested leakage-safe outer CV (cross_validate). alpha
    stays global (never tuned).
    """

    def test_per_class_cv_is_default(self):
        # A5 flips the constructor default from the A2 placeholder int to
        # "per_class_cv".
        assert MultiClassClassModel().n_components == "per_class_cv"

    def test_per_class_cv_resolves_per_class_components(self):
        # "per_class_cv" tunes each class's n_components by one-vs-rest CV and
        # records the choice in n_components_ (a {class: int} dict).
        X, y = _blobs([60, 60, 60])
        m = MultiClassClassModel(
            engine="pca-simca", n_components="per_class_cv", scaling="none"
        ).fit(X, y)
        assert set(m.n_components_) == set(m.classes_.tolist())
        for c in m.classes_:
            assert isinstance(m.n_components_[c], (int, np.integer))
            assert m.n_components_[c] >= 1
        P, A = m.decision_matrix(X)
        assert P.shape == (len(X), 3)

    def test_cross_validate_covers_all_samples(self):
        X, y = _blobs([60, 60, 60])
        m = MultiClassClassModel(engine="pca-simca", n_components=5, scaling="none")
        res = m.cross_validate(X, y, n_splits=5)
        assert len(res["labels"]) == len(X)
        P, A = res["decision_matrix"]
        assert P.shape == (len(X), 3) and A.shape == (len(X), 3)
        # every sample belongs to exactly one outer test fold
        assert sorted(np.concatenate([f for f in res["test_indices"]])) == list(range(len(X)))

    def test_cross_validate_no_outer_leakage(self):
        # Structural leakage pin: each sample's out-of-fold prediction must come
        # from a fold whose TRAINING set excluded it. (A.1 per-class tuning is
        # then nested-safe by construction, since fit() only sees fold-train.)
        X, y = _blobs([60, 60, 60])
        res = MultiClassClassModel(
            engine="pca-simca", n_components="per_class_cv", scaling="none"
        ).cross_validate(X, y, n_splits=5)
        for f, test_idx in enumerate(res["test_indices"]):
            train_idx = set(res["train_indices"][f].tolist())
            assert not (set(test_idx.tolist()) & train_idx)  # disjoint
        # and the union of train+test for each fold is the whole dataset
        for f, test_idx in enumerate(res["test_indices"]):
            assert len(res["train_indices"][f]) + len(test_idx) == len(X)
