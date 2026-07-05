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
from spectral_predict.model_io import load_model, predict_with_model, save_model
from spectral_predict.simca import (
    MultiClassClassModel,
    multiclass_simca_metrics,
    novelty_tradeoff_auc,
    wilson_ci,
)


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


def _graded(sizes, n_features=24, n_relevant=6, seed=0, sep=14.0):
    """K classes whose features carry a GRADED between-class separation (feature
    0 strongest, geometrically decreasing) plus a shared within-class latent
    structure. The gradient gives a genuine, recoverable discriminating-power
    ranking (so a stability test is meaningful) and high modeling power; the
    top-``n_relevant`` separation features are returned as "relevant"."""
    rng = np.random.RandomState(seed)
    strength = np.geomspace(sep, sep / 8.0, n_features)  # distinct, decreasing
    loading = rng.normal(size=n_features)
    Xs, ys = [], []
    for k, n in enumerate(sizes):
        X = rng.normal(scale=0.3, size=(n, n_features))
        t = rng.normal(size=(n, 1))
        X += t * loading            # shared within-class latent -> high MPOW
        X += (k + 1) * strength     # between-class separation, graded per feature
        Xs.append(X)
        ys.append(np.full(n, k))
    relevant = np.argsort(strength)[::-1][:n_relevant]
    return np.vstack(Xs), np.concatenate(ys), relevant


class TestWoldVarselB1:
    """B1: Wold (1976) modeling power + discriminating power variable selection
    (spec §5.6). MPOW_j = 1 - s_resid,j / s_total,j (classical Wold, distinct
    from the DD-SIMCA per-variable T^2/Q framework). Discriminating power =
    macro-averaged one-vs-rest cross-fit residual-RMS ratios. Selection modes:
    modeling / discriminating / balanced (max MPOW*DPOW). All varsel here is
    legitimate class-modeling-native math on the genuine multi-class label —
    the one-class UVE/iPLS exclusion (CLAUDE.md) does NOT apply to the true
    multi-class case.
    """

    def test_modeling_power_matches_reference(self):
        # MPOW hand/reference value on a fixed matrix: 1 - std(resid)/std(raw)
        # per variable, residual from an INDEPENDENT sklearn PCA reconstruction.
        from sklearn.decomposition import PCA

        from spectral_predict.simca import wold_modeling_power

        X = np.arange(1, 33, dtype=float).reshape(8, 4) % 7 + np.array(
            [0.0, 1.0, 2.0, 3.0]
        )
        nc = 2
        mp = wold_modeling_power(X, n_components=nc)
        pca = PCA(n_components=nc).fit(X)
        resid = X - pca.inverse_transform(pca.transform(X))
        expected = 1.0 - resid.std(axis=0, ddof=0) / X.std(axis=0, ddof=0)
        assert mp.shape == (4,)
        np.testing.assert_allclose(mp, expected, atol=1e-9)

    def test_modeling_power_zero_variance_guard(self):
        # A constant (zero-variance) column has no variance to model -> MPOW 0,
        # never NaN/inf from a divide-by-zero.
        from spectral_predict.simca import wold_modeling_power

        rng = np.random.RandomState(0)
        X = rng.normal(size=(20, 5))
        X[:, 2] = 4.0  # constant column
        mp = wold_modeling_power(X, n_components=2)
        assert np.all(np.isfinite(mp))
        assert mp[2] == pytest.approx(0.0)

    def test_discriminating_power_ranking_stable(self):
        # DPOW ranking is stable across resamples: each resample's per-variable
        # aggregate DPOW correlates with the consensus (mean) ranking at
        # Spearman rho >= 0.8 (median over 20 resamples). (spec §5.6 / §9.10)
        from scipy.stats import spearmanr

        from spectral_predict.simca import wold_variable_powers

        dpows = []
        for seed in range(20):
            X, y, _ = _graded([80, 80, 80], seed=seed)
            powers = wold_variable_powers(X, y, n_components=2, scaling="none")
            dpows.append(np.asarray(powers["discriminating_power"]))
        arr = np.vstack(dpows)
        consensus = arr.mean(axis=0)
        rhos = [spearmanr(consensus, arr[i]).correlation for i in range(len(dpows))]
        assert np.median(rhos) >= 0.8, f"median consensus rho={np.median(rhos):.3f}"

    def test_balanced_retains_relevant_variables(self):
        # "balanced" (max MPOW*DPOW) retains >= 2/3 of the known-relevant
        # variables in its selected set. (spec §5.6 / §9.10)
        from spectral_predict.simca import wold_variable_selection

        X, y, relevant = _graded([80, 80, 80], seed=1)
        mask = wold_variable_selection(
            X, y, mode="balanced", n_components=2, n_select=8, scaling="none"
        )
        assert mask.shape == (X.shape[1],)
        assert mask.dtype == bool
        kept = set(np.where(mask)[0].tolist())
        retained = len(kept & set(int(i) for i in relevant))
        assert retained >= int(np.ceil(2 / 3 * len(relevant))), (
            f"retained {retained}/{len(relevant)} relevant"
        )

    def test_selection_modes_return_masks(self):
        # All three modes return a boolean mask of the requested size.
        from spectral_predict.simca import wold_variable_selection

        X, y, _ = _graded([60, 60, 60], seed=2)
        for mode in ("modeling", "discriminating", "balanced"):
            mask = wold_variable_selection(
                X, y, mode=mode, n_components=2, n_select=10, scaling="none"
            )
            assert mask.shape == (X.shape[1],)
            assert mask.dtype == bool
            assert int(mask.sum()) == 10, mode

    def test_per_class_scaling_produces_finite_powers(self):
        # scaling="per_class" (SIMCA-textbook) path yields finite, sane powers.
        from spectral_predict.simca import wold_variable_powers

        X, y, _ = _graded([60, 60, 60], seed=3)
        powers = wold_variable_powers(X, y, n_components=2, scaling="per_class")
        for key in ("modeling_power", "discriminating_power"):
            arr = np.asarray(powers[key])
            assert arr.shape == (X.shape[1],)
            assert np.all(np.isfinite(arr))


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


class TestNoveltyModeA6:
    """A6: novelty evaluation. LOCO holds out each class (refit on the rest) and
    measures how often the held-out class is flagged "novel" (none-of-the-above);
    external mode fits on the known classes and scores a separate held-out set.
    """

    def test_loco_holds_out_vs_insample(self):
        X, y = _blobs([80, 80, 80], sep=12.0)
        params = dict(engine="pca-simca", n_components=5, scaling="per_class")
        loco = MultiClassClassModel(**params).evaluate_novelty(X, y, mode="loco")
        assert set(loco) == set(np.unique(y).tolist())
        assert min(loco.values()) >= 0.8  # a held-out class is flagged novel
        # in-sample (all classes modeled) accepts its own members
        insample = np.mean(
            MultiClassClassModel(**params).fit(X, y).predict(X) == "novel"
        )
        assert insample <= 0.15

    def test_external_novelty_mode(self):
        X, y = _blobs([80, 80, 80], sep=12.0)
        rng = np.random.RandomState(4)
        X_ext = rng.normal(size=(80, 20))
        X_ext[:, 9] += 120.0  # far, unseen region of feature space
        nov = MultiClassClassModel(
            engine="pca-simca", n_components=5, scaling="per_class"
        ).evaluate_novelty(X, y, mode="external", external_X=X_ext)
        assert nov >= 0.8

    def test_discriminant_baseline_forces_a_class(self):
        # Acceptance contrast (§9.1): SIMCA flags the novel class as "none of the
        # above"; a discriminant baseline (LDA, like PLS-DA) forces (almost) all
        # of it into a trained class — it can never abstain.
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X, y = _blobs([80, 80, 80], sep=12.0)
        rng = np.random.RandomState(4)
        X_ext = rng.normal(size=(80, 20))
        X_ext[:, 9] += 120.0
        simca = MultiClassClassModel(
            engine="pca-simca", n_components=5, scaling="per_class"
        ).fit(X, y)
        assert np.mean(simca.predict(X_ext) == "novel") >= 0.8
        lda = LinearDiscriminantAnalysis().fit(X, y)
        forced = np.mean(np.isin(lda.predict(X_ext), np.unique(y)))
        assert forced >= 0.9


class TestMetricsA7:
    """A7: dedicated multilabel/class-model metrics (spec section 7). NOT the
    inverted one_class_metrics, NOT single-label compute_imbalance_metrics.
    """

    # A fully hand-worked 6-sample, 2-class (classes [0,1]) decision matrix.
    # Rows: (true label, accept flags [class0, class1]). Label 9 = truly novel
    # (not in classes).
    _CLASSES = [0, 1]
    _Y = np.array([0, 0, 1, 1, 9, 9])
    _A = np.array(
        [
            [True, False],   # s0 true0 -> {0}         correct
            [False, False],  # s1 true0 -> {}          false-novel
            [False, True],   # s2 true1 -> {1}         correct
            [True, True],    # s3 true1 -> {0,1}       ambiguous
            [False, False],  # s4 novel -> {}          correct novelty
            [True, False],   # s5 novel -> {0}         missed novelty
        ]
    )

    def test_metrics_hand_computed(self):
        m = multiclass_simca_metrics(self._Y, self._A, self._CLASSES)
        assert m["per_class_sensitivity"][0] == pytest.approx(0.5)   # s0,s1 -> 1/2
        assert m["per_class_sensitivity"][1] == pytest.approx(1.0)   # s2,s3 -> 2/2
        assert m["per_class_specificity"][0] == pytest.approx(0.5)   # 2/4 non-0 rejected
        assert m["per_class_specificity"][1] == pytest.approx(1.0)   # 4/4 non-1 rejected
        assert m["novelty_detection_rate"] == pytest.approx(0.5)     # s4,s5 -> 1/2
        assert m["no_class_rate"] == pytest.approx(2 / 6)            # s1,s4
        assert m["ambiguity_rate"] == pytest.approx(1 / 6)           # s3
        assert m["exact_set_rate"] == pytest.approx(0.5)             # s0,s2,s4
        assert m["efficiency"] == pytest.approx(0.75)                # geomean(0.75,0.75)

    def test_wilson_ci_brackets_estimate(self):
        lo, hi = wilson_ci(5, 100)
        assert 0.0 <= lo < 0.05 < hi <= 1.0
        lo0, hi0 = wilson_ci(0, 10)
        assert lo0 >= 0.0 and hi0 > 0.0     # one-sided-ish; lower bound never negative

    def test_novelty_tradeoff_auc_bounds_and_separation(self):
        # Known class-0 samples score high p for class 0, ~0 for others; truly
        # novel samples score ~0 everywhere -> the alpha-sweep AUC is high.
        classes = [0, 1]
        y = np.array([0] * 50 + [9] * 50)
        P = np.zeros((100, 2))
        P[:50, 0] = 0.9          # known-0 accepted by class 0
        P[:50, 1] = 0.01
        P[50:, :] = 0.01         # novel: low everywhere
        auc = novelty_tradeoff_auc(y, P, classes)
        assert 0.0 <= auc <= 1.0
        assert auc >= 0.8


class TestPersistenceA8:
    """A8: .dasp round-trip for the whole MultiClassClassModel orchestrator +
    a decision-matrix prediction schema + a forward-compat task_type gate (§6).
    """

    def _fit(self):
        X, y = _blobs([60, 60, 60])
        m = MultiClassClassModel(
            engine="pca-simca", n_components=5, scaling="per_class"
        ).fit(X, y)
        return m, X, y

    def _meta(self, m):
        return {
            "model_name": "MultiClassSIMCA",
            "task_type": "multiclass_simca",
            "wavelengths": list(range(20)),
            "n_vars": 20,
            "class_names": [int(c) for c in m.classes_],
            "alpha": m.alpha,
            "scaling": m.scaling,
            "engine_family": m.engine,
        }

    def test_save_load_roundtrip_reproduces_decision_matrix(self, tmp_path):
        m, X, y = self._fit()
        fp = tmp_path / "mc.dasp"
        save_model(m, None, self._meta(m), fp)
        out = predict_with_model(load_model(fp), X, validate_wavelengths=False)
        assert set(out) >= {
            "p_values", "decision_matrix", "summary_label", "accepted_classes"
        }
        P_ref, A_ref = m.decision_matrix(X)
        np.testing.assert_allclose(out["p_values"], P_ref, equal_nan=True)
        np.testing.assert_array_equal(out["decision_matrix"], A_ref)
        np.testing.assert_array_equal(out["summary_label"], m.predict(X))
        assert len(out["accepted_classes"]) == len(X)

    def test_unknown_task_type_raises(self, tmp_path):
        m, X, y = self._fit()
        meta = self._meta(m)
        meta["task_type"] = "some_future_task"
        fp = tmp_path / "future.dasp"
        save_model(m, None, meta, fp)
        with pytest.raises(NotImplementedError):
            predict_with_model(load_model(fp), X, validate_wavelengths=False)


class TestPhaseAHardening:
    """Phase-A review-gate fixes (Codex + DeepSeek + Kimi 3-family panel):
    layered n_components-aware calibration floor + NaN-metric bugs + engine_params.
    """

    def test_nonsimca_requires_20_to_reject(self):
        # DeepSeek C1: empirical p floor is 1/(m+1); at m<20 it exceeds alpha=0.05
        # so non-SIMCA engines can never reject -> such classes are unmodelable.
        X, y = _blobs([15, 60, 60])
        m = MultiClassClassModel(
            engine="ocsvm", scaling="none", min_class_samples=10
        ).fit(X, y)
        assert 0 in m.unmodelable_          # n=15 < 20 -> can't reject -> unmodelable
        assert 1 not in m.unmodelable_ and 2 not in m.unmodelable_

    def test_simca_smalln_warns_but_models(self):
        # DD-SIMCA over-rejects at small n: warn (n_components-aware) but still model.
        X, y = _blobs([15, 60, 60])   # n=15 < max(20, 5*5=25)
        with pytest.warns(UserWarning):
            m = MultiClassClassModel(
                engine="pca-simca", n_components=5, scaling="none", min_class_samples=10
            ).fit(X, y)
        assert 0 not in m.unmodelable_    # modeled, just warned

    def test_all_unmodelable_raises(self):
        X, y = _blobs([5, 6, 7])          # all below hard floor 10
        with pytest.raises(ValueError):
            MultiClassClassModel(engine="pca-simca", min_class_samples=10).fit(X, y)

    def test_engine_params_forwarded(self):
        X, y = _blobs([60, 60, 60])
        m = MultiClassClassModel(
            engine="isolation-forest", engine_params={"n_estimators": 37}, scaling="none"
        ).fit(X, y)
        assert m.models_[0].n_estimators == 37

    def test_efficiency_ignores_absent_class(self):
        # class 2 present in `classes` but absent from y_true -> its sensitivity is
        # NaN; efficiency must stay finite (nanmean, not mean).
        y = np.array([0, 0, 1, 1])
        A = np.array(
            [[True, False, False], [True, False, False],
             [False, True, False], [False, True, False]]
        )
        met = multiclass_simca_metrics(y, A, [0, 1, 2])
        assert np.isnan(met["per_class_sensitivity"][2])
        assert np.isfinite(met["efficiency"])

    def test_novelty_auc_with_unmodelable_nan_column(self):
        # An all-NaN (unmodelable) class column must not collapse the AUC to 0.
        classes = [0, 1]
        y = np.array([0] * 50 + [9] * 50)
        P = np.full((100, 2), np.nan)
        P[:50, 0] = 0.9      # known-0 accepted by class 0
        P[50:, 0] = 0.01     # novel low on class 0
        # class 1 column stays all-NaN (unmodelable)
        auc = novelty_tradeoff_auc(y, P, classes)
        assert 0.0 <= auc <= 1.0 and auc >= 0.8
