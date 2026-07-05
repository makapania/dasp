"""Tests for spectral_predict.simca — T-31 multi-class class modeling.

A2: MultiClassClassModel core (per-class DD-SIMCA membership + decision matrix +
predict summary labels + min_class_samples unmodelable handling).

These tests define the A2 API contract. scaling="none" is used explicitly
throughout (per_class/global scaling modes arrive in A3, which also flips the
constructor default to "per_class").
"""

from __future__ import annotations

import warnings

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


class TestWoldPlotDataB2:
    """B2: Wold diagnostic plot data — per-variable MPOW/DPOW arrays in the
    correct (K, n_features) shape and class order for the Phase-D GUI to render.
    """

    def test_plot_data_shapes_and_class_order(self):
        from spectral_predict.simca import (
            wold_diagnostic_plot_data,
            wold_variable_powers,
        )

        X, y, _ = _graded([50, 50, 50], seed=0)
        K, p = 3, X.shape[1]
        data = wold_diagnostic_plot_data(X, y, n_components=2, scaling="none")
        assert list(data["classes"]) == list(np.unique(y))
        assert data["modeling_power"].shape == (K, p)
        assert data["discriminating_power"].shape == (K, p)
        assert data["modeling_power_agg"].shape == (p,)
        assert data["discriminating_power_agg"].shape == (p,)
        assert len(data["variables"]) == p
        # row k must correspond to class classes[k]
        powers = wold_variable_powers(X, y, n_components=2, scaling="none")
        for k, c in enumerate(data["classes"]):
            np.testing.assert_allclose(
                data["modeling_power"][k], powers["modeling_power_per_class"][c]
            )
            np.testing.assert_allclose(
                data["discriminating_power"][k],
                powers["discriminating_power_per_class"][c],
            )
        np.testing.assert_allclose(
            data["modeling_power_agg"], powers["modeling_power"]
        )

    def test_plot_data_wavelength_axis(self):
        from spectral_predict.simca import wold_diagnostic_plot_data

        X, y, _ = _graded([40, 40, 40], seed=1)
        wl = np.linspace(1000.0, 2500.0, X.shape[1])
        data = wold_diagnostic_plot_data(
            X, y, n_components=2, scaling="none", wavelengths=wl
        )
        np.testing.assert_allclose(np.asarray(data["variables"], dtype=float), wl)

    def test_plot_data_defaults_variables_to_indices(self):
        from spectral_predict.simca import wold_diagnostic_plot_data

        X, y, _ = _graded([40, 40, 40], seed=2)
        data = wold_diagnostic_plot_data(X, y, n_components=2, scaling="none")
        np.testing.assert_array_equal(
            np.asarray(data["variables"]), np.arange(X.shape[1])
        )


def _novel_split(seed=0, n_features=30, sep=14.0, n_far=56, n_inlier=24):
    """3 known graded classes (train) + an EXTERNAL novel set that is a mixture
    of broadly-shifted far-novel samples and a minority of genuine class-2-like
    inliers, so the external novelty rate is non-trivial (~0.73 baseline, not a
    trivial 1.0) — making the supervised-varsel novelty guard discriminating."""
    rng = np.random.RandomState(seed)
    strength = np.geomspace(sep, sep / 8.0, n_features)
    loading = rng.normal(size=n_features)
    Xs, ys = [], []
    for k in range(3):
        X = (
            rng.normal(scale=0.3, size=(80, n_features))
            + rng.normal(size=(80, 1)) * loading
            + (k + 1) * strength
        )
        Xs.append(X)
        ys.append(np.full(80, k))
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    far = (
        rng.normal(scale=0.3, size=(n_far, n_features))
        + rng.normal(size=(n_far, 1)) * loading
        + 6.0 * strength
    )
    inl = (
        rng.normal(scale=0.3, size=(n_inlier, n_features))
        + rng.normal(size=(n_inlier, 1)) * loading
        + 3.0 * strength
    )
    return Xtr, ytr, np.vstack([far, inl])


class TestVarselIntegrationB3:
    """B3: variable_selection integration on MultiClassClassModel — Wold modes
    (from B1) and a supervised prefilter on the genuine multi-class label,
    tagged varsel_path_, mask computed train-only at fit and applied before the
    per-class models. The supervised path is gated by an external-novel-class
    guard: it must NOT degrade novelty below the full-spectra baseline (spec §5
    guardrail / §9.9).
    """

    def test_no_varsel_is_default_and_untagged(self):
        X, y = _blobs([40, 40, 40])
        m = MultiClassClassModel(n_components=3, scaling="none").fit(X, y)
        assert m.varsel_path_ == "none"
        assert m.varsel_mask_ is None

    def test_wold_varsel_selects_subspace_and_predicts_on_full_width(self):
        X, y, _ = _graded([60, 60, 60], seed=0)
        m = MultiClassClassModel(
            n_components=3,
            scaling="none",
            variable_selection="wold_balanced",
            n_select=10,
        ).fit(X, y)
        assert m.varsel_path_ == "wold"
        assert m.varsel_mask_.shape == (X.shape[1],)
        assert m.varsel_mask_.dtype == bool
        assert int(m.varsel_mask_.sum()) == 10
        # per-class engines are fit on the 10-var subspace
        for c in m.classes_:
            if c not in m.unmodelable_:
                assert m.models_[c].pca_.n_features_in_ == 10
        # decision_matrix / predict accept FULL-width X and mask internally
        P, A = m.decision_matrix(X)
        assert P.shape == (len(X), 3)
        assert len(m.predict(X)) == len(X)

    def test_supervised_varsel_tagged_and_masked(self):
        X, y, _ = _graded([60, 60, 60], seed=1)
        m = MultiClassClassModel(
            n_components=3,
            scaling="none",
            variable_selection="importance",
            n_select=12,
            varsel_model_name="RandomForest",
        ).fit(X, y)
        assert m.varsel_path_ == "supervised"
        assert int(m.varsel_mask_.sum()) == 12
        for c in m.classes_:
            if c not in m.unmodelable_:
                assert m.models_[c].pca_.n_features_in_ == 12
        assert m.decision_matrix(X)[0].shape == (len(X), 3)

    def test_supervised_prefilter_does_not_degrade_novelty(self):
        # §9.9 guard, STRENGTHENED per the Phase-B gate (MiniMax H2 / Kimi M6):
        # multi-seed PAIRED comparison across several n_select values, deterministic
        # (RF importances are seeded via build_model random_state=42, and all data
        # is seeded). Assert the MEAN over-seeds gap does not degrade novelty by
        # more than 0.05 for any n_select. Empirically probed: mean gaps
        # {n_select 5: -0.029, 12: -0.004, 20: +0.001}; worst single-seed -0.088.
        params = dict(engine="pca-simca", n_components=5, scaling="per_class")
        for n_select in (5, 12, 20):
            gaps, fulls = [], []
            for seed in range(10):
                Xtr, ytr, Xnov = _novel_split(seed=seed)
                full = MultiClassClassModel(**params).evaluate_novelty(
                    Xtr, ytr, mode="external", external_X=Xnov
                )
                sup = MultiClassClassModel(
                    variable_selection="importance",
                    n_select=n_select,
                    varsel_model_name="RandomForest",
                    **params,
                ).evaluate_novelty(Xtr, ytr, mode="external", external_X=Xnov)
                gaps.append(sup - full)
                fulls.append(full)
            assert 0.6 <= np.mean(fulls) <= 0.9, (
                f"baseline novelty {np.mean(fulls):.3f} not a real test"
            )
            assert np.mean(gaps) >= -0.05, (
                f"n_select={n_select}: supervised degraded novelty by "
                f"{-np.mean(gaps):.3f} (mean over 10 seeds)"
            )

    def test_precomputed_mask_hook(self):
        # A boolean array is accepted directly as the mask (varsel_path_ =
        # "precomputed") — the extensibility hook so the C search layer can wire
        # ANY supervised method by computing the mask externally and passing it in.
        X, y, _ = _graded([60, 60, 60], seed=3)
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[[0, 1, 2, 3, 4]] = True
        m = MultiClassClassModel(
            n_components=3, scaling="none", variable_selection=mask
        ).fit(X, y)
        assert m.varsel_path_ == "precomputed"
        np.testing.assert_array_equal(m.varsel_mask_, mask)
        for c in m.classes_:
            if c not in m.unmodelable_:
                assert m.models_[c].pca_.n_features_in_ == 5
        assert m.decision_matrix(X)[0].shape == (len(X), 3)

    def test_unimplemented_supervised_method_raises(self):
        # Extensible dispatcher: model-layer supervised path ships "importance";
        # the fuller method set (spa/cars/ga/...) is enumerated in the C search
        # layer. An unwired name raises rather than silently mis-selecting.
        X, y, _ = _graded([50, 50, 50], seed=4)
        with pytest.raises((NotImplementedError, ValueError)):
            MultiClassClassModel(
                n_components=3, scaling="none", variable_selection="spa", n_select=10
            ).fit(X, y)

    def test_wold_varsel_recomputed_on_fold_train_only(self):
        # STRENGTHENED leakage pin (Codex LOW): spy on wold_variable_selection and
        # assert it is called once per fold on FOLD-TRAIN rows only (row count <
        # full n), never on the full dataset — catches a regression that cached a
        # full-data mask on self and reused it across folds.
        import spectral_predict.simca as simca_mod

        X, y, _ = _graded([60, 60, 60], seed=2)
        n_total = len(X)
        seen_row_counts = []
        orig = simca_mod.wold_variable_selection

        def _spy(X_arg, *a, **k):
            seen_row_counts.append(np.asarray(X_arg).shape[0])
            return orig(X_arg, *a, **k)

        simca_mod.wold_variable_selection = _spy
        try:
            res = MultiClassClassModel(
                n_components=3,
                scaling="none",
                variable_selection="wold_balanced",
                n_select=10,
            ).cross_validate(X, y, n_splits=5)
        finally:
            simca_mod.wold_variable_selection = orig
        assert len(res["labels"]) == n_total
        assert len(seen_row_counts) == 5  # once per fold
        assert all(rc < n_total for rc in seen_row_counts), seen_row_counts


class TestPhaseBGateFoldins:
    """Phase-B multi-family gate fold-ins (Codex + Kimi + MiniMax): varsel
    robustness (below-floor class crash, empty/invalid n_select), reproducibility,
    balanced-score normalization, and non-finite importance guarding.
    """

    def test_wold_varsel_below_floor_class_does_not_crash(self):
        # Codex HIGH / Kimi M3: Wold power estimation ran on ALL classes before
        # the unmodelable floor, so a below-floor class hit KFold(n_splits=1) and
        # crashed. Varsel must use only modelable classes; the small class is still
        # preserved as an unmodelable decision-matrix column.
        rng = np.random.RandomState(0)
        X = np.vstack(
            [
                rng.normal(size=(3, 24)),
                rng.normal(size=(30, 24)) + 6,
                rng.normal(size=(30, 24)) + 12,
            ]
        )
        y = np.array([0] * 3 + [1] * 30 + [2] * 30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MultiClassClassModel(
                n_components=3,
                scaling="none",
                min_class_samples=10,
                variable_selection="wold_balanced",
                n_select=8,
            ).fit(X, y)
        assert 0 in m.unmodelable_
        assert list(m.classes_) == [0, 1, 2]
        assert int(m.varsel_mask_.sum()) == 8
        P, _ = m.decision_matrix(X)
        assert P.shape[1] == 3 and np.all(np.isnan(P[:, 0]))

    @pytest.mark.parametrize("bad", [0, -3])
    def test_invalid_n_select_raises(self, bad):
        # Kimi H1/L8: n_select <= 0 must raise a clear error, not silently produce
        # an empty (n_select=0) or almost-all (negative slice) mask.
        X, y, _ = _graded([40, 40, 40], seed=1)
        with pytest.raises(ValueError):
            MultiClassClassModel(
                n_components=3,
                scaling="none",
                variable_selection="wold_balanced",
                n_select=bad,
            ).fit(X, y)

    def test_non_finite_importances_raise(self):
        # Kimi #7: a supervised model returning NaN/inf importances must raise,
        # not build an arbitrary mask from `imp >= mean(imp)`.
        import spectral_predict.unified_bayesian as ub

        X, y, _ = _graded([40, 40, 40], seed=2)
        orig = ub.compute_importances

        def _bad(*a, **k):
            out = np.ones(X.shape[1])
            out[0] = np.nan
            return out

        ub.compute_importances = _bad
        try:
            with pytest.raises(ValueError):
                MultiClassClassModel(
                    n_components=3,
                    scaling="none",
                    variable_selection="importance",
                    n_select=10,
                ).fit(X, y)
        finally:
            ub.compute_importances = orig

    def test_wold_pca_selection_is_deterministic(self):
        # Kimi H2: Wold PCA calls must pin random_state so the mask is identical
        # across runs even when sklearn would pick the randomized SVD solver.
        from spectral_predict.simca import wold_variable_selection

        rng = np.random.RandomState(0)
        X = rng.normal(size=(650, 40))
        X[:200, :6] += 8.0
        X[200:400, :6] += 4.0
        y = np.array([0] * 200 + [1] * 200 + [2] * 250)
        m1 = wold_variable_selection(X, y, mode="balanced", n_components=3, n_select=10)
        m2 = wold_variable_selection(X, y, mode="balanced", n_components=3, n_select=10)
        np.testing.assert_array_equal(m1, m2)


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
