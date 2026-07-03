"""Tests for the T-17 multi-target varsel adapters, guards, and grid orchestration."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20260702)


@pytest.fixture
def xy_multi(rng):
    n, p = 50, 40
    X = rng.standard_normal((n, p))
    base = X[:, :4] @ rng.standard_normal((4, 3))
    Y = base + 0.05 * rng.standard_normal((n, 3))
    wl = np.linspace(1000.0, 2000.0, p)
    return X, Y, wl


def test_ipls_selection_rejects_2d_y(xy_multi):
    from spectral_predict.variable_selection import ipls_selection

    X, Y, _wl = xy_multi
    with pytest.raises(NotImplementedError):
        ipls_selection(X, Y)


def test_ipls_selection_single_y_still_works(xy_multi):
    from spectral_predict.variable_selection import ipls_selection

    X, Y, _wl = xy_multi
    # single-column 2-D and 1-D must NOT raise (guard fires only on >1 column).
    out_1d = ipls_selection(X, Y[:, 0])
    out_col = ipls_selection(X, Y[:, [0]])
    assert out_1d is not None
    assert out_col is not None


def test_vcpa_iriv_rejects_2d_y(xy_multi):
    from spectral_predict.wavelength_selection import vcpa_iriv

    X, Y, _wl = xy_multi
    with pytest.raises(NotImplementedError):
        vcpa_iriv(X, Y, n_outer_iterations=1, n_inner_iterations=2, binary_matrix_samples=4)


def test_parse_ipls_subset_limit():
    from spectral_predict.multitarget_grid import _parse_ipls_subset_limit

    assert _parse_ipls_subset_limit("Top 5") == 5
    assert _parse_ipls_subset_limit("Top 20") == 20
    assert _parse_ipls_subset_limit("All") is None


def test_interval_subset_adapter_returns_truncated_subsets(xy_multi):
    from spectral_predict.multitarget_grid import _interval_subset_adapter

    X, Y, wl = xy_multi
    subs = _interval_subset_adapter(
        "ipls_forward", X, Y, wl, ipls_subset_limit="Top 5"
    )
    assert 1 <= len(subs) <= 5
    for s in subs:
        assert s["method"] == "ipls_forward"
        assert isinstance(s["indices"], np.ndarray)
        assert s["indices"].size >= 1
        assert s["indices"].size < X.shape[1] or "iPLS" in s["tag"]
        assert isinstance(s["tag"], str)


def test_verify_spa_multi_y_safe_true_on_informative_block(xy_multi):
    from spectral_predict.multitarget_grid import verify_spa_multi_y_safe

    X, Y, _wl = xy_multi
    assert verify_spa_multi_y_safe(X, Y, n_features=5) is True


def test_verify_spa_multi_y_safe_shape_and_finiteness():
    from spectral_predict.variable_selection import spa_selection

    rng = np.random.default_rng(9)
    X = rng.standard_normal((40, 20))
    Y = X[:, :3] @ rng.standard_normal((3, 2)) + 0.05 * rng.standard_normal((40, 2))
    imp = np.asarray(spa_selection(X, Y, n_features=5), dtype=float)
    assert imp.shape == (20,)
    assert np.all(np.isfinite(imp))


def test_importances_to_subsets_top_n_filtered(xy_multi):
    from spectral_predict.multitarget_grid import _importances_to_subsets

    X, _Y, _wl = xy_multi
    imp = np.arange(X.shape[1], dtype=float)  # 40 features, strictly increasing
    subs = _importances_to_subsets(
        imp, "spa", variable_counts=[5, 10, 999], n_features_sub=X.shape[1]
    )
    # 999 >= 40 is filtered out; 5 and 10 remain.
    sizes = sorted(s["indices"].size for s in subs)
    assert sizes == [5, 10]
    top5 = next(s for s in subs if s["indices"].size == 5)
    assert set(top5["indices"].tolist()) == {35, 36, 37, 38, 39}  # highest importances
    assert top5["method"] == "spa"
    assert "top5" in top5["tag"]


def test_model_independent_importances_spa_and_ga(xy_multi):
    from spectral_predict.multitarget_grid import _model_independent_importances

    X, Y, _wl = xy_multi
    imp_spa = _model_independent_importances("spa", X, Y)
    assert imp_spa is not None and imp_spa.shape == (X.shape[1],)
    imp_ga = _model_independent_importances("ga", X, Y)
    assert imp_ga is not None and imp_ga.shape == (X.shape[1],)


def test_importance_reference_fit_tree_model(xy_multi):
    from spectral_predict.multitarget_grid import _importance_reference_fit

    X, Y, _wl = xy_multi
    imp = _importance_reference_fit("RandomForest", X, Y, min_fold_train=X.shape[0] - 1)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))


def test_fipls_spa_preserves_2d_y_when_spa_safe():
    """Strengthened (T-17 FIX 7a): proves multi-Y fipls_spa genuinely uses BOTH
    targets, not just the stronger one.

    Builds a block where the two targets are driven by DISJOINT, well-separated
    feature bands (blockA=[4,5,6], blockB=[30,31,32]) with comparably strong
    signals. The 2-D fit must reach BOTH bands, and the multi-Y importance must
    NOT be identical to either single-target run. The old test only checked
    shape + finiteness, which passes even if SPA silently collapsed to one
    target's band.
    """
    from spectral_predict.multitarget_grid import verify_spa_multi_y_safe
    from spectral_predict.variable_selection import fipls_spa_selection

    rng = np.random.default_rng(20260702)
    n, p = 60, 40
    wl = np.linspace(1000.0, 2000.0, p)
    X = rng.standard_normal((n, p))
    blockA = [4, 5, 6]
    blockB = [30, 31, 32]  # disjoint from blockA, well separated
    # Two comparably strong signals, each driven by its own disjoint band.
    Y0 = X[:, blockA] @ rng.standard_normal((3,)) + 0.02 * rng.standard_normal(n)
    Y1 = X[:, blockB] @ rng.standard_normal((3,)) + 0.02 * rng.standard_normal(n)
    Y = np.column_stack([Y0, Y1])

    if not verify_spa_multi_y_safe(X, Y, n_features=5):
        pytest.skip("SPA not 2-D-safe here")

    imp_multi = np.asarray(fipls_spa_selection(X, Y, wl), dtype=float)
    imp_t0 = np.asarray(fipls_spa_selection(X, Y[:, [0]], wl), dtype=float)
    imp_t1 = np.asarray(fipls_spa_selection(X, Y[:, [1]], wl), dtype=float)

    # Shape + finiteness (the old weak assertions, kept).
    assert imp_multi.shape == (p,)
    assert imp_t0.shape == (p,)
    assert imp_t1.shape == (p,)
    assert np.all(np.isfinite(imp_multi))
    assert np.all(np.isfinite(imp_t0))
    assert np.all(np.isfinite(imp_t1))

    # CORE STRENGTHENING 1: the 2-D fit reaches BOTH targets' bands. fipls_spa
    # zeros non-selected features, so nonzero(imp_multi) is the selected set.
    sel = set(np.nonzero(imp_multi)[0].tolist())
    assert sel & set(blockA), (
        f"multi-Y fipls_spa did not reach blockA {blockA}; selected={sorted(sel)}"
    )
    assert sel & set(blockB), (
        f"multi-Y fipls_spa did not reach blockB {blockB}; selected={sorted(sel)}"
    )

    # CORE STRENGTHENING 2: the multi-Y importance is genuinely multi-Y -- it is
    # NOT identical to either single-target run (would indicate SPA ignored one
    # target column entirely).
    assert not np.allclose(imp_multi, imp_t0), (
        "multi-Y fipls_spa importance is identical to the target-0-only run; "
        "the second target was not used."
    )
    assert not np.allclose(imp_multi, imp_t1), (
        "multi-Y fipls_spa importance is identical to the target-1-only run; "
        "the first target was not used."
    )


def test_classify_varsel_method():
    from spectral_predict.multitarget_grid import classify_varsel_method

    assert classify_varsel_method("ipls_forward", enabled_models=["PLS"], spa_ok=True) == "subset"
    assert classify_varsel_method("mwpls", enabled_models=["RandomForest"], spa_ok=True) == "subset"
    assert classify_varsel_method("spa", enabled_models=["PLS"], spa_ok=True) == "importance"
    assert classify_varsel_method("spa", enabled_models=["PLS"], spa_ok=False) == "skip"
    assert classify_varsel_method("fipls_spa", enabled_models=["PLS"], spa_ok=False) == "skip"
    assert classify_varsel_method("importance", enabled_models=["RandomForest"], spa_ok=True) == "importance"
    # ga is linear-only: present linear model -> importance; tree-only -> skip.
    assert classify_varsel_method("ga", enabled_models=["PLS"], spa_ok=True) == "importance"
    assert classify_varsel_method("ga", enabled_models=["RandomForest"], spa_ok=True) == "skip"
    # UVE/CARS/legacy always skip.
    for m in ["uve", "cars", "cars-tree", "uve_cars", "ipls", "vcpa-iriv", "fipls_cars"]:
        assert classify_varsel_method(m, enabled_models=["PLS"], spa_ok=True) == "skip"


def test_build_varsel_subsets_full_plus_interval_and_skips(xy_multi):
    from spectral_predict.multitarget_grid import build_multitarget_varsel_subsets

    X, Y, wl = xy_multi
    cache = {}
    subs, skipped = build_multitarget_varsel_subsets(
        ["ipls_forward", "uve", "cars"], X, Y, wl,
        enabled_models=["PLS"], variable_counts=[5, 10],
        ipls_subset_limit="Top 5", spa_ok=True,
        cache=cache,
        preprocess_id="raw",
    )
    tags = [s["tag"] for s in subs]
    assert "full" in tags
    assert subs[0]["method"] == "full"
    assert any(s["method"] == "ipls_forward" for s in subs)
    assert set(skipped) == {"uve", "cars"}


def test_build_varsel_subsets_cache_hit(xy_multi):
    from spectral_predict.multitarget_grid import build_multitarget_varsel_subsets

    X, Y, wl = xy_multi
    cache = {}
    build_multitarget_varsel_subsets(
        ["spa"], X, Y, wl, enabled_models=["PLS"], variable_counts=[5],
        ipls_subset_limit="All", spa_ok=True,
        cache=cache, preprocess_id="raw",
    )
    assert ("raw", "spa") in cache  # importance memoized per (preprocess, method)


# --- T-17 FIX 1: varsel cache key collision + mutation poisoning guards ---

def test_preprocess_fingerprint_discriminates_window():
    """Two configs sharing name='deriv' but differing in window MUST fingerprint
    differently. Documents the bug: the OLD production key (pc['name']) collides."""
    from spectral_predict.multitarget_grid import (
        _preprocess_fingerprint,
        build_multitarget_preprocess_configs,
    )

    cfgs = build_multitarget_preprocess_configs({"sg1": True}, window_sizes=[11, 21])
    deriv_cfgs = [c for c in cfgs if c["name"] == "deriv"]
    assert len(deriv_cfgs) == 2, "expected two 'deriv' configs differing in window"
    a, b = deriv_cfgs
    # Document the bug: the OLD production cache key (pc["name"]) is identical.
    assert a["name"] == b["name"] == "deriv"
    # The fingerprint MUST discriminate them, and must be hashable.
    fpa, fpb = _preprocess_fingerprint(a), _preprocess_fingerprint(b)
    assert fpa != fpb
    hash(fpa)
    hash(fpb)


def test_varsel_cache_no_bleed_across_distinct_fingerprints():
    """Two DIFFERENT X blocks with distinct preprocess fingerprints must yield
    DIFFERENT ipls_forward subsets from a shared cache (no cache bleed)."""
    from spectral_predict.multitarget_grid import (
        _preprocess_fingerprint,
        build_multitarget_varsel_subsets,
    )

    rng = np.random.default_rng(20260702)
    n, p = 60, 40
    wl = np.linspace(1000.0, 2000.0, p)
    # X1 carries signal in columns 0-3; X2 in columns 20-23. Strong SNR.
    X1 = rng.standard_normal((n, p))
    Y1 = X1[:, :4] @ rng.standard_normal((4, 2)) + 0.02 * rng.standard_normal((n, 2))
    X2 = rng.standard_normal((n, p))
    Y2 = X2[:, 20:24] @ rng.standard_normal((4, 2)) + 0.02 * rng.standard_normal((n, 2))

    cfg_a = {"name": "deriv", "deriv": 1, "window": 11, "polyorder": 2}
    cfg_b = {"name": "deriv", "deriv": 1, "window": 21, "polyorder": 2}
    # OLD key collides (same name); fingerprints must not.
    assert cfg_a["name"] == cfg_b["name"]
    fpa, fpb = _preprocess_fingerprint(cfg_a), _preprocess_fingerprint(cfg_b)
    assert fpa != fpb

    cache: dict = {}
    subs_a, _ = build_multitarget_varsel_subsets(
        ["ipls_forward"], X1, Y1, wl, enabled_models=["PLS"], variable_counts=[5],
        ipls_subset_limit="Top 5", spa_ok=True,
        cache=cache, preprocess_id=fpa,
    )
    subs_b, _ = build_multitarget_varsel_subsets(
        ["ipls_forward"], X2, Y2, wl, enabled_models=["PLS"], variable_counts=[5],
        ipls_subset_limit="Top 5", spa_ok=True,
        cache=cache, preprocess_id=fpb,
    )

    ipls_a = [s for s in subs_a if s["method"] == "ipls_forward"]
    ipls_b = [s for s in subs_b if s["method"] == "ipls_forward"]
    assert ipls_a and ipls_b

    picked_a = np.unique(np.concatenate([np.asarray(s["indices"]) for s in ipls_a]))
    picked_b = np.unique(np.concatenate([np.asarray(s["indices"]) for s in ipls_b]))
    # Core anti-bleed assertion: the two blocks select DIFFERENT index sets
    # (each reflects its own informative region, not the other's cache entry).
    assert not np.array_equal(picked_a, picked_b)
    # Directional lean: each block must reach its own signal region.
    assert picked_a.min() < 10, "block A must select a low-index (signal) feature"
    assert picked_b.max() >= 20, "block B must select a high-index (signal) feature"
    # Cache holds two distinct entries (fingerprints did not collide).
    assert (fpa, "ipls_forward") in cache
    assert (fpb, "ipls_forward") in cache
    assert len({(fpa, "ipls_forward"), (fpb, "ipls_forward")}) == 2


def test_varsel_cache_returned_subset_mutation_isolated():
    """Mutating a returned subset's indices in place must NOT poison the cache;
    a second identical call returns clean (un-mutated) indices."""
    from spectral_predict.multitarget_grid import build_multitarget_varsel_subsets

    rng = np.random.default_rng(20260702)
    n, p = 60, 40
    wl = np.linspace(1000.0, 2000.0, p)
    X = rng.standard_normal((n, p))
    Y = X[:, :4] @ rng.standard_normal((4, 2)) + 0.02 * rng.standard_normal((n, 2))

    cache: dict = {}
    args = dict(
        methods=["ipls_forward"], X_pp=X, Y=Y, wavelengths=wl,
        enabled_models=["PLS"], variable_counts=[5],
        ipls_subset_limit="Top 5", spa_ok=True,
        cache=cache, preprocess_id="raw",
    )
    subs1, _ = build_multitarget_varsel_subsets(**args)
    target = next(s for s in subs1 if s["method"] == "ipls_forward")
    assert target["indices"].size >= 1

    original = np.array(target["indices"], copy=True)
    # Poison the returned indices in place AND via rebind+contents mutation.
    target["indices"][:] = -1
    target["indices"] = np.array([999])
    target["indices"][...] = -1

    subs2, _ = build_multitarget_varsel_subsets(**args)
    twin = next(s for s in subs2 if s.get("tag") == target["tag"])
    # Freshly returned indices must be UNCHANGED (not -1, not [999]).
    assert np.array_equal(twin["indices"], original)
    assert (-1 not in twin["indices"])


# --- T-17 FIX 2: fipls_spa importance adapter must not collapse to index [0] ---

def test_fipls_spa_model_independent_importances_full_width():
    """Direct unit assertion: the fipls_spa branch of _model_independent_importances
    returns a full-width importance array whose peak sits on the signal band.

    On the BUGGY code this either returns None (no wavelengths kwarg accepted ->
    TypeError) or, if the kwarg existed but the branch didn't, a degenerate nan.
    """
    from spectral_predict.multitarget_grid import (
        _model_independent_importances,
        verify_spa_multi_y_safe,
    )

    rng = np.random.default_rng(20260702)
    n, p = 60, 40
    wl = np.linspace(1000.0, 2000.0, p)
    X = rng.standard_normal((n, p))
    informative = [18, 19, 20, 21]
    Y = X[:, informative] @ rng.standard_normal((4, 2)) + 0.02 * rng.standard_normal((n, 2))

    if not verify_spa_multi_y_safe(X, Y, n_features=5):
        pytest.skip("SPA not 2-D-safe here")

    imp = _model_independent_importances("fipls_spa", X, Y, wavelengths=wl)
    assert imp is not None
    imp = np.asarray(imp, dtype=float)
    assert imp.shape == (p,)
    assert np.all(np.isfinite(imp))
    assert np.count_nonzero(imp) > 1
    assert (set(np.argsort(imp)[-4:].tolist()) & set(informative)) or \
           (np.argmax(imp) in informative)


def test_fipls_spa_importance_adapter_selects_informative_not_index0():
    """Discriminating end-to-end test through build_multitarget_varsel_subsets.

    Signal lives in columns {18,19,20,21} -- NOT at index 0. A correctly-wired
    fipls_spa adapter must produce subsets that include at least one informative
    feature and are NOT the degenerate lone-index {0} subset.

    On the BUGGY code, _model_independent_importances returns None for fipls_spa;
    _importances_to_subsets turns np.asarray(None,float)=nan into argsort([nan])
    -> index [0] only -> union collapses to {0}, size 1.
    """
    from spectral_predict.multitarget_grid import (
        build_multitarget_varsel_subsets,
        verify_spa_multi_y_safe,
    )

    rng = np.random.default_rng(20260702)
    n, p = 60, 40
    wl = np.linspace(1000.0, 2000.0, p)
    X = rng.standard_normal((n, p))
    informative = [18, 19, 20, 21]
    Y = X[:, informative] @ rng.standard_normal((4, 2)) + 0.02 * rng.standard_normal((n, 2))

    if not verify_spa_multi_y_safe(X, Y, n_features=5):
        pytest.skip("SPA not 2-D-safe here")

    subs, _skipped = build_multitarget_varsel_subsets(
        ["fipls_spa"], X, Y, wl,
        enabled_models=["PLS"], variable_counts=[5, 10],
        ipls_subset_limit="All", spa_ok=True,
        cache={}, preprocess_id="raw",
    )

    fs = [s for s in subs if s["method"] == "fipls_spa"]
    assert fs, "expected at least one fipls_spa subset"

    # At least one fipls_spa subset must carry more than a lone index.
    assert any(s["indices"].size > 1 for s in fs), (
        f"fipls_spa subsets are degenerate (all size 1): "
        f"{[s['indices'].tolist() for s in fs]}"
    )

    union = set(np.concatenate([s["indices"] for s in fs]).tolist())
    assert union & set(informative), (
        f"fipls_spa union {union} does not include any informative feature "
        f"from {informative}"
    )
    assert union != {0}, f"fipls_spa union collapsed to degenerate {{0}}: {union}"


# --- T-17 FIX 4: MLP model-specific importance must not fall back to np.ones ---

def test_importance_reference_fit_mlp_not_uniform():
    """BEHAVIORAL: MLP importance must NOT collapse to a uniform np.ones vector.

    sklearn MLPRegressor exposes coefs_ (a list), NOT coef_ and NOT
    feature_importances_, so the buggy else-branch in _importance_reference_fit
    falls through to matrix = np.ones((n_features, 1)) -> a perfectly uniform
    aggregated importance -> meaningless top-N varsel subsets. The fix delegates
    to models.get_feature_importances (|first-layer weights|), matching single-Y.
    """
    from spectral_predict.multitarget_grid import _importance_reference_fit

    rng = np.random.default_rng(7)
    n, p = 60, 12
    X = rng.standard_normal((n, p))
    informative = [2, 3, 4]
    Y = X[:, informative] @ rng.standard_normal((3, 2)) + 0.05 * rng.standard_normal((n, 2))

    imp = _importance_reference_fit("MLP", X, Y, min_fold_train=n - 1)
    imp = np.asarray(imp, dtype=float)

    assert imp.shape == (p,)
    assert np.all(np.isfinite(imp))
    # CORE: the importance vector is NOT all-equal. On buggy code imp is a
    # uniform (aggregated np.ones) vector -> all-equal -> this FAILS. On fixed
    # code it reflects |first-layer weights| -> non-uniform -> PASSES.
    assert not np.allclose(imp, imp[0]), (
        f"MLP importance is uniform (all-equal={imp[0]}): the np.ones fallback "
        f"is still in place."
    )


def test_importance_reference_fit_mlp_delegates_to_get_feature_importances(monkeypatch):
    """PATH/SPY: the MLP else-branch MUST delegate to the shared single-Y
    importance extractor models.get_feature_importances.

    _importance_reference_fit re-imports get_feature_importances from .models
    at call time, so patching the module attribute takes effect. On buggy code
    the else-branch uses np.ones and never calls get_feature_importances for
    MLP -> the spy stays empty -> this FAILS.
    """
    import spectral_predict.models as models_module
    from spectral_predict.multitarget_grid import _importance_reference_fit

    rng = np.random.default_rng(7)
    n, p = 60, 12
    X = rng.standard_normal((n, p))
    informative = [2, 3, 4]
    Y = X[:, informative] @ rng.standard_normal((3, 2)) + 0.05 * rng.standard_normal((n, 2))

    called: list[str] = []

    def spy(model, name, X, y):
        called.append(name)
        return np.linspace(1, 2, X.shape[1])

    monkeypatch.setattr(models_module, "get_feature_importances", spy)

    imp = _importance_reference_fit("MLP", X, Y, min_fold_train=n - 1)
    imp = np.asarray(imp, dtype=float)

    # The spy WAS called with model_name "MLP". On buggy code the else-branch
    # never calls get_feature_importances for MLP -> called stays empty -> FAILS.
    assert "MLP" in called, (
        f"get_feature_importances was not called for MLP (called={called}); "
        f"the np.ones fallback is still in place."
    )
    # Sanity: returned array matches the spy's non-uniform values shape.
    expected = np.linspace(1, 2, p)
    assert imp.shape == expected.shape
    assert not np.allclose(imp, imp[0])
