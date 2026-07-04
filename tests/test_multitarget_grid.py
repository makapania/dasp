"""Tests for the T-17 multi-target grid orchestrator (multitarget_grid.py)."""
from __future__ import annotations

import numpy as np
import pytest


def test_preprocess_configs_raw_and_snv():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({"raw": True, "snv": True})
    names = [c["name"] for c in cfgs]
    assert "raw" in names
    assert "snv" in names
    raw = next(c for c in cfgs if c["name"] == "raw")
    assert raw["deriv"] is None and raw["window"] is None and raw["polyorder"] is None


def test_preprocess_configs_sg_polyorder_pairing():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs(
        {"sg1": True, "sg2": True}, window_sizes=[11]
    )
    derivs = {(c["deriv"], c["polyorder"]) for c in cfgs if c["name"] == "deriv"}
    assert (1, 2) in derivs  # sg1 -> deriv 1 / poly 2
    assert (2, 3) in derivs  # sg2 -> deriv 2 / poly 3


def test_describe_preprocess_config_includes_deriv_and_window():
    """FIX 3: the leaderboard/CSV preprocessing label must carry the SG
    derivative order + window (+ polyorder), not just the bare ``name`` — so
    ``snv_deriv d1 w11`` is distinguishable from ``snv_deriv d2 w17``."""
    from spectral_predict.multitarget_grid import (
        _describe_preprocess_config,
        build_multitarget_preprocess_configs,
    )

    cfgs = build_multitarget_preprocess_configs({"sg2": True}, window_sizes=[17])
    pc = next(c for c in cfgs if c["deriv"] == 2 and c["window"] == 17)
    label = _describe_preprocess_config(pc)
    assert "d2" in label, f"deriv order missing from label: {label!r}"
    assert "w17" in label, f"SG window missing from label: {label!r}"
    assert "p3" in label, f"polyorder missing from label: {label!r}"
    # A raw config yields the bare name with no numeric suffixes.
    assert _describe_preprocess_config({"name": "raw"}) == "raw"


def test_preprocess_configs_autoscale_doubling():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({"raw": True}, autoscale=True)
    names = [c["name"] for c in cfgs]
    assert "raw" in names                 # without-autoscale copy
    assert "raw+autoscale" in names       # with-autoscale copy
    assert any(c.get("autoscale") is True for c in cfgs)
    assert any(c.get("autoscale") is False for c in cfgs)


def test_preprocess_configs_baseline_doubling():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs(
        {"raw": True}, baseline_method="als", baseline_params={"lam": 1e5}
    )
    names = [c["name"] for c in cfgs]
    assert "raw" in names
    assert "als+raw" in names
    without = next(c for c in cfgs if c["name"] == "raw")
    assert without["baseline_method"] is None


def test_preprocess_configs_empty_falls_back_to_raw():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({})
    assert [c["name"] for c in cfgs] == ["raw"]


@pytest.fixture
def rng():
    return np.random.default_rng(4242)


@pytest.fixture
def grid_xy(rng):
    n, p = 45, 30
    X = rng.standard_normal((n, p))
    base = X[:, :4] @ rng.standard_normal((4, 2))
    Y = base + 0.05 * rng.standard_normal((n, 2))
    wl = np.linspace(1000.0, 2000.0, p)
    return X, Y, wl


def test_grid_search_grid_only_assert(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    with pytest.raises(ValueError):
        run_multitarget_grid_search(
            X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
            preprocessing_methods={"raw": True}, autoscale=False,
            variable_selection_methods=[], tier="quick",
            cv="kfold", n_folds=3, n_repeats=1, optimization_method="unified",
        )


def test_grid_search_end_to_end_ranks_and_skips(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True, "snv": True}, autoscale=False,
        variable_selection_methods=["ipls_forward", "uve"], variable_counts=[5, 10],
        ipls_subset_limit="Top 3", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, random_state=42,
    )
    # Ranked, NaN-safe: best has a finite joint_q2.
    assert len(out.results) >= 4
    assert np.isfinite(out.results[0].joint_q2)
    # Both preprocessing states and >1 varsel tag appear.
    assert {r.preprocessing for r in out.results} >= {"raw", "snv"}
    assert any(r.varsel_method == "ipls_forward" for r in out.results)
    assert any(r.varsel_method == "full" for r in out.results)
    # UVE skip surfaced.
    assert "uve" in out.skipped


def test_grid_search_apply_uve_prefilter_surfaces_skip(grid_xy):
    """FIX A: apply_uve_prefilter=True must flow from run_multitarget_grid_search
    into build_multitarget_varsel_subsets so the skip-notice branch fires and
    'apply_uve_prefilter' appears in out.skipped (UVE-on-y is a discrimination
    method, not a multi-Y method -- greyed out, surfaced as a skip notice).

    On unfixed code the run function has no apply_uve_prefilter param -> the call
    raises TypeError and this test fails.
    """
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=["ipls_forward"], variable_counts=[5],
        ipls_subset_limit="Top 3", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, apply_uve_prefilter=True,
    )
    assert "apply_uve_prefilter" in out.skipped
    # The run still completes and ranks (the skip notice does not abort search).
    assert out.results
    assert np.isfinite(out.results[0].joint_q2)


def test_grid_search_no_uve_prefilter_no_notice(grid_xy):
    """FIX A (negative control): default apply_uve_prefilter=False must NOT add
    the skip notice (guards against an always-on notice)."""
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=["ipls_forward"], variable_counts=[5],
        ipls_subset_limit="Top 3", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1,
    )
    assert "apply_uve_prefilter" not in out.skipped


def test_grid_search_progress_callback_dict_shape(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    seen = []
    run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=[], tier="quick",
        cv="kfold", n_folds=3, n_repeats=1,
        progress_callback=lambda info: seen.append(info),
    )
    assert seen
    last = seen[-1]
    assert set(["message", "current", "total"]).issubset(last.keys())
    assert "best_model" in last


def test_grid_varsel_producer_failure_is_skipped_not_fatal(grid_xy, monkeypatch):
    """FIX C: one raising varsel producer must NOT abort the whole search.

    The per-method interval adapter runs with no try/except on unfixed code, so
    a single raising (preprocess, method) aborts run_multitarget_grid_search
    entirely. The fix appends the method to out.skipped and continues; the 'full'
    cells and every other result still rank.
    """
    import spectral_predict.multitarget_grid as mtg
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    def _boom(*a, **k):
        raise RuntimeError("interval adapter blew up")

    monkeypatch.setattr(mtg, "_interval_subset_adapter", _boom)

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=["ipls_forward"], variable_counts=[5],
        ipls_subset_limit="Top 3", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1,
    )
    # Search COMPLETED despite the producer raising.
    assert "ipls_forward" in out.skipped
    # 'full' cells still built + ranked.
    assert out.results
    assert np.isfinite(out.results[0].joint_q2)
    assert any(r.varsel_method == "full" for r in out.results)


def test_grid_importance_reference_fit_failure_is_skipped_not_fatal(grid_xy, monkeypatch):
    """FIX C: a raising _importance_reference_fit (model-specific importance fit)
    must NOT abort the search. The offending method+model is recorded in
    out.skipped and the run completes with other cells still ranking."""
    import spectral_predict.multitarget_grid as mtg
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    def _boom(model_name, *a, **k):
        raise RuntimeError(f"importance fit blew up for {model_name}")

    monkeypatch.setattr(mtg, "_importance_reference_fit", _boom)

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=["importance"], variable_counts=[5],
        ipls_subset_limit="All", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1,
    )
    # Search COMPLETED; importance failure surfaced in skipped.
    assert any("importance" in s for s in out.skipped)
    # 'full' cells still built + ranked.
    assert out.results
    assert np.isfinite(out.results[0].joint_q2)
    assert any(r.varsel_method == "full" for r in out.results)


def test_grid_pls_ncomponents_capped_per_subset_and_reported(grid_xy):
    """T-17 FIX 5: PLS n_components must be capped PER-SUBSET (to the
    cap_components limit) BEFORE dedup, so a narrow top-N subset does not
    spawn duplicate effective cells whose reported params overstate
    n_components. Spec (design doc line 186): params holds the EFFECTIVE
    (post-cap) hyperparameters."""
    from spectral_predict.models import get_model_grids
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy

    # PRECONDITION (documents the grid CAN request >5): at max_n_components=10
    # the PLS grid emits some n_components in the 6..10 range.
    grids = get_model_grids(
        task_type="regression", n_features=30, tier="standard",
        enabled_models=["PLS"], max_n_components=10, max_iter=500,
    )
    requested = {p.get("n_components") for (_e, p) in grids["PLS"]}
    assert max(nc for nc in requested if nc is not None) > 5

    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=["spa"], variable_counts=[5],
        ipls_subset_limit="All", tier="standard", max_n_components=10,
        cv="kfold", n_folds=3, n_repeats=1, random_state=42,
    )
    # SPA must be 2-D-safe on this block (not skipped).
    assert "spa" not in out.skipped

    narrow = [r for r in out.results if r.model_name == "PLS" and r.n_variables == 5]
    assert narrow, "expected PLS cells on the top-5 SPA subset"

    # CORE DISCRIMINATOR (b): EVERY narrow PLS cell reports n_components <= 5
    # (the per-subset cap), never the requested 6..10.
    assert all(r.params.get("n_components", 0) <= 5 for r in narrow), (
        "overstated n_components on narrow subset: "
        + str(sorted(r.params.get("n_components") for r in narrow))
    )

    # SECONDARY (a) dedup: no two narrow PLS rows share identical effective
    # (preprocessing, varsel_tag, params).
    keys = [
        (r.preprocessing, r.varsel_tag, frozenset(r.params.items()))
        for r in narrow
    ]
    assert len(keys) == len(set(keys))


def test_grid_importance_varsel_produces_narrow_cells(grid_xy):
    """FIX E(1): the model-specific 'importance' varsel path (exercised via
    _importance_reference_fit + _importances_to_subsets) must produce leaderboard
    cells with n_variables < full and a finite best, across a JOINT model (PLS)
    and an INDEPENDENT model (Ridge)."""
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy  # p == 30
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=["importance"], variable_counts=[5, 10],
        ipls_subset_limit="All", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, random_state=42,
    )
    imp_cells = [r for r in out.results if r.varsel_method == "importance"]
    assert imp_cells, "expected 'importance' varsel cells in the leaderboard"
    assert all(r.n_variables < X.shape[1] for r in imp_cells)
    assert set(r.n_variables for r in imp_cells) <= {5, 10}
    assert np.isfinite(out.results[0].joint_q2)


def test_grid_sg1_completes_with_finite_best(grid_xy):
    """FIX G(1): a Savitzky-Golay-derivative preprocessing config must run end to
    end (exercises the deriv/edge-mask branch that trims X_pp + wl_pp) and yield a
    finite best."""
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"sg1": True}, autoscale=False, window_sizes=[11],
        variable_selection_methods=[], tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, random_state=42,
    )
    assert out.results
    assert np.isfinite(out.best.joint_q2)
    # FIX 3: the label now carries deriv order + SG window (not the bare name).
    assert any(
        r.preprocessing.startswith("deriv") and "d1" in r.preprocessing
        and "w11" in r.preprocessing
        for r in out.results
    )


def test_preprocess_fingerprint_discriminates_deriv_and_polyorder():
    """FIX G(2): two configs sharing name but differing in deriv/polyorder must
    receive DISTINCT fingerprints (no cache collision)."""
    from spectral_predict.multitarget_grid import _preprocess_fingerprint

    base = {"name": "deriv", "window": 11}
    a = {**base, "deriv": 1, "polyorder": 2}
    b = {**base, "deriv": 2, "polyorder": 3}
    c = {**base, "deriv": 1, "polyorder": 4}  # same deriv, different polyorder
    assert a["name"] == b["name"] == c["name"]
    fps = {_preprocess_fingerprint(a), _preprocess_fingerprint(b),
           _preprocess_fingerprint(c)}
    assert len(fps) == 3  # all distinct


def test_dedup_keyset_equals_consumed_no_config_lost():
    from spectral_predict.multitarget_grid import _dedup_model_configs

    # Two RF configs differing only in a consumed key (bootstrap) must NOT collapse.
    class _E: pass
    grids = {"RandomForest": [
        (_E(), {"n_estimators": 50, "bootstrap": True}),
        (_E(), {"n_estimators": 50, "bootstrap": False}),
        (_E(), {"n_estimators": 50, "bootstrap": True}),  # exact dup -> collapses
    ]}
    out = _dedup_model_configs(grids)
    assert len(out) == 2
    assert {c["params"]["bootstrap"] for c in out} == {True, False}


# --------------------------------------------------------------------------- #
# T-17 FIX 6b: wavelength restriction that selects zero columns must raise
# --------------------------------------------------------------------------- #
def test_grid_wavelength_restriction_empty_raises(grid_xy):
    """A wavelength restriction with NO overlap must raise a clear ValueError
    (single-Y parity, search.py:~2791/~2830) instead of silently proceeding
    into a degenerate (n, 0) feature block.
    """
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    # wl spans 1000-2000 nm; 99990-99999 has zero overlap.
    with pytest.raises(ValueError, match="excludes all wavelengths"):
        run_multitarget_grid_search(
            X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
            wavelength_restriction={"min": 99990.0, "max": 99999.0},
            preprocessing_methods={"raw": True}, autoscale=False,
            variable_selection_methods=[], tier="quick",
            cv="kfold", n_folds=3, n_repeats=1,
        )


def test_grid_wavelength_restriction_nonempty_does_not_raise(grid_xy):
    """POSITIVE control: a NON-empty restriction must NOT raise (guards against
    an over-broad raise that would also trip on valid restrictions).
    """
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        wavelength_restriction={"min": 1200.0, "max": 1800.0},
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=[], tier="quick",
        cv="kfold", n_folds=3, n_repeats=1,
    )
    assert out.results  # non-empty, finite best
    assert np.isfinite(out.best.joint_q2)


# --------------------------------------------------------------------------- #
# T-17 FIX 6c: Cancel must interrupt the preprocessing/subset-build loop
# --------------------------------------------------------------------------- #
class _Stopped:
    """Fake controller whose check_and_wait() always returns False (stop)."""
    def __init__(self):
        self.calls = 0

    def check_and_wait(self) -> bool:
        self.calls += 1
        return False


class _Running:
    """Fake controller whose check_and_wait() always returns True (continue)."""
    def __init__(self):
        self.calls = 0

    def check_and_wait(self) -> bool:
        self.calls += 1
        return True


def test_grid_controller_prestopped_early_exit(grid_xy):
    """A controller that reports stop BEFORE the first preprocessing iteration
    must interrupt the cell-BUILD loop so NO cells are built or evaluated.

    On buggy code: (1) the preprocess loop had no controller check, and (2) the
    eval-loop check ignored the False return value -- so the full search ran
    and out.results was NON-empty.
    """
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    ctrl = _Stopped()
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True, "snv": True}, autoscale=False,
        variable_selection_methods=["ipls_forward"], variable_counts=[5],
        ipls_subset_limit="Top 3", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, controller=ctrl,
    )
    # EARLY EXIT: no cells built or evaluated.
    assert out.results == []
    assert ctrl.calls >= 1


def test_grid_controller_running_completes(grid_xy):
    """POSITIVE control: a controller whose check_and_wait() always returns True
    must NOT early-exit -- the same run yields non-empty out.results (guards
    against an always-break over-fix).
    """
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    ctrl = _Running()
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=[], tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, controller=ctrl,
    )
    assert out.results  # non-empty
