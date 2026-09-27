"""Top-N subsets of sparse selectors must not pad with zero-importance variables.

CARS (and the other selectors in ``SPARSE_SELECTOR_METHODS``) return a score array
whose zeros mean "not selected". Taking ``np.argsort(importances, kind="stable")[-n:]``
with ``n`` above the non-zero count used to fill the gap with the highest-index
zeros, i.e. the longest wavelengths, while the row was still labelled ``top{n}_cars``.
The fix caps ``n`` at the selected count: the tag keeps the requested count, the
row's ``n_vars`` records the fitted one, and repeat subsets are not re-run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.variable_selection import SPARSE_SELECTOR_METHODS, _cap_top_n

N_FEATURES = 120
# Selected variables sit at the LOW end, so any padding (highest-index zeros) is visible.
SELECTED = np.arange(5, 20)  # 15 variables
WAVELENGTHS = np.arange(1000, 1000 + N_FEATURES)
SELECTED_WL = {f"{float(w):g}" for w in WAVELENGTHS[SELECTED]}


def _sparse_importances(n_features: int) -> np.ndarray:
    imp = np.zeros(n_features)
    sel = SELECTED[SELECTED < n_features]
    imp[sel] = np.linspace(1.0, 2.0, len(sel))
    return imp


def _fake_cars(X, y, *args, **kwargs):
    return _sparse_importances(np.asarray(X).shape[1])


def _regression_data(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = rng.randn(40, N_FEATURES)
    y = X[:, SELECTED] @ rng.randn(len(SELECTED)) + 0.1 * rng.randn(40)
    return pd.DataFrame(X, columns=[str(w) for w in WAVELENGTHS]), pd.Series(y)


def _vars(cell: str) -> set[str]:
    return set(str(cell).split(","))


class TestCapTopN:
    def test_sparse_method_caps_at_nonzero(self):
        imp = _sparse_importances(N_FEATURES)
        assert _cap_top_n(imp, 10, "cars") == 10
        assert _cap_top_n(imp, 250, "cars") == len(SELECTED)

    def test_dense_method_unchanged(self):
        imp = _sparse_importances(N_FEATURES)
        assert "importance" not in SPARSE_SELECTOR_METHODS
        assert _cap_top_n(imp, 100, "importance") == 100

    def test_all_zero_sparse_selects_nothing(self):
        # 0, not the requested count: callers skip rather than slice [-0:] (every column).
        assert _cap_top_n(np.zeros(N_FEATURES), 50, "cars") == 0

    def test_all_zero_dense_unchanged(self):
        assert _cap_top_n(np.zeros(N_FEATURES), 50, "importance") == 50

    def test_exact_and_single_variable(self):
        imp = _sparse_importances(N_FEATURES)
        assert _cap_top_n(imp, len(SELECTED), "cars") == len(SELECTED)
        one = np.zeros(N_FEATURES)
        one[3] = 0.5
        assert _cap_top_n(one, 100, "cars") == 1


def test_grid_search_never_pads_cars_subsets(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", _fake_cars)
    X, y = _regression_data()
    results, _ = search.run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=True,
        enable_region_subsets=False,
        variable_selection_methods=["cars"],
        variable_counts=[10, 50, 100],
        tier="quick",
    )
    # "top{n}_cars" rows plus the method-optimal row, tagged plain "cars".
    tags = results["SubsetTag"].astype(str)
    cars_rows = results[tags.str.endswith("_cars") | (tags == "cars")]
    assert not cars_rows.empty

    for _, row in cars_rows.iterrows():
        assert _vars(row["all_vars"]) <= SELECTED_WL, row["SubsetTag"]
        assert row["n_vars"] == len(_vars(row["all_vars"]))

    # top-50 and top-100 both cap to the 15 selected vars, which is also the
    # method-optimal count: that subset runs only as the method-optimal row.
    assert set(cars_rows["n_vars"]) == {10, len(SELECTED)}
    assert not tags.isin(["top50_cars", "top100_cars"]).any()
    assert not cars_rows.duplicated(["Model", "Params", "n_vars"]).any()


def test_one_class_grid_never_pads_cars_subsets(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", _fake_cars)
    rng = np.random.RandomState(1)
    X = np.vstack([rng.randn(30, N_FEATURES) * 0.3, rng.randn(8, N_FEATURES) + 3.0])
    X = pd.DataFrame(X, columns=[str(w) for w in WAVELENGTHS])
    y = pd.Series(["clean"] * 30 + ["contaminated"] * 8)

    results = search.run_one_class_search(
        X=X,
        y=y,
        inlier_class_label="clean",
        folds=3,
        preprocessing_methods=["raw"],
        window_sizes=[17],
        enabled_models=["IsolationForest"],
        variable_selection_methods=["cars"],
        variable_counts=[10, 50, 100],
    )
    cars_rows = results[results["SubsetTag"].astype(str).str.startswith("cars_top")]
    assert not cars_rows.empty

    for _, row in cars_rows.iterrows():
        assert _vars(row["all_vars"]) <= SELECTED_WL, row["SubsetTag"]
        assert row["n_vars"] == len(_vars(row["all_vars"]))
    # top50 and top100 cap to the same 15 variables: only the first is fitted.
    assert set(cars_rows["SubsetTag"]) == {"cars_top10", "cars_top50"}
    assert set(cars_rows["n_vars"]) == {10, len(SELECTED)}


def test_bayesian_cars_trials_never_pad(monkeypatch):
    optuna = pytest.importorskip("optuna")
    import spectral_predict.unified_bayesian as ub

    monkeypatch.setattr(ub, "cars_selection", _fake_cars)
    X, y = _regression_data()
    _, study = ub.run_unified_bayesian(
        X=X.to_numpy(),
        y=y.to_numpy(),
        wavelengths=WAVELENGTHS.astype(float),
        model_name="PLS",
        task_type="regression",
        n_trials=40,
        cv_folds=3,
        random_state=0,
        verbose=False,
        enable_sqlite_persistence="never",
    )
    cars_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.params.get("subset_type") == "cars"
        and t.params.get("n_vars") != "full"
    ]
    over_requested = [t for t in cars_trials if t.params["n_vars"] > len(SELECTED)]
    assert over_requested, "seed produced no CARS trial requesting more than CARS kept"

    # Derivative preprocessing remaps wavelengths, so check counts, not names:
    # padding would show up as more fitted variables than CARS selected.
    # A replayed duplicate returns before the subset attrs are set.
    cars_trials = [t for t in cars_trials if "selected_wavelengths" in t.user_attrs]
    over_requested = [t for t in over_requested if "selected_wavelengths" in t.user_attrs]
    for t in cars_trials:
        n_fit = len(_vars(t.user_attrs["selected_wavelengths"]))
        assert n_fit == t.user_attrs["n_vars"] <= len(SELECTED), t.params
    for t in over_requested:
        assert t.user_attrs["n_vars"] == len(SELECTED)
        # The tag keeps the requested count (clamped to the preprocessed width).
        requested = int(t.user_attrs["subset_tag"].removeprefix("top").removesuffix("_cars"))
        assert len(SELECTED) < requested <= t.params["n_vars"]


def test_grid_search_never_pads_ga_subsets(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "ga_pls_selection", _fake_cars)
    X, y = _regression_data()
    results, _ = search.run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=True,
        enable_region_subsets=False,
        variable_selection_methods=["ga"],
        variable_counts=[10, 50, 100],
        tier="quick",
    )
    tags = results["SubsetTag"].astype(str)
    ga_rows = results[tags.str.endswith("_ga")]
    assert not ga_rows.empty
    for _, row in ga_rows.iterrows():
        assert _vars(row["all_vars"]) <= SELECTED_WL, row["SubsetTag"]
    # top50 and top100 both cap to the 15 selected vars: only top50 is fitted.
    assert set(ga_rows["SubsetTag"]) == {"top10_ga", "top50_ga"}
    assert set(ga_rows["n_vars"]) == {10, len(SELECTED)}


def test_multiclass_mask_never_pads_cars(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", _fake_cars)
    rng = np.random.RandomState(2)
    X = rng.randn(45, N_FEATURES)
    y = np.repeat(["a", "b", "c"], 15)
    mask = search.multiclass_varsel_mask(X, y, WAVELENGTHS.astype(float), "cars", n_select=100)
    assert np.flatnonzero(mask).tolist() == SELECTED.tolist()


def _edge_only_cars(X, y, *args, **kwargs):
    # Scores only inside the SG derivative edge zone, so edge masking removes them all.
    imp = np.zeros(np.asarray(X).shape[1])
    imp[:3] = [1.0, 2.0, 3.0]
    return imp


def test_grid_skips_cars_when_edge_masking_removes_every_selection(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", _edge_only_cars)
    X, y = _regression_data()
    results, _ = search.run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"sg1": True},
        window_sizes=[17],
        enable_variable_subsets=True,
        enable_region_subsets=False,
        variable_selection_methods=["cars"],
        variable_counts=[10, 50],
        tier="quick",
    )
    tags = results["SubsetTag"].astype(str)
    assert not (tags.str.endswith("_cars") | (tags == "cars")).any()


def test_multiclass_mask_rejects_empty_sparse_selection(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", lambda X, y, *a, **k: np.zeros(X.shape[1]))
    rng = np.random.RandomState(2)
    X = rng.randn(45, N_FEATURES)
    y = np.repeat(["a", "b", "c"], 15)
    with pytest.raises(search.MulticlassVarselUnsupported):
        search.multiclass_varsel_mask(X, y, WAVELENGTHS.astype(float), "cars", n_select=10)


@pytest.mark.parametrize(
    "n_components", [0.99, {"a": 2, "b": 2, "c": 2}], ids=["fraction", "per_class_dict"]
)
def test_multiclass_search_fits_each_capped_mask_once(monkeypatch, n_components):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", _fake_cars)
    rng = np.random.RandomState(3)
    X = np.vstack([rng.randn(15, N_FEATURES) + shift for shift in (0.0, 2.0, 4.0)])
    y = np.repeat(["a", "b", "c"], 15)
    results = search.run_multiclass_simca_search(
        X,
        y,
        wavelengths=WAVELENGTHS.astype(float),
        engines=["pca-simca"],
        preprocessing_methods={"raw": True},
        varsel_paths=["cars"],
        variable_selection_n_select=[10, 50, 100],
        n_components=n_components,
        min_class_samples=5,
        cv_splits=3,
    )
    cars_rows = results[results["varsel_path"] == "cars"]
    # 50 and 100 both cap to the 15 selected variables: fitted once.
    assert sorted(cars_rows["NSelect"]) == [10, 50]
    assert sorted(cars_rows["n_vars"]) == [10, len(SELECTED)]


class _FixedSampler:
    """Every param at its first choice / lower bound, except n_vars alternating 50/100
    and subset_type forced to CARS, so consecutive trials differ only in a requested
    count that caps to the same selection."""

    def __new__(cls):
        import optuna

        class _Sampler(optuna.samplers.RandomSampler):
            def sample_independent(self, study, trial, param_name, param_distribution):
                if param_name == "n_vars":
                    return 50 if trial.number % 2 == 0 else 100
                if param_name == "subset_type":
                    return "cars"
                if isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
                    return param_distribution.choices[0]
                return param_distribution.low

        return _Sampler(seed=0)


@pytest.mark.parametrize("task_type", ["regression", "one_class"])
def test_bayesian_capped_requests_replay_one_fit(monkeypatch, task_type):
    optuna = pytest.importorskip("optuna")
    import spectral_predict.unified_bayesian as ub

    monkeypatch.setattr(ub, "cars_selection", _fake_cars)
    monkeypatch.setattr(ub, "TPESampler", lambda *a, **k: _FixedSampler())
    if task_type == "one_class":
        rng = np.random.RandomState(4)
        X = np.vstack([rng.randn(30, N_FEATURES) * 0.3, rng.randn(8, N_FEATURES) + 3.0])
        y = np.array(["clean"] * 30 + ["contaminated"] * 8)
        model_name, extra = "IsolationForest", {"inlier_class_label": "clean"}
    else:
        X_df, y_s = _regression_data()
        X, y = X_df.to_numpy(), y_s.to_numpy()
        model_name, extra = "PLS", {}

    results, study = ub.run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=WAVELENGTHS.astype(float),
        model_name=model_name,
        task_type=task_type,
        n_trials=2,
        cv_folds=3,
        random_state=0,
        verbose=False,
        enable_sqlite_persistence="never",
        **extra,
    )
    first, second = study.trials[:2]
    assert (first.params["n_vars"], second.params["n_vars"]) == (50, 100)
    assert first.params["subset_type"] == second.params["subset_type"] == "cars"
    assert second.user_attrs.get(ub.DUPLICATE_OF_TRIAL_ATTR) == first.number
    assert second.value == first.value
    assert len(results) == 1


def _one_class_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(1)
    X = np.vstack([rng.randn(30, N_FEATURES) * 0.3, rng.randn(8, N_FEATURES) + 3.0])
    return (
        pd.DataFrame(X, columns=[str(w) for w in WAVELENGTHS]),
        pd.Series(["clean"] * 30 + ["contaminated"] * 8),
    )


def _run_one_class(search, preprocessing: str) -> tuple[pd.DataFrame, dict]:
    X, y = _one_class_data()
    messages = []
    results = search.run_one_class_search(
        X=X,
        y=y,
        inlier_class_label="clean",
        folds=3,
        preprocessing_methods=[preprocessing],
        window_sizes=[17],
        enabled_models=["IsolationForest"],
        variable_selection_methods=["cars"],
        variable_counts=[10, 50, 100],
        progress_callback=messages.append,
    )
    return results, messages[-1]


def test_one_class_reports_capped_counts_as_skipped(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", _fake_cars)
    _, final = _run_one_class(search, "raw")
    assert final["current"] == final["total"]
    # top100 caps to the same 15 vars as top50: its model params are skipped, not "0 skipped".
    assert not final["message"].endswith(" 0 skipped"), final["message"]


def test_one_class_skips_cars_when_edge_masking_removes_every_selection(monkeypatch):
    import spectral_predict.search as search

    monkeypatch.setattr(search, "cars_selection", _edge_only_cars)
    results, final = _run_one_class(search, "deriv1")
    assert not results["SubsetTag"].astype(str).str.startswith("cars_top").any()
    assert final["current"] == final["total"]
    assert not final["message"].endswith(" 0 skipped"), final["message"]


def test_bayesian_empty_sparse_selection_is_penalised_not_padded(monkeypatch):
    pytest.importorskip("optuna")
    import spectral_predict.unified_bayesian as ub

    monkeypatch.setattr(ub, "cars_selection", lambda X, y, *a, **k: np.zeros(np.asarray(X).shape[1]))
    monkeypatch.setattr(ub, "TPESampler", lambda *a, **k: _FixedSampler())
    X_df, y_s = _regression_data()
    results, study = ub.run_unified_bayesian(
        X=X_df.to_numpy(),
        y=y_s.to_numpy(),
        wavelengths=WAVELENGTHS.astype(float),
        model_name="PLS",
        task_type="regression",
        n_trials=2,
        cv_folds=3,
        random_state=0,
        verbose=False,
        enable_sqlite_persistence="never",
    )
    assert [t.value for t in study.trials] == [1e10, 1e10]
    assert len(results) == 0
