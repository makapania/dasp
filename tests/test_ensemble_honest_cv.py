"""Honest ensemble cross-validation and failed-member handling (R002, R018, R021, R105).

* R002: ensemble CV must refit the base models inside each outer fold; memorising base
  models on pure noise must not score near R2 = 1.
* R018: targets are a label-indexed Series in the GUI. Positional CV indices must reach
  them by position (string IDs, gapped and permuted integer IDs). The previous suite
  passed numpy ``y`` only, so it never saw this.
* R021: the weight / meta-model fit must see genuinely held-out base predictions.
* R105: a member whose out-of-fold refit fails is removed consistently.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from spectral_predict.ensemble import (
    OOF_FAILURE_WARNING,
    MixtureOfExpertsEnsemble,
    RegionAwareWeightedEnsemble,
    StackingEnsemble,
    create_auto_ensembles,
    create_ensemble,
    cross_validate_ensembles,
)

ALL_TYPES = ["simple_average", "region_weighted", "mixture_experts", "stacking", "region_stacking"]


def _signal_data(n: int = 60, p: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 2.0 * X[:, 0] - X[:, 3] + 0.5 * X[:, 7] + 0.3 * rng.standard_normal(n)
    return X, y


def _fitted(models, X, y):
    return [m.fit(X, y) for m in models]


def _index_variants(n: int) -> dict[str, pd.Index]:
    rng = np.random.default_rng(1)
    return {
        "string_ids": pd.Index([f"S{i:03d}" for i in range(n)]),
        "gapped_int": pd.Index(np.arange(n) * 3 + 7),
        "permuted_int": pd.Index(rng.permutation(n)),
    }


# --- R018: label-indexed targets ---------------------------------------------------


@pytest.mark.parametrize("variant", ["string_ids", "gapped_int", "permuted_int"])
def test_cross_validate_ensembles_indexes_series_targets_by_position(variant):
    X, y = _signal_data()
    index = _index_variants(len(y))[variant]
    X_df = pd.DataFrame(X, index=index, columns=[f"{1000 + i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, index=index)
    models = _fitted([Ridge(alpha=1.0), PLSRegression(n_components=3)], X_df, y)

    from_series = cross_validate_ensembles(
        models, ["Ridge", "PLS"], X_df, y_ser, ALL_TYPES, n_regions=3
    )
    from_array = cross_validate_ensembles(models, ["Ridge", "PLS"], X, y, ALL_TYPES, n_regions=3)

    assert from_series.errors == {}
    for etype in ALL_TYPES:
        np.testing.assert_allclose(from_series.predictions[etype], from_array.predictions[etype])


@pytest.mark.parametrize("variant", ["string_ids", "gapped_int", "permuted_int"])
def test_create_auto_ensembles_indexes_series_targets_by_position(variant):
    X, y = _signal_data(n=80)
    index = _index_variants(len(y))[variant]
    X_df = pd.DataFrame(X, index=index, columns=[f"wl_{i}" for i in range(X.shape[1])])
    results_df = pd.DataFrame(
        {
            "Model": ["Ridge", "PLS"],
            "regional_rmse": [
                {"Q1": 0.1, "Q2": 0.2, "Q3": 0.1, "Q4": 0.2},
                {"Q1": 0.2, "Q2": 0.1, "Q3": 0.2, "Q4": 0.1},
            ],
        }
    )

    def reconstruct(row, X_train, y_train):
        model = Ridge(alpha=1.0) if row["Model"] == "Ridge" else PLSRegression(n_components=3)
        return model.fit(X_train, y_train), row["Model"]

    kwargs = dict(
        results_df=results_df,
        task_type="regression",
        reconstruct_func=reconstruct,
        all_wavelengths=list(X_df.columns),
    )
    from_series = create_auto_ensembles(X_train=X_df, y_train=pd.Series(y, index=index), **kwargs)
    from_array = create_auto_ensembles(X_train=X_df, y_train=y, **kwargs)

    assert from_series, "the Series target must not break auto-ensembles"
    for name, info in from_array.items():
        assert from_series[name]["metrics"]["r2"] == pytest.approx(info["metrics"]["r2"])
        assert info["metrics"]["r2"] > 0.8


def test_create_auto_ensembles_reports_nan_instead_of_full_data_fallback():
    """A fold that cannot rebuild 2 models must not be scored by the full-data ensemble."""
    X, y = _signal_data(n=40)
    results_df = pd.DataFrame(
        {
            "Model": ["Ridge", "Fragile"],
            "regional_rmse": [
                {"Q1": 0.1, "Q2": 0.2, "Q3": 0.1, "Q4": 0.2},
                {"Q1": 0.2, "Q2": 0.1, "Q3": 0.2, "Q4": 0.1},
            ],
        }
    )

    def reconstruct(row, X_train, y_train):
        if row["Model"] == "Fragile" and len(y_train) < len(y):
            raise ValueError("cannot fit on a CV training fold")
        return Ridge(alpha=1.0).fit(X_train, y_train), row["Model"]

    with pytest.warns(UserWarning):
        out = create_auto_ensembles(
            results_df, X, y, "regression", reconstruct, all_wavelengths=list(range(X.shape[1]))
        )
    assert out
    for info in out.values():
        assert np.isnan(info["metrics"]["r2"]) and np.isnan(info["metrics"]["rmse"])


# --- R002: honest outer CV ------------------------------------------------------------


@pytest.mark.parametrize("etype", ALL_TYPES)
def test_memorising_base_models_on_noise_do_not_score_high(etype):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 30))
    y = rng.standard_normal(80)
    members = [KNeighborsRegressor(n_neighbors=1), DecisionTreeRegressor(random_state=0)]
    models = _fitted(members, X, y)
    # Sanity: the full-data members memorise (in-sample R2 = 1), which the old loop scored.
    assert r2_score(y, models[0].predict(X)) == pytest.approx(1.0)

    result = cross_validate_ensembles(models, ["1NN", "Tree"], X, y, [etype], n_regions=3)

    assert result.errors == {}
    assert r2_score(y, result.predictions[etype]) < 0.1


def test_cross_validation_never_scores_a_row_with_a_model_fitted_on_it():
    """Instrumented members record the rows they were fitted on; none may include the test row."""
    X, y = _signal_data(n=30)
    X = np.column_stack([np.arange(len(y), dtype=float), X])  # column 0 = row id

    class LeakDetector(BaseEstimator, RegressorMixin):
        """Raises if asked to predict a row it was fitted on."""

        def fit(self, X, y):
            self.rows_ = set(np.asarray(X)[:, 0].astype(int))
            self.ridge_ = Ridge(alpha=1.0).fit(np.asarray(X)[:, 1:], y)
            return self

        def predict(self, X):
            X = np.asarray(X)
            leaked = set(X[:, 0].astype(int)) & self.rows_
            if leaked:
                raise RuntimeError(f"predicted rows {sorted(leaked)} it was fitted on")
            return self.ridge_.predict(X[:, 1:])

    full = [LeakDetector().fit(X, y), LeakDetector().fit(X, y)]
    result = cross_validate_ensembles(full, ["a", "b"], X, y, ALL_TYPES, n_regions=2)

    # A leak would surface as a per-type error (outer scoring) or an excluded member
    # (inner weight fit), so both must be empty.
    assert result.errors == {}
    assert result.notes == []
    assert set(result.predictions) == set(ALL_TYPES)


def test_cross_validate_ensembles_rejects_full_data_boundaries():
    X, y = _signal_data()
    models = _fitted([Ridge(), PLSRegression(2)], X, y)
    with pytest.raises(ValueError, match="y_percentiles"):
        cross_validate_ensembles(
            models, ["a", "b"], X, y, ["region_weighted"], y_percentiles=[0, 1, 2]
        )


def test_cross_validate_ensembles_leaves_input_models_untouched():
    X, y = _signal_data()
    models = _fitted([Ridge(alpha=1.0), PLSRegression(n_components=3)], X, y)
    before = [m.predict(X).ravel().copy() for m in models]
    cross_validate_ensembles(models, ["a", "b"], X, y, ALL_TYPES, n_regions=3)
    for m, b in zip(models, before):
        np.testing.assert_array_equal(m.predict(X).ravel(), b)


# --- R021: genuinely held-out inner predictions ---------------------------------------


class _RecordingMeta(Ridge):
    """Stacking meta-model that keeps the features it was fitted on."""

    def fit(self, X, y, sample_weight=None):
        self.seen_X_ = np.array(X, copy=True)
        return super().fit(X, y, sample_weight=sample_weight)


def test_stacking_meta_model_sees_out_of_fold_predictions():
    X, y = _signal_data(n=50)
    members = _fitted([KNeighborsRegressor(n_neighbors=1), Ridge(alpha=1.0)], X, y)
    ens = StackingEnsemble(
        members, ["1NN", "Ridge"], meta_model=_RecordingMeta(alpha=1.0), region_aware=False, cv=5
    ).fit(X, y)
    meta = ens.meta_model  # the fitted clone

    expected = np.zeros((len(y), 2))
    for train_idx, val_idx in KFold(5, shuffle=True, random_state=42).split(X):
        for j, m in enumerate(members):
            expected[val_idx, j] = clone(m).fit(X[train_idx], y[train_idx]).predict(X[val_idx])
    np.testing.assert_allclose(meta.seen_X_, expected)
    # In-sample 1-NN predictions would equal y exactly.
    assert not np.allclose(meta.seen_X_[:, 0], y)


def test_refit_base_models_false_warns_that_weights_are_in_sample():
    X, y = _signal_data()
    members = _fitted([Ridge(), PLSRegression(2)], X, y)
    with pytest.warns(UserWarning, match="in-sample"):
        RegionAwareWeightedEnsemble(members, ["a", "b"], n_regions=2, refit_base_models=False).fit(
            X, y
        )


# --- R105: failed members ---------------------------------------------------------------


class _FailsOnSmallFits(BaseEstimator, RegressorMixin):
    """Fits only with at least ``min_rows`` rows (e.g. many PLS components on a small fold).

    The full-data fit succeeds and then predicts noise, so a failed member that kept any
    weight would visibly corrupt the ensemble.
    """

    def __init__(self, min_rows=40):
        self.min_rows = min_rows

    def fit(self, X, y):
        if len(y) < self.min_rows:
            raise ValueError(f"needs {self.min_rows} rows, got {len(y)}")
        self.scale_ = float(np.std(y)) * 10
        return self

    def predict(self, X):
        return np.random.default_rng(0).standard_normal(len(X)) * self.scale_


def _ensemble(etype, models, names, explicit_bounds, y):
    kwargs = dict(n_regions=3, cv=5)
    if explicit_bounds:
        kwargs["y_percentiles"] = np.percentile(y, [0, 33, 67, 100])
    if etype == "region_weighted":
        return RegionAwareWeightedEnsemble(models, names, **kwargs)
    if etype == "mixture_experts":
        return MixtureOfExpertsEnsemble(models, names, **kwargs)
    region_aware = etype == "region_stacking"
    return StackingEnsemble(models, names, region_aware=region_aware, **kwargs)


@pytest.mark.parametrize("position", ["first", "last"])
@pytest.mark.parametrize(
    "explicit_bounds", [False, True], ids=["default_bounds", "explicit_bounds"]
)
@pytest.mark.parametrize(
    "etype", ["region_weighted", "mixture_experts", "stacking", "region_stacking"]
)
def test_failed_member_is_removed_consistently(etype, explicit_bounds, position):
    X, y = _signal_data(n=40)
    good = _fitted([Ridge(alpha=1.0), PLSRegression(n_components=3)], X, y)
    bad = _FailsOnSmallFits(min_rows=40).fit(X, y)
    models = [bad, *good] if position == "first" else [*good, bad]
    names = ["Bad", "Ridge", "PLS"] if position == "first" else ["Ridge", "PLS", "Bad"]
    caller_models = list(models)

    with pytest.warns(UserWarning, match=OOF_FAILURE_WARNING):
        ens = _ensemble(etype, models, names, explicit_bounds, y).fit(X, y)
    reference = _ensemble(etype, good, ["Ridge", "PLS"], explicit_bounds, y).fit(X, y)

    assert ens.model_names == ["Ridge", "PLS"]
    assert len(ens.models) == 2 and all(m is not bad for m in ens.models)
    assert [name for name, _ in ens.excluded_models_] == ["Bad"]
    assert models == caller_models, "the caller's list must not be mutated"
    weights = getattr(ens, "weights_", None)
    if etype in ("region_weighted", "mixture_experts"):
        assert weights.shape[0] == 2 and np.all(np.isfinite(weights))
    preds = ens.predict(X)
    assert np.all(np.isfinite(preds))
    np.testing.assert_allclose(preds, reference.predict(X))


def test_real_pls_with_too_many_components_for_inner_folds_is_excluded():
    """Codex's trigger: 15-component PLS valid on 16 rows, invalid on ~12-row inner folds."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 40))
    y = X[:, 0] + 0.1 * rng.standard_normal(16)
    pls15 = PLSRegression(n_components=15).fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)

    with pytest.warns(UserWarning, match=OOF_FAILURE_WARNING):
        ens = StackingEnsemble([pls15, ridge], ["PLS15", "Ridge"], region_aware=False, cv=4).fit(
            X, y
        )
    assert ens.model_names == ["Ridge"]
    assert np.all(np.isfinite(ens.predict(X)))


@pytest.mark.parametrize(
    "etype", ["region_weighted", "mixture_experts", "stacking", "region_stacking"]
)
def test_all_members_failing_raises(etype):
    X, y = _signal_data(n=40)
    bad = [_FailsOnSmallFits(40).fit(X, y), _FailsOnSmallFits(40).fit(X, y)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="every base model"):
            _ensemble(etype, bad, ["a", "b"], False, y).fit(X, y)


def test_outer_cv_drops_member_that_cannot_fit_outer_fold_and_notes_it():
    X, y = _signal_data(n=40)
    members = [_FailsOnSmallFits(40).fit(X, y), *_fitted([Ridge(), PLSRegression(3)], X, y)]
    result = cross_validate_ensembles(
        members, ["Bad", "Ridge", "PLS"], X, y, ALL_TYPES, n_regions=3
    )
    assert result.errors == {}
    assert any("Bad" in note for note in result.notes)
    for etype in ALL_TYPES:
        assert np.all(np.isfinite(result.predictions[etype]))
        assert r2_score(y, result.predictions[etype]) > 0.8


def test_outer_cv_reports_error_when_no_member_fits():
    X, y = _signal_data(n=40)
    members = [_FailsOnSmallFits(40).fit(X, y), _FailsOnSmallFits(40).fit(X, y)]
    result = cross_validate_ensembles(members, ["a", "b"], X, y, ["simple_average", "stacking"])
    assert set(result.errors) == {"simple_average", "stacking"}
    assert result.predictions == {}


def test_create_ensemble_still_accepts_series_targets():
    X, y = _signal_data()
    index = pd.Index([f"S{i}" for i in range(len(y))])
    X_df = pd.DataFrame(X, index=index)
    models = _fitted([Ridge(), PLSRegression(2)], X_df, y)
    for etype in ALL_TYPES:
        ens = create_ensemble(
            models, ["a", "b"], X_df, pd.Series(y, index=index), etype, n_regions=3
        )
        assert np.all(np.isfinite(ens.predict(X_df)))


# --- Review round 1 ----------------------------------------------------------------------


class _RefusesRefit(Ridge):
    """Meta-model that raises if fitted twice, i.e. if fold state could carry over."""

    def fit(self, X, y, sample_weight=None):
        if hasattr(self, "coef_"):
            raise RuntimeError("meta-model instance reused across fits")
        return super().fit(X, y, sample_weight=sample_weight)


def test_cross_validation_clones_caller_meta_model_per_fold():
    X, y = _signal_data()
    models = _fitted([Ridge(alpha=1.0), PLSRegression(n_components=3)], X, y)
    meta = _RefusesRefit(alpha=1.0)
    result = cross_validate_ensembles(
        models, ["a", "b"], X, y, ["stacking", "region_stacking"], n_regions=3, meta_model=meta
    )
    assert result.errors == {}
    assert not hasattr(meta, "coef_"), "the caller's meta-model must stay unfitted"


def test_stacking_refit_starts_from_fresh_meta_model():
    X, y = _signal_data()
    models = _fitted([Ridge(alpha=1.0), PLSRegression(n_components=3)], X, y)
    ens = StackingEnsemble(models, ["a", "b"], meta_model=_RefusesRefit(), region_aware=False)
    ens.fit(X, y)
    ens.fit(X[:40], y[:40])  # a second fit must not see the first fit's state
    assert np.all(np.isfinite(ens.predict(X)))


@pytest.mark.parametrize("key", ["preprocessors", "preprocessor_configs"])
def test_cross_validate_ensembles_rejects_separate_preprocessing(key):
    from sklearn.preprocessing import StandardScaler

    X, y = _signal_data()
    models = _fitted([Ridge(), PLSRegression(2)], X, y)
    scalers = [StandardScaler().fit(X), StandardScaler().fit(X)]
    with pytest.raises(ValueError, match=key):
        cross_validate_ensembles(models, ["a", "b"], X, y, ["simple_average"], **{key: scalers})


def test_cross_validate_ensembles_is_regression_only():
    X, y = _signal_data()
    models = _fitted([Ridge(), PLSRegression(2)], X, y)
    labels = np.where(y > np.median(y), "high", "low")
    with pytest.raises(ValueError, match="regression targets only"):
        cross_validate_ensembles(models, ["a", "b"], X, labels, ["simple_average"])


class _Shift:
    """Row-wise 'preprocessor' that shifts every value; obvious if mis-assigned."""

    def __init__(self, offset: float):
        self.offset = offset

    def transform(self, X):
        return np.asarray(X) + self.offset


@pytest.mark.parametrize("bad_position", [0, 1])
@pytest.mark.parametrize("attr", ["preprocessors", "preprocessor_configs"])
def test_dropping_a_member_keeps_survivors_preprocessing_with_short_lists(bad_position, attr):
    """Short lists: a survivor must not inherit the dropped member's preprocessing."""
    X, y = _signal_data(n=40)
    ridge = Ridge(alpha=1.0).fit(X, y)
    pls = PLSRegression(n_components=3).fit(X + 5.0, y)  # its preprocessing is +5
    bad = _FailsOnSmallFits(min_rows=40).fit(X, y)
    if bad_position == 0:
        # [bad, pls, ridge] with a 2-long list: bad -> +100, pls -> +5, ridge -> raw
        models, names, prep = [bad, pls, ridge], ["Bad", "PLS", "Ridge"], [_Shift(100), _Shift(5)]
    else:
        # [pls, bad, ridge] with a 2-long list: pls -> +5, bad -> +100, ridge -> raw
        models, names, prep = [pls, bad, ridge], ["PLS", "Bad", "Ridge"], [_Shift(5), _Shift(100)]

    with pytest.warns(UserWarning, match=OOF_FAILURE_WARNING):
        ens = StackingEnsemble(models, names, region_aware=False, cv=5, **{attr: prep}).fit(X, y)
    reference = StackingEnsemble(
        [pls, ridge], ["PLS", "Ridge"], region_aware=False, cv=5, **{attr: [_Shift(5)]}
    ).fit(X, y)

    assert ens.model_names == ["PLS", "Ridge"]
    assert ens._get_preprocessor(1) is None
    np.testing.assert_allclose(ens.predict(X), reference.predict(X))


def test_create_auto_ensembles_single_sample_gives_nan_not_calibration():
    X, y = _signal_data(n=1)
    results_df = pd.DataFrame(
        {
            "Model": ["A", "B"],
            "regional_rmse": [
                {"Q1": 0.1, "Q2": 0.2, "Q3": 0.1, "Q4": 0.2},
                {"Q1": 0.2, "Q2": 0.1, "Q3": 0.2, "Q4": 0.1},
            ],
        }
    )

    def reconstruct(row, X_train, y_train):
        return Ridge(alpha=1.0).fit(X_train, y_train), row["Model"]

    with pytest.warns(UserWarning, match="fewer than 2 samples"):
        out = create_auto_ensembles(results_df, X, y, "regression", reconstruct, list(range(12)))
    assert out
    for info in out.values():
        assert np.isnan(info["metrics"]["r2"]) and np.isnan(info["metrics"]["rmse"])


@pytest.mark.parametrize("etype", ["region_weighted", "mixture_experts", "stacking"])
def test_ensemble_with_excluded_member_round_trips_through_save_and_load(tmp_path, etype):
    from spectral_predict.model_io import load_ensemble, save_ensemble

    X, y = _signal_data(n=40)
    good = _fitted([Ridge(alpha=1.0), PLSRegression(n_components=3)], X, y)
    bad = _FailsOnSmallFits(min_rows=40).fit(X, y)
    with pytest.warns(UserWarning, match=OOF_FAILURE_WARNING):
        ens = _ensemble(etype, [good[0], bad, good[1]], ["Ridge", "Bad", "PLS"], False, y)
        ens.fit(X, y)
    path = tmp_path / f"{etype}.dasp"
    wavelengths = [1000.0 + i for i in range(X.shape[1])]
    save_ensemble(
        ens,
        str(path),
        {
            "ensemble_type": etype,
            "task_type": "regression",
            "wavelengths": wavelengths,
            "n_vars": len(wavelengths),
        },
    )

    loaded = load_ensemble(str(path))
    assert loaded["model_names"] == ["Ridge", "PLS"]
    assert loaded["config"]["n_models"] == 2
    np.testing.assert_allclose(loaded["ensemble"].predict(X), ens.predict(X))
