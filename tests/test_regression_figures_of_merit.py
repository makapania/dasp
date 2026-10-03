"""Regression figures of merit (QW5): scoring.regression_figures_of_merit.

Definitions follow Bellon-Maurel et al. 2010, TrAC 29(9):1073-1081 (Eqs. 1-4,
RPIQ in §5.3 / Table 2) with SEP = bias-corrected, n-1 df, and the slope test
regressing the reference on the prediction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from spectral_predict.scoring import create_results_dataframe, regression_figures_of_merit


def test_known_residual_vector_hand_computed():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array([0.1, -0.2, 0.3, 0.0, 0.3])
    fom = regression_figures_of_merit(y, y + e, context="validation")

    # e_bar = 0.1; centred SS = 0.18; mean(e^2) = 0.046
    assert fom["n"] == 5
    assert fom["Bias"] == pytest.approx(0.1)
    assert fom["RMSE"] == pytest.approx(math.sqrt(0.046))
    assert fom["SEP"] == pytest.approx(math.sqrt(0.18 / 4))
    assert fom["SEPc"] == pytest.approx(math.sqrt(0.18 / 5))
    assert fom["MAE"] == pytest.approx(0.18)
    # ddof=0 SD of 1..5 = sqrt(2); quartiles (linear) 2 and 4; range 4
    assert fom["SD_ref"] == pytest.approx(math.sqrt(2))
    assert fom["IQR_ref"] == pytest.approx(2.0)
    assert fom["RPD"] == pytest.approx(math.sqrt(2) / math.sqrt(0.046))
    assert fom["RPIQ"] == pytest.approx(2.0 / math.sqrt(0.046))
    assert fom["RER"] == pytest.approx(4.0 / math.sqrt(0.046))
    t = 0.1 * math.sqrt(5) / math.sqrt(0.045)
    assert fom["Bias_t"] == pytest.approx(t)
    assert fom["Bias_p"] == pytest.approx(2 * stats.t.sf(t, 4))
    assert fom["test_role"] == "acceptance"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 3, 10, 57])
def test_rmse_decomposition_identities(seed, n):
    rng = np.random.default_rng(seed)
    y = rng.normal(10, 3, n)
    yhat = y + rng.normal(0.4, 1.0, n)
    fom = regression_figures_of_merit(y, yhat, context="cv")
    # Bellon-Maurel Eq. 2 (m denominators) and the n-1 SEP form
    assert fom["RMSE"] ** 2 == pytest.approx(fom["Bias"] ** 2 + fom["SEPc"] ** 2)
    assert fom["RMSE"] ** 2 == pytest.approx(fom["Bias"] ** 2 + (n - 1) * fom["SEP"] ** 2 / n)
    assert fom["SEP"] == pytest.approx(np.std(yhat - y, ddof=1))
    assert fom["Bias"] == pytest.approx(np.mean(yhat - y))


def test_additive_bias_detected_slope_not():
    rng = np.random.default_rng(3)
    y = rng.uniform(0, 20, 60)
    yhat = y + 2.0 + rng.normal(0, 0.3, 60)
    fom = regression_figures_of_merit(y, yhat)
    assert fom["Bias"] == pytest.approx(2.0, abs=0.15)
    assert fom["Bias_t"] > 0  # positive = over-prediction
    assert fom["Bias_p"] < 1e-6
    assert fom["Slope"] == pytest.approx(1.0, abs=0.02)
    assert fom["Slope_p"] > 0.01
    # Intercept is not the bias: reference = a + b*prediction gives a ~ -2
    assert fom["Intercept"] == pytest.approx(-2.0, abs=0.3)


def test_slope_distortion_reference_on_prediction_orientation():
    rng = np.random.default_rng(4)
    y = rng.uniform(0, 20, 80)
    yhat = y.mean() + 0.5 * (y - y.mean()) + rng.normal(0, 0.1, 80)
    fom = regression_figures_of_merit(y, yhat)
    # y regressed on yhat: slope ~ 2 (regressing yhat on y would give ~0.5)
    assert fom["Slope"] == pytest.approx(2.0, abs=0.1)
    assert fom["Slope_p"] < 1e-6
    assert abs(fom["Bias"]) < 0.1
    assert fom["Bias_p"] > 0.01
    expected = stats.linregress(yhat, y)
    assert fom["Slope"] == pytest.approx(expected.slope)
    assert fom["Intercept"] == pytest.approx(expected.intercept)
    t_expected = (expected.slope - 1.0) / expected.stderr
    assert fom["Slope_t"] == pytest.approx(t_expected)


def test_perfect_prediction_ratios_are_inf_not_zero():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    fom = regression_figures_of_merit(y, y.copy())
    assert fom["RMSE"] == 0.0
    assert fom["SEP"] == 0.0
    assert fom["RPD"] == math.inf
    assert fom["RPIQ"] == math.inf
    assert fom["RER"] == math.inf
    assert fom["R2"] == 1.0
    assert fom["CCC"] == 1.0
    assert fom["Slope"] == pytest.approx(1.0)
    assert math.isnan(fom["Bias_t"]) and math.isnan(fom["Bias_p"])  # 0/0


def test_pure_offset_has_infinite_bias_t():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    fom = regression_figures_of_merit(y, y + 0.5)
    assert fom["SEP"] == pytest.approx(0.0, abs=1e-12)
    assert fom["Bias_p"] == pytest.approx(0.0, abs=1e-6)


def test_constant_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fom = regression_figures_of_merit(y, np.full(5, 3.0))
    assert math.isnan(fom["Slope"]) and math.isnan(fom["Intercept"])
    assert math.isnan(fom["Slope_p"])
    assert fom["CCC"] == 0.0
    assert fom["R2"] == pytest.approx(0.0)
    assert np.isfinite(fom["RPD"])


def test_constant_reference_perfect_prediction_is_nan_ratio():
    y = np.full(4, 3.0)
    fom = regression_figures_of_merit(y, y.copy())
    assert math.isnan(fom["RPD"]) and math.isnan(fom["RPIQ"]) and math.isnan(fom["RER"])


def test_small_n():
    one = regression_figures_of_merit([2.0], [2.5])
    assert one["n"] == 1
    assert one["Bias"] == pytest.approx(0.5)
    assert one["RMSE"] == pytest.approx(0.5)
    for k in ("SEP", "R2", "Bias_t", "Bias_p", "Slope", "Slope_p"):
        assert math.isnan(one[k]), k
    two = regression_figures_of_merit([1.0, 2.0], [1.5, 2.1])
    assert np.isfinite(two["SEP"]) and np.isfinite(two["Slope"])
    assert math.isnan(two["Slope_p"])  # needs n >= 3
    empty = regression_figures_of_merit([], [])
    assert empty["n"] == 0 and math.isnan(empty["RMSE"])


def test_nan_input_propagates_to_every_metric():
    fom = regression_figures_of_merit([1.0, 2.0, np.nan, 4.0], [1.1, 2.0, 3.0, 4.2])
    assert fom["n"] == 4
    numeric = [k for k, v in fom.items() if isinstance(v, float)]
    assert numeric and all(math.isnan(fom[k]) for k in numeric)


def test_context_and_shape_validation():
    assert regression_figures_of_merit([1, 2, 3], [1, 2, 3], context="cv")["test_role"] == (
        "diagnostic"
    )
    assert (
        regression_figures_of_merit([1, 2, 3], [1, 2, 3], context="calibration")["test_role"]
        == "diagnostic"
    )
    with pytest.raises(ValueError):
        regression_figures_of_merit([1, 2], [1, 2], context="test")
    with pytest.raises(ValueError):
        regression_figures_of_merit([1, 2, 3], [1, 2])


def test_results_columns_declared():
    cols = list(create_results_dataframe("regression").columns)
    for c in ("RMSEcv", "RPD", "Bias", "RER", "CCCcv", "SECV", "RPIQ"):
        assert c in cols


# ---------------------------------------------------------------------------
# Wiring through run_search: existing columns keep their values, new columns
# obey the definitions, external validation gets the same FoM.
# ---------------------------------------------------------------------------


def _regression_data(n=36, p=40, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, :5] @ np.array([1.0, -0.5, 0.8, 0.3, 0.2]) + rng.normal(0, 0.4, n) + 5.0
    Xdf = pd.DataFrame(X, columns=[float(1100 + 2 * i) for i in range(p)])
    return Xdf, pd.Series(y)


def _run(Xdf, y, **kw):
    from spectral_predict.search import run_search

    df, enc = run_search(
        Xdf,
        y,
        "regression",
        folds=4,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        max_n_components=3,
        enable_variable_subsets=False,
        enable_region_subsets=False,
        **kw,
    )
    assert enc is None
    return df


@pytest.mark.parametrize("cv_strategy", ["kfold", "repeated_kfold", "loo"])
def test_run_search_cv_fom_columns(cv_strategy):
    Xdf, y = _regression_data()
    df = _run(Xdf, y, cv_strategy=cv_strategy, cv_n_repeats=3)
    n = len(y)
    yv = y.to_numpy()
    for _, r in df.iterrows():
        rmse = r["RMSEcv"]
        # existing columns: unchanged definitions
        assert r["RPD"] == pytest.approx(np.std(yv) / rmse)
        assert r["RER"] == pytest.approx(np.ptp(yv) / rmse)
        # new columns; n is the sample count even under repeated CV (one
        # averaged prediction per sample), not n * repeats
        assert rmse**2 == pytest.approx(r["Bias"] ** 2 + (n - 1) * r["SECV"] ** 2 / n)
        q1, q3 = np.percentile(yv, [25, 75])
        assert r["RPIQ"] == pytest.approx((q3 - q1) / rmse)


def test_run_search_external_validation_fom():
    Xdf, y = _regression_data(n=48)
    X_cal, y_cal = Xdf.iloc[:36], y.iloc[:36].reset_index(drop=True)
    X_val, y_val = Xdf.iloc[36:].to_numpy(), y.iloc[36:].to_numpy()
    df = _run(
        X_cal.reset_index(drop=True),
        y_cal,
        cv_strategy="kfold",
        X_validation=X_val,
        y_validation=y_val,
        compute_validation=True,
        validation_top_n=5,
    )
    for c in (
        "RMSEP",
        "R2pred",
        "SEP",
        "Biaspred",
        "RPDpred",
        "RPIQpred",
        "RERpred",
        "CCCpred",
        "Slopepred",
        "Interceptpred",
        "Bias_p_pred",
        "Slope_p_pred",
    ):
        assert c in df.columns, c
    rows = df[df["RMSEP"].notna()]
    assert len(rows) > 0
    m = len(y_val)
    q1, q3 = np.percentile(y_val, [25, 75])
    for _, r in rows.iterrows():
        assert r["RMSEP"] ** 2 == pytest.approx(r["Biaspred"] ** 2 + (m - 1) * r["SEP"] ** 2 / m)
        assert r["RPIQpred"] == pytest.approx((q3 - q1) / r["RMSEP"])
        assert r["RPDpred"] == pytest.approx(np.std(y_val) / r["RMSEP"])
