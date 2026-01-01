"""
Smoke tests for Spectral Predict.

Quick sanity checks that verify basic functionality without extensive computation.
All tests in this module should complete in < 30 seconds total.
"""

import pytest
import numpy as np


@pytest.mark.smoke
def test_import_spectral_predict():
    """Package imports without error."""
    import spectral_predict
    assert spectral_predict is not None


@pytest.mark.smoke
def test_import_search():
    """Core search module imports."""
    from spectral_predict import search
    assert hasattr(search, "run_search")
    assert callable(search.run_search)


@pytest.mark.smoke
def test_import_models():
    """Models module imports."""
    from spectral_predict import models
    assert hasattr(models, "PLSRegression")
    assert hasattr(models, "Ridge")
    assert hasattr(models, "Lasso")
    assert hasattr(models, "RandomForestRegressor")


@pytest.mark.smoke
def test_import_preprocess():
    """Preprocessing module imports."""
    from spectral_predict import preprocess
    assert hasattr(preprocess, "SNV")
    assert hasattr(preprocess, "SavgolDerivative")
    assert hasattr(preprocess, "build_preprocessing_pipeline")


@pytest.mark.smoke
def test_import_io():
    """IO module imports."""
    from spectral_predict import io
    assert hasattr(io, "read_spectra")
    assert callable(io.read_spectra)
    assert hasattr(io, "read_csv_spectra")
    assert hasattr(io, "read_excel_spectra")


@pytest.mark.smoke
def test_import_variable_selection():
    """Variable selection module imports."""
    from spectral_predict import variable_selection
    # Check for any exported functions/classes
    exports = [x for x in dir(variable_selection) if not x.startswith("_")]
    assert len(exports) > 0


@pytest.mark.smoke
def test_import_calibration_transfer():
    """Calibration transfer module imports."""
    from spectral_predict import calibration_transfer
    # Check for any exported functions/classes
    exports = [x for x in dir(calibration_transfer) if not x.startswith("_")]
    assert len(exports) > 0


@pytest.mark.smoke
def test_basic_pls_fit():
    """PLS model can fit minimal data."""
    from spectral_predict.models import PLSRegression

    # Create minimal synthetic data
    np.random.seed(42)
    X = np.random.randn(20, 10)
    y = np.random.randn(20)

    # Create and fit model
    model = PLSRegression(n_components=2)
    model.fit(X, y)

    # Verify model has been fitted
    assert hasattr(model, "coef_")
    assert model.coef_ is not None

    # Verify prediction works
    y_pred = model.predict(X)
    # PLS can return either (20,) or (20, 1) depending on sklearn version
    assert y_pred.shape in [(20,), (20, 1)]
    assert not np.isnan(y_pred).any()


@pytest.mark.smoke
def test_basic_ridge_fit():
    """Ridge model can fit minimal data."""
    from spectral_predict.models import Ridge

    # Create minimal synthetic data
    np.random.seed(42)
    X = np.random.randn(20, 10)
    y = np.random.randn(20)

    # Create and fit model
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Verify model has been fitted
    assert hasattr(model, "coef_")
    assert model.coef_ is not None

    # Verify prediction works
    y_pred = model.predict(X)
    assert y_pred.shape == (20,)
    assert not np.isnan(y_pred).any()


@pytest.mark.smoke
def test_snv_preprocessing():
    """SNV preprocessing works on minimal data."""
    from spectral_predict.preprocess import SNV

    # Create minimal synthetic spectra
    np.random.seed(42)
    X = np.random.randn(10, 50) + 10  # Add offset to avoid zeros

    # Apply SNV preprocessing
    snv = SNV()
    X_preprocessed = snv.fit_transform(X)

    # Verify output shape is correct
    assert X_preprocessed.shape == X.shape

    # Verify SNV properties: each row should have mean ~0 and std ~1
    row_means = np.mean(X_preprocessed, axis=1)
    row_stds = np.std(X_preprocessed, axis=1, ddof=1)

    assert np.allclose(row_means, 0, atol=1e-8)
    assert np.allclose(row_stds, 1, atol=0.05)  # Relaxed tolerance for std


@pytest.mark.smoke
def test_savgol_preprocessing():
    """Savitzky-Golay preprocessing works on minimal data."""
    from spectral_predict.preprocess import SavgolDerivative

    # Create minimal synthetic spectra
    np.random.seed(42)
    X = np.random.randn(10, 50)

    # Apply Savitzky-Golay preprocessing (1st derivative)
    savgol = SavgolDerivative(deriv=1, window=7, polyorder=2)

    X_preprocessed = savgol.fit_transform(X)

    # Verify output shape is correct
    assert X_preprocessed.shape == X.shape

    # Verify not all values are the same (derivative should vary)
    assert not np.allclose(X_preprocessed, X_preprocessed[0, 0])


@pytest.mark.smoke
def test_model_registry():
    """Model registry contains expected models."""
    from spectral_predict.model_registry import get_supported_models, ALL_MODELS

    # Verify registry exists and has expected models
    model_names = get_supported_models()
    assert len(model_names) > 0
    # Verify ALL_MODELS is populated
    assert len(ALL_MODELS) > 0
    # The registry should have some common models (case-insensitive check)
    assert any("pls" in name.lower() for name in model_names)


@pytest.mark.smoke
def test_numpy_scipy_available():
    """Required scientific computing libraries are available."""
    import numpy
    import scipy
    import pandas
    import sklearn

    # Verify basic functionality
    arr = numpy.array([1, 2, 3])
    assert arr.sum() == 6

    from scipy import stats
    assert hasattr(stats, "pearsonr")

    df = pandas.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3


@pytest.mark.smoke
def test_ml_libraries_available():
    """Required ML libraries are available."""
    import xgboost
    import lightgbm
    import catboost

    # Verify basic instantiation works
    xgb_model = xgboost.XGBRegressor(n_estimators=1, random_state=42)
    lgb_model = lightgbm.LGBMRegressor(n_estimators=1, random_state=42, verbose=-1)
    cat_model = catboost.CatBoostRegressor(iterations=1, random_state=42, verbose=False)

    assert xgb_model is not None
    assert lgb_model is not None
    assert cat_model is not None
