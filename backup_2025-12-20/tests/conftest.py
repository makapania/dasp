"""
Pytest configuration and shared fixtures for Spectral Predict tests.

This module provides:
- Path fixtures for project directories
- Synthetic data fixtures for testing
- Model fixtures with pre-trained models
- Example data fixtures from the example/ directory
- Custom pytest markers for test categorization
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# Import synthetic data generators
from tests.fixtures.synthetic_data import (
    generate_classification_spectra,
    generate_imbalanced_data,
    generate_outlier_data,
    generate_spectral_data,
)


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers",
        "smoke: Quick smoke tests that verify basic functionality"
    )
    config.addinivalue_line(
        "markers",
        "unit: Unit tests for individual functions and methods"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests that test multiple components together"
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that take a long time to run (>10 seconds)"
    )
    config.addinivalue_line(
        "markers",
        "numerical: Tests that verify numerical accuracy and stability"
    )
    config.addinivalue_line(
        "markers",
        "regression: Regression tests that verify bug fixes stay fixed"
    )
    config.addinivalue_line(
        "markers",
        "io: Tests for file input/output operations"
    )
    config.addinivalue_line(
        "markers",
        "gui: Tests for GUI components (may require display)"
    )


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    The project root contains pyproject.toml and is the parent of the tests/ directory.

    Returns
    -------
    Path
        Absolute path to project root

    Examples
    --------
    >>> def test_config_exists(project_root):
    ...     assert (project_root / "pyproject.toml").exists()
    """
    # tests/conftest.py -> tests/ -> project_root/
    return Path(__file__).parent.parent.resolve()


@pytest.fixture(scope="session")
def example_data_dir(project_root: Path) -> Path:
    """
    Return the path to the example/ directory containing sample data.

    Returns
    -------
    Path
        Absolute path to example/ directory

    Examples
    --------
    >>> def test_bone_collagen_exists(example_data_dir):
    ...     assert (example_data_dir / "BoneCollagen.csv").exists()
    """
    return project_root / "example"


@pytest.fixture(scope="session")
def gold_standard_dir(project_root: Path) -> Path:
    """
    Return the path to tests/gold_standards/ for reference results.

    This directory can contain known-good outputs for regression testing.

    Returns
    -------
    Path
        Absolute path to tests/gold_standards/ directory

    Examples
    --------
    >>> def test_pls_predictions(gold_standard_dir):
    ...     ref = pd.read_csv(gold_standard_dir / "pls_predictions.csv")
    ...     # Compare with current predictions
    """
    gold_dir = project_root / "tests" / "gold_standards"
    gold_dir.mkdir(exist_ok=True)
    return gold_dir


# =============================================================================
# Synthetic Data Fixtures - Small
# =============================================================================


@pytest.fixture
def synthetic_spectra_small() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate small synthetic spectral dataset for quick tests.

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (50, 200)
        Wavelengths from 1000-2500nm
        Columns are wavelength strings (e.g., "1000.0")
    y : pd.Series
        Target values with shape (50,)

    Notes
    -----
    - Uses seed=42 for reproducibility
    - 50 samples, 200 wavelengths
    - 5 informative wavelengths
    - Low noise level (0.1)
    - Ideal for quick unit tests

    Examples
    --------
    >>> def test_snv(synthetic_spectra_small):
    ...     X, y = synthetic_spectra_small
    ...     from spectral_predict.preprocess import snv
    ...     X_snv = snv(X)
    ...     assert X_snv.shape == X.shape
    """
    n_wavelengths = 200
    wavelengths = np.linspace(1000, 2500, n_wavelengths)
    wavelength_names = [str(wl) for wl in wavelengths]

    # Generate using the standard generator
    X, y = generate_spectral_data(
        n_samples=50,
        n_wavelengths=n_wavelengths,
        n_informative=5,
        noise_level=0.1,
        seed=42,
    )

    # Replace column names with 1000-2500nm range
    X.columns = wavelength_names

    return X, y


@pytest.fixture
def synthetic_spectra_medium() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate medium synthetic spectral dataset for standard tests.

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (100, 500)
        Wavelengths from 400-2400nm (visible + NIR)
        Columns are wavelength strings
    y : pd.Series
        Target values with shape (100,)

    Notes
    -----
    - Uses seed=42 for reproducibility
    - 100 samples, 500 wavelengths
    - Covers visible (400-780nm) and NIR (780-2400nm)
    - Good for testing preprocessing chains

    Examples
    --------
    >>> def test_savgol(synthetic_spectra_medium):
    ...     X, y = synthetic_spectra_medium
    ...     from spectral_predict.preprocess import savgol_derivative
    ...     X_d1 = savgol_derivative(X, deriv=1)
    ...     assert X_d1.shape == X.shape
    """
    n_wavelengths = 500
    wavelengths = np.linspace(400, 2400, n_wavelengths)
    wavelength_names = [str(wl) for wl in wavelengths]

    X, y = generate_spectral_data(
        n_samples=100,
        n_wavelengths=n_wavelengths,
        n_informative=7,
        noise_level=0.15,
        seed=42,
    )

    X.columns = wavelength_names

    return X, y


@pytest.fixture
def synthetic_spectra_large() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate large synthetic spectral dataset matching typical NIR spectrometers.

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (200, 2151)
        Wavelengths from 350-2500nm at 1nm intervals
        Columns are wavelength strings
    y : pd.Series
        Target values with shape (200,)

    Notes
    -----
    - Uses seed=42 for reproducibility
    - 200 samples, 2151 wavelengths (typical NIR spectrometer)
    - Matches common instrument specifications
    - Use for performance and scalability tests

    Examples
    --------
    >>> def test_variable_selection_performance(synthetic_spectra_large):
    ...     X, y = synthetic_spectra_large
    ...     from spectral_predict.variable_selection import run_spa
    ...     selected = run_spa(X, y, n_vars=20)
    ...     assert len(selected) == 20
    """
    n_wavelengths = 2151  # 350-2500nm at 1nm intervals
    wavelengths = np.linspace(350, 2500, n_wavelengths)
    wavelength_names = [str(wl) for wl in wavelengths]

    X, y = generate_spectral_data(
        n_samples=200,
        n_wavelengths=n_wavelengths,
        n_informative=10,
        noise_level=0.2,
        seed=42,
    )

    X.columns = wavelength_names

    return X, y


# =============================================================================
# Synthetic Data Fixtures - Classification
# =============================================================================


@pytest.fixture
def classification_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate balanced binary classification dataset.

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (100, 200)
        Columns are wavelength strings
    y : pd.Series
        Binary class labels (0 or 1) with shape (100,)
        50 samples per class

    Notes
    -----
    - Uses seed=42 for reproducibility
    - 100 samples total (50 per class)
    - 200 wavelengths
    - Medium separability (separation=1.0)
    - Good for testing classification models

    Examples
    --------
    >>> def test_pls_da(classification_data):
    ...     X, y = classification_data
    ...     from spectral_predict.models import train_pls_da
    ...     model = train_pls_da(X, y, n_components=5)
    ...     assert model is not None
    """
    X, y = generate_classification_spectra(
        n_samples=100,
        n_wavelengths=200,
        n_classes=2,
        separation=1.0,
        seed=42,
    )

    return X, y


@pytest.fixture
def imbalanced_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate severely imbalanced binary classification dataset.

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (110, 200)
        Columns are wavelength strings
    y : pd.Series
        Binary class labels with 10:1 imbalance
        Class 0: 100 samples, Class 1: 10 samples

    Notes
    -----
    - Uses seed=42 for reproducibility
    - 110 samples total (100 majority, 10 minority)
    - 200 wavelengths
    - 10:1 imbalance ratio
    - Useful for testing SMOTE and class weighting

    Examples
    --------
    >>> def test_smote(imbalanced_data):
    ...     X, y = imbalanced_data
    ...     from spectral_predict.imbalance import apply_smote
    ...     X_res, y_res = apply_smote(X, y)
    ...     assert y_res.value_counts()[0] == y_res.value_counts()[1]
    """
    X, y = generate_imbalanced_data(
        n_samples=110,
        n_features=200,
        imbalance_ratio=10.0,
        n_classes=2,
        seed=42,
    )

    return X, y


@pytest.fixture
def outlier_data() -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Generate spectral data with known outliers.

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (100, 200)
        Contains 5 outlier samples at known positions
    y : pd.Series
        Target values with shape (100,)
    outlier_indices : np.ndarray
        Indices of outlier samples [0, 1, 2, 3, 4]

    Notes
    -----
    - Uses seed=42 for reproducibility
    - 100 samples total (95 clean, 5 outliers)
    - 200 wavelengths
    - Outliers are at indices [0, 1, 2, 3, 4]
    - Mixed outlier types (spectral + leverage)
    - Useful for testing outlier detection algorithms

    Examples
    --------
    >>> def test_outlier_detection(outlier_data):
    ...     X, y, true_outliers = outlier_data
    ...     from spectral_predict.outliers import detect_outliers
    ...     detected = detect_outliers(X, method="isolation_forest")
    ...     # Check overlap with true outliers
    ...     overlap = len(set(detected) & set(true_outliers))
    ...     assert overlap >= 3  # Should detect at least 3/5
    """
    X, y, outlier_indices = generate_outlier_data(
        n_samples=100,
        n_wavelengths=200,
        n_outliers=5,
        outlier_type="both",
        seed=42,
    )

    return X, y, outlier_indices


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def trained_pls_model(synthetic_spectra_small) -> Tuple[object, list]:
    """
    Return a pre-trained PLS model with wavelength list.

    Returns
    -------
    model : PLSRegression or dict
        Trained PLS model with 5 components
    wavelengths : list
        List of wavelength strings used for training

    Notes
    -----
    - Trained on synthetic_spectra_small (50 samples, 200 wavelengths)
    - 5 PLS components
    - Deterministic (seed=42)
    - Useful for testing prediction, serialization, and transfer

    Examples
    --------
    >>> def test_model_predictions(trained_pls_model, synthetic_spectra_small):
    ...     model, wavelengths = trained_pls_model
    ...     X, y = synthetic_spectra_small
    ...     predictions = model.predict(X)
    ...     assert len(predictions) == len(y)
    """
    from sklearn.cross_decomposition import PLSRegression

    X, y = synthetic_spectra_small
    wavelengths = X.columns.tolist()

    # Train PLS model
    model = PLSRegression(n_components=5)
    model.fit(X, y)

    return model, wavelengths


# =============================================================================
# Example Data Fixtures
# =============================================================================


@pytest.fixture
def bone_collagen_csv(example_data_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """
    Load the BoneCollagen.csv example dataset.

    Returns
    -------
    data : pd.DataFrame
        Full BoneCollagen dataset with columns:
        - File Number
        - Sample no.
        - %Collagen (continuous target)
        - CollagenCat (categorical target: Low/Medium/High)
    file_path : Path
        Path to the CSV file

    Notes
    -----
    - Contains 50 NIR spectra of bone samples
    - Both regression (%Collagen) and classification (CollagenCat) targets
    - First 4 columns are metadata, rest are wavelengths
    - Only loaded if file exists (will skip test otherwise)

    Examples
    --------
    >>> def test_bone_collagen_loading(bone_collagen_csv):
    ...     data, path = bone_collagen_csv
    ...     assert "%Collagen" in data.columns
    ...     assert "CollagenCat" in data.columns
    ...     assert len(data) == 50
    """
    csv_path = example_data_dir / "BoneCollagen.csv"

    if not csv_path.exists():
        pytest.skip(f"BoneCollagen.csv not found at {csv_path}")

    # Load with BOM handling (file starts with UTF-8 BOM)
    data = pd.read_csv(csv_path, encoding="utf-8-sig")

    return data, csv_path


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory for test outputs.

    Returns
    -------
    Path
        Temporary directory that will be cleaned up after test

    Notes
    -----
    - Automatically created and cleaned up by pytest
    - Use for saving test outputs, plots, models, etc.
    - Each test gets a unique temporary directory

    Examples
    --------
    >>> def test_save_model(trained_pls_model, temp_output_dir):
    ...     model, wavelengths = trained_pls_model
    ...     output_path = temp_output_dir / "model.pkl"
    ...     save_model(model, output_path)
    ...     assert output_path.exists()
    """
    return tmp_path


@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Reset numpy random seed before each test for reproducibility.

    This ensures tests are deterministic even if they modify the global
    random state. Applied automatically to all tests.
    """
    np.random.seed(42)
    yield
    # Cleanup after test (if needed)


# =============================================================================
# Session-scoped fixtures for expensive operations
# =============================================================================


@pytest.fixture(scope="session")
def example_asd_files(example_data_dir: Path) -> list:
    """
    Return list of example ASD files for testing file I/O.

    Returns
    -------
    list of Path
        List of paths to .asd files in example/ directory

    Notes
    -----
    - Only returns files that actually exist
    - Empty list if no ASD files found (test should handle gracefully)
    - Session-scoped for efficiency

    Examples
    --------
    >>> def test_asd_reading(example_asd_files):
    ...     if not example_asd_files:
    ...         pytest.skip("No ASD files available")
    ...     from spectral_predict.io import read_asd
    ...     data = read_asd(example_asd_files[0])
    ...     assert data is not None
    """
    asd_files = list(example_data_dir.glob("*.asd"))
    return sorted(asd_files)


@pytest.fixture(scope="session")
def example_spc_files(example_data_dir: Path) -> list:
    """
    Return list of example SPC files for testing file I/O.

    Returns
    -------
    list of Path
        List of paths to .spc files in example/ directory

    Examples
    --------
    >>> def test_spc_reading(example_spc_files):
    ...     if not example_spc_files:
    ...         pytest.skip("No SPC files available")
    ...     from spectral_predict.io import read_spc
    ...     data = read_spc(example_spc_files[0])
    ...     assert data is not None
    """
    spc_files = list(example_data_dir.glob("*.spc"))
    return sorted(spc_files)
