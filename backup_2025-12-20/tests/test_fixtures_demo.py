"""
Demonstration tests showing how to use the pytest fixtures.

This file serves as both documentation and a smoke test for the fixture infrastructure.
Run with: pytest tests/test_fixtures_demo.py -v
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.smoke
def test_synthetic_spectra_small(synthetic_spectra_small):
    """Test that small synthetic data has correct shape and properties."""
    X, y = synthetic_spectra_small

    # Check shapes
    assert X.shape == (50, 200), "Should have 50 samples and 200 wavelengths"
    assert y.shape == (50,), "Should have 50 target values"

    # Check wavelength range (1000-2500nm)
    wavelengths = X.columns.astype(float)
    assert wavelengths.min() == pytest.approx(1000.0, rel=1e-3)
    assert wavelengths.max() == pytest.approx(2500.0, rel=1e-3)

    # Check data types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


@pytest.mark.smoke
def test_synthetic_spectra_medium(synthetic_spectra_medium):
    """Test that medium synthetic data has correct shape and properties."""
    X, y = synthetic_spectra_medium

    assert X.shape == (100, 500)
    assert y.shape == (100,)

    # Check wavelength range (400-2400nm)
    wavelengths = X.columns.astype(float)
    assert wavelengths.min() == pytest.approx(400.0, rel=1e-3)
    assert wavelengths.max() == pytest.approx(2400.0, rel=1e-3)


@pytest.mark.smoke
def test_synthetic_spectra_large(synthetic_spectra_large):
    """Test that large synthetic data matches typical NIR spectrometer."""
    X, y = synthetic_spectra_large

    assert X.shape == (200, 2151)
    assert y.shape == (200,)

    # Check wavelength range (350-2500nm)
    wavelengths = X.columns.astype(float)
    assert wavelengths.min() == pytest.approx(350.0, rel=1e-3)
    assert wavelengths.max() == pytest.approx(2500.0, rel=1e-3)


@pytest.mark.smoke
def test_classification_data(classification_data):
    """Test that classification data is balanced."""
    X, y = classification_data

    assert X.shape == (100, 200)
    assert y.shape == (100,)

    # Check class balance
    class_counts = y.value_counts()
    assert len(class_counts) == 2, "Should have 2 classes"
    assert class_counts[0] == 50, "Class 0 should have 50 samples"
    assert class_counts[1] == 50, "Class 1 should have 50 samples"


@pytest.mark.smoke
def test_imbalanced_data(imbalanced_data):
    """Test that imbalanced data has correct class distribution."""
    X, y = imbalanced_data

    assert X.shape == (110, 200)
    assert y.shape == (110,)

    # Check class imbalance
    class_counts = y.value_counts()
    assert class_counts[0] == 100, "Majority class should have 100 samples"
    assert class_counts[1] == 10, "Minority class should have 10 samples"

    # Verify 10:1 ratio
    ratio = class_counts[0] / class_counts[1]
    assert ratio == pytest.approx(10.0)


@pytest.mark.smoke
def test_outlier_data(outlier_data):
    """Test that outlier data has known outlier positions."""
    X, y, outlier_indices = outlier_data

    assert X.shape == (100, 200)
    assert y.shape == (100,)
    assert len(outlier_indices) == 5, "Should have 5 outliers"

    # Check outlier indices are at expected positions
    expected_indices = np.array([0, 1, 2, 3, 4])
    np.testing.assert_array_equal(outlier_indices, expected_indices)


@pytest.mark.smoke
def test_trained_pls_model(trained_pls_model, synthetic_spectra_small):
    """Test that trained PLS model can make predictions."""
    model, wavelengths = trained_pls_model
    X, y = synthetic_spectra_small

    # Check wavelengths match
    assert wavelengths == X.columns.tolist()

    # Test predictions
    predictions = model.predict(X)
    assert predictions.shape == (50,) or predictions.shape == (50, 1)

    # Model should have reasonable R² on training data
    from sklearn.metrics import r2_score

    y_pred = predictions.ravel()
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"Model should fit training data reasonably (R²={r2:.3f})"


@pytest.mark.io
def test_bone_collagen_csv(bone_collagen_csv):
    """Test that BoneCollagen.csv loads correctly."""
    data, path = bone_collagen_csv

    # Check expected columns exist
    assert "File Number" in data.columns
    assert "%Collagen" in data.columns
    assert "CollagenCat" in data.columns

    # Check data has expected shape (49 spectra - Spectrum 00024 is .spc format)
    assert len(data) == 49

    # Check categorical values
    categories = data["CollagenCat"].unique()
    assert set(categories).issubset({"Low", "Medium", "High"})

    # Check collagen values are numeric and in reasonable range
    collagen_values = data["%Collagen"]
    assert collagen_values.min() > 0
    assert collagen_values.max() < 100


@pytest.mark.smoke
def test_project_root_exists(project_root):
    """Test that project root contains expected files."""
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / "src").is_dir()


@pytest.mark.smoke
def test_example_data_dir_exists(example_data_dir):
    """Test that example data directory exists and has files."""
    assert example_data_dir.exists()
    assert example_data_dir.is_dir()

    # Should contain BoneCollagen.csv
    assert (example_data_dir / "BoneCollagen.csv").exists()


@pytest.mark.smoke
def test_gold_standard_dir_created(gold_standard_dir):
    """Test that gold standards directory is created."""
    assert gold_standard_dir.exists()
    assert gold_standard_dir.is_dir()


@pytest.mark.smoke
def test_temp_output_dir(temp_output_dir):
    """Test that temporary output directory works."""
    assert temp_output_dir.exists()
    assert temp_output_dir.is_dir()

    # Test writing to it
    test_file = temp_output_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()


@pytest.mark.smoke
def test_deterministic_fixtures(synthetic_spectra_small):
    """Test that fixtures are deterministic across runs."""
    # Get data twice
    X1, y1 = synthetic_spectra_small

    # Import fixture function directly to generate again
    from tests.fixtures.synthetic_data import generate_spectral_data

    n_wavelengths = 200
    wavelengths = np.linspace(1000, 2500, n_wavelengths)
    wavelength_names = [str(wl) for wl in wavelengths]

    X2, y2 = generate_spectral_data(
        n_samples=50,
        n_wavelengths=n_wavelengths,
        n_informative=5,
        noise_level=0.1,
        seed=42,
    )
    X2.columns = wavelength_names

    # Should be identical
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


@pytest.mark.numerical
def test_spectral_data_has_realistic_values(synthetic_spectra_small):
    """Test that synthetic spectra have realistic absorbance values."""
    X, y = synthetic_spectra_small

    # Absorbance values typically 0-2 for NIR
    assert X.min().min() >= 0, "Absorbance should be non-negative"
    assert X.max().max() < 5, "Absorbance should be realistic (<5)"

    # Check that spectra are somewhat smooth (not random noise)
    for i in range(len(X)):
        spectrum = X.iloc[i].values
        # Calculate smoothness as std of first derivative
        first_deriv = np.diff(spectrum)
        smoothness = np.std(first_deriv)
        assert smoothness < 0.1, "Spectra should be relatively smooth"
