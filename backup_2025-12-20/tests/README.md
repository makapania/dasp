# Spectral Predict Test Infrastructure

This directory contains the test suite for Spectral Predict, including comprehensive pytest fixtures and synthetic data generators.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run only smoke tests (fast)
pytest tests/ -m smoke -v

# Run tests with coverage
pytest tests/ --cov=src/spectral_predict --cov-report=html

# Run specific test file
pytest tests/test_fixtures_demo.py -v
```

## Directory Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── fixtures/
│   ├── __init__.py
│   └── synthetic_data.py    # Synthetic data generators
├── gold_standards/          # Reference outputs for regression testing
└── test_*.py                # Test modules
```

## Available Fixtures

### Path Fixtures

- **`project_root()`** - Returns project root Path
- **`example_data_dir()`** - Returns example/ directory path
- **`gold_standard_dir()`** - Returns tests/gold_standards/ path
- **`temp_output_dir()`** - Temporary directory for test outputs

### Synthetic Data Fixtures

#### Small Dataset (Quick Tests)
- **`synthetic_spectra_small()`** - 50 samples, 200 wavelengths (1000-2500nm)
  - Ideal for unit tests
  - Low noise, deterministic (seed=42)

#### Medium Dataset (Standard Tests)
- **`synthetic_spectra_medium()`** - 100 samples, 500 wavelengths (400-2400nm)
  - Covers visible + NIR range
  - Good for preprocessing chains

#### Large Dataset (Performance Tests)
- **`synthetic_spectra_large()`** - 200 samples, 2151 wavelengths (350-2500nm)
  - Matches typical NIR spectrometer
  - Use for scalability tests

#### Classification Data
- **`classification_data()`** - Balanced binary classification (100 samples, 2 classes)
- **`imbalanced_data()`** - Severely imbalanced (110 samples, 10:1 ratio)

#### Special Cases
- **`outlier_data()`** - 100 samples with 5 known outliers at indices [0, 1, 2, 3, 4]

### Model Fixtures

- **`trained_pls_model()`** - Pre-trained PLS model with 5 components

### Example Data Fixtures

- **`bone_collagen_csv()`** - Loads BoneCollagen.csv from example/ directory
- **`example_asd_files()`** - List of ASD files for I/O testing
- **`example_spc_files()`** - List of SPC files for I/O testing

## Custom Markers

Tests can be marked with custom categories:

```python
@pytest.mark.smoke      # Quick smoke tests
@pytest.mark.unit       # Unit tests
@pytest.mark.integration  # Integration tests
@pytest.mark.slow       # Tests taking >10 seconds
@pytest.mark.numerical  # Numerical accuracy tests
@pytest.mark.regression # Regression tests
@pytest.mark.io         # File I/O tests
@pytest.mark.gui        # GUI component tests
```

Run tests by marker:
```bash
pytest tests/ -m smoke      # Only smoke tests
pytest tests/ -m "not slow" # Exclude slow tests
```

## Synthetic Data Generators

Direct access to data generators (in `tests/fixtures/synthetic_data.py`):

```python
from tests.fixtures.synthetic_data import (
    generate_spectral_data,
    generate_outlier_data,
    generate_imbalanced_data,
    generate_baseline_data,
    generate_classification_spectra,
)

# Generate custom data
X, y = generate_spectral_data(
    n_samples=100,
    n_wavelengths=300,
    n_informative=5,
    noise_level=0.1,
    seed=42
)
```

All generators:
- Return `(X_df, y_series)` tuples
- Use deterministic seeding for reproducibility
- Include realistic spectral patterns
- Have comprehensive docstrings

## Example Usage

### Using Fixtures in Tests

```python
def test_snv_preprocessing(synthetic_spectra_small):
    """Test SNV normalization."""
    X, y = synthetic_spectra_small

    from spectral_predict.preprocess import snv
    X_snv = snv(X)

    # SNV should preserve shape
    assert X_snv.shape == X.shape

    # Each spectrum should have mean ≈ 0 and std ≈ 1
    assert abs(X_snv.mean(axis=1).mean()) < 0.01
    assert abs(X_snv.std(axis=1).mean() - 1.0) < 0.01
```

### Using Multiple Fixtures

```python
def test_model_serialization(trained_pls_model, temp_output_dir):
    """Test saving and loading models."""
    model, wavelengths = trained_pls_model

    # Save model
    model_path = temp_output_dir / "pls_model.pkl"
    save_model(model, model_path)

    # Load model
    loaded_model = load_model(model_path)

    # Verify it works
    assert loaded_model is not None
```

### Direct Generator Usage

```python
def test_outlier_detection():
    """Test outlier detection with known outliers."""
    from tests.fixtures.synthetic_data import generate_outlier_data

    X, y, true_outliers = generate_outlier_data(
        n_samples=100,
        n_wavelengths=200,
        n_outliers=5,
        outlier_type="both",
        seed=42
    )

    # Run detection
    detected = detect_outliers(X, method="isolation_forest")

    # Check overlap with true outliers
    overlap = len(set(detected) & set(true_outliers))
    assert overlap >= 3  # Should detect at least 60%
```

## Testing Best Practices

1. **Use appropriate fixture size**
   - Unit tests: `synthetic_spectra_small`
   - Integration tests: `synthetic_spectra_medium`
   - Performance tests: `synthetic_spectra_large`

2. **Mark tests appropriately**
   ```python
   @pytest.mark.smoke
   def test_basic_functionality():
       ...

   @pytest.mark.slow
   @pytest.mark.integration
   def test_full_workflow():
       ...
   ```

3. **Test determinism**
   - All fixtures use seed=42
   - Tests should be reproducible
   - Use `reset_random_seed()` fixture (applied automatically)

4. **Use temp_output_dir for file operations**
   ```python
   def test_export(synthetic_spectra_small, temp_output_dir):
       X, y = synthetic_spectra_small
       output_path = temp_output_dir / "results.csv"
       X.to_csv(output_path)
       assert output_path.exists()
   ```

5. **Skip tests gracefully when data unavailable**
   ```python
   def test_asd_reading(example_asd_files):
       if not example_asd_files:
           pytest.skip("No ASD files available")
       # ... test code
   ```

## Continuous Integration

For CI pipelines:

```bash
# Quick smoke tests only
pytest tests/ -m smoke --maxfail=3

# Full test suite with coverage
pytest tests/ --cov=src/spectral_predict --cov-report=xml

# Exclude slow tests
pytest tests/ -m "not slow" --maxfail=5
```

## Adding New Fixtures

To add new fixtures:

1. Add to `tests/conftest.py` for shared fixtures
2. Add to `tests/fixtures/synthetic_data.py` for data generators
3. Document in this README
4. Add demonstration test in `test_fixtures_demo.py`

Example:

```python
@pytest.fixture
def my_new_fixture(synthetic_spectra_small):
    """Create specialized test data."""
    X, y = synthetic_spectra_small
    # Transform data
    return X_transformed, y_transformed
```

## Troubleshooting

**Import errors with fixtures:**
- Ensure you're running pytest from project root
- Check that `tests/fixtures/__init__.py` exists

**Fixtures not found:**
- Verify `conftest.py` is in `tests/` directory
- Check pytest discovery: `pytest --collect-only`

**Determinism issues:**
- All fixtures use seed=42
- `reset_random_seed()` runs before each test automatically
- Verify no global state modifications

**Memory issues with large fixtures:**
- Use session scope for expensive setup: `@pytest.fixture(scope="session")`
- Clean up large objects in fixture teardown
