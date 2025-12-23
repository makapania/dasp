# Pytest Fixtures Quick Reference

## Data Size Guide

| Fixture | Samples | Wavelengths | Range (nm) | Use Case |
|---------|---------|-------------|------------|----------|
| `synthetic_spectra_small` | 50 | 200 | 1000-2500 | Unit tests, quick validation |
| `synthetic_spectra_medium` | 100 | 500 | 400-2400 | Standard tests, preprocessing chains |
| `synthetic_spectra_large` | 200 | 2151 | 350-2500 | Performance tests, realistic data |

## Common Patterns

### Basic Fixture Usage

```python
def test_something(synthetic_spectra_small):
    X, y = synthetic_spectra_small
    # X is pd.DataFrame with wavelength column names
    # y is pd.Series with target values
    assert X.shape == (50, 200)
```

### Multiple Fixtures

```python
def test_with_model(trained_pls_model, synthetic_spectra_small):
    model, wavelengths = trained_pls_model
    X, y = synthetic_spectra_small
    predictions = model.predict(X)
```

### Temporary Files

```python
def test_save_file(synthetic_spectra_small, temp_output_dir):
    X, y = synthetic_spectra_small
    output = temp_output_dir / "data.csv"
    X.to_csv(output)
    assert output.exists()
```

### Example Data

```python
def test_real_data(bone_collagen_csv):
    data, path = bone_collagen_csv
    # data is full DataFrame
    # path is Path to the file
```

## Fixture Return Types

| Fixture | Returns | Example |
|---------|---------|---------|
| Spectral data | `(pd.DataFrame, pd.Series)` | `X, y = synthetic_spectra_small()` |
| Outlier data | `(pd.DataFrame, pd.Series, np.ndarray)` | `X, y, outliers = outlier_data()` |
| Model | `(model, list)` | `model, wavelengths = trained_pls_model()` |
| Paths | `Path` | `root = project_root()` |
| Example data | `(pd.DataFrame, Path)` | `data, path = bone_collagen_csv()` |

## Test Markers

```python
@pytest.mark.smoke        # Quick tests (~200ms each)
@pytest.mark.unit         # Unit tests for single functions
@pytest.mark.integration  # Multi-component tests
@pytest.mark.slow         # Tests >10 seconds
@pytest.mark.numerical    # Numerical accuracy tests
@pytest.mark.io           # File I/O tests
@pytest.mark.gui          # GUI component tests
```

Run by marker:
```bash
pytest -m smoke           # Only smoke tests
pytest -m "not slow"      # Exclude slow tests
pytest -m "smoke or unit" # Multiple markers
```

## All Available Fixtures

### Paths
- `project_root()` → Path to C:/Users/sponheim/git/dasp
- `example_data_dir()` → Path to example/
- `gold_standard_dir()` → Path to tests/gold_standards/
- `temp_output_dir()` → Temporary directory (auto-cleanup)

### Synthetic Spectral Data
- `synthetic_spectra_small()` → (X: 50×200, y: 50)
- `synthetic_spectra_medium()` → (X: 100×500, y: 100)
- `synthetic_spectra_large()` → (X: 200×2151, y: 200)

### Classification Data
- `classification_data()` → (X: 100×200, y: 100) - balanced
- `imbalanced_data()` → (X: 110×200, y: 110) - 10:1 ratio
- `outlier_data()` → (X: 100×200, y: 100, outliers: [0,1,2,3,4])

### Models
- `trained_pls_model()` → (model, wavelengths)

### Example Data
- `bone_collagen_csv()` → (DataFrame, Path)
- `example_asd_files()` → list[Path]
- `example_spc_files()` → list[Path]

### Utilities
- `reset_random_seed()` → Auto-applied to all tests

## Common Test Patterns

### Test Preprocessing

```python
@pytest.mark.unit
def test_snv(synthetic_spectra_small):
    X, y = synthetic_spectra_small
    from spectral_predict.preprocess import snv

    X_snv = snv(X)

    assert X_snv.shape == X.shape
    assert abs(X_snv.mean(axis=1).mean()) < 0.01
```

### Test Model Training

```python
@pytest.mark.unit
def test_pls_training(synthetic_spectra_medium):
    X, y = synthetic_spectra_medium
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=5)
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape[0] == len(y)
```

### Test Outlier Detection

```python
@pytest.mark.numerical
def test_outlier_detection(outlier_data):
    X, y, true_outliers = outlier_data

    detected = my_outlier_detector(X)

    # Check detection rate
    overlap = len(set(detected) & set(true_outliers))
    assert overlap >= 3  # 60% detection rate
```

### Test Variable Selection

```python
@pytest.mark.integration
def test_spa_selection(synthetic_spectra_medium):
    X, y = synthetic_spectra_medium
    from spectral_predict.variable_selection import run_spa

    selected = run_spa(X, y, n_vars=20)

    assert len(selected) == 20
    assert all(w in X.columns for w in selected)
```

### Test File I/O

```python
@pytest.mark.io
def test_read_asd(example_asd_files):
    if not example_asd_files:
        pytest.skip("No ASD files available")

    from spectral_predict.io import read_asd

    data = read_asd(example_asd_files[0])
    assert data is not None
```

### Test Classification

```python
@pytest.mark.unit
def test_classification(classification_data):
    X, y = classification_data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    accuracy = lda.score(X, y)
    assert accuracy > 0.7  # Should achieve 70%+ on training
```

### Test Imbalance Handling

```python
@pytest.mark.integration
def test_smote(imbalanced_data):
    X, y = imbalanced_data
    from spectral_predict.imbalance import apply_smote

    assert y.value_counts()[0] == 100
    assert y.value_counts()[1] == 10

    X_res, y_res = apply_smote(X, y)

    # Should be balanced now
    assert y_res.value_counts()[0] == y_res.value_counts()[1]
```

## Direct Generator Usage

When you need custom parameters:

```python
from tests.fixtures.synthetic_data import generate_spectral_data

def test_custom_data():
    # Create data with specific parameters
    X, y = generate_spectral_data(
        n_samples=200,
        n_wavelengths=1000,
        n_informative=10,
        noise_level=0.05,
        seed=123
    )

    # Your test code here
```

## Tips

1. **Start with small fixtures** - Use `synthetic_spectra_small` unless you need larger data
2. **Use markers** - Mark tests so you can run subsets during development
3. **Check determinism** - All fixtures use seed=42, tests should be reproducible
4. **Clean up** - Use `temp_output_dir` for file outputs
5. **Skip gracefully** - If optional data missing, use `pytest.skip()`
