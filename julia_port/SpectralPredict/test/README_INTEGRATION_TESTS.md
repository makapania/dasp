# Integration Test Suite - SpectralPredict.jl

## Overview

The integration test suite (`test_integration.jl`) provides comprehensive end-to-end testing of all major SpectralPredict modules working together. It validates full workflows from data generation through model training and evaluation.

## Test Coverage

### 1. Variable Selection Methods Integration
- **UVE (Uninformative Variable Elimination)**: Tests noise-based filtering
- **SPA (Successive Projections Algorithm)**: Tests collinearity reduction
- **iPLS (Interval PLS)**: Tests spectral region-based selection
- **UVE-SPA Hybrid**: Tests combined approach

**What's tested:**
- Variable selection reduces dimensionality correctly
- Selected variables include informative features
- Results DataFrame has proper structure
- Methods execute without errors
- Performance is reasonable (< 30s per method)

### 2. MSC Preprocessing Integration
- Tests Multiplicative Scatter Correction application
- Validates MSC + search pipeline
- Ensures scatter effects are corrected

**What's tested:**
- MSC transforms data correctly
- No NaN or Inf values produced
- Integration with full search pipeline
- Best model performance is reasonable

### 3. NeuralBoosted Model Integration
- Tests direct usage of NeuralBoosted regressor
- Validates integration with search pipeline
- Ensures gradient boosting with MLP weak learners works

**What's tested:**
- Model fits and predicts correctly
- Convergence occurs (early stopping works)
- Integration with preprocessing
- Performance metrics are valid

### 4. Combined Features Integration
- Tests multiple features working together:
  - MSC + Variable Selection
  - Variable Selection + Derivatives + NeuralBoosted
  - All preprocessing + all models

**What's tested:**
- Complex pipelines execute without errors
- Results include diverse model types
- Best models are correctly identified
- Ranking system works properly

### 5. DataFrame Structure Validation
- Validates results DataFrame structure
- Checks column presence and data types
- Verifies ranking correctness
- Ensures subset diversity

**Required columns:**
- `Model`, `Preprocess`, `Deriv`, `Window`, `Poly`
- `SubsetTag`, `n_vars`, `full_vars`
- `RMSE`, `R2`, `CompositeScore`, `Rank`

### 6. Error Handling and Edge Cases
- Small datasets
- Limited features with variable selection
- Variable counts exceeding available features
- All methods executing together

**What's tested:**
- No crashes or exceptions
- Graceful handling of edge cases
- Proper validation of inputs
- Informative error messages

### 7. Performance Benchmarks
- **Minimal search**: < 10 seconds
- **Standard search**: < 30 seconds
- **NeuralBoosted**: 30-60 seconds (use small datasets)

## Running the Tests

### Run Full Integration Suite

```bash
cd julia_port/SpectralPredict
julia --project=. test/test_integration.jl
```

### Run with Test Framework

```julia
using Test
include("test/test_integration.jl")
```

### Run Specific Test Groups

Edit `test_integration.jl` to comment out test groups you don't want to run, or use Julia's test filtering:

```julia
using Test
@testset "Specific Tests" begin
    include("test/test_integration.jl")
end
```

## Expected Runtime

| Test Group | Configurations | Expected Time |
|------------|---------------|---------------|
| Variable Selection (UVE) | ~12 | 8-12s |
| Variable Selection (SPA) | ~6 | 5-8s |
| Variable Selection (iPLS) | ~6 | 5-8s |
| Variable Selection (UVE-SPA) | ~6 | 6-10s |
| MSC Preprocessing | ~4 | 3-5s |
| NeuralBoosted Direct | 1 | 5-10s |
| NeuralBoosted Search | ~2 | 15-25s |
| Combined Features (MSC + UVE) | ~18 | 10-15s |
| Combined (All features) | ~12 | 20-30s |
| DataFrame Validation | ~50 | 15-20s |
| Error Handling | ~20 | 8-12s |
| Performance Benchmarks | ~15 | 10-15s |
| **TOTAL** | **~150** | **2-3 minutes** |

**Note**: Times may vary based on hardware. Tests use small datasets (50-150 samples, 60-150 features) for speed.

## Synthetic Data Generation

The integration tests use realistic synthetic NIR spectral data with:

- **Wavelength range**: 1100-2500 nm (typical NIR)
- **Samples**: 50-150 (adjustable)
- **Wavelengths**: 60-150 (adjustable)
- **Structure**:
  - Smooth baseline with multiplicative scatter
  - Gaussian absorption bands at informative wavelengths
  - Correlated noise structure
  - Known linear relationship to reference values

**Why synthetic data?**
- Reproducible results (fixed random seed)
- Known ground truth (informative wavelengths)
- Controllable complexity
- Fast generation (< 1s)
- No external dependencies

## Test Output

Each test group prints:
- Progress messages
- Timing information
- Configuration counts
- Best model performance
- Validation checkmarks (✓)

Example output:
```
================================================================================
TEST 1: Variable Selection Methods
--------------------------------------------------------------------------------
Generated NIR data: 96 train samples, 24 test samples
Wavelengths: 150 (1100.0-2500.0 nm)
Known informative wavelengths at indices: [25, 50, 75, 100, 125]

  Testing UVE selection...
    ✓ UVE selection completed in 8.32s
    ✓ Created 12 UVE subset configurations
    ✓ UVE reduced to: [10, 25, 50] variables
    ✓ Best model: PLS uve25 (RMSE=2.456, R²=0.823)
```

## Validation Criteria

### Performance Metrics
- **RMSE**: > 0 (positive)
- **R²**: ≤ 1.0 (valid range)
- **CompositeScore**: Finite values
- **Rank**: Sequential from 1

### Dimensionality Reduction
- Variable selection: `n_vars < n_features`
- Selected variables: Include informative features
- Reduction levels: Match requested counts

### Error Handling
- No crashes or exceptions
- Graceful degradation for edge cases
- Proper validation messages
- Informative warnings

## Limitations and Known Issues

### Current Limitations
1. **NeuralBoosted is slow**: Use small datasets (n < 100) for tests
2. **No GPU acceleration**: Tests run on CPU only
3. **Sequential execution**: Tests run one at a time (no parallelization)
4. **Fixed random seed**: Results are deterministic but may vary across Julia versions

### What's NOT Tested
- Real spectral data formats (SPC, CSV parsing)
- Multi-class classification (only regression and binary)
- Extremely large datasets (n > 1000, p > 1000)
- Parallel cross-validation
- Model serialization/deserialization
- GUI integration

### Future Improvements
1. Add classification integration tests
2. Test with real spectral datasets
3. Add parallel execution benchmarks
4. Test model persistence
5. Add memory usage tracking
6. Test with different random seeds

## Troubleshooting

### Tests Fail with "Out of Memory"
- Reduce `n_samples` and `n_wavelengths` in `generate_nir_data()`
- Use fewer folds (`n_folds=3` instead of 5)
- Reduce `variable_counts` arrays

### Tests Take Too Long (> 5 minutes)
- Skip NeuralBoosted tests (comment out Test 3)
- Reduce dataset sizes
- Use fewer models (`models=["Ridge"]` only)
- Disable region subsets (`enable_region_subsets=false`)

### Variable Selection Tests Fail
- Check that `variable_selection.jl` is properly loaded
- Verify UVE/SPA/iPLS functions are exported
- Ensure input data has sufficient samples (n > 20)

### NeuralBoosted Tests Fail
- Reduce `n_estimators` (e.g., 10-20)
- Increase `learning_rate` (e.g., 0.2-0.3)
- Reduce dataset size (n < 80, p < 80)
- Check Flux.jl installation

### Results DataFrame Issues
- Verify all columns are present in `run_search()` output
- Check that ranking function is called
- Ensure composite scoring is computed

## Integration Test Philosophy

### What Makes a Good Integration Test?

1. **Tests Real Workflows**: Not individual functions, but complete user journeys
2. **Uses Realistic Data**: Synthetic but representative of actual use cases
3. **Fast Enough**: Completes in minutes, not hours
4. **Comprehensive Coverage**: Tests all major features working together
5. **Clear Validation**: Easy to understand what passed/failed
6. **Reproducible**: Same results every run (fixed random seed)
7. **Well-Documented**: Clear explanations of what's being tested

### This Test Suite Follows These Principles

✓ Tests complete workflows (data → preprocessing → model → results)
✓ Uses NIR-like synthetic data (realistic structure)
✓ Completes in 2-3 minutes (reasonable for CI)
✓ Covers all major modules (variable selection, preprocessing, models, search)
✓ Clear pass/fail criteria (performance metrics, structure validation)
✓ Fixed random seed (Random.seed!(42))
✓ Extensive documentation (this README!)

## Contributing

When adding new integration tests:

1. **Create realistic test data**: Use `generate_nir_data()` or similar
2. **Keep tests fast**: Target < 30s per test group
3. **Validate thoroughly**: Check results structure and metrics
4. **Document expected behavior**: Print progress and results
5. **Handle edge cases**: Test with small/large/edge datasets
6. **Add timing information**: Use `@elapsed` for benchmarks

## References

- Main documentation: `../README.md`
- Module tests: `test_models.jl`, `test_cv.jl`, `test_search.jl`
- Variable selection: `../VARIABLE_SELECTION_INTEGRATION.md`
- NeuralBoosted: `../NEURAL_BOOSTED_INTEGRATION_SUMMARY.md`
- Search pipeline: `../SEARCH_IMPLEMENTATION_SUMMARY.md`

## Contact

For questions or issues with integration tests, contact the Spectral Predict team.

---

**Last Updated**: November 2025
**Test Suite Version**: 1.0
**Julia Compatibility**: 1.9+
