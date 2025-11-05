# Test Suite Documentation

## Overview

This directory contains comprehensive unit tests for all newly implemented Julia modules in SpectralPredict:

1. **test_variable_selection.jl** - Tests for UVE, SPA, iPLS, and UVE-SPA variable selection methods
2. **test_diagnostics.jl** - Tests for diagnostic functions (residuals, leverage, Q-Q plots, jackknife)
3. **test_neural_boosted.jl** - Tests for Neural Boosted Regressor (gradient boosting with MLP weak learners)
4. **test_msc.jl** - Tests for Multiplicative Scatter Correction (MSC) preprocessing

## Test Coverage Summary

### Variable Selection Tests (test_variable_selection.jl)
- **Total Tests**: ~80 tests across 5 main test sets
- **Coverage**:
  - UVE Selection: Basic functionality, reproducibility, parameter variations, edge cases, numerical stability
  - SPA Selection: Basic functionality, collinearity reduction, reproducibility, edge cases, selection order
  - iPLS Selection: Basic functionality, interval identification, parameter variations, edge cases
  - UVE-SPA Hybrid: Basic functionality, two-stage process, reproducibility, edge cases
  - Integration tests comparing all methods

### Diagnostics Tests (test_diagnostics.jl)
- **Total Tests**: ~70 tests across 4 main test sets
- **Coverage**:
  - compute_residuals: Basic functionality, perfect predictions, outlier detection, edge cases
  - compute_leverage: Basic functionality, high leverage detection, SVD vs direct methods
  - qq_plot_data: Normal/heavy-tailed/light-tailed distributions, small samples, numerical properties
  - jackknife_prediction_intervals: Basic functionality, confidence levels, reproducibility, edge cases
  - Integration tests for full diagnostic pipeline

### Neural Boosted Tests (test_neural_boosted.jl)
- **Total Tests**: ~90 tests across 8 main test sets
- **Coverage**:
  - Model construction and parameter validation
  - Fitting with early stopping and without
  - Prediction on training and test data
  - Feature importances computation
  - Different activation functions (tanh, relu, sigmoid, identity)
  - Different loss functions (MSE, Huber)
  - Learning rate effects and boosting behavior
  - Edge cases (small datasets, wide data, constant targets)
  - Reproducibility with random seeds
  - Numerical stability

### MSC Tests (test_msc.jl)
- **Total Tests**: ~75 tests across 7 main test sets
- **Coverage**:
  - Basic MSC functionality (mean, median, custom reference)
  - Scatter correction properties (removes additive/multiplicative effects)
  - fit_msc function and train/test consistency
  - Edge cases (constant spectra, single sample, large datasets, collinear features)
  - Numerical stability (small/large values, near-singular cases)
  - Parameter validation
  - Comparison with SNV preprocessing
  - Integration tests for full preprocessing workflow

## Running the Tests

### Option 1: Run All Tests

```bash
cd julia_port/SpectralPredict
julia --project=. test/runtests.jl
```

This will run all test suites and provide a comprehensive summary.

### Option 2: Run Individual Test Files

```bash
cd julia_port/SpectralPredict

# Test variable selection methods
julia --project=. test/test_variable_selection.jl

# Test diagnostic functions
julia --project=. test/test_diagnostics.jl

# Test neural boosted regressor
julia --project=. test/test_neural_boosted.jl

# Test MSC preprocessing
julia --project=. test/test_msc.jl
```

### Option 3: Run from Julia REPL

```julia
# Start Julia with project
cd("julia_port/SpectralPredict")
using Pkg
Pkg.activate(".")

# Run tests interactively
include("test/test_variable_selection.jl")
include("test/test_diagnostics.jl")
include("test/test_neural_boosted.jl")
include("test/test_msc.jl")
```

### Option 4: Run with Julia Package Manager

```julia
using Pkg
Pkg.activate("julia_port/SpectralPredict")
Pkg.test()
```

## Test Structure

Each test file follows the same structure:

1. **Test Data Generators**: Functions to create synthetic spectral data with known properties
2. **Basic Functionality Tests**: Verify core functionality works as expected
3. **Edge Case Tests**: Test boundary conditions (empty data, single sample, large datasets, etc.)
4. **Numerical Correctness Tests**: Verify outputs have correct dimensions, ranges, and properties
5. **Reproducibility Tests**: Ensure same seed produces same results
6. **Error Handling Tests**: Verify appropriate errors are raised for invalid inputs
7. **Integration Tests**: Test interactions between functions and realistic workflows

## Test Principles

All tests follow these principles:

- **Independence**: Each test is independent and can run in any order
- **Descriptive Names**: Test names clearly explain what is being tested
- **Fast Execution**: Most tests run in < 1 second; full suite completes in < 2 minutes
- **Realistic Data**: Test data generators create realistic spectral data patterns
- **Comprehensive Coverage**: Tests cover normal cases, edge cases, and error conditions
- **Documentation**: Comments explain complex test logic

## Expected Test Output

Successful test run output will look like:

```
======================================================================
SpectralPredict.jl - Comprehensive Test Suite
======================================================================

Running 4 test suites...

======================================================================
Testing: Variable Selection
======================================================================
Test Summary:                            | Pass  Total
Variable Selection Tests                 |   80     80
✓ Variable Selection: PASSED

======================================================================
Testing: Diagnostics
======================================================================
Test Summary:                            | Pass  Total
Diagnostics Tests                        |   70     70
✓ Diagnostics: PASSED

======================================================================
Testing: Neural Boosted
======================================================================
Test Summary:                            | Pass  Total
Neural Boosted Tests                     |   90     90
✓ Neural Boosted: PASSED

======================================================================
Testing: MSC Preprocessing
======================================================================
Test Summary:                            | Pass  Total
MSC Tests                                |   75     75
✓ MSC Preprocessing: PASSED

======================================================================
Test Summary
======================================================================
  ✓ PASSED: Variable Selection
  ✓ PASSED: Diagnostics
  ✓ PASSED: Neural Boosted
  ✓ PASSED: MSC Preprocessing

Total: 4/4 test suites passed
======================================================================

🎉 All tests passed!
```

## Troubleshooting

### Issue: Module not found error

**Solution**: Make sure you're running from the SpectralPredict directory with the project activated:

```bash
cd julia_port/SpectralPredict
julia --project=.
```

### Issue: Package dependencies missing

**Solution**: Install dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Issue: Tests are slow

**Cause**: Some tests involve cross-validation or iterative optimization.

**Solution**: Tests are designed to be fast. If tests are slow:
- Check if you're running in debug mode
- Ensure Julia is compiled (first run may be slower due to JIT compilation)
- Consider running tests in parallel: `julia -t auto --project=. test/runtests.jl`

### Issue: Flux.jl or neural network tests fail

**Cause**: Neural network training can be sensitive to initialization.

**Solution**: Tests use fixed random seeds for reproducibility. If tests still fail:
- Check Flux.jl version is compatible (Project.toml)
- Verify GPU/CPU settings if applicable
- Review test output for specific failure messages

### Issue: Numerical precision errors

**Cause**: Floating-point arithmetic can vary slightly across systems.

**Solution**: Tests use appropriate tolerances (e.g., `≈` with `atol` parameter). If precision errors occur:
- Check if running on different hardware (ARM vs x86)
- Verify BLAS/LAPACK libraries are correctly installed
- Adjust tolerances if necessary for your system

## Performance Benchmarks

Expected test execution times on a typical system (Intel i7, 16GB RAM):

- **test_variable_selection.jl**: ~20-30 seconds
- **test_diagnostics.jl**: ~15-20 seconds
- **test_neural_boosted.jl**: ~30-40 seconds (includes neural network training)
- **test_msc.jl**: ~10-15 seconds

**Total Suite**: ~1.5-2 minutes

## Contributing New Tests

When adding new tests, follow these guidelines:

1. **Use @testset**: Group related tests in named test sets
2. **Test Normal and Edge Cases**: Cover both typical usage and boundary conditions
3. **Check Numerical Properties**: Verify no NaN/Inf, correct dimensions, reasonable ranges
4. **Document Complex Tests**: Add comments explaining non-obvious test logic
5. **Keep Tests Fast**: Aim for < 1 second per test when possible
6. **Use Realistic Data**: Generate synthetic data that mimics real spectral data
7. **Test Error Handling**: Verify functions raise appropriate errors for invalid inputs

Example test structure:

```julia
@testset "New Feature Tests" begin
    @testset "Basic Functionality" begin
        # Test normal usage
        @test function_works_correctly()
    end

    @testset "Edge Cases" begin
        # Test boundary conditions
        @test handles_empty_input()
        @test handles_single_sample()
    end

    @testset "Error Handling" begin
        # Test invalid inputs
        @test_throws ArgumentError invalid_input()
    end
end
```

## Test Data Generators

Each test file includes helper functions to generate realistic test data:

### Variable Selection
- `generate_spectral_data()`: Synthetic spectral data with known informative variables
- `generate_collinear_data()`: Data with groups of correlated variables
- `generate_interval_data()`: Data with distinct informative spectral intervals

### Diagnostics
- `generate_regression_data()`: Standard regression data with optional outliers
- `generate_perfect_predictions()`: Data with zero prediction error
- `generate_predictions_with_outliers()`: Predictions containing outliers

### Neural Boosted
- `generate_boosting_data()`: Non-linear regression data suitable for boosting
- `generate_data_with_outliers()`: Data with known outliers for robust loss testing
- `generate_linear_data()`: Simple linear regression data

### MSC
- `generate_spectral_data()`: Realistic NIR-like spectra
- `generate_scattered_data()`: Clean and scattered versions of same spectra
- `generate_constant_spectra()`: Edge case with identical spectra

## CI/CD Integration

To integrate these tests into a CI/CD pipeline:

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@v1
        with:
          version: '1.9'
      - uses: julia-actions/julia-buildpkg@v1
      - name: Run tests
        run: |
          cd julia_port/SpectralPredict
          julia --project=. test/runtests.jl
```

## Test Maintenance

- **Review Regularly**: Update tests when module functionality changes
- **Monitor Performance**: Track test execution time to catch performance regressions
- **Update Data Generators**: Improve test data to better reflect real-world scenarios
- **Expand Coverage**: Add tests for newly discovered edge cases or bug fixes
- **Refactor When Needed**: Keep test code clean and maintainable

## Additional Resources

- [Julia Test Documentation](https://docs.julialang.org/en/v1/stdlib/Test/)
- [Testing Best Practices](https://docs.julialang.org/en/v1/manual/workflow-tips/#Testing-and-debugging)
- SpectralPredict Module Documentation: See individual module source files for detailed API documentation

## Contact

For questions about the test suite or to report issues:
- Check existing tests for examples
- Review module documentation in source files
- Create an issue if tests fail unexpectedly
