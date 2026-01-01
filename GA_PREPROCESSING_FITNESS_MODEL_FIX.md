# GA Preprocessing Fitness Model Fix - Summary

**Date**: 2025-12-20
**Status**: ✅ COMPLETE

## Problem

The `optimize_preprocessing()` function in `src/spectral_predict/ga_preprocessing.py` only supported PLS for fitness evaluation, but `search.py` was calling it with a `fitness_model` parameter that didn't exist, causing:

```
TypeError: optimize_preprocessing() got an unexpected keyword argument 'fitness_model'
```

## Solution

Added full support for multiple fitness models in GA preprocessing optimization:

### 1. LightGBM Import (Lines 31-38)
```python
# Import LightGBM for fitness evaluation
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LIGHTGBM_PREPROC = True
except ImportError:
    HAS_LIGHTGBM_PREPROC = False
    LGBMRegressor = None
    LGBMClassifier = None
```

### 2. Updated `evaluate_fitness()` Function
- Added `fitness_model: str = 'pls'` parameter (line 225)
- Implemented LightGBM fitness evaluation (lines 287-324)
- Falls back to PLS if LightGBM requested but not available
- Supports both regression (RMSECV) and classification (accuracy)

### 3. Updated `optimize_preprocessing()` Function
- Added `fitness_model: str = 'pls'` parameter (line 432)
- Updated docstring to document new parameter
- Passes `fitness_model` to all `evaluate_fitness()` calls
- Displays fitness model in verbose output with fallback warning

### 4. Enhanced Module Documentation
- Updated module docstring to explain fitness model options
- Documents 'pls' (default, always available) and 'lightgbm' options
- Notes that LightGBM is better for tree-based models

## Files Modified

1. **src/spectral_predict/ga_preprocessing.py**
   - Added LightGBM import with fallback
   - Updated `evaluate_fitness()` to support multiple fitness models
   - Updated `optimize_preprocessing()` signature and implementation
   - Enhanced documentation

2. **tests/test_ga_preprocessing.py**
   - Removed hardcoded skip condition on LightGBM test
   - Test now runs when LightGBM is available

## Testing

### Unit Tests
All 27 tests in `tests/test_ga_preprocessing.py` pass:
- ✅ Chromosome encoding tests (5 tests)
- ✅ Fitness evaluation tests (5 tests, including new LightGBM test)
- ✅ Genetic operators tests (6 tests)
- ✅ GA optimization tests (8 tests)
- ✅ Convenience function tests (2 tests)
- ✅ Edge cases (2 tests)

### Integration Testing
Created and ran integration test demonstrating:
- ✅ PLS fitness model works correctly
- ✅ LightGBM fitness model works correctly
- ✅ Different fitness models can find different optimal preprocessing
- ✅ Fallback behavior when LightGBM unavailable

## Interface

### Function Signature
```python
def optimize_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 32,
    n_generations: int = 50,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    cv_folds: int = 5,
    n_components: int = 10,
    elitism: int = 2,
    task_type: str = 'regression',
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    fitness_model: str = 'pls'  # NEW PARAMETER
) -> Dict[str, Any]:
```

### Usage Examples

**Using PLS fitness (default):**
```python
result = optimize_preprocessing(X, y, fitness_model='pls')
```

**Using LightGBM fitness:**
```python
result = optimize_preprocessing(X, y, fitness_model='lightgbm')
```

**Automatic fallback:**
```python
# If LightGBM not available, automatically falls back to PLS
result = optimize_preprocessing(X, y, fitness_model='lightgbm')
```

## Backward Compatibility

✅ **MAINTAINED** - The `fitness_model` parameter defaults to `'pls'`, so all existing code continues to work without modification.

## Performance Notes

From integration testing:
- PLS fitness: Fast, suitable for linear models
- LightGBM fitness: Slightly slower, but better for optimizing preprocessing for tree-based models
- Both find valid but potentially different optimal preprocessing configurations

## Integration with search.py

The implementation now correctly supports `search.py` calls:

```python
# For linear models (line 582)
ga_result_linear = optimize_preprocessing(
    X.values, y.values,
    fitness_model='pls',
    ...
)

# For tree-based models (line 607)
ga_result_tree = optimize_preprocessing(
    X.values, y.values,
    fitness_model='lightgbm',  # Falls back to PLS if unavailable
    ...
)
```

## Notes

- The LightGBM fitness evaluation uses simple hyperparameters (n_estimators=100, max_depth=5, learning_rate=0.1)
- LightGBM configured with `verbosity=-1` and `force_col_wise=True` for quiet operation
- Both regression and classification tasks are fully supported
- The `n_components` parameter only affects PLS fitness evaluation
