# R Validation Quick Start Guide

## TL;DR

Validate that Python models match R package quality in 4 steps:

```bash
# Step 1: Generate sample data
python r_validation_scripts/generate_sample_data.py

# Step 2: Run Python tests
pytest tests/test_r_validation.py -v -s

# Step 3: Run R scripts
Rscript r_validation_scripts/pls_comparison.R
Rscript r_validation_scripts/random_forest_comparison.R
Rscript r_validation_scripts/xgboost_comparison.R
Rscript r_validation_scripts/glmnet_comparison.R

# Step 4: Compare results
python r_validation_scripts/compare_results.py --model all
```

## What Gets Validated

| Model | Python Package | R Package | Expected Match |
|-------|---------------|-----------|----------------|
| PLS | sklearn.cross_decomposition | pls | Exact (< 1e-10) |
| Random Forest | sklearn.ensemble | randomForest | High correlation (> 0.95) |
| XGBoost | xgboost | xgboost | Very close (< 1e-4) |
| Ridge | sklearn.linear_model | glmnet | Very close (< 1e-6) |
| Lasso | sklearn.linear_model | glmnet | Very close (< 1e-6) |
| ElasticNet | sklearn.linear_model | glmnet | Very close (< 1e-6) |

## Passing Criteria

✓ **PASS:** Predictions match within tolerance
⚠ **WARNING:** Predictions differ slightly but acceptably
✗ **FAIL:** Predictions differ significantly - investigate!

### Deterministic Models (PLS, Ridge, Lasso, ElasticNet, XGBoost)
- Predictions: < 1e-6 difference
- RMSE/R²: < 1e-4 difference
- Coefficients: < 1e-6 difference

### Stochastic Models (Random Forest)
- Predictions correlation: > 0.95
- RMSE difference: < 10%
- Feature importance correlation: > 0.9

## Install R Packages

```R
install.packages(c("pls", "randomForest", "xgboost", "glmnet", "jsonlite"))
```

## Output Location

Results are saved to:
- `r_validation_scripts/results/python/` - Python model results (JSON)
- `r_validation_scripts/results/r/` - R model results (JSON)
- `data/` - Sample spectral datasets

## What to Check

When comparing results, look for:

1. **Predictions:** Should match very closely (see table above)
2. **RMSE/R²:** Should be identical or very close
3. **Feature importances:** Should correlate highly (> 0.9)
4. **Top features:** Should have good overlap (> 70%)

## Common Issues

### Random Forest shows differences
✓ **Normal** - Different RNG implementations. Look for correlation > 0.95.

### PLS loadings have opposite signs
✓ **Normal** - Sign is arbitrary, magnitude is what matters.

### glmnet parameters confusing
✓ **Read this:**
```
sklearn Ridge(alpha=X)     = glmnet(alpha=0, lambda=X)
sklearn Lasso(alpha=X)     = glmnet(alpha=1, lambda=X)
sklearn ElasticNet(        = glmnet(
  alpha=X, l1_ratio=Y)        alpha=Y, lambda=X)
```

## Full Documentation

See `README_R_VALIDATION.md` for:
- Detailed installation instructions
- Troubleshooting guide
- Advanced usage
- FAQ
- References
