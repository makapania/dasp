# Bayesian Hyperparameter Optimization for DASP

## Problem

Grid search tests EVERY combination of hyperparameters:
- XGBoost comprehensive: 5,832 configs
- Takes 11+ days to complete

## Solution

Use Bayesian optimization (Optuna) to find optimal parameters in 20-50 trials instead of 5,000+.

### How it works

1. **Sample intelligently**: Try promising regions of hyperparameter space
2. **Learn from failures**: Skip configurations similar to bad ones
3. **Converge fast**: Usually finds near-optimal params in 20-50 trials

### Expected Speedup

- Grid search: 5,832 trials
- Bayesian: 20-50 trials
- **Speedup: 100-300x**

### Implementation

```python
import optuna
from src.spectral_predict.models import build_model

def objective(trial, X, y, model_name):
    """Optuna objective function for one model type."""

    # Sample hyperparameters from space
    if model_name == "XGBoost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        }
    elif model_name == "LightGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        }
    # ... other models

    # Train with cross-validation
    model = build_model(model_name, params)
    cv_scores = cross_validate(model, X, y, cv=5, scoring='r2')

    return cv_scores.mean()

# Optimize each model type
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X, y, "XGBoost"), n_trials=30)

# Best params found
best_params = study.best_params
print(f"Best R²: {study.best_value:.4f}")
print(f"Best params: {best_params}")
```

### Benefits

1. **100x faster**: 30 trials instead of 5,832
2. **Better results**: Continuous parameter space (learning_rate=0.127 instead of just 0.05, 0.1, 0.2)
3. **Early stopping**: Optuna can prune bad trials early
4. **Visualization**: Optuna provides parameter importance plots

### Integration with DASP

Add new search mode:

```python
# In spectral_predict_gui_optimized.py
optimization_mode = self.optimization_mode.get()  # "grid" or "bayesian"

if optimization_mode == "bayesian":
    results = run_bayesian_search(X, y, task_type, n_trials=30)
else:
    results = run_search(X, y, task_type, tier='standard')  # Existing grid search
```

### Migration Path

1. **Phase 1**: Implement Bayesian search as option (2-3 days)
2. **Phase 2**: Test with real data, compare results (1 day)
3. **Phase 3**: Make Bayesian default, keep grid as fallback (1 day)

**Total time: 1 week**
**Speedup: 100x**
**No Julia migration needed**

## Recommendation

Try this BEFORE Julia migration. If Bayesian optimization gets you from 11 days → 2-3 hours, you don't need Julia at all.

Only migrate to Julia if you need to go from 2-3 hours → 15-20 minutes.
