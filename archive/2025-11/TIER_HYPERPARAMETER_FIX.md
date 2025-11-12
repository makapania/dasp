# Tier Hyperparameter Fix

## Problem

When running a Quick tier regression analysis, ALL PLS results showed 5 latent variables (LVs), regardless of the optimal number. The issue was that the 'quick' tier was testing only [5, 10, 15] components instead of the full range [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50].

## Root Cause

The tier system was incorrectly designed to modify hyperparameter search grids. The original intent was:
- **Quick tier**: Test FEWER models and preprocessing methods, but use SAME hyperparameter grids
- **Standard tier**: Test MORE models and preprocessing methods, but use SAME hyperparameter grids
- **Comprehensive tier**: Test ALL models and preprocessing methods, with SAME hyperparameter grids

However, the implementation had different hyperparameter grids for each tier, causing:
1. Quick tier to find suboptimal hyperparameters (e.g., always selecting 5 LVs for PLS)
2. Inconsistent results across tiers for the same model type
3. Confusion about what "tier" actually controls

## Solution

**All tiers now use identical hyperparameter grids** based on the original 'comprehensive' tier defaults. Tiers now only control:
- Which models to include (e.g., quick might skip experimental models)
- Which preprocessing methods to test (e.g., quick might skip some derivative/SNV combinations)
- Number of variable subsets to evaluate

## Changes Made

Updated `src/spectral_predict/model_config.py` to make all three tiers (quick, standard, comprehensive) use the same hyperparameter grids:

### PLS
- **All tiers**: [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50] components (12 configs)
- **Previously**: Quick tier only tested [5, 10, 15] (3 configs)

### Ridge
- **All tiers**: [0.001, 0.01, 0.1, 1.0, 10.0] alpha values (5 configs)
- **Previously**: Quick tier only tested [0.1, 1.0] (2 configs)

### ElasticNet
- **All tiers**:
  - alpha: [0.001, 0.01, 0.1, 1.0]
  - l1_ratio: [0.1, 0.3, 0.5, 0.7, 0.9]
  - Grid size: 4×5 = 20 configs
- **Previously**: Quick tier only tested 2 alphas × 1 l1_ratio = 2 configs

### XGBoost
- **All tiers**:
  - n_estimators: [100, 200]
  - learning_rate: [0.05, 0.1]
  - max_depth: [3, 6, 9]
  - subsample: [0.7, 0.85, 1.0]
  - colsample_bytree: [0.7, 0.85, 1.0]
  - reg_alpha: [0, 0.1, 0.5]
  - reg_lambda: [1.0, 5.0]
  - Grid size: 2×2×3×3×3×3×2 = 648 configs
- **Previously**: Quick tier only tested 1 config with fixed parameters

### LightGBM
- **All tiers**:
  - n_estimators: [50, 100, 200]
  - learning_rate: [0.05, 0.1, 0.2]
  - num_leaves: [31, 50, 70]
  - Grid size: 3×3×3 = 27 configs
- **Previously**: Quick tier only tested 1 config

### SVR
- **All tiers**:
  - kernel: ['rbf', 'linear']
  - C: [0.1, 1.0, 10.0]
  - gamma: ['scale', 'auto']
  - Grid size: 9 configs
- **Previously**: Quick tier only tested 1 config

### NeuralBoosted
- **All tiers**:
  - n_estimators: [100, 150]
  - learning_rate: [0.1, 0.2, 0.3]
  - hidden_layer_size: [3, 5]
  - activation: ['tanh', 'identity']
  - Grid size: 2×3×2×2 = 24 configs
- **Previously**: Quick tier only tested 1 config

### CatBoost
- **All tiers**:
  - iterations: [50, 100, 200]
  - learning_rate: [0.05, 0.1, 0.2]
  - depth: [4, 6, 8]
  - Grid size: 3×3×3 = 27 configs
- **Previously**: Quick tier only tested 1 config

### RandomForest
- **All tiers**:
  - n_estimators: [100, 200, 500]
  - max_depth: [None, 15, 30]
  - Grid size: 3×3 = 9 configs
- **Previously**: Quick tier only tested 1 config

### MLP
- **All tiers**:
  - hidden_layer_sizes: [(64,), (128, 64)]
  - alpha: [1e-4, 1e-3]
  - learning_rate_init: [1e-3, 1e-2]
  - Grid size: 2×2×2 = 8 configs
- **Previously**: Quick tier only tested 1 config

### Lasso
- **All tiers**: [0.001, 0.01, 0.1, 1.0] alpha values (4 configs)
- **Previously**: Quick tier only tested [0.1] (1 config)

## Impact

After this fix:
- **Quick tier** will now test the full hyperparameter range for each model but may skip some models or preprocessing combinations
- **PLS analyses** will now properly optimize the number of latent variables instead of always returning 5
- **All models** will have consistent hyperparameter search across all tiers
- **Results** will be more reliable and comparable across tier levels

## Files Modified

- `src/spectral_predict/model_config.py` - Updated all model hyperparameter grids

## Testing Recommendation

Re-run your Quick tier regression analysis on the examples folder. You should now see:
- PLS results with varying numbers of LVs (not just 5)
- Optimal PLS configurations might have 8, 10, 12, or more components depending on the data
- All other models also finding optimal hyperparameters across their full search space
