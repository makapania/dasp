"""
DASP Regression Testing - Match R Implementation
=================================================

This script runs DASP regression models with parameters equivalent to R,
then exports predictions and metrics for direct comparison.

Datasets tested:
  1. Bone Collagen (n=49: 36 train, 13 test)
  2. Enamel d13C (n=140: 105 train, 35 test)

Models tested (matching R):
  - PLS Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - XGBoost

Usage: python dasp_regression.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent
R_DATA_DIR = BASE_DIR / "r_data"
RESULTS_DIR = BASE_DIR / "results" / "dasp_regression"

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate regression metrics matching R implementation."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'model': model_name,
        'rmse': float(rmse),
        'r2': float(r2),
        'mae': float(mae),
        'n_samples': len(y_true),
        'predictions': y_pred.flatten().tolist()
    }

def test_regression_models(dataset_name, data_dir, target_col, results_dir):
    """Test all regression models on a dataset."""

    print("=" * 80)
    print(f"TESTING DATASET: {dataset_name.upper()}")
    print("=" * 80)

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from: {data_dir}")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values

    y_train = pd.read_csv(data_dir / "y_train.csv")[target_col].values
    y_test = pd.read_csv(data_dir / "y_test.csv")[target_col].values

    print(f"  Train: {X_train.shape[0]} samples x {X_train.shape[1]} wavelengths")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Target range: {y_train.min():.2f} - {y_train.max():.2f}")

    all_results = {}

    # ========================================================================
    # Model 1: PLS Regression
    # ========================================================================

    print("\n" + "-" * 80)
    print("PLS Regression")
    print("-" * 80)

    pls_n_components = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50]
    max_comp = min(X_train.shape[0] - 1, X_train.shape[1])
    pls_n_components = [n for n in pls_n_components if n <= max_comp]

    print(f"\nTesting n_components: {pls_n_components}")

    pls_results = {}
    for n_comp in pls_n_components:
        print(f"  n_components={n_comp}... ", end="")

        model = PLSRegression(n_components=n_comp, max_iter=500, tol=1e-6, scale=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).flatten()

        metrics = calculate_metrics(y_test, y_pred, f"PLS_{n_comp}")
        pls_results[str(n_comp)] = metrics

        print(f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")

    all_results.update(pls_results)

    with open(results_dir / "pls_results.json", 'w') as f:
        json.dump(pls_results, f, indent=2)

    # ========================================================================
    # Model 2: Ridge Regression
    # ========================================================================

    print("\n" + "-" * 80)
    print("Ridge Regression")
    print("-" * 80)

    # Note: DASP uses 'alpha' for what R calls 'lambda'
    ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    print(f"\nTesting alpha values: {ridge_alphas}")

    ridge_results = {}
    for alpha in ridge_alphas:
        print(f"  alpha={alpha:.3f}... ", end="")

        model = Ridge(alpha=alpha, fit_intercept=True, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred, f"Ridge_{alpha:.3f}")
        ridge_results[str(alpha)] = metrics

        print(f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")

    all_results.update(ridge_results)

    with open(results_dir / "ridge_results.json", 'w') as f:
        json.dump(ridge_results, f, indent=2)

    # ========================================================================
    # Model 3: Lasso Regression
    # ========================================================================

    print("\n" + "-" * 80)
    print("Lasso Regression")
    print("-" * 80)

    lasso_alphas = [0.001, 0.01, 0.1, 1.0]
    print(f"\nTesting alpha values: {lasso_alphas}")

    lasso_results = {}
    for alpha in lasso_alphas:
        print(f"  alpha={alpha:.3f}... ", end="")

        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred, f"Lasso_{alpha:.3f}")
        lasso_results[str(alpha)] = metrics

        print(f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")

    all_results.update(lasso_results)

    with open(results_dir / "lasso_results.json", 'w') as f:
        json.dump(lasso_results, f, indent=2)

    # ========================================================================
    # Model 4: Random Forest
    # ========================================================================

    print("\n" + "-" * 80)
    print("Random Forest Regression")
    print("-" * 80)

    rf_n_estimators = [100, 200]
    rf_max_depths = [15, 30, None]

    rf_results = {}
    for n_est in rf_n_estimators:
        for max_depth in rf_max_depths:

            depth_label = "None" if max_depth is None else str(max_depth)
            config_name = f"RF_ntree{n_est}_depth{depth_label}"

            print(f"  n_estimators={n_est}, max_depth={depth_label}... ", end="")

            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = calculate_metrics(y_test, y_pred, config_name)
            metrics['n_estimators'] = n_est
            metrics['max_depth'] = max_depth
            rf_results[config_name] = metrics

            print(f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")

    all_results.update(rf_results)

    with open(results_dir / "rf_results.json", 'w') as f:
        json.dump(rf_results, f, indent=2)

    # ========================================================================
    # Model 5: XGBoost
    # ========================================================================

    print("\n" + "-" * 80)
    print("XGBoost Regression")
    print("-" * 80)

    xgb_n_estimators = [100, 200]
    xgb_learning_rates = [0.05, 0.1]
    xgb_max_depths = [3, 6]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_results = {}
    for n_est in xgb_n_estimators:
        for lr in xgb_learning_rates:
            for max_depth in xgb_max_depths:

                config_name = f"XGB_n{n_est}_lr{lr:.2f}_depth{max_depth}"

                print(f"  n_estimators={n_est}, lr={lr:.2f}, depth={max_depth}... ", end="")

                params = {
                    'objective': 'reg:squarederror',
                    'eta': lr,
                    'max_depth': max_depth,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'seed': RANDOM_SEED
                }

                model = xgb.train(params, dtrain, num_boost_round=n_est, verbose_eval=False)
                y_pred = model.predict(dtest)

                metrics = calculate_metrics(y_test, y_pred, config_name)
                metrics['n_estimators'] = n_est
                metrics['learning_rate'] = lr
                metrics['max_depth'] = max_depth
                xgb_results[config_name] = metrics

                print(f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")

    all_results.update(xgb_results)

    with open(results_dir / "xgboost_results.json", 'w') as f:
        json.dump(xgb_results, f, indent=2)

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print(f"Summary: {dataset_name}")
    print("=" * 80)

    all_r2 = {name: results['r2'] for name, results in all_results.items()}
    best_name = max(all_r2, key=all_r2.get)
    best_model = all_results[best_name]

    print(f"\nTotal models tested: {len(all_results)}")
    print(f"Best model: {best_model['model']}")
    print(f"  R² = {best_model['r2']:.4f}")
    print(f"  RMSE = {best_model['rmse']:.4f}")
    print(f"  MAE = {best_model['mae']:.4f}")

    # Save combined results
    with open(results_dir / "all_models_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")

    return all_results

def main():
    """Main execution function."""
    print("=" * 80)
    print("DASP Regression Testing - Match R Implementation")
    print("=" * 80)
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Results directory: {RESULTS_DIR}")

    start_time = time.time()

    # Test Dataset 1: Bone Collagen
    bone_results = test_regression_models(
        dataset_name="Bone Collagen",
        data_dir=R_DATA_DIR / "regression",
        target_col="%Collagen",
        results_dir=RESULTS_DIR / "bone_collagen"
    )

    # Test Dataset 2: Enamel d13C
    d13c_results = test_regression_models(
        dataset_name="Enamel d13C",
        data_dir=R_DATA_DIR / "d13c",
        target_col="d13C",
        results_dir=RESULTS_DIR / "d13c"
    )

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - ALL DATASETS")
    print("=" * 80)

    print("\nBone Collagen (n=49):")
    bone_r2 = {name: res['r2'] for name, res in bone_results.items()}
    best_bone_name = max(bone_r2, key=bone_r2.get)
    best_bone = bone_results[best_bone_name]
    print(f"  Best: {best_bone['model']} (R²={best_bone['r2']:.4f}, RMSE={best_bone['rmse']:.4f})")

    print("\nEnamel d13C (n=140):")
    d13c_r2 = {name: res['r2'] for name, res in d13c_results.items()}
    best_d13c_name = max(d13c_r2, key=d13c_r2.get)
    best_d13c = d13c_results[best_d13c_name]
    print(f"  Best: {best_d13c['model']} (R²={best_d13c['r2']:.4f}, RMSE={best_d13c['rmse']:.4f})")

    elapsed_time = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    print("\n" + "=" * 80)
    print("COMPLETE! All DASP regression tests finished.")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Compare with R: python compare_regression_results.py")
    print("  2. Generate visualizations: python visualize_comparisons.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
