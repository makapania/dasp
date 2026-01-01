"""
Diagnostic script to identify why CARS fails for tree models.
This tests CARS directly with different model_type values.
Also computes R² values for regression verification.
"""
import sys
import traceback
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

# Add src to path
sys.path.insert(0, 'src')

from spectral_predict.variable_selection import cars_selection


def compute_r2_with_selected_vars(X, y, importances, n_top=50, model_type='PLS'):
    """Compute R² using top-N selected variables."""
    if importances is None or np.all(importances == 0):
        return None

    # Select top N variables
    top_indices = np.argsort(importances)[-n_top:][::-1]
    X_subset = X[:, top_indices]

    # Fit model and compute R²
    if model_type == 'PLS':
        model = PLSRegression(n_components=min(5, n_top, X_subset.shape[0] - 1))
    else:  # Ridge
        model = Ridge(alpha=1.0)

    scores = cross_val_score(model, X_subset, y, cv=5, scoring='r2')
    return np.mean(scores)


def run_diagnostic():
    print("=" * 60)
    print("CARS DIAGNOSTIC TEST WITH R² VERIFICATION")
    print("=" * 60)

    # Generate synthetic spectral data
    print("\n1. Generating synthetic spectral data...")
    np.random.seed(42)

    n_samples = 100
    n_features = 500  # Typical spectral data size

    # Create informative features (first 50) and noise features (rest)
    X_informative = np.random.randn(n_samples, 50) * 2
    X_noise = np.random.randn(n_samples, n_features - 50) * 0.1
    X = np.hstack([X_informative, X_noise])

    # Target is a linear combination of first 10 features + noise
    y = X[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.5

    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   n_samples={X.shape[0]}, n_features={X.shape[1]}")

    # Test configurations
    test_configs = [
        {"name": "PLS (default)", "model_type": None},
        {"name": "PLS (explicit)", "model_type": "PLS"},
        {"name": "Ridge", "model_type": "Ridge"},
        {"name": "RandomForest", "model_type": "RandomForest"},
        {"name": "LightGBM", "model_type": "LightGBM"},
        {"name": "XGBoost", "model_type": "XGBoost"},
    ]

    results = {}

    for config in test_configs:
        name = config["name"]
        model_type = config["model_type"]

        print(f"\n{'=' * 60}")
        print(f"2. Testing CARS with model_type='{model_type}' ({name})")
        print("=" * 60)

        try:
            importances = cars_selection(
                X, y,
                n_iterations=10,  # Reduced for speed
                pls_components=5,
                cv_folds=5,
                monte_carlo_samples=80,
                random_state=42,
                model_type=model_type
            )

            n_selected = np.sum(importances > 0)
            print(f"\n   SUCCESS!")
            print(f"   - Importances shape: {importances.shape}")
            print(f"   - Non-zero importances: {n_selected}")
            print(f"   - Min: {np.min(importances):.4f}, Max: {np.max(importances):.4f}")

            # Compute R² for PLS and Ridge models
            if name in ["PLS (default)", "Ridge"]:
                r2_pls = compute_r2_with_selected_vars(X, y, importances, n_top=50, model_type='PLS')
                r2_ridge = compute_r2_with_selected_vars(X, y, importances, n_top=50, model_type='Ridge')
                print(f"   - R² (PLS, top-50): {r2_pls:.6f}")
                print(f"   - R² (Ridge, top-50): {r2_ridge:.6f}")
                results[name] = {"status": "SUCCESS", "r2_pls": r2_pls, "r2_ridge": r2_ridge}
            else:
                results[name] = {"status": "SUCCESS", "r2_pls": None, "r2_ridge": None}

        except Exception as e:
            print(f"\n   FAILED!")
            print(f"   - Exception type: {type(e).__name__}")
            print(f"   - Exception message: {e}")
            print(f"\n   Full traceback:")
            traceback.print_exc()
            results[name] = {"status": f"FAILED: {type(e).__name__}: {e}", "r2_pls": None, "r2_ridge": None}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "OK" if result["status"] == "SUCCESS" else "FAIL"
        r2_info = ""
        if result["r2_pls"] is not None:
            r2_info = f" | R²(PLS)={result['r2_pls']:.6f}, R²(Ridge)={result['r2_ridge']:.6f}"
        print(f"   [{status}] {name}: {result['status']}{r2_info}")

    # Save baseline R² values for comparison
    print("\n" + "=" * 60)
    print("BASELINE R² VALUES (SAVE THESE FOR COMPARISON)")
    print("=" * 60)
    for name in ["PLS (default)", "Ridge"]:
        if name in results and results[name]["r2_pls"] is not None:
            print(f"   {name}: R²(PLS)={results[name]['r2_pls']:.6f}, R²(Ridge)={results[name]['r2_ridge']:.6f}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_diagnostic()
