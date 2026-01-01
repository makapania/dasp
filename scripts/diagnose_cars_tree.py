"""
Diagnostic script to identify why CARS fails for tree models.
This tests CARS directly with different model_type values.
"""
import sys
import traceback
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from spectral_predict.variable_selection import cars_selection

def run_diagnostic():
    print("=" * 60)
    print("CARS DIAGNOSTIC TEST")
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
            results[name] = "SUCCESS"

        except Exception as e:
            print(f"\n   FAILED!")
            print(f"   - Exception type: {type(e).__name__}")
            print(f"   - Exception message: {e}")
            print(f"\n   Full traceback:")
            traceback.print_exc()
            results[name] = f"FAILED: {type(e).__name__}: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "OK" if result == "SUCCESS" else "FAIL"
        print(f"   [{status}] {name}: {result}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostic()
