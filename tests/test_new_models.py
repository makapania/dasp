"""
Test new ML models (ElasticNet, SVR, XGBoost, LightGBM, CatBoost) with bone collagen data.

This test verifies:
1. Each model can be instantiated
2. Each model can train on data
3. Each model can make predictions
4. Feature importance extraction works
5. Models perform reasonably on the example dataset
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spectral_predict.models import get_model, get_feature_importances
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_bone_collagen_data(n_samples=25):
    """
    Load bone collagen data for testing.

    Returns synthetic spectral data correlated with collagen content.
    """
    example_dir = Path(__file__).parent.parent / 'example'
    csv_path = example_dir / 'BoneCollagen.csv'

    # Load reference data
    df = pd.read_csv(csv_path)

    # Limit to n_samples
    df = df.head(n_samples)

    # Create synthetic spectral data (2151 wavelengths)
    np.random.seed(42)
    n_wavelengths = 2151
    X = np.random.randn(len(df), n_wavelengths) * 0.1

    # Add signal correlated with collagen content
    for i, collagen in enumerate(df['%Collagen'].values):
        baseline = collagen / 20.0
        X[i] += baseline * np.sin(np.linspace(0, 10, n_wavelengths))
        X[i] += baseline * 0.5 * np.cos(np.linspace(0, 5, n_wavelengths))
        X[i] += np.random.randn(n_wavelengths) * 0.05

    y = df['%Collagen'].values

    # Add some important features manually for testing feature importance
    # Feature 100, 500, 1000 will have strong correlation
    X[:, 100] = y + np.random.randn(len(y)) * 0.5
    X[:, 500] = y * 0.8 + np.random.randn(len(y)) * 0.6
    X[:, 1000] = y * 0.6 + np.random.randn(len(y)) * 0.8

    return X, y


def test_elasticnet():
    """Test ElasticNet regression model."""
    print("\n" + "="*80)
    print("TEST 1: ElasticNet Model")
    print("="*80)

    X, y = load_bone_collagen_data()
    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} % collagen")

    # Get model
    model = get_model("ElasticNet", task_type='regression')
    print(f"\nModel: {model}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train
    print("\nTraining...")
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train RMSE: {rmse_train:.3f}, R²: {r2_train:.3f}")
    print(f"  Test  RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

    # Feature importance
    print("\nExtracting feature importances...")
    importances = get_feature_importances(model, "ElasticNet", X_train, y_train)

    if importances is not None:
        print(f"  Shape: {importances.shape}")
        print(f"  Non-zero features: {np.sum(importances != 0)}")
        # ElasticNet should have sparse coefficients
        assert np.sum(importances != 0) < X.shape[1], "ElasticNet should be sparse"
        print("  ✓ Sparsity confirmed (L1 regularization working)")
    else:
        print("  ⚠ No feature importances extracted")

    assert rmse_test < 10.0, f"Test RMSE too high: {rmse_test:.3f}"
    print("\n✓ ElasticNet test passed")

    return {'model': 'ElasticNet', 'rmse': rmse_test, 'r2': r2_test}


def test_svr():
    """Test Support Vector Regression model."""
    print("\n" + "="*80)
    print("TEST 2: SVR (Support Vector Regression)")
    print("="*80)

    X, y = load_bone_collagen_data(n_samples=20)  # Smaller dataset for SVR speed
    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} features")

    # SVR needs scaled features - use a subset
    print("Using feature subset (first 100) for SVR speed...")
    X_subset = X[:, :100]

    model = get_model("SVR", task_type='regression')
    print(f"\nModel: {model}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.3, random_state=42
    )

    # Scale data (important for SVR)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining...")
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train RMSE: {rmse_train:.3f}, R²: {r2_train:.3f}")
    print(f"  Test  RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

    # SVR doesn't have direct feature importance
    print("\nFeature importance for SVR: Not directly available (uses kernel)")

    assert rmse_test < 10.0, f"Test RMSE too high: {rmse_test:.3f}"
    print("\n✓ SVR test passed")

    return {'model': 'SVR', 'rmse': rmse_test, 'r2': r2_test}


def test_xgboost():
    """Test XGBoost regression model."""
    print("\n" + "="*80)
    print("TEST 3: XGBoost")
    print("="*80)

    X, y = load_bone_collagen_data()
    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} features")

    model = get_model("XGBoost", task_type='regression')
    print(f"\nModel: {model}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train RMSE: {rmse_train:.3f}, R²: {r2_train:.3f}")
    print(f"  Test  RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

    # Feature importance
    print("\nExtracting feature importances...")
    importances = get_feature_importances(model, "XGBoost", X_train, y_train)

    if importances is not None:
        print(f"  Shape: {importances.shape}")
        top_10_idx = np.argsort(importances)[-10:][::-1]
        print(f"  Top 10 features: {top_10_idx}")
        print(f"  ✓ Feature importances extracted successfully")
    else:
        print("  ⚠ No feature importances extracted")

    assert rmse_test < 10.0, f"Test RMSE too high: {rmse_test:.3f}"
    print("\n✓ XGBoost test passed")

    return {'model': 'XGBoost', 'rmse': rmse_test, 'r2': r2_test}


def test_lightgbm():
    """Test LightGBM regression model."""
    print("\n" + "="*80)
    print("TEST 4: LightGBM")
    print("="*80)

    X, y = load_bone_collagen_data()
    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} features")

    model = get_model("LightGBM", task_type='regression')
    print(f"\nModel: {model}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\nTraining...")
    # LightGBM can be verbose, suppress output
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train RMSE: {rmse_train:.3f}, R²: {r2_train:.3f}")
    print(f"  Test  RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

    # Feature importance
    print("\nExtracting feature importances...")
    importances = get_feature_importances(model, "LightGBM", X_train, y_train)

    if importances is not None:
        print(f"  Shape: {importances.shape}")
        top_10_idx = np.argsort(importances)[-10:][::-1]
        print(f"  Top 10 features: {top_10_idx}")
        print("  ✓ Feature importances extracted")
    else:
        print("  ⚠ No feature importances extracted")

    assert rmse_test < 10.0, f"Test RMSE too high: {rmse_test:.3f}"
    print("\n✓ LightGBM test passed")

    return {'model': 'LightGBM', 'rmse': rmse_test, 'r2': r2_test}


def test_catboost():
    """Test CatBoost regression model."""
    print("\n" + "="*80)
    print("TEST 5: CatBoost")
    print("="*80)

    X, y = load_bone_collagen_data()
    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} features")

    model = get_model("CatBoost", task_type='regression')
    print(f"\nModel: {model}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\nTraining...")
    # CatBoost is verbose by default, suppress
    model.set_params(verbose=0)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train RMSE: {rmse_train:.3f}, R²: {r2_train:.3f}")
    print(f"  Test  RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

    # Feature importance
    print("\nExtracting feature importances...")
    importances = get_feature_importances(model, "CatBoost", X_train, y_train)

    if importances is not None:
        print(f"  Shape: {importances.shape}")
        top_10_idx = np.argsort(importances)[-10:][::-1]
        print(f"  Top 10 features: {top_10_idx}")
        print("  ✓ Feature importances extracted")
    else:
        print("  ⚠ No feature importances extracted")

    assert rmse_test < 10.0, f"Test RMSE too high: {rmse_test:.3f}"
    print("\n✓ CatBoost test passed")

    return {'model': 'CatBoost', 'rmse': rmse_test, 'r2': r2_test}


def test_model_comparison():
    """Compare all new models."""
    print("\n" + "="*80)
    print("TEST 6: Model Comparison")
    print("="*80)

    results = []

    # Run all tests
    results.append(test_elasticnet())
    results.append(test_svr())
    results.append(test_xgboost())
    results.append(test_lightgbm())
    results.append(test_catboost())

    # Summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # Best model
    best_idx = df['rmse'].idxmin()
    best_model = df.loc[best_idx, 'model']
    best_rmse = df.loc[best_idx, 'rmse']

    print(f"\nBest model: {best_model} (RMSE: {best_rmse:.3f})")

    # All models should work
    assert len(results) == 5, "All 5 models should complete"
    print("\n✓ All models tested successfully")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("NEW ML MODELS TEST SUITE")
    print("Testing: ElasticNet, SVR, XGBoost, LightGBM, CatBoost")
    print("="*80)

    try:
        test_elasticnet()
        test_svr()
        test_xgboost()
        test_lightgbm()
        test_catboost()
        test_model_comparison()

        print("\n" + "="*80)
        print("ALL MODEL TESTS PASSED ✓")
        print("="*80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
