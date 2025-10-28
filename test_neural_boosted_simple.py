"""Simple test script for Neural Boosted Regression (no pytest required)."""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.neural_boosted import NeuralBoostedRegressor

def test_basic_fit():
    """Test 1: Basic fit and predict."""
    print("\n" + "="*60)
    print("TEST 1: Basic Fit and Predict")
    print("="*60)

    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    model = NeuralBoostedRegressor(
        n_estimators=30,
        learning_rate=0.1,
        hidden_layer_size=5,
        max_iter=500,
        random_state=42,
        verbose=1
    )

    print("Fitting model...")
    model.fit(X, y)

    print(f"\nFitted {model.n_estimators_} weak learners")
    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    print(f"Training R² = {r2:.4f}")
    print(f"Training RMSE = {rmse:.4f}")

    assert r2 > 0.7, f"R² too low: {r2:.4f}"
    print("✓ Test 1 PASSED")

def test_early_stopping():
    """Test 2: Early stopping."""
    print("\n" + "="*60)
    print("TEST 2: Early Stopping")
    print("="*60)

    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    model = NeuralBoostedRegressor(
        n_estimators=100,
        learning_rate=0.1,
        hidden_layer_size=3,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
        verbose=0
    )

    print("Fitting with early stopping...")
    model.fit(X, y)

    print(f"Stopped at {model.n_estimators_}/{model.n_estimators} estimators")
    print(f"Validation scores: {len(model.validation_score_)} recorded")

    assert model.n_estimators_ < 100, "Early stopping should trigger"
    print("✓ Test 2 PASSED")

def test_feature_importances():
    """Test 3: Feature importance extraction."""
    print("\n" + "="*60)
    print("TEST 3: Feature Importances")
    print("="*60)

    np.random.seed(42)
    n_samples = 150
    n_features = 50

    # Create data where only features 5, 10, 15 matter
    X = np.random.randn(n_samples, n_features) * 0.1
    y = 5.0 * X[:, 5] + 3.0 * X[:, 10] - 2.0 * X[:, 15] + np.random.randn(n_samples) * 0.1

    model = NeuralBoostedRegressor(
        n_estimators=25,
        learning_rate=0.1,
        hidden_layer_size=5,
        random_state=42,
        verbose=0
    )

    print("Fitting model (only features 5, 10, 15 are important)...")
    model.fit(X, y)

    importances = model.get_feature_importances()
    top_5_indices = np.argsort(importances)[-5:][::-1]

    print(f"\nTop 5 features by importance: {top_5_indices}")
    print(f"Feature 5 importance: {importances[5]:.4f} (rank #{np.where(np.argsort(importances)[::-1] == 5)[0][0] + 1})")
    print(f"Feature 10 importance: {importances[10]:.4f} (rank #{np.where(np.argsort(importances)[::-1] == 10)[0][0] + 1})")
    print(f"Feature 15 importance: {importances[15]:.4f} (rank #{np.where(np.argsort(importances)[::-1] == 15)[0][0] + 1})")

    important_features = {5, 10, 15}
    detected = important_features.intersection(set(top_5_indices))
    print(f"\nDetected {len(detected)}/3 important features in top 5")

    assert len(detected) >= 2, f"Should detect at least 2 important features"
    print("✓ Test 3 PASSED")

def test_spectral_data():
    """Test 4: Simulated spectral data."""
    print("\n" + "="*60)
    print("TEST 4: Spectral-Like Data")
    print("="*60)

    np.random.seed(42)
    n_samples = 80
    n_wavelengths = 500

    # Simulate NIR spectrum
    X = np.random.randn(n_samples, n_wavelengths) * 0.05 + 1.0

    # Target based on specific wavelengths (absorption peaks)
    y = (2.0 * X[:, 100] +  # O-H peak
         1.5 * X[:, 200] +  # C-H peak
         -0.8 * X[:, 300] +  # N-H peak
         np.random.randn(n_samples) * 0.05)

    print(f"Data: {n_samples} samples × {n_wavelengths} wavelengths")
    print(f"Important wavelengths: 100, 200, 300")

    model = NeuralBoostedRegressor(
        n_estimators=30,
        learning_rate=0.1,
        hidden_layer_size=5,
        early_stopping=True,
        random_state=42,
        verbose=0
    )

    print("\nFitting model...")
    model.fit(X, y)

    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    print(f"\nUsed {model.n_estimators_} estimators (early stopping)")
    print(f"Training R² = {r2:.4f}")
    print(f"Training RMSE = {rmse:.4f}")

    # Check feature importances
    importances = model.get_feature_importances()
    top_10_indices = np.argsort(importances)[-10:][::-1]

    print(f"\nTop 10 wavelengths by importance: {top_10_indices}")

    important_wavelengths = {100, 200, 300}
    detected = important_wavelengths.intersection(set(top_10_indices))
    print(f"Detected {len(detected)}/3 important wavelengths in top 10: {detected}")

    assert r2 > 0.6, f"R² too low: {r2:.4f}"
    assert len(detected) >= 1, "Should detect at least 1 important wavelength"
    print("✓ Test 4 PASSED")

def test_generalization():
    """Test 5: Generalization to test data."""
    print("\n" + "="*60)
    print("TEST 5: Generalization")
    print("="*60)

    X, y = make_regression(n_samples=200, n_features=30, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    model = NeuralBoostedRegressor(
        n_estimators=50,
        learning_rate=0.1,
        hidden_layer_size=5,
        early_stopping=True,
        random_state=42,
        verbose=0
    )

    print("\nFitting model...")
    model.fit(X_train, y_train)

    # Training performance
    pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))

    # Test performance
    pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    print(f"\nUsed {model.n_estimators_} estimators")
    print(f"Training R² = {r2_train:.4f}, RMSE = {rmse_train:.4f}")
    print(f"Test R² = {r2_test:.4f}, RMSE = {rmse_test:.4f}")
    print(f"Gap: ΔR² = {r2_train - r2_test:.4f}")

    assert r2_test > 0.5, f"Test R² too low: {r2_test:.4f}"
    assert r2_train - r2_test < 0.3, "Overfitting detected (large train-test gap)"
    print("✓ Test 5 PASSED")

def test_huber_loss():
    """Test 6: Huber loss with outliers."""
    print("\n" + "="*60)
    print("TEST 6: Huber Loss (Outlier Robustness)")
    print("="*60)

    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    # Add outliers
    outlier_indices = [0, 1, 2, 3]
    y[outlier_indices] = y[outlier_indices] + np.array([15, -12, 18, -15])
    print(f"Added {len(outlier_indices)} outliers")

    # MSE model
    model_mse = NeuralBoostedRegressor(
        loss='mse',
        n_estimators=25,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )

    # Huber model
    model_huber = NeuralBoostedRegressor(
        loss='huber',
        huber_delta=1.35,
        n_estimators=25,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )

    print("\nFitting MSE model...")
    model_mse.fit(X, y)

    print("Fitting Huber model...")
    model_huber.fit(X, y)

    # Evaluate on clean data (excluding outliers)
    mask = np.ones(len(y), dtype=bool)
    mask[outlier_indices] = False
    X_clean = X[mask]
    y_clean = y[mask]

    r2_mse = r2_score(y_clean, model_mse.predict(X_clean))
    r2_huber = r2_score(y_clean, model_huber.predict(X_clean))

    print(f"\nPerformance on clean data (excluding outliers):")
    print(f"MSE model R² = {r2_mse:.4f}")
    print(f"Huber model R² = {r2_huber:.4f}")

    assert r2_huber > 0.5, "Huber should fit clean data reasonably"
    print("✓ Test 6 PASSED")

def main():
    """Run all tests."""
    print("="*60)
    print("Neural Boosted Regression - Test Suite")
    print("="*60)

    tests = [
        test_basic_fit,
        test_early_stopping,
        test_feature_importances,
        test_spectral_data,
        test_generalization,
        test_huber_loss
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test FAILED: {e}")
            failed += 1

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
