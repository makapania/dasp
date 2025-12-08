"""
Test script for model save/load functionality in Spectral Predict v3.
"""

import numpy as np
from spectral_predict_v3.core import model_io
from spectral_predict_v3.core.models import get_model
from spectral_predict_v3.core.preprocess import SNV, SavgolDerivative
import tempfile
import os

def test_model_save_load():
    """Test saving and loading a model."""
    print("Testing model save/load functionality...")

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 50
    n_wavelengths = 100
    X_train = np.random.randn(n_samples, n_wavelengths)
    y_train = np.random.randn(n_samples) * 10 + 50  # Random target values
    wavelengths = np.linspace(400, 800, n_wavelengths)

    # Train a simple model
    print("Training PLS model...")
    model = get_model('PLS', task_type='regression', n_components=5)
    model.fit(X_train, y_train)

    # Create model bundle
    print("Creating model bundle...")
    model_bundle = model_io.create_model_bundle(
        model=model,
        model_name='PLS',
        preprocessing='raw',
        wavelengths=wavelengths,
        target_name='Test_Target',
        task_type='regression',
        metrics={'RMSE': 2.5, 'R2': 0.85},
        params={'n_components': 5}
    )

    # Save model to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    temp_path = temp_file.name
    temp_file.close()

    try:
        print(f"Saving model to {temp_path}...")
        model_io.save_model(model_bundle, temp_path)
        print("Model saved successfully!")

        # Load model back
        print(f"Loading model from {temp_path}...")
        loaded_bundle = model_io.load_model(temp_path)
        print("Model loaded successfully!")

        # Verify loaded bundle
        print("\nVerifying loaded model bundle:")
        print(f"  Model name: {loaded_bundle['model_name']}")
        print(f"  Preprocessing: {loaded_bundle['preprocessing']}")
        print(f"  Task type: {loaded_bundle['task_type']}")
        print(f"  Target name: {loaded_bundle['target_name']}")
        print(f"  Wavelengths: {len(loaded_bundle['wavelengths'])} points")
        print(f"  Metrics: {loaded_bundle['metrics']}")

        # Test prediction with loaded model
        print("\nTesting prediction with loaded model...")
        X_test = np.random.randn(10, n_wavelengths)
        predictions, info = model_io.apply_model(loaded_bundle, X_test, wavelengths)
        print(f"Predictions generated: {len(predictions)} samples")
        print(f"  Interpolated: {info['interpolated']}")
        print(f"  Variables selected: {info['variables_selected']}")
        print(f"  Sample predictions: {predictions[:3]}")

        # Test with different wavelengths (interpolation)
        print("\nTesting with different wavelengths (should interpolate)...")
        wavelengths_new = np.linspace(420, 780, 80)  # Different range/resolution
        X_test_new = np.random.randn(5, len(wavelengths_new))
        predictions_new, info_new = model_io.apply_model(loaded_bundle, X_test_new, wavelengths_new)
        print(f"Predictions generated: {len(predictions_new)} samples")
        print(f"  Interpolated: {info_new['interpolated']}")
        print(f"  Sample predictions: {predictions_new[:3]}")

        print("\nAll tests passed!")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"\nCleaned up temporary file: {temp_path}")


def test_preprocessing_application():
    """Test preprocessing application in model_io."""
    print("\n" + "="*60)
    print("Testing preprocessing application...")

    np.random.seed(42)
    X = np.random.randn(10, 50) + 100  # Offset to test SNV

    # Test raw (no preprocessing)
    X_raw = model_io._apply_preprocessing(X, 'raw')
    assert np.allclose(X, X_raw), "Raw preprocessing should not change data"
    print("  raw: OK")

    # Test SNV
    X_snv = model_io._apply_preprocessing(X, 'snv')
    # Each row should have mean ~0 and std ~1
    means = np.mean(X_snv, axis=1)
    stds = np.std(X_snv, axis=1)
    assert np.allclose(means, 0, atol=1e-10), f"SNV means not near 0: {means}"
    assert np.allclose(stds, 1, atol=1e-10), f"SNV stds not near 1: {stds}"
    print("  snv: OK")

    # Test derivatives
    X_deriv1 = model_io._apply_preprocessing(X, 'deriv1_w7')
    assert X_deriv1.shape == X.shape, "Derivative should preserve shape"
    print("  deriv1_w7: OK")

    X_deriv2 = model_io._apply_preprocessing(X, 'deriv2_w19')
    assert X_deriv2.shape == X.shape, "Derivative should preserve shape"
    print("  deriv2_w19: OK")

    # Test SNV + derivative
    X_snv_deriv1 = model_io._apply_preprocessing(X, 'snv_deriv1_w7')
    assert X_snv_deriv1.shape == X.shape, "SNV+derivative should preserve shape"
    print("  snv_deriv1_w7: OK")

    X_snv_deriv2 = model_io._apply_preprocessing(X, 'snv_deriv2_w19')
    assert X_snv_deriv2.shape == X.shape, "SNV+derivative should preserve shape"
    print("  snv_deriv2_w19: OK")

    print("All preprocessing tests passed!")


if __name__ == '__main__':
    test_model_save_load()
    test_preprocessing_application()
    print("\n" + "="*60)
    print("All integration tests completed successfully!")
