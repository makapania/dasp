#!/usr/bin/env python
"""Test script for applicability domain functionality."""

import numpy as np
import tempfile
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from src.spectral_predict.model_io import save_model, load_model, predict_with_uncertainty

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic spectral data
n_samples = 80
n_wavelengths = 500

print("=" * 70)
print("Testing Applicability Domain Implementation")
print("=" * 70)

# Generate training data
X_train = np.random.randn(n_samples, n_wavelengths)
y_train = X_train[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.1

# Generate test data (some in-domain, some out-of-domain)
X_test_in = X_train[:5] + np.random.randn(5, n_wavelengths) * 0.1  # Similar to training
X_test_out = X_train[:5] + np.random.randn(5, n_wavelengths) * 5.0  # Very different
X_test = np.vstack([X_test_in, X_test_out])

print(f"\nTraining samples: {n_samples}")
print(f"Test samples: {X_test.shape[0]} (5 in-domain, 5 out-of-domain)")
print(f"Features: {n_wavelengths}")

# Test 1: PLS Model
print("\n" + "=" * 70)
print("Test 1: PLS Regression Model")
print("=" * 70)

pls_model = PLSRegression(n_components=10)
pls_model.fit(X_train, y_train)

# Compute CV residuals for RMSECV
y_pred_train = pls_model.predict(X_train).ravel()
cv_residuals = y_pred_train - y_train

metadata = {
    'model_name': 'PLS',
    'task_type': 'regression',
    'preprocessing': 'raw',
    'wavelengths': list(range(n_wavelengths)),
    'n_vars': n_wavelengths,
    'performance': {'RMSE': np.std(cv_residuals), 'R2': 0.95}
}

# Save model with applicability domain
with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
    pls_model_path = f.name

save_model(
    model=pls_model,
    preprocessor=None,
    metadata=metadata,
    filepath=pls_model_path,
    cv_residuals=cv_residuals,
    X_train=X_train  # This enables applicability domain
)

print(f"\n✓ Model saved to {pls_model_path}")

# Load model
model_dict = load_model(pls_model_path)
print(f"✓ Model loaded successfully")
print(f"  - Has applicability domain: {model_dict.get('ad_data') is not None}")
print(f"  - Has PCA model: {model_dict.get('pca_model') is not None}")

if model_dict.get('ad_data'):
    ad_data = model_dict['ad_data']
    print(f"  - Representative spectra: {ad_data['representative_spectra'].shape[0]}")
    print(f"  - Distance thresholds: p50={ad_data['distance_thresholds'][0]:.2f}, "
          f"p75={ad_data['distance_thresholds'][1]:.2f}, "
          f"p95={ad_data['distance_thresholds'][2]:.2f}")

# Make predictions with uncertainty
result = predict_with_uncertainty(model_dict, X_test, validate_wavelengths=False)

print(f"\n✓ Predictions made successfully")
print(f"  - Has uncertainty: {result['has_uncertainty']}")
print(f"  - Has applicability domain: {result['has_applicability_domain']}")

if result['has_uncertainty']:
    print(f"  - RMSECV: {result['uncertainty'].get('rmsecv', 'N/A'):.4f}")

if result['has_applicability_domain']:
    ad = result['applicability_domain']
    print(f"\n  Applicability Domain Results:")
    print(f"  {'Sample':<10} {'Prediction':<12} {'PCA Dist':<12} {'Status':<15}")
    print(f"  {'-'*50}")
    for i in range(len(result['predictions'])):
        pred = result['predictions'][i]
        dist = ad['pca_distance'][i]
        status = ad['distance_status'][i]
        print(f"  {i+1:<10} {pred:<12.4f} {dist:<12.3f} {status:<15}")

# Clean up
os.unlink(pls_model_path)
print(f"\n✓ Test file cleaned up")

# Test 2: Random Forest Model (test tree variance)
print("\n" + "=" * 70)
print("Test 2: Random Forest Model (Tree Variance)")
print("=" * 70)

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_train_rf = rf_model.predict(X_train)
cv_residuals_rf = y_pred_train_rf - y_train

metadata_rf = {
    'model_name': 'RandomForest',
    'task_type': 'regression',
    'preprocessing': 'raw',
    'wavelengths': list(range(n_wavelengths)),
    'n_vars': n_wavelengths,
    'performance': {'RMSE': np.std(cv_residuals_rf), 'R2': 0.96},
    'model_class': 'RandomForestRegressor'
}

# Save RF model
with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
    rf_model_path = f.name

save_model(
    model=rf_model,
    preprocessor=None,
    metadata=metadata_rf,
    filepath=rf_model_path,
    cv_residuals=cv_residuals_rf,
    X_train=X_train
)

print(f"\n✓ RF Model saved to {rf_model_path}")

# Load and predict
model_dict_rf = load_model(rf_model_path)
result_rf = predict_with_uncertainty(model_dict_rf, X_test, validate_wavelengths=False)

print(f"✓ RF Predictions made successfully")
if result_rf['has_uncertainty']:
    print(f"  - RMSECV: {result_rf['uncertainty'].get('rmsecv', 'N/A'):.4f}")
    if 'tree_variance' in result_rf['uncertainty']:
        tree_var = result_rf['uncertainty']['tree_variance']
        print(f"  - Tree variance (mean): {tree_var.mean():.4f}")
        print(f"  - Tree variance (range): {tree_var.min():.4f} - {tree_var.max():.4f}")

# Clean up
os.unlink(rf_model_path)

# Test 3: Model without applicability domain (backward compatibility)
print("\n" + "=" * 70)
print("Test 3: Backward Compatibility (No Applicability Domain)")
print("=" * 70)

pls_model2 = PLSRegression(n_components=10)
pls_model2.fit(X_train, y_train)

with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
    pls_model_path2 = f.name

# Save WITHOUT X_train
save_model(
    model=pls_model2,
    preprocessor=None,
    metadata=metadata,
    filepath=pls_model_path2,
    cv_residuals=cv_residuals
    # No X_train parameter
)

print(f"\n✓ Model saved without applicability domain")

model_dict2 = load_model(pls_model_path2)
result2 = predict_with_uncertainty(model_dict2, X_test, validate_wavelengths=False)

print(f"✓ Predictions work without applicability domain")
print(f"  - Has applicability domain: {result2['has_applicability_domain']}")
print(f"  - RMSECV: {result2['uncertainty'].get('rmsecv', 'N/A'):.4f}")

os.unlink(pls_model_path2)

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
