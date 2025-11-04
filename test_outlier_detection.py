"""Quick validation test for outlier_detection module."""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from spectral_predict.outlier_detection import (
    run_pca_outlier_detection,
    compute_q_residuals,
    compute_mahalanobis_distance,
    check_y_data_consistency,
    generate_outlier_report
)

print("=" * 60)
print("Testing Outlier Detection Module")
print("=" * 60)

# Create synthetic spectral data
np.random.seed(42)
n_samples = 100
n_wavelengths = 200

# Normal samples
X_normal = np.random.randn(n_samples - 5, n_wavelengths)

# Add 5 clear outliers with different characteristics
outlier1 = np.random.randn(1, n_wavelengths) * 5  # High variance
outlier2 = np.random.randn(1, n_wavelengths) + 10  # Shifted mean
outlier3 = np.random.randn(1, n_wavelengths) * 3 + 5  # Both
outlier4 = np.random.randn(1, n_wavelengths) - 8  # Negative shift
outlier5 = np.random.randn(1, n_wavelengths) * 4  # High variance

X = np.vstack([X_normal, outlier1, outlier2, outlier3, outlier4, outlier5])

# Create Y values (reference)
y_normal = 10 + 2 * np.random.randn(n_samples - 5)
y_outliers = np.array([50, 30, -5, 25, 35])  # Some Y outliers
y = np.concatenate([y_normal, y_outliers])

print(f"\nTest Data:")
print(f"  Samples: {X.shape[0]}")
print(f"  Wavelengths: {X.shape[1]}")
print(f"  Y range: {y.min():.2f} to {y.max():.2f}")
print(f"  Expected outliers: samples 95-99")

# Test 1: PCA outlier detection
print("\n" + "=" * 60)
print("Test 1: PCA Outlier Detection (Hotelling T²)")
print("=" * 60)

pca_results = run_pca_outlier_detection(X, y, n_components=5)

print(f"\nPCA Results:")
print(f"  Components: {pca_results['pca_model'].n_components_}")
print(f"  Variance explained: {pca_results['variance_explained'][:3]}")
print(f"  T² threshold: {pca_results['t2_threshold']:.2f}")
print(f"  Outliers detected: {pca_results['n_outliers']}")
print(f"  Outlier indices: {pca_results['outlier_indices']}")

assert pca_results['n_outliers'] >= 3, "Should detect at least 3 outliers"
print("  [PASS] Test passed")

# Test 2: Q-residuals
print("\n" + "=" * 60)
print("Test 2: Q-Residuals (SPE)")
print("=" * 60)

q_results = compute_q_residuals(X, pca_results['pca_model'], n_components=5)

print(f"\nQ-Residuals Results:")
print(f"  Q threshold: {q_results['q_threshold']:.2e}")
print(f"  Outliers detected: {q_results['n_outliers']}")
print(f"  Outlier indices: {q_results['outlier_indices']}")

assert q_results['n_outliers'] >= 3, "Should detect at least 3 outliers"
print("  [PASS] Test passed")

# Test 3: Mahalanobis distance
print("\n" + "=" * 60)
print("Test 3: Mahalanobis Distance")
print("=" * 60)

maha_results = compute_mahalanobis_distance(pca_results['scores'])

print(f"\nMahalanobis Results:")
print(f"  Median distance: {maha_results['median']:.2f}")
print(f"  MAD: {maha_results['mad']:.2f}")
print(f"  Threshold: {maha_results['threshold']:.2f}")
print(f"  Outliers detected: {maha_results['n_outliers']}")
print(f"  Outlier indices: {maha_results['outlier_indices']}")

assert maha_results['n_outliers'] >= 2, "Should detect at least 2 outliers"
print("  [PASS] Test passed")

# Test 4: Y data consistency
print("\n" + "=" * 60)
print("Test 4: Y Data Consistency")
print("=" * 60)

y_results = check_y_data_consistency(y, lower_bound=0, upper_bound=20)

print(f"\nY Consistency Results:")
print(f"  Mean: {y_results['mean']:.2f}")
print(f"  Std: {y_results['std']:.2f}")
print(f"  Median: {y_results['median']:.2f}")
print(f"  Range: {y_results['min']:.2f} to {y_results['max']:.2f}")
print(f"  Z-score outliers: {np.sum(y_results['z_outliers'])}")
print(f"  Range outliers: {np.sum(y_results['range_outliers'])}")
print(f"  Total outliers: {y_results['n_outliers']}")
print(f"  Outlier indices: {y_results['outlier_indices']}")

assert y_results['n_outliers'] >= 4, "Should detect at least 4 Y outliers"
print("  [PASS] Test passed")

# Test 5: Combined report
print("\n" + "=" * 60)
print("Test 5: Combined Outlier Report")
print("=" * 60)

report = generate_outlier_report(X, y, n_pca_components=5,
                                y_lower_bound=0, y_upper_bound=20)

print(f"\nCombined Report:")
print(f"  Total samples: {len(report['outlier_summary'])}")
print(f"  High confidence outliers (3+ flags): {len(report['high_confidence_outliers'])}")
print(f"  Moderate confidence outliers (2 flags): {len(report['moderate_confidence_outliers'])}")
print(f"  Low confidence outliers (1 flag): {len(report['low_confidence_outliers'])}")

print(f"\nSummary DataFrame columns:")
print(f"  {list(report['outlier_summary'].columns)}")

# Show high confidence outliers
if len(report['high_confidence_outliers']) > 0:
    print(f"\nHigh Confidence Outliers:")
    print(report['high_confidence_outliers'][['Sample_Index', 'Y_Value', 'Total_Flags',
                                               'T2_Outlier', 'Q_Outlier', 'Maha_Outlier', 'Y_Outlier']])

assert len(report['high_confidence_outliers']) >= 2, "Should detect at least 2 high-confidence outliers"
print("  [PASS] Test passed")

# Test 6: Edge cases
print("\n" + "=" * 60)
print("Test 6: Edge Cases")
print("=" * 60)

# Test with DataFrame input
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

report_df = generate_outlier_report(X_df, y_series, n_pca_components=3)
print(f"  [PASS] Works with DataFrame/Series input")

# Test with small dataset
X_small = X[:10, :]
y_small = y[:10]
report_small = generate_outlier_report(X_small, y_small, n_pca_components=3)
print(f"  [PASS] Works with small dataset (n=10)")

# Test with single component
pca_single = run_pca_outlier_detection(X, y, n_components=1)
print(f"  [PASS] Works with single component")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)

print("\n[SUCCESS] Module validation successful!")
print("  - All 5 core functions working correctly")
print("  - Edge cases handled properly")
print("  - Ready for GUI integration")
