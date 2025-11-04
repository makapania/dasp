"""
Example usage of outlier_detection module for GUI integration.

This demonstrates how to use the outlier detection functions in a typical
workflow and shows the data structures returned for GUI visualization.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from spectral_predict.outlier_detection import generate_outlier_report

# Example: Load spectral data (replace with actual GUI data loading)
# Assume user has uploaded:
# - X: spectral data (samples x wavelengths)
# - y: reference values
# - wavelengths: array of wavelength values

# For demonstration, create synthetic data
np.random.seed(42)
n_samples = 50
n_wavelengths = 100

X = np.random.randn(n_samples, n_wavelengths)
y = 10 + 2 * np.random.randn(n_samples)
wavelengths = np.linspace(1000, 2500, n_wavelengths)

# Add some outliers for demonstration
X[48] = X[48] * 5  # Spectral outlier
y[49] = 50  # Y outlier

print("=" * 70)
print("OUTLIER DETECTION - GUI INTEGRATION EXAMPLE")
print("=" * 70)

# ============================================================================
# Step 1: User clicks "Run Outlier Detection" button in GUI
# ============================================================================

print("\n[1] Running outlier detection...")

# Get parameters from GUI controls
n_pca_components = 5  # From spinbox/dropdown
y_lower_bound = 0.0   # From optional text entry (None if empty)
y_upper_bound = 20.0  # From optional text entry (None if empty)

# Run comprehensive outlier detection
report = generate_outlier_report(
    X=X,
    y=y,
    n_pca_components=n_pca_components,
    y_lower_bound=y_lower_bound,
    y_upper_bound=y_upper_bound
)

print(f"   Complete! Analyzed {len(report['outlier_summary'])} samples")

# ============================================================================
# Step 2: Display results in GUI - Summary Statistics
# ============================================================================

print("\n[2] Summary Statistics (display in GUI):")
print("-" * 70)

summary_text = f"""
Detection Results:
  • Hotelling T² outliers: {report['pca']['n_outliers']}
  • Q-residuals outliers: {report['q_residuals']['n_outliers']}
  • Mahalanobis outliers: {report['mahalanobis']['n_outliers']}
  • Y-value outliers: {report['y_consistency']['n_outliers']}

Confidence Levels:
  • High confidence (3+ methods): {len(report['high_confidence_outliers'])} samples
  • Moderate confidence (2 methods): {len(report['moderate_confidence_outliers'])} samples
  • Low confidence (1 method): {len(report['low_confidence_outliers'])} samples

PCA Variance Explained:
  • PC1: {report['pca']['variance_explained'][0]:.1%}
  • PC2: {report['pca']['variance_explained'][1]:.1%}
  • PC3: {report['pca']['variance_explained'][2]:.1%}
  • Total (5 PCs): {report['pca']['variance_explained'].sum():.1%}
"""

print(summary_text)

# ============================================================================
# Step 3: Create interactive plots (use matplotlib in GUI)
# ============================================================================

print("\n[3] Data for interactive plots:")
print("-" * 70)

# Plot 1: PCA Score Plot (PC1 vs PC2)
print("\n  Plot 1: PCA Score Plot")
print("  - X data: PC1 scores =", report['pca']['scores'][:, 0].shape)
print("  - Y data: PC2 scores =", report['pca']['scores'][:, 1].shape)
print("  - Colors: Y values")
print("  - Sizes: Hotelling T² values")
print("  - Hover labels: Sample indices")

# Plot 2: Hotelling T² Chart
print("\n  Plot 2: Hotelling T² Chart")
print("  - X data: Sample indices")
print("  - Y data: T² values =", report['pca']['hotelling_t2'].shape)
print("  - Threshold line:", f"{report['pca']['t2_threshold']:.2f}")
print("  - Highlight outliers in red")

# Plot 3: Q-Residuals Chart
print("\n  Plot 3: Q-Residuals Chart")
print("  - X data: Sample indices")
print("  - Y data: Q-residuals =", report['q_residuals']['q_residuals'].shape)
print("  - Threshold line:", f"{report['q_residuals']['q_threshold']:.2e}")

# Plot 4: Mahalanobis Distance
print("\n  Plot 4: Mahalanobis Distance")
print("  - X data: Sample indices")
print("  - Y data: Distances =", report['mahalanobis']['distances'].shape)
print("  - Threshold line:", f"{report['mahalanobis']['threshold']:.2f}")

# Plot 5: Y Value Distribution
print("\n  Plot 5: Y Value Distribution")
print("  - Histogram data: Y values")
print("  - Box plot data: Y values")
print("  - Mark outliers from z-score and range checks")

# ============================================================================
# Step 4: Display outlier summary table in GUI
# ============================================================================

print("\n[4] Outlier Summary Table (display in Treeview/Table widget):")
print("-" * 70)

# Show first few rows
df = report['outlier_summary']
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head().to_string())

# Show high confidence outliers
if len(report['high_confidence_outliers']) > 0:
    print("\n\nHIGH CONFIDENCE OUTLIERS (recommend review):")
    print(report['high_confidence_outliers'][
        ['Sample_Index', 'Y_Value', 'Total_Flags',
         'T2_Outlier', 'Q_Outlier', 'Maha_Outlier', 'Y_Outlier']
    ].to_string())

# ============================================================================
# Step 5: User selects samples to exclude
# ============================================================================

print("\n[5] Sample Selection for Exclusion:")
print("-" * 70)

# In GUI, user can:
# - Check "Auto-select high confidence outliers" checkbox
# - Manually check/uncheck individual samples in table
# - Click "Mark for Exclusion" button

# For this example, auto-select high confidence outliers
samples_to_exclude = report['high_confidence_outliers']['Sample_Index'].tolist()

print(f"  Selected for exclusion: {samples_to_exclude}")

# ============================================================================
# Step 6: Create filtered dataset
# ============================================================================

print("\n[6] Creating Filtered Dataset:")
print("-" * 70)

# Create boolean mask
keep_mask = np.ones(len(X), dtype=bool)
keep_mask[samples_to_exclude] = False

X_filtered = X[keep_mask]
y_filtered = y[keep_mask]

print(f"  Original dataset: {X.shape[0]} samples")
print(f"  Excluded: {len(samples_to_exclude)} samples")
print(f"  Filtered dataset: {X_filtered.shape[0]} samples")

# ============================================================================
# Step 7: Export exclusion report (optional)
# ============================================================================

print("\n[7] Export Exclusion Report:")
print("-" * 70)

# Create exclusion log
exclusion_log = report['outlier_summary'].iloc[samples_to_exclude].copy()

# Add reason column (in GUI, users would fill this in)
exclusion_log['Exclusion_Reason'] = 'Multiple outlier detection methods flagged'
exclusion_log['Timestamp'] = pd.Timestamp.now()
exclusion_log['User'] = 'analyst_name'  # From GUI session

print(f"  Exclusion log created with {len(exclusion_log)} entries")
print("\n  Log preview:")
print(exclusion_log[['Sample_Index', 'Y_Value', 'Total_Flags', 'Exclusion_Reason']].to_string())

# Save to CSV (in GUI, user clicks "Export Report" button)
# exclusion_log.to_csv('outlier_exclusion_log.csv', index=False)
print("\n  Ready to export to CSV")

# ============================================================================
# Step 8: Proceed to analysis with filtered data
# ============================================================================

print("\n[8] Ready for Main Analysis:")
print("-" * 70)

print(f"""
  Filtered data prepared:
    • X_filtered: {X_filtered.shape}
    • y_filtered: {y_filtered.shape}
    • Original data preserved for reference
    • Exclusion log created

  User can now proceed to "Analysis Configuration" tab
  with clean, quality-controlled data.
""")

# ============================================================================
# Bonus: Accessing individual method results
# ============================================================================

print("\n[BONUS] Accessing Individual Method Results:")
print("-" * 70)

print("\nPCA Results:")
print(f"  - Scores shape: {report['pca']['scores'].shape}")
print(f"  - Loadings shape: {report['pca']['loadings'].shape}")
print(f"  - Variance explained: {report['pca']['variance_explained']}")
print(f"  - T² outliers: {report['pca']['outlier_indices']}")

print("\nQ-Residuals Results:")
print(f"  - Q-residuals: {report['q_residuals']['q_residuals'][:5]} ...")
print(f"  - Threshold: {report['q_residuals']['q_threshold']:.2e}")
print(f"  - Outliers: {report['q_residuals']['outlier_indices']}")

print("\nMahalanobis Results:")
print(f"  - Distances: {report['mahalanobis']['distances'][:5]} ...")
print(f"  - Median: {report['mahalanobis']['median']:.2f}")
print(f"  - MAD: {report['mahalanobis']['mad']:.2f}")
print(f"  - Outliers: {report['mahalanobis']['outlier_indices']}")

print("\nY Consistency Results:")
print(f"  - Mean: {report['y_consistency']['mean']:.2f}")
print(f"  - Std: {report['y_consistency']['std']:.2f}")
print(f"  - Z-score outliers: {report['y_consistency']['outlier_indices']}")

print("\n" + "=" * 70)
print("EXAMPLE COMPLETE - Module ready for GUI integration!")
print("=" * 70)
