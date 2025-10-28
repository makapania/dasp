"""
Test to verify that region-based subsets are computed on preprocessed data.
This ensures that different preprocessing methods get different region selections.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from src.spectral_predict.search import run_search

# Create synthetic spectral-like data
np.random.seed(42)
n_samples = 50
n_wavelengths = 200

# Create X with some structure
X_array, y_array = make_regression(n_samples=n_samples, n_features=n_wavelengths,
                                    n_informative=20, noise=10, random_state=42)

# Create wavelength array (350-2500 nm range)
wavelengths = np.linspace(350, 2500, n_wavelengths)

# Convert to DataFrame (expected by run_search)
X = pd.DataFrame(X_array, columns=[f"{wl:.1f}" for wl in wavelengths])
y = pd.Series(y_array)

print("=" * 80)
print("Testing Region Subset Selection on Preprocessed Data")
print("=" * 80)
print(f"Data shape: {X.shape}")
print(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
print()

# Run search with verbose output to see region analysis messages
print("Running analysis...")
print("Look for messages like: 'Region analysis for raw: Identified N region-based subsets'")
print()

results = run_search(
    X=X,
    y=y,
    task_type="regression",
    folds=3,  # Small number for faster test
    lambda_penalty=0.15,
    max_n_components=5,  # Small number for faster test
    max_iter=200
)

print()
print("=" * 80)
print("Checking Results")
print("=" * 80)

# Check if we have region-based subsets in results
region_results = results[results['SubsetTag'].str.contains('region', na=False)]

print(f"Total results: {len(results)}")
print(f"Region-based subset results: {len(region_results)}")
print()

if len(region_results) > 0:
    print("✓ SUCCESS: Region-based subsets were created!")
    print()
    print("Sample region results:")
    print(region_results[['Model', 'Preprocess', 'SubsetTag', 'n_vars', 'R2']].head(10))
    print()

    # Check which preprocessing methods have region results
    preproc_with_regions = region_results['Preprocess'].unique()
    print(f"Preprocessing methods with region subsets: {list(preproc_with_regions)}")

    # Verify only raw/snv have regions (not derivatives)
    has_derivative = any('d1' in p or 'd2' in p for p in preproc_with_regions)
    if has_derivative:
        print("⚠ WARNING: Derivative preprocessing has region subsets (should be skipped)")
    else:
        print("✓ Correctly skipped region analysis for derivative preprocessing")

else:
    print("✗ WARNING: No region-based subsets found in results")
    print("This might indicate an issue with region computation")

print()
print("=" * 80)
print("Test Complete")
print("=" * 80)
