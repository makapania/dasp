"""Test that NeuralBoosted results appear in output."""

import numpy as np
import pandas as pd
from src.spectral_predict.search import run_search

# Create synthetic spectral data
np.random.seed(42)
n_samples = 50
n_wavelengths = 100

# Generate spectral data
X_data = np.random.randn(n_samples, n_wavelengths) + 5.0
wavelengths = np.linspace(400, 2500, n_wavelengths)

# Create target with some correlation to spectral data
y_data = 2.0 * X_data[:, 10] + 1.5 * X_data[:, 50] + np.random.randn(n_samples) * 0.5

# Convert to DataFrame
X_df = pd.DataFrame(X_data, columns=[f"{w:.1f}" for w in wavelengths])
y_series = pd.Series(y_data, name="target")

print("Testing NeuralBoosted output fix...")
print(f"Data shape: X={X_df.shape}, y={y_series.shape}")
print()

# Run search with only NeuralBoosted model
print("Running search with NeuralBoosted model only...")
results_df = run_search(
    X_df,
    y_series,
    task_type="regression",
    folds=3,
    lambda_penalty=0.15,
    max_n_components=10,
    max_iter=100,
    models_to_test=["NeuralBoosted"]
)

print()
print(f"Total results generated: {len(results_df)}")
print()

# Check for NeuralBoosted results
neuralboosted_results = results_df[results_df["Model"] == "NeuralBoosted"]
print(f"NeuralBoosted results found: {len(neuralboosted_results)}")

if len(neuralboosted_results) > 0:
    print("\n✓ SUCCESS: NeuralBoosted results are being generated!")
    print("\nSample NeuralBoosted results:")
    print(neuralboosted_results[['Model', 'Preprocess', 'SubsetTag', 'n_vars', 'RMSE', 'R2']].head(10))

    # Check for subset results
    subset_results = neuralboosted_results[neuralboosted_results["SubsetTag"] != "full"]
    print(f"\nNeuralBoosted subset results: {len(subset_results)}")

    if len(subset_results) > 0:
        print("✓ NeuralBoosted wavelength subset models are working!")
        print("\nSample subset results:")
        print(subset_results[['Model', 'SubsetTag', 'n_vars', 'RMSE', 'R2']].head(5))
    else:
        print("✗ WARNING: No NeuralBoosted subset results found")
else:
    print("\n✗ FAIL: No NeuralBoosted results in output!")
    print("\nAll models in results:")
    print(results_df["Model"].value_counts())
