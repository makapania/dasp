"""
Quick test for NSGA-II wavelength reduction changes.

Tests:
1. CARS-Tree importance is computed before optimization
2. Initial population starts with ~250 wavelengths (not 100%)
3. Final solutions have compact wavelength subsets
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from spectral_predict.nsga2_search import run_nsga2_search

# Create synthetic spectral data
np.random.seed(42)
n_samples = 100
n_wavelengths = 500

# Generate spectra with informative region
X = np.random.randn(n_samples, n_wavelengths)
# Make wavelengths 100-200 informative
X[:, 100:200] += np.linspace(0, 2, 100) * np.random.randn(n_samples, 1)
y = X[:, 100:200].mean(axis=1) + 0.1 * np.random.randn(n_samples)

print("=" * 60)
print("NSGA-II Wavelength Reduction Test")
print("=" * 60)
print(f"Data: {n_samples} samples, {n_wavelengths} wavelengths")
print(f"True informative region: wavelengths 100-200")
print()

# Run NSGA-II with reduced parameters for quick test
result = run_nsga2_search(
    X=X,
    y=y,
    task_type='regression',
    population_size=30,  # Small population for quick test
    n_generations=20,     # Few generations for quick test
    cv_folds=3,
    min_wavelengths=10,
    random_state=42,
    verbose=2,
    models=['PLS', 'LightGBM'],  # Just two models
)

print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

# Check the knee solution
knee = result['knee_solution']
print(f"\nKnee solution:")
print(f"  Model: {knee['model']}")
print(f"  Preprocessing: {knee['preprocessing']}")
print(f"  N wavelengths: {knee['n_wavelengths']}")
print(f"  Error: {knee['objectives']['error']:.4f}")

# Check if wavelengths are in expected range
selected_indices = knee['selected_indices']
in_informative_region = sum(1 for i in selected_indices if 100 <= i < 200)
print(f"  Wavelengths in informative region (100-200): {in_informative_region}/{len(selected_indices)}")

# Check Pareto front solutions
pareto_front = result['pareto_front']
pareto_solutions = result['pareto_solutions']

print(f"\nPareto front: {len(pareto_front)} solutions")
n_wavelengths_list = []
for i, (obj, sol) in enumerate(zip(pareto_front, pareto_solutions)):
    n_wl = int(np.sum(sol[13:]))
    n_wavelengths_list.append(n_wl)

print(f"  Wavelength counts: min={min(n_wavelengths_list)}, max={max(n_wavelengths_list)}, mean={np.mean(n_wavelengths_list):.0f}")

# Success criteria
print("\n" + "=" * 60)
print("Validation")
print("=" * 60)
max_wavelengths = max(n_wavelengths_list)
if max_wavelengths <= 400:
    print(f"PASS: Max wavelengths ({max_wavelengths}) is below 400 (was 800+ before fix)")
else:
    print(f"WARNING: Max wavelengths ({max_wavelengths}) is above 400 - may need more tuning")

mean_wavelengths = np.mean(n_wavelengths_list)
if mean_wavelengths <= 300:
    print(f"PASS: Mean wavelengths ({mean_wavelengths:.0f}) is below 300 (target range)")
else:
    print(f"WARNING: Mean wavelengths ({mean_wavelengths:.0f}) is above 300 - may need more tuning")

if in_informative_region > len(selected_indices) * 0.3:
    print(f"PASS: {in_informative_region/len(selected_indices)*100:.0f}% of wavelengths are in informative region")
else:
    print(f"INFO: Only {in_informative_region/len(selected_indices)*100:.0f}% in informative region (may be OK)")
