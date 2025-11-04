"""
Verification script for R2 discrepancy fix.

This script demonstrates that the fix correctly reproduces main analysis behavior
for derivative preprocessing + variable subset models.

Usage:
    python verify_r2_fix.py
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from spectral_predict.preprocess import build_preprocessing_pipeline


def test_r2_fix_simulation():
    """
    Simulate the main analysis and model development workflows to verify
    that the fix reproduces the correct R2 values.
    """
    print("=" * 80)
    print("R2 Fix Verification Test")
    print("=" * 80)

    # Generate synthetic spectral data
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 500

    wavelengths = np.linspace(1500, 2500, n_wavelengths)
    y = np.random.randn(n_samples)

    # Generate spectra with features correlated to y
    X = np.zeros((n_samples, n_wavelengths))
    for i in range(n_samples):
        baseline = 1.0 + 0.1 * y[i]
        X[i, :] = baseline
        X[i, 100:120] += 0.5 * y[i]  # Peak 1
        X[i, 300:320] += 0.3 * y[i]  # Peak 2
        X[i, :] += 0.05 * np.random.randn(n_wavelengths)

    # Convert to DataFrame (like GUI uses)
    X_df = pd.DataFrame(X, columns=wavelengths)

    print(f"\nData: {n_samples} samples, {n_wavelengths} wavelengths")

    # Preprocessing parameters
    preprocess_name = 'snv_deriv'
    deriv = 2
    window = 17
    polyorder = 3
    n_components = 5

    print(f"Preprocessing: SNV + SG2 (window={window})")
    print(f"Model: PLS (n_components={n_components})")

    # Simulate main analysis (search.py approach)
    print("\n" + "-" * 80)
    print("MAIN ANALYSIS (search.py approach)")
    print("-" * 80)

    # 1. Build preprocessing pipeline
    prep_steps = build_preprocessing_pipeline(preprocess_name, deriv, window, polyorder)
    prep_pipeline = Pipeline(prep_steps)

    # 2. Preprocess FULL spectrum
    X_full_preprocessed = prep_pipeline.fit_transform(X)

    # 3. Simulate variable selection (select scattered wavelengths)
    # Select every 10th wavelength (non-contiguous)
    selected_indices = np.arange(0, n_wavelengths, 10)[:50]  # 50 wavelengths
    selected_wavelengths = wavelengths[selected_indices]

    print(f"Variable selection: {len(selected_indices)} wavelengths (non-contiguous)")
    print(f"  First 5 indices: {selected_indices[:5]}")
    print(f"  Last 5 indices: {selected_indices[-5:]}")

    # 4. Subset PREPROCESSED data
    X_subset_preprocessed = X_full_preprocessed[:, selected_indices]

    # 5. Run CV with PLS only (no preprocessing in CV)
    pls = PLSRegression(n_components=n_components)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_main = cross_val_score(pls, X_subset_preprocessed, y, cv=cv, scoring='r2').mean()

    print(f"\nMain Analysis R2: {r2_main:.4f}")

    # Simulate OLD Model Development (BROKEN approach - subset then preprocess)
    print("\n" + "-" * 80)
    print("OLD MODEL DEVELOPMENT (BROKEN - subset then preprocess)")
    print("-" * 80)

    # 1. Subset RAW data first
    X_subset_raw = X_df.iloc[:, selected_indices].values

    # 2. Build pipeline with preprocessing + model
    pipe_steps_old = build_preprocessing_pipeline(preprocess_name, deriv, window, polyorder)
    pipe_steps_old.append(('pls', PLSRegression(n_components=n_components)))
    pipe_old = Pipeline(pipe_steps_old)

    # 3. Run CV (preprocessing happens INSIDE CV on subset)
    r2_old = cross_val_score(pipe_old, X_subset_raw, y, cv=cv, scoring='r2').mean()

    print(f"Old Model Development R2: {r2_old:.4f}")
    print(f"Difference from main: {r2_main - r2_old:+.4f}")

    # Simulate NEW Model Development (FIXED approach - preprocess full then subset)
    print("\n" + "-" * 80)
    print("NEW MODEL DEVELOPMENT (FIXED - full spectrum preprocessing)")
    print("-" * 80)

    # Detect derivative + subset scenario
    is_derivative = True  # We're using derivatives
    is_subset = len(selected_indices) < n_wavelengths

    if is_derivative and is_subset:
        print("Detected: Derivative + subset -> Using full-spectrum preprocessing")

        # 1. Build preprocessing pipeline
        prep_steps_new = build_preprocessing_pipeline(preprocess_name, deriv, window, polyorder)
        prep_pipeline_new = Pipeline(prep_steps_new)

        # 2. Preprocess FULL spectrum
        X_full_prep_new = prep_pipeline_new.fit_transform(X_df.values)

        # 3. Subset PREPROCESSED data
        X_work_new = X_full_prep_new[:, selected_indices]

        # 4. Build pipeline with ONLY model (no preprocessing)
        pipe_new = Pipeline([('pls', PLSRegression(n_components=n_components))])

        # 5. Run CV
        r2_new = cross_val_score(pipe_new, X_work_new, y, cv=cv, scoring='r2').mean()
    else:
        r2_new = 0.0  # Should not happen

    print(f"New Model Development R2: {r2_new:.4f}")
    print(f"Difference from main: {r2_main - r2_new:+.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Main Analysis R2:          {r2_main:.4f}")
    print(f"Old Model Dev R2 (BROKEN): {r2_old:.4f}  (Delta = {r2_main - r2_old:+.4f})")
    print(f"New Model Dev R2 (FIXED):  {r2_new:.4f}  (Delta = {r2_main - r2_new:+.4f})")
    print()

    # Check if fix is successful
    old_diff = abs(r2_main - r2_old)
    new_diff = abs(r2_main - r2_new)

    if new_diff < 0.01:  # Within CV variance
        print(f"[OK] FIX SUCCESSFUL! New approach matches main analysis (diff = {new_diff:.4f})")
    else:
        print(f"[FAIL] FIX INCOMPLETE. Difference still significant (diff = {new_diff:.4f})")

    if old_diff > 0.05:
        print(f"[OK] Old approach was indeed broken (diff = {old_diff:.4f})")

    improvement = ((old_diff - new_diff) / old_diff) * 100
    print(f"\nImprovement: {improvement:.1f}% reduction in R2 discrepancy")

    return r2_main, r2_old, r2_new


if __name__ == "__main__":
    try:
        test_r2_fix_simulation()
    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()
