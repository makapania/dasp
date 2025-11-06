"""
Test script to diagnose model save/load/predict issue.
Compare models saved from Python vs Julia results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Check if a .dasp file path was provided
if len(sys.argv) < 2:
    print("Usage: python test_model_save_load.py <path_to_model.dasp>")
    print("\nThis script will:")
    print("1. Load the model")
    print("2. Show all metadata")
    print("3. Create test data")
    print("4. Try making predictions")
    print("5. Show what went wrong (if anything)")
    sys.exit(1)

model_path = sys.argv[1]

print("="*70)
print("Model Save/Load/Predict Diagnostic")
print("="*70)
print(f"\nModel file: {model_path}")
print()

try:
    from spectral_predict.model_io import load_model, predict_with_model

    # Load the model
    print("Step 1: Loading model...")
    model_dict = load_model(model_path)
    print("  SUCCESS: Model loaded")
    print()

    # Show metadata
    print("Step 2: Checking metadata...")
    metadata = model_dict['metadata']

    critical_fields = ['wavelengths', 'n_vars', 'model_name', 'preprocessing', 'task_type']
    print("  Critical fields:")
    for field in critical_fields:
        value = metadata.get(field, "MISSING!")
        if field == 'wavelengths' and isinstance(value, list):
            print(f"    {field}: {len(value)} wavelengths (first 5: {value[:5] if len(value) >= 5 else value})")
        else:
            print(f"    {field}: {value}")

    print()
    print("  Full metadata:")
    for key, value in sorted(metadata.items()):
        if key == 'wavelengths':
            continue  # Already shown above
        if isinstance(value, (list, np.ndarray)) and len(str(value)) > 100:
            print(f"    {key}: <{len(value)} items>")
        else:
            print(f"    {key}: {value}")

    print()

    # Check if model and preprocessor are present
    print("Step 3: Checking model components...")
    print(f"  Model object: {type(model_dict['model']).__name__}")
    print(f"  Preprocessor: {type(model_dict['preprocessor']).__name__ if model_dict['preprocessor'] else 'None (raw data)'}")
    print()

    # Create test data matching the wavelengths
    print("Step 4: Creating test data...")
    wavelengths = metadata['wavelengths']
    n_test_samples = 10

    # Create DataFrame with correct wavelength columns
    test_data = pd.DataFrame(
        np.random.rand(n_test_samples, len(wavelengths)),
        columns=[str(wl) for wl in wavelengths]
    )
    print(f"  Created {n_test_samples} test samples with {len(wavelengths)} wavelengths")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print()

    # Try making predictions
    print("Step 5: Making predictions...")
    predictions = predict_with_model(model_dict, test_data, validate_wavelengths=True)

    print(f"  SUCCESS: Predictions generated!")
    print(f"  Shape: {predictions.shape}")
    print(f"  Predictions: {predictions}")
    print()

    # Check for issues
    print("Step 6: Checking for issues...")
    issues = []

    if len(predictions) == 0:
        issues.append("PREDICTIONS ARE EMPTY!")

    if np.all(np.isnan(predictions)):
        issues.append("ALL PREDICTIONS ARE NaN!")

    if np.any(np.isnan(predictions)):
        n_nan = np.sum(np.isnan(predictions))
        issues.append(f"{n_nan} out of {len(predictions)} predictions are NaN")

    if np.all(predictions == 0):
        issues.append("ALL PREDICTIONS ARE ZERO!")

    if len(issues) == 0:
        print("  No issues found - model appears to be working correctly!")
    else:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")

    print()
    print("="*70)
    print("Diagnostic Complete")
    print("="*70)

except Exception as e:
    print()
    print("="*70)
    print("ERROR OCCURRED")
    print("="*70)
    print()
    import traceback
    traceback.print_exc()
    print()
    print(f"Error: {e}")
