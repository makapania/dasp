#!/usr/bin/env python3
"""
Diagnostic Test: Tab 7 Model Development Bug

This script runs a minimal analysis with BoneCollagen data and traces
the alpha extraction issue through diagnostic logging.

Usage:
    python test_tab7_diagnostics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy
from spectral_predict.search import run_search

print("="*80)
print("TAB 7 DIAGNOSTIC TEST: Alpha Extraction Bug")
print("="*80)
print()

# ===================================================================
# STEP 1: Load BoneCollagen Data
# ===================================================================
print("[STEP 1] Loading BoneCollagen data...")

example_dir = Path(__file__).parent / "example"
if not example_dir.exists():
    print(f"❌ ERROR: Example directory not found: {example_dir}")
    sys.exit(1)

try:
    X = read_asd_dir(str(example_dir))
    print(f"  ✓ Loaded {len(X)} spectra with {len(X.columns)} wavelengths")
except Exception as e:
    print(f"❌ ERROR loading ASD files: {e}")
    sys.exit(1)

try:
    ref = read_reference_csv(str(example_dir / "BoneCollagen.csv"), "File Number")
    print(f"  ✓ Loaded reference data with {len(ref)} samples")
except Exception as e:
    print(f"❌ ERROR loading reference CSV: {e}")
    sys.exit(1)

try:
    X_aligned, y = align_xy(X, ref, "File Number", "%Collagen")
    print(f"  ✓ Aligned data: {len(X_aligned)} samples matched")
except Exception as e:
    print(f"❌ ERROR aligning X and y: {e}")
    sys.exit(1)

print(f"\n  Data summary:")
print(f"    X shape: {X_aligned.shape}")
print(f"    y shape: {y.shape}")
print(f"    y range: {y.min():.2f} to {y.max():.2f}")
print()

# ===================================================================
# STEP 2: Run Analysis with Lasso
# ===================================================================
print("[STEP 2] Running analysis with Lasso model...")
print("  Model: Lasso")
print("  Preprocessing: snv_sg2 (SNV + 2nd derivative)")
print("  Window: 17")
print("  Subset: top 50 wavelengths (importance)")
print("  CV folds: 5")
print()

try:
    results_df = run_search(
        X_aligned, y,
        task_type='regression',
        folds=5,
        models_to_test=['Lasso'],
        preprocessing_methods={'snv_sg2': True},  # SNV + 2nd derivative
        window_sizes=[17],
        enable_variable_subsets=True,
        variable_counts=[50],
        variable_selection_methods=['importance'],
        lambda_penalty=0.15,
        max_n_components=20,
        max_iter=100
    )
    print(f"  ✓ Analysis complete: {len(results_df)} results")
except Exception as e:
    print(f"❌ ERROR running analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===================================================================
# STEP 3: Examine Top Result
# ===================================================================
print("\n[STEP 3] Examining top result...")

if len(results_df) == 0:
    print("❌ ERROR: No results returned from analysis")
    sys.exit(1)

# Get top model (rank 1)
top_model = results_df.iloc[0]

print(f"\nTop Model Configuration:")
print(f"  Rank: {top_model['Rank']}")
print(f"  Model: {top_model['Model']}")
print(f"  R²: {top_model['R2']:.4f}")
print(f"  RMSE: {top_model['RMSE']:.4f}")
print(f"  Preprocessing: {top_model['Preprocess']}")
print(f"  Deriv: {top_model.get('Deriv', 'N/A')}")
print(f"  Window: {top_model['Window']}")
print(f"  n_vars: {top_model['n_vars']}")
print(f"  SubsetTag: {top_model['SubsetTag']}")

# ===================================================================
# CRITICAL: Check Alpha field
# ===================================================================
print(f"\n🔍 CRITICAL: Alpha Field Examination")
print(f"="*80)

alpha_fields_to_check = ['Alpha', 'alpha', 'Params']
for field in alpha_fields_to_check:
    if field in top_model.index:
        value = top_model[field]
        value_type = type(value).__name__
        if pd.isna(value):
            value_display = "NaN"
        elif value is None:
            value_display = "None"
        else:
            value_display = str(value)
        print(f"  {field:15s} = {value_display:30s} (type: {value_type})")
    else:
        print(f"  {field:15s} = FIELD NOT FOUND")

print(f"="*80)

# ===================================================================
# STEP 4: Simulate Tab 7 Loading
# ===================================================================
print(f"\n[STEP 4] Simulating Tab 7 configuration loading...")

config = top_model.to_dict()

print(f"\n🔍 Full config dict (first 20 keys):")
for i, (key, value) in enumerate(sorted(config.items())[:20]):
    value_type = type(value).__name__
    if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
        value_display = "NaN"
    elif value is None:
        value_display = "None"
    else:
        value_display = str(value)[:50]
    print(f"  {key:20s} = {value_display:30s} (type: {value_type})")

# Simulate _get_config_value function
def _get_config_value(keys):
    for k in keys:
        if k in config:
            v = config.get(k)
            if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v) != 'N/A':
                return v
    return None

# Test alpha extraction (same as GUI code)
print(f"\n🔍 Testing alpha extraction (same logic as GUI):")
alpha_val = _get_config_value(['Alpha', 'alpha'])
print(f"  alpha_val = {alpha_val}")
print(f"  type = {type(alpha_val)}")
print(f"  is None? {alpha_val is None}")
if alpha_val is not None:
    print(f"  is NaN? {pd.isna(alpha_val) if isinstance(alpha_val, (float, np.floating)) else False}")
    print(f"  bool(alpha_val)? {bool(alpha_val)}")

# ===================================================================
# STEP 5: Diagnosis
# ===================================================================
print(f"\n[STEP 5] DIAGNOSIS")
print(f"="*80)

if alpha_val is None:
    print("❌ BUG IDENTIFIED: alpha_val is None!")
    print("   This means the 'Alpha' field is either:")
    print("   1. Not present in config")
    print("   2. Has value None")
    print("   3. Has value NaN")
    print("   4. Has value 'N/A'")
    print()
    print("   Impact: Tab 7 will use default alpha=1.0 instead of optimized value")
    print("   Result: Wrong R² predictions!")
elif alpha_val == 0:
    print("⚠️  WARNING: alpha_val is 0 (falsy value)")
    print("   This will fail the 'if alpha_val:' check in GUI")
    print("   Impact: Widget will not be set, default 1.0 will be used")
else:
    print(f"✅ Alpha extracted successfully: {alpha_val}")
    print(f"   This value should be loaded into Tab 7 widget")

print(f"="*80)

# ===================================================================
# STEP 6: Recommendations
# ===================================================================
print(f"\n[STEP 6] RECOMMENDATIONS")
print(f"="*80)

if alpha_val is None or alpha_val == 0:
    print("1. Check src/spectral_predict/search.py:")
    print("   - Find where result['Alpha'] is set (around line 746)")
    print("   - Verify it's storing the optimized alpha value")
    print()
    print("2. Check spectral_predict_gui_optimized.py:")
    print("   - Line 3346: Change 'if alpha_val:' to 'if alpha_val is not None:'")
    print("   - This handles alpha=0 case (though rare for Lasso)")
    print()
    print("3. Add FAIL LOUD validation:")
    print("   - Raise error if alpha_val is None for Ridge/Lasso models")
    print("   - Don't silently fall back to default 1.0")
else:
    print("✅ Alpha extraction appears to work correctly in this test")
    print("   The bug may be in a different scenario or data-dependent")
    print("   Run the GUI to see full diagnostic logs")

print(f"="*80)
print()
print("Test complete! Run the GUI to see full diagnostic logs in action.")
print()
