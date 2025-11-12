"""
Test to demonstrate the ranking bug.

The issue: pandas .rank() with default ascending=True assigns:
- LOWER ranks to LOWER values
- HIGHER ranks to HIGHER values

But our CompositeScore is designed with "lower is better".
So a model with CompositeScore=-2.0 (BEST) should get Rank=1,
but actually gets a low rank number because -2.0 is a low value.

Expected: Best model (lowest CompositeScore) → Rank 1
Actual: Best model (lowest CompositeScore) → Rank 1 ✓ (works correctly!)

Wait... let me verify this more carefully.
"""

import pandas as pd
import numpy as np

# Simulate 4 models with different R² values and CompositeScores
data = {
    'Model': ['ModelA', 'ModelB', 'ModelC', 'ModelD'],
    'R2': [0.95, 0.85, 0.75, 0.65],  # ModelA is best (highest R²)
    'RMSE': [0.10, 0.20, 0.30, 0.40],  # ModelA is best (lowest RMSE)
}

df = pd.DataFrame(data)

# Compute z-scores (like in scoring.py)
z_rmse = (df["RMSE"] - df["RMSE"].mean()) / df["RMSE"].std()
z_r2 = (df["R2"] - df["R2"].mean()) / df["R2"].std()

# Performance score (lower is better)
# For best model (ModelA): z_rmse is very negative (good), z_r2 is very positive (good)
# So: performance_score = 0.5 * (negative) - 0.5 * (positive) = very negative
df["performance_score"] = 0.5 * z_rmse - 0.5 * z_r2
df["CompositeScore"] = df["performance_score"]

print("=" * 80)
print("RANKING BUG TEST")
print("=" * 80)
print("\nModel Performance (ModelA is BEST):")
print(df[['Model', 'R2', 'RMSE', 'CompositeScore']])

print("\n" + "=" * 80)
print("TESTING PANDAS .rank() BEHAVIOR")
print("=" * 80)

# Test 1: Default behavior (ascending=True)
df["Rank_default"] = df["CompositeScore"].rank(method="min")
print("\n1. With ascending=True (DEFAULT):")
print(df[['Model', 'CompositeScore', 'Rank_default']])
print("\nInterpretation: Lower CompositeScore → LOWER rank number")
print("Expected for 'lower is better': ModelA (best) should be Rank 1")
print(f"Result: ModelA is Rank {df.loc[df['Model']=='ModelA', 'Rank_default'].values[0]}")
if df.loc[df['Model']=='ModelA', 'Rank_default'].values[0] == 1:
    print("✓ CORRECT: Best model gets Rank 1")
else:
    print("✗ BUG: Best model does NOT get Rank 1")

# Test 2: With ascending=False
df["Rank_desc"] = df["CompositeScore"].rank(method="min", ascending=False)
print("\n2. With ascending=False:")
print(df[['Model', 'CompositeScore', 'Rank_desc']])
print("\nInterpretation: Lower CompositeScore → HIGHER rank number")
print(f"Result: ModelA is Rank {df.loc[df['Model']=='ModelA', 'Rank_desc'].values[0]}")
if df.loc[df['Model']=='ModelA', 'Rank_desc'].values[0] == 1:
    print("✓ CORRECT: Best model gets Rank 1")
else:
    print("✗ BUG: Best model does NOT get Rank 1")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\nFor 'lower is better' scores:")
print("- ascending=True (default) assigns Rank 1 to LOWEST value ✓ CORRECT")
print("- ascending=False assigns Rank 1 to HIGHEST value ✗ WRONG")
print("\nCurrent code uses: df['Rank'] = df['CompositeScore'].rank(method='min')")
print("This is CORRECT for 'lower is better' scoring.")
print("\nIf ranking is broken, the bug is elsewhere!")
