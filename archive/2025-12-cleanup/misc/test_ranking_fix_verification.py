"""
Verification script for ranking bug fix.

This demonstrates that the ranking system now correctly ranks high-performance models
above low-performance models, fixing the issue where R²=0.45 models ranked #1
while R²>0.95 models ranked #1200+.
"""

import pandas as pd
import numpy as np
from spectral_predict.scoring import compute_composite_score

# Create test scenario matching user's bug report
print("\n" + "="*80)
print("RANKING FIX VERIFICATION")
print("="*80)
print("\nScenario: PLS model with R²=0.45 vs other models with R²>0.95")
print("Before fix: R²=0.45 ranked #1, R²>0.95 ranked #1200+")
print("After fix: Should rank by performance correctly\n")

# Create test data
n_models = 50
models_data = []

# Add one terrible PLS model (the bug case)
models_data.append({
    "Model": "PLS",
    "R2": 0.45,
    "RMSE": 0.35,
    "n_vars": 10,
    "full_vars": 2151,
    "LVs": 3,
    "Params": "{}",
    "Preprocess": "raw",
    "Deriv": 0,
    "Window": 0,
    "Poly": 0,
    "SubsetTag": "full",
    "top_vars": None
})

# Add many excellent models
for i in range(n_models - 1):
    models_data.append({
        "Model": np.random.choice(["PLS", "Ridge", "RandomForest", "XGBoost"]),
        "R2": np.random.uniform(0.92, 0.98),
        "RMSE": np.random.uniform(0.05, 0.15),
        "n_vars": np.random.randint(100, 2000),
        "full_vars": 2151,
        "LVs": np.random.randint(0, 20) if np.random.rand() > 0.5 else 0,
        "Params": "{}",
        "Preprocess": np.random.choice(["raw", "snv", "deriv"]),
        "Deriv": np.random.randint(0, 3),
        "Window": 0,
        "Poly": 0,
        "SubsetTag": "full",
        "top_vars": None
    })

df = pd.DataFrame(models_data)

# Test with user's settings (penalty=2)
print("Testing with penalty=2 (user's setting):")
print("-"*80)

df_ranked = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2, verbose=True)

# Find the terrible model
terrible_model = df_ranked[df_ranked["R2"] < 0.50].iloc[0]
excellent_models = df_ranked[df_ranked["R2"] > 0.95]

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nTerrible model (R²={terrible_model['R2']:.3f}, RMSE={terrible_model['RMSE']:.3f}):")
print(f"  Rank: #{terrible_model['Rank']}")
print(f"  CompositeScore: {terrible_model['CompositeScore']:.4f}")

print(f"\nExcellent models (R²>0.95): {len(excellent_models)} total")
print(f"  Best rank: #{excellent_models['Rank'].min()}")
print(f"  Worst rank: #{excellent_models['Rank'].max()}")
print(f"  Average rank: #{excellent_models['Rank'].mean():.1f}")

if terrible_model['Rank'] > excellent_models['Rank'].mean():
    print("\n✓ PASS: Terrible model ranks WORSE than excellent models (bug is fixed!)")
else:
    print("\n✗ FAIL: Terrible model still ranks better than it should")

print("\n" + "="*80)
