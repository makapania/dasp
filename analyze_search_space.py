"""Analyze the full search space to understand why it's still slow."""
import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("SEARCH SPACE ANALYSIS")
print("=" * 70)

# Count preprocessing configs
preprocess_configs = [
    {"name": "raw", "deriv": None},
    {"name": "snv", "deriv": None},
]

# Add derivative configs
for deriv in [1, 2]:
    for window in [7, 19]:
        preprocess_configs.append({"name": "deriv", "deriv": deriv})
        preprocess_configs.append({"name": "snv_deriv", "deriv": deriv})
        preprocess_configs.append({"name": "deriv_snv", "deriv": deriv})

print(f"\n1. Preprocessing configurations: {len(preprocess_configs)}")
print("   Breakdown:")
print("   - raw: 1")
print("   - snv: 1")
print("   - Derivatives (1st, 2nd) x Windows (7, 19) x Methods (3): 2 x 2 x 3 = 12")
print("   TOTAL: 14 preprocessing methods")

# Count model configs
from spectral_predict.models import get_model_grids

grids = get_model_grids('regression', n_features=500, max_iter=100)

print(f"\n2. Model configurations (per preprocessing method):")
for model_name, configs in grids.items():
    print(f"   - {model_name}: {len(configs)} configs")

nb_configs = len(grids.get('NeuralBoosted', []))

print(f"\n3. TOTAL search space for NeuralBoosted:")
print(f"   {nb_configs} model configs × {len(preprocess_configs)} preprocess = {nb_configs * len(preprocess_configs)} total configs")

print(f"\n4. Time estimates (with current optimizations):")
print(f"   Assuming ~5-10 seconds per config with 5-fold CV:")
print(f"   - Optimistic (5s/config): {nb_configs * len(preprocess_configs) * 5 / 60:.1f} minutes")
print(f"   - Realistic (10s/config): {nb_configs * len(preprocess_configs) * 10 / 60:.1f} minutes")

print(f"\n5. BEFORE optimizations (24 base configs):")
old_configs = 24
print(f"   {old_configs} model configs × {len(preprocess_configs)} preprocess = {old_configs * len(preprocess_configs)} total")
print(f"   Time estimate (20s/config with max_iter=500): {old_configs * len(preprocess_configs) * 20 / 60:.1f} minutes")

print(f"\n6. SPEEDUP ACHIEVED:")
old_time = old_configs * len(preprocess_configs) * 20
new_time = nb_configs * len(preprocess_configs) * 10
speedup = old_time / new_time
print(f"   {speedup:.1f}x faster than before!")
print(f"   ({old_time/60:.1f} min -> {new_time/60:.1f} min)")

print("\n" + "=" * 70)
print("FURTHER OPTIMIZATION RECOMMENDATIONS:")
print("=" * 70)
print("\n1. REDUCE PREPROCESSING CONFIGS (biggest impact):")
print("   - Current: 14 preprocessing methods")
print("   - Recommended: 6-8 preprocessing methods")
print("   - Drop one derivative window: 14 -> 8 configs")
print("   - Additional speedup: ~1.75x")
print("")
print("2. REDUCE DERIVATIVE OPTIONS:")
print("   Current: 2 derivatives × 2 windows × 3 combos = 12 configs")
print("   Option A: Use only window=11 (middle ground)")
print("      Result: 2 × 1 × 3 = 6 derivative configs -> 8 total preprocess")
print("   Option B: Drop one SNV combo")
print("      Result: 2 × 2 × 2 = 8 derivative configs -> 10 total preprocess")
print("")
print("3. IF STILL TOO SLOW - MOST AGGRESSIVE:")
print("   - raw: 1")
print("   - snv: 1")
print("   - 1st deriv (window=11): 1")
print("   - 2nd deriv (window=11): 1")
print("   TOTAL: 4 preprocessing methods")
print(f"   Result: {nb_configs} × 4 = {nb_configs * 4} configs = ~{nb_configs * 4 * 10 / 60:.0f} minutes")
print("=" * 70)
