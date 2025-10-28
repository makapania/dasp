"""Quick test that NeuralBoosted is available for regression."""

from src.spectral_predict.models import get_model_grids

# Test that NeuralBoosted is available for regression
print("Testing NeuralBoosted availability in model grids...")
print()

# Get regression model grids
regression_grids = get_model_grids(task_type="regression", n_features=100, max_n_components=20, max_iter=100)

print("Available models for REGRESSION:")
for model_name in regression_grids.keys():
    n_configs = len(regression_grids[model_name])
    print(f"  - {model_name}: {n_configs} configurations")
print()

# Check if NeuralBoosted is in regression grids
if "NeuralBoosted" in regression_grids:
    print("✓ SUCCESS: NeuralBoosted is available for regression tasks!")
    print(f"  Number of configurations: {len(regression_grids['NeuralBoosted'])}")
    print()
    print("  Sample configuration:")
    model, params = regression_grids['NeuralBoosted'][0]
    print(f"    Model: {model}")
    print(f"    Params: {params}")
else:
    print("✗ FAIL: NeuralBoosted is NOT available for regression tasks!")
    print()

# Also check classification (should work too)
print()
classification_grids = get_model_grids(task_type="classification", n_features=100, max_n_components=20, max_iter=100)

print("Available models for CLASSIFICATION:")
for model_name in classification_grids.keys():
    n_configs = len(classification_grids[model_name])
    print(f"  - {model_name}: {n_configs} configurations")
