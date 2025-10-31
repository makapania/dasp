"""Check that optimizations are actually being used at runtime."""
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from spectral_predict.models import get_model_grids

print("=" * 60)
print("RUNTIME CONFIGURATION CHECK")
print("=" * 60)

# Get grids
grids = get_model_grids('regression', n_features=500, max_iter=100)
nb_configs = grids.get('NeuralBoosted', [])

print(f"\n1. Number of NeuralBoosted configs: {len(nb_configs)}")
print(f"   Expected: 8 configs")
print(f"   Status: {'[OK]' if len(nb_configs) == 8 else '[ISSUE]'}")

print("\n2. Neural Boosted Configurations:")
for i, (model, name) in enumerate(nb_configs[:8]):
    print(f"   {i+1}. n_est={model.n_estimators}, lr={model.learning_rate}, "
          f"hidden={model.hidden_layer_size}, act={model.activation}, "
          f"max_iter={model.max_iter}")

# Check that max_iter is properly set
print("\n3. Checking max_iter values:")
max_iters = [model.max_iter for model, _ in nb_configs]
all_100 = all(m == 100 for m in max_iters)
print(f"   All models have max_iter=100: {all_100}")
print(f"   Status: {'[OK]' if all_100 else '[ISSUE]'}")

# Check the actual weak learner creation to see if tol is correct
print("\n4. Checking tolerance in source code:")
with open('src/spectral_predict/neural_boosted.py', 'r') as f:
    content = f.read()
    if 'tol=5e-4' in content:
        print("   tol=5e-4 found in source: [OK]")
    else:
        print("   tol=5e-4 NOT found in source: [ISSUE]")

print("\n" + "=" * 60)
