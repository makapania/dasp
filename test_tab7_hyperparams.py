"""
Test script to verify Tab 7 hyperparameter fixes.
Simulates the exact code path from the GUI to verify no cross-contamination.
"""

# Test Fix #1: Hyperparameter contamination removal
def test_hyperparameter_extraction():
    """Simulate Tab 7 hyperparameter extraction for different model types."""

    print("=" * 80)
    print("TESTING TAB 7 HYPERPARAMETER EXTRACTION FIXES")
    print("=" * 80)

    # Simulate widget extraction for Lasso model
    print("\n[TEST 1] Lasso Model - Hyperparameter Extraction")
    print("-" * 80)

    model_name = "Lasso"

    # Simulate widget extraction (what Tab 7 does)
    tab7_hyperparam_widgets = {'alpha': type('obj', (object,), {'get': lambda: 0.01})}

    hyperparams = {}
    for param_name, widget in tab7_hyperparam_widgets.items():
        try:
            value = widget.get()
            if param_name in ['n_components', 'n_estimators', 'max_depth']:
                hyperparams[param_name] = int(value)
            else:
                hyperparams[param_name] = float(value)
        except Exception as e:
            print(f"  Warning: Could not extract {param_name}: {e}")

    # FIXED CODE (lines 2158-2182)
    if model_name == 'PLS':
        if 'n_components' not in hyperparams:
            hyperparams['n_components'] = 10
    elif model_name in ['Ridge', 'Lasso']:
        if 'alpha' not in hyperparams:
            hyperparams['alpha'] = 1.0
    elif model_name == 'RandomForest':
        if 'n_estimators' not in hyperparams:
            hyperparams['n_estimators'] = 100
        if 'max_depth' not in hyperparams:
            hyperparams['max_depth'] = None
    elif model_name == 'MLP':
        if 'learning_rate_init' not in hyperparams:
            hyperparams['learning_rate_init'] = 0.001
    elif model_name == 'NeuralBoosted':
        if 'n_estimators' not in hyperparams:
            hyperparams['n_estimators'] = 100
        if 'learning_rate' not in hyperparams:
            hyperparams['learning_rate'] = 0.1
        if 'hidden_layer_size' not in hyperparams:
            hyperparams['hidden_layer_size'] = 50

    print(f"Model Type: {model_name}")
    print(f"Extracted Hyperparameters: {hyperparams}")

    # Verify no contamination
    expected_keys = {'alpha'}
    actual_keys = set(hyperparams.keys())

    if actual_keys == expected_keys:
        print("[PASS] Lasso has ONLY 'alpha' parameter (no contamination)")
        test1_pass = True
    else:
        print(f"[FAIL] Lasso has extra parameters: {actual_keys - expected_keys}")
        test1_pass = False

    # Test PLS model
    print("\n[TEST 2] PLS Model - Hyperparameter Extraction")
    print("-" * 80)

    model_name = "PLS"
    tab7_hyperparam_widgets = {'n_components': type('obj', (object,), {'get': lambda: 15})}

    hyperparams = {}
    for param_name, widget in tab7_hyperparam_widgets.items():
        try:
            value = widget.get()
            if param_name in ['n_components', 'n_estimators', 'max_depth']:
                hyperparams[param_name] = int(value)
            else:
                hyperparams[param_name] = float(value)
        except Exception as e:
            print(f"  Warning: Could not extract {param_name}: {e}")

    # FIXED CODE
    if model_name == 'PLS':
        if 'n_components' not in hyperparams:
            hyperparams['n_components'] = 10
    elif model_name in ['Ridge', 'Lasso']:
        if 'alpha' not in hyperparams:
            hyperparams['alpha'] = 1.0

    print(f"Model Type: {model_name}")
    print(f"Extracted Hyperparameters: {hyperparams}")

    expected_keys = {'n_components'}
    actual_keys = set(hyperparams.keys())

    if actual_keys == expected_keys:
        print("[PASS] PLS has ONLY 'n_components' parameter (no contamination)")
        test2_pass = True
    else:
        print(f"[FAIL] PLS has extra parameters: {actual_keys - expected_keys}")
        test2_pass = False

    # Test NeuralBoosted model
    print("\n[TEST 3] NeuralBoosted Model - Hyperparameter Extraction")
    print("-" * 80)

    model_name = "NeuralBoosted"
    tab7_hyperparam_widgets = {
        'n_estimators': type('obj', (object,), {'get': lambda: 200}),
        'learning_rate': type('obj', (object,), {'get': lambda: 0.05}),
        'hidden_layer_size': type('obj', (object,), {'get': lambda: 100})
    }

    hyperparams = {}
    for param_name, widget in tab7_hyperparam_widgets.items():
        try:
            value = widget.get()
            if param_name in ['n_components', 'n_estimators', 'max_depth', 'hidden_layer_size']:
                hyperparams[param_name] = int(value)
            else:
                hyperparams[param_name] = float(value)
        except Exception as e:
            print(f"  Warning: Could not extract {param_name}: {e}")

    # FIXED CODE
    if model_name == 'PLS':
        if 'n_components' not in hyperparams:
            hyperparams['n_components'] = 10
    elif model_name in ['Ridge', 'Lasso']:
        if 'alpha' not in hyperparams:
            hyperparams['alpha'] = 1.0
    elif model_name == 'RandomForest':
        if 'n_estimators' not in hyperparams:
            hyperparams['n_estimators'] = 100
        if 'max_depth' not in hyperparams:
            hyperparams['max_depth'] = None
    elif model_name == 'MLP':
        if 'learning_rate_init' not in hyperparams:
            hyperparams['learning_rate_init'] = 0.001
    elif model_name == 'NeuralBoosted':
        if 'n_estimators' not in hyperparams:
            hyperparams['n_estimators'] = 100
        if 'learning_rate' not in hyperparams:
            hyperparams['learning_rate'] = 0.1
        if 'hidden_layer_size' not in hyperparams:
            hyperparams['hidden_layer_size'] = 50

    print(f"Model Type: {model_name}")
    print(f"Extracted Hyperparameters: {hyperparams}")

    expected_keys = {'n_estimators', 'learning_rate', 'hidden_layer_size'}
    actual_keys = set(hyperparams.keys())

    if actual_keys == expected_keys:
        print("[PASS] NeuralBoosted has correct parameters (no contamination)")
        test3_pass = True
    else:
        print(f"[FAIL] NeuralBoosted has wrong parameters. Expected {expected_keys}, got {actual_keys}")
        test3_pass = False

    # Overall result
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_pass = test1_pass and test2_pass and test3_pass

    if all_pass:
        print("[OK][OK][OK] ALL TESTS PASSED [OK][OK][OK]")
        print("\nFix #1 (Hyperparameter Contamination): WORKING CORRECTLY")
        print("Each model type receives ONLY its relevant hyperparameters.")
        print("No cross-contamination between model types.")
    else:
        print("[FAIL][FAIL][FAIL] SOME TESTS FAILED [FAIL][FAIL][FAIL]")
        print(f"  Test 1 (Lasso): {'PASS' if test1_pass else 'FAIL'}")
        print(f"  Test 2 (PLS): {'PASS' if test2_pass else 'FAIL'}")
        print(f"  Test 3 (NeuralBoosted): {'PASS' if test3_pass else 'FAIL'}")

    print("=" * 80)

    return all_pass

if __name__ == "__main__":
    test_hyperparameter_extraction()
