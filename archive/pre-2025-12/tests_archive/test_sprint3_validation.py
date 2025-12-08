"""Quick validation test for Sprint 3: MLP & SVR hyperparameter implementation with conditional logic"""

import numpy as np
from src.spectral_predict.models import get_model_grids
from src.spectral_predict.model_config import get_hyperparameters

def test_mlp_parameters():
    """Test that MLP parameters are properly exposed with conditional momentum logic"""
    print("Testing MLP parameter exposure...")

    # Get MLP config
    mlp_config = get_hyperparameters('MLP', 'standard')

    # Check all 8 parameters are defined (3 original + 5 new)
    required_params = ['hidden_layer_sizes', 'alpha', 'learning_rate_init',
                      'activation', 'solver', 'batch_size', 'learning_rate', 'momentum']

    for param in required_params:
        assert param in mlp_config, f"Missing parameter: {param}"
        assert isinstance(mlp_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {mlp_config[param]}")

    # Test that standard tier uses single-value defaults for new params
    assert len(mlp_config['activation']) == 1
    assert len(mlp_config['solver']) == 1
    assert len(mlp_config['batch_size']) == 1
    assert len(mlp_config['learning_rate']) == 1
    assert len(mlp_config['momentum']) == 1

    print("  [OK] Grid size maintained with default single values")

    # Test grid generation with conditional momentum logic
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['MLP'])
    mlp_grids = grids['MLP']

    # Verify parameters are in the param dict
    first_model, first_params = mlp_grids[0]
    assert 'activation' in first_params, "activation should be in params"
    assert 'solver' in first_params, "solver should be in params"
    assert 'batch_size' in first_params, "batch_size should be in params"
    assert 'learning_rate' in first_params, "learning_rate should be in params"

    # Check if momentum is conditionally included
    if first_params['solver'] == 'sgd':
        assert 'momentum' in first_params, "momentum should be in params when solver='sgd'"

    print(f"  [OK] Grid generated: {len(mlp_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] MLP parameters test PASSED\n")

def test_svr_parameters():
    """Test that SVR parameters are properly exposed with conditional kernel logic"""
    print("Testing SVR parameter exposure...")

    svr_config = get_hyperparameters('SVR', 'standard')

    # Check all 7 parameters are defined (3 original + 4 new)
    required_params = ['kernel', 'C', 'gamma', 'epsilon', 'degree', 'coef0', 'shrinking']

    for param in required_params:
        assert param in svr_config, f"Missing parameter: {param}"
        assert isinstance(svr_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {svr_config[param]}")

    # Test that standard tier uses single-value defaults for new params
    assert len(svr_config['epsilon']) == 1
    assert len(svr_config['degree']) == 1
    assert len(svr_config['coef0']) == 1
    assert len(svr_config['shrinking']) == 1

    print("  [OK] Grid size maintained with default single values")

    # Test grid generation with conditional kernel logic
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['SVR'])
    svr_grids = grids['SVR']

    # Verify parameters are conditionally in the param dict
    first_model, first_params = svr_grids[0]
    assert 'epsilon' in first_params, "epsilon should be in params"
    assert 'shrinking' in first_params, "shrinking should be in params"

    # Check conditional parameters based on kernel
    kernel = first_params['kernel']
    if kernel in ['rbf', 'poly']:
        assert 'gamma' in first_params, f"gamma should be in params for {kernel} kernel"
    if kernel == 'poly':
        assert 'degree' in first_params, "degree should be in params for poly kernel"
        assert 'coef0' in first_params, "coef0 should be in params for poly kernel"
    if kernel == 'sigmoid':
        assert 'coef0' in first_params, "coef0 should be in params for sigmoid kernel"

    print(f"  [OK] Grid generated: {len(svr_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] SVR parameters test PASSED\n")

def test_mlp_comprehensive_with_sgd():
    """Test that MLP comprehensive tier properly includes momentum with SGD solver"""
    print("Testing MLP comprehensive tier with SGD solver...")

    mlp_config = get_hyperparameters('MLP', 'comprehensive')

    # Verify comprehensive tier has multiple values
    assert len(mlp_config['activation']) > 1, "Comprehensive should have multiple activation values"
    assert len(mlp_config['solver']) > 1, "Comprehensive should have multiple solver values"
    assert 'sgd' in mlp_config['solver'], "Comprehensive should include sgd solver"
    assert len(mlp_config['momentum']) > 1, "Comprehensive should have multiple momentum values"

    print("  [OK] MLP comprehensive tier has expanded parameters including sgd solver")

    # Test grid generation includes momentum for sgd configs
    grids = get_model_grids('regression', 100, tier='comprehensive', enabled_models=['MLP'])
    mlp_grids = grids['MLP']

    # Find a config with sgd solver and verify momentum is included
    sgd_configs = [cfg for cfg in mlp_grids if cfg[1]['solver'] == 'sgd']
    assert len(sgd_configs) > 0, "Should have configs with sgd solver"

    sgd_model, sgd_params = sgd_configs[0]
    assert 'momentum' in sgd_params, "sgd configs should include momentum"
    print(f"  [OK] SGD config includes momentum: {sgd_params}")

    # Find a config with non-sgd solver and verify momentum is NOT included
    non_sgd_configs = [cfg for cfg in mlp_grids if cfg[1]['solver'] != 'sgd']
    if len(non_sgd_configs) > 0:
        non_sgd_model, non_sgd_params = non_sgd_configs[0]
        assert 'momentum' not in non_sgd_params, "non-sgd configs should NOT include momentum"
        print(f"  [OK] Non-SGD config excludes momentum: {non_sgd_params}")

    print("[PASS] MLP conditional momentum test PASSED\n")

def test_svr_comprehensive_kernel_conditionals():
    """Test that SVR comprehensive tier properly handles kernel-conditional parameters"""
    print("Testing SVR comprehensive tier with kernel-conditional parameters...")

    svr_config = get_hyperparameters('SVR', 'comprehensive')

    # Verify comprehensive tier has multiple kernels
    assert len(svr_config['kernel']) > 2, "Comprehensive should have multiple kernel values"
    assert 'poly' in svr_config['kernel'], "Comprehensive should include poly kernel"
    assert 'sigmoid' in svr_config['kernel'], "Comprehensive should include sigmoid kernel"
    assert len(svr_config['degree']) > 1, "Comprehensive should have multiple degree values"
    assert len(svr_config['coef0']) > 1, "Comprehensive should have multiple coef0 values"

    print("  [OK] SVR comprehensive tier has expanded parameters including poly/sigmoid kernels")

    # Test grid generation includes conditional parameters
    grids = get_model_grids('regression', 100, tier='comprehensive', enabled_models=['SVR'])
    svr_grids = grids['SVR']

    # Find a poly config and verify degree and coef0 are included
    poly_configs = [cfg for cfg in svr_grids if cfg[1]['kernel'] == 'poly']
    if len(poly_configs) > 0:
        poly_model, poly_params = poly_configs[0]
        assert 'degree' in poly_params, "poly configs should include degree"
        assert 'coef0' in poly_params, "poly configs should include coef0"
        assert 'gamma' in poly_params, "poly configs should include gamma"
        print(f"  [OK] Poly config includes degree, coef0, gamma: {poly_params}")

    # Find a sigmoid config and verify coef0 is included but not degree
    sigmoid_configs = [cfg for cfg in svr_grids if cfg[1]['kernel'] == 'sigmoid']
    if len(sigmoid_configs) > 0:
        sigmoid_model, sigmoid_params = sigmoid_configs[0]
        assert 'coef0' in sigmoid_params, "sigmoid configs should include coef0"
        assert 'degree' not in sigmoid_params, "sigmoid configs should NOT include degree"
        assert 'gamma' not in sigmoid_params, "sigmoid configs should NOT include gamma"
        print(f"  [OK] Sigmoid config includes coef0 but excludes degree/gamma: {sigmoid_params}")

    # Find a linear config and verify conditional params are excluded
    linear_configs = [cfg for cfg in svr_grids if cfg[1]['kernel'] == 'linear']
    if len(linear_configs) > 0:
        linear_model, linear_params = linear_configs[0]
        assert 'degree' not in linear_params, "linear configs should NOT include degree"
        assert 'coef0' not in linear_params, "linear configs should NOT include coef0"
        assert 'gamma' not in linear_params, "linear configs should NOT include gamma"
        print(f"  [OK] Linear config excludes conditional params: {linear_params}")

    print("[PASS] SVR conditional kernel parameters test PASSED\n")

if __name__ == '__main__':
    print("=" * 60)
    print("SPRINT 3 VALIDATION TEST")
    print("Testing MLP & SVR Hyperparameters with Conditional Logic")
    print("=" * 60 + "\n")

    test_mlp_parameters()
    test_svr_parameters()
    test_mlp_comprehensive_with_sgd()
    test_svr_comprehensive_kernel_conditionals()

    print("=" * 60)
    print("ALL SPRINT 3 TESTS PASSED [PASS]")
    print("=" * 60)
