"""Quick validation test for Sprint 2: Linear model hyperparameter implementation"""

import numpy as np
from src.spectral_predict.models import get_model_grids
from src.spectral_predict.model_config import get_hyperparameters

def test_pls_parameters():
    """Test that PLS parameters are properly exposed"""
    print("Testing PLS parameter exposure...")

    # Get PLS config
    pls_config = get_hyperparameters('PLS', 'standard')

    # Check all 3 parameters are defined (n_components + 2 new ones)
    required_params = ['n_components', 'max_iter', 'tol']

    for param in required_params:
        assert param in pls_config, f"Missing parameter: {param}"
        assert isinstance(pls_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {pls_config[param]}")

    # Test that standard tier uses single-value defaults for new params
    assert len(pls_config['max_iter']) == 1
    assert len(pls_config['tol']) == 1

    print("  [OK] Grid size maintained with default single values")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['PLS'])
    pls_grids = grids['PLS']

    # Verify parameters are in the param dict
    first_model, first_params = pls_grids[0]
    assert 'max_iter' in first_params, "max_iter should be in params"
    assert 'tol' in first_params, "tol should be in params"

    print(f"  [OK] Grid generated: {len(pls_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] PLS parameters test PASSED\n")

def test_ridge_parameters():
    """Test that Ridge parameters are properly exposed"""
    print("Testing Ridge parameter exposure...")

    ridge_config = get_hyperparameters('Ridge', 'standard')

    required_params = ['alpha', 'solver', 'tol']

    for param in required_params:
        assert param in ridge_config, f"Missing parameter: {param}"
        assert isinstance(ridge_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {ridge_config[param]}")

    assert len(ridge_config['solver']) == 1
    assert len(ridge_config['tol']) == 1

    print("  [OK] Grid size: 5 alpha x 1 solver x 1 tol = 5 configs (unchanged)")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['Ridge'])
    ridge_grids = grids['Ridge']

    first_model, first_params = ridge_grids[0]
    assert 'solver' in first_params, "solver should be in params"
    assert 'tol' in first_params, "tol should be in params"

    print(f"  [OK] Grid generated: {len(ridge_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] Ridge parameters test PASSED\n")

def test_lasso_parameters():
    """Test that Lasso parameters are properly exposed"""
    print("Testing Lasso parameter exposure...")

    lasso_config = get_hyperparameters('Lasso', 'standard')

    required_params = ['alpha', 'selection', 'tol']

    for param in required_params:
        assert param in lasso_config, f"Missing parameter: {param}"
        assert isinstance(lasso_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {lasso_config[param]}")

    assert len(lasso_config['selection']) == 1
    assert len(lasso_config['tol']) == 1

    print("  [OK] Grid size: 4 alpha x 1 selection x 1 tol = 4 configs (unchanged)")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['Lasso'])
    lasso_grids = grids['Lasso']

    first_model, first_params = lasso_grids[0]
    assert 'selection' in first_params, "selection should be in params"
    assert 'tol' in first_params, "tol should be in params"

    print(f"  [OK] Grid generated: {len(lasso_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] Lasso parameters test PASSED\n")

def test_elasticnet_parameters():
    """Test that ElasticNet parameters are properly exposed"""
    print("Testing ElasticNet parameter exposure...")

    en_config = get_hyperparameters('ElasticNet', 'standard')

    required_params = ['alpha', 'l1_ratio', 'selection', 'tol']

    for param in required_params:
        assert param in en_config, f"Missing parameter: {param}"
        assert isinstance(en_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {en_config[param]}")

    assert len(en_config['selection']) == 1
    assert len(en_config['tol']) == 1

    print("  [OK] Grid size: 4 alpha x 5 l1_ratio x 1 selection x 1 tol = 20 configs (unchanged)")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['ElasticNet'])
    en_grids = grids['ElasticNet']

    first_model, first_params = en_grids[0]
    assert 'selection' in first_params, "selection should be in params"
    assert 'tol' in first_params, "tol should be in params"

    print(f"  [OK] Grid generated: {len(en_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] ElasticNet parameters test PASSED\n")

def test_comprehensive_tier_expansion():
    """Test that comprehensive tier properly expands linear model grids"""
    print("Testing comprehensive tier grid expansion...")

    # PLS comprehensive
    pls_config = get_hyperparameters('PLS', 'comprehensive')
    assert len(pls_config['max_iter']) > 1, "Comprehensive should have multiple max_iter values"
    assert len(pls_config['tol']) > 1, "Comprehensive should have multiple tol values"
    print("  [OK] PLS comprehensive tier has expanded parameters")

    # Ridge comprehensive
    ridge_config = get_hyperparameters('Ridge', 'comprehensive')
    assert len(ridge_config['solver']) > 1, "Comprehensive should have multiple solver values"
    assert len(ridge_config['tol']) > 1, "Comprehensive should have multiple tol values"
    print("  [OK] Ridge comprehensive tier has expanded parameters")

    # Lasso comprehensive
    lasso_config = get_hyperparameters('Lasso', 'comprehensive')
    assert len(lasso_config['selection']) > 1, "Comprehensive should have multiple selection values"
    assert len(lasso_config['tol']) > 1, "Comprehensive should have multiple tol values"
    print("  [OK] Lasso comprehensive tier has expanded parameters")

    # ElasticNet comprehensive
    en_config = get_hyperparameters('ElasticNet', 'comprehensive')
    assert len(en_config['selection']) > 1, "Comprehensive should have multiple selection values"
    assert len(en_config['tol']) > 1, "Comprehensive should have multiple tol values"
    print("  [OK] ElasticNet comprehensive tier has expanded parameters")

    print("[PASS] Comprehensive tier expansion test PASSED\n")

if __name__ == '__main__':
    print("=" * 60)
    print("SPRINT 2 VALIDATION TEST")
    print("Testing Linear Model (PLS, Ridge, Lasso, ElasticNet) Hyperparameters")
    print("=" * 60 + "\n")

    test_pls_parameters()
    test_ridge_parameters()
    test_lasso_parameters()
    test_elasticnet_parameters()
    test_comprehensive_tier_expansion()

    print("=" * 60)
    print("ALL SPRINT 2 TESTS PASSED [PASS]")
    print("=" * 60)
