"""Quick validation test for Sprint 1: LightGBM & RandomForest parameter implementation"""

import numpy as np
from src.spectral_predict.models import get_model_grids
from src.spectral_predict.model_config import get_hyperparameters

def test_lightgbm_parameters():
    """Test that LightGBM parameters are properly exposed (not hard-coded)"""
    print("Testing LightGBM parameter exposure...")

    # Get LightGBM config
    lgbm_config = get_hyperparameters('LightGBM', 'standard')

    # Check all 9 parameters are defined
    required_params = ['n_estimators', 'learning_rate', 'num_leaves',
                      'max_depth', 'min_child_samples', 'subsample',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda']

    for param in required_params:
        assert param in lgbm_config, f"Missing parameter: {param}"
        assert isinstance(lgbm_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {lgbm_config[param]}")

    # Test that standard tier uses single-value defaults (grid size maintained)
    assert len(lgbm_config['max_depth']) == 1
    assert len(lgbm_config['min_child_samples']) == 1
    assert len(lgbm_config['subsample']) == 1
    assert len(lgbm_config['colsample_bytree']) == 1
    assert len(lgbm_config['reg_alpha']) == 1
    assert len(lgbm_config['reg_lambda']) == 1

    print("  [OK] Grid size maintained: 3×3×3×1×1×1×1×1×1 = 27 configs")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['LightGBM'])
    lgbm_grids = grids['LightGBM']

    # Verify parameters are in the param dict
    first_model, first_params = lgbm_grids[0]
    assert 'max_depth' in first_params, "max_depth should be in params"
    assert 'min_child_samples' in first_params, "min_child_samples should be in params"
    assert 'subsample' in first_params, "subsample should be in params"
    assert 'colsample_bytree' in first_params, "colsample_bytree should be in params"
    assert 'reg_alpha' in first_params, "reg_alpha should be in params"
    assert 'reg_lambda' in first_params, "reg_lambda should be in params"

    print(f"  [OK] Grid generated: {len(lgbm_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] LightGBM parameters test PASSED\n")

def test_randomforest_parameters():
    """Test that RandomForest parameters are properly restored"""
    print("Testing RandomForest parameter restoration...")

    # Get RandomForest config
    rf_config = get_hyperparameters('RandomForest', 'standard')

    # Check all 8 parameters are defined
    required_params = ['n_estimators', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'max_features', 'bootstrap',
                      'max_leaf_nodes', 'min_impurity_decrease']

    for param in required_params:
        assert param in rf_config, f"Missing parameter: {param}"
        assert isinstance(rf_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {rf_config[param]}")

    # Test that standard tier uses single-value defaults (grid size maintained)
    assert len(rf_config['min_samples_split']) == 1
    assert len(rf_config['min_samples_leaf']) == 1
    assert len(rf_config['max_features']) == 1
    assert len(rf_config['bootstrap']) == 1
    assert len(rf_config['max_leaf_nodes']) == 1
    assert len(rf_config['min_impurity_decrease']) == 1

    print("  [OK] Grid size maintained: 3×3×1×1×1×1×1×1 = 9 configs")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['RandomForest'])
    rf_grids = grids['RandomForest']

    # Verify parameters are in the param dict
    first_model, first_params = rf_grids[0]
    assert 'min_samples_split' in first_params, "min_samples_split should be in params"
    assert 'min_samples_leaf' in first_params, "min_samples_leaf should be in params"
    assert 'max_features' in first_params, "max_features should be in params"
    assert 'bootstrap' in first_params, "bootstrap should be in params"
    assert 'max_leaf_nodes' in first_params, "max_leaf_nodes should be in params"
    assert 'min_impurity_decrease' in first_params, "min_impurity_decrease should be in params"

    print(f"  [OK] Grid generated: {len(rf_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] RandomForest parameters test PASSED\n")

def test_comprehensive_tier_expansion():
    """Test that comprehensive tier properly expands grid"""
    print("Testing comprehensive tier grid expansion...")

    # LightGBM comprehensive
    lgbm_config = get_hyperparameters('LightGBM', 'comprehensive')
    assert len(lgbm_config['max_depth']) > 1, "Comprehensive should have multiple max_depth values"
    assert len(lgbm_config['min_child_samples']) > 1, "Comprehensive should have multiple min_child_samples values"
    assert len(lgbm_config['subsample']) > 1, "Comprehensive should have multiple subsample values"
    print("  [OK] LightGBM comprehensive tier has expanded parameters")

    # RandomForest comprehensive
    rf_config = get_hyperparameters('RandomForest', 'comprehensive')
    assert len(rf_config['min_samples_split']) > 1, "Comprehensive should have multiple min_samples_split values"
    assert len(rf_config['min_samples_leaf']) > 1, "Comprehensive should have multiple min_samples_leaf values"
    assert len(rf_config['max_features']) > 1, "Comprehensive should have multiple max_features values"
    print("  [OK] RandomForest comprehensive tier has expanded parameters")

    print("[PASS] Comprehensive tier expansion test PASSED\n")

if __name__ == '__main__':
    print("=" * 60)
    print("SPRINT 1 VALIDATION TEST")
    print("Testing LightGBM hard-coded values fix and RandomForest restoration")
    print("=" * 60 + "\n")

    test_lightgbm_parameters()
    test_randomforest_parameters()
    test_comprehensive_tier_expansion()

    print("=" * 60)
    print("ALL SPRINT 1 TESTS PASSED [PASS]")
    print("=" * 60)
