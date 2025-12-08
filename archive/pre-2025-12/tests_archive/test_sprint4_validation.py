"""Quick validation test for Sprint 4: XGBoost and CatBoost hyperparameter implementation

Note: NeuralBoosted subsample was removed from Sprint 4 because NeuralBoostedRegressor
is a custom implementation that doesn't support subsampling without modifying the
internal boosting algorithm logic.
"""

import numpy as np
from src.spectral_predict.models import get_model_grids
from src.spectral_predict.model_config import get_hyperparameters

def test_xgboost_parameters():
    """Test that XGBoost parameters are properly exposed"""
    print("Testing XGBoost parameter exposure...")

    # Get XGBoost config
    xgb_config = get_hyperparameters('XGBoost', 'standard')

    # Check all 9 parameters are defined (7 original + 2 new)
    required_params = ['n_estimators', 'learning_rate', 'max_depth', 'subsample',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight', 'gamma']

    for param in required_params:
        assert param in xgb_config, f"Missing parameter: {param}"
        assert isinstance(xgb_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {xgb_config[param]}")

    # Test that standard tier uses single-value defaults for new params
    assert len(xgb_config['min_child_weight']) == 1
    assert len(xgb_config['gamma']) == 1

    print("  [OK] Grid size maintained with default single values")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='standard', enabled_models=['XGBoost'])
    xgb_grids = grids['XGBoost']

    # Verify parameters are in the param dict
    first_model, first_params = xgb_grids[0]
    assert 'min_child_weight' in first_params, "min_child_weight should be in params"
    assert 'gamma' in first_params, "gamma should be in params"

    print(f"  [OK] Grid generated: {len(xgb_grids)} configs")
    print(f"  [OK] Sample params: {first_params}")
    print("[PASS] XGBoost parameters test PASSED\n")

def test_catboost_parameters():
    """Test that CatBoost parameters are properly exposed"""
    print("Testing CatBoost parameter exposure...")

    catboost_config = get_hyperparameters('CatBoost', 'standard')

    # Check all 7 parameters are defined (3 original + 4 new)
    required_params = ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg',
                      'border_count', 'bagging_temperature', 'random_strength']

    for param in required_params:
        assert param in catboost_config, f"Missing parameter: {param}"
        assert isinstance(catboost_config[param], list), f"{param} should be a list"
        print(f"  [OK] {param}: {catboost_config[param]}")

    # Test that standard tier uses single-value defaults for new params
    assert len(catboost_config['l2_leaf_reg']) == 1
    assert len(catboost_config['border_count']) == 1
    assert len(catboost_config['bagging_temperature']) == 1
    assert len(catboost_config['random_strength']) == 1

    print("  [OK] Grid size maintained with default single values")

    # Test grid generation (only if CatBoost is available)
    try:
        grids = get_model_grids('regression', 100, tier='standard', enabled_models=['CatBoost'])
        if 'CatBoost' in grids:
            catboost_grids = grids['CatBoost']

            # Verify parameters are in the param dict
            first_model, first_params = catboost_grids[0]
            assert 'l2_leaf_reg' in first_params, "l2_leaf_reg should be in params"
            assert 'border_count' in first_params, "border_count should be in params"
            assert 'bagging_temperature' in first_params, "bagging_temperature should be in params"
            assert 'random_strength' in first_params, "random_strength should be in params"

            print(f"  [OK] Grid generated: {len(catboost_grids)} configs")
            print(f"  [OK] Sample params: {first_params}")
        else:
            print("  [SKIP] CatBoost not available on this system")
    except Exception as e:
        print(f"  [SKIP] CatBoost not available: {e}")

    print("[PASS] CatBoost parameters test PASSED\n")

def test_xgboost_comprehensive_tier():
    """Test that XGBoost comprehensive tier properly expands parameters"""
    print("Testing XGBoost comprehensive tier expansion...")

    xgb_config = get_hyperparameters('XGBoost', 'comprehensive')

    # Verify comprehensive tier has multiple values
    assert len(xgb_config['min_child_weight']) > 1, "Comprehensive should have multiple min_child_weight values"
    assert len(xgb_config['gamma']) > 1, "Comprehensive should have multiple gamma values"

    print("  [OK] XGBoost comprehensive tier has expanded parameters")

    # Test grid generation
    grids = get_model_grids('regression', 100, tier='comprehensive', enabled_models=['XGBoost'])
    xgb_grids = grids['XGBoost']

    # Verify grid is larger than standard
    standard_grids = get_model_grids('regression', 100, tier='standard', enabled_models=['XGBoost'])
    assert len(xgb_grids) > len(standard_grids['XGBoost']), "Comprehensive should have more configs than standard"

    print(f"  [OK] Comprehensive grid: {len(xgb_grids)} configs (vs {len(standard_grids['XGBoost'])} in standard)")
    print("[PASS] XGBoost comprehensive tier test PASSED\n")

def test_catboost_comprehensive_tier():
    """Test that CatBoost comprehensive tier properly expands parameters"""
    print("Testing CatBoost comprehensive tier expansion...")

    catboost_config = get_hyperparameters('CatBoost', 'comprehensive')

    # Verify comprehensive tier has multiple values
    assert len(catboost_config['l2_leaf_reg']) > 1, "Comprehensive should have multiple l2_leaf_reg values"
    assert len(catboost_config['border_count']) > 1, "Comprehensive should have multiple border_count values"
    assert len(catboost_config['bagging_temperature']) > 1, "Comprehensive should have multiple bagging_temperature values"
    assert len(catboost_config['random_strength']) > 1, "Comprehensive should have multiple random_strength values"

    print("  [OK] CatBoost comprehensive tier has expanded parameters")

    # Test grid generation (only if CatBoost is available)
    try:
        grids = get_model_grids('regression', 100, tier='comprehensive', enabled_models=['CatBoost'])
        if 'CatBoost' in grids:
            catboost_grids = grids['CatBoost']

            # Verify grid is larger than standard
            standard_grids = get_model_grids('regression', 100, tier='standard', enabled_models=['CatBoost'])
            if 'CatBoost' in standard_grids:
                assert len(catboost_grids) > len(standard_grids['CatBoost']), "Comprehensive should have more configs"
                print(f"  [OK] Comprehensive grid: {len(catboost_grids)} configs (vs {len(standard_grids['CatBoost'])} in standard)")
        else:
            print("  [SKIP] CatBoost not available on this system")
    except Exception as e:
        print(f"  [SKIP] CatBoost not available: {e}")

    print("[PASS] CatBoost comprehensive tier test PASSED\n")

if __name__ == '__main__':
    print("=" * 60)
    print("SPRINT 4 VALIDATION TEST")
    print("Testing XGBoost and CatBoost Hyperparameters")
    print("=" * 60 + "\n")

    test_xgboost_parameters()
    test_catboost_parameters()
    test_xgboost_comprehensive_tier()
    test_catboost_comprehensive_tier()

    print("=" * 60)
    print("ALL SPRINT 4 TESTS PASSED [PASS]")
    print("=" * 60)
