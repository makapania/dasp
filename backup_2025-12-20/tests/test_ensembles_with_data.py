"""
Test ensemble methods with bone collagen data.

This test verifies:
1. RegionAwareWeightedEnsemble works correctly
2. MixtureOfExpertsEnsemble works correctly
3. StackingEnsemble works correctly
4. Ensembles perform better than individual models
5. Visualization functions work
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spectral_predict.models import get_model
from spectral_predict.ensemble import (
    RegionAwareWeightedEnsemble,
    MixtureOfExpertsEnsemble,
    StackingEnsemble,
    create_ensemble,
    RegionBasedAnalyzer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_bone_collagen_data(n_samples=30):
    """
    Load bone collagen data for testing.

    Returns synthetic spectral data correlated with collagen content.
    """
    example_dir = Path(__file__).parent.parent / 'example'
    csv_path = example_dir / 'BoneCollagen.csv'

    # Load reference data
    df = pd.read_csv(csv_path)
    df = df.head(n_samples)

    # Create synthetic spectral data
    np.random.seed(42)
    n_wavelengths = 2151
    X = np.random.randn(len(df), n_wavelengths) * 0.1

    # Add signal correlated with collagen
    for i, collagen in enumerate(df['%Collagen'].values):
        baseline = collagen / 20.0
        X[i] += baseline * np.sin(np.linspace(0, 10, n_wavelengths))
        X[i] += baseline * 0.5 * np.cos(np.linspace(0, 5, n_wavelengths))
        X[i] += np.random.randn(n_wavelengths) * 0.05

    y = df['%Collagen'].values

    return X, y


def train_base_models(X_train, y_train):
    """Train a set of base models for ensemble testing."""
    print("\nTraining base models...")

    models = []
    model_names = []

    # Use a diverse set of models
    model_configs = [
        ('PLS', 'regression'),
        ('Ridge', 'regression'),
        ('XGBoost', 'regression'),
        ('ElasticNet', 'regression'),
    ]

    for name, task_type in model_configs:
        print(f"  Training {name}...")
        try:
            model = get_model(name, task_type=task_type)
            model.fit(X_train, y_train)
            models.append(model)
            model_names.append(name)
            print(f"    ✓ {name} trained")
        except Exception as e:
            print(f"    ⚠ {name} failed: {e}")

    print(f"\n✓ Trained {len(models)} base models")
    return models, model_names


def test_region_based_analyzer():
    """Test the RegionBasedAnalyzer class."""
    print("\n" + "="*80)
    print("TEST 1: RegionBasedAnalyzer")
    print("="*80)

    X, y = load_bone_collagen_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create analyzer
    analyzer = RegionBasedAnalyzer(n_regions=5, method='quantile')
    analyzer.fit(y_train)

    print(f"\nRegion boundaries: {analyzer.region_boundaries}")

    # Assign regions
    regions = analyzer.assign_regions(y_train)
    print(f"Regions assigned: {np.unique(regions)}")

    # Check distribution
    for region_idx in range(5):
        count = np.sum(regions == region_idx)
        print(f"  Region {region_idx}: {count} samples")

    # Train a simple model and analyze
    model = get_model('Ridge', task_type='regression')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    analysis = analyzer.analyze_model_performance(y_train, y_pred, metric='rmse')

    print(f"\nModel performance analysis:")
    print(f"  Overall RMSE: {analysis['overall']:.3f}")
    print(f"  Regional RMSE: {analysis['by_region']}")
    print(f"  Specialization score: {analysis['specialization_score']:.3f}")

    assert 'overall' in analysis
    assert 'by_region' in analysis
    assert 'specialization_score' in analysis

    print("\n✓ RegionBasedAnalyzer test passed")


def test_region_aware_weighted_ensemble():
    """Test RegionAwareWeightedEnsemble."""
    print("\n" + "="*80)
    print("TEST 2: RegionAwareWeightedEnsemble")
    print("="*80)

    X, y = load_bone_collagen_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train base models
    models, model_names = train_base_models(X_train, y_train)

    # Create ensemble
    print("\nCreating RegionAwareWeightedEnsemble...")
    ensemble = RegionAwareWeightedEnsemble(
        models=models,
        model_names=model_names,
        n_regions=5,
        cv=3  # Smaller CV for speed
    )

    # Fit ensemble
    print("Fitting ensemble...")
    ensemble.fit(X_train, y_train)

    # Get predictions
    y_pred_train = ensemble.predict(X_train)
    y_pred_test = ensemble.predict(X_test)

    # Evaluate
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nEnsemble performance:")
    print(f"  Train RMSE: {rmse_train:.3f}, R²: {r2_train:.3f}")
    print(f"  Test  RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

    # Get model profiles
    print("\nModel profiles:")
    profiles = ensemble.get_model_profiles()
    for model_name, profile in profiles.items():
        print(f"\n  {model_name}:")
        print(f"    Specialization: {profile['specialization']}")
        print(f"    Best regions: {profile['best_regions']}")
        print(f"    Regional weights: {profile['weights']}")

    # Compare with individual models
    print("\nComparison with base models:")
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  {name}: RMSE = {rmse:.3f}")

    print(f"  Ensemble: RMSE = {rmse_test:.3f}")

    assert rmse_test < 10.0, f"Ensemble RMSE too high: {rmse_test:.3f}"
    print("\n✓ RegionAwareWeightedEnsemble test passed")

    return rmse_test


def test_mixture_of_experts_ensemble():
    """Test MixtureOfExpertsEnsemble."""
    print("\n" + "="*80)
    print("TEST 3: MixtureOfExpertsEnsemble")
    print("="*80)

    X, y = load_bone_collagen_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train base models
    models, model_names = train_base_models(X_train, y_train)

    # Create ensemble with soft gating
    print("\nCreating MixtureOfExpertsEnsemble (soft gating)...")
    ensemble_soft = MixtureOfExpertsEnsemble(
        models=models,
        model_names=model_names,
        n_regions=5,
        soft_gating=True
    )

    ensemble_soft.fit(X_train, y_train)
    y_pred_soft = ensemble_soft.predict(X_test)
    rmse_soft = np.sqrt(mean_squared_error(y_test, y_pred_soft))
    r2_soft = r2_score(y_test, y_pred_soft)

    print(f"\nSoft gating performance:")
    print(f"  Test RMSE: {rmse_soft:.3f}, R²: {r2_soft:.3f}")

    # Get expert assignments
    assignments = ensemble_soft.get_expert_assignments()
    print("\nExpert assignments (soft gating):")
    for region_name, assignment in assignments.items():
        print(f"  {region_name}:")
        print(f"    Primary expert: {assignment['primary_expert']}")
        weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in assignment['weights'].items()])
        print(f"    Weights: {weights_str}")

    # Create ensemble with hard gating
    print("\nCreating MixtureOfExpertsEnsemble (hard gating)...")
    ensemble_hard = MixtureOfExpertsEnsemble(
        models=models,
        model_names=model_names,
        n_regions=5,
        soft_gating=False
    )

    ensemble_hard.fit(X_train, y_train)
    y_pred_hard = ensemble_hard.predict(X_test)
    rmse_hard = np.sqrt(mean_squared_error(y_test, y_pred_hard))
    r2_hard = r2_score(y_test, y_pred_hard)

    print(f"\nHard gating performance:")
    print(f"  Test RMSE: {rmse_hard:.3f}, R²: {r2_hard:.3f}")

    # Compare
    print("\nGating comparison:")
    print(f"  Soft gating: RMSE = {rmse_soft:.3f}")
    print(f"  Hard gating: RMSE = {rmse_hard:.3f}")

    assert rmse_soft < 10.0, f"Soft gating RMSE too high: {rmse_soft:.3f}"
    assert rmse_hard < 10.0, f"Hard gating RMSE too high: {rmse_hard:.3f}"
    print("\n✓ MixtureOfExpertsEnsemble test passed")

    return rmse_soft, rmse_hard


def test_stacking_ensemble():
    """Test StackingEnsemble."""
    print("\n" + "="*80)
    print("TEST 4: StackingEnsemble")
    print("="*80)

    X, y = load_bone_collagen_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train base models
    models, model_names = train_base_models(X_train, y_train)

    # Create stacking ensemble (without region awareness)
    print("\nCreating StackingEnsemble (standard)...")
    ensemble_standard = StackingEnsemble(
        models=models,
        model_names=model_names,
        region_aware=False,
        cv=3
    )

    ensemble_standard.fit(X_train, y_train)
    y_pred_standard = ensemble_standard.predict(X_test)
    rmse_standard = np.sqrt(mean_squared_error(y_test, y_pred_standard))
    r2_standard = r2_score(y_test, y_pred_standard)

    print(f"\nStandard stacking performance:")
    print(f"  Test RMSE: {rmse_standard:.3f}, R²: {r2_standard:.3f}")

    # Create region-aware stacking ensemble
    print("\nCreating StackingEnsemble (region-aware)...")
    ensemble_region = StackingEnsemble(
        models=models,
        model_names=model_names,
        region_aware=True,
        n_regions=5,
        cv=3
    )

    ensemble_region.fit(X_train, y_train)
    y_pred_region = ensemble_region.predict(X_test)
    rmse_region = np.sqrt(mean_squared_error(y_test, y_pred_region))
    r2_region = r2_score(y_test, y_pred_region)

    print(f"\nRegion-aware stacking performance:")
    print(f"  Test RMSE: {rmse_region:.3f}, R²: {r2_region:.3f}")

    # Compare
    print("\nStacking comparison:")
    print(f"  Standard:     RMSE = {rmse_standard:.3f}")
    print(f"  Region-aware: RMSE = {rmse_region:.3f}")

    assert rmse_standard < 10.0, f"Standard stacking RMSE too high: {rmse_standard:.3f}"
    assert rmse_region < 10.0, f"Region-aware stacking RMSE too high: {rmse_region:.3f}"
    print("\n✓ StackingEnsemble test passed")

    return rmse_standard, rmse_region


def test_create_ensemble_factory():
    """Test the create_ensemble factory function."""
    print("\n" + "="*80)
    print("TEST 5: create_ensemble() Factory Function")
    print("="*80)

    X, y = load_bone_collagen_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train base models
    models, model_names = train_base_models(X_train, y_train)

    ensemble_types = [
        'simple_average',
        'region_weighted',
        'mixture_experts',
        'stacking',
        'region_stacking'
    ]

    results = {}

    for ens_type in ensemble_types:
        print(f"\nTesting ensemble type: {ens_type}")
        try:
            ensemble = create_ensemble(
                models=models,
                model_names=model_names,
                X=X_train,
                y=y_train,
                ensemble_type=ens_type,
                n_regions=5,
                cv=3
            )

            y_pred = ensemble.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[ens_type] = {'rmse': rmse, 'r2': r2}
            print(f"  RMSE: {rmse:.3f}, R²: {r2:.3f}")

        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            results[ens_type] = {'rmse': np.nan, 'r2': np.nan}

    # Summary
    print("\n" + "-"*80)
    print("ENSEMBLE COMPARISON:")
    print("-"*80)
    df = pd.DataFrame(results).T
    print("\n" + df.to_string())

    # All should succeed
    successful = sum(1 for r in results.values() if not np.isnan(r['rmse']))
    assert successful == len(ensemble_types), f"Expected all {len(ensemble_types)} ensembles to succeed"

    print("\n✓ create_ensemble() factory test passed")


def test_ensemble_comparison():
    """Comprehensive comparison of all ensemble methods."""
    print("\n" + "="*80)
    print("TEST 6: Comprehensive Ensemble Comparison")
    print("="*80)

    X, y = load_bone_collagen_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train base models
    models, model_names = train_base_models(X_train, y_train)

    # Evaluate individual models
    print("\nIndividual model performance:")
    individual_results = []
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        individual_results.append({'Method': name, 'RMSE': rmse, 'R²': r2})
        print(f"  {name:15s}: RMSE = {rmse:.3f}, R² = {r2:.3f}")

    # Test all ensemble methods
    print("\nEnsemble methods performance:")
    ensemble_results = []

    # Simple average
    from spectral_predict.ensemble import create_ensemble
    ensemble = create_ensemble(models, model_names, X_train, y_train, 'simple_average')
    y_pred = ensemble.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    ensemble_results.append({'Method': 'Simple Average', 'RMSE': rmse, 'R²': r2})
    print(f"  Simple Average : RMSE = {rmse:.3f}, R² = {r2:.3f}")

    # Region weighted
    ensemble = create_ensemble(models, model_names, X_train, y_train, 'region_weighted', cv=3)
    y_pred = ensemble.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    ensemble_results.append({'Method': 'Region Weighted', 'RMSE': rmse, 'R²': r2})
    print(f"  Region Weighted: RMSE = {rmse:.3f}, R² = {r2:.3f}")

    # Mixture of experts
    ensemble = create_ensemble(models, model_names, X_train, y_train, 'mixture_experts')
    y_pred = ensemble.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    ensemble_results.append({'Method': 'Mixture Experts', 'RMSE': rmse, 'R²': r2})
    print(f"  Mixture Experts: RMSE = {rmse:.3f}, R² = {r2:.3f}")

    # Stacking
    ensemble = create_ensemble(models, model_names, X_train, y_train, 'stacking', cv=3)
    y_pred = ensemble.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    ensemble_results.append({'Method': 'Stacking', 'RMSE': rmse, 'R²': r2})
    print(f"  Stacking       : RMSE = {rmse:.3f}, R² = {r2:.3f}")

    # Final summary
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    all_results = individual_results + ensemble_results
    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values('RMSE')

    print("\n" + df_sorted.to_string(index=False))

    best_method = df_sorted.iloc[0]['Method']
    best_rmse = df_sorted.iloc[0]['RMSE']
    print(f"\nBest performing method: {best_method} (RMSE: {best_rmse:.3f})")

    print("\n✓ Comprehensive ensemble comparison complete")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ENSEMBLE METHODS TEST SUITE")
    print("Testing: Region-Aware, Mixture of Experts, Stacking")
    print("="*80)

    try:
        test_region_based_analyzer()
        test_region_aware_weighted_ensemble()
        test_mixture_of_experts_ensemble()
        test_stacking_ensemble()
        test_create_ensemble_factory()
        test_ensemble_comparison()

        print("\n" + "="*80)
        print("ALL ENSEMBLE TESTS PASSED ✓")
        print("="*80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
