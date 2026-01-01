"""
V1 Comprehensive Testing Script
===============================

End-to-end testing of V1 spectral predict functionality using bone collagen data.

Data: C:/Users/sponheim/Desktop/bone
- 49 samples with %Collagen (regression) and CollagenCat (classification)
- Train: 41 samples, Test: 8 samples via SPXY

Stages:
1. Data Loading & SPXY Split
2. Grid Search Baseline (PLS, Ridge, LightGBM)
3. Variable Selection Methods Testing
4. Bayesian Optimization Testing
5. NSGA2 Multi-Objective Testing
6. Model Save/Load Verification
7. Holdout Validation

Each stage pauses for user approval before continuing.
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Parse command line args
parser = argparse.ArgumentParser(description='V1 Comprehensive Testing')
parser.add_argument('--no-prompt', action='store_true', help='Skip confirmation prompts')
parser.add_argument('--stage', type=int, default=0, help='Run only specific stage (1-7)')
ARGS, _ = parser.parse_known_args()

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import spectral_predict modules
from spectral_predict.io import read_asd_dir, read_reference_csv
from spectral_predict.sample_selection import spxy
from spectral_predict.search import run_search, run_bayesian_search
from spectral_predict.nsga2_search import run_nsga2_search
from spectral_predict.model_io import save_model, load_model, predict_with_model
from spectral_predict.models import get_model, build_model


# Configuration
DATA_DIR = Path(r"C:\Users\sponheim\Desktop\bone")
REFERENCE_FILE = DATA_DIR / "BoneCollagen.csv"
TEST_SIZE = 8  # 8-sample holdout (~17%)
RANDOM_STATE = 42


def wait_for_approval(stage_name: str):
    """Pause and wait for user to approve continuing."""
    print("\n" + "=" * 60)
    print(f"CHECKPOINT: {stage_name} Complete")
    print("=" * 60)

    if ARGS.no_prompt:
        print("\n[--no-prompt] Auto-continuing...")
        return

    response = input("\nContinue to next stage? [Y/n/q]: ").strip().lower()
    if response in ['n', 'q', 'quit', 'exit']:
        print("Testing stopped by user.")
        sys.exit(0)
    print()


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60 + "\n")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n--- {text} ---\n")


# =============================================================================
# STAGE 1: Data Loading & SPXY Split
# =============================================================================

def stage1_load_data():
    """Load spectral data and create SPXY train/test split."""
    print_header("STAGE 1: Data Loading & SPXY Split")

    # Load spectral data from ASD files
    print("Loading spectral data from ASD files...")
    spectra_df, spec_metadata = read_asd_dir(DATA_DIR)
    print(f"  Loaded {len(spectra_df)} spectra")
    print(f"  Wavelength range: {spectra_df.columns.min():.1f} - {spectra_df.columns.max():.1f} nm")
    print(f"  Number of wavelengths: {len(spectra_df.columns)}")

    # Load reference file
    print("\nLoading reference data...")
    ref_df = read_reference_csv(REFERENCE_FILE, id_column='File Number')
    print(f"  Loaded {len(ref_df)} reference entries")
    print(f"  Columns: {list(ref_df.columns)}")

    # Clean up index matching
    # Spectra: "Spectrum00001" -> "Spectrum 00001" (add space after "Spectrum")
    spectra_df.index = spectra_df.index.str.replace('.asd', '', regex=False)
    spectra_df.index = spectra_df.index.str.replace(r'Spectrum(\d)', r'Spectrum \1', regex=True)

    # Match spectra to reference
    common_ids = spectra_df.index.intersection(ref_df.index)
    print(f"\nMatched {len(common_ids)} samples")

    if len(common_ids) < len(spectra_df):
        missing = set(spectra_df.index) - set(ref_df.index)
        print(f"  Missing from reference: {list(missing)[:5]}...")

    # Align data
    X = spectra_df.loc[common_ids]
    y_regression = ref_df.loc[common_ids, '%Collagen'].astype(float)
    y_classification = ref_df.loc[common_ids, 'CollagenCat']

    print(f"\nRegression target (%Collagen):")
    print(f"  Range: {y_regression.min():.2f} - {y_regression.max():.2f}")
    print(f"  Mean: {y_regression.mean():.2f}, Std: {y_regression.std():.2f}")

    print(f"\nClassification target (CollagenCat):")
    print(f"  Classes: {y_classification.value_counts().to_dict()}")

    # Create SPXY split
    print(f"\nCreating SPXY split ({TEST_SIZE} test samples)...")
    test_indices = spxy(X.values, y_regression.values, n_samples=TEST_SIZE)
    train_mask = ~np.isin(np.arange(len(X)), test_indices)

    X_train = X.iloc[train_mask]
    X_test = X.iloc[test_indices]
    y_reg_train = y_regression.iloc[train_mask]
    y_reg_test = y_regression.iloc[test_indices]
    y_cls_train = y_classification.iloc[train_mask]
    y_cls_test = y_classification.iloc[test_indices]

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"  Regression range: {y_reg_train.min():.2f} - {y_reg_train.max():.2f}")
    print(f"  Classification: {y_cls_train.value_counts().to_dict()}")

    print(f"\nTest set: {len(X_test)} samples")
    print(f"  Regression range: {y_reg_test.min():.2f} - {y_reg_test.max():.2f}")
    print(f"  Classification: {y_cls_test.value_counts().to_dict()}")

    # Verify test set covers target range
    if y_reg_test.min() > y_reg_train.min() * 1.5 or y_reg_test.max() < y_reg_train.max() * 0.7:
        print("\n  WARNING: Test set may not adequately cover target range!")
    else:
        print("\n  Test set coverage looks good.")

    # Package results
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_reg_train': y_reg_train,
        'y_reg_test': y_reg_test,
        'y_cls_train': y_cls_train,
        'y_cls_test': y_cls_test,
        'wavelengths': list(X.columns),
    }

    return data


# =============================================================================
# STAGE 2: Grid Search Baseline
# =============================================================================

def stage2_grid_search(data: dict):
    """Run grid search baseline with PLS, Ridge, LightGBM."""
    print_header("STAGE 2: Grid Search Baseline")

    results = {'regression': {}, 'classification': {}}

    # 2A: Regression
    print_subheader("2A: Regression (%Collagen)")

    models_to_test = ['PLS', 'Ridge', 'LightGBM']

    print(f"Running grid search with models: {models_to_test}")
    print(f"Using 5-fold CV on {len(data['X_train'])} training samples...")

    reg_results, _ = run_search(
        X=data['X_train'],
        y=data['y_reg_train'],
        task_type='regression',
        folds=5,
        tier='standard',
        models_to_test=models_to_test,
        enable_variable_subsets=False,  # Full spectrum only for baseline
        enable_region_subsets=False,
    )

    print(f"\nGrid search completed: {len(reg_results)} configurations tested")

    # Show best results by model type
    print("\nBest results by model type:")
    print("-" * 70)
    print(f"{'Model':<15} {'Preprocess':<20} {'RMSE':<12} {'R2':<12}")
    print("-" * 70)

    for model in models_to_test:
        model_results = reg_results[reg_results['Model'] == model]
        if len(model_results) > 0:
            best = model_results.loc[model_results['RMSE'].idxmin()]
            results['regression'][model] = {
                'rmse': best['RMSE'],
                'r2': best['R2'],
                'preprocessing': best['Preprocess'],
                'full_result': best,
            }
            print(f"{model:<15} {best['Preprocess']:<20} {best['RMSE']:<12.4f} {best['R2']:<12.4f}")

    # 2B: Classification
    print_subheader("2B: Classification (CollagenCat)")

    # Encode labels
    le = LabelEncoder()
    y_cls_encoded = pd.Series(
        le.fit_transform(data['y_cls_train']),
        index=data['y_cls_train'].index
    )

    models_to_test_cls = ['PLS-DA', 'Ridge', 'LightGBM']

    print(f"Running grid search with models: {models_to_test_cls}")
    print(f"Classes: {list(le.classes_)}")

    cls_results, _ = run_search(
        X=data['X_train'],
        y=y_cls_encoded,
        task_type='classification',
        folds=5,
        tier='standard',
        models_to_test=models_to_test_cls,
        enable_variable_subsets=False,
        enable_region_subsets=False,
    )

    print(f"\nGrid search completed: {len(cls_results)} configurations tested")

    # Show best results by model type
    print("\nBest results by model type:")
    print("-" * 70)
    print(f"{'Model':<15} {'Preprocess':<20} {'Accuracy':<12} {'AUC':<12}")
    print("-" * 70)

    for model in models_to_test_cls:
        model_key = 'PLS' if model == 'PLS-DA' else model
        model_results = cls_results[cls_results['Model'].str.contains(model_key, case=False)]
        if len(model_results) > 0:
            best = model_results.loc[model_results['Accuracy'].idxmax()]
            results['classification'][model] = {
                'accuracy': best['Accuracy'],
                'auc': best.get('ROC_AUC', np.nan),
                'preprocessing': best['Preprocess'],
                'full_result': best,
            }
            auc_val = best.get('ROC_AUC', np.nan)
            print(f"{model:<15} {best['Preprocess']:<20} {best['Accuracy']:<12.4f} {auc_val:<12.4f}")

    # Store for later comparison
    results['reg_full'] = reg_results
    results['cls_full'] = cls_results
    results['label_encoder'] = le

    return results


# =============================================================================
# STAGE 3: Variable Selection Methods Testing
# =============================================================================

def stage3_variable_selection(data: dict, baseline_results: dict):
    """Test variable selection methods."""
    print_header("STAGE 3: Variable Selection Methods Testing")

    # Variable selection methods to test (excluding GA per plan)
    varsel_methods = ['importance', 'spa', 'uve', 'uve_spa', 'ipls', 'cars']
    variable_counts = [10, 20, 50, 100, 250]

    results = {}

    print_subheader("3A: Individual Methods (Regression)")
    print(f"Testing methods: {varsel_methods}")
    print(f"Variable counts: {variable_counts}")

    baseline_rmse = min(r['rmse'] for r in baseline_results['regression'].values())
    print(f"\nBaseline (full spectrum) best RMSE: {baseline_rmse:.4f}")

    for method in varsel_methods:
        print(f"\n  Testing {method}...")
        try:
            method_results, _ = run_search(
                X=data['X_train'],
                y=data['y_reg_train'],
                task_type='regression',
                folds=5,
                tier='quick',  # Quick tier for speed
                models_to_test=['PLS', 'Ridge'],
                enable_variable_subsets=True,
                variable_counts=variable_counts,
                variable_selection_methods=[method],
                enable_region_subsets=False,
            )

            # Find best subset result (SubsetTag contains variable selection info)
            subset_results = method_results[method_results['SubsetTag'] != 'full']
            if len(subset_results) > 0:
                best_subset = subset_results.loc[subset_results['RMSE'].idxmin()]
                results[method] = {
                    'best_rmse': best_subset['RMSE'],
                    'best_n_vars': best_subset.get('n_vars', 'unknown'),
                    'model': best_subset['Model'],
                    'beats_baseline': best_subset['RMSE'] < baseline_rmse,
                }

                status = "IMPROVED" if best_subset['RMSE'] < baseline_rmse else "no improvement"
                print(f"    Best: RMSE={best_subset['RMSE']:.4f} ({status})")
            else:
                results[method] = {'error': 'No subset results'}
                print(f"    WARNING: No subset results returned")

        except Exception as e:
            results[method] = {'error': str(e)}
            print(f"    ERROR: {e}")

    # Summary
    print_subheader("Variable Selection Summary")
    print("-" * 60)
    print(f"{'Method':<15} {'Best RMSE':<12} {'Beats Baseline?':<15}")
    print("-" * 60)

    any_improvement = False
    for method, res in results.items():
        if 'error' in res:
            print(f"{method:<15} {'ERROR':<12} {res['error'][:20]}")
        else:
            status = "YES" if res['beats_baseline'] else "no"
            if res['beats_baseline']:
                any_improvement = True
            print(f"{method:<15} {res['best_rmse']:<12.4f} {status:<15}")

    if not any_improvement:
        print("\nWARNING: No variable selection method beat the full-spectrum baseline!")
        print("This could indicate an issue with variable selection implementation.")

    # 3B: Test method combinations
    print_subheader("3B: Method Combinations")

    combo_methods = ['spa', 'uve_spa', 'ipls']
    print(f"Testing combination: {combo_methods}")

    try:
        combo_results, _ = run_search(
            X=data['X_train'],
            y=data['y_reg_train'],
            task_type='regression',
            folds=5,
            tier='quick',
            models_to_test=['PLS', 'Ridge'],
            enable_variable_subsets=True,
            variable_counts=[20, 50, 100],
            variable_selection_methods=combo_methods,
            enable_region_subsets=False,
        )

        print(f"Combination search completed: {len(combo_results)} results")

        # Check if different methods are represented
        methods_found = set()
        for vs in combo_results['SubsetTag'].unique():
            if vs != 'full':
                for m in combo_methods:
                    if m in vs.lower():
                        methods_found.add(m)

        print(f"Methods represented in results: {methods_found}")

        if len(methods_found) < len(combo_methods):
            missing = set(combo_methods) - methods_found
            print(f"WARNING: Missing methods: {missing}")

    except Exception as e:
        print(f"ERROR: {e}")

    return results


# =============================================================================
# STAGE 4: Bayesian Optimization Testing
# =============================================================================

def stage4_bayesian(data: dict, baseline_results: dict):
    """Test Bayesian optimization."""
    print_header("STAGE 4: Bayesian Optimization Testing")

    results = {}

    # Get quick tier baseline for comparison
    print("Running quick-tier grid search for comparison baseline...")
    quick_baseline, _ = run_search(
        X=data['X_train'],
        y=data['y_reg_train'],
        task_type='regression',
        folds=5,
        tier='quick',
        models_to_test=['PLS', 'Ridge'],
        enable_variable_subsets=False,
    )
    quick_best_rmse = quick_baseline['RMSE'].min()
    print(f"Quick tier best RMSE: {quick_best_rmse:.4f}")

    # 4A: Regression
    print_subheader("4A: Bayesian Optimization (Regression)")

    models_to_test = ['PLS', 'Ridge', 'LightGBM']
    print(f"Running Bayesian search with models: {models_to_test}")
    print("n_trials=50 per model...")

    try:
        bayes_results, _ = run_bayesian_search(
            X=data['X_train'],
            y=data['y_reg_train'],
            task_type='regression',
            models_to_test=models_to_test,
            n_trials=50,
            folds=5,
            tier='standard',
        )

        print(f"\nBayesian search completed: {len(bayes_results)} results")

        # Check diversity of results
        models_found = bayes_results['Model'].unique()
        print(f"Models in results: {list(models_found)}")

        # Check if all models appear
        for model in models_to_test:
            if model not in models_found:
                print(f"WARNING: {model} not found in results!")

        # Check preprocessing diversity
        preproc_found = bayes_results['Preprocess'].unique()
        print(f"Preprocessing methods: {list(preproc_found)}")

        # Best result
        best = bayes_results.loc[bayes_results['RMSE'].idxmin()]
        results['regression'] = {
            'best_rmse': best['RMSE'],
            'best_model': best['Model'],
            'best_preprocessing': best['Preprocess'],
            'beats_quick': best['RMSE'] <= quick_best_rmse,
            'n_results': len(bayes_results),
        }

        print(f"\nBest Bayesian result:")
        print(f"  Model: {best['Model']}")
        print(f"  RMSE: {best['RMSE']:.4f}")
        print(f"  Beats quick tier: {'YES' if best['RMSE'] <= quick_best_rmse else 'NO'}")

        if best['RMSE'] > quick_best_rmse:
            print(f"\nWARNING: Bayesian failed to beat quick tier!")
            print(f"  Bayesian: {best['RMSE']:.4f}")
            print(f"  Quick tier: {quick_best_rmse:.4f}")

        # Check parameter diversity (if available)
        if 'Params' in bayes_results.columns:
            print("\nParameter diversity check:")
            for model in models_found:
                model_params = bayes_results[bayes_results['Model'] == model]['Params'].apply(str).unique()
                n_unique = len(model_params)
                print(f"  {model}: {n_unique} unique parameter sets")

    except Exception as e:
        print(f"ERROR: {e}")
        results['regression'] = {'error': str(e)}

    # 4B: Classification
    print_subheader("4B: Bayesian Optimization (Classification)")

    le = LabelEncoder()
    y_cls_encoded = pd.Series(
        le.fit_transform(data['y_cls_train']),
        index=data['y_cls_train'].index
    )

    models_to_test_cls = ['PLS', 'Ridge', 'LightGBM']
    print(f"Running Bayesian search with models: {models_to_test_cls}")

    try:
        bayes_cls_results, _ = run_bayesian_search(
            X=data['X_train'],
            y=y_cls_encoded,
            task_type='classification',
            models_to_test=models_to_test_cls,
            n_trials=50,
            folds=5,
            tier='standard',
        )

        print(f"\nBayesian classification completed: {len(bayes_cls_results)} results")

        best = bayes_cls_results.loc[bayes_cls_results['Accuracy'].idxmax()]
        results['classification'] = {
            'best_accuracy': best['Accuracy'],
            'best_model': best['Model'],
            'n_results': len(bayes_cls_results),
        }

        print(f"\nBest classification result:")
        print(f"  Model: {best['Model']}")
        print(f"  Accuracy: {best['Accuracy']:.4f}")

    except Exception as e:
        print(f"ERROR: {e}")
        results['classification'] = {'error': str(e)}

    return results


# =============================================================================
# STAGE 5: NSGA2 Multi-Objective Testing
# =============================================================================

def stage5_nsga2(data: dict, baseline_results: dict):
    """Test NSGA2 multi-objective optimization."""
    print_header("STAGE 5: NSGA2 Multi-Objective Testing")

    results = {}

    # Get quick tier baseline
    quick_best_rmse = min(r['rmse'] for r in baseline_results['regression'].values())
    print(f"Quick tier best RMSE: {quick_best_rmse:.4f}")

    # 5A: Regression
    print_subheader("5A: NSGA2 (Regression)")

    models = ['PLS', 'Ridge', 'LightGBM']
    print(f"Running NSGA2 with models: {models}")
    print("population_size=50, n_generations=50...")

    try:
        nsga2_results = run_nsga2_search(
            X=data['X_train'].values,
            y=data['y_reg_train'].values,
            task_type='regression',
            models=models,
            population_size=50,
            n_generations=50,
            cv_folds=5,
            random_state=RANDOM_STATE,
            verbose=1,
        )

        # NSGA2 returns dict with 'pareto_front', 'all_solutions', etc.
        if 'pareto_front' in nsga2_results:
            pareto = nsga2_results['pareto_front']
            print(f"\nPareto front size: {len(pareto)}")

            # Check model diversity in Pareto front
            if isinstance(pareto, pd.DataFrame) and 'Model' in pareto.columns:
                models_in_pareto = pareto['Model'].unique()
                print(f"Models in Pareto front: {list(models_in_pareto)}")

                if len(models_in_pareto) < len(models):
                    missing = set(models) - set(models_in_pareto)
                    print(f"WARNING: Missing models: {missing}")

                # Best prediction error
                if 'RMSE' in pareto.columns:
                    best_rmse = pareto['RMSE'].min()
                else:
                    best_rmse = None
                    print("WARNING: Cannot find RMSE column in results")

                if best_rmse is not None:
                    results['regression'] = {
                        'best_rmse': best_rmse,
                        'pareto_size': len(pareto),
                        'models_in_pareto': list(models_in_pareto),
                        'beats_quick': best_rmse <= quick_best_rmse,
                    }

                    print(f"\nBest RMSE in Pareto: {best_rmse:.4f}")
                    print(f"Beats quick tier: {'YES' if best_rmse <= quick_best_rmse else 'NO'}")

                    if best_rmse > quick_best_rmse:
                        print(f"\nWARNING: NSGA2 failed to beat quick tier!")
            else:
                print("Pareto front format unexpected")
                results['regression'] = {'raw_results': nsga2_results}
        else:
            print("No pareto_front in results")
            results['regression'] = nsga2_results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['regression'] = {'error': str(e)}

    # 5B: Classification
    print_subheader("5B: NSGA2 (Classification)")

    le = LabelEncoder()
    y_cls_encoded = le.fit_transform(data['y_cls_train'])

    print(f"Running NSGA2 classification...")

    try:
        nsga2_cls_results = run_nsga2_search(
            X=data['X_train'].values,
            y=y_cls_encoded,
            task_type='classification',
            models=['PLS', 'Ridge', 'LightGBM'],
            population_size=50,
            n_generations=50,
            cv_folds=5,
            random_state=RANDOM_STATE,
            verbose=1,
        )

        if 'pareto_front' in nsga2_cls_results:
            pareto = nsga2_cls_results['pareto_front']
            print(f"Pareto front size: {len(pareto)}")
            results['classification'] = {'pareto_size': len(pareto)}
        else:
            results['classification'] = nsga2_cls_results

    except Exception as e:
        print(f"ERROR: {e}")
        results['classification'] = {'error': str(e)}

    return results


# =============================================================================
# STAGE 6: Model Save/Load Verification
# =============================================================================

def stage6_save_load(data: dict):
    """Verify model save/load works for all model types."""
    print_header("STAGE 6: Model Save/Load Verification")

    model_types = ['PLS', 'Ridge', 'LightGBM', 'ElasticNet', 'RandomForest', 'XGBoost']
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for model_name in model_types:
            print(f"\nTesting {model_name}...")

            try:
                # 1. Create and train model
                model = get_model(model_name, task_type='regression')
                model.fit(data['X_train'].values, data['y_reg_train'].values)

                # 2. Predict on test set
                preds_before = model.predict(data['X_test'].values)

                # 3. Save model
                save_path = tmpdir / f"test_{model_name}.dasp"
                metadata = {
                    'model_name': model_name,
                    'task_type': 'regression',
                    'wavelengths': list(data['X_train'].columns),
                    'n_vars': len(data['X_train'].columns),
                }
                save_model(model, None, metadata, save_path)

                # 4. Load model
                loaded = load_model(save_path)
                loaded_model = loaded['model']

                # 5. Predict with loaded model
                preds_after = loaded_model.predict(data['X_test'].values)

                # 6. Compare predictions
                max_diff = np.abs(preds_before - preds_after).max()
                match = max_diff < 1e-6

                results[model_name] = {
                    'save': True,
                    'load': True,
                    'predictions_match': match,
                    'max_diff': max_diff,
                }

                status = "OK" if match else f"MISMATCH (diff={max_diff:.2e})"
                print(f"  Save: OK, Load: OK, Predictions: {status}")

            except Exception as e:
                results[model_name] = {'error': str(e)}
                print(f"  ERROR: {e}")

        # Test classification model (PLS-DA)
        print(f"\nTesting PLS-DA (classification)...")
        try:
            le = LabelEncoder()
            y_cls_encoded = le.fit_transform(data['y_cls_train'])

            model = get_model('PLS', task_type='classification')
            model.fit(data['X_train'].values, y_cls_encoded)

            preds_before = model.predict(data['X_test'].values)

            save_path = tmpdir / "test_PLS-DA.dasp"
            metadata = {
                'model_name': 'PLS-DA',
                'task_type': 'classification',
                'wavelengths': list(data['X_train'].columns),
                'n_vars': len(data['X_train'].columns),
            }
            save_model(model, None, metadata, save_path, label_encoder=le)

            loaded = load_model(save_path)
            loaded_model = loaded['model']

            preds_after = loaded_model.predict(data['X_test'].values)

            match = np.array_equal(preds_before, preds_after)
            results['PLS-DA'] = {
                'save': True,
                'load': True,
                'predictions_match': match,
                'label_encoder_saved': loaded.get('label_encoder') is not None,
            }

            status = "OK" if match else "MISMATCH"
            le_status = "OK" if loaded.get('label_encoder') is not None else "MISSING"
            print(f"  Save: OK, Load: OK, Predictions: {status}, Label Encoder: {le_status}")

        except Exception as e:
            results['PLS-DA'] = {'error': str(e)}
            print(f"  ERROR: {e}")

    # Summary
    print_subheader("Save/Load Summary")
    print("-" * 60)

    all_ok = True
    for model, res in results.items():
        if 'error' in res:
            print(f"{model}: FAILED - {res['error']}")
            all_ok = False
        elif not res.get('predictions_match', True):
            print(f"{model}: PREDICTION MISMATCH")
            all_ok = False
        else:
            print(f"{model}: OK")

    if all_ok:
        print("\nAll models passed save/load verification!")
    else:
        print("\nWARNING: Some models failed save/load verification!")

    return results


# =============================================================================
# STAGE 7: Holdout Validation
# =============================================================================

def _apply_preprocessing(X, deriv, window, poly):
    """Apply preprocessing based on grid search best config."""
    from spectral_predict.preprocess import SavgolDerivative, SNV, build_preprocessing_pipeline

    if deriv is None or deriv == 0:
        # No derivative, just return as-is (or apply SNV if that was the config)
        return X if not hasattr(X, 'values') else X.values

    # Apply Savitzky-Golay derivative
    sg = SavgolDerivative(deriv=deriv, window=window, polyorder=poly)
    X_arr = X.values if hasattr(X, 'values') else X
    return sg.fit_transform(X_arr)


def stage7_holdout(data: dict, baseline_results: dict):
    """Final validation on holdout test set using BEST config from grid search."""
    print_header("STAGE 7: Holdout Validation")

    results = {}

    print("Training best models from grid search on full training set...")
    print(f"Training: {len(data['X_train'])} samples")
    print(f"Testing: {len(data['X_test'])} samples")
    print("\nNOTE: Using best preprocessing config from grid search for each model.")

    # Regression
    print_subheader("Regression Holdout Results")

    for model_name, baseline in baseline_results['regression'].items():
        try:
            # Get best preprocessing config from grid search result
            best_result = baseline.get('full_result', {})
            deriv = best_result.get('Deriv', 2)
            window = best_result.get('Window', 15)
            poly = best_result.get('Poly', 3)
            n_lvs = best_result.get('LVs', 10)

            print(f"\n{model_name}:")
            print(f"  Best config: deriv={deriv}, window={window}, poly={poly}, LVs={n_lvs}")

            # Apply preprocessing to train and test data
            X_train_prep = _apply_preprocessing(data['X_train'], deriv, window, poly)
            X_test_prep = _apply_preprocessing(data['X_test'], deriv, window, poly)

            # Build model with correct parameters
            if model_name == 'PLS':
                model = get_model('PLS', task_type='regression', n_components=n_lvs)
            else:
                model = get_model(model_name, task_type='regression')

            model.fit(X_train_prep, data['y_reg_train'].values)
            preds = model.predict(X_test_prep)

            # Calculate metrics
            rmse = np.sqrt(np.mean((preds - data['y_reg_test'].values) ** 2))
            ss_res = np.sum((preds - data['y_reg_test'].values) ** 2)
            ss_tot = np.sum((data['y_reg_test'].values - data['y_reg_test'].values.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            results[model_name] = {
                'cv_rmse': baseline['rmse'],
                'holdout_rmse': rmse,
                'holdout_r2': r2,
                'preprocessing': f"deriv={deriv}, w={window}, p={poly}",
            }

            print(f"  CV RMSE (from search): {baseline['rmse']:.4f}")
            print(f"  Holdout RMSE: {rmse:.4f}")
            print(f"  Holdout R2: {r2:.4f}")

            # Check consistency
            if rmse > baseline['rmse'] * 2:
                print(f"  WARNING: Holdout RMSE is much worse than CV!")

        except Exception as e:
            print(f"\n{model_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

    # Classification
    print_subheader("Classification Holdout Results")

    le = LabelEncoder()
    y_cls_train_enc = le.fit_transform(data['y_cls_train'])
    y_cls_test_enc = le.transform(data['y_cls_test'])

    for model_name, baseline in baseline_results['classification'].items():
        try:
            # Get best preprocessing config from grid search result
            best_result = baseline.get('full_result', {})
            deriv = best_result.get('Deriv', 2)
            window = best_result.get('Window', 15)
            poly = best_result.get('Poly', 3)
            n_lvs = best_result.get('LVs', 5)

            print(f"\n{model_name}:")
            print(f"  Best config: deriv={deriv}, window={window}, poly={poly}, LVs={n_lvs}")

            # Apply preprocessing to train and test data
            X_train_prep = _apply_preprocessing(data['X_train'], deriv, window, poly)
            X_test_prep = _apply_preprocessing(data['X_test'], deriv, window, poly)

            # Build model with correct parameters
            model_key = 'PLS' if model_name == 'PLS-DA' else model_name

            if model_key == 'PLS':
                # PLS-DA needs PLSTransformer + LogisticRegression pipeline
                from sklearn.pipeline import Pipeline
                from sklearn.linear_model import LogisticRegression
                from spectral_predict.models import PLSTransformer

                model = Pipeline([
                    ('pls', PLSTransformer(n_components=n_lvs, scale=False)),
                    ('lr', LogisticRegression(max_iter=1000, random_state=42))
                ])
            else:
                model = get_model(model_key, task_type='classification')

            model.fit(X_train_prep, y_cls_train_enc)
            preds = model.predict(X_test_prep)

            accuracy = np.mean(preds == y_cls_test_enc)
            n_correct = int(accuracy * len(y_cls_test_enc))

            results[f"{model_name}_cls"] = {
                'cv_accuracy': baseline['accuracy'],
                'holdout_accuracy': accuracy,
                'n_correct': n_correct,
                'n_total': len(y_cls_test_enc),
                'preprocessing': f"deriv={deriv}, w={window}, p={poly}",
            }

            print(f"  CV Accuracy (from search): {baseline['accuracy']:.4f}")
            print(f"  Holdout Accuracy: {accuracy:.4f} ({n_correct}/{len(y_cls_test_enc)})")

        except Exception as e:
            print(f"\n{model_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results[f"{model_name}_cls"] = {'error': str(e)}

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all test stages."""
    print("\n" + "=" * 70)
    print(" V1 COMPREHENSIVE TESTING")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    # Stage 1
    data = stage1_load_data()
    all_results['stage1'] = {'samples_train': len(data['X_train']),
                             'samples_test': len(data['X_test'])}
    wait_for_approval("Stage 1: Data Loading")

    # Stage 2
    baseline_results = stage2_grid_search(data)
    all_results['stage2'] = baseline_results
    wait_for_approval("Stage 2: Grid Search Baseline")

    # Stage 3
    varsel_results = stage3_variable_selection(data, baseline_results)
    all_results['stage3'] = varsel_results
    wait_for_approval("Stage 3: Variable Selection")

    # Stage 4
    bayes_results = stage4_bayesian(data, baseline_results)
    all_results['stage4'] = bayes_results
    wait_for_approval("Stage 4: Bayesian Optimization")

    # Stage 5
    nsga2_results = stage5_nsga2(data, baseline_results)
    all_results['stage5'] = nsga2_results
    wait_for_approval("Stage 5: NSGA2")

    # Stage 6
    saveload_results = stage6_save_load(data)
    all_results['stage6'] = saveload_results
    wait_for_approval("Stage 6: Save/Load")

    # Stage 7
    holdout_results = stage7_holdout(data, baseline_results)
    all_results['stage7'] = holdout_results

    # Final Summary
    print_header("TESTING COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Flag any issues
    issues = []

    # Check Bayesian
    if 'regression' in bayes_results and 'error' not in bayes_results['regression']:
        if not bayes_results['regression'].get('beats_quick', True):
            issues.append("Bayesian regression failed to beat quick tier")

    # Check NSGA2
    if 'regression' in nsga2_results and 'error' not in nsga2_results['regression']:
        if not nsga2_results['regression'].get('beats_quick', True):
            issues.append("NSGA2 regression failed to beat quick tier")

    # Check save/load
    for model, res in saveload_results.items():
        if 'error' in res:
            issues.append(f"{model} save/load failed")
        elif not res.get('predictions_match', True):
            issues.append(f"{model} predictions don't match after load")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo major issues found!")

    return all_results


if __name__ == "__main__":
    results = main()
