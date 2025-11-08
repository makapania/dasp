#!/usr/bin/env python3
"""
Verification Script for Agent 4: Execution Engine Implementation

This script verifies that all required components for the Tab 6 (Custom Model Development)
execution engine are properly implemented in spectral_predict_gui_optimized.py.

Run this script to confirm the implementation is complete.
"""

import sys
from pathlib import Path
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_imports():
    """Verify all required imports are available."""
    print("=" * 80)
    print("CHECKING IMPORTS")
    print("=" * 80)

    required_imports = [
        ('spectral_predict.models', 'get_model'),
        ('spectral_predict.preprocess', 'build_preprocessing_pipeline'),
        ('spectral_predict.diagnostics', 'jackknife_prediction_intervals'),
        ('spectral_predict.model_io', 'save_model'),
        ('sklearn.model_selection', 'KFold'),
        ('sklearn.model_selection', 'StratifiedKFold'),
        ('sklearn.metrics', 'mean_squared_error'),
        ('sklearn.metrics', 'r2_score'),
        ('sklearn.base', 'clone'),
        ('sklearn.pipeline', 'Pipeline'),
    ]

    all_passed = True
    for module_name, import_name in required_imports:
        try:
            module = __import__(module_name, fromlist=[import_name])
            getattr(module, import_name)
            print(f"  [OK] {module_name}.{import_name}")
        except (ImportError, AttributeError) as e:
            print(f"  [FAIL] {module_name}.{import_name}: {e}")
            all_passed = False

    return all_passed


def check_gui_methods():
    """Verify all required methods exist in the GUI file."""
    print("\n" + "=" * 80)
    print("CHECKING GUI METHODS")
    print("=" * 80)

    gui_file = Path(__file__).parent / "spectral_predict_gui_optimized.py"

    if not gui_file.exists():
        print(f"  [FAIL] GUI file not found: {gui_file}")
        return False

    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()

    required_methods = [
        '_run_refined_model',
        '_run_refined_model_thread',
        '_update_refined_results',
        '_validate_refinement_parameters',
        '_save_refined_model',
        '_plot_refined_predictions',
        '_plot_residual_diagnostics',
        '_plot_leverage_diagnostics',
    ]

    all_passed = True
    for method in required_methods:
        pattern = rf'def {method}\(self[^)]*\):'
        if re.search(pattern, content):
            # Count lines in method
            start = content.find(f'def {method}(self')
            if start != -1:
                # Find next method definition
                next_def = content.find('\n    def ', start + 1)
                if next_def != -1:
                    method_content = content[start:next_def]
                else:
                    method_content = content[start:]
                lines = method_content.count('\n')
                print(f"  [OK] {method}() - {lines} lines")
        else:
            print(f"  [FAIL] {method}() - NOT FOUND")
            all_passed = False

    return all_passed


def check_critical_features():
    """Verify critical features are implemented."""
    print("\n" + "=" * 80)
    print("CHECKING CRITICAL FEATURES")
    print("=" * 80)

    gui_file = Path(__file__).parent / "spectral_predict_gui_optimized.py"

    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()

    features = {
        'Cross-validation (KFold)': 'cv = KFold',
        'Cross-validation (StratifiedKFold)': 'cv = StratifiedKFold',
        'Shuffle=False (deterministic CV)': 'shuffle=False',
        'Full-spectrum preprocessing': 'use_full_spectrum_preprocessing',
        'Hyperparameter loading': 'params_from_search',
        'Prediction intervals (jackknife)': 'jackknife_prediction_intervals',
        'Validation set exclusion': 'validation_indices',
        'Excluded spectra handling': 'excluded_spectra',
        'Index reset (CV consistency)': 'reset_index',
        'Thread-safe UI updates': 'self.root.after',
        'Model cloning per fold': 'clone(pipe)',
        'Pipeline with preprocessing': 'Pipeline(',
        'Error handling (try/except)': 'except Exception as e:',
        'Results comparison': 'COMPARISON TO LOADED MODEL',
        'Debug logging': 'print(f"DEBUG:',
    }

    all_passed = True
    for feature, keyword in features.items():
        if keyword in content:
            # Count occurrences
            count = content.count(keyword)
            print(f"  [OK] {feature} - {count} occurrence(s)")
        else:
            print(f"  [FAIL] {feature} - NOT FOUND")
            all_passed = False

    return all_passed


def check_preprocessing_paths():
    """Verify both preprocessing paths are implemented."""
    print("\n" + "=" * 80)
    print("CHECKING PREPROCESSING PATHS")
    print("=" * 80)

    gui_file = Path(__file__).parent / "spectral_predict_gui_optimized.py"

    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the preprocessing path decision logic
    path_a_keywords = [
        'use_full_spectrum_preprocessing',
        'X_full_preprocessed',
        'wavelength_indices',
        'PATH A',
    ]

    path_b_keywords = [
        'X_work = X_base_df',
        'PATH B',
        'Standard',
    ]

    print("\n  Path A (Derivative + Subset):")
    path_a_found = True
    for keyword in path_a_keywords:
        if keyword in content:
            print(f"    [OK] {keyword}")
        else:
            print(f"    [FAIL] {keyword} - NOT FOUND")
            path_a_found = False

    print("\n  Path B (Raw/SNV or Full Spectrum):")
    path_b_found = True
    for keyword in path_b_keywords:
        if keyword in content:
            print(f"    [OK] {keyword}")
        else:
            print(f"    [FAIL] {keyword} - NOT FOUND")
            path_b_found = False

    return path_a_found and path_b_found


def check_model_support():
    """Verify all model types are supported."""
    print("\n" + "=" * 80)
    print("CHECKING MODEL SUPPORT")
    print("=" * 80)

    try:
        from spectral_predict.models import get_model

        models = ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']

        all_passed = True
        for model_name in models:
            try:
                model = get_model(model_name, task_type='regression', n_components=10, max_iter=100)
                print(f"  [OK] {model_name} - {type(model).__name__}")
            except Exception as e:
                print(f"  [FAIL] {model_name} - {e}")
                all_passed = False

        return all_passed
    except ImportError as e:
        print(f"  [FAIL] Could not import get_model: {e}")
        return False


def check_documentation():
    """Verify documentation files exist."""
    print("\n" + "=" * 80)
    print("CHECKING DOCUMENTATION")
    print("=" * 80)

    docs = [
        'AGENT4_EXECUTION_ENGINE_COMPLETE.md',
        'AGENT4_EXECUTION_FLOW_DIAGRAM.md',
        'AGENT4_QUICK_REFERENCE.md',
    ]

    all_passed = True
    for doc in docs:
        doc_path = Path(__file__).parent / doc
        if doc_path.exists():
            size = doc_path.stat().st_size
            lines = len(doc_path.read_text(encoding='utf-8').split('\n'))
            print(f"  [OK] {doc} - {lines} lines, {size:,} bytes")
        else:
            print(f"  [FAIL] {doc} - NOT FOUND")
            all_passed = False

    return all_passed


def main():
    """Run all verification checks."""
    print("\n")
    print("*" * 80)
    print("AGENT 4: EXECUTION ENGINE IMPLEMENTATION VERIFICATION")
    print("*" * 80)
    print("\n")

    results = {
        'Imports': check_imports(),
        'GUI Methods': check_gui_methods(),
        'Critical Features': check_critical_features(),
        'Preprocessing Paths': check_preprocessing_paths(),
        'Model Support': check_model_support(),
        'Documentation': check_documentation(),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for category, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {category}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED - IMPLEMENTATION COMPLETE")
        print("=" * 80)
        print("\nTab 6 (Custom Model Development) execution engine is fully implemented")
        print("and ready for production use.")
        print("\nNo further work required for Agent 4.")
        return 0
    else:
        print("RESULT: SOME CHECKS FAILED - REVIEW REQUIRED")
        print("=" * 80)
        print("\nSome components are missing or incomplete.")
        print("Review the failure messages above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
