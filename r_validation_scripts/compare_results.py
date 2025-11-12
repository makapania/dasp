#!/usr/bin/env python3
"""
Compare Python and R model results to validate equivalence.

This script loads results from Python and R model runs and performs
detailed comparisons to ensure they produce equivalent results.

Usage:
    python r_validation_scripts/compare_results.py --model all
    python r_validation_scripts/compare_results.py --model pls_regression
    python r_validation_scripts/compare_results.py --model rf_regression
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr


RESULTS_DIR = Path(__file__).parent / 'results'
PYTHON_DIR = RESULTS_DIR / 'python'
R_DIR = RESULTS_DIR / 'r'


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print("="*80)


def print_subheader(text):
    """Print a formatted subheader."""
    print(f"\n{Colors.BLUE}{text}{Colors.END}")
    print("-"*80)


def print_pass(text):
    """Print a pass message."""
    print(f"{Colors.GREEN}✓ PASS{Colors.END}: {text}")


def print_warning(text):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ WARNING{Colors.END}: {text}")


def print_fail(text):
    """Print a fail message."""
    print(f"{Colors.RED}✗ FAIL{Colors.END}: {text}")


def load_json_results(model_name, source):
    """
    Load JSON results file.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'pls_regression')
    source : str
        'python' or 'r'

    Returns
    -------
    dict
        Results dictionary
    """
    if source == 'python':
        filepath = PYTHON_DIR / f'{model_name}.json'
    else:
        filepath = R_DIR / f'{model_name}.json'

    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def compare_predictions(python_results, r_results, tolerance=1e-6):
    """
    Compare predictions between Python and R.

    Parameters
    ----------
    python_results : dict
        Python results
    r_results : dict
        R results
    tolerance : float
        Maximum allowed difference

    Returns
    -------
    dict
        Comparison metrics
    """
    print_subheader("Prediction Comparison")

    results = {}

    for split in ['train', 'test']:
        py_pred = np.array(python_results['predictions'][split])
        r_pred = np.array(r_results['predictions'][split])

        # Calculate differences
        abs_diff = np.abs(py_pred - r_pred)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        std_diff = np.std(abs_diff)

        # Calculate correlation
        correlation, _ = pearsonr(py_pred, r_pred)

        # Calculate relative error
        rel_error = np.mean(abs_diff / (np.abs(r_pred) + 1e-10)) * 100

        print(f"\n{split.upper()} Predictions:")
        print(f"  Samples: {len(py_pred)}")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  Std absolute difference: {std_diff:.2e}")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Mean relative error: {rel_error:.3f}%")

        # Check if within tolerance
        if max_diff < tolerance:
            print_pass(f"{split} predictions match within tolerance ({tolerance:.2e})")
        elif max_diff < tolerance * 10:
            print_warning(f"{split} predictions differ by {max_diff:.2e} (tolerance: {tolerance:.2e})")
        else:
            print_fail(f"{split} predictions differ significantly: {max_diff:.2e}")

        results[split] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'correlation': correlation,
            'rel_error': rel_error,
            'passes': max_diff < tolerance
        }

    return results


def compare_metrics(python_results, r_results, tolerance=1e-6):
    """
    Compare performance metrics between Python and R.

    Parameters
    ----------
    python_results : dict
        Python results
    r_results : dict
        R results
    tolerance : float
        Maximum allowed difference

    Returns
    -------
    dict
        Comparison results
    """
    print_subheader("Performance Metrics Comparison")

    py_info = python_results['model_info']
    r_info = r_results['model_info']

    metrics = ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']
    results = {}

    for metric in metrics:
        if metric in py_info and metric in r_info:
            py_val = py_info[metric]
            r_val = r_info[metric]
            diff = abs(py_val - r_val)
            rel_diff = diff / (abs(r_val) + 1e-10) * 100

            print(f"\n{metric}:")
            print(f"  Python: {py_val:.6f}")
            print(f"  R:      {r_val:.6f}")
            print(f"  Difference: {diff:.6f} ({rel_diff:.3f}%)")

            if diff < tolerance:
                print_pass(f"{metric} matches within tolerance")
            elif diff < tolerance * 10:
                print_warning(f"{metric} differs by {diff:.6f}")
            else:
                print_fail(f"{metric} differs significantly: {diff:.6f}")

            results[metric] = {
                'python': py_val,
                'r': r_val,
                'diff': diff,
                'rel_diff': rel_diff,
                'passes': diff < tolerance
            }

    return results


def compare_feature_importances(model_name, tolerance=0.1):
    """
    Compare feature importances between Python and R.

    Parameters
    ----------
    model_name : str
        Model name
    tolerance : float
        Minimum required correlation

    Returns
    -------
    dict or None
        Comparison results or None if files don't exist
    """
    py_file = PYTHON_DIR / f'{model_name}_importance.csv'
    r_file = R_DIR / f'{model_name}_importance.csv'

    if not (py_file.exists() and r_file.exists()):
        print_subheader("Feature Importances")
        print("  Importance files not found - skipping comparison")
        return None

    print_subheader("Feature Importances Comparison")

    py_imp = pd.read_csv(py_file)
    r_imp = pd.read_csv(r_file)

    # Get importance values
    py_values = py_imp['importance'].values
    r_values = r_imp['importance'].values

    # Calculate correlation
    correlation, p_value = pearsonr(py_values, r_values)

    # Calculate differences
    abs_diff = np.abs(py_values - r_values)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    # Top features
    n_top = min(10, len(py_values))
    py_top_idx = np.argsort(py_values)[-n_top:][::-1]
    r_top_idx = np.argsort(r_values)[-n_top:][::-1]
    top_overlap = len(set(py_top_idx) & set(r_top_idx))

    print(f"  Features: {len(py_values)}")
    print(f"  Correlation: {correlation:.4f} (p={p_value:.2e})")
    print(f"  Max absolute difference: {max_diff:.4f}")
    print(f"  Mean absolute difference: {mean_diff:.4f}")
    print(f"  Top {n_top} overlap: {top_overlap}/{n_top}")

    if correlation > 0.9:
        print_pass(f"Feature importances highly correlated ({correlation:.4f})")
    elif correlation > tolerance:
        print_warning(f"Feature importances moderately correlated ({correlation:.4f})")
    else:
        print_fail(f"Feature importances poorly correlated ({correlation:.4f})")

    return {
        'correlation': correlation,
        'p_value': p_value,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'top_overlap': top_overlap,
        'n_top': n_top,
        'passes': correlation > tolerance
    }


def compare_coefficients(model_name, tolerance=1e-6):
    """
    Compare model coefficients between Python and R.

    Parameters
    ----------
    model_name : str
        Model name
    tolerance : float
        Maximum allowed difference

    Returns
    -------
    dict or None
        Comparison results or None if files don't exist
    """
    py_file = PYTHON_DIR / f'{model_name}_coefs.csv'
    r_file = R_DIR / f'{model_name}_coefs.csv'

    if not (py_file.exists() and r_file.exists()):
        print_subheader("Coefficients")
        print("  Coefficient files not found - skipping comparison")
        return None

    print_subheader("Coefficients Comparison")

    py_coef = pd.read_csv(py_file)
    r_coef = pd.read_csv(r_file)

    # Get coefficient values
    py_values = py_coef['coefficient'].values
    r_values = r_coef['coefficient'].values

    # Calculate differences
    abs_diff = np.abs(py_values - r_values)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    std_diff = np.std(abs_diff)

    # Calculate correlation
    correlation, _ = pearsonr(py_values, r_values)

    # Non-zero coefficients
    py_nonzero = np.sum(np.abs(py_values) > 1e-10)
    r_nonzero = np.sum(np.abs(r_values) > 1e-10)

    print(f"  Features: {len(py_values)}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Std absolute difference: {std_diff:.2e}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Non-zero coefficients:")
    print(f"    Python: {py_nonzero}/{len(py_values)}")
    print(f"    R: {r_nonzero}/{len(r_values)}")

    if max_diff < tolerance:
        print_pass(f"Coefficients match within tolerance ({tolerance:.2e})")
    elif max_diff < tolerance * 10:
        print_warning(f"Coefficients differ by {max_diff:.2e}")
    else:
        print_fail(f"Coefficients differ significantly: {max_diff:.2e}")

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'correlation': correlation,
        'py_nonzero': py_nonzero,
        'r_nonzero': r_nonzero,
        'passes': max_diff < tolerance * 10  # More lenient for coefficients
    }


def compare_model(model_name, pred_tolerance=1e-6):
    """
    Compare a single model between Python and R.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'pls_regression')
    pred_tolerance : float
        Tolerance for prediction comparison

    Returns
    -------
    dict
        Summary of comparison results
    """
    print_header(f"COMPARING: {model_name.upper()}")

    try:
        # Load results
        print("\nLoading results...")
        python_results = load_json_results(model_name, 'python')
        r_results = load_json_results(model_name, 'r')
        print(f"  Python: {PYTHON_DIR / f'{model_name}.json'}")
        print(f"  R: {R_DIR / f'{model_name}.json'}")

        # Compare predictions
        pred_results = compare_predictions(python_results, r_results, pred_tolerance)

        # Compare metrics
        metric_results = compare_metrics(python_results, r_results, pred_tolerance)

        # Compare feature importances (if available)
        importance_results = compare_feature_importances(model_name)

        # Compare coefficients (if available)
        coef_results = compare_coefficients(model_name)

        # Overall assessment
        print_subheader("Overall Assessment")

        all_pass = True
        for split_results in pred_results.values():
            if not split_results['passes']:
                all_pass = False

        for metric_result in metric_results.values():
            if not metric_result['passes']:
                all_pass = False

        if importance_results and not importance_results['passes']:
            all_pass = False

        if coef_results and not coef_results['passes']:
            all_pass = False

        if all_pass:
            print_pass("All comparisons passed!")
        else:
            print_warning("Some comparisons failed or showed warnings")

        return {
            'model': model_name,
            'predictions': pred_results,
            'metrics': metric_results,
            'importances': importance_results,
            'coefficients': coef_results,
            'all_pass': all_pass
        }

    except FileNotFoundError as e:
        print_fail(f"Missing results file: {e}")
        return None
    except Exception as e:
        print_fail(f"Error comparing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_report(comparison_results):
    """
    Generate a summary report of all comparisons.

    Parameters
    ----------
    comparison_results : list
        List of comparison result dictionaries
    """
    print_header("VALIDATION SUMMARY REPORT")

    print("\n" + "="*80)
    print(f"{'Model':<30} {'Predictions':<15} {'Metrics':<15} {'Overall':<10}")
    print("="*80)

    for result in comparison_results:
        if result is None:
            continue

        model_name = result['model']

        # Check predictions
        pred_status = "✓" if all(r['passes'] for r in result['predictions'].values()) else "✗"

        # Check metrics
        metric_status = "✓" if all(r['passes'] for r in result['metrics'].values()) else "✗"

        # Overall
        overall_status = "PASS" if result['all_pass'] else "FAIL"

        print(f"{model_name:<30} {pred_status:<15} {metric_status:<15} {overall_status:<10}")

    print("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare Python and R model results')
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        help='Model to compare (e.g., pls_regression, rf_regression) or "all"'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Tolerance for prediction comparison'
    )

    args = parser.parse_args()

    # List of models to compare
    if args.model == 'all':
        models = [
            'pls_regression',
            'rf_regression',
            'xgb_regression',
            'ridge_regression',
            'lasso_regression',
            'elasticnet_regression'
        ]
    else:
        models = [args.model]

    # Compare each model
    comparison_results = []
    for model in models:
        result = compare_model(model, pred_tolerance=args.tolerance)
        if result:
            comparison_results.append(result)

    # Generate summary report
    if len(comparison_results) > 1:
        generate_report(comparison_results)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nCompared {len(comparison_results)} model(s)")
    print(f"Passed: {sum(1 for r in comparison_results if r['all_pass'])}/{len(comparison_results)}")


if __name__ == '__main__':
    main()
