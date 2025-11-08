"""
Results DataFrame Validation Module

This module provides validation functions to ensure that Results DataFrames
contain all critical fields needed for Model Development Tab 7.

Usage:
    from validate_results_dataframe import validate_results_dataframe, print_validation_report

    # Validate a results DataFrame
    report = validate_results_dataframe(results_df, task_type="regression", backend="python")
    if not report['valid']:
        print(f"Missing fields: {report['missing']}")

    # Print formatted report
    print_validation_report(results_df)
"""

import pandas as pd


def validate_results_dataframe(results_df, task_type="regression", backend="python"):
    """
    Validate that a results DataFrame contains all fields needed for Model Development.

    Parameters
    ----------
    results_df : pd.DataFrame
        The results DataFrame to validate
    task_type : str, default="regression"
        Either "regression" or "classification"
    backend : str, default="python"
        Either "python" or "julia"

    Returns
    -------
    dict
        Validation report with keys:
        - 'valid': bool (True if all required fields present)
        - 'missing': list of str (missing field names)
        - 'present': list of str (present field names)
        - 'warnings': list of str (non-critical issues)
        - 'summary': str (one-line summary)

    Examples
    --------
    >>> import pandas as pd
    >>> results_df = pd.read_csv("results.csv")
    >>> report = validate_results_dataframe(results_df)
    >>> print(report['summary'])
    ✅ VALID - 26/26 required fields present

    >>> if not report['valid']:
    ...     print(f"Missing: {', '.join(report['missing'])}")
    """
    # Core fields required for ALL models
    core_fields = [
        "Model",           # Model type (PLS, Ridge, RandomForest, etc.)
        "Preprocess",      # Preprocessing method (raw, snv, deriv, etc.)
        "Deriv",           # Derivative order (None, 1, 2)
        "Window",          # Savitzky-Golay window size
        "SubsetTag",       # Subset identifier (full, top10, top50, etc.)
        "n_vars",          # Number of wavelengths used
        "full_vars",       # Total available wavelengths
        "all_vars",        # Complete wavelength list (CSV string) - CRITICAL for loading
        "n_folds",         # CV fold count - CRITICAL for reproducibility
    ]

    # Task-specific metrics
    if task_type == "regression":
        metric_fields = ["R2", "RMSE"]
    else:  # classification
        metric_fields = ["Accuracy", "ROC_AUC"]

    # Model-specific hyperparameters (optional but recommended)
    # These are used to reproduce exact model configurations
    model_hyperparameters = {
        "PLS": ["LVs"],                                      # Number of latent variables
        "Ridge": ["Alpha"],                                  # Regularization strength
        "Lasso": ["Alpha"],                                  # Regularization strength
        "RandomForest": ["n_estimators", "max_depth"],       # Trees, depth
        "MLP": ["Hidden", "LR_init"],                        # Layer sizes, learning rate
        "NeuralBoosted": ["n_estimators", "LearningRate", "HiddenSize", "Activation"],
    }

    required_fields = core_fields + metric_fields

    # Check which fields are present
    present = [f for f in required_fields if f in results_df.columns]
    missing = [f for f in required_fields if f not in results_df.columns]

    warnings = []

    # Check model-specific hyperparameters (non-critical but recommended)
    if "Model" in results_df.columns:
        for model_type in results_df["Model"].unique():
            if model_type in model_hyperparameters:
                for hyperparam in model_hyperparameters[model_type]:
                    if hyperparam not in results_df.columns:
                        warnings.append(
                            f"Model-specific field '{hyperparam}' missing for {model_type} "
                            f"(non-critical but recommended)"
                        )

    # Backend-specific checks
    if backend == "python" and "n_folds" not in results_df.columns:
        if "n_folds" not in missing:
            missing.append("n_folds")
        warnings.append(
            "CRITICAL: Python backend missing 'n_folds' field - "
            "this will cause issues with Model Development reproducibility"
        )

    # Check for empty DataFrame
    if len(results_df) == 0:
        warnings.append("WARNING: Results DataFrame is empty (no rows)")

    # Check for NaN values in critical fields
    critical_fields = ["Model", "all_vars", "n_vars"]
    for field in critical_fields:
        if field in results_df.columns:
            nan_count = results_df[field].isna().sum()
            if nan_count > 0:
                warnings.append(
                    f"WARNING: {nan_count} NaN values found in critical field '{field}'"
                )

    valid = len(missing) == 0

    return {
        'valid': valid,
        'missing': missing,
        'present': present,
        'warnings': warnings,
        'summary': f"{'✅ VALID' if valid else '❌ INVALID'} - "
                  f"{len(present)}/{len(required_fields)} required fields present"
    }


def print_validation_report(results_df, task_type="regression", backend="python"):
    """
    Print a formatted validation report to console.

    Parameters
    ----------
    results_df : pd.DataFrame
        The results DataFrame to validate
    task_type : str, default="regression"
        Either "regression" or "classification"
    backend : str, default="python"
        Either "python" or "julia"

    Returns
    -------
    dict
        Validation report (same as validate_results_dataframe)

    Examples
    --------
    >>> results_df = pd.read_csv("results.csv")
    >>> report = print_validation_report(results_df)
    ====================================================================
    RESULTS DATAFRAME VALIDATION REPORT
    ====================================================================
    Status: ✅ VALID - 26/26 required fields present
    ...
    """
    report = validate_results_dataframe(results_df, task_type, backend)

    print("\n" + "="*70)
    print("RESULTS DATAFRAME VALIDATION REPORT")
    print("="*70)
    print(f"\nBackend: {backend}")
    print(f"Task Type: {task_type}")
    print(f"Rows: {len(results_df)}")
    print(f"Columns: {len(results_df.columns)}")
    print(f"\nStatus: {report['summary']}\n")

    if report['present']:
        print(f"✅ Present Fields ({len(report['present'])}):")
        for field in sorted(report['present']):
            # Show sample value if available
            if len(results_df) > 0:
                sample_val = results_df[field].iloc[0]
                if pd.isna(sample_val):
                    val_str = "(NaN)"
                elif isinstance(sample_val, str) and len(str(sample_val)) > 30:
                    val_str = f"(e.g., {str(sample_val)[:27]}...)"
                else:
                    val_str = f"(e.g., {sample_val})"
            else:
                val_str = ""
            print(f"   • {field:20s} {val_str}")

    if report['missing']:
        print(f"\n❌ Missing Fields ({len(report['missing'])}):")
        for field in sorted(report['missing']):
            print(f"   • {field}")

    if report['warnings']:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"   • {warning}")

    print("\n" + "="*70 + "\n")

    return report


def check_model_loading_readiness(results_df, row_index):
    """
    Check if a specific row has all data needed to load a model in Tab 7.

    Parameters
    ----------
    results_df : pd.DataFrame
        The results DataFrame
    row_index : int
        Index of the row to check

    Returns
    -------
    dict
        Readiness report with keys:
        - 'ready': bool (True if row can be loaded)
        - 'missing': list of str (missing fields)
        - 'issues': list of str (data quality issues)

    Examples
    --------
    >>> readiness = check_model_loading_readiness(results_df, 0)
    >>> if readiness['ready']:
    ...     print("✅ Model ready to load")
    ... else:
    ...     print(f"❌ Missing: {readiness['missing']}")
    """
    # Critical fields for model loading
    critical_for_loading = [
        "all_vars",      # Wavelength list
        "n_folds",       # CV configuration
        "Model",         # Model type
        "Preprocess",    # Preprocessing method
        "Deriv",         # Derivative order
        "Window",        # SG window size
    ]

    if row_index >= len(results_df):
        return {
            'ready': False,
            'missing': [],
            'issues': [f"Row index {row_index} out of range (DataFrame has {len(results_df)} rows)"]
        }

    row = results_df.iloc[row_index]

    missing = []
    issues = []

    for field in critical_for_loading:
        if field not in results_df.columns:
            missing.append(field)
        elif pd.isna(row[field]):
            issues.append(f"Field '{field}' is NaN")

    # Check all_vars format (should be comma-separated list)
    if "all_vars" in results_df.columns and not pd.isna(row["all_vars"]):
        all_vars_str = str(row["all_vars"])
        if all_vars_str == "N/A":
            issues.append("all_vars is 'N/A' - cannot load wavelengths")
        elif "," not in all_vars_str and len(all_vars_str) > 0:
            # Single wavelength is okay, but check if it's a valid number
            try:
                float(all_vars_str)
            except ValueError:
                issues.append(f"all_vars has invalid format: '{all_vars_str}'")

    # Check n_folds is a valid integer
    if "n_folds" in results_df.columns and not pd.isna(row["n_folds"]):
        try:
            n_folds_val = int(row["n_folds"])
            if n_folds_val < 2:
                issues.append(f"n_folds={n_folds_val} is invalid (must be >= 2)")
        except (ValueError, TypeError):
            issues.append(f"n_folds has invalid value: {row['n_folds']}")

    ready = len(missing) == 0 and len(issues) == 0

    return {
        'ready': ready,
        'missing': missing,
        'issues': issues
    }


# Command-line interface for testing
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Validate a Results DataFrame CSV file")
    parser.add_argument("csv_file", help="Path to results CSV file")
    parser.add_argument("--task", default="regression", choices=["regression", "classification"],
                       help="Task type (default: regression)")
    parser.add_argument("--backend", default="python", choices=["python", "julia"],
                       help="Backend used (default: python)")
    parser.add_argument("--check-row", type=int, metavar="INDEX",
                       help="Check if specific row is ready for model loading")

    args = parser.parse_args()

    try:
        results_df = pd.read_csv(args.csv_file)
        print(f"\nLoaded results from: {args.csv_file}")

        # Print validation report
        report = print_validation_report(results_df, task_type=args.task, backend=args.backend)

        # Check specific row if requested
        if args.check_row is not None:
            print("\n" + "="*70)
            print(f"MODEL LOADING READINESS CHECK (Row {args.check_row})")
            print("="*70)
            readiness = check_model_loading_readiness(results_df, args.check_row)

            if readiness['ready']:
                print("✅ READY - This model can be loaded in Model Development Tab 7")
            else:
                print("❌ NOT READY - Issues must be resolved before loading")

                if readiness['missing']:
                    print(f"\nMissing fields:")
                    for field in readiness['missing']:
                        print(f"   • {field}")

                if readiness['issues']:
                    print(f"\nData issues:")
                    for issue in readiness['issues']:
                        print(f"   • {issue}")

            print("="*70 + "\n")

        # Exit code based on validation
        sys.exit(0 if report['valid'] else 1)

    except FileNotFoundError:
        print(f"ERROR: File not found: {args.csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
