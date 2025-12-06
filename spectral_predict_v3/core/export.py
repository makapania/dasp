"""
Export utilities for Spectral Predict v3.

Provides functions to export:
- Results tables to CSV/Excel
- Plots to PNG
- Preprocessed data to CSV
- Model predictions to files
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings


def export_results_to_csv(
    results: pd.DataFrame,
    filepath: str,
    include_index: bool = True
) -> bool:
    """
    Export results DataFrame to CSV file.

    Parameters
    ----------
    results : pd.DataFrame
        Results data to export
    filepath : str
        Output file path
    include_index : bool
        Whether to include the index column

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        # Ensure .csv extension
        if filepath.suffix.lower() != '.csv':
            filepath = filepath.with_suffix('.csv')

        # Export to CSV
        results.to_csv(filepath, index=include_index, encoding='utf-8')

        print(f"Results exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False


def export_results_to_excel(
    results: pd.DataFrame,
    filepath: str,
    sheet_name: str = 'Results',
    include_index: bool = True,
    additional_sheets: Optional[Dict[str, pd.DataFrame]] = None
) -> bool:
    """
    Export results DataFrame to Excel file with optional multiple sheets.

    Parameters
    ----------
    results : pd.DataFrame
        Main results data to export
    filepath : str
        Output file path
    sheet_name : str
        Name for the main results sheet
    include_index : bool
        Whether to include the index column
    additional_sheets : dict of {str: pd.DataFrame}, optional
        Additional sheets to include {sheet_name: dataframe}

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        # Ensure .xlsx extension
        if filepath.suffix.lower() not in ['.xlsx', '.xls']:
            filepath = filepath.with_suffix('.xlsx')

        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Write main results
            results.to_excel(writer, sheet_name=sheet_name, index=include_index)

            # Write additional sheets if provided
            if additional_sheets:
                for name, df in additional_sheets.items():
                    df.to_excel(writer, sheet_name=name, index=include_index)

        print(f"Results exported to Excel: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False


def export_predictions_to_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepath: str,
    sample_names: Optional[List[str]] = None,
    include_residuals: bool = True
) -> bool:
    """
    Export predictions and actual values to CSV.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    filepath : str
        Output file path
    sample_names : list of str, optional
        Sample names for the index
    include_residuals : bool
        Whether to include residuals column

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.csv':
            filepath = filepath.with_suffix('.csv')

        # Create DataFrame
        data = {
            'Actual': y_true,
            'Predicted': y_pred
        }

        if include_residuals:
            data['Residual'] = y_true - y_pred

        df = pd.DataFrame(data)

        # Add sample names if provided
        if sample_names is not None:
            df.index = sample_names

        # Export
        df.to_csv(filepath, index=True, encoding='utf-8')

        print(f"Predictions exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting predictions: {e}")
        return False


def export_preprocessed_data_to_csv(
    X: np.ndarray,
    wavelengths: np.ndarray,
    filepath: str,
    sample_names: Optional[List[str]] = None,
    y: Optional[np.ndarray] = None,
    y_name: str = 'Target'
) -> bool:
    """
    Export preprocessed spectral data to CSV.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_wavelengths)
    wavelengths : np.ndarray
        Wavelength values for column headers
    filepath : str
        Output file path
    sample_names : list of str, optional
        Sample names for row index
    y : np.ndarray, optional
        Target values to include as first column
    y_name : str
        Column name for target values

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.csv':
            filepath = filepath.with_suffix('.csv')

        # Create column names from wavelengths
        columns = [f"{w:.2f}" for w in wavelengths]

        # Create DataFrame
        df = pd.DataFrame(X, columns=columns)

        # Add target values if provided
        if y is not None:
            df.insert(0, y_name, y)

        # Add sample names if provided
        if sample_names is not None:
            df.index = sample_names

        # Export
        df.to_csv(filepath, index=True, encoding='utf-8')

        print(f"Preprocessed data exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting preprocessed data: {e}")
        return False


def export_variable_selection_to_csv(
    selected_indices: np.ndarray,
    wavelengths: np.ndarray,
    importances: Optional[np.ndarray] = None,
    filepath: str = 'selected_variables.csv'
) -> bool:
    """
    Export selected variables to CSV.

    Parameters
    ----------
    selected_indices : np.ndarray
        Indices of selected variables
    wavelengths : np.ndarray
        All wavelength values
    importances : np.ndarray, optional
        Importance scores for selected variables
    filepath : str
        Output file path

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.csv':
            filepath = filepath.with_suffix('.csv')

        # Create data dictionary
        data = {
            'Index': selected_indices,
            'Wavelength': wavelengths[selected_indices]
        }

        if importances is not None:
            data['Importance'] = importances[selected_indices]

        df = pd.DataFrame(data)

        # Export
        df.to_csv(filepath, index=False, encoding='utf-8')

        print(f"Selected variables exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting variable selection: {e}")
        return False


def export_plot_data_to_csv(
    plot_data: Dict[str, np.ndarray],
    filepath: str
) -> bool:
    """
    Export plot data (e.g., ROC curve, learning curve) to CSV.

    Parameters
    ----------
    plot_data : dict
        Dictionary of arrays to export {column_name: array}
    filepath : str
        Output file path

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.csv':
            filepath = filepath.with_suffix('.csv')

        # Create DataFrame
        df = pd.DataFrame(plot_data)

        # Export
        df.to_csv(filepath, index=False, encoding='utf-8')

        print(f"Plot data exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting plot data: {e}")
        return False


def export_confusion_matrix_to_csv(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    filepath: str = 'confusion_matrix.csv'
) -> bool:
    """
    Export confusion matrix to CSV.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes)
    class_names : list of str, optional
        Class names for labels
    filepath : str
        Output file path

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.csv':
            filepath = filepath.with_suffix('.csv')

        # Create class names if not provided
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(cm.shape[0])]

        # Create DataFrame with class names
        df = pd.DataFrame(cm, index=class_names, columns=class_names)

        # Add row labels
        df.index.name = 'True \\ Predicted'

        # Export
        df.to_csv(filepath, encoding='utf-8')

        print(f"Confusion matrix exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting confusion matrix: {e}")
        return False


def export_model_summary(
    model_info: Dict[str, Any],
    filepath: str
) -> bool:
    """
    Export model summary information to text file.

    Parameters
    ----------
    model_info : dict
        Dictionary containing model information:
        - 'model_name': str
        - 'hyperparameters': dict
        - 'performance': dict (e.g., {'R2': 0.95, 'RMSE': 0.05})
        - 'n_features': int
        - 'n_samples': int
        - etc.
    filepath : str
        Output file path

    Returns
    -------
    bool
        True if export succeeded
    """
    try:
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.txt':
            filepath = filepath.with_suffix('.txt')

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            # Model name
            if 'model_name' in model_info:
                f.write(f"Model: {model_info['model_name']}\n\n")

            # Hyperparameters
            if 'hyperparameters' in model_info:
                f.write("Hyperparameters:\n")
                for key, value in model_info['hyperparameters'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Dataset info
            if 'n_samples' in model_info or 'n_features' in model_info:
                f.write("Dataset:\n")
                if 'n_samples' in model_info:
                    f.write(f"  Samples: {model_info['n_samples']}\n")
                if 'n_features' in model_info:
                    f.write(f"  Features: {model_info['n_features']}\n")
                f.write("\n")

            # Performance metrics
            if 'performance' in model_info:
                f.write("Performance Metrics:\n")
                for key, value in model_info['performance'].items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Additional info
            for key, value in model_info.items():
                if key not in ['model_name', 'hyperparameters', 'performance', 'n_samples', 'n_features']:
                    f.write(f"{key}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"Model summary exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error exporting model summary: {e}")
        return False


# Batch export function
def export_all_results(
    results_dir: str,
    results_df: pd.DataFrame,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    wavelengths: Optional[np.ndarray] = None,
    model_info: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Export all available results to a directory.

    Creates a directory and exports:
    - results.csv (main results table)
    - results.xlsx (Excel format)
    - predictions.csv (if y_true and y_pred provided)
    - preprocessed_data.csv (if X and wavelengths provided)
    - model_summary.txt (if model_info provided)

    Parameters
    ----------
    results_dir : str
        Directory to create and export to
    results_df : pd.DataFrame
        Main results DataFrame
    y_true : np.ndarray, optional
        True values
    y_pred : np.ndarray, optional
        Predictions
    X : np.ndarray, optional
        Preprocessed spectral data
    wavelengths : np.ndarray, optional
        Wavelength values
    model_info : dict, optional
        Model information

    Returns
    -------
    bool
        True if all exports succeeded
    """
    try:
        # Create directory
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        success = True

        # Export main results
        success &= export_results_to_csv(
            results_df,
            results_path / 'results.csv'
        )
        success &= export_results_to_excel(
            results_df,
            results_path / 'results.xlsx'
        )

        # Export predictions if available
        if y_true is not None and y_pred is not None:
            success &= export_predictions_to_csv(
                y_true, y_pred,
                results_path / 'predictions.csv'
            )

        # Export preprocessed data if available
        if X is not None and wavelengths is not None:
            success &= export_preprocessed_data_to_csv(
                X, wavelengths,
                results_path / 'preprocessed_data.csv'
            )

        # Export model summary if available
        if model_info is not None:
            success &= export_model_summary(
                model_info,
                results_path / 'model_summary.txt'
            )

        if success:
            print(f"\nAll results exported to: {results_path}")

        return success

    except Exception as e:
        print(f"Error in batch export: {e}")
        return False
