"""Model serialization and persistence for DASP.

This module provides functionality to save and load trained spectral prediction
models with all associated metadata, preprocessing pipelines, and configuration.

File Format:
-----------
Models are saved as .dasp files (ZIP archives) containing:
- metadata.json: Model configuration, wavelengths, performance metrics
- model.pkl: Joblib-serialized sklearn model
- preprocessor.pkl: Joblib-serialized preprocessing pipeline (if applicable)

Example Usage:
-------------
```python
# Save a trained model
save_model(
    model=fitted_pls_model,
    preprocessor=preprocessing_pipeline,
    metadata={
        'model_name': 'PLS',
        'wavelengths': [1500.0, 1520.0, ...],
        'performance': {'R2': 0.987, 'RMSE': 0.125}
    },
    filepath='my_model.dasp'
)

# Load and use the model
model_dict = load_model('my_model.dasp')
predictions = predict_with_model(model_dict, new_X_data)
```
"""

import joblib
import json
import zipfile
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


__version__ = '1.0.0'


def save_model(
    model: Any,
    preprocessor: Optional[Any],
    metadata: Dict[str, Any],
    filepath: Union[str, Path]
) -> None:
    """
    Save a trained model with all metadata to a .dasp file.

    Parameters
    ----------
    model : sklearn estimator or similar
        Fitted model object (e.g., PLSRegression, Ridge, RandomForest, etc.)
    preprocessor : sklearn Pipeline or None
        Fitted preprocessing pipeline (e.g., SNV, derivatives).
        Can be None if model was trained on raw data.
    metadata : dict
        Model metadata. Should include:
        - 'model_name' (str): Model type (e.g., 'PLS', 'Ridge')
        - 'task_type' (str): 'regression' or 'classification'
        - 'preprocessing' (str): Preprocessing method (e.g., 'snv', 'sg1')
        - 'wavelengths' (list): Wavelengths used for training
        - 'n_vars' (int): Number of variables/wavelengths
        - 'performance' (dict): Performance metrics (R2, RMSE, etc.)
        Optional fields:
        - 'window' (int): Savgol window size
        - 'polyorder' (int): Savgol polynomial order
        - 'params' (dict): Model hyperparameters
        - 'training_stats' (dict): Training data statistics
    filepath : str or Path
        Output file path. Will append .dasp extension if not present.

    Raises
    ------
    ValueError
        If metadata is missing required fields
    IOError
        If file cannot be written

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> model = PLSRegression(n_components=5)
    >>> model.fit(X_train, y_train)
    >>> save_model(
    ...     model=model,
    ...     preprocessor=None,
    ...     metadata={
    ...         'model_name': 'PLS',
    ...         'task_type': 'regression',
    ...         'preprocessing': 'raw',
    ...         'wavelengths': [1500.0, 1501.0, ...],
    ...         'n_vars': 800,
    ...         'performance': {'R2': 0.95, 'RMSE': 0.12}
    ...     },
    ...     filepath='my_pls_model.dasp'
    ... )
    """
    # Validate metadata
    required_fields = ['model_name', 'task_type', 'wavelengths', 'n_vars']
    missing_fields = [f for f in required_fields if f not in metadata]
    if missing_fields:
        raise ValueError(f"Metadata missing required fields: {missing_fields}")

    # Add version and timestamp
    metadata_complete = metadata.copy()
    metadata_complete['created'] = datetime.now().isoformat()
    metadata_complete['dasp_version'] = __version__
    metadata_complete['model_class'] = str(type(model).__name__)

    # Ensure filepath has .dasp extension
    filepath = Path(filepath)
    if filepath.suffix != '.dasp':
        filepath = filepath.with_suffix('.dasp')

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Save metadata as JSON
        metadata_path = tmppath / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_complete, f, indent=2, default=_json_serializer)

        # Save model using joblib
        model_path = tmppath / 'model.pkl'
        joblib.dump(model, model_path, compress=3)

        # Save preprocessor if present
        preprocessor_path = tmppath / 'preprocessor.pkl'
        if preprocessor is not None:
            joblib.dump(preprocessor, preprocessor_path, compress=3)

        # Create ZIP archive
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(metadata_path, 'metadata.json')
            zf.write(model_path, 'model.pkl')
            if preprocessor is not None:
                zf.write(preprocessor_path, 'preprocessor.pkl')


def load_model(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a saved model from a .dasp file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .dasp model file

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Fitted model object
        - 'preprocessor': Fitted preprocessing pipeline (or None)
        - 'metadata': Dictionary with all model metadata

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    IOError
        If file cannot be read or is corrupted
    ValueError
        If file format is invalid

    Examples
    --------
    >>> model_dict = load_model('my_pls_model.dasp')
    >>> print(model_dict['metadata']['model_name'])
    'PLS'
    >>> print(model_dict['metadata']['performance'])
    {'R2': 0.95, 'RMSE': 0.12}
    >>> predictions = model_dict['model'].predict(X_new)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    if not zipfile.is_zipfile(filepath):
        raise ValueError(f"File is not a valid .dasp (ZIP) file: {filepath}")

    # Create temporary directory to extract files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Extract all files from ZIP
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(tmppath)

        # Load metadata
        metadata_path = tmppath / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError("Invalid .dasp file: missing metadata.json")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load model
        model_path = tmppath / 'model.pkl'
        if not model_path.exists():
            raise ValueError("Invalid .dasp file: missing model.pkl")

        model = joblib.load(model_path)

        # Load preprocessor if present
        preprocessor = None
        preprocessor_path = tmppath / 'preprocessor.pkl'
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)

    return {
        'model': model,
        'preprocessor': preprocessor,
        'metadata': metadata
    }


def predict_with_model(
    model_dict: Dict[str, Any],
    X_new: Union[pd.DataFrame, np.ndarray],
    validate_wavelengths: bool = True
) -> np.ndarray:
    """
    Make predictions with a loaded model on new spectral data.

    This function handles:
    - Wavelength validation and selection
    - Preprocessing application
    - Prediction generation

    Parameters
    ----------
    model_dict : dict
        Dictionary returned from load_model(), containing:
        - 'model': Fitted model
        - 'preprocessor': Fitted preprocessing pipeline (or None)
        - 'metadata': Model metadata with wavelengths
    X_new : pd.DataFrame or np.ndarray
        New spectral data.
        If DataFrame: columns should be wavelengths (as strings or floats)
        If ndarray: shape should be (n_samples, n_wavelengths) matching
                    the wavelengths in metadata (in correct order)
    validate_wavelengths : bool, default=True
        If True, validate that X_new contains all required wavelengths.
        If False, assume X_new columns/features are in correct order.

    Returns
    -------
    np.ndarray
        Predicted values, shape (n_samples,) for regression or
        (n_samples, n_classes) for classification

    Raises
    ------
    ValueError
        If required wavelengths are missing from X_new
        If X_new has wrong shape/format

    Examples
    --------
    >>> # Load model
    >>> model_dict = load_model('my_model.dasp')
    >>>
    >>> # Load new data
    >>> X_new = pd.read_csv('new_spectra.csv', index_col=0)
    >>>
    >>> # Make predictions
    >>> predictions = predict_with_model(model_dict, X_new)
    >>> print(predictions)
    array([15.2, 18.7, 12.3, ...])
    """
    # Extract components
    model = model_dict['model']
    preprocessor = model_dict['preprocessor']
    metadata = model_dict['metadata']

    # Get required wavelengths from metadata
    if 'wavelengths' not in metadata:
        raise ValueError("Model metadata missing 'wavelengths' field")

    required_wl = metadata['wavelengths']

    # Convert to numpy array if needed
    if isinstance(X_new, pd.DataFrame):
        if validate_wavelengths:
            # Validate and select wavelengths
            X_selected = _select_wavelengths_from_dataframe(X_new, required_wl)
        else:
            X_selected = X_new.values
    elif isinstance(X_new, np.ndarray):
        # Assume array is already in correct format
        if validate_wavelengths and X_new.shape[1] != len(required_wl):
            raise ValueError(
                f"X_new has {X_new.shape[1]} features but model requires "
                f"{len(required_wl)} wavelengths"
            )
        X_selected = X_new
    else:
        raise TypeError(f"X_new must be DataFrame or ndarray, got {type(X_new)}")

    # Apply preprocessing if present
    if preprocessor is not None:
        X_processed = preprocessor.transform(X_selected)
    else:
        X_processed = X_selected

    # Make predictions
    predictions = model.predict(X_processed)

    return predictions


def get_model_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get model information without loading the full model (faster).

    Only loads the metadata.json file, not the model pkl files.

    Parameters
    ----------
    filepath : str or Path
        Path to the .dasp model file

    Returns
    -------
    dict
        Model metadata

    Examples
    --------
    >>> info = get_model_info('my_model.dasp')
    >>> print(f"Model: {info['model_name']}, R²: {info['performance']['R2']}")
    Model: PLS, R²: 0.987
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with zipfile.ZipFile(filepath, 'r') as zf:
        with zf.open('metadata.json') as f:
            metadata = json.load(f)

    return metadata


def _select_wavelengths_from_dataframe(
    df: pd.DataFrame,
    required_wavelengths: list
) -> np.ndarray:
    """
    Select and order wavelengths from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Spectral data with wavelengths as columns
    required_wavelengths : list
        List of required wavelengths (floats)

    Returns
    -------
    np.ndarray
        Selected data in correct order, shape (n_samples, n_wavelengths)

    Raises
    ------
    ValueError
        If required wavelengths are missing
    """
    # Convert DataFrame columns to floats for comparison
    try:
        available_wl = df.columns.astype(float).values
    except (ValueError, TypeError):
        raise ValueError("DataFrame columns must be numeric wavelengths")

    # Check for missing wavelengths
    required_set = set(required_wavelengths)
    available_set = set(available_wl)
    missing_wl = required_set - available_set

    if missing_wl:
        n_missing = len(missing_wl)
        sample_missing = list(missing_wl)[:5]
        raise ValueError(
            f"Missing {n_missing} required wavelengths. "
            f"Examples: {sample_missing}"
        )

    # Select wavelengths in correct order
    # Use string matching to handle floating point comparison
    selected_cols = []
    for required_wl in required_wavelengths:
        # Find matching column (allowing small floating point differences)
        matching_cols = [
            col for col in df.columns
            if abs(float(col) - required_wl) < 0.01
        ]
        if not matching_cols:
            raise ValueError(f"Required wavelength {required_wl} not found")
        selected_cols.append(matching_cols[0])

    return df[selected_cols].values


def _json_serializer(obj):
    """
    Custom JSON serializer for numpy types.

    Parameters
    ----------
    obj : any
        Object to serialize

    Returns
    -------
    serializable object

    Raises
    ------
    TypeError
        If object cannot be serialized
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Module exports
__all__ = [
    'save_model',
    'load_model',
    'predict_with_model',
    'get_model_info'
]
