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
    filepath: Union[str, Path],
    label_encoder: Optional[Any] = None,
    cv_residuals: Optional[np.ndarray] = None,
    cv_predictions: Optional[np.ndarray] = None,
    cv_actuals: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None
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
    label_encoder : sklearn.preprocessing.LabelEncoder or None
        Label encoder for classification with text labels (e.g., "low", "medium", "high").
        Used to convert between text labels and numeric codes.
    cv_residuals : np.ndarray or None
        Cross-validation residuals (predictions - actuals) for uncertainty estimation.
        Shape: (n_cv_samples,) for regression or (n_cv_samples, n_classes) for classification probabilities.
    cv_predictions : np.ndarray or None
        Cross-validation predictions for uncertainty analysis.
        Shape: (n_cv_samples,) for regression or (n_cv_samples,) for classification.
    cv_actuals : np.ndarray or None
        Cross-validation actual values for uncertainty analysis.
        Shape: (n_cv_samples,)
    X_train : np.ndarray or None
        Training data (preprocessed) for applicability domain assessment.
        Shape: (n_samples, n_features)
        Used to store representative spectra and fit PCA for distance calculations.
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

    # Add label encoder information if present
    if label_encoder is not None:
        metadata_complete['has_label_encoder'] = True
        metadata_complete['label_classes'] = label_encoder.classes_.tolist()
        metadata_complete['label_mapping'] = dict(zip(
            label_encoder.classes_,
            label_encoder.transform(label_encoder.classes_).tolist()
        ))
    else:
        metadata_complete['has_label_encoder'] = False

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

        # Save label_encoder if present
        label_encoder_path = tmppath / 'label_encoder.pkl'
        if label_encoder is not None:
            joblib.dump(label_encoder, label_encoder_path, compress=3)

        # Save CV data if present (for uncertainty estimation)
        cv_data_path = tmppath / 'cv_data.npz'
        if cv_residuals is not None or cv_predictions is not None or cv_actuals is not None:
            cv_data_dict = {}
            if cv_residuals is not None:
                cv_data_dict['cv_residuals'] = cv_residuals
            if cv_predictions is not None:
                cv_data_dict['cv_predictions'] = cv_predictions
            if cv_actuals is not None:
                cv_data_dict['cv_actuals'] = cv_actuals
            np.savez_compressed(cv_data_path, **cv_data_dict)
            metadata_complete['has_cv_data'] = True
        else:
            metadata_complete['has_cv_data'] = False

        # Save applicability domain data if training data provided
        ad_data_path = tmppath / 'applicability_domain.npz'
        if X_train is not None:
            from sklearn.decomposition import PCA
            from scipy.spatial.distance import pdist

            n_samples, n_features = X_train.shape

            # Adaptive representative selection
            if n_samples <= 100:
                # Store all training spectra for small datasets
                representative_spectra = X_train
                representative_indices = np.arange(n_samples)
                print(f"Applicability domain: storing all {n_samples} training spectra")
            else:
                # Use Kennard-Stone to select ~150 representative samples for large datasets
                from src.spectral_predict.sample_selection import kennard_stone
                n_representatives = min(150, n_samples)
                representative_indices = kennard_stone(X_train, n_samples=n_representatives)
                representative_spectra = X_train[representative_indices]
                print(f"Applicability domain: selected {n_representatives} representative spectra from {n_samples} using Kennard-Stone")

            # Fit PCA for dimensionality reduction (capture ~99% variance)
            # Use min to avoid having more components than samples or features
            n_components = min(20, n_samples - 1, n_features)
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)

            # Calculate distance thresholds from training data (for coloring predictions)
            # Use Euclidean distance in PCA space
            pca_distances = pdist(X_train_pca, metric='euclidean')
            distance_thresholds = {
                'p50': float(np.percentile(pca_distances, 50)),
                'p75': float(np.percentile(pca_distances, 75)),
                'p95': float(np.percentile(pca_distances, 95)),
                'max': float(np.max(pca_distances))
            }

            # Save applicability domain data
            ad_data_dict = {
                'representative_spectra': representative_spectra,
                'representative_indices': representative_indices,
                'training_pca_scores': X_train_pca[representative_indices] if n_samples > 100 else X_train_pca,
                'distance_thresholds': np.array([distance_thresholds['p50'],
                                                  distance_thresholds['p75'],
                                                  distance_thresholds['p95'],
                                                  distance_thresholds['max']])
            }
            np.savez_compressed(ad_data_path, **ad_data_dict)

            # Store PCA model separately
            pca_model_path = tmppath / 'pca_model.pkl'
            joblib.dump(pca, pca_model_path, compress=3)

            metadata_complete['has_applicability_domain'] = True
            metadata_complete['n_representatives'] = len(representative_indices)
            metadata_complete['pca_components'] = n_components
            metadata_complete['distance_thresholds'] = distance_thresholds

            print(f"Applicability domain: PCA with {n_components} components (explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
        else:
            metadata_complete['has_applicability_domain'] = False

        # Create ZIP archive
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(metadata_path, 'metadata.json')
            zf.write(model_path, 'model.pkl')
            if preprocessor is not None:
                zf.write(preprocessor_path, 'preprocessor.pkl')
            if label_encoder is not None:
                zf.write(label_encoder_path, 'label_encoder.pkl')
            if cv_data_path.exists():
                zf.write(cv_data_path, 'cv_data.npz')
            if ad_data_path.exists():
                zf.write(ad_data_path, 'applicability_domain.npz')
            if (tmppath / 'pca_model.pkl').exists():
                zf.write(tmppath / 'pca_model.pkl', 'pca_model.pkl')


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

        # Load label_encoder if present
        label_encoder = None
        label_encoder_path = tmppath / 'label_encoder.pkl'
        if label_encoder_path.exists():
            label_encoder = joblib.load(label_encoder_path)

        # Load CV data if present (for uncertainty estimation)
        cv_data = None
        cv_data_path = tmppath / 'cv_data.npz'
        if cv_data_path.exists():
            with np.load(cv_data_path) as npz_file:
                # Convert to dict for easier access
                cv_data = {key: npz_file[key] for key in npz_file.files}

        # Load applicability domain data if present
        ad_data = None
        pca_model = None
        ad_data_path = tmppath / 'applicability_domain.npz'
        pca_model_path = tmppath / 'pca_model.pkl'
        if ad_data_path.exists():
            with np.load(ad_data_path) as npz_file:
                ad_data = {key: npz_file[key] for key in npz_file.files}
        if pca_model_path.exists():
            pca_model = joblib.load(pca_model_path)

    return {
        'model': model,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'metadata': metadata,
        'cv_data': cv_data,
        'ad_data': ad_data,
        'pca_model': pca_model
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

    # Check if model uses full-spectrum preprocessing (derivative + subset case)
    use_full_spectrum_preprocessing = metadata.get('use_full_spectrum_preprocessing', False)
    full_wavelengths = metadata.get('full_wavelengths', None)

    # Convert to numpy array if needed
    if isinstance(X_new, pd.DataFrame):
        if validate_wavelengths:
            # For derivative + subset: select ALL wavelengths for preprocessing, then subset
            if use_full_spectrum_preprocessing and full_wavelengths is not None:
                # Step 1: Select ALL wavelengths needed for preprocessing
                X_full = _select_wavelengths_from_dataframe(X_new, full_wavelengths)

                # Step 2: Apply preprocessing to full spectrum
                if preprocessor is not None:
                    X_full_preprocessed = preprocessor.transform(X_full)
                else:
                    X_full_preprocessed = X_full

                # Step 3: Find indices of subset wavelengths in full wavelengths
                wavelength_indices = []
                for wl in required_wl:
                    idx = np.where(np.abs(np.array(full_wavelengths) - wl) < 0.01)[0]
                    if len(idx) > 0:
                        wavelength_indices.append(idx[0])
                    else:
                        raise ValueError(f"Required wavelength {wl} not found in full_wavelengths")

                # Step 4: Subset the preprocessed data
                X_processed = X_full_preprocessed[:, wavelength_indices]
            else:
                # Standard case: select subset wavelengths, then preprocess
                X_selected = _select_wavelengths_from_dataframe(X_new, required_wl)

                # Apply preprocessing if present
                if preprocessor is not None:
                    X_processed = preprocessor.transform(X_selected)
                else:
                    X_processed = X_selected
        else:
            X_selected = X_new.values
            # Apply preprocessing if present
            if preprocessor is not None:
                X_processed = preprocessor.transform(X_selected)
            else:
                X_processed = X_selected
    elif isinstance(X_new, np.ndarray):
        # Assume array is already in correct format
        if validate_wavelengths:
            expected_features = len(full_wavelengths) if use_full_spectrum_preprocessing and full_wavelengths else len(required_wl)
            if X_new.shape[1] != expected_features:
                raise ValueError(
                    f"X_new has {X_new.shape[1]} features but model requires "
                    f"{expected_features} wavelengths"
                )

        # For arrays, preprocessing still needs to be applied
        if use_full_spectrum_preprocessing and full_wavelengths is not None:
            # Apply preprocessing, then subset
            if preprocessor is not None:
                X_full_preprocessed = preprocessor.transform(X_new)
            else:
                X_full_preprocessed = X_new

            # Find indices of subset wavelengths
            wavelength_indices = []
            for wl in required_wl:
                idx = np.where(np.abs(np.array(full_wavelengths) - wl) < 0.01)[0]
                if len(idx) > 0:
                    wavelength_indices.append(idx[0])

            X_processed = X_full_preprocessed[:, wavelength_indices]
        else:
            # Standard case
            if preprocessor is not None:
                X_processed = preprocessor.transform(X_new)
            else:
                X_processed = X_new
    else:
        raise TypeError(f"X_new must be DataFrame or ndarray, got {type(X_new)}")

    # Make predictions
    predictions = model.predict(X_processed)

    # If label_encoder exists, convert predictions back to original text labels
    if 'label_encoder' in model_dict and model_dict['label_encoder'] is not None:
        label_encoder = model_dict['label_encoder']
        # Check if predictions are already text labels (some models decode internally)
        if predictions.dtype == object or predictions.dtype.kind == 'U':
            # Already decoded text labels, return as-is
            pass
        else:
            # Numeric predictions that need decoding
            predictions = label_encoder.inverse_transform(predictions.astype(int))

    return predictions


def predict_with_uncertainty(
    model_dict: Dict[str, Any],
    X_new: Union[pd.DataFrame, np.ndarray],
    validate_wavelengths: bool = True
) -> Dict[str, Any]:
    """
    Make predictions with a loaded model and compute uncertainty estimates.

    This function extends predict_with_model() by also returning:
    - For classification: class probabilities and confidence scores
    - For regression: model RMSECV and applicability domain metrics
    - Applicability domain: distance to training data for all models

    Parameters
    ----------
    model_dict : dict
        Dictionary returned from load_model(), containing model, metadata, and optionally cv_data
    X_new : pd.DataFrame or np.ndarray
        New spectral data to predict on
    validate_wavelengths : bool, default=True
        Whether to validate wavelengths match model requirements

    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': np.ndarray of predictions (same as predict_with_model())
        - 'uncertainty': dict with uncertainty metrics:
            For classification:
                - 'probabilities': np.ndarray, shape (n_samples, n_classes)
                - 'confidence': np.ndarray, shape (n_samples,) - max probability
                - 'class_names': list of class names (if label_encoder exists)
            For regression:
                - 'rmsecv': float - overall model error from CV (if available)
                - 'tree_variance': np.ndarray, shape (n_samples,) - only for RandomForest
        - 'applicability_domain': dict with distance metrics:
            - 'pca_distance': np.ndarray, shape (n_samples,) - distance to nearest training sample in PCA space
            - 'spectral_distance': np.ndarray, shape (n_samples,) - Euclidean distance in spectral space
            - 'nearest_sample_idx': np.ndarray, shape (n_samples,) - index of nearest training sample
            - 'distance_status': np.ndarray, shape (n_samples,) - 'good', 'caution', 'extrapolation'
        - 'has_uncertainty': bool - whether uncertainty data is available
        - 'has_applicability_domain': bool - whether applicability domain data is available

    Examples
    --------
    >>> model_dict = load_model('my_model.dasp')
    >>> result = predict_with_uncertainty(model_dict, X_new)
    >>> print(result['predictions'])
    array([15.2, 18.7, 12.3])
    >>> print(result['uncertainty']['rmsecv'])
    0.34
    >>> print(result['applicability_domain']['pca_distance'])
    array([0.82, 2.45, 5.67])
    >>> print(result['applicability_domain']['distance_status'])
    array(['good', 'caution', 'extrapolation'])
    """
    # Get standard predictions first
    predictions = predict_with_model(model_dict, X_new, validate_wavelengths)

    model = model_dict['model']
    metadata = model_dict['metadata']
    task_type = metadata.get('task_type', 'regression')

    uncertainty = {}
    has_uncertainty = False

    # Extract components needed for preprocessing
    preprocessor = model_dict['preprocessor']
    required_wl = metadata['wavelengths']
    use_full_spectrum_preprocessing = metadata.get('use_full_spectrum_preprocessing', False)
    full_wavelengths = metadata.get('full_wavelengths', None)

    # Preprocess X_new to get X_processed (same logic as predict_with_model)
    if isinstance(X_new, pd.DataFrame):
        if validate_wavelengths:
            if use_full_spectrum_preprocessing and full_wavelengths is not None:
                X_full = _select_wavelengths_from_dataframe(X_new, full_wavelengths)
                if preprocessor is not None:
                    X_full_preprocessed = preprocessor.transform(X_full)
                else:
                    X_full_preprocessed = X_full
                wavelength_indices = []
                for wl in required_wl:
                    idx = np.where(np.abs(np.array(full_wavelengths) - wl) < 0.01)[0]
                    if len(idx) > 0:
                        wavelength_indices.append(idx[0])
                X_processed = X_full_preprocessed[:, wavelength_indices]
            else:
                X_selected = _select_wavelengths_from_dataframe(X_new, required_wl)
                if preprocessor is not None:
                    X_processed = preprocessor.transform(X_selected)
                else:
                    X_processed = X_selected
        else:
            X_selected = X_new.values
            if preprocessor is not None:
                X_processed = preprocessor.transform(X_selected)
            else:
                X_processed = X_selected
    elif isinstance(X_new, np.ndarray):
        if use_full_spectrum_preprocessing and full_wavelengths is not None:
            if preprocessor is not None:
                X_full_preprocessed = preprocessor.transform(X_new)
            else:
                X_full_preprocessed = X_new
            wavelength_indices = []
            for wl in required_wl:
                idx = np.where(np.abs(np.array(full_wavelengths) - wl) < 0.01)[0]
                if len(idx) > 0:
                    wavelength_indices.append(idx[0])
            X_processed = X_full_preprocessed[:, wavelength_indices]
        else:
            if preprocessor is not None:
                X_processed = preprocessor.transform(X_new)
            else:
                X_processed = X_new
    else:
        raise TypeError(f"X_new must be DataFrame or ndarray, got {type(X_new)}")

    # Classification: get probabilities
    if task_type == 'classification':
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_processed)
                confidence = np.max(probabilities, axis=1)
                uncertainty['probabilities'] = probabilities
                uncertainty['confidence'] = confidence
                has_uncertainty = True

                # Add class names if label_encoder exists
                if 'label_encoder' in model_dict and model_dict['label_encoder'] is not None:
                    uncertainty['class_names'] = model_dict['label_encoder'].classes_.tolist()
                else:
                    # Try to get from model classes if available
                    if hasattr(model, 'classes_'):
                        uncertainty['class_names'] = model.classes_.tolist()
            except Exception as e:
                # Model doesn't support predict_proba or failed
                uncertainty['error'] = f"Could not compute probabilities: {str(e)}"

    # Regression: report model-level error, not per-sample CI
    else:  # regression
        cv_data = model_dict.get('cv_data', None)

        if cv_data is not None and 'cv_residuals' in cv_data:
            # Use CV residuals to calculate RMSECV (model-level metric)
            residuals = cv_data['cv_residuals']
            rmsecv = np.sqrt(np.mean(residuals**2))
            uncertainty['rmsecv'] = float(rmsecv)
            has_uncertainty = True
        elif 'performance' in metadata and 'RMSE' in metadata['performance']:
            # Fallback to RMSE from metadata
            uncertainty['rmsecv'] = float(metadata['performance']['RMSE'])
            has_uncertainty = True

        # For Random Forest: calculate per-sample tree variance
        model_class = metadata.get('model_class', '')
        if 'RandomForest' in model_class and hasattr(model, 'estimators_'):
            # Get predictions from each tree
            tree_predictions = np.array([tree.predict(X_processed) for tree in model.estimators_])
            # Calculate variance across trees for each sample
            tree_variance = np.std(tree_predictions, axis=0)
            uncertainty['tree_variance'] = tree_variance
            has_uncertainty = True

    # Calculate applicability domain metrics (for all model types)
    applicability_domain = {}
    has_applicability_domain = False

    # Check if this is an ensemble with base model dicts
    is_ensemble = 'base_model_dicts' in model_dict and model_dict['base_model_dicts']

    if is_ensemble:
        # Aggregate applicability domain from base models
        base_model_dicts = model_dict['base_model_dicts']
        aggregated_ad = _aggregate_ensemble_applicability_domain(base_model_dicts, X_processed)

        if aggregated_ad is not None:
            applicability_domain = aggregated_ad
            has_applicability_domain = True

    elif 'ad_data' in model_dict and model_dict['ad_data'] is not None:
        from scipy.spatial.distance import cdist

        ad_data = model_dict['ad_data']
        pca_model = model_dict.get('pca_model')

        if pca_model is not None:
            # Transform prediction data to PCA space
            X_pred_pca = pca_model.transform(X_processed)
            training_pca_scores = ad_data['training_pca_scores']

            # Calculate distances in PCA space
            pca_distances = cdist(X_pred_pca, training_pca_scores, metric='euclidean')
            min_pca_distance = np.min(pca_distances, axis=1)
            nearest_idx = np.argmin(pca_distances, axis=1)

            applicability_domain['pca_distance'] = min_pca_distance
            applicability_domain['nearest_sample_idx'] = nearest_idx

            # Calculate distances in spectral space (optional, for comparison)
            representative_spectra = ad_data['representative_spectra']
            spectral_distances = cdist(X_processed, representative_spectra, metric='euclidean')
            min_spectral_distance = np.min(spectral_distances, axis=1)
            applicability_domain['spectral_distance'] = min_spectral_distance

            # Get distance thresholds for coloring
            if 'distance_thresholds' in ad_data:
                thresholds = ad_data['distance_thresholds']
                p50, p75, p95, max_dist = thresholds

                # Assign status based on PCA distance
                distance_status = np.empty(len(min_pca_distance), dtype=object)
                distance_status[min_pca_distance <= p75] = 'good'
                distance_status[(min_pca_distance > p75) & (min_pca_distance <= p95)] = 'caution'
                distance_status[min_pca_distance > p95] = 'extrapolation'

                applicability_domain['distance_status'] = distance_status
                applicability_domain['thresholds'] = {
                    'p50': float(p50),
                    'p75': float(p75),
                    'p95': float(p95),
                    'max': float(max_dist)
                }

            has_applicability_domain = True

    return {
        'predictions': predictions,
        'uncertainty': uncertainty,
        'has_uncertainty': has_uncertainty,
        'applicability_domain': applicability_domain,
        'has_applicability_domain': has_applicability_domain
    }


def _aggregate_ensemble_applicability_domain(base_model_dicts, X_processed):
    """
    Aggregate applicability domain information from multiple base models in an ensemble.

    Strategy: Use the worst-case (maximum) distance across all base models.
    This ensures conservative warnings - if ANY base model considers a prediction
    as extrapolation, the ensemble prediction will be flagged.

    Parameters
    ----------
    base_model_dicts : list of dict
        List of model dictionaries for each base model
    X_processed : ndarray
        Preprocessed input data

    Returns
    -------
    dict or None
        Aggregated applicability domain info, or None if no base models have AD data
    """
    from scipy.spatial.distance import cdist

    all_pca_distances = []
    all_spectral_distances = []
    has_any_ad = False
    aggregated_thresholds = None

    # Collect applicability domain info from each base model
    for model_dict in base_model_dicts:
        if 'ad_data' not in model_dict or model_dict['ad_data'] is None:
            continue

        ad_data = model_dict['ad_data']
        pca_model = model_dict.get('pca_model')

        if pca_model is None:
            continue

        has_any_ad = True

        # Transform to PCA space
        X_pred_pca = pca_model.transform(X_processed)
        training_pca_scores = ad_data['training_pca_scores']

        # Calculate distances
        pca_distances = cdist(X_pred_pca, training_pca_scores, metric='euclidean')
        min_pca_distance = np.min(pca_distances, axis=1)
        all_pca_distances.append(min_pca_distance)

        # Spectral distances
        representative_spectra = ad_data['representative_spectra']
        spectral_distances = cdist(X_processed, representative_spectra, metric='euclidean')
        min_spectral_distance = np.min(spectral_distances, axis=1)
        all_spectral_distances.append(min_spectral_distance)

        # Get thresholds (use first model's thresholds as reference)
        if aggregated_thresholds is None and 'distance_thresholds' in ad_data:
            aggregated_thresholds = ad_data['distance_thresholds']

    if not has_any_ad:
        return None

    # Aggregate: use maximum (worst-case) distance across all base models
    all_pca_distances = np.array(all_pca_distances)
    all_spectral_distances = np.array(all_spectral_distances)

    max_pca_distance = np.max(all_pca_distances, axis=0)
    max_spectral_distance = np.max(all_spectral_distances, axis=0)

    # Find which model contributed the max distance for each sample
    nearest_model_idx = np.argmax(all_pca_distances, axis=0)

    applicability_domain = {
        'pca_distance': max_pca_distance,
        'spectral_distance': max_spectral_distance,
        'nearest_model_idx': nearest_model_idx  # Which base model had worst distance
    }

    # Assign status based on aggregated thresholds
    if aggregated_thresholds is not None:
        p50, p75, p95, max_dist = aggregated_thresholds

        distance_status = np.empty(len(max_pca_distance), dtype=object)
        distance_status[max_pca_distance <= p75] = 'good'
        distance_status[(max_pca_distance > p75) & (max_pca_distance <= p95)] = 'caution'
        distance_status[max_pca_distance > p95] = 'extrapolation'

        applicability_domain['distance_status'] = distance_status
        applicability_domain['thresholds'] = {
            'p50': float(p50),
            'p75': float(p75),
            'p95': float(p95),
            'max': float(max_dist)
        }

    return applicability_domain


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


def _convert_to_serializable(obj):
    """
    Convert objects to JSON-serializable format for ensemble config.

    This is an alias for _json_serializer to maintain consistency.
    """
    return _json_serializer(obj)


def save_ensemble(ensemble: Any, filepath: str, metadata: Dict[str, Any]) -> None:
    """
    Save an ensemble model to a .dasp file.

    Parameters
    ----------
    ensemble : Any
        Trained ensemble object (RegionAwareWeightedEnsemble, MixtureOfExpertsEnsemble, etc.)
    filepath : str
        Path where the ensemble .dasp file will be saved
    metadata : dict
        Additional metadata including:
        - 'ensemble_type': Type of ensemble ('simple_average', 'region_weighted', etc.)
        - 'ensemble_name': Display name
        - 'task_type': 'regression' or 'classification' (REQUIRED)
        - 'preprocessing': Preprocessing method (REQUIRED)
        - 'wavelengths': List of wavelengths (REQUIRED)
        - 'n_vars': Number of variables (REQUIRED)
        - 'performance': Performance metrics dict
        - 'use_full_spectrum_preprocessing': Boolean for derivative+subset case
        - 'full_wavelengths': Full wavelength list if using derivative+subset
        - 'window': Savgol window size (if applicable)
        - 'X_train': Training data for applicability domain (optional)
        - 'cv_residuals', 'cv_predictions', 'cv_actuals': CV data for uncertainty (optional)

    Returns
    -------
    None

    Notes
    -----
    Ensemble .dasp files are ZIP archives containing:
    - ensemble_config.json: Ensemble configuration and metadata
    - base_model_0.dasp, base_model_1.dasp, ...: Individual model files
    - ensemble_state.pkl: Ensemble-specific state (weights, analyzer, etc.)

    Raises
    ------
    ValueError
        If metadata is missing required fields (task_type, wavelengths, n_vars)
    """
    # Validate required metadata fields
    required_fields = ['task_type', 'wavelengths', 'n_vars']
    missing_fields = [f for f in required_fields if f not in metadata]
    if missing_fields:
        raise ValueError(f"Ensemble metadata missing required fields: {missing_fields}")

    filepath = Path(filepath)

    # Extract optional training data for applicability domain
    X_train = metadata.pop('X_train', None)
    cv_residuals = metadata.pop('cv_residuals', None)
    cv_predictions = metadata.pop('cv_predictions', None)
    cv_actuals = metadata.pop('cv_actuals', None)
    preprocessor = metadata.pop('preprocessor', None)
    label_encoder = metadata.pop('label_encoder', None)

    # Create temporary directory for base model files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save each base model
        base_model_files = []
        for i, (model, model_name) in enumerate(zip(ensemble.models, ensemble.model_names)):
            base_model_path = tmpdir_path / f"base_model_{i}.dasp"

            # Build comprehensive base model metadata with all required fields
            base_metadata = {
                'model_name': model_name,
                'model_index': i,
                'is_base_model': True,
                # Required fields
                'task_type': metadata['task_type'],
                'wavelengths': metadata['wavelengths'],
                'n_vars': metadata['n_vars'],
                # Optional but important fields
                'preprocessing': metadata.get('preprocessing', 'unknown'),
                'window': metadata.get('window', None),
                'performance': metadata.get('performance', {}),
                'use_full_spectrum_preprocessing': metadata.get('use_full_spectrum_preprocessing', False),
                'full_wavelengths': metadata.get('full_wavelengths', None),
                'n_samples': metadata.get('n_training_samples', 0),
                'ensemble_parent': True,  # Flag to indicate this is from an ensemble
            }

            # Save individual model with all metadata and optional training data
            save_model(
                model=model,
                preprocessor=preprocessor,
                metadata=base_metadata,
                filepath=str(base_model_path),
                label_encoder=label_encoder,
                cv_residuals=cv_residuals,
                cv_predictions=cv_predictions,
                cv_actuals=cv_actuals,
                X_train=X_train
            )
            base_model_files.append(f"base_model_{i}.dasp")

        # Create ensemble config
        ensemble_config = {
            'format_version': '1.0',
            'ensemble_type': metadata.get('ensemble_type', 'unknown'),
            'ensemble_name': metadata.get('ensemble_name', 'Ensemble'),
            'n_models': len(ensemble.models),
            'model_names': ensemble.model_names,
            'base_model_files': base_model_files,
            'metadata': metadata,
            'save_date': datetime.now().isoformat()
        }

        # Save ensemble-specific state
        ensemble_state = {}

        # Save weights if present (for weighted ensembles)
        if hasattr(ensemble, 'weights_'):
            ensemble_state['weights'] = ensemble.weights_

        # Save analyzer if present (for region-aware ensembles)
        if hasattr(ensemble, 'analyzer_'):
            ensemble_state['analyzer'] = ensemble.analyzer_

        # Save meta_model if present (for stacking)
        if hasattr(ensemble, 'meta_model_'):
            ensemble_state['meta_model'] = ensemble.meta_model_

        # Save region info if present
        if hasattr(ensemble, 'n_regions'):
            ensemble_state['n_regions'] = ensemble.n_regions

        # Pickle ensemble state
        ensemble_state_path = tmpdir_path / "ensemble_state.pkl"
        joblib.dump(ensemble_state, ensemble_state_path)

        # Save config as JSON
        config_path = tmpdir_path / "ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2, default=_convert_to_serializable)

        # Create ZIP archive
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add config
            zf.write(config_path, "ensemble_config.json")

            # Add ensemble state
            zf.write(ensemble_state_path, "ensemble_state.pkl")

            # Add all base models
            for base_file in base_model_files:
                zf.write(tmpdir_path / base_file, base_file)


def load_ensemble(filepath: str) -> Dict[str, Any]:
    """
    Load an ensemble model from a .dasp file.

    Parameters
    ----------
    filepath : str
        Path to the ensemble .dasp file

    Returns
    -------
    dict
        Dictionary containing:
        - 'ensemble': Reconstructed ensemble object
        - 'metadata': Ensemble metadata
        - 'model_names': List of base model names
        - 'config': Full ensemble configuration

    Raises
    ------
    FileNotFoundError
        If the .dasp file doesn't exist
    ValueError
        If the file is not a valid ensemble .dasp file
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Ensemble file not found: {filepath}")

    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Extract ZIP contents
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(tmpdir_path)

        # Load config
        config_path = tmpdir_path / "ensemble_config.json"
        if not config_path.exists():
            raise ValueError(f"Not a valid ensemble file: missing ensemble_config.json")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load base models
        base_models = []
        base_model_dicts = []  # Keep full model_dicts for applicability domain
        base_model_files = config['base_model_files']
        model_names = config['model_names']

        for base_file in base_model_files:
            base_model_path = tmpdir_path / base_file
            model_dict = load_model(str(base_model_path))
            base_models.append(model_dict['model'])
            base_model_dicts.append(model_dict)  # Store full dict

        # Load ensemble state
        ensemble_state_path = tmpdir_path / "ensemble_state.pkl"
        ensemble_state = joblib.load(ensemble_state_path)

        # Reconstruct ensemble object
        from spectral_predict.ensemble import (
            RegionAwareWeightedEnsemble,
            MixtureOfExpertsEnsemble,
            StackingEnsemble
        )

        ensemble_type = config['ensemble_type']

        # Create appropriate ensemble object
        if ensemble_type == 'region_weighted':
            ensemble = RegionAwareWeightedEnsemble(
                models=base_models,
                model_names=model_names,
                n_regions=ensemble_state.get('n_regions', 5)
            )
            # Restore weights and analyzer
            if 'weights' in ensemble_state:
                ensemble.weights_ = ensemble_state['weights']
            if 'analyzer' in ensemble_state:
                ensemble.analyzer_ = ensemble_state['analyzer']

        elif ensemble_type == 'mixture_experts':
            ensemble = MixtureOfExpertsEnsemble(
                models=base_models,
                model_names=model_names,
                n_regions=ensemble_state.get('n_regions', 5)
            )
            # Restore analyzer
            if 'analyzer' in ensemble_state:
                ensemble.analyzer_ = ensemble_state['analyzer']

        elif ensemble_type in ['stacking', 'region_stacking']:
            region_aware = (ensemble_type == 'region_stacking')
            ensemble = StackingEnsemble(
                models=base_models,
                model_names=model_names,
                region_aware=region_aware,
                n_regions=ensemble_state.get('n_regions', 5) if region_aware else None
            )
            # Restore meta_model
            if 'meta_model' in ensemble_state:
                ensemble.meta_model_ = ensemble_state['meta_model']

        elif ensemble_type == 'simple_average':
            # Simple average - just store models
            class SimpleAverageEnsemble:
                def __init__(self, models, model_names):
                    self.models = models
                    self.model_names = model_names

                def predict(self, X):
                    predictions = np.array([model.predict(X) for model in self.models])
                    return predictions.mean(axis=0)

            ensemble = SimpleAverageEnsemble(base_models, model_names)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

        return {
            'ensemble': ensemble,
            'metadata': config['metadata'],
            'model_names': model_names,
            'config': config,
            'base_model_dicts': base_model_dicts  # Include for applicability domain
        }


# Module exports
__all__ = [
    'save_model',
    'load_model',
    'predict_with_model',
    'predict_with_uncertainty',
    'get_model_info',
    'save_ensemble',
    'load_ensemble'
]
