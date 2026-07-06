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
import logging
import warnings
import zipfile
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

from . import __version__
from .resource_paths import is_frozen

logger = logging.getLogger(__name__)


def _ensure_pipeline_fitted(pipeline):
    """
    Ensure a Pipeline is marked as fitted for PyInstaller bundle compatibility.

    This is a workaround for sklearn's check_is_fitted() behaving differently
    in PyInstaller bundles. Only called when is_frozen() is True.

    Also handles TransformedTargetRegressor wrapping a Pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import TransformedTargetRegressor

    # If model is a TransformedTargetRegressor, patch its inner regressor
    if isinstance(pipeline, TransformedTargetRegressor):
        if hasattr(pipeline, 'regressor_'):
            _ensure_pipeline_fitted(pipeline.regressor_)
        return

    if not isinstance(pipeline, Pipeline):
        return

    # Set Pipeline's fitted indicator
    if not hasattr(pipeline, '_final_estimator'):
        # Pipeline stores final step reference after fitting
        if pipeline.steps:
            pipeline._final_estimator = pipeline.steps[-1][1]

    # Ensure each step has fitted attributes
    for name, step in pipeline.steps:
        if step is not None and not hasattr(step, 'n_features_in_'):
            step.n_features_in_ = None
        if step is not None and not hasattr(step, '_is_fitted'):
            step._is_fitted = True


def save_model(
    model: Any,
    preprocessor: Optional[Any],
    metadata: Dict[str, Any],
    filepath: Union[str, Path],
    label_encoder: Optional[Any] = None,
    cv_residuals: Optional[np.ndarray] = None,
    cv_predictions: Optional[np.ndarray] = None,
    cv_actuals: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None,
    bias_correction: Optional[Dict[str, Any]] = None
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

        # Extract one-class auxiliary objects BEFORE JSON serialization
        # (sklearn objects are not JSON-serializable)
        oc_scaler = None
        oc_pca_reducer = None
        if metadata_complete.get("task_type") == "one_class":
            oc_scaler = metadata_complete.pop("scaler", None)
            oc_pca_reducer = metadata_complete.pop("pca_reducer", None)
            metadata_complete["has_scaler"] = oc_scaler is not None
            metadata_complete["has_pca_reducer"] = oc_pca_reducer is not None

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

        # Save one-class auxiliary objects if present
        if oc_scaler is not None:
            joblib.dump(oc_scaler, tmppath / "scaler.pkl", compress=3)
        if oc_pca_reducer is not None:
            joblib.dump(oc_pca_reducer, tmppath / "pca_reducer.pkl", compress=3)

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

            # Pre-compute Q-threshold from full training data
            train_reconstructed_full = pca.inverse_transform(X_train_pca)
            train_q_full = np.sum((X_train - train_reconstructed_full) ** 2, axis=1)

            # Compute T² for all training samples (for reliability score calibration)
            mu_train = np.mean(X_train_pca, axis=0)
            cov_train = np.cov(X_train_pca.T)
            if X_train_pca.shape[1] == 1:
                cov_train = np.atleast_2d(cov_train)
            try:
                inv_cov_train = np.linalg.inv(cov_train)
            except np.linalg.LinAlgError:
                cov_train += np.eye(cov_train.shape[0]) * 1e-6
                inv_cov_train = np.linalg.inv(cov_train)
            training_t2 = np.array([(x - mu_train) @ inv_cov_train @ (x - mu_train) for x in X_train_pca])

            # Save applicability domain data
            ad_data_dict = {
                'representative_spectra': representative_spectra,
                'representative_indices': representative_indices,
                'training_pca_scores': X_train_pca,  # Full training PCA scores for correct T² statistics
                'distance_thresholds': np.array([distance_thresholds['p50'],
                                                  distance_thresholds['p75'],
                                                  distance_thresholds['p95'],
                                                  distance_thresholds['max']]),
                'q_threshold': np.array([np.percentile(train_q_full, 99)]),
                'training_t2_values': training_t2,
                'training_q_values': train_q_full,
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

        # Save bias correction if provided
        bias_correction_path = tmppath / 'bias_correction.json'
        if bias_correction is not None:
            with open(bias_correction_path, 'w', encoding='utf-8') as f:
                json.dump(bias_correction, f, indent=2)
            metadata_complete['has_bias_correction'] = True
        else:
            metadata_complete['has_bias_correction'] = False

        # Re-write metadata (may have been updated with has_bias_correction etc.)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_complete, f, indent=2, default=_json_serializer)

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
            if bias_correction_path.exists():
                zf.write(bias_correction_path, 'bias_correction.json')
            # One-class auxiliary files
            scaler_zip_path = tmppath / "scaler.pkl"
            if scaler_zip_path.exists():
                zf.write(scaler_zip_path, "scaler.pkl")
            pca_reducer_zip_path = tmppath / "pca_reducer.pkl"
            if pca_reducer_zip_path.exists():
                zf.write(pca_reducer_zip_path, "pca_reducer.pkl")


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
        # Bundle compatibility fix: ensure Pipeline is marked as fitted
        if model is not None and is_frozen():
            _ensure_pipeline_fitted(model)

        # Load preprocessor if present
        preprocessor = None
        preprocessor_path = tmppath / 'preprocessor.pkl'
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)

            # Bundle compatibility fix: ensure Pipeline is marked as fitted
            if preprocessor is not None and is_frozen():
                _ensure_pipeline_fitted(preprocessor)

        # Load label_encoder if present
        label_encoder = None
        label_encoder_path = tmppath / 'label_encoder.pkl'
        if label_encoder_path.exists():
            label_encoder = joblib.load(label_encoder_path)

        # Load one-class auxiliary objects if present
        scaler = None
        scaler_path = tmppath / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        pca_reducer = None
        pca_reducer_path = tmppath / "pca_reducer.pkl"
        if pca_reducer_path.exists():
            pca_reducer = joblib.load(pca_reducer_path)

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

        # Load bias correction if present
        bias_correction = None
        bias_correction_path = tmppath / 'bias_correction.json'
        if bias_correction_path.exists():
            with open(bias_correction_path, 'r', encoding='utf-8') as f:
                bias_correction = json.load(f)

    return {
        'model': model,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'metadata': metadata,
        'cv_data': cv_data,
        'ad_data': ad_data,
        'pca_model': pca_model,
        'bias_correction': bias_correction,
        'scaler': scaler,
        'pca_reducer': pca_reducer,
    }


def _compute_reliability_scores(
    t2_values: np.ndarray,
    q_values: np.ndarray,
    t2_threshold: float,
    q_threshold: float,
    training_t2: np.ndarray | None = None,
    training_q: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-sample prediction reliability scores (0-95%).

    Scores reflect how spectrally similar each sample is to training data,
    which correlates with expected prediction quality. Uses training T²/Q
    distributions when available for calibrated percentile-based scoring.
    Falls back to ratio-based scoring for older models.

    Score ranges:
        80-95%  High reliability — spectrally very similar to training data
        50-79%  Moderate — within training variability, toward edges
        35-49%  Low — at or beyond training boundary
         5-34%  Very low — well outside training domain
    """
    if training_t2 is not None and training_q is not None:
        # Vectorized percentile computation via searchsorted
        sorted_t2 = np.sort(training_t2)
        sorted_q = np.sort(training_q)
        n_train = len(sorted_t2)

        t2_pctl = np.searchsorted(sorted_t2, t2_values, side='right') / n_train
        q_pctl = np.searchsorted(sorted_q, q_values, side='right') / n_train
        worst_pctl = np.maximum(t2_pctl, q_pctl)

        # Detect samples beyond training max
        beyond_t2 = t2_values > sorted_t2[-1]
        beyond_q = q_values > sorted_q[-1]
        beyond_training = beyond_t2 | beyond_q

        # Piecewise linear mapping for within-training-range samples
        scores = np.where(
            worst_pctl <= 0.90,
            95 - (worst_pctl / 0.90) * 15,               # 95 -> 80
            np.where(
                worst_pctl <= 0.99,
                80 - ((worst_pctl - 0.90) / 0.09) * 30,  # 80 -> 50
                50 - ((worst_pctl - 0.99) / 0.01) * 15   # 50 -> 35
            )
        )

        # Override for beyond-training samples: exponential decay from 35
        t2_ratio = np.where(t2_threshold > 0, t2_values / t2_threshold, 0.0)
        q_ratio = np.where(q_threshold > 0, q_values / q_threshold, 0.0)
        worst_ratio = np.maximum(t2_ratio, q_ratio)
        beyond_scores = 35 * np.exp(-0.5 * np.maximum(0, worst_ratio - 1))

        scores = np.where(beyond_training, beyond_scores, scores)
    else:
        # Fallback for older models: ratio-based (no training distribution)
        t2_ratio = np.where(t2_threshold > 0, t2_values / t2_threshold, 0.0)
        q_ratio = np.where(q_threshold > 0, q_values / q_threshold, 0.0)
        worst_ratio = np.maximum(t2_ratio, q_ratio)
        scores = 100 * np.exp(-0.7 * worst_ratio)

    return np.clip(np.round(scores), 5, 95).astype(int)


# Recognized metadata["task_type"] values (spec §6). An explicit value outside
# this set is rejected by ``predict_with_model`` as a forward-compat gate; a
# missing or null task_type is treated as a legacy model (no gate).
_SUPPORTED_TASK_TYPES = frozenset(
    {"regression", "classification", "one_class", "multiclass_simca"}
)


def predict_with_model(
    model_dict: Dict[str, Any],
    X_new: Union[pd.DataFrame, np.ndarray],
    validate_wavelengths: bool = True,
    _internals: dict | None = None,
) -> Union[np.ndarray, dict]:
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
    np.ndarray or dict
        - For ``task_type`` in {``regression``, ``classification``, ``one_class``}:
          an ``ndarray`` of predicted values, shape ``(n_samples,)`` for
          regression / one-class or ``(n_samples, n_classes)`` for
          classification.
        - For ``task_type == "multiclass_simca"``: a ``dict`` with the schema
          ``{"p_values": ndarray (n_samples, K) float, "decision_matrix":
          ndarray (n_samples, K) bool, "summary_label": ndarray (n_samples,)
          object of single-class-label / "multiple" / "novel",
          "accepted_classes": list[list] of per-row accepted class labels}``.

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
    # Suppress sklearn warnings that are cosmetic only:
    # - "X does not have valid feature names" (model fitted with DataFrame, predicting with array)
    # - "Pipeline instance is not fitted yet" (FutureWarning, predictions still work)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")

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
            # validate_wavelengths=False: trust the caller's column order. Still
            # honor the full-spectrum edge-mask handshake (derivative + subset) —
            # otherwise a model trained on the SG-edge-trimmed axis receives the
            # full-width matrix and its per-class scaler/PCA get the wrong feature
            # count (crashed the multiclass save -> predict-on-raw path).
            if use_full_spectrum_preprocessing and full_wavelengths is not None:
                X_full = X_new.values
                X_full_preprocessed = (
                    preprocessor.transform(X_full) if preprocessor is not None else X_full
                )
                wavelength_indices = []
                for wl in required_wl:
                    idx = np.where(np.abs(np.array(full_wavelengths) - wl) < 0.01)[0]
                    if len(idx) > 0:
                        wavelength_indices.append(idx[0])
                X_processed = X_full_preprocessed[:, wavelength_indices]
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

    # --- task-type dispatch (spec §6) -----------------------------------------
    task_type = metadata.get("task_type", "regression")

    # Forward-compat gate: reject genuinely-unknown task types. A missing
    # task_type defaults to "regression" above (legacy regression models) and an
    # explicit null is also left alone (legacy); only an explicit value outside
    # the recognized set raises. This must NOT fire for any of the four
    # recognized values or for absent/null task_type.
    if task_type is not None and task_type not in _SUPPORTED_TASK_TYPES:
        raise NotImplementedError(
            f"task_type={task_type!r} is not supported by this build "
            f"(expected one of {sorted(_SUPPORTED_TASK_TYPES)} or absent)."
        )

    # Multi-class SIMCA (task A8, spec §6): the model is a MultiClassClassModel
    # orchestrator that owns all per-class state. Return its decision matrix,
    # per-row summary label, and per-row accepted-class lists as a dict — NOT
    # the ndarray that regression / classification / one_class return.
    if task_type == "multiclass_simca":
        P, A = model.decision_matrix(X_processed)
        labels = model.predict(X_processed)
        accepted_classes = [
            [c for c, accepted in zip(model.classes_, A[i]) if accepted]
            for i in range(A.shape[0])
        ]
        return {
            "p_values": P,
            "decision_matrix": A,
            "summary_label": labels,
            "accepted_classes": accepted_classes,
        }

    # One-class prediction branch
    if task_type == "one_class":
        # Apply one-class scaler if present
        oc_scaler = model_dict.get("scaler")
        if oc_scaler is not None:
            X_processed = oc_scaler.transform(X_processed)

        # Apply one-class PCA reducer if present
        oc_pca_reducer = model_dict.get("pca_reducer")
        if oc_pca_reducer is not None:
            X_processed = oc_pca_reducer.transform(X_processed)

        # Capture transformed X for decision score extraction (backward-compatible)
        if _internals is not None:
            _internals['X_processed'] = X_processed

        # Predict labels (+1 inlier, -1 outlier)
        predictions = model.predict(X_processed)
        return predictions

    # Make predictions
    predictions = model.predict(X_processed)

    # If label_encoder exists, convert predictions back to original text labels
    if 'label_encoder' in model_dict and model_dict['label_encoder'] is not None:
        label_encoder = model_dict['label_encoder']
        # Check if predictions are already text labels (some models decode internally)
        if pd.api.types.is_string_dtype(predictions.dtype):
            # Already decoded text labels, return as-is
            pass
        else:
            # Numeric predictions that need decoding
            predictions = label_encoder.inverse_transform(predictions.astype(int))

    # NOTE: Bias correction is applied after model.predict(), which returns
    # original-scale values even when TransformedTargetRegressor is used.
    bias_correction = model_dict.get('bias_correction')
    if bias_correction is not None:
        from .bias_correction import apply_correction
        predictions = apply_correction(predictions, bias_correction)

    return predictions


def predict_with_uncertainty(
    model_dict: Dict[str, Any],
    X_new: Union[pd.DataFrame, np.ndarray],
    validate_wavelengths: bool = True,
    prediction_data_type: Optional[str] = None
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
    prediction_data_type : str, optional
        Type of prediction data ('absorbance' or 'reflectance'). If provided and differs
        from model's training data type, a warning will be included in the result.

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
        - 'data_type_warning': str or None - warning message if prediction data type differs from training data type

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
    # Suppress sklearn warnings that are cosmetic only
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")

    model = model_dict['model']
    metadata = model_dict['metadata']
    task_type = metadata.get('task_type', 'regression')

    # Check for data type mismatch
    data_type_warning = None
    model_data_type = metadata.get('data_type')

    if prediction_data_type and model_data_type:
        if prediction_data_type.lower() != model_data_type.lower():
            data_type_warning = (
                f"Model trained on {model_data_type.upper()} data, "
                f"but prediction data is {prediction_data_type.upper()}."
            )

    # Multi-class class-modeling (SIMCA): predict_with_model already returns the
    # per-sample decision schema (p-values + accept matrix + accepted class sets
    # + summary labels). Surface it through the uncertainty envelope so the GUI
    # predict path gets a well-formed result instead of an ndarray-shaped one
    # (T-31 Phase D fold-in). Applicability domain is not defined per-model here
    # (each class has its own model), so it is left empty.
    if task_type == 'multiclass_simca':
        pred = predict_with_model(model_dict, X_new, validate_wavelengths)
        return {
            'predictions': pred['summary_label'],
            'uncertainty': {
                'p_values': pred['p_values'],
                'decision_matrix': pred['decision_matrix'],
                'accepted_classes': pred['accepted_classes'],
                'class_names': list(getattr(model, 'classes_', [])),
            },
            'applicability_domain': {},
            'has_uncertainty': True,
            'has_applicability_domain': False,
            'data_type_warning': data_type_warning,
        }

    # One-class models: extract decision scores for uncertainty/applicability domain
    if task_type == 'one_class':
        internals: dict = {}
        predictions = predict_with_model(model_dict, X_new, validate_wavelengths, _internals=internals)

        # Extract decision scores from the already-transformed data.
        # Failure here used to be silently swallowed (decision_scores=None,
        # empty uncertainty/AD payloads, no warning) — the user got
        # predictions back with no signal that the score-extraction path
        # broke. Now we log + surface a structured error flag so the GUI
        # can display "predictions worked, uncertainty broke" alongside
        # the predictions.
        decision_scores = None
        decision_score_error = None
        X_proc = internals.get('X_processed')
        if X_proc is not None:
            try:
                if hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_proc)
                elif hasattr(model, 'score_samples'):
                    decision_scores = model.score_samples(X_proc)
                else:
                    decision_score_error = (
                        f"{type(model).__name__} exposes neither "
                        f"decision_function nor score_samples"
                    )
            except Exception as exc:
                logger.warning(
                    "OC decision-score extraction failed for %s: %s: %s",
                    type(model).__name__, type(exc).__name__, exc,
                )
                decision_score_error = f"{type(exc).__name__}: {exc}"
                decision_scores = None

        result: dict = {
            'predictions': predictions,
            'uncertainty': {},
            'applicability_domain': {},
            'has_uncertainty': False,
            'has_applicability_domain': False,
            'data_type_warning': data_type_warning,
            'decision_score_error': decision_score_error,
        }

        if decision_scores is not None:
            scores = decision_scores

            # Use training-derived thresholds when available, fall back to batch percentiles
            oc_stats = metadata.get('oc_score_stats')
            if oc_stats is not None:
                q10, q25 = oc_stats['q10'], oc_stats['q25']
            else:
                q10, q25 = np.percentile(scores, [10, 25])

            status = np.where(
                scores >= q25, 'good',
                np.where(scores >= q10, 'caution', 'extrapolation')
            )

            # Compute confidence: higher = more in-domain (positive scores = inlier)
            if oc_stats is not None:
                center = oc_stats['mean']
                scale = max(oc_stats['std'], 1e-10)
                confidence = 1.0 / (1.0 + np.exp(-(scores - center) / scale))
            else:
                s_min, s_max = scores.min(), scores.max()
                if s_max > s_min:
                    confidence = (scores - s_min) / (s_max - s_min)
                else:
                    confidence = np.full_like(scores, 0.5)

            result['uncertainty'] = {
                'decision_scores': scores,
                'confidence': confidence,
            }
            result['applicability_domain'] = {
                'anomaly_score': scores,
                'distance_status': status,
            }
            result['has_uncertainty'] = True
            result['has_applicability_domain'] = True

        return result

    # Get standard predictions for non-OC models
    predictions = predict_with_model(model_dict, X_new, validate_wavelengths)

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
        # Aggregate applicability domain information from base models
        base_model_dicts = model_dict['base_model_dicts']
        aggregated_ad = _aggregate_ensemble_applicability_domain(base_model_dicts, X_processed)

        if aggregated_ad is not None:
            applicability_domain.update(aggregated_ad)
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

            # --- Hotelling T² ---
            mu = np.mean(training_pca_scores, axis=0)
            cov_matrix = np.cov(training_pca_scores.T)
            if training_pca_scores.shape[1] == 1:
                cov_matrix = np.atleast_2d(cov_matrix)
            try:
                inv_cov = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                inv_cov = np.linalg.inv(cov_matrix)

            n_train = training_pca_scores.shape[0]
            n_comp = training_pca_scores.shape[1]
            from scipy import stats as sp_stats
            if n_train > n_comp:
                t2_threshold = (
                    n_comp * (n_train - 1) / (n_train - n_comp)
                    * sp_stats.f.ppf(0.99, n_comp, n_train - n_comp)
                )
            else:
                t2_threshold = sp_stats.chi2.ppf(0.99, n_comp)

            diff = X_pred_pca - mu
            t2_values = np.array([d @ inv_cov @ d for d in diff])

            # --- Q-residuals (SPE) ---
            # Use pre-computed Q-threshold from full training data if available
            if 'q_threshold' in ad_data:
                q_threshold = float(ad_data['q_threshold'].item())
            else:
                # Fallback for older models: compute from representative spectra
                train_reconstructed = pca_model.inverse_transform(
                    pca_model.transform(representative_spectra)
                )
                train_q = np.sum((representative_spectra - train_reconstructed) ** 2, axis=1)
                q_threshold = np.percentile(train_q, 99)

            new_reconstructed = pca_model.inverse_transform(X_pred_pca)
            q_values = np.sum((X_processed - new_reconstructed) ** 2, axis=1)

            # --- Four-zone classification ---
            t2_flag = t2_values > t2_threshold
            q_flag = q_values > q_threshold
            domain_status = np.empty(len(X_processed), dtype=object)
            domain_status[(~t2_flag) & (~q_flag)] = 'within_domain'
            domain_status[(t2_flag) & (~q_flag)] = 'influential'
            domain_status[(~t2_flag) & (q_flag)] = 'new_features'
            domain_status[(t2_flag) & (q_flag)] = 'outside_domain'

            applicability_domain['t2_values'] = t2_values
            applicability_domain['t2_threshold'] = float(t2_threshold)
            applicability_domain['q_values'] = q_values
            applicability_domain['q_threshold'] = float(q_threshold)
            applicability_domain['domain_status'] = domain_status

            # Compute reliability scores
            training_t2 = ad_data.get('training_t2_values')
            training_q = ad_data.get('training_q_values')
            reliability_scores = _compute_reliability_scores(
                t2_values, q_values, t2_threshold, q_threshold,
                training_t2=training_t2, training_q=training_q
            )
            applicability_domain['reliability_scores'] = reliability_scores

            has_applicability_domain = True

    return {
        'predictions': predictions,
        'uncertainty': uncertainty,
        'has_uncertainty': has_uncertainty,
        'applicability_domain': applicability_domain,
        'has_applicability_domain': has_applicability_domain,
        'data_type_warning': data_type_warning
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
    first_ad_model = None  # For T²/Q computation

    # Collect applicability domain info from each base model
    for model_dict in base_model_dicts:
        if 'ad_data' not in model_dict or model_dict['ad_data'] is None:
            continue

        ad_data = model_dict['ad_data']
        pca_model = model_dict.get('pca_model')

        if pca_model is None:
            continue

        has_any_ad = True
        if first_ad_model is None:
            first_ad_model = model_dict

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

    # --- Hotelling T² and Q-residuals from first base model with AD ---
    if first_ad_model is not None:
        ad_data = first_ad_model['ad_data']
        pca_model = first_ad_model['pca_model']
        training_pca_scores = ad_data['training_pca_scores']
        representative_spectra = ad_data['representative_spectra']

        X_pred_pca = pca_model.transform(X_processed)

        mu = np.mean(training_pca_scores, axis=0)
        cov_matrix = np.cov(training_pca_scores.T)
        if training_pca_scores.shape[1] == 1:
            cov_matrix = np.atleast_2d(cov_matrix)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(cov_matrix)

        n_train = training_pca_scores.shape[0]
        n_comp = training_pca_scores.shape[1]
        from scipy import stats as sp_stats
        if n_train > n_comp:
            t2_threshold = (
                n_comp * (n_train - 1) / (n_train - n_comp)
                * sp_stats.f.ppf(0.99, n_comp, n_train - n_comp)
            )
        else:
            t2_threshold = sp_stats.chi2.ppf(0.99, n_comp)

        diff = X_pred_pca - mu
        t2_values = np.array([d @ inv_cov @ d for d in diff])

        # Use pre-computed Q-threshold from full training data if available
        if 'q_threshold' in ad_data:
            q_threshold = float(ad_data['q_threshold'].item())
        else:
            # Fallback for older models: compute from representative spectra
            train_reconstructed = pca_model.inverse_transform(
                pca_model.transform(representative_spectra)
            )
            train_q = np.sum((representative_spectra - train_reconstructed) ** 2, axis=1)
            q_threshold = np.percentile(train_q, 99)

        new_reconstructed = pca_model.inverse_transform(X_pred_pca)
        q_values = np.sum((X_processed - new_reconstructed) ** 2, axis=1)

        t2_flag = t2_values > t2_threshold
        q_flag = q_values > q_threshold
        domain_status = np.empty(len(X_processed), dtype=object)
        domain_status[(~t2_flag) & (~q_flag)] = 'within_domain'
        domain_status[(t2_flag) & (~q_flag)] = 'influential'
        domain_status[(~t2_flag) & (q_flag)] = 'new_features'
        domain_status[(t2_flag) & (q_flag)] = 'outside_domain'

        applicability_domain['t2_values'] = t2_values
        applicability_domain['t2_threshold'] = float(t2_threshold)
        applicability_domain['q_values'] = q_values
        applicability_domain['q_threshold'] = float(q_threshold)
        applicability_domain['domain_status'] = domain_status

        # Compute reliability scores
        training_t2 = ad_data.get('training_t2_values')
        training_q = ad_data.get('training_q_values')
        reliability_scores = _compute_reliability_scores(
            t2_values, q_values, t2_threshold, q_threshold,
            training_t2=training_t2, training_q=training_q
        )
        applicability_domain['reliability_scores'] = reliability_scores

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
    # Filter out any non-numeric columns first (metadata columns, etc.)
    numeric_cols = []
    non_numeric_cols = []
    for col in df.columns:
        try:
            float(col)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"Note: Ignoring {len(non_numeric_cols)} non-wavelength columns during prediction: {non_numeric_cols[:3]}{'...' if len(non_numeric_cols) > 3 else ''}")

    if not numeric_cols:
        raise ValueError("DataFrame has no numeric wavelength columns. Check that your data contains spectral data.")

    # Use only numeric columns for wavelength matching
    df_numeric = df[numeric_cols]
    available_wl = df_numeric.columns.astype(float).values

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
        matching_cols = []
        for col in df_numeric.columns:
            col_float = float(col)  # Safe now - we filtered to numeric only
            if abs(col_float - required_wl) < 0.01:
                matching_cols.append(col)
        if not matching_cols:
            raise ValueError(f"Required wavelength {required_wl} not found")
        selected_cols.append(matching_cols[0])

    return df_numeric[selected_cols].values


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

    # Validate ensemble attributes
    if not hasattr(ensemble, 'models'):
        raise ValueError("Ensemble must have 'models' attribute containing list of base models")

    if not hasattr(ensemble, 'model_names') or ensemble.model_names is None:
        # Auto-generate model names if missing
        ensemble.model_names = [f"Model_{i}" for i in range(len(ensemble.models))]
        print(f"Warning: ensemble.model_names was missing, auto-generated {len(ensemble.models)} names")

    if len(ensemble.models) != len(ensemble.model_names):
        raise ValueError(
            f"Ensemble models and model_names length mismatch: "
            f"{len(ensemble.models)} models but {len(ensemble.model_names)} names"
        )

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

        # Extract preprocessors from base model dicts
        base_preprocessors = [md.get('preprocessor') for md in base_model_dicts]

        # Load ensemble state
        ensemble_state_path = tmpdir_path / "ensemble_state.pkl"
        ensemble_state = joblib.load(ensemble_state_path)

        # Reconstruct ensemble object
        from spectral_predict.ensemble import (
            SimpleAverageEnsemble,
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
                n_regions=ensemble_state.get('n_regions', 5),
                preprocessors=base_preprocessors
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
                n_regions=ensemble_state.get('n_regions', 5),
                preprocessors=base_preprocessors
            )
            # Restore weights and analyzer
            if 'weights' in ensemble_state:
                ensemble.weights_ = ensemble_state['weights']
            if 'analyzer' in ensemble_state:
                ensemble.analyzer_ = ensemble_state['analyzer']

        elif ensemble_type in ['stacking', 'region_stacking']:
            region_aware = (ensemble_type == 'region_stacking')
            ensemble = StackingEnsemble(
                models=base_models,
                model_names=model_names,
                region_aware=region_aware,
                n_regions=ensemble_state.get('n_regions', 5) if region_aware else None,
                preprocessors=base_preprocessors
            )
            # Restore meta_model and analyzer
            if 'meta_model' in ensemble_state:
                ensemble.meta_model_ = ensemble_state['meta_model']
            if 'analyzer' in ensemble_state:
                ensemble.analyzer_ = ensemble_state['analyzer']

        elif ensemble_type == 'simple_average':
            # Simple average - use imported class from ensemble.py
            ensemble = SimpleAverageEnsemble(
                models=base_models,
                model_names=model_names,
                preprocessors=base_preprocessors
            )
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
