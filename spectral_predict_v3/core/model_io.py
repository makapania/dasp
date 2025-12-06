"""
Model I/O functionality for Spectral Predict v3.

Save, load, and apply trained models with complete preprocessing pipelines.
"""

import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .preprocess import SNV, SavgolDerivative


def save_model(
    model_bundle: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save a model bundle to disk.

    Parameters
    ----------
    model_bundle : dict
        Complete model bundle containing:
        - 'model': fitted sklearn-compatible model
        - 'model_name': str, model name (e.g., 'PLS', 'Ridge')
        - 'preprocessing': str, preprocessing method (e.g., 'snv_deriv1_w7')
        - 'wavelengths': np.ndarray, expected wavelengths
        - 'target_name': str, target variable name
        - 'task_type': str, 'regression' or 'classification'
        - 'metrics': dict, training metrics (RMSE, R2, etc.)
        - 'params': dict, model hyperparameters
        - 'variable_indices': np.ndarray or None, indices if variable selection used
        - 'created': str, ISO timestamp
        - 'version': str, software version
    filepath : str
        Path to save the model bundle (.pkl)

    Raises
    ------
    ValueError
        If model_bundle is missing required keys
    IOError
        If save fails
    """
    required_keys = ['model', 'model_name', 'preprocessing', 'wavelengths',
                     'target_name', 'task_type']

    for key in required_keys:
        if key not in model_bundle:
            raise ValueError(f"model_bundle missing required key: {key}")

    # Add metadata if not present
    if 'created' not in model_bundle:
        model_bundle['created'] = datetime.now().isoformat()
    if 'version' not in model_bundle:
        model_bundle['version'] = '3.0'

    # Ensure path has .pkl extension
    filepath = str(filepath)
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'

    # Save using pickle
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model_bundle, f)
    except Exception as e:
        raise IOError(f"Failed to save model to {filepath}: {e}")


def load_model(filepath: str) -> Dict[str, Any]:
    """
    Load a model bundle from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model bundle (.pkl)

    Returns
    -------
    model_bundle : dict
        Complete model bundle with all metadata

    Raises
    ------
    FileNotFoundError
        If filepath does not exist
    IOError
        If load fails
    ValueError
        If loaded bundle is invalid
    """
    filepath = str(filepath)

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            model_bundle = pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load model from {filepath}: {e}")

    # Validate bundle
    required_keys = ['model', 'model_name', 'preprocessing', 'wavelengths',
                     'target_name', 'task_type']
    for key in required_keys:
        if key not in model_bundle:
            raise ValueError(f"Loaded model bundle is missing required key: {key}")

    return model_bundle


def apply_model(
    model_bundle: Dict[str, Any],
    X_new: np.ndarray,
    wavelengths_new: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply a trained model to new spectral data.

    Handles wavelength interpolation, preprocessing, variable selection,
    and prediction in the correct order.

    Parameters
    ----------
    model_bundle : dict
        Loaded model bundle from load_model()
    X_new : np.ndarray
        New spectral data, shape (n_samples, n_features)
    wavelengths_new : np.ndarray, optional
        Wavelengths for new spectra. If provided and different from
        training wavelengths, interpolation will be performed.

    Returns
    -------
    predictions : np.ndarray
        Model predictions, shape (n_samples,)
    info : dict
        Information about the prediction process:
        - 'interpolated': bool, whether wavelength interpolation was performed
        - 'preprocessing_applied': str, preprocessing method applied
        - 'variables_selected': int or None, number of variables selected

    Raises
    ------
    ValueError
        If wavelength mismatch cannot be resolved
        If preprocessing fails
        If prediction fails
    """
    X_new = np.asarray(X_new)
    info = {
        'interpolated': False,
        'preprocessing_applied': model_bundle['preprocessing'],
        'variables_selected': None
    }

    # Extract model bundle components
    model = model_bundle['model']
    preprocessing = model_bundle['preprocessing']
    wavelengths_train = model_bundle['wavelengths']
    variable_indices = model_bundle.get('variable_indices', None)

    # Step 1: Handle wavelength mismatch if wavelengths provided
    if wavelengths_new is not None:
        wavelengths_new = np.asarray(wavelengths_new)

        # Check if interpolation is needed
        if not np.array_equal(wavelengths_new, wavelengths_train):
            X_new = _interpolate_spectra(
                X_new, wavelengths_new, wavelengths_train
            )
            info['interpolated'] = True

    # Step 2: Apply preprocessing
    try:
        X_processed = _apply_preprocessing(X_new, preprocessing)
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")

    # Step 3: Apply variable selection if used during training
    if variable_indices is not None:
        variable_indices = np.asarray(variable_indices)

        # Validate indices
        if np.max(variable_indices) >= X_processed.shape[1]:
            raise ValueError(
                f"Variable indices exceed feature count. "
                f"Max index: {np.max(variable_indices)}, "
                f"n_features: {X_processed.shape[1]}"
            )

        X_processed = X_processed[:, variable_indices]
        info['variables_selected'] = len(variable_indices)

    # Step 4: Make predictions
    try:
        predictions = model.predict(X_processed)
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

    return predictions, info


def _interpolate_spectra(
    X: np.ndarray,
    wavelengths_from: np.ndarray,
    wavelengths_to: np.ndarray
) -> np.ndarray:
    """
    Interpolate spectra from one wavelength grid to another.

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_features_from)
    wavelengths_from : np.ndarray
        Current wavelengths, shape (n_features_from,)
    wavelengths_to : np.ndarray
        Target wavelengths, shape (n_features_to,)

    Returns
    -------
    X_interp : np.ndarray
        Interpolated spectra, shape (n_samples, n_features_to)

    Raises
    ------
    ValueError
        If wavelength ranges don't overlap sufficiently
    """
    # Validate wavelength compatibility
    overlap_min = max(wavelengths_from.min(), wavelengths_to.min())
    overlap_max = min(wavelengths_from.max(), wavelengths_to.max())

    if overlap_min >= overlap_max:
        raise ValueError(
            f"No wavelength overlap. "
            f"New range: [{wavelengths_from.min():.1f}, {wavelengths_from.max():.1f}], "
            f"Required range: [{wavelengths_to.min():.1f}, {wavelengths_to.max():.1f}]"
        )

    # Check if target wavelengths are mostly within source range
    n_outside = np.sum((wavelengths_to < wavelengths_from.min()) |
                       (wavelengths_to > wavelengths_from.max()))

    if n_outside > len(wavelengths_to) * 0.1:  # More than 10% outside
        raise ValueError(
            f"Insufficient wavelength overlap. "
            f"{n_outside}/{len(wavelengths_to)} target wavelengths are outside source range. "
            f"New range: [{wavelengths_from.min():.1f}, {wavelengths_from.max():.1f}], "
            f"Required range: [{wavelengths_to.min():.1f}, {wavelengths_to.max():.1f}]"
        )

    # Interpolate each spectrum
    X_interp = np.zeros((X.shape[0], len(wavelengths_to)))

    for i in range(X.shape[0]):
        # Use linear interpolation, extrapolate for small gaps
        f = interp1d(
            wavelengths_from, X[i, :],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        X_interp[i, :] = f(wavelengths_to)

    return X_interp


def _apply_preprocessing(X: np.ndarray, preprocessing: str) -> np.ndarray:
    """
    Apply preprocessing transformation to spectra.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data, shape (n_samples, n_features)
    preprocessing : str
        Preprocessing method name, e.g.:
        - 'raw': no preprocessing
        - 'snv': Standard Normal Variate
        - 'deriv1_w7': 1st derivative with window 7
        - 'deriv2_w19': 2nd derivative with window 19
        - 'snv_deriv1_w7': SNV then 1st derivative
        - 'snv_deriv2_w19': SNV then 2nd derivative

    Returns
    -------
    X_processed : np.ndarray
        Preprocessed spectra, same shape as X

    Raises
    ------
    ValueError
        If preprocessing method is unknown or fails
    """
    if preprocessing == 'raw':
        return X

    # Parse preprocessing string
    parts = preprocessing.split('_')

    # Check for SNV prefix
    apply_snv = False
    if parts[0] == 'snv':
        apply_snv = True
        parts = parts[1:]  # Remove 'snv' from parts

    # Apply SNV if needed
    if apply_snv:
        X = SNV().fit_transform(X)

    # If only SNV, return now
    if len(parts) == 0:
        return X

    # Check for derivative
    if parts[0].startswith('deriv'):
        # Extract derivative order and window
        deriv_part = parts[0]

        # Parse derivative order (deriv1 or deriv2)
        if deriv_part == 'deriv1':
            deriv_order = 1
        elif deriv_part == 'deriv2':
            deriv_order = 2
        else:
            raise ValueError(f"Unknown derivative type: {deriv_part}")

        # Parse window size (from w7, w19, etc.)
        window = 7  # default
        if len(parts) > 1 and parts[1].startswith('w'):
            try:
                window = int(parts[1][1:])  # Extract number after 'w'
            except ValueError:
                raise ValueError(f"Invalid window size: {parts[1]}")

        # Apply derivative
        X = SavgolDerivative(deriv=deriv_order, window=window).fit_transform(X)

    return X


def kennard_stone_selection(X: np.ndarray, n_select: int) -> np.ndarray:
    """
    Select diverse representative samples using Kennard-Stone algorithm.

    This algorithm iteratively selects samples that are maximally different
    from already selected samples, ensuring good coverage of the input space.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape (n_samples, n_features)
    n_select : int
        Number of samples to select

    Returns
    -------
    selected_indices : np.ndarray
        Indices of selected samples, shape (n_select,)

    Notes
    -----
    - First sample: the one with maximum total distance to all others
    - Subsequent samples: the one with maximum distance to nearest selected sample
    """
    if n_select >= X.shape[0]:
        return np.arange(X.shape[0])

    # Compute pairwise distances
    distances = pairwise_distances(X, metric='euclidean')

    # Select first sample: most different from all others
    selected = [int(np.argmax(np.sum(distances, axis=1)))]

    # Iteratively select remaining samples
    while len(selected) < n_select:
        # Get indices of remaining samples
        remaining = [i for i in range(len(X)) if i not in selected]

        # For each remaining sample, find distance to nearest selected sample
        min_dists = distances[remaining][:, selected].min(axis=1)

        # Select the sample with maximum distance to nearest selected
        next_idx = remaining[int(np.argmax(min_dists))]
        selected.append(next_idx)

    return np.array(selected, dtype=int)


def _compute_applicability_domain(
    X_train: np.ndarray,
    max_representatives: int = 150,
    n_components: int = 10
) -> Dict[str, Any]:
    """
    Compute applicability domain data for a training set.

    Uses PCA projection and distance-based thresholds to define the domain
    where the model's predictions are reliable.

    Parameters
    ----------
    X_train : np.ndarray
        Training spectra (preprocessed), shape (n_samples, n_features)
    max_representatives : int, optional
        Maximum number of representative samples to store (default: 150)
        If n_samples > max_representatives, uses Kennard-Stone selection
    n_components : int, optional
        Number of PCA components to use (default: 10)

    Returns
    -------
    ad_data : dict
        Dictionary containing:
        - 'representative_spectra': Selected training spectra for comparison
        - 'pca_model': Fitted PCA model
        - 'representative_pca_scores': PCA scores of representative samples
        - 'distance_thresholds': Percentile-based distance thresholds
          {'p50': float, 'p75': float, 'p95': float, 'max': float}
    """
    n_samples = X_train.shape[0]

    # Select representative samples
    if n_samples <= max_representatives:
        # Use all training samples
        representative_indices = np.arange(n_samples)
        X_repr = X_train.copy()
    else:
        # Use Kennard-Stone to select diverse subset
        representative_indices = kennard_stone_selection(X_train, max_representatives)
        X_repr = X_train[representative_indices, :]

    # Fit PCA on all training data
    n_components_actual = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components_actual)
    train_pca_scores = pca.fit_transform(X_train)

    # Get PCA scores for representatives
    repr_pca_scores = train_pca_scores[representative_indices, :]

    # Compute pairwise distances in PCA space (within training set)
    pairwise_dists = pairwise_distances(train_pca_scores, metric='euclidean')

    # For each training sample, find distance to nearest OTHER training sample
    # (set diagonal to infinity to exclude self-distance)
    np.fill_diagonal(pairwise_dists, np.inf)
    min_distances = pairwise_dists.min(axis=1)

    # Compute percentile thresholds
    thresholds = {
        'p50': float(np.percentile(min_distances, 50)),
        'p75': float(np.percentile(min_distances, 75)),
        'p95': float(np.percentile(min_distances, 95)),
        'max': float(np.max(min_distances))
    }

    return {
        'representative_spectra': X_repr,
        'pca_model': pca,
        'representative_pca_scores': repr_pca_scores,
        'distance_thresholds': thresholds
    }


def compute_applicability_status(
    model_bundle: Dict[str, Any],
    X_new: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute applicability domain status for new spectra.

    Compares new spectra to the training domain using PCA distance.

    Parameters
    ----------
    model_bundle : dict
        Model bundle containing applicability domain data
    X_new : np.ndarray
        New spectra (preprocessed), shape (n_samples, n_features)

    Returns
    -------
    ad_results : dict
        Dictionary containing:
        - 'pca_distance': Distance to nearest training sample in PCA space
        - 'status': Classification as 'good', 'caution', or 'extrapolation'
        - 'nearest_sample_idx': Index of nearest representative sample

    Notes
    -----
    Status thresholds:
    - 'good': distance <= p75
    - 'caution': p75 < distance <= p95
    - 'extrapolation': distance > p95
    """
    # Check if model has applicability domain data
    if not model_bundle.get('has_applicability_domain', False):
        # Return None values if no AD data
        n = X_new.shape[0]
        return {
            'pca_distance': np.full(n, np.nan),
            'status': np.array(['N/A'] * n),
            'nearest_sample_idx': np.full(n, -1, dtype=int)
        }

    # Extract AD components
    pca_model = model_bundle['pca_model']
    repr_pca_scores = model_bundle['representative_pca_scores']
    thresholds = model_bundle['distance_thresholds']

    # Project new spectra to PCA space
    X_new_pca = pca_model.transform(X_new)

    # Compute distances to all representative samples
    distances = cdist(X_new_pca, repr_pca_scores, metric='euclidean')

    # Find minimum distance and nearest sample for each new spectrum
    min_distances = distances.min(axis=1)
    nearest_indices = distances.argmin(axis=1)

    # Assign status based on thresholds
    status = np.empty(len(min_distances), dtype='<U15')
    for i, dist in enumerate(min_distances):
        if dist <= thresholds['p75']:
            status[i] = 'good'
        elif dist <= thresholds['p95']:
            status[i] = 'caution'
        else:
            status[i] = 'extrapolation'

    return {
        'pca_distance': min_distances,
        'status': status,
        'nearest_sample_idx': nearest_indices
    }


def create_model_bundle(
    model,
    model_name: str,
    preprocessing: str,
    wavelengths: np.ndarray,
    target_name: str,
    task_type: str,
    metrics: Dict[str, float],
    params: Dict[str, Any] = None,
    variable_indices: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Create a complete model bundle for saving.

    This is a convenience function to package all model components
    together before saving.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible model
    model_name : str
        Model name (e.g., 'PLS', 'Ridge', 'RandomForest')
    preprocessing : str
        Preprocessing method applied (e.g., 'snv_deriv1_w7')
    wavelengths : np.ndarray
        Training wavelengths
    target_name : str
        Name of the target variable
    task_type : str
        'regression' or 'classification'
    metrics : dict
        Training metrics (e.g., {'RMSE': 0.45, 'R2': 0.92})
    params : dict, optional
        Model hyperparameters
    variable_indices : np.ndarray, optional
        Indices of selected variables (if variable selection was used)
    X_train : np.ndarray, optional
        Preprocessed training spectra for applicability domain computation
        If provided, AD data will be computed and included in bundle

    Returns
    -------
    model_bundle : dict
        Complete model bundle ready for saving
    """
    bundle = {
        'model': model,
        'model_name': model_name,
        'preprocessing': preprocessing,
        'wavelengths': np.asarray(wavelengths),
        'target_name': target_name,
        'task_type': task_type,
        'metrics': metrics,
        'params': params or {},
        'variable_indices': np.asarray(variable_indices) if variable_indices is not None else None,
        'created': datetime.now().isoformat(),
        'version': '3.0',
        # Applicability domain fields
        'has_applicability_domain': False,
        'representative_spectra': None,
        'pca_model': None,
        'representative_pca_scores': None,
        'distance_thresholds': None
    }

    # Compute applicability domain if training data provided
    if X_train is not None:
        try:
            ad_data = _compute_applicability_domain(X_train)
            bundle.update({
                'has_applicability_domain': True,
                'representative_spectra': ad_data['representative_spectra'],
                'pca_model': ad_data['pca_model'],
                'representative_pca_scores': ad_data['representative_pca_scores'],
                'distance_thresholds': ad_data['distance_thresholds']
            })
        except Exception as e:
            # If AD computation fails, continue without it
            print(f"Warning: Could not compute applicability domain: {e}")
            bundle['has_applicability_domain'] = False

    return bundle
