"""
Preprocessing Discovery Module - Smart Preprocessing Selection

Replaces the broken GA preprocessing with an NSGA-II-inspired approach:
1. Exhaustively test preprocessing combinations
2. Use user-selectable importance methods for wavelength selection
3. Return top N diverse configurations for grid search

This module does NOT modify grid search - it only provides preprocessing configs.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple, Any
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder

from .constants import RANDOM_STATE


# =============================================================================
# Preprocessing Candidates
# =============================================================================

# All preprocessing types to test (includes all 4 derivatives per user request)
PREPROCESSING_CANDIDATES = [
    # (name, deriv_order, requires_window)
    ('raw', None, False),
    ('snv', None, False),
    # 1st derivative
    ('deriv1', 1, True),
    ('snv_deriv1', 1, True),
    ('deriv1_snv', 1, True),
    # 2nd derivative
    ('deriv2', 2, True),
    ('snv_deriv2', 2, True),
    ('deriv2_snv', 2, True),
    # 3rd derivative
    ('deriv3', 3, True),
    ('snv_deriv3', 3, True),
    ('deriv3_snv', 3, True),
    # 4th derivative
    ('deriv4', 4, True),
    ('snv_deriv4', 4, True),
    ('deriv4_snv', 4, True),
]

# Window sizes to test (5 representative values for fast discovery)
# Covers small (7), medium (11, 17), and large (25, 31) windows
WINDOW_SIZES = [7, 11, 17, 25, 31]

# Wavelength subset sizes to test
SUBSET_SIZES = [50, 100, 200, 300]

# Complexity scores for ranking
PREPROCESSING_COMPLEXITY = {
    'raw': 0.0,
    'snv': 0.1,
    'deriv1': 0.2,
    'snv_deriv1': 0.3,
    'deriv1_snv': 0.3,
    'deriv2': 0.4,
    'snv_deriv2': 0.5,
    'deriv2_snv': 0.5,
    'deriv3': 0.6,
    'snv_deriv3': 0.7,
    'deriv3_snv': 0.7,
    'deriv4': 0.8,
    'snv_deriv4': 0.9,
    'deriv4_snv': 0.9,
}


# =============================================================================
# Importance Methods
# =============================================================================

IMPORTANCE_METHODS = {
    'cars_tree': {
        'name': 'CARS-Tree (Hybrid)',
        'description': 'LightGBM split+gain blend. Dense, stable. Good for tree models.',
    },
    'model_specific': {
        'name': 'Model-Specific',
        'description': 'VIP for PLS, tree importance for RF/LightGBM, coefficients for Ridge.',
    },
    'lightgbm': {
        'name': 'LightGBM Gain',
        'description': 'Native LightGBM importance (gain-based). Fast, sparse.',
    },
    'vip': {
        'name': 'PLS VIP',
        'description': 'Variable Importance in Projection. Chemometrics standard.',
    }
}


def compute_importance(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'model_specific',
    model_name: Optional[str] = None,
    task_type: str = 'regression'
) -> np.ndarray:
    """
    Compute wavelength importance using user-selected method.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    method : str
        One of: 'cars_tree', 'model_specific', 'lightgbm', 'vip'
    model_name : str, optional
        Model name for model-specific importance
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    importance : np.ndarray
        Normalized importance scores [0, 1] for each wavelength
    """
    n_samples, n_features = X.shape

    # Handle classification labels
    y_encoded = y
    if task_type == 'classification' and y.dtype == object:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

    if method == 'cars_tree':
        importance = _compute_cars_tree_importance(X, y_encoded, task_type)

    elif method == 'model_specific':
        importance = _compute_model_specific_importance(
            X, y_encoded, model_name, task_type
        )

    elif method == 'lightgbm':
        importance = _compute_lightgbm_importance(X, y_encoded, task_type)

    elif method == 'vip':
        importance = _compute_vip_importance(X, y_encoded)

    else:
        raise ValueError(f"Unknown importance method: {method}")

    # Normalize to [0, 1]
    importance = np.asarray(importance, dtype=float)
    max_imp = importance.max()
    if max_imp > 0:
        importance = importance / max_imp

    return importance


def _compute_cars_tree_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str
) -> np.ndarray:
    """CARS-Tree hybrid importance using LightGBM split+gain blend."""
    try:
        from .variable_selection import cars_selection

        n_samples = X.shape[0]
        cv_folds = min(5, max(2, n_samples // 5))

        importance = cars_selection(
            X, y,
            n_iterations=50,
            model_type='LightGBM',
            use_hybrid_importance=True,
            hybrid_importance_weight=0.5,
            cv_folds=cv_folds,
            random_state=RANDOM_STATE
        )
        return importance

    except Exception as e:
        print(f"CARS-Tree failed ({e}), falling back to LightGBM importance")
        return _compute_lightgbm_importance(X, y, task_type)


def _compute_lightgbm_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str
) -> np.ndarray:
    """Native LightGBM feature importance (gain-based)."""
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier

        if task_type == 'classification':
            model = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1
            )
        else:
            model = LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1
            )

        model.fit(X, y)
        return model.feature_importances_

    except Exception as e:
        print(f"LightGBM importance failed ({e}), falling back to VIP")
        return _compute_vip_importance(X, y)


def _compute_vip_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """PLS Variable Importance in Projection (VIP) scores."""
    try:
        from .models import compute_vip

        n_samples, n_features = X.shape
        n_components = min(10, n_features // 10, n_samples // 2)
        n_components = max(2, n_components)

        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y)

        return compute_vip(pls, X, y)

    except Exception as e:
        print(f"VIP computation failed ({e}), using uniform importance")
        return np.ones(X.shape[1])


def _compute_model_specific_importance(
    X: np.ndarray,
    y: np.ndarray,
    model_name: Optional[str],
    task_type: str
) -> np.ndarray:
    """Compute importance using native method for specific model type."""

    if model_name is None:
        # Default to LightGBM for tree models, VIP for linear
        return _compute_lightgbm_importance(X, y, task_type)

    # PLS models - use VIP
    if model_name in ('PLS', 'PLS-DA'):
        return _compute_vip_importance(X, y)

    # Linear models - use coefficient-based importance
    elif model_name in ('Ridge', 'Lasso', 'ElasticNet'):
        return _compute_coefficient_importance(X, y, model_name)

    # Tree models - use native tree importance
    elif model_name in ('RandomForest', 'LightGBM', 'XGBoost', 'CatBoost'):
        return _compute_tree_importance(X, y, model_name, task_type)

    # Neural models - use first layer weights
    elif model_name in ('MLP', 'NeuralBoosted'):
        return _compute_neural_importance(X, y, task_type)

    # SVM - use support vector based importance
    elif model_name in ('SVR', 'SVC'):
        return _compute_svm_importance(X, y, task_type)

    else:
        # Default to LightGBM
        return _compute_lightgbm_importance(X, y, task_type)


def _compute_coefficient_importance(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str
) -> np.ndarray:
    """Coefficient-based importance for linear models."""
    try:
        from sklearn.linear_model import Ridge, Lasso, ElasticNet

        if model_name == 'Ridge':
            model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        elif model_name == 'Lasso':
            model = Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000)
        else:  # ElasticNet
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000)

        model.fit(X, y)
        return np.abs(model.coef_).ravel()

    except Exception as e:
        print(f"Coefficient importance failed ({e}), falling back to VIP")
        return _compute_vip_importance(X, y)


def _compute_tree_importance(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task_type: str
) -> np.ndarray:
    """Native tree model importance."""
    try:
        if model_name == 'RandomForest':
            if task_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1
                )

        elif model_name == 'XGBoost':
            from xgboost import XGBRegressor, XGBClassifier
            if task_type == 'classification':
                model = XGBClassifier(
                    n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=1, verbosity=0
                )
            else:
                model = XGBRegressor(
                    n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=1, verbosity=0
                )

        elif model_name == 'CatBoost':
            from catboost import CatBoostRegressor, CatBoostClassifier
            if task_type == 'classification':
                model = CatBoostClassifier(
                    n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=0
                )
            else:
                model = CatBoostRegressor(
                    n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=0
                )
        else:
            # Default to LightGBM
            return _compute_lightgbm_importance(X, y, task_type)

        model.fit(X, y)
        return model.feature_importances_

    except Exception as e:
        print(f"Tree importance failed ({e}), falling back to LightGBM")
        return _compute_lightgbm_importance(X, y, task_type)


def _compute_neural_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str
) -> np.ndarray:
    """Neural network importance using first layer weights."""
    try:
        from sklearn.neural_network import MLPRegressor, MLPClassifier

        if task_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE
            )

        model.fit(X, y)

        # First layer weights: (n_features, n_hidden)
        weights = model.coefs_[0]
        return np.mean(np.abs(weights), axis=1)

    except Exception as e:
        print(f"Neural importance failed ({e}), falling back to LightGBM")
        return _compute_lightgbm_importance(X, y, task_type)


def _compute_svm_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str
) -> np.ndarray:
    """SVM importance using coefficients or support vectors."""
    try:
        from sklearn.svm import SVR, SVC
        from sklearn.preprocessing import StandardScaler

        # Scale data for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task_type == 'classification':
            model = SVC(kernel='linear', random_state=RANDOM_STATE)
        else:
            model = SVR(kernel='linear')

        model.fit(X_scaled, y)

        # For linear kernel, use coefficients
        if hasattr(model, 'coef_'):
            return np.abs(model.coef_).ravel()
        else:
            # For non-linear, fall back to LightGBM
            return _compute_lightgbm_importance(X, y, task_type)

    except Exception as e:
        print(f"SVM importance failed ({e}), falling back to LightGBM")
        return _compute_lightgbm_importance(X, y, task_type)


# =============================================================================
# Preprocessing Application
# =============================================================================

def apply_preprocessing(
    X: np.ndarray,
    preproc_name: str,
    window: Optional[int] = None
) -> np.ndarray:
    """
    Apply preprocessing to spectral data.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    preproc_name : str
        Preprocessing name (e.g., 'snv_deriv1', 'deriv2_snv')
    window : int, optional
        Window size for derivatives

    Returns
    -------
    X_preproc : np.ndarray
        Preprocessed spectral data
    """
    from .preprocess import SNV, SavgolDerivative

    X = np.asarray(X)

    if preproc_name == 'raw':
        return X.copy()

    elif preproc_name == 'snv':
        return SNV().transform(X)

    # Parse preprocessing name for derivative order
    deriv_order = None
    for d in [4, 3, 2, 1]:  # Check higher derivatives first
        if f'deriv{d}' in preproc_name:
            deriv_order = d
            break

    if deriv_order is None:
        raise ValueError(f"Unknown preprocessing: {preproc_name}")

    # Determine polyorder (deriv + 1)
    polyorder = deriv_order + 1

    # Ensure window is valid
    if window is None:
        window = 17  # Default
    if window % 2 == 0:
        window += 1
    if window < polyorder + 2:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1

    # Apply based on ordering
    if preproc_name.startswith('snv_deriv'):
        # SNV first, then derivative
        X_snv = SNV().transform(X)
        return SavgolDerivative(deriv=deriv_order, window=window, polyorder=polyorder).transform(X_snv)

    elif preproc_name.startswith('deriv') and preproc_name.endswith('_snv'):
        # Derivative first, then SNV
        X_deriv = SavgolDerivative(deriv=deriv_order, window=window, polyorder=polyorder).transform(X)
        return SNV().transform(X_deriv)

    elif preproc_name.startswith('deriv'):
        # Derivative only
        return SavgolDerivative(deriv=deriv_order, window=window, polyorder=polyorder).transform(X)

    else:
        raise ValueError(f"Unknown preprocessing: {preproc_name}")


def get_edge_zone(preproc_name: str, window: Optional[int]) -> int:
    """Get edge zone size for preprocessing (wavelengths to exclude)."""
    if window is None:
        return 0

    # Only derivatives need edge masking
    for d in [4, 3, 2, 1]:
        if f'deriv{d}' in preproc_name:
            return window // 2

    return 0


# =============================================================================
# Wavelength Selection
# =============================================================================

def select_wavelengths_by_importance(
    importance: np.ndarray,
    target_n: int = 200,
    edge_zone: int = 0
) -> np.ndarray:
    """
    Select top N wavelengths by importance, excluding edges.

    Parameters
    ----------
    importance : np.ndarray
        Importance scores for each wavelength
    target_n : int
        Number of wavelengths to select
    edge_zone : int
        Number of edge wavelengths to exclude on each side

    Returns
    -------
    indices : np.ndarray
        Selected wavelength indices (in spectral order)
    """
    importance = importance.copy()

    # Zero out edge zones
    if edge_zone > 0:
        importance[:edge_zone] = 0
        importance[-edge_zone:] = 0

    # Get available (non-zero) wavelengths
    available = np.where(importance > 0)[0]

    if len(available) == 0:
        # All wavelengths masked, fall back to center region
        n_features = len(importance)
        center_start = edge_zone if edge_zone > 0 else n_features // 10
        center_end = n_features - center_start
        indices = np.arange(center_start, min(center_start + target_n, center_end))
        return indices

    # Select top N by importance
    n_select = min(target_n, len(available))

    # Sort by importance (descending) and take top N
    sorted_idx = np.argsort(importance)[::-1]
    selected = sorted_idx[:n_select]

    # Return in spectral order for reproducibility
    return np.sort(selected)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_preprocessing_config(
    X: np.ndarray,
    y: np.ndarray,
    preproc_name: str,
    window: Optional[int],
    importance_method: str,
    model_name: Optional[str],
    task_type: str,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Evaluate one preprocessing config using ALL wavelengths (like grid search does).

    NOTE: We evaluate with full wavelengths to match grid search behavior.
    Importance-based wavelength selection is computed but stored separately
    for potential use in variable selection.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data
    y : np.ndarray
        Target values
    preproc_name : str
        Preprocessing name
    window : int, optional
        Window size for derivatives
    importance_method : str
        Importance method to use
    model_name : str, optional
        Model name for model-specific importance
    task_type : str
        'regression' or 'classification'
    cv_folds : int
        Number of CV folds

    Returns
    -------
    result : dict
        Configuration results
    """
    try:
        # 1. Apply preprocessing
        X_preproc = apply_preprocessing(X, preproc_name, window)

        # 2. Compute edge zone and apply edge masking (like grid search does)
        edge_zone = get_edge_zone(preproc_name, window)
        if edge_zone > 0:
            X_eval = X_preproc[:, edge_zone:-edge_zone]
        else:
            X_eval = X_preproc

        # 3. Evaluate with ALL wavelengths (after edge masking) - matches grid search
        score = _quick_evaluate(X_eval, y, task_type, cv_folds)

        # 4. Compute importance for potential variable selection use
        # (stored but not used for evaluation score)
        try:
            importance = compute_importance(
                X_eval, y, method=importance_method,
                model_name=model_name, task_type=task_type
            )
            # Select wavelengths using default subset size (200)
            selected_wavelengths = select_wavelengths_by_importance(
                importance, target_n=200, edge_zone=0  # Edge already applied
            )
            # Adjust indices to account for edge masking
            if edge_zone > 0:
                selected_wavelengths = selected_wavelengths + edge_zone
        except Exception:
            selected_wavelengths = None

        # Parse deriv order from name
        deriv_order = None
        for d in [4, 3, 2, 1]:
            if f'deriv{d}' in preproc_name:
                deriv_order = d
                break

        return {
            'preprocessing': preproc_name,
            'window': window,
            'deriv': deriv_order,
            'polyorder': deriv_order + 1 if deriv_order else None,
            'selected_wavelengths': selected_wavelengths,  # For potential variable selection
            'n_wavelengths': X_eval.shape[1],  # Actual wavelengths used for evaluation
            'score': score,  # RMSE or accuracy (on full wavelengths)
            'importance_method': importance_method,
            'model_name': model_name
        }

    except Exception as e:
        print(f"  Error evaluating {preproc_name} w={window}: {e}")
        return None


def _quick_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int
) -> float:
    """Quick cross-validated evaluation using fast model."""
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier

        n_samples = X.shape[0]
        cv_folds = min(cv_folds, n_samples // 2)
        cv_folds = max(2, cv_folds)

        if task_type == 'classification':
            model = LGBMClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1
            )
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            return scores.mean()  # Higher is better
        else:
            model = LGBMRegressor(
                n_estimators=50,
                max_depth=4,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1
            )
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
            return -scores.mean()  # Return positive RMSE (lower is better)

    except Exception:
        # Fall back to PLS
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score

        n_components = min(10, X.shape[1] // 10, X.shape[0] // 2)
        n_components = max(2, n_components)

        pls = PLSRegression(n_components=n_components)
        scores = cross_val_score(pls, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
        return -scores.mean()


# =============================================================================
# Scoring and Ranking
# =============================================================================

def score_config(config: Dict, all_configs: List[Dict], task_type: str) -> float:
    """
    Score a config based on error, complexity, and wavelength count.

    Lower score = better config.
    """
    scores = [c['score'] for c in all_configs]
    wavelengths = [c['n_wavelengths'] for c in all_configs]

    # Normalize score (0 = best, 1 = worst)
    if task_type == 'regression':
        # For regression, lower RMSE is better
        min_score, max_score = min(scores), max(scores)
        if max_score > min_score:
            score_norm = (config['score'] - min_score) / (max_score - min_score)
        else:
            score_norm = 0
    else:
        # For classification, higher accuracy is better
        min_score, max_score = min(scores), max(scores)
        if max_score > min_score:
            score_norm = 1 - (config['score'] - min_score) / (max_score - min_score)
        else:
            score_norm = 0

    # Wavelength count (prefer fewer)
    wavelength_norm = config['n_wavelengths'] / max(wavelengths)

    # Preprocessing complexity
    complexity_norm = PREPROCESSING_COMPLEXITY.get(config['preprocessing'], 0.5)

    # Weighted combination (error most important)
    return 0.6 * score_norm + 0.25 * wavelength_norm + 0.15 * complexity_norm


def select_diverse_configs(
    configs: List[Dict],
    n_top: int,
    task_type: str
) -> List[Dict]:
    """
    Select top N diverse configurations.

    Ensures diversity in preprocessing types, not just top N by score.
    """
    if len(configs) <= n_top:
        return configs

    # Score all configs
    for config in configs:
        config['_combined_score'] = score_config(config, configs, task_type)

    # Sort by combined score
    sorted_configs = sorted(configs, key=lambda c: c['_combined_score'])

    # Select ensuring diversity
    selected = []
    selected_preprocs = set()

    # First pass: select best from each preprocessing type
    for config in sorted_configs:
        preproc = config['preprocessing']
        if preproc not in selected_preprocs:
            selected.append(config)
            selected_preprocs.add(preproc)
            if len(selected) >= n_top:
                break

    # Second pass: fill remaining slots with best overall
    if len(selected) < n_top:
        for config in sorted_configs:
            if config not in selected:
                selected.append(config)
                if len(selected) >= n_top:
                    break

    # Clean up temporary score
    for config in selected:
        if '_combined_score' in config:
            del config['_combined_score']

    return selected


# =============================================================================
# Main Discovery Function
# =============================================================================

def discover_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    models_to_test: Optional[List[str]] = None,
    task_type: str = 'regression',
    importance_method: str = 'model_specific',
    n_top: int = 10,
    cv_folds: int = 5,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Discover top N preprocessing configurations using NSGA-II-style intelligence.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    models_to_test : list, optional
        Models that will be tested (for model-specific importance)
    task_type : str
        'regression' or 'classification'
    importance_method : str
        One of: 'cars_tree', 'model_specific', 'lightgbm', 'vip'
    n_top : int
        Number of configs to return
    cv_folds : int
        Number of CV folds
    progress_callback : callable, optional
        Progress callback function(current, total, message)

    Returns
    -------
    configs : list of dict
        Top N preprocessing configs with selected wavelengths
    """
    X = np.asarray(X)
    y = np.asarray(y)

    print(f"\n=== Smart Preprocessing Discovery ===")
    print(f"Importance method: {importance_method}")
    print(f"Task type: {task_type}")
    print(f"Data shape: {X.shape}")
    print(f"Looking for top {n_top} configurations")

    # Build list of all preprocessing combinations to test
    combinations = []
    for preproc_name, deriv_order, requires_window in PREPROCESSING_CANDIDATES:
        if requires_window:
            for window in WINDOW_SIZES:
                combinations.append((preproc_name, window))
        else:
            combinations.append((preproc_name, None))

    print(f"Testing {len(combinations)} preprocessing combinations...")

    # For initial evaluation, use LightGBM importance (fast) to get scores
    # Model-specific importance will be computed later for each model
    initial_importance_method = 'lightgbm' if importance_method == 'model_specific' else importance_method

    # Evaluate all combinations
    all_configs = []
    total = len(combinations)

    for i, (preproc_name, window) in enumerate(combinations):
        # Progress update
        if progress_callback:
            msg = f"Evaluating {preproc_name}" + (f" w={window}" if window else "")
            progress_callback(i + 1, total, msg)

        # Print progress
        window_str = f"w={window}" if window else ""
        print(f"  [{i+1}/{total}] {preproc_name} {window_str}...", end=" ", flush=True)

        # Evaluate this config (uses fast importance for initial ranking)
        result = evaluate_preprocessing_config(
            X, y,
            preproc_name=preproc_name,
            window=window,
            importance_method=initial_importance_method,
            model_name=None,
            task_type=task_type,
            cv_folds=cv_folds
        )

        if result is not None:
            all_configs.append(result)
            score_str = f"RMSE={result['score']:.4f}" if task_type == 'regression' else f"Acc={result['score']:.4f}"
            print(f"{score_str}, {result['n_wavelengths']} wavelengths")
        else:
            print("FAILED")

    if not all_configs:
        print("ERROR: No valid preprocessing configurations found!")
        return []

    # Select diverse top N
    print(f"\nSelecting top {n_top} diverse configurations...")
    top_configs = select_diverse_configs(all_configs, n_top, task_type)

    # For model_specific importance with multiple models, compute per-model importance
    if importance_method == 'model_specific' and models_to_test and len(models_to_test) > 1:
        print(f"\n=== Computing Model-Specific Importance ===")
        print(f"Models: {models_to_test}")

        expanded_configs = []
        for config in top_configs:
            preproc_name = config['preprocessing']
            window = config.get('window')

            # Apply preprocessing to get X for importance calculation
            X_preproc = apply_preprocessing(X, preproc_name, window)
            edge_zone = get_edge_zone(preproc_name, window)
            if edge_zone > 0:
                X_eval = X_preproc[:, edge_zone:-edge_zone]
            else:
                X_eval = X_preproc

            # Compute importance for each model
            for model_name in models_to_test:
                print(f"  {preproc_name} w={window} -> {model_name}...", end=" ", flush=True)
                try:
                    importance = _compute_model_specific_importance(
                        X_eval, y, model_name, task_type
                    )
                    selected_wavelengths = select_wavelengths_by_importance(
                        importance, target_n=200, edge_zone=0
                    )
                    if edge_zone > 0:
                        selected_wavelengths = selected_wavelengths + edge_zone

                    # Create model-specific config copy
                    model_config = config.copy()
                    model_config['selected_wavelengths'] = selected_wavelengths
                    model_config['importance_method'] = 'model_specific'
                    model_config['model_name'] = model_name
                    expanded_configs.append(model_config)
                    print(f"OK ({len(selected_wavelengths)} wavelengths)")
                except Exception as e:
                    print(f"FAILED ({e})")
                    # Fall back to original config for this model
                    model_config = config.copy()
                    model_config['model_name'] = model_name
                    expanded_configs.append(model_config)

        top_configs = expanded_configs
        print(f"\nExpanded to {len(top_configs)} model-specific configurations")

    # Print summary
    print(f"\n=== Top {len(top_configs)} Configurations ===")
    for i, config in enumerate(top_configs):
        window_str = f"w={config['window']}" if config['window'] else ""
        score_str = f"RMSE={config['score']:.4f}" if task_type == 'regression' else f"Acc={config['score']:.4f}"
        model_str = f" ({config.get('model_name', 'all')})" if config.get('model_name') else ""
        print(f"  {i+1}. {config['preprocessing']} {window_str}{model_str}: {score_str}, {config['n_wavelengths']} wavelengths")

    return top_configs
