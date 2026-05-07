"""
Exhaustive preprocessing optimization (V1).

This module implements two preprocessing-search strategies:

* ``exhaustive_search``: enumerates all 14 × 17 = 238 (preproc, window)
  combinations with parallel fitness evaluation.
* ``smart_exhaustive_search``: derivative-aware two-stage exhaustive
  search with multi-seed robust fitness; smaller candidate set (~80 cells)
  covering only legal (preproc, window) pairings.

Both share the same chromosome encoding (2 genes — preproc index, window
index) and fitness evaluator. The legacy genetic-algorithm path was
removed in 2026-05-06: with parallel exhaustive completing 238 fits in
seconds on a multi-core machine, GA evolution offered no coverage
advantage over enumeration. Helpers ``random_chromosome``,
``get_seed_chromosomes``, ``evaluate_fitness_robust`` are retained — they
are used by ``smart_exhaustive_search`` and by the planned phase-2
multi-seed rescore in ``phase2_rescore.py``.

Fitness Models
--------------
- 'pls': Uses PLS regression (default, always available)
- 'mlp': Uses Multi-Layer Perceptron neural network (for Neural/SVM models)
- 'lightgbm': Uses LightGBM (better for tree-based models, requires LightGBM)
- 'neuralboosted': Uses NeuralBoosted hybrid (falls back to PLS if absent)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Callable, Optional, Dict, Any, List
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score

# Import V1 preprocessing transformers
from .preprocess import SNV, SavgolDerivative

# Import LightGBM for fitness evaluation (required dependency)
from lightgbm import LGBMRegressor, LGBMClassifier


# =============================================================================
# CHROMOSOME ENCODING (SIMPLIFIED: 2 GENES ONLY)
# =============================================================================

# Gene 0: Preprocessing type (14 options)
# All SNV + derivative combinations for derivatives 1-4
PREPROC_TYPES = [
    'raw',           # 0
    'snv',           # 1
    'deriv1',        # 2
    'deriv2',        # 3
    'deriv3',        # 4
    'deriv4',        # 5
    'snv_deriv1',    # 6
    'snv_deriv2',    # 7
    'snv_deriv3',    # 8
    'snv_deriv4',    # 9
    'deriv1_snv',    # 10
    'deriv2_snv',    # 11
    'deriv3_snv',    # 12
    'deriv4_snv',    # 13
]

# Gene 1: S-G window sizes (odd values only, 17 options)
WINDOW_SIZES = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 51]

# Derivative-specific window ranges (smarter search - avoid nonsensical combinations)
# 1st derivative: small windows for sharp peaks
# Higher derivatives: larger windows for smoothing (SG needs more points)
DERIVATIVE_WINDOW_RANGES = {
    'deriv1': [5, 7, 9, 11, 13, 15, 17, 19, 21],
    'deriv2': [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    'deriv3': [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35],
    'deriv4': [15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 51],
}

# Map model names to proxy types for smart proxy selection
MODEL_TO_PROXY = {
    # Linear models -> PLS proxy
    'PLS': 'pls', 'PLS-DA': 'pls', 'Ridge': 'pls', 'Lasso': 'pls', 'ElasticNet': 'pls',
    # Tree models -> LightGBM proxy
    'RandomForest': 'lightgbm', 'LightGBM': 'lightgbm', 'XGBoost': 'lightgbm', 'CatBoost': 'lightgbm',
    # Neural/kernel models -> MLP proxy
    'MLP': 'mlp', 'SVM': 'mlp', 'SVR': 'mlp', 'SVC': 'mlp',
}

# Seeds for multi-seed robustness testing
ROBUSTNESS_SEEDS = [42, 123, 456, 789, 999]

# Variance penalty coefficient (higher = prefer more stable preprocessing)
VARIANCE_PENALTY = 0.1

# Total search space: 14 × 17 = 238 combinations
N_GENES = 2
TOTAL_COMBINATIONS = len(PREPROC_TYPES) * len(WINDOW_SIZES)  # 238


def random_chromosome(rng: np.random.RandomState) -> np.ndarray:
    """Generate a random chromosome with 2 genes."""
    return np.array([
        rng.randint(0, len(PREPROC_TYPES)),
        rng.randint(0, len(WINDOW_SIZES)),
    ], dtype=np.int32)


def get_seed_chromosomes() -> List[np.ndarray]:
    """
    Return proven good preprocessing configurations as seed chromosomes.

    These seeds ensure the GA population includes common, effective
    preprocessing methods used in spectroscopy.
    """
    seeds = []

    def make_seed(preproc: str, window: int = 17) -> np.ndarray:
        """Helper to create a seed chromosome from named parameters."""
        return np.array([
            PREPROC_TYPES.index(preproc),
            WINDOW_SIZES.index(window) if window in WINDOW_SIZES else 6,  # default idx 6 = 17
        ], dtype=np.int32)

    # Proven good configurations from literature and practice
    seeds.append(make_seed('raw'))                          # Baseline
    seeds.append(make_seed('snv'))                          # Standard scatter correction
    seeds.append(make_seed('deriv1', window=17))            # 1st derivative, common window
    seeds.append(make_seed('deriv2', window=17))            # 2nd derivative, common window
    seeds.append(make_seed('deriv3', window=21))            # 3rd derivative (needs larger window)
    seeds.append(make_seed('deriv4', window=25))            # 4th derivative (needs even larger window)
    seeds.append(make_seed('snv_deriv1', window=17))        # SNV then 1st deriv
    seeds.append(make_seed('snv_deriv2', window=17))        # SNV then 2nd deriv
    seeds.append(make_seed('snv_deriv3', window=21))        # SNV then 3rd deriv
    seeds.append(make_seed('snv_deriv4', window=25))        # SNV then 4th deriv
    seeds.append(make_seed('deriv1_snv', window=17))        # 1st deriv then SNV
    seeds.append(make_seed('deriv2_snv', window=17))        # 2nd deriv then SNV
    seeds.append(make_seed('deriv3_snv', window=21))        # 3rd deriv then SNV
    seeds.append(make_seed('deriv4_snv', window=25))        # 4th deriv then SNV

    return seeds


def chromosome_to_transform(genes: np.ndarray) -> Tuple[str, Optional[Callable]]:
    """
    Convert chromosome to (name, transform_func) tuple.

    The returned tuple is compatible with _build_preprocessing_configs() output
    in search.py, making integration seamless.

    Parameters
    ----------
    genes : np.ndarray
        Integer-encoded chromosome [preproc_type, window]

    Returns
    -------
    name : str
        Human-readable name for the preprocessing configuration
    transform_func : callable or None
        Function that takes X and returns preprocessed X, or None for 'raw'
    """
    preproc_idx = genes[0]
    window_idx = genes[1]

    preproc_type = PREPROC_TYPES[preproc_idx]
    window = WINDOW_SIZES[window_idx]

    # Build name
    if preproc_type in ['raw', 'snv']:
        name = preproc_type
    else:
        name = f"{preproc_type}_w{window}"

    # Return early for raw (no transform needed)
    if preproc_type == 'raw':
        return (name, None)

    # Build transform function
    def transform(X, pt=preproc_type, w=window):
        X_out = np.asarray(X, dtype=np.float64)

        if pt == 'snv':
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv1':
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
        elif pt == 'deriv2':
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
        elif pt == 'deriv3':
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
        elif pt == 'deriv4':
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
        elif pt == 'snv_deriv1':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
        elif pt == 'snv_deriv2':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
        elif pt == 'deriv1_snv':
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv2_snv':
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == 'snv_deriv3':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
        elif pt == 'snv_deriv4':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
        elif pt == 'deriv3_snv':
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv4_snv':
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)

        return X_out

    return (name, transform)


def get_config_description(genes: np.ndarray) -> str:
    """Get human-readable description of chromosome configuration."""
    preproc_type = PREPROC_TYPES[genes[0]]
    window = WINDOW_SIZES[genes[1]]

    if preproc_type in ['raw', 'snv']:
        return f"Preproc: {preproc_type}"
    else:
        return f"Preproc: {preproc_type}, Window: {window}"


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

def evaluate_fitness(
    genes: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    n_components: int = 10,
    task_type: str = 'regression',
    random_state: int = 42,
    fitness_model: str = 'pls',
    model_config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Evaluate fitness of a preprocessing configuration.

    Uses cross-validated RMSECV (regression) or accuracy (classification) as fitness.

    Parameters
    ----------
    genes : np.ndarray
        Chromosome encoding preprocessing configuration [preproc_type, window]
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cv_folds : int
        Number of CV folds
    n_components : int
        Max PLS components to test (used for PLS fitness model)
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random state for CV splitting
    fitness_model : str
        Model to use for fitness evaluation: 'pls', 'lightgbm', 'mlp', or 'neuralboosted'
        Ignored if model_config is provided.
    model_config : dict, optional
        If provided, uses actual model for fitness evaluation.
        Dict with keys: 'name' (str), 'params' (dict of hyperparameters)

    Returns
    -------
    fitness : float
        For regression: negative RMSECV (higher = better)
        For classification: accuracy (higher = better)
    """
    try:
        # Get transform function
        name, transform_func = chromosome_to_transform(genes)

        # Apply preprocessing
        if transform_func is not None:
            X_preproc = transform_func(X)
        else:
            X_preproc = X

        # Check for invalid values
        if not np.isfinite(X_preproc).all():
            return -np.inf

        # Check for zero variance columns
        if np.any(np.std(X_preproc, axis=0) < 1e-10):
            return -np.inf

        # Determine n_components
        n_samples, n_features = X_preproc.shape
        n_comp = min(n_components, n_features // 2, n_samples // 2)
        n_comp = max(1, n_comp)

        # Set up CV based on task type
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # If model_config provided, try actual model for fitness (with fallback)
        if model_config is not None:
            actual_result = _evaluate_with_actual_model(
                X_preproc, y, cv, task_type,
                model_config['name'], model_config.get('params', {}), random_state
            )
            # If actual model succeeded, return its result
            if actual_result > -np.inf:
                return actual_result
            # Fall through to proxy models if actual model failed

        # Use proxy fitness model (or as fallback if actual model failed)
        if fitness_model == 'lightgbm':
            return _evaluate_lightgbm(X_preproc, y, cv, task_type, random_state)
        elif fitness_model == 'mlp':
            return _evaluate_mlp(X_preproc, y, cv, task_type, random_state)
        elif fitness_model == 'neuralboosted':
            return _evaluate_neuralboosted(X_preproc, y, cv, n_comp, task_type, random_state)
        else:
            # Default: PLS
            return _evaluate_pls(X_preproc, y, cv, n_comp, task_type)

    except Exception:
        # Any error = very poor fitness
        return -np.inf


def _evaluate_with_actual_model(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    task_type: str,
    model_name: str,
    model_params: Dict[str, Any],
    random_state: int
) -> float:
    """
    Evaluate fitness using actual model with user hyperparameters.

    This ensures preprocessing is optimized for the ACTUAL model the user wants
    to test, not a proxy model with hardcoded hyperparameters.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data
    y : np.ndarray
        Target values
    cv : CV splitter
        Cross-validation strategy
    task_type : str
        'regression' or 'classification'
    model_name : str
        Name of model (e.g., 'PLS', 'LightGBM', 'XGBoost', 'PLS-DA')
    model_params : dict
        User hyperparameters for the model
    random_state : int
        Random state for reproducibility

    Returns
    -------
    fitness : float
        Negative RMSECV (regression) or accuracy (classification)
    """
    from .models import get_model
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        # Prepare y for classification
        if task_type == 'classification':
            y_class = np.asarray(y)
            if not pd.api.types.is_numeric_dtype(y_class.dtype):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                if y_class.dtype == object:
                    y_class = y_class.astype(str)
                y_class = le.fit_transform(y_class)
            else:
                y_class = y_class.astype(int)
            y_for_cv = y_class
        else:
            y_for_cv = y

        # Get n_components from params if available (for PLS-based models)
        n_components = model_params.get('n_components', 10) if model_params else 10

        # Get the model instance (get_model doesn't accept random_state)
        model = get_model(model_name, task_type=task_type, n_components=n_components)

        # Apply user hyperparameters (excluding lr_ prefixed params for PLS-DA)
        if model_params:
            # Filter out lr_ params (LogisticRegression params for PLS-DA)
            base_params = {k: v for k, v in model_params.items()
                          if not k.startswith('lr_')}
            if base_params:
                try:
                    model.set_params(**base_params)
                except Exception:
                    pass  # Ignore params that don't apply

        # For PLS-DA classification, build Pipeline with PLSTransformer + LogisticRegression
        # This matches how PLS-DA is built during search (search.py:3598-3621)
        if task_type == 'classification' and model_name.upper() in ['PLS-DA', 'PLS']:
            from sklearn.linear_model import LogisticRegression

            # Extract LogisticRegression parameters from config (prefixed with lr_)
            lr_C = model_params.get('lr_C', 1.0) if model_params else 1.0
            lr_solver = model_params.get('lr_solver', 'lbfgs') if model_params else 'lbfgs'
            lr_max_iter = model_params.get('lr_max_iter', 1000) if model_params else 1000

            model = Pipeline([
                ('pls', model),
                ('scaler', StandardScaler()),  # Scale PLS scores for LogisticRegression
                ('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter,
                                          random_state=random_state))
            ])

        # Cross-validated prediction
        y_pred = cross_val_predict(model, X, y_for_cv, cv=cv)

        # Calculate fitness
        if task_type == 'classification':
            # Ensure y_pred is integer class labels
            y_pred_class = np.asarray(y_pred)
            if y_pred_class.dtype != int:
                y_pred_class = y_pred_class.astype(int)

            return accuracy_score(y_class, y_pred_class)
        else:
            rmsecv = np.sqrt(mean_squared_error(y, y_pred))
            return -rmsecv

    except Exception as e:
        # Model evaluation failed - return very poor fitness
        return -np.inf


def _evaluate_pls(X, y, cv, n_comp, task_type):
    """Evaluate fitness using PLS."""
    pls = PLSRegression(n_components=n_comp, scale=False)
    y_pred = cross_val_predict(pls, X, y, cv=cv)

    if task_type == 'classification':
        y_class = np.asarray(y)
        if not pd.api.types.is_numeric_dtype(y_class.dtype):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            if y_class.dtype == object:
                y_class = y_class.astype(str)
            y_class = le.fit_transform(y_class)
        y_pred_class = (y_pred > np.median(y_pred)).astype(int).ravel()
        return accuracy_score(y_class, y_pred_class)
    else:
        rmsecv = np.sqrt(mean_squared_error(y, y_pred))
        return -rmsecv


def _evaluate_lightgbm(X, y, cv, task_type, random_state):
    """Evaluate fitness using LightGBM."""
    if task_type == 'classification':
        y_class = np.asarray(y)
        if not pd.api.types.is_numeric_dtype(y_class.dtype):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            if y_class.dtype == object:
                y_class = y_class.astype(str)
            y_class = le.fit_transform(y_class)
        else:
            y_class = y_class.astype(int)

        model = LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=random_state, verbosity=-1, force_col_wise=True
        )
        y_pred = cross_val_predict(model, X, y_class, cv=cv)
        return accuracy_score(y_class, y_pred)
    else:
        model = LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=random_state, verbosity=-1, force_col_wise=True
        )
        y_pred = cross_val_predict(model, X, y, cv=cv)
        rmsecv = np.sqrt(mean_squared_error(y, y_pred))
        return -rmsecv


def _evaluate_mlp(X, y, cv, task_type, random_state):
    """Evaluate fitness using MLP."""
    from sklearn.neural_network import MLPRegressor, MLPClassifier

    if task_type == 'classification':
        y_class = np.asarray(y)
        if not pd.api.types.is_numeric_dtype(y_class.dtype):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            if y_class.dtype == object:
                y_class = y_class.astype(str)
            y_class = le.fit_transform(y_class)
        else:
            y_class = y_class.astype(int)

        model = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500,
            random_state=random_state, early_stopping=True, validation_fraction=0.1
        )
        y_pred = cross_val_predict(model, X, y_class, cv=cv)
        return accuracy_score(y_class, y_pred)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500,
            random_state=random_state, early_stopping=True, validation_fraction=0.1
        )
        y_pred = cross_val_predict(model, X, y, cv=cv)
        rmsecv = np.sqrt(mean_squared_error(y, y_pred))
        return -rmsecv


def _evaluate_neuralboosted(X, y, cv, n_comp, task_type, random_state):
    """Evaluate fitness using NeuralBoosted (falls back to PLS if unavailable)."""
    try:
        from .neural_boosted import NeuralBoostedRegressor, NeuralBoostedClassifier

        if task_type == 'classification':
            y_class = np.asarray(y)
            if not pd.api.types.is_numeric_dtype(y_class.dtype):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                if y_class.dtype == object:
                    y_class = y_class.astype(str)
                y_class = le.fit_transform(y_class)
            else:
                y_class = y_class.astype(int)

            model = NeuralBoostedClassifier(random_state=random_state)
            y_pred = cross_val_predict(model, X, y_class, cv=cv)
            return accuracy_score(y_class, y_pred)
        else:
            model = NeuralBoostedRegressor(random_state=random_state)
            y_pred = cross_val_predict(model, X, y, cv=cv)
            rmsecv = np.sqrt(mean_squared_error(y, y_pred))
            return -rmsecv
    except ImportError:
        # Fall back to PLS
        return _evaluate_pls(X, y, cv, n_comp, task_type)


# =============================================================================
# SMART PROXY SELECTION
# =============================================================================

def get_proxy_for_model(model_name: Optional[str]) -> str:
    """
    Auto-select appropriate proxy model based on target model type.

    Linear models (PLS, Ridge, etc.) -> PLS proxy
    Tree models (RF, LightGBM, etc.) -> LightGBM proxy
    Neural/kernel models (MLP, SVM) -> MLP proxy

    Parameters
    ----------
    model_name : str or None
        Name of target model, or None for default

    Returns
    -------
    proxy : str
        One of 'pls', 'lightgbm', 'mlp'
    """
    if model_name is None:
        return 'lightgbm'  # Default to LightGBM (fast, general purpose)

    return MODEL_TO_PROXY.get(model_name, 'lightgbm')


def get_smart_window_range(preproc_type: str) -> List[int]:
    """
    Get appropriate window sizes for a preprocessing type.

    Higher derivatives need larger windows for smoothing.
    Non-derivative preprocessing returns all window sizes (not used anyway).

    Parameters
    ----------
    preproc_type : str
        Preprocessing type name

    Returns
    -------
    windows : list of int
        Appropriate window sizes for this preprocessing
    """
    # Extract derivative order from preprocessing name
    for deriv_order in ['deriv4', 'deriv3', 'deriv2', 'deriv1']:
        if deriv_order in preproc_type:
            return DERIVATIVE_WINDOW_RANGES[deriv_order]

    # Non-derivative preprocessing - window doesn't matter but return default
    return [17]  # Single default, won't be used


def get_smart_combinations() -> List[Tuple[int, int]]:
    """
    Generate smart preprocessing + window combinations.

    Uses derivative-specific window ranges to avoid nonsensical combinations
    like 4th derivative with window=5.

    Returns
    -------
    combinations : list of (preproc_idx, window_idx) tuples
    """
    combinations = []

    for p_idx, preproc_type in enumerate(PREPROC_TYPES):
        if preproc_type in ['raw', 'snv']:
            # Non-derivative: single entry with default window
            combinations.append((p_idx, WINDOW_SIZES.index(17)))
        else:
            # Derivative: use smart window range
            smart_windows = get_smart_window_range(preproc_type)
            for window in smart_windows:
                if window in WINDOW_SIZES:
                    w_idx = WINDOW_SIZES.index(window)
                    combinations.append((p_idx, w_idx))

    return combinations


# =============================================================================
# ROBUST FITNESS EVALUATION (Multi-seed + Variance Penalty)
# =============================================================================

def evaluate_fitness_robust(
    genes: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    n_components: int = 10,
    task_type: str = 'regression',
    fitness_model: str = 'pls',
    model_config: Optional[Dict[str, Any]] = None,
    n_seeds: int = 5,
    variance_penalty: float = VARIANCE_PENALTY
) -> Tuple[float, float, float]:
    """
    Evaluate fitness with multi-seed robustness and variance penalty.

    Runs evaluation across multiple random seeds to detect unstable
    preprocessing configurations. Penalizes high variance.

    Parameters
    ----------
    genes : np.ndarray
        Chromosome encoding preprocessing configuration
    X : np.ndarray
        Raw spectral data
    y : np.ndarray
        Target values
    cv_folds : int
        Number of CV folds
    n_components : int
        Max PLS components
    task_type : str
        'regression' or 'classification'
    fitness_model : str
        Proxy model to use
    model_config : dict, optional
        Actual model config if provided
    n_seeds : int
        Number of seeds to test (default 5)
    variance_penalty : float
        Penalty coefficient for variance (default 0.1)

    Returns
    -------
    robust_fitness : float
        Variance-penalized fitness (mean - penalty * std)
    mean_fitness : float
        Mean fitness across seeds
    std_fitness : float
        Standard deviation across seeds
    """
    seeds = ROBUSTNESS_SEEDS[:n_seeds]
    fitness_scores = []

    for seed in seeds:
        score = evaluate_fitness(
            genes, X, y,
            cv_folds=cv_folds,
            n_components=n_components,
            task_type=task_type,
            random_state=seed,
            fitness_model=fitness_model,
            model_config=model_config
        )
        if score > -np.inf:
            fitness_scores.append(score)

    if len(fitness_scores) == 0:
        return -np.inf, -np.inf, np.inf

    mean_fitness = np.mean(fitness_scores)
    std_fitness = np.std(fitness_scores) if len(fitness_scores) > 1 else 0.0

    # Penalize high variance (unstable preprocessing)
    robust_fitness = mean_fitness - variance_penalty * std_fitness

    return robust_fitness, mean_fitness, std_fitness


# =============================================================================
# DIVERSITY SELECTION FOR EXHAUSTIVE SEARCH
# =============================================================================

def select_diverse_exhaustive_configs(
    all_genes: List[np.ndarray],
    all_fitness: List[float],
    n_top: int = 5,
    exclude_raw_from_diversity: bool = True
) -> List[int]:
    """
    Select top-N diverse configurations ensuring variety across preprocessing types.

    This prevents the top-N from being all similar configs (e.g., all 2nd derivative
    with similar windows). Instead, we get the best config from each preprocessing
    type first, then fill remaining slots with best overall.

    Parameters
    ----------
    all_genes : list of np.ndarray
        All chromosomes evaluated
    all_fitness : list of float
        Fitness scores for each chromosome
    n_top : int
        Number of configurations to return
    exclude_raw_from_diversity : bool
        If True, "raw" preprocessing is NOT given a diversity slot.
        It will only be included if it scores well enough to be in top-N
        without the diversity boost. Default True because raw rarely
        produces the best models.

    Returns
    -------
    selected_indices : list of int
        Indices of selected configurations in original order
    """
    # Create list of (index, genes, fitness, preproc_type)
    configs = []
    for i, (genes, fitness) in enumerate(zip(all_genes, all_fitness)):
        if fitness > -np.inf:  # Skip invalid configs
            preproc_type = PREPROC_TYPES[genes[0]]
            configs.append({
                'index': i,
                'genes': genes,
                'fitness': fitness,
                'preproc_type': preproc_type
            })

    if len(configs) <= n_top:
        return [c['index'] for c in configs]

    # Sort by fitness (descending - higher is better)
    sorted_configs = sorted(configs, key=lambda c: c['fitness'], reverse=True)

    # Select ensuring diversity across preprocessing types
    selected = []
    selected_preprocs = set()

    # First pass: select best from each preprocessing TYPE (not window variation)
    # This ensures we get deriv1, deriv2, snv_deriv1, etc. rather than just
    # multiple window sizes of the same preprocessing
    for config in sorted_configs:
        preproc_type = config['preproc_type']

        # Skip raw in diversity selection if configured (raw rarely produces best models)
        if exclude_raw_from_diversity and preproc_type == 'raw':
            continue

        if preproc_type not in selected_preprocs:
            selected.append(config)
            selected_preprocs.add(preproc_type)
            if len(selected) >= n_top:
                break

    # Second pass: fill remaining slots with best overall (including raw)
    if len(selected) < n_top:
        for config in sorted_configs:
            if config not in selected:
                selected.append(config)
                if len(selected) >= n_top:
                    break

    # Return indices sorted by fitness (best first)
    selected = sorted(selected, key=lambda c: c['fitness'], reverse=True)
    return [c['index'] for c in selected]


# =============================================================================
# EXHAUSTIVE SEARCH (FOR SMALL SEARCH SPACE)
# =============================================================================

def exhaustive_search(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    n_components: int = 10,
    task_type: str = 'regression',
    random_state: int = 42,
    fitness_model: str = 'pls',
    n_jobs: int = 1,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    top_n: int = 5,
    model_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Exhaustively search all 170 preprocessing combinations.

    With only 170 combinations (10 preprocessing types × 17 window sizes),
    exhaustive search is feasible and guarantees finding the optimal solution.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cv_folds : int
        Number of CV folds for fitness evaluation
    n_components : int
        Max PLS components for evaluation
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random seed for reproducibility
    fitness_model : str
        Model to use for fitness evaluation
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Progress callback function
    top_n : int
        Number of top preprocessing configs to return (default=5)
    model_config : dict, optional
        Dict with 'name' and 'params' for actual model fitness evaluation
        If None, uses proxy fitness_model instead

    Returns
    -------
    result : dict
        Same format as optimize_preprocessing()
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    if verbose >= 1:
        print(f"Exhaustive Preprocessing Search")
        print(f"  Data: {n_samples} samples, {n_features} features")
        print(f"  Task: {task_type}")
        print(f"  Fitness model: {fitness_model.upper()}")
        print(f"  Total combinations: {TOTAL_COMBINATIONS}")

    # Generate all possible chromosomes
    all_genes = [
        np.array([p, w], dtype=np.int32)
        for p in range(len(PREPROC_TYPES))
        for w in range(len(WINDOW_SIZES))
    ]

    # Evaluate all combinations
    if n_jobs != 1:
        # Parallel evaluation
        try:
            from joblib import Parallel, delayed

            if verbose >= 1:
                print(f"  Parallel evaluation with n_jobs={n_jobs}")

            # Use 'threading' in frozen apps (avoids PyInstaller process spawn issues)
            # Use 'loky' in dev mode (faster multiprocessing)
            from spectral_predict.search import _frozen_needs_threading_fallback
            backend = 'threading' if _frozen_needs_threading_fallback() else 'loky'
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(evaluate_fitness)(
                    genes, X, y, cv_folds, n_components, task_type, random_state, fitness_model, model_config
                )
                for genes in all_genes
            )
        except ImportError:
            # Fall back to sequential
            if verbose >= 1:
                print("  joblib not available, using sequential evaluation")
            results = []
            for i, genes in enumerate(all_genes):
                fitness = evaluate_fitness(
                    genes, X, y, cv_folds, n_components, task_type, random_state, fitness_model, model_config
                )
                results.append(fitness)
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback({
                        'algorithm': 'exhaustive_preprocessing',
                        'current': i + 1,
                        'total': len(all_genes),
                        'best_fitness': max(results),
                        'message': f"Tested {i+1}/{len(all_genes)} combinations"
                    })
    else:
        # Sequential evaluation
        results = []
        for i, genes in enumerate(all_genes):
            fitness = evaluate_fitness(
                genes, X, y, cv_folds, n_components, task_type, random_state, fitness_model, model_config
            )
            results.append(fitness)

            if verbose >= 2 and (i + 1) % 50 == 0:
                print(f"  Tested {i+1}/{len(all_genes)} combinations")

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback({
                    'algorithm': 'exhaustive_preprocessing',
                    'current': i + 1,
                    'total': len(all_genes),
                    'best_fitness': max(results),
                    'message': f"Tested {i+1}/{len(all_genes)} combinations"
                })

    # Use diversity selection to ensure variety across preprocessing types
    # This prevents selecting 5 similar configs (e.g., all deriv2 with different windows)
    diverse_indices = select_diverse_exhaustive_configs(
        all_genes, results, n_top=top_n,
        exclude_raw_from_diversity=True  # Raw rarely produces best models
    )

    if verbose >= 1:
        print(f"  Diversity selection: ensuring variety across preprocessing types")

    # Build top-N configs list using diversity-selected indices
    configs = []
    for idx in diverse_indices:
        genes = all_genes[idx]
        fitness = results[idx]
        name, transform = chromosome_to_transform(genes)
        config_desc = get_config_description(genes)

        # Extract deriv and window from genes for search.py integration
        preproc_type = PREPROC_TYPES[genes[0]]
        window = WINDOW_SIZES[genes[1]]

        # Determine derivative order from preprocessing type name
        deriv_order = None
        if 'deriv1' in preproc_type:
            deriv_order = 1
        elif 'deriv2' in preproc_type:
            deriv_order = 2
        elif 'deriv3' in preproc_type:
            deriv_order = 3
        elif 'deriv4' in preproc_type:
            deriv_order = 4

        # Polyorder for Savitzky-Golay = deriv_order - 1 (minimum 0)
        polyorder = max(deriv_order - 1, 0) if deriv_order else None

        # Convert fitness to RMSECV/error score format
        if task_type == 'classification':
            score = 1.0 - fitness  # Convert accuracy to error rate
        else:
            score = -fitness  # Convert negative RMSECV to positive RMSECV

        configs.append({
            'genes': genes,
            'name': name,
            'transform': transform,
            'rmsecv': score,
            'config': config_desc,
            'fitness': fitness,
            'deriv': deriv_order,
            'window': window,
            'polyorder': polyorder
        })

    # Best is first in list
    best_genes = configs[0]['genes']
    best_name = configs[0]['name']
    best_transform = configs[0]['transform']
    best_score = configs[0]['rmsecv']
    best_config = configs[0]['config']
    best_fitness = configs[0]['fitness']

    if verbose >= 1:
        print(f"\nExhaustive search complete!")
        if task_type == 'classification':
            print(f"  Best Accuracy: {best_fitness:.4f}")
        else:
            print(f"  Best RMSECV: {best_score:.4f}")
        print(f"  Best config: {best_config}")
        print(f"  Returning top {len(configs)} diverse configs")

    return {
        'configs': configs,  # NEW: List of top-N configs
        'best_genes': best_genes,  # Backward compatibility
        'best_name': best_name,
        'best_transform': best_transform,
        'best_rmsecv': best_score,
        'best_config': best_config,
        'history': [{'combination': i, 'fitness': f} for i, f in enumerate(results)],
        'task_type': task_type,
        'method': 'exhaustive'
    }


# =============================================================================
# SMART TWO-STAGE EXHAUSTIVE SEARCH
# =============================================================================

def smart_exhaustive_search(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    n_components: int = 10,
    task_type: str = 'regression',
    fitness_model: str = 'auto',
    target_model: Optional[str] = None,
    n_jobs: int = 1,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    top_n: int = 5,
    model_config: Optional[Dict[str, Any]] = None,
    stage1_top_k: int = 20,
    robust_validation: bool = True
) -> Dict[str, Any]:
    """
    Smart two-stage exhaustive search with derivative-specific windows.

    Stage 1: Fast screening
      - Uses derivative-specific window ranges (avoids nonsensical combinations)
      - 3-fold CV for speed
      - Selects top K candidates

    Stage 2: Thorough validation
      - 5-fold CV on top candidates
      - Multi-seed robustness testing (optional)
      - Variance-penalized ranking

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cv_folds : int
        Number of CV folds for stage 2 (stage 1 uses 3)
    n_components : int
        Max PLS components for evaluation
    task_type : str
        'regression' or 'classification'
    fitness_model : str
        'auto' to auto-select based on target_model, or 'pls', 'lightgbm', 'mlp'
    target_model : str, optional
        Target model name for smart proxy selection (e.g., 'PLS', 'LightGBM')
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Progress callback function
    top_n : int
        Number of final configs to return
    model_config : dict, optional
        Dict with 'name' and 'params' for actual model fitness
    stage1_top_k : int
        Number of candidates to advance from stage 1 (default 20)
    robust_validation : bool
        Whether to use multi-seed robustness in stage 2 (default True)

    Returns
    -------
    result : dict
        Same format as exhaustive_search(), plus 'stability' for each config
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    # Auto-select proxy based on target model
    if fitness_model == 'auto':
        if model_config and 'name' in model_config:
            fitness_model = get_proxy_for_model(model_config['name'])
        elif target_model:
            fitness_model = get_proxy_for_model(target_model)
        else:
            fitness_model = 'lightgbm'

    # Get smart combinations (derivative-specific windows)
    smart_combos = get_smart_combinations()
    n_smart_combos = len(smart_combos)

    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"SMART TWO-STAGE PREPROCESSING SEARCH")
        print(f"{'='*60}")
        print(f"  Data: {n_samples} samples, {n_features} features")
        print(f"  Task: {task_type}")
        print(f"  Proxy model: {fitness_model.upper()}")
        print(f"  Smart combinations: {n_smart_combos} (vs {TOTAL_COMBINATIONS} exhaustive)")
        print(f"  Stage 1: Fast screening (3-fold CV)")
        print(f"  Stage 2: Thorough validation ({cv_folds}-fold CV" +
              (", multi-seed)" if robust_validation else ")"))

    # =========================================================================
    # STAGE 1: Fast Screening
    # =========================================================================
    if verbose >= 1:
        print(f"\n--- Stage 1: Fast Screening ({n_smart_combos} combinations) ---")

    stage1_genes = [
        np.array([p_idx, w_idx], dtype=np.int32)
        for p_idx, w_idx in smart_combos
    ]

    stage1_cv_folds = 3  # Faster than full CV

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            if verbose >= 1:
                print(f"  Parallel evaluation with n_jobs={n_jobs}")

            # Use 'threading' in frozen apps (avoids PyInstaller process spawn issues)
            # Use 'loky' in dev mode (faster multiprocessing)
            from spectral_predict.search import _frozen_needs_threading_fallback
            backend = 'threading' if _frozen_needs_threading_fallback() else 'loky'
            stage1_results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(evaluate_fitness)(
                    genes, X, y, stage1_cv_folds, n_components, task_type, 42, fitness_model, model_config
                )
                for genes in stage1_genes
            )
        except ImportError:
            stage1_results = [
                evaluate_fitness(genes, X, y, stage1_cv_folds, n_components, task_type, 42, fitness_model, model_config)
                for genes in stage1_genes
            ]
    else:
        stage1_results = []
        for i, genes in enumerate(stage1_genes):
            fitness = evaluate_fitness(
                genes, X, y, stage1_cv_folds, n_components, task_type, 42, fitness_model, model_config
            )
            stage1_results.append(fitness)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback({
                    'algorithm': 'smart_preprocessing',
                    'stage': 1,
                    'current': i + 1,
                    'total': n_smart_combos,
                    'best_fitness': max(stage1_results),
                    'message': f"Stage 1: {i+1}/{n_smart_combos}"
                })

            if verbose >= 2 and (i + 1) % 20 == 0:
                print(f"  Stage 1: {i+1}/{n_smart_combos} tested")

    # Select top K candidates for stage 2
    stage1_array = np.array(stage1_results)
    top_k_indices = np.argsort(stage1_array)[-stage1_top_k:][::-1]

    if verbose >= 1:
        print(f"  Stage 1 complete: Selected top {len(top_k_indices)} candidates")
        best_stage1 = stage1_array[top_k_indices[0]]
        if task_type == 'classification':
            print(f"  Best stage 1 accuracy: {best_stage1:.4f}")
        else:
            print(f"  Best stage 1 RMSECV: {-best_stage1:.4f}")

    # =========================================================================
    # STAGE 2: Thorough Validation
    # =========================================================================
    if verbose >= 1:
        print(f"\n--- Stage 2: Thorough Validation ({len(top_k_indices)} candidates) ---")

    stage2_results = []

    for i, idx in enumerate(top_k_indices):
        genes = stage1_genes[idx]

        if robust_validation:
            # Multi-seed robustness evaluation
            robust_fitness, mean_fitness, std_fitness = evaluate_fitness_robust(
                genes, X, y,
                cv_folds=cv_folds,
                n_components=n_components,
                task_type=task_type,
                fitness_model=fitness_model,
                model_config=model_config,
                n_seeds=5,
                variance_penalty=VARIANCE_PENALTY
            )
            stage2_results.append({
                'genes': genes,
                'robust_fitness': robust_fitness,
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness
            })
        else:
            # Single evaluation with full CV
            fitness = evaluate_fitness(
                genes, X, y, cv_folds, n_components, task_type, 42, fitness_model, model_config
            )
            stage2_results.append({
                'genes': genes,
                'robust_fitness': fitness,
                'mean_fitness': fitness,
                'std_fitness': 0.0
            })

        if progress_callback:
            progress_callback({
                'algorithm': 'smart_preprocessing',
                'stage': 2,
                'current': i + 1,
                'total': len(top_k_indices),
                'message': f"Stage 2: {i+1}/{len(top_k_indices)}"
            })

        if verbose >= 2:
            preproc = PREPROC_TYPES[genes[0]]
            window = WINDOW_SIZES[genes[1]]
            r = stage2_results[-1]
            if task_type == 'classification':
                print(f"  {preproc} w={window}: Acc={r['mean_fitness']:.4f} ± {r['std_fitness']:.4f}")
            else:
                print(f"  {preproc} w={window}: RMSECV={-r['mean_fitness']:.4f} ± {r['std_fitness']:.4f}")

    # Sort by robust fitness
    stage2_results.sort(key=lambda x: x['robust_fitness'], reverse=True)

    # Build output configs
    configs = []
    for i, result in enumerate(stage2_results[:top_n]):
        genes = result['genes']
        name, transform = chromosome_to_transform(genes)
        config_desc = get_config_description(genes)

        preproc_type = PREPROC_TYPES[genes[0]]
        window = WINDOW_SIZES[genes[1]]

        # Determine derivative order
        deriv_order = None
        for d in [4, 3, 2, 1]:
            if f'deriv{d}' in preproc_type:
                deriv_order = d
                break

        polyorder = max(deriv_order + 1, 2) if deriv_order else None

        # Convert fitness to score format
        if task_type == 'classification':
            score = 1.0 - result['mean_fitness']
        else:
            score = -result['mean_fitness']

        configs.append({
            'genes': genes,
            'name': name,
            'transform': transform,
            'rmsecv': score,
            'config': config_desc,
            'fitness': result['robust_fitness'],
            'mean_fitness': result['mean_fitness'],
            'std_fitness': result['std_fitness'],
            'stability': 1.0 / (1.0 + result['std_fitness']),  # Higher = more stable
            'deriv': deriv_order,
            'window': window,
            'polyorder': polyorder
        })

    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"SMART SEARCH COMPLETE")
        print(f"{'='*60}")
        best = configs[0]
        if task_type == 'classification':
            print(f"  Best: {best['config']}")
            print(f"  Accuracy: {best['mean_fitness']:.4f} ± {best['std_fitness']:.4f}")
        else:
            print(f"  Best: {best['config']}")
            print(f"  RMSECV: {best['rmsecv']:.4f} ± {best['std_fitness']:.4f}")
        print(f"  Stability: {best['stability']:.3f}")
        print(f"  Returning top {len(configs)} configs")

    return {
        'configs': configs,
        'best_genes': configs[0]['genes'],
        'best_name': configs[0]['name'],
        'best_transform': configs[0]['transform'],
        'best_rmsecv': configs[0]['rmsecv'],
        'best_config': configs[0]['config'],
        'history': stage1_results,
        'task_type': task_type,
        'method': 'smart_exhaustive',
        'stage1_tested': n_smart_combos,
        'stage2_tested': len(top_k_indices)
    }


# =============================================================================
# MAIN GA FUNCTION
# =============================================================================

def optimize_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'exhaustive',
    cv_folds: int = 5,
    n_components: int = 10,
    task_type: str = 'regression',
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    fitness_model: str = 'pls',
    n_jobs: int = -1,
    top_n: int = 5,
    model_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize spectral preprocessing via exhaustive enumeration.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    method : str
        Optimization method:

        * ``"exhaustive"`` (default): tests all 238 (preproc, window) cells.
        * ``"smart"``: derivative-aware two-stage exhaustive with multi-seed
          robust fitness on the survivors. Returns the same result shape as
          ``"exhaustive"``.

        The legacy ``"ga"`` (genetic algorithm) mode was removed in 2026-05-06.
        With ``n_jobs=-1`` parallelism completing 238 cells in seconds, GA
        evolution had no coverage or speed advantage over enumeration.
    cv_folds : int
        Number of CV folds for fitness evaluation
    n_components : int
        Max PLS components for evaluation
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random seed for reproducibility
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Function called with progress dict
    fitness_model : str
        Proxy model for fitness evaluation: 'pls', 'lightgbm', 'mlp',
        or 'neuralboosted'. Ignored when ``model_config`` is provided.
    n_jobs : int
        Number of parallel jobs for exhaustive search (default ``-1``,
        i.e. all cores)
    top_n : int
        Number of top preprocessing configs to return (default=5)
    model_config : dict, optional
        Dict with 'name' and 'params' for actual model fitness evaluation.
        When provided, exhaustive uses the actual model with ``params`` instead
        of the ``fitness_model`` proxy.

    Returns
    -------
    result : dict
        Dictionary containing:

        * ``"configs"``: list of top-N preprocessing configs (each a dict with
          ``genes``, ``name``, ``transform``, ``rmsecv``, ``config``, etc.)
        * ``"best_genes"``: np.ndarray — best chromosome (backward compat)
        * ``"best_name"``: str
        * ``"best_transform"``: callable
        * ``"best_rmsecv"``: float
        * ``"best_config"``: str — human-readable configuration
        * ``"method"``: str — ``"exhaustive"`` or ``"smart"``
    """
    if method == 'exhaustive':
        return exhaustive_search(
            X, y, cv_folds, n_components, task_type, random_state,
            fitness_model, n_jobs, verbose, progress_callback, top_n, model_config
        )

    if method == 'smart':
        target_model = model_config.get('name') if model_config else None
        return smart_exhaustive_search(
            X, y,
            cv_folds=cv_folds,
            n_components=n_components,
            task_type=task_type,
            fitness_model='auto',
            target_model=target_model,
            n_jobs=n_jobs,
            verbose=verbose,
            progress_callback=progress_callback,
            top_n=top_n,
            model_config=model_config,
            stage1_top_k=20,
            robust_validation=True,
        )

    raise ValueError(
        f"method must be 'exhaustive' or 'smart', got {method!r}. "
        "Legacy 'ga' mode was removed in 2026-05-06."
    )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_optimized_preproc_config(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool = True,
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[str, Optional[Callable]]:
    """
    Get optimized preprocessing as (name, transform_func) tuple.

    This is a convenience function that returns the result in the same
    format as _build_preprocessing_configs() for easy integration.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data
    y : np.ndarray
        Target values
    quick : bool
        If True, use quick settings (fewer generations)
    random_state : int
        Random seed
    verbose : int
        Verbosity level

    Returns
    -------
    name : str
        Preprocessing configuration name
    transform_func : callable or None
        Transform function, or None for 'raw'
    """
    result = optimize_preprocessing(
        X, y,
        method='exhaustive',
        cv_folds=3 if quick else 5,
        random_state=random_state,
        verbose=verbose,
    )

    return (result['best_name'], result['best_transform'])
