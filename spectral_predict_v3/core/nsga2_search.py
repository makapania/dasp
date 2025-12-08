"""
NSGA-II Multi-Objective Optimization for Spectral Predict v3.

This module implements NSGA-II (Non-dominated Sorting Genetic Algorithm II)
for Pareto optimization of multiple conflicting objectives in spectroscopy:

1. Minimize prediction error (RMSE for regression, 1-Accuracy for classification)
2. Minimize number of wavelengths (parsimony)
3. Minimize model complexity (latent variables, model type)

Returns a Pareto front of non-dominated solutions with knee point detection
for automatic "best compromise" selection.

References:
- Deb et al. (2002) "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
- pymoo library: https://pymoo.org/
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

# Optional pymoo import - graceful degradation if not available
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


def check_pymoo_available():
    """Check if pymoo is available and raise informative error if not."""
    if not PYMOO_AVAILABLE:
        raise ImportError(
            "pymoo is required for NSGA-II optimization. "
            "Install with: pip install pymoo>=0.6.0"
        )


# =============================================================================
# Preprocessing options (same encoding as ga_preprocessing.py)
# =============================================================================

PREPROC_TYPES = [
    'raw',           # 0
    'snv',           # 1
    'deriv1',        # 2
    'deriv2',        # 3
    'snv_deriv1',    # 4
    'snv_deriv2',    # 5
    'deriv1_snv',    # 6
    'deriv2_snv',    # 7
    'deriv3',        # 8
    'deriv4',        # 9
]

WINDOW_SIZES = list(range(5, 35, 2))  # 5, 7, 9, ..., 33

MODEL_TYPES = [
    'PLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest',
    'LightGBM', 'XGBoost', 'CatBoost', 'SVR', 'MLP', 'NeuralBoosted'
]  # All supported models for NSGA-II


def _get_preprocessing_transform(preproc_idx: int, window_idx: int):
    """
    Get preprocessing transformation function.

    Parameters
    ----------
    preproc_idx : int
        Index into PREPROC_TYPES
    window_idx : int
        Index into WINDOW_SIZES

    Returns
    -------
    transform : callable or None
        Transform function or None for raw
    """
    from .preprocess import SNV, SavgolDerivative

    preproc_type = PREPROC_TYPES[min(preproc_idx, len(PREPROC_TYPES) - 1)]
    window = WINDOW_SIZES[min(window_idx, len(WINDOW_SIZES) - 1)]

    if preproc_type == 'raw':
        return None
    elif preproc_type == 'snv':
        snv = SNV()
        return lambda X: snv.fit_transform(X)
    elif preproc_type.startswith('deriv'):
        # Extract derivative order
        if 'deriv1' in preproc_type:
            deriv_order = 1
        elif 'deriv2' in preproc_type:
            deriv_order = 2
        elif 'deriv3' in preproc_type:
            deriv_order = 3
        elif 'deriv4' in preproc_type:
            deriv_order = 4
        else:
            deriv_order = 1

        sg = SavgolDerivative(deriv=deriv_order, window=window)
        snv = SNV()

        if preproc_type.startswith('snv_'):
            # SNV then derivative
            def transform(X):
                X_snv = snv.fit_transform(X)
                return sg.fit_transform(X_snv)
            return transform
        elif preproc_type.endswith('_snv'):
            # Derivative then SNV
            def transform(X):
                X_deriv = sg.fit_transform(X)
                return snv.fit_transform(X_deriv)
            return transform
        else:
            # Just derivative
            return lambda X: sg.fit_transform(X)

    return None


# =============================================================================
# NSGA-II Problem Definition
# =============================================================================

class SpectralOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for spectral calibration.

    Decision variables (chromosome):
    - Gene 0: Preprocessing type (0-9)
    - Gene 1: S-G window size index (0-14)
    - Gene 2: Model type (0=PLS, 1=Ridge)
    - Gene 3: PLS components / Ridge alpha index (0-14)
    - Gene 4-N: Binary wavelength selection (0/1)

    Objectives (all minimized):
    1. Prediction error (RMSE or 1-Accuracy)
    2. Number of selected wavelengths (normalized)
    3. Model complexity score (normalized)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'regression',
        cv_folds: int = 5,
        min_wavelengths: int = 10,
        random_state: int = 42,
        cache_enabled: bool = True,
        models: Optional[List[str]] = None,
    ):
        """
        Initialize the optimization problem.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavelengths)
            Spectral data
        y : ndarray, shape (n_samples,)
            Target values
        task_type : str
            'regression' or 'classification'
        cv_folds : int
            Number of CV folds for fitness evaluation
        min_wavelengths : int
            Minimum number of wavelengths that must be selected
        random_state : int
            Random state for reproducibility
        cache_enabled : bool
            If True, cache fitness evaluations
        models : list of str, optional
            Model types to consider. If None, uses default MODEL_TYPES.
        """
        self.X = X
        self.y = y
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.min_wavelengths = min_wavelengths
        self.random_state = random_state
        self.n_wavelengths = X.shape[1]

        # Use user-specified models or defaults
        self.model_types = models if models is not None else MODEL_TYPES

        # Encode labels for classification
        self.label_encoder = None
        if task_type == 'classification':
            if y.dtype == object or not np.issubdtype(y.dtype, np.number):
                self.label_encoder = LabelEncoder()
                self.y = self.label_encoder.fit_transform(y)

        # Fitness cache
        self.cache_enabled = cache_enabled
        self._cache = {}
        self._eval_count = 0

        # Decision variables:
        # [preproc_type, window_idx, model_type, model_param, wl_0, wl_1, ..., wl_n]
        n_vars = 4 + self.n_wavelengths

        # Variable bounds
        xl = np.zeros(n_vars)
        xu = np.array([
            len(PREPROC_TYPES) - 1,  # preproc_type
            len(WINDOW_SIZES) - 1,   # window_idx
            len(self.model_types) - 1,    # model_type (use instance var)
            14,                       # model_param (LVs 1-15 or alpha index)
        ] + [1] * self.n_wavelengths)  # wavelength selection

        super().__init__(
            n_var=n_vars,
            n_obj=3,  # 3 objectives
            n_ieq_constr=1,  # Constraint: min wavelengths
            xl=xl,
            xu=xu,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate fitness for a population of solutions.

        Parameters
        ----------
        X : ndarray, shape (pop_size, n_vars)
            Population of solutions
        out : dict
            Output dictionary for objectives and constraints
        """
        pop_size = X.shape[0]
        F = np.zeros((pop_size, 3))  # 3 objectives
        G = np.zeros((pop_size, 1))  # 1 constraint

        for i in range(pop_size):
            chromosome = X[i].astype(int)

            # Check cache
            cache_key = tuple(chromosome)
            if self.cache_enabled and cache_key in self._cache:
                F[i], G[i] = self._cache[cache_key]
                continue

            # Decode chromosome
            preproc_idx = chromosome[0]
            window_idx = chromosome[1]
            model_idx = chromosome[2]
            model_param = chromosome[3]
            wavelength_mask = chromosome[4:].astype(bool)

            # Count selected wavelengths
            n_selected = np.sum(wavelength_mask)

            # Constraint: minimum wavelengths
            G[i, 0] = self.min_wavelengths - n_selected  # <= 0 means feasible

            # If too few wavelengths, use penalty
            if n_selected < self.min_wavelengths:
                F[i, 0] = 1e6  # Very high error
                F[i, 1] = 1.0  # Normalized wavelength count
                F[i, 2] = 1.0  # Normalized complexity

                if self.cache_enabled:
                    self._cache[cache_key] = (F[i].copy(), G[i].copy())
                continue

            # Objective 1: Prediction error
            error = self._compute_prediction_error(
                preproc_idx, window_idx, model_idx, model_param, wavelength_mask
            )
            F[i, 0] = error

            # Objective 2: Wavelength count (normalized to 0-1)
            F[i, 1] = n_selected / self.n_wavelengths

            # Objective 3: Model complexity (normalized to 0-1)
            complexity = self._compute_complexity(model_idx, model_param, preproc_idx)
            F[i, 2] = complexity

            self._eval_count += 1

            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = (F[i].copy(), G[i].copy())

        out["F"] = F
        out["G"] = G

    def _compute_prediction_error(
        self,
        preproc_idx: int,
        window_idx: int,
        model_idx: int,
        model_param: int,
        wavelength_mask: np.ndarray,
    ) -> float:
        """Compute CV prediction error for a configuration."""
        try:
            # Apply preprocessing
            transform = _get_preprocessing_transform(preproc_idx, window_idx)
            if transform is not None:
                X_proc = transform(self.X)
            else:
                X_proc = self.X.copy()

            # Select wavelengths
            X_subset = X_proc[:, wavelength_mask]

            # Check for degenerate cases
            if X_subset.shape[1] == 0:
                return 1e6
            if np.any(np.isnan(X_subset)) or np.any(np.isinf(X_subset)):
                return 1e6

            # Build model
            model_type = self.model_types[min(model_idx, len(self.model_types) - 1)]

            if model_type == 'PLS':
                n_components = min(model_param + 1, X_subset.shape[1], X_subset.shape[0] - 1)
                n_components = max(1, n_components)
                model = PLSRegression(n_components=n_components)
            elif model_type == 'Ridge':
                # Exponential alpha scale: 1e-4 to 1e3
                alpha = 10 ** (model_param / 3 - 2)  # param 0-14 -> alpha 0.01 to 1000
                model = Ridge(alpha=alpha)
            elif model_type == 'LightGBM':
                from .models import get_model, HAS_LIGHTGBM
                if not HAS_LIGHTGBM:
                    return 1e6
                # model_param controls n_estimators (50-200) and learning_rate
                n_estimators = 50 + model_param * 10  # 50-190
                learning_rate = 0.05 if model_param < 7 else 0.1
                # Task-aware default: 15 for classification, 31 for regression (V1)
                num_leaves = 15 if self.task_type == 'classification' else 31
                model = get_model('LightGBM', self.task_type, self.random_state,
                                  n_estimators=n_estimators, learning_rate=learning_rate,
                                  num_leaves=num_leaves, reg_lambda=0.1)
            elif model_type == 'XGBoost':
                from .models import get_model, HAS_XGBOOST
                if not HAS_XGBOOST:
                    return 1e6
                # model_param controls n_estimators, max_depth, and regularization
                n_estimators = 50 + model_param * 10  # 50-190
                max_depth = 3 + (model_param % 5)  # 3-7
                # Vary subsample/colsample based on model_param: lower params = less reg, higher = more reg
                subsample = 1.0 if model_param < 7 else 0.8
                colsample = 1.0 if model_param < 7 else 0.8
                model = get_model('XGBoost', self.task_type, self.random_state,
                                  n_estimators=n_estimators, learning_rate=0.1,
                                  max_depth=max_depth, reg_lambda=0.1,
                                  subsample=subsample, colsample_bytree=colsample)
            elif model_type == 'Lasso':
                from .models import get_model
                # alpha from 1e-3 to 100 (log scale)
                alpha = 10 ** (model_param / 3 - 3)  # param 0-14 -> alpha ~0.001 to 100
                model = get_model('Lasso', self.task_type, self.random_state, alpha=alpha)
            elif model_type == 'ElasticNet':
                from .models import get_model
                # alpha from 0.01 to 100, l1_ratio from 0.2 to 0.8
                alpha = 10 ** (model_param / 3 - 2)
                l1_ratio = 0.2 + (model_param % 5) * 0.15  # 0.2 to 0.8
                model = get_model('ElasticNet', self.task_type, self.random_state,
                                  alpha=alpha, l1_ratio=l1_ratio)
            elif model_type == 'RandomForest':
                from .models import get_model
                n_estimators = 50 + model_param * 15  # 50-260
                max_depth = None if model_param < 5 else 10 + model_param * 3  # None or 25-67
                model = get_model('RandomForest', self.task_type, self.random_state,
                                  n_estimators=n_estimators, max_depth=max_depth)
            elif model_type == 'CatBoost':
                from .models import get_model, HAS_CATBOOST
                if not HAS_CATBOOST:
                    return 1e6
                iterations = 50 + model_param * 15  # 50-260
                depth = 4 + (model_param % 5)  # 4-8
                learning_rate = 0.05 if model_param < 7 else 0.1
                model = get_model('CatBoost', self.task_type, self.random_state,
                                  iterations=iterations, depth=depth, learning_rate=learning_rate)
            elif model_type == 'SVR':
                from .models import get_model
                kernel = 'rbf' if model_param < 10 else 'linear'
                C = 10 ** (model_param / 3 - 1)  # 0.1 to ~1000
                model = get_model('SVR', self.task_type, self.random_state, kernel=kernel, C=C)
            elif model_type == 'MLP':
                from .models import get_model
                # Layer size based on model_param
                layer_size = 30 + model_param * 10  # 30-170
                n_layers = 1 if model_param < 7 else 2
                hidden_layer_sizes = (layer_size,) if n_layers == 1 else (layer_size, layer_size // 2)
                alpha = 10 ** (model_param / 5 - 4)  # 1e-4 to 0.01
                model = get_model('MLP', self.task_type, self.random_state,
                                  hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
            elif model_type == 'NeuralBoosted':
                from .models import get_model, HAS_NEURAL_BOOSTED
                if not HAS_NEURAL_BOOSTED:
                    return 1e6
                n_estimators = 30 + model_param * 10  # 30-170
                hidden_layer_size = 3 + (model_param % 5)  # 3-7
                learning_rate = 0.05 + (model_param / 14) * 0.2  # 0.05-0.25
                model = get_model('NeuralBoosted', self.task_type, self.random_state,
                                  n_estimators=n_estimators, hidden_layer_size=hidden_layer_size,
                                  learning_rate=learning_rate)
            else:
                return 1e6

            # Cross-validation
            if self.task_type == 'regression':
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    scores = cross_val_score(
                        model, X_subset, self.y, cv=cv, scoring='neg_mean_squared_error'
                    )
                rmse = np.sqrt(-np.mean(scores))
                return rmse
            else:
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    scores = cross_val_score(
                        model, X_subset, self.y, cv=cv, scoring='accuracy'
                    )
                # Return 1 - accuracy (to minimize)
                return 1.0 - np.mean(scores)

        except Exception:
            return 1e6  # Return high error on failure

    def _compute_complexity(
        self,
        model_idx: int,
        model_param: int,
        preproc_idx: int,
    ) -> float:
        """
        Compute normalized model complexity (0-1).

        Based on _compute_unified_complexity from scoring.py.
        """
        # Model complexity (0-1)
        model_type = self.model_types[min(model_idx, len(self.model_types) - 1)]
        if model_type == 'PLS':
            # LVs complexity: 1 LV = 0, 15 LVs = 1
            model_complexity = model_param / 14.0
        elif model_type == 'Ridge':
            # Ridge: higher alpha = simpler, lower alpha = more complex
            model_complexity = 1.0 - model_param / 14.0
        elif model_type in ['LightGBM', 'XGBoost']:
            # Boosting: more estimators = more complex
            model_complexity = 0.3 + (model_param / 14.0) * 0.5  # Base 0.3, up to 0.8
        elif model_type == 'Lasso':
            # Simple linear model with L1 regularization
            model_complexity = 0.3
        elif model_type == 'ElasticNet':
            # Slightly more complex than Lasso (two regularization terms)
            model_complexity = 0.35
        elif model_type == 'RandomForest':
            # Forest complexity scales with estimators
            model_complexity = 0.4 + (model_param / 14.0) * 0.4  # Base 0.4, up to 0.8
        elif model_type == 'CatBoost':
            # Gradient boosting with added complexity
            model_complexity = 0.35 + (model_param / 14.0) * 0.45  # Base 0.35, up to 0.8
        elif model_type == 'SVR':
            # Kernel complexity
            model_complexity = 0.5 + (model_param / 14.0) * 0.3  # Base 0.5, up to 0.8
        elif model_type == 'MLP':
            # Neural net complexity
            model_complexity = 0.5 + (model_param / 14.0) * 0.4  # Base 0.5, up to 0.9
        elif model_type == 'NeuralBoosted':
            # Ensemble neural - between NN and boosting
            model_complexity = 0.45 + (model_param / 14.0) * 0.4  # Base 0.45, up to 0.85
        else:
            model_complexity = 0.5

        # Preprocessing complexity (0-1)
        preproc_type = PREPROC_TYPES[min(preproc_idx, len(PREPROC_TYPES) - 1)]
        if preproc_type == 'raw':
            preproc_complexity = 0.0
        elif preproc_type == 'snv':
            preproc_complexity = 0.2
        elif 'deriv1' in preproc_type:
            preproc_complexity = 0.5 if 'snv' in preproc_type else 0.4
        elif 'deriv2' in preproc_type:
            preproc_complexity = 0.7 if 'snv' in preproc_type else 0.6
        elif 'deriv3' in preproc_type or 'deriv4' in preproc_type:
            preproc_complexity = 0.8
        else:
            preproc_complexity = 0.3

        # Combined complexity (model weighted more)
        complexity = 0.7 * model_complexity + 0.3 * preproc_complexity
        return complexity


# =============================================================================
# Knee Point Detection
# =============================================================================

def find_knee_point(pareto_front: np.ndarray) -> int:
    """
    Find the knee point in a Pareto front using the maximum curvature method.

    The knee point represents the "best compromise" solution where
    improving one objective significantly worsens others.

    Parameters
    ----------
    pareto_front : ndarray, shape (n_solutions, n_objectives)
        Pareto front objective values (all minimized)

    Returns
    -------
    knee_idx : int
        Index of the knee point solution
    """
    if len(pareto_front) <= 2:
        return 0

    # Normalize objectives to [0, 1]
    pf_min = pareto_front.min(axis=0)
    pf_max = pareto_front.max(axis=0)
    pf_range = pf_max - pf_min
    pf_range[pf_range == 0] = 1  # Avoid division by zero

    pf_norm = (pareto_front - pf_min) / pf_range

    # For 2D: find maximum perpendicular distance to line from first to last
    if pareto_front.shape[1] == 2:
        # Sort by first objective
        sort_idx = np.argsort(pf_norm[:, 0])
        pf_sorted = pf_norm[sort_idx]

        # Line from first to last point
        p1 = pf_sorted[0]
        p2 = pf_sorted[-1]

        # Perpendicular distance for each point
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return sort_idx[len(sort_idx) // 2]

        line_unit = line_vec / line_len

        distances = []
        for i, p in enumerate(pf_sorted):
            vec = p - p1
            proj_len = np.dot(vec, line_unit)
            proj = p1 + proj_len * line_unit
            dist = np.linalg.norm(p - proj)
            distances.append(dist)

        # Return original index of maximum distance point
        max_dist_sorted_idx = np.argmax(distances)
        return sort_idx[max_dist_sorted_idx]

    # For 3D+: find point closest to ideal point (utopia point)
    # The ideal point is the minimum of each objective (impossible to achieve all at once)
    ideal = pf_norm.min(axis=0)

    # Distance to ideal point
    distances = np.sqrt(np.sum((pf_norm - ideal) ** 2, axis=1))

    return int(np.argmin(distances))


def weighted_selection(
    pareto_front: np.ndarray,
    weights: List[float] = None,
) -> int:
    """
    Select solution using weighted sum of normalized objectives.

    Parameters
    ----------
    pareto_front : ndarray, shape (n_solutions, n_objectives)
        Pareto front objective values
    weights : list of float, optional
        Weights for each objective (default: equal weights)

    Returns
    -------
    idx : int
        Index of selected solution
    """
    n_obj = pareto_front.shape[1]

    if weights is None:
        weights = [1.0 / n_obj] * n_obj

    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    # Normalize objectives
    pf_min = pareto_front.min(axis=0)
    pf_max = pareto_front.max(axis=0)
    pf_range = pf_max - pf_min
    pf_range[pf_range == 0] = 1

    pf_norm = (pareto_front - pf_min) / pf_range

    # Weighted sum
    weighted_scores = np.sum(pf_norm * weights, axis=1)

    return int(np.argmin(weighted_scores))


# =============================================================================
# Main NSGA-II Function
# =============================================================================

def run_nsga2_search(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    population_size: int = 50,
    n_generations: int = 100,
    cv_folds: int = 5,
    min_wavelengths: int = 10,
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run NSGA-II multi-objective optimization for spectral calibration.

    Optimizes three objectives:
    1. Prediction error (RMSE or 1-Accuracy)
    2. Number of wavelengths (parsimony)
    3. Model complexity

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_wavelengths)
        Spectral data
    y : ndarray, shape (n_samples,)
        Target values
    task_type : str
        'regression' or 'classification'
    population_size : int
        NSGA-II population size
    n_generations : int
        Number of generations
    cv_folds : int
        Number of CV folds
    min_wavelengths : int
        Minimum wavelengths to select
    random_state : int
        Random state
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Callback function(dict) for progress updates

    Returns
    -------
    result : dict
        Dictionary with:
        - 'pareto_front': ndarray, objective values for Pareto solutions
        - 'pareto_solutions': ndarray, decision variables for Pareto solutions
        - 'knee_idx': int, index of knee point solution
        - 'knee_solution': dict, decoded knee point solution
        - 'n_evaluations': int, total fitness evaluations
        - 'history': list, best objective per generation
    """
    check_pymoo_available()

    # Use user-specified models or defaults
    if models is None:
        models = MODEL_TYPES

    if verbose >= 1:
        print("NSGA-II Multi-Objective Optimization")
        print(f"  Data: {X.shape[0]} samples, {X.shape[1]} wavelengths")
        print(f"  Task: {task_type}")
        print(f"  Models: {models}")
        print(f"  Population: {population_size}, Generations: {n_generations}")
        print(f"  CV folds: {cv_folds}, Min wavelengths: {min_wavelengths}")

    # Create problem with user-specified models
    problem = SpectralOptimizationProblem(
        X=X,
        y=y,
        task_type=task_type,
        cv_folds=cv_folds,
        min_wavelengths=min_wavelengths,
        random_state=random_state,
        models=models,
    )

    # Configure NSGA-II
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.1, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    # Termination criterion
    termination = get_termination("n_gen", n_generations)

    # Track history for progress
    history = []

    class ProgressCallback:
        def __init__(self, total_gen, callback, verbose):
            self.total_gen = total_gen
            self.callback = callback
            self.verbose = verbose
            self.gen = 0
            self.best_error = None
            self.n_pareto = 0

        def __call__(self, algorithm):
            self.gen += 1

            # Get current Pareto front
            if algorithm.pop is not None and len(algorithm.pop) > 0:
                F = algorithm.pop.get("F")
                if F is not None and len(F) > 0:
                    # Best error (first objective)
                    self.best_error = F[:, 0].min()
                    self.n_pareto = len(F)
                    history.append(self.best_error)

                    if self.verbose >= 1 and self.gen % 10 == 0:
                        print(f"  Gen {self.gen}/{self.total_gen}: "
                              f"Pareto size={self.n_pareto}, Best error={self.best_error:.4f}")

            if self.callback is not None:
                msg = f"Gen {self.gen}/{self.total_gen}"
                if self.best_error is not None:
                    msg = f"Gen {self.gen}/{self.total_gen}: Pareto size={self.n_pareto}, Best error={self.best_error:.4f}"
                self.callback({
                    'algorithm': 'nsga2',
                    'generation': self.gen,
                    'total_generations': self.total_gen,
                    'best_fitness': -self.best_error if self.best_error is not None else None,
                    'message': msg,
                })

    callback = ProgressCallback(n_generations, progress_callback, verbose)

    # Run optimization
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=random_state,
            callback=callback,
            verbose=False,
        )

    # Extract results
    pareto_front = res.F  # Objective values
    pareto_solutions = res.X  # Decision variables

    if pareto_front is None or len(pareto_front) == 0:
        if verbose >= 1:
            print("Warning: No feasible solutions found")
        return {
            'pareto_front': np.array([]),
            'pareto_solutions': np.array([]),
            'knee_idx': -1,
            'knee_solution': None,
            'n_evaluations': problem._eval_count,
            'history': history,
            'model_types': models,  # Store for later use
        }

    # Find knee point
    knee_idx = find_knee_point(pareto_front)

    # Decode knee solution (pass models for correct model name lookup)
    knee_chromosome = pareto_solutions[knee_idx].astype(int)
    knee_solution = decode_solution(knee_chromosome, problem.n_wavelengths, models)
    knee_solution['objectives'] = {
        'error': pareto_front[knee_idx, 0],
        'n_wavelengths': pareto_front[knee_idx, 1] * problem.n_wavelengths,
        'complexity': pareto_front[knee_idx, 2],
    }

    if verbose >= 1:
        print(f"\nOptimization complete!")
        print(f"  Pareto front size: {len(pareto_front)}")
        print(f"  Total evaluations: {problem._eval_count}")
        print(f"\nKnee point solution:")
        print(f"  Preprocessing: {knee_solution['preprocessing']}")
        print(f"  Model: {knee_solution['model']} ({knee_solution['model_params']})")
        print(f"  Wavelengths: {knee_solution['n_wavelengths']} selected")
        print(f"  Error: {knee_solution['objectives']['error']:.4f}")
        print(f"  Complexity: {knee_solution['objectives']['complexity']:.4f}")

    return {
        'pareto_front': pareto_front,
        'pareto_solutions': pareto_solutions,
        'knee_idx': knee_idx,
        'knee_solution': knee_solution,
        'n_evaluations': problem._eval_count,
        'history': history,
        'model_types': models,  # Store for later use
    }


def decode_solution(chromosome: np.ndarray, n_wavelengths: int, model_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Decode a chromosome into human-readable solution description.

    Parameters
    ----------
    chromosome : ndarray
        Integer-encoded solution
    n_wavelengths : int
        Total number of wavelengths
    model_types : list of str, optional
        Model types used in optimization. If None, uses default MODEL_TYPES.

    Returns
    -------
    solution : dict
        Decoded solution with preprocessing, model, wavelengths info
    """
    if model_types is None:
        model_types = MODEL_TYPES

    preproc_idx = int(chromosome[0])
    window_idx = int(chromosome[1])
    model_idx = int(chromosome[2])
    model_param = int(chromosome[3])
    wavelength_mask = chromosome[4:].astype(bool)

    # Preprocessing
    preproc_type = PREPROC_TYPES[min(preproc_idx, len(PREPROC_TYPES) - 1)]
    window = WINDOW_SIZES[min(window_idx, len(WINDOW_SIZES) - 1)]

    if preproc_type == 'raw':
        preproc_name = 'raw'
    elif preproc_type == 'snv':
        preproc_name = 'snv'
    else:
        preproc_name = f"{preproc_type}_w{window}"

    # Model
    model_type = model_types[min(model_idx, len(model_types) - 1)]
    if model_type == 'PLS':
        n_components = model_param + 1
        model_params = f"n_components={n_components}"
    elif model_type == 'Ridge':
        alpha = 10 ** (model_param / 3 - 2)
        model_params = f"alpha={alpha:.2e}"
    elif model_type == 'LightGBM':
        n_estimators = 50 + model_param * 10
        learning_rate = 0.05 if model_param < 7 else 0.1
        model_params = f"n_estimators={n_estimators}, lr={learning_rate}"
    elif model_type == 'XGBoost':
        n_estimators = 50 + model_param * 10
        max_depth = 3 + (model_param % 5)
        subsample = 1.0 if model_param < 7 else 0.8
        model_params = f"n_estimators={n_estimators}, max_depth={max_depth}, subsample={subsample}"
    elif model_type == 'Lasso':
        alpha = 10 ** (model_param / 3 - 3)
        model_params = f"alpha={alpha:.2e}"
    elif model_type == 'ElasticNet':
        alpha = 10 ** (model_param / 3 - 2)
        l1_ratio = 0.2 + (model_param % 5) * 0.15
        model_params = f"alpha={alpha:.2e}, l1_ratio={l1_ratio:.2f}"
    elif model_type == 'RandomForest':
        n_estimators = 50 + model_param * 15
        max_depth = None if model_param < 5 else 10 + model_param * 3
        model_params = f"n_estimators={n_estimators}, max_depth={max_depth}"
    elif model_type == 'CatBoost':
        iterations = 50 + model_param * 15
        depth = 4 + (model_param % 5)
        model_params = f"iterations={iterations}, depth={depth}"
    elif model_type == 'SVR':
        kernel = 'rbf' if model_param < 10 else 'linear'
        C = 10 ** (model_param / 3 - 1)
        model_params = f"kernel={kernel}, C={C:.2e}"
    elif model_type == 'MLP':
        layer_size = 30 + model_param * 10
        n_layers = 1 if model_param < 7 else 2
        model_params = f"layers={n_layers}, size={layer_size}"
    elif model_type == 'NeuralBoosted':
        n_estimators = 30 + model_param * 10
        hidden_layer_size = 3 + (model_param % 5)
        model_params = f"n_estimators={n_estimators}, hidden={hidden_layer_size}"
    else:
        model_params = ""

    # Wavelengths
    selected_indices = np.where(wavelength_mask)[0].tolist()

    return {
        'preprocessing': preproc_name,
        'preproc_idx': preproc_idx,
        'window_idx': window_idx,
        'model': model_type,
        'model_idx': model_idx,
        'model_param': model_param,
        'model_params': model_params,
        'wavelength_mask': wavelength_mask,
        'selected_indices': selected_indices,
        'n_wavelengths': len(selected_indices),
    }


def pareto_to_dataframe(
    result: Dict[str, Any],
    n_wavelengths: int,
    task_type: str = 'regression',
    model_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert NSGA-II Pareto front to results DataFrame.

    Parameters
    ----------
    result : dict
        Result from run_nsga2_search
    n_wavelengths : int
        Total number of wavelengths
    task_type : str
        'regression' or 'classification'
    model_types : list of str, optional
        Model types used. If None, uses result['model_types'] or defaults.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with one row per Pareto solution
    """
    if len(result['pareto_front']) == 0:
        return pd.DataFrame()

    # Get model_types from result if not provided
    if model_types is None:
        model_types = result.get('model_types', MODEL_TYPES)

    rows = []
    for i, (objectives, solution) in enumerate(zip(
        result['pareto_front'],
        result['pareto_solutions']
    )):
        decoded = decode_solution(solution, n_wavelengths, model_types)

        row = {
            'Model': decoded['model'],
            'Preprocessing': decoded['preprocessing'],
            'Variables': 'nsga2',
            'N_Variables': decoded['n_wavelengths'],
            'Params': decoded['model_params'],
            'Complexity': objectives[2],
            'Is_Knee': i == result['knee_idx'],
        }

        if task_type == 'regression':
            row['RMSE'] = objectives[0]
            row['R2'] = None  # Would need recomputation
        else:
            row['Accuracy'] = 1.0 - objectives[0]
            row['ROC_AUC'] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by error (best first)
    error_col = 'RMSE' if task_type == 'regression' else 'Accuracy'
    ascending = task_type == 'regression'  # Lower RMSE is better, higher Accuracy is better
    df = df.sort_values(error_col, ascending=ascending).reset_index(drop=True)

    return df


# =============================================================================
# Convenience wrapper for integration
# =============================================================================

def run_nsga2_auto(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    preset: str = 'standard',
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run NSGA-II with preset configurations.

    Parameters
    ----------
    X : ndarray
        Spectral data
    y : ndarray
        Target values
    task_type : str
        'regression' or 'classification'
    preset : str
        'quick' (pop=30, gen=50), 'standard' (pop=50, gen=100),
        'thorough' (pop=100, gen=200)
    cv_folds : int
        Number of CV folds
    random_state : int
        Random state
    verbose : int
        Verbosity level
    progress_callback : callable, optional
        Progress callback

    Returns
    -------
    df : pd.DataFrame
        Results DataFrame
    result : dict
        Full NSGA-II result
    """
    presets = {
        'quick': {'population_size': 30, 'n_generations': 50},
        'standard': {'population_size': 50, 'n_generations': 100},
        'thorough': {'population_size': 100, 'n_generations': 200},
    }

    params = presets.get(preset, presets['standard'])

    result = run_nsga2_search(
        X=X,
        y=y,
        task_type=task_type,
        population_size=params['population_size'],
        n_generations=params['n_generations'],
        cv_folds=cv_folds,
        min_wavelengths=max(10, X.shape[1] // 20),  # At least 5% or 10
        random_state=random_state,
        verbose=verbose,
        progress_callback=progress_callback,
    )

    df = pareto_to_dataframe(result, X.shape[1], task_type)

    return df, result
