"""
NSGA-II Multi-Objective Optimization for Spectral Predict V1.

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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder

# pymoo imports (required dependency)
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# V1 imports - use local modules
from .preprocess import SNV, SavgolDerivative
from .models import get_feature_importances

# All model libraries are required dependencies
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from .neural_boosted import NeuralBoostedRegressor, NeuralBoostedClassifier




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


def _get_edge_zone_size(preproc_idx: int, window_idx: int) -> int:
    """
    Get the size of the edge zone to exclude for derivative preprocessing.

    For derivative preprocessing (Savitzky-Golay), edge wavelengths are
    unreliable due to interpolation effects. Edge zone = window // 2.

    Parameters
    ----------
    preproc_idx : int
        Index into PREPROC_TYPES
    window_idx : int
        Index into WINDOW_SIZES

    Returns
    -------
    edge_zone : int
        Number of wavelengths to exclude on each edge (0 if no derivative)
    """
    preproc_type = PREPROC_TYPES[min(preproc_idx, len(PREPROC_TYPES) - 1)]

    # Check if this preprocessing uses derivatives
    if 'deriv' not in preproc_type:
        return 0

    # Get window size
    window = WINDOW_SIZES[min(window_idx, len(WINDOW_SIZES) - 1)]

    # Edge zone is half the window size
    return window // 2


class SeededWavelengthSampling(Sampling):
    """
    Custom sampling that seeds initial population with strategic solutions.

    Includes:
    - All-wavelengths solutions (one per model type) to ensure baseline performance
    - Random solutions for exploration

    This addresses the issue where pure random sampling makes it very unlikely
    to generate solutions with all wavelengths selected, causing NSGA to miss
    the high-performance corner of the Pareto front.
    """

    def __init__(self, n_wavelengths: int, model_types: List[str], n_preproc: int = 10, n_window: int = 15):
        super().__init__()
        self.n_wavelengths = n_wavelengths
        self.model_types = model_types
        self.n_preproc = n_preproc
        self.n_window = n_window

    def _do(self, problem, n_samples, **kwargs):
        """Generate initial population with seeded all-wavelengths solutions."""
        # Variable structure: [preproc_idx, window_idx, model_idx, model_param, lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
        #                      subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene, wl_0, wl_1, ..., wl_n]
        n_vars = 13 + self.n_wavelengths

        # Initialize with random values
        X = np.random.randint(
            problem.xl,
            problem.xu + 1,
            size=(n_samples, n_vars)
        )

        # Seed first few solutions with ALL wavelengths and different model types
        # This ensures NSGA has "full wavelength" baselines to work from
        n_seeded = min(len(self.model_types), n_samples // 4, 5)  # Seed up to 5 solutions

        for i in range(n_seeded):
            model_idx = i % len(self.model_types)
            # Use raw preprocessing (idx=0) and default window (idx=6 = window 17)
            X[i, 0] = 0   # raw preprocessing
            X[i, 1] = 6   # default window
            X[i, 2] = model_idx  # cycle through model types
            X[i, 3] = 7   # middle model_param value (good default)
            # Use Grid Search default hyperparameters for seeded solutions
            X[i, 4] = 10  # lr ≈ 0.1 (0.01 * 30^(10/14) ≈ 0.105)
            X[i, 5] = 12  # reg_alpha ≈ 0.07 (close to Grid Search 0.1)
            X[i, 6] = 14  # reg_lambda = 1.0 (matches Grid Search exactly)
            X[i, 7] = 7   # l1_ratio = 0.5 (middle, reasonable default)
            X[i, 8] = 10  # subsample ≈ 0.86 (close to Grid Search 0.8)
            X[i, 9] = 10  # colsample ≈ 0.86 (close to Grid Search 0.8)
            X[i, 10] = 2  # min_samples ≈ 5 (1 + (2/14)*29 ≈ 5, matches Grid Search)
            X[i, 11] = 0  # gamma = 0 (no penalty)
            X[i, 12] = 7  # max_features = 0.3 (reasonable default)
            X[i, 13:] = 1  # ALL wavelengths selected

        # Also add some solutions with SNV preprocessing + all wavelengths
        for i in range(n_seeded, min(n_seeded * 2, n_samples // 2)):
            model_idx = (i - n_seeded) % len(self.model_types)
            X[i, 0] = 1   # SNV preprocessing
            X[i, 1] = 6   # default window
            X[i, 2] = model_idx
            X[i, 3] = 7   # middle model_param value
            # Use Grid Search default hyperparameters for seeded solutions
            X[i, 4] = 10  # lr ≈ 0.1 (matches Grid Search)
            X[i, 5] = 12  # reg_alpha ≈ 0.07 (close to Grid Search 0.1)
            X[i, 6] = 14  # reg_lambda = 1.0 (matches Grid Search exactly)
            X[i, 7] = 7   # l1_ratio = 0.5 (reasonable default)
            X[i, 8] = 10  # subsample ≈ 0.86 (close to Grid Search 0.8)
            X[i, 9] = 10  # colsample ≈ 0.86 (close to Grid Search 0.8)
            X[i, 10] = 2  # min_samples ≈ 5 (matches Grid Search)
            X[i, 11] = 0  # gamma = 0 (no penalty)
            X[i, 12] = 7  # max_features = 0.3 (reasonable default)
            X[i, 13:] = 1  # ALL wavelengths selected

        return X


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
    preproc_type = PREPROC_TYPES[min(preproc_idx, len(PREPROC_TYPES) - 1)]
    window = WINDOW_SIZES[min(window_idx, len(WINDOW_SIZES) - 1)]

    if preproc_type == 'raw':
        return None
    elif preproc_type == 'snv':
        snv = SNV()
        return lambda X: snv.fit_transform(X)
    elif preproc_type.startswith('deriv') or 'deriv' in preproc_type:
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




def _decode_hyperparameter_genes(
    lr_gene: int,
    reg_alpha_gene: int,
    reg_lambda_gene: int,
    l1_gene: int,
    subsample_gene: int = 7,
    colsample_gene: int = 7,
    min_samples_gene: int = 7,
    gamma_gene: int = 0,
    max_features_gene: int = 7,
) -> Dict[str, float]:
    """
    Decode hyperparameter genes to actual values.

    Parameters
    ----------
    lr_gene : int (0-14)
        Learning rate gene
    reg_alpha_gene : int (0-14)
        L1 regularization gene
    reg_lambda_gene : int (0-14)
        L2 regularization gene
    l1_gene : int (0-14)
        ElasticNet l1_ratio gene
    subsample_gene : int (0-14)
        Subsample ratio gene for tree models
    colsample_gene : int (0-14)
        Column sample ratio gene for tree models
    min_samples_gene : int (0-14)
        Minimum samples gene for tree models
    gamma_gene : int (0-14)
        Gamma/penalty gene for XGBoost, CatBoost, SVR
    max_features_gene : int (0-14)
        Max features gene for RandomForest

    Returns
    -------
    params : dict
        Decoded hyperparameters with keys:
        - learning_rate: float, range 0.01 to 0.3 (log scale)
        - reg_alpha: float, range 1e-8 to 10.0 (log scale)
        - reg_lambda: float, range 1e-8 to 10.0 (log scale)
        - l1_ratio: float, range 0.1 to 0.9 (linear scale)
        - subsample: float, range 0.5 to 1.0 (linear scale)
        - colsample_bytree: float, range 0.5 to 1.0 (linear scale)
        - min_samples: int, range 1 to 30 (linear scale)
        - gamma: float, range 0 to 5 (linear scale)
        - max_features: float, range 0.1 to 0.5 (linear scale)
    """
    # Learning rate: log scale 0.01 to 0.3
    # Formula: lr = 0.01 * (30.0 ** (gene / 14))
    # gene=0 -> 0.01, gene=10 -> 0.1 (Grid Search default), gene=14 -> 0.3
    lr = 0.01 * (30.0 ** (lr_gene / 14.0))

    # Regularization: log scale 1e-8 to 1.0
    # Formula: reg = 10 ** ((gene/14) * 8 - 8)
    # gene=0 -> 1e-8, gene=7 -> 1e-4, gene=14 -> 1.0 (Grid Search default)
    reg_alpha = 10 ** (reg_alpha_gene / 14.0 * 8 - 8)
    reg_lambda = 10 ** (reg_lambda_gene / 14.0 * 8 - 8)

    # l1_ratio: linear scale 0.1 to 0.9
    # Formula: l1_ratio = 0.1 + (gene / 14) * 0.8
    # gene=0 -> 0.1, gene=7 -> 0.5, gene=14 -> 0.9
    l1_ratio = 0.1 + (l1_gene / 14.0) * 0.8

    # subsample: linear scale 0.5 to 1.0
    # gene=0 -> 0.5, gene=7 -> 0.75, gene=14 -> 1.0
    subsample = 0.5 + (subsample_gene / 14.0) * 0.5

    # colsample_bytree: linear scale 0.5 to 1.0
    # gene=0 -> 0.5, gene=7 -> 0.75, gene=14 -> 1.0
    colsample = 0.5 + (colsample_gene / 14.0) * 0.5

    # min_samples: linear scale 1 to 30
    # gene=0 -> 1, gene=7 -> 15, gene=14 -> 30
    min_samples = 1 + int((min_samples_gene / 14.0) * 29)

    # gamma: linear scale 0 to 5
    # gene=0 -> 0, gene=7 -> 2.5, gene=14 -> 5
    gamma = (gamma_gene / 14.0) * 5.0

    # max_features: linear scale 0.1 to 0.5
    # gene=0 -> 0.1, gene=7 -> 0.3, gene=14 -> 0.5
    max_features = 0.1 + (max_features_gene / 14.0) * 0.4

    return {
        'learning_rate': lr,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'l1_ratio': l1_ratio,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'min_samples': min_samples,
        'gamma': gamma,
        'max_features': max_features,
    }

def _build_model(model_type: str, model_param: int, task_type: str, random_state: int, hyperparams: Optional[Dict[str, float]] = None):
    """
    Build a model instance based on type and parameter encoding.

    Parameters
    ----------
    model_type : str
        Model type name
    model_param : int
        Encoded model parameter (0-14)
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random state for reproducibility

    Returns
    -------
    model : sklearn estimator or None
        Model instance or None if not available
    """
    if model_type == 'PLS':
        n_components = model_param + 1  # 1-15
        return PLSRegression(n_components=n_components, scale=False)

    elif model_type == 'Ridge':
        # Exponential alpha scale: 1e-4 to 1e3
        alpha = 10 ** (model_param / 3 - 2)  # param 0-14 -> alpha 0.01 to 1000
        return Ridge(alpha=alpha, random_state=random_state)

    elif model_type == 'Lasso':
        # alpha from 1e-3 to 100 (log scale)
        alpha = 10 ** (model_param / 3 - 3)  # param 0-14 -> alpha ~0.001 to 100
        if task_type == 'regression':
            return Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
        else:
            # For classification, use LogisticRegression with L1
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(penalty='l1', C=1/max(alpha, 1e-6), solver='saga',
                                      random_state=random_state, max_iter=1000)

    elif model_type == 'ElasticNet':
        # alpha from 0.01 to 100
        alpha = 10 ** (model_param / 3 - 2)
        # Use independent l1_ratio from hyperparams (fixes BUG 6)
        l1_ratio = hyperparams.get('l1_ratio', 0.5) if hyperparams else 0.5
        if task_type == 'regression':
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(penalty='elasticnet', C=1/max(alpha, 1e-6), l1_ratio=l1_ratio,
                                      solver='saga', random_state=random_state, max_iter=1000)

    elif model_type == 'RandomForest':
        n_estimators = 50 + model_param * 15  # 50-260
        max_depth = None if model_param < 5 else 10 + model_param * 3  # None or 25-67
        # Use new hyperparameter genes for regularization
        max_features = hyperparams.get('max_features', 0.3) if hyperparams else 'sqrt'
        min_samples = hyperparams.get('min_samples', 5) if hyperparams else 5
        min_samples_leaf = max(1, min_samples // 5)  # Scale down for leaf param
        if task_type == 'regression':
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=2,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                random_state=random_state,
                n_jobs=1
            )
        else:
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=2,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                random_state=random_state,
                n_jobs=1
            )

    elif model_type == 'LightGBM':
        n_estimators = 50 + model_param * 10  # 50-190
        # Use hyperparams for all regularization parameters
        learning_rate = hyperparams.get('learning_rate', 0.1) if hyperparams else 0.1
        reg_lambda = hyperparams.get('reg_lambda', 1.0) if hyperparams else 1.0
        subsample = hyperparams.get('subsample', 0.8) if hyperparams else 0.8
        colsample = hyperparams.get('colsample_bytree', 0.8) if hyperparams else 0.8
        min_child_samples = hyperparams.get('min_samples', 5) if hyperparams else 5
        num_leaves = 15 if task_type == 'classification' else 31
        if task_type == 'regression':
            return LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                subsample_freq=1,  # Required when subsample < 1
                colsample_bytree=colsample,
                reg_alpha=0.1,  # Fixed at Grid Search default
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=1,
                verbose=-1
            )
        else:
            return LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                subsample_freq=1,
                colsample_bytree=colsample,
                reg_alpha=0.1,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=1,
                verbose=-1
            )

    elif model_type == 'XGBoost':
        n_estimators = 50 + model_param * 10  # 50-190
        max_depth = 3 + (model_param % 5)  # 3-7
        # Use hyperparams for all regularization parameters
        learning_rate = hyperparams.get('learning_rate', 0.1) if hyperparams else 0.1
        reg_alpha = hyperparams.get('reg_alpha', 0.1) if hyperparams else 0.1
        reg_lambda = hyperparams.get('reg_lambda', 1.0) if hyperparams else 1.0
        subsample = hyperparams.get('subsample', 0.8) if hyperparams else 0.8
        colsample = hyperparams.get('colsample_bytree', 0.8) if hyperparams else 0.8
        gamma = hyperparams.get('gamma', 0) if hyperparams else 0
        min_child_weight = max(1, hyperparams.get('min_samples', 5) // 3) if hyperparams else 1
        if task_type == 'regression':
            return XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample,
                gamma=gamma,
                min_child_weight=min_child_weight,
                tree_method='hist',  # Fixed at Grid Search default
                random_state=random_state,
                n_jobs=1,
                verbosity=0
            )
        else:
            return XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample,
                gamma=gamma,
                min_child_weight=min_child_weight,
                tree_method='hist',
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )

    elif model_type == 'CatBoost':
        iterations = 50 + model_param * 15  # 50-260
        depth = 4 + (model_param % 5)  # 4-8
        # Use hyperparams for regularization
        learning_rate = hyperparams.get('learning_rate', 0.1) if hyperparams else 0.1
        # Reuse gamma gene for l2_leaf_reg: gamma 0-5 maps to l2_leaf_reg 1-11
        l2_leaf_reg = 1.0 + (hyperparams.get('gamma', 0) * 2) if hyperparams else 3.0
        if task_type == 'regression':
            return CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_state=random_state,
                verbose=0,
                thread_count=1
            )
        else:
            return CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_state=random_state,
                verbose=0,
                thread_count=1
            )

    elif model_type == 'SVR':
        kernel = 'rbf' if model_param < 10 else 'linear'
        C = 10 ** (model_param / 3 - 1)  # 0.1 to ~1000
        # Reuse gamma gene for epsilon: gamma 0-5 maps to epsilon 0.01-0.3
        epsilon = 0.01 + (hyperparams.get('gamma', 0) / 5.0) * 0.29 if hyperparams else 0.1
        if task_type == 'regression':
            return SVR(kernel=kernel, C=C, gamma='scale', epsilon=epsilon, max_iter=5000)
        else:
            return SVC(kernel=kernel, C=C, gamma='scale', random_state=random_state, probability=True, max_iter=5000)

    elif model_type == 'MLP':
        layer_size = 30 + model_param * 10  # 30-170
        n_layers = 1 if model_param < 7 else 2
        hidden_layer_sizes = (layer_size,) if n_layers == 1 else (layer_size, layer_size // 2)
        alpha = 10 ** (model_param / 5 - 4)  # 1e-4 to 0.01
        # Repurpose learning_rate gene for learning_rate_init
        lr_init = hyperparams.get('learning_rate', 0.001) if hyperparams else 0.001
        if task_type == 'regression':
            return MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=lr_init,
                activation='relu',
                solver='adam',
                random_state=random_state,
                max_iter=500,
                early_stopping=True
            )
        else:
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=lr_init,
                activation='relu',
                solver='adam',
                random_state=random_state,
                max_iter=500,
                early_stopping=True
            )

    elif model_type == 'NeuralBoosted':
        n_estimators = 30 + model_param * 10  # 30-170
        hidden_layer_size = 3 + (model_param % 5)  # 3-7
        learning_rate = 0.05 + (model_param / 14) * 0.2  # 0.05-0.25
        if task_type == 'regression':
            return NeuralBoostedRegressor(n_estimators=n_estimators, hidden_layer_size=hidden_layer_size,
                                          learning_rate=learning_rate, random_state=random_state)
        else:
            return NeuralBoostedClassifier(n_estimators=n_estimators, hidden_layer_size=hidden_layer_size,
                                           learning_rate=learning_rate, random_state=random_state)

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
    - Gene 2: Model type (0=PLS, 1=Ridge, ...)
    - Gene 3: Model parameter (0-14)
    - Gene 4: Learning rate gene (0-14)
    - Gene 5: L1 regularization gene (0-14)
    - Gene 6: L2 regularization gene (0-14)
    - Gene 7: ElasticNet l1_ratio gene (0-14)
    - Gene 8: Subsample gene (0-14) -> [0.5, 1.0]
    - Gene 9: Colsample gene (0-14) -> [0.5, 1.0]
    - Gene 10: Min samples gene (0-14) -> [1, 30]
    - Gene 11: Gamma gene (0-14) -> [0, 5]
    - Gene 12: Max features gene (0-14) -> [0.1, 0.5]
    - Gene 13-N: Binary wavelength selection (0/1)

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
        # [preproc_type, window_idx, model_type, model_param, lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
        #  subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene, wl_0, wl_1, ..., wl_n]
        n_vars = 13 + self.n_wavelengths

        # Variable bounds
        xl = np.zeros(n_vars)
        xu = np.array([
            len(PREPROC_TYPES) - 1,  # 0: preproc_type
            len(WINDOW_SIZES) - 1,   # 1: window_idx
            len(self.model_types) - 1,  # 2: model_type
            14,  # 3: model_param (LVs 1-15 or alpha index)
            14,  # 4: lr_gene (0-14)
            14,  # 5: reg_alpha_gene (0-14)
            14,  # 6: reg_lambda_gene (0-14)
            14,  # 7: l1_gene (0-14)
            14,  # 8: subsample_gene (0-14) -> [0.5, 1.0]
            14,  # 9: colsample_gene (0-14) -> [0.5, 1.0]
            14,  # 10: min_samples_gene (0-14) -> [1, 30]
            14,  # 11: gamma_gene (0-14) -> [0, 5]
            14,  # 12: max_features_gene (0-14) -> [0.1, 0.5]
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
            lr_gene = chromosome[4]
            reg_alpha_gene = chromosome[5]
            reg_lambda_gene = chromosome[6]
            l1_gene = chromosome[7]
            subsample_gene = chromosome[8]
            colsample_gene = chromosome[9]
            min_samples_gene = chromosome[10]
            gamma_gene = chromosome[11]
            max_features_gene = chromosome[12]
            wavelength_mask = chromosome[13:].astype(bool)

            # Mask out edge wavelengths for derivative preprocessing
            # Edge zone = window // 2 on each side (unreliable due to SG interpolation)
            edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
            if edge_zone > 0:
                wavelength_mask = wavelength_mask.copy()  # Don't modify original
                wavelength_mask[:edge_zone] = False
                wavelength_mask[-edge_zone:] = False

            # Count selected wavelengths (after edge masking)
            n_selected = np.sum(wavelength_mask)

            # Constraint: minimum wavelengths
            G[i, 0] = self.min_wavelengths - n_selected  # <= 0 means feasible

            # If too few wavelengths, use penalty
            if n_selected < self.min_wavelengths:
                F[i, 0] = 1e10  # Very high error (matches Bayesian)
                F[i, 1] = 1.0  # Normalized wavelength count
                F[i, 2] = 1.0  # Normalized complexity

                if self.cache_enabled:
                    self._cache[cache_key] = (F[i].copy(), G[i].copy())
                continue

            # Objective 1: Prediction error
            error = self._compute_prediction_error(
                preproc_idx, window_idx, model_idx, model_param, wavelength_mask,
                lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
                subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene
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
        lr_gene: int,
        reg_alpha_gene: int,
        reg_lambda_gene: int,
        l1_gene: int,
        subsample_gene: int = 7,
        colsample_gene: int = 7,
        min_samples_gene: int = 7,
        gamma_gene: int = 0,
        max_features_gene: int = 7,
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
                return 1e10
            if np.any(np.isnan(X_subset)) or np.any(np.isinf(X_subset)):
                return 1e10

            # Build model
            model_type = self.model_types[min(model_idx, len(self.model_types) - 1)]

            # Decode hyperparameter genes (including new regularization genes)
            hyperparams = _decode_hyperparameter_genes(
                lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
                subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene
            )

            # Special handling for PLS - limit components and use scale=False
            # scale=False matches get_model() in models.py for consistency with Model Development
            if model_type == 'PLS':
                n_components = min(model_param + 1, X_subset.shape[1], X_subset.shape[0] - 1)
                n_components = max(1, n_components)
                model = PLSRegression(n_components=n_components, scale=False)
            else:
                model = _build_model(model_type, model_param, self.task_type, self.random_state, hyperparams)

            if model is None:
                return 1e10

            # Cross-validation
            if self.task_type == 'regression':
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    scores = cross_val_score(
                        model, X_subset, self.y, cv=cv, scoring='neg_mean_squared_error'
                    )
                # Use mean(sqrt(MSE)) for optimization and display consistency
                # This matches Bayesian optimization and Model Development exactly.
                rmse_per_fold = np.sqrt(-scores)
                rmse = float(np.mean(rmse_per_fold))
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
            return 1e10  # Return high error on failure (matches Bayesian)

    def _compute_complexity(
        self,
        model_idx: int,
        model_param: int,
        preproc_idx: int,
    ) -> float:
        """
        Compute normalized model complexity (0-1).
        """
        # Model complexity (0-1)
        # Values: Fixed ordering + 50% reduction to allow tree models to compete fairly
        model_type = self.model_types[min(model_idx, len(self.model_types) - 1)]
        if model_type == 'PLS':
            # PLS: 1 LV = 0, 15 LVs = 0.25 (capped, linear model shouldn't be most complex)
            model_complexity = (model_param / 14.0) * 0.25
        elif model_type == 'Ridge':
            # Ridge: higher alpha = simpler, capped at 0.2 (linear model)
            model_complexity = (1.0 - model_param / 14.0) * 0.2
        elif model_type in ['LightGBM', 'XGBoost']:
            # Boosting: 0.15 to 0.4 (50% reduction from 0.3-0.8)
            model_complexity = 0.15 + (model_param / 14.0) * 0.25
        elif model_type == 'Lasso':
            # Simple linear model - lower than boosting
            model_complexity = 0.075
        elif model_type == 'ElasticNet':
            # Slightly more complex than Lasso
            model_complexity = 0.1
        elif model_type == 'RandomForest':
            # Forest: 0.2 to 0.4 (50% reduction from 0.4-0.8)
            model_complexity = 0.2 + (model_param / 14.0) * 0.2
        elif model_type == 'CatBoost':
            # Gradient boosting: 0.175 to 0.4 (50% reduction from 0.35-0.8)
            model_complexity = 0.175 + (model_param / 14.0) * 0.225
        elif model_type == 'SVR':
            # Kernel: 0.25 to 0.4 (50% reduction from 0.5-0.8)
            model_complexity = 0.25 + (model_param / 14.0) * 0.15
        elif model_type == 'MLP':
            # Neural net: 0.25 to 0.45 (50% reduction from 0.5-0.9)
            model_complexity = 0.25 + (model_param / 14.0) * 0.2
        elif model_type == 'NeuralBoosted':
            # Ensemble neural: 0.225 to 0.425 (50% reduction from 0.45-0.85)
            model_complexity = 0.225 + (model_param / 14.0) * 0.2
        else:
            model_complexity = 0.25

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
    controller=None,
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
    models : list of str, optional
        Model types to consider. If None, uses all available.
    controller : SearchController, optional
        Controller for cancellation. If provided and cancelled, optimization stops early.

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

    # Configure NSGA-II with custom seeded sampling
    # SeededWavelengthSampling ensures we include "all wavelengths" solutions
    # in the initial population, which helps NSGA find the high-performance
    # corner of the Pareto front (competitive with Bayesian/Grid Search)
    custom_sampling = SeededWavelengthSampling(
        n_wavelengths=problem.n_wavelengths,
        model_types=models,
        n_preproc=len(PREPROC_TYPES),
        n_window=len(WINDOW_SIZES),
    )

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=custom_sampling,
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.1, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    # Termination criterion
    termination = get_termination("n_gen", n_generations)

    # Track history for progress
    history = []

    class ProgressCallback:
        def __init__(self, total_gen, callback, verbose, ctrl):
            self.total_gen = total_gen
            self.callback = callback
            self.verbose = verbose
            self.ctrl = ctrl
            self.gen = 0
            self.best_error = None
            self.n_pareto = 0
            self.cancelled = False

        def __call__(self, algorithm):
            self.gen += 1

            # Check for cancellation
            if self.ctrl and not self.ctrl.check_and_wait():
                self.cancelled = True
                algorithm.termination.force_termination = True
                return

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

    callback = ProgressCallback(n_generations, progress_callback, verbose, controller)

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

    # Check if cancelled
    if callback.cancelled:
        if verbose >= 1:
            print("NSGA-II optimization cancelled by user")
        return {
            'pareto_front': np.array([]),
            'pareto_solutions': np.array([]),
            'knee_idx': -1,
            'knee_solution': None,
            'n_evaluations': problem._eval_count,
            'history': history,
            'model_types': models,
            'label_encoder': problem.label_encoder,
            'cancelled': True,
        }

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
            'model_types': models,
            'label_encoder': problem.label_encoder,
            'cancelled': False,
        }

    # Find knee point
    knee_idx = find_knee_point(pareto_front)

    # Decode knee solution
    knee_chromosome = pareto_solutions[knee_idx].astype(int)
    knee_solution = decode_solution(knee_chromosome, problem.n_wavelengths, models, task_type)
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
        'model_types': models,
        'label_encoder': problem.label_encoder,
        'cancelled': False,
    }


def decode_solution(chromosome: np.ndarray, n_wavelengths: int, model_types: Optional[List[str]] = None, task_type: str = 'regression') -> Dict[str, Any]:
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
    task_type : str, default='regression'
        Task type ('regression' or 'classification')

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
    lr_gene = int(chromosome[4])
    reg_alpha_gene = int(chromosome[5])
    reg_lambda_gene = int(chromosome[6])
    l1_gene = int(chromosome[7])
    subsample_gene = int(chromosome[8])
    colsample_gene = int(chromosome[9])
    min_samples_gene = int(chromosome[10])
    gamma_gene = int(chromosome[11])
    max_features_gene = int(chromosome[12])
    wavelength_mask = chromosome[13:].astype(bool)

    # Decode hyperparameter genes (including new regularization genes)
    hyperparams = _decode_hyperparameter_genes(
        lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
        subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene
    )

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
    # Build model using get_model() to match Model Development, then apply NSGA overrides
    # This ensures stored params are COMPLETE and identical to what Model Development uses
    model_type = model_types[min(model_idx, len(model_types) - 1)]
    random_state = 42  # Standard random state used throughout NSGA-II

    # Compute NSGA-specific parameter overrides
    if model_type == 'PLS':
        n_components = model_param + 1
        nsga_overrides = {'n_components': n_components}
    elif model_type == 'Ridge':
        alpha = 10 ** (model_param / 3 - 2)
        nsga_overrides = {'alpha': alpha}
    elif model_type == 'Lasso':
        alpha = 10 ** (model_param / 3 - 3)
        nsga_overrides = {'alpha': alpha, 'max_iter': 10000}
    elif model_type == 'ElasticNet':
        alpha = 10 ** (model_param / 3 - 2)
        l1_ratio = hyperparams['l1_ratio']  # Use decoded l1_ratio
        nsga_overrides = {'alpha': alpha, 'l1_ratio': l1_ratio, 'max_iter': 10000}
    elif model_type == 'RandomForest':
        n_estimators = 50 + model_param * 15
        max_depth = None if model_param < 5 else 10 + model_param * 3
        nsga_overrides = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': hyperparams['max_features'],
            'min_samples_leaf': max(1, hyperparams['min_samples'] // 5),
            'n_jobs': 1,
        }
    elif model_type == 'LightGBM':
        n_estimators = 50 + model_param * 10
        nsga_overrides = {
            'n_estimators': n_estimators,
            'learning_rate': hyperparams['learning_rate'],
            'reg_lambda': hyperparams['reg_lambda'],
            'subsample': hyperparams['subsample'],
            'colsample_bytree': hyperparams['colsample_bytree'],
            'min_child_samples': hyperparams['min_samples'],
            'n_jobs': 1,
        }
    elif model_type == 'XGBoost':
        n_estimators = 50 + model_param * 10
        max_depth = 3 + (model_param % 5)
        nsga_overrides = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': hyperparams['subsample'],
            'colsample_bytree': hyperparams['colsample_bytree'],
            'learning_rate': hyperparams['learning_rate'],
            'reg_alpha': hyperparams['reg_alpha'],
            'reg_lambda': hyperparams['reg_lambda'],
            'gamma': hyperparams['gamma'],
            'min_child_weight': max(1, hyperparams['min_samples'] // 3),
            'tree_method': 'hist',
            'n_jobs': 1,
        }
    elif model_type == 'CatBoost':
        iterations = 50 + model_param * 15
        depth = 4 + (model_param % 5)
        nsga_overrides = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': hyperparams['learning_rate'],
            'l2_leaf_reg': 1.0 + hyperparams['gamma'] * 2,  # gamma 0-5 -> l2_leaf_reg 1-11
            'thread_count': 1
        }
    elif model_type == 'SVR':
        kernel = 'rbf' if model_param < 10 else 'linear'
        C = 10 ** (model_param / 3 - 1)
        nsga_overrides = {
            'kernel': kernel,
            'C': C,
            'gamma': 'scale',
            'epsilon': 0.01 + (hyperparams['gamma'] / 5.0) * 0.29,  # gamma 0-5 -> epsilon 0.01-0.3
            'max_iter': 5000
        }
    elif model_type == 'MLP':
        layer_size = 30 + model_param * 10
        n_layers = 1 if model_param < 7 else 2
        hidden_layer_sizes = (layer_size,) if n_layers == 1 else (layer_size, layer_size // 2)
        alpha = 10 ** (model_param / 5 - 4)
        nsga_overrides = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'alpha': alpha,
            'learning_rate_init': hyperparams['learning_rate'],
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
        }
    elif model_type == 'NeuralBoosted':
        n_estimators = 30 + model_param * 10
        hidden_layer_size = 3 + (model_param % 5)
        learning_rate = 0.05 + (model_param / 14) * 0.2
        nsga_overrides = {
            'n_estimators': n_estimators,
            'hidden_layer_size': hidden_layer_size,
            'learning_rate': learning_rate,
        }
    else:
        nsga_overrides = {}

    # Build model using get_model() and apply NSGA overrides to get COMPLETE params
    try:
        from spectral_predict.models import get_model
        model = get_model(model_type, task_type=task_type, n_jobs=1)
        if model is not None:
            # Apply NSGA overrides
            valid_params = model.get_params()
            filtered_overrides = {k: v for k, v in nsga_overrides.items() if k in valid_params}
            if filtered_overrides:
                model.set_params(**filtered_overrides)
            # Get COMPLETE params (includes all defaults from get_model)
            params_dict = model.get_params()
        else:
            params_dict = nsga_overrides
    except (ImportError, ValueError):
        # Fallback to NSGA overrides only
        params_dict = nsga_overrides

    # Convert dict to string for storage (parseable by ast.literal_eval)
    model_params = str(params_dict)

    # Apply edge masking for derivative preprocessing (same as fitness evaluation)
    # Edge wavelengths are unreliable due to Savitzky-Golay interpolation artifacts
    edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
    if edge_zone > 0:
        wavelength_mask = wavelength_mask.copy()  # Don't modify original
        wavelength_mask[:edge_zone] = False
        wavelength_mask[-edge_zone:] = False

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
        'edge_zone': edge_zone,
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
        decoded = decode_solution(solution, n_wavelengths, model_types, task_type)

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


def _compute_solution_r2(
    X: np.ndarray,
    y: np.ndarray,
    solution: np.ndarray,
    n_wavelengths: int,
    model_types: List[str],
    task_type: str,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Optional[float]:
    """
    Compute R2 for a single NSGA-II solution via cross-validation.

    Uses get_model() from models.py to ensure model construction matches
    Model Development tab exactly (same defaults, same parameters).

    Parameters
    ----------
    X : np.ndarray
        Input data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    solution : np.ndarray
        Chromosome encoding the solution
    n_wavelengths : int
        Total number of wavelengths
    model_types : list of str
        List of model types used in optimization
    task_type : str
        'regression' or 'classification'
    cv_folds : int
        Number of CV folds
    random_state : int
        Random state for reproducibility

    Returns
    -------
    r2 : float or None
        R2 score, or None if computation failed
    """
    if task_type != 'regression':
        return None  # R2 only for regression

    try:
        # Decode solution
        preproc_idx = int(solution[0])
        window_idx = int(solution[1])
        model_idx = int(solution[2])
        model_param = int(solution[3])
        lr_gene = int(solution[4])
        reg_alpha_gene = int(solution[5])
        reg_lambda_gene = int(solution[6])
        l1_gene = int(solution[7])
        subsample_gene = int(solution[8])
        colsample_gene = int(solution[9])
        min_samples_gene = int(solution[10])
        gamma_gene = int(solution[11])
        max_features_gene = int(solution[12])
        wavelength_mask = solution[13:].astype(bool)

        # Decode hyperparameter genes (including new regularization genes)
        hyperparams = _decode_hyperparameter_genes(
            lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
            subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene
        )

        # Apply preprocessing
        transform = _get_preprocessing_transform(preproc_idx, window_idx)
        if transform is not None:
            X_proc = transform(X)
        else:
            X_proc = X.copy()

        # Select wavelengths
        X_subset = X_proc[:, wavelength_mask]

        if X_subset.shape[1] == 0:
            return None

        # Get model type
        model_type = model_types[min(model_idx, len(model_types) - 1)]

        # For PLS, limit n_components to valid range and use scale=False to match Model Development
        if model_type == 'PLS':
            n_components = min(model_param + 1, X_subset.shape[1], X_subset.shape[0] - 1)
            n_components = max(1, n_components)
            # scale=False matches get_model() in models.py for consistent R² between NSGA and Model Development
            model = PLSRegression(n_components=n_components, scale=False)
        else:
            # Use _build_model for all other models with hyperparams
            model = _build_model(model_type, model_param, task_type, random_state, hyperparams)

        # Cross-validation for R2
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores = cross_val_score(model, X_subset, y, cv=cv, scoring='r2')

        return float(np.mean(scores))

    except Exception:
        return None


def _compute_display_rmse(
    X: np.ndarray,
    y: np.ndarray,
    solution: np.ndarray,
    n_wavelengths: int,
    model_types: List[str],
    task_type: str,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Optional[float]:
    """
    Compute RMSE using mean(sqrt(MSE)) formula to match Model Development display.

    NSGA optimization uses sqrt(mean(MSE)) for better Pareto trade-offs, but
    this function recomputes RMSE using the Model Development formula so
    that displayed values match between Results and Model Development tabs.

    Parameters
    ----------
    X : np.ndarray
        Input data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    solution : np.ndarray
        Chromosome encoding the solution
    n_wavelengths : int
        Total number of wavelengths
    model_types : list of str
        List of model types used in optimization
    task_type : str
        'regression' or 'classification'
    cv_folds : int
        Number of CV folds
    random_state : int
        Random state for reproducibility

    Returns
    -------
    rmse : float or None
        RMSE computed as mean(sqrt(MSE)), or None if computation failed
    """
    if task_type != 'regression':
        return None  # RMSE only for regression

    try:
        # Decode solution
        preproc_idx = int(solution[0])
        window_idx = int(solution[1])
        model_idx = int(solution[2])
        model_param = int(solution[3])
        lr_gene = int(solution[4])
        reg_alpha_gene = int(solution[5])
        reg_lambda_gene = int(solution[6])
        l1_gene = int(solution[7])
        subsample_gene = int(solution[8])
        colsample_gene = int(solution[9])
        min_samples_gene = int(solution[10])
        gamma_gene = int(solution[11])
        max_features_gene = int(solution[12])
        wavelength_mask = solution[13:].astype(bool)

        # Decode hyperparameter genes (including new regularization genes)
        hyperparams = _decode_hyperparameter_genes(
            lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
            subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene
        )

        # Apply preprocessing
        transform = _get_preprocessing_transform(preproc_idx, window_idx)
        if transform is not None:
            X_proc = transform(X)
        else:
            X_proc = X.copy()

        # Select wavelengths
        X_subset = X_proc[:, wavelength_mask]

        if X_subset.shape[1] == 0:
            return None

        # Get model type
        model_type = model_types[min(model_idx, len(model_types) - 1)]

        # For PLS, limit n_components to valid range and use scale=False to match Model Development
        if model_type == 'PLS':
            n_components = min(model_param + 1, X_subset.shape[1], X_subset.shape[0] - 1)
            n_components = max(1, n_components)
            model = PLSRegression(n_components=n_components, scale=False)
        else:
            # Use _build_model for all other models with hyperparams
            model = _build_model(model_type, model_param, task_type, random_state, hyperparams)

        # Cross-validation for RMSE
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores = cross_val_score(model, X_subset, y, cv=cv, scoring='neg_mean_squared_error')

        # Use mean(sqrt(MSE)) to match Model Development
        rmse_per_fold = np.sqrt(-scores)
        return float(np.mean(rmse_per_fold))

    except Exception:
        return None


def _indices_to_wavelength_str(indices: List[int], wavelengths: np.ndarray = None) -> str:
    """Convert wavelength indices to comma-separated wavelength string.

    Parameters
    ----------
    indices : list of int
        Indices into wavelengths array
    wavelengths : np.ndarray, optional
        Array of actual wavelength values

    Returns
    -------
    str
        Comma-separated wavelength values like "1500,1520,1540"
        or comma-separated indices if wavelengths not provided
    """
    if not indices:
        return 'N/A'
    if wavelengths is not None:
        # Convert indices to actual wavelength values
        selected_wl = [wavelengths[i] for i in indices if i < len(wavelengths)]
        return ','.join([f"{w:.0f}" for w in selected_wl])
    else:
        # Fallback to indices if wavelengths not available
        return ','.join([str(i) for i in indices])


def _compute_top_variables(
    X: np.ndarray,
    y: np.ndarray,
    decoded: Dict[str, Any],
    model_types: List[str],
    task_type: str,
    wavelengths: np.ndarray = None,
    top_n: int = 30,
    random_state: int = 42,
) -> str:
    """
    Compute top variables by feature importance (not index order).

    This matches Grid Search behavior which computes VIP scores, coefficients,
    or feature_importances_ and ranks variables by actual importance.

    Parameters
    ----------
    X : np.ndarray
        Input data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    decoded : Dict
        Decoded solution dictionary containing model info, preprocessing, wavelength_mask
    model_types : List[str]
        Available model types
    task_type : str
        'regression' or 'classification'
    wavelengths : np.ndarray, optional
        Array of actual wavelength values
    top_n : int, default 30
        Number of top variables to return
    random_state : int, default 42
        Random state for model training

    Returns
    -------
    str
        Comma-separated wavelength values ordered by importance (most important first)
    """
    try:
        selected_indices = decoded.get('selected_indices', [])
        if not selected_indices:
            return 'N/A'

        # Apply preprocessing
        preproc_idx = decoded.get('preproc_idx', 0)
        window_idx = decoded.get('window_idx', 6)
        transform = _get_preprocessing_transform(preproc_idx, window_idx)
        X_proc = transform(X) if transform else X.copy()

        # Select wavelengths
        wavelength_mask = decoded.get('wavelength_mask')
        if wavelength_mask is not None:
            X_subset = X_proc[:, wavelength_mask]
        else:
            X_subset = X_proc[:, selected_indices]

        # Build and train model
        model_type = decoded.get('model', 'PLS')
        model_param = decoded.get('model_param', {})
        model = _build_model(model_type, model_param, task_type, random_state)
        if model is None:
            # Fallback to first N indices
            return _indices_to_wavelength_str(selected_indices[:top_n], wavelengths)

        model.fit(X_subset, y)

        # Get feature importances using the same function as Grid Search
        importances = get_feature_importances(model, model_type)
        if importances is None or len(importances) == 0:
            # Fallback to first N indices
            return _indices_to_wavelength_str(selected_indices[:top_n], wavelengths)

        # Sort by importance descending
        n_to_select = min(top_n, len(importances), len(selected_indices))
        top_local_indices = np.argsort(importances, kind='stable')[-n_to_select:][::-1]

        # Map back to original wavelength indices
        top_original_indices = [selected_indices[i] for i in top_local_indices if i < len(selected_indices)]

        return _indices_to_wavelength_str(top_original_indices, wavelengths)

    except Exception:
        # Fallback to first N indices on any error
        selected_indices = decoded.get('selected_indices', [])
        return _indices_to_wavelength_str(selected_indices[:top_n], wavelengths) if selected_indices else 'N/A'


def convert_nsga2_to_v1_format(
    result: Dict[str, Any],
    n_wavelengths: int,
    task_type: str,
    folds: int = 5,
    excluded_count: int = 0,
    validation_count: int = 0,
    total_samples_original: int = None,
    wavelengths: np.ndarray = None,
    X: np.ndarray = None,
    y: np.ndarray = None,
    compute_r2: bool = True,
) -> pd.DataFrame:
    """
    Convert NSGA-II results to V1 results DataFrame format.

    This ensures compatibility with the existing V1 GUI results display.

    Parameters
    ----------
    wavelengths : np.ndarray, optional
        Array of actual wavelength values. If provided, all_vars and top_vars
        will contain wavelength values instead of indices.
    X : np.ndarray, optional
        Input data for R2 computation (n_samples, n_wavelengths)
    y : np.ndarray, optional
        Target values for R2 computation
    compute_r2 : bool, default True
        Whether to compute R2 for regression tasks (adds ~1s per solution)
    """
    if len(result['pareto_front']) == 0:
        return pd.DataFrame()

    model_types = result.get('model_types', MODEL_TYPES)
    n_samples = total_samples_original if total_samples_original else 0
    n_calibration = n_samples - excluded_count - validation_count if n_samples else 0

    rows = []
    for i, (objectives, solution) in enumerate(zip(
        result['pareto_front'],
        result['pareto_solutions']
    )):
        decoded = decode_solution(solution, n_wavelengths, model_types, task_type)

        # Parse preprocessing info for derivative/window details
        preproc = decoded['preprocessing']
        deriv_order = None
        window_size = None
        if 'deriv1' in preproc or 'deriv2' in preproc or 'deriv3' in preproc or 'deriv4' in preproc:
            # Extract derivative order from preprocessing name
            for d in ['deriv4', 'deriv3', 'deriv2', 'deriv1']:
                if d in preproc:
                    deriv_order = int(d[-1])
                    break
            # Window size is in decoded
            window_idx = decoded.get('window_idx', 6)  # Default to index 6 = window 17
            window_size = 5 + window_idx * 2  # Window sizes: 5, 7, 9, ..., 33

        row = {
            'Task': task_type,
            'Model': decoded['model'],
            'Preprocessing': decoded['preprocessing'],
            'Preprocess': decoded['preprocessing'],  # Alias for compatibility
            'Folds': folds,
            'N_Calibration': n_calibration,
            'N_Excluded': excluded_count,
            'N_Validation': validation_count,
            'Parameters': decoded['model_params'],
            'Params': decoded['model_params'],  # Alias for compatibility
            'Variables': f"nsga2_{decoded['n_wavelengths']}",
            'full_vars': n_wavelengths,  # Total wavelengths available
            'SubsetTag': 'nsga2',
            'Imbalance': 'none',
            'top_vars': _compute_top_variables(X, y, decoded, model_types, task_type, wavelengths, 30, 42) if (X is not None and y is not None and decoded['selected_indices']) else (_indices_to_wavelength_str(decoded['selected_indices'][:30], wavelengths) if decoded['selected_indices'] else 'N/A'),
            'all_vars': _indices_to_wavelength_str(decoded['selected_indices'], wavelengths) if decoded['selected_indices'] else 'N/A',
            'n_vars': decoded['n_wavelengths'],
            'Deriv': deriv_order,
            'Window': window_size,
            'Poly': 2 if deriv_order else None,  # Default polyorder for Savgol
            'LVs': int(solution[3]) + 1 if decoded['model'] == 'PLS' else None,  # n_components for PLS
            'Complexity': objectives[2],
            'Is_Knee': i == result['knee_idx'],
        }

        if task_type == 'regression':
            # Compute display RMSE using mean(sqrt(MSE)) to match Model Development
            # NSGA optimization uses sqrt(mean(MSE)) for better Pareto trade-offs,
            # but displayed values should match what Model Development shows
            if X is not None and y is not None:
                display_rmse = _compute_display_rmse(
                    X, y, solution, n_wavelengths, model_types, task_type, folds, 42
                )
                row['RMSE'] = display_rmse if display_rmse is not None else objectives[0]
            else:
                row['RMSE'] = objectives[0]  # Fallback to optimization RMSE

            # Compute R2 if X and y are provided
            if compute_r2 and X is not None and y is not None:
                r2 = _compute_solution_r2(
                    X, y, solution, n_wavelengths, model_types, task_type, folds, 42
                )
                row['R2'] = r2
            else:
                row['R2'] = None
            row['CompositeScore'] = row['RMSE']  # Use display RMSE as composite
        else:
            row['Accuracy'] = 1.0 - objectives[0]
            row['ROC_AUC'] = None
            row['CompositeScore'] = 1.0 - objectives[0]  # Use accuracy as composite

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort and rank
    if task_type == 'regression':
        df = df.sort_values('RMSE', ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    df['Rank'] = range(1, len(df) + 1)

    return df
