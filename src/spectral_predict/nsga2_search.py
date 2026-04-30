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

import ast
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold

# Import early stopping CV utilities
from .cv_utils import (
    cross_val_score_with_early_stopping,
    cross_val_predict_with_early_stopping,
    is_boosting_model,
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, log_loss
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeClassifier, Lasso, ElasticNet
from .models import PLSTransformer  # Wrapper that ensures 2D output for PLS-DA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder

# pymoo imports (required dependency)
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# V1 imports - use local modules
from .preprocess import SNV, SavgolDerivative
from .models import get_feature_importances
from .variable_selection import cars_selection
from .scoring import compute_specificity, lins_ccc

# Imbalance handling imports
from imblearn.pipeline import Pipeline as ImbPipeline
from .imbalance import build_imbalance_transformer, validate_classification_config

# All model libraries are required dependencies
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from .neural_boosted import NeuralBoostedRegressor, NeuralBoostedClassifier

import logging

logger = logging.getLogger(__name__)




# =============================================================================
# Preprocessing options (same encoding as ga_preprocessing.py)
# =============================================================================

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

WINDOW_SIZES = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 37, 43, 51]

MODEL_TYPES = [
    'PLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest',
    'LightGBM', 'XGBoost', 'CatBoost', 'SVR', 'MLP', 'NeuralBoosted'
]  # All supported models for NSGA-II

# Hyperparameter genes (indices 4-12) relevance per model type
# Genes: 4=lr, 5=reg_alpha, 6=reg_lambda, 7=l1_ratio, 8=subsample,
#        9=colsample, 10=min_samples, 11=gamma, 12=max_features
MODEL_ACTIVE_GENES = {
    'PLS':          [],                           # Uses only n_components (gene 3)
    'Ridge':        [],                           # Uses only alpha (gene 3)
    'Lasso':        [],                           # Uses only alpha (gene 3)
    'ElasticNet':   [7],                          # l1_ratio
    'RandomForest': [10, 12],                     # min_samples, max_features
    'LightGBM':     [4, 5, 6, 8, 9, 10],          # lr, reg_alpha, reg_lambda, subsample, colsample, min_samples
    'XGBoost':      [4, 5, 6, 8, 9, 10, 11],      # + gamma
    'CatBoost':     [4, 8, 11],                   # lr, subsample, gamma (as l2_leaf_reg)
    'SVR':          [11],                         # gamma (kernel parameter)
    'MLP':          [4],                          # learning_rate
    'NeuralBoosted': [4, 6],                      # lr, reg_lambda
}


def _needs_resampling_pipeline(imbalance_method: Optional[str], task_type: str) -> bool:
    """Determine if the imbalance method requires imblearn Pipeline.

    Parameters
    ----------
    imbalance_method : str or None
        The imbalance handling method
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    bool
        True if the method requires imblearn Pipeline (uses fit_resample)
    """
    if imbalance_method is None:
        return False
    if imbalance_method == 'class_weight':
        return False

    if task_type == 'classification':
        resampling_methods = {'smote', 'adasyn', 'borderline_smote',
                              'random_undersampler', 'tomek_links', 'smote_tomek', 'smote_enn'}
        return imbalance_method.lower().replace('-', '_') in resampling_methods
    elif task_type == 'regression':
        resampling_methods = {'undersample', 'oversample', 'smogn', 'smotetomek'}
        return imbalance_method.lower() in resampling_methods
    return False


class SmartMutation(Mutation):
    """
    Custom mutation with two key features:
    1. Only mutates hyperparameter genes relevant to the model type
    2. Importance-biased wavelength mutation (optional)

    Gene layout:
    - 0: preprocessing type (always mutate)
    - 1: window size (always mutate)
    - 2: model type (always mutate)
    - 3: model param (always mutate)
    - 4-12: hyperparameters (only mutate if relevant to model)
    - 13+: wavelengths (biased mutation based on importance if available)

    Importance-biased mutation:
    - High importance wavelengths: protected from being dropped
    - Low importance wavelengths: easier to drop
    """

    def __init__(self, prob: float = 0.1, eta: float = 20, importance_scores: Optional[np.ndarray] = None,
                 sparsity_bias: float = 2.0):
        super().__init__()
        self.prob = prob
        self.eta = eta
        self.pm = PM(prob=prob, eta=eta, vtype=float, repair=RoundingRepair())
        self.importance_scores = importance_scores
        self.sparsity_bias = sparsity_bias  # Drop is N times more likely than add
        self._rng = np.random.default_rng()

    def set_importance_scores(self, importance: np.ndarray):
        """Update the wavelength importance scores for biased mutation."""
        self.importance_scores = importance

    def _do(self, problem, X, **kwargs):
        # Apply standard PM mutation to control genes (0-12) first
        X_mutated = self.pm._do(problem, X.copy(), **kwargs)

        # For each individual, restore hyperparameter genes that aren't relevant
        for i in range(len(X)):
            model_idx = int(X[i, 2])
            model_type = MODEL_TYPES[min(model_idx, len(MODEL_TYPES) - 1)]
            active_genes = MODEL_ACTIVE_GENES.get(model_type, [])

            # For genes 4-12, restore original if not active for this model
            for gene_idx in range(4, 13):
                if gene_idx not in active_genes:
                    X_mutated[i, gene_idx] = X[i, gene_idx]

        # Apply importance-biased mutation to wavelength genes if available
        if self.importance_scores is not None:
            X_mutated = self._biased_wavelength_mutation(X, X_mutated)

        return X_mutated

    def _biased_wavelength_mutation(self, X_orig: np.ndarray, X_mutated: np.ndarray) -> np.ndarray:
        """
        Apply importance-biased mutation with sparsity pressure to wavelength genes.

        Logic:
        - If wavelength selected (1): P(drop) = base_prob * sparsity_bias * (1 - importance)
          High importance → low drop chance; sparsity_bias increases drop pressure
        - If wavelength not selected (0): P(add) = base_prob * (1/sparsity_bias) * importance
          High importance → high add chance; sparsity_bias reduces add likelihood

        With sparsity_bias=2.0, dropping is 2x more likely than adding (encourages smaller subsets).

        IMPORTANT: First restores original wavelength values (undoes PM mutation),
        then applies biased mutation. This ensures only biased mutation controls
        wavelength genes, not the random PM mutation.
        """
        n_wl = len(self.importance_scores)

        for i in range(len(X_orig)):
            for j in range(n_wl):
                gene_idx = 13 + j
                current = int(X_orig[i, gene_idx])
                imp = self.importance_scores[j]

                # FIRST: Restore original value (undo PM mutation for wavelengths)
                X_mutated[i, gene_idx] = current

                # THEN: Apply biased mutation with sparsity pressure
                if current == 1:
                    # Currently selected: probability of dropping
                    # High importance → low drop probability
                    # sparsity_bias multiplier increases drop pressure
                    drop_prob = self.prob * self.sparsity_bias * (1.0 - imp)
                    if self._rng.random() < drop_prob:
                        X_mutated[i, gene_idx] = 0
                else:
                    # Currently not selected: probability of adding
                    # High importance → high add probability
                    # 1/sparsity_bias reduces add likelihood (encourages sparsity)
                    add_prob = self.prob * (1.0 / self.sparsity_bias) * imp
                    if self._rng.random() < add_prob:
                        X_mutated[i, gene_idx] = 1

        return X_mutated


class ImportanceTracker:
    """
    Tracks best preprocessing and manages adaptive importance score computation.

    Updates importance scores every N generations based on the current best
    preprocessing configuration from the Pareto front.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'regression',
        update_interval: int = 10,
        random_state: int = 42,
    ):
        self.X = X
        self.y = y
        self.task_type = task_type
        self.update_interval = update_interval
        self.random_state = random_state

        self.n_wavelengths = X.shape[1]
        # Start with uniform importance
        self.importance_scores = np.ones(self.n_wavelengths) / self.n_wavelengths
        self.last_update_gen = 0

        # Initial best guesses (raw preprocessing, window 17, PLS model)
        self.best_preproc_idx = 0
        self.best_window_idx = 6
        self.best_model_type = 'PLS'

    def update_best_from_population(self, population: np.ndarray, objectives: np.ndarray):
        """
        Update best preprocessing configuration from current population.

        Finds the solution with lowest error and extracts its preprocessing settings.
        """
        if len(objectives) == 0:
            return

        # Find solution with minimum error (first objective)
        best_idx = np.argmin(objectives[:, 0])
        best_chromosome = population[best_idx]

        self.best_preproc_idx = int(best_chromosome[0])
        self.best_window_idx = int(best_chromosome[1])
        model_idx = int(best_chromosome[2])
        self.best_model_type = MODEL_TYPES[min(model_idx, len(MODEL_TYPES) - 1)]

    def should_update(self, generation: int) -> bool:
        """Check if importance scores should be recomputed."""
        return (generation - self.last_update_gen) >= self.update_interval

    def compute_importance(self) -> np.ndarray:
        """Compute importance scores using current best preprocessing configuration."""
        self.importance_scores = _compute_wavelength_importance(
            X=self.X,
            y=self.y,
            preproc_idx=self.best_preproc_idx,
            window_idx=self.best_window_idx,
            model_type=self.best_model_type,
            task_type=self.task_type,
            random_state=self.random_state,
        )
        return self.importance_scores


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

    Uses CARS/CARS-Tree importance scores for importance-weighted wavelength selection:
    - Seeded solutions start with ~target_n_wavelengths selected (based on importance)
    - Random solutions get importance-weighted random selection
    - This encourages NSGA-II to explore compact, high-quality subsets from the start

    Parameters
    ----------
    importance_scores : np.ndarray, optional
        CARS/CARS-Tree importance scores (higher = more important).
        If None, uses uniform importance (all wavelengths equally likely).
    target_n_wavelengths : int, default 250
        Target number of wavelengths for seeded solutions.
    """

    def __init__(self, n_wavelengths: int, model_types: List[str], n_preproc: int = 10, n_window: int = 15,
                 importance_scores: Optional[np.ndarray] = None, target_n_wavelengths: int = 250):
        super().__init__()
        self.n_wavelengths = n_wavelengths
        self.model_types = model_types
        self.n_preproc = n_preproc
        self.n_window = n_window
        self.importance_scores = importance_scores
        self.target_n_wavelengths = min(target_n_wavelengths, n_wavelengths)
        self._rng = np.random.default_rng()

    def _do(self, problem, n_samples, **kwargs):
        """Generate initial population with importance-weighted wavelength selection."""
        # Variable structure: [preproc_idx, window_idx, model_idx, model_param, lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
        #                      subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene, wl_0, wl_1, ..., wl_n]
        n_vars = 13 + self.n_wavelengths

        # Initialize with random values
        X = np.random.randint(
            problem.xl,
            problem.xu + 1,
            size=(n_samples, n_vars)
        )

        # Compute selection probabilities from importance scores
        if self.importance_scores is not None:
            # Normalize to probabilities (higher importance = higher selection probability)
            probs = self.importance_scores.copy()
            probs = np.maximum(probs, 1e-10)  # Ensure no zeros
            probs = probs / probs.sum()
        else:
            # Uniform probability if no importance scores
            probs = np.ones(self.n_wavelengths) / self.n_wavelengths

        # Seed with diverse preprocessing types including derivatives (commonly best)
        # PREPROC_TYPES indices: 0=raw, 1=snv, 2=deriv1, 3=deriv2, 4=deriv3, 5=deriv4,
        #                        6=snv_deriv1, 7=snv_deriv2, 8=snv_deriv3, 9=snv_deriv4,
        #                        10=deriv1_snv, 11=deriv2_snv, 12=deriv3_snv, 13=deriv4_snv
        # Prioritize snv_deriv combinations (commonly best), then plain derivs, then raw/snv
        SEED_PREPROC = [6, 7, 8, 9, 2, 3, 4, 5, 10, 11, 0, 1]  # 12 preprocessing types

        # Default hyperparameters (Grid Search defaults)
        DEFAULT_HYPERPARAMS = {
            'lr': 10,        # lr ≈ 0.1
            'reg_alpha': 12, # reg_alpha ≈ 0.07
            'reg_lambda': 14,# reg_lambda = 1.0
            'l1_ratio': 7,   # l1_ratio = 0.5
            'subsample': 10, # subsample ≈ 0.86
            'colsample': 10, # colsample ≈ 0.86
            'min_samples': 2,# min_samples ≈ 5
            'gamma': 0,      # gamma = 0
            'max_features': 7# max_features = 0.3
        }

        # Seed solutions: one per (preprocessing × model) combination
        solution_idx = 0
        for preproc_idx in SEED_PREPROC:
            if solution_idx >= n_samples // 2:  # Don't seed more than half the population
                break
            for model_idx in range(len(self.model_types)):
                if solution_idx >= n_samples // 2:
                    break

                X[solution_idx, 0] = preproc_idx
                X[solution_idx, 1] = 6   # window index 6 = window 17 (good default for derivatives)
                X[solution_idx, 2] = model_idx
                X[solution_idx, 3] = 7   # middle model_param value
                X[solution_idx, 4] = DEFAULT_HYPERPARAMS['lr']
                X[solution_idx, 5] = DEFAULT_HYPERPARAMS['reg_alpha']
                X[solution_idx, 6] = DEFAULT_HYPERPARAMS['reg_lambda']
                X[solution_idx, 7] = DEFAULT_HYPERPARAMS['l1_ratio']
                X[solution_idx, 8] = DEFAULT_HYPERPARAMS['subsample']
                X[solution_idx, 9] = DEFAULT_HYPERPARAMS['colsample']
                X[solution_idx, 10] = DEFAULT_HYPERPARAMS['min_samples']
                X[solution_idx, 11] = DEFAULT_HYPERPARAMS['gamma']
                X[solution_idx, 12] = DEFAULT_HYPERPARAMS['max_features']

                # Use importance-weighted selection instead of all wavelengths
                X[solution_idx, 13:] = 0  # Start with none selected
                selected_indices = self._rng.choice(
                    self.n_wavelengths,
                    size=self.target_n_wavelengths,
                    replace=False,
                    p=probs
                )
                X[solution_idx, 13 + selected_indices] = 1

                solution_idx += 1

        # For remaining (random) solutions, also use importance-weighted wavelength selection
        for i in range(solution_idx, n_samples):
            # Random number of wavelengths between target/2 and target*1.5
            n_wl_to_select = self._rng.integers(
                max(10, self.target_n_wavelengths // 2),
                min(self.n_wavelengths, int(self.target_n_wavelengths * 1.5)) + 1
            )
            X[i, 13:] = 0  # Start with none selected
            selected_indices = self._rng.choice(
                self.n_wavelengths,
                size=n_wl_to_select,
                replace=False,
                p=probs
            )
            X[i, 13 + selected_indices] = 1

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


# Tree-based models that should use LightGBM importance
TREE_MODELS = {'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'}


def _compute_wavelength_importance(
    X: np.ndarray,
    y: np.ndarray,
    preproc_idx: int,
    window_idx: int,
    model_type: str,
    task_type: str = 'regression',
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute normalized wavelength importance scores for biased mutation.

    For tree-based models (LightGBM, XGBoost, CatBoost, RandomForest):
        Uses LightGBM feature_importances_ (fast, good proxy for all tree models)

    For linear models (PLS, Ridge, Lasso, ElasticNet, SVR, MLP, NeuralBoosted):
        Uses PLS VIP scores (captures linear relationships)

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    preproc_idx : int
        Index into PREPROC_TYPES for current best preprocessing
    window_idx : int
        Index into WINDOW_SIZES for current best preprocessing
    model_type : str
        Model type name from MODEL_TYPES
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random state for reproducibility

    Returns
    -------
    importance : np.ndarray
        Normalized importance scores in [0, 1] for each wavelength
        Higher values = more important wavelengths
    """
    n_wavelengths = X.shape[1]

    # Apply preprocessing
    try:
        transform = _get_preprocessing_transform(preproc_idx, window_idx)
        X_proc = transform(X) if transform is not None else X.copy()
    except Exception:
        X_proc = X.copy()

    # Handle edge zones - zero out importance for unreliable edge wavelengths
    edge_zone = _get_edge_zone_size(preproc_idx, window_idx)

    try:
        if model_type in TREE_MODELS:
            # Use LightGBM feature importance for tree-based models
            from lightgbm import LGBMRegressor, LGBMClassifier
            if task_type == 'regression':
                model = LGBMRegressor(
                    n_estimators=50, max_depth=5, random_state=random_state,
                    verbosity=-1, n_jobs=1
                )
            else:
                model = LGBMClassifier(
                    n_estimators=50, max_depth=5, random_state=random_state,
                    verbosity=-1, n_jobs=1
                )
            model.fit(X_proc, y)
            importance = model.feature_importances_.astype(float)
        else:
            # Use PLS VIP scores for linear models
            from sklearn.cross_decomposition import PLSRegression
            n_comp = min(5, X_proc.shape[1] // 2, X_proc.shape[0] - 1)
            n_comp = max(1, n_comp)

            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_proc, y)

            # Compute VIP scores per Wold (2001) canonical formula:
            # SSY_a = q_a^2 * (T_a' T_a) where q_a is the y-loading for component a
            W = np.asarray(pls.x_weights_)   # (n_features, n_components)
            T = np.asarray(pls.x_scores_)    # (n_samples, n_components)
            Q = np.asarray(pls.y_loadings_)  # sklearn shape: (n_targets, n_components)
            q = Q if Q.ndim == 1 else Q[0, :]

            ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)
            ssy_total = float(np.sum(ssy_comp))

            if ssy_total <= 0.0:
                importance = np.zeros(n_wavelengths, dtype=float)
            else:
                n_features = W.shape[0]
                col_norm_sq = np.sum(W ** 2, axis=0)
                col_norm_sq = np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)
                w_norm_sq = (W ** 2) / col_norm_sq
                importance = np.sqrt(n_features * (w_norm_sq @ ssy_comp) / ssy_total)

    except Exception:
        # Fallback to uniform importance if computation fails
        importance = np.ones(n_wavelengths)

    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()

    # Zero out edge zones (these wavelengths can't be used anyway)
    if edge_zone > 0:
        importance[:edge_zone] = 0.0
        importance[-edge_zone:] = 0.0

    return importance


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


def _get_constrained_pls_components(model_param: int, n_features: int, n_samples: int = None) -> int:
    """Get constrained n_components for PLS model.

    Constraints:
    - n_components <= 15 (practical max in chemometrics)
    - n_components <= n_features - 1 (ensures dimensionality reduction, prevents OLS equivalence)
    - n_components <= n_samples - 1 (mathematical requirement)

    Parameters
    ----------
    model_param : int
        Raw gene value (0-14)
    n_features : int
        Number of selected features/wavelengths
    n_samples : int, optional
        Number of samples. If None, only n_features constraint is applied.

    Returns
    -------
    int
        Constrained n_components value (>= 1)
    """
    # Practical cap: 15 is rarely exceeded in chemometrics
    # n_features - 1 ensures PLS doesn't degenerate to OLS
    n_components = min(model_param + 1, 15, n_features - 1)

    # Apply sample constraint if n_samples is provided
    if n_samples is not None and n_samples > 1:
        n_components = min(n_components, n_samples - 1)

    # Ensure at least 1 component (handles edge case where n_features <= 1)
    return max(1, n_components)


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
    if model_type in ('PLS', 'PLS-DA'):
        n_components = model_param + 1  # 1-15
        if task_type == 'classification':
            # PLS-DA: PLSTransformer + StandardScaler + LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression

            # Extract LogisticRegression parameters from hyperparams (prefixed with lr_)
            lr_C = hyperparams.get('lr_C', 1.0) if hyperparams else 1.0
            lr_solver = hyperparams.get('lr_solver', 'lbfgs') if hyperparams else 'lbfgs'
            lr_max_iter = hyperparams.get('lr_max_iter', 1000) if hyperparams else 1000

            pls_transformer = PLSTransformer(n_components=n_components, scale=False)
            return Pipeline([
                ('pls', pls_transformer),
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=random_state))
            ])
        else:
            # Regression: use PLSRegression directly
            return PLSRegression(n_components=n_components, scale=False)

    elif model_type == 'Ridge':
        # Exponential alpha scale: 1e-4 to 1e3
        alpha = 10 ** (model_param / 3 - 2)  # param 0-14 -> alpha 0.01 to 1000
        if task_type == 'regression':
            return Ridge(alpha=alpha, random_state=random_state)
        else:
            return RidgeClassifier(alpha=alpha, random_state=random_state)

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
        imbalance_method: Optional[str] = None,
        imbalance_params: Optional[Dict[str, Any]] = None,
        # PLS-DA LogisticRegression parameters (Stage 2)
        plsda_lr_C: float = 1.0,
        plsda_lr_solver: str = 'lbfgs',
        plsda_lr_max_iter: int = 1000,
        # Early stopping for boosting models
        early_stopping_rounds: Optional[int] = 40,
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
        imbalance_method : str, optional
            Imbalance handling method (e.g., 'smote', 'class_weight')
        imbalance_params : dict, optional
            Parameters for the imbalance method
        plsda_lr_C : float, default=1.0
            LogisticRegression inverse regularization strength for PLS-DA
        plsda_lr_solver : str, default='lbfgs'
            LogisticRegression solver for PLS-DA
        plsda_lr_max_iter : int, default=1000
            LogisticRegression maximum iterations for PLS-DA
        early_stopping_rounds : int, optional, default=50
            Number of rounds without improvement before stopping for boosting models.
            Set to None or 0 to disable.
        """
        self.X = X
        self.y = y
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.min_wavelengths = min_wavelengths
        self.random_state = random_state
        self.n_wavelengths = X.shape[1]

        # Imbalance handling settings
        self.imbalance_method = imbalance_method
        self.imbalance_params = imbalance_params if imbalance_params is not None else {}

        # PLS-DA LogisticRegression parameters (Stage 2)
        self.plsda_lr_C = plsda_lr_C
        self.plsda_lr_solver = plsda_lr_solver
        self.plsda_lr_max_iter = plsda_lr_max_iter

        # Early stopping for boosting models
        self.early_stopping_rounds = early_stopping_rounds

        # Use user-specified models or defaults
        self.model_types = models if models is not None else MODEL_TYPES

        # Encode labels for classification
        # ALWAYS encode classification labels to ensure consistent 0..n-1 range
        # even for numeric labels (e.g., [1,2,3] -> [0,1,2])
        self.label_encoder = None
        if task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y_arr = np.asarray(y)
            if y_arr.dtype == object:
                y_arr = y_arr.astype(str)
            self.y = self.label_encoder.fit_transform(y_arr)

        # Fitness cache
        self.cache_enabled = cache_enabled
        self._cache = {}
        self._eval_count = 0
        # Track ALL evaluations for bias=0 (minimum error) selection
        self._all_solutions = []    # All chromosomes evaluated
        self._all_objectives = []   # Their objective values [error, wavelengths, complexity]
        self._failure_counts = {}   # Track failures per model type for debugging
        self._model_eval_counts = {}  # Track successful evaluations per model type

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

            # Objective 2: Wavelength count (quadratic penalty for large counts)
            # Using (n/total)^2 so 800 wavelengths is penalized 4x more than 400
            # This encourages NSGA-II to find more compact solutions
            F[i, 1] = (n_selected / self.n_wavelengths) ** 2

            # Objective 3: Model complexity (normalized to 0-1)
            complexity = self._compute_complexity(model_idx, model_param, preproc_idx)
            F[i, 2] = complexity

            self._eval_count += 1

            # Track this evaluation for bias=0 selection
            self._all_solutions.append(X[i].copy())
            self._all_objectives.append(F[i].copy())

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

            # Track model evaluation attempts
            if model_type not in self._model_eval_counts:
                self._model_eval_counts[model_type] = 0
            self._model_eval_counts[model_type] += 1

            # Decode hyperparameter genes (including new regularization genes)
            hyperparams = _decode_hyperparameter_genes(
                lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
                subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene
            )

            # Special handling for PLS - limit components and use scale=False
            # scale=False matches get_model() in models.py for consistency with Model Development
            # Practical cap: 15 is rarely exceeded in chemometrics
            # n_features - 1 ensures PLS doesn't degenerate to OLS
            # CRITICAL: Use CV fold training size, not full dataset size
            # sklearn PLS requires n_components <= min(n_features, n_samples_in_training_fold) - 1
            if model_type in ('PLS', 'PLS-DA'):
                min_train_samples = X_subset.shape[0] * (self.cv_folds - 1) // self.cv_folds
                n_components = min(model_param + 1, 15, X_subset.shape[1] - 1, min_train_samples - 1)
                n_components = max(1, n_components)
                model = PLSRegression(n_components=n_components, scale=False)
            else:
                model = _build_model(model_type, model_param, self.task_type, self.random_state, hyperparams)

            if model is None:
                return 1e10

            # Scale-sensitive models need StandardScaler (matches search.py behavior)
            # For classification: PLS needs StandardScaler + LogisticRegression wrapper
            # For both tasks: SVC/SVR, MLP, NeuralBoosted need StandardScaler wrapper
            SCALE_SENSITIVE_MODELS = {'SVR', 'MLP', 'NeuralBoosted', 'Ridge', 'Lasso', 'ElasticNet'}

            # Build pipeline steps with imbalance handling support
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            pipe_steps = []

            # Step 1: Add imbalance handling (must be first for fit_resample)
            if self.imbalance_method is not None and self.imbalance_method != 'class_weight':
                imbalance_transformer = build_imbalance_transformer(
                    method=self.imbalance_method,
                    task_type=self.task_type,
                    random_state=self.random_state,
                    **self.imbalance_params
                )
                pipe_steps.append(('imbalance', imbalance_transformer))

            # Step 2: Handle class_weight for models that support it
            if self.imbalance_method == 'class_weight' and self.task_type == 'classification':
                if hasattr(model, 'class_weight'):
                    try:
                        model.set_params(class_weight='balanced')
                    except Exception:
                        pass

            # Step 3: Build model pipeline based on model type
            if self.task_type == 'classification' and model_type in ('PLS', 'PLS-DA'):
                # PLS-DA: PLSTransformer (2D output) + StandardScaler + LogisticRegression
                # PLSTransformer ensures transform() returns 2D arrays (fixes sklearn 3D output bug)
                from sklearn.linear_model import LogisticRegression
                # CRITICAL: Use CV fold training size, not full dataset size
                min_train_samples = X_subset.shape[0] * (self.cv_folds - 1) // self.cv_folds
                n_components = min(model_param + 1, 15, X_subset.shape[1] - 1, min_train_samples - 1)
                n_components = max(1, n_components)
                pls_transformer = PLSTransformer(n_components=n_components, scale=False)
                pipe_steps.append(('pls', pls_transformer))
                pipe_steps.append(('scaler', StandardScaler()))  # Scale PLS scores for LogisticRegression
                # Use configurable LogisticRegression parameters (Stage 2 of PLS-DA)
                lr = LogisticRegression(
                    C=self.plsda_lr_C,
                    solver=self.plsda_lr_solver,
                    max_iter=self.plsda_lr_max_iter,
                    random_state=self.random_state
                )
                # Apply class_weight to LogisticRegression if specified
                if self.imbalance_method == 'class_weight':
                    lr.set_params(class_weight='balanced')
                pipe_steps.append(('lr', lr))
            elif model_type in SCALE_SENSITIVE_MODELS:
                # Scale-sensitive models: StandardScaler + Model (matches search.py lines 3427-3429)
                pipe_steps.append(('scaler', StandardScaler()))
                pipe_steps.append(('model', model))
            else:
                # Other models don't need scaling
                pipe_steps.append(('model', model))

            # Step 4: Create pipeline with correct class (ImbPipeline for resampling methods)
            needs_resampling = _needs_resampling_pipeline(self.imbalance_method, self.task_type)
            if needs_resampling:
                pipeline_model = ImbPipeline(pipe_steps)
            else:
                pipeline_model = Pipeline(pipe_steps)

            # Cross-validation
            # Use early stopping for boosting models (XGBoost, LightGBM, CatBoost)
            use_early_stopping = (
                self.early_stopping_rounds is not None and
                self.early_stopping_rounds > 0 and
                is_boosting_model(model)  # Check the underlying model, not pipeline
            )

            if self.task_type == 'regression':
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if use_early_stopping:
                        scores = cross_val_score_with_early_stopping(
                            pipeline_model, X_subset, self.y, cv=cv,
                            scoring='neg_mean_squared_error',
                            early_stopping_rounds=self.early_stopping_rounds
                        )
                    else:
                        scores = cross_val_score(
                            pipeline_model, X_subset, self.y, cv=cv, scoring='neg_mean_squared_error'
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
                    if use_early_stopping:
                        scores = cross_val_score_with_early_stopping(
                            pipeline_model, X_subset, self.y, cv=cv,
                            scoring='accuracy',
                            early_stopping_rounds=self.early_stopping_rounds
                        )
                    else:
                        scores = cross_val_score(
                            pipeline_model, X_subset, self.y, cv=cv, scoring='accuracy'
                        )
                # Return 1 - accuracy (to minimize)
                return 1.0 - np.mean(scores)

        except Exception as e:
            # Track and log model failures for debugging
            model_type = self.model_types[min(model_idx, len(self.model_types) - 1)]

            if model_type not in self._failure_counts:
                self._failure_counts[model_type] = 0
            self._failure_counts[model_type] += 1

            # Warn on first failure of each model type (visible by default)
            if self._failure_counts[model_type] == 1:
                logger.warning(
                    f"NSGA-II: First failure for {model_type}: {type(e).__name__}: {e}"
                )

            # Debug level for subsequent failures
            logger.debug(
                f"NSGA-II model failed: model={model_type}, preproc={preproc_idx}, "
                f"window={window_idx}, n_wavelengths={np.sum(wavelength_mask)}, "
                f"error={type(e).__name__}: {e}"
            )

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
        if model_type in ('PLS', 'PLS-DA'):
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

def find_knee_point(pareto_front: np.ndarray, selection_bias: float = 2.0) -> int:
    """
    Find the knee point in a Pareto front using the maximum curvature method.

    The knee point represents the "best compromise" solution where
    improving one objective significantly worsens others.

    Parameters
    ----------
    pareto_front : ndarray, shape (n_solutions, n_objectives)
        Pareto front objective values (all minimized)
    selection_bias : float, default 2.0
        Selection bias parameter:
        - 0.0: Select minimum error solution (best R²)
        - 2.0: Select knee point (default behavior)
        - 1.0: Weighted compromise between min-error and knee point
        Values are clamped to [0, 2] range.

    Returns
    -------
    knee_idx : int
        Index of the knee point solution
    """
    # Clamp selection_bias to [0, 2] range
    selection_bias = np.clip(selection_bias, 0.0, 2.0)

    # Handle minimum error selection (bias <= 0) - works for any size front
    if selection_bias <= 0.0:
        # Return solution with minimum error (first objective)
        return int(np.argmin(pareto_front[:, 0]))

    # For knee point calculation, need at least 3 solutions
    if len(pareto_front) <= 2:
        return 0

    # Normalize objectives to [0, 1]
    pf_min = pareto_front.min(axis=0)
    pf_max = pareto_front.max(axis=0)
    pf_range = pf_max - pf_min
    pf_range[pf_range == 0] = 1  # Avoid division by zero

    pf_norm = (pareto_front - pf_min) / pf_range

    # Compute knee point scores (distance to line or ideal point)
    if pareto_front.shape[1] == 2:
        # For 2D: find maximum perpendicular distance to line from first to last
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

        # Convert to array and map back to original order
        distances_array = np.array(distances)
        knee_scores = np.zeros(len(pareto_front))
        for i, orig_idx in enumerate(sort_idx):
            knee_scores[orig_idx] = distances_array[i]

    else:
        # For 3D+: find point closest to ideal point (utopia point)
        # The ideal point is the minimum of each objective (impossible to achieve all at once)
        ideal = pf_norm.min(axis=0)

        # Distance to ideal point (invert so larger is better for knee point)
        distances = np.sqrt(np.sum((pf_norm - ideal) ** 2, axis=1))
        # Invert: smaller distance to ideal = larger knee score
        knee_scores = np.max(distances) - distances

    # Handle full knee point selection (bias >= 2)
    if selection_bias >= 2.0:
        # Return original knee point logic
        return int(np.argmax(knee_scores))

    # Weighted selection (0 < bias < 2)
    # Compute error ranks (lower error = lower rank = better)
    error_values = pareto_front[:, 0]
    error_ranks = np.argsort(np.argsort(error_values))  # Ranks from 0 to n-1

    # Normalize both to [0, 1]
    n = len(pareto_front)
    error_ranks_norm = error_ranks / max(n - 1, 1)

    # Normalize knee scores to [0, 1]
    knee_min = knee_scores.min()
    knee_max = knee_scores.max()
    knee_range = knee_max - knee_min
    if knee_range > 0:
        knee_scores_norm = (knee_scores - knee_min) / knee_range
    else:
        knee_scores_norm = np.zeros(len(knee_scores))

    # Combine: weight = bias / 2.0
    # bias=0 -> weight=0 -> only error_ranks
    # bias=1 -> weight=0.5 -> equal weighting
    # bias=2 -> weight=1.0 -> only knee_scores
    weight = selection_bias / 2.0
    # Lower combined score is better (minimization)
    combined = (1 - weight) * error_ranks_norm + weight * (1 - knee_scores_norm)

    return int(np.argmin(combined))


# =============================================================================
# Main NSGA-II Function
# =============================================================================

def run_nsga2_search(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    population_size: int = 60,
    n_generations: int = 120,
    cv_folds: int = 5,
    min_wavelengths: int = 10,
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    models: Optional[List[str]] = None,
    controller=None,
    selection_bias: float = 0.0,
    use_guidance: bool = True,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: Optional[int] = 40,
    cv_strategy: str = 'kfold',
    cv_n_repeats: int = 5,
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
    selection_bias : float, default 2.0
        Selection bias for knee point selection:
        - 0.0: Select minimum error solution (best R²)
        - 2.0: Select knee point (default behavior)
        - 1.0: Weighted compromise between min-error and knee point
        Values are clamped to [0, 2] range.
    use_guidance : bool, default True
        If True, use CARS-Tree importance for guided wavelength selection (SeededWavelengthSampling + SmartMutation).
        If False, use standard NSGA-II (IntegerRandomSampling + PM mutation).
    imbalance_method : str, optional
        Imbalance handling method (e.g., 'smote', 'class_weight')
    imbalance_params : dict, optional
        Parameters for the imbalance method

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
    # CV strategy fallback for NSGA-II (LOO/Repeated K-Fold not yet supported)
    if cv_strategy not in ('kfold', None):
        logger.warning(
            "NSGA-II search does not yet support %s CV strategy; falling back to K-fold for this run.",
            cv_strategy,
        )
        print(f"Warning: NSGA-II search does not support {cv_strategy} CV; falling back to K-fold.")
        cv_strategy = 'kfold'

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
        if imbalance_method:
            print(f"  Imbalance handling: {imbalance_method}")

    # Guard against None params
    if imbalance_params is None:
        imbalance_params = {}

    # Substitute regression sample weighting methods that don't work with cross_val_score
    # These methods (binning, rare_boost, balanced) require manual sample_weight extraction
    # which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
    UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
    if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
        original_method = imbalance_method
        imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
        warn_msg = f"'{original_method}' requires Grid Search. Using 'smogn' instead for NSGA-II."
        logger.warning(warn_msg)
        if progress_callback:
            progress_callback({'message': f"[Warning] {warn_msg}"})
        elif verbose >= 1:
            print(f"  Note: {warn_msg}")

    # Validate imbalance configuration for classification
    if task_type == 'classification' and imbalance_method is not None:
        # Need to encode y first for validation
        y_for_validation = y
        if not pd.api.types.is_numeric_dtype(y.dtype):
            from sklearn.preprocessing import LabelEncoder as LE
            y_arr = np.asarray(y)
            if y_arr.dtype == object:
                y_arr = y_arr.astype(str)
            y_for_validation = LE().fit_transform(y_arr)
        validate_classification_config(
            y=y_for_validation,
            imbalance_method=imbalance_method,
            imbalance_params=imbalance_params,
            n_folds=cv_folds
        )

    # Create problem with user-specified models
    problem = SpectralOptimizationProblem(
        X=X,
        y=y,
        task_type=task_type,
        cv_folds=cv_folds,
        min_wavelengths=min_wavelengths,
        random_state=random_state,
        models=models,
        imbalance_method=imbalance_method,
        imbalance_params=imbalance_params,
        early_stopping_rounds=early_stopping_rounds,
    )

    # Encode labels for classification before CARS importance calculation
    # (CARS-Tree uses numeric targets internally)
    y_for_importance = y
    if task_type == 'classification':
        if not pd.api.types.is_numeric_dtype(y.dtype):
            label_encoder_for_importance = LabelEncoder()
            y_arr = np.asarray(y)
            if y_arr.dtype == object:
                y_arr = y_arr.astype(str)
            y_for_importance = label_encoder_for_importance.fit_transform(y_arr)

    # Compute CARS-Tree importance for principled wavelength guidance (if use_guidance=True)
    # CARS-Tree uses hybrid split+gain importance (denser than plain CARS)
    if use_guidance:
        if verbose >= 1:
            print("  Computing CARS-Tree importance scores for wavelength guidance...")
        try:
            cars_importance = cars_selection(
                X, y_for_importance,
                n_iterations=50,
                pls_components=min(10, X.shape[0] - 1),
                cv_folds=min(cv_folds, X.shape[0] // 2),  # Ensure enough samples per fold
                monte_carlo_samples=80,
                random_state=random_state,
                model_type='LightGBM',  # Use tree model for denser importance distribution
                use_hybrid_importance=True,  # CARS-Tree mode: blend split+gain importance
                hybrid_importance_weight=0.5,
                task_type=task_type  # Pass task type for correct model selection
            )
            # Normalize to [0, 1]
            if cars_importance.max() > 0:
                cars_importance = cars_importance / cars_importance.max()
            else:
                cars_importance = np.ones(X.shape[1]) / X.shape[1]
            if verbose >= 1:
                n_nonzero = np.sum(cars_importance > 0.01)
                print(f"  CARS-Tree selected {n_nonzero} important wavelengths (>1% importance)")
        except Exception as e:
            if verbose >= 1:
                print(f"  CARS-Tree failed ({e}), using uniform importance")
            cars_importance = np.ones(X.shape[1]) / X.shape[1]
    else:
        # Standard NSGA-II: No importance guidance
        if verbose >= 1:
            print("  Using standard NSGA-II (no importance guidance)")
        cars_importance = None

    # Configure NSGA-II operators based on use_guidance flag
    if use_guidance:
        # Guided NSGA-II: Use CARS importance for seeded sampling and smart mutation
        # Population is initialized with importance-weighted wavelength selection
        # (targets ~250 wavelengths per solution instead of all wavelengths)
        custom_sampling = SeededWavelengthSampling(
            n_wavelengths=problem.n_wavelengths,
            model_types=models,
            n_preproc=len(PREPROC_TYPES),
            n_window=len(WINDOW_SIZES),
            importance_scores=cars_importance,
            target_n_wavelengths=250,  # Start with compact subsets
        )

        # Create mutation operator with CARS importance and sparsity bias
        # sparsity_bias=1.4 means dropping is 1.4x more likely than adding (mild preference for compact subsets)
        mutation_operator = SmartMutation(
            prob=0.1,
            eta=20,
            importance_scores=cars_importance,
            sparsity_bias=1.4,
        )
    else:
        # Standard NSGA-II: Use random sampling and polynomial mutation
        custom_sampling = IntegerRandomSampling()
        mutation_operator = PM(prob=0.1, eta=20, vtype=float, repair=RoundingRepair())

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=custom_sampling,
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=mutation_operator,
        eliminate_duplicates=True,
    )

    # Termination criterion
    termination = get_termination("n_gen", n_generations)

    # Track history for progress
    history = []

    class ProgressCallback:
        def __init__(self, total_gen, callback, verbose, ctrl,
                     importance_tracker=None, mutation_operator=None):
            self.total_gen = total_gen
            self.callback = callback
            self.verbose = verbose
            self.ctrl = ctrl
            self.gen = 0
            self.best_error = None
            self.n_pareto = 0
            self.cancelled = False
            self.importance_tracker = importance_tracker
            self.mutation_operator = mutation_operator

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
                X_pop = algorithm.pop.get("X")

                if F is not None and len(F) > 0:
                    # Best error (first objective)
                    self.best_error = F[:, 0].min()
                    self.n_pareto = len(F)
                    history.append(self.best_error)

                    # Update importance scores adaptively every N generations
                    if self.importance_tracker is not None and X_pop is not None:
                        # Track best preprocessing from current population
                        self.importance_tracker.update_best_from_population(X_pop, F)

                        # Recompute importance if interval elapsed
                        if self.importance_tracker.should_update(self.gen):
                            new_importance = self.importance_tracker.compute_importance()
                            self.importance_tracker.last_update_gen = self.gen

                            # Update mutation operator with new importance scores
                            if self.mutation_operator is not None:
                                self.mutation_operator.set_importance_scores(new_importance)

                            if self.verbose >= 2:
                                print(f"    Importance scores updated (gen {self.gen})")

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

    # Instantiate callback (no adaptive importance updates - CARS importance is static)
    callback = ProgressCallback(
        n_generations,
        progress_callback,
        verbose,
        controller,
        importance_tracker=None,  # CARS importance is used statically, no adaptive updates
        mutation_operator=mutation_operator,
    )

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

    # Log model evaluation distribution
    if verbose >= 1 and problem._model_eval_counts:
        print("\nNSGA-II model evaluation distribution:")
        for model_type, count in sorted(problem._model_eval_counts.items()):
            failures = problem._failure_counts.get(model_type, 0)
            success_rate = (count - failures) / count * 100 if count > 0 else 0
            print(f"  {model_type}: {count} evals ({failures} failures, {success_rate:.1f}% success)")

    # Log failure summary if any models failed
    if problem._failure_counts:
        if verbose >= 1:
            print("\nNSGA-II model failure summary:")
            for model_type, count in sorted(problem._failure_counts.items()):
                print(f"  {model_type}: {count} evaluation failures")
            print("  (First failure for each model type was logged as warning)")
        logger.info(
            f"NSGA-II failures: {dict(problem._failure_counts)}"
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
            'imbalance_method': imbalance_method,
            'imbalance_params': imbalance_params,
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
            'imbalance_method': imbalance_method,
            'imbalance_params': imbalance_params,
        }

    # Summarize model diversity in Pareto front
    if verbose >= 1 and pareto_solutions is not None and len(pareto_solutions) > 0:
        pareto_model_counts = {}
        for sol in pareto_solutions:
            model_idx = int(sol[2])
            model_name = models[min(model_idx, len(models) - 1)]
            pareto_model_counts[model_name] = pareto_model_counts.get(model_name, 0) + 1
        print("\nPareto front model diversity:")
        for model_name, count in sorted(pareto_model_counts.items(), key=lambda x: -x[1]):
            print(f"  {model_name}: {count} solutions ({count/len(pareto_solutions)*100:.1f}%)")

    # Select solution based on bias
    if selection_bias <= 0 and len(problem._all_objectives) > 0:
        # bias=0: Select from ALL evaluated solutions (not just Pareto front)
        all_objectives = np.array(problem._all_objectives)
        all_solutions = np.array(problem._all_solutions)

        # Find minimum error across ALL evaluations
        best_idx = int(np.argmin(all_objectives[:, 0]))

        # Use this solution
        knee_chromosome = all_solutions[best_idx].astype(int)
        knee_solution = decode_solution(knee_chromosome, problem.n_wavelengths, models, task_type, n_samples=problem.X.shape[0])
        knee_solution['objectives'] = {
            'error': all_objectives[best_idx, 0],
            'n_wavelengths': all_objectives[best_idx, 1] * problem.n_wavelengths,
            'complexity': all_objectives[best_idx, 2],
        }
        knee_idx = -1  # Not from Pareto front

        if verbose >= 1:
            print(f"\nOptimization complete!")
            print(f"  Pareto front size: {len(pareto_front)}")
            print(f"  Total evaluations: {problem._eval_count}")
            print(f"\nMinimum error solution (from {len(all_objectives)} evaluations):")
            print(f"  Preprocessing: {knee_solution['preprocessing']}")
            print(f"  Model: {knee_solution['model']} ({knee_solution['model_params']})")
            print(f"  Wavelengths: {knee_solution['n_wavelengths']} selected")
            print(f"  Error: {knee_solution['objectives']['error']:.4f}")
            print(f"  Complexity: {knee_solution['objectives']['complexity']:.4f}")
    else:
        # bias=1 or 2: Select from Pareto front
        knee_idx = find_knee_point(pareto_front, selection_bias=selection_bias)

        # Decode knee solution
        knee_chromosome = pareto_solutions[knee_idx].astype(int)
        knee_solution = decode_solution(knee_chromosome, problem.n_wavelengths, models, task_type, n_samples=problem.X.shape[0])
        knee_solution['objectives'] = {
            'error': pareto_front[knee_idx, 0],
            'n_wavelengths': pareto_front[knee_idx, 1] * problem.n_wavelengths,
            'complexity': pareto_front[knee_idx, 2],
        }

        if verbose >= 1:
            print(f"\nOptimization complete!")
            print(f"  Pareto front size: {len(pareto_front)}")
            print(f"  Total evaluations: {problem._eval_count}")

            # Determine selection method based on bias value
            if selection_bias >= 2:
                selection_method = "Knee point solution"
            else:
                selection_method = f"Weighted (bias={selection_bias:.1f}) solution"

            print(f"\n{selection_method}:")
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
        'imbalance_method': imbalance_method,
        'imbalance_params': imbalance_params,
        'early_stopping_rounds': early_stopping_rounds,
    }


def _format_imbalance_display(imbalance_method: Optional[str]) -> str:
    """Format imbalance method for display in Results tab."""
    if imbalance_method is None:
        return "—"
    return imbalance_method


def decode_solution(chromosome: np.ndarray, n_wavelengths: int, model_types: Optional[List[str]] = None, task_type: str = 'regression', n_samples: int = None) -> Dict[str, Any]:
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
    n_samples : int, optional
        Number of samples. If provided, constrains PLS n_components appropriately.
        PLS requires n_components <= min(n_features, n_samples - 1).

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

    # Number of selected features (for PLS constraint)
    n_selected_features = int(np.sum(wavelength_mask))

    # Compute NSGA-specific parameter overrides
    if model_type in ('PLS', 'PLS-DA'):
        # Apply constraints: n_components <= min(n_features, n_samples - 1)
        n_components = _get_constrained_pls_components(model_param, n_selected_features, n_samples)
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
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
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
    imbalance_method : str, optional
        Imbalance handling method (e.g., 'smogn', 'smote')
    imbalance_params : dict, optional
        Parameters for imbalance method

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

        # Apply edge masking (same as _compute_prediction_error)
        # Edge zone = window // 2 on each side (unreliable due to SG interpolation)
        edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
        if edge_zone > 0:
            wavelength_mask = wavelength_mask.copy()
            wavelength_mask[:edge_zone] = False
            wavelength_mask[-edge_zone:] = False

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
        # Practical cap: 15 is rarely exceeded in chemometrics
        # n_features - 1 ensures PLS doesn't degenerate to OLS
        # CRITICAL: Use CV fold training size, not full dataset size
        if model_type in ('PLS', 'PLS-DA'):
            min_train_samples = X_subset.shape[0] * (cv_folds - 1) // cv_folds
            n_components = min(model_param + 1, 15, X_subset.shape[1] - 1, min_train_samples - 1)
            n_components = max(1, n_components)
            # scale=False matches get_model() in models.py for consistent R² between NSGA and Model Development
            model = PLSRegression(n_components=n_components, scale=False)
        else:
            # Use _build_model for all other models with hyperparams
            model = _build_model(model_type, model_param, task_type, random_state, hyperparams)

        # Build pipeline with imbalance handling if specified
        # This ensures R2cv matches what Model Development computes
        if imbalance_method is not None and imbalance_method != 'class_weight':
            from sklearn.pipeline import Pipeline
            from spectral_predict.imbalance import build_imbalance_transformer

            imb_params = imbalance_params if imbalance_params else {}
            imbalance_transformer = build_imbalance_transformer(
                method=imbalance_method,
                task_type=task_type,
                random_state=random_state,
                **imb_params
            )

            pipe_steps = [("imbalance", imbalance_transformer), ("model", model)]

            # Use ImbPipeline for resampling methods
            if _needs_resampling_pipeline(imbalance_method, task_type):
                from imblearn.pipeline import Pipeline as ImbPipeline
                model = ImbPipeline(pipe_steps)
            else:
                model = Pipeline(pipe_steps)

        # Cross-validation for R2 using aggregated predictions (not per-fold averages)
        # Averaging per-fold R² is mathematically incorrect due to different SS_tot per fold
        # This matches the method used in search.py for consistency with Model Development
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y_pred = cross_val_predict(model, X_subset, y, cv=cv)

        return float(r2_score(y, y_pred))

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
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Compute RMSE for display, with edge masking to match optimization.

    This function recomputes RMSE using the same edge masking as optimization so
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
    imbalance_method : str, optional
        Imbalance handling method (e.g., 'smogn', 'smote')
    imbalance_params : dict, optional
        Parameters for imbalance method

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

        # Apply edge masking (same as _compute_prediction_error)
        # Edge zone = window // 2 on each side (unreliable due to SG interpolation)
        edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
        if edge_zone > 0:
            wavelength_mask = wavelength_mask.copy()
            wavelength_mask[:edge_zone] = False
            wavelength_mask[-edge_zone:] = False

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
        # Practical cap: 15 is rarely exceeded in chemometrics
        # n_features - 1 ensures PLS doesn't degenerate to OLS
        # CRITICAL: Use CV fold training size, not full dataset size
        if model_type in ('PLS', 'PLS-DA'):
            min_train_samples = X_subset.shape[0] * (cv_folds - 1) // cv_folds
            n_components = min(model_param + 1, 15, X_subset.shape[1] - 1, min_train_samples - 1)
            n_components = max(1, n_components)
            model = PLSRegression(n_components=n_components, scale=False)
        else:
            # Use _build_model for all other models with hyperparams
            model = _build_model(model_type, model_param, task_type, random_state, hyperparams)

        # Build pipeline with imbalance handling if specified
        # This ensures RMSEcv matches what Model Development computes
        if imbalance_method is not None and imbalance_method != 'class_weight':
            from sklearn.pipeline import Pipeline
            from spectral_predict.imbalance import build_imbalance_transformer

            imb_params = imbalance_params if imbalance_params else {}
            imbalance_transformer = build_imbalance_transformer(
                method=imbalance_method,
                task_type=task_type,
                random_state=random_state,
                **imb_params
            )

            pipe_steps = [("imbalance", imbalance_transformer), ("model", model)]

            # Use ImbPipeline for resampling methods
            if _needs_resampling_pipeline(imbalance_method, task_type):
                from imblearn.pipeline import Pipeline as ImbPipeline
                model = ImbPipeline(pipe_steps)
            else:
                model = Pipeline(pipe_steps)

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


def _compute_nir_metrics(
    X: np.ndarray,
    y: np.ndarray,
    solution: np.ndarray,
    n_wavelengths: int,
    model_types: List[str],
    task_type: str,
    cv_folds: int = 5,
    random_state: int = 42,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute NIR-specific metrics (MAEcv, Bias, RPD, RER) for display.

    Uses cross-validated predictions to compute metrics that match Model Development.

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
    imbalance_method : str, optional
        Imbalance handling method
    imbalance_params : dict, optional
        Parameters for imbalance method

    Returns
    -------
    metrics : dict
        Dictionary with NIR metrics: {'MAEcv', 'Bias', 'RPD', 'RER'}
        Values are np.nan if computation failed.
    """
    default_metrics = {'MAEcv': np.nan, 'Bias': np.nan, 'RPD': np.nan, 'RER': np.nan, 'CCCcv': np.nan}

    if task_type != 'regression':
        return default_metrics

    try:
        # Decode solution (same as _compute_display_rmse)
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

        # Apply edge masking (same as _compute_display_rmse)
        edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
        if edge_zone > 0:
            wavelength_mask = wavelength_mask.copy()
            wavelength_mask[:edge_zone] = False
            wavelength_mask[-edge_zone:] = False

        # Decode hyperparameter genes
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
            return default_metrics

        # Get model type
        model_type = model_types[min(model_idx, len(model_types) - 1)]

        # Build model (same logic as _compute_display_rmse)
        if model_type in ('PLS', 'PLS-DA'):
            min_train_samples = X_subset.shape[0] * (cv_folds - 1) // cv_folds
            n_components = min(model_param + 1, 15, X_subset.shape[1] - 1, min_train_samples - 1)
            n_components = max(1, n_components)
            model = PLSRegression(n_components=n_components, scale=False)
        else:
            model = _build_model(model_type, model_param, task_type, random_state, hyperparams)

        # Build pipeline with imbalance handling if specified
        if imbalance_method is not None and imbalance_method != 'class_weight':
            from sklearn.pipeline import Pipeline
            from spectral_predict.imbalance import build_imbalance_transformer

            imb_params = imbalance_params if imbalance_params else {}
            imbalance_transformer = build_imbalance_transformer(
                method=imbalance_method,
                task_type=task_type,
                random_state=random_state,
                **imb_params
            )

            pipe_steps = [("imbalance", imbalance_transformer), ("model", model)]

            if _needs_resampling_pipeline(imbalance_method, task_type):
                from imblearn.pipeline import Pipeline as ImbPipeline
                model = ImbPipeline(pipe_steps)
            else:
                model = Pipeline(pipe_steps)

        # Cross-validation to get predictions
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y_pred_cv = cross_val_predict(model, X_subset, y, cv=cv)

        # Compute NIR metrics from CV predictions
        # MAEcv: Mean Absolute Error
        mae_cv = mean_absolute_error(y, y_pred_cv)
        # Bias: Mean prediction error (positive = systematic overprediction)
        bias_cv = float(np.mean(y_pred_cv - y))
        # RPD and RER need RMSE (compute from predictions for consistency)
        rmse_cv = np.sqrt(np.mean((y - y_pred_cv) ** 2))
        # RPD: Ratio of Performance to Deviation
        y_std = float(np.std(y))
        rpd = y_std / rmse_cv if rmse_cv > 0 else 0.0
        # RER: Range Error Ratio
        y_range = float(np.ptp(y))
        rer = y_range / rmse_cv if rmse_cv > 0 else 0.0

        return {
            'MAEcv': float(mae_cv),
            'Bias': bias_cv,
            'RPD': rpd,
            'RER': rer,
            'CCCcv': float(lins_ccc(y, y_pred_cv.ravel())),
        }

    except Exception:
        return default_metrics


def _compute_classification_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    solution: np.ndarray,
    n_wavelengths: int,
    model_types: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute classification CV metrics (F1, ROC_AUC, Precision, Recall) for a solution.

    This performs manual cross-validation to compute all metrics within each fold,
    then returns the mean values. This matches how Grid Search computes CV metrics.

    Parameters
    ----------
    X : np.ndarray
        Input data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values (class labels)
    solution : np.ndarray
        Chromosome encoding the solution
    n_wavelengths : int
        Total number of wavelengths
    model_types : list of str
        List of model types used in optimization
    cv_folds : int
        Number of CV folds
    random_state : int
        Random state for reproducibility
    imbalance_method : str, optional
        Imbalance handling method (e.g., 'smote', 'adasyn')
    imbalance_params : dict, optional
        Parameters for imbalance method

    Returns
    -------
    metrics : dict
        Dictionary with CV metrics: {'F1cv', 'ROC_AUCcv', 'Precisioncv', 'Recallcv',
        'Specificitycv', 'Kappacv', 'MCCcv', 'BalancedAcccv', 'BERcv', 'LogLosscv'}
        Values are None if computation failed.
    """
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Encode labels for classification (PLS-DA requires numeric y)
        le = LabelEncoder()
        y_arr = np.asarray(y)
        if y_arr.dtype == object:
            y_arr = y_arr.astype(str)
        y = le.fit_transform(y_arr)

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

        # Apply edge masking (same as _compute_prediction_error)
        edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
        if edge_zone > 0:
            wavelength_mask = wavelength_mask.copy()
            wavelength_mask[:edge_zone] = False
            wavelength_mask[-edge_zone:] = False

        # Decode hyperparameter genes
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
            return {'F1cv': None, 'ROC_AUCcv': None, 'Precisioncv': None, 'Recallcv': None}

        # Get model type
        model_type = model_types[min(model_idx, len(model_types) - 1)]

        # Detect binary vs multiclass
        n_classes = len(np.unique(y))
        is_binary = n_classes == 2

        # Set averaging method based on number of classes
        average = 'binary' if is_binary else 'macro'

        # Build imbalance transformer if specified
        imbalance_transformer = None
        if imbalance_method is not None and imbalance_method != 'class_weight':
            from spectral_predict.imbalance import build_imbalance_transformer
            imb_params = imbalance_params if imbalance_params else {}
            imbalance_transformer = build_imbalance_transformer(
                method=imbalance_method,
                task_type='classification',
                random_state=random_state,
                **imb_params
            )

        # Manual cross-validation to compute all metrics
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        f1_scores = []
        roc_auc_scores = []
        precision_scores = []
        recall_scores = []
        specificity_scores = []
        kappa_scores = []
        mcc_scores = []
        balanced_acc_scores = []
        ber_scores = []
        logloss_scores = []

        for train_idx, test_idx in cv.split(X_subset, y):
            X_train, X_test = X_subset[train_idx], X_subset[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply imbalance resampling to training data (inside CV fold)
            X_train_resampled, y_train_resampled = X_train, y_train
            if imbalance_transformer is not None and _needs_resampling_pipeline(imbalance_method, 'classification'):
                try:
                    from sklearn.base import clone
                    imb_clone = clone(imbalance_transformer)
                    X_train_resampled, y_train_resampled = imb_clone.fit_resample(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Imbalance resampling failed in fold: {type(e).__name__}: {e}")

            # Build model for this fold
            if model_type in ('PLS', 'PLS-DA'):
                # PLS-DA: PLSTransformer + StandardScaler + LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                n_components = min(model_param + 1, 15, X_train_resampled.shape[1] - 1, X_train_resampled.shape[0] - 1)
                n_components = max(1, n_components)
                pls_transformer = PLSTransformer(n_components=n_components, scale=False)
                # Extract LogisticRegression parameters from hyperparams (prefixed with lr_)
                lr_C = hyperparams.get('lr_C', 1.0) if hyperparams else 1.0
                lr_solver = hyperparams.get('lr_solver', 'lbfgs') if hyperparams else 'lbfgs'
                lr_max_iter = hyperparams.get('lr_max_iter', 1000) if hyperparams else 1000
                lr = LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=random_state)
                # Apply class_weight if specified
                if imbalance_method == 'class_weight':
                    lr.set_params(class_weight='balanced')
                model = Pipeline([
                    ('pls', pls_transformer),
                    ('scaler', StandardScaler()),
                    ('lr', lr)
                ])
            else:
                model = _build_model(model_type, model_param, 'classification', random_state, hyperparams)
                # Apply class_weight if specified and model supports it
                if imbalance_method == 'class_weight' and model is not None and hasattr(model, 'class_weight'):
                    try:
                        model.set_params(class_weight='balanced')
                    except Exception:
                        pass

            if model is None:
                continue

            # Fit and predict
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X_train_resampled, y_train_resampled)
                y_pred = model.predict(X_test)

            # Compute F1, Precision, Recall
            try:
                f1_scores.append(f1_score(y_test, y_pred, average=average, zero_division=0))
                precision_scores.append(precision_score(y_test, y_pred, average=average, zero_division=0))
                recall_scores.append(recall_score(y_test, y_pred, average=average, zero_division=0))
            except Exception as e:
                logger.warning(f"F1/Precision/Recall failed in fold: {type(e).__name__}: {e}")

            # Compute ROC_AUC and Log Loss if model has predict_proba
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    if is_binary:
                        roc_auc_scores.append(roc_auc_score(y_test, y_proba[:, 1]))
                    else:
                        # Multi-class: use ovr average
                        roc_auc_scores.append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))

                    # Log Loss
                    try:
                        logloss_scores.append(log_loss(y_test, y_proba))
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"ROC_AUC failed in fold: {type(e).__name__}: {e}")

            # Compute additional classification metrics
            try:
                specificity_scores.append(compute_specificity(y_test, y_pred, average='macro'))
            except Exception:
                pass

            try:
                kappa_scores.append(cohen_kappa_score(y_test, y_pred))
            except Exception:
                pass

            try:
                mcc_scores.append(matthews_corrcoef(y_test, y_pred))
            except Exception:
                pass

            try:
                balanced_acc = balanced_accuracy_score(y_test, y_pred)
                balanced_acc_scores.append(balanced_acc)
                ber_scores.append(1.0 - balanced_acc)
            except Exception:
                pass

        # Return mean values
        return {
            'F1cv': float(np.mean(f1_scores)) if f1_scores else np.nan,
            'ROC_AUCcv': float(np.mean(roc_auc_scores)) if roc_auc_scores else np.nan,
            'Precisioncv': float(np.mean(precision_scores)) if precision_scores else np.nan,
            'Recallcv': float(np.mean(recall_scores)) if recall_scores else np.nan,
            'Specificitycv': float(np.mean(specificity_scores)) if specificity_scores else np.nan,
            'Kappacv': float(np.mean(kappa_scores)) if kappa_scores else np.nan,
            'MCCcv': float(np.mean(mcc_scores)) if mcc_scores else np.nan,
            'BalancedAcccv': float(np.mean(balanced_acc_scores)) if balanced_acc_scores else np.nan,
            'BERcv': float(np.mean(ber_scores)) if ber_scores else np.nan,
            'LogLosscv': float(np.mean(logloss_scores)) if logloss_scores else np.nan,
        }

    except Exception as e:
        logger.debug(f"_compute_classification_cv_metrics failed: {type(e).__name__}: {e}")
        return {
            'F1cv': np.nan, 'ROC_AUCcv': np.nan, 'Precisioncv': np.nan, 'Recallcv': np.nan,
            'Specificitycv': np.nan, 'Kappacv': np.nan, 'MCCcv': np.nan,
            'BalancedAcccv': np.nan, 'BERcv': np.nan, 'LogLosscv': np.nan
        }


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
        return ','.join([f"{w:.1f}" for w in selected_wl])
    else:
        # Fallback to indices if wavelengths not available
        return ','.join([str(i) for i in indices])


def _normalize_preprocess_name(name: str) -> str:
    """Convert NSGA-II preprocessing names to standard names for validation.

    NSGA-II uses detailed names like 'deriv1_w17', 'snv_deriv2_w21', 'deriv3_snv_w15',
    but build_preprocessing_pipeline() expects 'deriv', 'snv_deriv', 'deriv_snv'.

    Parameters
    ----------
    name : str
        NSGA-II preprocessing name (e.g., 'deriv1_w17', 'snv_deriv2_w21')

    Returns
    -------
    str
        Normalized name compatible with build_preprocessing_pipeline()
    """
    if name in ('raw', 'snv'):
        return name

    # Strip window suffix like '_w17', '_w43' first
    import re
    name_no_window = re.sub(r'_w\d+$', '', name)

    # deriv1, deriv2, deriv3, deriv4 → deriv
    if name_no_window.startswith('deriv') and '_' not in name_no_window:
        return 'deriv'
    # snv_deriv1, snv_deriv2, etc. → snv_deriv
    if name_no_window.startswith('snv_deriv'):
        return 'snv_deriv'
    # deriv1_snv, deriv2_snv, etc. → deriv_snv
    if name_no_window.startswith('deriv') and name_no_window.endswith('_snv'):
        return 'deriv_snv'
    return name  # fallback for unknown names


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

        # Cap n_components for PLS when X_subset has fewer features than expected
        n_features_subset = X_subset.shape[1]
        if hasattr(model, 'n_components') and model.n_components is not None:
            if model.n_components >= n_features_subset:
                from sklearn.base import clone
                model = clone(model)
                capped = max(1, n_features_subset - 1)
                model.set_params(n_components=capped)

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


def _compute_calibration_metrics(
    X: np.ndarray,
    y: np.ndarray,
    solution: np.ndarray,
    n_wavelengths: int,
    model_types: List[str],
    task_type: str,
) -> Dict[str, float]:
    """
    Compute calibration (training set) metrics for a single NSGA-II solution.

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

    Returns
    -------
    metrics : dict
        Dictionary with calibration metrics (RMSE, R2 for regression; Accuracy, etc. for classification)
    """
    from sklearn.metrics import (
        mean_squared_error, r2_score, accuracy_score,
        roc_auc_score, f1_score, precision_score, recall_score
    )

    try:
        # Encode labels for classification (PLS-DA requires numeric y)
        if task_type == 'classification':
            le = LabelEncoder()
            y_arr = np.asarray(y)
            if y_arr.dtype == object:
                y_arr = y_arr.astype(str)
            y = le.fit_transform(y_arr)

        # Decode solution directly (same pattern as _compute_display_rmse)
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

        # Apply edge masking (same as _compute_display_rmse)
        edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
        if edge_zone > 0:
            wavelength_mask = wavelength_mask.copy()
            wavelength_mask[:edge_zone] = False
            wavelength_mask[-edge_zone:] = False

        # Decode hyperparameters
        hyperparams = _decode_hyperparameter_genes(
            lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene,
            subsample_gene, colsample_gene, min_samples_gene, gamma_gene, max_features_gene
        )

        # Apply preprocessing FIRST (before wavelength selection)
        transform = _get_preprocessing_transform(preproc_idx, window_idx)
        if transform is not None:
            X_proc = transform(X)
        else:
            X_proc = X.copy()

        # Then select wavelengths
        X_subset = X_proc[:, wavelength_mask]

        if X_subset.shape[1] == 0:
            if task_type == 'regression':
                return {'RMSE': np.nan, 'R2': np.nan}
            else:
                return {
                    'Accuracy': np.nan, 'ROC_AUC': np.nan, 'F1': np.nan,
                    'Precision': np.nan, 'Recall': np.nan,
                    'Specificity': np.nan, 'Kappa': np.nan, 'MCC': np.nan,
                    'BalancedAcc': np.nan, 'BER': np.nan, 'LogLoss': np.nan
                }

        # Get model type and build model (same as _compute_display_rmse)
        model_type = model_types[min(model_idx, len(model_types) - 1)]

        # For PLS, limit n_components to valid range
        # Practical cap: 15 is rarely exceeded in chemometrics
        # n_features - 1 ensures PLS doesn't degenerate to OLS
        if model_type in ('PLS', 'PLS-DA'):
            n_components = min(model_param + 1, 15, X_subset.shape[1] - 1, X_subset.shape[0] - 1)
            n_components = max(1, n_components)
            if task_type == 'classification':
                # PLS-DA: PLSTransformer + StandardScaler + LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                pls_transformer = PLSTransformer(n_components=n_components, scale=False)
                # Extract LogisticRegression parameters from hyperparams (prefixed with lr_)
                lr_C = hyperparams.get('lr_C', 1.0) if hyperparams else 1.0
                lr_solver = hyperparams.get('lr_solver', 'lbfgs') if hyperparams else 'lbfgs'
                lr_max_iter = hyperparams.get('lr_max_iter', 1000) if hyperparams else 1000
                model = Pipeline([
                    ('pls', pls_transformer),
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=42))
                ])
            else:
                # Regression: use PLSRegression directly
                model = PLSRegression(n_components=n_components, scale=False)
        else:
            model = _build_model(model_type, model_param, task_type, 42, hyperparams)

        # Check if model was built successfully
        if model is None:
            logger.error(f"_build_model returned None for {model_type} in _compute_calibration_metrics")
            if task_type == 'regression':
                return {'RMSE': np.nan, 'R2': np.nan}
            else:
                return {'Accuracy': np.nan, 'ROC_AUC': np.nan, 'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan}

        # Fit on full training data
        model.fit(X_subset, y)

        # Predict on training data
        y_pred = model.predict(X_subset)

        # Compute metrics
        metrics = {}
        if task_type == 'regression':
            metrics['RMSE'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics['R2'] = r2_score(y, y_pred)
            metrics['CCC'] = lins_ccc(y, y_pred.ravel())
        else:
            metrics['Accuracy'] = accuracy_score(y, y_pred)

            # ROC AUC if probabilities available
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_subset)
                    n_classes = len(np.unique(y))
                    if n_classes == 2:
                        metrics['ROC_AUC'] = roc_auc_score(y, y_proba[:, 1])
                    else:
                        metrics['ROC_AUC'] = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
                else:
                    metrics['ROC_AUC'] = np.nan
            except Exception:
                metrics['ROC_AUC'] = np.nan

            # F1, Precision, Recall
            try:
                metrics['F1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
                metrics['Precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
                metrics['Recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            except Exception:
                metrics['F1'] = np.nan
                metrics['Precision'] = np.nan
                metrics['Recall'] = np.nan

            # Additional classification metrics
            try:
                metrics['Specificity'] = compute_specificity(y, y_pred, average='macro')
            except Exception:
                metrics['Specificity'] = np.nan

            try:
                metrics['Kappa'] = cohen_kappa_score(y, y_pred)
            except Exception:
                metrics['Kappa'] = np.nan

            try:
                metrics['MCC'] = matthews_corrcoef(y, y_pred)
            except Exception:
                metrics['MCC'] = np.nan

            try:
                metrics['BalancedAcc'] = balanced_accuracy_score(y, y_pred)
                metrics['BER'] = 1.0 - metrics['BalancedAcc']
            except Exception:
                metrics['BalancedAcc'] = np.nan
                metrics['BER'] = np.nan

            # Log Loss
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_subset)
                    metrics['LogLoss'] = log_loss(y, y_proba)
                else:
                    metrics['LogLoss'] = np.nan
            except Exception:
                metrics['LogLoss'] = np.nan

        return metrics

    except Exception as e:
        # Log the error for diagnostics before returning empty dict
        logger.error(f"_compute_calibration_metrics FAILED: task={task_type}, model_idx={int(solution[2])}, error={type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


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
    include_best_from_all: bool = True,
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
    include_best_from_all : bool, default True
        When selection_bias=0, the minimum error solution may not be on the
        Pareto front. If True, include this solution in results even if dominated.
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
        decoded = decode_solution(solution, n_wavelengths, model_types, task_type, n_samples=n_samples)

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
            # Window size is in decoded - use actual WINDOW_SIZES array lookup
            window_idx = decoded.get('window_idx', 6)  # Default to index 6 = window 17
            window_size = WINDOW_SIZES[min(window_idx, len(WINDOW_SIZES) - 1)]

        row = {
            'Task': task_type,
            'Model': decoded['model'],
            'Preprocessing': decoded['preprocessing'],
            'Preprocess': _normalize_preprocess_name(decoded['preprocessing']),  # Normalized for validation
            'Folds': folds,
            'N_Calibration': n_calibration,
            'N_Excluded': excluded_count,
            'N_Validation': validation_count,
            'Parameters': decoded['model_params'],
            'Params': decoded['model_params'],  # Alias for compatibility
            'Variables': f"nsga2_{decoded['n_wavelengths']}",
            'full_vars': n_wavelengths,  # Total wavelengths available
            'SubsetTag': 'nsga2',
            'Imbalance': _format_imbalance_display(result.get('imbalance_method')),
            'early_stopping_rounds': result.get('early_stopping_rounds'),
            'imbalance_method': result.get('imbalance_method'),
            'imbalance_params': result.get('imbalance_params'),
            'top_vars': _compute_top_variables(X, y, decoded, model_types, task_type, wavelengths, 30, 42) if (X is not None and y is not None and decoded['selected_indices']) else (_indices_to_wavelength_str(decoded['selected_indices'][:30], wavelengths) if decoded['selected_indices'] else 'N/A'),
            'all_vars': _indices_to_wavelength_str(decoded['selected_indices'], wavelengths) if decoded['selected_indices'] else 'N/A',
            'n_vars': decoded['n_wavelengths'],
            'Deriv': deriv_order,
            'Window': int(window_size) if window_size is not None else None,
            'Poly': deriv_order + 1 if deriv_order else None,  # polyorder = deriv + 1
            'LVs': _get_constrained_pls_components(int(solution[3]), decoded['n_wavelengths'], n_samples) if decoded['model'] in ('PLS', 'PLS-DA') else None,
            'Complexity': objectives[2],
            'Is_Knee': i == result['knee_idx'],
        }

        # Get imbalance settings from result for CV metric computation
        imbalance_method = result.get('imbalance_method')
        imbalance_params = result.get('imbalance_params')

        if task_type == 'regression':
            # Compute calibration metrics (training data)
            if X is not None and y is not None:
                cal_metrics = _compute_calibration_metrics(
                    X, y, solution, n_wavelengths, model_types, task_type
                )
                row['RMSE'] = cal_metrics.get('RMSE', np.nan)
                row['R2'] = cal_metrics.get('R2', np.nan)
                row['CCC'] = cal_metrics.get('CCC', np.nan)
            else:
                row['RMSE'] = np.nan
                row['R2'] = np.nan
                row['CCC'] = np.nan

            # Compute CV metrics (cross-validation) with imbalance handling
            if X is not None and y is not None:
                display_rmse = _compute_display_rmse(
                    X, y, solution, n_wavelengths, model_types, task_type, folds, 42,
                    imbalance_method=imbalance_method,
                    imbalance_params=imbalance_params
                )
                row['RMSEcv'] = display_rmse if display_rmse is not None else objectives[0]

                if compute_r2:
                    r2_cv = _compute_solution_r2(
                        X, y, solution, n_wavelengths, model_types, task_type, folds, 42,
                        imbalance_method=imbalance_method,
                        imbalance_params=imbalance_params
                    )
                    row['R2cv'] = r2_cv
                else:
                    row['R2cv'] = None

                # Compute NIR-specific metrics
                nir_metrics = _compute_nir_metrics(
                    X, y, solution, n_wavelengths, model_types, task_type, folds, 42,
                    imbalance_method=imbalance_method,
                    imbalance_params=imbalance_params
                )
                row['MAEcv'] = nir_metrics['MAEcv']
                row['Bias'] = nir_metrics['Bias']
                row['RPD'] = nir_metrics['RPD']
                row['RER'] = nir_metrics['RER']
                row['CCCcv'] = nir_metrics['CCCcv']
            else:
                row['RMSEcv'] = objectives[0]  # Fallback to optimization RMSE
                row['R2cv'] = None
                row['MAEcv'] = np.nan
                row['Bias'] = np.nan
                row['RPD'] = np.nan
                row['RER'] = np.nan
                row['CCCcv'] = np.nan

            row['CompositeScore'] = row['RMSEcv']  # Use CV RMSE as composite
        else:
            # Classification: compute calibration and CV metrics
            if X is not None and y is not None:
                cal_metrics = _compute_calibration_metrics(
                    X, y, solution, n_wavelengths, model_types, task_type
                )
                row['Accuracy'] = cal_metrics.get('Accuracy', np.nan)
                row['ROC_AUC'] = cal_metrics.get('ROC_AUC', np.nan)
                row['F1'] = cal_metrics.get('F1', np.nan)
                row['Precision'] = cal_metrics.get('Precision', np.nan)
                row['Recall'] = cal_metrics.get('Recall', np.nan)
                row['Specificity'] = cal_metrics.get('Specificity', np.nan)
                row['Kappa'] = cal_metrics.get('Kappa', np.nan)
                row['MCC'] = cal_metrics.get('MCC', np.nan)
                row['BalancedAcc'] = cal_metrics.get('BalancedAcc', np.nan)
                row['BER'] = cal_metrics.get('BER', np.nan)
                row['LogLoss'] = cal_metrics.get('LogLoss', np.nan)
            else:
                row['Accuracy'] = np.nan
                row['ROC_AUC'] = np.nan
                row['F1'] = np.nan
                row['Precision'] = np.nan
                row['Recall'] = np.nan
                row['Specificity'] = np.nan
                row['Kappa'] = np.nan
                row['MCC'] = np.nan
                row['BalancedAcc'] = np.nan
                row['BER'] = np.nan
                row['LogLoss'] = np.nan

            # CV metrics: compute actual CV metrics for F1, ROC_AUC, Precision, Recall with imbalance handling
            row['Accuracycv'] = 1.0 - objectives[0]  # From optimization objective
            if X is not None and y is not None:
                cv_metrics = _compute_classification_cv_metrics(
                    X, y, solution, n_wavelengths, model_types, folds, 42,
                    imbalance_method=imbalance_method,
                    imbalance_params=imbalance_params
                )
                row['ROC_AUCcv'] = cv_metrics.get('ROC_AUCcv')
                row['F1cv'] = cv_metrics.get('F1cv')
                row['Precisioncv'] = cv_metrics.get('Precisioncv')
                row['Recallcv'] = cv_metrics.get('Recallcv')
                row['Specificitycv'] = cv_metrics.get('Specificitycv')
                row['Kappacv'] = cv_metrics.get('Kappacv')
                row['MCCcv'] = cv_metrics.get('MCCcv')
                row['BalancedAcccv'] = cv_metrics.get('BalancedAcccv')
                row['BERcv'] = cv_metrics.get('BERcv')
                row['LogLosscv'] = cv_metrics.get('LogLosscv')
            else:
                row['ROC_AUCcv'] = None
                row['F1cv'] = None
                row['Precisioncv'] = None
                row['Recallcv'] = None
                row['Specificitycv'] = None
                row['Kappacv'] = None
                row['MCCcv'] = None
                row['BalancedAcccv'] = None
                row['BERcv'] = None
                row['LogLosscv'] = None

            row['CompositeScore'] = row['Accuracycv']  # Use CV accuracy as composite

            # Add GUI-required classification columns
            row['per_class_metrics'] = {}  # GUI checks: 'per_class_metrics' in results_df.columns
            row['class_labels'] = list(np.unique(y)) if y is not None else []
            # Per-class F1 columns
            if y is not None:
                for class_idx, class_label in enumerate(np.unique(y)):
                    row[f'F1_Class{class_idx}'] = row.get('F1', np.nan)

        rows.append(row)

    # Include best solution from all evaluations if not already in Pareto front
    # This happens when selection_bias=0 and the minimum error solution is dominated
    if include_best_from_all and result.get('knee_solution') is not None:
        knee_sol = result['knee_solution']
        knee_idx = result.get('knee_idx', -1)

        # knee_idx == -1 means the solution came from all evaluations, not Pareto front
        if knee_idx == -1 and knee_sol.get('objectives'):
            # Check if this solution has better error than anything in the Pareto front
            knee_error = knee_sol['objectives'].get('error', float('inf'))
            pareto_min_error = min(obj[0] for obj in result['pareto_front']) if len(result['pareto_front']) > 0 else float('inf')

            # Only add if it's actually better than what's in the Pareto front
            if knee_error < pareto_min_error:
                # Use explicit values from knee_sol if available
                preproc = knee_sol.get('preprocessing', 'raw')
                deriv_order = knee_sol.get('deriv_order')
                window_size = knee_sol.get('window')
                polyorder = knee_sol.get('polyorder')

                # Fallback to parsing if explicit values not available (backward compat)
                if deriv_order is None and ('deriv' in preproc):
                    for d in ['deriv4', 'deriv3', 'deriv2', 'deriv1']:
                        if d in preproc:
                            deriv_order = int(d[-1])
                            break
                if window_size is None and deriv_order is not None:
                    window_idx = knee_sol.get('window_idx', 6)
                    window_size = WINDOW_SIZES[min(window_idx, len(WINDOW_SIZES) - 1)]
                if polyorder is None and deriv_order is not None:
                    polyorder = deriv_order + 1

                best_row = {
                    'Task': task_type,
                    'Model': knee_sol.get('model', 'Unknown'),
                    'Preprocessing': preproc,
                    'Preprocess': _normalize_preprocess_name(preproc),  # Normalized for validation
                    'Folds': folds,
                    'N_Calibration': n_calibration,
                    'N_Excluded': excluded_count,
                    'N_Validation': validation_count,
                    'Parameters': knee_sol.get('model_params', ''),
                    'Params': knee_sol.get('model_params', ''),
                    'Variables': f"nsga2_{knee_sol.get('n_wavelengths', 0)}",
                    'full_vars': n_wavelengths,
                    'SubsetTag': 'nsga2_best',  # Mark as best from all evaluations
                    'Imbalance': 'none',
                    'early_stopping_rounds': result.get('early_stopping_rounds'),
                    'top_vars': _indices_to_wavelength_str(knee_sol.get('selected_indices', [])[:30], wavelengths) if knee_sol.get('selected_indices') else 'N/A',
                    'all_vars': _indices_to_wavelength_str(knee_sol.get('selected_indices', []), wavelengths) if knee_sol.get('selected_indices') else 'N/A',
                    'n_vars': knee_sol.get('n_wavelengths', 0),
                    'Deriv': deriv_order,
                    'Window': int(window_size) if window_size is not None else None,
                    'Poly': polyorder,
                    'LVs': knee_sol.get('model_params', {}).get('n_components') if knee_sol.get('model') in ('PLS', 'PLS-DA') else None,
                    'Complexity': knee_sol['objectives'].get('complexity', 0),
                    'Is_Knee': False,
                    'Is_Best_Error': True,  # Mark this as the minimum error solution
                }

                if task_type == 'regression':
                    # Compute calibration metrics
                    if X is not None and y is not None:
                        # Reconstruct solution array from knee_sol for calibration metrics
                        # This is a simplified version - may not work for all cases
                        # But provides basic calibration metrics
                        try:
                            from .models import get_model
                            from sklearn.metrics import mean_squared_error, r2_score

                            selected_indices = knee_sol.get('selected_indices', [])
                            if selected_indices:
                                X_selected = X[:, selected_indices]
                                preproc_idx = knee_sol.get('preproc_idx', 0)
                                window_idx = knee_sol.get('window_idx', 6)
                                transform = _get_preprocessing_transform(preproc_idx, window_idx)
                                X_proc = transform(X_selected) if transform else X_selected

                                model = get_model(knee_sol.get('model', 'PLS'), task_type, knee_sol.get('stored_params', {}))
                                model.fit(X_proc, y)
                                y_pred_cal = model.predict(X_proc)

                                best_row['RMSE'] = np.sqrt(mean_squared_error(y, y_pred_cal))
                                best_row['R2'] = r2_score(y, y_pred_cal)
                            else:
                                best_row['RMSE'] = np.nan
                                best_row['R2'] = np.nan
                        except Exception:
                            best_row['RMSE'] = np.nan
                            best_row['R2'] = np.nan
                    else:
                        best_row['RMSE'] = np.nan
                        best_row['R2'] = np.nan

                    # CV metrics
                    best_row['RMSEcv'] = knee_error
                    if compute_r2 and X is not None and y is not None:
                        # Compute R2cv via cross-validation
                        try:
                            from sklearn.model_selection import cross_val_predict, KFold
                            from .models import get_model

                            selected_indices = knee_sol.get('selected_indices', [])
                            if selected_indices:
                                X_selected = X[:, selected_indices]
                            else:
                                X_selected = X

                            # Apply preprocessing
                            preproc_idx = knee_sol.get('preproc_idx', 0)
                            window_idx = knee_sol.get('window_idx', 6)
                            transform = _get_preprocessing_transform(preproc_idx, window_idx)
                            X_proc = transform(X_selected) if transform else X_selected

                            # Get model
                            model = get_model(knee_sol.get('model', 'PLS'), task_type, knee_sol.get('stored_params', {}))

                            # Cross-val predict
                            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
                            y_pred = cross_val_predict(model, X_proc, y, cv=kf)

                            # Compute R2cv
                            ss_res = np.sum((y - y_pred) ** 2)
                            ss_tot = np.sum((y - np.mean(y)) ** 2)
                            r2_cv = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                            best_row['R2cv'] = r2_cv
                        except Exception:
                            best_row['R2cv'] = None
                    else:
                        best_row['R2cv'] = None
                    best_row['CompositeScore'] = knee_error
                else:
                    # Classification calibration metrics
                    best_row['Accuracy'] = np.nan
                    best_row['ROC_AUC'] = np.nan
                    best_row['F1'] = np.nan
                    best_row['Precision'] = np.nan
                    best_row['Recall'] = np.nan
                    # Classification CV metrics
                    best_row['Accuracycv'] = 1.0 - knee_error
                    best_row['ROC_AUCcv'] = None
                    best_row['F1cv'] = None
                    best_row['Precisioncv'] = None
                    best_row['Recallcv'] = None
                    best_row['CompositeScore'] = 1.0 - knee_error

                    # Add GUI-required classification columns
                    best_row['per_class_metrics'] = {}  # GUI checks: 'per_class_metrics' in results_df.columns
                    best_row['class_labels'] = list(np.unique(y)) if y is not None else []
                    # Per-class F1 columns
                    if y is not None:
                        for class_idx, class_label in enumerate(np.unique(y)):
                            best_row[f'F1_Class{class_idx}'] = best_row.get('F1', np.nan)

                rows.append(best_row)

    df = pd.DataFrame(rows)

    # Sort and rank using CV metrics (unbiased performance estimates)
    if task_type == 'regression':
        df = df.sort_values('RMSEcv', ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values('Accuracycv', ascending=False).reset_index(drop=True)

    df['Rank'] = range(1, len(df) + 1)

    # Remove duplicate columns (keep 'Params' not 'Parameters', keep 'Preprocess' not 'Preprocessing')
    if 'Parameters' in df.columns:
        df = df.drop(columns=['Parameters'])
    if 'Preprocessing' in df.columns:
        df = df.drop(columns=['Preprocessing'])

    # Reorder columns to match Grid Search format
    # Preprocessing columns early (Deriv, Window, Poly, LVs, n_vars), metrics in middle, top_vars/all_vars at end
    base_cols = ['Rank', 'Task', 'Model', 'Params', 'Preprocess', 'Deriv', 'Window',
                 'Poly', 'LVs', 'n_vars', 'Variables', 'full_vars', 'SubsetTag', 'Imbalance',
                 'early_stopping_rounds']

    # Performance metrics after Imbalance (calibration first, then CV, then NIR-specific)
    if task_type == 'regression':
        perf_cols = ['RMSE', 'R2', 'RMSEcv', 'R2cv', 'MAEcv', 'RPD', 'Bias', 'RER', 'CompositeScore']
    else:
        perf_cols = [
            # Calibration metrics
            'Accuracy', 'ROC_AUC', 'F1', 'Precision', 'Recall',
            'Specificity', 'Kappa', 'MCC', 'BalancedAcc', 'BER', 'LogLoss',
            # Cross-validation metrics
            'Accuracycv', 'ROC_AUCcv', 'F1cv', 'Precisioncv', 'Recallcv',
            'Specificitycv', 'Kappacv', 'MCCcv', 'BalancedAcccv', 'BERcv', 'LogLosscv',
            # Additional columns
            'CompositeScore'
        ]

    # Variable columns at end
    other_cols = ['top_vars', 'all_vars']

    # NSGA-specific columns at end
    nsga_cols = ['Complexity', 'Is_Knee', 'Folds', 'N_Calibration', 'N_Excluded', 'N_Validation']
    if 'Is_Best_Error' in df.columns:
        nsga_cols.append('Is_Best_Error')

    # Build final column order (only include columns that exist in df)
    final_cols = []
    for col_list in [base_cols, perf_cols, other_cols, nsga_cols]:
        final_cols.extend([col for col in col_list if col in df.columns])

    # Add any remaining columns not in our explicit lists
    remaining = [col for col in df.columns if col not in final_cols]
    final_cols.extend(remaining)

    df = df[final_cols]

    # Convert integer columns to nullable Int64 to avoid float display (e.g., 1.0 -> 1)
    int_cols = ['Deriv', 'Window', 'Poly', 'LVs', 'n_vars', 'Rank', 'Folds',
                'N_Calibration', 'N_Excluded', 'N_Validation', 'full_vars',
                'early_stopping_rounds']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    return df
