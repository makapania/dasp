"""Model definitions and grid search configurations."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.base import BaseEstimator, TransformerMixin
from .neural_boosted import NeuralBoostedRegressor, NeuralBoostedClassifier

# Import gradient boosting libraries
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# CatBoost is optional (requires Visual Studio on Windows)
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    CatBoostRegressor = None
    CatBoostClassifier = None
    print("Warning: CatBoost not available (requires Visual Studio 2022 on Windows). CatBoost models will be disabled.")

# Import tiered configuration
from .model_config import (
    OPTIMIZED_HYPERPARAMETERS,
    get_tier_models,
    get_hyperparameters,
    get_execution_defaults
)


class PLSTransformer(BaseEstimator, TransformerMixin):
    """Wrapper for PLSRegression that ensures transform() returns 2D output for classification.

    This fixes the issue where PLSRegression.transform() can return 3D arrays when
    fitted with 2D y, which breaks downstream classifiers like LogisticRegression.
    """

    def __init__(self, n_components=2, max_iter=500, tol=1e-6, scale=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.pls_ = None

    def fit(self, X, y):
        """Fit PLS model, ensuring y is 1D for classification."""
        self.pls_ = PLSRegression(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            scale=self.scale
        )
        # Ensure y is 1D (sklearn classifiers expect this)
        y_1d = np.ravel(y) if y.ndim > 1 else y
        self.pls_.fit(X, y_1d)
        return self

    def transform(self, X):
        """Transform X to PLS scores (always returns 2D: n_samples x n_components)."""
        if self.pls_ is None:
            raise ValueError("PLSTransformer must be fitted before transform")
        # Use transform() which returns X_scores (2D)
        X_scores = self.pls_.transform(X)
        # Ensure 2D output
        if X_scores.ndim == 1:
            X_scores = X_scores.reshape(-1, 1)
        elif X_scores.ndim > 2:
            # If somehow 3D, take first output dimension
            X_scores = X_scores[:, :, 0] if X_scores.shape[2] == 1 else X_scores.reshape(X_scores.shape[0], -1)
        return X_scores

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'scale': self.scale
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __getattr__(self, name):
        """Forward attribute access to underlying PLS model."""
        # Avoid infinite recursion for our own attributes
        if name == 'pls_':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Forward to underlying PLS model if fitted
        pls = object.__getattribute__(self, 'pls_')
        if pls is not None:
            return getattr(pls, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (model not fitted)")

    def __sklearn_is_fitted__(self):
        """Check if the transformer is fitted.

        Returns
        -------
        bool
            True if the underlying PLS model is fitted, False otherwise
        """
        return self.pls_ is not None and hasattr(self.pls_, 'x_weights_')


def get_model(model_name, task_type='regression', n_components=10, max_n_components=8, max_iter=500):
    """
    Get a single model instance with default hyperparameters.

    Parameters
    ----------
    model_name : str
        Model type name ('PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted', 'PLS-DA')
    task_type : str, default='regression'
        'regression' or 'classification'
    n_components : int, default=10
        Number of components for PLS models
    max_n_components : int, default=8
        Maximum number of PLS components to test
    max_iter : int, default=500
        Maximum iterations for neural network models

    Returns
    -------
    model : estimator
        Configured model instance ready for fitting
    """
    # Clip n_components to max_n_components
    n_components = min(n_components, max_n_components)

    if task_type == "regression":
        if model_name == "PLS":
            return PLSRegression(n_components=n_components, scale=False)

        elif model_name == "Ridge":
            return Ridge(alpha=1.0, random_state=42)

        elif model_name == "Lasso":
            return Lasso(alpha=1.0, random_state=42, max_iter=max_iter)

        elif model_name == "ElasticNet":
            return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=max_iter)

        elif model_name == "RandomForest":
            exec_defaults = get_execution_defaults('RandomForest')
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                **exec_defaults
            )

        elif model_name == "MLP":
            exec_defaults = get_execution_defaults('MLP')
            return MLPRegressor(
                hidden_layer_sizes=(64,),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                random_state=42,
                **exec_defaults
            )

        elif model_name == "NeuralBoosted":
            exec_defaults = get_execution_defaults('NeuralBoosted')
            return NeuralBoostedRegressor(
                n_estimators=100,
                learning_rate=0.1,
                hidden_layer_size=3,
                activation='tanh',
                random_state=42,
                **exec_defaults
            )

        elif model_name == "SVR":
            return SVR(kernel='rbf', C=1.0, gamma='scale')

        elif model_name == "XGBoost":
            exec_defaults = get_execution_defaults('XGBoost')
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,  # XGBoost default (original working value)
                subsample=0.8,  # Original working value for spectroscopy
                colsample_bytree=0.8,  # Original working value for high-dim data
                reg_alpha=0.1,  # Light L1 regularization for feature selection
                reg_lambda=1.0,  # XGBoost default L2 regularization
                random_state=42,
                **exec_defaults
            )

        elif model_name == "LightGBM":
            exec_defaults = get_execution_defaults('LightGBM')
            return LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,  # LightGBM default
                max_depth=-1,  # No depth limit (controlled by num_leaves)
                min_child_samples=10,  # Balanced: 5 too aggressive for large datasets, 20 too conservative for small
                subsample=0.8,  # Row sampling to prevent overfitting (like XGBoost)
                colsample_bytree=0.8,  # Feature sampling for high-dim data (like XGBoost)
                reg_alpha=0.1,  # L1 regularization for feature selection (like XGBoost)
                reg_lambda=1.0,  # L2 regularization to prevent overfitting (like XGBoost)
                random_state=42,
                **exec_defaults
            )

        elif model_name == "CatBoost":
            if not HAS_CATBOOST:
                raise ValueError("CatBoost is not available. Install Visual Studio 2022 Build Tools and run: pip install catboost")
            return CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )

        else:
            raise ValueError(f"Unknown regression model: {model_name}")

    else:  # classification
        if model_name in ["PLS-DA", "PLS"]:
            # For classification, PLS is used as a transformer
            return PLSTransformer(n_components=n_components, scale=False)

        elif model_name == "RandomForest":
            exec_defaults = get_execution_defaults('RandomForest')
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                **exec_defaults
            )

        elif model_name == "MLP":
            exec_defaults = get_execution_defaults('MLP')
            return MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                random_state=42,
                **exec_defaults
            )

        elif model_name == "NeuralBoosted":
            exec_defaults = get_execution_defaults('NeuralBoosted')
            return NeuralBoostedClassifier(
                n_estimators=100,
                learning_rate=0.1,
                hidden_layer_size=5,
                activation='tanh',
                early_stopping_metric='accuracy',  # User-selectable metric (default: accuracy)
                class_weight=None,  # Can be 'balanced' for imbalanced datasets
                random_state=42,
                **exec_defaults
            )

        elif model_name == "SVM":
            return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)

        elif model_name == "XGBoost":
            exec_defaults = get_execution_defaults('XGBoost')
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,  # Better default for spectroscopy (correlated samples)
                colsample_bytree=0.8,  # Better default for 2000+ features (tree diversity)
                reg_alpha=0.1,  # L1 regularization for implicit feature selection
                random_state=42,
                **exec_defaults
            )

        elif model_name == "LightGBM":
            exec_defaults = get_execution_defaults('LightGBM')
            return LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=15,  # Reduced for small datasets (was 31)
                max_depth=-1,  # No limit (controlled by num_leaves)
                min_child_samples=10,  # Balanced: 5 too aggressive for large datasets, 20 too conservative for small
                subsample=0.8,  # Row sampling like XGBoost
                colsample_bytree=0.8,  # Feature sampling for high-dimensional data
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                **exec_defaults
            )

        elif model_name == "CatBoost":
            if not HAS_CATBOOST:
                raise ValueError("CatBoost is not available. Install Visual Studio 2022 Build Tools and run: pip install catboost")
            return CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )

        else:
            raise ValueError(f"Unknown classification model: {model_name}")

    return model


def get_model_grids(task_type, n_features, max_n_components=8, max_iter=500,
                    n_estimators_list=None, learning_rates=None,
                    pls_max_iter_list=None, pls_tol_list=None,
                    rf_n_trees_list=None,
                    rf_max_depth_list=None, rf_min_samples_split_list=None,
                    rf_min_samples_leaf_list=None, rf_max_features_list=None,
                    rf_bootstrap_list=None, rf_max_leaf_nodes_list=None,
                    rf_min_impurity_decrease_list=None,
                    ridge_alphas_list=None, ridge_solver_list=None, ridge_tol_list=None,
                    lasso_alphas_list=None, lasso_selection_list=None, lasso_tol_list=None,
                    xgb_n_estimators_list=None, xgb_learning_rates=None, xgb_max_depths=None,
                    xgb_subsample=None, xgb_colsample_bytree=None, xgb_reg_alpha=None, xgb_reg_lambda=None,
                    xgb_min_child_weight_list=None, xgb_gamma_list=None,
                    elasticnet_alphas_list=None, elasticnet_l1_ratios=None,
                    elasticnet_selection_list=None, elasticnet_tol_list=None,
                    lightgbm_n_estimators_list=None, lightgbm_learning_rates=None, lightgbm_num_leaves_list=None,
                    lightgbm_max_depth_list=None, lightgbm_min_child_samples_list=None,
                    lightgbm_subsample_list=None, lightgbm_colsample_bytree_list=None,
                    lightgbm_reg_alpha_list=None, lightgbm_reg_lambda_list=None,
                    catboost_iterations_list=None, catboost_learning_rates=None, catboost_depths=None,
                    catboost_l2_leaf_reg_list=None, catboost_border_count_list=None,
                    catboost_bagging_temperature_list=None, catboost_random_strength_list=None,
                    svr_kernels=None, svr_C_list=None, svr_gamma_list=None,
                    svr_epsilon_list=None, svr_degree_list=None, svr_coef0_list=None, svr_shrinking_list=None,
                    mlp_hidden_layer_sizes_list=None, mlp_alphas_list=None, mlp_learning_rate_inits=None,
                    mlp_activation_list=None, mlp_solver_list=None, mlp_batch_size_list=None,
                    mlp_learning_rate_schedule_list=None, mlp_momentum_list=None,
                    tier='standard', enabled_models=None):
    """
    Get model grids for hyperparameter search with tiered defaults.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'
    n_features : int
        Number of input features
    max_n_components : int, default=8
        Maximum number of PLS components to test
    max_iter : int, default=500
        Maximum iterations for MLP
    n_estimators_list : list of int, optional
        List of n_estimators values for NeuralBoosted. If None, uses tier defaults
    learning_rates : list of float, optional
        List of learning rates for NeuralBoosted. If None, uses tier defaults
    rf_n_trees_list : list of int, optional
        List of n_estimators values for RandomForest. If None, uses tier defaults
    rf_max_depth_list : list of int or None, optional
        List of max_depth values for RandomForest. If None, uses tier defaults
    ridge_alphas_list : list of float, optional
        List of alpha values for Ridge. If None, uses tier defaults
    lasso_alphas_list : list of float, optional
        List of alpha values for Lasso. If None, uses tier defaults
    tier : str, default='standard'
        Model tier: 'quick', 'standard', 'comprehensive', or 'experimental'
        This sets optimized defaults for all hyperparameters
    enabled_models : list of str, optional
        List of specific models to include. If None, uses all models in tier

    Returns
    -------
    grids : dict
        Dictionary mapping model names to lists of (model, param_dict) tuples

    Notes
    -----
    The tier system provides optimized defaults:
    - 'quick': Minimal configs for rapid testing (3-5 min)
    - 'standard': Core models with balanced grids (10-15 min) [DEFAULT]
    - 'comprehensive': Extended grids for thorough analysis (20-30 min)
    - 'experimental': All models with full grids (45+ min)

    You can override any tier defaults by specifying explicit hyperparameter lists.
    """
    # Get tier-specific hyperparameters from config
    # Users can override any of these by passing explicit lists

    # Determine which models to include
    if enabled_models is None:
        # Use tier defaults if no explicit model list provided
        enabled_models = get_tier_models(tier)

    # NeuralBoosted defaults (tier-aware)
    if n_estimators_list is None:
        nb_config = get_hyperparameters('NeuralBoosted', tier)
        n_estimators_list = nb_config.get('n_estimators', [100])
    if learning_rates is None:
        nb_config = get_hyperparameters('NeuralBoosted', tier)
        learning_rates = nb_config.get('learning_rate', [0.1, 0.2])

    # RandomForest defaults (tier-aware)
    if rf_n_trees_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_n_trees_list = rf_config.get('n_estimators', [200, 500])
    if rf_max_depth_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_max_depth_list = rf_config.get('max_depth', [None, 30])
    if rf_min_samples_split_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_min_samples_split_list = rf_config.get('min_samples_split', [2])
    if rf_min_samples_leaf_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_min_samples_leaf_list = rf_config.get('min_samples_leaf', [1])
    if rf_max_features_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_max_features_list = rf_config.get('max_features', ['sqrt'])
    if rf_bootstrap_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_bootstrap_list = rf_config.get('bootstrap', [True])
    if rf_max_leaf_nodes_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_max_leaf_nodes_list = rf_config.get('max_leaf_nodes', [None])
    if rf_min_impurity_decrease_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_min_impurity_decrease_list = rf_config.get('min_impurity_decrease', [0.0])

    # Ridge defaults (tier-aware)
    if ridge_alphas_list is None:
        ridge_config = get_hyperparameters('Ridge', tier)
        ridge_alphas_list = ridge_config.get('alpha', [0.01, 0.1, 1.0, 10.0])
    if ridge_solver_list is None:
        ridge_config = get_hyperparameters('Ridge', tier)
        ridge_solver_list = ridge_config.get('solver', ['auto'])
    if ridge_tol_list is None:
        ridge_config = get_hyperparameters('Ridge', tier)
        ridge_tol_list = ridge_config.get('tol', [1e-4])

    # Lasso defaults (tier-aware)
    if lasso_alphas_list is None:
        lasso_config = get_hyperparameters('Lasso', tier)
        lasso_alphas_list = lasso_config.get('alpha', [0.01, 0.1, 1.0])
    if lasso_selection_list is None:
        lasso_config = get_hyperparameters('Lasso', tier)
        lasso_selection_list = lasso_config.get('selection', ['cyclic'])
    if lasso_tol_list is None:
        lasso_config = get_hyperparameters('Lasso', tier)
        lasso_tol_list = lasso_config.get('tol', [1e-4])

    # XGBoost defaults (tier-aware)
    if xgb_n_estimators_list is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_n_estimators_list = xgb_config.get('n_estimators', [100, 200])
    if xgb_learning_rates is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_learning_rates = xgb_config.get('learning_rate', [0.05, 0.1])
    if xgb_max_depths is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_max_depths = xgb_config.get('max_depth', [3, 6])
    if xgb_subsample is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_subsample = xgb_config.get('subsample', [0.8, 1.0])
    if xgb_colsample_bytree is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_colsample_bytree = xgb_config.get('colsample_bytree', [0.8, 1.0])
    if xgb_reg_alpha is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_reg_alpha = xgb_config.get('reg_alpha', [0, 0.1])
    if xgb_reg_lambda is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_reg_lambda = xgb_config.get('reg_lambda', [1.0])  # Default if not in tier
    if xgb_min_child_weight_list is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_min_child_weight_list = xgb_config.get('min_child_weight', [1])
    if xgb_gamma_list is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_gamma_list = xgb_config.get('gamma', [0])

    # ElasticNet defaults (tier-aware)
    if elasticnet_alphas_list is None:
        en_config = get_hyperparameters('ElasticNet', tier)
        elasticnet_alphas_list = en_config.get('alpha', [0.01, 0.1, 1.0])
    if elasticnet_l1_ratios is None:
        en_config = get_hyperparameters('ElasticNet', tier)
        elasticnet_l1_ratios = en_config.get('l1_ratio', [0.3, 0.5, 0.7])
    if elasticnet_selection_list is None:
        en_config = get_hyperparameters('ElasticNet', tier)
        elasticnet_selection_list = en_config.get('selection', ['cyclic'])
    if elasticnet_tol_list is None:
        en_config = get_hyperparameters('ElasticNet', tier)
        elasticnet_tol_list = en_config.get('tol', [1e-4])

    # LightGBM defaults (tier-aware)
    if lightgbm_n_estimators_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_n_estimators_list = lgbm_config.get('n_estimators', [100, 200])
    if lightgbm_learning_rates is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_learning_rates = lgbm_config.get('learning_rate', [0.1])
    if lightgbm_num_leaves_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_num_leaves_list = lgbm_config.get('num_leaves', [31, 50])
    if lightgbm_max_depth_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_max_depth_list = lgbm_config.get('max_depth', [-1])
    if lightgbm_min_child_samples_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_min_child_samples_list = lgbm_config.get('min_child_samples', [10])
    if lightgbm_subsample_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_subsample_list = lgbm_config.get('subsample', [0.8])
    if lightgbm_colsample_bytree_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_colsample_bytree_list = lgbm_config.get('colsample_bytree', [0.8])
    if lightgbm_reg_alpha_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_reg_alpha_list = lgbm_config.get('reg_alpha', [0.1])
    if lightgbm_reg_lambda_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_reg_lambda_list = lgbm_config.get('reg_lambda', [1.0])

    # CatBoost defaults (tier-aware)
    if catboost_iterations_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_iterations_list = cb_config.get('iterations', [100, 200])
    if catboost_learning_rates is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_learning_rates = cb_config.get('learning_rate', [0.1])
    if catboost_depths is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_depths = cb_config.get('depth', [4, 6])
    if catboost_l2_leaf_reg_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_l2_leaf_reg_list = cb_config.get('l2_leaf_reg', [3.0])
    if catboost_border_count_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_border_count_list = cb_config.get('border_count', [128])
    if catboost_bagging_temperature_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_bagging_temperature_list = cb_config.get('bagging_temperature', [1.0])
    if catboost_random_strength_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_random_strength_list = cb_config.get('random_strength', [1.0])

    # SVR defaults (tier-aware)
    if svr_kernels is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_kernels = svr_config.get('kernel', ['rbf', 'linear'])
    if svr_C_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_C_list = svr_config.get('C', [1.0, 10.0])
    if svr_gamma_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_gamma_list = svr_config.get('gamma', ['scale'])
    if svr_epsilon_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_epsilon_list = svr_config.get('epsilon', [0.1])
    if svr_degree_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_degree_list = svr_config.get('degree', [3])
    if svr_coef0_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_coef0_list = svr_config.get('coef0', [0.0])
    if svr_shrinking_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_shrinking_list = svr_config.get('shrinking', [True])

    # MLP defaults (tier-aware)
    if mlp_hidden_layer_sizes_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_hidden_layer_sizes_list = mlp_config.get('hidden_layer_sizes', [(64,), (128, 64)])
    if mlp_alphas_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_alphas_list = mlp_config.get('alpha', [1e-3])
    if mlp_learning_rate_inits is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_learning_rate_inits = mlp_config.get('learning_rate_init', [1e-3])
    if mlp_activation_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_activation_list = mlp_config.get('activation', ['relu'])
    if mlp_solver_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_solver_list = mlp_config.get('solver', ['adam'])
    if mlp_batch_size_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_batch_size_list = mlp_config.get('batch_size', ['auto'])
    if mlp_learning_rate_schedule_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_learning_rate_schedule_list = mlp_config.get('learning_rate_schedule', ['constant'])
    if mlp_momentum_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_momentum_list = mlp_config.get('momentum', [0.9])

    grids = {}

    # PLS components grid - test ALL integer values from 2 to max allowed
    # Max is limited by both n_features and max_n_components (which is adjusted for CV fold size)
    pls_max = min(n_features, max_n_components)
    pls_components = list(range(2, pls_max + 1)) if pls_max >= 2 else [2]

    # PLS additional hyperparameters (tier-aware)
    if pls_max_iter_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_max_iter_list = pls_config.get('max_iter', [500])
    if pls_tol_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_tol_list = pls_config.get('tol', [1e-6])

    if task_type == "regression":
        # PLS Regression (only if in enabled_models)
        if 'PLS' in enabled_models:
            pls_configs = []
            for nc in pls_components:
                for max_iter_val in pls_max_iter_list:
                    for tol_val in pls_tol_list:
                        pls_configs.append(
                            (
                                PLSRegression(
                                    n_components=nc,
                                    max_iter=max_iter_val,
                                    tol=tol_val,
                                    scale=False
                                ),
                                {
                                    "n_components": nc,
                                    "max_iter": max_iter_val,
                                    "tol": tol_val
                                }
                            )
                        )
            grids["PLS"] = pls_configs

        # Ridge Regression (tier-aware)
        if 'Ridge' in enabled_models:
            ridge_configs = []
            for alpha in ridge_alphas_list:
                for solver_val in ridge_solver_list:
                    for tol_val in ridge_tol_list:
                        ridge_configs.append(
                            (
                                Ridge(
                                    alpha=alpha,
                                    solver=solver_val,
                                    tol=tol_val,
                                    random_state=42
                                ),
                                {
                                    "alpha": alpha,
                                    "solver": solver_val,
                                    "tol": tol_val
                                }
                            )
                        )
            grids["Ridge"] = ridge_configs

        # Lasso Regression (tier-aware)
        if 'Lasso' in enabled_models:
            lasso_configs = []
            for alpha in lasso_alphas_list:
                for selection_val in lasso_selection_list:
                    for tol_val in lasso_tol_list:
                        lasso_configs.append(
                            (
                                Lasso(
                                    alpha=alpha,
                                    selection=selection_val,
                                    tol=tol_val,
                                    max_iter=max_iter,
                                    random_state=42
                                ),
                                {
                                    "alpha": alpha,
                                    "selection": selection_val,
                                    "tol": tol_val
                                }
                            )
                        )
            grids["Lasso"] = lasso_configs

        # ElasticNet Regression (tier-aware with UI overrides)
        if 'ElasticNet' in enabled_models:
            elasticnet_configs = []
            for alpha in elasticnet_alphas_list:
                for l1_ratio in elasticnet_l1_ratios:
                    for selection_val in elasticnet_selection_list:
                        for tol_val in elasticnet_tol_list:
                            elasticnet_configs.append(
                                (
                                    ElasticNet(
                                        alpha=alpha,
                                        l1_ratio=l1_ratio,
                                        selection=selection_val,
                                        tol=tol_val,
                                        max_iter=max_iter,
                                        random_state=42
                                    ),
                                    {
                                        "alpha": alpha,
                                        "l1_ratio": l1_ratio,
                                        "selection": selection_val,
                                        "tol": tol_val
                                    }
                                )
                            )
            grids["ElasticNet"] = elasticnet_configs

        # Random Forest (tier-aware)
        if 'RandomForest' in enabled_models:
            rf_configs = []
            for n_est in rf_n_trees_list:
                for max_d in rf_max_depth_list:
                    for min_split in rf_min_samples_split_list:
                        for min_leaf in rf_min_samples_leaf_list:
                            for max_feat in rf_max_features_list:
                                for bootstrap in rf_bootstrap_list:
                                    for max_leaf_nodes in rf_max_leaf_nodes_list:
                                        for min_impurity_dec in rf_min_impurity_decrease_list:
                                            rf_configs.append(
                                                (
                                                    RandomForestRegressor(
                                                        n_estimators=n_est,
                                                        max_depth=max_d,
                                                        min_samples_split=min_split,
                                                        min_samples_leaf=min_leaf,
                                                        max_features=max_feat,
                                                        bootstrap=bootstrap,
                                                        max_leaf_nodes=max_leaf_nodes,
                                                        min_impurity_decrease=min_impurity_dec,
                                                        random_state=42,
                                                        n_jobs=-1
                                                    ),
                                                    {
                                                        "n_estimators": n_est,
                                                        "max_depth": max_d,
                                                        "min_samples_split": min_split,
                                                        "min_samples_leaf": min_leaf,
                                                        "max_features": max_feat,
                                                        "bootstrap": bootstrap,
                                                        "max_leaf_nodes": max_leaf_nodes,
                                                        "min_impurity_decrease": min_impurity_dec
                                                    },
                                                )
                                            )
            grids["RandomForest"] = rf_configs

        # MLP (tier-aware with UI overrides)
        if 'MLP' in enabled_models:
            mlp_configs = []
            for hidden in mlp_hidden_layer_sizes_list:
                for alpha in mlp_alphas_list:
                    for lr_init in mlp_learning_rate_inits:
                        for activation in mlp_activation_list:
                            for solver in mlp_solver_list:
                                for batch_size in mlp_batch_size_list:
                                    for lr_schedule in mlp_learning_rate_schedule_list:
                                        # momentum only applies when solver='sgd'
                                        if solver == 'sgd':
                                            for momentum in mlp_momentum_list:
                                                mlp_configs.append(
                                                    (
                                                        MLPRegressor(
                                                            hidden_layer_sizes=hidden,
                                                            alpha=alpha,
                                                            learning_rate_init=lr_init,
                                                            activation=activation,
                                                            solver=solver,
                                                            batch_size=batch_size,
                                                            learning_rate=lr_schedule,
                                                            momentum=momentum,
                                                            max_iter=max_iter,
                                                            random_state=42,
                                                            early_stopping=True,
                                                        ),
                                                        {
                                                            "hidden_layer_sizes": hidden,
                                                            "alpha": alpha,
                                                            "learning_rate_init": lr_init,
                                                            "activation": activation,
                                                            "solver": solver,
                                                            "batch_size": batch_size,
                                                            "learning_rate": lr_schedule,
                                                            "momentum": momentum,
                                                        },
                                                    )
                                                )
                                        else:
                                            # For non-sgd solvers, don't include momentum
                                            mlp_configs.append(
                                                (
                                                    MLPRegressor(
                                                        hidden_layer_sizes=hidden,
                                                        alpha=alpha,
                                                        learning_rate_init=lr_init,
                                                        activation=activation,
                                                        solver=solver,
                                                        batch_size=batch_size,
                                                        learning_rate=lr_schedule,
                                                        max_iter=max_iter,
                                                        random_state=42,
                                                        early_stopping=True,
                                                    ),
                                                    {
                                                        "hidden_layer_sizes": hidden,
                                                        "alpha": alpha,
                                                        "learning_rate_init": lr_init,
                                                        "activation": activation,
                                                        "solver": solver,
                                                        "batch_size": batch_size,
                                                        "learning_rate": lr_schedule,
                                                    },
                                                )
                                            )
            grids["MLP"] = mlp_configs

        # Neural Boosted Regression (tier-aware)
        if 'NeuralBoosted' in enabled_models:
            nb_config = get_hyperparameters('NeuralBoosted', tier)
            hidden_sizes = nb_config.get('hidden_layer_size', [3, 5])
            activations = nb_config.get('activation', ['tanh', 'identity'])

            nbr_configs = []
            for n_est in n_estimators_list:
                for lr in learning_rates:
                    for hidden in hidden_sizes:
                        for activation in activations:
                            nbr_configs.append(
                                (
                                    NeuralBoostedRegressor(
                                        n_estimators=n_est,
                                        learning_rate=lr,
                                        hidden_layer_size=hidden,
                                        activation=activation,
                                        early_stopping=True,
                                        validation_fraction=0.15,
                                        n_iter_no_change=10,
                                        alpha=1e-4,
                                        random_state=42,
                                        verbose=0
                                    ),
                                    {
                                        "n_estimators": n_est,
                                        "learning_rate": lr,
                                        "hidden_layer_size": hidden,
                                        "activation": activation
                                    }
                                )
                            )
            grids["NeuralBoosted"] = nbr_configs

        # Support Vector Regression (SVR) - tier-aware
        if 'SVR' in enabled_models:
            svr_config = get_hyperparameters('SVR', tier)
            svr_kernels = svr_config.get('kernel', ['rbf', 'linear'])
            svr_Cs = svr_config.get('C', [1.0, 10.0])
            svr_gammas = svr_config.get('gamma', ['scale'])

            svr_configs = []
            for kernel in svr_kernels:
                for C in svr_Cs:
                    for epsilon in svr_epsilon_list:
                        for shrinking in svr_shrinking_list:
                            # gamma applies to rbf and poly kernels
                            if kernel in ['rbf', 'poly']:
                                for gamma in svr_gammas:
                                    # degree only applies to poly kernel
                                    if kernel == 'poly':
                                        for degree in svr_degree_list:
                                            for coef0 in svr_coef0_list:
                                                svr_configs.append(
                                                    (
                                                        SVR(
                                                            kernel=kernel,
                                                            C=C,
                                                            epsilon=epsilon,
                                                            gamma=gamma,
                                                            degree=degree,
                                                            coef0=coef0,
                                                            shrinking=shrinking
                                                        ),
                                                        {
                                                            "kernel": kernel,
                                                            "C": C,
                                                            "epsilon": epsilon,
                                                            "gamma": gamma,
                                                            "degree": degree,
                                                            "coef0": coef0,
                                                            "shrinking": shrinking
                                                        }
                                                    )
                                                )
                                    else:  # rbf kernel
                                        svr_configs.append(
                                            (
                                                SVR(
                                                    kernel=kernel,
                                                    C=C,
                                                    epsilon=epsilon,
                                                    gamma=gamma,
                                                    shrinking=shrinking
                                                ),
                                                {
                                                    "kernel": kernel,
                                                    "C": C,
                                                    "epsilon": epsilon,
                                                    "gamma": gamma,
                                                    "shrinking": shrinking
                                                }
                                            )
                                        )
                            elif kernel == 'sigmoid':
                                # sigmoid uses coef0 but not gamma or degree
                                for coef0 in svr_coef0_list:
                                    svr_configs.append(
                                        (
                                            SVR(
                                                kernel=kernel,
                                                C=C,
                                                epsilon=epsilon,
                                                coef0=coef0,
                                                shrinking=shrinking
                                            ),
                                            {
                                                "kernel": kernel,
                                                "C": C,
                                                "epsilon": epsilon,
                                                "coef0": coef0,
                                                "shrinking": shrinking
                                            }
                                        )
                                    )
                            else:  # linear or other kernels
                                svr_configs.append(
                                    (
                                        SVR(
                                            kernel=kernel,
                                            C=C,
                                            epsilon=epsilon,
                                            shrinking=shrinking
                                        ),
                                        {
                                            "kernel": kernel,
                                            "C": C,
                                            "epsilon": epsilon,
                                            "shrinking": shrinking
                                        }
                                    )
                                )
            grids["SVR"] = svr_configs

        # XGBoost Regression - tier-aware with UI overrides
        if 'XGBoost' in enabled_models:
            xgb_configs = []
            for n_est in xgb_n_estimators_list:
                for lr in xgb_learning_rates:
                    for max_depth in xgb_max_depths:
                        for subsample in xgb_subsample:
                            for colsample in xgb_colsample_bytree:
                                for reg_alpha in xgb_reg_alpha:
                                    for reg_lambda in xgb_reg_lambda:
                                        for min_child_weight in xgb_min_child_weight_list:
                                            for gamma in xgb_gamma_list:
                                                xgb_configs.append(
                                                    (
                                                        XGBRegressor(
                                                            n_estimators=n_est,
                                                            learning_rate=lr,
                                                            max_depth=max_depth,
                                                            subsample=subsample,
                                                            colsample_bytree=colsample,
                                                            reg_alpha=reg_alpha,
                                                            reg_lambda=reg_lambda,
                                                            min_child_weight=min_child_weight,
                                                            gamma=gamma,
                                                            tree_method='hist',  # Faster for high-dimensional data
                                                            random_state=42,
                                                            n_jobs=-1,
                                                            verbosity=0
                                                        ),
                                                        {
                                                            "n_estimators": n_est,
                                                            "learning_rate": lr,
                                                            "max_depth": max_depth,
                                                            "subsample": subsample,
                                                            "colsample_bytree": colsample,
                                                            "reg_alpha": reg_alpha,
                                                            "reg_lambda": reg_lambda,
                                                            "min_child_weight": min_child_weight,
                                                            "gamma": gamma,
                                                            "tree_method": "hist"
                                                        }
                                                    )
                                                )
            grids["XGBoost"] = xgb_configs

        # LightGBM Regression - tier-aware with UI overrides
        if 'LightGBM' in enabled_models:
            lgbm_configs = []
            for n_est in lightgbm_n_estimators_list:
                for lr in lightgbm_learning_rates:
                    for num_leaves in lightgbm_num_leaves_list:
                        for max_depth in lightgbm_max_depth_list:
                            for min_child_samples in lightgbm_min_child_samples_list:
                                for subsample in lightgbm_subsample_list:
                                    for colsample_bytree in lightgbm_colsample_bytree_list:
                                        for reg_alpha in lightgbm_reg_alpha_list:
                                            for reg_lambda in lightgbm_reg_lambda_list:
                                                lgbm_configs.append(
                                                    (
                                                        LGBMRegressor(
                                                            n_estimators=n_est,
                                                            learning_rate=lr,
                                                            num_leaves=num_leaves,
                                                            max_depth=max_depth,
                                                            min_child_samples=min_child_samples,
                                                            subsample=subsample,
                                                            bagging_freq=1,  # Required when subsample < 1.0
                                                            colsample_bytree=colsample_bytree,
                                                            reg_alpha=reg_alpha,
                                                            reg_lambda=reg_lambda,
                                                            random_state=42,
                                                            n_jobs=-1,
                                                            verbosity=-1
                                                        ),
                                                        {
                                                            "n_estimators": n_est,
                                                            "learning_rate": lr,
                                                            "num_leaves": num_leaves,
                                                            "max_depth": max_depth,
                                                            "min_child_samples": min_child_samples,
                                                            "subsample": subsample,
                                                            "bagging_freq": 1,
                                                            "colsample_bytree": colsample_bytree,
                                                            "reg_alpha": reg_alpha,
                                                            "reg_lambda": reg_lambda
                                                        }
                                                    )
                                                )
            grids["LightGBM"] = lgbm_configs

        # CatBoost Regression - tier-aware with UI overrides (optional - requires Visual Studio)
        if 'CatBoost' in enabled_models and HAS_CATBOOST:
            catboost_configs = []
            for iterations in catboost_iterations_list:
                for lr in catboost_learning_rates:
                    for depth in catboost_depths:
                        for l2_leaf_reg in catboost_l2_leaf_reg_list:
                            for border_count in catboost_border_count_list:
                                for bagging_temp in catboost_bagging_temperature_list:
                                    for random_str in catboost_random_strength_list:
                                        catboost_configs.append(
                                            (
                                                CatBoostRegressor(
                                                    iterations=iterations,
                                                    learning_rate=lr,
                                                    depth=depth,
                                                    l2_leaf_reg=l2_leaf_reg,
                                                    border_count=border_count,
                                                    bagging_temperature=bagging_temp,
                                                    random_strength=random_str,
                                                    random_state=42,
                                                    verbose=False
                                                ),
                                                {
                                                    "iterations": iterations,
                                                    "learning_rate": lr,
                                                    "depth": depth,
                                                    "l2_leaf_reg": l2_leaf_reg,
                                                    "border_count": border_count,
                                                    "bagging_temperature": bagging_temp,
                                                    "random_strength": random_str
                                                }
                                            )
                                        )
            grids["CatBoost"] = catboost_configs
        elif 'CatBoost' in enabled_models and not HAS_CATBOOST:
            print("Warning: CatBoost requested but not available. Skipping CatBoost models.")

    else:  # classification
        # PLS-DA (PLS + LogisticRegression) - tier-aware
        if 'PLS-DA' in enabled_models or 'PLS' in enabled_models:
            pls_da_configs = []
            for nc in pls_components:
                for max_iter_val in pls_max_iter_list:
                    for tol_val in pls_tol_list:
                        pls_da_configs.append(
                            (
                                PLSTransformer(
                                    n_components=nc,
                                    max_iter=max_iter_val,
                                    tol=tol_val,
                                    scale=False
                                ),
                                {
                                    "n_components": nc,
                                    "max_iter": max_iter_val,
                                    "tol": tol_val
                                }
                            )
                        )
            grids["PLS-DA"] = pls_da_configs

        # Random Forest Classifier - tier-aware
        if 'RandomForest' in enabled_models:
            rf_configs = []
            for n_est in rf_n_trees_list:
                for max_d in rf_max_depth_list:
                    for min_split in rf_min_samples_split_list:
                        for min_leaf in rf_min_samples_leaf_list:
                            for max_feat in rf_max_features_list:
                                for bootstrap in rf_bootstrap_list:
                                    for max_leaf_nodes in rf_max_leaf_nodes_list:
                                        for min_impurity_dec in rf_min_impurity_decrease_list:
                                            rf_configs.append(
                                                (
                                                    RandomForestClassifier(
                                                        n_estimators=n_est,
                                                        max_depth=max_d,
                                                        min_samples_split=min_split,
                                                        min_samples_leaf=min_leaf,
                                                        max_features=max_feat,
                                                        bootstrap=bootstrap,
                                                        max_leaf_nodes=max_leaf_nodes,
                                                        min_impurity_decrease=min_impurity_dec,
                                                        random_state=42,
                                                        n_jobs=-1
                                                    ),
                                                    {
                                                        "n_estimators": n_est,
                                                        "max_depth": max_d,
                                                        "min_samples_split": min_split,
                                                        "min_samples_leaf": min_leaf,
                                                        "max_features": max_feat,
                                                        "bootstrap": bootstrap,
                                                        "max_leaf_nodes": max_leaf_nodes,
                                                        "min_impurity_decrease": min_impurity_dec
                                                    },
                                                )
                                            )
            grids["RandomForest"] = rf_configs

        # MLP Classifier - tier-aware with UI overrides
        if 'MLP' in enabled_models:
            mlp_configs = []
            for hidden in mlp_hidden_layer_sizes_list:
                for alpha in mlp_alphas_list:
                    for lr_init in mlp_learning_rate_inits:
                        for activation in mlp_activation_list:
                            for solver in mlp_solver_list:
                                for batch_size in mlp_batch_size_list:
                                    for lr_schedule in mlp_learning_rate_schedule_list:
                                        # momentum only applies when solver='sgd'
                                        if solver == 'sgd':
                                            for momentum in mlp_momentum_list:
                                                mlp_configs.append(
                                                    (
                                                        MLPClassifier(
                                                            hidden_layer_sizes=hidden,
                                                            alpha=alpha,
                                                            learning_rate_init=lr_init,
                                                            activation=activation,
                                                            solver=solver,
                                                            batch_size=batch_size,
                                                            learning_rate=lr_schedule,
                                                            momentum=momentum,
                                                            max_iter=max_iter,
                                                            random_state=42,
                                                            early_stopping=True,
                                                        ),
                                                        {
                                                            "hidden_layer_sizes": hidden,
                                                            "alpha": alpha,
                                                            "learning_rate_init": lr_init,
                                                            "activation": activation,
                                                            "solver": solver,
                                                            "batch_size": batch_size,
                                                            "learning_rate": lr_schedule,
                                                            "momentum": momentum,
                                                        },
                                                    )
                                                )
                                        else:
                                            # For non-sgd solvers, don't include momentum
                                            mlp_configs.append(
                                                (
                                                    MLPClassifier(
                                                        hidden_layer_sizes=hidden,
                                                        alpha=alpha,
                                                        learning_rate_init=lr_init,
                                                        activation=activation,
                                                        solver=solver,
                                                        batch_size=batch_size,
                                                        learning_rate=lr_schedule,
                                                        max_iter=max_iter,
                                                        random_state=42,
                                                        early_stopping=True,
                                                    ),
                                                    {
                                                        "hidden_layer_sizes": hidden,
                                                        "alpha": alpha,
                                                        "learning_rate_init": lr_init,
                                                        "activation": activation,
                                                        "solver": solver,
                                                        "batch_size": batch_size,
                                                        "learning_rate": lr_schedule,
                                                    },
                                                )
                                            )
            grids["MLP"] = mlp_configs

        # Neural Boosted Classifier - tier-aware (optimized grid size)
        if 'NeuralBoosted' in enabled_models:
            nb_config = get_hyperparameters('NeuralBoosted', tier)
            hidden_sizes = nb_config.get('hidden_layer_size', [3, 5])  # Aligned with regression
            activations = nb_config.get('activation', ['tanh', 'identity'])  # Aligned with regression

            nbc_configs = []
            for n_est in n_estimators_list:
                for lr in learning_rates:
                    for hidden in hidden_sizes:
                        for activation in activations:
                            # NOTE: early_stopping_metric and class_weight removed from grid
                            # These are user settings, not hyperparameters to search
                            # Users can modify via GUI if needed
                            nbc_configs.append(
                                (
                                    NeuralBoostedClassifier(
                                        n_estimators=n_est,
                                        learning_rate=lr,
                                        hidden_layer_size=hidden,
                                        activation=activation,
                                        early_stopping=True,
                                        validation_fraction=0.15,
                                        n_iter_no_change=10,
                                        early_stopping_metric='accuracy',  # Default (user preference)
                                        class_weight=None,  # Default (users enable 'balanced' if needed)
                                        alpha=1e-4,
                                        random_state=42,
                                        verbose=0
                                    ),
                                    {
                                        "n_estimators": n_est,
                                        "learning_rate": lr,
                                        "hidden_layer_size": hidden,
                                        "activation": activation
                                    }
                                )
                            )
            grids["NeuralBoosted"] = nbc_configs

        # Support Vector Machine (SVM) for classification - tier-aware
        if 'SVM' in enabled_models or 'SVR' in enabled_models:  # SVR config works for SVM too
            svm_config = get_hyperparameters('SVR', tier)  # Reuse SVR config
            svm_kernels = svm_config.get('kernel', ['rbf', 'linear'])
            svm_Cs = svm_config.get('C', [1.0, 10.0])
            svm_gammas = svm_config.get('gamma', ['scale'])

            svm_configs = []
            for kernel in svm_kernels:
                for C in svm_Cs:
                    if kernel == 'rbf':
                        for gamma in svm_gammas:
                            svm_configs.append(
                                (
                                    SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42),
                                    {"kernel": kernel, "C": C, "gamma": gamma}
                                )
                            )
                    else:
                        svm_configs.append(
                            (
                                SVC(kernel=kernel, C=C, probability=True, random_state=42),
                                {"kernel": kernel, "C": C}
                            )
                        )
            grids["SVM"] = svm_configs

        # XGBoost Classification - tier-aware with UI overrides
        if 'XGBoost' in enabled_models:
            xgb_configs = []
            for n_est in xgb_n_estimators_list:
                for lr in xgb_learning_rates:
                    for max_depth in xgb_max_depths:
                        for subsample in xgb_subsample:
                            for colsample in xgb_colsample_bytree:
                                for reg_alpha in xgb_reg_alpha:
                                    for reg_lambda in xgb_reg_lambda:
                                        xgb_configs.append(
                                            (
                                                XGBClassifier(
                                                    n_estimators=n_est,
                                                    learning_rate=lr,
                                                    max_depth=max_depth,
                                                    subsample=subsample,
                                                    colsample_bytree=colsample,
                                                    reg_alpha=reg_alpha,
                                                    reg_lambda=reg_lambda,
                                                    tree_method='hist',  # Faster for high-dimensional data
                                                    random_state=42,
                                                    n_jobs=-1,
                                                    verbosity=0
                                                ),
                                                {
                                                    "n_estimators": n_est,
                                                    "learning_rate": lr,
                                                    "max_depth": max_depth,
                                                    "subsample": subsample,
                                                    "colsample_bytree": colsample,
                                                    "reg_alpha": reg_alpha,
                                                    "reg_lambda": reg_lambda,
                                                    "tree_method": "hist"
                                                }
                                            )
                                        )
            grids["XGBoost"] = xgb_configs

        # LightGBM Classification - tier-aware with UI overrides
        if 'LightGBM' in enabled_models:
            lgbm_configs = []
            for n_est in lightgbm_n_estimators_list:
                for lr in lightgbm_learning_rates:
                    for num_leaves in lightgbm_num_leaves_list:
                        for max_depth in lightgbm_max_depth_list:
                            for min_child_samples in lightgbm_min_child_samples_list:
                                for subsample in lightgbm_subsample_list:
                                    for colsample_bytree in lightgbm_colsample_bytree_list:
                                        for reg_alpha in lightgbm_reg_alpha_list:
                                            for reg_lambda in lightgbm_reg_lambda_list:
                                                lgbm_configs.append(
                                                    (
                                                        LGBMClassifier(
                                                            n_estimators=n_est,
                                                            learning_rate=lr,
                                                            num_leaves=num_leaves,
                                                            max_depth=max_depth,
                                                            min_child_samples=min_child_samples,
                                                            subsample=subsample,
                                                            bagging_freq=1,  # Required when subsample < 1.0
                                                            colsample_bytree=colsample_bytree,
                                                            reg_alpha=reg_alpha,
                                                            reg_lambda=reg_lambda,
                                                            random_state=42,
                                                            n_jobs=-1,
                                                            verbosity=-1
                                                        ),
                                                        {
                                                            "n_estimators": n_est,
                                                            "learning_rate": lr,
                                                            "num_leaves": num_leaves,
                                                            "max_depth": max_depth,
                                                            "min_child_samples": min_child_samples,
                                                            "subsample": subsample,
                                                            "bagging_freq": 1,
                                                            "colsample_bytree": colsample_bytree,
                                                            "reg_alpha": reg_alpha,
                                                            "reg_lambda": reg_lambda
                                                        }
                                                    )
                                                )
            grids["LightGBM"] = lgbm_configs

        # CatBoost Classification - tier-aware with UI overrides (optional - requires Visual Studio)
        if 'CatBoost' in enabled_models and HAS_CATBOOST:
            catboost_configs = []
            for iterations in catboost_iterations_list:
                for lr in catboost_learning_rates:
                    for depth in catboost_depths:
                        catboost_configs.append(
                            (
                                CatBoostClassifier(
                                    iterations=iterations,
                                    learning_rate=lr,
                                    depth=depth,
                                    random_state=42,
                                    verbose=False
                                ),
                                {"iterations": iterations, "learning_rate": lr, "depth": depth}
                            )
                        )
            grids["CatBoost"] = catboost_configs
        elif 'CatBoost' in enabled_models and not HAS_CATBOOST:
            print("Warning: CatBoost requested but not available. Skipping CatBoost models.")

    return grids


def compute_vip(pls_model, X, y):
    """
    Compute Variable Importance in Projection (VIP) scores for a fitted PLS model.

    Parameters
    ----------
    pls_model : PLSRegression or PLSTransformer
        Fitted PLS model
    X : array-like
        Training data
    y : array-like
        Target values

    Returns
    -------
    vip_scores : ndarray
        VIP score for each variable
    """
    # Handle PLSTransformer wrapper
    if isinstance(pls_model, PLSTransformer):
        pls_model = pls_model.pls_

    # Get PLS components
    W = pls_model.x_weights_  # (n_features, n_components)
    T = pls_model.x_scores_  # (n_samples, n_components)

    # Get explained variance by each component
    # SSY: sum of squares of y explained by each component
    y = np.asarray(y).reshape(-1, 1)
    ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)

    # Total SSY
    ssy_total = np.sum(ssy_comp)

    # VIP calculation (vectorized for performance)
    n_features = W.shape[0]
    n_components = W.shape[1]

    # Vectorized version: same math, but uses broadcasting instead of loop
    weight = np.sum((W ** 2) * ssy_comp, axis=1)  # Sum over components for each feature
    vip_scores = np.sqrt(n_features * weight / ssy_total)

    return vip_scores


def get_feature_importances(model, model_name, X, y):
    """
    Extract feature importances from a fitted model.

    Parameters
    ----------
    model : estimator
        Fitted model
    model_name : str
        Model type name
    X : array-like
        Training data
    y : array-like
        Target values

    Returns
    -------
    importances : ndarray
        Feature importance scores (higher = more important)
    """
    if model_name in ["PLS", "PLS-DA"]:
        # Use VIP scores
        return compute_vip(model, X, y)

    elif model_name in ["Ridge", "Lasso", "ElasticNet"]:
        # Get coefficients (linear models)
        coefs = np.abs(model.coef_)
        if len(coefs.shape) > 1:
            coefs = coefs[0]  # Handle multi-output case

        # Return absolute coefficient values as importance
        return coefs

    elif model_name in ["SVR", "SVM"]:
        # For SVR/SVM, use coefficient-based importance for linear kernel
        # For RBF kernel, use absolute mean of support vector weights
        if hasattr(model, 'coef_') and model.coef_ is not None:
            coefs = np.abs(model.coef_)
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            return coefs
        else:
            # For non-linear kernels, compute feature importance from support vectors
            # This is an approximation: weighted sum of absolute support vectors
            support_vectors = model.support_vectors_
            dual_coef = np.abs(model.dual_coef_)
            if len(dual_coef.shape) > 1:
                dual_coef = dual_coef[0]
            # Weight each support vector by its coefficient and average
            weighted_sv = np.dot(dual_coef, np.abs(support_vectors))
            return weighted_sv / np.sum(weighted_sv)  # Normalize

    elif model_name == "RandomForest":
        # Use built-in feature importances
        return model.feature_importances_

    elif model_name == "MLP":
        # For MLP, use absolute average weight of first layer
        # This is a simple heuristic
        weights = model.coefs_[0]  # (n_features, n_hidden)
        return np.mean(np.abs(weights), axis=1)

    elif model_name == "NeuralBoosted":
        # For Neural Boosted, aggregate importances across all weak learners
        return model.get_feature_importances()

    elif model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        # All gradient boosting models have built-in feature importances
        return model.feature_importances_

    else:
        raise ValueError(f"Unknown model type: {model_name}")
