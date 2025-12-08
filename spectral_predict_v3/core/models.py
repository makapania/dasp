"""
Model definitions and factory for Spectral Predict v3 (standalone).

Forked from v1 - simplified for v3's numpy-first approach.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Optional imports with fallbacks
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    LGBMRegressor = None
    LGBMClassifier = None

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBRegressor = None
    XGBClassifier = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    CatBoostRegressor = None
    CatBoostClassifier = None

# Neural Boosted models (v3 local)
try:
    from .neural_boosted import NeuralBoostedRegressor, NeuralBoostedClassifier
    HAS_NEURAL_BOOSTED = True
except ImportError:
    HAS_NEURAL_BOOSTED = False
    NeuralBoostedRegressor = None
    NeuralBoostedClassifier = None


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS Discriminant Analysis wrapper for classification."""

    def __init__(self, n_components=10):
        self.n_components = n_components
        self._pls = None
        self._classes = None
        self._label_encoder = None

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder

        # Store original classes and encode if string labels
        self._classes = np.unique(y)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._pls = PLSRegression(n_components=n_comp)
        self._pls.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_pred = self._pls.predict(X).ravel()
        # Clip to valid class range and round
        n_classes = len(self._classes)
        y_pred_int = np.round(y_pred).clip(0, n_classes - 1).astype(int)
        # Decode back to original labels
        return self._label_encoder.inverse_transform(y_pred_int)

    @property
    def coef_(self):
        """Expose PLS coefficients for feature importance extraction."""
        if self._pls is not None:
            return self._pls.coef_
        return None

    def get_params(self, deep=True):
        return {'n_components': self.n_components}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def get_model(model_name, task_type='regression', random_state=42, n_jobs=-1, **params):
    """
    Get a model instance by name with specified parameters.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'PLS', 'Ridge', 'RandomForest', 'LightGBM')
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    **params
        Additional model-specific parameters

    Returns
    -------
    model : estimator or None
        sklearn-compatible estimator, or None if model not available
    """
    # Regression models
    if task_type == 'regression':
        if model_name == 'PLS':
            n_components = params.get('n_components', 10)
            return PLSRegression(n_components=n_components)

        elif model_name == 'Ridge':
            alpha = params.get('alpha', 1.0)
            return Ridge(alpha=alpha, random_state=random_state)

        elif model_name == 'Lasso':
            alpha = params.get('alpha', 0.1)
            max_iter = params.get('max_iter', 5000)
            return Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)

        elif model_name == 'ElasticNet':
            alpha = params.get('alpha', 0.1)
            l1_ratio = params.get('l1_ratio', 0.5)
            max_iter = params.get('max_iter', 5000)
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state)

        elif model_name == 'RandomForest':
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs
            )

        elif model_name == 'LightGBM':
            if not HAS_LIGHTGBM:
                return None
            return LGBMRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                min_child_samples=params.get('min_child_samples', 5),  # Match V1 (was 20)
                subsample=params.get('subsample', 0.8),  # Match V1 (was 1.0)
                bagging_freq=1,  # CRITICAL: Required for subsample to work!
                colsample_bytree=params.get('colsample_bytree', 0.8),  # Match V1 (was 1.0)
                reg_alpha=params.get('reg_alpha', 0.1),  # Match V1 (was 0.0)
                reg_lambda=params.get('reg_lambda', 1.0),  # Match V1 (was 0.0)
                random_state=random_state,
                verbose=-1,
                n_jobs=n_jobs
            )

        elif model_name == 'XGBoost':
            if not HAS_XGBOOST:
                return None
            return XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 0.8),  # V1 spectral-optimized (was 1.0)
                colsample_bytree=params.get('colsample_bytree', 0.8),  # V1 spectral-optimized (was 1.0)
                reg_alpha=params.get('reg_alpha', 0.1),  # V1 spectral-optimized (was 0.0)
                reg_lambda=params.get('reg_lambda', 1.0),  # V1 spectral-optimized (was 0.0)
                gamma=params.get('gamma', 0.0),
                min_child_weight=params.get('min_child_weight', 1),
                tree_method=params.get('tree_method', 'hist'),  # V1 spectral-optimized
                random_state=random_state,
                verbosity=0,
                n_jobs=n_jobs
            )

        elif model_name == 'CatBoost':
            if not HAS_CATBOOST:
                return None
            iterations = params.get('iterations', 100)
            learning_rate = params.get('learning_rate', 0.1)
            depth = params.get('depth', 6)
            return CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                random_state=random_state,
                verbose=0
            )

        elif model_name == 'SVR':
            kernel = params.get('kernel', 'rbf')
            C = params.get('C', 1.0)
            max_iter = params.get('max_iter', 5000)
            return SVR(kernel=kernel, C=C, max_iter=max_iter)

        elif model_name == 'NeuralBoosted':
            if not HAS_NEURAL_BOOSTED:
                return None
            n_estimators = params.get('n_estimators', 100)
            learning_rate = params.get('learning_rate', 0.1)
            hidden_layer_size = params.get('hidden_layer_size', 3)
            return NeuralBoostedRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                hidden_layer_size=hidden_layer_size,
                random_state=random_state
            )

        elif model_name == 'MLP':
            hidden_layer_sizes = params.get('hidden_layer_sizes', (100,))
            activation = params.get('activation', 'relu')
            alpha = params.get('alpha', 0.0001)
            max_iter = params.get('max_iter', 2000)
            return MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                n_iter_no_change=10
            )

    # Classification models
    else:
        if model_name == 'PLS-DA':
            n_components = params.get('n_components', 10)
            return PLSDA(n_components=n_components)

        elif model_name == 'RandomForest':
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs
            )

        elif model_name == 'LightGBM':
            if not HAS_LIGHTGBM:
                return None
            return LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 15),  # Classification: 15 (V1 default)
                max_depth=params.get('max_depth', -1),
                min_child_samples=params.get('min_child_samples', 5),  # Match V1 (was 20)
                subsample=params.get('subsample', 0.8),  # Match V1 (was 1.0)
                bagging_freq=1,  # CRITICAL: Required for subsample to work!
                colsample_bytree=params.get('colsample_bytree', 0.8),  # Match V1 (was 1.0)
                reg_alpha=params.get('reg_alpha', 0.1),  # Match V1 (was 0.0)
                reg_lambda=params.get('reg_lambda', 1.0),  # Match V1 (was 0.0)
                random_state=random_state,
                verbose=-1,
                n_jobs=n_jobs
            )

        elif model_name == 'XGBoost':
            if not HAS_XGBOOST:
                return None
            return XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 0.8),  # V1 spectral-optimized (was 1.0)
                colsample_bytree=params.get('colsample_bytree', 0.8),  # V1 spectral-optimized (was 1.0)
                reg_alpha=params.get('reg_alpha', 0.1),  # V1 spectral-optimized (was 0.0)
                reg_lambda=params.get('reg_lambda', 1.0),  # V1 spectral-optimized (was 0.0)
                gamma=params.get('gamma', 0.0),
                min_child_weight=params.get('min_child_weight', 1),
                tree_method=params.get('tree_method', 'hist'),  # V1 spectral-optimized
                random_state=random_state,
                verbosity=0,
                n_jobs=n_jobs
            )

        elif model_name == 'CatBoost':
            if not HAS_CATBOOST:
                return None
            iterations = params.get('iterations', 100)
            learning_rate = params.get('learning_rate', 0.1)
            depth = params.get('depth', 6)
            return CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                random_state=random_state,
                verbose=0
            )

        elif model_name == 'SVM':
            kernel = params.get('kernel', 'rbf')
            C = params.get('C', 1.0)
            max_iter = params.get('max_iter', 5000)
            return SVC(kernel=kernel, C=C, max_iter=max_iter, random_state=random_state)

        elif model_name == 'NeuralBoosted':
            if not HAS_NEURAL_BOOSTED:
                return None
            n_estimators = params.get('n_estimators', 100)
            learning_rate = params.get('learning_rate', 0.1)
            hidden_layer_size = params.get('hidden_layer_size', 5)
            class_weight = params.get('class_weight', 'balanced')
            return NeuralBoostedClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                hidden_layer_size=hidden_layer_size,
                class_weight=class_weight,
                random_state=random_state
            )

        elif model_name == 'MLP':
            hidden_layer_sizes = params.get('hidden_layer_sizes', (100,))
            activation = params.get('activation', 'relu')
            alpha = params.get('alpha', 0.0001)
            max_iter = params.get('max_iter', 2000)
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                n_iter_no_change=10
            )

    return None


def get_feature_importances(model, model_name):
    """
    Extract feature importances from a fitted model.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible model
    model_name : str
        Model name to determine importance extraction method

    Returns
    -------
    importances : np.ndarray or None
        Feature importance scores, or None if not available
    """
    try:
        # Tree-based models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_

        # NeuralBoosted models have get_feature_importances()
        if hasattr(model, 'get_feature_importances'):
            return model.get_feature_importances()

        # PLS has coef_ with shape (n_targets, n_features)
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim == 2:
                # For PLS, shape is (n_targets, n_features) - we want features
                return np.abs(coef.ravel())
            return np.abs(coef.ravel())

        # Linear models have coef_
        if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            if hasattr(model, 'coef_'):
                return np.abs(model.coef_)

    except Exception:
        pass

    return None


def get_available_models(task_type='regression'):
    """
    Get list of available models based on installed packages.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    models : list of str
        List of available model names
    """
    if task_type == 'regression':
        models = ['PLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'SVR', 'MLP']
    else:
        models = ['PLS-DA', 'RandomForest', 'SVM', 'MLP']

    # Add optional models if available
    if HAS_LIGHTGBM:
        models.append('LightGBM')
    if HAS_XGBOOST:
        models.append('XGBoost')
    if HAS_CATBOOST:
        models.append('CatBoost')
    if HAS_NEURAL_BOOSTED:
        models.append('NeuralBoosted')

    return models
