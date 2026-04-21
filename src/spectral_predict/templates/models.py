"""
Model instantiation templates for generated scripts.
"""

# Model-specific import statements
MODEL_IMPORTS = {
    'PLS': 'from sklearn.cross_decomposition import PLSRegression',
    'PLSDA': 'from sklearn.cross_decomposition import PLSRegression',
    'Ridge': 'from sklearn.linear_model import Ridge',
    'Lasso': 'from sklearn.linear_model import Lasso',
    'ElasticNet': 'from sklearn.linear_model import ElasticNet',
    'RandomForest': 'from sklearn.ensemble import RandomForestRegressor',
    'RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',
    'LightGBM': 'from lightgbm import LGBMRegressor',
    'LightGBMClassifier': 'from lightgbm import LGBMClassifier',
    'XGBoost': 'from xgboost import XGBRegressor',
    'XGBoostClassifier': 'from xgboost import XGBClassifier',
    'CatBoost': 'from catboost import CatBoostRegressor',
    'CatBoostClassifier': 'from catboost import CatBoostClassifier',
    'SVR': 'from sklearn.svm import SVR',
    'SVC': 'from sklearn.svm import SVC',
    'MLP': 'from sklearn.neural_network import MLPRegressor',
    'MLPRegressor': 'from sklearn.neural_network import MLPRegressor',
    'MLPClassifier': 'from sklearn.neural_network import MLPClassifier',
    'IsolationForest': 'from sklearn.ensemble import IsolationForest',
    'OneClassSVM': 'from sklearn.svm import OneClassSVM',
    'EllipticEnvelope': 'from sklearn.covariance import EllipticEnvelope',
    'LOF': 'from sklearn.neighbors import LocalOutlierFactor',
    'PCA-SIMCA': '',
}

# Model instantiation templates
MODEL_TEMPLATES = {
    'PLS': '''
# PLS Regression
model = PLSRegression(
    n_components={n_components},
    scale={scale}
)
''',

    'PLSDA': '''
# PLS-DA (PLS for Discriminant Analysis)
# Note: sklearn's PLSRegression can be used for classification
# by treating class labels as continuous targets
model = PLSRegression(
    n_components={n_components},
    scale={scale}
)
''',

    'Ridge': '''
# Ridge Regression
model = Ridge(
    alpha={alpha},
    fit_intercept={fit_intercept}
)
''',

    'Lasso': '''
# Lasso Regression
model = Lasso(
    alpha={alpha},
    max_iter={max_iter},
    tol={tol},
    fit_intercept={fit_intercept}
)
''',

    'ElasticNet': '''
# ElasticNet Regression
model = ElasticNet(
    alpha={alpha},
    l1_ratio={l1_ratio},
    max_iter={max_iter},
    tol={tol},
    fit_intercept={fit_intercept}
)
''',

    'RandomForest': '''
# Random Forest Regressor
model = RandomForestRegressor(
    n_estimators={n_estimators},
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    min_samples_leaf={min_samples_leaf},
    random_state={random_state},
    n_jobs=-1
)
''',

    'RandomForestClassifier': '''
# Random Forest Classifier
model = RandomForestClassifier(
    n_estimators={n_estimators},
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    min_samples_leaf={min_samples_leaf},
    random_state={random_state},
    n_jobs=-1
)
''',

    'LightGBM': '''
# LightGBM Regressor
model = LGBMRegressor(
    n_estimators={n_estimators},
    max_depth={max_depth},
    learning_rate={learning_rate},
    num_leaves={num_leaves},
    random_state={random_state},
    verbose=-1
)
''',

    'LightGBMClassifier': '''
# LightGBM Classifier
model = LGBMClassifier(
    n_estimators={n_estimators},
    max_depth={max_depth},
    learning_rate={learning_rate},
    num_leaves={num_leaves},
    random_state={random_state},
    verbose=-1
)
''',

    'XGBoost': '''
# XGBoost Regressor
model = XGBRegressor(
    n_estimators={n_estimators},
    max_depth={max_depth},
    learning_rate={learning_rate},
    random_state={random_state},
    verbosity=0
)
''',

    'XGBoostClassifier': '''
# XGBoost Classifier
model = XGBClassifier(
    n_estimators={n_estimators},
    max_depth={max_depth},
    learning_rate={learning_rate},
    random_state={random_state},
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss'
)
''',

    'CatBoost': '''
# CatBoost Regressor
model = CatBoostRegressor(
    n_estimators={n_estimators},
    max_depth={max_depth},
    learning_rate={learning_rate},
    random_state={random_state},
    verbose=0
)
''',

    'CatBoostClassifier': '''
# CatBoost Classifier
model = CatBoostClassifier(
    n_estimators={n_estimators},
    max_depth={max_depth},
    learning_rate={learning_rate},
    random_state={random_state},
    verbose=0
)
''',

    'SVR': '''
# Support Vector Regression
model = SVR(
    C={C},
    epsilon={epsilon},
    kernel='{kernel}',
    gamma='{gamma}'
)
''',

    'SVC': '''
# Support Vector Classification
model = SVC(
    C={C},
    kernel='{kernel}',
    gamma='{gamma}',
    probability=True
)
''',

    'MLP': '''
# Multi-Layer Perceptron Regressor
model = MLPRegressor(
    hidden_layer_sizes={hidden_layer_sizes},
    activation='{activation}',
    solver='{solver}',
    alpha={alpha},
    learning_rate='{learning_rate}',
    max_iter={max_iter},
    random_state={random_state},
    early_stopping={early_stopping}
)
''',

    'MLPRegressor': '''
# Multi-Layer Perceptron Regressor
model = MLPRegressor(
    hidden_layer_sizes={hidden_layer_sizes},
    activation='{activation}',
    solver='{solver}',
    alpha={alpha},
    learning_rate='{learning_rate}',
    max_iter={max_iter},
    random_state={random_state},
    early_stopping={early_stopping}
)
''',

    'MLPClassifier': '''
# Multi-Layer Perceptron Classifier
model = MLPClassifier(
    hidden_layer_sizes={hidden_layer_sizes},
    activation='{activation}',
    solver='{solver}',
    alpha={alpha},
    learning_rate='{learning_rate}',
    max_iter={max_iter},
    random_state={random_state},
    early_stopping={early_stopping}
)
''',

    'IsolationForest': '''
model = IsolationForest(
    n_estimators={n_estimators},
    contamination={contamination},
    random_state={random_state},
    n_jobs={n_jobs}
)
''',

    'OneClassSVM': '''
model = OneClassSVM(
    kernel='{kernel}',
    gamma='{gamma}',
    nu={nu}
)
''',

    'EllipticEnvelope': '''
model = EllipticEnvelope(
    contamination={contamination},
    random_state={random_state}
)
''',

    'LOF': '''
model = LocalOutlierFactor(
    n_neighbors={n_neighbors},
    contamination={contamination},
    novelty=True,
    n_jobs={n_jobs}
)
''',

    'PCA-SIMCA': '''
model = PCASIMCA(
    n_components={n_components},
    alpha={alpha}
)
''',
}

# Default parameters for each model
DEFAULT_PARAMS = {
    'PLS': {'n_components': 10, 'scale': False},
    'PLSDA': {'n_components': 10, 'scale': False},
    'Ridge': {'alpha': 1.0, 'fit_intercept': True},
    'Lasso': {'alpha': 0.1, 'max_iter': 1000, 'tol': 0.0001, 'fit_intercept': True},
    'ElasticNet': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000, 'tol': 0.0001, 'fit_intercept': True},
    'RandomForest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42},
    'RandomForestClassifier': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42},
    'LightGBM': {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': 42},
    'LightGBMClassifier': {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': 42},
    'XGBoost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
    'XGBoostClassifier': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
    'CatBoost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
    'CatBoostClassifier': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
    'SVR': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
    'SVC': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
    'MLP': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'learning_rate': 'constant', 'max_iter': 200, 'random_state': 42, 'early_stopping': True},
    'MLPRegressor': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'learning_rate': 'constant', 'max_iter': 200, 'random_state': 42, 'early_stopping': True},
    'MLPClassifier': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'learning_rate': 'constant', 'max_iter': 200, 'random_state': 42, 'early_stopping': True},
    'IsolationForest': {'n_estimators': 200, 'contamination': 0.05, 'random_state': 42, 'n_jobs': 1},
    'OneClassSVM': {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
    'EllipticEnvelope': {'contamination': 0.05, 'random_state': 42},
    'LOF': {'n_neighbors': 20, 'contamination': 0.05, 'novelty': True, 'n_jobs': 1},
    'PCA-SIMCA': {'n_components': 5, 'alpha': 0.05},
}

ONE_CLASS_MODELS = {'IsolationForest', 'OneClassSVM', 'EllipticEnvelope', 'LOF', 'PCA-SIMCA'}

ONE_CLASS_NEEDS_SCALING = {'OneClassSVM', 'EllipticEnvelope', 'LOF'}

PCASIMCA_CLASS_TEMPLATE = '''
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from scipy import stats


class PCASIMCA(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components=5, alpha=0.05):
        if not (isinstance(alpha, (int, float)) and 0 < float(alpha) < 1):
            raise ValueError(f"alpha must be in (0, 1), got {{alpha!r}}")
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        if n_samples < 3:
            raise ValueError(f"Need at least 3 clean samples to fit DD-SIMCA, got {{n_samples}}")
        max_components = min(n_samples - 1, n_features)
        if max_components < 1:
            raise ValueError(f"Cannot fit DD-SIMCA with n_samples={{n_samples}}, n_features={{n_features}}")
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            pca_probe = PCA(n_components=self.n_components)
            pca_probe.fit(X)
            n_comp = min(pca_probe.n_components_, max_components)
        else:
            n_comp = min(int(self.n_components), max_components)
        self.n_components_ = n_comp
        self.pca_ = PCA(n_components=n_comp)
        self.scores_ = self.pca_.fit_transform(X)
        self.n_train_ = n_samples
        self.eigenvalues_ = self.pca_.explained_variance_
        t2_train = self._compute_t2(self.scores_)
        self.t2_train_ = t2_train
        self.t2_dof_, _, self.t2_scale_ = self._fit_chi2(t2_train)
        q_train = self._compute_q_residuals(X)
        if np.max(q_train) > 1e-10:
            self.q_dof_, _, self.q_scale_ = self._fit_chi2(q_train)
            self.q_threshold_method_ = "chi2_fit"
        else:
            self.q_dof_ = 2.0
            self.q_scale_ = 1e-10
            self.q_threshold_method_ = "zero_guard"
        self.joint_threshold_ = stats.chi2.ppf(1 - self.alpha, 4)
        self.t2_threshold_ = stats.chi2.ppf(1 - self.alpha, self.t2_dof_, loc=0, scale=self.t2_scale_)
        self.q_threshold_ = stats.chi2.ppf(1 - self.alpha, self.q_dof_, loc=0, scale=self.q_scale_)
        return self

    def _fit_chi2(self, values):
        try:
            dof, loc, scale = stats.chi2.fit(values, floc=0, method='mm')
            return dof, loc, scale
        except (RuntimeError, ValueError, TypeError):
            pass
        mean_val = np.mean(values)
        var_val = np.var(values)
        if var_val > 1e-10 and mean_val > 1e-10:
            dof = 2 * mean_val**2 / var_val
            scale = var_val / (2 * mean_val)
        else:
            dof = 2.0
            scale = max(mean_val / 2.0, 1e-10)
        return dof, 0.0, scale

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        pc_scores = self.pca_.transform(X)
        t2 = self._compute_t2(pc_scores)
        q = self._compute_q_residuals_from_scores(X, pc_scores)
        p_t2 = 1.0 - stats.chi2.cdf(t2, self.t2_dof_, loc=0, scale=self.t2_scale_)
        p_q = 1.0 - stats.chi2.cdf(q, self.q_dof_, loc=0, scale=self.q_scale_)
        p_t2 = np.clip(p_t2, 1e-300, 1.0)
        p_q = np.clip(p_q, 1e-300, 1.0)
        fisher_stat = -2.0 * (np.log(p_t2) + np.log(p_q))
        return self.joint_threshold_ - fisher_stat

    def score_samples(self, X):
        return self.decision_function(X)

    def _compute_t2(self, scores):
        lam = np.maximum(self.eigenvalues_, 1e-10)
        return np.sum(scores ** 2 / lam, axis=1)

    def _compute_q_residuals(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = self.pca_.transform(X)
        return self._compute_q_residuals_from_scores(X, scores)

    def _compute_q_residuals_from_scores(self, X, pc_scores):
        X = np.asarray(X, dtype=np.float64)
        X_reconstructed = pc_scores @ self.pca_.components_ + self.pca_.mean_
        residuals = X - X_reconstructed
        return np.sum(residuals ** 2, axis=1)
'''


def get_model_imports(model_name: str) -> str:
    """
    Get the import statement for a model.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'PLS', 'Ridge', 'RandomForest')

    Returns
    -------
    str
        Import statement
    """
    # Normalize model name
    normalized = model_name.replace(' ', '').replace('-', '')

    # Map common aliases
    aliases = {
        'RF': 'RandomForest',
        'LGBM': 'LightGBM',
        'XGB': 'XGBoost',
        'CB': 'CatBoost',
        'SVM': 'SVR',
        'NN': 'MLP',
    }

    if normalized.upper() in aliases:
        normalized = aliases[normalized.upper()]

    return MODEL_IMPORTS.get(normalized, '')


def get_model_template(model_name: str, params: dict = None, task_type: str = 'regression') -> str:
    """
    Get the model instantiation template with parameters.

    Parameters
    ----------
    model_name : str
        Model name
    params : dict, optional
        Model parameters. If None, uses defaults.
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    str
        Model instantiation code
    """
    # Normalize model name
    normalized = model_name.replace(' ', '').replace('-', '')

    # Map aliases
    aliases = {
        'RF': 'RandomForest',
        'LGBM': 'LightGBM',
        'XGB': 'XGBoost',
        'CB': 'CatBoost',
        'SVM': 'SVR',
        'NN': 'MLP',
    }

    if normalized.upper() in aliases:
        normalized = aliases[normalized.upper()]

    # Adjust for classification
    if task_type == 'classification':
        if normalized == 'RandomForest':
            normalized = 'RandomForestClassifier'
        elif normalized == 'LightGBM':
            normalized = 'LightGBMClassifier'
        elif normalized == 'XGBoost':
            normalized = 'XGBoostClassifier'
        elif normalized == 'CatBoost':
            normalized = 'CatBoostClassifier'
        elif normalized == 'SVR':
            normalized = 'SVC'
        elif normalized == 'PLS':
            normalized = 'PLSDA'
        elif normalized == 'MLP':
            normalized = 'MLPClassifier'

    # Get template
    template = MODEL_TEMPLATES.get(normalized, '')
    if not template:
        return f'# Model: {model_name}\n# (Template not available - please instantiate manually)\n'

    # Merge defaults with provided params
    defaults = DEFAULT_PARAMS.get(normalized, {}).copy()
    if params:
        defaults.update(params)

    # Format template
    try:
        return template.format(**defaults)
    except KeyError as e:
        # Missing parameter - return with placeholder
        return f'# Model: {model_name}\n# Note: Missing parameter {e}\n' + template
