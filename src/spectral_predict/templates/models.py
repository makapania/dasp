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
}

# Default parameters for each model
DEFAULT_PARAMS = {
    'PLS': {'n_components': 8, 'scale': False},
    'PLSDA': {'n_components': 8, 'scale': False},
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
}


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
