"""Coupled preprocessing and model hyperparameter optimization using Optuna.

This module provides joint optimization of preprocessing methods and model hyperparameters
in a single Optuna search, enabling discovery of optimal preprocessing-model combinations.

Key Features:
- Joint optimization: preprocessing type, window sizes, baseline methods, model hyperparameters
- Preprocessing options: raw, snv, deriv1-4, snv_deriv1-4, deriv1-4_snv
- Baseline correction: none, als, polynomial, airpls
- Model support: PLS, Ridge, Lasso, ElasticNet, RF, LightGBM, XGBoost, CatBoost, SVR, MLP, NeuralBoosted
- Single objective: minimize RMSECV (regression) or maximize accuracy (classification)
- Cross-validation with configurable folds
- Standalone implementation (does not modify existing search.py or bayesian_config.py)

Example:
--------
>>> from spectral_predict.coupled_search import run_coupled_search
>>> best_params, best_score, study = run_coupled_search(
...     X, y, task_type='regression', n_trials=100, cv_folds=5
... )
>>> print(f"Best RMSECV: {best_score:.4f}")
>>> print(f"Best preprocessing: {best_params['preprocessing']}")
>>> print(f"Best model: {best_params['model']}")
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna import Trial
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import clone

# Import existing transformers (no modifications to existing code)
from spectral_predict.preprocess import SNV, SavgolDerivative
from spectral_predict.baseline import BaselineALS, BaselinePolynomial, BaselineAirPLS

# Import model builders
from spectral_predict.models import build_model


def build_preprocessing_steps(trial: Trial, n_features: int) -> list:
    """
    Suggest preprocessing steps using Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for hyperparameter suggestions
    n_features : int
        Number of spectral features (wavelengths)

    Returns
    -------
    steps : list
        List of (name, transformer) tuples for sklearn Pipeline
    """
    steps = []

    # 1. Baseline correction (optional)
    baseline_method = trial.suggest_categorical(
        'baseline_method', ['none', 'als', 'polynomial', 'airpls']
    )

    if baseline_method == 'als':
        lam = trial.suggest_float('baseline_als_lambda', 1e2, 1e9, log=True)
        p = trial.suggest_float('baseline_als_p', 0.001, 0.1, log=True)
        steps.append(('baseline', BaselineALS(lambda_=lam, p=p, niter=10)))

    elif baseline_method == 'polynomial':
        degree = trial.suggest_int('baseline_poly_degree', 1, 5)
        steps.append(('baseline', BaselinePolynomial(degree=degree)))

    elif baseline_method == 'airpls':
        lam = trial.suggest_float('baseline_airpls_lam', 1e2, 1e9, log=True)
        steps.append(('baseline', BaselineAirPLS(lam=lam, max_iter=15, tol=1e-3)))

    # 2. Spectral preprocessing (SNV/derivatives)
    preprocess_type = trial.suggest_categorical(
        'preprocessing',
        ['raw', 'snv', 'deriv1', 'deriv2', 'deriv3', 'deriv4',
         'snv_deriv1', 'snv_deriv2', 'snv_deriv3', 'snv_deriv4',
         'deriv1_snv', 'deriv2_snv', 'deriv3_snv', 'deriv4_snv']
    )

    # Extract derivative order from preprocessing type
    if 'deriv' in preprocess_type:
        # Extract derivative order (deriv1, deriv2, etc.)
        deriv_order = int(preprocess_type[-1]) if preprocess_type[-1].isdigit() else 1

        # Window size (must be odd, >= polyorder + 2)
        window = trial.suggest_int('savgol_window', 5, min(51, n_features), step=2)

        # Polyorder: deriv_order + 1 for stability
        polyorder = deriv_order + 1

        # Ensure window >= polyorder + 2
        if window < polyorder + 2:
            window = polyorder + 2
            # Ensure odd
            if window % 2 == 0:
                window += 1
            # Ensure within bounds
            window = min(window, n_features)

        savgol = SavgolDerivative(deriv=deriv_order, window=window, polyorder=polyorder)

        if preprocess_type.startswith('snv_'):
            # SNV then derivative
            steps.append(('snv', SNV()))
            steps.append(('savgol', savgol))
        elif preprocess_type.startswith('deriv') and '_snv' in preprocess_type:
            # Derivative then SNV
            steps.append(('savgol', savgol))
            steps.append(('snv', SNV()))
        else:
            # Just derivative
            steps.append(('savgol', savgol))

    elif preprocess_type == 'snv':
        steps.append(('snv', SNV()))

    # 'raw' = no preprocessing steps

    return steps


def build_model_step(trial: Trial, model_name: str, task_type: str, n_features: int):
    """
    Suggest model hyperparameters using Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for hyperparameter suggestions
    model_name : str
        Model type ('pls', 'ridge', 'xgboost', etc.)
    task_type : str
        'regression' or 'classification'
    n_features : int
        Number of input features (after preprocessing)

    Returns
    -------
    model : estimator
        Configured model instance
    """
    model_name = model_name.lower()

    if model_name == 'pls':
        n_components = trial.suggest_int('pls_n_components', 2, min(20, n_features))
        max_iter = trial.suggest_int('pls_max_iter', 100, 1000, log=True)
        params = {'n_components': n_components, 'max_iter': max_iter}
        if task_type == 'regression':
            return build_model('PLS', params, task_type='regression')
        else:
            return build_model('PLS', params, task_type='classification')

    elif model_name == 'ridge':
        alpha = trial.suggest_float('ridge_alpha', 1e-4, 1e2, log=True)
        params = {'alpha': alpha}
        return build_model('Ridge', params, task_type=task_type)

    elif model_name == 'lasso':
        alpha = trial.suggest_float('lasso_alpha', 1e-4, 1e2, log=True)
        params = {'alpha': alpha, 'max_iter': 5000}
        return build_model('Lasso', params, task_type=task_type)

    elif model_name == 'elasticnet':
        alpha = trial.suggest_float('elasticnet_alpha', 1e-4, 1e2, log=True)
        l1_ratio = trial.suggest_float('elasticnet_l1_ratio', 0.1, 0.9)
        params = {'alpha': alpha, 'l1_ratio': l1_ratio, 'max_iter': 5000}
        return build_model('ElasticNet', params, task_type=task_type)

    elif model_name == 'rf' or model_name == 'randomforest':
        n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
        max_depth = trial.suggest_int('rf_max_depth', 3, 30)
        min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 10)
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
        }
        return build_model('RandomForest', params, task_type=task_type)

    elif model_name == 'xgboost':
        n_estimators = trial.suggest_int('xgb_n_estimators', 50, 500)
        learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
        subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
        reg_alpha = trial.suggest_float('xgb_reg_alpha', 1e-5, 10.0, log=True)
        reg_lambda = trial.suggest_float('xgb_reg_lambda', 1e-5, 10.0, log=True)
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
        }
        return build_model('XGBoost', params, task_type=task_type)

    elif model_name == 'lightgbm':
        n_estimators = trial.suggest_int('lgbm_n_estimators', 50, 500)
        learning_rate = trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int('lgbm_num_leaves', 15, 100)
        min_child_samples = trial.suggest_int('lgbm_min_child_samples', 5, 50)
        subsample = trial.suggest_float('lgbm_subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0)
        reg_alpha = trial.suggest_float('lgbm_reg_alpha', 1e-5, 10.0, log=True)
        reg_lambda = trial.suggest_float('lgbm_reg_lambda', 1e-5, 10.0, log=True)
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'bagging_freq': 1,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
        }
        return build_model('LightGBM', params, task_type=task_type)

    elif model_name == 'catboost':
        iterations = trial.suggest_int('catboost_iterations', 50, 500)
        learning_rate = trial.suggest_float('catboost_learning_rate', 0.01, 0.3, log=True)
        depth = trial.suggest_int('catboost_depth', 3, 10)
        l2_leaf_reg = trial.suggest_float('catboost_l2_leaf_reg', 1.0, 10.0)
        params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
        }
        return build_model('CatBoost', params, task_type=task_type)

    elif model_name == 'svr' or model_name == 'svm':
        kernel = trial.suggest_categorical('svm_kernel', ['rbf', 'linear', 'poly'])
        C = trial.suggest_float('svm_C', 0.1, 100.0, log=True)
        if kernel in ['rbf', 'poly']:
            gamma = trial.suggest_categorical('svm_gamma', ['scale', 'auto'])
        else:
            gamma = 'scale'

        params = {'kernel': kernel, 'C': C}
        if kernel in ['rbf', 'poly']:
            params['gamma'] = gamma
        if task_type == 'regression':
            epsilon = trial.suggest_float('svr_epsilon', 0.01, 1.0)
            params['epsilon'] = epsilon
            return build_model('SVR', params, task_type='regression')
        else:
            return build_model('SVM', params, task_type='classification')

    elif model_name == 'mlp':
        hidden_size = trial.suggest_int('mlp_hidden_size', 32, 256, log=True)
        n_layers = trial.suggest_int('mlp_n_layers', 1, 3)
        alpha = trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('mlp_learning_rate_init', 1e-4, 1e-2, log=True)

        if n_layers == 1:
            hidden_layer_sizes = (hidden_size,)
        elif n_layers == 2:
            hidden_layer_sizes = (hidden_size, hidden_size // 2)
        else:
            hidden_layer_sizes = (hidden_size, hidden_size // 2, hidden_size // 4)

        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
        }
        return build_model('MLP', params, task_type=task_type)

    elif model_name == 'neuralboosted':
        n_estimators = trial.suggest_int('nb_n_estimators', 50, 300)
        learning_rate = trial.suggest_float('nb_learning_rate', 0.05, 0.3)
        hidden_layer_size = trial.suggest_int('nb_hidden_layer_size', 3, 10)
        activation = trial.suggest_categorical('nb_activation', ['tanh', 'identity', 'logistic'])
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'hidden_layer_size': hidden_layer_size,
            'activation': activation,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 10,
            'alpha': 1e-4,
        }
        return build_model('NeuralBoosted', params, task_type=task_type)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_coupled_search(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    model_name: str = 'xgboost',
    n_trials: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple:
    """
    Run coupled optimization of preprocessing and model hyperparameters.

    This function jointly optimizes preprocessing methods (baseline correction, SNV,
    Savitzky-Golay derivatives) and model hyperparameters using Optuna's Bayesian
    optimization. Unlike grid search or sequential optimization, this approach explores
    the full preprocessing × model hyperparameter space efficiently.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Spectral data matrix
    y : np.ndarray, shape (n_samples,)
        Target values (continuous for regression, discrete for classification)
    task_type : str, default='regression'
        Task type: 'regression' or 'classification'
    model_name : str, default='xgboost'
        Model type to optimize. Options:
        - 'pls': Partial Least Squares
        - 'ridge': Ridge regression/classification
        - 'lasso': Lasso regression
        - 'elasticnet': ElasticNet regression
        - 'rf' or 'randomforest': Random Forest
        - 'xgboost': XGBoost
        - 'lightgbm': LightGBM
        - 'catboost': CatBoost
        - 'svr' or 'svm': Support Vector Machine
        - 'mlp': Multi-layer Perceptron
        - 'neuralboosted': Neural Boosted model
    n_trials : int, default=100
        Number of Optuna trials (hyperparameter combinations to test)
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all cores)
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    best_params : dict
        Best hyperparameters found (includes preprocessing and model params)
    best_score : float
        Best cross-validation score (RMSECV for regression, accuracy for classification)
    study : optuna.Study
        Optuna study object containing all trial results

    Examples
    --------
    >>> from spectral_predict.coupled_search import run_coupled_search
    >>> # Optimize XGBoost with all preprocessing options
    >>> best_params, best_score, study = run_coupled_search(
    ...     X, y, model_name='xgboost', n_trials=100
    ... )
    >>> print(f"Best RMSECV: {best_score:.4f}")
    >>> print(f"Preprocessing: {best_params['preprocessing']}")
    >>> print(f"Baseline: {best_params['baseline_method']}")

    >>> # Optimize PLS for classification
    >>> best_params, best_score, study = run_coupled_search(
    ...     X, y, task_type='classification', model_name='pls', n_trials=50
    ... )
    >>> print(f"Best accuracy: {best_score:.4f}")

    Notes
    -----
    - Preprocessing search space:
      * Baseline: none, ALS, polynomial, airPLS
      * Spectral: raw, SNV, derivatives (1st-4th order), SNV+deriv, deriv+SNV
      * Window sizes: 5-51 (automatically adjusted for derivative order)
    - Model hyperparameters are optimized simultaneously with preprocessing
    - Uses cross-validation to prevent overfitting
    - Automatically handles feature dimension changes from preprocessing
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    # Determine direction (minimize RMSECV for regression, maximize accuracy for classification)
    if task_type == 'regression':
        direction = 'minimize'
        scoring = 'neg_root_mean_squared_error'
    else:
        direction = 'maximize'
        scoring = 'accuracy'

    # Define objective function
    def objective(trial: Trial) -> float:
        """Optuna objective: build pipeline, evaluate via CV."""
        # 1. Build preprocessing steps
        preprocess_steps = build_preprocessing_steps(trial, n_features)

        # 2. Build model
        model = build_model_step(trial, model_name, task_type, n_features)

        # 3. Create pipeline
        pipeline_steps = preprocess_steps + [('model', model)]
        pipeline = Pipeline(pipeline_steps)

        # 4. Cross-validation
        if task_type == 'regression':
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        try:
            scores = cross_val_score(
                pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score='raise'
            )
            mean_score = scores.mean()

            # Return positive value for minimization (RMSECV), score as-is for maximization
            if task_type == 'regression':
                # cross_val_score returns negative RMSE, negate to get RMSECV
                return -mean_score
            else:
                return mean_score

        except Exception as e:
            # If pipeline fails (e.g., invalid hyperparameters), return worst score
            if verbose:
                print(f"Trial {trial.number} failed: {e}")
            if task_type == 'regression':
                return 1e10  # Large RMSECV (bad)
            else:
                return 0.0  # Zero accuracy (bad)

    # Create Optuna study
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))

    # Run optimization
    if verbose:
        print(f"\n{'='*60}")
        print(f"Coupled Preprocessing + Model Optimization")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Task: {task_type}")
        print(f"Trials: {n_trials}")
        print(f"CV Folds: {cv_folds}")
        print(f"{'='*60}\n")

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)

    # Extract best results
    best_params = study.best_params
    best_score = study.best_value

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimization Complete")
        print(f"{'='*60}")
        if task_type == 'regression':
            print(f"Best RMSECV: {best_score:.6f}")
        else:
            print(f"Best Accuracy: {best_score:.6f}")
        print(f"\nBest Preprocessing:")
        print(f"  - Type: {best_params.get('preprocessing', 'raw')}")
        print(f"  - Baseline: {best_params.get('baseline_method', 'none')}")
        if 'savgol_window' in best_params:
            print(f"  - Savgol Window: {best_params['savgol_window']}")
        print(f"\nBest Model Hyperparameters:")
        model_params = {k: v for k, v in best_params.items()
                       if not k.startswith('baseline_') and k not in ['preprocessing', 'savgol_window']}
        for key, value in sorted(model_params.items()):
            print(f"  - {key}: {value}")
        print(f"{'='*60}\n")

    return best_params, best_score, study


# Self-test using synthetic data
if __name__ == "__main__":
    print("Testing coupled_search.py with synthetic data...")

    # Generate synthetic spectral data
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 200

    # Simulate spectra with baseline + peaks + noise
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    X = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        # Baseline (polynomial)
        baseline = 0.5 + 0.0001 * wavelengths - 0.00000005 * wavelengths ** 2

        # Add Gaussian peaks
        peak1 = 0.3 * np.exp(-((wavelengths - 1000) ** 2) / (2 * 50 ** 2))
        peak2 = 0.5 * np.exp(-((wavelengths - 1500) ** 2) / (2 * 80 ** 2))

        # Noise
        noise = 0.02 * np.random.randn(n_wavelengths)

        X[i, :] = baseline + peak1 + peak2 + noise

    # Create regression target (sum of peak intensities + noise)
    y_regression = (
        X[:, np.argmin(np.abs(wavelengths - 1000))]
        + X[:, np.argmin(np.abs(wavelengths - 1500))]
        + 0.1 * np.random.randn(n_samples)
    )

    # Create classification target (binary based on median)
    y_classification = (y_regression > np.median(y_regression)).astype(int)

    # Test 1: Regression with XGBoost (quick test: 10 trials)
    print("\n" + "=" * 60)
    print("Test 1: Regression with XGBoost")
    print("=" * 60)
    best_params_reg, best_score_reg, study_reg = run_coupled_search(
        X, y_regression,
        task_type='regression',
        model_name='xgboost',
        n_trials=10,
        cv_folds=3,
        verbose=True
    )

    # Test 2: Classification with RandomForest (quick test: 10 trials)
    print("\n" + "=" * 60)
    print("Test 2: Classification with RandomForest")
    print("=" * 60)
    best_params_clf, best_score_clf, study_clf = run_coupled_search(
        X, y_classification,
        task_type='classification',
        model_name='rf',
        n_trials=10,
        cv_folds=3,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
