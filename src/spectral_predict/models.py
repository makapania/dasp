"""Model definitions and grid search configurations."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from .neural_boosted import NeuralBoostedRegressor

# Import gradient boosting libraries
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


def get_model(model_name, task_type='regression', n_components=10, max_n_components=24, max_iter=500):
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
    max_n_components : int, default=24
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
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )

        elif model_name == "MLP":
            return MLPRegressor(
                hidden_layer_sizes=(64,),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True
            )

        elif model_name == "NeuralBoosted":
            return NeuralBoostedRegressor(
                n_estimators=100,
                learning_rate=0.1,
                hidden_layer_size=3,
                activation='tanh',
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                alpha=1e-4,
                random_state=42,
                verbose=0
            )

        elif model_name == "SVR":
            return SVR(kernel='rbf', C=1.0, gamma='scale')

        elif model_name == "XGBoost":
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

        elif model_name == "LightGBM":
            return LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

        elif model_name == "CatBoost":
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
            return PLSRegression(n_components=n_components, scale=False)

        elif model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )

        elif model_name == "MLP":
            return MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True
            )

        elif model_name == "SVM":
            return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)

        elif model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

        elif model_name == "LightGBM":
            return LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

        elif model_name == "CatBoost":
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


def get_model_grids(task_type, n_features, max_n_components=24, max_iter=500,
                    n_estimators_list=None, learning_rates=None, rf_n_trees_list=None,
                    rf_max_depth_list=None, ridge_alphas_list=None, lasso_alphas_list=None):
    """
    Get model grids for hyperparameter search.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'
    n_features : int
        Number of input features
    max_n_components : int, default=24
        Maximum number of PLS components to test
    max_iter : int, default=500
        Maximum iterations for MLP
    n_estimators_list : list of int, optional
        List of n_estimators values for NeuralBoosted. If None, uses [100]
    learning_rates : list of float, optional
        List of learning rates for NeuralBoosted. If None, uses [0.1, 0.2]
    rf_n_trees_list : list of int, optional
        List of n_estimators values for RandomForest. If None, uses [200, 500]
    rf_max_depth_list : list of int or None, optional
        List of max_depth values for RandomForest. If None, uses [None, 30]
        (safer and faster than previous default of [None, 15, 30])

    Returns
    -------
    grids : dict
        Dictionary mapping model names to lists of (model, param_dict) tuples
    """
    # Set defaults for NeuralBoosted hyperparameters
    if n_estimators_list is None:
        n_estimators_list = [100]
    if learning_rates is None:
        learning_rates = [0.1, 0.2]

    # Set defaults for RandomForest hyperparameters
    if rf_n_trees_list is None:
        rf_n_trees_list = [200, 500]
    if rf_max_depth_list is None:
        # UPDATED: Changed from [None, 15, 30] to [None, 30] for safety and speed
        # This reduces RF grid configs from 6 to 4 per preprocessing method
        rf_max_depth_list = [None, 30]

    # Set defaults for Ridge hyperparameters
    if ridge_alphas_list is None:
        ridge_alphas_list = [0.001, 0.01, 0.1, 1.0, 10.0]

    # Set defaults for Lasso hyperparameters
    if lasso_alphas_list is None:
        lasso_alphas_list = [0.001, 0.01, 0.1, 1.0]

    grids = {}

    # PLS components grid (clip to n_features and max_n_components)
    pls_components = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50]
    pls_components = [c for c in pls_components if c <= n_features and c <= max_n_components]

    if task_type == "regression":
        # PLS Regression
        grids["PLS"] = [
            (PLSRegression(n_components=nc, scale=False), {"n_components": nc})
            for nc in pls_components
        ]

        # Ridge Regression - uses configurable alpha from GUI or defaults
        ridge_configs = []
        for alpha in ridge_alphas_list:
            ridge_configs.append(
                (
                    Ridge(alpha=alpha, random_state=42),
                    {"alpha": alpha}
                )
            )
        grids["Ridge"] = ridge_configs

        # Lasso Regression - uses configurable alpha from GUI or defaults
        lasso_configs = []
        for alpha in lasso_alphas_list:
            lasso_configs.append(
                (
                    Lasso(alpha=alpha, random_state=42, max_iter=max_iter),
                    {"alpha": alpha}
                )
            )
        grids["Lasso"] = lasso_configs

        # ElasticNet Regression - combines L1 and L2 regularization
        elasticnet_configs = []
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            for l1_ratio in [0.3, 0.5, 0.7]:  # Balance between L1 and L2
                elasticnet_configs.append(
                    (
                        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=max_iter),
                        {"alpha": alpha, "l1_ratio": l1_ratio}
                    )
                )
        grids["ElasticNet"] = elasticnet_configs

        # Random Forest - uses configurable max_depth from GUI or defaults
        rf_configs = []
        for n_est in rf_n_trees_list:
            for max_d in rf_max_depth_list:  # ✅ Now configurable!
                rf_configs.append(
                    (
                        RandomForestRegressor(
                            n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1
                        ),
                        {"n_estimators": n_est, "max_depth": max_d},
                    )
                )
        grids["RandomForest"] = rf_configs

        # MLP
        mlp_configs = []
        for hidden in [(64,), (128, 64)]:
            for alpha in [1e-4, 1e-3]:
                for lr in [1e-3, 1e-2]:
                    mlp_configs.append(
                        (
                            MLPRegressor(
                                hidden_layer_sizes=hidden,
                                alpha=alpha,
                                learning_rate_init=lr,
                                max_iter=max_iter,
                                random_state=42,
                                early_stopping=True,
                            ),
                            {
                                "hidden_layer_sizes": hidden,
                                "alpha": alpha,
                                "learning_rate_init": lr,
                            },
                        )
                    )
        grids["MLP"] = mlp_configs

        # Neural Boosted Regression
        nbr_configs = []

        # Use user-specified hyperparameters (or defaults from function parameters)
        # Default: [100] for n_estimators, [0.1, 0.2] for learning_rates
        # User can enable more via GUI for comprehensive search

        # Hidden layer sizes: keep small (weak learner property)
        hidden_sizes = [3, 5]

        # Activations: tanh (JMP default) and identity (linear)
        activations = ['tanh', 'identity']

        # Grid size: len(n_estimators_list) × len(learning_rates) × 2 × 2
        # Default: 1 × 2 × 2 × 2 = 8 configs
        # Full: 2 × 3 × 2 × 2 = 24 configs (if user enables all)

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
                                    alpha=1e-4,  # Light L2 regularization
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
        # Total configurations: 2 * 3 * 2 * 2 = 24 per preprocessing method

        # Support Vector Regression (SVR)
        svr_configs = []
        for kernel in ['rbf', 'linear']:
            for C in [0.1, 1.0, 10.0]:
                if kernel == 'rbf':
                    for gamma in ['scale', 'auto']:
                        svr_configs.append(
                            (
                                SVR(kernel=kernel, C=C, gamma=gamma),
                                {"kernel": kernel, "C": C, "gamma": gamma}
                            )
                        )
                else:
                    svr_configs.append(
                        (
                            SVR(kernel=kernel, C=C),
                            {"kernel": kernel, "C": C}
                        )
                    )
        grids["SVR"] = svr_configs

        # XGBoost Regression
        xgb_configs = []
        for n_est in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for max_depth in [3, 6, 9]:
                    xgb_configs.append(
                        (
                            XGBRegressor(
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=max_depth,
                                random_state=42,
                                n_jobs=-1,
                                verbosity=0
                            ),
                            {"n_estimators": n_est, "learning_rate": lr, "max_depth": max_depth}
                        )
                    )
        grids["XGBoost"] = xgb_configs

        # LightGBM Regression
        lgbm_configs = []
        for n_est in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for num_leaves in [31, 50, 70]:
                    lgbm_configs.append(
                        (
                            LGBMRegressor(
                                n_estimators=n_est,
                                learning_rate=lr,
                                num_leaves=num_leaves,
                                random_state=42,
                                n_jobs=-1,
                                verbosity=-1
                            ),
                            {"n_estimators": n_est, "learning_rate": lr, "num_leaves": num_leaves}
                        )
                    )
        grids["LightGBM"] = lgbm_configs

        # CatBoost Regression
        catboost_configs = []
        for iterations in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for depth in [4, 6, 8]:
                    catboost_configs.append(
                        (
                            CatBoostRegressor(
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

    else:  # classification
        # PLS-DA (PLS + LogisticRegression)
        grids["PLS-DA"] = [
            (PLSRegression(n_components=nc, scale=False), {"n_components": nc})
            for nc in pls_components
        ]

        # Random Forest - uses configurable max_depth from GUI or defaults
        rf_configs = []
        for n_est in rf_n_trees_list:
            for max_d in rf_max_depth_list:  # ✅ Now configurable!
                rf_configs.append(
                    (
                        RandomForestClassifier(
                            n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1
                        ),
                        {"n_estimators": n_est, "max_depth": max_d},
                    )
                )
        grids["RandomForest"] = rf_configs

        # MLP
        mlp_configs = []
        for hidden in [(64,), (128, 64)]:
            for alpha in [1e-4, 1e-3]:
                for lr in [1e-3, 1e-2]:
                    mlp_configs.append(
                        (
                            MLPClassifier(
                                hidden_layer_sizes=hidden,
                                alpha=alpha,
                                learning_rate_init=lr,
                                max_iter=max_iter,
                                random_state=42,
                                early_stopping=True,
                            ),
                            {
                                "hidden_layer_sizes": hidden,
                                "alpha": alpha,
                                "learning_rate_init": lr,
                            },
                        )
                    )
        grids["MLP"] = mlp_configs

        # Support Vector Machine (SVM) for classification
        svm_configs = []
        for kernel in ['rbf', 'linear']:
            for C in [0.1, 1.0, 10.0]:
                if kernel == 'rbf':
                    for gamma in ['scale', 'auto']:
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

        # XGBoost Classification
        xgb_configs = []
        for n_est in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for max_depth in [3, 6, 9]:
                    xgb_configs.append(
                        (
                            XGBClassifier(
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=max_depth,
                                random_state=42,
                                n_jobs=-1,
                                verbosity=0
                            ),
                            {"n_estimators": n_est, "learning_rate": lr, "max_depth": max_depth}
                        )
                    )
        grids["XGBoost"] = xgb_configs

        # LightGBM Classification
        lgbm_configs = []
        for n_est in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for num_leaves in [31, 50, 70]:
                    lgbm_configs.append(
                        (
                            LGBMClassifier(
                                n_estimators=n_est,
                                learning_rate=lr,
                                num_leaves=num_leaves,
                                random_state=42,
                                n_jobs=-1,
                                verbosity=-1
                            ),
                            {"n_estimators": n_est, "learning_rate": lr, "num_leaves": num_leaves}
                        )
                    )
        grids["LightGBM"] = lgbm_configs

        # CatBoost Classification
        catboost_configs = []
        for iterations in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for depth in [4, 6, 8]:
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

    return grids


def compute_vip(pls_model, X, y):
    """
    Compute Variable Importance in Projection (VIP) scores for a fitted PLS model.

    Parameters
    ----------
    pls_model : PLSRegression
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
