"""Model definitions and grid search configurations."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from .neural_boosted import NeuralBoostedRegressor


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

        else:
            raise ValueError(f"Unknown classification model: {model_name}")

    return model


def get_model_grids(task_type, n_features, max_n_components=24, max_iter=500,
                    n_estimators_list=None, learning_rates=None):
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

        # Ridge Regression
        ridge_configs = []
        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            ridge_configs.append(
                (
                    Ridge(alpha=alpha, random_state=42),
                    {"alpha": alpha}
                )
            )
        grids["Ridge"] = ridge_configs

        # Lasso Regression
        lasso_configs = []
        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            lasso_configs.append(
                (
                    Lasso(alpha=alpha, random_state=42, max_iter=max_iter),
                    {"alpha": alpha}
                )
            )
        grids["Lasso"] = lasso_configs

        # Random Forest
        rf_configs = []
        for n_est in [200, 500]:
            for max_d in [None, 15, 30]:
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

    else:  # classification
        # PLS-DA (PLS + LogisticRegression)
        grids["PLS-DA"] = [
            (PLSRegression(n_components=nc, scale=False), {"n_components": nc})
            for nc in pls_components
        ]

        # Random Forest
        rf_configs = []
        for n_est in [200, 500]:
            for max_d in [None, 15, 30]:
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

    # VIP calculation
    n_features = W.shape[0]
    n_components = W.shape[1]

    vip_scores = np.zeros(n_features)
    for i in range(n_features):
        weight = np.sum((W[i, :] ** 2) * ssy_comp)
        vip_scores[i] = np.sqrt(n_features * weight / ssy_total)

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

    elif model_name in ["Ridge", "Lasso"]:
        # Get coefficients (linear models)
        coefs = np.abs(model.coef_)
        if len(coefs.shape) > 1:
            coefs = coefs[0]  # Handle multi-output case

        # Return absolute coefficient values as importance
        return coefs

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

    else:
        raise ValueError(f"Unknown model type: {model_name}")
