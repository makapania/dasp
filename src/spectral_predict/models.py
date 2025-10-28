"""Model definitions and grid search configurations."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
from .neural_boosted import NeuralBoostedRegressor


def get_model_grids(task_type, n_features, max_n_components=24, max_iter=500):
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

    Returns
    -------
    grids : dict
        Dictionary mapping model names to lists of (model, param_dict) tuples
    """
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

        # Neural Boosted Regression
        nbr_configs = []

        # Grid: conservative to moderate learning rates
        learning_rates = [0.05, 0.1, 0.2]

        # Number of estimators: early stopping will find optimal
        n_estimators_list = [50, 100]

        # Hidden layer sizes: keep small (weak learner property)
        hidden_sizes = [3, 5]

        # Activations: tanh (JMP default) and identity (linear)
        activations = ['tanh', 'identity']

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
