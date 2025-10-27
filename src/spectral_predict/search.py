"""Model search with cross-validation and subset selection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

from .preprocess import build_preprocessing_pipeline
from .models import get_model_grids, get_feature_importances
from .scoring import create_results_dataframe, add_result


def run_search(X, y, task_type, folds=5, lambda_penalty=0.15):
    """
    Run comprehensive model search with preprocessing, CV, and subset selection.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (n_samples, n_features)
    y : pd.Series
        Target values
    task_type : str
        'regression' or 'classification'
    folds : int
        Number of CV folds
    lambda_penalty : float
        Complexity penalty weight

    Returns
    -------
    df_ranked : pd.DataFrame
        Ranked results with all model runs
    """
    X_np = X.values
    y_np = y.values
    wavelengths = X.columns.values
    n_features = X_np.shape[1]

    # Create results container
    df_results = create_results_dataframe(task_type)

    # Determine if classification is binary or multiclass
    is_binary_classification = False
    if task_type == "classification":
        n_classes = len(np.unique(y_np))
        is_binary_classification = n_classes == 2

    # Get model grids
    model_grids = get_model_grids(task_type, n_features)

    # Define preprocessing configurations
    preprocess_configs = [
        {"name": "raw", "deriv": None, "window": None, "polyorder": None},
        {"name": "snv", "deriv": None, "window": None, "polyorder": None},
    ]

    # Add derivative configs
    for deriv in [1, 2]:
        for window in [7, 19]:
            polyorder = 2 if deriv == 1 else 3
            preprocess_configs.append(
                {"name": "deriv", "deriv": deriv, "window": window, "polyorder": polyorder}
            )
            preprocess_configs.append(
                {"name": "snv_deriv", "deriv": deriv, "window": window, "polyorder": polyorder}
            )
            preprocess_configs.append(
                {"name": "deriv_snv", "deriv": deriv, "window": window, "polyorder": polyorder}
            )

    # Create CV splitter
    if task_type == "regression":
        cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        cv_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    print(f"Running {task_type} search with {folds}-fold CV...")
    print(f"Models: {list(model_grids.keys())}")
    print(f"Preprocessing configs: {len(preprocess_configs)}")

    # Main search loop
    for preprocess_cfg in preprocess_configs:
        for model_name, model_configs in model_grids.items():
            for model, params in model_configs:
                # Run full model first
                result = _run_single_config(
                    X_np,
                    y_np,
                    wavelengths,
                    model,
                    model_name,
                    params,
                    preprocess_cfg,
                    cv_splitter,
                    task_type,
                    is_binary_classification,
                    subset_indices=None,
                    subset_tag="full",
                )
                df_results = add_result(df_results, result)

                # For PLS and RF, compute feature importances and run subsets
                if model_name in ["PLS", "PLS-DA", "RandomForest"]:
                    # Refit on full data to get importances
                    pipe_steps = build_preprocessing_pipeline(
                        preprocess_cfg["name"],
                        preprocess_cfg["deriv"],
                        preprocess_cfg["window"],
                        preprocess_cfg["polyorder"],
                    )
                    pipe_steps.append(("model", model))
                    pipe = Pipeline(pipe_steps) if pipe_steps else model

                    pipe.fit(X_np, y_np)

                    # Get model from pipeline
                    fitted_model = (
                        pipe.named_steps["model"] if hasattr(pipe, "named_steps") else pipe
                    )

                    # Get transformed X for importance calculation
                    if hasattr(pipe, "named_steps") and len(pipe_steps) > 1:
                        # Transform through preprocessing
                        X_transformed = X_np
                        for step_name, transformer in pipe.steps[:-1]:
                            X_transformed = transformer.transform(X_transformed)
                    else:
                        X_transformed = X_np

                    # Get importances
                    try:
                        importances = get_feature_importances(
                            fitted_model, model_name, X_transformed, y_np
                        )

                        # Run subsets: top-20, top-5, top-3
                        for n_top in [20, 5, 3]:
                            if n_top >= n_features:
                                continue

                            top_indices = np.argsort(importances)[-n_top:][::-1]

                            subset_result = _run_single_config(
                                X_np,
                                y_np,
                                wavelengths,
                                model,
                                model_name,
                                params,
                                preprocess_cfg,
                                cv_splitter,
                                task_type,
                                is_binary_classification,
                                subset_indices=top_indices,
                                subset_tag=f"top{n_top}",
                            )
                            df_results = add_result(df_results, subset_result)
                    except Exception as e:
                        print(f"Warning: Could not compute importances for {model_name}: {e}")

    # Compute composite scores and rank
    from .scoring import compute_composite_score

    df_ranked = compute_composite_score(df_results, task_type, lambda_penalty)

    return df_ranked


def _run_single_config(
    X,
    y,
    wavelengths,
    model,
    model_name,
    params,
    preprocess_cfg,
    cv_splitter,
    task_type,
    is_binary_classification,
    subset_indices=None,
    subset_tag="full",
):
    """
    Run a single model configuration with CV.

    Returns a dictionary with results.
    """
    # Apply subset if specified
    if subset_indices is not None:
        X = X[:, subset_indices]
        n_vars = len(subset_indices)
    else:
        n_vars = X.shape[1]

    full_vars = len(wavelengths)

    # Build preprocessing pipeline
    pipe_steps = build_preprocessing_pipeline(
        preprocess_cfg["name"],
        preprocess_cfg["deriv"],
        preprocess_cfg["window"],
        preprocess_cfg["polyorder"],
    )

    # For PLS-DA, we need PLS + LogisticRegression
    if model_name == "PLS-DA":
        pipe_steps.append(("pls", model))
        pipe_steps.append(("lr", LogisticRegression(max_iter=1000, random_state=42)))
    else:
        pipe_steps.append(("model", model))

    pipe = Pipeline(pipe_steps) if pipe_steps else model

    # Run CV
    cv_metrics = []
    for train_idx, test_idx in cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit and predict
        pipe.fit(X_train, y_train)

        if task_type == "regression":
            y_pred = pipe.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            cv_metrics.append({"RMSE": rmse, "R2": r2})
        else:  # classification
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # ROC AUC
            try:
                if is_binary_classification:
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                else:
                    y_proba = pipe.predict_proba(X_test)
                    y_test_bin = label_binarize(y_test, classes=np.unique(y))
                    auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
            except Exception:
                auc = np.nan

            cv_metrics.append({"Accuracy": acc, "ROC_AUC": auc})

    # Average metrics
    if task_type == "regression":
        mean_rmse = np.mean([m["RMSE"] for m in cv_metrics])
        mean_r2 = np.mean([m["R2"] for m in cv_metrics])
    else:
        mean_acc = np.mean([m["Accuracy"] for m in cv_metrics])
        mean_auc = np.mean([m["ROC_AUC"] for m in cv_metrics if not np.isnan(m["ROC_AUC"])])

    # Extract LVs (for PLS models)
    lvs = params.get("n_components", np.nan)

    # Build result dictionary
    result = {
        "Task": task_type,
        "Model": model_name,
        "Params": str(params),
        "Preprocess": preprocess_cfg["name"],
        "Deriv": preprocess_cfg["deriv"],
        "Window": preprocess_cfg["window"],
        "Poly": preprocess_cfg["polyorder"],
        "LVs": lvs,
        "n_vars": n_vars,
        "full_vars": full_vars,
        "SubsetTag": subset_tag,
    }

    if task_type == "regression":
        result["RMSE"] = mean_rmse
        result["R2"] = mean_r2
    else:
        result["Accuracy"] = mean_acc
        result["ROC_AUC"] = mean_auc

    return result
