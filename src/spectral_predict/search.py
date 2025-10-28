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
from .regions import create_region_subsets, format_region_report


def run_search(X, y, task_type, folds=5, lambda_penalty=0.15, progress_callback=None):
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
    progress_callback : callable, optional
        Function to call with progress updates. Should accept dict with keys:
        - 'stage': Current stage (e.g., 'preprocessing', 'model_testing')
        - 'message': Status message
        - 'current': Current item number
        - 'total': Total items
        - 'best_model': Best model found so far (dict with RMSE/R2 or Acc/AUC)

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
    print()

    # Analyze spectral regions
    print("Analyzing spectral regions...")
    if progress_callback:
        progress_callback({
            'stage': 'region_analysis',
            'message': 'Analyzing spectral regions...',
            'current': 0,
            'total': 1
        })

    region_subsets = create_region_subsets(X_np, y_np, wavelengths, n_top_regions=5)
    if len(region_subsets) > 0:
        print(f"  Identified {len(region_subsets)} region-based variable subsets")
        for subset in region_subsets[:3]:  # Show top 3
            print(f"  - {subset['description']}")
    else:
        print("  Warning: Could not identify spectral regions")
    print()

    # Calculate total number of configurations for progress tracking
    total_configs = 0
    for model_name, model_configs in model_grids.items():
        total_configs += len(model_configs) * len(preprocess_configs)

    current_config = 0
    best_model_so_far = None

    # Main search loop
    for preprocess_cfg in preprocess_configs:
        for model_name, model_configs in model_grids.items():
            for model, params in model_configs:
                current_config += 1

                # Progress update
                prep_name = preprocess_cfg["name"]
                if preprocess_cfg["deriv"]:
                    prep_name += f"_d{preprocess_cfg['deriv']}"

                progress_msg = f"Testing {model_name} with {prep_name} preprocessing"
                print(f"[{current_config}/{total_configs}] {progress_msg}")

                if progress_callback:
                    progress_callback({
                        'stage': 'model_testing',
                        'message': progress_msg,
                        'current': current_config,
                        'total': total_configs,
                        'best_model': best_model_so_far
                    })

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

                # Update best model tracker
                if best_model_so_far is None:
                    best_model_so_far = result
                else:
                    if task_type == "regression":
                        if result["RMSE"] < best_model_so_far["RMSE"]:
                            best_model_so_far = result
                    else:  # classification
                        if result.get("ROC_AUC", 0) > best_model_so_far.get("ROC_AUC", 0):
                            best_model_so_far = result

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

                        # Expanded variable selection grid
                        # Logarithmic spacing: 10, 20, 50, 100, 250, 500, 1000, etc.
                        variable_counts = [10, 20, 50, 100, 250, 500, 1000]

                        # Only test counts that are less than total features
                        variable_counts = [n for n in variable_counts if n < n_features]

                        # Run subsets with expanded grid
                        for n_top in variable_counts:
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

                        # Run region-based subsets (only for raw and snv preprocessing)
                        # Skip for derivative preprocessing to avoid redundancy
                        if preprocess_cfg["deriv"] is None and len(region_subsets) > 0:
                            for region_subset in region_subsets:
                                region_result = _run_single_config(
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
                                    subset_indices=region_subset['indices'],
                                    subset_tag=region_subset['tag'],
                                )
                                df_results = add_result(df_results, region_result)

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
