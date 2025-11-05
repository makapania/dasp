"""Model search with cross-validation and subset selection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
from joblib import Parallel, delayed

from .preprocess import build_preprocessing_pipeline
from .models import get_model_grids, get_feature_importances
from .scoring import create_results_dataframe, add_result
from .regions import create_region_subsets, format_region_report
from .variable_selection import spa_selection, uve_selection, uve_spa_selection, ipls_selection


def run_search(X, y, task_type, folds=5, lambda_penalty=0.15, max_n_components=24,
               max_iter=500, models_to_test=None, preprocessing_methods=None,
               window_sizes=None, n_estimators_list=None, learning_rates=None,
               enable_variable_subsets=True, variable_counts=None,
               enable_region_subsets=True, n_top_regions=5, progress_callback=None,
               variable_selection_methods=None, apply_uve_prefilter=False,
               uve_cutoff_multiplier=1.0, uve_n_components=None,
               spa_n_random_starts=10, ipls_n_intervals=20):
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
    max_n_components : int, default=24
        Maximum number of PLS components to test
    max_iter : int, default=500
        Maximum iterations for MLP
    models_to_test : list of str, optional
        List of model names to test (e.g., ['PLS', 'RandomForest', 'MLP', 'NeuralBoosted'])
        If None, all models are tested
    enable_variable_subsets : bool, default=True
        Enable top-N variable subset analysis
    variable_counts : list of int, optional
        Variable counts to test (e.g., [10, 20, 50])
    enable_region_subsets : bool, default=True
        Enable spectral region subset analysis
    n_top_regions : int, default=5
        Number of top regions to analyze (5, 10, 15, or 20)
    progress_callback : callable, optional
        Function to call with progress updates. Should accept dict with keys:
        - 'stage': Current stage (e.g., 'preprocessing', 'model_testing')
        - 'message': Status message
        - 'current': Current item number
        - 'total': Total items
        - 'best_model': Best model found so far (dict with RMSE/R2 or Acc/AUC)
    variable_selection_methods : list of str or None, default=None
        List of variable selection methods to use. Can include multiple methods:
        'importance', 'spa'/'SPA', 'uve'/'UVE', 'uve_spa'/'UVE-SPA', 'ipls'/'iPLS'.
        Accepts both Python-style (lowercase) and Julia-style (uppercase) naming.
        If None, defaults to ['importance']. All methods are fully functional.
    apply_uve_prefilter : bool, default=False
        Placeholder flag indicating whether to run a UVE prefilter step.
    uve_cutoff_multiplier : float, default=1.0
        Placeholder parameter for UVE cutoff scaling.
    uve_n_components : int or None, default=None
        Placeholder for specifying component count for UVE.
    spa_n_random_starts : int, default=10
        Placeholder for SPA random restarts.
    ipls_n_intervals : int, default=20
        Placeholder for interval count in iPLS selection.

    Returns
    -------
    df_ranked : pd.DataFrame
        Ranked results with all model runs
    """
    X_np = X.values
    y_np = y.values
    wavelengths = X.columns.values
    n_features = X_np.shape[1]
    n_samples = X_np.shape[0]

    # Create results container
    df_results = create_results_dataframe(task_type)

    # Handle variable selection methods (support multiple methods)
    if variable_selection_methods is None or not variable_selection_methods:
        variable_selection_methods = ['importance']

    # Normalize method names to lowercase and handle Julia-style names
    # Julia uses: 'SPA', 'UVE', 'iPLS', 'UVE-SPA'
    # Python uses: 'spa', 'uve', 'ipls', 'uve_spa'
    method_map = {
        'SPA': 'spa',
        'UVE': 'uve',
        'IPLS': 'ipls',
        'iPLS': 'ipls',
        'UVE-SPA': 'uve_spa',
        'UVE_SPA': 'uve_spa',
        'importance': 'importance'
    }

    # Normalize input methods
    normalized_methods = []
    for m in variable_selection_methods:
        # Try exact match first (case-sensitive)
        if m in method_map:
            normalized_methods.append(method_map[m])
        # Try lowercase version
        elif m.lower() in ['importance', 'spa', 'uve', 'ipls', 'uve_spa']:
            normalized_methods.append(m.lower())
        else:
            # Unknown method, keep as-is for error reporting
            normalized_methods.append(m)

    variable_selection_methods = normalized_methods

    # Filter to only implemented methods
    implemented_methods = ['importance', 'spa', 'uve', 'uve_spa', 'ipls']  # All methods now functional
    selected_methods = [m for m in variable_selection_methods if m in implemented_methods]

    # Warn about unimplemented methods
    unimplemented = [m for m in variable_selection_methods if m not in implemented_methods]
    if unimplemented:
        print(f"Info: Variable selection methods {unimplemented} are not yet implemented.")
        print(f"      Continuing with implemented methods: {selected_methods}")

    # Ensure at least one method is selected
    if not selected_methods:
        selected_methods = ['importance']
        print("Info: No implemented methods selected. Defaulting to 'importance'.")
    if apply_uve_prefilter or uve_n_components or uve_cutoff_multiplier != 1.0:
        print("Info: UVE prefilter parameters are currently placeholders in the Python backend.")
    if spa_n_random_starts != 10:
        print("Info: SPA random starts parameter is noted but not yet applied in the Python backend.")
    if ipls_n_intervals != 20:
        print("Info: iPLS interval parameter is noted but not yet applied in the Python backend.")

    # Determine if classification is binary or multiclass
    is_binary_classification = False
    if task_type == "classification":
        n_classes = len(np.unique(y_np))
        is_binary_classification = n_classes == 2

    # Adjust max_n_components based on minimum CV fold size
    # PLS requires n_components <= min(n_features, n_samples_in_fold)
    # With k-fold CV, smallest fold has roughly n_samples / folds samples
    min_fold_samples = n_samples // folds
    # Use slightly lower bound to be safe (some folds might have fewer samples)
    safe_max_components = min(max_n_components, min_fold_samples - 1, n_features)

    if safe_max_components < max_n_components:
        print(f"Note: Reducing max components from {max_n_components} to {safe_max_components} " +
              f"due to dataset size (n_samples={n_samples}, min_fold_size~{min_fold_samples})")

    # Get model grids (pass n_estimators_list and learning_rates for NeuralBoosted)
    model_grids = get_model_grids(task_type, n_features, safe_max_components, max_iter,
                                   n_estimators_list=n_estimators_list, learning_rates=learning_rates)

    # Filter models if models_to_test is specified
    if models_to_test is not None:
        # Filter to only requested models
        model_grids = {name: configs for name, configs in model_grids.items()
                      if name in models_to_test}

        if not model_grids:
            raise ValueError(f"No valid models found. Available: {list(get_model_grids(task_type, n_features, safe_max_components, max_iter).keys())}, Requested: {models_to_test}")

    # Define preprocessing configurations based on user selections
    # Use preprocessing_methods dict if provided, otherwise default to all
    if preprocessing_methods is None:
        preprocessing_methods = {
            'raw': True,
            'snv': True,
            'sg1': True,
            'sg2': True,
            'deriv_snv': True
        }

    # Use window_sizes list if provided, otherwise default to [7, 19]
    if window_sizes is None:
        window_sizes = [7, 19]

    preprocess_configs = []

    # Add raw if selected
    if preprocessing_methods.get('raw', False):
        preprocess_configs.append({"name": "raw", "deriv": None, "window": None, "polyorder": None})

    # Add SNV if selected
    if preprocessing_methods.get('snv', False):
        preprocess_configs.append({"name": "snv", "deriv": None, "window": None, "polyorder": None})

    # Add derivative configs based on user selections
    # For each derivative type, we create:
    # 1. Pure derivative (deriv)
    # 2. SNV then derivative (snv_deriv) - if SNV is also selected
    # 3. Derivative then SNV (deriv_snv) - if deriv_snv checkbox is selected

    if preprocessing_methods.get('sg1', False):
        # 1st derivative only
        for window in window_sizes:
            preprocess_configs.append(
                {"name": "deriv", "deriv": 1, "window": window, "polyorder": 2}
            )

        # If SNV is also selected, add SNV → derivative combination
        if preprocessing_methods.get('snv', False):
            for window in window_sizes:
                preprocess_configs.append(
                    {"name": "snv_deriv", "deriv": 1, "window": window, "polyorder": 2}
                )

        # If deriv_snv is selected, add derivative → SNV combination for 1st deriv
        if preprocessing_methods.get('deriv_snv', False):
            for window in window_sizes:
                preprocess_configs.append(
                    {"name": "deriv_snv", "deriv": 1, "window": window, "polyorder": 2}
                )

    if preprocessing_methods.get('sg2', False):
        # 2nd derivative only
        for window in window_sizes:
            preprocess_configs.append(
                {"name": "deriv", "deriv": 2, "window": window, "polyorder": 3}
            )

        # If SNV is also selected, add SNV → derivative combination
        if preprocessing_methods.get('snv', False):
            for window in window_sizes:
                preprocess_configs.append(
                    {"name": "snv_deriv", "deriv": 2, "window": window, "polyorder": 3}
                )

        # If deriv_snv is selected, add derivative → SNV combination for 2nd deriv
        if preprocessing_methods.get('deriv_snv', False):
            for window in window_sizes:
                preprocess_configs.append(
                    {"name": "deriv_snv", "deriv": 2, "window": window, "polyorder": 3}
                )

    # If no preprocessing methods selected, default to raw
    if not preprocess_configs:
        print("Warning: No preprocessing methods selected. Defaulting to raw.")
        preprocess_configs.append({"name": "raw", "deriv": None, "window": None, "polyorder": None})

    # Create CV splitter
    if task_type == "regression":
        cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        cv_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    print(f"Running {task_type} search with {folds}-fold CV...")
    print(f"Models: {list(model_grids.keys())}")
    print(f"Preprocessing configs: {len(preprocess_configs)}")
    print(f"\nPreprocessing breakdown:")
    for cfg in preprocess_configs:
        cfg_name = cfg['name']
        if cfg['deriv'] is not None:
            print(f"  - {cfg_name} (deriv={cfg['deriv']}, window={cfg['window']})")
        else:
            print(f"  - {cfg_name}")
    print(f"\nEnable variable subsets: {enable_variable_subsets}")
    print(f"Variable counts: {variable_counts}")
    print(f"Enable region subsets: {enable_region_subsets}")
    print()

    # Note: Spectral region analysis is now done per preprocessing method
    # (inside the main loop) to ensure regions are computed on preprocessed data

    # Calculate total number of configurations for progress tracking
    total_configs = 0
    for model_name, model_configs in model_grids.items():
        total_configs += len(model_configs) * len(preprocess_configs)

    current_config = 0
    best_model_so_far = None

    # Main search loop
    for preprocess_cfg in preprocess_configs:
        # Compute region subsets on preprocessed data for this preprocessing method
        # This ensures regions are based on the actual preprocessed features
        # Region subsets now work for ALL preprocessing types including derivatives
        region_subsets = []
        X_preprocessed_for_regions = None
        if enable_region_subsets:
            try:
                # Build preprocessing pipeline
                prep_pipe_steps = build_preprocessing_pipeline(
                    preprocess_cfg["name"],
                    preprocess_cfg["deriv"],
                    preprocess_cfg["window"],
                    preprocess_cfg["polyorder"],
                )

                # Transform X through preprocessing
                X_preprocessed_for_regions = X_np.copy()
                if prep_pipe_steps:
                    prep_pipeline = Pipeline(prep_pipe_steps)
                    X_preprocessed_for_regions = prep_pipeline.fit_transform(X_preprocessed_for_regions, y_np)

                # Compute region subsets on preprocessed data
                # For derivatives: regions will be in derivative feature space
                # Convert wavelengths to float (they come from DataFrame columns as strings)
                wavelengths_float = np.array([float(w) for w in wavelengths])
                region_subsets = create_region_subsets(X_preprocessed_for_regions, y_np, wavelengths_float, n_top_regions=n_top_regions)

                if len(region_subsets) > 0:
                    prep_name = str(preprocess_cfg.get("name", "unknown"))
                    deriv_info = f"_d{preprocess_cfg['deriv']}" if preprocess_cfg["deriv"] else ""
                    print(f"  Region analysis for {prep_name}{deriv_info}: Identified {len(region_subsets)} region-based subsets")
            except Exception as e:
                prep_name = str(preprocess_cfg.get("name", "unknown"))
                print(f"  Warning: Could not compute region subsets for {prep_name}: {e}")
                # Uncomment for debugging:
                # import traceback
                # traceback.print_exc()
                region_subsets = []

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

                # For PLS, Ridge, Lasso, RF, MLP, and NeuralBoosted: compute feature importances and run subsets
                # IMPORTANT: Importances are computed on PREPROCESSED data, ensuring that
                # wavelength selection reflects the actual transformed features the model sees
                if model_name in ["PLS", "PLS-DA", "Ridge", "Lasso", "RandomForest", "MLP", "NeuralBoosted"]:
                    if not enable_variable_subsets:
                        print(f"  ⊗ Skipping subset analysis for {model_name} (variable subsets disabled)")
                    else:
                        print(f"  -> Computing feature importances for {model_name} subset analysis...")

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
                        # This ensures importances are based on PREPROCESSED features
                        if hasattr(pipe, "named_steps") and len(pipe_steps) > 1:
                            # Transform through preprocessing
                            X_transformed = X_np
                            for step_name, transformer in pipe.steps[:-1]:
                                X_transformed = transformer.transform(X_transformed)
                        else:
                            X_transformed = X_np

                        # Loop over each selected variable selection method
                        for varsel_method in selected_methods:
                            # Get importances computed on preprocessed data
                            try:
                                if varsel_method == 'importance':
                                    importances = get_feature_importances(
                                        fitted_model, model_name, X_transformed, y_np
                                    )

                                elif varsel_method == 'spa':
                                    # SPA: Successive Projections Algorithm - reduces collinearity
                                    # Select minimally correlated variables
                                    n_to_select = min(n_top, n_features_for_validation)
                                    importances = spa_selection(
                                        X_transformed, y_np,
                                        n_features=n_to_select,
                                        n_random_starts=spa_n_random_starts,
                                        cv_folds=folds
                                    )

                                elif varsel_method == 'uve':
                                    # UVE: Uninformative Variable Elimination - filters noise
                                    importances = uve_selection(
                                        X_transformed, y_np,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        n_components=uve_n_components,
                                        cv_folds=folds
                                    )

                                elif varsel_method == 'uve_spa':
                                    # UVE-SPA: Hybrid method - filters noise then reduces collinearity
                                    n_to_select = min(n_top, n_features_for_validation)
                                    importances = uve_spa_selection(
                                        X_transformed, y_np,
                                        n_features=n_to_select,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        uve_n_components=uve_n_components,
                                        uve_cv_folds=folds,
                                        spa_n_random_starts=spa_n_random_starts,
                                        spa_cv_folds=folds
                                    )

                                elif varsel_method == 'ipls':
                                    # iPLS: Interval PLS - selects based on spectral regions
                                    importances = ipls_selection(
                                        X_transformed, y_np,
                                        n_intervals=ipls_n_intervals,
                                        n_components=uve_n_components,
                                        cv_folds=folds
                                    )

                                else:
                                    # This shouldn't happen due to filtering, but handle gracefully
                                    print(f"  -> Skipping unimplemented method '{varsel_method}'")
                                    continue

                                # Use user-specified variable counts, or default if not provided
                                if variable_counts is None:
                                    user_variable_counts = [10, 20, 50, 100, 250, 500, 1000]
                                else:
                                    user_variable_counts = variable_counts

                                # For validation, use the feature count from the PREPROCESSED data
                                # (derivatives reduce feature count, so use transformed shape)
                                n_features_for_validation = X_transformed.shape[1]

                                # Only test counts that are less than total features
                                valid_variable_counts = [n for n in user_variable_counts if n < n_features_for_validation]

                                print(f"  -> User variable counts: {user_variable_counts}")
                                print(f"  -> Valid variable counts (< {n_features_for_validation} features): {valid_variable_counts}")
                                print(f"  -> Variable selection method: {varsel_method}")

                                if not valid_variable_counts:
                                    print(f"  ⚠ Warning: No valid variable counts to test (all selected counts >= {n_features_for_validation} features)")

                                # Run subsets with user-selected counts
                                for n_top in valid_variable_counts:
                                    print(f"  -> Testing top-{n_top} variable subset (method: {varsel_method})...")
                                    # Select top N most important features based on preprocessed importances
                                    top_indices = np.argsort(importances)[-n_top:][::-1]

                                    # For derivative preprocessing: importances are computed on transformed features
                                    # We must use the TRANSFORMED data and skip reapplying preprocessing
                                    # Otherwise window size (e.g., 17) > n_features (e.g., 10) causes errors
                                    if preprocess_cfg["deriv"] is not None:
                                        # Use preprocessed data, skip reprocessing
                                        # Keep original preprocess_cfg for correct labeling in results
                                        subset_result = _run_single_config(
                                            X_transformed,
                                            y_np,
                                            wavelengths,
                                            model,
                                            model_name,
                                            params,
                                            preprocess_cfg,  # Keep original config for labeling
                                            cv_splitter,
                                            task_type,
                                            is_binary_classification,
                                            subset_indices=top_indices,
                                            subset_tag=f"top{n_top}_{varsel_method}",
                                            skip_preprocessing=True,  # Flag to skip reapplying
                                        )
                                    else:
                                        # For raw/SNV: indices map to original wavelengths, can reapply preprocessing
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
                                            subset_tag=f"top{n_top}_{varsel_method}",
                                        )
                                    df_results = add_result(df_results, subset_result)

                            except Exception as e:
                                print(f"Warning: Could not compute importances for {model_name} with method '{varsel_method}': {e}")

                # Run region-based subsets for ALL models (not just PLS/RF/MLP/NeuralBoosted)
                # For derivatives: use preprocessed data to avoid reapplying preprocessing
                # For raw/SNV: use raw data and reapply preprocessing
                if enable_region_subsets and len(region_subsets) > 0:
                    print(f"  -> Testing {len(region_subsets)} region-based subsets...")
                    for region_subset in region_subsets:
                        if preprocess_cfg["deriv"] is not None:
                            # For derivatives: use preprocessed data, skip reprocessing
                            # Keep original preprocess_cfg for correct labeling
                            region_result = _run_single_config(
                                X_preprocessed_for_regions,
                                y_np,
                                wavelengths,
                                model,
                                model_name,
                                params,
                                preprocess_cfg,  # Keep original config for labeling
                                cv_splitter,
                                task_type,
                                is_binary_classification,
                                subset_indices=region_subset['indices'],
                                subset_tag=region_subset['tag'],
                                skip_preprocessing=True,  # Flag to skip reapplying
                            )
                        else:
                            # For raw/SNV: use raw data, reapply preprocessing
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

    # Compute composite scores and rank
    from .scoring import compute_composite_score

    df_ranked = compute_composite_score(df_results, task_type, lambda_penalty)

    return df_ranked


def _run_single_fold(pipe, X, y, train_idx, test_idx, task_type, is_binary_classification):
    """
    Run a single CV fold in parallel.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline to fit (will be cloned)
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    train_idx : ndarray
        Training indices
    test_idx : ndarray
        Test indices
    task_type : str
        'regression' or 'classification'
    is_binary_classification : bool
        Whether this is binary classification

    Returns
    -------
    metrics : dict
        Dictionary with fold metrics
    """
    # Clone pipeline to avoid thread-safety issues
    pipe_clone = clone(pipe)

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit and predict
    pipe_clone.fit(X_train, y_train)

    if task_type == "regression":
        y_pred = pipe_clone.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": rmse, "R2": r2}
    else:  # classification
        y_pred = pipe_clone.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # ROC AUC
        try:
            if is_binary_classification:
                y_proba = pipe_clone.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            else:
                y_proba = pipe_clone.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=np.unique(y))
                auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        except Exception:
            auc = np.nan

        return {"Accuracy": acc, "ROC_AUC": auc}


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
    top_n_vars=30,
    skip_preprocessing=False,
):
    """
    Run a single model configuration with CV.

    Parameters
    ----------
    skip_preprocessing : bool, default=False
        If True, skip building preprocessing pipeline (data is already preprocessed).
        Used for derivative subsets where preprocessing was already applied.

    Returns
    -------
    dict
        Dictionary with results, including top important variables.
    """
    # Apply subset if specified
    if subset_indices is not None:
        X = X[:, subset_indices]
        n_vars = len(subset_indices)
    else:
        n_vars = X.shape[1]

    full_vars = len(wavelengths)

    # Build preprocessing pipeline (skip if data is already preprocessed)
    if skip_preprocessing:
        pipe_steps = []
    else:
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

    # Run CV in parallel (use n_jobs=-1 to use all available cores)
    cv_metrics = Parallel(n_jobs=-1, backend='loky')(
        delayed(_run_single_fold)(
            pipe, X, y, train_idx, test_idx, task_type, is_binary_classification
        )
        for train_idx, test_idx in cv_splitter.split(X, y)
    )

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

    # Extract individual hyperparameters for easier viewing in GUI
    # These will appear as separate columns in the results table

    # Ridge/Lasso: alpha (regularization strength)
    if "alpha" in params:
        result["Alpha"] = params["alpha"]

    # RandomForest: n_estimators and max_depth
    if "n_estimators" in params:
        result["n_estimators"] = params["n_estimators"]
    if "max_depth" in params:
        result["max_depth"] = params["max_depth"] if params["max_depth"] is not None else "None"

    # MLP: hidden_layer_sizes, alpha, learning_rate_init
    if "hidden_layer_sizes" in params:
        # Convert tuple to readable string (e.g., (64,) -> "64", (128,64) -> "128-64")
        hidden = params["hidden_layer_sizes"]
        if isinstance(hidden, tuple):
            result["Hidden"] = "-".join(map(str, hidden))
        else:
            result["Hidden"] = str(hidden)
    if "learning_rate_init" in params:
        result["LR_init"] = params["learning_rate_init"]

    # NeuralBoosted: n_estimators, learning_rate, hidden_layer_size, activation
    if "learning_rate" in params:  # NeuralBoosted uses "learning_rate" not "learning_rate_init"
        result["LearningRate"] = params["learning_rate"]
    if "hidden_layer_size" in params:  # NeuralBoosted uses singular "hidden_layer_size"
        result["HiddenSize"] = params["hidden_layer_size"]
    if "activation" in params:
        result["Activation"] = params["activation"]

    if task_type == "regression":
        result["RMSE"] = mean_rmse
        result["R2"] = mean_r2
    else:
        result["Accuracy"] = mean_acc
        result["ROC_AUC"] = mean_auc

    # Extract top important variables/wavelengths
    # Refit on full data to get feature importances
    if model_name in ["PLS", "PLS-DA", "Ridge", "Lasso", "RandomForest", "MLP", "NeuralBoosted"]:
        try:
            # Refit the pipeline on full data
            pipe.fit(X, y)

            # Get the fitted model from pipeline
            fitted_model = (
                pipe.named_steps["model"] if hasattr(pipe, "named_steps") else pipe
            )

            # For PLS-DA, get the PLS component
            if model_name == "PLS-DA" and hasattr(pipe, "named_steps"):
                fitted_model = pipe.named_steps["pls"]

            # Get transformed X for importance calculation
            if hasattr(pipe, "named_steps") and len(pipe.steps) > 1:
                X_transformed = X
                for step_name, transformer in pipe.steps[:-1]:
                    if step_name != "lr":  # Skip logistic regression for PLS-DA
                        X_transformed = transformer.transform(X_transformed)
            else:
                X_transformed = X

            # Compute importances
            importances = get_feature_importances(
                fitted_model, model_name, X_transformed, y
            )

            # For subset models: save ALL wavelengths used (not just top 30)
            # This fixes the variable count mismatch when loading models for refinement
            if subset_tag != "full" and subset_indices is not None:
                # Save ALL wavelengths used in the subset model
                all_indices = np.arange(len(importances))
                if subset_indices is not None:
                    original_wavelengths_all = wavelengths[subset_indices]
                    all_wavelengths = original_wavelengths_all[all_indices]
                else:
                    all_wavelengths = wavelengths[all_indices]

                all_vars_str = ','.join([f"{w:.1f}" for w in all_wavelengths])
                result['all_vars'] = all_vars_str
            else:
                result['all_vars'] = 'N/A'

            # Get top N features for display purposes (always top 30)
            n_to_select = min(top_n_vars, len(importances))
            top_indices = np.argsort(importances)[-n_to_select:][::-1]

            # Map back to original wavelengths
            if subset_indices is not None:
                # We're working with a subset, map indices back to original wavelengths
                original_wavelengths = wavelengths[subset_indices]
                top_wavelengths = original_wavelengths[top_indices]
            else:
                # Full spectrum
                top_wavelengths = wavelengths[top_indices]

            # Format as comma-separated string
            top_vars_str = ','.join([f"{w:.1f}" for w in top_wavelengths])
            result['top_vars'] = top_vars_str

        except Exception as e:
            # If anything fails, just mark as N/A
            result['top_vars'] = 'N/A'
            result['all_vars'] = 'N/A'
    else:
        # For models that don't support importance extraction
        result['top_vars'] = 'N/A'
        result['all_vars'] = 'N/A'

    return result
