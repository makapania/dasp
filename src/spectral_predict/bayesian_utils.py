"""
Utility functions for Bayesian hyperparameter optimization with Optuna.

This module provides helper functions for:
    - Creating reproducible Optuna studies
    - Converting parameters between formats
    - Handling pruning and early stopping
    - Error handling and validation
"""

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, PercentilePruner
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
from .regions import create_region_subsets


# Configure logging for Optuna (suppress verbose output)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_optuna_study(
    direction: str = 'minimize',
    sampler: str = 'TPE',
    pruner: Optional[str] = None,
    random_state: int = 42,
    study_name: Optional[str] = None
) -> optuna.Study:
    """
    Create an Optuna study with specified configuration.

    Parameters
    ----------
    direction : str, default='minimize'
        Optimization direction ('minimize' for RMSE, 'maximize' for R²)
    sampler : str, default='TPE'
        Sampling algorithm:
            - 'TPE': Tree-structured Parzen Estimator (recommended, smart sampling)
            - 'Random': Random sampling (baseline)
    pruner : str or None, default=None
        Pruning strategy for early stopping:
            - None: No pruning
            - 'Median': Stop if trial is worse than median
            - 'Halving': Successive halving (aggressive)
    random_state : int, default=42
        Random seed for reproducibility
    study_name : str, optional
        Name for the study (for logging)

    Returns
    -------
    study : optuna.Study
        Configured Optuna study object

    Notes
    -----
    TPE sampler:
        - Uses Bayesian optimization to suggest promising parameters
        - Learns from previous trials
        - Typically finds good solutions in 20-50 trials

    Median pruner:
        - Stops trials that perform worse than median after K steps
        - Saves computation time on unpromising configurations
        - Safe default for most cases
    """
    # Configure sampler
    if sampler == 'TPE':
        sampler_obj = TPESampler(
            seed=random_state,
            n_startup_trials=10,  # Random exploration first
            n_ei_candidates=24,   # Number of candidates for expected improvement
            multivariate=True     # Consider parameter interactions
        )
    elif sampler == 'Random':
        sampler_obj = RandomSampler(seed=random_state)
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Use 'TPE' or 'Random'")

    # Configure pruner
    if pruner == 'Median':
        # Use PercentilePruner with 25th percentile (less aggressive than MedianPruner)
        # This keeps top 75% of trials instead of top 50%, avoiding over-pruning
        pruner_obj = PercentilePruner(
            percentile=25,       # Keep trials in top 75% (less aggressive)
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=2     # Wait 2 CV folds before pruning
        )
    elif pruner == 'Halving':
        pruner_obj = SuccessiveHalvingPruner(
            min_resource=1,      # Start with 1 fold
            reduction_factor=3   # Keep top 1/3 trials
        )
    elif pruner is None:
        pruner_obj = None
    else:
        raise ValueError(f"Unknown pruner: {pruner}. Use 'Median', 'Halving', or None")

    # Create study (in-memory, no database)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler_obj,
        pruner=pruner_obj,
        study_name=study_name,
        storage=None  # In-memory (no SQLite locking issues)
    )

    return study


def create_objective_function(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    preprocess_cfg: Dict,
    cv_splitter,
    task_type: str,
    is_binary_classification: bool,
    run_single_config_fn: Callable,
    tier: str = 'standard',
    n_features: int = None,
    max_n_components: int = 8,
    enable_variable_subsets: bool = True,
    variable_counts: list = None,
    variable_selection_methods: list = None,
    enable_region_subsets: bool = False,
    n_top_regions: int = 5,
    **kwargs
) -> Callable:
    """
    Create objective function for Optuna optimization.

    Parameters
    ----------
    model_name : str
        Name of model to optimize
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    wavelengths : np.ndarray
        Wavelength values
    preprocess_cfg : dict
        Preprocessing configuration
    cv_splitter : sklearn splitter
        Cross-validation splitter
    task_type : str
        'regression' or 'classification'
    is_binary_classification : bool
        Whether classification is binary
    run_single_config_fn : callable
        Function to run single model config (from search.py)
    tier : str
        Model tier level
    n_features : int, optional
        Number of features
    max_n_components : int
        Maximum PLS components
    enable_variable_subsets : bool, default=True
        Whether to test variable subsets (like grid search)
    variable_counts : list, optional
        List of variable counts to test (e.g., [10, 20, 50, 100, 250, 500, 1000])
    variable_selection_methods : list, optional
        List of variable selection methods to use (e.g., ['importance', 'spa', 'uve'])
    enable_region_subsets : bool, default=False
        Whether to test regional subsets (like grid search)
    n_top_regions : int, default=5
        Number of top spectral regions to test
    **kwargs : dict
        Additional parameters for run_single_config

    Returns
    -------
    objective : callable
        Objective function for Optuna

    Notes
    -----
    The objective function:
    1. Suggests hyperparameters using Optuna trial
    2. Trains full model with CV using existing infrastructure
    3. If variable subsets enabled: tests variable subsets (like grid search)
    4. If region subsets enabled: tests regional subsets (like grid search)
    5. Returns best metric (from full model or any subset)

    When variable subsets are enabled, each trial tests:
    - Full model (all features)
    - Top-10, top-20, top-50, etc. (configurable via variable_counts)

    When region subsets are enabled, each trial tests:
    - Individual spectral regions (e.g., NIR protein region 1900-2100nm)

    This mirrors grid search behavior with multiple configurations per hyperparameter setting.
    """
    from .bayesian_config import get_bayesian_search_space
    from .models import build_model
    from .model_registry import supports_subset_analysis
    from .models import get_feature_importances
    from .variable_selection import spa_selection, uve_selection, uve_spa_selection, ipls_selection, cars_selection
    from .wavelength_selection import vcpa_iriv

    # Calculate n_classes for classification tasks
    n_classes = len(np.unique(y)) if task_type == 'classification' else 2

    # Set default variable counts if not provided
    if variable_counts is None:
        variable_counts = [10, 20, 50, 100, 250, 500, 1000]

    # Set default variable selection methods if not provided
    # Bayesian should automatically explore multiple methods to find the best
    if variable_selection_methods is None:
        variable_selection_methods = ['importance', 'spa', 'uve', 'cars']

    # Compute regional subsets if enabled (ONCE, before trials start)
    region_subsets = []
    if enable_region_subsets:
        try:
            wavelengths_float = wavelengths.astype(float)
            region_subsets = create_region_subsets(
                X, y, wavelengths_float,
                n_top_regions=n_top_regions
            )
            print(f"  Identified {len(region_subsets)} spectral regions for testing")
        except Exception as e:
            logging.warning(f"Region creation failed: {e}")
            region_subsets = []

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object

        Returns
        -------
        metric : float
            Best metric across all configurations (full + subsets)
            For regression: minimum RMSE
            For classification: maximum accuracy (returned as negative for minimization)
        """
        # Get hyperparameters from Optuna
        params = get_bayesian_search_space(
            model_name=model_name,
            trial=trial,
            tier=tier,
            n_features=n_features,
            max_n_components=max_n_components,
            task_type=task_type,
            n_classes=n_classes
        )

        # Build model with suggested parameters
        model = build_model(model_name, params, task_type=task_type)

        # Filter kwargs to only include parameters that _run_single_config accepts
        # Remove progress_callback and n_trials which are used by Bayesian optimization but not by _run_single_config
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['progress_callback', 'n_trials']}

        # Track all results for this trial (full model + subsets)
        trial_results = []

        # Run cross-validation using existing infrastructure
        try:
            # === STEP 1: Test full model (all features) ===
            full_result = run_single_config_fn(
                X, y, wavelengths,
                model, model_name, params,
                preprocess_cfg, cv_splitter, task_type,
                is_binary_classification,
                skip_preprocessing=True,  # Already preprocessed
                subset_tag="full",
                **filtered_kwargs
            )
            trial_results.append(full_result)

            # Get base metric from full model
            if task_type == 'regression':
                best_metric = full_result['RMSE']
                best_r2 = full_result.get('R2', np.nan)
            else:
                best_metric = -full_result['Accuracy']  # Negative for minimization
                best_auc = full_result.get('ROC_AUC', np.nan)

            # === STEP 2: Test variable subsets (if enabled and model supports it) ===
            if enable_variable_subsets and supports_subset_analysis(model_name):
                # Fit model on full data to compute feature importances
                from sklearn.pipeline import Pipeline

                # Build model-only pipeline (data is already preprocessed)
                pipe_steps = [("model", model)]
                pipe = Pipeline(pipe_steps)
                pipe.fit(X, y)
                fitted_model = pipe.named_steps["model"]

                # Filter to valid variable counts (< total features)
                n_features_available = X.shape[1]
                valid_variable_counts = [n for n in variable_counts if n < n_features_available]

                # Only proceed if there are valid variable counts to test
                if valid_variable_counts:
                    # Loop over each variable selection method
                    for varsel_method in variable_selection_methods:
                        try:
                            # Compute feature importances based on method
                            if varsel_method == 'importance':
                                importances = get_feature_importances(
                                    fitted_model, model_name, X, y
                                )
                            elif varsel_method == 'spa':
                                # SPA: Successive Projections Algorithm
                                n_to_select = min(max(valid_variable_counts), n_features_available)
                                folds = filtered_kwargs.get('folds', 5)
                                random_state = 42  # Use fixed random state
                                spa_n_random_starts = 10  # Use default
                                importances = spa_selection(
                                    X, y,
                                    n_features=n_to_select,
                                    n_random_starts=spa_n_random_starts,
                                    cv_folds=folds,
                                    random_state=random_state
                                )
                            elif varsel_method == 'uve':
                                # UVE: Uninformative Variable Elimination
                                folds = filtered_kwargs.get('folds', 5)
                                random_state = 42  # Use fixed random state
                                uve_cutoff_multiplier = 1.0  # Use default
                                uve_n_components = 5  # Use default
                                importances = uve_selection(
                                    X, y,
                                    cutoff_multiplier=uve_cutoff_multiplier,
                                    n_components=uve_n_components,
                                    cv_folds=folds,
                                    random_state=random_state
                                )
                            elif varsel_method == 'uve_spa':
                                # UVE-SPA: Hybrid method
                                n_to_select = min(max(valid_variable_counts), n_features_available)
                                folds = filtered_kwargs.get('folds', 5)
                                random_state = 42  # Use fixed random state
                                uve_cutoff_multiplier = 1.0  # Use default
                                uve_n_components = 5  # Use default
                                spa_n_random_starts = 10  # Use default
                                importances = uve_spa_selection(
                                    X, y,
                                    n_features=n_to_select,
                                    cutoff_multiplier=uve_cutoff_multiplier,
                                    uve_n_components=uve_n_components,
                                    uve_cv_folds=folds,
                                    spa_n_random_starts=spa_n_random_starts,
                                    spa_cv_folds=folds,
                                    random_state=random_state
                                )
                            elif varsel_method == 'ipls':
                                # iPLS: Interval PLS
                                folds = filtered_kwargs.get('folds', 5)
                                random_state = 42  # Use fixed random state
                                ipls_n_intervals = 20  # Use default
                                uve_n_components = 5  # Use default
                                importances = ipls_selection(
                                    X, y,
                                    n_intervals=ipls_n_intervals,
                                    n_components=uve_n_components,
                                    cv_folds=folds,
                                    random_state=random_state
                                )
                            elif varsel_method in ('cars', 'cars-aware'):
                                # CARS: Competitive Adaptive Reweighted Sampling
                                # cars-aware: Use model-appropriate fitness (LightGBM for tree models)
                                folds = filtered_kwargs.get('folds', 5)
                                random_state = 42  # Use fixed random state
                                uve_n_components = 5  # Use default
                                model_type_for_cars = model_name if varsel_method == 'cars-aware' else None
                                if model_type_for_cars:
                                    print(f"    -> Running Model-Aware CARS for {model_name}")
                                importances = cars_selection(
                                    X, y,
                                    n_iterations=50,
                                    pls_components=uve_n_components,
                                    cv_folds=folds,
                                    monte_carlo_samples=80,
                                    random_state=random_state,
                                    model_type=model_type_for_cars
                                )
                            elif varsel_method == 'vcpa-iriv':
                                # VCPA-IRIV: Variable Combination Population Analysis
                                folds = filtered_kwargs.get('folds', 5)
                                random_state = 42  # Use fixed random state
                                uve_n_components = 5  # Use default
                                print(f"    -> Running VCPA-IRIV (n_outer=10, n_inner=50)")
                                result = vcpa_iriv(
                                    X, y,
                                    n_outer_iterations=10,
                                    n_inner_iterations=50,
                                    pls_components=uve_n_components,
                                    cv_folds=folds,
                                    random_state=random_state
                                )
                                # Extract importance scores from result dict
                                importances = result.get('importance_scores', result.get('importances', None))

                                # VCPA returns importance_scores for ACTIVE indices only
                                # We need to create full-length importance array using selected_indices
                                selected = result.get('selected_indices', [])
                                if importances is not None and len(importances) == len(selected):
                                    # Map importance scores back to full wavelength array
                                    full_importances = np.zeros(X.shape[1])
                                    for idx, imp in zip(selected, importances):
                                        if idx < len(full_importances):
                                            full_importances[idx] = imp
                                    importances = full_importances
                                elif importances is None:
                                    logging.warning("VCPA-IRIV returned no importance scores, skipping")
                                    continue
                            else:
                                # Warn about unsupported methods
                                logging.warning(f"Variable selection method '{varsel_method}' not supported in Bayesian optimization - skipping")
                                continue

                            # Test each variable count
                            for n_top in valid_variable_counts:
                                # Select top N features based on importances
                                top_indices = np.argsort(importances, kind='stable')[-n_top:][::-1]

                                # Build new model with same hyperparameters
                                subset_model = build_model(model_name, params, task_type=task_type)

                                # Test subset
                                subset_result = run_single_config_fn(
                                    X, y, wavelengths,
                                    subset_model, model_name, params,
                                    preprocess_cfg, cv_splitter, task_type,
                                    is_binary_classification,
                                    skip_preprocessing=True,  # Already preprocessed
                                    subset_indices=top_indices,
                                    subset_tag=f"top{n_top}_{varsel_method}",
                                    **filtered_kwargs
                                )
                                trial_results.append(subset_result)

                                # Update best metric if this subset is better
                                if task_type == 'regression':
                                    if subset_result['RMSE'] < best_metric:
                                        best_metric = subset_result['RMSE']
                                        best_r2 = subset_result.get('R2', np.nan)
                                else:
                                    subset_acc = -subset_result['Accuracy']  # Negative for minimization
                                    if subset_acc < best_metric:  # Lower is better (more negative = higher accuracy)
                                        best_metric = subset_acc
                                        best_auc = subset_result.get('ROC_AUC', np.nan)

                        except Exception as e:
                            # If variable selection method fails, skip it
                            logging.warning(f"Variable selection method '{varsel_method}' failed: {type(e).__name__}: {e}")
                            continue

            # === STEP 3: Test regional subsets (if enabled) ===
            if enable_region_subsets and len(region_subsets) > 0:
                for region in region_subsets:
                    region_indices = region['indices']
                    region_tag = region['tag']

                    # Build new model with same hyperparameters
                    region_model = build_model(model_name, params, task_type=task_type)

                    # Test region
                    region_result = run_single_config_fn(
                        X, y, wavelengths,
                        region_model, model_name, params,
                        preprocess_cfg, cv_splitter, task_type,
                        is_binary_classification,
                        skip_preprocessing=True,
                        subset_indices=region_indices,
                        subset_tag=region_tag,
                        **filtered_kwargs
                    )
                    trial_results.append(region_result)

                    # Update best metric if this region is better
                    if task_type == 'regression':
                        if region_result['RMSE'] < best_metric:
                            best_metric = region_result['RMSE']
                            best_r2 = region_result.get('R2', np.nan)
                    else:
                        region_acc = -region_result['Accuracy']
                        if region_acc < best_metric:
                            best_metric = region_acc
                            best_auc = region_result.get('ROC_AUC', np.nan)

            # Store best R²/AUC as user attribute for reporting
            if task_type == 'regression':
                trial.set_user_attr('R2', best_r2)
                trial.set_user_attr('n_configs_tested', len(trial_results))
            else:
                trial.set_user_attr('ROC_AUC', best_auc)
                trial.set_user_attr('n_configs_tested', len(trial_results))

            # CRITICAL: Store ALL trial results (full + subsets) for later retrieval
            # This ensures subset analysis results are not thrown away
            trial.set_user_attr('all_results', trial_results)

            return best_metric

        except Exception as e:
            # If model training fails, return large penalty value
            # This marks the trial as completed but with worst score
            logging.warning(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
            # Return very large penalty (for minimization) or very negative (for maximization)
            if task_type == 'regression':
                return 1e10  # Large RMSE penalty
            else:
                return 1e10  # Large penalty (negative accuracy is being minimized)

    return objective


def convert_optuna_result_to_dasp_format(
    study: optuna.Study,
    model_name: str,
    preprocess_cfg: Dict,
    task_type: str,
    wavelengths: np.ndarray = None,
    n_vars: int = None,
    excluded_count: int = 0,
    validation_count: int = 0,
    total_samples_original: int = None,
    folds: int = 5,
    imbalance_method: str = None
):
    """
    Convert Optuna study results to DASP result format.

    CRITICAL CHANGE: Now returns ALL configurations tested (full + all subsets) from ALL trials.
    Previously only returned the best trial's full model result.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    model_name : str
        Model name
    preprocess_cfg : dict
        Preprocessing configuration
    task_type : str
        'regression' or 'classification'
    wavelengths : np.ndarray, optional
        Wavelength values
    n_vars : int, optional
        Number of variables (features)
    excluded_count : int
        Number of excluded samples
    validation_count : int
        Number of validation samples
    total_samples_original : int, optional
        Original total sample count
    folds : int
        Number of CV folds
    imbalance_method : str, optional
        Imbalance handling method

    Returns
    -------
    results : List[Dict]
        List of result dictionaries in DASP format (compatible with results DataFrame)
        Each trial contributes multiple results (full model + subsets)
    """
    from typing import List

    # Format imbalance display
    if imbalance_method is None:
        imbalance_display = "—"
    elif imbalance_method == 'class_weight':
        imbalance_display = "class_weight"
    else:
        imbalance_display = imbalance_method

    # Collect all results from all trials
    all_results = []

    # Calculate total optimization time
    optimization_time = sum(t.duration.total_seconds() for t in study.trials if t.duration)

    # Loop through all completed trials
    for trial in study.trials:
        # Skip failed/pruned trials
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Get trial parameters
        trial_params = trial.params
        lvs = trial_params.get('n_components', np.nan)

        # Get all results stored for this trial (full + subsets)
        trial_results = trial.user_attrs.get('all_results', [])

        # If no results stored (shouldn't happen), skip this trial
        if not trial_results:
            logging.warning(f"Trial {trial.number} has no stored results, skipping")
            continue

        # Convert each configuration result to DASP format
        for config_result in trial_results:
            # Extract subset information
            subset_tag = config_result.get('SubsetTag', 'full')
            config_n_vars = config_result.get('n_vars', n_vars)

            # Build base result dictionary
            result = {
                'Task': task_type,
                'Model': model_name,
                'Params': str(trial_params),
                'Preprocess': preprocess_cfg.get('name', 'unknown'),
                'Deriv': preprocess_cfg.get('deriv', 0),
                'Window': preprocess_cfg.get('window', 0),
                'Poly': preprocess_cfg.get('polyorder', 0),
                'LVs': lvs,
                'n_vars': config_n_vars,
                'full_vars': n_vars if n_vars is not None else len(wavelengths) if wavelengths is not None else np.nan,
                'SubsetTag': subset_tag,
                'Imbalance': imbalance_display,
                'all_vars': config_result.get('all_vars', 'N/A'),
                'top_vars': config_result.get('top_vars', 'N/A'),
                'n_trials': len(study.trials),
                'trial_number': trial.number,
                'optimization_time': optimization_time
            }

            # Add training configuration
            result['training_config'] = {
                'folds': folds,
                'n_samples_used': total_samples_original - excluded_count - validation_count if total_samples_original else np.nan,
                'n_samples_total': total_samples_original if total_samples_original else np.nan,
                'excluded_count': excluded_count,
                'validation_count': validation_count,
                'n_features_used': config_n_vars,
                'random_state': 42,
            }

            # Add task-specific metrics
            if task_type == 'regression':
                result['RMSE'] = config_result.get('RMSE', np.nan)
                result['R2'] = config_result.get('R2', np.nan)
                # Bayesian optimization doesn't compute regional RMSE (would need all predictions)
                result['regional_rmse'] = config_result.get('regional_rmse', None)
                result['y_quartiles'] = config_result.get('y_quartiles', None)
            else:
                result['Accuracy'] = config_result.get('Accuracy', np.nan)
                result['ROC_AUC'] = config_result.get('ROC_AUC', np.nan)

            all_results.append(result)

    return all_results


def print_optimization_summary(study: optuna.Study, model_name: str):
    """
    Print summary of Bayesian optimization results.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    model_name : str
        Model name
    """
    print(f"\n{'='*70}")
    print(f"Bayesian Optimization Summary: {model_name}")
    print(f"{'='*70}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Print trial statistics
    values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if values:
        print(f"\nTrial statistics:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std:  {np.std(values):.4f}")
        print(f"  Min:  {np.min(values):.4f}")
        print(f"  Max:  {np.max(values):.4f}")

    print(f"{'='*70}\n")


def get_param_importance(study: optuna.Study, top_n: int = 5) -> Dict[str, float]:
    """
    Calculate hyperparameter importance using fANOVA.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    top_n : int
        Number of top parameters to return

    Returns
    -------
    importance : dict
        Dictionary mapping parameter names to importance scores
    """
    try:
        from optuna.importance import get_param_importances

        importance = get_param_importances(study)

        # Get top N
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:top_n])

    except Exception:
        # If importance calculation fails, return empty dict
        return {}


def save_optimization_plots(study: optuna.Study, output_dir: str, model_name: str):
    """
    Save Optuna visualization plots.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    output_dir : str
        Directory to save plots
    model_name : str
        Model name (for filename)

    Notes
    -----
    Requires plotly to be installed. Silently skips if not available.
    """
    try:
        import optuna.visualization as vis
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_history.html'))

        # Parameter importances
        if len(study.trials) > 10:  # Need enough trials
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(output_dir, f'{model_name}_importance.html'))

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_parallel.html'))

    except ImportError:
        # Plotly not installed - silently skip
        pass
    except Exception as e:
        logging.warning(f"Could not save plots: {e}")


class ProgressCallback:
    """
    Callback for progress reporting during Bayesian optimization.

    Parameters
    ----------
    progress_callback : callable, optional
        Function to call with progress updates (for GUI)
    model_name : str
        Model name for logging
    preprocess_name : str
        Preprocessing method name for logging
    n_trials : int
        Total number of trials for this model
    task_type : str
        'regression' or 'classification'
    global_offset : int
        Offset for global progress tracking (trials completed before this model)
    global_total : int
        Total number of trials across all models
    """

    def __init__(self, progress_callback: Optional[Callable] = None, model_name: str = '',
                 preprocess_name: str = '', n_trials: int = 30, task_type: str = 'regression',
                 global_offset: int = 0, global_total: int = 0):
        self.progress_callback = progress_callback
        self.model_name = model_name
        self.preprocess_name = preprocess_name
        self.n_trials = n_trials
        self.task_type = task_type
        self.global_offset = global_offset
        self.global_total = global_total if global_total > 0 else n_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Called after each trial completes."""
        if self.progress_callback:
            # Calculate global progress
            global_current = self.global_offset + trial.number + 1

            # Format score based on task type
            if trial.value is not None:
                if self.task_type == 'regression':
                    score_str = f"RMSE: {trial.value:.4f}"
                else:
                    # Classification: value is negative accuracy (minimization)
                    score_str = f"Acc: {-trial.value:.4f}"
            else:
                score_str = "N/A"

            # Build best_model dict for GUI display
            if study.best_trial is not None and study.best_value is not None:
                if self.task_type == 'regression':
                    best_model = {
                        'Model': self.model_name,
                        'Preprocess': self.preprocess_name,
                        'RMSE': study.best_value,
                        'R2': study.best_trial.user_attrs.get('R2', np.nan)
                    }
                else:
                    best_model = {
                        'Model': self.model_name,
                        'Preprocess': self.preprocess_name,
                        'Accuracy': -study.best_value,  # Negate back to positive
                        'ROC_AUC': study.best_trial.user_attrs.get('ROC_AUC', np.nan)
                    }
            else:
                best_model = None

            self.progress_callback({
                'stage': 'bayesian_optimization',
                'message': f'{self.model_name} ({self.preprocess_name}): Trial {trial.number + 1}/{self.n_trials} - {score_str}',
                'current': global_current,
                'total': self.global_total,
                'best_model': best_model
            })


def handle_failed_trial(trial: optuna.Trial, exception: Exception) -> float:
    """
    Handle failed trials gracefully.

    Parameters
    ----------
    trial : optuna.Trial
        Failed trial
    exception : Exception
        Exception that caused failure

    Returns
    -------
    penalty : float
        Large penalty value to mark trial as failed

    Notes
    -----
    Common failure causes:
    - Model doesn't converge
    - Invalid hyperparameter combination
    - Numerical instability
    """
    logging.warning(f"Trial {trial.number} failed: {type(exception).__name__}: {exception}")

    # Return very large value (worst possible score)
    return 1e10  # Will be marked as worst trial


if __name__ == '__main__':
    # Example usage
    print("Bayesian Optimization Utilities - Example")
    print("=" * 70)

    # Create study
    study = create_optuna_study(
        direction='minimize',
        sampler='TPE',
        pruner='Median',
        random_state=42
    )

    print(f"✓ Created study: {study.study_name}")
    print(f"  Sampler: {type(study.sampler).__name__}")
    print(f"  Pruner: {type(study.pruner).__name__}")
    print(f"  Direction: {study.direction}")

    # Example objective function
    def simple_objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return x**2 + y**2

    # Optimize
    study.optimize(simple_objective, n_trials=20, show_progress_bar=False)

    # Print summary
    print_optimization_summary(study, "Simple Quadratic")

    # Parameter importance
    importance = get_param_importance(study)
    if importance:
        print("Parameter Importance:")
        for param, score in importance.items():
            print(f"  {param}: {score:.3f}")
