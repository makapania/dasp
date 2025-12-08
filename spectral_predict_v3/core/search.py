"""
Model search functionality for Spectral Predict v3 (standalone).

Full automation with preprocessing combinations, variable selection,
hyperparameter grids, and region analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Dict, Any, Tuple
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

from .model_config import (
    get_tier_models,
    get_hyperparameter_grid,
    get_preprocessing_config,
    get_variable_selection_config,
)
from .models import get_model, get_feature_importances
from .preprocess import SNV, SavgolDerivative, Absorbance
from .regions import create_region_subsets
# Import baseline correction classes
try:
    from .baseline import BaselinePolynomial, BaselineAsLS, BaselineAirPLS, SavgolSmooth
    HAS_BASELINE = True
except ImportError:
    HAS_BASELINE = False
from .variable_selection import ipls_forward as run_ipls_forward, ipls_backward as run_ipls_backward

# Import Bayesian search components
try:
    from .bayesian_search import SearchMode, run_bayesian_search, HAS_OPTUNA
except ImportError:
    # Bayesian search not available
    SearchMode = None
    run_bayesian_search = None
    HAS_OPTUNA = False

# Import NSGA-II multi-objective search
try:
    from .nsga2_search import run_nsga2_search, pareto_to_dataframe, PYMOO_AVAILABLE
except ImportError:
    run_nsga2_search = None
    pareto_to_dataframe = None
    PYMOO_AVAILABLE = False


def run_auto_search(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    task_type: str = 'regression',
    tier: str = 'standard',
    folds: int = 5,
    random_state: int = 42,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    # Search mode selection
    search_mode: str = 'grid',
    bayesian_n_trials: int = 50,
    # User-specified options (override tier defaults)
    preproc_methods: Optional[List[str]] = None,
    window_sizes: Optional[List[int]] = None,
    varsel_methods: Optional[List[str]] = None,
    varsel_params: Optional[Dict[str, Dict[str, Any]]] = None,
    var_counts: Optional[List[int]] = None,
    enable_regions: bool = False,
    n_regions: int = 5,
    # iPLS settings
    enable_ipls: bool = False,
    ipls_n_intervals: int = 20,
    ipls_forward: bool = True,
    ipls_backward: bool = True,
    pls_max_lv: Optional[int] = None,
    custom_models: Optional[List[str]] = None,
    # Custom hyperparameter grids (override defaults)
    custom_hyperparam_grids: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    # Wavelength range selection
    wl_range_min: Optional[float] = None,
    wl_range_max: Optional[float] = None,
    # Data conversion
    convert_absorbance: bool = False,
    # Baseline correction
    baseline_method: Optional[str] = None,
    baseline_params: Optional[Dict[str, Any]] = None,
    # Smoothing
    enable_smoothing: bool = False,
    smoothing_window: int = 11,
    smoothing_order: int = 2,
    # GA Preprocessing Optimization
    ga_preprocess: bool = False,
    ga_preprocess_population: int = 32,
    ga_preprocess_generations: int = 50,
    ga_preprocess_cv_folds: int = 5,
    # GA-PLS Wavelength Selection
    ga_pls: bool = False,
    ga_pls_population: int = 64,
    ga_pls_generations: int = 100,
    ga_pls_n_runs: int = 5,
    ga_pls_threshold: float = 0.6,
    # NSGA-II Multi-Objective Optimization
    nsga2_population: int = 50,
    nsga2_generations: int = 100,
    nsga2_min_wavelengths: int = 10,
    # Combined GA Workflow
    combined_ga_workflow: bool = False,
) -> pd.DataFrame:
    """
    Run automated model search with full preprocessing, variable selection,
    and hyperparameter tuning.

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray, optional
        Wavelength values for region analysis
    task_type : str
        'regression' or 'classification'
    tier : str
        'quick', 'standard', or 'comprehensive' (only affects which models are tested)
    folds : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility
    progress_callback : callable, optional
        Called with progress updates: {'stage', 'message', 'current', 'total', 'best_score'}
    search_mode : str, default='grid'
        Search mode: 'grid' for exhaustive grid search or 'bayesian' for Optuna-based
        Bayesian optimization. Bayesian search is faster for large hyperparameter spaces.
    bayesian_n_trials : int, default=50
        Number of trials for Bayesian optimization (only used if search_mode='bayesian')
    preproc_methods : list, optional
        Preprocessing methods to test (e.g., ['snv', 'deriv1', 'snv_deriv1'])
    window_sizes : list, optional
        SG window sizes for derivatives (e.g., [7, 19])
    varsel_methods : list, optional
        Variable selection methods (e.g., ['importance', 'spa'])
    var_counts : list, optional
        Variable counts to test (e.g., [20, 50, 100, 250])
    enable_regions : bool
        Whether to enable region-based analysis
    n_regions : int
        Number of top regions for region analysis
    enable_ipls : bool
        Whether to enable iPLS interval analysis
    ipls_n_intervals : int
        Number of intervals for iPLS (10-40, default 20)
    ipls_forward : bool
        Enable forward iPLS (iteratively add best intervals)
    ipls_backward : bool
        Enable backward iPLS (iteratively remove worst intervals)

    Returns
    -------
    pd.DataFrame
        Ranked results with columns: Model, Preprocessing, Variables, RMSE/Accuracy, R2/AUC, etc.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Apply absorbance conversion if requested (before any preprocessing)
    if convert_absorbance:
        if progress_callback:
            progress_callback({'message': 'Converting to absorbance (log10(1/R))...', 'stage': 'conversion'})
        X = Absorbance().fit_transform(X)

    # Apply baseline correction if requested (after absorbance, before other preprocessing)
    if baseline_method and HAS_BASELINE:
        if progress_callback:
            progress_callback({'message': f'Applying baseline correction ({baseline_method})...', 'stage': 'baseline'})
        params = baseline_params or {}
        if baseline_method.lower() == 'polynomial':
            baseline_corrector = BaselinePolynomial(**params)
        elif baseline_method.lower() == 'asls':
            baseline_corrector = BaselineAsLS(**params)
        elif baseline_method.lower() == 'airpls':
            baseline_corrector = BaselineAirPLS(**params)
        else:
            baseline_corrector = None
        if baseline_corrector is not None:
            X = baseline_corrector.fit_transform(X)

    # Apply smoothing if requested (after baseline, before other preprocessing)
    if enable_smoothing and HAS_BASELINE:
        if progress_callback:
            progress_callback({'message': 'Applying Savitzky-Golay smoothing...', 'stage': 'smoothing'})
        smoother = SavgolSmooth(window_length=smoothing_window, polyorder=smoothing_order)
        X = smoother.fit_transform(X)

    # Get models from tier or use custom list
    if custom_models:
        models_to_test = custom_models
    else:
        models_to_test = get_tier_models(tier, task_type)

    # Capture whether user explicitly requested variable selection for Bayesian mode
    # (before defaults are applied for grid search)
    user_requested_varsel = varsel_methods is not None and len(varsel_methods) > 0

    # Use user-specified options or defaults
    if preproc_methods is None:
        preproc_methods = ['snv', 'deriv1', 'deriv2', 'snv_deriv1']
        print(f"[Search] Using default preprocessing: {preproc_methods}")
    else:
        print(f"[Search] Using user-specified preprocessing: {preproc_methods}")
    if window_sizes is None:
        window_sizes = [7, 19]
        print(f"[Search] Using default window sizes: {window_sizes}")
    else:
        print(f"[Search] Using user-specified window sizes: {window_sizes}")
    if varsel_methods is None:
        varsel_methods = ['importance', 'spa']  # Default for grid search
    if var_counts is None:
        var_counts = [20, 50, 100, 250]

    # Build preprocessing config from user selections
    preproc_config = {
        'methods': preproc_methods,
        'window_sizes': window_sizes,
    }

    # Build variable selection config from user selections
    # Note: 'ipls' is handled separately via enable_ipls, not in varsel_methods
    filtered_varsel_methods = [m for m in (varsel_methods or []) if m != 'ipls']
    varsel_config = {
        'methods': filtered_varsel_methods,
        'counts': var_counts,
        'enable_regions': enable_regions,
        'n_top_regions': n_regions,
        'params': varsel_params or {},  # Method-specific parameters
    }

    # GA-PLS convenience integration - add to varsel_methods if ga_pls=True
    if ga_pls:
        if 'ga_pls' not in varsel_config['methods']:
            varsel_config['methods'].append('ga_pls')
        # Set GA-PLS parameters
        varsel_config['params']['ga_pls'] = {
            'population_size': ga_pls_population,
            'n_generations': ga_pls_generations,
            'n_runs': ga_pls_n_runs,
            'selection_threshold': ga_pls_threshold,
        }
        print(f"[GA-PLS] Enabled with pop={ga_pls_population}, gen={ga_pls_generations}, "
              f"runs={ga_pls_n_runs}, threshold={ga_pls_threshold}")

    # iPLS config (handled as Phase 4, separate from generic variable selection)
    ipls_config = {
        'enabled': enable_ipls and wavelengths is not None and task_type == 'regression',
        'n_intervals': ipls_n_intervals,
        'forward': ipls_forward,
        'backward': ipls_backward,
    }

    # Build preprocessing configurations
    preproc_configs = _build_preprocessing_configs(preproc_config)

    # GA PREPROCESSING OPTIMIZATION - Find optimal preprocessing before search
    if ga_preprocess:
        print(f"[GA Preprocessing] Starting optimization (pop={ga_preprocess_population}, gen={ga_preprocess_generations})")
        print(f"[GA Preprocessing] Exploring 10 preproc types, 17 window sizes, 4 baseline methods, 9 smoothing options")
        try:
            from .ga_preprocessing import optimize_preprocessing
            GA_PREPROC_AVAILABLE = True
        except ImportError:
            GA_PREPROC_AVAILABLE = False

        if GA_PREPROC_AVAILABLE:
            if progress_callback:
                progress_callback({
                    'stage': 'ga_preprocessing',
                    'message': 'Optimizing preprocessing with GA...',
                    'current': 0,
                    'total': ga_preprocess_generations
                })

            # Run GA preprocessing optimization
            ga_result = optimize_preprocessing(
                X, y,
                population_size=ga_preprocess_population,
                n_generations=ga_preprocess_generations,
                cv_folds=ga_preprocess_cv_folds,
                random_state=random_state,
                verbose=1
            )

            # Use only the optimized preprocessing config
            preproc_configs = [(ga_result['best_name'], ga_result['best_transform'])]

            if progress_callback:
                progress_callback({
                    'stage': 'ga_preprocessing',
                    'message': f"GA Preprocessing: {ga_result['best_config']} (RMSECV={ga_result['best_rmsecv']:.4f})",
                    'current': ga_preprocess_generations,
                    'total': ga_preprocess_generations
                })

            print(f"[GA Preprocessing] Using optimized config: {ga_result['best_name']}")
        else:
            print("[GA Preprocessing] Module not available, using standard preprocessing")

    # COMBINED GA WORKFLOW - Run GA-PLS on preprocessed data once
    # This is more efficient than running GA-PLS during each model configuration
    ga_selected_mask = None  # Will hold wavelength selection mask
    if combined_ga_workflow:
        try:
            from .ga_pls import ga_pls_selection
            GA_PLS_AVAILABLE = True
        except ImportError:
            GA_PLS_AVAILABLE = False

        if GA_PLS_AVAILABLE:
            # Apply best preprocessing (if GA preprocessing was used)
            if preproc_configs and preproc_configs[0][1] is not None:
                X_preproc = preproc_configs[0][1](X)
                print(f"[Combined GA] Applied preprocessing: {preproc_configs[0][0]}")
            else:
                X_preproc = X

            if progress_callback:
                progress_callback({
                    'stage': 'ga_pls_combined',
                    'message': 'Running GA-PLS wavelength selection...',
                    'current': 0,
                    'total': ga_pls_generations
                })

            # Run GA-PLS
            importances = ga_pls_selection(
                X_preproc, y,
                population_size=ga_pls_population,
                n_generations=ga_pls_generations,
                n_runs=ga_pls_n_runs,
                task_type=task_type,
                cv_folds=folds,
                random_state=random_state,
                verbose=1
            )

            # Create selection mask
            ga_selected_mask = importances >= ga_pls_threshold
            n_selected = np.sum(ga_selected_mask)

            if progress_callback:
                progress_callback({
                    'stage': 'ga_pls_combined',
                    'message': f'GA-PLS: {n_selected} wavelengths selected (threshold={ga_pls_threshold})',
                    'current': ga_pls_generations,
                    'total': ga_pls_generations
                })

            print(f"[Combined GA] GA-PLS selected {n_selected}/{X.shape[1]} wavelengths")

            # Reduce X to selected wavelengths for all subsequent processing
            if n_selected > 0 and n_selected < X.shape[1]:
                X = X[:, ga_selected_mask]
                if wavelengths is not None:
                    wavelengths = wavelengths[ga_selected_mask]
                print(f"[Combined GA] Data reduced to {X.shape[1]} wavelengths")

            # Disable per-config GA-PLS since we already did it
            if 'ga_pls' in varsel_config['methods']:
                varsel_config['methods'].remove('ga_pls')
                print("[Combined GA] Disabled per-config GA-PLS (already applied)")
        else:
            print("[Combined GA] GA-PLS not available, skipping wavelength selection")

    # BAYESIAN SEARCH MODE - Route to Bayesian optimization if requested
    # Bayesian mode is TRUE AUTOML: Optuna explores all preprocessing and variable selection
    # automatically. User only controls which models to consider.
    if search_mode.lower() == 'bayesian':
        if not HAS_OPTUNA:
            raise ImportError(
                "Bayesian search requires Optuna. Install with: pip install optuna\n"
                "Falling back to grid search mode."
            )

        # === AUTOML MODE: Override preprocessing with comprehensive search space ===
        # Bayesian mode explores ALL preprocessing options automatically
        bayesian_preproc_methods = ['raw', 'snv', 'deriv1', 'deriv2', 'snv_deriv1', 'snv_deriv2']
        bayesian_window_sizes = [7, 11, 17, 21]

        # Include baseline correction methods if available
        if HAS_BASELINE:
            bayesian_preproc_methods += ['baseline_poly', 'baseline_asls']

        print(f"[Bayesian] Using comprehensive preprocessing: {bayesian_preproc_methods}")
        print(f"[Bayesian] Window sizes: {bayesian_window_sizes}")

        # Build comprehensive preprocessing config for Bayesian search
        bayesian_preproc_config = {
            'methods': bayesian_preproc_methods,
            'window_sizes': bayesian_window_sizes,
        }
        bayesian_preproc_configs = _build_preprocessing_configs(bayesian_preproc_config)
        print(f"[Bayesian] Built {len(bayesian_preproc_configs)} preprocessing configurations")

        # Always include variable selection in Bayesian AutoML mode
        include_varsel = True
        bayesian_varsel_methods = ['importance', 'vip', 'spa']
        bayesian_var_counts = [20, 50, 100, 250]

        if progress_callback:
            progress_callback({
                'stage': 'bayesian_search',
                'message': 'Starting Bayesian AutoML (exploring all preprocessing + variable selection)...',
                'current': 0,
                'total': bayesian_n_trials
            })

        # Run Bayesian search with comprehensive search space
        best_config, study = run_bayesian_search(
            X=X,
            y=y,
            task_type=task_type,
            models_to_try=models_to_test,  # User controls which models
            preprocess_configs=bayesian_preproc_configs,  # AutoML: all preprocessing
            n_trials=bayesian_n_trials,
            cv_folds=folds,
            random_state=random_state,
            progress_callback=progress_callback,
            # AutoML: always include variable selection
            include_variable_selection=include_varsel,
            varsel_methods=bayesian_varsel_methods,
            varsel_counts=bayesian_var_counts,
            varsel_params=varsel_params
        )

        # Extract variable selection info if present
        varsel_info = best_config.get('variable_selection', {})
        varsel_method = varsel_info.get('method', 'full')
        varsel_n_vars = varsel_info.get('n_vars')

        # Build variable tag for DataFrame
        if varsel_method != 'full' and varsel_n_vars is not None:
            var_tag = f'{varsel_method}_top{varsel_n_vars}'
            n_vars = varsel_n_vars
        else:
            var_tag = 'full'
            n_vars = X.shape[1]

        # Convert to DataFrame format matching grid search
        if task_type == 'regression':
            df = pd.DataFrame([{
                'Model': best_config['model'],
                'Preprocessing': best_config['preprocessing'],
                'Variables': var_tag,
                'N_Variables': n_vars,
                'Params': _format_params(best_config['params']),
                'RMSE': best_config['score'],
                'R2': None,  # Would need to recompute
                'Top_Variables': '',
            }])
        else:
            df = pd.DataFrame([{
                'Model': best_config['model'],
                'Preprocessing': best_config['preprocessing'],
                'Variables': var_tag,
                'N_Variables': n_vars,
                'Params': _format_params(best_config['params']),
                'Accuracy': best_config['score'],
                'ROC_AUC': None,  # Would need to recompute
                'Top_Variables': '',
            }])

        return df

    # NSGA-II MULTI-OBJECTIVE SEARCH MODE
    # Pareto optimization of prediction error vs wavelength count vs model complexity
    if search_mode.lower() == 'nsga2':
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "NSGA-II search requires pymoo. Install with: pip install pymoo>=0.6.0\n"
                "Falling back to grid search mode."
            )

        print(f"[NSGA-II] Starting multi-objective optimization (pop={nsga2_population}, gen={nsga2_generations})")
        print(f"[NSGA-II] Using internal comprehensive preprocessing (10 types, 15 window sizes)")

        if progress_callback:
            progress_callback({
                'stage': 'nsga2_search',
                'message': 'Starting NSGA-II multi-objective optimization...',
                'current': 0,
                'total': nsga2_generations
            })

        # Run NSGA-II search with user-selected models
        nsga2_result = run_nsga2_search(
            X=X,
            y=y,
            task_type=task_type,
            population_size=nsga2_population,
            n_generations=nsga2_generations,
            cv_folds=folds,
            min_wavelengths=nsga2_min_wavelengths,
            random_state=random_state,
            verbose=1,
            progress_callback=progress_callback,
            models=models_to_test,  # Pass user-selected models
        )

        # Convert Pareto front to DataFrame
        df = pareto_to_dataframe(nsga2_result, X.shape[1], task_type)

        # Store full NSGA-II result as attribute for GUI access
        if len(df) > 0:
            df.attrs['nsga2_result'] = nsga2_result

        return df

    # GRID SEARCH MODE (default) - Continue with existing logic
    # Handle categorical labels for classification
    label_encoder = None
    if task_type == 'classification':
        if y.dtype == object or not np.issubdtype(y.dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

    # Set up cross-validation
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    results = []
    best_score = None
    config_num = 0

    # Calculate actual total configurations
    n_preproc = len(preproc_configs)

    # Phase 1 total: preprocessing × models × hyperparameters
    phase1_total = 0
    for model_name in models_to_test:
        if model_name in ['PLS', 'PLS-DA'] and pls_max_lv is not None:
            n_hyperparams = len(get_hyperparameter_grid(model_name, max_lv=pls_max_lv))
        else:
            n_hyperparams = len(get_hyperparameter_grid(model_name))
        phase1_total += n_preproc * n_hyperparams

    # Phase 2 total: variable selection (if enabled)
    # Each method is run separately, so multiply by number of methods
    phase2_total = 0
    n_varsel_methods = len(varsel_config.get('methods', []))
    if n_varsel_methods > 0 and len(varsel_config['counts']) > 0:
        n_valid_counts = sum(1 for c in varsel_config['counts'] if c < X.shape[1])
        # Each: preprocessing × methods × var_count × models (only first hyperparams)
        phase2_total = n_preproc * n_varsel_methods * n_valid_counts * len(models_to_test)

    # Phase 3 total: region analysis (if enabled)
    phase3_total = 0
    if varsel_config.get('enable_regions', False) and wavelengths is not None:
        n_top_regions = varsel_config.get('n_top_regions', 5)
        # Estimate number of region subsets (individual regions + combinations)
        # For n_top_regions=5: ~3 individual + 2 combinations = 5 subsets
        # For n_top_regions=10: ~5 individual + 3 combinations = 8 subsets
        # For n_top_regions=20: ~10 individual + 4 combinations = 14 subsets
        if n_top_regions <= 5:
            n_subsets = 5
        elif n_top_regions <= 10:
            n_subsets = 8
        elif n_top_regions <= 15:
            n_subsets = 11
        else:
            n_subsets = 14
        # Each: preprocessing × subsets × models (only first hyperparams)
        phase3_total = n_preproc * n_subsets * len(models_to_test)

    # Phase 4 total: iPLS analysis (if enabled)
    phase4_total = 0
    if ipls_config['enabled']:
        n_intervals = ipls_config['n_intervals']
        # Estimate iPLS subsets:
        # Forward: n_intervals individual + ~5 combinations = ~n_intervals + 5
        # Backward: ~n_intervals/2 progressive removals + 1 final = ~n_intervals/2 + 1
        fwd_subsets = (n_intervals + 5) if ipls_config['forward'] else 0
        bwd_subsets = (n_intervals // 2 + 2) if ipls_config['backward'] else 0
        n_ipls_subsets = fwd_subsets + bwd_subsets
        # Each: preprocessing × subsets × models (only first hyperparams)
        phase4_total = n_preproc * n_ipls_subsets * len(models_to_test)

    total_configs = phase1_total + phase2_total + phase3_total + phase4_total

    # Prepare wavelength range filtering
    wl_mask = None
    wavelengths_filtered = wavelengths
    if wavelengths is not None and (wl_range_min is not None or wl_range_max is not None):
        wl_mask = np.ones(len(wavelengths), dtype=bool)
        if wl_range_min is not None:
            wl_mask &= (wavelengths >= wl_range_min)
        if wl_range_max is not None:
            wl_mask &= (wavelengths <= wl_range_max)
        wavelengths_filtered = wavelengths[wl_mask]

        n_filtered = wl_mask.sum()
        if progress_callback:
            progress_callback({
                'stage': 'info',
                'message': f'Wavelength range: {wl_range_min:.0f}-{wl_range_max:.0f} nm ({n_filtered} of {len(wavelengths)} wavelengths)',
                'current': 0,
                'total': total_configs,
                'best_score': None
            })

        # Check for valid variable selection methods with reduced wavelengths
        if n_filtered < 50:
            # Filter out methods that need many wavelengths
            methods_needing_many_wl = {'uve', 'cars', 'uve-spa'}
            original_methods = varsel_config.get('methods', [])
            varsel_config['methods'] = [m for m in original_methods if m.lower() not in methods_needing_many_wl]
            if len(varsel_config['methods']) < len(original_methods):
                if progress_callback:
                    progress_callback({
                        'stage': 'info',
                        'message': f'Note: Some variable selection methods disabled due to narrow wavelength range',
                        'current': 0,
                        'total': total_configs,
                        'best_score': None
                    })

    # Phase 1: Test all preprocessing + model combinations
    for preproc_name, preproc_func in preproc_configs:
        # Apply preprocessing to FULL spectrum
        try:
            X_processed = preproc_func(X) if preproc_func else X
        except Exception as e:
            print(f"Preprocessing {preproc_name} failed: {e}")
            continue

        # Apply wavelength range filtering AFTER preprocessing
        if wl_mask is not None:
            X_processed = X_processed[:, wl_mask]

        # Test each model
        for model_name in models_to_test:
            # Get hyperparameter grid - check for custom grid first
            if custom_hyperparam_grids and model_name in custom_hyperparam_grids:
                # Use user-provided custom hyperparameter grid
                hyperparam_grid = custom_hyperparam_grids[model_name]
            elif model_name in ['PLS', 'PLS-DA'] and pls_max_lv is not None:
                hyperparam_grid = get_hyperparameter_grid(model_name, max_lv=pls_max_lv)
            else:
                hyperparam_grid = get_hyperparameter_grid(model_name)

            for params in hyperparam_grid:
                config_num += 1

                if progress_callback:
                    progress_callback({
                        'stage': 'model_testing',
                        'message': f'{preproc_name} + {model_name} ({_format_params(params)})',
                        'current': config_num,
                        'total': total_configs,
                        'best_score': best_score
                    })

                result = _test_single_config(
                    X_processed, y, model_name, task_type,
                    preproc_name, 'full', params, cv, random_state,
                    wavelengths=wavelengths_filtered
                )

                if result:
                    results.append(result)
                    # Track best score
                    if task_type == 'regression':
                        if best_score is None or result['RMSE'] < best_score:
                            best_score = result['RMSE']
                    else:
                        if best_score is None or result['Accuracy'] > best_score:
                            best_score = result['Accuracy']

    # Phase 2: Variable selection (if enabled for this tier)
    if varsel_config['methods'] and len(varsel_config['counts']) > 0:
        varsel_results = _run_variable_selection(
            X, y, wavelengths_filtered, task_type, tier,
            preproc_configs, models_to_test,
            varsel_config, cv, random_state,
            progress_callback, best_score,
            start_config_num=config_num, total_configs=total_configs,
            wl_mask=wl_mask
        )
        results.extend(varsel_results)
        config_num += len(varsel_results)

        # Update best score from variable selection results
        for result in varsel_results:
            if task_type == 'regression':
                if best_score is None or result['RMSE'] < best_score:
                    best_score = result['RMSE']
            else:
                if best_score is None or result['Accuracy'] > best_score:
                    best_score = result['Accuracy']

    # Phase 3: Region analysis (if enabled and wavelengths available)
    if varsel_config.get('enable_regions', False) and wavelengths_filtered is not None:
        region_results = _run_region_analysis(
            X, y, wavelengths_filtered, task_type, tier,
            preproc_configs, models_to_test,
            varsel_config, cv, random_state,
            progress_callback, best_score,
            start_config_num=config_num, total_configs=total_configs,
            wl_mask=wl_mask
        )
        results.extend(region_results)
        config_num += len(region_results)

        # Update best score from region results
        for result in region_results:
            if task_type == 'regression':
                if best_score is None or result['RMSE'] < best_score:
                    best_score = result['RMSE']
            else:
                if best_score is None or result['Accuracy'] > best_score:
                    best_score = result['Accuracy']

    # Phase 4: iPLS interval analysis (if enabled, regression only)
    if ipls_config['enabled']:
        ipls_results = _run_ipls_analysis(
            X, y, wavelengths_filtered, task_type,
            preproc_configs, models_to_test,
            ipls_config, cv, random_state,
            progress_callback, best_score,
            start_config_num=config_num, total_configs=total_configs,
            wl_mask=wl_mask
        )
        results.extend(ipls_results)

    # Create results DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        return df

    # Sort by primary metric
    if task_type == 'regression':
        df = df.sort_values('RMSE', ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    return df


def run_manual_search(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task_type: str = 'regression',
    preprocessing: str = 'raw',
    window_size: int = 7,
    model_params: Optional[Dict[str, Any]] = None,
    folds: int = 5,
    random_state: int = 42,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    wavelengths: Optional[np.ndarray] = None,
    wl_range_min: Optional[float] = None,
    wl_range_max: Optional[float] = None,
    convert_absorbance: bool = False,
    # Baseline correction
    baseline_method: Optional[str] = None,
    baseline_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Run manual model training with user-specified settings.

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_features)
    y : np.ndarray
        Target values
    model_name : str
        Model name (e.g., 'PLS', 'Ridge', 'RandomForest')
    task_type : str
        'regression' or 'classification'
    preprocessing : str
        Preprocessing method ('raw', 'snv', 'deriv1', 'deriv2', etc.)
    window_size : int
        Window size for derivative preprocessing
    model_params : dict, optional
        Model hyperparameters
    folds : int
        Number of CV folds
    random_state : int
        Random seed
    progress_callback : callable, optional
        Progress callback

    Returns
    -------
    pd.DataFrame
        Single-row results with model performance
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Apply absorbance conversion if requested (before any preprocessing)
    if convert_absorbance:
        if progress_callback:
            progress_callback({'message': 'Converting to absorbance (log10(1/R))...', 'stage': 'conversion'})
        X = Absorbance().fit_transform(X)

    # Apply baseline correction if requested (after absorbance, before other preprocessing)
    if baseline_method and HAS_BASELINE:
        if progress_callback:
            progress_callback({'message': f'Applying baseline correction ({baseline_method})...', 'stage': 'baseline'})
        params = baseline_params or {}
        if baseline_method.lower() == 'polynomial':
            baseline_corrector = BaselinePolynomial(**params)
        elif baseline_method.lower() == 'asls':
            baseline_corrector = BaselineAsLS(**params)
        elif baseline_method.lower() == 'airpls':
            baseline_corrector = BaselineAirPLS(**params)
        else:
            baseline_corrector = None
        if baseline_corrector is not None:
            X = baseline_corrector.fit_transform(X)

    if model_params is None:
        model_params = {}

    # Handle categorical labels for classification
    label_encoder = None
    if task_type == 'classification':
        if y.dtype == object or not np.issubdtype(y.dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

    # Set up cross-validation
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    # Apply preprocessing
    preproc_name = preprocessing
    if preprocessing == 'raw':
        X_processed = X
    elif preprocessing == 'snv':
        X_processed = SNV().fit_transform(X)
    elif preprocessing == 'deriv1':
        X_processed = SavgolDerivative(deriv=1, window=window_size).fit_transform(X)
        preproc_name = f'deriv1_w{window_size}'
    elif preprocessing == 'deriv2':
        X_processed = SavgolDerivative(deriv=2, window=window_size).fit_transform(X)
        preproc_name = f'deriv2_w{window_size}'
    elif preprocessing == 'snv_deriv1':
        X_snv = SNV().fit_transform(X)
        X_processed = SavgolDerivative(deriv=1, window=window_size).fit_transform(X_snv)
        preproc_name = f'snv_deriv1_w{window_size}'
    elif preprocessing == 'snv_deriv2':
        X_snv = SNV().fit_transform(X)
        X_processed = SavgolDerivative(deriv=2, window=window_size).fit_transform(X_snv)
        preproc_name = f'snv_deriv2_w{window_size}'
    else:
        X_processed = X

    # Apply wavelength range filtering AFTER preprocessing
    wavelengths_filtered = wavelengths
    if wavelengths is not None and (wl_range_min is not None or wl_range_max is not None):
        wl_mask = np.ones(len(wavelengths), dtype=bool)
        if wl_range_min is not None:
            wl_mask &= (wavelengths >= wl_range_min)
        if wl_range_max is not None:
            wl_mask &= (wavelengths <= wl_range_max)
        X_processed = X_processed[:, wl_mask]
        wavelengths_filtered = wavelengths[wl_mask]

        if progress_callback:
            progress_callback({
                'stage': 'info',
                'message': f'Wavelength range: {wl_range_min:.0f}-{wl_range_max:.0f} nm ({wl_mask.sum()} wavelengths)',
                'current': 0,
                'total': 1
            })

    if progress_callback:
        progress_callback({
            'stage': 'model_testing',
            'message': f'Training {model_name} with {preproc_name}',
            'current': 1,
            'total': 1
        })

    # Test the configuration
    result = _test_single_config(
        X_processed, y, model_name, task_type,
        preproc_name, 'full', model_params, cv, random_state,
        wavelengths=wavelengths_filtered
    )

    if result:
        return pd.DataFrame([result])
    else:
        return pd.DataFrame()


def _build_preprocessing_configs(preproc_config: Dict[str, Any]) -> List[Tuple[str, Callable]]:
    """
    Build list of preprocessing configurations from tier config.

    Returns list of (name, transform_function) tuples.
    """
    configs = []
    methods = preproc_config.get('methods', ['raw'])
    window_sizes = preproc_config.get('window_sizes', [7])

    for method in methods:
        if method == 'raw':
            configs.append(('raw', None))

        elif method == 'snv':
            configs.append(('snv', lambda X: SNV().fit_transform(X)))

        elif method == 'deriv1':
            for window in window_sizes:
                name = f'deriv1_w{window}'
                # Use default args to capture window value
                configs.append((name, lambda X, w=window: SavgolDerivative(deriv=1, window=w).fit_transform(X)))

        elif method == 'deriv2':
            for window in window_sizes:
                name = f'deriv2_w{window}'
                configs.append((name, lambda X, w=window: SavgolDerivative(deriv=2, window=w).fit_transform(X)))

        elif method == 'snv_deriv1':
            for window in window_sizes:
                name = f'snv_deriv1_w{window}'
                def transform(X, w=window):
                    X_snv = SNV().fit_transform(X)
                    return SavgolDerivative(deriv=1, window=w).fit_transform(X_snv)
                configs.append((name, transform))

        elif method == 'snv_deriv2':
            for window in window_sizes:
                name = f'snv_deriv2_w{window}'
                def transform(X, w=window):
                    X_snv = SNV().fit_transform(X)
                    return SavgolDerivative(deriv=2, window=w).fit_transform(X_snv)
                configs.append((name, transform))

        elif method == 'deriv1_snv':
            for window in window_sizes:
                name = f'deriv1_snv_w{window}'
                def transform(X, w=window):
                    X_deriv = SavgolDerivative(deriv=1, window=w).fit_transform(X)
                    return SNV().fit_transform(X_deriv)
                configs.append((name, transform))

        elif method == 'deriv2_snv':
            for window in window_sizes:
                name = f'deriv2_snv_w{window}'
                def transform(X, w=window):
                    X_deriv = SavgolDerivative(deriv=2, window=w).fit_transform(X)
                    return SNV().fit_transform(X_deriv)
                configs.append((name, transform))

        elif method == 'deriv3':
            for window in window_sizes:
                name = f'deriv3_w{window}'
                configs.append((name, lambda X, w=window: SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X)))

        elif method == 'deriv4':
            for window in window_sizes:
                name = f'deriv4_w{window}'
                configs.append((name, lambda X, w=window: SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X)))

        elif method == 'snv_deriv3':
            for window in window_sizes:
                name = f'snv_deriv3_w{window}'
                def transform(X, w=window):
                    X_snv = SNV().fit_transform(X)
                    return SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_snv)
                configs.append((name, transform))

        elif method == 'snv_deriv4':
            for window in window_sizes:
                name = f'snv_deriv4_w{window}'
                def transform(X, w=window):
                    X_snv = SNV().fit_transform(X)
                    return SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_snv)
                configs.append((name, transform))

        elif method == 'deriv3_snv':
            for window in window_sizes:
                name = f'deriv3_snv_w{window}'
                def transform(X, w=window):
                    X_deriv = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X)
                    return SNV().fit_transform(X_deriv)
                configs.append((name, transform))

        elif method == 'deriv4_snv':
            for window in window_sizes:
                name = f'deriv4_snv_w{window}'
                def transform(X, w=window):
                    X_deriv = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X)
                    return SNV().fit_transform(X_deriv)
                configs.append((name, transform))

    return configs


def _test_single_config(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task_type: str,
    preproc_name: str,
    variable_tag: str,
    params: Dict[str, Any],
    cv,
    random_state: int,
    wavelengths: Optional[np.ndarray] = None,
    variable_indices: Optional[np.ndarray] = None,
    n_top_vars: int = 50
) -> Optional[Dict[str, Any]]:
    """
    Test a single model configuration and return results.

    Parameters
    ----------
    wavelengths : array, optional
        Full wavelength array for mapping indices to wavelength values
    variable_indices : array, optional
        Indices of variables used (for variable selection runs)
    n_top_vars : int
        Number of top variables to report (default 50)
    """
    try:
        model = get_model(model_name, task_type, random_state, **params)
        if model is None:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(model, X, y, cv=cv)

        # Fit model to get feature importances for top_vars
        top_vars_str = ""
        try:
            model_fitted = get_model(model_name, task_type, random_state, **params)
            model_fitted.fit(X, y)
            importances = get_feature_importances(model_fitted, model_name)

            if importances is not None and len(importances) > 0:
                # Get top N variable indices (within the current X)
                n_top = min(n_top_vars, len(importances))
                top_idx_local = np.argsort(importances)[-n_top:][::-1]

                # Map to wavelengths if available
                if wavelengths is not None:
                    if variable_indices is not None:
                        # Variable selection was applied - map through indices
                        actual_indices = variable_indices[top_idx_local]
                        top_wavelengths = wavelengths[actual_indices]
                    else:
                        # Full spectrum - direct mapping
                        top_wavelengths = wavelengths[top_idx_local]
                    top_vars_str = ", ".join([f"{int(round(w))}" for w in top_wavelengths])
                else:
                    # No wavelengths - just show indices
                    if variable_indices is not None:
                        actual_indices = variable_indices[top_idx_local]
                        top_vars_str = ", ".join([str(i) for i in actual_indices])
                    else:
                        top_vars_str = ", ".join([str(i) for i in top_idx_local])
        except:
            pass

        if task_type == 'regression':
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            return {
                'Model': model_name,
                'Preprocessing': preproc_name,
                'Variables': variable_tag,
                'N_Variables': X.shape[1],
                'Params': _format_params(params),
                'RMSE': rmse,
                'R2': r2,
                'Top_Variables': top_vars_str,
                'variable_indices': variable_indices,  # Store for diagnostics/export
            }
        else:
            acc = accuracy_score(y, y_pred)
            try:
                if len(np.unique(y)) == 2:
                    auc = roc_auc_score(y, y_pred)
                else:
                    auc = None
            except:
                auc = None

            return {
                'Model': model_name,
                'Preprocessing': preproc_name,
                'Variables': variable_tag,
                'N_Variables': X.shape[1],
                'Params': _format_params(params),
                'Accuracy': acc,
                'ROC_AUC': auc,
                'Top_Variables': top_vars_str,
                'variable_indices': variable_indices,  # Store for diagnostics/export
            }

    except Exception as e:
        print(f"Error testing {model_name} with {preproc_name}: {e}")
        return None


def _run_variable_selection(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: Optional[np.ndarray],
    task_type: str,
    tier: str,
    preproc_configs: List[Tuple[str, Callable]],
    models_to_test: List[str],
    varsel_config: Dict[str, Any],
    cv,
    random_state: int,
    progress_callback: Optional[Callable],
    current_best: Optional[float],
    start_config_num: int = 0,
    total_configs: int = 0,
    wl_mask: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Run variable selection methods and test reduced models.

    Each method is run separately, and results are tagged with the method name
    so users can see which variable selection approach was used.
    """
    results = []
    best_score = current_best
    config_num = start_config_num

    methods = varsel_config.get('methods', [])
    counts = varsel_config.get('counts', [])
    varsel_method_params = varsel_config.get('params', {})

    # Filter counts to only those less than n_features
    n_features = X.shape[1]
    valid_counts = [c for c in counts if c < n_features]

    if not valid_counts or not methods:
        return results

    # Method name abbreviations for cleaner tags
    method_abbrevs = {
        'importance': 'RF',      # RandomForest importance
        'vip': 'VIP',            # PLS Variable Importance in Projection
        'spa': 'SPA',            # Successive Projections Algorithm
        'uve': 'UVE',            # Uninformative Variable Elimination
        'uve_spa': 'UVE-SPA',    # UVE + SPA hybrid
        'ipls': 'iPLS',          # Interval PLS
    }

    for preproc_name, preproc_func in preproc_configs:
        # Apply preprocessing to FULL spectrum
        try:
            X_processed = preproc_func(X) if preproc_func else X
        except:
            continue

        # Apply wavelength range filtering AFTER preprocessing
        if wl_mask is not None:
            X_processed = X_processed[:, wl_mask]

        # Run each variable selection method separately
        for method in methods:
            # Get importances for this specific method
            importances = _get_single_method_importances(
                X_processed, y, task_type, method, random_state, varsel_method_params
            )

            if importances is None:
                continue

            method_abbrev = method_abbrevs.get(method, method.upper())

            # Test each variable count
            for n_vars in valid_counts:
                # Select top variables
                top_indices = np.argsort(importances)[-n_vars:]
                X_selected = X_processed[:, top_indices]
                var_tag = f'{method_abbrev}_top{n_vars}'

                # Test with each model (using first hyperparams for variable selection)
                for model_name in models_to_test:
                    # Only use first hyperparameter config for speed
                    hyperparam_grid = get_hyperparameter_grid(model_name)
                    params = hyperparam_grid[0] if hyperparam_grid else {}

                    config_num += 1

                    if progress_callback:
                        progress_callback({
                            'stage': 'variable_selection',
                            'message': f'{preproc_name} + {model_name} ({var_tag})',
                            'current': config_num,
                            'total': total_configs,
                            'best_score': best_score
                        })

                    result = _test_single_config(
                        X_selected, y, model_name, task_type,
                        preproc_name, var_tag, params, cv, random_state,
                        wavelengths=wavelengths,
                        variable_indices=top_indices
                    )

                    if result:
                        results.append(result)
                        # Track best score
                        if task_type == 'regression':
                            if best_score is None or result['RMSE'] < best_score:
                                best_score = result['RMSE']
                        else:
                            if best_score is None or result['Accuracy'] > best_score:
                                best_score = result['Accuracy']

    return results


def _get_single_method_importances(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    method: str,
    random_state: int,
    varsel_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Optional[np.ndarray]:
    """
    Get variable importances using a single specified method.

    Parameters
    ----------
    method : str
        Variable selection method: 'importance', 'spa', 'uve', 'uve_spa', 'ipls'
    varsel_params : dict, optional
        Method-specific parameters

    Returns
    -------
    importances : np.ndarray or None
        Importance scores for each variable
    """
    varsel_params = varsel_params or {}

    try:
        if method == 'importance':
            # Use RandomForest for feature importance
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            if task_type == 'regression':
                rf = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
            else:
                rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)

            rf.fit(X, y)
            return rf.feature_importances_

        elif method == 'vip':
            # Use PLS VIP scores (chemometrics standard for PLS variable selection)
            from .variable_selection import compute_vip
            return compute_vip(X, y)

        elif method == 'spa':
            from .variable_selection import spa_selection
            params = varsel_params.get('spa', {})
            n_features = params.get('n_features', min(100, X.shape[1]))
            n_random_starts = params.get('n_random_starts', 10)
            return spa_selection(X, y, n_features=n_features, n_random_starts=n_random_starts, random_state=random_state)

        elif method == 'uve':
            from .variable_selection import uve_selection
            params = varsel_params.get('uve', {})
            cutoff = params.get('cutoff_multiplier', 1.0)
            return uve_selection(X, y, cutoff_multiplier=cutoff, random_state=random_state)

        elif method == 'uve_spa':
            from .variable_selection import uve_spa_selection
            params = varsel_params.get('uve_spa', {})
            cutoff = params.get('cutoff_multiplier', 1.0)
            n_features = params.get('n_features', min(100, X.shape[1]))
            return uve_spa_selection(X, y, n_features=n_features, cutoff_multiplier=cutoff, random_state=random_state)

        elif method == 'ipls':
            from .variable_selection import ipls_selection
            params = varsel_params.get('ipls', {})
            n_intervals = params.get('n_intervals', 20)
            return ipls_selection(X, y, n_intervals=n_intervals, random_state=random_state)

        elif method == 'ga_pls':
            # Genetic Algorithm PLS wavelength selection
            from .variable_selection import GA_PLS_AVAILABLE
            if not GA_PLS_AVAILABLE:
                print("GA-PLS not available. Ensure ga_pls.py is present.")
                return None
            from .ga_pls import ga_pls_selection
            params = varsel_params.get('ga_pls', {})
            # Extract GA-PLS parameters with defaults
            population_size = params.get('population_size', 64)
            n_generations = params.get('n_generations', 100)
            n_runs = params.get('n_runs', 5)
            crossover_rate = params.get('crossover_rate', 0.6)
            mutation_rate = params.get('mutation_rate', 0.01)
            selection_threshold = params.get('selection_threshold', 0.6)
            enable_permutation_test = params.get('enable_permutation_test', False)
            n_components = params.get('n_components', 10)
            cv_folds = params.get('cv_folds', 5)
            verbose = params.get('verbose', 1)
            return ga_pls_selection(
                X, y,
                population_size=population_size,
                n_generations=n_generations,
                n_runs=n_runs,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                selection_threshold=selection_threshold,
                enable_permutation_test=enable_permutation_test,
                n_components=n_components,
                cv_folds=cv_folds,
                task_type=task_type,
                random_state=random_state,
                verbose=verbose
            )

        else:
            print(f"Unknown variable selection method: {method}")
            return None

    except Exception as e:
        print(f"Variable selection method {method} failed: {e}")
        return None


def _get_variable_importances(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    methods: List[str],
    random_state: int,
    varsel_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Optional[np.ndarray]:
    """
    Get variable importances using the specified methods.
    Returns combined importance scores.

    Parameters
    ----------
    varsel_params : dict, optional
        Method-specific parameters, e.g.:
        {'spa': {'n_features': 50, 'n_random_starts': 10},
         'uve': {'cutoff_multiplier': 1.0},
         'ipls': {'n_intervals': 20}}
    """
    importances = None
    varsel_params = varsel_params or {}

    for method in methods:
        try:
            if method == 'importance':
                # Use RandomForest for feature importance
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

                if task_type == 'regression':
                    rf = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
                else:
                    rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)

                rf.fit(X, y)
                imp = rf.feature_importances_

            elif method == 'spa':
                from .variable_selection import spa_selection
                params = varsel_params.get('spa', {})
                n_features = params.get('n_features', min(100, X.shape[1]))
                n_random_starts = params.get('n_random_starts', 10)
                imp = spa_selection(X, y, n_features=n_features, n_random_starts=n_random_starts, random_state=random_state)

            elif method == 'uve':
                from .variable_selection import uve_selection
                params = varsel_params.get('uve', {})
                cutoff = params.get('cutoff_multiplier', 1.0)
                imp = uve_selection(X, y, cutoff_multiplier=cutoff, random_state=random_state)

            elif method == 'uve_spa':
                from .variable_selection import uve_spa_selection
                params = varsel_params.get('uve_spa', {})
                cutoff = params.get('cutoff_multiplier', 1.0)
                n_features = params.get('n_features', min(100, X.shape[1]))
                imp = uve_spa_selection(X, y, n_features=n_features, cutoff_multiplier=cutoff, random_state=random_state)

            elif method == 'ipls':
                from .variable_selection import ipls_selection
                params = varsel_params.get('ipls', {})
                n_intervals = params.get('n_intervals', 20)
                imp = ipls_selection(X, y, n_intervals=n_intervals, random_state=random_state)

            elif method == 'ga_pls':
                # Genetic Algorithm PLS wavelength selection
                from .variable_selection import GA_PLS_AVAILABLE
                if not GA_PLS_AVAILABLE:
                    print("GA-PLS not available. Skipping.")
                    continue
                from .ga_pls import ga_pls_selection
                params = varsel_params.get('ga_pls', {})
                population_size = params.get('population_size', 64)
                n_generations = params.get('n_generations', 100)
                n_runs = params.get('n_runs', 5)
                crossover_rate = params.get('crossover_rate', 0.6)
                mutation_rate = params.get('mutation_rate', 0.01)
                selection_threshold = params.get('selection_threshold', 0.6)
                enable_permutation_test = params.get('enable_permutation_test', False)
                n_components = params.get('n_components', 10)
                cv_folds = params.get('cv_folds', 5)
                verbose = params.get('verbose', 1)
                imp = ga_pls_selection(
                    X, y,
                    population_size=population_size,
                    n_generations=n_generations,
                    n_runs=n_runs,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    selection_threshold=selection_threshold,
                    enable_permutation_test=enable_permutation_test,
                    n_components=n_components,
                    cv_folds=cv_folds,
                    task_type=task_type,
                    random_state=random_state,
                    verbose=verbose
                )

            else:
                continue

            # Combine importances (average across methods)
            if importances is None:
                importances = imp
            else:
                importances = (importances + imp) / 2

        except Exception as e:
            print(f"Variable selection method {method} failed: {e}")
            continue

    return importances


def _run_region_analysis(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    task_type: str,
    tier: str,
    preproc_configs: List[Tuple[str, Callable]],
    models_to_test: List[str],
    varsel_config: Dict[str, Any],
    cv,
    random_state: int,
    progress_callback: Optional[Callable],
    current_best: Optional[float],
    start_config_num: int = 0,
    total_configs: int = 0,
    wl_mask: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Run region-based analysis and test models on spectral regions.

    This divides the spectrum into overlapping regions, computes importance
    scores (correlation for regression, Fisher ratio for classification),
    and tests models on top individual regions and combinations.

    Parameters
    ----------
    X : np.ndarray
        Original spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    task_type : str
        'regression' or 'classification'
    tier : str
        Search tier (affects which models to test)
    preproc_configs : list
        List of (name, transform_func) tuples
    models_to_test : list
        List of model names to test
    varsel_config : dict
        Variable selection config with 'n_top_regions' key
    cv : cross-validator
        Scikit-learn cross-validator
    random_state : int
        Random seed
    progress_callback : callable, optional
        Progress callback
    current_best : float, optional
        Current best score for progress display
    start_config_num : int
        Starting configuration number
    total_configs : int
        Total configurations (for progress)

    Returns
    -------
    results : list
        List of result dictionaries
    """
    results = []
    best_score = current_best
    config_num = start_config_num

    n_top_regions = varsel_config.get('n_top_regions', 5)

    for preproc_name, preproc_func in preproc_configs:
        # Apply preprocessing to FULL spectrum
        try:
            X_processed = preproc_func(X) if preproc_func else X
        except Exception as e:
            print(f"Preprocessing {preproc_name} failed in region analysis: {e}")
            continue

        # Apply wavelength range filtering AFTER preprocessing
        if wl_mask is not None:
            X_processed = X_processed[:, wl_mask]

        # Create region subsets for this preprocessed data
        try:
            region_subsets = create_region_subsets(
                X_processed, y, wavelengths,
                n_top_regions=n_top_regions,
                task_type=task_type
            )
        except Exception as e:
            print(f"Region subset creation failed for {preproc_name}: {e}")
            continue

        if not region_subsets:
            continue

        # Test each region subset
        for subset in region_subsets:
            indices = subset['indices']
            tag = subset['tag']

            if len(indices) == 0:
                continue

            X_region = X_processed[:, indices]

            # Test with each model (using first hyperparams for speed)
            for model_name in models_to_test:
                hyperparam_grid = get_hyperparameter_grid(model_name)
                params = hyperparam_grid[0] if hyperparam_grid else {}

                config_num += 1

                if progress_callback:
                    progress_callback({
                        'stage': 'region_analysis',
                        'message': f'{preproc_name} + {model_name} ({tag})',
                        'current': config_num,
                        'total': total_configs,
                        'best_score': best_score
                    })

                result = _test_single_config(
                    X_region, y, model_name, task_type,
                    preproc_name, tag, params, cv, random_state,
                    wavelengths=wavelengths,
                    variable_indices=np.asarray(indices)
                )

                if result:
                    results.append(result)
                    # Track best score
                    if task_type == 'regression':
                        if best_score is None or result['RMSE'] < best_score:
                            best_score = result['RMSE']
                    else:
                        if best_score is None or result['Accuracy'] > best_score:
                            best_score = result['Accuracy']

    return results


def _run_ipls_analysis(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    task_type: str,
    preproc_configs: List[Tuple[str, Callable]],
    models_to_test: List[str],
    ipls_config: Dict[str, Any],
    cv,
    random_state: int,
    progress_callback: Optional[Callable],
    current_best: Optional[float],
    start_config_num: int = 0,
    total_configs: int = 0,
    wl_mask: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Run iPLS interval analysis with proper interval-based subsets.

    This function implements proper Interval PLS (iPLS) that returns
    actual spectral intervals, not individual "top N" variables.

    Algorithm:
    1. For each preprocessing configuration:
       a. Run forward iPLS (if enabled) - iteratively add best intervals
       b. Run backward iPLS (if enabled) - iteratively remove worst intervals
       c. Each returns subsets with interval-based tags (e.g., 'fwd_iPLS_1400-1500nm')
    2. Test each subset with each model
    3. Return results with proper interval-based labeling

    Parameters
    ----------
    X : np.ndarray
        Original spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    task_type : str
        'regression' or 'classification' (iPLS only for regression)
    preproc_configs : list
        List of (name, transform_func) tuples
    models_to_test : list
        List of model names to test
    ipls_config : dict
        iPLS config with keys: 'n_intervals', 'forward', 'backward'
    cv : cross-validator
        Scikit-learn cross-validator
    random_state : int
        Random seed
    progress_callback : callable, optional
        Progress callback
    current_best : float, optional
        Current best score for progress display
    start_config_num : int
        Starting configuration number
    total_configs : int
        Total configurations (for progress)

    Returns
    -------
    results : list
        List of result dictionaries with interval-based variable tags
    """
    # iPLS only works for regression
    if task_type != 'regression':
        return []

    results = []
    best_score = current_best
    config_num = start_config_num

    n_intervals = ipls_config.get('n_intervals', 20)
    run_forward = ipls_config.get('forward', True)
    run_backward = ipls_config.get('backward', True)

    for preproc_name, preproc_func in preproc_configs:
        # Apply preprocessing to FULL spectrum
        try:
            X_processed = preproc_func(X) if preproc_func else X
        except Exception as e:
            print(f"Preprocessing {preproc_name} failed in iPLS analysis: {e}")
            continue

        # Apply wavelength range filtering AFTER preprocessing
        if wl_mask is not None:
            X_processed = X_processed[:, wl_mask]

        # Collect all iPLS subsets for this preprocessing
        all_subsets = []

        # Run forward iPLS
        if run_forward:
            try:
                fwd_subsets = run_ipls_forward(
                    X_processed, y, wavelengths,
                    n_intervals=n_intervals,
                    max_combine=5,
                    cv_folds=5,
                    random_state=random_state
                )
                all_subsets.extend(fwd_subsets)
            except Exception as e:
                print(f"Forward iPLS failed for {preproc_name}: {e}")

        # Run backward iPLS
        if run_backward:
            try:
                bwd_subsets = run_ipls_backward(
                    X_processed, y, wavelengths,
                    n_intervals=n_intervals,
                    cv_folds=5,
                    random_state=random_state
                )
                all_subsets.extend(bwd_subsets)
            except Exception as e:
                print(f"Backward iPLS failed for {preproc_name}: {e}")

        if not all_subsets:
            continue

        # Test each subset with each model
        for subset in all_subsets:
            indices = subset['indices']
            tag = subset['tag']

            if len(indices) == 0:
                continue

            X_subset = X_processed[:, indices]

            # Test with each model (using first hyperparams for speed)
            for model_name in models_to_test:
                hyperparam_grid = get_hyperparameter_grid(model_name)
                params = hyperparam_grid[0] if hyperparam_grid else {}

                config_num += 1

                if progress_callback:
                    progress_callback({
                        'stage': 'ipls_analysis',
                        'message': f'{preproc_name} + {model_name} ({tag})',
                        'current': config_num,
                        'total': total_configs,
                        'best_score': best_score
                    })

                result = _test_single_config(
                    X_subset, y, model_name, task_type,
                    preproc_name, tag, params, cv, random_state,
                    wavelengths=wavelengths,
                    variable_indices=np.asarray(indices)
                )

                if result:
                    results.append(result)
                    # Track best score
                    if best_score is None or result['RMSE'] < best_score:
                        best_score = result['RMSE']

    return results


def _format_params(params: Dict[str, Any]) -> str:
    """Format parameter dict as a readable string."""
    if not params:
        return 'default'

    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")

    return ', '.join(parts)


# =============================================================================
# SIMPLE RUN FUNCTION (for backwards compatibility)
# =============================================================================

def run_search(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    mode: str = 'auto',
    tier: str = 'standard',
    folds: int = 5,
    models_to_test: Optional[List[str]] = None,
    random_state: int = 42,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Run model search - unified interface for auto and manual modes.

    Parameters
    ----------
    X : np.ndarray
        Spectral data
    y : np.ndarray
        Target values
    task_type : str
        'regression' or 'classification'
    mode : str
        'auto' for full automation, 'manual' for single model
    tier : str
        'quick', 'standard', or 'comprehensive' (auto mode only)
    folds : int
        Number of CV folds
    models_to_test : list, optional
        Specific models to test (overrides tier defaults)
    random_state : int
        Random seed
    progress_callback : callable, optional
        Progress callback function
    **kwargs
        Additional arguments for manual mode (model_name, preprocessing, etc.)

    Returns
    -------
    pd.DataFrame
        Ranked results
    """
    if mode == 'manual':
        return run_manual_search(
            X, y,
            model_name=kwargs.get('model_name', 'PLS'),
            task_type=task_type,
            preprocessing=kwargs.get('preprocessing', 'raw'),
            window_size=kwargs.get('window_size', 7),
            model_params=kwargs.get('model_params'),
            folds=folds,
            random_state=random_state,
            progress_callback=progress_callback,
            wavelengths=kwargs.get('wavelengths'),
            wl_range_min=kwargs.get('wl_range_min'),
            wl_range_max=kwargs.get('wl_range_max'),
            convert_absorbance=kwargs.get('convert_absorbance', False),
            # Baseline correction
            baseline_method=kwargs.get('baseline_method'),
            baseline_params=kwargs.get('baseline_params'),
        )
    else:
        return run_auto_search(
            X, y,
            wavelengths=kwargs.get('wavelengths'),
            task_type=task_type,
            tier=tier,
            folds=folds,
            random_state=random_state,
            progress_callback=progress_callback,
            preproc_methods=kwargs.get('preproc_methods'),
            window_sizes=kwargs.get('window_sizes'),
            varsel_methods=kwargs.get('varsel_methods'),
            varsel_params=kwargs.get('varsel_params'),
            var_counts=kwargs.get('var_counts'),
            enable_regions=kwargs.get('enable_regions', False),
            n_regions=kwargs.get('n_regions', 5),
            enable_ipls=kwargs.get('enable_ipls', False),
            ipls_n_intervals=kwargs.get('ipls_n_intervals', 20),
            ipls_forward=kwargs.get('ipls_forward', True),
            ipls_backward=kwargs.get('ipls_backward', True),
            pls_max_lv=kwargs.get('pls_max_lv'),
            custom_models=kwargs.get('custom_models'),
            custom_hyperparam_grids=kwargs.get('custom_hyperparam_grids'),
            wl_range_min=kwargs.get('wl_range_min'),
            wl_range_max=kwargs.get('wl_range_max'),
            convert_absorbance=kwargs.get('convert_absorbance', False),
            # Baseline correction
            baseline_method=kwargs.get('baseline_method'),
            baseline_params=kwargs.get('baseline_params'),
            enable_smoothing=kwargs.get('enable_smoothing', False),
            smoothing_window=kwargs.get('smoothing_window', 11),
            smoothing_order=kwargs.get('smoothing_order', 2),
        )
