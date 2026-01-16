"""Model search with cross-validation and subset selection."""

import os
import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from joblib import Parallel, delayed

from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocess import build_preprocessing_pipeline
from .models import get_model_grids, get_feature_importances
from .scoring import create_results_dataframe, add_result
from .regions import create_region_subsets, format_region_report
from .variable_selection import spa_selection, uve_selection, uve_spa_selection, ipls_selection, cars_selection
from .wavelength_selection import vcpa_iriv
from .ga_pls import ga_pls_selection
from .ga_lightgbm import ga_lightgbm_selection
from .model_registry import supports_subset_analysis, supports_feature_importance
from .constants import RANDOM_STATE

from .ga_preprocessing import optimize_preprocessing, PREPROC_TYPES, WINDOW_SIZES
from .preprocessing_discovery import discover_preprocessing, IMPORTANCE_METHODS

# NSGA-II import
from .nsga2_search import run_nsga2_search, convert_nsga2_to_v1_format

# Model categories for GA preprocessing (4 specialized groups)
# Each group uses a fitness model that best represents its characteristics

# PLS-based models: Linear regression with dimension reduction
PLS_MODELS = {'PLS', 'PLS-DA', 'Ridge', 'Lasso', 'ElasticNet'}

# Neural/SVM models: Non-linear, kernel-based or neural network models
NEURAL_SVM_MODELS = {'MLP', 'SVR', 'SVC'}

# Tree models: Gradient boosting and ensemble tree methods
TREE_MODELS = {'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'}

# Neural-boosted hybrid model (single specialized model)
NEURALBOOSTED_MODELS = {'NeuralBoosted'}

# Backward compatibility: LINEAR_MODELS is union of PLS + Neural/SVM
LINEAR_MODELS = PLS_MODELS | NEURAL_SVM_MODELS


def _apply_edge_mask(importances: np.ndarray, preprocess_cfg: dict) -> np.ndarray:
    """Zero out edge importances affected by Savitzky-Golay derivatives.

    Savitzky-Golay derivatives with window W create boundary artifacts in the
    first/last W//2 wavelengths. This function masks those edge regions by
    setting their importance scores to zero, preventing variable selection
    methods from selecting artifact-affected variables.

    Parameters
    ----------
    importances : np.ndarray
        Feature importance scores with shape (n_features,)
    preprocess_cfg : dict
        Preprocessing configuration containing 'deriv' and 'window' keys

    Returns
    -------
    np.ndarray
        Importance scores with edge regions zeroed out if SG derivative is used

    Examples
    --------
    >>> importances = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    >>> cfg = {'deriv': 1, 'window': 5}
    >>> masked = _apply_edge_mask(importances, cfg)
    >>> # First and last 2 elements (5//2 = 2) are zeroed
    """
    deriv = preprocess_cfg.get("deriv")
    window = preprocess_cfg.get("window")

    # No masking needed if no derivative or window specified
    if not deriv or not window:
        return importances.copy()

    edge_margin = window // 2

    # Safety check: prevent zeroing entire array
    if 2 * edge_margin >= len(importances):
        return importances.copy()

    # Create masked copy
    masked = importances.copy()
    masked[:edge_margin] = 0.0
    masked[-edge_margin:] = 0.0

    return masked


def _get_edge_zone_size(preprocess_cfg: dict) -> int:
    """Get the size of edge zone to exclude for derivative preprocessing.

    For Savitzky-Golay derivatives with window W, the first and last W//2
    wavelengths are unreliable due to boundary effects. This function
    returns the edge zone size based on preprocessing configuration.

    Parameters
    ----------
    preprocess_cfg : dict
        Preprocessing configuration containing 'deriv' and 'window' keys

    Returns
    -------
    int
        Number of wavelengths to exclude on each edge (0 if no derivative)
    """
    deriv = preprocess_cfg.get("deriv")
    window = preprocess_cfg.get("window")

    # No edge masking needed if no derivative or window specified
    if not deriv or not window:
        return 0

    return window // 2


def _apply_edge_mask_to_data(
    X: np.ndarray,
    wavelengths: np.ndarray,
    preprocess_cfg: dict
) -> tuple:
    """Remove edge wavelengths affected by Savitzky-Golay derivatives.

    For derivative preprocessing, the first and last edge_zone wavelengths
    are unreliable. This function removes them from both data and wavelength
    arrays to ensure consistent model training across all search methods.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data with shape (n_samples, n_features)
    wavelengths : np.ndarray
        Wavelength values with shape (n_features,)
    preprocess_cfg : dict
        Preprocessing configuration containing 'deriv' and 'window' keys

    Returns
    -------
    tuple of (X_masked, wavelengths_masked, edge_zone)
        X_masked : np.ndarray with edge columns removed
        wavelengths_masked : np.ndarray with edge wavelengths removed
        edge_zone : int, the edge zone size applied
    """
    edge_zone = _get_edge_zone_size(preprocess_cfg)

    if edge_zone == 0:
        return X, wavelengths, 0

    # Safety check: ensure we keep at least some wavelengths
    if 2 * edge_zone >= X.shape[1]:
        print(f"  Warning: Edge zone ({edge_zone} per side) would remove all {X.shape[1]} wavelengths. Skipping edge masking.")
        return X, wavelengths, 0

    # Remove edge wavelengths from both data and wavelength array
    X_masked = X[:, edge_zone:-edge_zone]
    wavelengths_masked = wavelengths[edge_zone:-edge_zone]

    return X_masked, wavelengths_masked, edge_zone


def _supports_sample_weight(model):
    """Check if model.fit() accepts sample_weight parameter.

    Models like PLSRegression don't support sample_weight, so we need to check
    before passing it to avoid TypeError.
    """
    try:
        sig = inspect.signature(model.fit)
        return 'sample_weight' in sig.parameters
    except (ValueError, TypeError):
        return False


def _needs_resampling_pipeline(imbalance_method, task_type):
    """
    Determine if we need imblearn Pipeline for resampling.

    Standard sklearn Pipeline doesn't support fit_resample() methods.
    We need imblearn.pipeline.Pipeline for:
    - Classification: SMOTE, ADASYN, RandomUnderSampler, TomekLinks, etc.
    - Regression: undersample, oversample, smogn (resampling methods)

    We DON'T need it for:
    - Classification: class_weight (handled by model parameter)
    - Regression: binning, rare_boost, balanced (use RegressionSampleWeighter which only uses fit/transform)

    Parameters
    ----------
    imbalance_method : str or None
        The imbalance handling method
    task_type : str
        'classification' or 'regression'

    Returns
    -------
    bool
        True if imblearn Pipeline is needed
    """
    if imbalance_method is None:
        return False

    # class_weight doesn't need resampling pipeline (it's a model parameter)
    if imbalance_method == 'class_weight':
        return False

    # Classification resampling methods need imblearn Pipeline
    if task_type == 'classification':
        resampling_methods = ['smote', 'adasyn', 'borderline_smote',
                              'random_undersampler', 'tomek_links',
                              'smote_tomek', 'smote_enn']
        return imbalance_method.lower().replace('-', '_') in resampling_methods

    # Regression: resampling methods need imblearn Pipeline (fit_resample)
    # 'binning', 'rare_boost', 'balanced' use RegressionSampleWeighter (fit/transform only)
    if task_type == 'regression':
        resampling_methods = ['undersample', 'oversample', 'smogn']
        return imbalance_method.lower() in resampling_methods

    return False


def _rebuild_model_from_row(row: pd.Series, task_type: str):
    """Rebuild sklearn model from results row metadata.

    This function recreates the exact model configuration used during search,
    matching how Model Dev tab does it (ast.literal_eval + set_params).

    Parameters
    ----------
    row : pd.Series
        A row from the results DataFrame containing model configuration
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    model : sklearn estimator
        Model instance with correct hyperparameters applied
    """
    import ast
    from .models import get_model

    # Get model info
    model_name = row.get('Model', 'PLS')
    params_str = row.get('Params', '')
    n_lvs = row.get('LVs', None)

    # Parse params using ast.literal_eval (same as Model Dev tab)
    model_kwargs = {}
    if params_str and isinstance(params_str, str) and params_str.strip():
        try:
            parsed = ast.literal_eval(params_str)
            if isinstance(parsed, dict):
                model_kwargs = parsed
        except (ValueError, SyntaxError):
            pass  # Keep empty dict if parsing fails

    # Get model instance with n_components
    n_components = int(n_lvs) if n_lvs and not pd.isna(n_lvs) and n_lvs > 0 else 10
    # Use max_n_components high enough to not clip n_components
    model = get_model(model_name, task_type=task_type, n_components=n_components,
                      max_n_components=max(n_components, 20))

    # Apply parameters using set_params (same as Model Dev tab)
    if model_kwargs:
        try:
            model.set_params(**model_kwargs)
        except Exception as e:
            print(f"  [Warning] Could not apply params {model_kwargs}: {e}")

    # For PLS-DA classification, wrap PLSTransformer with LogisticRegression
    # This matches how PLS-DA is built during search (search.py:2933-2940)
    if task_type == 'classification' and model_name == 'PLS-DA':
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        pls_lr_pipeline = Pipeline([
            ('pls', model),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ])
        return pls_lr_pipeline

    return model


def compute_validation_metrics_for_top_models(
    df_results: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
    wavelengths: np.ndarray,
    top_n: int = 100,
    progress_callback=None
) -> pd.DataFrame:
    """Compute validation metrics for top N models.

    CRITICAL: This function matches Model Dev tab's behavior exactly:
    1. Preprocess FULL spectrum first (matching search.py behavior)
    2. THEN subset to model-specific wavelengths
    3. Fit model on preprocessed+subset data
    4. Calculate validation metrics

    Parameters
    ----------
    df_results : pd.DataFrame
        Ranked results DataFrame (must have CompositeScore column and all_vars column)
    X_train : np.ndarray
        Training spectral data (full spectrum, all wavelengths)
    y_train : np.ndarray
        Training target values
    X_val : np.ndarray
        Validation spectral data (full spectrum, all wavelengths)
    y_val : np.ndarray
        Validation target values
    task_type : str
        'regression' or 'classification'
    wavelengths : np.ndarray
        Array of wavelengths corresponding to columns in X_train and X_val
    top_n : int
        Number of top models to compute validation for
    progress_callback : callable, optional
        Progress callback function

    Returns
    -------
    pd.DataFrame
        Results with val_RMSE, val_R2 (or val_Accuracy) columns added
    """
    # Initialize columns
    if task_type == 'regression':
        df_results['val_RMSE'] = np.nan
        df_results['val_R2'] = np.nan
    else:
        df_results['val_Accuracy'] = np.nan
        df_results['val_ROC_AUC'] = np.nan
        df_results['val_F1'] = np.nan
        df_results['val_Precision'] = np.nan
        df_results['val_Recall'] = np.nan

    # Get top N indices by CompositeScore (lower is better)
    n_to_process = min(top_n, len(df_results))
    if 'CompositeScore' in df_results.columns:
        top_indices = df_results.nsmallest(n_to_process, 'CompositeScore').index
    else:
        # Fallback to first n rows
        top_indices = df_results.head(n_to_process).index

    print(f"\n[Validation] Computing validation metrics for top {n_to_process} models...")

    # Cache preprocessed data by preprocessing config to avoid redundant computation
    preprocess_cache = {}

    for i, idx in enumerate(top_indices):
        row = df_results.loc[idx]

        try:
            # === STEP 1: Get preprocessing config ===
            preprocess_name = row.get('Preprocess', 'raw')
            deriv = row.get('Deriv', 0)
            window = row.get('Window', None)
            poly = row.get('Poly', None)

            # Convert to proper types
            deriv = int(deriv) if deriv and not pd.isna(deriv) and deriv > 0 else None
            window = int(window) if window and not pd.isna(window) and window > 0 else None
            poly = int(poly) if poly and not pd.isna(poly) and poly > 0 else None

            # Create cache key
            cache_key = (preprocess_name, deriv, window, poly)

            # === STEP 2: Preprocess FULL spectrum (matching search.py and Model Dev) ===
            if cache_key in preprocess_cache:
                X_train_preprocessed, X_val_preprocessed = preprocess_cache[cache_key]
            else:
                # Build preprocessing pipeline
                prep_steps = build_preprocessing_pipeline(
                    preprocess_name,
                    deriv=deriv,
                    window=window,
                    polyorder=poly
                )

                if prep_steps:
                    prep_pipeline = Pipeline(list(prep_steps))
                    X_train_preprocessed = prep_pipeline.fit_transform(X_train)
                    X_val_preprocessed = prep_pipeline.transform(X_val)
                else:
                    X_train_preprocessed = X_train
                    X_val_preprocessed = X_val

                # Cache for reuse
                preprocess_cache[cache_key] = (X_train_preprocessed, X_val_preprocessed)

            # === STEP 3: Parse all_vars and subset to model wavelengths ===
            all_vars_str = row.get('all_vars', 'N/A')

            if all_vars_str != 'N/A' and all_vars_str and isinstance(all_vars_str, str):
                # Parse wavelengths from all_vars (e.g., "1520.0, 1540.0, 1560.0, ...")
                try:
                    model_wavelengths = [float(w.strip()) for w in all_vars_str.split(',') if w.strip()]
                except Exception as e:
                    print(f"  [Warning] Could not parse all_vars for model {i+1}: {e}")
                    model_wavelengths = None
            else:
                # Full spectrum model - use all wavelengths
                model_wavelengths = None

            # Subset AFTER preprocessing (matching Model Dev behavior)
            if model_wavelengths is not None and len(model_wavelengths) > 0:
                # Create mapping from wavelength to column index
                # CRITICAL: Do NOT sort - preserve the order from all_vars
                wl_to_idx = {float(wl): idx_wl for idx_wl, wl in enumerate(wavelengths)}

                # Get column indices for model wavelengths (in order)
                col_indices = []
                for wl in model_wavelengths:
                    if wl in wl_to_idx:
                        col_indices.append(wl_to_idx[wl])

                if len(col_indices) != len(model_wavelengths):
                    print(f"  [Warning] Only found {len(col_indices)}/{len(model_wavelengths)} wavelengths for model {i+1}")

                if not col_indices:
                    print(f"  [Warning] No wavelengths found for model {i+1}, skipping")
                    continue

                # Subset the PREPROCESSED data to selected wavelengths
                X_train_final = X_train_preprocessed[:, col_indices]
                X_val_final = X_val_preprocessed[:, col_indices]
            else:
                # Full spectrum model - use all preprocessed data
                X_train_final = X_train_preprocessed
                X_val_final = X_val_preprocessed

            # === STEP 4: Rebuild model and fit ===
            model = _rebuild_model_from_row(row, task_type)

            # Fit on training data
            model.fit(X_train_final, y_train)

            # Predict on validation data
            y_pred = model.predict(X_val_final)

            # === STEP 5: Calculate metrics ===
            if task_type == 'regression':
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_r2 = r2_score(y_val, y_pred)
                df_results.loc[idx, 'val_RMSE'] = val_rmse
                df_results.loc[idx, 'val_R2'] = val_r2
            else:
                # Accuracy
                val_acc = accuracy_score(y_val, y_pred)
                df_results.loc[idx, 'val_Accuracy'] = val_acc

                # Determine if binary or multiclass based on training data classes
                # Use 'macro' for multiclass to treat all classes equally (consistent with CV metrics)
                n_classes_train = len(np.unique(y_train))
                average_method = 'binary' if n_classes_train == 2 else 'macro'

                # F1 Score
                try:
                    val_f1 = f1_score(y_val, y_pred, average=average_method, zero_division=0)
                    df_results.loc[idx, 'val_F1'] = val_f1
                except Exception as e:
                    # Fallback to weighted if binary fails
                    try:
                        val_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                        df_results.loc[idx, 'val_F1'] = val_f1
                    except Exception as e2:
                        print(f"  [Warning] Could not compute F1 for model {i+1}: {e2}")

                # Precision
                try:
                    val_precision = precision_score(y_val, y_pred, average=average_method, zero_division=0)
                    df_results.loc[idx, 'val_Precision'] = val_precision
                except Exception as e:
                    try:
                        val_precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                        df_results.loc[idx, 'val_Precision'] = val_precision
                    except Exception as e2:
                        print(f"  [Warning] Could not compute Precision for model {i+1}: {e2}")

                # Recall
                try:
                    val_recall = recall_score(y_val, y_pred, average=average_method, zero_division=0)
                    df_results.loc[idx, 'val_Recall'] = val_recall
                except Exception as e:
                    try:
                        val_recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                        df_results.loc[idx, 'val_Recall'] = val_recall
                    except Exception as e2:
                        print(f"  [Warning] Could not compute Recall for model {i+1}: {e2}")

                # ROC AUC (requires predict_proba and at least 2 classes in validation)
                try:
                    val_classes = np.unique(y_val)
                    n_classes_val = len(val_classes)

                    if n_classes_val < 2:
                        # ROC AUC undefined with only one class - skip
                        pass
                    elif hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_val_final)
                        model_classes = model.classes_ if hasattr(model, 'classes_') else np.unique(y_train)

                        # Always subset to classes present in validation
                        # This handles: binary, multiclass, and class-mismatch cases uniformly
                        col_indices = []
                        for c in val_classes:
                            matches = np.where(model_classes == c)[0]
                            if len(matches) > 0:
                                col_indices.append(matches[0])

                        if len(col_indices) == n_classes_val:  # All validation classes found in model
                            y_proba_subset = y_proba[:, col_indices]
                            # ALWAYS renormalize to sum to 1 (even for binary)
                            # This is needed when validation has fewer classes than training
                            y_proba_subset = y_proba_subset / y_proba_subset.sum(axis=1, keepdims=True)

                            if n_classes_val == 2:
                                # Binary: use probability of second class (positive)
                                val_roc_auc = roc_auc_score(y_val, y_proba_subset[:, 1])
                            else:
                                # Multiclass: compute OvR
                                val_roc_auc = roc_auc_score(y_val, y_proba_subset, multi_class='ovr', average='macro')

                            df_results.loc[idx, 'val_ROC_AUC'] = val_roc_auc
                except Exception as e:
                    print(f"  [Warning] Could not compute ROC AUC for model {i+1}: {e}")

        except Exception as e:
            print(f"  [Warning] Failed to compute validation for model {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Progress update
        if progress_callback and (i + 1) % 10 == 0:
            progress_callback({
                'stage': 'validation_metrics',
                'message': f'Computing validation metrics ({i+1}/{n_to_process})',
                'current': i + 1,
                'total': n_to_process
            })

    print(f"[Validation] Completed validation metrics for {n_to_process} models")

    # Reorder columns to place metrics in logical order:
    # Calibration metrics first, then validation metrics
    cols = list(df_results.columns)
    if task_type == 'regression' and 'val_RMSE' in cols and 'R2' in cols:
        # Move val_RMSE and val_R2 after R2
        cols.remove('val_RMSE')
        cols.remove('val_R2')
        r2_idx = cols.index('R2')
        cols.insert(r2_idx + 1, 'val_RMSE')
        cols.insert(r2_idx + 2, 'val_R2')
        df_results = df_results[cols]
    elif task_type == 'classification':
        # Order: Accuracy, ROC_AUC, F1, Precision, Recall (calibration)
        #        val_Accuracy, val_ROC_AUC, val_F1, val_Precision, val_Recall (validation)
        cal_cols = ['Accuracy', 'ROC_AUC', 'F1', 'Precision', 'Recall']
        val_cols = ['val_Accuracy', 'val_ROC_AUC', 'val_F1', 'val_Precision', 'val_Recall']

        # Remove all metric columns that exist
        for col in cal_cols + val_cols:
            if col in cols:
                cols.remove(col)

        # Find insertion point (after Imbalance column, or after common metadata)
        if 'Imbalance' in cols:
            insert_idx = cols.index('Imbalance') + 1
        elif 'SubsetTag' in cols:
            insert_idx = cols.index('SubsetTag') + 1
        else:
            insert_idx = 0

        # Insert calibration metrics first, then validation metrics
        for i, col in enumerate(cal_cols):
            if col in df_results.columns:
                cols.insert(insert_idx + i, col)

        # Insert validation metrics after calibration metrics
        cal_count = sum(1 for c in cal_cols if c in df_results.columns)
        for i, col in enumerate(val_cols):
            if col in df_results.columns:
                cols.insert(insert_idx + cal_count + i, col)

        df_results = df_results[cols]

    return df_results


def run_search(X, y, task_type, folds=5, excluded_count=0, validation_count=0,
               total_samples_original=None, variable_penalty=0, complexity_penalty=0,
               max_n_components=8, max_iter=500, models_to_test=None, preprocessing_methods=None,
               interference_settings=None,
               window_sizes=None, n_estimators_list=None, learning_rates=None,
               neuralboosted_hidden_sizes=None, neuralboosted_activations=None,
               pls_max_iter_list=None, pls_tol_list=None,
               rf_n_trees_list=None, rf_max_depth_list=None,
               rf_min_samples_split_list=None, rf_min_samples_leaf_list=None,
               rf_max_features_list=None, rf_bootstrap_list=None,
               rf_max_leaf_nodes_list=None, rf_min_impurity_decrease_list=None,
               ridge_alphas_list=None, ridge_solver_list=None, ridge_tol_list=None,
               lasso_alphas_list=None, lasso_selection_list=None, lasso_tol_list=None,
               xgb_n_estimators_list=None, xgb_learning_rates=None, xgb_max_depths=None,
               xgb_subsample=None, xgb_colsample_bytree=None, xgb_reg_alpha=None, xgb_reg_lambda=None,
               xgb_min_child_weight_list=None, xgb_gamma_list=None,
               elasticnet_alphas_list=None, elasticnet_l1_ratios=None,
               elasticnet_selection_list=None, elasticnet_tol_list=None,
               lightgbm_n_estimators_list=None, lightgbm_learning_rates=None, lightgbm_num_leaves_list=None,
               lightgbm_max_depth_list=None, lightgbm_min_child_samples_list=None,
               lightgbm_subsample_list=None, lightgbm_colsample_bytree_list=None,
               lightgbm_reg_alpha_list=None, lightgbm_reg_lambda_list=None,
               catboost_iterations_list=None, catboost_learning_rates=None, catboost_depths=None,
               catboost_l2_leaf_reg_list=None, catboost_border_count_list=None,
               catboost_bagging_temperature_list=None, catboost_random_strength_list=None,
               svr_kernels=None, svr_C_list=None, svr_gamma_list=None,
               svr_epsilon_list=None, svr_degree_list=None, svr_coef0_list=None, svr_shrinking_list=None,
               mlp_hidden_layer_sizes_list=None, mlp_alphas_list=None, mlp_learning_rate_inits=None,
               mlp_activation_list=None, mlp_solver_list=None, mlp_batch_size_list=None,
               mlp_learning_rate_schedule_list=None, mlp_momentum_list=None,
               enable_variable_subsets=True, variable_counts=None,
               enable_region_subsets=True, n_top_regions=10, progress_callback=None,
               variable_selection_methods=None, apply_uve_prefilter=False,
               uve_cutoff_multiplier=1.0, uve_n_components=None,
               spa_n_random_starts=10, ipls_n_intervals=20,
               tier='standard', enabled_models=None,
               analysis_wl_min=None, analysis_wl_max=None,
               analysis_wl_regions=None,  # List of (min, max) tuples for multi-region support
               imbalance_method=None, imbalance_params=None, enable_class_weight=False,
               ga_preprocess=False,
               ga_preprocess_method='ga',
               ga_preprocess_population=48,
               ga_preprocess_generations=30,
               ga_preprocess_cv_folds=5,
               ga_quick_mode=False,
               # Smart preprocessing discovery parameters (NEW - replaces GA)
               smart_preprocess=False,
               smart_preprocess_importance='model_specific',
               smart_preprocess_n_top=10,
               # GA variable selection parameters
               ga_population_size=64,
               ga_generations=100,
               ga_n_runs=5,
               # Baseline and smoothing parameters
               baseline_method=None,
               baseline_params=None,
               smoothing=False,
               smoothing_window=17,
               smoothing_polyorder=2,
               # Search control (pause/resume/stop)
               controller=None,
               # Validation metrics parameters
               X_validation=None,
               y_validation=None,
               compute_validation=False,
               validation_top_n=100):
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
    variable_penalty : int (0-10), default=3
        Penalty for using many variables (0=ignore, 10=strong penalty)
    complexity_penalty : int (0-10), default=5
        Penalty for model complexity (0=ignore, 10=strong penalty)
    max_n_components : int, default=8
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
    n_top_regions : int, default=10
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
        'importance', 'spa', 'uve', 'uve_spa', 'ipls'. If None, defaults to ['importance'].
        Note: Currently only 'importance' is implemented; others are placeholders.
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
    tier : str, default='standard'
        Model tier: 'quick', 'standard', 'comprehensive', or 'experimental'
        This sets optimized defaults for all hyperparameters
    enabled_models : list of str, optional
        List of specific models to include. If None, uses all models in tier.
        Takes precedence over tier if both are specified.

    Returns
    -------
    df_ranked : pd.DataFrame
        Ranked results with all model runs
    """
    # Fixed random state used throughout codebase
    random_state = RANDOM_STATE

    # Use all cores for parallel execution
    n_jobs = -1

    X_np = X.values
    y_np = y.values
    wavelengths = X.columns.values
    n_features = X_np.shape[1]
    n_samples = X_np.shape[0]

    # Handle categorical labels for classification
    label_encoder = None
    if task_type == "classification":
        # Check if labels are non-numeric (text labels like "low", "medium", "high")
        if y_np.dtype == object or not np.issubdtype(y_np.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_original = y_np.copy()  # Keep original for logging
            y_np = label_encoder.fit_transform(y_np)
            # Log the label mapping
            label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            print(f"\n{'='*70}")
            print(f"CATEGORICAL LABEL ENCODING")
            print(f"{'='*70}")
            print(f"Detected non-numeric classification labels.")
            print(f"Encoding mapping:")
            for label, code in sorted(label_mapping.items(), key=lambda x: x[1]):
                print(f"  '{label}' -> {code}")
            print(f"{'='*70}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # UPFRONT VALIDATION FOR CLASSIFICATION IMBALANCE METHODS
    # Validate configuration BEFORE starting training to give immediate feedback
    # ═══════════════════════════════════════════════════════════════════════════
    if task_type == 'classification' and imbalance_method is not None:
        from .imbalance import validate_classification_config
        try:
            validate_classification_config(
                y=y_np,
                imbalance_method=imbalance_method,
                imbalance_params=imbalance_params,
                n_folds=folds
            )
            print(f"[OK] Imbalance configuration validated: {imbalance_method} with {folds}-fold CV")
        except ValueError as e:
            # Re-raise with clear indication this is an upfront validation error
            raise ValueError(f"Configuration Error (detected before training):\n\n{e}") from None

    # Create results container
    df_results = create_results_dataframe(task_type)

    # Handle variable selection methods (support multiple methods)
    if variable_selection_methods is None or not variable_selection_methods:
        variable_selection_methods = ['importance']

    # Filter to only implemented methods
    implemented_methods = ['importance', 'spa', 'uve', 'uve_spa', 'ipls', 'cars', 'cars-aware', 'cars-tree', 'vcpa-iriv', 'ga']
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

    # Adjust max_n_components based on CV training fold size
    # For REGRESSION: PLS requires n_components <= min(n_features, n_samples_in_training_fold)
    # For CLASSIFICATION: PLS-DA uses PLS as dimensionality reduction before LR classifier,
    #                     so we can be less strict (LR can handle more components than samples)
    # Use TRAINING fold size (not test fold) since PLS is fit on training data
    min_train_samples = n_samples * (folds - 1) // folds

    if task_type == "regression":
        # Strict constraint for PLS regression: n_components <= min(n_samples_train, n_features)
        safe_max_components = min(max_n_components, min_train_samples, n_features)
    else:
        # More relaxed constraint for PLS-DA classification
        # PLS transforms to latent space, then LR classifies
        # Allow more components since LR can handle high-dimensional input
        safe_max_components = min(max_n_components, n_features)
        # Still warn if components exceed training fold size (not recommended but allowed)
        if max_n_components > min_train_samples:
            print(f"Note: Using {max_n_components} PLS components with min_train_size~{min_train_samples}. " +
                  f"This is acceptable for PLS-DA (classification) but may cause instability.")

    if safe_max_components < max_n_components:
        print(f"Note: Reducing max components from {max_n_components} to {safe_max_components} " +
              f"due to dataset constraints (n_samples={n_samples}, n_features={n_features}, " +
              f"min_train_size~{min_train_samples}, task={task_type})")

    # Get model grids (pass n_estimators_list and learning_rates for NeuralBoosted,
    # rf_n_trees_list and rf_max_depth_list for RandomForest,
    # ridge_alphas_list and lasso_alphas_list for Ridge and Lasso,
    # xgb_* for XGBoost, elasticnet_* for ElasticNet, lightgbm_* for LightGBM, etc.,
    # tier for tiered defaults, and enabled_models for custom model selection)
    model_grids = get_model_grids(task_type, n_features, safe_max_components, max_iter,
                                   n_estimators_list=n_estimators_list, learning_rates=learning_rates,
                                   neuralboosted_hidden_sizes=neuralboosted_hidden_sizes,
                                   neuralboosted_activations=neuralboosted_activations,
                                   pls_max_iter_list=pls_max_iter_list, pls_tol_list=pls_tol_list,
                                   rf_n_trees_list=rf_n_trees_list, rf_max_depth_list=rf_max_depth_list,
                                   rf_min_samples_split_list=rf_min_samples_split_list,
                                   rf_min_samples_leaf_list=rf_min_samples_leaf_list,
                                   rf_max_features_list=rf_max_features_list,
                                   rf_bootstrap_list=rf_bootstrap_list,
                                   rf_max_leaf_nodes_list=rf_max_leaf_nodes_list,
                                   rf_min_impurity_decrease_list=rf_min_impurity_decrease_list,
                                   ridge_alphas_list=ridge_alphas_list, ridge_solver_list=ridge_solver_list,
                                   ridge_tol_list=ridge_tol_list,
                                   lasso_alphas_list=lasso_alphas_list, lasso_selection_list=lasso_selection_list,
                                   lasso_tol_list=lasso_tol_list,
                                   xgb_n_estimators_list=xgb_n_estimators_list, xgb_learning_rates=xgb_learning_rates,
                                   xgb_max_depths=xgb_max_depths, xgb_subsample=xgb_subsample,
                                   xgb_colsample_bytree=xgb_colsample_bytree, xgb_reg_alpha=xgb_reg_alpha,
                                   xgb_reg_lambda=xgb_reg_lambda,
                                   xgb_min_child_weight_list=xgb_min_child_weight_list, xgb_gamma_list=xgb_gamma_list,
                                   elasticnet_alphas_list=elasticnet_alphas_list, elasticnet_l1_ratios=elasticnet_l1_ratios,
                                   elasticnet_selection_list=elasticnet_selection_list, elasticnet_tol_list=elasticnet_tol_list,
                                   lightgbm_n_estimators_list=lightgbm_n_estimators_list,
                                   lightgbm_learning_rates=lightgbm_learning_rates,
                                   lightgbm_num_leaves_list=lightgbm_num_leaves_list,
                                   lightgbm_max_depth_list=lightgbm_max_depth_list,
                                   lightgbm_min_child_samples_list=lightgbm_min_child_samples_list,
                                   lightgbm_subsample_list=lightgbm_subsample_list,
                                   lightgbm_colsample_bytree_list=lightgbm_colsample_bytree_list,
                                   lightgbm_reg_alpha_list=lightgbm_reg_alpha_list,
                                   lightgbm_reg_lambda_list=lightgbm_reg_lambda_list,
                                   catboost_iterations_list=catboost_iterations_list,
                                   catboost_learning_rates=catboost_learning_rates, catboost_depths=catboost_depths,
                                   catboost_l2_leaf_reg_list=catboost_l2_leaf_reg_list,
                                   catboost_border_count_list=catboost_border_count_list,
                                   catboost_bagging_temperature_list=catboost_bagging_temperature_list,
                                   catboost_random_strength_list=catboost_random_strength_list,
                                   svr_kernels=svr_kernels, svr_C_list=svr_C_list, svr_gamma_list=svr_gamma_list,
                                   svr_epsilon_list=svr_epsilon_list, svr_degree_list=svr_degree_list,
                                   svr_coef0_list=svr_coef0_list, svr_shrinking_list=svr_shrinking_list,
                                   mlp_hidden_layer_sizes_list=mlp_hidden_layer_sizes_list,
                                   mlp_alphas_list=mlp_alphas_list, mlp_learning_rate_inits=mlp_learning_rate_inits,
                                   mlp_activation_list=mlp_activation_list, mlp_solver_list=mlp_solver_list,
                                   mlp_batch_size_list=mlp_batch_size_list,
                                   mlp_learning_rate_schedule_list=mlp_learning_rate_schedule_list,
                                   mlp_momentum_list=mlp_momentum_list,
                                   tier=tier, enabled_models=enabled_models, n_jobs=n_jobs)

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
            'sg3': False,  # Higher-order derivatives not default
            'sg4': False,  # Higher-order derivatives not default
            'deriv_snv': True
        }

    # Use window_sizes list if provided, otherwise default to [7, 19]
    if window_sizes is None:
        window_sizes = [7, 19]

    # Helper function to check if any interference method is actually enabled
    def _has_enabled_interference(interference):
        """Check if any interference removal method is actually enabled."""
        if interference is None or not isinstance(interference, dict):
            return False

        # Check basic methods
        if interference.get('msc', False):
            return True
        if interference.get('wavelength_exclusion', {}).get('enabled', False):
            return True
        if interference.get('osc', {}).get('enabled', False):
            return True

        # Check advanced methods
        advanced = interference.get('advanced', {})
        if isinstance(advanced, dict):
            if advanced.get('epo', {}).get('enabled', False):
                return True
            if advanced.get('dosc', {}).get('enabled', False):
                return True
            if advanced.get('glsw', {}).get('enabled', False):
                return True

        return False

    # Process interference removal settings (Phase 3)
    # ONLY include interference settings if user actually enabled methods
    # This preserves backward compatibility and reproducibility
    if interference_settings is None or not _has_enabled_interference(interference_settings):
        # No interference methods enabled - don't add to configs
        # This preserves the OLD behavior (before interference code was added)
        interference_to_add = None
    else:
        # Interference methods are enabled - include in configs
        interference_to_add = interference_settings

    # Initialize preprocessing control flag
    skip_normal_preprocessing = False

    # ═══════════════════════════════════════════════════════════════════════════
    # SMART PREPROCESSING DISCOVERY (NEW - replaces GA preprocessing)
    # Uses NSGA-II-style importance-guided wavelength selection
    # ═══════════════════════════════════════════════════════════════════════════
    if smart_preprocess:
        if progress_callback:
            progress_callback({
                'stage': 'smart_preprocessing',
                'message': 'Discovering optimal preprocessing configurations...',
                'current': 0,
                'total': 62  # Approximate number of combinations
            })

        print(f"\n{'='*70}")
        print("SMART PREPROCESSING DISCOVERY")
        print(f"{'='*70}")
        print(f"  Importance method: {smart_preprocess_importance}")
        print(f"  Number of top configs: {smart_preprocess_n_top}")
        print(f"  CV folds: {folds}")
        print(f"  Task type: {task_type}")
        print(f"  Note: This REPLACES user-selected preprocessing methods")
        print(f"{'='*70}\n")

        # Wrap progress callback for discovery
        def discovery_progress(current, total, message):
            if progress_callback:
                progress_callback({
                    'stage': 'smart_preprocessing',
                    'message': message,
                    'current': current,
                    'total': total
                })

        # Run smart preprocessing discovery
        discovered_configs = discover_preprocessing(
            X.values,  # Convert DataFrame to numpy
            y.values,  # Convert Series to numpy
            models_to_test=models_to_test,
            task_type=task_type,
            importance_method=smart_preprocess_importance,
            n_top=smart_preprocess_n_top,
            cv_folds=folds,
            progress_callback=discovery_progress
        )

        if not discovered_configs:
            print("WARNING: Smart preprocessing discovery found no valid configs!")
            print("Falling back to default preprocessing...")
            smart_preprocess = False  # Fall through to normal preprocessing
        else:
            # Convert discovered configs to format expected by rest of search.py
            preprocess_configs = []

            for i, cfg in enumerate(discovered_configs):
                # Build preprocessing name in format expected by build_preprocessing_pipeline
                base_name = cfg['preprocessing']
                window = cfg.get('window')
                deriv = cfg.get('deriv')

                # Determine base name for pipeline builder
                if base_name in ('raw', 'snv'):
                    pipeline_name = base_name
                elif base_name.startswith('snv_deriv'):
                    pipeline_name = 'snv_deriv'
                elif base_name.endswith('_snv'):
                    pipeline_name = 'deriv_snv'
                elif base_name.startswith('deriv'):
                    pipeline_name = 'deriv'
                else:
                    pipeline_name = base_name

                # Display name includes window
                if window:
                    display_name = f"{base_name}_w{window}"
                else:
                    display_name = base_name

                # Include model name in display if model-specific
                model_name = cfg.get('model_name')
                if model_name:
                    display_name = f"{display_name}_{model_name}"

                preprocess_configs.append({
                    "name": display_name,
                    "base_name": pipeline_name,
                    "deriv": deriv,
                    "window": window,
                    "polyorder": cfg.get('polyorder'),
                    "interference": interference_to_add,
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder,
                    # Smart preprocessing specific fields
                    "smart_selected_wavelengths": cfg.get('selected_wavelengths'),
                    "smart_n_wavelengths": cfg.get('n_wavelengths'),
                    "smart_score": cfg.get('score'),
                    "smart_importance_method": cfg.get('importance_method'),
                    "smart_model_name": model_name,  # Which model this was optimized for
                })

            print(f"\nCreated {len(preprocess_configs)} preprocessing configurations for grid search")
            print(f"{'='*70}\n")

            # Skip normal preprocessing config building AND old GA preprocessing
            skip_normal_preprocessing = True
            ga_preprocess = False  # Disable old GA since we're using smart preprocessing

    # ═══════════════════════════════════════════════════════════════════════════
    # GA PREPROCESSING OPTIMIZATION (LEGACY - kept for backward compatibility)
    # When enabled, this REPLACES user-selected preprocessing with GA-optimized config
    # ═══════════════════════════════════════════════════════════════════════════
    if ga_preprocess and not smart_preprocess:
        if progress_callback:
            progress_callback({
                'stage': 'ga_preprocessing',
                'message': 'Optimizing preprocessing parameters with GA...',
                'current': 0,
                'total': ga_preprocess_generations
            })

        print(f"\n{'='*70}")
        print("GA PREPROCESSING OPTIMIZATION")
        print(f"{'='*70}")
        print(f"  Search method: {ga_preprocess_method.upper()}")
        print(f"  Search space: 238 combinations (14 preproc x 17 windows)")
        if ga_preprocess_method == 'ga':
            print(f"  Population size: {ga_preprocess_population}")
            print(f"  Generations: {ga_preprocess_generations}")
        print(f"  CV folds: {ga_preprocess_cv_folds}")
        print(f"  Task type: {task_type}")
        print(f"  Note: This REPLACES user-selected preprocessing methods")
        print(f"{'='*70}\n")

        # Helper function to extract first hyperparameter set for each model
        def _extract_first_hyperparams(model_name: str, model_grids: dict) -> dict:
            """
            Extract the first hyperparameter configuration for a given model.

            This is used to create a representative model instance for GA fitness evaluation.
            Using the first hyperparameter set ensures consistency and reproducibility.

            Parameters
            ----------
            model_name : str
                Name of the model (e.g., 'PLS', 'LightGBM')
            model_grids : dict
                Model grids dict from get_model_grids()

            Returns
            -------
            params : dict
                First hyperparameter configuration for the model
            """
            if model_name in model_grids and len(model_grids[model_name]) > 0:
                return model_grids[model_name][0]
            return {}

        # NEW: Run GA optimization per-model (not per-model-group)
        # This ensures each model gets preprocessing optimized for its actual hyperparameters
        print(f"Running {ga_preprocess_method.upper()} optimization per-model with actual hyperparameters...")
        print(f"Models selected: {models_to_test}")
        print(f"")

        # Storage for GA results (one per model)
        ga_results = {}

        # Run GA optimization for each selected model
        # GA uses PROXY fitness models (PLS, LightGBM, MLP) for speed
        # After GA finds optimal preprocessing, Grid Search runs with user's full settings
        for model_name in models_to_test:
            print(f"Optimizing preprocessing for {model_name}...")

            # Determine which proxy fitness model to use based on model type
            if model_name.lower() in ['pls', 'ridge', 'lasso', 'elasticnet']:
                fitness_model = 'pls'
            elif model_name.lower() in ['lightgbm', 'xgboost', 'catboost', 'randomforest']:
                fitness_model = 'lightgbm'
            elif model_name.lower() in ['mlp', 'svr', 'svc']:
                fitness_model = 'mlp'
            elif model_name.lower() == 'neuralboosted':
                fitness_model = 'neuralboosted'
            else:
                fitness_model = 'pls'  # Default

            # Run GA/Exhaustive optimization with proxy fitness model
            ga_result = optimize_preprocessing(
                X.values,  # Convert DataFrame to numpy
                y.values,  # Convert Series to numpy
                method=ga_preprocess_method,
                population_size=ga_preprocess_population,
                n_generations=ga_preprocess_generations,
                cv_folds=folds,  # Use same CV folds as main search
                n_components=safe_max_components,  # Match grid search components
                task_type=task_type,
                random_state=random_state,
                verbose=1,
                progress_callback=progress_callback,
                fitness_model=fitness_model,  # Use proxy model for fast evaluation
                top_n=5,  # Return top 5 preprocessing configs
                n_jobs=-1 if ga_preprocess_method == 'exhaustive' else 1
            )

            ga_results[model_name] = ga_result
            print(f"  {model_name} optimization complete!")
            print(f"  Best config: {ga_result['best_config']}")
            print(f"  Best RMSECV: {ga_result['best_rmsecv']:.4f}")
            print(f"  Returning top {len(ga_result.get('configs', []))} configs\n")

        # Create preprocessing configs from all GA results
        # Each model contributes its top-N preprocessing configs
        preprocess_configs = []

        for model_name, ga_result in ga_results.items():
            configs_list = ga_result.get('configs', [])
            if not configs_list:
                # Fallback for backward compatibility (shouldn't happen with new code)
                configs_list = [{
                    'genes': ga_result['best_genes'],
                    'name': ga_result['best_name'],
                    'transform': ga_result['best_transform'],
                    'config': ga_result['best_config'],
                    'deriv': None,
                    'window': None,
                    'polyorder': None
                }]

            # Add all top-N configs for this model
            for i, cfg in enumerate(configs_list):
                base_name = cfg.get('name', 'unknown')
                preprocess_configs.append({
                    "name": f"{base_name}_{model_name}_{i+1}",  # Display name with suffix
                    "base_name": base_name,  # Base name for build_preprocessing_pipeline
                    "deriv": cfg.get('deriv'),
                    "window": cfg.get('window'),
                    "polyorder": cfg.get('polyorder'),
                    "interference": interference_to_add,
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder,
                    "ga_transform": cfg.get('transform'),
                    "ga_config": cfg.get('config'),
                    "ga_model_type": model_name,  # Track which model this was optimized for
                    "ga_genes": cfg.get('genes'),
                })

        print(f"Total preprocessing configs: {len(preprocess_configs)}")
        print(f"Breakdown: {len(models_to_test)} models × up to 5 configs each")
        print(f"{'='*70}\n")

        # Skip normal preprocessing config building
        skip_normal_preprocessing = True
    # Note: skip_normal_preprocessing is initialized to False before the blocks,
    # so we don't need an else clause here - it's already False if neither
    # smart_preprocess nor ga_preprocess set it to True

    if not skip_normal_preprocessing:
        preprocess_configs = []

        # Add raw if selected
        if preprocessing_methods.get('raw', False):
            preprocess_configs.append({
                "name": "raw",
                "deriv": None,
                "window": None,
                "polyorder": None,
                "interference": interference_to_add,  # Phase 3: Add interference settings only if enabled
                "baseline_method": baseline_method,
                "baseline_params": baseline_params,
                "smoothing": smoothing,
                "smoothing_window": smoothing_window,
                "smoothing_polyorder": smoothing_polyorder
            })

        # Add SNV if selected
        if preprocessing_methods.get('snv', False):
            preprocess_configs.append({
                "name": "snv",
                "deriv": None,
                "window": None,
                "polyorder": None,
                "interference": interference_to_add,  # Phase 3: Add interference settings only if enabled
                "baseline_method": baseline_method,
                "baseline_params": baseline_params,
                "smoothing": smoothing,
                "smoothing_window": smoothing_window,
                "smoothing_polyorder": smoothing_polyorder
            })

        # Add derivative configs based on user selections
        # For each derivative type, we create:
        # 1. Pure derivative (deriv)
        # 2. SNV then derivative (snv_deriv) - if SNV is also selected
        # 3. Derivative then SNV (deriv_snv) - if deriv_snv checkbox is selected

        if preprocessing_methods.get('sg1', False):
            # 1st derivative only
            for window in window_sizes:
                preprocess_configs.append({
                    "name": "deriv",
                    "deriv": 1,
                    "window": window,
                    "polyorder": 2,
                    "interference": interference_to_add,
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder
                })

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get('snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "snv_deriv",
                        "deriv": 1,
                        "window": window,
                        "polyorder": 2,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })

            # If deriv_snv is selected, add derivative -> SNV combination for 1st deriv
            if preprocessing_methods.get('deriv_snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "deriv_snv",
                        "deriv": 1,
                        "window": window,
                        "polyorder": 2,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })
    
        if preprocessing_methods.get('sg2', False):
            # 2nd derivative only
            for window in window_sizes:
                preprocess_configs.append({
                    "name": "deriv",
                    "deriv": 2,
                    "window": window,
                    "polyorder": 3,
                    "interference": interference_to_add,
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder
                })

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get('snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "snv_deriv",
                        "deriv": 2,
                        "window": window,
                        "polyorder": 3,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })

            # If deriv_snv is selected, add derivative -> SNV combination for 2nd deriv
            if preprocessing_methods.get('deriv_snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "deriv_snv",
                        "deriv": 2,
                        "window": window,
                        "polyorder": 3,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })
    
        if preprocessing_methods.get('sg3', False):
            # 3rd derivative only
            for window in window_sizes:
                preprocess_configs.append({
                    "name": "deriv",
                    "deriv": 3,
                    "window": window,
                    "polyorder": 4,
                    "interference": interference_to_add,
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder
                })

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get('snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "snv_deriv",
                        "deriv": 3,
                        "window": window,
                        "polyorder": 4,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })

            # If deriv_snv is selected, add derivative -> SNV combination for 3rd deriv
            if preprocessing_methods.get('deriv_snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "deriv_snv",
                        "deriv": 3,
                        "window": window,
                        "polyorder": 4,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })
    
        if preprocessing_methods.get('sg4', False):
            # 4th derivative only
            for window in window_sizes:
                preprocess_configs.append({
                    "name": "deriv",
                    "deriv": 4,
                    "window": window,
                    "polyorder": 5,
                    "interference": interference_to_add,
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder
                })

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get('snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "snv_deriv",
                        "deriv": 4,
                        "window": window,
                        "polyorder": 5,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })

            # If deriv_snv is selected, add derivative -> SNV combination for 4th deriv
            if preprocessing_methods.get('deriv_snv', False):
                for window in window_sizes:
                    preprocess_configs.append({
                        "name": "deriv_snv",
                        "deriv": 4,
                        "window": window,
                        "polyorder": 5,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder
                    })
    
        # If no preprocessing methods selected, default to raw
        if not preprocess_configs:
            print("Warning: No preprocessing methods selected. Defaulting to raw.")
            preprocess_configs.append({
                "name": "raw",
                "deriv": None,
                "window": None,
                "polyorder": None,
                "interference": interference_to_add,
                "baseline_method": baseline_method,
                "baseline_params": baseline_params,
                "smoothing": smoothing,
                "smoothing_window": smoothing_window,
                "smoothing_polyorder": smoothing_polyorder
            })

    # Create CV splitter
    if task_type == "regression":
        cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        cv_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

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
        # Check for pause/stop
        if controller and not controller.check_and_wait():
            print("Search stopped by user")
            break

        # Compute region subsets on preprocessed data for this preprocessing method
        # This ensures regions are based on the actual preprocessed features
        # ═══════════════════════════════════════════════════════════════════════════
        # GLOBAL WAVELENGTH FILTERING
        # Preprocess full spectrum first (SNV/derivatives need full range)
        # Then apply wavelength filtering for analysis
        # This ensures ALL models (full + subsets) use the same filtered data
        # ═══════════════════════════════════════════════════════════════════════════

        # Step 1: Build spectral preprocessing pipeline (NO imbalance yet)
        # Phase 3: Extract wavelengths for interference removal
        wavelengths = X.columns.astype(float).values if hasattr(X, 'columns') else None

        # Check if this is a GA-optimized preprocessing config
        if 'ga_transform' in preprocess_cfg and preprocess_cfg['ga_transform'] is not None:
            # Use GA transform directly (it already includes all preprocessing)
            X_preprocessed = preprocess_cfg['ga_transform'](X_np)
        else:
            # Use standard preprocessing pipeline
            # Use base_name if available (for GA configs), otherwise use name
            preprocess_name = preprocess_cfg.get("base_name", preprocess_cfg["name"])
            prep_pipe_steps = build_preprocessing_pipeline(
                preprocess_name,
                preprocess_cfg["deriv"],
                preprocess_cfg["window"],
                preprocess_cfg["polyorder"],
                imbalance_method=None,  # Imbalance will be added later inside CV folds
                imbalance_params=None,
                task_type=task_type,
                interference=preprocess_cfg.get("interference"),  # Phase 3
                wavelengths=wavelengths,  # Phase 3
                baseline_method=preprocess_cfg.get("baseline_method"),
                baseline_params=preprocess_cfg.get("baseline_params"),
                smoothing=preprocess_cfg.get("smoothing", False),
                smoothing_window=preprocess_cfg.get("smoothing_window", 17),
                smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2)
            )

            # Step 2: Apply preprocessing to full spectrum
            X_preprocessed = X_np.copy()
            if prep_pipe_steps:
                prep_pipeline = Pipeline(prep_pipe_steps)
                X_preprocessed = prep_pipeline.fit_transform(X_preprocessed, y_np)

        # Store original wavelength count before filtering (needed for Model Dev tab)
        n_original_wavelengths = len(wavelengths)

        # Step 3: Apply wavelength filtering to preprocessed data
        # ═══════════════════════════════════════════════════════════════════════════
        # WAVELENGTH FILTERING (multi-region or single range)
        # Custom regions take precedence over min/max if specified
        # ═══════════════════════════════════════════════════════════════════════════
        if analysis_wl_regions is not None and len(analysis_wl_regions) > 0:
            # Multi-region filtering (custom wavelength regions)
            wavelengths_float = wavelengths.astype(float)
            wl_mask = np.zeros(len(wavelengths), dtype=bool)

            # Build OR-mask across all regions
            for region_min, region_max in analysis_wl_regions:
                region_mask = (wavelengths_float >= region_min) & (wavelengths_float <= region_max)
                wl_mask |= region_mask

            # Validate non-empty selection
            n_wavelengths_selected = wl_mask.sum()
            if n_wavelengths_selected == 0:
                regions_str = ", ".join([f"{r[0]:.0f}-{r[1]:.0f}" for r in analysis_wl_regions])
                raise ValueError(
                    f"Custom wavelength regions [{regions_str}] nm exclude all wavelengths. "
                    f"Available range: {wavelengths_float.min():.1f}-{wavelengths_float.max():.1f} nm"
                )

            # Create filtered variables (scoped to this preprocessing config)
            X_for_models = X_preprocessed[:, wl_mask]
            wavelengths_for_models = wavelengths[wl_mask]

            # Generate preprocessing name for display
            prep_display = preprocess_cfg["name"]
            if preprocess_cfg["deriv"] is not None:
                prep_display += f"_d{preprocess_cfg['deriv']}"

            # Print filtering summary (shown once per preprocessing method)
            print(f"\n{'='*70}")
            print(f"WAVELENGTH FILTERING - CUSTOM REGIONS (after {prep_display} preprocessing)")
            print(f"{'='*70}")
            n_regions = len(analysis_wl_regions)
            for i, (r_min, r_max) in enumerate(analysis_wl_regions):
                region_count = ((wavelengths_float >= r_min) & (wavelengths_float <= r_max)).sum()
                print(f"  Region {i+1}: {r_min:.0f} - {r_max:.0f} nm ({region_count} wavelengths)")
            print(f"  Total wavelengths kept: {len(wavelengths_for_models)} of {len(wavelengths)}")
            print(f"  Note: Preprocessing applied to FULL spectrum before filtering")
            print(f"{'='*70}\n")

        elif analysis_wl_min is not None or analysis_wl_max is not None:
            # Single-range filtering (original behavior)
            wavelengths_float = wavelengths.astype(float)
            wl_mask = np.ones(len(wavelengths), dtype=bool)

            if analysis_wl_min is not None:
                wl_mask &= (wavelengths_float >= analysis_wl_min)
            if analysis_wl_max is not None:
                wl_mask &= (wavelengths_float <= analysis_wl_max)

            # Validate non-empty selection
            n_wavelengths_selected = wl_mask.sum()
            if n_wavelengths_selected == 0:
                raise ValueError(
                    f"Wavelength range {analysis_wl_min}-{analysis_wl_max} nm excludes all wavelengths. "
                    f"Available range: {wavelengths_float.min():.1f}-{wavelengths_float.max():.1f} nm"
                )

            # Create filtered variables (scoped to this preprocessing config)
            X_for_models = X_preprocessed[:, wl_mask]
            wavelengths_for_models = wavelengths[wl_mask]

            # Generate preprocessing name for display
            prep_display = preprocess_cfg["name"]
            if preprocess_cfg["deriv"] is not None:
                prep_display += f"_d{preprocess_cfg['deriv']}"

            # Print filtering summary (shown once per preprocessing method)
            print(f"\n{'='*70}")
            print(f"WAVELENGTH FILTERING (after {prep_display} preprocessing)")
            print(f"{'='*70}")
            print(f"  Range: {analysis_wl_min or 'min'} - {analysis_wl_max or 'max'} nm")
            print(f"  Wavelengths kept: {len(wavelengths_for_models)} of {len(wavelengths)}")
            print(f"  Note: Preprocessing applied to FULL spectrum before filtering")
            print(f"{'='*70}\n")
        else:
            # No filtering - use preprocessed data as-is
            X_for_models = X_preprocessed
            wavelengths_for_models = wavelengths

        # End of wavelength filtering block

        # Track if wavelength restriction is active
        # When active, skip edge masking because:
        # 1. Preprocessing (derivatives) was applied to FULL spectrum first
        # 2. Edge artifacts only exist at edges of ORIGINAL spectrum
        # 3. Restricted wavelengths are from MIDDLE of spectrum, not SG boundaries
        wavelength_restriction_active = bool(
            analysis_wl_regions or analysis_wl_min is not None or analysis_wl_max is not None
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # EDGE MASKING FOR DERIVATIVE PREPROCESSING
        # Savitzky-Golay derivatives create boundary artifacts at spectrum edges.
        # Edge zone = window // 2 on each side (unreliable due to SG interpolation).
        # This matches NSGA-II behavior for consistent R2 values across methods.
        # SKIP when wavelength restriction is active - those are middle wavelengths.
        # ═══════════════════════════════════════════════════════════════════════════
        edge_zone_applied = 0
        if preprocess_cfg.get("deriv") and preprocess_cfg.get("window") and not wavelength_restriction_active:
            X_for_models, wavelengths_for_models, edge_zone_applied = _apply_edge_mask_to_data(
                X_for_models, wavelengths_for_models, preprocess_cfg
            )
            if edge_zone_applied > 0:
                prep_name = preprocess_cfg.get("name", "unknown")
                deriv_info = f"_d{preprocess_cfg['deriv']}" if preprocess_cfg["deriv"] else ""
                print(f"\n{'='*70}")
                print(f"EDGE MASKING (after {prep_name}{deriv_info} preprocessing)")
                print(f"{'='*70}")
                print(f"  Derivative window: {preprocess_cfg['window']}")
                print(f"  Edge zone: {edge_zone_applied} wavelengths on each side")
                print(f"  Wavelengths after masking: {len(wavelengths_for_models)}")
                print(f"  Range: {wavelengths_for_models[0]:.1f} - {wavelengths_for_models[-1]:.1f} nm")
                print(f"{'='*70}\n")

        # ═══════════════════════════════════════════════════════════════════════════
        # REGION SUBSET COMPUTATION (after wavelength filtering)
        # Compute regions on filtered+preprocessed data
        # This ensures regions respect user's wavelength range selection
        # ═══════════════════════════════════════════════════════════════════════════
        region_subsets = []
        if enable_region_subsets:
            try:
                # Use filtered wavelengths for region computation
                wavelengths_float = wavelengths_for_models.astype(float)
                region_subsets = create_region_subsets(
                    X_for_models,              # Use filtered+preprocessed data
                    y_np,
                    wavelengths_float,         # Use filtered wavelengths
                    n_top_regions=n_top_regions
                )

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

        # End of region computation block

        for model_name, model_configs in model_grids.items():
            # Check for pause/stop
            if controller and not controller.check_and_wait():
                break

            # ═══════════════════════════════════════════════════════════════════════════
            # GA PREPROCESSING: Skip incompatible preprocessing configs
            # When GA preprocessing is enabled, only use the config appropriate for this model group
            # ═══════════════════════════════════════════════════════════════════════════
            if ga_preprocess and 'ga_model_type' in preprocess_cfg:
                # Determine this model's group
                if model_name in PLS_MODELS:
                    required_ga_type = "pls"
                elif model_name in NEURAL_SVM_MODELS:
                    required_ga_type = "neural_svm"
                elif model_name in TREE_MODELS:
                    required_ga_type = "tree"
                elif model_name in NEURALBOOSTED_MODELS:
                    required_ga_type = "neuralboosted"
                else:
                    # Unknown model type, use pls by default
                    required_ga_type = "pls"

                # Skip if this preprocessing config doesn't match the model group
                if preprocess_cfg['ga_model_type'] != required_ga_type:
                    continue

            for model, params in model_configs:
                # Check for pause/stop at each config
                if controller and not controller.check_and_wait():
                    print("Search stopped by user")
                    break

                current_config += 1

                # Progress update
                prep_name = preprocess_cfg["name"]
                # Only add derivative suffix if name doesn't already include it
                if preprocess_cfg["deriv"] and f"deriv{preprocess_cfg['deriv']}" not in prep_name:
                    prep_name += f"_d{preprocess_cfg['deriv']}"

                # Show parameters being tested (more informative)
                param_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:2]])  # Show first 2 params
                if len(params) > 2:
                    param_str += "..."

                progress_msg = f"Testing {model_name} ({param_str}) + {prep_name}"

                # Add best model so far to progress
                best_info = ""
                if best_model_so_far is not None:
                    if task_type == "regression":
                        best_info = f" | Best: R²={best_model_so_far['R2']:.3f}, RMSE={best_model_so_far['RMSE']:.3f}"
                    else:
                        best_info = f" | Best: AUC={best_model_so_far.get('ROC_AUC', 0):.3f}"

                print(f"[{current_config}/{total_configs}] {progress_msg}{best_info}")

                if progress_callback:
                    progress_callback({
                        'stage': 'model_testing',
                        'message': progress_msg,
                        'current': current_config,
                        'total': total_configs,
                        'best_model': best_model_so_far
                    })

                # Run full model first (using preprocessed + filtered data)
                result = _run_single_config(
                    X_for_models,           # Preprocessed + filtered data
                    y_np,
                    wavelengths_for_models, # Filtered wavelengths
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
                    skip_spectral_preprocessing=True,  # Skip spectral preprocessing (already done), keep imbalance handling
                    excluded_count=excluded_count,
                    imbalance_method=imbalance_method,
                    imbalance_params=imbalance_params,
                    validation_count=validation_count,
                    total_samples_original=total_samples_original,
                    folds=folds,
                    full_vars_original=n_original_wavelengths,
                    n_jobs_cv=n_jobs,
                    wavelength_restriction_active=wavelength_restriction_active,
                )
                df_results = add_result(df_results, result)

                # Show full model result
                if task_type == "regression":
                    print(f"     Full model: R²={result['R2']:.3f}, RMSE={result['RMSE']:.3f}")
                else:
                    print(f"     Full model: AUC={result.get('ROC_AUC', 0):.3f}, Acc={result.get('Accuracy', 0):.3f}")

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

                # For models that support feature importance: compute importances and run subsets
                # IMPORTANT: Importances are computed on PREPROCESSED data, ensuring that
                # wavelength selection reflects the actual transformed features the model sees
                if supports_subset_analysis(model_name):
                    if not enable_variable_subsets:
                        print(f"  -> Skipping subset analysis for {model_name} (variable subsets disabled)")
                    else:
                        print(f"  -> Computing feature importances for {model_name} subset analysis...")

                        # Cap n_components for PLS when fitting on filtered data
                        # (model was created with n_components based on original feature count,
                        # but X_for_models may have fewer features after wavelength filtering)
                        n_features_filtered = X_for_models.shape[1]
                        if hasattr(model, 'n_components') and model.n_components is not None:
                            if model.n_components >= n_features_filtered:
                                model = clone(model)
                                capped = max(1, n_features_filtered - 1)
                                model.set_params(n_components=capped)
                                print(f"     Note: Capped PLS n_components to {capped} for importance computation (only {n_features_filtered} features)")

                        # Build model-only pipeline (data is already preprocessed and filtered)
                        pipe_steps = []
                        pipe_steps.append(("model", model))
                        pipe = Pipeline(pipe_steps)

                        # Fit on preprocessed+filtered data
                        pipe.fit(X_for_models, y_np)

                        # Get model from pipeline
                        fitted_model = pipe.named_steps["model"]

                        # X_for_models is already preprocessed and filtered - use directly
                        X_transformed_varsel = X_for_models
                        wavelengths_varsel = wavelengths_for_models
                        n_features_varsel = X_for_models.shape[1]
                        n_features_for_validation = n_features_varsel  # Define early for SPA/UVE-SPA methods

                        # Loop over each selected variable selection method
                        # DEBUG: Print what methods will be processed
                        print(f"[DEBUG] Processing variable selection methods: {selected_methods}")
                        for varsel_method in selected_methods:
                            # Check for pause/stop
                            if controller and not controller.check_and_wait():
                                break

                            # Get importances computed on preprocessed data
                            try:
                                if varsel_method == 'importance':
                                    importances = get_feature_importances(
                                        fitted_model, model_name, X_transformed_varsel, y_np
                                    )

                                elif varsel_method == 'spa':
                                    # SPA: Successive Projections Algorithm - reduces collinearity
                                    # Select minimally correlated variables
                                    # Use max variable count as default for SPA feature selection
                                    default_n_select = max(variable_counts) if variable_counts else 100
                                    n_to_select = min(default_n_select, n_features_for_validation)
                                    importances = spa_selection(
                                        X_transformed_varsel, y_np,
                                        n_features=n_to_select,
                                        n_random_starts=spa_n_random_starts,
                                        cv_folds=folds,
                                        random_state=random_state
                                    )

                                elif varsel_method == 'uve':
                                    # UVE: Uninformative Variable Elimination - filters noise
                                    importances = uve_selection(
                                        X_transformed_varsel, y_np,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        n_components=uve_n_components,
                                        cv_folds=folds,
                                        random_state=random_state
                                    )

                                elif varsel_method == 'uve_spa':
                                    # UVE-SPA: Hybrid method - filters noise then reduces collinearity
                                    # Use max variable count as default for UVE-SPA feature selection
                                    default_n_select = max(variable_counts) if variable_counts else 100
                                    n_to_select = min(default_n_select, n_features_for_validation)
                                    print(f"    -> Running UVE-SPA (target: {n_to_select} features)")
                                    importances = uve_spa_selection(
                                        X_transformed_varsel, y_np,
                                        n_features=n_to_select,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        uve_n_components=uve_n_components,
                                        uve_cv_folds=folds,
                                        spa_n_random_starts=spa_n_random_starts,
                                        spa_cv_folds=folds,
                                        random_state=random_state
                                    )
                                    n_nonzero = np.sum(importances > 0) if importances is not None else 0
                                    print(f"    -> UVE-SPA completed: {n_nonzero} variables with non-zero importance")

                                elif varsel_method == 'ipls':
                                    # iPLS: Interval PLS - selects based on spectral regions
                                    importances = ipls_selection(
                                        X_transformed_varsel, y_np,
                                        n_intervals=ipls_n_intervals,
                                        n_components=uve_n_components,
                                        cv_folds=folds,
                                        random_state=random_state
                                    )

                                elif varsel_method in ('cars', 'cars-aware', 'cars-tree'):
                                    # CARS: Competitive Adaptive Reweighted Sampling
                                    # Monte Carlo-based method with exponential decay
                                    # cars-aware: Use model-appropriate fitness (LightGBM for tree models)
                                    # cars-tree: Hybrid importance (split+gain) for tree models
                                    if varsel_method == 'cars':
                                        model_type_for_cars = None
                                        use_hybrid = False
                                    elif varsel_method == 'cars-aware':
                                        model_type_for_cars = model_name
                                        use_hybrid = False
                                        print(f"    -> Running Model-Aware CARS for {model_name}")
                                    else:  # cars-tree
                                        model_type_for_cars = model_name
                                        use_hybrid = True
                                        print(f"    -> Running CARS-Tree (hybrid importance) for {model_name}")

                                    importances = cars_selection(
                                        X_transformed_varsel, y_np,
                                        n_iterations=50,
                                        pls_components=uve_n_components if uve_n_components is not None else 5,
                                        cv_folds=folds,
                                        monte_carlo_samples=80,
                                        random_state=random_state,
                                        model_type=model_type_for_cars,
                                        use_hybrid_importance=use_hybrid,
                                        hybrid_importance_weight=0.5
                                    )

                                elif varsel_method == 'vcpa-iriv':
                                    # VCPA-IRIV: Variable Combination Population Analysis
                                    # Iterative elimination with binary matrix sampling
                                    print(f"    -> Running VCPA-IRIV (n_outer=10, n_inner=50)")
                                    result = vcpa_iriv(
                                        X_transformed_varsel, y_np,
                                        n_outer_iterations=10,
                                        n_inner_iterations=50,
                                        pls_components=uve_n_components if uve_n_components is not None else 5,
                                        cv_folds=folds,
                                        random_state=random_state
                                    )
                                    # Extract importance scores from result dict
                                    # Note: vcpa_iriv returns 'importance_scores', not 'importances'
                                    importances = result.get('importance_scores', result.get('importances', None))

                                    # VCPA returns importance_scores for ACTIVE indices only
                                    # We need to create full-length importance array using selected_indices
                                    selected = result.get('selected_indices', [])
                                    if importances is not None and len(importances) == len(selected):
                                        # Map importance scores back to full wavelength array
                                        full_importances = np.zeros(X_transformed_varsel.shape[1])
                                        full_importances[selected] = importances
                                        importances = full_importances
                                        print(f"    -> VCPA-IRIV selected {len(selected)} variables with importance scores")
                                    elif len(selected) > 0:
                                        # Fallback: create binary mask from selected_indices
                                        importances = np.zeros(X_transformed_varsel.shape[1])
                                        importances[selected] = 1.0
                                        print(f"    -> VCPA-IRIV selected {len(selected)} variables (binary mask fallback)")
                                    else:
                                        # No variables selected - use uniform importances
                                        print(f"    -> WARNING: VCPA-IRIV selected no variables, using uniform importances")
                                        importances = np.ones(X_transformed_varsel.shape[1])

                                elif varsel_method == 'ga':
                                    # GA Variable Selection: Use model-appropriate fitness
                                    # Linear models use PLS fitness, tree models use LightGBM fitness

                                    # Determine GA parameters based on quick mode or user settings
                                    if ga_quick_mode:
                                        ga_pop, ga_gen, ga_runs, ga_early = 32, 50, 2, 10
                                        print(f"    -> Quick GA Mode: pop={ga_pop}, gen={ga_gen}, runs={ga_runs}")
                                    else:
                                        # Use user-specified parameters
                                        ga_pop = ga_population_size
                                        ga_gen = ga_generations
                                        ga_runs = ga_n_runs
                                        ga_early = 20  # Default early stopping
                                        print(f"    -> GA Mode: pop={ga_pop}, gen={ga_gen}, runs={ga_runs}")

                                    if model_name in LINEAR_MODELS:
                                        print(f"    -> Using GA-PLS for {model_name} (linear model)")
                                        importances = ga_pls_selection(
                                            X_transformed_varsel, y_np,
                                            task_type=task_type,
                                            n_components=uve_n_components if uve_n_components is not None else 10,
                                            cv=folds,
                                            population_size=ga_pop,
                                            n_generations=ga_gen,
                                            n_runs=ga_runs,
                                            early_stopping=ga_early,
                                            random_state=random_state,
                                            progress_callback=progress_callback
                                        )
                                    elif model_name in TREE_MODELS:
                                        print(f"    -> Using GA-LightGBM for {model_name} (tree model)")
                                        importances = ga_lightgbm_selection(
                                            X_transformed_varsel, y_np,
                                            task_type=task_type,
                                            cv_folds=folds,
                                            n_estimators=50,
                                            num_leaves=15 if task_type == 'classification' else 31,
                                            population_size=ga_pop,
                                            n_generations=ga_gen,
                                            n_runs=ga_runs,
                                            early_stopping=ga_early,
                                            random_state=random_state,
                                            progress_callback=progress_callback
                                        )
                                    else:
                                        # Default to GA-PLS for unknown model types
                                        print(f"    -> Using GA-PLS for {model_name} (default)")
                                        importances = ga_pls_selection(
                                            X_transformed_varsel, y_np,
                                            task_type=task_type,
                                            n_components=uve_n_components if uve_n_components is not None else 10,
                                            cv=folds,
                                            population_size=ga_pop,
                                            n_generations=ga_gen,
                                            n_runs=ga_runs,
                                            early_stopping=ga_early,
                                            random_state=random_state,
                                            progress_callback=progress_callback
                                        )

                                else:
                                    # This shouldn't happen due to filtering, but handle gracefully
                                    print(f"  -> Skipping unimplemented method '{varsel_method}'")
                                    continue

                                # Track if uniform fallback was used (for debugging/filtering results)
                                used_uniform_fallback = False

                                # Validate importances array before proceeding
                                if importances is None:
                                    print(f"  -> ERROR: {varsel_method} returned None importances, skipping")
                                    continue
                                if len(importances) != X_transformed_varsel.shape[1]:
                                    print(f"  -> ERROR: {varsel_method} returned wrong-sized importances "
                                          f"({len(importances)} vs {X_transformed_varsel.shape[1]}), skipping")
                                    continue
                                if np.all(importances == 0):
                                    print(f"  -> WARNING: {varsel_method} returned all-zero importances, using uniform")
                                    importances = np.ones(X_transformed_varsel.shape[1])
                                    used_uniform_fallback = True

                                # Use user-specified variable counts, or default if not provided
                                if variable_counts is None:
                                    user_variable_counts = [10, 20, 50, 100, 250, 500, 1000]
                                else:
                                    user_variable_counts = variable_counts

                                # For validation, use the feature count from the FILTERED PREPROCESSED data
                                # (derivatives reduce feature count, wavelength filtering further reduces)
                                n_features_for_validation = n_features_varsel

                                # Only test counts that are less than total features
                                valid_variable_counts = [n for n in user_variable_counts if n < n_features_for_validation]

                                print(f"  -> User variable counts: {user_variable_counts}")
                                print(f"  -> Valid variable counts (< {n_features_for_validation} features): {valid_variable_counts}")
                                print(f"  -> Variable selection method: {varsel_method}")

                                if not valid_variable_counts:
                                    print(f"  WARNING: No valid variable counts to test (all selected counts >= {n_features_for_validation} features)")

                                # DEBUG: Show importances summary for this method
                                print(f"  [DEBUG] {varsel_method} importances: min={np.min(importances):.4f}, max={np.max(importances):.4f}, std={np.std(importances):.4f}")

                                # Apply edge masking for Savitzky-Golay derivatives
                                # SKIP when wavelength restriction is active - restricted wavelengths
                                # are from middle of spectrum, not SG boundary edges
                                if not wavelength_restriction_active:
                                    importances = _apply_edge_mask(importances, preprocess_cfg)

                                # Run subsets with user-selected counts
                                results_added_for_method = 0
                                for n_top in valid_variable_counts:
                                    print(f"  -> Testing top-{n_top} vars ({varsel_method})...", end=" ")
                                    # Select top N most important features based on preprocessed importances
                                    # Use stable sort to ensure deterministic feature ordering when importances are tied
                                    top_indices = np.argsort(importances, kind='stable')[-n_top:][::-1]

                                    # DEBUG: Show first 5 selected wavelengths for comparison
                                    if n_top == valid_variable_counts[0]:  # Only for first subset size
                                        selected_wls = wavelengths_varsel[top_indices[:5]]
                                        print(f"\n      [DEBUG] Top 5 wavelengths for {varsel_method}: {selected_wls}")

                                    # For derivative preprocessing: importances are computed on transformed features
                                    # We must use the TRANSFORMED data and skip reapplying preprocessing
                                    # Otherwise window size (e.g., 17) > n_features (e.g., 10) causes errors
                                    if preprocess_cfg["deriv"] is not None:
                                        # Use preprocessed+filtered data (already done globally)
                                        # Keep original preprocess_cfg for correct labeling in results
                                        subset_result = _run_single_config(
                                            X_transformed_varsel,  # Already preprocessed + filtered
                                            y_np,
                                            wavelengths_varsel,
                                            model,
                                            model_name,
                                            params,
                                            preprocess_cfg,  # Keep original config for labeling
                                            cv_splitter,
                                            task_type,
                                            is_binary_classification,
                                            subset_indices=top_indices,
                                            subset_tag=f"top{n_top}_{varsel_method}",
                                            top_n_vars=30,
                                            skip_preprocessing=False,
                                            skip_spectral_preprocessing=True,  # Spectral already done, keep imbalance
                                            excluded_count=excluded_count,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            validation_count=validation_count,
                                            total_samples_original=total_samples_original,
                                            folds=folds,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=n_jobs,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                        )
                                    else:
                                        # For raw/SNV: use filtered data since indices reference filtered array
                                        # Data is already preprocessed and filtered globally
                                        subset_result = _run_single_config(
                                            X_transformed_varsel,  # Already preprocessed + filtered
                                            y_np,
                                            wavelengths_varsel,
                                            model,
                                            model_name,
                                            params,
                                            preprocess_cfg,
                                            cv_splitter,
                                            task_type,
                                            is_binary_classification,
                                            subset_indices=top_indices,
                                            subset_tag=f"top{n_top}_{varsel_method}",
                                            top_n_vars=30,
                                            skip_preprocessing=False,
                                            skip_spectral_preprocessing=True,  # Spectral already done, keep imbalance
                                            excluded_count=excluded_count,
                                            validation_count=validation_count,
                                            total_samples_original=total_samples_original,
                                            folds=folds,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=n_jobs,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                        )

                                    # Track if uniform fallback was used for this result
                                    subset_result["uniform_fallback"] = used_uniform_fallback

                                    df_results = add_result(df_results, subset_result)
                                    results_added_for_method += 1

                                    # Show result immediately
                                    if task_type == "regression":
                                        print(f"R²={subset_result['R2']:.3f}, RMSE={subset_result['RMSE']:.3f}")
                                    else:
                                        print(f"AUC={subset_result.get('ROC_AUC', 0):.3f}, Acc={subset_result.get('Accuracy', 0):.3f}")

                                    # Update best model tracker for subset results
                                    if best_model_so_far is None:
                                        best_model_so_far = subset_result
                                    else:
                                        if task_type == "regression":
                                            if subset_result["RMSE"] < best_model_so_far["RMSE"]:
                                                best_model_so_far = subset_result
                                        else:  # classification
                                            if subset_result.get("ROC_AUC", 0) > best_model_so_far.get("ROC_AUC", 0):
                                                best_model_so_far = subset_result

                                # Summary for this variable selection method
                                print(f"  [SUMMARY] {varsel_method}: Added {results_added_for_method} results to dataframe")

                            except Exception as e:
                                import traceback
                                print(f"Warning: Could not compute importances for {model_name} with method '{varsel_method}': {e}")
                                print(f"  Full traceback:\n{traceback.format_exc()}")

                # Run region-based subsets for ALL models (not just PLS/RF/MLP/NeuralBoosted)
                # For derivatives: use preprocessed data to avoid reapplying preprocessing
                # For raw/SNV: use raw data and reapply preprocessing
                if enable_region_subsets and len(region_subsets) > 0:
                    print(f"  -> Testing {len(region_subsets)} spectral regions:")
                    for i, region_subset in enumerate(region_subsets, 1):
                        print(f"     Region {i}/{len(region_subsets)} ({region_subset['tag']})...", end=" ")
                        # Use filtered+preprocessed data for ALL preprocessing types
                        # Region indices were computed on filtered data, so this is correct
                        region_result = _run_single_config(
                            X_for_models,              # Filtered+preprocessed data
                            y_np,
                            wavelengths_for_models,    # Filtered wavelengths
                            model,
                            model_name,
                            params,
                            preprocess_cfg,  # Keep original config for labeling
                            cv_splitter,
                            task_type,
                            is_binary_classification,
                            subset_indices=region_subset['indices'],
                            subset_tag=region_subset['tag'],
                            top_n_vars=30,
                            skip_preprocessing=False,
                            skip_spectral_preprocessing=True,  # Spectral preprocessing already done
                            excluded_count=excluded_count,
                            validation_count=validation_count,
                            total_samples_original=total_samples_original,
                            folds=folds,
                            imbalance_method=imbalance_method,
                            imbalance_params=imbalance_params,
                            full_vars_original=n_original_wavelengths,
                            n_jobs_cv=n_jobs,
                            wavelength_restriction_active=wavelength_restriction_active,
                        )
                        df_results = add_result(df_results, region_result)

                        # Show result immediately
                        if task_type == "regression":
                            print(f"R²={region_result['R2']:.3f}, RMSE={region_result['RMSE']:.3f}")
                        else:
                            print(f"AUC={region_result.get('ROC_AUC', 0):.3f}, Acc={region_result.get('Accuracy', 0):.3f}")

                        # Update best model tracker for region subset results
                        if best_model_so_far is None:
                            best_model_so_far = region_result
                        else:
                            if task_type == "regression":
                                if region_result["RMSE"] < best_model_so_far["RMSE"]:
                                    best_model_so_far = region_result
                            else:  # classification
                                if region_result.get("ROC_AUC", 0) > best_model_so_far.get("ROC_AUC", 0):
                                    best_model_so_far = region_result

    # Compute composite scores and rank
    from .scoring import compute_composite_score

    # Check for subset contamination (mixing full-spectrum with subset models)
    if "SubsetTag" in df_results.columns:
        subset_counts = df_results["SubsetTag"].value_counts()
        if len(subset_counts) > 1:
            print("\n[WARNING] Ranking includes multiple subset types:")
            for subset_type, count in subset_counts.items():
                print(f"  - {subset_type}: {count} models")
            print("  Subset models may rank higher due to lower variable counts.")
            print("  Consider filtering by SubsetTag before ranking for fairer comparison.\n")

    # DEBUG: Count results before scoring
    print(f"\n[DEBUG] Results before scoring: {len(df_results)} rows")
    if "SubsetTag" in df_results.columns:
        print(f"[DEBUG] SubsetTag counts BEFORE scoring:\n{df_results['SubsetTag'].value_counts().to_string()}")

    df_ranked = compute_composite_score(df_results, task_type, variable_penalty, complexity_penalty)

    # DEBUG: Count results after scoring
    print(f"\n[DEBUG] Results after scoring: {len(df_ranked)} rows")
    if "SubsetTag" in df_ranked.columns:
        print(f"[DEBUG] SubsetTag counts AFTER scoring:\n{df_ranked['SubsetTag'].value_counts().to_string()}")

    # =========================================================================
    # COMPUTE VALIDATION METRICS FOR TOP MODELS (if validation set provided)
    # =========================================================================
    if compute_validation and X_validation is not None and y_validation is not None:
        # Convert X to numpy if it's a DataFrame
        X_train_for_val = X.values if hasattr(X, 'values') else X
        X_val_for_val = X_validation if isinstance(X_validation, np.ndarray) else np.array(X_validation)
        y_val_for_val = y_validation if isinstance(y_validation, np.ndarray) else np.array(y_validation)

        # CRITICAL: Use encoded training labels (y_np) for consistency
        # y_np was encoded earlier if label_encoder exists, so model training
        # and validation must use the same encoding
        y_train_for_val = y_np  # Use the (possibly encoded) training labels

        # CRITICAL: Encode validation labels using the same encoder as training
        if label_encoder is not None:
            try:
                y_val_for_val = label_encoder.transform(y_val_for_val)
                print(f"[Validation] Encoded validation labels using training label encoder")
            except ValueError as e:
                print(f"[Warning] Could not encode validation labels: {e}")
                print(f"          Validation labels may contain classes not seen during training")

        # Get wavelengths for subsetting
        wavelengths_for_validation = X.columns.astype(float).values if hasattr(X, 'columns') else np.arange(X.shape[1])

        df_ranked = compute_validation_metrics_for_top_models(
            df_ranked,
            X_train_for_val,
            y_train_for_val,
            X_val_for_val,
            y_val_for_val,
            task_type,
            wavelengths_for_validation,
            top_n=validation_top_n,
            progress_callback=progress_callback
        )

    # Return results along with label_encoder (for classification with text labels)
    return df_ranked, label_encoder


def run_bayesian_search(X, y, task_type, models_to_test=None, preprocessing_methods=None,
                        n_trials=None, folds=5, excluded_count=0, validation_count=0,
                        total_samples_original=None, max_n_components=12, tier='standard',
                        imbalance_method=None, imbalance_params=None,
                        progress_callback=None,
                        enable_variable_subsets=True, variable_counts=None,
                        enable_region_subsets=False, n_top_regions=5,
                        variable_selection_methods=None,
                        # Baseline and smoothing parameters (same as run_grid_search)
                        baseline_method=None,
                        baseline_params=None,
                        smoothing=False,
                        smoothing_window=17,
                        smoothing_polyorder=2,
                        # Validation metrics parameters
                        X_validation=None,
                        y_validation=None,
                        compute_validation=False,
                        validation_top_n=100):
    """
    Run Bayesian hyperparameter optimization using Optuna.

    Uses Tree-structured Parzen Estimator (TPE) to find optimal hyperparameters
    in 30-50 trials instead of testing 5,832+ grid combinations.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (n_samples, n_features)
    y : pd.Series
        Target values
    task_type : str
        'regression' or 'classification'
    models_to_test : list of str, optional
        List of model names to optimize (e.g., ['XGBoost', 'LightGBM', 'MLP'])
        If None, optimizes all models in tier
    preprocessing_methods : list of dict, optional
        Preprocessing configurations to test
        If None, uses defaults: [{'name': 'snv', 'deriv': 2}, {'name': 'none', 'deriv': 0}]
    n_trials : int
        Number of Optuna trials per model (GUI default is 100)
    folds : int, default=5
        Number of CV folds
    excluded_count : int, default=0
        Number of excluded samples (for tracking)
    validation_count : int, default=0
        Number of validation samples (for tracking)
    total_samples_original : int, optional
        Original total sample count (for tracking)
    max_n_components : int, default=8
        Maximum PLS components (constrained by min(n_samples, n_features))
    tier : str, default='standard'
        Model tier: 'quick', 'standard', or 'comprehensive'
    imbalance_method : str, optional
        Imbalance handling method ('smote', 'rare_boost', 'class_weight', etc.)
    imbalance_params : dict, optional
        Parameters for imbalance method
    progress_callback : callable, optional
        Function to call with progress updates

    Returns
    -------
    df_ranked : pd.DataFrame
        Ranked results with best hyperparameters found for each model
    label_encoder : LabelEncoder or None
        Label encoder for classification with text labels

    Examples
    --------
    >>> # Optimize XGBoost and LightGBM with 30 trials each
    >>> results, _ = run_bayesian_search(
    ...     X, y,
    ...     task_type='regression',
    ...     models_to_test=['XGBoost', 'LightGBM'],
    ...     n_trials=30,
    ...     tier='standard'
    ... )

    Notes
    -----
    - Bayesian optimization finds optimal parameters 100x faster than grid search
    - Search spaces are continuous (e.g., learning_rate=0.127 instead of [0.05, 0.1, 0.2])
    - Uses existing DASP infrastructure (_run_single_config, preprocessing, CV)
    - Results are compatible with grid search results (same DataFrame format)
    - Does NOT replace grid search - runs as alternative method when selected in GUI
    """
    # Use fixed random state (ignore parameter - hardcoded throughout codebase)
    random_state = RANDOM_STATE

    from .bayesian_utils import create_optuna_study, create_objective_function, convert_optuna_result_to_dasp_format, ProgressCallback
    from .bayesian_config import get_bayesian_search_space
    from .models import build_model
    import optuna

    # Suppress Optuna logging (use DASP progress callback instead)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Validate n_trials - must be provided by caller (GUI controls this)
    if n_trials is None:
        raise ValueError("n_trials must be specified (GUI default is 100)")

    # Prepare data
    X_np = X.values
    y_np = y.values
    wavelengths = X.columns.values
    n_features = X_np.shape[1]
    n_samples = X_np.shape[0]

    # Handle categorical labels for classification
    label_encoder = None
    if task_type == "classification":
        if y_np.dtype == object or not np.issubdtype(y_np.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_original = y_np.copy()
            y_np = label_encoder.fit_transform(y_np)

            # Log label mapping
            label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            print(f"\n{'='*70}")
            print(f"BAYESIAN OPTIMIZATION - CATEGORICAL LABEL ENCODING")
            print(f"{'='*70}")
            print(f"Detected non-numeric classification labels.")
            print(f"Encoding mapping:")
            for label, code in sorted(label_mapping.items(), key=lambda x: x[1]):
                print(f"  '{label}' -> {code}")
            print(f"{'='*70}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # UPFRONT VALIDATION FOR CLASSIFICATION IMBALANCE METHODS
    # Validate configuration BEFORE starting Bayesian optimization
    # ═══════════════════════════════════════════════════════════════════════════
    if task_type == 'classification' and imbalance_method is not None:
        from .imbalance import validate_classification_config
        try:
            validate_classification_config(
                y=y_np,
                imbalance_method=imbalance_method,
                imbalance_params=imbalance_params,
                n_folds=folds
            )
            print(f"[OK] Imbalance configuration validated: {imbalance_method} with {folds}-fold CV")
        except ValueError as e:
            raise ValueError(f"Configuration Error (detected before optimization):\n\n{e}") from None

    # Determine binary classification status
    is_binary_classification = (task_type == "classification" and len(np.unique(y_np)) == 2)

    # Adjust max_n_components based on data constraints (same logic as run_search)
    # For small wavelength subsets, PLS n_components must be capped
    # Use TRAINING fold size (not test fold) since PLS is fit on training data
    min_train_samples = n_samples * (folds - 1) // folds

    if task_type == "regression":
        # Strict constraint for PLS regression: n_components <= min(n_samples_train, n_features)
        safe_max_components = min(max_n_components, min_train_samples, n_features)
    else:
        # More relaxed for PLS-DA classification
        safe_max_components = min(max_n_components, n_features)

    if safe_max_components < max_n_components:
        print(f"Note: Bayesian search reducing max PLS components from {max_n_components} to {safe_max_components} "
              f"(n_features={n_features}, min_train_size~{min_train_samples})")
        max_n_components = safe_max_components

    # Ensure at least 1 component (edge case with very small datasets)
    max_n_components = max(1, max_n_components)

    # Create results container
    df_results = create_results_dataframe(task_type)

    # GA preprocessing not currently supported in Bayesian search
    ga_preprocess = False

    # Default preprocessing methods
    # polyorder must be deriv + 1: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    if preprocessing_methods is None:
        preprocessing_methods = [
            {'name': 'snv', 'deriv': 2, 'window': 15, 'polyorder': 3, 'interference': None},
            {'name': 'snv', 'deriv': 1, 'window': 15, 'polyorder': 2, 'interference': None},
            {'name': 'snv', 'deriv': 0, 'window': 0, 'polyorder': 0, 'interference': None},
            {'name': 'none', 'deriv': 0, 'window': 0, 'polyorder': 0, 'interference': None},
        ]
    elif isinstance(preprocessing_methods, dict):
        # Convert GUI dictionary format {'raw': True, 'snv': True, ...} to list format
        # This handles the format passed from the GUI
        preprocess_configs = []

        # Define window sizes based on derivative order
        # Lower derivatives can use smaller windows; higher derivatives need larger windows
        # to avoid noise amplification
        def get_windows_for_deriv(deriv_order):
            if deriv_order >= 3:
                return [15, 23, 31, 41]  # Higher derivatives need larger windows
            else:
                return [7, 13, 21, 31]  # 1st/2nd derivatives can use smaller windows

        # Add raw if selected
        if preprocessing_methods.get('raw', False):
            preprocess_configs.append({
                'name': 'raw',
                'deriv': 0,
                'window': 0,
                'polyorder': 0,
                'interference': None,
                'baseline_method': baseline_method,
                'baseline_params': baseline_params,
                'smoothing': smoothing,
                'smoothing_window': smoothing_window,
                'smoothing_polyorder': smoothing_polyorder
            })

        # Add SNV if selected
        if preprocessing_methods.get('snv', False):
            preprocess_configs.append({
                'name': 'snv',
                'deriv': 0,
                'window': 0,
                'polyorder': 0,
                'interference': None,
                'baseline_method': baseline_method,
                'baseline_params': baseline_params,
                'smoothing': smoothing,
                'smoothing_window': smoothing_window,
                'smoothing_polyorder': smoothing_polyorder
            })

        # Add SG1 (1st derivative) if selected - test multiple window sizes
        if preprocessing_methods.get('sg1', False):
            for window in get_windows_for_deriv(1):
                preprocess_configs.append({
                    'name': 'snv',
                    'deriv': 1,
                    'window': window,
                    'polyorder': 2,
                    'interference': None,
                    'baseline_method': baseline_method,
                    'baseline_params': baseline_params,
                    'smoothing': smoothing,
                    'smoothing_window': smoothing_window,
                    'smoothing_polyorder': smoothing_polyorder
                })

        # Add SG2 (2nd derivative) if selected - test multiple window sizes
        if preprocessing_methods.get('sg2', False):
            for window in get_windows_for_deriv(2):
                preprocess_configs.append({
                    'name': 'snv',
                    'deriv': 2,
                    'window': window,
                    'polyorder': 3,  # polyorder = deriv + 1
                    'interference': None,
                    'baseline_method': baseline_method,
                    'baseline_params': baseline_params,
                    'smoothing': smoothing,
                    'smoothing_window': smoothing_window,
                    'smoothing_polyorder': smoothing_polyorder
                })

        # Add SG3 (3rd derivative) if selected - test multiple window sizes
        if preprocessing_methods.get('sg3', False):
            for window in get_windows_for_deriv(3):
                preprocess_configs.append({
                    'name': 'snv',
                    'deriv': 3,
                    'window': window,
                    'polyorder': 4,  # polyorder = deriv + 1
                    'interference': None,
                    'baseline_method': baseline_method,
                    'baseline_params': baseline_params,
                    'smoothing': smoothing,
                    'smoothing_window': smoothing_window,
                    'smoothing_polyorder': smoothing_polyorder
                })

        # Add SG4 (4th derivative) if selected - test multiple window sizes
        if preprocessing_methods.get('sg4', False):
            for window in get_windows_for_deriv(4):
                preprocess_configs.append({
                    'name': 'snv',
                    'deriv': 4,
                    'window': window,
                    'polyorder': 5,  # polyorder = deriv + 1
                    'interference': None,
                    'baseline_method': baseline_method,
                    'baseline_params': baseline_params,
                    'smoothing': smoothing,
                    'smoothing_window': smoothing_window,
                    'smoothing_polyorder': smoothing_polyorder
                })

        # Add deriv_snv if selected - test multiple window sizes
        if preprocessing_methods.get('deriv_snv', False):
            for window in get_windows_for_deriv(2):
                preprocess_configs.append({
                    'name': 'deriv_snv',
                    'deriv': 2,
                    'window': window,
                    'polyorder': 3,  # polyorder = deriv + 1
                    'interference': None
                })

        preprocessing_methods = preprocess_configs

    # Default models based on tier
    if models_to_test is None:
        # Use tier system to determine which models to test
        from .model_config import get_tier_models
        tier_config = get_tier_models(tier, task_type)
        models_to_test = [m for m, enabled in tier_config.items() if enabled]

    # Create CV splitter
    from sklearn.model_selection import StratifiedKFold, KFold
    if task_type == "classification":
        cv_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    # Track progress
    total_tasks = len(models_to_test) * len(preprocessing_methods)
    total_trials = total_tasks * n_trials  # For global progress tracking
    current_task = 0
    trials_completed = 0  # Track global trial count for elapsed time calculation

    print(f"\n{'='*70}")
    print(f"BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Task: {task_type}")
    print(f"Models: {models_to_test}")
    print(f"Preprocessing methods: {len(preprocessing_methods)}")
    print(f"Trials per model×preprocessing: {n_trials}")
    print(f"  → Total trials per model: {n_trials * len(preprocessing_methods)}")
    print(f"  → Total trials overall: {total_trials}")
    print(f"Total optimizations: {total_tasks} (models × preprocessing)")
    print(f"CV folds: {folds}")
    print(f"Tier: {tier}")
    print(f"{'='*70}\n")

    # Loop over models and preprocessing methods
    for model_name in models_to_test:
        for preprocess_cfg in preprocessing_methods:
            # ═══════════════════════════════════════════════════════════════════════════
            # GA PREPROCESSING: Skip incompatible preprocessing configs
            # When GA preprocessing is enabled, only use the config appropriate for this model group
            # ═══════════════════════════════════════════════════════════════════════════
            if ga_preprocess and 'ga_model_type' in preprocess_cfg:
                # Determine this model's group
                if model_name in PLS_MODELS:
                    required_ga_type = "pls"
                elif model_name in NEURAL_SVM_MODELS:
                    required_ga_type = "neural_svm"
                elif model_name in TREE_MODELS:
                    required_ga_type = "tree"
                elif model_name in NEURALBOOSTED_MODELS:
                    required_ga_type = "neuralboosted"
                else:
                    # Unknown model type, use pls by default
                    required_ga_type = "pls"

                # Skip if this preprocessing config doesn't match the model group
                if preprocess_cfg['ga_model_type'] != required_ga_type:
                    continue

            current_task += 1

            print(f"\n{'='*70}")
            print(f"Optimizing {model_name} [{current_task}/{total_tasks}]")
            print(f"Preprocessing: {preprocess_cfg['name']} (deriv={preprocess_cfg['deriv']})")
            print(f"{'='*70}")

            # ═══════════════════════════════════════════════════════════════════════════
            # CRITICAL FIX: Apply preprocessing BEFORE Bayesian optimization
            # This matches the grid search pattern (lines 689-705)
            # ═══════════════════════════════════════════════════════════════════════════

            # Check if this is a GA-optimized preprocessing config
            if 'ga_transform' in preprocess_cfg and preprocess_cfg['ga_transform'] is not None:
                # Use GA transform directly (it already includes all preprocessing)
                X_preprocessed = preprocess_cfg['ga_transform'](X_np)
            else:
                # Step 1: Build spectral preprocessing pipeline (NO imbalance yet)
                # Use base_name if available (for GA configs), otherwise use name
                preprocess_name = preprocess_cfg.get("base_name", preprocess_cfg["name"])
                prep_pipe_steps = build_preprocessing_pipeline(
                    preprocess_name,
                    preprocess_cfg["deriv"],
                    preprocess_cfg["window"],
                    preprocess_cfg["polyorder"],
                    imbalance_method=None,  # Imbalance will be added later inside CV folds
                    imbalance_params=None,
                    task_type=task_type,
                    interference=preprocess_cfg.get("interference"),  # Phase 3: interference removal
                    wavelengths=wavelengths,  # Phase 3: needed for interference removal
                    baseline_method=preprocess_cfg.get("baseline_method"),
                    baseline_params=preprocess_cfg.get("baseline_params"),
                    smoothing=preprocess_cfg.get("smoothing", False),
                    smoothing_window=preprocess_cfg.get("smoothing_window", 17),
                    smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2)
                )

                # Step 2: Apply preprocessing to full spectrum
                X_preprocessed = X_np.copy()
                if prep_pipe_steps:
                    prep_pipeline = Pipeline(prep_pipe_steps)
                    X_preprocessed = prep_pipeline.fit_transform(X_preprocessed, y_np)

            # Edge masking will be handled in _run_single_config() to avoid double masking
            wavelengths_for_model = wavelengths
            n_features_for_model = n_features

            # Update progress callback
            if progress_callback:
                progress_callback({
                    'stage': 'bayesian_optimization',
                    'message': f'Optimizing {model_name} ({preprocess_cfg["name"]}, deriv={preprocess_cfg["deriv"]})',
                    'current': current_task,
                    'total': total_tasks,
                })

            # Create Optuna study
            direction = 'minimize' if task_type == 'regression' else 'maximize'
            study = create_optuna_study(
                direction=direction,
                sampler='TPE',
                random_state=random_state,
                study_name=f"{model_name}_{preprocess_cfg['name']}_deriv{preprocess_cfg['deriv']}"
            )

            # Create objective function (pass preprocessed + edge-masked data)
            objective_fn = create_objective_function(
                model_name=model_name,
                X=X_preprocessed,  # Preprocessed + edge-masked data
                y=y_np,
                wavelengths=wavelengths_for_model,  # Edge-masked wavelengths
                preprocess_cfg=preprocess_cfg,
                cv_splitter=cv_splitter,
                task_type=task_type,
                is_binary_classification=is_binary_classification,
                run_single_config_fn=_run_single_config,  # Use existing infrastructure
                tier=tier,
                n_features=n_features_for_model,  # Edge-masked feature count
                max_n_components=max_n_components,
                enable_variable_subsets=enable_variable_subsets,
                variable_counts=variable_counts,
                variable_selection_methods=variable_selection_methods,
                enable_region_subsets=enable_region_subsets,
                n_top_regions=n_top_regions,
                excluded_count=excluded_count,
                validation_count=validation_count,
                total_samples_original=total_samples_original,
                folds=folds,
                imbalance_method=imbalance_method,
                imbalance_params=imbalance_params,
                progress_callback=progress_callback,
                n_trials=n_trials
            )

            # Run optimization
            try:
                # Create progress callback for per-trial updates
                optuna_progress_callback = None
                if progress_callback:
                    preprocess_name = preprocess_cfg['name']
                    if preprocess_cfg['deriv']:
                        preprocess_name += f"_d{preprocess_cfg['deriv']}"

                    optuna_progress_callback = ProgressCallback(
                        progress_callback=progress_callback,
                        model_name=model_name,
                        preprocess_name=preprocess_name,
                        n_trials=n_trials,
                        task_type=task_type,
                        global_offset=trials_completed,
                        global_total=total_trials
                    )

                study.optimize(
                    objective_fn,
                    n_trials=n_trials,
                    show_progress_bar=False,
                    callbacks=[optuna_progress_callback] if optuna_progress_callback else None
                )

                # Update global trial count for next model
                trials_completed += len(study.trials)

                # Convert Optuna results to DASP format
                # CRITICAL: Now returns a LIST of ALL configurations tested (full + subsets)
                results_list = convert_optuna_result_to_dasp_format(
                    study=study,
                    model_name=model_name,
                    preprocess_cfg=preprocess_cfg,
                    task_type=task_type,
                    wavelengths=wavelengths_for_model,  # Edge-masked wavelengths
                    n_vars=n_features_for_model,  # Edge-masked feature count
                    excluded_count=excluded_count,
                    validation_count=validation_count,
                    total_samples_original=total_samples_original,
                    folds=folds,
                    imbalance_method=imbalance_method
                )

                # Add ALL results to dataframe (not just one)
                for result in results_list:
                    df_results = pd.concat([df_results, pd.DataFrame([result])], ignore_index=True)

                # Print summary (find best overall result)
                print(f"[OK] Collected {len(results_list)} configurations from {len(study.trials)} trials")

                # Find and print best result
                best_result = study.best_trial
                if task_type == 'regression':
                    best_r2 = best_result.user_attrs.get('R2', np.nan)
                    print(f"  Best trial #{best_result.number}: RMSE={best_result.value:.4f}, R²={best_r2:.4f}")
                else:
                    best_auc = best_result.user_attrs.get('ROC_AUC', np.nan)
                    print(f"  Best trial #{best_result.number}: Accuracy={-best_result.value:.4f}, ROC_AUC={best_auc:.4f}")
                print(f"  Parameters: {best_result.params}")

            except Exception as e:
                print(f"[X] Optimization failed for {model_name}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Rank results
    print(f"\n{'='*70}")
    print(f"RANKING RESULTS")
    print(f"{'='*70}")

    from .scoring import compute_composite_score
    df_ranked = compute_composite_score(
        df_results,
        task_type=task_type,
        variable_penalty=0,  # Bayesian optimization doesn't penalize variables
        complexity_penalty=0  # Bayesian optimization doesn't penalize complexity
    )

    # Rename CompositeScore to Score for consistency with grid search
    if 'CompositeScore' in df_ranked.columns:
        df_ranked = df_ranked.rename(columns={'CompositeScore': 'Score'})

    print(f"\n[OK] Bayesian optimization complete!")
    print(f"  Total models optimized: {len(df_ranked)}")
    if len(df_ranked) > 0:
        best_model = df_ranked.iloc[0]
        if task_type == 'regression':
            print(f"  Best model: {best_model['Model']} (RMSE={best_model['RMSE']:.4f}, R²={best_model['R2']:.4f})")
        else:
            print(f"  Best model: {best_model['Model']} (Accuracy={best_model['Accuracy']:.4f}, ROC_AUC={best_model['ROC_AUC']:.4f})")

    print(f"{'='*70}\n")

    # =========================================================================
    # COMPUTE VALIDATION METRICS FOR TOP MODELS (if validation set provided)
    # =========================================================================
    if compute_validation and X_validation is not None and y_validation is not None:
        # Convert X to numpy if it's a DataFrame
        X_train_for_val = X.values if hasattr(X, 'values') else X
        X_val_for_val = X_validation if isinstance(X_validation, np.ndarray) else np.array(X_validation)
        y_val_for_val = y_validation if isinstance(y_validation, np.ndarray) else np.array(y_validation)

        # CRITICAL: Use encoded training labels (y_np) for consistency
        # y_np was encoded earlier if label_encoder exists, so model training
        # and validation must use the same encoding
        y_train_for_val = y_np  # Use the (possibly encoded) training labels

        # CRITICAL: Encode validation labels using the same encoder as training
        if label_encoder is not None:
            try:
                y_val_for_val = label_encoder.transform(y_val_for_val)
                print(f"[Validation] Encoded validation labels using training label encoder")
            except ValueError as e:
                print(f"[Warning] Could not encode validation labels: {e}")
                print(f"          Validation labels may contain classes not seen during training")

        # Get wavelengths for subsetting
        wavelengths_for_validation = X.columns.astype(float).values if hasattr(X, 'columns') else np.arange(X.shape[1])

        df_ranked = compute_validation_metrics_for_top_models(
            df_ranked,
            X_train_for_val,
            y_train_for_val,
            X_val_for_val,
            y_val_for_val,
            task_type,
            wavelengths_for_validation,
            top_n=validation_top_n,
            progress_callback=progress_callback
        )

    return df_ranked, label_encoder


def _run_single_fold(pipe, X, y, train_idx, test_idx, task_type, is_binary_classification,
                     use_sample_weight_for_classification=False):
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
    use_sample_weight_for_classification : bool
        If True, compute and apply sample_weight for classification models
        that don't support class_weight but do support sample_weight (e.g., Ridge)

    Returns
    -------
    metrics : dict
        Dictionary with fold metrics (includes y_test and y_pred for regional analysis)
    """
    # Clone pipeline to avoid thread-safety issues
    pipe_clone = clone(pipe)

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Track whether we used manual fitting (to avoid Pipeline not fitted warning)
    # These must be initialized BEFORE any sample weight handling blocks
    manual_fit_used = False
    fitted_steps = []  # Store fitted preprocessing steps for manual transform
    final_model = None  # Store final model for manual predict

    # Check if pipeline has sample weighting transformer
    # This handles regression weighting methods (rare_boost, binning, balanced)
    sample_weight_train = None
    if hasattr(pipe_clone, 'named_steps') and 'imbalance' in pipe_clone.named_steps:
        imbalance_step = pipe_clone.named_steps['imbalance']
        if hasattr(imbalance_step, 'sample_weight_'):
            # This is a weighting transformer (RegressionSampleWeighter)
            # Fit the pipeline first to compute weights on this fold's training data
            pipe_clone.fit(X_train, y_train)

            # Extract computed weights (they're for the training fold)
            sample_weight_train = imbalance_step.sample_weight_

            # Get transformed X by applying all transformers except the model
            X_train_transformed = X_train
            for step_name, step in pipe_clone.steps[:-1]:
                X_train_transformed = step.transform(X_train_transformed)

            # Refit ONLY the final model step with weights (if supported)
            final_model = pipe_clone.steps[-1][1]
            if _supports_sample_weight(final_model):
                final_model.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)
            else:
                # Model doesn't support sample_weight (e.g., PLS) - fit without it
                final_model.fit(X_train_transformed, y_train)
            # Note: manual_fit_used stays False here because pipe_clone.fit() was called,
            # so the pipeline is officially fitted and predict() will work without warning

    # Handle classification sample weights (for models like Ridge that support sample_weight but not class_weight)

    if sample_weight_train is None and use_sample_weight_for_classification and task_type == 'classification':
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight_train = compute_sample_weight('balanced', y_train)

        # Get final model from pipeline
        if hasattr(pipe_clone, 'steps'):
            manual_fit_used = True
            # Transform X through all steps except the model
            X_train_transformed = X_train
            for step_name, step in pipe_clone.steps[:-1]:
                if hasattr(step, 'fit_resample'):
                    # For imblearn resamplers, apply fit_resample
                    X_train_transformed, y_train_for_model = step.fit_resample(X_train_transformed, y_train)
                    # Recompute sample weights for resampled data
                    sample_weight_train = compute_sample_weight('balanced', y_train_for_model)
                    fitted_steps.append((step_name, step, 'resample'))
                elif hasattr(step, 'transform'):
                    step.fit(X_train_transformed, y_train)
                    X_train_transformed = step.transform(X_train_transformed)
                    fitted_steps.append((step_name, step, 'transform'))

            # Fit the final model with sample weights (if supported)
            final_model = pipe_clone.steps[-1][1]
            if _supports_sample_weight(final_model):
                final_model.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)
            else:
                final_model.fit(X_train_transformed, y_train)
        else:
            # No pipeline, just the model
            if _supports_sample_weight(pipe_clone):
                pipe_clone.fit(X_train, y_train, sample_weight=sample_weight_train)
            else:
                pipe_clone.fit(X_train, y_train)
        sample_weight_train = 'applied'  # Flag that we've already fit

    # Standard path: fit if not already done above
    if sample_weight_train is None:
        pipe_clone.fit(X_train, y_train)

    # Helper function to transform and predict when manual fitting was used
    def _manual_transform_predict(X_data):
        """Transform X through manually fitted steps and predict with final model."""
        X_transformed = X_data
        for step_name, step, step_type in fitted_steps:
            if step_type == 'transform' and hasattr(step, 'transform'):
                X_transformed = step.transform(X_transformed)
            # Skip resample steps for test data - they only apply to training
        return final_model.predict(X_transformed), X_transformed

    def _manual_transform_predict_proba(X_data):
        """Transform X through manually fitted steps and predict_proba with final model."""
        X_transformed = X_data
        for step_name, step, step_type in fitted_steps:
            if step_type == 'transform' and hasattr(step, 'transform'):
                X_transformed = step.transform(X_transformed)
        return final_model.predict_proba(X_transformed)

    if task_type == "regression":
        if manual_fit_used:
            y_pred, _ = _manual_transform_predict(X_test)
        else:
            y_pred = pipe_clone.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": rmse, "R2": r2, "y_test": y_test, "y_pred": y_pred}
    else:  # classification
        if manual_fit_used:
            y_pred, _ = _manual_transform_predict(X_test)
        else:
            y_pred = pipe_clone.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Use is_binary_classification flag (determined from full dataset) for consistent averaging
        # This avoids issues where a CV fold might have missing classes
        # Use 'macro' for multiclass to treat all classes equally (consistent with ROC_AUC)
        average_method = 'binary' if is_binary_classification else 'macro'

        # F1, Precision, Recall
        try:
            f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
        except Exception:
            f1 = np.nan
        try:
            precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
        except Exception:
            precision = np.nan
        try:
            recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
        except Exception:
            recall = np.nan

        # ROC AUC (requires at least 2 classes in test fold)
        n_classes_test = len(np.unique(y_test))
        if n_classes_test < 2:
            # Single class in this CV fold - ROC AUC undefined
            auc = np.nan
        else:
            try:
                if manual_fit_used:
                    y_proba = _manual_transform_predict_proba(X_test)
                    model_classes = final_model.classes_ if hasattr(final_model, 'classes_') else None
                else:
                    y_proba = pipe_clone.predict_proba(X_test)
                    model_classes = pipe_clone.classes_ if hasattr(pipe_clone, 'classes_') else None

                if is_binary_classification:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # Explicitly tell roc_auc_score the column order matches model's classes_
                    if model_classes is not None:
                        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro', labels=model_classes)
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            except Exception:
                auc = np.nan

        return {"Accuracy": acc, "ROC_AUC": auc, "F1": f1, "Precision": precision, "Recall": recall}


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
    skip_spectral_preprocessing=False,
    excluded_count=0,
    validation_count=0,
    total_samples_original=None,
    folds=5,
    imbalance_method=None,
    imbalance_params=None,
    full_vars_original=None,
    n_jobs_cv=1,
    wavelength_restriction_active=False,
):
    """
    Run a single model configuration with CV.

    Parameters
    ----------
    skip_preprocessing : bool, default=False
        If True, skip building preprocessing pipeline (data is already preprocessed).
        Used for derivative subsets where preprocessing was already applied.
    wavelength_restriction_active : bool, default=False
        If True, skip edge masking for importance calculation. Used when wavelengths
        are restricted to a subset of the spectrum (from middle, not edges).

    Returns
    -------
    dict
        Dictionary with results, including top important variables.
    """
    # Use fixed random state (ignore parameter - hardcoded throughout codebase)
    random_state = RANDOM_STATE

    # Apply subset if specified
    if subset_indices is not None:
        X = X[:, subset_indices]
        n_vars = len(subset_indices)
    else:
        n_vars = X.shape[1]

    # Cap n_components to actual feature count (for PLS with small wavelength subsets)
    # Clone model first to avoid affecting other iterations (model passed by reference)
    if hasattr(model, 'n_components') and model.n_components is not None and model.n_components > n_vars:
        model = clone(model)
        capped_n_components = max(1, n_vars - 1)
        model.set_params(n_components=capped_n_components)

    # Use original wavelength count if provided (for wavelength filtering case)
    # Otherwise use current wavelength array length
    full_vars = full_vars_original if full_vars_original is not None else len(wavelengths)

    # Build preprocessing pipeline (skip if data is already preprocessed)
    if skip_preprocessing:
        # Old behavior: skip everything (for backward compatibility)
        pipe_steps = []
    elif skip_spectral_preprocessing:
        # NEW: Skip spectral preprocessing but ADD imbalance handling
        # This is used when spectral preprocessing (SNV/deriv) was already done globally,
        # but we still need to add imbalance transformers inside CV folds
        pipe_steps = []
        if imbalance_method is not None and imbalance_method != 'class_weight':
            # Add ONLY imbalance handling (spectral preprocessing already done)
            from spectral_predict.imbalance import build_imbalance_transformer
            if imbalance_params is None:
                imbalance_params = {}
            imbalance_transformer = build_imbalance_transformer(
                method=imbalance_method,
                task_type=task_type,
                random_state=random_state,  # CRITICAL for reproducibility
                **imbalance_params
            )
            pipe_steps.append(("imbalance", imbalance_transformer))
    else:
        # Normal behavior: build full pipeline (spectral + imbalance)
        # Phase 3: Extract wavelengths for interference removal
        # Use wavelengths from parameter if available, otherwise try to extract from DataFrame columns
        wavelengths_for_interference = wavelengths if wavelengths is not None else (
            X.columns.astype(float).values if hasattr(X, 'columns') else None
        )

        # Use base_name if available (for GA configs), otherwise use name
        preprocess_name = preprocess_cfg.get("base_name", preprocess_cfg["name"])
        pipe_steps = build_preprocessing_pipeline(
            preprocess_name,
            preprocess_cfg["deriv"],
            preprocess_cfg["window"],
            preprocess_cfg["polyorder"],
            imbalance_method=imbalance_method,
            imbalance_params=imbalance_params,
            task_type=task_type,
            interference=preprocess_cfg.get("interference"),  # Phase 3
            wavelengths=wavelengths_for_interference,  # Phase 3
            baseline_method=preprocess_cfg.get("baseline_method"),
            baseline_params=preprocess_cfg.get("baseline_params"),
            smoothing=preprocess_cfg.get("smoothing", False),
            smoothing_window=preprocess_cfg.get("smoothing_window", 17),
            smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2)
        )

    # Handle class_weight for imbalanced classification
    # Only apply if user selected 'class_weight' method AND model supports it
    # Note: MLP doesn't support class_weight OR sample_weight - users should use SMOTE instead
    use_sample_weight_for_classification = False
    if imbalance_method == 'class_weight' and task_type == 'classification':
        if hasattr(model, 'class_weight'):
            try:
                model.set_params(class_weight='balanced')
            except Exception as e:
                import warnings
                warnings.warn(
                    f"{model_name} has class_weight attribute but set_params failed: {e}. "
                    f"Consider using SMOTE or other resampling method.",
                    UserWarning
                )
        else:
            # Check if model supports sample_weight in fit() (e.g., RidgeClassifier)
            import inspect
            model_fit_sig = inspect.signature(model.fit) if hasattr(model, 'fit') else None
            if model_fit_sig and 'sample_weight' in model_fit_sig.parameters:
                # Model supports sample_weight - we'll compute and apply during _run_single_fold
                use_sample_weight_for_classification = True
            else:
                # Model doesn't support class_weight OR sample_weight
                import warnings
                if model_name in ['MLP', 'MLPClassifier']:
                    warnings.warn(
                        f"{model_name} does not support class_weight or sample_weight. "
                        f"For imbalanced classification with MLP, use SMOTE or other resampling methods instead.",
                        UserWarning
                    )
                else:
                    warnings.warn(
                        f"{model_name} does not support class_weight. "
                        f"Consider using SMOTE or other resampling methods for imbalanced data.",
                        UserWarning
                    )

    # For PLS-DA, we need PLS + LogisticRegression
    if model_name == "PLS-DA":
        pipe_steps.append(("pls", model))
        # Apply class_weight to LogisticRegression if requested
        if imbalance_method == 'class_weight' and task_type == 'classification':
            pipe_steps.append(("lr", LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')))
        else:
            pipe_steps.append(("lr", LogisticRegression(max_iter=1000, random_state=random_state)))
    else:
        pipe_steps.append(("model", model))

    # Choose correct Pipeline class based on whether resampling is needed
    # Standard sklearn Pipeline doesn't support fit_resample() methods
    # Use imblearn Pipeline for resampling transformers (SMOTE, RegressionUndersampler, etc.)
    if pipe_steps:
        needs_resampling = _needs_resampling_pipeline(imbalance_method, task_type)

        if needs_resampling:
            pipe = ImbPipeline(pipe_steps)
        else:
            pipe = Pipeline(pipe_steps)
    else:
        pipe = model

    # Run CV (serial if n_jobs_cv=1 for reproducibility, parallel otherwise)
    if n_jobs_cv == 1:
        # Serial execution for reproducibility (deterministic fold ordering)
        cv_metrics = [
            _run_single_fold(
                pipe, X, y, train_idx, test_idx, task_type, is_binary_classification,
                use_sample_weight_for_classification
            )
            for train_idx, test_idx in cv_splitter.split(X, y)
        ]
    else:
        # Parallel execution for speed
        cv_metrics = Parallel(n_jobs=n_jobs_cv, backend='loky')(
            delayed(_run_single_fold)(
                pipe, X, y, train_idx, test_idx, task_type, is_binary_classification,
                use_sample_weight_for_classification
            )
            for train_idx, test_idx in cv_splitter.split(X, y)
        )

    # Print summary if imbalance handling was used
    if imbalance_method is not None:
        if imbalance_method == 'class_weight':
            print(f"  [OK] Imbalance handling: class_weight applied to model")
        else:
            print(f"  [OK] Imbalance handling: {imbalance_method} applied successfully")

    # Average metrics
    if task_type == "regression":
        mean_rmse = np.mean([m["RMSE"] for m in cv_metrics])
        mean_r2 = np.mean([m["R2"] for m in cv_metrics])

        # Compute regional performance (quartile-based) for consensus predictions
        # Collect all CV predictions and true values
        all_y_test = np.concatenate([m["y_test"] for m in cv_metrics])
        all_y_pred = np.concatenate([m["y_pred"] for m in cv_metrics])

        # Compute quartiles based on true values
        quartiles = np.percentile(all_y_test, [25, 50, 75])

        # Compute RMSE for each quartile region
        regional_rmse = {}
        for i, (lower, upper) in enumerate([
            (-np.inf, quartiles[0]),  # Q1
            (quartiles[0], quartiles[1]),  # Q2
            (quartiles[1], quartiles[2]),  # Q3
            (quartiles[2], np.inf)  # Q4
        ]):
            mask = (all_y_test >= lower) & (all_y_test < upper if i < 3 else all_y_test >= lower)
            if mask.sum() > 0:
                regional_rmse[f'Q{i+1}'] = np.sqrt(mean_squared_error(
                    all_y_test[mask], all_y_pred[mask]
                ))
            else:
                regional_rmse[f'Q{i+1}'] = np.nan
    else:
        mean_acc = np.mean([m["Accuracy"] for m in cv_metrics])
        mean_auc = np.mean([m["ROC_AUC"] for m in cv_metrics if not np.isnan(m["ROC_AUC"])])
        mean_f1 = np.mean([m["F1"] for m in cv_metrics if not np.isnan(m["F1"])])
        mean_precision = np.mean([m["Precision"] for m in cv_metrics if not np.isnan(m["Precision"])])
        mean_recall = np.mean([m["Recall"] for m in cv_metrics if not np.isnan(m["Recall"])])
        regional_rmse = None  # Not applicable for classification

    # Capture complete parameters for ALL models (not just those with feature importance)
    # This fixes R² reproducibility issue for Ridge, Lasso, ElasticNet, PLS, etc.
    # Also compute calibration metrics on training data
    cal_rmse = None
    cal_r2 = None
    cal_acc = None
    cal_auc = None
    cal_f1 = None
    cal_precision = None
    cal_recall = None

    try:
        # Refit the pipeline on full data to get final fitted parameters
        pipe.fit(X, y)

        # Get the fitted model from pipeline
        fitted_model = (
            pipe.named_steps["model"] if hasattr(pipe, "named_steps") else pipe
        )

        # Compute calibration metrics (training data performance)
        y_pred_cal = pipe.predict(X)

        if task_type == "regression":
            cal_rmse = np.sqrt(mean_squared_error(y, y_pred_cal))
            cal_r2 = r2_score(y, y_pred_cal)
        else:
            # Classification metrics
            cal_acc = accuracy_score(y, y_pred_cal)

            # Compute ROC AUC if probabilities available
            try:
                if hasattr(pipe, "predict_proba"):
                    y_pred_proba_cal = pipe.predict_proba(X)
                    n_classes = len(np.unique(y))
                    if n_classes == 2:
                        cal_auc = roc_auc_score(y, y_pred_proba_cal[:, 1])
                    else:
                        cal_auc = roc_auc_score(y, y_pred_proba_cal,
                                               multi_class='ovr', average='macro')
                else:
                    cal_auc = np.nan
            except Exception:
                cal_auc = np.nan

            # Compute F1, Precision, Recall
            try:
                cal_f1 = f1_score(y, y_pred_cal, average='weighted', zero_division=0)
                cal_precision = precision_score(y, y_pred_cal, average='weighted', zero_division=0)
                cal_recall = recall_score(y, y_pred_cal, average='weighted', zero_division=0)
            except Exception:
                cal_f1 = np.nan
                cal_precision = np.nan
                cal_recall = np.nan

        # Capture ALL parameters
        print(f"\n{'='*80}")
        print(f"DIAGNOSTIC - {model_name} Training (Results Tab)")
        print(f"{'='*80}")
        try:
            all_params = fitted_model.get_params()
            print(f"ALL {model_name} parameters after training:")
            for key in sorted(all_params.keys()):
                print(f"  {key}: {all_params[key]}")
            print(f"\nOld params dict (incomplete - only grid search params):")
            print(f"  {params}")

            # CRITICAL FIX: Replace params with complete parameter set
            # Filter out non-serializable parameters and convert numpy types
            filtered_params = {}
            for key, value in all_params.items():
                # Skip callables and complex objects
                if callable(value) or hasattr(value, '__dict__'):
                    continue

                # Convert value to Python-native type for reliable serialization
                try:
                    # Handle numpy scalar types (np.int64, np.float64, etc.)
                    if hasattr(value, 'item'):
                        value = value.item()

                    # Skip nan values - they can't be serialized with ast.literal_eval
                    # and XGBoost will use its default (nan) anyway
                    if isinstance(value, float) and np.isnan(value):
                        continue

                    # Test if value can be round-tripped through str() -> ast.literal_eval()
                    test_str = str({key: value})
                    import ast
                    ast.literal_eval(test_str)

                    # Value passed the test, include it
                    filtered_params[key] = value
                except:
                    # Skip values that can't be serialized/deserialized
                    continue

            params = filtered_params  # Replace params with complete set

            print(f"\nNew params dict (complete - ALL parameters):")
            print(f"  {params}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"ERROR capturing {model_name} params: {e}\n")
            print(f"Continuing with original params dict\n")

        # Store the fitted_model for feature importance calculation below
        fitted_model_for_importance = fitted_model

    except Exception as e:
        print(f"Warning: Could not fit model for parameter capture: {e}")
        fitted_model_for_importance = None

    # Extract LVs (for PLS models) - must be done before building result dict
    # Use int to avoid decimal display, None for non-PLS models
    n_comp = params.get("n_components")
    lvs = int(n_comp) if n_comp is not None else None

    # Format imbalance handling indicator for display
    if imbalance_method is None:
        imbalance_display = "—"
    elif imbalance_method == 'class_weight':
        imbalance_display = "class_weight"
    else:
        imbalance_display = imbalance_method

    # Build result dictionary AFTER capturing complete params
    # Use base_name for Preprocess column (for validation compatibility)
    # but store full display name for reference
    preprocess_display = preprocess_cfg["name"]
    preprocess_base = preprocess_cfg.get("base_name", preprocess_cfg["name"])
    result = {
        "Task": task_type,
        "Model": model_name,
        "Params": str(params),  # Now includes complete parameter set
        "Preprocess": preprocess_base,  # Use base name for pipeline building
        "PreprocessDisplay": preprocess_display,  # Full name for display
        "Deriv": preprocess_cfg["deriv"],
        "Window": preprocess_cfg["window"],
        "Poly": preprocess_cfg["polyorder"],
        "LVs": lvs,
        "n_vars": n_vars,
        "full_vars": full_vars,
        "SubsetTag": subset_tag,
        "Imbalance": imbalance_display,
    }

    # Add training configuration for tracking data state
    # This helps identify when Model Development tab uses different data
    result["training_config"] = {
        "folds": cv_splitter.n_splits if hasattr(cv_splitter, 'n_splits') else folds,
        "n_samples_used": len(X),  # Number of samples used for training (after filtering)
        "n_samples_total": total_samples_original if total_samples_original else len(X),
        "excluded_count": excluded_count,  # Number of excluded samples
        "validation_count": validation_count,  # Number of validation samples
        "n_features_used": X.shape[1],  # Number of features/wavelengths used
        "random_state": 42,  # CV random state (always 42 in this codebase)
    }

    # Store GA preprocessing genes if present (for Model Development reconstruction)
    if 'ga_genes' in preprocess_cfg and preprocess_cfg['ga_genes'] is not None:
        result["ga_genes"] = preprocess_cfg['ga_genes'].tolist()  # Serialize numpy array
        result["ga_model_type"] = preprocess_cfg.get("ga_model_type", "linear")
        result["ga_config"] = preprocess_cfg.get("ga_config", "")

    if task_type == "regression":
        # Calibration metrics (training data)
        result["RMSE"] = cal_rmse if cal_rmse is not None else np.nan
        result["R2"] = cal_r2 if cal_r2 is not None else np.nan
        # Cross-validation metrics (test fold averages)
        result["RMSEcv"] = mean_rmse
        result["R2cv"] = mean_r2
        # Add regional performance for consensus predictions
        result["regional_rmse"] = regional_rmse
        result["y_quartiles"] = quartiles.tolist()  # Save quartile thresholds
    else:
        # Calibration metrics (training data)
        result["Accuracy"] = cal_acc if cal_acc is not None else np.nan
        result["ROC_AUC"] = cal_auc if cal_auc is not None else np.nan
        result["F1"] = cal_f1 if cal_f1 is not None else np.nan
        result["Precision"] = cal_precision if cal_precision is not None else np.nan
        result["Recall"] = cal_recall if cal_recall is not None else np.nan
        # Cross-validation metrics (test fold averages)
        result["Accuracycv"] = mean_acc
        result["ROC_AUCcv"] = mean_auc
        result["F1cv"] = mean_f1
        result["Precisioncv"] = mean_precision
        result["Recallcv"] = mean_recall

    # Save all_vars for ALL models (including full spectrum)
    # This ensures Model Development can reconstruct the exact wavelengths used
    # CRITICAL: For full models, 'wavelengths' is already filtered by wl_min/wl_max
    # so we must save it to allow exact replication
    if subset_tag != "full" and subset_indices is not None:
        # Subset model: save only the subset wavelengths
        subset_wavelengths = wavelengths[subset_indices]
        all_vars_str = ','.join([f"{w:.0f}" for w in subset_wavelengths])
        result['all_vars'] = all_vars_str
    else:
        # Full model: save ALL wavelengths used (may be filtered by wl_min/wl_max)
        all_vars_str = ','.join([f"{w:.0f}" for w in wavelengths])
        result['all_vars'] = all_vars_str

    # Continue with feature importance extraction if model was already fitted above
    if supports_feature_importance(model_name) and fitted_model_for_importance is not None:
        try:
            # Use the already-fitted model from parameter capture above
            fitted_model = fitted_model_for_importance

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

            # Apply edge masking for Savitzky-Golay derivatives (consistent with variable selection)
            # SKIP when wavelength restriction is active - restricted wavelengths
            # are from middle of spectrum, not SG boundary edges
            if not wavelength_restriction_active:
                importances = _apply_edge_mask(importances, preprocess_cfg)

            # Get top N features for display purposes (always top 30)
            n_to_select = min(top_n_vars, len(importances))
            # Use stable sort to ensure deterministic feature ordering when importances are tied
            top_indices = np.argsort(importances, kind='stable')[-n_to_select:][::-1]

            # Map back to original wavelengths
            if subset_indices is not None:
                # We're working with a subset, map indices back to original wavelengths
                original_wavelengths = wavelengths[subset_indices]
                top_wavelengths = original_wavelengths[top_indices]
            else:
                # Full spectrum
                top_wavelengths = wavelengths[top_indices]

            # Format as comma-separated string
            top_vars_str = ','.join([f"{w:.0f}" for w in top_wavelengths])
            result['top_vars'] = top_vars_str

        except Exception as e:
            # If anything fails, just mark as N/A
            result['top_vars'] = 'N/A'
            # Keep all_vars that we already set above
    else:
        # For models that don't support importance extraction
        result['top_vars'] = 'N/A'
        # Keep all_vars that we already set above

    return result
