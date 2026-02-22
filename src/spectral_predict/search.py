"""Model search with cross-validation and subset selection."""

import os
import sys
import inspect
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score, classification_report,
    mean_absolute_error, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from joblib import Parallel, delayed

from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocess import build_preprocessing_pipeline
from .models import get_model_grids, get_feature_importances
from .scoring import create_results_dataframe, add_result, compute_specificity
from .regions import create_region_subsets, format_region_report
from .variable_selection import (
    spa_selection, uve_selection, uve_spa_selection,
    ipls_selection, ipls_forward, ipls_backward, cars_selection,
    get_uve_threshold, uve_cars_selection, uve_cars_spa_selection,
    fipls_spa_selection, fipls_cars_selection,
    mc_sipls, mwpls
)
from .wavelength_selection import vcpa_iriv
from .ga_pls import ga_pls_selection
from .ga_lightgbm import ga_lightgbm_selection
from .model_registry import supports_subset_analysis, supports_feature_importance
from .constants import RANDOM_STATE

# Import early stopping CV utilities
from .cv_utils import is_boosting_model, _fit_with_early_stopping

from .ga_preprocessing import optimize_preprocessing, PREPROC_TYPES, WINDOW_SIZES
from .preprocessing_discovery import discover_preprocessing, IMPORTANCE_METHODS

# NSGA-II import
from .nsga2_search import run_nsga2_search, convert_nsga2_to_v1_format

logger = logging.getLogger(__name__)

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

# Scale-sensitive models: These use gradient descent or kernel methods
# that are sensitive to feature scale and benefit from StandardScaler
SCALE_SENSITIVE_MODELS = {'SVC', 'SVR', 'MLP', 'NeuralBoosted', 'Ridge', 'Lasso', 'ElasticNet'}

# Models that are slower with parallel CV due to threading conflicts or low overhead
# SVM: internal multi-threading conflicts with sklearn's CV parallelization
# PLS/PLS-DA: so fast that joblib overhead dominates (0.08s serial vs 0.29s parallel)
# Ridge/Lasso/ElasticNet: linear solve is ~5ms, joblib spawn overhead is ~1s on Windows
MODELS_PREFER_SERIAL_CV = {'SVM', 'PLS', 'PLS-DA', 'Ridge', 'Lasso', 'ElasticNet'}

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
        resampling_methods = ['undersample', 'oversample', 'smogn', 'smotetomek']
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
    # This matches how PLS-DA is built during search (search.py:3420-3443)
    if task_type == 'classification' and model_name == 'PLS-DA':
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        # Extract LogisticRegression parameters from config (prefixed with lr_)
        lr_C = model_kwargs.get('lr_C', 1.0)
        lr_solver = model_kwargs.get('lr_solver', 'lbfgs')
        lr_max_iter = model_kwargs.get('lr_max_iter', 1000)

        pls_lr_pipeline = Pipeline([
            ('pls', model),
            ('scaler', StandardScaler()),  # Scale PLS scores for LogisticRegression
            ('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=42))
        ])
        return pls_lr_pipeline

    # For scale-sensitive models (SVC/SVR, MLP, NeuralBoosted), add StandardScaler
    # These use gradient descent or kernel methods that are sensitive to feature scale
    if model_name in SCALE_SENSITIVE_MODELS:
        from sklearn.pipeline import Pipeline
        scaled_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        return scaled_pipeline

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
        Results with RMSEP, R2pred (or val_Accuracy) columns added
    """
    # Initialize columns
    if task_type == 'regression':
        df_results['RMSEP'] = np.nan
        df_results['R2pred'] = np.nan
    else:
        df_results['val_Accuracy'] = np.nan
        df_results['val_ROC_AUC'] = np.nan
        df_results['val_F1'] = np.nan
        df_results['val_Precision'] = np.nan
        df_results['val_Recall'] = np.nan

    # Get top N indices by CompositeScore (lower is better)
    n_to_process = min(top_n, len(df_results))
    if 'CompositeScore' in df_results.columns:
        # Ensure CompositeScore is numeric (may be object dtype from CSV)
        df_results['CompositeScore'] = pd.to_numeric(df_results['CompositeScore'], errors='coerce')
        top_indices = df_results.nsmallest(n_to_process, 'CompositeScore').index
    else:
        # Fallback to first n rows
        top_indices = df_results.head(n_to_process).index

    print(f"\n[Validation] Computing validation metrics for top {n_to_process} models...")

    # For classification, check class distribution in validation set and warn if problematic
    if task_type == 'classification':
        val_class_counts = pd.Series(y_val).value_counts()
        train_class_counts = pd.Series(y_train).value_counts()
        classes_in_train = set(train_class_counts.index)
        classes_in_val = set(val_class_counts.index)
        missing_classes = classes_in_train - classes_in_val

        if missing_classes:
            print(f"\n[Validation Warning] {len(missing_classes)} class(es) not represented in validation set: {missing_classes}")
            print(f"  Training class distribution: {dict(train_class_counts)}")
            print(f"  Validation class distribution: {dict(val_class_counts)}")
            print(f"  Some metrics (ROC AUC) will be NaN. Consider using more validation samples or fewer classes.\n")

        # Check for critically small class samples in validation
        min_samples_per_class = val_class_counts.min() if len(val_class_counts) > 0 else 0
        if min_samples_per_class < 2:
            print(f"[Validation Warning] Some classes have <2 samples in validation. Metrics may be unreliable.\n")

    # Cache preprocessed data by preprocessing config to avoid redundant computation
    preprocess_cache = {}

    for i, idx in enumerate(top_indices):
        row = df_results.loc[idx]

        try:
            # === STEP 1: Get preprocessing config ===
            # Use PreprocessBase (clean pipeline name) if available, fall back to Preprocess
            preprocess_name = row.get('PreprocessBase', row.get('Preprocess', 'raw'))
            # Strip baseline prefix (e.g., "als+snv" → "snv") as fallback
            baseline_method = None
            if '+' in str(preprocess_name):
                baseline_method, preprocess_name = str(preprocess_name).split('+', 1)
            deriv = row.get('Deriv', 0)
            window = row.get('Window', None)
            poly = row.get('Poly', None)

            # Check for GA preprocessing genes (needs reconstruction)
            ga_genes_str = row.get('ga_genes', None)
            use_ga_transform = False
            ga_transform = None
            ga_genes = None

            # Handle ga_genes_str being None, empty string, NaN scalar, list, or array
            ga_genes_is_valid = False
            if ga_genes_str is not None:
                if isinstance(ga_genes_str, (list, np.ndarray)):
                    ga_genes_is_valid = len(ga_genes_str) > 0
                elif isinstance(ga_genes_str, str):
                    ga_genes_is_valid = ga_genes_str != ''
                else:
                    try:
                        ga_genes_is_valid = not pd.isna(ga_genes_str)
                    except (ValueError, TypeError):
                        ga_genes_is_valid = True

            if ga_genes_is_valid:
                try:
                    # Parse genes from string (stored as list representation)
                    import ast
                    if isinstance(ga_genes_str, str):
                        ga_genes = np.array(ast.literal_eval(ga_genes_str))
                    else:
                        ga_genes = np.array(ga_genes_str)

                    # Import GA reconstruction function
                    from spectral_predict.ga_preprocessing import chromosome_to_transform

                    # Reconstruct transform from genes
                    _, ga_transform = chromosome_to_transform(ga_genes)
                    use_ga_transform = True
                except Exception as e:
                    genes_preview = str(ga_genes_str)[:100] if isinstance(ga_genes_str, str) else str(ga_genes_str)
                    print(f"  [Warning] Could not reconstruct GA transform: {e}")
                    print(f"            GA genes data: {genes_preview}")
                    use_ga_transform = False

            # Convert to proper types (only needed if not using GA transform)
            if not use_ga_transform:
                deriv = int(deriv) if deriv and not pd.isna(deriv) and deriv > 0 else None
                window = int(window) if window and not pd.isna(window) and window > 0 else None
                poly = int(poly) if poly and not pd.isna(poly) and poly > 0 else None

            # Create cache key
            if use_ga_transform:
                # GA preprocessing: cache by genes hash
                cache_key = ('ga', tuple(ga_genes))
            else:
                cache_key = (preprocess_name, deriv, window, poly, baseline_method)

            # === STEP 2: Preprocess FULL spectrum (matching search.py and Model Dev) ===
            if cache_key in preprocess_cache:
                X_train_preprocessed, X_val_preprocessed = preprocess_cache[cache_key]
            else:
                if use_ga_transform:
                    # Apply GA transform directly
                    X_train_preprocessed = ga_transform(X_train)
                    X_val_preprocessed = ga_transform(X_val)
                else:
                    # Build standard preprocessing pipeline
                    prep_steps = build_preprocessing_pipeline(
                        preprocess_name,
                        deriv=deriv,
                        window=window,
                        polyorder=poly,
                        baseline_method=baseline_method,
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

            # === STEP 3: Parse wavelength selection ===
            # Note: smart_selected_wavelengths is metadata only - smart preprocessing
            # does NOT subset wavelengths during training (see preprocessing_discovery.py
            # lines 580-584), so we don't subset during validation either.
            # Only all_vars (from variable selection methods like UVE, SPA, iPLS, CARS,
            # GA-PLS) actually subsets wavelengths during training.
            col_indices = None

            # Check for variable selection wavelengths (all_vars stores wavelength VALUES)
            all_vars_str = row.get('all_vars', 'N/A')
            if all_vars_str != 'N/A' and all_vars_str and isinstance(all_vars_str, str):
                # Parse wavelengths from all_vars (e.g., "1520.0, 1540.0, 1560.0, ...")
                try:
                    model_wavelengths = [float(w.strip()) for w in all_vars_str.split(',') if w.strip()]
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
                        col_indices = None
                except Exception as e:
                    print(f"  [Warning] Could not parse all_vars for model {i+1}: {e}")
                    col_indices = None

            # Subset AFTER preprocessing (matching Model Dev behavior)
            if col_indices is not None and len(col_indices) > 0:
                # Validate indices are within bounds
                max_idx = X_train_preprocessed.shape[1] - 1
                valid_indices = [idx for idx in col_indices if 0 <= idx <= max_idx]
                if len(valid_indices) != len(col_indices):
                    print(f"  [Warning] {len(col_indices) - len(valid_indices)} indices out of bounds for model {i+1}")
                if not valid_indices:
                    print(f"  [Warning] No valid indices for model {i+1}, skipping")
                    continue
                # Subset the PREPROCESSED data to selected columns
                X_train_final = X_train_preprocessed[:, valid_indices]
                X_val_final = X_val_preprocessed[:, valid_indices]
            else:
                # Full spectrum model - use all preprocessed data
                X_train_final = X_train_preprocessed
                X_val_final = X_val_preprocessed

            # === STEP 4: Rebuild model and fit ===
            model = _rebuild_model_from_row(row, task_type)

            # Safety check: Skip if n_components > n_features (can happen with variable selection)
            if hasattr(model, 'n_components') and model.n_components > X_train_final.shape[1]:
                print(f"  [Warning] Skipping model {i+1}: n_components ({model.n_components}) > n_features ({X_train_final.shape[1]})")
                continue

            # Fit on training data
            model.fit(X_train_final, y_train)

            # Predict on validation data
            y_pred = model.predict(X_val_final)
            y_pred = np.ravel(y_pred)  # Ensure 1D for metrics

            # === STEP 5: Calculate metrics ===
            if task_type == 'regression':
                rmsep = np.sqrt(mean_squared_error(y_val, y_pred))
                r2pred = r2_score(y_val, y_pred)
                df_results.loc[idx, 'RMSEP'] = rmsep
                df_results.loc[idx, 'R2pred'] = r2pred
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
                        # ROC AUC undefined with only one class
                        # Only log for first model to avoid spam
                        if i == 0:
                            print(f"  [Info] ROC AUC skipped - validation set has only 1 class (need at least 2)")
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
    if task_type == 'regression' and 'RMSEP' in cols and 'R2cv' in cols:
        # Move RMSEP and R2pred after R2cv
        cols.remove('RMSEP')
        cols.remove('R2pred')
        r2cv_idx = cols.index('R2cv')
        cols.insert(r2cv_idx + 1, 'RMSEP')
        cols.insert(r2cv_idx + 2, 'R2pred')
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
               total_samples_original=None, variable_penalty=0, gap_penalty=0,
               max_n_components=10, max_iter=500, models_to_test=None, preprocessing_methods=None,
               interference_settings=None,
               window_sizes=None, n_estimators_list=None, learning_rates=None,
               neuralboosted_hidden_sizes=None, neuralboosted_activations=None,
               pls_max_iter_list=None, pls_tol_list=None,
               plsda_lr_C_list=None, plsda_lr_solver_list=None, plsda_lr_max_iter_list=None,
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
               enable_region_subsets=True, n_top_regions=10,
               region_test_all_individual=False, region_test_pairwise=False,
               progress_callback=None,
               variable_selection_methods=None, apply_uve_prefilter=False,
               uve_cutoff_multiplier=1.0, uve_n_components=None,
               spa_n_random_starts=10, ipls_n_intervals=20,
               ipls_max_combine=5, ipls_subset_limit="Top 10",
               sipls_n_combinations=500,
               mwpls_window_sizes=None, mwpls_step_size=None,
               tier='standard', enabled_models=None,
               analysis_wl_min=None, analysis_wl_max=None,
               analysis_wl_regions=None,  # List of (min, max) tuples for multi-region support
               imbalance_method=None, imbalance_params=None, enable_class_weight=False,
               ga_preprocess=False,
               ga_preprocess_method='exhaustive',
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
               validation_top_n=100,
               # Early stopping for boosting models
               early_stopping_rounds=40):
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
    variable_penalty : int (0-10), default=0
        Penalty for using many variables (0=ignore, 10=strong penalty)
    gap_penalty : int (0-10), default=0
        Penalty for calibration-CV gap (0=ignore, 10=strong penalty)
    max_n_components : int, default=10
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

    # Use all cores for parallel execution (will be overridden per-model for CatBoost/SVM)
    # In frozen (PyInstaller) apps, force n_jobs=1 to prevent spawning new instances
    is_frozen = getattr(sys, 'frozen', False) or '__compiled__' in dir()
    n_jobs_default = 1 if is_frozen else -1

    # Drop rows where y is NaN (safety net for data with empty rows)
    nan_mask = y.isna()
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(f"Warning: Dropping {n_dropped} sample(s) with NaN target values before analysis.")
        X = X[~nan_mask]
        y = y[~nan_mask]

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
    implemented_methods = ['importance', 'spa', 'uve', 'uve_spa', 'ipls', 'ipls_forward', 'ipls_backward', 'mc_sipls', 'mwpls', 'cars', 'cars-aware', 'cars-tree', 'vcpa-iriv', 'ga', 'uve_cars', 'uve_cars_tree', 'uve_cars_spa', 'fipls_spa', 'fipls_cars']
    selected_methods = [m for m in variable_selection_methods if m in implemented_methods]

    # If UVE-hybrid variant is selected alongside base method, drop the base (hybrid subsumes it)
    if 'uve_cars' in selected_methods and 'cars' in selected_methods:
        selected_methods.remove('cars')
        print("Info: Removed 'cars' — 'uve_cars' includes CARS with UVE pre-filtering")
    if 'uve_cars_tree' in selected_methods and 'cars-tree' in selected_methods:
        selected_methods.remove('cars-tree')
        print("Info: Removed 'cars-tree' — 'uve_cars_tree' includes CARS-Tree with UVE pre-filtering")

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
                                   plsda_lr_C_list=plsda_lr_C_list, plsda_lr_solver_list=plsda_lr_solver_list,
                                   plsda_lr_max_iter_list=plsda_lr_max_iter_list,
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
                                   tier=tier, enabled_models=enabled_models, n_jobs=n_jobs_default)

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

                display_name = pipeline_name
                model_name = cfg.get('model_name')

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
        # Use model_grids.keys() since it's already filtered by enabled_models or models_to_test
        models_for_ga = list(model_grids.keys())
        print(f"Running {ga_preprocess_method.upper()} optimization per-model with actual hyperparameters...")
        print(f"Models selected: {models_for_ga}")
        print(f"")

        # Storage for GA results (one per model)
        ga_results = {}

        # Run GA optimization for each selected model
        # Uses ACTUAL model evaluation (via model_config) for accurate preprocessing optimization
        # Falls back to proxy fitness models if actual model fails
        for model_name in models_for_ga:
            # Send progress update to GUI before starting this model
            if progress_callback:
                model_idx = models_for_ga.index(model_name) + 1
                progress_callback({
                    'algorithm': 'preprocessing_optimization',
                    'current': 0,
                    'total': len(models_for_ga),
                    'message': f"Optimizing preprocessing for {model_name} ({model_idx}/{len(models_for_ga)})..."
                })

            print(f"Optimizing preprocessing for {model_name}...")

            # Determine which proxy fitness model to use as fallback
            if model_name.lower() in ['pls', 'pls-da', 'ridge', 'lasso', 'elasticnet']:
                fitness_model = 'pls'
            elif model_name.lower() in ['lightgbm', 'xgboost', 'catboost', 'randomforest']:
                fitness_model = 'lightgbm'
            elif model_name.lower() in ['mlp', 'svr', 'svc']:
                fitness_model = 'mlp'
            elif model_name.lower() == 'neuralboosted':
                fitness_model = 'neuralboosted'
            else:
                fitness_model = 'pls'  # Default

            # Get first hyperparameter set for this model (for actual model evaluation)
            # model_grids is dict mapping model_name -> list of (model_instance, params_dict)
            first_params = {}
            if model_name in model_grids and model_grids[model_name]:
                # Extract params from first config tuple: (model_instance, params_dict)
                first_params = model_grids[model_name][0][1] if len(model_grids[model_name][0]) > 1 else {}

            # Build model_config for actual model evaluation
            model_config = {
                'name': model_name,
                'params': first_params
            }

            # Run GA/Exhaustive optimization with actual model evaluation
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
                fitness_model=fitness_model,  # Fallback if model_config fails
                top_n=5,  # Return top 5 preprocessing configs
                n_jobs=-1 if ga_preprocess_method == 'exhaustive' else 1,
                model_config=model_config  # Use actual model for fitness evaluation
            )

            ga_results[model_name] = ga_result
            print(f"  {model_name} optimization complete!")
            print(f"  Best config: {ga_result['best_config']}")
            if task_type == 'classification':
                # For classification, fitness is accuracy (positive, higher = better)
                best_fitness = ga_result['configs'][0]['fitness'] if ga_result.get('configs') else 0
                print(f"  Best Accuracy: {best_fitness:.4f}")
            else:
                # For regression, best_rmsecv is already the RMSECV value
                print(f"  Best RMSECV: {ga_result['best_rmsecv']:.4f}")
            print(f"  Returning top {len(ga_result.get('configs', []))} configs\n")

            # Send completion update to GUI after this model finishes
            if progress_callback:
                model_idx = models_for_ga.index(model_name) + 1
                best_score = ga_result['configs'][0]['fitness'] if ga_result.get('configs') else 0
                if task_type == 'classification':
                    score_str = f"Best Accuracy: {best_score:.4f}"
                else:
                    score_str = f"Best RMSECV: {ga_result['best_rmsecv']:.4f}"
                progress_callback({
                    'algorithm': 'preprocessing_optimization',
                    'current': model_idx,
                    'total': len(models_for_ga),
                    'message': f"  ✓ {model_name} preprocessing complete - {score_str}"
                })

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
                # Clean display name: strip derivative order
                if base_name in ('raw', 'snv'):
                    clean_name = base_name
                elif base_name.startswith('snv_deriv'):
                    clean_name = 'snv_deriv'
                elif base_name.endswith('_snv'):
                    clean_name = 'deriv_snv'
                elif base_name.startswith('deriv'):
                    clean_name = 'deriv'
                else:
                    clean_name = base_name
                preprocess_configs.append({
                    "name": clean_name,
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
        print(f"Breakdown: {len(models_for_ga)} models × up to 5 configs each")
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

    # --- Baseline toggle: when enabled, test both WITH and WITHOUT baseline ---
    if baseline_method is not None and preprocess_configs:
        configs_without = []
        configs_with = []
        for cfg in preprocess_configs:
            # Without baseline
            cfg_no = dict(cfg)
            cfg_no["baseline_method"] = None
            cfg_no["baseline_params"] = None
            configs_without.append(cfg_no)
            # With baseline
            cfg_bl = dict(cfg)
            cfg_bl["base_name"] = cfg.get("base_name", cfg["name"])
            cfg_bl["name"] = f"{baseline_method}+{cfg['name']}"
            configs_with.append(cfg_bl)
        preprocess_configs = configs_without + configs_with

    # --- Smoothing toggle: when enabled, test both WITH and WITHOUT smoothing ---
    if smoothing and preprocess_configs:
        configs_without_smooth = []
        configs_with_smooth = []
        for cfg in preprocess_configs:
            # Without smoothing
            cfg_no = dict(cfg)
            cfg_no["smoothing"] = False
            configs_without_smooth.append(cfg_no)
            # With smoothing
            cfg_sm = dict(cfg)
            cfg_sm["base_name"] = cfg.get("base_name", cfg["name"])
            name = cfg["name"]
            if "+" in name:
                parts = name.split("+", 1)
                cfg_sm["name"] = f"{parts[0]}+sg0+{parts[1]}"
            else:
                cfg_sm["name"] = f"sg0+{name}"
            configs_with_smooth.append(cfg_sm)
        preprocess_configs = configs_without_smooth + configs_with_smooth

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
                    n_top_regions=n_top_regions,
                    test_all_individual=region_test_all_individual,
                    test_pairwise=region_test_pairwise
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
            # When GA preprocessing is enabled, only use the config optimized for this specific model
            # Each model gets its own set of top-N preprocessing configs from GA optimization
            # ═══════════════════════════════════════════════════════════════════════════
            if ga_preprocess and 'ga_model_type' in preprocess_cfg:
                # Skip if this preprocessing config was optimized for a different model
                # ga_model_type stores the actual model name (e.g., "LightGBM", "PLS")
                if preprocess_cfg['ga_model_type'] != model_name:
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

                # Add best model so far to progress (show CV metrics for consistency with ranking)
                best_info = ""
                if best_model_so_far is not None:
                    if task_type == "regression":
                        best_info = f" | Best CV: R²cv={best_model_so_far['R2cv']:.3f}, RMSEcv={best_model_so_far['RMSEcv']:.3f}"
                    else:
                        best_info = f" | Best CV: AUCcv={best_model_so_far.get('ROC_AUCcv', 0):.3f}"

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
                    n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                    wavelength_restriction_active=wavelength_restriction_active,
                    early_stopping_rounds=early_stopping_rounds,
                )
                if result is None:
                    print(f"  [WARNING] Full model invalid for {model_name}, skipping")
                    continue
                df_results = add_result(df_results, result)

                # Show full model result (CV metrics for consistency)
                if task_type == "regression":
                    print(f"     Full model: R²cv={result['R2cv']:.3f}, RMSEcv={result['RMSEcv']:.3f}")
                else:
                    print(f"     Full model: AUCcv={result.get('ROC_AUCcv', 0):.3f}, Acccv={result.get('Accuracycv', 0):.3f}")

                # Update best model tracker (use CV metrics for consistency with ranking)
                if best_model_so_far is None:
                    best_model_so_far = result
                else:
                    if task_type == "regression":
                        if result["RMSEcv"] < best_model_so_far["RMSEcv"]:
                            best_model_so_far = result
                    else:  # classification
                        if result.get("ROC_AUCcv", 0) > best_model_so_far.get("ROC_AUCcv", 0):
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

                            # ===== Subset-returning methods (iPLS, MC-siPLS, MWPLS) =====
                            if varsel_method in ('ipls_forward', 'ipls_backward', 'mc_sipls', 'mwpls'):
                                print(f"  -> Running {varsel_method}...")

                                # Call appropriate function
                                if varsel_method == 'ipls_forward':
                                    subsets = ipls_forward(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        max_combine=ipls_max_combine,
                                        cv_folds=folds,
                                        random_state=random_state
                                    )
                                elif varsel_method == 'ipls_backward':
                                    subsets = ipls_backward(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        cv_folds=folds,
                                        random_state=random_state
                                    )
                                elif varsel_method == 'mc_sipls':
                                    subsets = mc_sipls(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        n_combinations=sipls_n_combinations,
                                        max_combine=ipls_max_combine,
                                        cv_folds=folds,
                                        random_state=random_state
                                    )
                                elif varsel_method == 'mwpls':
                                    subsets = mwpls(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        window_sizes=mwpls_window_sizes,
                                        step_size=mwpls_step_size,
                                        cv_folds=folds
                                    )

                                if subsets is None or len(subsets) == 0:
                                    print(f"  -> {varsel_method} returned no subsets, skipping")
                                    continue

                                # Sort by rmsecv (best first) and apply limit
                                subsets_sorted = sorted(subsets, key=lambda s: s.get('rmsecv', float('inf')))

                                # Apply subset limit from dropdown
                                if ipls_subset_limit == "Top 5":
                                    subsets_to_test = subsets_sorted[:5]
                                elif ipls_subset_limit == "Top 10":
                                    subsets_to_test = subsets_sorted[:10]
                                elif ipls_subset_limit == "Top 20":
                                    subsets_to_test = subsets_sorted[:20]
                                else:  # "All"
                                    subsets_to_test = subsets_sorted

                                print(f"  -> Testing {len(subsets_to_test)} of {len(subsets)} subsets...")

                                # Test each subset
                                for subset_dict in subsets_to_test:
                                    if controller and not controller.check_and_wait():
                                        break

                                    subset_indices = subset_dict['indices']
                                    subset_tag = subset_dict['tag']

                                    # Use existing _run_single_config (same as top-N path)
                                    if preprocess_cfg["deriv"] is not None:
                                        subset_result = _run_single_config(
                                            X_transformed_varsel, y_np, wavelengths_varsel,
                                            model, model_name, params, preprocess_cfg,
                                            cv_splitter, task_type, is_binary_classification,
                                            subset_indices=subset_indices,
                                            subset_tag=subset_tag,
                                            top_n_vars=len(subset_indices),
                                            skip_preprocessing=False,
                                            skip_spectral_preprocessing=True,
                                            excluded_count=excluded_count,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            validation_count=validation_count,
                                            total_samples_original=total_samples_original,
                                            folds=folds,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )
                                    else:
                                        subset_result = _run_single_config(
                                            X_transformed_varsel, y_np, wavelengths_varsel,
                                            model, model_name, params, preprocess_cfg,
                                            cv_splitter, task_type, is_binary_classification,
                                            subset_indices=subset_indices,
                                            subset_tag=subset_tag,
                                            top_n_vars=len(subset_indices),
                                            skip_preprocessing=False,
                                            skip_spectral_preprocessing=True,
                                            excluded_count=excluded_count,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            validation_count=validation_count,
                                            total_samples_original=total_samples_original,
                                            folds=folds,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )

                                    if subset_result is None:
                                        continue
                                    df_results = add_result(df_results, subset_result)

                                    if task_type == "regression":
                                        print(f"    {subset_tag}: R²={subset_result['R2']:.3f}")
                                    else:
                                        print(f"    {subset_tag}: AUC={subset_result.get('ROC_AUC', 0):.3f}")

                                continue  # Skip to next method (don't fall through to importance path)

                            # ===== EXISTING CODE: Standard importance-returning methods =====
                            # Get importances computed on preprocessed data
                            uve_selected_mask = None  # Captured by UVE for method-optimal count
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
                                    # Use get_uve_threshold to also capture selected_mask for method-optimal count
                                    importances, _uve_threshold, uve_selected_mask = get_uve_threshold(
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
                                        hybrid_importance_weight=0.5,
                                        task_type=task_type
                                    )

                                elif varsel_method in ('uve_cars', 'uve_cars_tree'):
                                    # UVE-CARS / UVE-CARS-Tree: Noise filtering + adaptive selection
                                    if varsel_method == 'uve_cars':
                                        mt_for_cars = None
                                        uh_for_cars = False
                                        print(f"    -> Running UVE-CARS")
                                    else:
                                        mt_for_cars = model_name
                                        uh_for_cars = True
                                        print(f"    -> Running UVE-CARS-Tree for {model_name}")

                                    importances = uve_cars_selection(
                                        X_transformed_varsel, y_np,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        uve_n_components=uve_n_components,
                                        uve_cv_folds=folds,
                                        n_iterations=50,
                                        pls_components=uve_n_components if uve_n_components is not None else 5,
                                        cars_cv_folds=folds,
                                        monte_carlo_samples=80,
                                        random_state=random_state,
                                        model_type=mt_for_cars,
                                        use_hybrid_importance=uh_for_cars,
                                        hybrid_importance_weight=0.5,
                                        task_type=task_type
                                    )

                                elif varsel_method == 'uve_cars_spa':
                                    # UVE-CARS-SPA: 3-stage hybrid
                                    print(f"    -> Running UVE-CARS-SPA (3-stage)")
                                    importances = uve_cars_spa_selection(
                                        X_transformed_varsel, y_np,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        uve_n_components=uve_n_components,
                                        uve_cv_folds=folds,
                                        n_iterations=50,
                                        pls_components=uve_n_components if uve_n_components is not None else 5,
                                        cars_cv_folds=folds,
                                        monte_carlo_samples=80,
                                        spa_n_features=None,
                                        spa_n_random_starts=spa_n_random_starts,
                                        spa_cv_folds=folds,
                                        random_state=random_state,
                                        task_type=task_type
                                    )

                                elif varsel_method == 'fipls_spa':
                                    # Forward iPLS → SPA: Region selection + collinearity reduction
                                    print(f"    -> Running Forward iPLS-SPA")
                                    importances = fipls_spa_selection(
                                        X_transformed_varsel, y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        max_combine=5,
                                        ipls_cv_folds=folds,
                                        spa_n_features=None,
                                        spa_n_random_starts=spa_n_random_starts,
                                        spa_cv_folds=folds,
                                        random_state=random_state
                                    )

                                elif varsel_method == 'fipls_cars':
                                    # Forward iPLS → CARS: Region selection + adaptive selection
                                    print(f"    -> Running Forward iPLS-CARS")
                                    importances = fipls_cars_selection(
                                        X_transformed_varsel, y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        max_combine=5,
                                        ipls_cv_folds=folds,
                                        n_iterations=50,
                                        pls_components=uve_n_components if uve_n_components is not None else 5,
                                        cars_cv_folds=folds,
                                        monte_carlo_samples=80,
                                        random_state=random_state,
                                        task_type=task_type
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

                                # Compute method-optimal variable count (natural cutoff from method)
                                n_method_optimal = 0
                                method_has_natural_optimal = False

                                if not used_uniform_fallback:
                                    if varsel_method in ('cars', 'cars-aware', 'cars-tree', 'uve_spa',
                                                         'uve_cars', 'uve_cars_tree', 'uve_cars_spa',
                                                         'fipls_spa', 'fipls_cars'):
                                        n_method_optimal = int(np.count_nonzero(importances))
                                        method_has_natural_optimal = True
                                    elif varsel_method == 'uve' and uve_selected_mask is not None:
                                        n_method_optimal = int(np.sum(uve_selected_mask))
                                        method_has_natural_optimal = True
                                    elif varsel_method == 'vcpa-iriv':
                                        n_method_optimal = int(np.count_nonzero(importances))
                                        method_has_natural_optimal = True

                                if method_has_natural_optimal:
                                    if n_method_optimal <= 0 or n_method_optimal >= n_features_for_validation:
                                        method_has_natural_optimal = False
                                    elif n_method_optimal in valid_variable_counts:
                                        print(f"  -> Method-optimal for {varsel_method}: {n_method_optimal} already in counts, skipping")
                                        method_has_natural_optimal = False
                                    else:
                                        print(f"  -> Method-optimal for {varsel_method}: {n_method_optimal} vars (will test)")

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
                                            n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
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
                                            n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )

                                    if subset_result is None:
                                        print(f"[SKIPPED]")
                                        continue

                                    # Track if uniform fallback was used for this result
                                    subset_result["uniform_fallback"] = used_uniform_fallback

                                    df_results = add_result(df_results, subset_result)
                                    results_added_for_method += 1

                                    # Show result immediately (CV metrics for consistency)
                                    if task_type == "regression":
                                        print(f"R²cv={subset_result['R2cv']:.3f}, RMSEcv={subset_result['RMSEcv']:.3f}")
                                    else:
                                        print(f"AUCcv={subset_result.get('ROC_AUCcv', 0):.3f}, Acccv={subset_result.get('Accuracycv', 0):.3f}")

                                    # Update best model tracker for subset results (use CV metrics for consistency)
                                    if best_model_so_far is None:
                                        best_model_so_far = subset_result
                                    else:
                                        if task_type == "regression":
                                            if subset_result["RMSEcv"] < best_model_so_far["RMSEcv"]:
                                                best_model_so_far = subset_result
                                        else:  # classification
                                            if subset_result.get("ROC_AUCcv", 0) > best_model_so_far.get("ROC_AUCcv", 0):
                                                best_model_so_far = subset_result

                                # Run method-optimal subset if applicable
                                if method_has_natural_optimal and n_method_optimal > 0:
                                    print(f"  -> Testing method-optimal {n_method_optimal} vars ({varsel_method})...", end=" ")
                                    top_indices_opt = np.argsort(importances, kind='stable')[-n_method_optimal:][::-1]

                                    if preprocess_cfg["deriv"] is not None:
                                        opt_result = _run_single_config(
                                            X_transformed_varsel, y_np,
                                            wavelengths_varsel,
                                            model, model_name, params,
                                            preprocess_cfg, cv_splitter,
                                            task_type, is_binary_classification,
                                            subset_indices=top_indices_opt,
                                            subset_tag=f"{varsel_method}",
                                            top_n_vars=30,
                                            skip_preprocessing=False,
                                            skip_spectral_preprocessing=True,
                                            excluded_count=excluded_count,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            validation_count=validation_count,
                                            total_samples_original=total_samples_original,
                                            folds=folds,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )
                                    else:
                                        opt_result = _run_single_config(
                                            X_transformed_varsel, y_np,
                                            wavelengths_varsel,
                                            model, model_name, params,
                                            preprocess_cfg, cv_splitter,
                                            task_type, is_binary_classification,
                                            subset_indices=top_indices_opt,
                                            subset_tag=f"{varsel_method}",
                                            top_n_vars=30,
                                            skip_preprocessing=False,
                                            skip_spectral_preprocessing=True,
                                            excluded_count=excluded_count,
                                            validation_count=validation_count,
                                            total_samples_original=total_samples_original,
                                            folds=folds,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )

                                    if opt_result is None:
                                        print(f"[SKIPPED]")
                                    else:
                                        opt_result["uniform_fallback"] = used_uniform_fallback
                                        df_results = add_result(df_results, opt_result)
                                        results_added_for_method += 1

                                        if task_type == "regression":
                                            print(f"R²cv={opt_result['R2cv']:.3f}, RMSEcv={opt_result['RMSEcv']:.3f} (method-optimal)")
                                        else:
                                            print(f"AUCcv={opt_result.get('ROC_AUCcv', 0):.3f}, Acccv={opt_result.get('Accuracycv', 0):.3f} (method-optimal)")

                                        if best_model_so_far is None:
                                            best_model_so_far = opt_result
                                        else:
                                            if task_type == "regression":
                                                if opt_result["RMSEcv"] < best_model_so_far["RMSEcv"]:
                                                    best_model_so_far = opt_result
                                            else:
                                                if opt_result.get("ROC_AUCcv", 0) > best_model_so_far.get("ROC_AUCcv", 0):
                                                    best_model_so_far = opt_result

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
                            n_jobs_cv=1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default,
                            wavelength_restriction_active=wavelength_restriction_active,
                            early_stopping_rounds=early_stopping_rounds,
                        )
                        if region_result is None:
                            print(f"[SKIPPED]")
                            continue
                        df_results = add_result(df_results, region_result)

                        # Show result immediately (CV metrics for consistency)
                        if task_type == "regression":
                            print(f"R²cv={region_result['R2cv']:.3f}, RMSEcv={region_result['RMSEcv']:.3f}")
                        else:
                            print(f"AUCcv={region_result.get('ROC_AUCcv', 0):.3f}, Acccv={region_result.get('Accuracycv', 0):.3f}")

                        # Update best model tracker for region subset results (use CV metrics for consistency)
                        if best_model_so_far is None:
                            best_model_so_far = region_result
                        else:
                            if task_type == "regression":
                                if region_result["RMSEcv"] < best_model_so_far["RMSEcv"]:
                                    best_model_so_far = region_result
                            else:  # classification
                                if region_result.get("ROC_AUCcv", 0) > best_model_so_far.get("ROC_AUCcv", 0):
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

    df_ranked = compute_composite_score(df_results, task_type, variable_penalty, gap_penalty)

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
    max_n_components : int, default=10
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

    # Drop rows where y is NaN (safety net for data with empty rows)
    nan_mask = y.isna()
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(f"Warning: Dropping {n_dropped} sample(s) with NaN target values before Bayesian optimization.")
        X = X[~nan_mask]
        y = y[~nan_mask]

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
            # When GA preprocessing is enabled, only use the config optimized for this specific model
            # Each model gets its own set of top-N preprocessing configs from GA optimization
            # ═══════════════════════════════════════════════════════════════════════════
            if ga_preprocess and 'ga_model_type' in preprocess_cfg:
                # Skip if this preprocessing config was optimized for a different model
                # ga_model_type stores the actual model name (e.g., "LightGBM", "PLS")
                if preprocess_cfg['ga_model_type'] != model_name:
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

            # Apply edge masking for SG derivatives (matches grid search at line 1894)
            # SG derivatives create boundary artifacts at first/last window//2 wavelengths
            wavelengths_for_model = wavelengths
            n_features_for_model = n_features
            if preprocess_cfg.get("deriv") and preprocess_cfg.get("window"):
                X_preprocessed, wavelengths_for_model, edge_zone_applied = _apply_edge_mask_to_data(
                    X_preprocessed, wavelengths_for_model, preprocess_cfg
                )
                n_features_for_model = X_preprocessed.shape[1]
                if edge_zone_applied > 0:
                    prep_name = preprocess_cfg.get("name", "unknown")
                    deriv_info = f"_d{preprocess_cfg['deriv']}"
                    print(f"\n{'='*70}")
                    print(f"BAYESIAN EDGE MASKING (after {prep_name}{deriv_info} preprocessing)")
                    print(f"{'='*70}")
                    print(f"  Derivative window: {preprocess_cfg['window']}")
                    print(f"  Edge zone: {edge_zone_applied} wavelengths on each side")
                    print(f"  Wavelengths after masking: {len(wavelengths_for_model)}")
                    print(f"  Range: {wavelengths_for_model[0]:.1f} - {wavelengths_for_model[-1]:.1f} nm")
                    print(f"{'='*70}\n")

            # Recompute max_n_components with edge-masked feature count
            config_max_n_components = max_n_components
            if n_features_for_model < n_features:
                config_max_n_components = min(max_n_components, n_features_for_model)
                config_max_n_components = max(1, config_max_n_components)

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
                max_n_components=config_max_n_components,
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
                    imbalance_method=imbalance_method,
                    imbalance_params=imbalance_params
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
        gap_penalty=0  # Bayesian optimization doesn't penalize gap
    )

    # Rename CompositeScore to Score for consistency with grid search
    if 'CompositeScore' in df_ranked.columns:
        df_ranked = df_ranked.rename(columns={'CompositeScore': 'Score'})

    print(f"\n[OK] Bayesian optimization complete!")
    print(f"  Total models optimized: {len(df_ranked)}")
    if len(df_ranked) > 0:
        best_model = df_ranked.iloc[0]
        if task_type == 'regression':
            print(f"  Best model: {best_model['Model']} (RMSEcv={best_model['RMSEcv']:.4f}, R²cv={best_model['R2cv']:.4f})")
        else:
            print(f"  Best model: {best_model['Model']} (Accuracycv={best_model['Accuracycv']:.4f}, ROC_AUCcv={best_model['ROC_AUCcv']:.4f})")

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
                     use_sample_weight_for_classification=False,
                     early_stopping_rounds=None):
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
    early_stopping_rounds : int, optional
        Number of rounds without improvement before stopping for boosting models.
        If None or 0, early stopping is disabled.

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
        # Check if we should use early stopping for boosting models
        use_early_stopping = (
            early_stopping_rounds is not None and
            early_stopping_rounds > 0
        )

        if use_early_stopping:
            # Get final model from pipeline
            if hasattr(pipe_clone, 'steps'):
                final_model_es = pipe_clone.steps[-1][1]
            else:
                final_model_es = pipe_clone

            # Check if final model is a boosting model
            if is_boosting_model(final_model_es):
                manual_fit_used = True

                # Transform training data through preprocessing steps
                X_train_transformed = X_train.copy()
                X_test_transformed = X_test.copy()

                if hasattr(pipe_clone, 'steps'):
                    for step_name, step in pipe_clone.steps[:-1]:
                        if hasattr(step, 'fit_resample'):
                            # For imblearn resamplers, apply fit_resample (only to training data)
                            X_train_transformed, y_train = step.fit_resample(X_train_transformed, y_train)
                            fitted_steps.append((step_name, step, 'resample'))
                            # Note: Don't transform test data - resampling only applies to training
                        elif hasattr(step, 'transform'):
                            step.fit(X_train_transformed, y_train)
                            X_train_transformed = step.transform(X_train_transformed)
                            X_test_transformed = step.transform(X_test_transformed)
                            fitted_steps.append((step_name, step, 'transform'))

                    final_model = final_model_es
                else:
                    final_model = final_model_es

                # Fit with early stopping
                _fit_with_early_stopping(
                    final_model,
                    X_train_transformed, y_train,
                    X_test_transformed, y_test,
                    early_stopping_rounds
                )
            else:
                # Not a boosting model - standard fit
                pipe_clone.fit(X_train, y_train)
        else:
            # No early stopping
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
        y_pred = np.ravel(y_pred)  # Ensure 1D for metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": rmse, "R2": r2, "y_test": y_test, "y_pred": y_pred}
    else:  # classification
        if manual_fit_used:
            y_pred, _ = _manual_transform_predict(X_test)
        else:
            y_pred = pipe_clone.predict(X_test)
        y_pred = np.ravel(y_pred)  # Ensure 1D for metrics

        # PLS-DA Debug logging - diagnose all-same-class predictions
        if hasattr(pipe_clone, 'named_steps') and 'pls' in pipe_clone.named_steps:
            pls = pipe_clone.named_steps['pls']
            lr = pipe_clone.named_steps['lr']
            X_scores_test = pls.transform(X_test)
            y_train_int = y_train.astype(int) if hasattr(y_train, 'astype') else np.array(y_train).astype(int)
            print(f"[PLS-DA DEBUG]")
            print(f"  y_train unique: {np.unique(y_train)}, counts: {np.bincount(y_train_int)}")
            print(f"  PLS scores shape: {X_scores_test.shape}")
            print(f"  PLS scores mean per sample (first 5): {X_scores_test.mean(axis=1)[:5]}")
            print(f"  PLS scores std: {X_scores_test.std():.6f}")
            print(f"  LR classes_: {lr.classes_}")
            print(f"  LR coef_ sum: {np.abs(lr.coef_).sum():.6f}")
            print(f"  LR intercept_: {lr.intercept_}")
            print(f"  y_pred unique: {np.unique(y_pred)}, counts: {np.bincount(y_pred.astype(int))}")

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
            logloss = np.nan
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

                # Log Loss (requires predict_proba)
                try:
                    logloss = log_loss(y_test, y_proba, labels=model_classes if model_classes is not None else None)
                except Exception:
                    logloss = np.nan
            except Exception:
                auc = np.nan
                logloss = np.nan

        # Compute additional classification metrics
        try:
            specificity = compute_specificity(y_test, y_pred, average='macro')
        except Exception:
            specificity = np.nan

        try:
            kappa = cohen_kappa_score(y_test, y_pred)
        except Exception:
            kappa = np.nan

        try:
            mcc = matthews_corrcoef(y_test, y_pred)
        except Exception:
            mcc = np.nan

        try:
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            ber = 1.0 - balanced_acc
        except Exception:
            balanced_acc = np.nan
            ber = np.nan

        return {
            "Accuracy": acc, "ROC_AUC": auc, "F1": f1, "Precision": precision, "Recall": recall,
            "Specificity": specificity, "Kappa": kappa, "MCC": mcc,
            "BalancedAcc": balanced_acc, "BER": ber, "LogLoss": logloss,
            "y_test": y_test, "y_pred": y_pred
        }


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
    early_stopping_rounds=None,
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

    # Skip invalid PLS n_components / feature count combinations
    # When n_components >= n_vars, PLS is degenerate or invalid — skip instead of silently clamping
    if hasattr(model, 'n_components') and model.n_components is not None and model.n_components >= n_vars:
        print(f"  [SKIP] {model_name} n_components={model.n_components} >= n_vars={n_vars}, invalid combination")
        return None

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

    # For PLS-DA, we need PLS + StandardScaler + LogisticRegression
    # StandardScaler normalizes PLS scores to fix numerical instability with derivatives
    if model_name == "PLS-DA":
        pipe_steps.append(("pls", model))
        pipe_steps.append(("scaler", StandardScaler()))  # Scale PLS scores for LogisticRegression

        # Extract LogisticRegression parameters from config (prefixed with lr_)
        lr_C = params.get('lr_C', 1.0) if params else 1.0
        lr_solver = params.get('lr_solver', 'lbfgs') if params else 'lbfgs'
        lr_max_iter = params.get('lr_max_iter', 1000) if params else 1000

        # Build LogisticRegression with configurable parameters
        lr_kwargs = {
            'C': lr_C,
            'solver': lr_solver,
            'max_iter': lr_max_iter,
            'random_state': random_state
        }

        # Apply class_weight to LogisticRegression if requested
        if imbalance_method == 'class_weight' and task_type == 'classification':
            lr_kwargs['class_weight'] = 'balanced'

        pipe_steps.append(("lr", LogisticRegression(**lr_kwargs)))
    # For scale-sensitive models (SVC/SVR, MLP, NeuralBoosted), add StandardScaler before model
    # These use gradient descent or kernel methods that are sensitive to feature scale
    elif model_name in SCALE_SENSITIVE_MODELS:
        pipe_steps.append(("scaler", StandardScaler()))
        pipe_steps.append(("model", model))
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
                use_sample_weight_for_classification,
                early_stopping_rounds=early_stopping_rounds
            )
            for train_idx, test_idx in cv_splitter.split(X, y)
        ]
    else:
        # Parallel execution for speed
        # Use 'threading' in frozen apps (avoids PyInstaller process spawn issues)
        # Use 'loky' in dev mode (faster multiprocessing)
        import sys
        is_frozen = getattr(sys, 'frozen', False) or '__compiled__' in dir()
        backend = 'threading' if is_frozen else 'loky'
        cv_metrics = Parallel(n_jobs=n_jobs_cv, backend=backend)(
            delayed(_run_single_fold)(
                pipe, X, y, train_idx, test_idx, task_type, is_binary_classification,
                use_sample_weight_for_classification,
                early_stopping_rounds=early_stopping_rounds
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

        # Compute regional performance (quartile-based) for consensus predictions
        # Collect all CV predictions and true values
        all_y_test = np.concatenate([m["y_test"] for m in cv_metrics])
        all_y_pred = np.concatenate([m["y_pred"] for m in cv_metrics])

        # Compute R² from aggregated predictions (not per-fold averages)
        # Averaging per-fold R² is mathematically incorrect due to different SS_tot per fold
        mean_r2 = r2_score(all_y_test, all_y_pred)

        # Compute additional NIR spectroscopy metrics from aggregated CV predictions
        # MAEcv: Mean Absolute Error - less sensitive to outliers than RMSE
        mae_cv = mean_absolute_error(all_y_test, all_y_pred)
        # Bias: Mean prediction error (positive = systematic overprediction)
        bias_cv = float(np.mean(all_y_pred - all_y_test))
        # RPD: Ratio of Performance to Deviation (std(y) / RMSEcv)
        # Industry standard for NIR model fitness assessment
        # RPD < 1.5: Poor, 1.5-2: Screening only, 2-3: Acceptable, > 3: Good
        y_std = float(np.std(y))
        rpd = y_std / mean_rmse if mean_rmse > 0 else 0.0
        # RER: Range Error Ratio (range(y) / RMSEcv)
        # Alternative to RPD, uses data range instead of standard deviation
        y_range = float(np.ptp(y))  # max(y) - min(y)
        rer = y_range / mean_rmse if mean_rmse > 0 else 0.0

        # Compute quartiles based on TRUE Y values
        # Regional selection identifies which models excel in different value ranges
        # The auto-ensemble now uses stacking (not routing), so true Y is correct
        quartiles = np.percentile(all_y_test, [25, 50, 75])

        # Compute RMSE for each quartile region
        # Note: Regional R² is not computed because it's mathematically misleading
        # when Y values are restricted to a narrow range (low variance → negative R²)
        regional_rmse = {}
        for i, (lower, upper) in enumerate([
            (-np.inf, quartiles[0]),  # Q1
            (quartiles[0], quartiles[1]),  # Q2
            (quartiles[1], quartiles[2]),  # Q3
            (quartiles[2], np.inf)  # Q4
        ]):
            # Use true Y values for mask (auto-ensemble uses stacking, not routing)
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

        # New classification metrics
        mean_specificity = np.mean([m["Specificity"] for m in cv_metrics if not np.isnan(m["Specificity"])])
        mean_kappa = np.mean([m["Kappa"] for m in cv_metrics if not np.isnan(m["Kappa"])])
        mean_mcc = np.mean([m["MCC"] for m in cv_metrics if not np.isnan(m["MCC"])])
        mean_balanced_acc = np.mean([m["BalancedAcc"] for m in cv_metrics if not np.isnan(m["BalancedAcc"])])
        mean_ber = np.mean([m["BER"] for m in cv_metrics if not np.isnan(m["BER"])])
        mean_logloss = np.mean([m["LogLoss"] for m in cv_metrics if not np.isnan(m["LogLoss"])])

        regional_rmse = None  # Not applicable for classification

        # Collect all CV predictions and true values (same as regression)
        all_y_test = np.concatenate([m["y_test"] for m in cv_metrics])
        all_y_pred = np.concatenate([m["y_pred"] for m in cv_metrics])

        # Compute per-class metrics for classification (analogous to regional RMSE for regression)
        per_class_metrics = {}
        class_labels = None
        try:
            # Get per-class metrics from aggregated CV predictions
            report = classification_report(all_y_test, all_y_pred, output_dict=True, zero_division=0)
            class_labels = sorted([k for k in report.keys()
                                   if k not in ['accuracy', 'macro avg', 'weighted avg']])
            for class_label in class_labels:
                class_key = str(class_label)
                if class_key in report:
                    per_class_metrics[class_key] = {
                        'F1': report[class_key]['f1-score'],
                        'Precision': report[class_key]['precision'],
                        'Recall': report[class_key]['recall'],
                        'Support': int(report[class_key]['support'])
                    }
        except Exception as e:
            print(f"Warning: Could not compute per-class metrics: {e}")
            per_class_metrics = {}
            class_labels = None

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
    cal_specificity = None
    cal_kappa = None
    cal_mcc = None
    cal_balanced_acc = None
    cal_ber = None
    cal_logloss = None

    try:
        # Refit the pipeline on full data to get final fitted parameters
        pipe.fit(X, y)

        # Get the fitted model from pipeline for parameter capture
        # IMPORTANT: For PLS-DA and other multi-step pipelines without "model" step,
        # we need the FULL pipeline to capture all parameters (including n_components).
        # Only extract specific model step if it's a standard "model" named step.
        if hasattr(pipe, "named_steps"):
            if "model" in pipe.named_steps:
                fitted_model = pipe.named_steps["model"]
            else:
                # For PLS-DA and other pipelines without "model" step,
                # use full pipeline to capture ALL parameters (pls__n_components, etc.)
                fitted_model = pipe
        else:
            fitted_model = pipe

        # Compute calibration metrics (training data performance)
        y_pred_cal = pipe.predict(X)
        y_pred_cal = np.ravel(y_pred_cal)  # Ensure 1D for metrics

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
            except Exception as e:
                logger.debug(f"Failed to compute calibration ROC AUC: {e}")
                cal_auc = np.nan

            # Compute F1, Precision, Recall
            try:
                cal_f1 = f1_score(y, y_pred_cal, average='weighted', zero_division=0)
                cal_precision = precision_score(y, y_pred_cal, average='weighted', zero_division=0)
                cal_recall = recall_score(y, y_pred_cal, average='weighted', zero_division=0)
            except Exception as e:
                logger.debug(f"Failed to compute calibration F1/Precision/Recall: {e}")
                cal_f1 = np.nan
                cal_precision = np.nan
                cal_recall = np.nan

            # Compute new classification metrics
            try:
                cal_specificity = compute_specificity(y, y_pred_cal, average='macro')
            except Exception as e:
                logger.debug(f"Failed to compute calibration Specificity: {e}")
                cal_specificity = np.nan

            try:
                cal_kappa = cohen_kappa_score(y, y_pred_cal)
            except Exception as e:
                logger.debug(f"Failed to compute calibration Kappa: {e}")
                cal_kappa = np.nan

            try:
                cal_mcc = matthews_corrcoef(y, y_pred_cal)
            except Exception as e:
                logger.debug(f"Failed to compute calibration MCC: {e}")
                cal_mcc = np.nan

            try:
                cal_balanced_acc = balanced_accuracy_score(y, y_pred_cal)
                cal_ber = 1.0 - cal_balanced_acc
            except Exception as e:
                logger.debug(f"Failed to compute calibration BalancedAcc/BER: {e}")
                cal_balanced_acc = np.nan
                cal_ber = np.nan

            # Compute Log Loss
            try:
                if hasattr(pipe, "predict_proba"):
                    y_pred_proba_cal = pipe.predict_proba(X)
                    cal_logloss = log_loss(y, y_pred_proba_cal)
                else:
                    cal_logloss = np.nan
            except Exception as e:
                logger.debug(f"Failed to compute calibration LogLoss: {e}")
                cal_logloss = np.nan

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

            # Pipeline-specific params that shouldn't be saved/restored
            # These are Pipeline meta-parameters, not model parameters
            # verbose must be bool in newer scikit-learn, but get_params() returns int
            PIPELINE_META_PARAMS = {'verbose', 'memory', 'steps', 'transform_input'}

            filtered_params = {}
            for key, value in all_params.items():
                # Skip Pipeline-specific parameters that cause issues when re-applied
                if key in PIPELINE_META_PARAMS:
                    continue

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
    # CRITICAL FIX: Also check for pls__n_components (PLS-DA pipeline prefixed params)
    n_comp = params.get("n_components") or params.get("pls__n_components")
    lvs = int(n_comp) if n_comp is not None else None

    # Format imbalance handling indicator for display
    if imbalance_method is None:
        imbalance_display = "—"
    elif imbalance_method == 'class_weight':
        imbalance_display = "class_weight"
    else:
        imbalance_display = imbalance_method

    # Build result dictionary AFTER capturing complete params
    preprocess_display = preprocess_cfg["name"]
    preprocess_base = preprocess_cfg.get("base_name", preprocess_cfg["name"])
    result = {
        "Task": task_type,
        "Model": model_name,
        "Params": str(params),  # Now includes complete parameter set
        "Preprocess": preprocess_display,  # Full name including baseline prefix (e.g., als+snv)
        "PreprocessBase": preprocess_base,  # Clean pipeline name for build_preprocessing_pipeline()
        "Deriv": preprocess_cfg["deriv"],
        "Window": preprocess_cfg["window"],
        "Poly": preprocess_cfg["polyorder"],
        "LVs": lvs,
        "n_vars": n_vars,
        "full_vars": full_vars,
        "SubsetTag": subset_tag,
        "Imbalance": imbalance_display,
        # Track early stopping to allow Model Development to reproduce boosted results
        "early_stopping_rounds": early_stopping_rounds if model_name in ("XGBoost", "LightGBM", "CatBoost") else None,
        # Store actual imbalance settings for Model Development tab to use
        # (imbalance_display is for UI, these are for exact pipeline reconstruction)
        "imbalance_method": imbalance_method,
        "imbalance_params": imbalance_params,
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

    # Store Smart preprocessing metadata if present (for validation reconstruction)
    if 'smart_selected_wavelengths' in preprocess_cfg and preprocess_cfg['smart_selected_wavelengths'] is not None:
        result["smart_selected_wavelengths"] = preprocess_cfg['smart_selected_wavelengths']
        result["smart_n_wavelengths"] = preprocess_cfg.get("smart_n_wavelengths")
        result["smart_score"] = preprocess_cfg.get("smart_score")
        result["smart_importance_method"] = preprocess_cfg.get("smart_importance_method")
        result["smart_model_name"] = preprocess_cfg.get("smart_model_name")

    if task_type == "regression":
        # Calibration metrics (training data)
        result["RMSE"] = cal_rmse if cal_rmse is not None else np.nan
        result["R2"] = cal_r2 if cal_r2 is not None else np.nan
        # Cross-validation metrics (test fold averages)
        result["RMSEcv"] = mean_rmse
        result["R2cv"] = mean_r2
        # NIR-specific metrics (computed from aggregated CV predictions)
        result["MAEcv"] = mae_cv
        result["RPD"] = rpd
        result["Bias"] = bias_cv
        result["RER"] = rer
        # Add regional performance for consensus predictions (dict format for ensemble)
        result["regional_rmse"] = regional_rmse
        result["y_quartiles"] = quartiles.tolist()  # Save quartile thresholds
        # Add individual quartile columns for display/sorting
        if regional_rmse is not None:
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                result[f'RMSE_{q}'] = regional_rmse.get(q, np.nan)
    else:
        # Calibration metrics (training data)
        result["Accuracy"] = cal_acc if cal_acc is not None else np.nan
        result["ROC_AUC"] = cal_auc if cal_auc is not None else np.nan
        result["F1"] = cal_f1 if cal_f1 is not None else np.nan
        result["Precision"] = cal_precision if cal_precision is not None else np.nan
        result["Recall"] = cal_recall if cal_recall is not None else np.nan
        result["Specificity"] = cal_specificity if cal_specificity is not None else np.nan
        result["Kappa"] = cal_kappa if cal_kappa is not None else np.nan
        result["MCC"] = cal_mcc if cal_mcc is not None else np.nan
        result["BalancedAcc"] = cal_balanced_acc if cal_balanced_acc is not None else np.nan
        result["BER"] = cal_ber if cal_ber is not None else np.nan
        result["LogLoss"] = cal_logloss if cal_logloss is not None else np.nan
        # Cross-validation metrics (test fold averages)
        result["Accuracycv"] = mean_acc
        result["ROC_AUCcv"] = mean_auc
        result["F1cv"] = mean_f1
        result["Precisioncv"] = mean_precision
        result["Recallcv"] = mean_recall
        result["Specificitycv"] = mean_specificity
        result["Kappacv"] = mean_kappa
        result["MCCcv"] = mean_mcc
        result["BalancedAcccv"] = mean_balanced_acc
        result["BERcv"] = mean_ber
        result["LogLosscv"] = mean_logloss
        # Per-class metrics for regional analysis (analogous to regional_rmse for regression)
        result["per_class_metrics"] = per_class_metrics if per_class_metrics else None
        result["class_labels"] = class_labels
        # Add individual class F1 columns for display/sorting (like RMSE_Q1 for regression)
        if per_class_metrics:
            for class_label, metrics in per_class_metrics.items():
                result[f'F1_Class{class_label}'] = metrics['F1']

    # Save all_vars for ALL models (including full spectrum)
    # This ensures Model Development can reconstruct the exact wavelengths used
    # CRITICAL: For full models, 'wavelengths' is already filtered by wl_min/wl_max
    # so we must save it to allow exact replication
    if subset_tag != "full" and subset_indices is not None:
        # Subset model: save only the subset wavelengths
        subset_wavelengths = wavelengths[subset_indices]
        all_vars_str = ','.join([f"{w:.1f}" for w in subset_wavelengths])
        result['all_vars'] = all_vars_str
    else:
        # Full model: save ALL wavelengths used (may be filtered by wl_min/wl_max)
        all_vars_str = ','.join([f"{w:.1f}" for w in wavelengths])
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
            top_vars_str = ','.join([f"{w:.1f}" for w in top_wavelengths])
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
