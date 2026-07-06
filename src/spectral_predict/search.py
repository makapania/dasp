"""Model search with cross-validation and subset selection."""

import os
import sys
import inspect
import logging
from typing import Optional


def _frozen_needs_threading_fallback() -> bool:
    """Whether the current frozen build needs the threading-backend workaround.

    PyInstaller windowed bundles cannot safely use loky's spawn method
    regardless of Python version.  The frozen runtime hook's
    multiprocessing.freeze_support() crashes on argv parsing in child
    processes ("ValueError: not enough values to unpack (expected 2, got
    1)"), and the parent retries spawning → fork-bomb of GUI windows.
    Falling back to the threading backend avoids the broken spawn entirely.
    """
    is_frozen = getattr(sys, "frozen", False) or "__compiled__" in globals()
    return is_frozen


import numpy as np
import pandas as pd


def _normalize_mixed_type_labels(labels):
    """Normalize mixed-type class labels so numeric-equivalent values collapse.

    Accepts pd.Series, np.ndarray, or list; returns the same container type.
    NaN is preserved; never stringified.
    """

    def _norm(v):
        if pd.isna(v):
            return v
        if isinstance(v, str):
            v = v.strip()
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except (ValueError, TypeError):
            return str(v)

    if isinstance(labels, pd.Series):
        return labels.apply(_norm)
    if isinstance(labels, np.ndarray):
        return np.array([_norm(v) for v in labels], dtype=object)
    if isinstance(labels, list):
        return [_norm(v) for v in labels]
    return type(labels)(_norm(v) for v in labels)


from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    mean_absolute_error,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from joblib import Parallel, delayed

from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocess import build_preprocessing_pipeline
from .models import get_model_grids, get_feature_importances
from .scoring import (
    add_result,
    compute_cv_anova_pvalue,
    compute_specificity,
    create_results_dataframe,
    lins_ccc,
)
from .regions import create_region_subsets, format_region_report
from .variable_selection import (
    spa_selection,
    uve_selection,
    uve_spa_selection,
    ipls_selection,
    ipls_forward,
    ipls_backward,
    cars_selection,
    get_uve_threshold,
    uve_cars_selection,
    uve_cars_spa_selection,
    fipls_spa_selection,
    fipls_cars_selection,
    mc_sipls,
    mwpls,
)
from .wavelength_selection import vcpa_iriv
from .ga_pls import ga_pls_selection
from .ga_lightgbm import ga_lightgbm_selection
from .model_registry import supports_subset_analysis, supports_feature_importance
from .constants import RANDOM_STATE

# Import early stopping CV utilities
from .cv_utils import is_boosting_model, _fit_with_early_stopping, build_cv_splitter

from .ga_preprocessing import optimize_preprocessing, PREPROC_TYPES, WINDOW_SIZES
from .preprocessing_discovery import discover_preprocessing, IMPORTANCE_METHODS

# NSGA-II import
from .nsga2_search import run_nsga2_search, convert_nsga2_to_v1_format

logger = logging.getLogger(__name__)

# Model categories for GA preprocessing (4 specialized groups)
# Each group uses a fitness model that best represents its characteristics

# PLS-based models: Linear regression with dimension reduction
PLS_MODELS = {"PLS", "PLS-DA", "Ridge", "Lasso", "ElasticNet"}

# Neural/SVM models: Non-linear, kernel-based or neural network models
NEURAL_SVM_MODELS = {"MLP", "SVR", "SVC"}

# Tree models: Gradient boosting and ensemble tree methods
TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}

# Neural-boosted hybrid model (single specialized model)
NEURALBOOSTED_MODELS = {"NeuralBoosted"}

# Scale-sensitive models: These use gradient descent or kernel methods
# that are sensitive to feature scale and benefit from StandardScaler
SCALE_SENSITIVE_MODELS = {"SVC", "SVR", "MLP", "NeuralBoosted", "Ridge", "Lasso", "ElasticNet"}

# Models that are slower with parallel CV due to threading conflicts or low overhead
# SVM: internal multi-threading conflicts with sklearn's CV parallelization
# PLS/PLS-DA: so fast that joblib overhead dominates (0.08s serial vs 0.29s parallel)
# Ridge/Lasso/ElasticNet: linear solve is ~5ms, joblib spawn overhead is ~1s on Windows
MODELS_PREFER_SERIAL_CV = {"SVM", "PLS", "PLS-DA", "Ridge", "Lasso", "ElasticNet"}

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


def _apply_edge_mask_to_data(X: np.ndarray, wavelengths: np.ndarray, preprocess_cfg: dict) -> tuple:
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
        print(
            f"  Warning: Edge zone ({edge_zone} per side) would remove all {X.shape[1]} wavelengths. Skipping edge masking."
        )
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
        return "sample_weight" in sig.parameters
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

    # class_weight / auto don't need resampling pipeline — they're model-parameter
    # sentinels (auto is resolved into class_weight or None at run-entry; both
    # short-circuit here as defense-in-depth so a future refactor that delays
    # resolution can't accidentally wrap them in ImbPipeline).
    if imbalance_method in ("class_weight", "auto"):
        return False

    # Classification resampling methods need imblearn Pipeline
    if task_type == "classification":
        resampling_methods = [
            "smote",
            "adasyn",
            "borderline_smote",
            "random_undersampler",
            "tomek_links",
            "smote_tomek",
            "smote_enn",
        ]
        return imbalance_method.lower().replace("-", "_") in resampling_methods

    # Regression: resampling methods need imblearn Pipeline (fit_resample)
    # 'binning', 'rare_boost', 'balanced' use RegressionSampleWeighter (fit/transform only)
    if task_type == "regression":
        resampling_methods = ["undersample", "oversample", "smogn", "smotetomek"]
        return imbalance_method.lower() in resampling_methods

    return False


def _rebuild_model_from_row(row: pd.Series, task_type: str, *, autoscale: bool = False):
    """Rebuild sklearn model from results row metadata.

    This function recreates the exact model configuration used during search,
    matching how Model Dev tab does it (ast.literal_eval + set_params).

    Parameters
    ----------
    row : pd.Series
        A row from the results DataFrame containing model configuration
    task_type : str
        'regression' or 'classification'
    autoscale : bool, default False
        T-36 fix (post-merge review): when True, the preprocessing pipeline
        already includes a StandardScaler, so the per-model scaler that this
        function would otherwise wrap around scale-sensitive estimators is
        skipped to avoid double-scaling on the validation rebuild path.

    Returns
    -------
    model : sklearn estimator
        Model instance with correct hyperparameters applied
    """
    import ast
    from .models import get_model

    # Get model info
    model_name = row.get("Model", "PLS")
    params_str = row.get("Params", "")
    n_lvs = row.get("LVs", None)

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
    model = get_model(
        model_name,
        task_type=task_type,
        n_components=n_components,
        max_n_components=max(n_components, 20),
    )

    # Strip Pipeline prefixes from stored params (e.g., 'model__n_estimators' → 'n_estimators').
    # PLS-DA captured params carry 'pls__' as the inner-step prefix. At this point
    # `model` is still the bare estimator (PLSTransformer for PLS-DA, before the
    # outer Pipeline wrap below), so 'pls__n_components' must be stripped to bare
    # 'n_components' before set_params, otherwise sklearn raises and the rebuild
    # falls back to the inflated default n_components from the (potentially stale)
    # LVs column. Other Pipeline wrappers (scaler__, lr__) are still skipped because
    # those sub-estimators are constructed fresh during the wrap, not via set_params.
    if model_kwargs:
        normalized = {}
        for key, value in model_kwargs.items():
            if key.startswith("model__"):
                normalized[key[7:]] = value
            elif key.startswith("pls__"):
                normalized[key[5:]] = value
            elif "__" in key:
                continue  # Skip remaining Pipeline wrapper params (scaler__, lr__)
            else:
                normalized[key] = value
        model_kwargs = normalized

    # Apply parameters using set_params (same as Model Dev tab)
    if model_kwargs:
        try:
            model.set_params(**model_kwargs)
        except Exception as e:
            print(f"  [Warning] Could not apply params {model_kwargs}: {e}")

    # For PLS-DA classification, wrap PLSTransformer with LogisticRegression
    # This matches how PLS-DA is built during search (search.py:3420-3443)
    if task_type == "classification" and model_name == "PLS-DA":
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        # Extract LogisticRegression parameters from config (prefixed with lr_)
        lr_C = model_kwargs.get("lr_C", 1.0)
        lr_solver = model_kwargs.get("lr_solver", "lbfgs")
        lr_max_iter = model_kwargs.get("lr_max_iter", 1000)

        pls_lr_pipeline = Pipeline(
            [
                ("pls", model),
                ("scaler", StandardScaler()),  # Scale PLS scores for LogisticRegression
                (
                    "lr",
                    LogisticRegression(
                        C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=42
                    ),
                ),
            ]
        )
        return pls_lr_pipeline

    # For scale-sensitive models (SVC/SVR, MLP, NeuralBoosted), add StandardScaler.
    # T-36 fix (post-merge review): when autoscale was active during search, the
    # preprocessing pipeline already StandardScaler'd the inputs — adding another
    # scaler here would double-scale and produce different validation scores than
    # search produced. Skip the per-model scaler in that case.
    if model_name in SCALE_SENSITIVE_MODELS and not autoscale:
        from sklearn.pipeline import Pipeline

        scaled_pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        return scaled_pipeline

    return model


def _apply_class_weight_discriminator_for_rebuilt_model(
    model,
    model_name: str,
    task_type: str,
    y_train: np.ndarray,
    imbalance_method: Optional[str] = None,
) -> dict:
    """Apply the class_weight discriminator to a model rebuilt from a result row.

    Mirrors the canonical class_weight discriminator block in
    ``_run_single_config`` (CatBoost / ``hasattr(class_weight)`` / sample_weight
    fallback), with sister-site implementations in ``unified_bayesian.objective``,
    ``nsga2_search.SearchProblem._evaluate``, ``nsga2_search._compute_classification_cv_metrics``,
    and ``nsga2_search._compute_calibration_metrics``. Used by rebuild paths
    that reconstruct a model from the result-row Params dict — XGBoost's
    fit-time sample_weight and PLS-DA's LR sub-step class_weight are not in
    Params and would be silently lost (training UNWEIGHTED) without
    re-application here.

    Returns fit_kwargs ready to splat into ``model.fit(X, y, **fit_kwargs)``.
    For models whose class_weight is a constructor kwarg, ``set_params`` is
    applied in place and ``{}`` is returned. For models that need fit-time
    sample_weight (XGBoost; sklearn estimators without a class_weight kwarg
    that accept sample_weight in fit), the computed sample_weight is returned
    keyed appropriately for either bare estimators (``sample_weight``) or
    Pipeline-wrapped estimators (``model__sample_weight``).

    Normalizes ``'auto'`` → ``'class_weight'`` internally so direct GUI callers
    that pass the raw user selection don't bypass the discriminator.
    """
    if task_type != "classification" or not imbalance_method:
        return {}
    method = imbalance_method.lower() if isinstance(imbalance_method, str) else None
    if method == "auto":
        method = "class_weight"
    if method != "class_weight":
        return {}

    # 1. class_weight via constructor kwarg — probe deep params in priority
    #    order. PLS-DA Pipeline exposes `lr__class_weight`; scale-sensitive
    #    Pipeline (`('scaler', ...), ('model', est)`) exposes
    #    `model__class_weight`; bare estimators expose `class_weight`.
    deep_params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    for key in ("lr__class_weight", "model__class_weight", "class_weight"):
        if key in deep_params:
            try:
                model.set_params(**{key: "balanced"})
                return {}
            except Exception as e:
                import warnings

                warnings.warn(
                    f"set_params({key}='balanced') failed during rebuild: {e}. "
                    f"Validation model will train UNWEIGHTED.",
                    UserWarning,
                )
                break

    # 2. CatBoost: class_weight is exposed as `class_weights` (plural) /
    #    `auto_class_weights`, which the loop above won't catch. Match the
    #    canonical pattern's CatBoost branch.
    if model_name == "CatBoost":
        try:
            model.set_params(auto_class_weights="Balanced")
            return {}
        except Exception as e:
            import warnings

            warnings.warn(
                f"CatBoost set_params(auto_class_weights='Balanced') failed during "
                f"rebuild: {e}. Validation model will train UNWEIGHTED.",
                UserWarning,
            )

    # 3. sample_weight fallback for estimators whose fit() accepts it (XGBoost,
    #    RidgeClassifier, etc.). Route via Pipeline kwarg if wrapped.
    import inspect

    final_estimator = model
    sample_weight_kwarg = "sample_weight"
    if hasattr(model, "named_steps"):
        if "model" in model.named_steps:
            final_estimator = model.named_steps["model"]
            sample_weight_kwarg = "model__sample_weight"
        else:
            # Pipeline whose final step isn't named 'model' (e.g., PLS-DA's
            # 'lr' was already addressed in step 1; if we got here, set_params
            # failed and there is no clean fallback).
            return {}

    fit_sig = inspect.signature(final_estimator.fit) if hasattr(final_estimator, "fit") else None
    if fit_sig and "sample_weight" in fit_sig.parameters:
        from sklearn.utils.class_weight import compute_sample_weight

        sw = compute_sample_weight("balanced", y_train)
        return {sample_weight_kwarg: sw}

    # 4. No mechanism available — warn (mirrors the no-mechanism branch in
    #    the canonical _run_single_config discriminator).
    import warnings

    if model_name in ("MLP", "MLPClassifier"):
        warnings.warn(
            f"{model_name} does not support class_weight or sample_weight. "
            f"Validation model trains UNWEIGHTED. For imbalanced classification "
            f"with MLP, use SMOTE or other resampling methods.",
            UserWarning,
        )
    else:
        warnings.warn(
            f"{model_name} does not support class_weight in any supported form. "
            f"Validation model trains UNWEIGHTED. Consider SMOTE.",
            UserWarning,
        )
    return {}


def compute_validation_metrics_for_top_models(
    df_results: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
    wavelengths: np.ndarray,
    top_n: int = 100,
    progress_callback=None,
    imbalance_method: Optional[str] = None,
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
    imbalance_method : str, optional
        The imbalance method that was active during search. Threaded through
        so the rebuilt model receives the same class_weight / sample_weight
        treatment as the search-time model — without it, XGBoost retrains
        UNWEIGHTED (its class_weight lives only in fit-time sample_weight)
        and PLS-DA retrains UNWEIGHTED (its LR sub-step's class_weight is
        not serialized into the result-row Params dict).

    Returns
    -------
    pd.DataFrame
        Results with RMSEP, R2pred (or val_Accuracy) columns added
    """
    # Drop samples with NaN target values (safety net — upstream should filter but may not)
    train_nan_mask = pd.isna(y_train)
    if np.any(train_nan_mask):
        n_dropped = int(np.sum(train_nan_mask))
        print(f"[Validation] Dropping {n_dropped} training sample(s) with NaN target values")
        X_train = X_train[~train_nan_mask]
        y_train = y_train[~train_nan_mask]

    val_nan_mask = pd.isna(y_val)
    if np.any(val_nan_mask):
        n_dropped = int(np.sum(val_nan_mask))
        print(f"[Validation] Dropping {n_dropped} validation sample(s) with NaN target values")
        X_val = X_val[~val_nan_mask]
        y_val = y_val[~val_nan_mask]

    # Initialize columns
    if task_type == "regression":
        df_results["RMSEP"] = np.nan
        df_results["R2pred"] = np.nan
    else:
        df_results["val_Accuracy"] = np.nan
        df_results["val_ROC_AUC"] = np.nan
        df_results["val_F1"] = np.nan
        df_results["val_Precision"] = np.nan
        df_results["val_Recall"] = np.nan

    # Get top N indices by CompositeScore (lower is better)
    n_to_process = min(top_n, len(df_results))
    if "CompositeScore" in df_results.columns:
        # Ensure CompositeScore is numeric (may be object dtype from CSV)
        df_results["CompositeScore"] = pd.to_numeric(df_results["CompositeScore"], errors="coerce")
        top_indices = df_results.nsmallest(n_to_process, "CompositeScore").index
    else:
        # Fallback to first n rows
        top_indices = df_results.head(n_to_process).index

    print(f"\n[Validation] Computing validation metrics for top {n_to_process} models...")

    # Coerce mixed-type labels to strings for classification / one-class
    if task_type in ("classification", "one_class"):
        if getattr(y_train, "dtype", None) == object:
            _types = {type(v).__name__ for v in y_train}
            if len(_types) > 1:
                y_train = _normalize_mixed_type_labels(y_train)
        if getattr(y_val, "dtype", None) == object:
            _types = {type(v).__name__ for v in y_val}
            if len(_types) > 1:
                y_val = _normalize_mixed_type_labels(y_val)

    # For classification, check class distribution in validation set and warn if problematic
    if task_type == "classification":
        val_class_counts = pd.Series(y_val).value_counts()
        train_class_counts = pd.Series(y_train).value_counts()
        classes_in_train = set(train_class_counts.index)
        classes_in_val = set(val_class_counts.index)
        missing_classes = classes_in_train - classes_in_val

        if missing_classes:
            print(
                f"\n[Validation Warning] {len(missing_classes)} class(es) not represented in validation set: {missing_classes}"
            )
            print(f"  Training class distribution: {dict(train_class_counts)}")
            print(f"  Validation class distribution: {dict(val_class_counts)}")
            print(
                f"  Some metrics (ROC AUC) will be NaN. Consider using more validation samples or fewer classes.\n"
            )

        # Check for critically small class samples in validation
        min_samples_per_class = val_class_counts.min() if len(val_class_counts) > 0 else 0
        if min_samples_per_class < 2:
            print(
                f"[Validation Warning] Some classes have <2 samples in validation. Metrics may be unreliable.\n"
            )

    # Cache preprocessed data by preprocessing config to avoid redundant computation
    preprocess_cache = {}

    for i, idx in enumerate(top_indices):
        row = df_results.loc[idx]

        try:
            # === STEP 1: Get preprocessing config ===
            # Use PreprocessBase (clean pipeline name) if available, fall back to Preprocess
            preprocess_name = row.get("PreprocessBase", row.get("Preprocess", "raw"))

            # Read explicit metadata columns (stored by Bayesian search paths)
            baseline_method = row.get("baseline_method", None)
            if isinstance(baseline_method, float) and pd.isna(baseline_method):
                baseline_method = None
            smoothing = bool(row.get("smoothing", False))
            if isinstance(smoothing, float):
                smoothing = smoothing > 0
            smoothing_window = (
                int(row.get("smoothing_window", 17))
                if not (
                    isinstance(row.get("smoothing_window"), float)
                    and pd.isna(row.get("smoothing_window"))
                )
                else 17
            )
            smoothing_polyorder = (
                int(row.get("smoothing_polyorder", 2))
                if not (
                    isinstance(row.get("smoothing_polyorder"), float)
                    and pd.isna(row.get("smoothing_polyorder"))
                )
                else 2
            )
            # T-36: autoscale flag must be read so the validation rebuild matches the
            # search pipeline. Old .dasp files without the column default to False.
            # Parse robustly: handles bool, numpy.bool_, NaN-float, int 0/1, and the
            # quoted-string forms ("True"/"False"/"1"/"0") that hand-edited CSVs may
            # contain. Note: bool("False") == True in Python, so a naive bool() cast
            # is wrong on the string path.
            autoscale_raw = row.get("Autoscale", False)
            if isinstance(autoscale_raw, float) and pd.isna(autoscale_raw):
                autoscale = False
            elif isinstance(autoscale_raw, str):
                autoscale = autoscale_raw.strip().lower() in ("true", "1", "yes")
            else:
                autoscale = bool(autoscale_raw)

            # Fallback: parse display name for old results without explicit columns.
            # Sets autoscale=True ONLY when the explicit column read above produced False
            # (so an explicit column always wins over a name suffix). The smoothing flag
            # follows the same pattern (sg0 prefix only sets it when not already True).
            if "+" in str(preprocess_name):
                parts = str(preprocess_name).split("+")
                core_parts = []
                for part in parts:
                    if part in ("als", "polynomial", "rubber_band", "airpls", "advanced"):
                        if baseline_method is None:
                            baseline_method = part
                    elif part == "sg0":
                        smoothing = True
                    elif part == "autoscale":
                        if not autoscale:
                            autoscale = True
                    else:
                        core_parts.append(part)
                preprocess_name = "_".join(core_parts) if core_parts else "raw"

            deriv = row.get("Deriv", 0)
            window = row.get("Window", None)
            poly = row.get("Poly", None)

            # Check for exhaustive-preprocessing chromosome (needs reconstruction).
            # Column rename 2026-05-06: this used to be `ga_genes`, but the GUI's
            # Refine tab uses `ga_genes` for GA-PLS / GA-LightGBM wavelength
            # selection (an entirely different artifact). Reading
            # preprocessing-chromosome data through the wavelength-index path
            # produced silent garbage (X[:, [3,5,1]] instead of a transform).
            # Read `preprocess_chromosome` first; fall back to `ga_genes` for
            # result CSVs written before the rename.
            ga_genes_str = row.get("preprocess_chromosome", None)
            if ga_genes_str is None:
                ga_genes_str = row.get("ga_genes", None)
            use_ga_transform = False
            ga_transform = None
            ga_genes = None

            # Handle ga_genes_str being None, empty string, NaN scalar, list, or array
            ga_genes_is_valid = False
            if ga_genes_str is not None:
                if isinstance(ga_genes_str, (list, np.ndarray)):
                    ga_genes_is_valid = len(ga_genes_str) > 0
                elif isinstance(ga_genes_str, str):
                    ga_genes_is_valid = ga_genes_str != ""
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
                    from spectral_predict.ga_preprocessing import (
                        chromosome_to_transform,
                        _decode_autoscale_gene,
                    )

                    # Reconstruct transform from genes
                    _, ga_transform = chromosome_to_transform(ga_genes)
                    use_ga_transform = True

                    # Backfill autoscale from the chromosome itself when the
                    # row's "Autoscale" column is missing or False but the
                    # 3-gene chromosome encodes autoscale=True. Pre-fix
                    # CSVs and any saved-model artefacts written before this
                    # path stopped baking the scaler into the closure could
                    # otherwise rebuild without the post-closure scaler step.
                    if not autoscale and _decode_autoscale_gene(ga_genes):
                        autoscale = True
                except Exception as e:
                    genes_preview = (
                        str(ga_genes_str)[:100]
                        if isinstance(ga_genes_str, str)
                        else str(ga_genes_str)
                    )
                    print(f"  [Warning] Could not reconstruct GA transform: {e}")
                    print(f"            GA genes data: {genes_preview}")
                    use_ga_transform = False

            # Convert to proper types (only needed if not using GA transform)
            if not use_ga_transform:
                deriv = int(deriv) if deriv and not pd.isna(deriv) and deriv > 0 else None
                window = int(window) if window and not pd.isna(window) and window > 0 else None
                poly = int(poly) if poly and not pd.isna(poly) and poly > 0 else None

            # T-36 fix (post-merge review v2): persist baseline_params from the
            # row so non-default ALS/polynomial settings survive the regression /
            # classification validation roundtrip rather than silently snapping
            # back to defaults — same shape as the one-class fix in
            # contamination.py:~1141.
            baseline_params_raw = row.get("baseline_params", None)
            baseline_params = None
            if baseline_params_raw is not None:
                if isinstance(baseline_params_raw, dict):
                    baseline_params = baseline_params_raw
                elif isinstance(baseline_params_raw, str) and baseline_params_raw.strip():
                    try:
                        import ast as _ast_local

                        parsed = _ast_local.literal_eval(baseline_params_raw)
                        if isinstance(parsed, dict):
                            baseline_params = parsed
                    except (ValueError, SyntaxError):
                        baseline_params = None

            # Create cache key
            if use_ga_transform:
                # GA preprocessing: cache by genes hash
                cache_key = ("ga", tuple(ga_genes))
            else:
                cache_key = (
                    preprocess_name,
                    deriv,
                    window,
                    poly,
                    baseline_method,
                    smoothing,
                    smoothing_window,
                    smoothing_polyorder,
                    autoscale,
                )  # T-36: must vary key to avoid cache collision

            # === STEP 2: Preprocess FULL spectrum (matching search.py and Model Dev) ===
            if cache_key in preprocess_cache:
                X_train_preprocessed, X_val_preprocessed = preprocess_cache[cache_key]
            else:
                if use_ga_transform:
                    # GA closure handles per-spectrum operations only. raw
                    # chromosomes return None — treat as identity at this
                    # step; autoscale (when set) is applied below with a
                    # train-fitted scaler.
                    if ga_transform is not None:
                        X_train_preprocessed = ga_transform(X_train)
                        X_val_preprocessed = ga_transform(X_val)
                    else:
                        X_train_preprocessed = np.asarray(X_train, dtype=np.float64)
                        X_val_preprocessed = np.asarray(X_val, dtype=np.float64)

                    # The bug this branch was carrying: pre-fix the closure
                    # itself called StandardScaler().fit_transform(), so each
                    # call (one for train, one for val) refit the scaler on
                    # its own input and produced different scaler params.
                    # Validation features were centred to *val's* means,
                    # collapsing R²pred on every autoscale=True row. Fix:
                    # fit StandardScaler on TRAIN only, reuse on VAL.
                    if autoscale:
                        from sklearn.preprocessing import StandardScaler

                        _scaler = StandardScaler().fit(X_train_preprocessed)
                        X_train_preprocessed = _scaler.transform(X_train_preprocessed)
                        X_val_preprocessed = _scaler.transform(X_val_preprocessed)
                else:
                    # Build standard preprocessing pipeline
                    prep_steps = build_preprocessing_pipeline(
                        preprocess_name,
                        deriv=deriv,
                        window=window,
                        polyorder=poly,
                        baseline_method=baseline_method,
                        baseline_params=baseline_params,  # T-36 fix v2
                        smoothing=smoothing,
                        smoothing_window=smoothing_window,
                        smoothing_polyorder=smoothing_polyorder,
                        autoscale=autoscale,
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
            all_vars_str = row.get("all_vars", "N/A")
            if all_vars_str != "N/A" and all_vars_str and isinstance(all_vars_str, str):
                # Parse wavelengths from all_vars (e.g., "1520.0, 1540.0, 1560.0, ...")
                try:
                    model_wavelengths = [
                        float(w.strip()) for w in all_vars_str.split(",") if w.strip()
                    ]
                    # Create mapping from wavelength to column index
                    # CRITICAL: Do NOT sort - preserve the order from all_vars
                    wl_to_idx = {float(wl): idx_wl for idx_wl, wl in enumerate(wavelengths)}
                    # Get column indices for model wavelengths (in order)
                    col_indices = []
                    for wl in model_wavelengths:
                        if wl in wl_to_idx:
                            col_indices.append(wl_to_idx[wl])
                    if len(col_indices) != len(model_wavelengths):
                        print(
                            f"  [Warning] Only found {len(col_indices)}/{len(model_wavelengths)} wavelengths for model {i+1}"
                        )
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
                    print(
                        f"  [Warning] {len(col_indices) - len(valid_indices)} indices out of bounds for model {i+1}"
                    )
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
            # T-36 fix (post-merge review): pass the parsed autoscale flag so
            # _rebuild_model_from_row can skip its per-model StandardScaler when
            # the preprocessing pipeline already scaled the inputs.
            model = _rebuild_model_from_row(row, task_type, autoscale=autoscale)

            # Safety check: Skip if n_components > n_features (can happen with variable selection)
            if hasattr(model, "n_components") and model.n_components > X_train_final.shape[1]:
                print(
                    f"  [Warning] Skipping model {i+1}: n_components ({model.n_components}) > n_features ({X_train_final.shape[1]})"
                )
                continue

            # Apply class_weight discriminator before fit. Without this, XGBoost
            # rebuilt for validation trains UNWEIGHTED (sample_weight is fit-time,
            # not in Params) and PLS-DA trains UNWEIGHTED (LR sub-step's
            # class_weight is not in Params). See helper docstring.
            model_name = row.get("Model", "PLS")
            fit_kwargs = _apply_class_weight_discriminator_for_rebuilt_model(
                model,
                model_name,
                task_type,
                y_train,
                imbalance_method=imbalance_method,
            )

            # Fit on training data
            model.fit(X_train_final, y_train, **fit_kwargs)

            # Predict on validation data
            y_pred = model.predict(X_val_final)
            y_pred = np.ravel(y_pred)  # Ensure 1D for metrics

            # === STEP 5: Calculate metrics ===
            if task_type == "regression":
                rmsep = np.sqrt(mean_squared_error(y_val, y_pred))
                r2pred = r2_score(y_val, y_pred)
                df_results.loc[idx, "RMSEP"] = rmsep
                df_results.loc[idx, "R2pred"] = r2pred
            else:
                # Accuracy
                val_acc = accuracy_score(y_val, y_pred)
                df_results.loc[idx, "val_Accuracy"] = val_acc

                # Determine if binary or multiclass based on training data classes
                # Use 'macro' for multiclass to treat all classes equally (consistent with CV metrics)
                n_classes_train = len(np.unique(y_train))
                average_method = "binary" if n_classes_train == 2 else "macro"

                # F1 Score
                try:
                    val_f1 = f1_score(y_val, y_pred, average=average_method, zero_division=0)
                    df_results.loc[idx, "val_F1"] = val_f1
                except Exception as e:
                    # Fallback to weighted if binary fails
                    try:
                        val_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
                        df_results.loc[idx, "val_F1"] = val_f1
                    except Exception as e2:
                        print(f"  [Warning] Could not compute F1 for model {i+1}: {e2}")

                # Precision
                try:
                    val_precision = precision_score(
                        y_val, y_pred, average=average_method, zero_division=0
                    )
                    df_results.loc[idx, "val_Precision"] = val_precision
                except Exception as e:
                    try:
                        val_precision = precision_score(
                            y_val, y_pred, average="weighted", zero_division=0
                        )
                        df_results.loc[idx, "val_Precision"] = val_precision
                    except Exception as e2:
                        print(f"  [Warning] Could not compute Precision for model {i+1}: {e2}")

                # Recall
                try:
                    val_recall = recall_score(
                        y_val, y_pred, average=average_method, zero_division=0
                    )
                    df_results.loc[idx, "val_Recall"] = val_recall
                except Exception as e:
                    try:
                        val_recall = recall_score(
                            y_val, y_pred, average="weighted", zero_division=0
                        )
                        df_results.loc[idx, "val_Recall"] = val_recall
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
                            print(
                                f"  [Info] ROC AUC skipped - validation set has only 1 class (need at least 2)"
                            )
                    elif hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_val_final)
                        model_classes = (
                            model.classes_ if hasattr(model, "classes_") else np.unique(y_train)
                        )

                        # Always subset to classes present in validation
                        # This handles: binary, multiclass, and class-mismatch cases uniformly
                        col_indices = []
                        for c in val_classes:
                            matches = np.where(model_classes == c)[0]
                            if len(matches) > 0:
                                col_indices.append(matches[0])

                        if (
                            len(col_indices) == n_classes_val
                        ):  # All validation classes found in model
                            y_proba_subset = y_proba[:, col_indices]
                            # ALWAYS renormalize to sum to 1 (even for binary)
                            # This is needed when validation has fewer classes than training
                            y_proba_subset = y_proba_subset / y_proba_subset.sum(
                                axis=1, keepdims=True
                            )

                            if n_classes_val == 2:
                                # Binary: use probability of second class (positive)
                                val_roc_auc = roc_auc_score(y_val, y_proba_subset[:, 1])
                            else:
                                # Multiclass: compute OvR
                                val_roc_auc = roc_auc_score(
                                    y_val, y_proba_subset, multi_class="ovr", average="macro"
                                )

                            df_results.loc[idx, "val_ROC_AUC"] = val_roc_auc
                except Exception as e:
                    print(f"  [Warning] Could not compute ROC AUC for model {i+1}: {e}")

        except Exception as e:
            print(f"  [Warning] Failed to compute validation for model {i+1}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Progress update
        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(
                {
                    "stage": "validation_metrics",
                    "message": f"Computing validation metrics ({i+1}/{n_to_process})",
                    "current": i + 1,
                    "total": n_to_process,
                }
            )

    print(f"[Validation] Completed validation metrics for {n_to_process} models")

    # Reorder columns to place metrics in logical order:
    # Calibration metrics first, then validation metrics
    cols = list(df_results.columns)
    if task_type == "regression" and "RMSEP" in cols and "R2cv" in cols:
        # Move RMSEP and R2pred after R2cv
        cols.remove("RMSEP")
        cols.remove("R2pred")
        r2cv_idx = cols.index("R2cv")
        cols.insert(r2cv_idx + 1, "RMSEP")
        cols.insert(r2cv_idx + 2, "R2pred")
        df_results = df_results[cols]
    elif task_type == "classification":
        # Order: Accuracy, ROC_AUC, F1, Precision, Recall (calibration)
        #        val_Accuracy, val_ROC_AUC, val_F1, val_Precision, val_Recall (validation)
        cal_cols = ["Accuracy", "ROC_AUC", "F1", "Precision", "Recall"]
        val_cols = ["val_Accuracy", "val_ROC_AUC", "val_F1", "val_Precision", "val_Recall"]

        # Remove all metric columns that exist
        for col in cal_cols + val_cols:
            if col in cols:
                cols.remove(col)

        # Find insertion point (after Imbalance column, or after common metadata)
        if "Imbalance" in cols:
            insert_idx = cols.index("Imbalance") + 1
        elif "SubsetTag" in cols:
            insert_idx = cols.index("SubsetTag") + 1
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


def run_search(
    X,
    y,
    task_type,
    folds=5,
    cv_strategy="kfold",
    cv_n_repeats=5,
    excluded_count=0,
    validation_count=0,
    total_samples_original=None,
    variable_penalty=0,
    gap_penalty=0,
    max_n_components=10,
    max_iter=500,
    models_to_test=None,
    preprocessing_methods=None,
    interference_settings=None,
    window_sizes=None,
    n_estimators_list=None,
    learning_rates=None,
    neuralboosted_hidden_sizes=None,
    neuralboosted_activations=None,
    pls_max_iter_list=None,
    pls_tol_list=None,
    plsda_lr_C_list=None,
    plsda_lr_solver_list=None,
    plsda_lr_max_iter_list=None,
    rf_n_trees_list=None,
    rf_max_depth_list=None,
    rf_min_samples_split_list=None,
    rf_min_samples_leaf_list=None,
    rf_max_features_list=None,
    rf_bootstrap_list=None,
    rf_max_leaf_nodes_list=None,
    rf_min_impurity_decrease_list=None,
    ridge_alphas_list=None,
    ridge_solver_list=None,
    ridge_tol_list=None,
    lasso_alphas_list=None,
    lasso_selection_list=None,
    lasso_tol_list=None,
    xgb_n_estimators_list=None,
    xgb_learning_rates=None,
    xgb_max_depths=None,
    xgb_subsample=None,
    xgb_colsample_bytree=None,
    xgb_reg_alpha=None,
    xgb_reg_lambda=None,
    xgb_min_child_weight_list=None,
    xgb_gamma_list=None,
    elasticnet_alphas_list=None,
    elasticnet_l1_ratios=None,
    elasticnet_selection_list=None,
    elasticnet_tol_list=None,
    lightgbm_n_estimators_list=None,
    lightgbm_learning_rates=None,
    lightgbm_num_leaves_list=None,
    lightgbm_max_depth_list=None,
    lightgbm_min_child_samples_list=None,
    lightgbm_subsample_list=None,
    lightgbm_colsample_bytree_list=None,
    lightgbm_reg_alpha_list=None,
    lightgbm_reg_lambda_list=None,
    catboost_iterations_list=None,
    catboost_learning_rates=None,
    catboost_depths=None,
    catboost_l2_leaf_reg_list=None,
    catboost_border_count_list=None,
    catboost_bagging_temperature_list=None,
    catboost_random_strength_list=None,
    svr_kernels=None,
    svr_C_list=None,
    svr_gamma_list=None,
    svr_epsilon_list=None,
    svr_degree_list=None,
    svr_coef0_list=None,
    svr_shrinking_list=None,
    mlp_hidden_layer_sizes_list=None,
    mlp_alphas_list=None,
    mlp_learning_rate_inits=None,
    mlp_activation_list=None,
    mlp_solver_list=None,
    mlp_batch_size_list=None,
    mlp_learning_rate_schedule_list=None,
    mlp_momentum_list=None,
    enable_variable_subsets=True,
    variable_counts=None,
    enable_region_subsets=True,
    n_top_regions=10,
    region_test_all_individual=False,
    region_test_pairwise=False,
    progress_callback=None,
    variable_selection_methods=None,
    apply_uve_prefilter=False,
    uve_cutoff_multiplier=1.0,
    uve_n_components=None,
    ipls_n_intervals=20,
    ipls_max_combine=5,
    ipls_subset_limit="Top 10",
    sipls_n_combinations=500,
    mwpls_window_sizes=None,
    mwpls_step_size=None,
    tier="standard",
    enabled_models=None,
    analysis_wl_min=None,
    analysis_wl_max=None,
    analysis_wl_regions=None,  # List of (min, max) tuples for multi-region support
    imbalance_method=None,
    imbalance_params=None,
    enable_class_weight=False,
    ga_preprocess=False,
    ga_preprocess_cv_folds=5,
    ga_preprocess_autoscale=True,  # Phase 3: search both autoscale on/off
    ga_preprocess_phase2_n_seeds=5,  # Phase 2: 0 disables, 5 default
    ga_preprocess_phase2_max_pool_multiplier=8,
    ga_quick_mode=False,
    # Smart preprocessing discovery parameters (NEW - replaces GA)
    smart_preprocess=False,
    smart_preprocess_importance="model_specific",
    smart_preprocess_n_top=10,
    # TPE preprocessing discovery parameters (T-37 — supersedes smart + GA)
    tpe_preprocess=False,
    tpe_preprocess_n_trials=75,
    tpe_preprocess_n_top=10,
    tpe_enable_autoscale=True,
    tpe_multistart=False,  # Phase 4 (2026-05-06): multi-start + multi-seed rescore
    tpe_n_starts=5,
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
    # Autoscale (UV scaling) toggle — doubles preprocess_configs (T-36)
    autoscale=False,
    # Search control (pause/resume/stop)
    controller=None,
    # Validation metrics parameters
    X_validation=None,
    y_validation=None,
    compute_validation=False,
    validation_top_n=100,
    # Early stopping for boosting models
    early_stopping_rounds=40,
):
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

    # Use all cores for parallel execution. Python 3.11 frozen builds need
    # cpu_count() because loky's process spawn is broken there; 3.12 frozen
    # builds work like dev mode (n_jobs=-1).
    n_jobs_default = (os.cpu_count() or -1) if _frozen_needs_threading_fallback() else -1

    # Drop rows where y is NaN (safety net for data with empty rows)
    nan_mask = y.isna()
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(f"Warning: Dropping {n_dropped} sample(s) with NaN target values before analysis.")
        X = X[~nan_mask]
        y = y[~nan_mask]

    # Auto-mode imbalance resolution: classify the run-level y once and
    # substitute 'class_weight' or None for the rest of the search. Resolution
    # is run-level (not per-fold) — stratified CV preserves global ratios
    # tightly enough that per-fold drift rarely flips the decision. Audit
    # message goes through both logger.info (T-45 file handler) and stdout
    # (console runs). NaN-dropping happens inside resolve_auto_imbalance.
    if imbalance_method == "auto" and task_type == "classification":
        from spectral_predict.imbalance import resolve_auto_imbalance, format_auto_imbalance_message

        resolved, info = resolve_auto_imbalance(y.values, task_type=task_type)
        message = format_auto_imbalance_message(info)
        logger.info(message)
        print(f"  {message}")
        imbalance_method = resolved

    X_np = X.values
    y_np = y.values
    wavelengths = X.columns.values
    n_features = X_np.shape[1]
    n_samples = X_np.shape[0]

    # Handle categorical labels for classification
    label_encoder = None
    if task_type == "classification":
        # Check if labels are non-numeric (text labels like "low", "medium", "high")
        if not pd.api.types.is_numeric_dtype(y_np.dtype):
            from sklearn.preprocessing import LabelEncoder

            label_encoder = LabelEncoder()
            y_original = y_np.copy()  # Keep original for logging
            y_np = label_encoder.fit_transform(y_np)
            # Log the label mapping
            label_mapping = dict(
                zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
            )
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
    if task_type == "classification" and imbalance_method is not None:
        from .imbalance import validate_classification_config

        try:
            validate_classification_config(
                y=y_np,
                imbalance_method=imbalance_method,
                imbalance_params=imbalance_params,
                n_folds=folds,
            )
            print(
                f"[OK] Imbalance configuration validated: {imbalance_method} with {folds}-fold CV"
            )
        except ValueError as e:
            # Re-raise with clear indication this is an upfront validation error
            raise ValueError(f"Configuration Error (detected before training):\n\n{e}") from None

    # Backend guard for CV strategy vs class distribution (runs regardless of
    # imbalance_method — GUI can't be the only layer of defense for scripted
    # callers that bypass the GUI).
    from .cv_utils import validate_cv_strategy_for_task

    try:
        validate_cv_strategy_for_task(
            strategy=cv_strategy,
            task_type=task_type,
            y=y_np if task_type == "classification" else np.asarray(y),
            n_folds=folds,
            n_repeats=cv_n_repeats,
        )
    except ValueError as e:
        raise ValueError(f"Configuration Error (detected before training):\n\n{e}") from None

    # Create results container
    df_results = create_results_dataframe(task_type)

    # Handle variable selection methods (support multiple methods)
    if variable_selection_methods is None or not variable_selection_methods:
        variable_selection_methods = ["importance"]

    # Filter to only implemented methods
    implemented_methods = [
        "importance",
        "spa",
        "uve",
        "uve_spa",
        "ipls",
        "ipls_forward",
        "ipls_backward",
        "mc_sipls",
        "mwpls",
        "cars",
        "cars-aware",
        "cars-tree",
        "vcpa-iriv",
        "ga",
        "uve_cars",
        "uve_cars_tree",
        "uve_cars_spa",
        "fipls_spa",
        "fipls_cars",
    ]
    selected_methods = [m for m in variable_selection_methods if m in implemented_methods]

    # If UVE-hybrid variant is selected alongside base method, drop the base (hybrid subsumes it)
    if "uve_cars" in selected_methods and "cars" in selected_methods:
        selected_methods.remove("cars")
        print("Info: Removed 'cars' — 'uve_cars' includes CARS with UVE pre-filtering")
    if "uve_cars_tree" in selected_methods and "cars-tree" in selected_methods:
        selected_methods.remove("cars-tree")
        print(
            "Info: Removed 'cars-tree' — 'uve_cars_tree' includes CARS-Tree with UVE pre-filtering"
        )

    # Warn about unimplemented methods
    unimplemented = [m for m in variable_selection_methods if m not in implemented_methods]
    if unimplemented:
        print(f"Info: Variable selection methods {unimplemented} are not yet implemented.")
        print(f"      Continuing with implemented methods: {selected_methods}")

    # Ensure at least one method is selected
    if not selected_methods:
        selected_methods = ["importance"]
        print("Info: No implemented methods selected. Defaulting to 'importance'.")
    if ipls_n_intervals != 20:
        print("Info: iPLS interval parameter is noted but not yet applied in the Python backend.")

    # Determine if classification is binary or multiclass
    is_binary_classification = False
    if task_type == "classification":
        n_classes = len(np.unique(y_np))
        is_binary_classification = n_classes == 2

    # Adjust max_n_components based on CV training fold size.
    # T-10: cv_strategy-aware. K-fold/RepeatedKFold use the exact
    # floor `n_samples * (folds - 1) // folds`. LOO has train-fold size n-1.
    # Group splitters are not yet supported (T-15 follow-up).
    # For REGRESSION: PLS requires n_components <= min(n_features, n_samples_train_fold)
    # For CLASSIFICATION: PLS-DA uses PLS as dimensionality reduction before LR classifier,
    #                     so we can be less strict (LR can handle more components than samples)
    if n_samples >= 2:
        from .cv_utils import compute_min_train_fold_size

        min_train_samples = compute_min_train_fold_size(
            cv_strategy=cv_strategy,
            n_samples=n_samples,
            n_folds=folds,
        )
    else:
        min_train_samples = 0

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
            print(
                f"Note: Using {max_n_components} PLS components with min_train_size~{min_train_samples}. "
                + f"This is acceptable for PLS-DA (classification) but may cause instability."
            )

    if safe_max_components < max_n_components:
        print(
            f"Note: Reducing max components from {max_n_components} to {safe_max_components} "
            + f"due to dataset constraints (n_samples={n_samples}, n_features={n_features}, "
            + f"min_train_size~{min_train_samples}, task={task_type})"
        )

    # Get model grids (pass n_estimators_list and learning_rates for NeuralBoosted,
    # rf_n_trees_list and rf_max_depth_list for RandomForest,
    # ridge_alphas_list and lasso_alphas_list for Ridge and Lasso,
    # xgb_* for XGBoost, elasticnet_* for ElasticNet, lightgbm_* for LightGBM, etc.,
    # tier for tiered defaults, and enabled_models for custom model selection)
    model_grids = get_model_grids(
        task_type,
        n_features,
        safe_max_components,
        max_iter,
        n_estimators_list=n_estimators_list,
        learning_rates=learning_rates,
        neuralboosted_hidden_sizes=neuralboosted_hidden_sizes,
        neuralboosted_activations=neuralboosted_activations,
        pls_max_iter_list=pls_max_iter_list,
        pls_tol_list=pls_tol_list,
        plsda_lr_C_list=plsda_lr_C_list,
        plsda_lr_solver_list=plsda_lr_solver_list,
        plsda_lr_max_iter_list=plsda_lr_max_iter_list,
        rf_n_trees_list=rf_n_trees_list,
        rf_max_depth_list=rf_max_depth_list,
        rf_min_samples_split_list=rf_min_samples_split_list,
        rf_min_samples_leaf_list=rf_min_samples_leaf_list,
        rf_max_features_list=rf_max_features_list,
        rf_bootstrap_list=rf_bootstrap_list,
        rf_max_leaf_nodes_list=rf_max_leaf_nodes_list,
        rf_min_impurity_decrease_list=rf_min_impurity_decrease_list,
        ridge_alphas_list=ridge_alphas_list,
        ridge_solver_list=ridge_solver_list,
        ridge_tol_list=ridge_tol_list,
        lasso_alphas_list=lasso_alphas_list,
        lasso_selection_list=lasso_selection_list,
        lasso_tol_list=lasso_tol_list,
        xgb_n_estimators_list=xgb_n_estimators_list,
        xgb_learning_rates=xgb_learning_rates,
        xgb_max_depths=xgb_max_depths,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
        xgb_reg_alpha=xgb_reg_alpha,
        xgb_reg_lambda=xgb_reg_lambda,
        xgb_min_child_weight_list=xgb_min_child_weight_list,
        xgb_gamma_list=xgb_gamma_list,
        elasticnet_alphas_list=elasticnet_alphas_list,
        elasticnet_l1_ratios=elasticnet_l1_ratios,
        elasticnet_selection_list=elasticnet_selection_list,
        elasticnet_tol_list=elasticnet_tol_list,
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
        catboost_learning_rates=catboost_learning_rates,
        catboost_depths=catboost_depths,
        catboost_l2_leaf_reg_list=catboost_l2_leaf_reg_list,
        catboost_border_count_list=catboost_border_count_list,
        catboost_bagging_temperature_list=catboost_bagging_temperature_list,
        catboost_random_strength_list=catboost_random_strength_list,
        svr_kernels=svr_kernels,
        svr_C_list=svr_C_list,
        svr_gamma_list=svr_gamma_list,
        svr_epsilon_list=svr_epsilon_list,
        svr_degree_list=svr_degree_list,
        svr_coef0_list=svr_coef0_list,
        svr_shrinking_list=svr_shrinking_list,
        mlp_hidden_layer_sizes_list=mlp_hidden_layer_sizes_list,
        mlp_alphas_list=mlp_alphas_list,
        mlp_learning_rate_inits=mlp_learning_rate_inits,
        mlp_activation_list=mlp_activation_list,
        mlp_solver_list=mlp_solver_list,
        mlp_batch_size_list=mlp_batch_size_list,
        mlp_learning_rate_schedule_list=mlp_learning_rate_schedule_list,
        mlp_momentum_list=mlp_momentum_list,
        tier=tier,
        enabled_models=enabled_models,
        n_jobs=n_jobs_default,
    )

    # Filter models if models_to_test is specified
    if models_to_test is not None:
        # Filter to only requested models
        model_grids = {
            name: configs for name, configs in model_grids.items() if name in models_to_test
        }

        if not model_grids:
            raise ValueError(
                f"No valid models found. Available: {list(get_model_grids(task_type, n_features, safe_max_components, max_iter).keys())}, Requested: {models_to_test}"
            )

    # Define preprocessing configurations based on user selections
    # Use preprocessing_methods dict if provided, otherwise default to all
    if preprocessing_methods is None:
        preprocessing_methods = {
            "raw": True,
            "snv": True,
            "sg1": True,
            "sg2": True,
            "sg3": False,  # Higher-order derivatives not default
            "sg4": False,  # Higher-order derivatives not default
            "deriv_snv": True,
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
        if interference.get("msc", False):
            return True
        if interference.get("wavelength_exclusion", {}).get("enabled", False):
            return True
        if interference.get("osc", {}).get("enabled", False):
            return True

        # Check advanced methods
        advanced = interference.get("advanced", {})
        if isinstance(advanced, dict):
            if advanced.get("epo", {}).get("enabled", False):
                return True
            if advanced.get("dosc", {}).get("enabled", False):
                return True
            if advanced.get("glsw", {}).get("enabled", False):
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

    # T-37 fix (post-merge review): explicit mutual-exclusion guard for
    # preprocessing-discovery flags. The per-branch gates below all use
    # `flag and not other_flag` shorthand, which silently drops into
    # normal-preprocessing fallback when callers (tests, scripts) accidentally
    # pass two flags True. Raise here so the caller learns about it.
    _discovery_flags = sum(bool(f) for f in (smart_preprocess, tpe_preprocess, ga_preprocess))
    if _discovery_flags > 1:
        raise ValueError(
            "smart_preprocess, tpe_preprocess, and ga_preprocess are mutually "
            "exclusive — set at most one to True"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SMART PREPROCESSING DISCOVERY (NEW - replaces GA preprocessing)
    # Uses NSGA-II-style importance-guided wavelength selection
    # ═══════════════════════════════════════════════════════════════════════════
    if smart_preprocess and not tpe_preprocess:
        if progress_callback:
            progress_callback(
                {
                    "stage": "smart_preprocessing",
                    "message": "Discovering optimal preprocessing configurations...",
                    "current": 0,
                    "total": 62,  # Approximate number of combinations
                }
            )

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
                progress_callback(
                    {
                        "stage": "smart_preprocessing",
                        "message": message,
                        "current": current,
                        "total": total,
                    }
                )

        # Run smart preprocessing discovery
        discovered_configs = discover_preprocessing(
            X.values,  # Convert DataFrame to numpy
            y.values,  # Convert Series to numpy
            models_to_test=models_to_test,
            task_type=task_type,
            importance_method=smart_preprocess_importance,
            n_top=smart_preprocess_n_top,
            cv_folds=folds,
            progress_callback=discovery_progress,
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
                base_name = cfg["preprocessing"]
                window = cfg.get("window")
                deriv = cfg.get("deriv")

                # Determine base name for pipeline builder
                if base_name in ("raw", "snv"):
                    pipeline_name = base_name
                elif base_name.startswith("snv_deriv"):
                    pipeline_name = "snv_deriv"
                elif base_name.endswith("_snv"):
                    pipeline_name = "deriv_snv"
                elif base_name.startswith("deriv"):
                    pipeline_name = "deriv"
                else:
                    pipeline_name = base_name

                display_name = pipeline_name
                model_name = cfg.get("model_name")

                preprocess_configs.append(
                    {
                        "name": display_name,
                        "base_name": pipeline_name,
                        "deriv": deriv,
                        "window": window,
                        "polyorder": cfg.get("polyorder"),
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                        # Smart preprocessing specific fields
                        "smart_selected_wavelengths": cfg.get("selected_wavelengths"),
                        "smart_n_wavelengths": cfg.get("n_wavelengths"),
                        "smart_score": cfg.get("score"),
                        "smart_importance_method": cfg.get("importance_method"),
                        "smart_model_name": model_name,  # Which model this was optimized for
                    }
                )

            print(
                f"\nCreated {len(preprocess_configs)} preprocessing configurations for grid search"
            )
            print(f"{'='*70}\n")

            # Skip normal preprocessing config building AND old GA preprocessing
            skip_normal_preprocessing = True
            ga_preprocess = False  # Disable old GA since we're using smart preprocessing

    # ═══════════════════════════════════════════════════════════════════════════
    # TPE PREPROCESSING DISCOVERY (T-37 — supersedes smart + GA)
    # Uses Optuna TPE to search a 5-D space (preproc × window × autoscale ×
    # baseline × smoothing) with a LightGBM proxy.  Returns top-N diverse
    # configs tested against ALL enabled models — preserves model diversity.
    # ═══════════════════════════════════════════════════════════════════════════
    if tpe_preprocess and not smart_preprocess:
        if progress_callback:
            progress_callback(
                {
                    "stage": "tpe_preprocessing",
                    "message": "TPE preprocessing discovery...",
                    "current": 0,
                    "total": tpe_preprocess_n_trials,
                }
            )

        print(f"\n{'='*70}")
        print("TPE PREPROCESSING DISCOVERY")
        print(f"{'='*70}")
        print(f"  Trials: {tpe_preprocess_n_trials}")
        print(f"  Top-N configs: {tpe_preprocess_n_top}")
        print(f"  CV folds: {folds}")
        print(f"  Task type: {task_type}")
        print(f"{'='*70}\n")

        from .tpe_preprocessing_discovery import (
            run_tpe_preprocessing_discovery,
            run_tpe_multistart_preprocessing_discovery,
            resolve_tpe_proxy_family,
        )

        def tpe_progress(current, total, message):
            if progress_callback:
                progress_callback(
                    {
                        "stage": "tpe_preprocessing",
                        "message": message,
                        "current": current,
                        "total": total,
                    }
                )

        # Pick proxy family from the user's enabled-models list. Tree-only
        # downstream → tree proxy (LightGBM with adaptive min_child_samples);
        # anything else → linear proxy (PLS / LogReg). See resolve_tpe_proxy_family
        # docstring for the mixed/unknown rule.
        tpe_proxy_family = resolve_tpe_proxy_family(models_to_test)
        print(f"  TPE proxy family: {tpe_proxy_family} (resolved from models_to_test={models_to_test})")

        # When tpe_multistart=True, run M independent TPE studies and rescore
        # the union with multi-seed CV. Closes the TPE-drift problem documented
        # in tools/bayesian_topk_stability.py. Both call sites
        # (regression/classification here and the one_class call site in
        # run_one_class_search) gate on the same flag.
        if tpe_multistart:
            discovered_configs = run_tpe_multistart_preprocessing_discovery(
                X.values,
                y.values,
                task_type=task_type,
                n_trials=tpe_preprocess_n_trials,
                n_top=tpe_preprocess_n_top,
                cv_folds=folds,
                enable_autoscale=tpe_enable_autoscale,
                enable_baseline=(baseline_method is not None),
                enable_smoothing=smoothing,
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder,
                n_starts=tpe_n_starts,
                progress_callback=tpe_progress,
                controller=controller,
                proxy_family=tpe_proxy_family,
            )
        else:
            discovered_configs = run_tpe_preprocessing_discovery(
                X.values,
                y.values,
                task_type=task_type,
                n_trials=tpe_preprocess_n_trials,
                n_top=tpe_preprocess_n_top,
                cv_folds=folds,
                enable_autoscale=tpe_enable_autoscale,
                enable_baseline=(baseline_method is not None),
                enable_smoothing=smoothing,
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder,
                progress_callback=tpe_progress,
                proxy_family=tpe_proxy_family,
            )

        if not discovered_configs:
            print("WARNING: TPE preprocessing discovery found no valid configs!")
            print("Falling back to default preprocessing...")
            tpe_preprocess = False
        else:
            preprocess_configs = []
            for cfg in discovered_configs:
                base_name = cfg["preprocessing"]
                window = cfg.get("window")
                deriv = cfg.get("deriv")

                if base_name in ("raw", "snv"):
                    pipeline_name = base_name
                elif base_name.startswith("snv_deriv"):
                    pipeline_name = "snv_deriv"
                elif base_name.endswith("_snv"):
                    pipeline_name = "deriv_snv"
                elif base_name.startswith("deriv"):
                    pipeline_name = "deriv"
                else:
                    pipeline_name = base_name

                display_name = base_name
                if window:
                    display_name += f"_w{window}"
                if cfg.get("_tpe_baseline_method"):
                    display_name = f"{cfg['_tpe_baseline_method']}+{display_name}"
                if cfg.get("_tpe_smoothing"):
                    display_name = f"sg0+{display_name}"
                if cfg.get("_tpe_autoscale"):
                    display_name = f"{display_name}+autoscale"

                preprocess_configs.append(
                    {
                        "name": display_name,
                        "base_name": pipeline_name,
                        "deriv": deriv,
                        "window": window,
                        "polyorder": cfg.get("polyorder"),
                        "interference": interference_to_add,
                        "baseline_method": cfg.get("_tpe_baseline_method"),
                        "baseline_params": cfg.get("_tpe_baseline_params"),
                        "smoothing": cfg.get("_tpe_smoothing", False),
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                        "autoscale": cfg.get("_tpe_autoscale", False),
                        "tpe_score": cfg.get("score"),
                        # Phase 4 fix (2026-05-06): propagate multistart halt
                        # reason through to result rows so users can see
                        # whether the rescore converged. Single-start TPE
                        # configs don't carry this key; default to None.
                        "tpe_multistart_halt_reason": cfg.get(
                            "_tpe_multistart_halt_reason"
                        ),
                        # 2026-05-08: model-family-aware proxy audit trail.
                        # Records which family/model the TPE proxy used so
                        # the result CSV captures whether ranking came from
                        # a tree (LightGBM) or linear (PLS/LogReg) proxy.
                        "tpe_proxy_family": cfg.get("_tpe_proxy_family"),
                        "tpe_proxy_model_name": cfg.get("_tpe_proxy_model_name"),
                    }
                )

            print(
                f"\nCreated {len(preprocess_configs)} preprocessing configurations for grid search"
            )
            print(f"{'='*70}\n")

            skip_normal_preprocessing = True
            ga_preprocess = False
            smart_preprocess = False
            baseline_method = None  # TPE configs already have per-config baseline
            autoscale = False  # TPE configs already have per-config autoscale
            smoothing = False  # TPE configs already have per-config smoothing

    # ═══════════════════════════════════════════════════════════════════════════
    # GA PREPROCESSING OPTIMIZATION (LEGACY - kept for backward compatibility)
    # When enabled, this REPLACES user-selected preprocessing with GA-optimized config
    # ═══════════════════════════════════════════════════════════════════════════
    if ga_preprocess and not smart_preprocess and not tpe_preprocess:
        if progress_callback:
            progress_callback(
                {
                    "stage": "exhaustive_preprocessing",
                    "message": "Optimizing preprocessing parameters (exhaustive search)...",
                    "current": 0,
                    "total": 238,  # 14 preproc x 17 windows
                }
            )

        print(f"\n{'='*70}")
        print("EXHAUSTIVE PREPROCESSING OPTIMIZATION")
        print(f"{'='*70}")
        print(f"  Search space: 238 combinations (14 preproc x 17 windows)")
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
        print(
            "Running EXHAUSTIVE optimization per-model with actual hyperparameters..."
        )
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
                progress_callback(
                    {
                        "algorithm": "preprocessing_optimization",
                        "current": 0,
                        "total": len(models_for_ga),
                        "message": f"Optimizing preprocessing for {model_name} ({model_idx}/{len(models_for_ga)})...",
                    }
                )

            print(f"Optimizing preprocessing for {model_name}...")

            # Determine which proxy fitness model to use as fallback
            if model_name.lower() in ["pls", "pls-da", "ridge", "lasso", "elasticnet"]:
                fitness_model = "pls"
            elif model_name.lower() in ["lightgbm", "xgboost", "catboost", "randomforest"]:
                fitness_model = "lightgbm"
            elif model_name.lower() in ["mlp", "svr", "svc"]:
                fitness_model = "mlp"
            elif model_name.lower() == "neuralboosted":
                fitness_model = "neuralboosted"
            else:
                fitness_model = "pls"  # Default

            # Get first hyperparameter set for this model (for actual model evaluation)
            # model_grids is dict mapping model_name -> list of (model_instance, params_dict)
            first_params = {}
            if model_name in model_grids and model_grids[model_name]:
                # Extract params from first config tuple: (model_instance, params_dict)
                first_params = (
                    model_grids[model_name][0][1] if len(model_grids[model_name][0]) > 1 else {}
                )

            # Build model_config for actual model evaluation
            model_config = {"name": model_name, "params": first_params}

            # Run exhaustive optimization with actual model evaluation
            ga_result = optimize_preprocessing(
                X.values,  # Convert DataFrame to numpy
                y.values,  # Convert Series to numpy
                method="exhaustive",
                cv_folds=folds,  # Use same CV folds as main search
                n_components=safe_max_components,  # Match grid search components
                task_type=task_type,
                random_state=random_state,
                verbose=1,
                progress_callback=progress_callback,
                fitness_model=fitness_model,  # Fallback if model_config fails
                top_n=5,  # Return top 5 preprocessing configs
                n_jobs=-1,  # Always parallel (was conditional on legacy GA mode)
                model_config=model_config,  # Use actual model for fitness evaluation
                apply_autoscale=ga_preprocess_autoscale,  # Phase 3 autoscale gene
                phase2_n_seeds=ga_preprocess_phase2_n_seeds,  # Phase 2 multi-seed rescore
                phase2_max_pool_multiplier=ga_preprocess_phase2_max_pool_multiplier,
            )

            ga_results[model_name] = ga_result
            print(f"  {model_name} optimization complete!")
            print(f"  Best config: {ga_result['best_config']}")
            if task_type == "classification":
                # For classification, fitness is accuracy (positive, higher = better)
                best_fitness = ga_result["configs"][0]["fitness"] if ga_result.get("configs") else 0
                print(f"  Best Accuracy: {best_fitness:.4f}")
            else:
                # For regression, best_rmsecv is already the RMSECV value
                print(f"  Best RMSECV: {ga_result['best_rmsecv']:.4f}")
            print(f"  Returning top {len(ga_result.get('configs', []))} configs\n")

            # Send completion update to GUI after this model finishes
            if progress_callback:
                model_idx = models_for_ga.index(model_name) + 1
                best_score = ga_result["configs"][0]["fitness"] if ga_result.get("configs") else 0
                if task_type == "classification":
                    score_str = f"Best Accuracy: {best_score:.4f}"
                else:
                    score_str = f"Best RMSECV: {ga_result['best_rmsecv']:.4f}"
                progress_callback(
                    {
                        "algorithm": "preprocessing_optimization",
                        "current": model_idx,
                        "total": len(models_for_ga),
                        "message": f"  ✓ {model_name} preprocessing complete - {score_str}",
                    }
                )

        # Create preprocessing configs from all GA results
        # Each model contributes its top-N preprocessing configs
        preprocess_configs = []

        for model_name, ga_result in ga_results.items():
            configs_list = ga_result.get("configs", [])
            if not configs_list:
                # Fallback for backward compatibility (shouldn't happen with new code)
                configs_list = [
                    {
                        "genes": ga_result["best_genes"],
                        "name": ga_result["best_name"],
                        "transform": ga_result["best_transform"],
                        "config": ga_result["best_config"],
                        "deriv": None,
                        "window": None,
                        "polyorder": None,
                    }
                ]

            # Add all top-N configs for this model
            for i, cfg in enumerate(configs_list):
                base_name = cfg.get("name", "unknown")
                # Strip the optional "+autoscale" tag from the chromosome name
                # before deriving the pipeline base_name; the autoscale flag
                # is propagated separately via the dict's "autoscale" key.
                if base_name.endswith("+autoscale"):
                    base_name = base_name[: -len("+autoscale")]
                # Clean display name: strip derivative order
                if base_name in ("raw", "snv"):
                    clean_name = base_name
                elif base_name.startswith("snv_deriv"):
                    clean_name = "snv_deriv"
                elif base_name.endswith("_snv"):
                    clean_name = "deriv_snv"
                elif base_name.startswith("deriv"):
                    clean_name = "deriv"
                else:
                    clean_name = base_name

                # Decode autoscale gene from chromosome (Phase 3, 2026-05-06).
                # Backward-compat: 2-gene chromosomes from saved CSVs and from
                # ga_preprocess_autoscale=False runs read as autoscale=False.
                genes = cfg.get("genes")
                from .ga_preprocessing import _decode_autoscale_gene
                autoscale_gene = (
                    _decode_autoscale_gene(genes)
                    if genes is not None
                    else False
                )

                preprocess_configs.append(
                    {
                        "name": clean_name,
                        "base_name": base_name,  # Base name for build_preprocessing_pipeline
                        "deriv": cfg.get("deriv"),
                        "window": cfg.get("window"),
                        "polyorder": cfg.get("polyorder"),
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                        "autoscale": autoscale_gene,  # Phase 3 autoscale dimension
                        # Phase 2 (2026-05-06): which path produced this config?
                        # 'converged' / 'cap' / 'single_iteration' / 'disabled'.
                        # Per-config because results CSV is row-shaped, but the
                        # value is per-search (all configs from one ga_result
                        # share the same halt_reason).
                        "phase2_halt_reason": ga_result.get("phase2_halt_reason", "disabled"),
                        "ga_transform": cfg.get("transform"),
                        "ga_config": cfg.get("config"),
                        "ga_model_type": model_name,  # Track which model this was optimized for
                        # Renamed from "ga_genes" 2026-05-06: collision with the
                        # GUI Refine tab's wavelength-index field of the same
                        # name. preprocess_chromosome is the dasp-internal name
                        # for [preproc_idx, window_idx] (legacy 2-gene) or
                        # [preproc_idx, window_idx, autoscale] (Phase 3).
                        "preprocess_chromosome": genes,
                    }
                )

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
        if preprocessing_methods.get("raw", False):
            preprocess_configs.append(
                {
                    "name": "raw",
                    "deriv": None,
                    "window": None,
                    "polyorder": None,
                    "interference": interference_to_add,  # Phase 3: Add interference settings only if enabled
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder,
                }
            )

        # Add SNV if selected
        if preprocessing_methods.get("snv", False):
            preprocess_configs.append(
                {
                    "name": "snv",
                    "deriv": None,
                    "window": None,
                    "polyorder": None,
                    "interference": interference_to_add,  # Phase 3: Add interference settings only if enabled
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder,
                }
            )

        # Add derivative configs based on user selections
        # For each derivative type, we create:
        # 1. Pure derivative (deriv)
        # 2. SNV then derivative (snv_deriv) - if SNV is also selected
        # 3. Derivative then SNV (deriv_snv) - if deriv_snv checkbox is selected

        if preprocessing_methods.get("sg1", False):
            # 1st derivative only
            for window in window_sizes:
                preprocess_configs.append(
                    {
                        "name": "deriv",
                        "deriv": 1,
                        "window": window,
                        "polyorder": 2,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                    }
                )

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get("snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "snv_deriv",
                            "deriv": 1,
                            "window": window,
                            "polyorder": 2,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

            # If deriv_snv is selected, add derivative -> SNV combination for 1st deriv
            if preprocessing_methods.get("deriv_snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "deriv_snv",
                            "deriv": 1,
                            "window": window,
                            "polyorder": 2,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

        if preprocessing_methods.get("sg2", False):
            # 2nd derivative only
            for window in window_sizes:
                preprocess_configs.append(
                    {
                        "name": "deriv",
                        "deriv": 2,
                        "window": window,
                        "polyorder": 3,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                    }
                )

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get("snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "snv_deriv",
                            "deriv": 2,
                            "window": window,
                            "polyorder": 3,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

            # If deriv_snv is selected, add derivative -> SNV combination for 2nd deriv
            if preprocessing_methods.get("deriv_snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "deriv_snv",
                            "deriv": 2,
                            "window": window,
                            "polyorder": 3,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

        if preprocessing_methods.get("sg3", False):
            # 3rd derivative only
            for window in window_sizes:
                preprocess_configs.append(
                    {
                        "name": "deriv",
                        "deriv": 3,
                        "window": window,
                        "polyorder": 4,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                    }
                )

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get("snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "snv_deriv",
                            "deriv": 3,
                            "window": window,
                            "polyorder": 4,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

            # If deriv_snv is selected, add derivative -> SNV combination for 3rd deriv
            if preprocessing_methods.get("deriv_snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "deriv_snv",
                            "deriv": 3,
                            "window": window,
                            "polyorder": 4,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

        if preprocessing_methods.get("sg4", False):
            # 4th derivative only
            for window in window_sizes:
                preprocess_configs.append(
                    {
                        "name": "deriv",
                        "deriv": 4,
                        "window": window,
                        "polyorder": 5,
                        "interference": interference_to_add,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                    }
                )

            # If SNV is also selected, add SNV -> derivative combination
            if preprocessing_methods.get("snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "snv_deriv",
                            "deriv": 4,
                            "window": window,
                            "polyorder": 5,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

            # If deriv_snv is selected, add derivative -> SNV combination for 4th deriv
            if preprocessing_methods.get("deriv_snv", False):
                for window in window_sizes:
                    preprocess_configs.append(
                        {
                            "name": "deriv_snv",
                            "deriv": 4,
                            "window": window,
                            "polyorder": 5,
                            "interference": interference_to_add,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )

        # If no preprocessing methods selected, default to raw
        if not preprocess_configs:
            print("Warning: No preprocessing methods selected. Defaulting to raw.")
            preprocess_configs.append(
                {
                    "name": "raw",
                    "deriv": None,
                    "window": None,
                    "polyorder": None,
                    "interference": interference_to_add,
                    "baseline_method": baseline_method,
                    "baseline_params": baseline_params,
                    "smoothing": smoothing,
                    "smoothing_window": smoothing_window,
                    "smoothing_polyorder": smoothing_polyorder,
                }
            )

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

    # --- Autoscale (UV scaling) toggle: when enabled, test both WITH and WITHOUT autoscale ---
    if autoscale and preprocess_configs:
        configs_without_autoscale = []
        configs_with_autoscale = []
        for cfg in preprocess_configs:
            cfg_no = dict(cfg)
            cfg_no["autoscale"] = False
            configs_without_autoscale.append(cfg_no)
            cfg_sc = dict(cfg)
            cfg_sc["autoscale"] = True
            cfg_sc["base_name"] = cfg.get("base_name", cfg["name"])
            cfg_sc["name"] = cfg["name"] + "+autoscale"
            configs_with_autoscale.append(cfg_sc)
        preprocess_configs = configs_without_autoscale + configs_with_autoscale

    # Create CV splitter via factory (supports kfold/repeated_kfold/loo)
    cv_splitter = build_cv_splitter(
        strategy=cv_strategy,
        n_folds=folds,
        task_type=task_type,
        n_repeats=cv_n_repeats,
        random_state=random_state,
    )

    print(
        f"Running {task_type} search with {cv_strategy} CV (folds={folds}, repeats={cv_n_repeats})..."
    )
    print(f"Models: {list(model_grids.keys())}")
    print(f"Preprocessing configs: {len(preprocess_configs)}")
    print(f"\nPreprocessing breakdown:")
    for cfg in preprocess_configs:
        cfg_name = cfg["name"]
        if cfg["deriv"] is not None:
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

    # ═══════════════════════════════════════════════════════════════════════════
    # VARIABLE SELECTION CACHE
    # Most variable selection methods (UVE, SPA, iPLS, CARS, GA-PLS) build their
    # own internal models and produce results independent of the outer model being
    # tested. Caching avoids redundant computation across models and hyperparameter
    # combos. The 'importance' method is NEVER cached (depends on fitted model).
    # ═══════════════════════════════════════════════════════════════════════════
    import threading as _threading

    _varsel_cache: dict = {}
    _varsel_cache_lock = _threading.Lock()

    # Determine model category for methods that distinguish tree vs linear
    LINEAR_MODELS_SET = set(LINEAR_MODELS)
    TREE_MODELS_SET = set(TREE_MODELS)

    def _varsel_cache_key(
        preprocess_cfg: dict,
        varsel_method: str,
        model_name: str,
        uve_prefilter_active: bool = False,
    ) -> tuple:
        """Build a hashable cache key for variable selection results.

        For model-independent methods (uve, spa, ipls, cars, ga, etc.), the key
        does not include model_name. For model-category-dependent methods
        (cars-aware, cars-tree, uve_cars_tree, ga), it includes the category.
        """
        # Canonical preprocessing hash: only fields that affect output.
        # Note: 'smoothing' need not be a separate field — the smoothing-doubling
        # block already encodes it in the 'name' (sg0+ prefix), so name discriminates.
        # 'autoscale' is encoded in 'name' too, but is added here defensively (and
        # because Bayesian-path Optuna trials toggle autoscale without renaming).
        prep_parts = (
            preprocess_cfg.get("name", ""),
            preprocess_cfg.get("base_name", ""),
            preprocess_cfg.get("deriv", 0),
            preprocess_cfg.get("window", 0),
            preprocess_cfg.get("polyorder", 0),
            str(preprocess_cfg.get("baseline_method", "")),
            preprocess_cfg.get("autoscale", False),  # T-36
        )

        # Methods that depend on model category (tree vs linear)
        if varsel_method in ("cars-aware", "cars-tree", "uve_cars_tree"):
            category = "tree" if model_name in TREE_MODELS_SET else "linear"
            return (prep_parts, varsel_method, category, uve_prefilter_active)
        elif varsel_method == "ga":
            # GA uses ga_pls for linear, ga_lightgbm for tree
            category = "tree" if model_name in TREE_MODELS_SET else "linear"
            return (prep_parts, varsel_method, category, uve_prefilter_active)
        else:
            # Model-independent: uve, spa, ipls, cars, uve_spa, etc.
            return (prep_parts, varsel_method, uve_prefilter_active)

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
        wavelengths = X.columns.astype(float).values if hasattr(X, "columns") else None

        # Check if this is a GA-optimized preprocessing config
        if "ga_transform" in preprocess_cfg and preprocess_cfg["ga_transform"] is not None:
            # GA closure now handles per-spectrum operations only. Autoscale
            # (column-wise StandardScaler) is applied separately so the
            # validation-rebuild path can fit-on-train / transform-on-val
            # without divergent scaler state. One-shot fit on full X_np
            # matches pre-fix CV behaviour and is chemometrics-acceptable.
            X_preprocessed = preprocess_cfg["ga_transform"](X_np)
            if preprocess_cfg.get("autoscale", False):
                from sklearn.preprocessing import StandardScaler
                X_preprocessed = StandardScaler().fit_transform(X_preprocessed)
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
                smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2),
                autoscale=preprocess_cfg.get("autoscale", False),
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
                wl_mask &= wavelengths_float >= analysis_wl_min
            if analysis_wl_max is not None:
                wl_mask &= wavelengths_float <= analysis_wl_max

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
        if (
            preprocess_cfg.get("deriv")
            and preprocess_cfg.get("window")
            and not wavelength_restriction_active
        ):
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
                print(
                    f"  Range: {wavelengths_for_models[0]:.1f} - {wavelengths_for_models[-1]:.1f} nm"
                )
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
                    X_for_models,  # Use filtered+preprocessed data
                    y_np,
                    wavelengths_float,  # Use filtered wavelengths
                    n_top_regions=n_top_regions,
                    test_all_individual=region_test_all_individual,
                    test_pairwise=region_test_pairwise,
                )

                if len(region_subsets) > 0:
                    prep_name = str(preprocess_cfg.get("name", "unknown"))
                    deriv_info = f"_d{preprocess_cfg['deriv']}" if preprocess_cfg["deriv"] else ""
                    print(
                        f"  Region analysis for {prep_name}{deriv_info}: Identified {len(region_subsets)} region-based subsets"
                    )
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
            if ga_preprocess and "ga_model_type" in preprocess_cfg:
                # Skip if this preprocessing config was optimized for a different model
                # ga_model_type stores the actual model name (e.g., "LightGBM", "PLS")
                if preprocess_cfg["ga_model_type"] != model_name:
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
                param_str = ", ".join(
                    [f"{k}={v}" for k, v in list(params.items())[:2]]
                )  # Show first 2 params
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
                    progress_callback(
                        {
                            "stage": "model_testing",
                            "message": progress_msg,
                            "current": current_config,
                            "total": total_configs,
                            "best_model": best_model_so_far,
                        }
                    )

                # Run full model first (using preprocessed + filtered data)
                result = _run_single_config(
                    X_for_models,  # Preprocessed + filtered data
                    y_np,
                    wavelengths_for_models,  # Filtered wavelengths
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
                    cv_strategy=cv_strategy,
                    cv_n_repeats=cv_n_repeats,
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
                    print(
                        f"     Full model: R²cv={result['R2cv']:.3f}, RMSEcv={result['RMSEcv']:.3f}"
                    )
                else:
                    print(
                        f"     Full model: AUCcv={result.get('ROC_AUCcv', 0):.3f}, Acccv={result.get('Accuracycv', 0):.3f}"
                    )

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
                        print(
                            f"  -> Skipping subset analysis for {model_name} (variable subsets disabled)"
                        )
                    else:
                        print(
                            f"  -> Computing feature importances for {model_name} subset analysis..."
                        )

                        # Cap n_components for PLS when fitting on filtered data
                        # (model was created with n_components based on original feature count,
                        # but X_for_models may have fewer features after wavelength filtering)
                        n_features_filtered = X_for_models.shape[1]
                        if hasattr(model, "n_components") and model.n_components is not None:
                            if model.n_components >= n_features_filtered:
                                model = clone(model)
                                capped = max(1, n_features_filtered - 1)
                                model.set_params(n_components=capped)
                                print(
                                    f"     Note: Capped PLS n_components to {capped} for importance computation (only {n_features_filtered} features)"
                                )

                        # Build model-only pipeline (data is already preprocessed and filtered)
                        # clone() prevents the importance-capture fit from mutating the shared
                        # `model` reference (leaving n_features_in_ set to the full-spectrum
                        # count, which then collides with subset fits downstream on sklearn 1.5.2).
                        pipe_steps = []
                        pipe_steps.append(("model", clone(model)))
                        pipe = Pipeline(pipe_steps)

                        # Fit on preprocessed+filtered data
                        pipe.fit(X_for_models, y_np)

                        # Get model from pipeline
                        fitted_model = pipe.named_steps["model"]

                        # X_for_models is already preprocessed and filtered - use directly
                        X_transformed_varsel = X_for_models
                        wavelengths_varsel = wavelengths_for_models
                        n_features_varsel = X_for_models.shape[1]
                        n_features_for_validation = (
                            n_features_varsel  # Define early for SPA/UVE-SPA methods
                        )

                        # --- UVE Prefilter: eliminate uninformative variables before varsel ---
                        _uve_prefilter_active = False
                        if apply_uve_prefilter and n_features_varsel >= 3:
                            try:
                                _uve_imp, _uve_thr, _uve_mask = get_uve_threshold(
                                    X_transformed_varsel,
                                    y_np,
                                    cutoff_multiplier=uve_cutoff_multiplier,
                                    n_components=uve_n_components,
                                    cv_folds=folds,
                                    random_state=random_state,
                                )
                                n_before = n_features_varsel
                                n_after = int(np.sum(_uve_mask))
                                if n_after < n_before:
                                    X_transformed_varsel = X_transformed_varsel[:, _uve_mask]
                                    wavelengths_varsel = wavelengths_varsel[_uve_mask]
                                    n_features_varsel = n_after
                                    n_features_for_validation = n_after
                                    _uve_prefilter_active = True
                                    print(
                                        f"     UVE prefilter: {n_before} -> {n_after} variables "
                                        f"({n_before - n_after} eliminated, threshold={_uve_thr:.4f})"
                                    )
                            except Exception as e:
                                print(
                                    f"     UVE prefilter failed ({e}), using all {n_features_varsel} variables"
                                )
                        elif apply_uve_prefilter and n_features_varsel < 3:
                            print(
                                f"     UVE prefilter skipped: only {n_features_varsel} features (min 3)"
                            )

                        # Loop over each selected variable selection method
                        for varsel_method in selected_methods:
                            # Check for pause/stop
                            if controller and not controller.check_and_wait():
                                break

                            # ===== Subset-returning methods (iPLS, MC-siPLS, MWPLS) =====
                            if varsel_method in (
                                "ipls_forward",
                                "ipls_backward",
                                "mc_sipls",
                                "mwpls",
                            ):
                                print(f"  -> Running {varsel_method}...")

                                # Call appropriate function
                                if varsel_method == "ipls_forward":
                                    subsets = ipls_forward(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        max_combine=ipls_max_combine,
                                        cv_folds=folds,
                                        random_state=random_state,
                                    )
                                elif varsel_method == "ipls_backward":
                                    subsets = ipls_backward(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        cv_folds=folds,
                                        random_state=random_state,
                                    )
                                elif varsel_method == "mc_sipls":
                                    subsets = mc_sipls(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        n_combinations=sipls_n_combinations,
                                        max_combine=ipls_max_combine,
                                        cv_folds=folds,
                                        random_state=random_state,
                                    )
                                elif varsel_method == "mwpls":
                                    subsets = mwpls(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        window_sizes=mwpls_window_sizes,
                                        step_size=mwpls_step_size,
                                        cv_folds=folds,
                                    )

                                if subsets is None or len(subsets) == 0:
                                    print(f"  -> {varsel_method} returned no subsets, skipping")
                                    continue

                                # Sort by rmsecv (best first) and apply limit
                                subsets_sorted = sorted(
                                    subsets, key=lambda s: s.get("rmsecv", float("inf"))
                                )

                                # Apply subset limit from dropdown
                                if ipls_subset_limit == "Top 5":
                                    subsets_to_test = subsets_sorted[:5]
                                elif ipls_subset_limit == "Top 10":
                                    subsets_to_test = subsets_sorted[:10]
                                elif ipls_subset_limit == "Top 20":
                                    subsets_to_test = subsets_sorted[:20]
                                else:  # "All"
                                    subsets_to_test = subsets_sorted

                                print(
                                    f"  -> Testing {len(subsets_to_test)} of {len(subsets)} subsets..."
                                )

                                # Test each subset
                                for subset_dict in subsets_to_test:
                                    if controller and not controller.check_and_wait():
                                        break

                                    subset_indices = subset_dict["indices"]
                                    subset_tag = subset_dict["tag"]

                                    # Use existing _run_single_config (same as top-N path)
                                    if preprocess_cfg["deriv"] is not None:
                                        subset_result = _run_single_config(
                                            X_transformed_varsel,
                                            y_np,
                                            wavelengths_varsel,
                                            model,
                                            model_name,
                                            params,
                                            preprocess_cfg,
                                            cv_splitter,
                                            task_type,
                                            is_binary_classification,
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
                                            cv_strategy=cv_strategy,
                                            cv_n_repeats=cv_n_repeats,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=(
                                                1
                                                if model_name in MODELS_PREFER_SERIAL_CV
                                                else n_jobs_default
                                            ),
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )
                                    else:
                                        subset_result = _run_single_config(
                                            X_transformed_varsel,
                                            y_np,
                                            wavelengths_varsel,
                                            model,
                                            model_name,
                                            params,
                                            preprocess_cfg,
                                            cv_splitter,
                                            task_type,
                                            is_binary_classification,
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
                                            cv_strategy=cv_strategy,
                                            cv_n_repeats=cv_n_repeats,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=(
                                                1
                                                if model_name in MODELS_PREFER_SERIAL_CV
                                                else n_jobs_default
                                            ),
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )

                                    if subset_result is None:
                                        continue
                                    df_results = add_result(df_results, subset_result)

                                    if task_type == "regression":
                                        print(f"    {subset_tag}: R²={subset_result['R2']:.3f}")
                                    else:
                                        print(
                                            f"    {subset_tag}: AUC={subset_result.get('ROC_AUC', 0):.3f}"
                                        )

                                continue  # Skip to next method (don't fall through to importance path)

                            # ===== EXISTING CODE: Standard importance-returning methods =====
                            # Get importances computed on preprocessed data
                            uve_selected_mask = None  # Captured by UVE for method-optimal count

                            # --- Variable selection cache lookup ---
                            # 'importance' is NEVER cached (depends on fitted model + hyperparams)
                            _cache_hit = False
                            if varsel_method != "importance":
                                _cache_key = _varsel_cache_key(
                                    preprocess_cfg, varsel_method, model_name, _uve_prefilter_active
                                )
                                with _varsel_cache_lock:
                                    if _cache_key in _varsel_cache:
                                        _cached = _varsel_cache[_cache_key]
                                        importances = _cached["importances"]
                                        uve_selected_mask = _cached.get("uve_selected_mask")
                                        _cache_hit = True
                                        print(f"  -> Using cached {varsel_method} result")

                            if _cache_hit:
                                pass  # Skip computation, use cached importances
                            else:
                                pass  # Fall through to compute importances

                            try:
                                if _cache_hit:
                                    pass  # Already have importances from cache

                                elif varsel_method == "importance":
                                    importances = get_feature_importances(
                                        fitted_model, model_name, X_transformed_varsel, y_np
                                    )

                                elif varsel_method == "spa":
                                    # SPA: Successive Projections Algorithm - reduces collinearity
                                    # Select minimally correlated variables
                                    # Use max variable count as default for SPA feature selection
                                    default_n_select = (
                                        max(variable_counts) if variable_counts else 100
                                    )
                                    n_to_select = min(default_n_select, n_features_for_validation)
                                    importances = spa_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        n_features=n_to_select,
                                        cv_folds=folds,
                                    )

                                elif varsel_method == "uve":
                                    # UVE: Uninformative Variable Elimination - filters noise
                                    # Use get_uve_threshold to also capture selected_mask for method-optimal count
                                    importances, _uve_threshold, uve_selected_mask = (
                                        get_uve_threshold(
                                            X_transformed_varsel,
                                            y_np,
                                            cutoff_multiplier=uve_cutoff_multiplier,
                                            n_components=uve_n_components,
                                            cv_folds=folds,
                                            random_state=random_state,
                                        )
                                    )

                                elif varsel_method == "uve_spa":
                                    # UVE-SPA: Hybrid method - filters noise then reduces collinearity
                                    # Use max variable count as default for UVE-SPA feature selection
                                    default_n_select = (
                                        max(variable_counts) if variable_counts else 100
                                    )
                                    n_to_select = min(default_n_select, n_features_for_validation)
                                    print(
                                        f"    -> Running UVE-SPA (target: {n_to_select} features)"
                                    )
                                    importances = uve_spa_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        n_features=n_to_select,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        uve_n_components=uve_n_components,
                                        uve_cv_folds=folds,
                                        spa_cv_folds=folds,
                                        random_state=random_state,
                                    )
                                    n_nonzero = (
                                        np.sum(importances > 0) if importances is not None else 0
                                    )
                                    print(
                                        f"    -> UVE-SPA completed: {n_nonzero} variables with non-zero importance"
                                    )

                                elif varsel_method == "ipls":
                                    # iPLS: Interval PLS - selects based on spectral regions
                                    importances = ipls_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        n_intervals=ipls_n_intervals,
                                        n_components=uve_n_components,
                                        cv_folds=folds,
                                        random_state=random_state,
                                    )

                                elif varsel_method in ("cars", "cars-aware", "cars-tree"):
                                    # CARS: Competitive Adaptive Reweighted Sampling
                                    # Monte Carlo-based method with exponential decay
                                    # cars-aware: Use model-appropriate fitness (LightGBM for tree models)
                                    # cars-tree: Hybrid importance (split+gain) for tree models
                                    if varsel_method == "cars":
                                        model_type_for_cars = None
                                        use_hybrid = False
                                    elif varsel_method == "cars-aware":
                                        model_type_for_cars = model_name
                                        use_hybrid = False
                                        print(f"    -> Running Model-Aware CARS for {model_name}")
                                    else:  # cars-tree
                                        model_type_for_cars = model_name
                                        use_hybrid = True
                                        print(
                                            f"    -> Running CARS-Tree (hybrid importance) for {model_name}"
                                        )

                                    importances = cars_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        n_iterations=50,
                                        pls_components=(
                                            uve_n_components if uve_n_components is not None else 5
                                        ),
                                        cv_folds=folds,
                                        monte_carlo_samples=80,
                                        random_state=random_state,
                                        model_type=model_type_for_cars,
                                        use_hybrid_importance=use_hybrid,
                                        hybrid_importance_weight=0.5,
                                        task_type=task_type,
                                    )

                                elif varsel_method in ("uve_cars", "uve_cars_tree"):
                                    # UVE-CARS / UVE-CARS-Tree: Noise filtering + adaptive selection
                                    if varsel_method == "uve_cars":
                                        mt_for_cars = None
                                        uh_for_cars = False
                                        print(f"    -> Running UVE-CARS")
                                    else:
                                        mt_for_cars = model_name
                                        uh_for_cars = True
                                        print(f"    -> Running UVE-CARS-Tree for {model_name}")

                                    importances = uve_cars_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        uve_n_components=uve_n_components,
                                        uve_cv_folds=folds,
                                        n_iterations=50,
                                        pls_components=(
                                            uve_n_components if uve_n_components is not None else 5
                                        ),
                                        cars_cv_folds=folds,
                                        monte_carlo_samples=80,
                                        random_state=random_state,
                                        model_type=mt_for_cars,
                                        use_hybrid_importance=uh_for_cars,
                                        hybrid_importance_weight=0.5,
                                        task_type=task_type,
                                    )

                                elif varsel_method == "uve_cars_spa":
                                    # UVE-CARS-SPA: 3-stage hybrid
                                    print(f"    -> Running UVE-CARS-SPA (3-stage)")
                                    importances = uve_cars_spa_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        cutoff_multiplier=uve_cutoff_multiplier,
                                        uve_n_components=uve_n_components,
                                        uve_cv_folds=folds,
                                        n_iterations=50,
                                        pls_components=(
                                            uve_n_components if uve_n_components is not None else 5
                                        ),
                                        cars_cv_folds=folds,
                                        monte_carlo_samples=80,
                                        spa_n_features=None,
                                        spa_cv_folds=folds,
                                        random_state=random_state,
                                        task_type=task_type,
                                    )

                                elif varsel_method == "fipls_spa":
                                    # Forward iPLS → SPA: Region selection + collinearity reduction
                                    print(f"    -> Running Forward iPLS-SPA")
                                    importances = fipls_spa_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        max_combine=5,
                                        ipls_cv_folds=folds,
                                        spa_n_features=None,
                                        spa_cv_folds=folds,
                                        random_state=random_state,
                                    )

                                elif varsel_method == "fipls_cars":
                                    # Forward iPLS → CARS: Region selection + adaptive selection
                                    print(f"    -> Running Forward iPLS-CARS")
                                    importances = fipls_cars_selection(
                                        X_transformed_varsel,
                                        y_np,
                                        wavelengths=wavelengths_varsel,
                                        n_intervals=ipls_n_intervals,
                                        max_combine=5,
                                        ipls_cv_folds=folds,
                                        n_iterations=50,
                                        pls_components=(
                                            uve_n_components if uve_n_components is not None else 5
                                        ),
                                        cars_cv_folds=folds,
                                        monte_carlo_samples=80,
                                        random_state=random_state,
                                        task_type=task_type,
                                    )

                                elif varsel_method == "vcpa-iriv":
                                    # VCPA-IRIV: Variable Combination Population Analysis
                                    # Iterative elimination with binary matrix sampling
                                    print(f"    -> Running VCPA-IRIV (n_outer=10, n_inner=50)")
                                    result = vcpa_iriv(
                                        X_transformed_varsel,
                                        y_np,
                                        n_outer_iterations=10,
                                        n_inner_iterations=50,
                                        pls_components=(
                                            uve_n_components if uve_n_components is not None else 5
                                        ),
                                        cv_folds=folds,
                                        random_state=random_state,
                                    )
                                    # Extract importance scores from result dict
                                    # Note: vcpa_iriv returns 'importance_scores', not 'importances'
                                    importances = result.get(
                                        "importance_scores", result.get("importances", None)
                                    )

                                    # VCPA returns importance_scores for ACTIVE indices only
                                    # We need to create full-length importance array using selected_indices
                                    selected = result.get("selected_indices", [])
                                    if importances is not None and len(importances) == len(
                                        selected
                                    ):
                                        # Map importance scores back to full wavelength array
                                        full_importances = np.zeros(X_transformed_varsel.shape[1])
                                        full_importances[selected] = importances
                                        importances = full_importances
                                        print(
                                            f"    -> VCPA-IRIV selected {len(selected)} variables with importance scores"
                                        )
                                    elif len(selected) > 0:
                                        # Fallback: create binary mask from selected_indices
                                        importances = np.zeros(X_transformed_varsel.shape[1])
                                        importances[selected] = 1.0
                                        print(
                                            f"    -> VCPA-IRIV selected {len(selected)} variables (binary mask fallback)"
                                        )
                                    else:
                                        # No variables selected - use uniform importances
                                        print(
                                            f"    -> WARNING: VCPA-IRIV selected no variables, using uniform importances"
                                        )
                                        importances = np.ones(X_transformed_varsel.shape[1])

                                elif varsel_method == "ga":
                                    # GA Variable Selection: Use model-appropriate fitness
                                    # Linear models use PLS fitness, tree models use LightGBM fitness

                                    # Determine GA parameters based on quick mode or user settings
                                    if ga_quick_mode:
                                        ga_pop, ga_gen, ga_runs, ga_early = 32, 50, 2, 10
                                        print(
                                            f"    -> Quick GA Mode: pop={ga_pop}, gen={ga_gen}, runs={ga_runs}"
                                        )
                                    else:
                                        # Use user-specified parameters
                                        ga_pop = ga_population_size
                                        ga_gen = ga_generations
                                        ga_runs = ga_n_runs
                                        ga_early = 20  # Default early stopping
                                        print(
                                            f"    -> GA Mode: pop={ga_pop}, gen={ga_gen}, runs={ga_runs}"
                                        )

                                    if model_name in LINEAR_MODELS:
                                        print(
                                            f"    -> Using GA-PLS for {model_name} (linear model)"
                                        )
                                        importances = ga_pls_selection(
                                            X_transformed_varsel,
                                            y_np,
                                            task_type=task_type,
                                            n_components=(
                                                uve_n_components
                                                if uve_n_components is not None
                                                else 10
                                            ),
                                            cv=folds,
                                            population_size=ga_pop,
                                            n_generations=ga_gen,
                                            n_runs=ga_runs,
                                            early_stopping=ga_early,
                                            random_state=random_state,
                                            progress_callback=progress_callback,
                                        )
                                    elif model_name in TREE_MODELS:
                                        print(
                                            f"    -> Using GA-LightGBM for {model_name} (tree model)"
                                        )
                                        importances = ga_lightgbm_selection(
                                            X_transformed_varsel,
                                            y_np,
                                            task_type=task_type,
                                            cv_folds=folds,
                                            n_estimators=50,
                                            num_leaves=15 if task_type == "classification" else 31,
                                            population_size=ga_pop,
                                            n_generations=ga_gen,
                                            n_runs=ga_runs,
                                            early_stopping=ga_early,
                                            random_state=random_state,
                                            progress_callback=progress_callback,
                                        )
                                    else:
                                        # Default to GA-PLS for unknown model types
                                        print(f"    -> Using GA-PLS for {model_name} (default)")
                                        importances = ga_pls_selection(
                                            X_transformed_varsel,
                                            y_np,
                                            task_type=task_type,
                                            n_components=(
                                                uve_n_components
                                                if uve_n_components is not None
                                                else 10
                                            ),
                                            cv=folds,
                                            population_size=ga_pop,
                                            n_generations=ga_gen,
                                            n_runs=ga_runs,
                                            early_stopping=ga_early,
                                            random_state=random_state,
                                            progress_callback=progress_callback,
                                        )

                                else:
                                    # This shouldn't happen due to filtering, but handle gracefully
                                    print(f"  -> Skipping unimplemented method '{varsel_method}'")
                                    continue

                                # --- Store result in cache ---
                                if not _cache_hit and varsel_method != "importance":
                                    with _varsel_cache_lock:
                                        _varsel_cache[_cache_key] = {
                                            "importances": importances,
                                            "uve_selected_mask": uve_selected_mask,
                                        }

                                # Track if uniform fallback was used (for debugging/filtering results)
                                used_uniform_fallback = False

                                # Validate importances array before proceeding
                                if importances is None:
                                    print(
                                        f"  -> ERROR: {varsel_method} returned None importances, skipping"
                                    )
                                    continue
                                if len(importances) != X_transformed_varsel.shape[1]:
                                    print(
                                        f"  -> ERROR: {varsel_method} returned wrong-sized importances "
                                        f"({len(importances)} vs {X_transformed_varsel.shape[1]}), skipping"
                                    )
                                    continue
                                if np.all(importances == 0):
                                    print(
                                        f"  -> WARNING: {varsel_method} returned all-zero importances, using uniform"
                                    )
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
                                valid_variable_counts = [
                                    n for n in user_variable_counts if n < n_features_for_validation
                                ]

                                print(f"  -> User variable counts: {user_variable_counts}")
                                print(
                                    f"  -> Valid variable counts (< {n_features_for_validation} features): {valid_variable_counts}"
                                )
                                print(f"  -> Variable selection method: {varsel_method}")

                                if not valid_variable_counts:
                                    print(
                                        f"  WARNING: No valid variable counts to test (all selected counts >= {n_features_for_validation} features)"
                                    )

                                # Apply edge masking for Savitzky-Golay derivatives
                                # SKIP when wavelength restriction is active - restricted wavelengths
                                # are from middle of spectrum, not SG boundary edges
                                if not wavelength_restriction_active and not _uve_prefilter_active:
                                    importances = _apply_edge_mask(importances, preprocess_cfg)

                                # Compute method-optimal variable count (natural cutoff from method)
                                n_method_optimal = 0
                                method_has_natural_optimal = False

                                if not used_uniform_fallback:
                                    if varsel_method in (
                                        "cars",
                                        "cars-aware",
                                        "cars-tree",
                                        "uve_spa",
                                        "uve_cars",
                                        "uve_cars_tree",
                                        "uve_cars_spa",
                                        "fipls_spa",
                                        "fipls_cars",
                                    ):
                                        n_method_optimal = int(np.count_nonzero(importances))
                                        method_has_natural_optimal = True
                                    elif varsel_method == "uve" and uve_selected_mask is not None:
                                        n_method_optimal = int(np.sum(uve_selected_mask))
                                        method_has_natural_optimal = True
                                    elif varsel_method == "vcpa-iriv":
                                        n_method_optimal = int(np.count_nonzero(importances))
                                        method_has_natural_optimal = True

                                if method_has_natural_optimal:
                                    if (
                                        n_method_optimal <= 0
                                        or n_method_optimal >= n_features_for_validation
                                    ):
                                        method_has_natural_optimal = False
                                    elif n_method_optimal in valid_variable_counts:
                                        print(
                                            f"  -> Method-optimal for {varsel_method}: {n_method_optimal} already in counts, skipping"
                                        )
                                        method_has_natural_optimal = False
                                    else:
                                        print(
                                            f"  -> Method-optimal for {varsel_method}: {n_method_optimal} vars (will test)"
                                        )

                                # Run subsets with user-selected counts
                                results_added_for_method = 0
                                for n_top in valid_variable_counts:
                                    print(
                                        f"  -> Testing top-{n_top} vars ({varsel_method})...",
                                        end=" ",
                                    )
                                    # Select top N most important features based on preprocessed importances
                                    # Use stable sort to ensure deterministic feature ordering when importances are tied
                                    top_indices = np.argsort(importances, kind="stable")[-n_top:][
                                        ::-1
                                    ]

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
                                            cv_strategy=cv_strategy,
                                            cv_n_repeats=cv_n_repeats,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=(
                                                1
                                                if model_name in MODELS_PREFER_SERIAL_CV
                                                else n_jobs_default
                                            ),
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
                                            cv_strategy=cv_strategy,
                                            cv_n_repeats=cv_n_repeats,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=(
                                                1
                                                if model_name in MODELS_PREFER_SERIAL_CV
                                                else n_jobs_default
                                            ),
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
                                        print(
                                            f"R²cv={subset_result['R2cv']:.3f}, RMSEcv={subset_result['RMSEcv']:.3f}"
                                        )
                                    else:
                                        print(
                                            f"AUCcv={subset_result.get('ROC_AUCcv', 0):.3f}, Acccv={subset_result.get('Accuracycv', 0):.3f}"
                                        )

                                    # Update best model tracker for subset results (use CV metrics for consistency)
                                    if best_model_so_far is None:
                                        best_model_so_far = subset_result
                                    else:
                                        if task_type == "regression":
                                            if (
                                                subset_result["RMSEcv"]
                                                < best_model_so_far["RMSEcv"]
                                            ):
                                                best_model_so_far = subset_result
                                        else:  # classification
                                            if subset_result.get(
                                                "ROC_AUCcv", 0
                                            ) > best_model_so_far.get("ROC_AUCcv", 0):
                                                best_model_so_far = subset_result

                                # Run method-optimal subset if applicable
                                if method_has_natural_optimal and n_method_optimal > 0:
                                    print(
                                        f"  -> Testing method-optimal {n_method_optimal} vars ({varsel_method})...",
                                        end=" ",
                                    )
                                    top_indices_opt = np.argsort(importances, kind="stable")[
                                        -n_method_optimal:
                                    ][::-1]

                                    if preprocess_cfg["deriv"] is not None:
                                        opt_result = _run_single_config(
                                            X_transformed_varsel,
                                            y_np,
                                            wavelengths_varsel,
                                            model,
                                            model_name,
                                            params,
                                            preprocess_cfg,
                                            cv_splitter,
                                            task_type,
                                            is_binary_classification,
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
                                            cv_strategy=cv_strategy,
                                            cv_n_repeats=cv_n_repeats,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=(
                                                1
                                                if model_name in MODELS_PREFER_SERIAL_CV
                                                else n_jobs_default
                                            ),
                                            wavelength_restriction_active=wavelength_restriction_active,
                                            early_stopping_rounds=early_stopping_rounds,
                                        )
                                    else:
                                        opt_result = _run_single_config(
                                            X_transformed_varsel,
                                            y_np,
                                            wavelengths_varsel,
                                            model,
                                            model_name,
                                            params,
                                            preprocess_cfg,
                                            cv_splitter,
                                            task_type,
                                            is_binary_classification,
                                            subset_indices=top_indices_opt,
                                            subset_tag=f"{varsel_method}",
                                            top_n_vars=30,
                                            skip_preprocessing=False,
                                            skip_spectral_preprocessing=True,
                                            excluded_count=excluded_count,
                                            validation_count=validation_count,
                                            total_samples_original=total_samples_original,
                                            folds=folds,
                                            cv_strategy=cv_strategy,
                                            cv_n_repeats=cv_n_repeats,
                                            imbalance_method=imbalance_method,
                                            imbalance_params=imbalance_params,
                                            full_vars_original=n_original_wavelengths,
                                            n_jobs_cv=(
                                                1
                                                if model_name in MODELS_PREFER_SERIAL_CV
                                                else n_jobs_default
                                            ),
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
                                            print(
                                                f"R²cv={opt_result['R2cv']:.3f}, RMSEcv={opt_result['RMSEcv']:.3f} (method-optimal)"
                                            )
                                        else:
                                            print(
                                                f"AUCcv={opt_result.get('ROC_AUCcv', 0):.3f}, Acccv={opt_result.get('Accuracycv', 0):.3f} (method-optimal)"
                                            )

                                        if best_model_so_far is None:
                                            best_model_so_far = opt_result
                                        else:
                                            if task_type == "regression":
                                                if (
                                                    opt_result["RMSEcv"]
                                                    < best_model_so_far["RMSEcv"]
                                                ):
                                                    best_model_so_far = opt_result
                                            else:
                                                if opt_result.get(
                                                    "ROC_AUCcv", 0
                                                ) > best_model_so_far.get("ROC_AUCcv", 0):
                                                    best_model_so_far = opt_result

                                # Summary for this variable selection method
                                print(
                                    f"  [SUMMARY] {varsel_method}: Added {results_added_for_method} results to dataframe"
                                )

                            except Exception as e:
                                import traceback

                                print(
                                    f"Warning: Could not compute importances for {model_name} with method '{varsel_method}': {e}"
                                )
                                print(f"  Full traceback:\n{traceback.format_exc()}")

                # Run region-based subsets for ALL models (not just PLS/RF/MLP/NeuralBoosted)
                # For derivatives: use preprocessed data to avoid reapplying preprocessing
                # For raw/SNV: use raw data and reapply preprocessing
                if enable_region_subsets and len(region_subsets) > 0:
                    print(f"  -> Testing {len(region_subsets)} spectral regions:")
                    for i, region_subset in enumerate(region_subsets, 1):
                        print(
                            f"     Region {i}/{len(region_subsets)} ({region_subset['tag']})...",
                            end=" ",
                        )
                        # Use filtered+preprocessed data for ALL preprocessing types
                        # Region indices were computed on filtered data, so this is correct
                        region_result = _run_single_config(
                            X_for_models,  # Filtered+preprocessed data
                            y_np,
                            wavelengths_for_models,  # Filtered wavelengths
                            model,
                            model_name,
                            params,
                            preprocess_cfg,  # Keep original config for labeling
                            cv_splitter,
                            task_type,
                            is_binary_classification,
                            subset_indices=region_subset["indices"],
                            subset_tag=region_subset["tag"],
                            top_n_vars=30,
                            skip_preprocessing=False,
                            skip_spectral_preprocessing=True,  # Spectral preprocessing already done
                            excluded_count=excluded_count,
                            validation_count=validation_count,
                            total_samples_original=total_samples_original,
                            folds=folds,
                            cv_strategy=cv_strategy,
                            cv_n_repeats=cv_n_repeats,
                            imbalance_method=imbalance_method,
                            imbalance_params=imbalance_params,
                            full_vars_original=n_original_wavelengths,
                            n_jobs_cv=(
                                1 if model_name in MODELS_PREFER_SERIAL_CV else n_jobs_default
                            ),
                            wavelength_restriction_active=wavelength_restriction_active,
                            early_stopping_rounds=early_stopping_rounds,
                        )
                        if region_result is None:
                            print(f"[SKIPPED]")
                            continue
                        df_results = add_result(df_results, region_result)

                        # Show result immediately (CV metrics for consistency)
                        if task_type == "regression":
                            print(
                                f"R²cv={region_result['R2cv']:.3f}, RMSEcv={region_result['RMSEcv']:.3f}"
                            )
                        else:
                            print(
                                f"AUCcv={region_result.get('ROC_AUCcv', 0):.3f}, Acccv={region_result.get('Accuracycv', 0):.3f}"
                            )

                        # Update best model tracker for region subset results (use CV metrics for consistency)
                        if best_model_so_far is None:
                            best_model_so_far = region_result
                        else:
                            if task_type == "regression":
                                if region_result["RMSEcv"] < best_model_so_far["RMSEcv"]:
                                    best_model_so_far = region_result
                            else:  # classification
                                if region_result.get("ROC_AUCcv", 0) > best_model_so_far.get(
                                    "ROC_AUCcv", 0
                                ):
                                    best_model_so_far = region_result

    # Compute composite scores and rank
    from .scoring import compute_composite_score

    # Check for subset mixing (full-spectrum with subset models)
    if "SubsetTag" in df_results.columns:
        subset_counts = df_results["SubsetTag"].value_counts()
        if len(subset_counts) > 1:
            print("\n[WARNING] Ranking includes multiple subset types:")
            for subset_type, count in subset_counts.items():
                print(f"  - {subset_type}: {count} models")
            print("  Subset models may rank higher due to lower variable counts.")
            print("  Consider filtering by SubsetTag before ranking for fairer comparison.\n")

    df_ranked = compute_composite_score(df_results, task_type, variable_penalty, gap_penalty)

    # =========================================================================
    # COMPUTE VALIDATION METRICS FOR TOP MODELS (if validation set provided)
    # =========================================================================
    if compute_validation and X_validation is not None and y_validation is not None:
        # Convert X to numpy if it's a DataFrame
        X_train_for_val = X.values if hasattr(X, "values") else X
        X_val_for_val = (
            X_validation if isinstance(X_validation, np.ndarray) else np.array(X_validation)
        )
        y_val_for_val = (
            y_validation if isinstance(y_validation, np.ndarray) else np.array(y_validation)
        )

        # Filter NaN from validation targets
        val_nan = pd.isna(y_val_for_val)
        if np.any(val_nan):
            n_val_dropped = int(np.sum(val_nan))
            print(
                f"[Validation] Dropping {n_val_dropped} validation sample(s) with NaN target values"
            )
            X_val_for_val = X_val_for_val[~val_nan]
            y_val_for_val = y_val_for_val[~val_nan]

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
        wavelengths_for_validation = (
            X.columns.astype(float).values if hasattr(X, "columns") else np.arange(X.shape[1])
        )

        df_ranked = compute_validation_metrics_for_top_models(
            df_ranked,
            X_train_for_val,
            y_train_for_val,
            X_val_for_val,
            y_val_for_val,
            task_type,
            wavelengths_for_validation,
            top_n=validation_top_n,
            progress_callback=progress_callback,
            imbalance_method=imbalance_method,
        )

    # Return results along with label_encoder (for classification with text labels)
    return df_ranked, label_encoder


def _run_single_fold(
    pipe,
    X,
    y,
    train_idx,
    test_idx,
    task_type,
    is_binary_classification,
    use_sample_weight_for_classification=False,
    early_stopping_rounds=None,
):
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
    if hasattr(pipe_clone, "named_steps") and "imbalance" in pipe_clone.named_steps:
        imbalance_step = pipe_clone.named_steps["imbalance"]
        if hasattr(imbalance_step, "sample_weight_"):
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

    if (
        sample_weight_train is None
        and use_sample_weight_for_classification
        and task_type == "classification"
    ):
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weight_train = compute_sample_weight("balanced", y_train)

        # Get final model from pipeline
        if hasattr(pipe_clone, "steps"):
            manual_fit_used = True
            # Transform X through all steps except the model
            X_train_transformed = X_train
            # T-32: track the post-resampling y so the final fit() sees a y
            # whose length matches X_train_transformed AND sample_weight_train.
            # Pre-fix, the fit() calls below passed the original y_train, which
            # mismatches X / sample_weight after a resampler runs. sklearn
            # raises ValueError on the length mismatch — the user got a hard
            # crash whenever they combined a resampler (SMOTE, ADASYN, ...)
            # with a sample_weight-supporting model (Ridge, LogisticRegression).
            y_train_for_model = y_train
            for step_name, step in pipe_clone.steps[:-1]:
                if hasattr(step, "fit_resample"):
                    # For imblearn resamplers, apply fit_resample
                    X_train_transformed, y_train_for_model = step.fit_resample(
                        X_train_transformed, y_train_for_model
                    )
                    # Recompute sample weights for resampled data
                    sample_weight_train = compute_sample_weight("balanced", y_train_for_model)
                    fitted_steps.append((step_name, step, "resample"))
                elif hasattr(step, "transform"):
                    step.fit(X_train_transformed, y_train_for_model)
                    X_train_transformed = step.transform(X_train_transformed)
                    fitted_steps.append((step_name, step, "transform"))

            # Fit the final model with sample weights (if supported)
            final_model = pipe_clone.steps[-1][1]
            if _supports_sample_weight(final_model):
                final_model.fit(
                    X_train_transformed, y_train_for_model, sample_weight=sample_weight_train
                )
            else:
                final_model.fit(X_train_transformed, y_train_for_model)
        else:
            # No pipeline, just the model
            if _supports_sample_weight(pipe_clone):
                pipe_clone.fit(X_train, y_train, sample_weight=sample_weight_train)
            else:
                pipe_clone.fit(X_train, y_train)
        sample_weight_train = "applied"  # Flag that we've already fit

    # Standard path: fit if not already done above
    if sample_weight_train is None:
        # Check if we should use early stopping for boosting models
        use_early_stopping = early_stopping_rounds is not None and early_stopping_rounds > 0

        if use_early_stopping:
            # Get final model from pipeline
            if hasattr(pipe_clone, "steps"):
                final_model_es = pipe_clone.steps[-1][1]
            else:
                final_model_es = pipe_clone

            # Check if final model is a boosting model
            if is_boosting_model(final_model_es):
                manual_fit_used = True

                # Transform training data through preprocessing steps
                X_train_transformed = X_train.copy()
                X_test_transformed = X_test.copy()

                if hasattr(pipe_clone, "steps"):
                    # T-32 fix-of-fixes (GLM/DeepSeek MEDIUM): use the same
                    # y_train_for_model threading pattern as the classification-
                    # sample-weight branch instead of in-place mutating y_train.
                    # Pre-fix this branch had the same architectural landmine
                    # (downstream code seeing silently-resampled y_train) — no
                    # crash today since boosting models on this path don't
                    # propagate sample_weight, but a future merge of the two
                    # branches OR adding a sample_weight-supporting boosting
                    # model would re-introduce T-32's class of bug.
                    y_train_for_model = y_train
                    for step_name, step in pipe_clone.steps[:-1]:
                        if hasattr(step, "fit_resample"):
                            # For imblearn resamplers, apply fit_resample (only to training data)
                            X_train_transformed, y_train_for_model = step.fit_resample(
                                X_train_transformed, y_train_for_model
                            )
                            fitted_steps.append((step_name, step, "resample"))
                            # Note: Don't transform test data - resampling only applies to training
                        elif hasattr(step, "transform"):
                            step.fit(X_train_transformed, y_train_for_model)
                            X_train_transformed = step.transform(X_train_transformed)
                            X_test_transformed = step.transform(X_test_transformed)
                            fitted_steps.append((step_name, step, "transform"))

                    final_model = final_model_es
                else:
                    final_model = final_model_es
                    y_train_for_model = y_train

                # Fit with early stopping
                _fit_with_early_stopping(
                    final_model,
                    X_train_transformed,
                    y_train_for_model,
                    X_test_transformed,
                    y_test,
                    early_stopping_rounds,
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
            if step_type == "transform" and hasattr(step, "transform"):
                X_transformed = step.transform(X_transformed)
            # Skip resample steps for test data - they only apply to training
        return final_model.predict(X_transformed), X_transformed

    def _manual_transform_predict_proba(X_data):
        """Transform X through manually fitted steps and predict_proba with final model."""
        X_transformed = X_data
        for step_name, step, step_type in fitted_steps:
            if step_type == "transform" and hasattr(step, "transform"):
                X_transformed = step.transform(X_transformed)
        return final_model.predict_proba(X_transformed)

    if task_type == "regression":
        if manual_fit_used:
            y_pred, _ = _manual_transform_predict(X_test)
        else:
            y_pred = pipe_clone.predict(X_test)
        y_pred = np.ravel(y_pred)  # Ensure 1D for metrics
        # Per-fold RMSE/R² are kept for debugging and tests only — headline metrics
        # are computed from pooled y_test/y_pred in the caller (see _aggregate_metrics
        # at search.py:4212+) to match IUPAC/chemometrics convention.
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": rmse, "R2": r2, "y_test": y_test, "y_pred": y_pred}
    else:  # classification
        if manual_fit_used:
            y_pred, _ = _manual_transform_predict(X_test)
        else:
            y_pred = pipe_clone.predict(X_test)
        y_pred = np.ravel(y_pred)  # Ensure 1D for metrics

        acc = accuracy_score(y_test, y_pred)

        # Use is_binary_classification flag (determined from full dataset) for consistent averaging
        # This avoids issues where a CV fold might have missing classes
        # Use 'macro' for multiclass to treat all classes equally (consistent with ROC_AUC)
        average_method = "binary" if is_binary_classification else "macro"

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
                    model_classes = (
                        final_model.classes_ if hasattr(final_model, "classes_") else None
                    )
                else:
                    y_proba = pipe_clone.predict_proba(X_test)
                    model_classes = pipe_clone.classes_ if hasattr(pipe_clone, "classes_") else None

                if is_binary_classification:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # Explicitly tell roc_auc_score the column order matches model's classes_
                    if model_classes is not None:
                        auc = roc_auc_score(
                            y_test,
                            y_proba,
                            multi_class="ovr",
                            average="macro",
                            labels=model_classes,
                        )
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

                # Log Loss (requires predict_proba)
                try:
                    logloss = log_loss(
                        y_test, y_proba, labels=model_classes if model_classes is not None else None
                    )
                except Exception:
                    logloss = np.nan
            except Exception:
                auc = np.nan
                logloss = np.nan

        # Compute additional classification metrics
        try:
            specificity = compute_specificity(y_test, y_pred, average="macro")
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
            "Accuracy": acc,
            "ROC_AUC": auc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "Kappa": kappa,
            "MCC": mcc,
            "BalancedAcc": balanced_acc,
            "BER": ber,
            "LogLoss": logloss,
            "y_test": y_test,
            "y_pred": y_pred,
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
    cv_strategy="kfold",
    cv_n_repeats=5,
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
    if (
        hasattr(model, "n_components")
        and model.n_components is not None
        and model.n_components >= n_vars
    ):
        print(
            f"  [SKIP] {model_name} n_components={model.n_components} >= n_vars={n_vars}, invalid combination"
        )
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
        if imbalance_method is not None and imbalance_method != "class_weight":
            # Add ONLY imbalance handling (spectral preprocessing already done)
            from spectral_predict.imbalance import build_imbalance_transformer

            if imbalance_params is None:
                imbalance_params = {}
            imbalance_transformer = build_imbalance_transformer(
                method=imbalance_method,
                task_type=task_type,
                random_state=random_state,  # CRITICAL for reproducibility
                **imbalance_params,
            )
            pipe_steps.append(("imbalance", imbalance_transformer))
    else:
        # Normal behavior: build full pipeline (spectral + imbalance)
        # Phase 3: Extract wavelengths for interference removal
        # Use wavelengths from parameter if available, otherwise try to extract from DataFrame columns
        wavelengths_for_interference = (
            wavelengths
            if wavelengths is not None
            else (X.columns.astype(float).values if hasattr(X, "columns") else None)
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
            smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2),
            autoscale=preprocess_cfg.get("autoscale", False),
        )

    # Handle class_weight for imbalanced classification.
    # Discrimination order matches the GUI dispatcher (c395317), the codegen
    # (code_generator.py:943-950 / 1704-1721), unified_bayesian.py:1238, and
    # nsga2_search.py:1400 / 3106 / 3437. The explicit CatBoost branch is
    # required for consistency: CatBoost.fit DOES expose sample_weight, but the
    # codebase-wide convention for CatBoost is auto_class_weights='Balanced'
    # (constructor mechanism). Without the explicit branch, CatBoost would fall
    # through to the sample_weight fallback — functionally weighted but mechanism-
    # divergent from the rest of the dispatchers (Codex MEDIUM on PR #38).
    use_sample_weight_for_classification = False
    if imbalance_method == "class_weight" and task_type == "classification":
        if model_name == "CatBoost":
            try:
                model.set_params(auto_class_weights="Balanced")
            except Exception as e:
                import warnings

                warnings.warn(
                    f"CatBoost set_params(auto_class_weights='Balanced') failed: {e}. "
                    f"Model will train UNWEIGHTED. Consider switching to SMOTE/ADASYN.",
                    UserWarning,
                )
        elif hasattr(model, "class_weight"):
            try:
                model.set_params(class_weight="balanced")
            except Exception as e:
                import warnings

                warnings.warn(
                    f"{model_name} has class_weight attribute but set_params failed: {e}. "
                    f"Consider using SMOTE or other resampling method.",
                    UserWarning,
                )
        else:
            # Check if model supports sample_weight in fit() (e.g., RidgeClassifier)
            import inspect

            model_fit_sig = inspect.signature(model.fit) if hasattr(model, "fit") else None
            if model_fit_sig and "sample_weight" in model_fit_sig.parameters:
                # Model supports sample_weight - we'll compute and apply during _run_single_fold
                use_sample_weight_for_classification = True
            else:
                # Model doesn't support class_weight OR sample_weight
                import warnings

                if model_name in ["MLP", "MLPClassifier"]:
                    warnings.warn(
                        f"{model_name} does not support class_weight or sample_weight. "
                        f"For imbalanced classification with MLP, use SMOTE or other resampling methods instead.",
                        UserWarning,
                    )
                else:
                    warnings.warn(
                        f"{model_name} does not support class_weight. "
                        f"Consider using SMOTE or other resampling methods for imbalanced data.",
                        UserWarning,
                    )

    # For PLS-DA, we need PLS + StandardScaler + LogisticRegression
    # StandardScaler normalizes PLS scores to fix numerical instability with derivatives
    if model_name == "PLS-DA":
        pipe_steps.append(("pls", clone(model)))
        pipe_steps.append(("scaler", StandardScaler()))  # Scale PLS scores for LogisticRegression

        # Extract LogisticRegression parameters from config (prefixed with lr_)
        lr_C = params.get("lr_C", 1.0) if params else 1.0
        lr_solver = params.get("lr_solver", "lbfgs") if params else "lbfgs"
        lr_max_iter = params.get("lr_max_iter", 1000) if params else 1000

        # Build LogisticRegression with configurable parameters
        lr_kwargs = {
            "C": lr_C,
            "solver": lr_solver,
            "max_iter": lr_max_iter,
            "random_state": random_state,
        }

        # Apply class_weight to LogisticRegression if requested
        if imbalance_method == "class_weight" and task_type == "classification":
            lr_kwargs["class_weight"] = "balanced"

        pipe_steps.append(("lr", LogisticRegression(**lr_kwargs)))
    # For scale-sensitive models (SVC/SVR, MLP, NeuralBoosted), add StandardScaler before model
    # These use gradient descent or kernel methods that are sensitive to feature scale.
    # T-36: When autoscale is active, the preprocessing pipeline already inserts a
    # StandardScaler on the spectral block, so the per-model scaler is redundant.
    elif model_name in SCALE_SENSITIVE_MODELS:
        if not preprocess_cfg.get("autoscale", False):
            pipe_steps.append(("scaler", StandardScaler()))
        pipe_steps.append(("model", clone(model)))
    else:
        pipe_steps.append(("model", clone(model)))

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

    # Realize splits once so we can both run folds AND pool by sample later.
    # Generators get consumed; we need the test indices for repeated-CV pooling.
    splits = list(cv_splitter.split(X, y))

    # Run CV (serial if n_jobs_cv=1 for reproducibility, parallel otherwise)
    if n_jobs_cv == 1:
        # Serial execution for reproducibility (deterministic fold ordering)
        cv_metrics = [
            _run_single_fold(
                pipe,
                X,
                y,
                train_idx,
                test_idx,
                task_type,
                is_binary_classification,
                use_sample_weight_for_classification,
                early_stopping_rounds=early_stopping_rounds,
            )
            for train_idx, test_idx in splits
        ]
    else:
        # Parallel execution for speed.
        # 3.11 frozen builds must use 'threading' — loky's process spawn is
        # broken in PyInstaller 5.x bundles on 3.11. Dev mode and 3.12 frozen
        # builds use 'loky' for real multiprocessing.
        backend = "threading" if _frozen_needs_threading_fallback() else "loky"
        cv_metrics = Parallel(n_jobs=n_jobs_cv, backend=backend)(
            delayed(_run_single_fold)(
                pipe,
                X,
                y,
                train_idx,
                test_idx,
                task_type,
                is_binary_classification,
                use_sample_weight_for_classification,
                early_stopping_rounds=early_stopping_rounds,
            )
            for train_idx, test_idx in splits
        )

    # Print summary if imbalance handling was used
    if imbalance_method is not None:
        if imbalance_method == "class_weight":
            print(f"  [OK] Imbalance handling: class_weight applied to model")
        else:
            print(f"  [OK] Imbalance handling: {imbalance_method} applied successfully")

    # Guard against all-folds-failed state: empty cv_metrics means no valid
    # predictions to aggregate. Fail loudly instead of silently reporting 0.0
    # accuracy / NaN RMSE.
    if not cv_metrics:
        raise ValueError(
            "All CV folds failed — cannot compute metrics. "
            "Check upstream fold errors for root cause."
        )

    # Pool predictions per sample so repeated-CV (RepeatedKFold/RepeatedStratifiedKFold)
    # produces one prediction per sample before scoring. Under plain K-Fold / LOO
    # this is a no-op (each sample appears in exactly one test fold). Under
    # repeated CV, naive concatenation duplicates rows and computes metrics from
    # correlated observations.
    from spectral_predict.cv_utils import _is_repeated_cv, reduce_repeated_cv_predictions

    repeated_cv = _is_repeated_cv(cv_splitter)

    # Average metrics
    if task_type == "regression":
        if repeated_cv:
            all_y_test, all_y_pred = reduce_repeated_cv_predictions(
                cv_metrics, splits, n_samples=len(y), task_type="regression"
            )
        else:
            all_y_test = np.concatenate([m["y_test"] for m in cv_metrics])
            all_y_pred = np.concatenate([m["y_pred"] for m in cv_metrics])

        # Compute RMSE from aggregated predictions (not per-fold averages).
        # Matches chemometrics convention (Unscrambler, PLS_Toolbox, SIMCA, IUPAC).
        # Under LOO this is required — per-fold RMSE on 1-sample folds degenerates to |y-ŷ|,
        # and averaging those gives MAE, not RMSE.
        mean_rmse = float(np.sqrt(mean_squared_error(all_y_test, all_y_pred)))

        # Compute R² from aggregated predictions (not per-fold averages)
        # Averaging per-fold R² is mathematically incorrect due to different SS_tot per fold
        mean_r2 = r2_score(all_y_test, all_y_pred)

        # Compute additional NIR spectroscopy metrics from aggregated CV predictions
        # MAEcv: Mean Absolute Error - less sensitive to outliers than RMSE
        mae_cv = mean_absolute_error(all_y_test, all_y_pred)
        # Bias: Mean prediction error (positive = systematic overprediction)
        bias_cv = float(np.mean(all_y_pred - all_y_test))
        ccc_cv = lins_ccc(all_y_test, all_y_pred)
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
        for i, (lower, upper) in enumerate(
            [
                (-np.inf, quartiles[0]),  # Q1
                (quartiles[0], quartiles[1]),  # Q2
                (quartiles[1], quartiles[2]),  # Q3
                (quartiles[2], np.inf),  # Q4
            ]
        ):
            # Use true Y values for mask (auto-ensemble uses stacking, not routing)
            mask = (all_y_test >= lower) & (all_y_test < upper if i < 3 else all_y_test >= lower)
            if mask.sum() > 0:
                regional_rmse[f"Q{i+1}"] = np.sqrt(
                    mean_squared_error(all_y_test[mask], all_y_pred[mask])
                )
            else:
                regional_rmse[f"Q{i+1}"] = np.nan
    else:
        # Headline label-based metrics: under repeated CV, derive from
        # majority-vote-pooled predictions per sample (averaging fold metrics
        # double-counts samples that appear in multiple test folds). AUC/LogLoss/BER
        # require probabilities and stay as mean-of-folds.
        if repeated_cv:
            all_y_test, all_y_pred = reduce_repeated_cv_predictions(
                cv_metrics, splits, n_samples=len(y), task_type="classification"
            )
            from sklearn.metrics import (
                accuracy_score as _acc,
                f1_score as _f1,
                precision_score as _ps,
                recall_score as _rs,
                balanced_accuracy_score as _bas,
                cohen_kappa_score as _kappa,
                matthews_corrcoef as _mcc,
            )

            avg = "binary" if is_binary_classification else "macro"
            mean_acc = float(_acc(all_y_test, all_y_pred))
            mean_f1 = float(_f1(all_y_test, all_y_pred, average=avg, zero_division=0))
            mean_precision = float(_ps(all_y_test, all_y_pred, average=avg, zero_division=0))
            mean_recall = float(_rs(all_y_test, all_y_pred, average=avg, zero_division=0))
            mean_balanced_acc = float(_bas(all_y_test, all_y_pred))
            mean_kappa = float(_kappa(all_y_test, all_y_pred))
            mean_mcc = float(_mcc(all_y_test, all_y_pred))
            # Specificity is only defined for binary; derive from confusion matrix.
            # Pass labels= explicitly so the matrix is always 2x2 even when
            # pooled predictions collapse to a single class (upstream y
            # validation guarantees both labels exist in y_true, but model
            # degeneracy can make y_pred single-class).
            if is_binary_classification:
                from sklearn.metrics import confusion_matrix as _cm

                binary_labels = np.unique(y)
                cm = _cm(all_y_test, all_y_pred, labels=binary_labels)
                tn, fp, fn, tp = cm.ravel()
                mean_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
            else:
                mean_specificity = float(
                    np.mean(
                        [m["Specificity"] for m in cv_metrics if not np.isnan(m["Specificity"])]
                    )
                )
            # BER = 1 - BalancedAccuracy, label-based, pools alongside BalancedAcc
            mean_ber = 1.0 - mean_balanced_acc
        else:
            mean_acc = np.mean([m["Accuracy"] for m in cv_metrics])
            mean_f1 = np.mean([m["F1"] for m in cv_metrics if not np.isnan(m["F1"])])
            mean_precision = np.mean(
                [m["Precision"] for m in cv_metrics if not np.isnan(m["Precision"])]
            )
            mean_recall = np.mean([m["Recall"] for m in cv_metrics if not np.isnan(m["Recall"])])
            mean_specificity = np.mean(
                [m["Specificity"] for m in cv_metrics if not np.isnan(m["Specificity"])]
            )
            mean_kappa = np.mean([m["Kappa"] for m in cv_metrics if not np.isnan(m["Kappa"])])
            mean_mcc = np.mean([m["MCC"] for m in cv_metrics if not np.isnan(m["MCC"])])
            mean_balanced_acc = np.mean(
                [m["BalancedAcc"] for m in cv_metrics if not np.isnan(m["BalancedAcc"])]
            )
            mean_ber = np.mean([m["BER"] for m in cv_metrics if not np.isnan(m["BER"])])

        # AUC and LogLoss require probabilities — keep as mean-of-folds
        mean_auc = np.mean([m["ROC_AUC"] for m in cv_metrics if not np.isnan(m["ROC_AUC"])])
        mean_logloss = np.mean([m["LogLoss"] for m in cv_metrics if not np.isnan(m["LogLoss"])])

        regional_rmse = None  # Not applicable for classification

        # Per-class report uses pooled predictions (one per sample under repeated CV)
        if not repeated_cv:
            all_y_test = np.concatenate([m["y_test"] for m in cv_metrics])
            all_y_pred = np.concatenate([m["y_pred"] for m in cv_metrics])

        # Compute per-class metrics for classification (analogous to regional RMSE for regression)
        per_class_metrics = {}
        class_labels = None
        try:
            # Get per-class metrics from aggregated CV predictions
            report = classification_report(
                all_y_test, all_y_pred, output_dict=True, zero_division=0
            )
            class_labels = sorted(
                [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
            )
            for class_label in class_labels:
                class_key = str(class_label)
                if class_key in report:
                    per_class_metrics[class_key] = {
                        "F1": report[class_key]["f1-score"],
                        "Precision": report[class_key]["precision"],
                        "Recall": report[class_key]["recall"],
                        "Support": int(report[class_key]["support"]),
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
    cal_ccc = None
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
            cal_ccc = lins_ccc(y, y_pred_cal)
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
                        cal_auc = roc_auc_score(
                            y, y_pred_proba_cal, multi_class="ovr", average="macro"
                        )
                else:
                    cal_auc = np.nan
            except Exception as e:
                logger.debug(f"Failed to compute calibration ROC AUC: {e}")
                cal_auc = np.nan

            # Compute F1, Precision, Recall
            try:
                cal_f1 = f1_score(y, y_pred_cal, average="weighted", zero_division=0)
                cal_precision = precision_score(y, y_pred_cal, average="weighted", zero_division=0)
                cal_recall = recall_score(y, y_pred_cal, average="weighted", zero_division=0)
            except Exception as e:
                logger.debug(f"Failed to compute calibration F1/Precision/Recall: {e}")
                cal_f1 = np.nan
                cal_precision = np.nan
                cal_recall = np.nan

            # Compute new classification metrics
            try:
                cal_specificity = compute_specificity(y, y_pred_cal, average="macro")
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
            PIPELINE_META_PARAMS = {"verbose", "memory", "steps", "transform_input"}

            filtered_params = {}
            for key, value in all_params.items():
                # Skip Pipeline-specific parameters that cause issues when re-applied
                if key in PIPELINE_META_PARAMS:
                    continue

                # Skip callables and complex objects
                if callable(value) or hasattr(value, "__dict__"):
                    continue

                # Convert value to Python-native type for reliable serialization
                try:
                    # Handle numpy scalar types (np.int64, np.float64, etc.)
                    if hasattr(value, "item"):
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
    # Explicit None check (not `or`) so n_components == 0 isn't silently
    # replaced by the pipeline-prefixed fallback.
    if "n_components" in params:
        n_comp = params["n_components"]
    elif "pls__n_components" in params:
        n_comp = params["pls__n_components"]
    else:
        n_comp = None
    lvs = int(n_comp) if n_comp is not None else None

    # Format imbalance handling indicator for display
    if imbalance_method is None:
        imbalance_display = "—"
    elif imbalance_method == "class_weight":
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
        "Autoscale": preprocess_cfg.get("autoscale", False),  # T-36: UV scaling toggle
        # T-36 fix (post-merge review v2): persist baseline / smoothing metadata
        # so the validation rebuild path (compute_validation_metrics_for_top_models)
        # can reconstruct the same pipeline that search ran. Previously only the
        # one-class path emitted these — regression/classification rows would
        # silently snap back to defaults during validation rebuild whenever the
        # user picked non-default ALS / smoothing settings.
        "baseline_method": preprocess_cfg.get("baseline_method"),
        "baseline_params": preprocess_cfg.get("baseline_params"),
        "smoothing": preprocess_cfg.get("smoothing", False),
        "smoothing_window": preprocess_cfg.get("smoothing_window", 17),
        "smoothing_polyorder": preprocess_cfg.get("smoothing_polyorder", 2),
        "LVs": lvs,
        "n_vars": n_vars,
        "full_vars": full_vars,
        "SubsetTag": subset_tag,
        "Imbalance": imbalance_display,
        # Track early stopping to allow Model Development to reproduce boosted results
        "early_stopping_rounds": (
            early_stopping_rounds if model_name in ("XGBoost", "LightGBM", "CatBoost") else None
        ),
        # Store actual imbalance settings for Model Development tab to use
        # (imbalance_display is for UI, these are for exact pipeline reconstruction)
        "imbalance_method": imbalance_method,
        "imbalance_params": imbalance_params,
        "tpe_score": preprocess_cfg.get("tpe_score"),
        # Phase 4: multistart halt reason. None for single-start TPE configs
        # and exhaustive configs; one of {converged, cap, single_iteration}
        # for multistart configs. Symmetric with phase2_halt_reason for the
        # exhaustive path (added below for the chromosome-bearing rows).
        "tpe_multistart_halt_reason": preprocess_cfg.get("tpe_multistart_halt_reason"),
        # 2026-05-08: model-family-aware proxy audit (None for non-TPE rows).
        "tpe_proxy_family": preprocess_cfg.get("tpe_proxy_family"),
        "tpe_proxy_model_name": preprocess_cfg.get("tpe_proxy_model_name"),
    }

    # Add training configuration for tracking data state
    # This helps identify when Model Development tab uses different data
    #
    # cv_strategy is the source of truth; effective_folds is a cross-machine
    # compat field that lets old binaries read a sane integer even for LOO /
    # repeated K-fold. Never rely on getattr(cv_splitter, 'n_splits', folds) —
    # LeaveOneOut has no n_splits attribute, so that fallback would return the
    # stale `folds` kwarg (typically 5) which is wrong under LOO.
    if cv_strategy == "loo":
        effective_folds = len(X)
    else:
        # 'kfold' and 'repeated_kfold' both carry the same per-fold semantics;
        # repeat count is stored separately in cv_n_repeats.
        effective_folds = folds

    result["training_config"] = {
        "cv_strategy": cv_strategy,
        "cv_n_repeats": cv_n_repeats,
        "folds": effective_folds,
        "n_samples_used": len(X),  # Number of samples used for training (after filtering)
        "n_samples_total": total_samples_original if total_samples_original else len(X),
        "excluded_count": excluded_count,  # Number of excluded samples
        "validation_count": validation_count,  # Number of validation samples
        "n_features_used": X.shape[1],  # Number of features/wavelengths used
        "random_state": 42,  # CV random state (always 42 in this codebase)
    }

    # Store exhaustive-preprocessing chromosome if present (for Model Development
    # reconstruction). Column renamed 2026-05-06: this used to be `ga_genes`, but
    # that name collides with the GUI Refine tab's wavelength-index field of the
    # same name (which comes from GA-PLS variable selection). Renaming the
    # preprocessing-chromosome column to `preprocess_chromosome` removes the
    # ambiguity. The chromosome-rebuild reader (in
    # compute_validation_metrics_for_top_models) falls back to `ga_genes` so
    # old result CSVs continue to rebuild correctly.
    if (
        "preprocess_chromosome" in preprocess_cfg
        and preprocess_cfg["preprocess_chromosome"] is not None
    ):
        result["preprocess_chromosome"] = preprocess_cfg[
            "preprocess_chromosome"
        ].tolist()  # Serialize numpy array
        result["ga_model_type"] = preprocess_cfg.get("ga_model_type", "linear")
        result["ga_config"] = preprocess_cfg.get("ga_config", "")

    # Phase 2 halt-reason visibility (closes Codex MED #2 from post-Phase-4
    # review). Per-row column so users can see which exhaustive-preprocessing
    # rows came from a 'converged' rescore vs a 'cap'-hit rescore vs the
    # legacy 'disabled' single-seed path. Per-search value duplicated to
    # every row from one ga_result, but row-shape is the only way to
    # surface it without a CSV schema split.
    if "phase2_halt_reason" in preprocess_cfg:
        result["phase2_halt_reason"] = preprocess_cfg["phase2_halt_reason"]

    # Store Smart preprocessing metadata if present (for validation reconstruction)
    if (
        "smart_selected_wavelengths" in preprocess_cfg
        and preprocess_cfg["smart_selected_wavelengths"] is not None
    ):
        result["smart_selected_wavelengths"] = preprocess_cfg["smart_selected_wavelengths"]
        result["smart_n_wavelengths"] = preprocess_cfg.get("smart_n_wavelengths")
        result["smart_score"] = preprocess_cfg.get("smart_score")
        result["smart_importance_method"] = preprocess_cfg.get("smart_importance_method")
        result["smart_model_name"] = preprocess_cfg.get("smart_model_name")

    if task_type == "regression":
        # Calibration metrics (training data)
        result["RMSE"] = cal_rmse if cal_rmse is not None else np.nan
        result["R2"] = cal_r2 if cal_r2 is not None else np.nan
        result["CCC"] = cal_ccc if cal_ccc is not None else np.nan
        # Cross-validation metrics (pooled across folds)
        result["RMSEcv"] = mean_rmse
        result["R2cv"] = mean_r2
        result["CCCcv"] = ccc_cv
        # NIR-specific metrics (computed from aggregated CV predictions)
        result["MAEcv"] = mae_cv
        result["RPD"] = rpd
        result["Bias"] = bias_cv
        result["RER"] = rer
        # CV-ANOVA F-test (Eriksson, Trygg & Wold 2008)
        if model_name == "PLS" and lvs is not None:
            result["cv_anova_pvalue"] = compute_cv_anova_pvalue(
                y_true=y, rmsecv=mean_rmse, n_components=lvs,
            )
        else:
            result["cv_anova_pvalue"] = np.nan
        # Add regional performance for consensus predictions (dict format for ensemble)
        result["regional_rmse"] = regional_rmse
        result["y_quartiles"] = quartiles.tolist()  # Save quartile thresholds
        # Add individual quartile columns for display/sorting
        if regional_rmse is not None:
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                result[f"RMSE_{q}"] = regional_rmse.get(q, np.nan)
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
                result[f"F1_Class{class_label}"] = metrics["F1"]

    # Save all_vars for ALL models (including full spectrum)
    # This ensures Model Development can reconstruct the exact wavelengths used
    # CRITICAL: For full models, 'wavelengths' is already filtered by wl_min/wl_max
    # so we must save it to allow exact replication
    if subset_tag != "full" and subset_indices is not None:
        # Subset model: save only the subset wavelengths
        subset_wavelengths = wavelengths[subset_indices]
        all_vars_str = ",".join([f"{w:g}" for w in subset_wavelengths])
        result["all_vars"] = all_vars_str
    else:
        # Full model: save ALL wavelengths used (may be filtered by wl_min/wl_max)
        all_vars_str = ",".join([f"{w:g}" for w in wavelengths])
        result["all_vars"] = all_vars_str

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
            importances = get_feature_importances(fitted_model, model_name, X_transformed, y)

            # Apply edge masking for Savitzky-Golay derivatives (consistent with variable selection)
            # SKIP when wavelength restriction is active - restricted wavelengths
            # are from middle of spectrum, not SG boundary edges
            if not wavelength_restriction_active:
                importances = _apply_edge_mask(importances, preprocess_cfg)

            # Get top N features for display purposes (always top 30)
            n_to_select = min(top_n_vars, len(importances))
            # Use stable sort to ensure deterministic feature ordering when importances are tied
            top_indices = np.argsort(importances, kind="stable")[-n_to_select:][::-1]

            # Map back to original wavelengths
            if subset_indices is not None:
                # We're working with a subset, map indices back to original wavelengths
                original_wavelengths = wavelengths[subset_indices]
                top_wavelengths = original_wavelengths[top_indices]
            else:
                # Full spectrum
                top_wavelengths = wavelengths[top_indices]

            # Format as comma-separated string
            top_vars_str = ",".join([f"{w:g}" for w in top_wavelengths])
            result["top_vars"] = top_vars_str

        except Exception as e:
            # If anything fails, just mark as N/A
            result["top_vars"] = "N/A"
            # Keep all_vars that we already set above
    else:
        # For models that don't support importance extraction
        result["top_vars"] = "N/A"
        # Keep all_vars that we already set above

    return result


# ============================================================================
# ONE-CLASS DETECTION SEARCH
# ============================================================================


def _resolve_one_class_model_grids(
    enabled_models,
    oc_model_param_overrides=None,
    oc_hyperparams=None,
):
    from itertools import product
    from .contamination import get_one_class_model_grids

    oc_grids = get_one_class_model_grids()

    if not oc_model_param_overrides:
        if oc_hyperparams:
            for model_name, param_list in oc_grids.items():
                for params in param_list:
                    if model_name == "OneClassSVM" and "nu" in oc_hyperparams:
                        params["nu"] = oc_hyperparams["nu"]
                    if model_name in ("IsolationForest", "EllipticEnvelope", "LOF"):
                        if "contamination" in oc_hyperparams:
                            params["contamination"] = oc_hyperparams["contamination"]
                    if model_name == "PCA-SIMCA":
                        if "alpha" in oc_hyperparams:
                            params["alpha"] = oc_hyperparams["alpha"]
                        if "n_components" in oc_hyperparams:
                            params["n_components"] = oc_hyperparams["n_components"]
        model_grids = {}
        for model_name, param_list in oc_grids.items():
            if model_name in enabled_models:
                model_grids[model_name] = param_list
        return model_grids

    model_grids = {}
    for model_name in enabled_models:
        if model_name not in oc_grids:
            continue
        if model_name in oc_model_param_overrides:
            overrides = oc_model_param_overrides[model_name]
            if model_name == "OneClassSVM":
                model_grids[model_name] = _build_ocsvm_custom_grid(
                    overrides,
                    oc_grids["OneClassSVM"],
                )
            elif model_name == "IsolationForest":
                model_grids[model_name] = _build_if_custom_grid(
                    overrides,
                    oc_grids["IsolationForest"],
                )
            elif model_name == "EllipticEnvelope":
                model_grids[model_name] = _build_ee_custom_grid(
                    overrides,
                    oc_grids["EllipticEnvelope"],
                )
            elif model_name == "LOF":
                model_grids[model_name] = _build_lof_custom_grid(
                    overrides,
                    oc_grids["LOF"],
                )
            elif model_name == "PCA-SIMCA":
                model_grids[model_name] = _build_simca_custom_grid(
                    overrides,
                    oc_grids["PCA-SIMCA"],
                )
            else:
                model_grids[model_name] = oc_grids[model_name]
        else:
            model_grids[model_name] = oc_grids[model_name]

    return model_grids


def _oc_extract_defaults(grid, key):
    vals = set()
    for p in grid:
        if key in p:
            vals.add(p[key])
    return sorted(vals, key=lambda x: (isinstance(x, str), x))


def _build_ocsvm_custom_grid(overrides, default_grid):
    from itertools import product

    kernels = overrides.get("kernel", [])
    if not kernels:
        kernels = _oc_extract_defaults(default_grid, "kernel")
    gammas = overrides.get("gamma", [])
    if not gammas:
        gammas = _oc_extract_defaults(default_grid, "gamma")
    nus = overrides.get("nu", [])
    if not nus:
        nus = _oc_extract_defaults(default_grid, "nu")
    degrees = overrides.get("degree", [])
    if not degrees:
        degrees = [2]

    combos = []
    for k, g, nu in product(kernels, gammas, nus):
        if k == "poly":
            for d in degrees:
                combos.append({"kernel": k, "gamma": g, "nu": nu, "degree": d})
        else:
            combos.append({"kernel": k, "gamma": g, "nu": nu})
    return combos if combos else default_grid


def _build_if_custom_grid(overrides, default_grid):
    from itertools import product

    n_est = overrides.get("n_estimators", [])
    if not n_est:
        n_est = _oc_extract_defaults(default_grid, "n_estimators")
    contam = overrides.get("contamination", [])
    if not contam:
        contam = _oc_extract_defaults(default_grid, "contamination")
    max_feat = overrides.get("max_features", [])
    if not max_feat:
        max_feat = _oc_extract_defaults(default_grid, "max_features")
    max_samp = overrides.get("max_samples", [])
    if not max_samp:
        max_samp = _oc_extract_defaults(default_grid, "max_samples")
    if not max_samp:
        max_samp = ["auto"]

    combos = [
        {"n_estimators": int(n), "contamination": c, "max_features": mf, "max_samples": ms}
        for n, c, mf, ms in product(n_est, contam, max_feat, max_samp)
    ]
    return combos if combos else default_grid


def _build_ee_custom_grid(overrides, default_grid):
    from itertools import product

    contam = overrides.get("contamination", [])
    if not contam:
        contam = _oc_extract_defaults(default_grid, "contamination")
    support_fracs = overrides.get("support_fraction", [])
    if not support_fracs:
        # Any entry in default_grid without the key represents support_fraction=None.
        has_implicit_none = any("support_fraction" not in p for p in default_grid)
        explicit = _oc_extract_defaults(default_grid, "support_fraction")
        support_fracs = ([None] if has_implicit_none else []) + explicit
    if not support_fracs:
        support_fracs = [None]

    combos = []
    for c, sf in product(contam, support_fracs):
        entry: dict = {"contamination": c}
        if sf is not None:
            entry["support_fraction"] = sf
        combos.append(entry)
    return combos if combos else default_grid


def _build_lof_custom_grid(overrides, default_grid):
    from itertools import product

    nn = overrides.get("n_neighbors", [])
    if not nn:
        nn = _oc_extract_defaults(default_grid, "n_neighbors")
    contam = overrides.get("contamination", [])
    if not contam:
        contam = _oc_extract_defaults(default_grid, "contamination")
    metrics = overrides.get("metric", [])
    if not metrics:
        raw = _oc_extract_defaults(default_grid, "metric")
        metrics = raw if raw else ["euclidean"]

    combos = [
        {"n_neighbors": int(n), "contamination": c, "metric": m}
        for n, c, m in product(nn, contam, metrics)
    ]
    return combos if combos else default_grid


def _build_simca_custom_grid(overrides, default_grid):
    from itertools import product

    n_comp = overrides.get("n_components", [])
    if not n_comp:
        n_comp = _oc_extract_defaults(default_grid, "n_components")
    alphas = overrides.get("alpha", [])
    if not alphas:
        alphas = _oc_extract_defaults(default_grid, "alpha")

    combos = [{"n_components": nc, "alpha": a} for nc, a in product(n_comp, alphas)]
    return combos if combos else default_grid


def run_one_class_search(
    X,
    y,
    inlier_class_label,
    folds=5,
    cv_strategy="kfold",
    cv_n_repeats=5,
    preprocessing_methods=None,
    window_sizes=None,
    tier="standard",
    enabled_models=None,
    variable_penalty=0,
    gap_penalty=0,
    analysis_wl_min=None,
    analysis_wl_max=None,
    progress_callback=None,
    controller=None,
    baseline_method=None,
    baseline_params=None,
    enable_smoothing=False,
    smoothing_window=17,
    smoothing_polyorder=2,
    oc_hyperparams=None,
    oc_model_param_overrides=None,
    smart_preprocess=False,
    smart_preprocess_importance="model_specific",
    smart_preprocess_n_top=10,
    # T-37: TPE preprocessing discovery (supersedes smart + GA)
    tpe_preprocess=False,
    tpe_preprocess_n_trials=75,
    tpe_preprocess_n_top=10,
    tpe_enable_autoscale=True,
    tpe_multistart=False,  # Phase 4 (2026-05-06): multi-start + multi-seed rescore
    tpe_n_starts=5,
    # Variable selection
    variable_selection_methods=None,
    variable_counts=None,
    apply_uve_prefilter=False,
    uve_cutoff_multiplier=1.0,
    uve_n_components=None,
    ga_population_size=64,
    ga_generations=100,
    ga_n_runs=5,
    # T-36: autoscale (UV scaling) toggle — doubles preprocess_configs
    autoscale=False,
):
    """Run one-class model search.

    Trains models ONLY on inlier (clean) samples and evaluates detection
    of outliers (out-of-class samples). Uses a fundamentally different CV
    strategy than standard classification: training folds contain only
    inliers, while test folds contain both inliers and outliers.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (samples x wavelengths).
    y : pd.Series
        Class labels. One class is designated as the inlier (clean) class.
    inlier_class_label : str or int
        The label in y that represents clean/inlier samples.
        All other labels are treated as outliers (contaminated).
    folds : int, default=5
        Number of CV folds for inlier data.
    preprocessing_methods : list of str, optional
        Preprocessing methods to test (e.g., ['raw', 'snv', 'deriv1']).
    window_sizes : list of int, optional
        Savitzky-Golay window sizes.
    tier : str, default='standard'
        Model tier ('quick', 'standard', 'comprehensive', 'experimental').
    enabled_models : list of str, optional
        Explicit list of one-class models to test.
    analysis_wl_min : float, optional
        Minimum wavelength for analysis range.
    analysis_wl_max : float, optional
        Maximum wavelength for analysis range.
    progress_callback : callable, optional
        Callback for progress updates.
    controller : object, optional
        Pause/stop controller.
    baseline_method : str, optional
        Baseline correction method.
    baseline_params : dict, optional
        Baseline correction parameters.
    enable_smoothing : bool, default=False
        Whether to apply pre-smoothing.
    smoothing_window : int, default=17
        Smoothing window size.
    smoothing_polyorder : int, default=2
        Smoothing polynomial order.
    smart_preprocess : bool, default=False
        Whether to use smart preprocessing discovery.
    smart_preprocess_importance : str, default='model_specific'
        Importance method for preprocessing discovery.
    smart_preprocess_n_top : int, default=10
        Number of top preprocessing configs to discover.
    variable_selection_methods : list of str, optional
        Variable selection methods (e.g., ['importance', 'spa', 'cars']).
        If None or empty, no variable selection is performed.
    variable_counts : list of int, optional
        Number of top variables to test. Default: [10, 20, 50, 100, 250, 500, 1000].
    apply_uve_prefilter : bool, default=False
        Whether to apply UVE prefilter before other methods. Coerced to
        False early in this function (before any UVE-touching path runs)
        with a warning — UVE prefilter is a y-driven discrimination
        method, not a one-class method per CLAUDE.md:66 / Pomerantsev
        et al. 2025 LOVE.
    uve_cutoff_multiplier : float, default=1.0
        UVE cutoff multiplier for uninformative variable elimination.
    uve_n_components : int, optional
        Number of PLS components for UVE.
    ga_population_size : int, default=64
        Population size for GA variable selection.
    ga_generations : int, default=100
        Number of generations for GA variable selection.
    ga_n_runs : int, default=5
        Number of GA runs.

    Returns
    -------
    df_results : pd.DataFrame
        Results dataframe ranked by balanced accuracy.
    """
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler
    from .contamination import (
        build_one_class_model,
        get_one_class_model_grids,
        one_class_metrics,
        run_one_class_cv,
    )
    from .model_config import get_tier_models
    from .scoring import create_results_dataframe, add_result

    random_state = RANDOM_STATE

    # Validate inputs
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    wavelengths = X.columns.values if hasattr(X, "columns") else np.arange(X_np.shape[1])

    # Convert labels to one-class format: +1 = inlier, -1 = outlier
    # Compare as strings to handle both numeric and text labels consistently
    y_str = np.asarray(y_np, dtype=str)
    y_oc = np.where(y_str == str(inlier_class_label), 1, -1)
    inlier_mask = y_oc == 1
    outlier_mask = y_oc == -1
    inlier_indices = np.where(inlier_mask)[0]
    outlier_indices = np.where(outlier_mask)[0]
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)

    # UVE prefilter is a y-driven discrimination filter (CLAUDE.md:66) and
    # has no place in one-class screening. The GUI clears this checkbox at
    # spectral_predict_gui_optimized.py:16671, but scripted callers bypass
    # the GUI. Coerce to False with a warning so backend behavior matches
    # the documented one-class contract regardless of caller.
    if apply_uve_prefilter:
        logger.warning(
            "apply_uve_prefilter=True is not supported for one-class "
            "screening (UVE prefilter is a y-driven discrimination method, "
            "not a one-class method). Forcing apply_uve_prefilter=False."
        )
        apply_uve_prefilter = False

    logger.info("=" * 70)
    logger.info("ONE-CLASS SCREENING")
    logger.info("=" * 70)
    logger.info("Inlier class: '%s' (%d samples)", inlier_class_label, n_inliers)
    logger.info("Outlier classes: %d samples", n_outliers)
    if n_outliers > 0:
        outlier_labels = np.unique(y_np[outlier_mask])
        for lbl in outlier_labels:
            count = np.sum(y_np == lbl)
            logger.info("  - '%s': %d samples", lbl, count)
    else:
        logger.warning("No outlier samples! Evaluation will only measure specificity.")
    logger.info("=" * 70)

    # Upfront CV-strategy guard (n_repeats >= 1, one-class inlier counts,
    # LOO min-2 inlier rule). Raises ValueError before any training starts.
    from .cv_utils import validate_cv_strategy_for_task

    validate_cv_strategy_for_task(
        strategy=cv_strategy,
        task_type="one_class",
        y=y_oc,
        n_folds=folds,
        n_repeats=cv_n_repeats,
        inlier_label=1,  # y_oc is already +1/-1 encoded above
    )

    # Store full wavelengths before any masking (2a)
    wavelengths_full = wavelengths.copy()

    # Build wavelength mask but defer application until after preprocessing (2a)
    wl_mask = None
    wavelength_restriction_active = False
    if analysis_wl_min is not None or analysis_wl_max is not None:
        wl_float = wavelengths.astype(float)
        wl_mask = np.ones(len(wavelengths), dtype=bool)
        if analysis_wl_min is not None:
            wl_mask &= wl_float >= analysis_wl_min
        if analysis_wl_max is not None:
            wl_mask &= wl_float <= analysis_wl_max
        wavelength_restriction_active = True
        masked_wl = wavelengths[wl_mask]
        logger.info(
            "Wavelength range: %.1f - %.1f (%d features)",
            masked_wl[0],
            masked_wl[-1],
            len(masked_wl),
        )

    # Build preprocessing configs
    if preprocessing_methods is None:
        preprocessing_methods = ["raw", "snv", "deriv1", "deriv2", "snv_deriv1", "snv_deriv2"]
    if window_sizes is None:
        window_sizes = [7, 19]

    # T-37 fix (post-merge review): explicit mutual-exclusion guard, mirroring
    # the same guard added to run_search above so scripted callers learn about
    # conflicting flags instead of silently falling back to normal preprocessing.
    if sum(bool(f) for f in (smart_preprocess, tpe_preprocess)) > 1:
        raise ValueError(
            "smart_preprocess and tpe_preprocess are mutually exclusive — "
            "set at most one to True"
        )

    preprocess_configs = []
    if smart_preprocess and not tpe_preprocess:
        from .preprocessing_discovery import discover_preprocessing

        # Wrap progress callback to match discovery's (current, total, msg) signature
        def discovery_progress(current, total, message):
            if progress_callback:
                progress_callback(
                    {
                        "stage": "smart_preprocessing",
                        "message": message,
                        "current": current,
                        "total": total,
                    }
                )

        discovered = discover_preprocessing(
            X_np,
            y_oc,
            task_type="one_class",
            importance_method=smart_preprocess_importance,
            n_top=smart_preprocess_n_top,
            cv_folds=folds,
            progress_callback=discovery_progress,
        )
        if discovered:
            # Translate discovery output format to search config format
            for cfg in discovered:
                disc_name = cfg.get("preprocessing", "raw")
                disc_deriv = cfg.get("deriv")
                disc_window = cfg.get("window")
                # Derive the pipeline method name from the preprocessing name
                pipeline_method = disc_name
                for d in [4, 3, 2, 1]:
                    pipeline_method = pipeline_method.replace(str(d), "")
                display_name = disc_name + (f"_w{disc_window}" if disc_window else "")
                preprocess_configs.append(
                    {
                        "method": pipeline_method,
                        "name": display_name,
                        "deriv": disc_deriv,
                        "window": disc_window,
                        "polyorder": cfg.get("polyorder"),
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": enable_smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                    }
                )
            logger.info("Smart preprocessing discovered %d configs", len(preprocess_configs))

    if tpe_preprocess and not smart_preprocess:
        from .tpe_preprocessing_discovery import (
            run_tpe_preprocessing_discovery,
            run_tpe_multistart_preprocessing_discovery,
            resolve_tpe_proxy_family,
        )

        def tpe_oc_progress(current, total, message):
            if progress_callback:
                progress_callback(
                    {
                        "stage": "tpe_preprocessing",
                        "message": message,
                        "current": current,
                        "total": total,
                    }
                )

        # one_class call site mirrors the regression/classification TPE call
        # site (search for `tpe_multistart` to find both). Same gating flag,
        # same wrapper dispatch.
        # Family routing is functionally a no-op for one_class (proxy is
        # always LGBM-supervised-on-y_oc with IF fallback regardless of
        # family), but we plumb the resolved value for audit-trail
        # consistency (`_tpe_proxy_family` column).
        tpe_proxy_family_oc = resolve_tpe_proxy_family(enabled_models)
        if tpe_multistart:
            discovered = run_tpe_multistart_preprocessing_discovery(
                X_np,
                y_oc,
                task_type="one_class",
                n_trials=tpe_preprocess_n_trials,
                n_top=tpe_preprocess_n_top,
                cv_folds=folds,
                enable_autoscale=tpe_enable_autoscale,
                enable_baseline=(baseline_method is not None),
                enable_smoothing=enable_smoothing,
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder,
                n_starts=tpe_n_starts,
                progress_callback=tpe_oc_progress,
                controller=controller,
                proxy_family=tpe_proxy_family_oc,
            )
        else:
            discovered = run_tpe_preprocessing_discovery(
                X_np,
                y_oc,
                task_type="one_class",
                n_trials=tpe_preprocess_n_trials,
                n_top=tpe_preprocess_n_top,
                cv_folds=folds,
                enable_autoscale=tpe_enable_autoscale,
                enable_baseline=(baseline_method is not None),
                enable_smoothing=enable_smoothing,
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder,
                progress_callback=tpe_oc_progress,
                proxy_family=tpe_proxy_family_oc,
            )
        if discovered:
            preprocess_configs = []
            for cfg in discovered:
                disc_name = cfg.get("preprocessing", "raw")
                disc_deriv = cfg.get("deriv")
                disc_window = cfg.get("window")
                pipeline_method = disc_name
                for d in [4, 3, 2, 1]:
                    pipeline_method = pipeline_method.replace(str(d), "")
                display_name = disc_name + (f"_w{disc_window}" if disc_window else "")
                if cfg.get("_tpe_baseline_method"):
                    display_name = f"{cfg['_tpe_baseline_method']}+{display_name}"
                if cfg.get("_tpe_smoothing"):
                    display_name = f"sg0+{display_name}"
                if cfg.get("_tpe_autoscale"):
                    display_name = f"{display_name}+autoscale"
                preprocess_configs.append(
                    {
                        "method": pipeline_method,
                        "name": display_name,
                        "deriv": disc_deriv,
                        "window": disc_window,
                        "polyorder": cfg.get("polyorder"),
                        "baseline_method": cfg.get("_tpe_baseline_method"),
                        "baseline_params": cfg.get("_tpe_baseline_params"),
                        "smoothing": cfg.get("_tpe_smoothing", False),
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                        "autoscale": cfg.get("_tpe_autoscale", False),
                        "tpe_score": cfg.get("score"),
                        # Phase 4: multistart halt-reason propagation
                        # (one_class call site mirror of regression/cls path)
                        "tpe_multistart_halt_reason": cfg.get(
                            "_tpe_multistart_halt_reason"
                        ),
                        # 2026-05-08: model-family-aware proxy audit trail
                        # (one_class is family-independent — proxy is always
                        # LGBM-supervised + IF fallback regardless — but we
                        # plumb the field for audit-trail consistency).
                        "tpe_proxy_family": cfg.get("_tpe_proxy_family"),
                        "tpe_proxy_model_name": cfg.get("_tpe_proxy_model_name"),
                    }
                )
            logger.info("TPE preprocessing discovered %d configs", len(preprocess_configs))
            autoscale = False  # TPE configs already have per-config autoscale

    if not preprocess_configs:
        for method in preprocessing_methods:
            if method in ("deriv1", "deriv2", "snv_deriv1", "snv_deriv2"):
                deriv_order = 1 if method.endswith("1") else 2
                # Map display names to build_preprocessing_pipeline names
                pipeline_method = method.replace("1", "").replace(
                    "2", ""
                )  # deriv1->deriv, snv_deriv2->snv_deriv
                for ws in window_sizes:
                    preprocess_configs.append(
                        {
                            "method": pipeline_method,  # 2d: base method name for build_preprocessing_pipeline
                            "name": f"{method}_w{ws}",
                            "deriv": deriv_order,
                            "window": ws,
                            "polyorder": None,  # 2c: let SavgolDerivative auto-detect via polyorder_map
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": enable_smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )
            else:
                preprocess_configs.append(
                    {
                        "method": method,  # 2d: base method name for build_preprocessing_pipeline
                        "name": method,
                        "deriv": None,
                        "window": None,
                        "polyorder": None,  # 2c: consistent with derivative configs
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": enable_smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                    }
                )

    # --- Autoscale (UV scaling) toggle: when enabled, test both WITH and WITHOUT autoscale ---
    # T-36: mirrors run_search's grid-doubling block. One-class configs use 'method'
    # as the clean pipeline name (search.py:5326) rather than 'base_name'; we leave
    # 'method' alone and only suffix the display 'name'.
    if autoscale and preprocess_configs:
        configs_without_autoscale = []
        configs_with_autoscale = []
        for cfg in preprocess_configs:
            cfg_no = dict(cfg)
            cfg_no["autoscale"] = False
            configs_without_autoscale.append(cfg_no)
            cfg_sc = dict(cfg)
            cfg_sc["autoscale"] = True
            cfg_sc["name"] = cfg["name"] + "+autoscale"
            configs_with_autoscale.append(cfg_sc)
        preprocess_configs = configs_without_autoscale + configs_with_autoscale

    # Get model grids
    if enabled_models is None:
        enabled_models = get_tier_models(tier, task_type="one_class")

    model_grids = _resolve_one_class_model_grids(
        enabled_models,
        oc_model_param_overrides=oc_model_param_overrides,
        oc_hyperparams=oc_hyperparams,
    )

    # Filter and validate variable selection methods for one-class.
    # UVE family is excluded per CLAUDE.md:66 — UVE-on-y_oc is a discrimination
    # method (Pomerantsev et al. 2025 LOVE / Forina modeling-power vs
    # discrimination-power), not a one-class method. iPLS family is also
    # implicitly excluded (not in this whitelist) since it requires PLS
    # internals not available for one-class models. The GUI mirrors both
    # exclusions at spectral_predict_gui_optimized.py:16667-16683.
    implemented_oc_varsel = {
        "importance",
        "spa",
        "cars",
        "cars-tree",
        "ga",
        "vcpa-iriv",
    }
    selected_varsel_methods = []
    if variable_selection_methods:
        selected_varsel_methods = [
            m for m in variable_selection_methods if m in implemented_oc_varsel
        ]
        unsupported = [m for m in variable_selection_methods if m not in implemented_oc_varsel]
        if unsupported:
            logger.warning(
                "Variable selection methods not supported for one-class, skipping: %s",
                unsupported,
            )
        if selected_varsel_methods:
            logger.info("Variable selection methods: %s", selected_varsel_methods)

    # Determine variable counts for variable selection
    if variable_counts is None:
        oc_variable_counts = [10, 20, 50, 100, 250, 500, 1000]
    else:
        oc_variable_counts = list(variable_counts)

    # Calculate total configurations
    n_model_params = sum(len(params) for params in model_grids.values())
    full_spectrum_configs = n_model_params * len(preprocess_configs)

    # Variable selection adds: preprocess * methods * valid_counts * model_params
    varsel_configs = 0
    if selected_varsel_methods:
        n_estimated_counts = len(oc_variable_counts)
        varsel_configs = (
            len(preprocess_configs)
            * len(selected_varsel_methods)
            * n_estimated_counts
            * n_model_params
        )

    total_configs = full_spectrum_configs + varsel_configs
    current_config = 0

    logger.info("Models: %s", list(model_grids.keys()))
    logger.info("Preprocessing configs: %d", len(preprocess_configs))
    logger.info("Full-spectrum configurations: %d", full_spectrum_configs)
    if varsel_configs > 0:
        logger.info("Variable selection configurations (estimated): %d", varsel_configs)
    logger.info("Total configurations: %d", total_configs)
    logger.info(
        "CV strategy: %s (folds=%d, repeats=%d) on inlier data (outliers in test only)",
        cv_strategy,
        folds,
        cv_n_repeats,
    )
    if progress_callback:
        progress_callback(
            {
                "stage": "info",
                "message": f"Starting one-class search: {total_configs} configurations",
                "current": 0,
                "total": total_configs,
            }
        )

    # Create results container
    df_results = create_results_dataframe("one_class")
    best_result = None
    skipped_configs = 0  # 2g: track skipped configurations

    # cv_strategy and cv_n_repeats are forwarded to contamination.run_one_class_cv
    # which builds the appropriate CV splitter via build_cv_splitter().

    # Cache preprocessed data to avoid recomputing in the variable selection loop
    _preprocess_result_cache = {}
    _user_stopped = False

    for preprocess_cfg in preprocess_configs:
        if _user_stopped:
            break
        # Apply preprocessing to full dataset
        pipe_steps = build_preprocessing_pipeline(
            preprocess_cfg["method"],  # 2d: use dedicated 'method' key
            preprocess_cfg["deriv"],
            preprocess_cfg["window"],
            preprocess_cfg["polyorder"],
            task_type="one_class",
            baseline_method=preprocess_cfg.get("baseline_method"),
            baseline_params=preprocess_cfg.get("baseline_params"),
            smoothing=preprocess_cfg.get("smoothing", False),
            smoothing_window=preprocess_cfg.get("smoothing_window", 17),
            smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2),
            autoscale=preprocess_cfg.get("autoscale", False),  # T-36
        )

        # Apply preprocessing (fit on inlier data, transform all)
        if pipe_steps:
            from sklearn.pipeline import Pipeline as SkPipeline

            prep_pipe = SkPipeline(pipe_steps)
            try:  # 2k: narrow to expected error types
                # Fit on inlier data only (important for some preprocessing methods)
                prep_pipe.fit(X_np[inlier_indices])
                X_preprocessed = prep_pipe.transform(X_np)
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning("Preprocessing '%s' failed: %s", preprocess_cfg["name"], e)
                # 2k: increment current_config for all skipped configs in this preprocessing group
                n_skipped = sum(len(pl) for pl in model_grids.values())
                current_config += n_skipped
                skipped_configs += n_skipped  # 2g
                continue
        else:
            X_preprocessed = X_np.copy()

        # 2a: Apply wavelength mask AFTER preprocessing (deferred from before the loop)
        wavelengths_current = wavelengths_full.copy()
        if wavelength_restriction_active and wl_mask is not None:
            X_preprocessed = X_preprocessed[:, wl_mask]
            wavelengths_current = wavelengths_full[wl_mask]

        # 2b: Apply edge mask for derivative preprocessing when no wavelength restriction
        if (
            preprocess_cfg.get("deriv")
            and preprocess_cfg.get("window")
            and not wavelength_restriction_active
        ):
            X_preprocessed, wavelengths_current, _ = _apply_edge_mask_to_data(
                X_preprocessed, wavelengths_current, preprocess_cfg
            )

        # Cache final preprocessed result for reuse in variable selection loop
        _cache_key = (
            preprocess_cfg["name"],
            preprocess_cfg.get("deriv", 0),
            preprocess_cfg.get("window", 0),
        )
        _preprocess_result_cache[_cache_key] = (X_preprocessed.copy(), wavelengths_current.copy())

        for model_name, param_list in model_grids.items():
            if controller and not controller.check_and_wait():
                _user_stopped = True
                break

            for params in param_list:
                if controller and not controller.check_and_wait():
                    _user_stopped = True
                    break

                current_config += 1
                param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
                prep_name = preprocess_cfg["name"]
                progress_msg = f"Testing {model_name} ({param_str}) + {prep_name}"

                best_info = ""
                if best_result is not None:
                    best_info = (
                        f" | Best: BalAcc={best_result.get('BalancedAcccv', 0):.3f}, "
                        f"Sens={best_result.get('Sensitivitycv', 0):.3f}"
                    )
                logger.info("[%d/%d] %s%s", current_config, total_configs, progress_msg, best_info)

                if progress_callback:
                    progress_callback(
                        {
                            "stage": "model_testing",
                            "message": progress_msg,
                            "current": current_config,
                            "total": total_configs,
                            "best_model": best_result,
                        }
                    )

                # Run one-class CV
                cv_result = run_one_class_cv(
                    X_preprocessed,
                    y_oc,
                    model_name,
                    params,
                    n_folds=folds,
                    cv_strategy=cv_strategy,
                    cv_n_repeats=cv_n_repeats,
                    random_state=42,
                    y_original=y_np,
                )

                if cv_result.get("skipped", False):
                    logger.warning(
                        "[SKIP] Too few successful folds for %s + %s",
                        model_name,
                        preprocess_cfg["name"],
                    )
                    skipped_configs += 1
                    continue

                mean_m = cv_result["mean_metrics"]
                cal_metrics = cv_result["cal_metrics"]
                bal_acc_cv = mean_m["balanced_accuracy"]

                # Build result dict
                n_vars = X_preprocessed.shape[1]

                result = {
                    "Task": "one_class",
                    "Model": model_name,
                    "Params": str(params),
                    "Preprocess": preprocess_cfg["name"],
                    "Deriv": preprocess_cfg["deriv"],
                    "Window": preprocess_cfg["window"],
                    "Poly": preprocess_cfg["polyorder"],
                    "Autoscale": preprocess_cfg.get("autoscale", False),  # T-36
                    # T-36 bundled fix: write baseline/smoothing metadata so the
                    # one-class validation rebuild path (contamination.py:~1100) reads
                    # actual values rather than silently falling back to defaults.
                    "baseline_method": preprocess_cfg.get("baseline_method"),
                    # T-36 fix (post-merge review): persist baseline_params so
                    # non-default ALS/polynomial settings survive the rebuild.
                    "baseline_params": preprocess_cfg.get("baseline_params"),
                    "smoothing": preprocess_cfg.get("smoothing", False),
                    "smoothing_window": preprocess_cfg.get("smoothing_window", 17),
                    "smoothing_polyorder": preprocess_cfg.get("smoothing_polyorder", 2),
                    "LVs": params.get("n_components") if model_name == "PCA-SIMCA" else None,
                    "n_vars": n_vars,
                    "full_vars": len(wavelengths_current),
                    "SubsetTag": "full",
                    "Imbalance": "—",
                    # Calibration metrics
                    "Sensitivity": cal_metrics.get("sensitivity", np.nan),
                    "Specificity": cal_metrics.get("specificity", np.nan),
                    "Precision": cal_metrics.get("precision", np.nan),
                    "F1": cal_metrics.get("f1", np.nan),
                    "Accuracy": cal_metrics.get("accuracy", np.nan),
                    "BalancedAcc": cal_metrics.get("balanced_accuracy", np.nan),
                    "AUC": cal_metrics.get("auc", np.nan),
                    # CV metrics
                    "Sensitivitycv": mean_m["sensitivity"],
                    "Specificitycv": mean_m["specificity"],
                    "Precisioncv": mean_m["precision"],
                    "F1cv": mean_m["f1"],
                    "Accuracycv": mean_m["accuracy"],
                    "BalancedAcccv": bal_acc_cv,
                    "AUCcv": mean_m["auc"],
                    # Metadata
                    "n_inliers": n_inliers,
                    "n_outliers": n_outliers,
                    "inlier_class_label": str(inlier_class_label),
                    # PreprocessBase is the clean pipeline name (e.g.
                    # 'snv_deriv') accepted by build_preprocessing_pipeline.
                    # Display name (Preprocess) carries the window suffix
                    # ('snv_deriv1_w11') which the pipeline builder rejects.
                    # Mirrors the classification grid path at search.py:4506-4507.
                    "PreprocessBase": preprocess_cfg.get("method", preprocess_cfg["name"]),
                    "top_vars": "N/A",
                    "all_vars": ",".join([f"{float(w):g}" for w in wavelengths_current]),
                    "per_contaminant_sensitivity": cal_metrics.get("per_contaminant", {}),
                    # Persist scaler/PCA/stats for model save/load
                    "scaler": cv_result.get("cal_scaler"),
                    "pca_reducer": cv_result.get("cal_pca_reducer"),
                    "oc_score_stats": cv_result.get("oc_score_stats"),
                    "tpe_score": preprocess_cfg.get("tpe_score"),
                    # Phase 4 propagation (one_class path) — closes Codex
                    # residual STRONG on Fix #5: regression/cls result rows
                    # already carried this; one_class did not.
                    "tpe_multistart_halt_reason": preprocess_cfg.get(
                        "tpe_multistart_halt_reason"
                    ),
                    # 2026-05-08: model-family-aware proxy audit (one_class).
                    "tpe_proxy_family": preprocess_cfg.get("tpe_proxy_family"),
                    "tpe_proxy_model_name": preprocess_cfg.get("tpe_proxy_model_name"),
                }

                # Training config for model reproducibility (mirrors regression/classification)
                _eff_folds = len(X) if cv_strategy == "loo" else folds
                result["training_config"] = {
                    "cv_strategy": cv_strategy,
                    "cv_n_repeats": cv_n_repeats,
                    "folds": _eff_folds,
                    "n_samples_used": len(X),
                    "random_state": random_state,
                }

                # Add per-contaminant columns for display
                per_contam = cal_metrics.get("per_contaminant", {})
                for contam_label, contam_sens in per_contam.items():
                    result[f"Cal_Sens_{contam_label}"] = contam_sens

                df_results = add_result(df_results, result)

                # Show result
                sens_cv = mean_m["sensitivity"]
                spec_cv = mean_m["specificity"]
                contam_info = ""
                if per_contam:
                    contam_parts = [f"{k}={v:.2f}" for k, v in per_contam.items()]
                    contam_info = f", Per-contam: [{', '.join(contam_parts)}]"
                logger.info(
                    "     Sens=%.3f, Spec=%.3f, BalAcc=%.3f%s",
                    sens_cv,
                    spec_cv,
                    bal_acc_cv,
                    contam_info,
                )

                # Update best tracker
                if best_result is None or bal_acc_cv > best_result.get("BalancedAcccv", 0):
                    best_result = result

    # =========================================================================
    # Variable Selection Loop
    # =========================================================================
    # Only run if variable selection methods were requested and validated
    if selected_varsel_methods:
        from .contamination import compute_one_class_importances

        logger.info("=" * 70)
        logger.info("ONE-CLASS VARIABLE SELECTION")
        logger.info("=" * 70)
        logger.info("Methods: %s", selected_varsel_methods)
        logger.info("Variable counts: %s", oc_variable_counts)

        # Cache for variable selection results (keyed by preprocess+method)
        _oc_varsel_cache: dict = {}

        for preprocess_cfg in preprocess_configs:
            if _user_stopped:
                break
            if controller and not controller.check_and_wait():
                _user_stopped = True
                break

            # Reuse cached preprocessing result from full-spectrum loop
            _cache_key = (
                preprocess_cfg["name"],
                preprocess_cfg.get("deriv", 0),
                preprocess_cfg.get("window", 0),
            )
            if _cache_key in _preprocess_result_cache:
                X_preprocessed, wavelengths_current = _preprocess_result_cache[_cache_key]
                X_preprocessed = X_preprocessed.copy()  # Don't mutate cache
                wavelengths_current = wavelengths_current.copy()
            else:
                # Fallback: recompute if not in cache (e.g., smart_preprocess changed configs)
                pipe_steps = build_preprocessing_pipeline(
                    preprocess_cfg["method"],
                    preprocess_cfg["deriv"],
                    preprocess_cfg["window"],
                    preprocess_cfg["polyorder"],
                    task_type="one_class",
                    baseline_method=preprocess_cfg.get("baseline_method"),
                    baseline_params=preprocess_cfg.get("baseline_params"),
                    smoothing=preprocess_cfg.get("smoothing", False),
                    smoothing_window=preprocess_cfg.get("smoothing_window", 17),
                    smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2),
                    autoscale=preprocess_cfg.get("autoscale", False),  # T-36
                )

                if pipe_steps:
                    from sklearn.pipeline import Pipeline as SkPipeline

                    prep_pipe = SkPipeline(pipe_steps)
                    try:
                        prep_pipe.fit(X_np[inlier_indices])
                        X_preprocessed = prep_pipe.transform(X_np)
                    except (ValueError, np.linalg.LinAlgError) as e:
                        logger.warning(
                            "Preprocessing '%s' failed in varsel: %s",
                            preprocess_cfg["name"],
                            e,
                        )
                        n_skip = (
                            len(selected_varsel_methods) * len(oc_variable_counts) * n_model_params
                        )
                        current_config += n_skip
                        skipped_configs += n_skip
                        continue
                else:
                    X_preprocessed = X_np.copy()

                wavelengths_current = wavelengths_full.copy()
                if wavelength_restriction_active and wl_mask is not None:
                    X_preprocessed = X_preprocessed[:, wl_mask]
                    wavelengths_current = wavelengths_full[wl_mask]

                if (
                    preprocess_cfg.get("deriv")
                    and preprocess_cfg.get("window")
                    and not wavelength_restriction_active
                ):
                    X_preprocessed, wavelengths_current, _ = _apply_edge_mask_to_data(
                        X_preprocessed, wavelengths_current, preprocess_cfg
                    )

            n_features_current = X_preprocessed.shape[1]

            # --- UVE Prefilter: eliminate uninformative variables before varsel ---
            _uve_prefilter_active = False
            if apply_uve_prefilter and n_features_current >= 3:
                _uve_pf_key = (preprocess_cfg["name"], "__uve_prefilter__")
                if _uve_pf_key in _oc_varsel_cache:
                    _uve_mask = _oc_varsel_cache[_uve_pf_key]
                else:
                    try:
                        _uve_imp, _uve_thr, _uve_mask = get_uve_threshold(
                            X_preprocessed,
                            y_oc,
                            cutoff_multiplier=uve_cutoff_multiplier,
                            n_components=uve_n_components,
                            cv_folds=folds,
                            random_state=random_state,
                        )
                        _oc_varsel_cache[_uve_pf_key] = _uve_mask
                    except Exception as e:
                        logger.warning(
                            "UVE prefilter failed for '%s': %s", preprocess_cfg["name"], e
                        )
                        _uve_mask = np.ones(n_features_current, dtype=bool)
                        _oc_varsel_cache[_uve_pf_key] = _uve_mask

                n_before = n_features_current
                n_after = int(np.sum(_uve_mask))
                if n_after < n_before:
                    X_preprocessed = X_preprocessed[:, _uve_mask]
                    wavelengths_current = wavelengths_current[_uve_mask]
                    n_features_current = n_after
                    _uve_prefilter_active = True
                    logger.info(
                        "  UVE prefilter: %d -> %d variables (%d eliminated)",
                        n_before,
                        n_after,
                        n_before - n_after,
                    )
            elif apply_uve_prefilter and n_features_current < 3:
                logger.info("  UVE prefilter skipped: only %d features (min 3)", n_features_current)

            for varsel_method in selected_varsel_methods:
                if _user_stopped:
                    break
                if controller and not controller.check_and_wait():
                    _user_stopped = True
                    break

                logger.info(
                    "Computing %s importances for preprocess '%s'...",
                    varsel_method,
                    preprocess_cfg["name"],
                )

                # Cache key: (preprocess_name, varsel_method)
                _cache_key = (preprocess_cfg["name"], varsel_method)
                importances = None
                uve_selected_mask = None

                if _cache_key in _oc_varsel_cache:
                    importances = _oc_varsel_cache[_cache_key]["importances"]
                    uve_selected_mask = _oc_varsel_cache[_cache_key].get("uve_selected_mask")
                    logger.info("  Using cached %s result", varsel_method)
                else:
                    try:
                        if varsel_method == "importance":
                            # Use LightGBM binary classifier on y_oc
                            importances = compute_one_class_importances(
                                X_preprocessed,
                                y_oc,
                                method="lightgbm",
                                random_state=random_state,
                            )

                        elif varsel_method == "spa":
                            default_n_select = (
                                max(oc_variable_counts) if oc_variable_counts else 100
                            )
                            n_to_select = min(default_n_select, n_features_current)
                            importances = spa_selection(
                                X_preprocessed,
                                y_oc,
                                n_features=n_to_select,
                                cv_folds=folds,
                            )

                        elif varsel_method == "uve":
                            importances, _uve_threshold, uve_selected_mask = get_uve_threshold(
                                X_preprocessed,
                                y_oc,
                                cutoff_multiplier=uve_cutoff_multiplier,
                                n_components=uve_n_components,
                                cv_folds=folds,
                                random_state=random_state,
                            )

                        elif varsel_method == "uve_spa":
                            default_n_select = (
                                max(oc_variable_counts) if oc_variable_counts else 100
                            )
                            n_to_select = min(default_n_select, n_features_current)
                            importances = uve_spa_selection(
                                X_preprocessed,
                                y_oc,
                                n_features=n_to_select,
                                cutoff_multiplier=uve_cutoff_multiplier,
                                uve_n_components=uve_n_components,
                                uve_cv_folds=folds,
                                spa_cv_folds=folds,
                                random_state=random_state,
                            )

                        elif varsel_method in ("cars", "cars-tree"):
                            # For one-class, always use hybrid importance
                            # (LightGBM-based) on binary y_oc
                            use_hybrid = varsel_method == "cars-tree"
                            importances = cars_selection(
                                X_preprocessed,
                                y_oc,
                                n_iterations=50,
                                pls_components=(
                                    uve_n_components if uve_n_components is not None else 5
                                ),
                                cv_folds=folds,
                                monte_carlo_samples=80,
                                random_state=random_state,
                                model_type=None,
                                use_hybrid_importance=use_hybrid,
                                hybrid_importance_weight=0.5,
                                task_type="classification",
                            )

                        elif varsel_method in ("uve_cars", "uve_cars_tree"):
                            use_hybrid = varsel_method == "uve_cars_tree"
                            importances = uve_cars_selection(
                                X_preprocessed,
                                y_oc,
                                cutoff_multiplier=uve_cutoff_multiplier,
                                uve_n_components=uve_n_components,
                                uve_cv_folds=folds,
                                n_iterations=50,
                                pls_components=(
                                    uve_n_components if uve_n_components is not None else 5
                                ),
                                cars_cv_folds=folds,
                                monte_carlo_samples=80,
                                random_state=random_state,
                                model_type=None,
                                use_hybrid_importance=use_hybrid,
                                hybrid_importance_weight=0.5,
                                task_type="classification",
                            )

                        elif varsel_method == "uve_cars_spa":
                            importances = uve_cars_spa_selection(
                                X_preprocessed,
                                y_oc,
                                cutoff_multiplier=uve_cutoff_multiplier,
                                uve_n_components=uve_n_components,
                                uve_cv_folds=folds,
                                n_iterations=50,
                                pls_components=(
                                    uve_n_components if uve_n_components is not None else 5
                                ),
                                cars_cv_folds=folds,
                                monte_carlo_samples=80,
                                spa_n_features=None,
                                spa_cv_folds=folds,
                                random_state=random_state,
                                task_type="classification",
                            )

                        elif varsel_method == "vcpa-iriv":
                            result_vcpa = vcpa_iriv(
                                X_preprocessed,
                                y_oc,
                                n_outer_iterations=10,
                                n_inner_iterations=50,
                                pls_components=(
                                    uve_n_components if uve_n_components is not None else 5
                                ),
                                cv_folds=folds,
                                random_state=random_state,
                            )
                            importances = result_vcpa.get(
                                "importance_scores",
                                result_vcpa.get("importances", None),
                            )
                            selected = result_vcpa.get("selected_indices", [])
                            if importances is not None and len(importances) == len(selected):
                                full_importances = np.zeros(n_features_current)
                                full_importances[selected] = importances
                                importances = full_importances
                            elif len(selected) > 0:
                                importances = np.zeros(n_features_current)
                                importances[selected] = 1.0
                            else:
                                importances = np.ones(n_features_current)

                        elif varsel_method == "ga":
                            # GA: use LightGBM fitness for one-class
                            # (binary classification on y_oc)
                            ga_pop = ga_population_size
                            ga_gen = ga_generations
                            ga_runs_val = ga_n_runs
                            ga_early = 20
                            importances = ga_lightgbm_selection(
                                X_preprocessed,
                                y_oc,
                                task_type="classification",
                                cv_folds=folds,
                                n_estimators=50,
                                num_leaves=15,
                                population_size=ga_pop,
                                n_generations=ga_gen,
                                n_runs=ga_runs_val,
                                early_stopping=ga_early,
                                random_state=random_state,
                                progress_callback=progress_callback,
                            )

                        else:
                            logger.warning(
                                "Unhandled varsel method '%s' for one-class, skipping",
                                varsel_method,
                            )
                            n_skip = len(oc_variable_counts) * n_model_params
                            current_config += n_skip
                            skipped_configs += n_skip
                            continue

                    except Exception as e:
                        logger.warning(
                            "Variable selection '%s' failed for preprocess '%s': %s",
                            varsel_method,
                            preprocess_cfg["name"],
                            e,
                        )
                        n_skip = len(oc_variable_counts) * n_model_params
                        current_config += n_skip
                        skipped_configs += n_skip
                        continue

                    # Cache the result
                    _oc_varsel_cache[_cache_key] = {
                        "importances": importances,
                        "uve_selected_mask": uve_selected_mask,
                    }

                # Validate importances
                if importances is None:
                    logger.warning("%s returned None importances, skipping", varsel_method)
                    n_skip = len(oc_variable_counts) * n_model_params
                    current_config += n_skip
                    skipped_configs += n_skip
                    continue

                if len(importances) != n_features_current:
                    logger.warning(
                        "%s returned wrong-sized importances (%d vs %d), skipping",
                        varsel_method,
                        len(importances),
                        n_features_current,
                    )
                    n_skip = len(oc_variable_counts) * n_model_params
                    current_config += n_skip
                    skipped_configs += n_skip
                    continue

                used_uniform_fallback = False
                if np.all(importances == 0):
                    logger.warning(
                        "%s returned all-zero importances, using uniform",
                        varsel_method,
                    )
                    importances = np.ones(n_features_current)
                    used_uniform_fallback = True

                # Apply edge mask for derivatives
                # Skip when UVE prefilter active — variables are non-contiguous
                if not wavelength_restriction_active and not _uve_prefilter_active:
                    importances = _apply_edge_mask(importances, preprocess_cfg)

                # Filter valid variable counts
                valid_counts = [c for c in oc_variable_counts if c < n_features_current]
                if not valid_counts:
                    logger.warning(
                        "No valid variable counts (all >= %d features), skipping %s",
                        n_features_current,
                        varsel_method,
                    )
                    continue

                logger.info(
                    "  Valid variable counts: %s (features: %d)",
                    valid_counts,
                    n_features_current,
                )

                for n_vars in valid_counts:
                    top_indices = np.argsort(importances, kind="stable")[-n_vars:]
                    X_subset = X_preprocessed[:, top_indices]
                    wavelengths_subset = wavelengths_current[top_indices]

                    for model_name, param_list in model_grids.items():
                        if _user_stopped:
                            break
                        if controller and not controller.check_and_wait():
                            _user_stopped = True
                            break

                        for params in param_list:
                            if _user_stopped:
                                break
                            if controller and not controller.check_and_wait():
                                _user_stopped = True
                                break

                            current_config += 1
                            param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
                            prep_name = preprocess_cfg["name"]
                            subset_tag = f"{varsel_method}_top{n_vars}"
                            progress_msg = (
                                f"Testing {model_name} ({param_str}) + "
                                f"{prep_name} [{subset_tag}]"
                            )

                            best_info = ""
                            if best_result is not None:
                                best_info = (
                                    f" | Best: BalAcc=" f"{best_result.get('BalancedAcccv', 0):.3f}"
                                )
                            logger.info(
                                "[%d/%d] %s%s",
                                current_config,
                                total_configs,
                                progress_msg,
                                best_info,
                            )

                            if progress_callback:
                                progress_callback(
                                    {
                                        "stage": "model_testing",
                                        "message": progress_msg,
                                        "current": current_config,
                                        "total": total_configs,
                                        "best_model": best_result,
                                    }
                                )

                            cv_result = run_one_class_cv(
                                X_subset,
                                y_oc,
                                model_name,
                                params,
                                n_folds=folds,
                                cv_strategy=cv_strategy,
                                cv_n_repeats=cv_n_repeats,
                                random_state=42,
                                y_original=y_np,
                            )

                            if cv_result.get("skipped", False):
                                logger.warning(
                                    "[SKIP] Too few successful folds for " "%s + %s [%s]",
                                    model_name,
                                    prep_name,
                                    subset_tag,
                                )
                                skipped_configs += 1
                                continue

                            mean_m = cv_result["mean_metrics"]
                            cal_metrics = cv_result["cal_metrics"]
                            bal_acc_cv = mean_m["balanced_accuracy"]

                            # Build result dict with variable selection info
                            result = {
                                "Task": "one_class",
                                "Model": model_name,
                                "Params": str(params),
                                "Preprocess": preprocess_cfg["name"],
                                # See full-spectrum branch at search.py:5136
                                # for why both Preprocess (display) and
                                # PreprocessBase (clean pipeline name) are
                                # required for downstream validation rebuild.
                                "PreprocessBase": preprocess_cfg.get(
                                    "method", preprocess_cfg["name"]
                                ),
                                "Deriv": preprocess_cfg["deriv"],
                                "Window": preprocess_cfg["window"],
                                "Poly": preprocess_cfg["polyorder"],
                                "Autoscale": preprocess_cfg.get("autoscale", False),  # T-36
                                # T-36 bundled fix: write baseline/smoothing metadata
                                # so the validation rebuild path reads real values.
                                "baseline_method": preprocess_cfg.get("baseline_method"),
                                # T-36 fix (post-merge review): persist baseline_params.
                                "baseline_params": preprocess_cfg.get("baseline_params"),
                                "smoothing": preprocess_cfg.get("smoothing", False),
                                "smoothing_window": preprocess_cfg.get("smoothing_window", 17),
                                "smoothing_polyorder": preprocess_cfg.get("smoothing_polyorder", 2),
                                "LVs": (
                                    params.get("n_components")
                                    if model_name == "PCA-SIMCA"
                                    else None
                                ),
                                "n_vars": n_vars,
                                "full_vars": n_features_current,
                                "SubsetTag": subset_tag,
                                "Imbalance": "—",
                                # Calibration metrics
                                "Sensitivity": cal_metrics.get("sensitivity", np.nan),
                                "Specificity": cal_metrics.get("specificity", np.nan),
                                "Precision": cal_metrics.get("precision", np.nan),
                                "F1": cal_metrics.get("f1", np.nan),
                                "Accuracy": cal_metrics.get("accuracy", np.nan),
                                "BalancedAcc": cal_metrics.get("balanced_accuracy", np.nan),
                                "AUC": cal_metrics.get("auc", np.nan),
                                # CV metrics
                                "Sensitivitycv": mean_m["sensitivity"],
                                "Specificitycv": mean_m["specificity"],
                                "Precisioncv": mean_m["precision"],
                                "F1cv": mean_m["f1"],
                                "Accuracycv": mean_m["accuracy"],
                                "BalancedAcccv": bal_acc_cv,
                                "AUCcv": mean_m["auc"],
                                # Metadata
                                "n_inliers": n_inliers,
                                "n_outliers": n_outliers,
                                "inlier_class_label": str(inlier_class_label),
                                # Both top_vars and all_vars must store the
                                # SELECTED subset (the wavelengths the model
                                # was actually trained on). Downstream
                                # consumers — Model Development reload at
                                # spectral_predict_gui_optimized.py:30556 and
                                # external validation at contamination.py:972
                                # — read all_vars as "the trained wavelength
                                # list". Storing the pre-subset working set
                                # there caused variable-selected grid-search
                                # one-class models to be reconstructed on the
                                # full spectrum, producing wrong predictions.
                                # Mirrors the Bayesian contract at
                                # unified_bayesian.py:1046-1050.
                                "top_vars": ",".join(
                                    [f"{float(w):g}" for w in wavelengths_subset]
                                ),
                                "all_vars": ",".join(
                                    [f"{float(w):g}" for w in wavelengths_subset]
                                ),
                                "uniform_fallback": used_uniform_fallback,
                                "per_contaminant_sensitivity": cal_metrics.get(
                                    "per_contaminant", {}
                                ),
                                # Persist scaler/PCA/stats for model save/load
                                "scaler": cv_result.get("cal_scaler"),
                                "pca_reducer": cv_result.get("cal_pca_reducer"),
                                "oc_score_stats": cv_result.get("oc_score_stats"),
                                "tpe_score": preprocess_cfg.get("tpe_score"),
                                # Phase 4 propagation — second one_class result
                                # site (mirror of the first oc-result-dict).
                                "tpe_multistart_halt_reason": preprocess_cfg.get(
                                    "tpe_multistart_halt_reason"
                                ),
                                # 2026-05-08: model-family-aware proxy audit
                                # (one_class variable-subset row mirror).
                                "tpe_proxy_family": preprocess_cfg.get("tpe_proxy_family"),
                                "tpe_proxy_model_name": preprocess_cfg.get("tpe_proxy_model_name"),
                            }

                            # Training config for model reproducibility
                            _eff_folds = len(X) if cv_strategy == "loo" else folds
                            result["training_config"] = {
                                "cv_strategy": cv_strategy,
                                "cv_n_repeats": cv_n_repeats,
                                "folds": _eff_folds,
                                "n_samples_used": len(X),
                                "random_state": random_state,
                            }

                            # Add per-contaminant columns
                            per_contam = cal_metrics.get("per_contaminant", {})
                            for contam_label, contam_sens in per_contam.items():
                                result[f"Cal_Sens_{contam_label}"] = contam_sens

                            df_results = add_result(df_results, result)

                            # Log result
                            sens_cv = mean_m["sensitivity"]
                            spec_cv = mean_m["specificity"]
                            contam_info = ""
                            if per_contam:
                                contam_parts = [f"{k}={v:.2f}" for k, v in per_contam.items()]
                                contam_info = f", Per-contam: " f"[{', '.join(contam_parts)}]"
                            logger.info(
                                "     Sens=%.3f, Spec=%.3f, BalAcc=%.3f%s",
                                sens_cv,
                                spec_cv,
                                bal_acc_cv,
                                contam_info,
                            )

                            # Update best tracker
                            if best_result is None or bal_acc_cv > best_result.get(
                                "BalancedAcccv", 0
                            ):
                                best_result = result

    # Rank results using composite score (consistent with regression/classification)
    if len(df_results) > 0:
        from .scoring import compute_composite_score

        df_results = compute_composite_score(df_results, "one_class", variable_penalty, gap_penalty)

    logger.info("=" * 70)
    logger.info("ONE-CLASS SEARCH COMPLETE")
    logger.info("=" * 70)
    logger.info("Total configurations tested: %d", len(df_results))
    logger.info("Skipped configurations: %d", skipped_configs)  # 2g
    if len(df_results) > 0:
        best = df_results.iloc[0]
        logger.info("Best model: %s + %s", best["Model"], best["Preprocess"])
        logger.info("  Sensitivity (CV): %.3f", best["Sensitivitycv"])
        logger.info("  Specificity (CV): %.3f", best["Specificitycv"])
        logger.info("  Balanced Accuracy (CV): %.3f", best["BalancedAcccv"])
        logger.info("  AUC (CV): %.3f", best["AUCcv"])
    logger.info("=" * 70)
    if progress_callback:
        progress_callback(
            {
                "stage": "info",
                "message": (
                    f"One-class search complete: {len(df_results)} results, "
                    f"{skipped_configs} skipped"
                ),
                "current": total_configs,
                "total": total_configs,
            }
        )

    return df_results


# ============================================================================
# T-31 Phase C / task C2: run_multiclass_simca_search
# ============================================================================

# Maps a search-layer varsel_path token to the MultiClassClassModel
# variable_selection constructor value (spec §7 / Phase-B guardrail #7). "none"
# -> no prefilter; the wold_* / importance strings pass straight through.
_MULTICLASS_VARSEL_PATHS: dict[str, str | None] = {
    "none": None,
    "wold_modeling": "wold_modeling",
    "wold_discriminating": "wold_discriminating",
    "wold_balanced": "wold_balanced",
    "importance": "importance",
}


def _multiclass_loco_novelty_auc(build_model_fn, X, y, cv_splits=5, oof_cv=None):
    """Leave-one-class-out novelty-vs-false-rejection AUC (spec §7 ranking metric).

    A training set has NO truly-novel samples, so
    :func:`~spectral_predict.simca.novelty_tradeoff_auc` on the in-sample OOF
    decision matrix returns NaN (its novel-sample denominator is zero). This
    helper synthesizes the novel population by leave-one-class-out (LOCO): each
    class in turn is treated as "novel" against a model fit on the remaining
    ``K-1`` classes, while the false-rejection axis comes from the in-sample OOF
    own-class p-values. The α-sweep AUC of ``novelty_rate`` vs
    ``false_rejection_rate`` is the ranking metric.

    Parameters
    ----------
    build_model_fn : callable
        Zero-arg factory returning a FRESH (unfitted)
        :class:`~spectral_predict.simca.MultiClassClassModel` configured exactly
        as the row under test (engine / alpha / scaling / varsel / ...). Called
        once for the OOF CV and once per held-out class.
    X : ndarray of shape (n_samples, n_features)
        Preprocessed spectra (per-spectrum ops already applied outside folds).
    y : array-like of shape (n_samples,)
        Class labels.
    cv_splits : int, default=5
        Outer folds for the OOF own-class p-values. IGNORED when ``oof_cv`` is
        supplied (the caller's precomputed CV is reused verbatim).
    oof_cv : dict, optional
        A precomputed ``MultiClassClassModel.cross_validate`` result (with keys
        ``decision_matrix`` and ``classes``). When given, step (1)'s OOF CV is
        reused instead of recomputed — the search loop already computes it for
        the leaderboard metrics, so passing it halves the per-config OOF CV cost.
        Must correspond to the SAME config ``build_model_fn`` produces.

    Returns
    -------
    float
        AUC in ``[0, 1]``; ``float("nan")`` if it cannot be computed (no finite
        own-class p-values, or every LOCO fit failed).

    The curve plots ``novelty_rate`` (y) against ``false_rejection_rate`` (x)
    over an α-sweep; equivalently novelty vs ``1 - correct-acceptance``. The
    trapezoidal AUC is direction-invariant (``trapz`` integrates whichever way
    the x-values are ordered), so the exact axis label does not affect ranking.

    LOCO is a WITHIN-DATASET novelty PROXY that OVER-estimates the spec §1
    held-out-foreign-class target: each held-out class is ruled on by only
    ``K-1`` models and is itself in-distribution, whereas a truly foreign class
    faces all ``K`` ruling models. Treat ``NoveltyAUC`` as an optimistic upper
    bound and verify on a genuine held-out novel class (Decision A).

    Notes
    -----
    - ``false_rejection_rate(α)`` = fraction of known rows whose OWN-class OOF
      p-value ``< α`` (rows with a NaN own-p — e.g. their class was unmodelable
      in every fold — are excluded from the denominator).
    - ``novelty_rate(α)`` is CLASS-BALANCED (Decision B, spec §7
      "small-class-robust"): per held-out class ``c`` it is the fraction of
      class-``c`` rows flagged novel (``max(finite foreign p) < α``); the overall
      rate is the mean over classes, so each class weighs equally regardless of
      size. A held row whose foreign p-values are ALL NaN (every ``K-1`` class
      unmodelable in that LOCO fit) is EXCLUDED from both numerator and
      denominator (B4 — it must not be treated as vacuously novel).
    - Threshold sweep = the sorted unique finite p-values (own + foreign) plus
      ``{0, 1, 2}`` (the ``2`` guarantees the false-rejection axis reaches 1 so a
      perfect separator does not collapse to AUC 0 — B1), downsampled to ≤2000
      thresholds for real-data scale; duplicate false-rejection x-values are
      deduped (max novelty kept) before the trapezoid.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)

    # (1) In-sample OOF own-class p-values -> false-rejection axis. Reuse the
    # caller's already-computed OOF CV when provided (the search loop computes it
    # for the leaderboard metrics), halving the per-config CV cost at real-data
    # scale (757x2151, K=10) instead of running cross_validate twice.
    if oof_cv is not None:
        cv = oof_cv
    else:
        try:
            cv = build_model_fn().cross_validate(X, y, n_splits=cv_splits)
        except Exception as exc:  # noqa: BLE001 — NaN-safe: any CV failure -> NaN AUC
            logger.warning("multiclass LOCO AUC: OOF cross_validate failed: %s", exc)
            return float("nan")
    P_oof, _ = cv["decision_matrix"]
    col_of = {c: k for k, c in enumerate(list(cv["classes"]))}
    own_p = np.array(
        [P_oof[i, col_of[y[i]]] if y[i] in col_of else np.nan for i in range(len(y))],
        dtype=np.float64,
    )
    own_p_finite = own_p[np.isfinite(own_p)]

    # (2) LOCO -> novelty axis. Each class held out as the novel population.
    # Per-class novelty statistics (max finite foreign p per row, or NaN if a row
    # has NO finite foreign p — B4: excluded from num+denom, never vacuously
    # novel). Kept per class so novelty_rate can be CLASS-BALANCED (B5).
    nov_by_class: dict = {}
    for c in classes:
        held_mask = y == c
        if not np.any(held_mask):
            continue
        try:
            m = build_model_fn()
            m.fit(X[y != c], y[y != c])
            P_c, _ = m.decision_matrix(X[held_mask])
        except Exception as exc:  # noqa: BLE001 — skip a failed LOCO fold, NaN-safe
            logger.warning("multiclass LOCO AUC: LOCO fit for class %r failed: %s", c, exc)
            continue
        stats_c = []
        for row in P_c:
            finite = row[np.isfinite(row)]
            stats_c.append(np.nan if finite.size == 0 else float(np.max(finite)))
        nov_by_class[c] = np.asarray(stats_c, dtype=np.float64)

    # Drop classes with no valid (finite) held rows; if none remain there is no
    # novel population to score.
    valid_by_class = [
        arr[np.isfinite(arr)] for arr in nov_by_class.values()
    ]
    valid_by_class = [arr for arr in valid_by_class if arr.size > 0]
    if own_p_finite.size == 0 or len(valid_by_class) == 0:
        return float("nan")

    # (3) Threshold sweep + trapezoid AUC.
    all_foreign = np.concatenate(valid_by_class)
    # The 2.0 anchor forces false_rej to reach 1.0 (B1: a perfect separator whose
    # own-p are all 1.0 otherwise never leaves x=0 -> AUC 0).
    thresholds = np.unique(
        np.concatenate([own_p_finite, all_foreign, [0.0, 1.0, 2.0]])
    )
    if thresholds.size > 2000:
        logger.info(
            "multiclass LOCO AUC: downsampling %d thresholds to 2000",
            thresholds.size,
        )
        idx = np.unique(np.linspace(0, thresholds.size - 1, 2000).astype(int))
        thresholds = thresholds[idx]

    def _class_balanced_novelty(a: float) -> float:
        # Mean over classes of each class's own-row novel fraction (B5).
        rates = [float(np.mean(arr < a)) for arr in valid_by_class]
        return float(np.mean(rates))

    false_rej = np.array([float(np.mean(own_p_finite < a)) for a in thresholds])
    novelty = np.array([_class_balanced_novelty(a) for a in thresholds])

    # Dedupe duplicate false_rej x-values keeping the MAX novelty (B1), then
    # trapezoid over the (sorted-unique) x.
    uniq_fr, inv = np.unique(false_rej, return_inverse=True)
    max_nov = np.full(uniq_fr.shape, -np.inf, dtype=np.float64)
    np.maximum.at(max_nov, inv, novelty)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz(max_nov, uniq_fr))
    return float(min(1.0, max(0.0, auc)))


def _multiclass_preprocess_matrix(X_np, preprocess_cfg, wavelengths_full):
    """Apply one multi-class preprocessing config to the FULL matrix.

    Per-spectrum ops (SNV / SG derivatives / baseline) are applied outside folds
    (chemometrics convention — NOT leakage); column-autoscale / calibration /
    varsel stay train-only inside the model. A malformed config returns
    ``(None, wavelengths, reason)`` instead of raising, so one bad config never
    aborts the whole search (spec §8 / Codex M2). Shared by the search loop and
    :func:`build_multiclass_decision_view` so both preprocess identically.

    Returns ``(X_pp | None, wavelengths_current, reason)``.
    """
    wavelengths_current = np.asarray(wavelengths_full).copy()
    try:
        pipe_steps = build_preprocessing_pipeline(
            preprocess_cfg["method"],
            preprocess_cfg["deriv"],
            preprocess_cfg["window"],
            preprocess_cfg["polyorder"],
            baseline_method=preprocess_cfg.get("baseline_method"),
            baseline_params=preprocess_cfg.get("baseline_params"),
            smoothing=preprocess_cfg.get("smoothing", False),
            smoothing_window=preprocess_cfg.get("smoothing_window", 17),
            smoothing_polyorder=preprocess_cfg.get("smoothing_polyorder", 2),
        )
        if pipe_steps:
            from sklearn.pipeline import Pipeline as SkPipeline

            prep_pipe = SkPipeline(pipe_steps)
            prep_pipe.fit(X_np)
            X_pp = prep_pipe.transform(X_np)
        else:
            X_pp = X_np.copy()

        if preprocess_cfg.get("deriv") and preprocess_cfg.get("window"):
            X_pp, wavelengths_current, _ = _apply_edge_mask_to_data(
                X_pp, wavelengths_current, preprocess_cfg
            )
        return X_pp, wavelengths_current, ""
    except Exception as exc:  # noqa: BLE001 — one bad config must not abort
        return (
            None,
            wavelengths_current,
            f"preprocessing_failed: {type(exc).__name__}: {exc}",
        )


def build_multiclass_decision_view(
    X,
    y,
    *,
    engine,
    preprocess_cfg,
    alpha=0.05,
    n_components=0.99,
    scaling="per_class",
    min_class_samples=10,
    variable_selection=None,
    n_select=None,
    wavelengths=None,
    sample_ids=None,
):
    """Fit ONE multi-class config on the full data and return its per-sample
    decision view (T-31 Phase D2 data provider; the GUI renders it).

    The model is fit on the full (preprocessed) training matrix and evaluated on
    it — this is the in-sample decision matrix shown to the user, NOT a
    leakage-safe estimate (those are the leaderboard's OOF/LOCO metrics). It is
    the "here is what every trained class model says about every sample" view.

    Returns a dict with keys:
        ``classes`` (list), ``p_values`` (n, K float), ``accept`` (n, K bool),
        ``labels`` (n,) single-class / ``"multiple"`` / ``"novel"``,
        ``true_labels`` (n,), ``sample_ids`` (n,), ``resolved_n_components``
        (dict class -> int), ``unmodelable_classes`` (list),
        ``wold`` (:func:`wold_diagnostic_plot_data` dict, or ``None`` if it
        could not be computed), ``preprocess_name`` (str), ``reason`` (str;
        non-empty only on preprocessing/fit failure).
    """
    from .simca import MultiClassClassModel, wold_diagnostic_plot_data

    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    X_np = np.asarray(X_np, dtype=np.float64)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    if wavelengths is not None:
        wavelengths_full = np.asarray(wavelengths)
    elif hasattr(X, "columns"):
        wavelengths_full = np.asarray(X.columns.values)
    else:
        wavelengths_full = np.arange(X_np.shape[1])
    if sample_ids is None:
        sample_ids = (
            list(X.index) if hasattr(X, "index") else list(range(X_np.shape[0]))
        )

    # Full config echoed back so the view is self-describing (Phase D3 export can
    # regenerate the exact decision matrix from it).
    config = {
        "engine": engine,
        "alpha": alpha,
        "n_components": n_components,
        "scaling": scaling,
        "min_class_samples": min_class_samples,
        "variable_selection": variable_selection,
        "n_select": n_select,
        "preprocess_cfg": dict(preprocess_cfg),
    }

    empty = {
        "classes": [],
        "p_values": np.empty((0, 0)),
        "accept": np.empty((0, 0), dtype=bool),
        "labels": [],
        "true_labels": list(y_np),
        "sample_ids": list(sample_ids),
        "resolved_n_components": {},
        "unmodelable_classes": [],
        "wold": None,
        "wold_error": "",
        "preprocess_name": preprocess_cfg.get("name", ""),
        "config": config,
        "reason": "",
    }

    X_pp, wl_current, reason = _multiclass_preprocess_matrix(
        X_np, preprocess_cfg, wavelengths_full
    )
    if X_pp is None:
        empty["reason"] = reason or "preprocessing_failed"
        return empty

    try:
        model = MultiClassClassModel(
            engine=engine,
            alpha=alpha,
            n_components=n_components,
            scaling=scaling,
            min_class_samples=min_class_samples,
            variable_selection=variable_selection,
            n_select=n_select,
        ).fit(X_pp, y_np)
        P, A = model.decision_matrix(X_pp)
        labels = list(model.predict(X_pp))
    except Exception as exc:  # noqa: BLE001 — surface as a reason, never crash the GUI
        empty["reason"] = f"{type(exc).__name__}: {exc}"
        return empty

    unmodelable = getattr(model, "unmodelable_", None)
    if unmodelable is not None and hasattr(unmodelable, "tolist"):
        unmodelable = unmodelable.tolist()
    unmodelable = sorted(unmodelable) if unmodelable else []

    # Wold MPOW/DPOW is a variable-space diagnostic (its own per-class PCA), so
    # it is meaningful regardless of the membership engine. Compute defensively.
    # wold_diagnostic_plot_data's per-class PCA is INT-ONLY: passing the model's
    # n_components verbatim would break it — a variance fraction like the 0.99
    # default int()s to 0 -> max(1,0)=1 PCA component (Wold computed on a 1-D
    # subspace, not the ~99%-variance one the models use), and "per_class_cv"
    # raises. Resolve to a representative int matching the fitted models.
    wold_nc = _resolve_wold_n_components(n_components, model, X_pp)
    wold_error = ""
    try:
        wold = wold_diagnostic_plot_data(
            X_pp, y_np, n_components=wold_nc, scaling=scaling,
            wavelengths=wl_current,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wold diagnostic plot data failed: %s", exc)
        wold = None
        wold_error = f"{type(exc).__name__}: {exc}"

    return {
        "classes": list(model.classes_),
        "p_values": np.asarray(P, dtype=np.float64),
        "accept": np.asarray(A, dtype=bool),
        "labels": labels,
        "true_labels": list(y_np),
        "sample_ids": list(sample_ids),
        "resolved_n_components": dict(getattr(model, "n_components_", {}) or {}),
        "unmodelable_classes": unmodelable,
        "wold": wold,
        "wold_error": wold_error,
        "preprocess_name": preprocess_cfg.get("name", ""),
        "config": config,
        "reason": "",
    }


def _resolve_wold_n_components(n_components, model, X_pp):
    """Resolve the model's ``n_components`` policy to a concrete int for the
    INT-ONLY Wold diagnostic PCA, matching the fitted models as closely as
    possible.

    - Prefer the model's RESOLVED per-class component counts (SIMCA populates
      ``n_components_``); use their max so the diagnostic subspace is at least as
      rich as any class model.
    - Else (non-SIMCA engines, empty ``n_components_``): resolve a variance
      fraction against pooled data via PCA; the ``"per_class_cv"`` sentinel and
      any non-fraction fall back to a bounded default.
    """
    resolved = getattr(model, "n_components_", {}) or {}
    max_pc = min(X_pp.shape[0] - 1, X_pp.shape[1])
    if resolved:
        return int(max(1, min(max(resolved.values()), max_pc)))
    if isinstance(n_components, str):  # "per_class_cv"
        return int(max(1, min(10, max_pc)))
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        from sklearn.decomposition import PCA

        pca = PCA(random_state=0).fit(X_pp)
        cum = np.cumsum(pca.explained_variance_ratio_)
        return int(max(1, min(int(np.searchsorted(cum, n_components)) + 1, max_pc)))
    return int(max(1, min(int(n_components), max_pc)))


def run_multiclass_simca_search(
    X,
    y,
    wavelengths=None,
    engines=None,
    preprocess_configs=None,
    preprocessing_methods=None,
    window_sizes=None,
    alpha=0.05,
    n_components=0.99,
    varsel_paths=None,
    variable_selection_n_select=None,
    min_class_samples=10,
    cv_splits=5,
    variable_penalty=0,
    gap_penalty=0,
    baseline_method=None,
    baseline_params=None,
    enable_smoothing=False,
    smoothing_window=17,
    smoothing_polyorder=2,
    progress_callback=None,
    controller=None,
    compute_top_decision_view=False,
):
    """T-31 multi-class SIMCA / class-modeling search (spec §7).

    Grid = ``preprocessing_configs × engines × varsel_paths`` (NO ``G^K``
    per-class ``n_components`` product — every row shares the single
    ``n_components`` policy, resolved per class INSIDE the row; the default is a
    novelty-oriented variance fraction, see the ``n_components`` param). Each
    row fits a :class:`~spectral_predict.simca.MultiClassClassModel`, computes
    OOF class-modeling metrics via ``cross_validate`` +
    :func:`~spectral_predict.simca.multiclass_simca_metrics`, and a
    leave-one-class-out ``NoveltyAUC`` (:func:`_multiclass_loco_novelty_auc`)
    that drives ranking (higher = better).

    Per-spectrum preprocessing (SNV / SG derivatives / baseline) is applied to
    ``X`` OUTSIDE the folds per chemometrics convention; column-autoscale,
    per-class calibration, and variable selection are fit train-only INSIDE
    ``MultiClassClassModel.fit`` / ``cross_validate`` (never re-fit here).

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Spectra.
    y : array-like of shape (n_samples,)
        Class labels.
    wavelengths : array-like, optional
        Feature axis; inferred from ``X.columns`` when ``X`` is a DataFrame,
        else ``arange(n_features)``.
    engines : list of str, optional
        MultiClassClassModel engine names; defaults to
        :data:`~spectral_predict.model_registry.MULTICLASS_ENGINES`.
    preprocess_configs : list of dict, optional
        Explicit preprocessing configs (each: ``method``/``name``/``deriv``/
        ``window``/``polyorder`` + optional baseline/smoothing). If omitted they
        are built from ``preprocessing_methods`` × ``window_sizes``.
    preprocessing_methods : list of str, optional
        Preprocessing method names (default ``["raw", "snv", "deriv1",
        "deriv2", "snv_deriv1", "snv_deriv2"]``) when ``preprocess_configs`` is
        None.
    window_sizes : list of int, optional
        Savitzky-Golay windows for derivative methods (default ``[7, 19]``).
    alpha : float, default=0.05
        Global significance level (never per-class; spec decision #6).
    n_components : float, int, dict, or "per_class_cv", default=0.99
        Per-class PCA components forwarded to each row's
        :class:`~spectral_predict.simca.MultiClassClassModel`. The default FLOAT
        ``0.99`` is a per-class variance-explained fraction — the
        NOVELTY-oriented choice (T-31 Decision D): on real held-out sites a
        99%-variance model flagged 100% of a foreign site novel at ~8% in-sample
        false-novel, versus only ~17% for ``"per_class_cv"``. ``"per_class_cv"``
        (one-vs-rest balanced-accuracy tuning) optimizes within-known
        DISCRIMINATION, not novelty, and under-detects held-out foreign classes;
        it remains selectable but is not the default.
    varsel_paths : list of str, optional
        Variable-selection paths to enumerate; each maps via
        :data:`_MULTICLASS_VARSEL_PATHS` to a ``variable_selection`` value.
        Default ``["none"]``.
    variable_selection_n_select : int, optional
        ``n_select`` forwarded to the model's variable selection (top-N).
    min_class_samples : int, default=10
        Hard modeling floor (spec §5.1 / §8); classes below it are flagged
        unmodelable, never dropped.
    cv_splits : int, default=5
        Outer CV folds for the OOF decision matrix + own-class p-values.
    variable_penalty, gap_penalty : int, default=0
        Forwarded to :func:`~spectral_predict.scoring.compute_composite_score`
        (gap penalty is a no-op for this task).
    baseline_method, baseline_params : optional
        Baseline correction forwarded to the preprocessing pipeline.
    enable_smoothing, smoothing_window, smoothing_polyorder : optional
        Savitzky-Golay pre-smoothing forwarded to the preprocessing pipeline.
    progress_callback : callable, optional
        Receives ``{stage, message, current, total}`` dicts.
    controller : object, optional
        Optional pause/stop controller with ``check_and_wait()``.

    Returns
    -------
    pd.DataFrame
        Results (schema of ``create_results_dataframe("multiclass_simca")``)
        ranked by :func:`~spectral_predict.scoring.compute_composite_score`.
    """
    import warnings

    from .model_registry import MULTICLASS_ENGINES
    from .scoring import add_result, compute_composite_score, create_results_dataframe
    from .simca import MultiClassClassModel, multiclass_simca_metrics

    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    X_np = np.asarray(X_np, dtype=np.float64)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    if wavelengths is not None:
        wavelengths_full = np.asarray(wavelengths)
    elif hasattr(X, "columns"):
        wavelengths_full = np.asarray(X.columns.values)
    else:
        wavelengths_full = np.arange(X_np.shape[1])

    def _as_list(v):
        if v is None:
            return [None]
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    alphas = _as_list(alpha)
    n_components_list = _as_list(n_components)
    n_select_list = _as_list(variable_selection_n_select)

    if engines is None:
        engines = list(MULTICLASS_ENGINES)
    if varsel_paths is None:
        varsel_paths = ["none"]
    for vp in varsel_paths:
        if vp not in _MULTICLASS_VARSEL_PATHS:
            raise ValueError(
                f"Unknown varsel_path {vp!r}; expected one of "
                f"{sorted(_MULTICLASS_VARSEL_PATHS)}."
            )

    # --- Build preprocessing configs (per-spectrum ops applied outside folds) --
    if preprocess_configs is None:
        if preprocessing_methods is None:
            preprocessing_methods = [
                "raw", "snv", "deriv1", "deriv2", "snv_deriv1", "snv_deriv2",
            ]
        if window_sizes is None:
            window_sizes = [7, 19]
        preprocess_configs = []
        for method in preprocessing_methods:
            if method in ("deriv1", "deriv2", "snv_deriv1", "snv_deriv2"):
                deriv_order = 1 if method.endswith("1") else 2
                pipeline_method = method.replace("1", "").replace("2", "")
                for ws in window_sizes:
                    preprocess_configs.append(
                        {
                            "method": pipeline_method,
                            "name": f"{method}_w{ws}",
                            "deriv": deriv_order,
                            "window": ws,
                            "polyorder": None,
                            "baseline_method": baseline_method,
                            "baseline_params": baseline_params,
                            "smoothing": enable_smoothing,
                            "smoothing_window": smoothing_window,
                            "smoothing_polyorder": smoothing_polyorder,
                        }
                    )
            else:
                preprocess_configs.append(
                    {
                        "method": method,
                        "name": method,
                        "deriv": None,
                        "window": None,
                        "polyorder": None,
                        "baseline_method": baseline_method,
                        "baseline_params": baseline_params,
                        "smoothing": enable_smoothing,
                        "smoothing_window": smoothing_window,
                        "smoothing_polyorder": smoothing_polyorder,
                    }
                )

    total_configs = (
        len(preprocess_configs)
        * len(engines)
        * len(varsel_paths)
        * len(alphas)
        * len(n_components_list)
        * len(n_select_list)
    )
    current_config = 0
    n_total_classes = int(len(np.unique(y_np)))

    logger.info("=" * 70)
    logger.info("MULTI-CLASS SIMCA SEARCH")
    logger.info("=" * 70)
    logger.info("Classes: %d | Engines: %s", n_total_classes, engines)
    logger.info("Preprocessing configs: %d | Varsel paths: %s", len(preprocess_configs), varsel_paths)
    logger.info("Total configurations: %d (alpha=%s)", total_configs, alpha)

    if progress_callback:
        progress_callback(
            {
                "stage": "info",
                "message": f"Starting multi-class SIMCA search: {total_configs} configurations",
                "current": 0,
                "total": total_configs,
            }
        )

    df_results = create_results_dataframe("multiclass_simca")
    _user_stopped = False

    for preprocess_cfg in preprocess_configs:
        if _user_stopped:
            break
        # Per-spectrum preprocessing on ALL rows (chemometrics convention: NOT
        # leakage). Column-autoscale/calibration/varsel stay train-only inside
        # the model. Build the pipeline and fit/transform the full matrix.
        # Build + fit/transform + edge-mask are ALL inside one guard (Codex M2):
        # a single malformed config (e.g. build_preprocessing_pipeline raising on
        # a bad window/polyorder) must NOT abort the whole search — it emits NaN
        # rows with a reason instead.
        X_pp, wavelengths_current, pp_reason = _multiclass_preprocess_matrix(
            X_np, preprocess_cfg, wavelengths_full
        )
        if X_pp is None:
            logger.warning(
                "Preprocessing '%s' failed: %s — emitting NaN rows for its configs",
                preprocess_cfg["name"], pp_reason,
            )

        for engine in engines:
            if _user_stopped:
                break
            for varsel_path in varsel_paths:
                if controller and not controller.check_and_wait():
                    _user_stopped = True
                    break

                for _alpha in alphas:
                    for _ncomp in n_components_list:
                        for _n_select in n_select_list:
                            current_config += 1
                            prep_name = preprocess_cfg["name"]
                            progress_msg = f"Testing {engine} + {prep_name} + varsel={varsel_path}"
                            logger.info("[%d/%d] %s", current_config, total_configs, progress_msg)
                            if progress_callback:
                                progress_callback(
                                    {
                                        "stage": "model_testing",
                                        "message": progress_msg,
                                        "current": current_config,
                                        "total": total_configs,
                                    }
                                )

                            varsel_value = _MULTICLASS_VARSEL_PATHS[varsel_path]

                            def _build():
                                return MultiClassClassModel(
                                    engine=engine,
                                    alpha=_alpha,
                                    n_components=_ncomp,
                                    scaling="per_class",
                                    min_class_samples=min_class_samples,
                                    variable_selection=varsel_value,
                                    n_select=_n_select,
                                )

                            # Base row (common cols + tags); metrics start NaN and are
                            # filled on success. Any failure records the row with NaN
                            # metrics + a reason (never crashes the whole search; spec §8).
                            full_vars = int(X_pp.shape[1]) if X_pp is not None else int(len(wavelengths_current))
                            row = {
                                "Task": "multiclass_simca",
                                "Model": engine,
                                "Params": f"alpha={_alpha}, scaling=per_class",
                                "Preprocess": prep_name,
                                "Deriv": preprocess_cfg.get("deriv"),
                                "Window": preprocess_cfg.get("window"),
                                "Poly": preprocess_cfg.get("polyorder"),
                                "LVs": "auto",
                                "n_vars": full_vars,
                                "full_vars": full_vars,
                                "SubsetTag": varsel_path,
                                "Imbalance": "—",
                                "NoveltyAUC": np.nan,
                                "Efficiency": np.nan,
                                "NoveltyRate": np.nan,
                                "NoClassRate": np.nan,
                                "AmbiguityRate": np.nan,
                                "ExactSetRate": np.nan,
                                "MeanSensitivity": np.nan,
                                "MeanSpecificity": np.nan,
                                "Alpha": _alpha,
                                "NComponents": _ncomp,
                                "MinClassN": np.nan,
                                "n_classes": n_total_classes,
                                "engine_family": engine,
                                "varsel_path": varsel_path,
                                "unmodelable_classes": "",
                                "reason": "",
                                "top_vars": "N/A",
                                "all_vars": "",
                            }

                            if X_pp is None:
                                row["reason"] = pp_reason or "preprocessing_failed"
                                df_results = add_result(df_results, row)
                                continue

                            try:
                                # Full fit -> per-class n_components, varsel mask, modeled set.
                                full_model = _build().fit(X_pp, y_np)
                                modeled = list(full_model.models_.keys())
                                if modeled:
                                    row["MinClassN"] = int(
                                        min(int(np.sum(y_np == c)) for c in modeled)
                                    )
                                if full_model.n_components_:
                                    row["LVs"] = str(full_model.n_components_)
                                if getattr(full_model, "varsel_mask_", None) is not None:
                                    row["n_vars"] = int(full_model.varsel_mask_.sum())
                                if full_model.unmodelable_:
                                    row["unmodelable_classes"] = str(
                                        sorted(full_model.unmodelable_.tolist()
                                               if hasattr(full_model.unmodelable_, "tolist")
                                               else full_model.unmodelable_)
                                    )

                                # OOF class-modeling metrics.
                                cv = _build().cross_validate(X_pp, y_np, n_splits=cv_splits)
                                _, A_oof = cv["decision_matrix"]
                                metrics = multiclass_simca_metrics(y_np, A_oof, list(cv["classes"]))
                                sens = np.asarray(
                                    list(metrics["per_class_sensitivity"].values()), dtype=np.float64
                                )
                                spec = np.asarray(
                                    list(metrics["per_class_specificity"].values()), dtype=np.float64
                                )
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", RuntimeWarning)
                                    mean_sens = (
                                        float(np.nanmean(sens))
                                        if sens.size and not np.all(np.isnan(sens))
                                        else np.nan
                                    )
                                    mean_spec = (
                                        float(np.nanmean(spec))
                                        if spec.size and not np.all(np.isnan(spec))
                                        else np.nan
                                    )
                                row["Efficiency"] = metrics["efficiency"]
                                row["NoveltyRate"] = metrics["novelty_detection_rate"]
                                row["NoClassRate"] = metrics["no_class_rate"]
                                row["AmbiguityRate"] = metrics["ambiguity_rate"]
                                row["ExactSetRate"] = metrics["exact_set_rate"]
                                row["MeanSensitivity"] = mean_sens
                                row["MeanSpecificity"] = mean_spec

                                # LOCO NoveltyAUC (the single-objective ranking metric, spec
                                # §7). Decision A: this is a within-dataset proxy that
                                # OVER-estimates true held-out-foreign novelty (see
                                # _multiclass_loco_novelty_auc docstring). Decision C: ranking
                                # stays single-objective on NoveltyAUC; a quality-vs-
                                # discrimination blend (e.g. NoveltyAUC·Efficiency**0.5)
                                # remains an unimplemented alternative the user is undecided
                                # on.
                                row["NoveltyAUC"] = _multiclass_loco_novelty_auc(
                                    _build, X_pp, y_np, cv_splits=cv_splits, oof_cv=cv
                                )
                            except Exception as exc:  # noqa: BLE001 — NaN-safe per-config guard
                                logger.warning(
                                    "Multi-class config %s + %s (varsel=%s) failed: %s",
                                    engine, prep_name, varsel_path, exc,
                                )
                                row["reason"] = f"{type(exc).__name__}: {exc}"

                            df_results = add_result(df_results, row)

    if len(df_results) > 0:
        # No-signal guard (Kimi H1/H2): if NOT ONE config produced a finite
        # NoveltyAUC the leaderboard has no ranking signal (every row ties on
        # NaN); warn loudly rather than return a clean-looking Rank=1 frame.
        if df_results["NoveltyAUC"].isna().all():
            logger.warning(
                "Multi-class SIMCA search: NO configuration produced a finite "
                "NoveltyAUC (every config failed or was unmodelable); the "
                "leaderboard carries no ranking signal."
            )
        df_results = compute_composite_score(
            df_results, "multiclass_simca", variable_penalty, gap_penalty
        )

    # Optionally build the per-sample decision view for the top-ranked config
    # (Phase D2). Attached via df.attrs so the return type stays a DataFrame for
    # every existing caller. The winning preprocess_cfg is looked up by name from
    # the authoritative in-scope list (no fragile name parsing).
    if compute_top_decision_view and len(df_results) > 0:
        try:
            top = df_results.iloc[0]
            top_cfg = next(
                (c for c in preprocess_configs if c["name"] == top["Preprocess"]),
                None,
            )
            if top_cfg is not None and not top.get("reason"):
                df_results.attrs["top_decision_view"] = build_multiclass_decision_view(
                    X_np,
                    y_np,
                    engine=str(top["engine_family"]),
                    preprocess_cfg=top_cfg,
                    alpha=top["Alpha"],
                    n_components=top["NComponents"],
                    scaling="per_class",
                    min_class_samples=min_class_samples,
                    variable_selection=_MULTICLASS_VARSEL_PATHS[str(top["varsel_path"])],
                    n_select=variable_selection_n_select,
                    wavelengths=wavelengths_full,
                    sample_ids=(list(X.index) if hasattr(X, "index") else None),
                )
        except Exception as exc:  # noqa: BLE001 — the leaderboard still returns
            logger.warning("Top decision-view build failed: %s", exc)

    logger.info("=" * 70)
    logger.info("MULTI-CLASS SIMCA SEARCH COMPLETE")
    logger.info("Total configurations: %d", len(df_results))
    logger.info("=" * 70)
    if progress_callback:
        progress_callback(
            {
                "stage": "info",
                "message": f"Multi-class SIMCA search complete: {len(df_results)} results",
                "current": total_configs,
                "total": total_configs,
            }
        )

    return df_results
