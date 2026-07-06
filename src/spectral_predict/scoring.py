"""Scoring and ranking functions."""

import ast
import logging
from bisect import bisect_left

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def compute_composite_score(df_results, task_type, variable_penalty=0, gap_penalty=0,
                            use_rmsep_gap=False, verbose=False):
    """
    Compute composite score with user-friendly penalty system.

    Penalties (0-10 scale):
    - variable_penalty: Penalty for using many variables
    - gap_penalty: Penalty for calibration-CV gap (overfitting indicator)
    - 0 = only performance matters (DEFAULT)
    - 10 = strong penalty

    Both penalties scale by the actual performance range of the result set,
    so same spinbox value = same relative impact regardless of dataset.

    Formula: Score = performance_score + var_penalty_term + gap_penalty_term

    Performance score (lower is better):
    - Regression: -R2cv
    - Classification: -Accuracycv - 0.0001 * F1cv

    Parameters
    ----------
    df_results : pd.DataFrame
        Results dataframe with metrics and complexity measures
    task_type : str
        'regression' or 'classification'
    variable_penalty : int (0-10)
        Penalty for using many variables (default: 0)
        Uses quadratic scaling for gentle impact at low values
    gap_penalty : int (0-10)
        Penalty for calibration-CV gap (default: 0)
        Measures RMSEcv/RMSE ratio (regression) or Accuracy/Accuracycv (classification)
        Higher ratio = more overfitting = bigger penalty
    use_rmsep_gap : bool
        If True and RMSEP/val_Accuracy column exists, use validation metrics
        for gap calculation instead of calibration metrics
    verbose : bool, default=False
        If True, print detailed scoring breakdown for top 20 models

    Returns
    -------
    df_scored : pd.DataFrame
        Results with CompositeScore and Rank columns added
    """
    df = df_results.copy()

    # Performance score (lower is better) — always use direct metric ranking
    if task_type == "regression":
        performance_score = -df["R2cv"]
    elif task_type == "one_class":
        bal_acc_cv = df["BalancedAcccv"].fillna(df.get("Specificitycv", 0))
        performance_score = -bal_acc_cv - 0.0001 * df["Sensitivitycv"].fillna(0)
    elif task_type == "multiclass_simca":
        # Rank by the alpha-sweep NoveltyAUC (higher = better -> lower score).
        # NaN AUC is preserved here (not fillna'd) so the lexicographic Rank
        # override below can force it LAST; the MinClassN tie-break is applied
        # there too (NOT via a fragile additive 1e-9 term — Codex H2/M1, Kimi M5).
        performance_score = -df["NoveltyAUC"].astype(np.float64)
    else:  # classification
        performance_score = -df["Accuracycv"] - 0.0001 * df["F1cv"]

    # Compute performance range for penalty scaling
    # Both penalties scale relative to this so same spinbox value = same max impact
    if task_type == "regression":
        perf_range = df["R2cv"].max() - df["R2cv"].min()
    elif task_type == "one_class":
        bal_acc_cv_range = df["BalancedAcccv"].fillna(df.get("Specificitycv", 0))
        perf_range = bal_acc_cv_range.max() - bal_acc_cv_range.min()
    elif task_type == "multiclass_simca":
        auc = df["NoveltyAUC"].astype(np.float64)
        perf_range = auc.max() - auc.min()
    else:
        perf_range = df["Accuracycv"].max() - df["Accuracycv"].min()

    # 1. Variable Count Penalty (0-10 scale)
    if variable_penalty > 0 and perf_range >= 0.001:
        n_vars_array = np.asarray(df["n_vars"], dtype=np.float64)
        full_vars_array = np.asarray(df["full_vars"], dtype=np.float64)
        var_fraction = n_vars_array / full_vars_array  # 0-1 scale

        penalty_scale = (variable_penalty / 10.0) ** 2  # quadratic 0-1
        var_penalty_term = penalty_scale * var_fraction * perf_range * 0.5
    else:
        var_penalty_term = 0

    # 2. Calibration-CV Gap Penalty (0-10 scale)
    # Uses actual measured metrics instead of hardcoded model family scores
    if gap_penalty > 0 and perf_range >= 0.001:
        penalty_scale = (gap_penalty / 10.0) ** 2  # quadratic 0-1

        if task_type == "regression":
            rmse = df["RMSE"].astype(np.float64)
            rmsecv = df["RMSEcv"].astype(np.float64)

            # RMSEP/RMSEcv mode (post-search re-ranking with validation data)
            if use_rmsep_gap and "RMSEP" in df.columns:
                rmsep = df["RMSEP"].astype(np.float64)
                has_rmsep = rmsep.notna() & (rmsecv > 1e-10)
                gap_ratio = np.where(
                    has_rmsep,
                    rmsep / rmsecv,
                    np.where(rmse > 1e-10, rmsecv / rmse, 1.0)
                )
            else:
                gap_ratio = np.where(
                    (rmse > 1e-10) & (rmsecv > 1e-10),
                    rmsecv / rmse,
                    1.0
                )

            # Normalize: ratio 1.0 = no gap (0 penalty), ratio 5.0 = max penalty (1.0)
            gap_fraction = np.clip((gap_ratio - 1.0) / 4.0, 0.0, 1.0)

        elif task_type == "one_class":
            # One-class: use balanced accuracy for gap calculation
            # Guard NaN: fall back to Specificity columns when BalancedAcc is NaN
            bal_acc = df["BalancedAcc"].fillna(df.get("Specificity", np.nan)).astype(np.float64)
            bal_acc_cv = df["BalancedAcccv"].fillna(df.get("Specificitycv", np.nan)).astype(np.float64)
            both_nan = bal_acc.isna() & bal_acc_cv.isna()
            bal_acc = bal_acc.fillna(0.0)
            bal_acc_cv = bal_acc_cv.fillna(0.0)
            gap_ratio = np.where(bal_acc_cv > 1e-10, bal_acc / bal_acc_cv, 1.0)
            gap_fraction = np.where(both_nan, 1.0, np.clip((gap_ratio - 1.0) / 0.2, 0.0, 1.0))

        elif task_type == "multiclass_simca":
            # No calibration-vs-CV gap concept for class modeling (metrics come
            # from a single OOF decision matrix); the gap penalty is a no-op.
            gap_fraction = 0.0

        else:  # classification
            acc = df["Accuracy"].astype(np.float64)
            acc_cv_col = "Accuracycv" if "Accuracycv" in df.columns else "AccuracyCV"
            acc_cv = df[acc_cv_col].astype(np.float64)

            # Validation mode
            if use_rmsep_gap and "val_Accuracy" in df.columns:
                val_acc = df["val_Accuracy"].astype(np.float64)
                has_val = val_acc.notna() & (acc_cv > 1e-10)
                gap_ratio = np.where(
                    has_val,
                    acc / val_acc,
                    np.where(acc_cv > 1e-10, acc / acc_cv, 1.0)
                )
            else:
                gap_ratio = np.where(acc_cv > 1e-10, acc / acc_cv, 1.0)

            # Accuracy gaps are tighter — normalize over 0.2 range
            # (Accuracy 1.0 vs Accuracycv 0.80 = ratio 1.25, maps to full penalty)
            gap_fraction = np.clip((gap_ratio - 1.0) / 0.2, 0.0, 1.0)

        gap_penalty_term = penalty_scale * gap_fraction * perf_range * 0.5
    else:
        gap_penalty_term = 0

    # Store individual components for diagnostics
    df["PerformanceScore"] = performance_score
    df["VarPenalty"] = var_penalty_term if not np.isscalar(var_penalty_term) else np.zeros(len(df))
    df["GapPenalty"] = gap_penalty_term if not np.isscalar(gap_penalty_term) else np.zeros(len(df))

    # Composite score (lower is better)
    df["CompositeScore"] = performance_score + var_penalty_term + gap_penalty_term

    if task_type == "multiclass_simca":
        # Lexicographic ranking (Codex H2/M1 + Kimi M5): NaN NoveltyAUC always
        # LAST; among finite rows lower CompositeScore (= higher penalty-adjusted
        # NoveltyAUC) wins; EXACT ties broken by the larger smallest-class n.
        # Encoded as a (CompositeScore, -MinClassN) sort key so a huge MinClassN
        # can never outweigh a real AUC gap, and a sub-1e-9 gap never flips.
        auc = df["NoveltyAUC"].astype(np.float64)
        nan_mask = auc.isna()
        df.loc[nan_mask, "CompositeScore"] = np.inf  # NaN rows sort last
        min_class_n = (
            df.get("MinClassN", pd.Series(0.0, index=df.index))
            .astype(np.float64)
            .fillna(0.0)
        )
        keys = list(zip(df["CompositeScore"].to_numpy(), (-min_class_n).to_numpy()))
        sorted_keys = sorted(keys)
        ranks = [bisect_left(sorted_keys, k) + 1 for k in keys]  # method="min"
        df["Rank"] = np.asarray(ranks, dtype=int)
    else:
        # Rank (1 = best)
        df["Rank"] = df["CompositeScore"].rank(method="min").astype(int)

    # Sort by rank and reset index to ensure sequential IDs for GUI display
    df = df.sort_values("Rank").reset_index(drop=True)

    # Reorder columns: Rank first, top_vars last
    cols = [c for c in df.columns if c not in ['Rank', 'top_vars']]
    new_col_order = ['Rank'] + cols + ['top_vars']
    df = df[new_col_order]

    # Add unified complexity score (0-100 scale, higher = more complex)
    # Informational column, doesn't affect ranking
    try:
        df["ComplexityScore"] = df.apply(_compute_unified_complexity, axis=1)
    except Exception as e:
        print(f"Warning: Unified complexity calculation failed: {e}")
        df["ComplexityScore"] = np.nan

    # Verbose diagnostic output
    if verbose:
        print("\n" + "="*80)
        print("RANKING DIAGNOSTIC - Top 20 Models")
        print("="*80)
        print(f"Penalty Settings: Variables={variable_penalty}/10, Gap={gap_penalty}/10")
        print(f"Task Type: {task_type}, RMSEP mode: {use_rmsep_gap}")
        print("\nScore Components (lower CompositeScore = better rank):")
        print("  - PerformanceScore: Based on -R2cv (or -Accuracycv)")
        print("  - VarPenalty: Penalty for using many wavelengths")
        print("  - GapPenalty: Penalty for calibration-CV gap (overfitting)")
        print("-"*80)

        top20 = df.head(20).copy()

        if task_type == "regression":
            display_cols = ["Rank", "Model", "R2", "RMSE", "R2cv", "RMSEcv", "n_vars",
                           "PerformanceScore", "VarPenalty", "GapPenalty", "CompositeScore"]
        elif task_type == "multiclass_simca":
            display_cols = ["Rank", "Model", "NoveltyAUC", "Efficiency", "MinClassN",
                           "n_vars", "PerformanceScore", "VarPenalty", "GapPenalty",
                           "CompositeScore"]
        else:
            display_cols = ["Rank", "Model", "Accuracy", "Accuracycv", "n_vars",
                           "PerformanceScore", "VarPenalty", "GapPenalty", "CompositeScore"]

        display_cols = [c for c in display_cols if c in top20.columns]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)

        print(top20[display_cols].to_string(index=False))
        print("="*80 + "\n")

    return df


def _compute_unified_complexity(row):
    """
    Compute unified complexity score (0-100 scale, higher = more complex).

    Formula: ComplexityScore = 0.25*Model + 0.30*Variables + 0.25*LVs + 0.20*Preprocessing

    Components:
    - Model Type (25%): Intrinsic model complexity
    - Variables (30%): Number of wavelengths selected (nonlinear penalty)
    - Latent Variables (25%): For PLS models, number of components
    - Preprocessing (20%): Derivative order and SNV

    Returns
    -------
    score : float
        Complexity score in range [0, 100]
    """
    # 1. Model Type Complexity (25% weight) - based on model complexity
    model = row.get("Model", "")
    model_scores = {
        "PLS": 20,
        "PCA-SIMCA": 20,
        "Ridge": 25,
        "EllipticEnvelope": 30,
        "Lasso": 30,
        "IsolationForest": 35,
        "LOF": 45,
        "OneClassSVM": 55,
        "RandomForest": 60,
        "MLP": 80,
        "NeuralBoosted": 85,
    }
    model_complexity = model_scores.get(model, 50)  # Default to 50 if unknown

    # 2. Variable Complexity (30% weight) - nonlinear penalty for many variables
    # Use sqrt-based nonlinear penalty: few vars = low penalty, many vars = high penalty
    n_vars = row.get("n_vars", 0)
    # Normalize: 10 vars ~ 2.0, 100 vars ~ 20, 500 vars ~ 100
    var_complexity = min(100, np.sqrt(n_vars) * 4.5)

    # 3. Latent Variable Complexity (25% weight) - for PLS models
    lvs = row.get("LVs", np.nan)
    # Multi-class SIMCA writes a non-numeric LVs ("auto" or a per-class dict
    # string like "{'A': 7, 'B': 9}"), which would blow up (lvs - 2) below with
    # a str - int TypeError. Coerce anything non-numeric to NaN so it falls to
    # the median-complexity default rather than aborting the whole column.
    if not isinstance(lvs, (int, float)):
        try:
            lvs = float(lvs)
        except (TypeError, ValueError):
            lvs = np.nan
    # PCA-SIMCA stores dimensionality as n_components in Params; LVs may be 0 or missing
    if (pd.isna(lvs) or lvs == 0) and model == "PCA-SIMCA":
        try:
            params_raw = row.get("Params", "{}")
            # Params can be a dict (in-memory result rows) or a str
            # (CSV-loaded rows). Handling only the str branch caused
            # PCA-SIMCA n_components to be lost for in-memory results,
            # which then collapsed lv_complexity to the median fallback.
            if isinstance(params_raw, dict):
                params_dict = params_raw
            elif isinstance(params_raw, str):
                params_dict = ast.literal_eval(params_raw) if params_raw.strip() else {}
            else:
                params_dict = {}
            lvs = params_dict.get("n_components", np.nan)
        except (ValueError, SyntaxError):
            lvs = np.nan
    if pd.isna(lvs) or lvs == 0:
        # Non-PLS models: use median complexity (50)
        lv_complexity = 50
    else:
        # Normalize LVs: 2 LVs = 0, 25 LVs = 100
        lv_complexity = min(100, (lvs - 2) * 100 / 23)

    # 4. Preprocessing Complexity (20% weight)
    preprocess = row.get("Preprocess", "raw")
    deriv = row.get("Deriv", 0)

    # Base preprocessing scores
    if preprocess == "raw":
        prep_base = 0
    elif preprocess == "snv":
        prep_base = 20
    elif preprocess == "deriv":
        if deriv == 1:
            prep_base = 50
        elif deriv == 2:
            prep_base = 70
        else:
            prep_base = 40  # Unknown derivative order
    elif preprocess == "deriv_snv":
        if deriv == 1:
            prep_base = 60
        elif deriv == 2:
            prep_base = 80
        else:
            prep_base = 50
    else:
        prep_base = 30  # Unknown preprocessing

    prep_complexity = min(100, prep_base)

    # Weighted sum (0-100 scale)
    complexity_score = (
        0.25 * model_complexity +
        0.30 * var_complexity +
        0.25 * lv_complexity +
        0.20 * prep_complexity
    )

    return round(complexity_score, 1)


def compute_specificity(y_true, y_pred, average='macro'):
    """
    Compute specificity (True Negative Rate) for classification.

    For binary classification: TN / (TN + FP)
    For multi-class: macro-averaged specificity across all classes

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str
        Averaging method ('macro' for equal weight per class)

    Returns
    -------
    specificity : float
        Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)

    # For each class, compute TN / (TN + FP)
    # TN for class i = sum of all cells except row i and column i
    # FP for class i = sum of column i except diagonal
    n_classes = cm.shape[0]

    if n_classes == 2:
        # Binary classification: simple formula
        tn = cm[0, 0]
        fp = cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Multi-class: compute per-class specificity then average
    specificities = []
    for i in range(n_classes):
        # True negatives: all correct predictions for other classes
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        # False positives: predicted as class i but actually other classes
        fp = np.sum(cm[:, i]) - cm[i, i]

        if (tn + fp) > 0:
            specificities.append(tn / (tn + fp))
        else:
            specificities.append(0.0)

    if average == 'macro':
        return np.mean(specificities)
    else:
        return specificities


def lins_ccc(y_true, y_pred) -> float:
    """Compute Lin's Concordance Correlation Coefficient.

    CCC measures agreement between paired observations along the 1:1 line.
    Unlike Pearson r (which is invariant to scale and bias) or R2 (which
    can be inflated under bias), CCC penalizes BOTH correlation departures
    AND systematic shift / scale-change of the predictions away from the
    identity line. Range: [-1, 1].

    Formula (Lin 1989):
        CCC = 2 * rho * sigma_x * sigma_y / (sigma_x^2 + sigma_y^2 + (mu_x - mu_y)^2)

    where rho is the Pearson correlation between x = y_true and y = y_pred.

    When one or both inputs have zero variance, this implementation returns
    0.0 (constant-vs-varying) or 1.0 (both constant equal). This is a
    DISPLAY CONVENTION for ranking; Lin (1989) defines CCC as undefined
    when variance is zero.

    Parameters
    ----------
    y_true : array-like
        Observed reference values.
    y_pred : array-like
        Predicted values, same length as y_true.

    Returns
    -------
    ccc : float
        Concordance correlation coefficient in [-1, 1].
        Returns 0.0 when either input has zero variance and the two
        arrays are not identical (degenerate-but-defined convention).
        Returns 1.0 when both inputs are equal constants.
        Returns NaN if either input contains NaN.

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.

    References
    ----------
    Lin, L. I. (1989). A concordance correlation coefficient to evaluate
    reproducibility. Biometrics, 45(1), 255-268.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return float("nan")

    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    if denominator == 0.0:
        return 1.0
    if var_true == 0.0 or var_pred == 0.0:
        return 0.0

    return float(2.0 * cov / denominator)


def compute_cv_anova_pvalue(
    y_true,
    rmsecv: float,
    n_components: int,
) -> float:
    """CV-ANOVA F-test p-value (Eriksson, Trygg & Wold 2008).

    Returns p-value for the null hypothesis that the PLS regression
    model's cross-validated PRESS is no better than mean-prediction PRESS.
    Only defined for single-Y PLS regression with pooled cross-validated
    predictions.

    For repeated K-fold CV, dasp reduces repeated predictions to a single
    per-sample average before computing RMSEcv, so PRESS = N * RMSEcv**2
    refers to the averaged-prediction vector. The p-value here therefore
    tests whether the averaged-CV prediction beats mean prediction — a
    sensible extension of Eriksson 2008 for repeated CV, not a literal
    application.

    Parameters
    ----------
    y_true : array-like
        Training target vector (1D, single Y).
    rmsecv : float
        Root mean squared error of cross-validation (pooled).
    n_components : int
        Number of PLS latent variables (A).

    Returns
    -------
    float
        p-value in [0, 1]. Returns 1.0 when PRESS >= SSY (model no better
        than mean — F-statistic <= 0). Returns nan on degenerate input:
        n_components < 1; rmsecv non-finite or <= 0; y_true not 1-D, fewer
        than 2 samples, or contains non-finite values; n_components >= N-1
        (over-parametrised); SSY <= 0 (zero-variance y).

    References
    ----------
    Eriksson, L., Trygg, J., & Wold, S. (2008). CV-ANOVA for significance
    testing of PLS and OPLS models. Journal of Chemometrics, 22(11-12),
    594-600.
    """
    if n_components is None or n_components < 1:
        return float("nan")
    if rmsecv is None or not np.isfinite(rmsecv):
        return float("nan")
    if rmsecv <= 0:
        # RMSEcv == 0 can be reached by perfectly predicted synthetic / demo
        # data; logging at debug avoids spurious warnings in test contexts.
        logger.debug(
            "compute_cv_anova_pvalue: rmsecv=%r is not strictly positive "
            "(perfectly-predicted CV or upstream sentinel). Returning nan.",
            rmsecv,
        )
        return float("nan")

    y = np.asarray(y_true)
    if y.ndim != 1:
        # dasp's PLS regression is single-Y today; multi-output here means
        # a dispatch bug somewhere upstream.
        logger.error(
            "compute_cv_anova_pvalue: y_true has shape=%s, expected 1-D. "
            "dasp's PLS regression is single-Y; this indicates a dispatch "
            "bug. Returning nan.",
            y.shape,
        )
        return float("nan")
    if y.size < 2:
        return float("nan")
    if not np.isfinite(y).all():
        # Upstream NaN-stripping uses pandas.isna which does NOT flag inf;
        # if non-finite values reach here, log so the upstream gap is visible.
        logger.warning(
            "compute_cv_anova_pvalue: y_true contains %d nan / %d inf. "
            "Upstream filtering may not catch inf. Returning nan.",
            int(np.isnan(y).sum()), int(np.isinf(y).sum()),
        )
        return float("nan")

    n = int(y.size)
    a = int(n_components)
    df2 = n - a - 1
    if df2 <= 0:
        return float("nan")

    ssy = float(np.sum((y - y.mean()) ** 2))
    if ssy <= 0:
        return float("nan")

    press = n * float(rmsecv) ** 2
    numerator = (ssy - press) / a
    denominator = press / df2
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")

    f_stat = numerator / denominator
    if f_stat <= 0:
        return 1.0  # PRESS >= SSY: model no better than mean.

    return float(f_dist.sf(f_stat, a, df2))


def create_results_dataframe(task_type):
    """
    Create an empty results dataframe with correct columns.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    df : pd.DataFrame
        Empty dataframe with appropriate columns
    """
    common_cols = [
        "Task",
        "Model",
        "Params",
        "Preprocess",
        "Deriv",
        "Window",
        "Poly",
        "LVs",
        "n_vars",
        "full_vars",
        "SubsetTag",
        "Imbalance",
    ]

    if task_type == "regression":
        # Calibration metrics first, then CV metrics, then NIR-specific metrics
        metric_cols = [
            "RMSE", "R2", "RMSEcv", "R2cv", "cv_anova_pvalue",
            "MAEcv", "RPD", "Bias", "RER", "CCC", "CCCcv",
        ]
    elif task_type == "one_class":
        # One-class detection screening metrics
        metric_cols = [
            # Calibration metrics
            "Sensitivity", "Specificity", "Precision", "F1",
            "Accuracy", "BalancedAcc", "AUC",
            # Cross-validation metrics
            "Sensitivitycv", "Specificitycv", "Precisioncv", "F1cv",
            "Accuracycv", "BalancedAcccv", "AUCcv",
        ]
    elif task_type == "multiclass_simca":
        # T-31 multi-class class-modeling metrics (NOT single-label — a sample
        # may be accepted by 0 / 1 / >=2 classes). Ranked by NoveltyAUC (the
        # alpha-sweep AUC of novelty-vs-false-rejection, spec §7). engine_family
        # + varsel_path are per-row tags (spec decision #3 / #5).
        metric_cols = [
            "NoveltyAUC", "Efficiency", "NoveltyRate", "NoClassRate",
            "AmbiguityRate", "ExactSetRate", "MeanSensitivity", "MeanSpecificity",
            "Alpha", "NComponents", "NSelect", "MinClassN", "n_classes",
            "engine_family", "varsel_path",
            # Emitted by run_multiclass_simca_search; declared so downstream
            # consumers stay in sync (Kimi M3).
            "unmodelable_classes", "reason",
        ]
    else:
        # Calibration metrics first, then CV metrics, then advanced metrics
        metric_cols = [
            # Calibration metrics
            "Accuracy", "ROC_AUC", "F1", "Precision", "Recall",
            "Specificity", "Kappa", "MCC", "BalancedAcc", "BER", "LogLoss",
            # Cross-validation metrics
            "Accuracycv", "ROC_AUCcv", "F1cv", "Precisioncv", "Recallcv",
            "Specificitycv", "Kappacv", "MCCcv", "BalancedAcccv", "BERcv", "LogLosscv"
        ]

    all_cols = common_cols + metric_cols + ["top_vars", "all_vars", "CompositeScore", "Rank"]

    return pd.DataFrame(columns=all_cols)


def add_result(df_results, result_dict):
    """
    Add a single result to the results dataframe.

    Parameters
    ----------
    df_results : pd.DataFrame
        Existing results dataframe
    result_dict : dict
        Dictionary with result information

    Returns
    -------
    df_results : pd.DataFrame
        Updated results dataframe
    """
    # Convert to DataFrame and append
    df_new = pd.DataFrame([result_dict])
    return pd.concat([df_results, df_new], ignore_index=True)


# ============================================================================
# IMBALANCE-AWARE METRICS
# ============================================================================

def compute_imbalance_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute metrics appropriate for imbalanced classification.

    These metrics provide better insight into model performance on imbalanced
    datasets compared to standard accuracy.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities (for ROC AUC)

    Returns
    -------
    metrics : dict
        Dictionary with:
        - 'balanced_accuracy': Macro-averaged recall (equal weight per class)
        - 'f1_weighted': F1 score weighted by class frequency
        - 'f1_macro': F1 score macro-averaged (equal weight per class)
        - 'precision_weighted': Weighted precision
        - 'recall_weighted': Weighted recall
        - 'roc_auc': ROC AUC score with macro averaging (equal weight per class, if y_pred_proba provided)

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier()
    >>> rf.fit(X_train, y_train)
    >>> y_pred = rf.predict(X_test)
    >>> y_proba = rf.predict_proba(X_test)
    >>> metrics = compute_imbalance_metrics(y_test, y_pred, y_proba)
    >>> print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score
    )

    metrics = {}

    # T-29 fix-of-fixes (DeepSeek HIGH-1): pre-compute diagnostic context once
    # so all warning messages can include n / n_classes without recomputing.
    # Also used for the balanced_accuracy wrap below — closes the asymmetry
    # where balanced_accuracy was the only metric outside a try/except.
    _y_true_arr = np.asarray(y_true)
    _diag_n = len(_y_true_arr)
    _diag_n_classes = len(np.unique(_y_true_arr)) if _diag_n else 0

    # Balanced accuracy (macro-averaged recall)
    # Adjusts for class imbalance - treats all classes equally
    # T-29 fix-of-fixes (DeepSeek HIGH-1): wrapped to match the other metrics —
    # was the only call outside a try/except, so a ValueError here crashed
    # the whole compute_imbalance_metrics call instead of returning a partial
    # dict like every other metric does.
    try:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    except Exception as exc:
        logger.warning(
            "T-29: balanced_accuracy failed (n=%d, n_classes=%d, exc=%s); "
            "using 0.0 sentinel", _diag_n, _diag_n_classes, exc,
        )
        metrics['balanced_accuracy'] = 0.0

    # F1 scores
    # Weighted: accounts for class frequency in the dataset
    # Macro: treats all classes equally regardless of frequency
    try:
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception as exc:
        # T-29: was a bare `except:` that swallowed KeyboardInterrupt and
        # SystemExit too — Ctrl-C during long searches hit this and got
        # eaten. `except Exception:` lets system-exits propagate. The
        # warning surfaces silent metric failure so a leaderboard 0.0 can
        # be distinguished from a real model-scored-badly 0.0. Diagnostic
        # context (n, n_classes) added in the fix-of-fixes pass per cross-
        # family review feedback (DeepSeek MEDIUM / GLM LOW).
        logger.warning(
            "T-29: f1 score failed (n=%d, n_classes=%d, exc=%s); "
            "using 0.0 sentinel", _diag_n, _diag_n_classes, exc,
        )
        metrics['f1_weighted'] = 0.0
        metrics['f1_macro'] = 0.0

    # Precision and recall (weighted)
    try:
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    except Exception as exc:
        logger.warning(
            "T-29: precision/recall failed (n=%d, n_classes=%d, exc=%s); "
            "using 0.0 sentinel", _diag_n, _diag_n_classes, exc,
        )
        metrics['precision_weighted'] = 0.0
        metrics['recall_weighted'] = 0.0

    # ROC AUC (if probabilities provided)
    # Using 'macro' average for consistency with CV folds (search.py) and
    # because macro gives equal weight to each class, which is preferred for imbalanced data
    if y_pred_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification - use proba for positive class
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multiclass - use one-vs-rest with macro averaging
                # Macro: equal weight per class (better for imbalanced data)
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba,
                                                   multi_class='ovr', average='macro')
        except Exception as exc:
            # ROC AUC commonly fails on small/extreme-imbalance CV folds
            # with "Only one class present in y_true" — log so the user
            # knows roc_auc=None means computation failed, not "no proba
            # provided" (the y_pred_proba-is-None branch returns None too).
            logger.warning(
                "T-29: roc_auc failed (n=%d, n_classes=%d, exc=%s); "
                "using None sentinel", _diag_n, _diag_n_classes, exc,
            )
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    return metrics


def print_imbalance_metrics_report(metrics, dataset_name="Test Set"):
    """
    Print a formatted report of imbalance-aware metrics.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_imbalance_metrics()
    dataset_name : str
        Name of the dataset (for display purposes)

    Example
    -------
    >>> metrics = compute_imbalance_metrics(y_test, y_pred, y_proba)
    >>> print_imbalance_metrics_report(metrics, "Validation Set")
    """
    print("\n" + "="*60)
    print(f"Imbalance-Aware Metrics Report: {dataset_name}")
    print("="*60)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)

    print(f"{'Balanced Accuracy':<30} {metrics['balanced_accuracy']:>10.4f}")
    print(f"{'F1 Score (Weighted)':<30} {metrics['f1_weighted']:>10.4f}")
    print(f"{'F1 Score (Macro)':<30} {metrics['f1_macro']:>10.4f}")
    print(f"{'Precision (Weighted)':<30} {metrics['precision_weighted']:>10.4f}")
    print(f"{'Recall (Weighted)':<30} {metrics['recall_weighted']:>10.4f}")

    if metrics['roc_auc'] is not None:
        print(f"{'ROC AUC':<30} {metrics['roc_auc']:>10.4f}")

    print("="*60)
    print("\nNotes:")
    print("- Balanced Accuracy: Macro-averaged recall (equal weight per class)")
    print("- Weighted metrics: Account for class frequency in dataset")
    print("- Macro metrics: Treat all classes equally regardless of frequency")
    print("="*60 + "\n")
