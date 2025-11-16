"""Scoring and ranking functions."""

import numpy as np
import pandas as pd


def compute_composite_score(df_results, task_type, variable_penalty=2, complexity_penalty=2, verbose=False):
    """
    Compute composite score with user-friendly penalty system.

    NEW: User-friendly 0-10 penalty system
    - variable_penalty (0-10): Penalty for using many variables
    - complexity_penalty (0-10): Penalty for model complexity (LVs, etc.)
    - 0 = only performance (R²) matters
    - 10 = strong penalty for variables/complexity

    Parameters
    ----------
    verbose : bool, default=False
        If True, print detailed scoring breakdown for top 20 models

    Formula: Score = performance_score + variable_penalty_term + complexity_penalty_term

    Performance score (lower is better):
    - Regression: 0.5*z(RMSE) - 0.5*z(R2)  [combine both metrics]
    - Classification: -z(ROC_AUC) - 0.3*z(Accuracy)

    Penalty terms scale linearly with user settings (0-10).

    Lower composite score is better.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results dataframe with metrics and complexity measures
    task_type : str
        'regression' or 'classification'
    variable_penalty : int (0-10)
        Penalty for using many variables (default: 2)
        0 = ignore variable count, 10 = strongly penalize many variables
        Uses cubic scaling for gentle penalty at low values (exploration-friendly)
    complexity_penalty : int (0-10)
        Penalty for model complexity (default: 2)
        0 = ignore complexity, 10 = strongly penalize complex models

    Returns
    -------
    df_scored : pd.DataFrame
        Results with CompositeScore and Rank columns added
    """
    df = df_results.copy()

    # Compute performance score (combining multiple metrics)
    if task_type == "regression":
        # Z-score for RMSE (lower is better, positive z = bad)
        z_rmse = (df["RMSE"] - df["RMSE"].mean()) / df["RMSE"].std()
        z_rmse = z_rmse.fillna(0)

        # Z-score for R2 (higher is better, negative z = bad)
        z_r2 = (df["R2"] - df["R2"].mean()) / df["R2"].std()
        z_r2 = z_r2.fillna(0)

        # Combined performance score (lower is better)
        # Weight RMSE and R2 equally, but negate R2 since higher is better
        performance_score = 0.5 * z_rmse - 0.5 * z_r2

    else:  # classification
        # Z-score for ROC_AUC (higher is better)
        z_auc = (df["ROC_AUC"] - df["ROC_AUC"].mean()) / df["ROC_AUC"].std()
        z_auc = z_auc.fillna(0)

        # Z-score for Accuracy (higher is better)
        z_acc = (df["Accuracy"] - df["Accuracy"].mean()) / df["Accuracy"].std()
        z_acc = z_acc.fillna(0)

        # Combined performance score (lower is better, so negate)
        performance_score = -z_auc - 0.3 * z_acc

    # NEW: User-friendly penalty system (0-10 scale)
    # Performance z-scores typically range ±3, so we scale penalties to be meaningful but not overwhelming

    # 1. Variable Count Penalty (0-10 scale)
    # Normalize: fraction of variables used, scaled by user preference
    if variable_penalty > 0:
        n_vars_array = np.asarray(df["n_vars"], dtype=np.float64)
        full_vars_array = np.asarray(df["full_vars"], dtype=np.float64)
        var_fraction = n_vars_array / full_vars_array  # 0-1 scale

        # RESTORED PENALTY: Scale by user penalty (0-10) with quadratic scaling for meaningful impact
        # At penalty=2, using all variables adds ~0.04 units (1-2% of z-score range)
        # At penalty=5, using all variables adds ~0.25 units (4-8% of z-score range)
        # At penalty=10, using all variables adds ~1.0 unit (comparable to ~0.3 std deviations in performance)
        # This provides meaningful variable selection pressure while still allowing full spectrum exploration
        var_penalty_term = ((variable_penalty / 10.0) ** 2) * var_fraction
    else:
        var_penalty_term = 0

    # 2. Model Complexity Penalty (0-10 scale)
    # Use model-type-specific complexity for fair comparison across different algorithms
    if complexity_penalty > 0:
        lvs = df["LVs"].fillna(0).infer_objects(copy=False).astype(np.float64)

        # Get model types
        models = df["Model"].astype(str)

        # For PLS models: use latent variables
        # Normalize LVs to [0, 1] scale (assume max useful LVs is ~25)
        lv_fraction = lvs / 25.0
        lv_fraction = np.minimum(lv_fraction, 1.0)  # Cap at 1.0

        # For non-PLS models: use intrinsic model complexity
        # Based on model type complexity scores (normalized to [0, 1])
        model_complexity_map = {
            "PLS": None,  # Use LVs for PLS
            "Ridge": 0.25,
            "Lasso": 0.30,
            "ElasticNet": 0.28,
            "RandomForest": 0.60,
            "XGBoost": 0.65,
            "LightGBM": 0.63,
            "CatBoost": 0.64,
            "MLP": 0.80,
            "NeuralBoosted": 0.85,
            "SVM": 0.55,
            "KNN": 0.45
        }

        # Calculate complexity for each model
        complexity_values = []
        for idx, (model, lv_val, lv_frac) in enumerate(zip(models, lvs, lv_fraction)):
            if lv_val > 0:  # PLS model with LVs
                complexity_values.append(lv_frac)
            else:
                # Non-PLS: use model-specific complexity
                complexity = model_complexity_map.get(model, 0.50)  # Default to 0.50 if unknown
                complexity_values.append(complexity)

        lv_fraction_adjusted = np.array(complexity_values, dtype=np.float64)

        # Scale by user penalty (0-10) with quadratic scaling for better behavior
        comp_penalty_term = ((complexity_penalty / 10.0) ** 2) * lv_fraction_adjusted
    else:
        comp_penalty_term = 0

    # Store individual components for diagnostics
    df["PerformanceScore"] = performance_score
    df["VarPenalty"] = var_penalty_term if not np.isscalar(var_penalty_term) else np.zeros(len(df))
    df["CompPenalty"] = comp_penalty_term if not np.isscalar(comp_penalty_term) else np.zeros(len(df))

    # Composite score (lower is better)
    # Performance score dominates, but penalties can affect ranking meaningfully
    df["CompositeScore"] = performance_score + var_penalty_term + comp_penalty_term

    # Rank (1 = best)
    df["Rank"] = df["CompositeScore"].rank(method="min").astype(int)

    # Sort by rank
    df = df.sort_values("Rank")

    # Reorder columns: Rank first, top_vars last
    # Get all columns except Rank and top_vars
    cols = [c for c in df.columns if c not in ['Rank', 'top_vars']]
    # Construct new column order: Rank first, then everything else, then top_vars
    new_col_order = ['Rank'] + cols + ['top_vars']
    df = df[new_col_order]

    # Add unified complexity score (0-100 scale, higher = more complex)
    # This is a new column for user convenience, doesn't affect ranking
    try:
        df["ComplexityScore"] = df.apply(_compute_unified_complexity, axis=1)
    except Exception as e:
        # If complexity calculation fails, set to NaN (don't break pipeline)
        print(f"Warning: Unified complexity calculation failed: {e}")
        df["ComplexityScore"] = np.nan

    # Verbose diagnostic output
    if verbose:
        print("\n" + "="*80)
        print("RANKING DIAGNOSTIC - Top 20 Models")
        print("="*80)
        print(f"Penalty Settings: Variables={variable_penalty}/10, Complexity={complexity_penalty}/10")
        print(f"Task Type: {task_type}")
        print("\nScore Components (lower CompositeScore = better rank):")
        print("  - PerformanceScore: Based on R²/RMSE (or AUC/Accuracy)")
        print("  - VarPenalty: Penalty for using many wavelengths")
        print("  - CompPenalty: Penalty for model complexity")
        print("-"*80)

        # Show top 20 models with score breakdown
        top20 = df.head(20).copy()

        if task_type == "regression":
            display_cols = ["Rank", "Model", "R2", "RMSE", "n_vars", "LVs",
                           "PerformanceScore", "VarPenalty", "CompPenalty", "CompositeScore"]
        else:
            display_cols = ["Rank", "Model", "ROC_AUC", "Accuracy", "n_vars", "LVs",
                           "PerformanceScore", "VarPenalty", "CompPenalty", "CompositeScore"]

        # Filter to available columns
        display_cols = [c for c in display_cols if c in top20.columns]

        # Format for display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)

        print(top20[display_cols].to_string(index=False))
        print("="*80 + "\n")

    return df


def _compute_unified_complexity(row):
    """
    Compute unified complexity score (0-100 scale, higher = more complex).

    Formula: ComplexityScore = 0.25×Model + 0.30×Variables + 0.25×LVs + 0.20×Preprocessing

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
        "Ridge": 25,
        "Lasso": 30,
        "RandomForest": 60,
        "MLP": 80,
        "NeuralBoosted": 85
    }
    model_complexity = model_scores.get(model, 50)  # Default to 50 if unknown

    # 2. Variable Complexity (30% weight) - nonlinear penalty for many variables
    # Use sqrt-based nonlinear penalty: few vars = low penalty, many vars = high penalty
    n_vars = row.get("n_vars", 0)
    # Normalize: 10 vars ≈ 2.0, 100 vars ≈ 20, 500 vars ≈ 100
    var_complexity = min(100, np.sqrt(n_vars) * 4.5)

    # 3. Latent Variable Complexity (25% weight) - for PLS models
    lvs = row.get("LVs", np.nan)
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
    ]

    if task_type == "regression":
        metric_cols = ["RMSE", "R2"]
    else:
        metric_cols = ["Accuracy", "ROC_AUC"]

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
