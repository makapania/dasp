"""Scoring and ranking functions."""

import numpy as np
import pandas as pd


def compute_composite_score(df_results, task_type, lambda_penalty=0.15):
    """
    Compute composite score with performance-dominant ranking.

    Formula: ~90% performance, ~10% complexity
    Score = performance_score + complexity_penalty

    Performance score (lower is better):
    - Regression: 0.5*z(RMSE) - 0.5*z(R2)  [combine both metrics]
    - Classification: -z(ROC_AUC) - 0.3*z(Accuracy)

    Complexity penalty (linear, no harsh sparsity penalties):
    - LV penalty: Small penalty for more latent variables
    - Variable penalty: Small penalty for more variables
    - Both weighted equally, scaled to ~10% of total score

    This ensures models with similar R² stay close in ranking,
    with complexity only as a tiebreaker.

    Lower composite score is better.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results dataframe with metrics and complexity measures
    task_type : str
        'regression' or 'classification'
    lambda_penalty : float
        Penalty weight for model complexity (default: 0.15)
        This is scaled down internally to keep complexity at ~10% influence.

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

    # Compute complexity penalty (simple linear penalties, no harsh sparsity penalties)
    # Scale complexity to be ~10% of performance (performance z-scores typically range ±3)
    # To make complexity ~10% of total: use small penalty weight

    # 1. LV penalty (for PLS models) - normalized to [0, 1]
    # Fewer LVs is better, so higher LVs = higher penalty
    lvs_penalty = df["LVs"].fillna(0).astype(np.float64) / 25.0

    # 2. Variable fraction penalty - normalized to [0, 1]
    # Fewer variables is better, so more variables = higher penalty
    n_vars_array = np.asarray(df["n_vars"], dtype=np.float64)
    full_vars_array = np.asarray(df["full_vars"], dtype=np.float64)
    vars_penalty = n_vars_array / full_vars_array

    # Scale complexity to ~10% of performance contribution
    # Performance z-scores range ~[-3, +3], so range is ~6
    # We want complexity to contribute ~0.6 units (10% of 6)
    # lvs_penalty + vars_penalty ranges [0, 2], so multiply by 0.3 to get ~0.6
    # Additionally, use user's lambda_penalty as a tuning factor
    complexity_scale = 0.3 * lambda_penalty / 0.15  # Normalize around default 0.15

    # Total complexity penalty
    complexity_penalty = complexity_scale * (lvs_penalty + vars_penalty)

    # Composite score (lower is better)
    df["CompositeScore"] = performance_score + complexity_penalty

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

    return df


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
