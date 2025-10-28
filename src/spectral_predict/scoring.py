"""Scoring and ranking functions."""

import numpy as np
import pandas as pd


def compute_composite_score(df_results, task_type, lambda_penalty=0.15):
    """
    Compute composite score with improved simplicity penalty.

    New formula balances performance and parsimony:
    Score = performance_score + complexity_penalty

    Performance score (lower is better):
    - Regression: 0.5*z(RMSE) - 0.5*z(R2)  [combine both metrics]
    - Classification: -z(ROC_AUC) - 0.3*z(Accuracy)

    Complexity penalty (non-linear for sparse models):
    - LV penalty: λ × (LVs/25)
    - Variable penalty: λ × (n_vars/full_vars)
    - Sparsity penalty: Additional penalty when using very few variables
      * If n_vars < 10: add λ × 2.0 (heavy penalty)
      * If n_vars < 25: add λ × 1.0 (moderate penalty)
      * If n_vars < 1% of full_vars: add λ × 1.5

    Lower composite score is better.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results dataframe with metrics and complexity measures
    task_type : str
        'regression' or 'classification'
    lambda_penalty : float
        Penalty weight for model complexity

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

    # Compute complexity penalty
    # 1. LV penalty (for PLS models)
    lvs_penalty = df["LVs"].fillna(0) / 25.0

    # 2. Variable fraction penalty
    vars_penalty = df["n_vars"] / df["full_vars"]

    # 3. Sparsity penalty (non-linear penalty for very sparse models)
    sparsity_penalty = np.zeros(len(df))

    # Heavy penalty for extremely sparse models
    very_sparse_mask = df["n_vars"] < 10
    sparsity_penalty[very_sparse_mask] += 2.0

    # Moderate penalty for sparse models
    sparse_mask = (df["n_vars"] >= 10) & (df["n_vars"] < 25)
    sparsity_penalty[sparse_mask] += 1.0

    # Additional penalty if using less than 1% of available variables
    percent_vars = (df["n_vars"] / df["full_vars"]) * 100
    ultra_sparse_mask = (percent_vars < 1.0) & (df["n_vars"] >= 10)
    sparsity_penalty[ultra_sparse_mask] += 1.5

    # Total complexity penalty
    complexity_penalty = lambda_penalty * (lvs_penalty + vars_penalty + sparsity_penalty)

    # Composite score (lower is better)
    df["CompositeScore"] = performance_score + complexity_penalty

    # Rank (1 = best)
    df["Rank"] = df["CompositeScore"].rank(method="min").astype(int)

    # Sort by rank
    df = df.sort_values("Rank")

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

    all_cols = common_cols + metric_cols + ["CompositeScore", "Rank"]

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
