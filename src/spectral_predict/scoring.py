"""Scoring and ranking functions."""

import numpy as np
import pandas as pd


def compute_composite_score(df_results, task_type, lambda_penalty=0.15):
    """
    Compute composite score with simplicity penalty.

    Score = z(primary_metric) + λ × (LVs/25 + n_vars/full_vars)

    For regression: primary_metric = RMSE (lower is better, so use +z)
    For classification: primary_metric = ROC_AUC (higher is better, so use -z)

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

    # Determine primary metric
    if task_type == "regression":
        metric_col = "RMSE"
        # For RMSE, lower is better, so we want positive z-scores for bad models
        z_metric = (df[metric_col] - df[metric_col].mean()) / df[metric_col].std()
    else:  # classification
        metric_col = "ROC_AUC"
        # For ROC_AUC, higher is better, so we negate to make lower scores better
        z_metric = -(df[metric_col] - df[metric_col].mean()) / df[metric_col].std()

    # Handle case where std = 0 (all same)
    if np.isnan(z_metric).any():
        z_metric = z_metric.fillna(0)

    # Compute complexity penalty
    # LVs: for non-PLS models, we'll use 0 or handle separately
    lvs_penalty = df["LVs"].fillna(0) / 25.0
    vars_penalty = df["n_vars"] / df["full_vars"]

    # Composite score
    df["CompositeScore"] = z_metric + lambda_penalty * (lvs_penalty + vars_penalty)

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
