import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, balanced_accuracy_score,
)


def one_class_metrics(y_true, y_pred, scores=None):
    """Compute metrics for one-class classification evaluation.

    Parameters
    ----------
    y_true : array-like
        True labels: +1 for inlier (clean), -1 for outlier (out-of-class).
    y_pred : array-like
        Predicted labels: +1 or -1.
    scores : array-like, optional
        Decision function scores for AUC computation.

    Returns
    -------
    metrics : dict
        Dictionary with sensitivity, specificity, precision, f1, accuracy,
        balanced_accuracy, auc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)

    metrics = {}
    n_true_outliers = y_true_bin.sum()
    n_true_inliers = (1 - y_true_bin).sum()

    metrics['sensitivity'] = recall_score(y_true_bin, y_pred_bin, zero_division=0) if n_true_outliers > 0 else np.nan
    metrics['specificity'] = recall_score(1 - y_true_bin, 1 - y_pred_bin, zero_division=0) if n_true_inliers > 0 else np.nan
    metrics['precision'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    metrics['f1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true_bin, y_pred_bin)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_bin, y_pred_bin) if (n_true_outliers > 0 and n_true_inliers > 0) else np.nan

    if scores is not None and n_true_outliers > 0 and n_true_inliers > 0:
        try:
            metrics['auc'] = roc_auc_score(y_true_bin, -np.asarray(scores))
        except (ValueError, TypeError):
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan

    return metrics
