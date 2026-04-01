"""Contamination detection models for NIR spectroscopic screening.

Implements one-class classification models that learn ONLY from clean/inlier
samples and flag anything outside that distribution as contaminated/suspect.
This solves the heterogeneous contaminant class problem where PLS-DA and
tree-based classifiers struggle with wildly variable positive classes.

Models provided:
- OneClassSVM: Kernel-based boundary around clean data
- IsolationForest: Isolation-based anomaly detection (fast, high-dim friendly)
- EllipticEnvelope: Mahalanobis-based (assumes roughly Gaussian clean class)
- LocalOutlierFactor: Density-based novelty detection
- PCA-SIMCA: PCA residuals + Hotelling T² (classic chemometrics approach)

Usage pattern:
    All models follow sklearn's one-class convention:
    - fit(X_clean) trains only on clean/inlier samples
    - predict(X) returns +1 (inlier/clean) or -1 (outlier/contaminated)
    - decision_function(X) returns continuous anomaly scores

References
----------
- Brereton, R.G. (2015). Pattern recognition applied to NIR.
- Oliveri, P. & Downey, G. (2012). DD-SIMCA for food authentication.
- Rodionova, O.Y. et al. (2016). Rigorous and compliant approaches to
  one-class classification.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


# ============================================================================
# PCA-SIMCA: Classic chemometrics one-class model
# ============================================================================

class PCASIMCA(BaseEstimator, ClassifierMixin):
    """PCA-based one-class classifier (SIMCA-style) for spectral data.

    Builds a PCA model on clean/inlier samples and uses the combined
    Hotelling T² and Q-residual (SPE) statistics to detect outliers.
    A sample is flagged as contaminated if EITHER statistic exceeds
    its confidence threshold.

    This is the classic chemometrics approach to contamination screening,
    equivalent to DD-SIMCA when using both T² and Q statistics with
    proper F-distribution thresholds.

    Parameters
    ----------
    n_components : int or float, default=5
        Number of PCA components. If float in (0, 1), selects components
        to explain that fraction of variance.
    alpha : float, default=0.05
        Significance level for the T² and Q thresholds (0.05 = 95% confidence).
    contamination : float, default=0.05
        Expected fraction of outliers. Used to calibrate thresholds when
        the F-distribution assumption is too conservative.
    """

    def __init__(self, n_components: int | float = 5, alpha: float = 0.05,
                 contamination: float = 0.05):
        self.n_components = n_components
        self.alpha = alpha
        self.contamination = contamination

    def fit(self, X, y=None):
        """Fit PCA model on clean/inlier samples only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (clean samples only).
        y : ignored
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Determine n_components
        max_components = min(n_samples - 1, n_features)
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Fit full PCA first to find how many components explain the variance
            pca_full = PCA(n_components=max_components)
            pca_full.fit(X)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cumvar, self.n_components) + 1)
            n_comp = min(n_comp, max_components)
        else:
            n_comp = min(int(self.n_components), max_components)

        self.n_components_ = n_comp
        self.pca_ = PCA(n_components=n_comp)
        self.scores_ = self.pca_.fit_transform(X)
        self.n_train_ = n_samples

        # Hotelling T² threshold (F-distribution)
        p = self.n_components_
        n = n_samples
        if n > p:
            self.t2_threshold_ = (
                p * (n - 1) / (n - p) *
                stats.f.ppf(1 - self.alpha, p, n - p)
            )
        else:
            self.t2_threshold_ = stats.chi2.ppf(1 - self.alpha, p)

        # Q-residual threshold (from training data distribution)
        q_residuals_train = self._compute_q_residuals(X)
        self.q_threshold_ = np.percentile(
            q_residuals_train, 100 * (1 - self.alpha)
        )

        # Compute training T² for score normalization
        self.t2_train_ = self._compute_t2(self.scores_)

        return self

    def predict(self, X):
        """Predict inlier (+1) or outlier (-1) for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            +1 for inlier (clean), -1 for outlier (contaminated).
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def decision_function(self, X):
        """Compute anomaly score for each sample.

        Higher values indicate more normal samples. Negative values
        indicate outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Positive = inlier, negative = outlier.
        """
        X = np.asarray(X, dtype=np.float64)
        pc_scores = self.pca_.transform(X)

        # Hotelling T² (normalized by threshold so 1.0 = on boundary)
        t2 = self._compute_t2(pc_scores)
        t2_ratio = t2 / self.t2_threshold_

        # Q-residuals (normalized by threshold)
        q = self._compute_q_residuals(X)
        q_ratio = q / self.q_threshold_

        # Combined score: worst of the two ratios
        # Score > 0 means inlier, < 0 means outlier
        worst_ratio = np.maximum(t2_ratio, q_ratio)
        return 1.0 - worst_ratio

    def score_samples(self, X):
        """Alias for decision_function (sklearn convention)."""
        return self.decision_function(X)

    def _compute_t2(self, scores):
        """Compute Hotelling T² for PC scores."""
        # Covariance from training scores
        cov = np.cov(self.scores_.T)
        if self.n_components_ == 1:
            var = np.var(self.scores_)
            if var < 1e-10:
                var = 1e-10
            return (scores.ravel() ** 2) / var
        else:
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov += np.eye(self.n_components_) * 1e-6
                inv_cov = np.linalg.inv(cov)
            return np.array([s @ inv_cov @ s.T for s in scores])

    def _compute_q_residuals(self, X):
        """Compute Q-residuals (SPE) for samples."""
        X = np.asarray(X, dtype=np.float64)
        scores = self.pca_.transform(X)
        X_reconstructed = scores @ self.pca_.components_ + self.pca_.mean_
        residuals = X - X_reconstructed
        return np.sum(residuals ** 2, axis=1)

    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'alpha': self.alpha,
            'contamination': self.contamination,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ============================================================================
# ONE-CLASS MODEL FACTORY
# ============================================================================

def get_one_class_model(model_name: str, **kwargs):
    """Get a one-class model instance with sensible defaults for NIR data.

    Parameters
    ----------
    model_name : str
        One of: 'OneClassSVM', 'IsolationForest', 'EllipticEnvelope',
        'LOF', 'PCA-SIMCA'
    **kwargs
        Override default hyperparameters.

    Returns
    -------
    model : estimator
        Configured one-class model instance.
    """
    if model_name == 'OneClassSVM':
        defaults = dict(kernel='rbf', gamma='scale', nu=0.05)
        defaults.update(kwargs)
        return OneClassSVM(**defaults)

    elif model_name == 'IsolationForest':
        defaults = dict(
            n_estimators=200, contamination=0.05,
            random_state=42, n_jobs=-1
        )
        defaults.update(kwargs)
        return IsolationForest(**defaults)

    elif model_name == 'EllipticEnvelope':
        defaults = dict(contamination=0.05, random_state=42)
        defaults.update(kwargs)
        return EllipticEnvelope(**defaults)

    elif model_name == 'LOF':
        defaults = dict(
            n_neighbors=20, contamination=0.05,
            novelty=True, n_jobs=-1
        )
        defaults.update(kwargs)
        return LocalOutlierFactor(**defaults)

    elif model_name == 'PCA-SIMCA':
        defaults = dict(n_components=5, alpha=0.05, contamination=0.05)
        defaults.update(kwargs)
        return PCASIMCA(**defaults)

    else:
        raise ValueError(
            f"Unknown one-class model: {model_name}. "
            f"Available: OneClassSVM, IsolationForest, EllipticEnvelope, LOF, PCA-SIMCA"
        )


def build_one_class_model(model_name: str, params: dict):
    """Build a one-class model with specific hyperparameters.

    Used by the search pipeline to instantiate models from grid search
    parameter combinations.

    Parameters
    ----------
    model_name : str
        Model name.
    params : dict
        Hyperparameters to pass to the model constructor.

    Returns
    -------
    model : estimator
        Configured model instance.
    """
    if model_name == 'OneClassSVM':
        return OneClassSVM(**params)
    elif model_name == 'IsolationForest':
        return IsolationForest(**{**params, 'random_state': 42})
    elif model_name == 'EllipticEnvelope':
        return EllipticEnvelope(**{**params, 'random_state': 42})
    elif model_name == 'LOF':
        return LocalOutlierFactor(**{**params, 'novelty': True})
    elif model_name == 'PCA-SIMCA':
        return PCASIMCA(**params)
    else:
        raise ValueError(f"Unknown one-class model: {model_name}")


def get_one_class_model_grids() -> dict:
    """Get hyperparameter grids for all one-class models.

    Returns
    -------
    grids : dict
        Mapping of model_name -> list of param dicts.
    """
    grids = {
        'OneClassSVM': [
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.01},
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
            {'kernel': 'rbf', 'gamma': 'auto', 'nu': 0.05},
            {'kernel': 'poly', 'gamma': 'scale', 'nu': 0.05, 'degree': 2},
        ],
        'IsolationForest': [
            {'n_estimators': 100, 'contamination': 0.01, 'max_features': 1.0},
            {'n_estimators': 200, 'contamination': 0.05, 'max_features': 1.0},
            {'n_estimators': 200, 'contamination': 0.05, 'max_features': 0.5},
            {'n_estimators': 200, 'contamination': 0.1, 'max_features': 1.0},
            {'n_estimators': 300, 'contamination': 0.05, 'max_features': 0.8},
        ],
        'EllipticEnvelope': [
            {'contamination': 0.01},
            {'contamination': 0.05},
            {'contamination': 0.1},
        ],
        'LOF': [
            {'n_neighbors': 10, 'contamination': 0.05},
            {'n_neighbors': 20, 'contamination': 0.05},
            {'n_neighbors': 30, 'contamination': 0.05},
            {'n_neighbors': 20, 'contamination': 0.01},
            {'n_neighbors': 20, 'contamination': 0.1},
        ],
        'PCA-SIMCA': [
            {'n_components': 3, 'alpha': 0.05},
            {'n_components': 5, 'alpha': 0.05},
            {'n_components': 7, 'alpha': 0.05},
            {'n_components': 5, 'alpha': 0.01},
            {'n_components': 0.95, 'alpha': 0.05},  # 95% variance explained
        ],
    }
    return grids


# ============================================================================
# ONE-CLASS EVALUATION METRICS
# ============================================================================

def one_class_metrics(y_true, y_pred, scores=None):
    """Compute metrics for one-class classification evaluation.

    Parameters
    ----------
    y_true : array-like
        True labels: +1 for inlier (clean), -1 for outlier (contaminated).
    y_pred : array-like
        Predicted labels: +1 or -1.
    scores : array-like, optional
        Decision function scores for AUC computation.

    Returns
    -------
    metrics : dict
        Dictionary with:
        - sensitivity: Proportion of true outliers detected (recall of -1 class)
        - specificity: Proportion of true inliers correctly identified (recall of +1 class)
        - precision: Precision of outlier predictions
        - f1: F1 score for outlier detection
        - accuracy: Overall accuracy
        - balanced_accuracy: Average of sensitivity and specificity
        - auc: ROC AUC if scores provided
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, balanced_accuracy_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Convert to binary: 1 = outlier (positive), 0 = inlier (negative)
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)

    metrics = {}

    # Sensitivity = recall of outlier class (how many contaminants did we catch?)
    n_true_outliers = y_true_bin.sum()
    if n_true_outliers > 0:
        metrics['sensitivity'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    else:
        metrics['sensitivity'] = np.nan

    # Specificity = recall of inlier class (how many clean samples stayed clean?)
    n_true_inliers = (1 - y_true_bin).sum()
    if n_true_inliers > 0:
        metrics['specificity'] = recall_score(
            1 - y_true_bin, 1 - y_pred_bin, zero_division=0
        )
    else:
        metrics['specificity'] = np.nan

    # Precision of outlier predictions
    metrics['precision'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)

    # F1 for outlier detection
    metrics['f1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true_bin, y_pred_bin)

    # Balanced accuracy
    if n_true_outliers > 0 and n_true_inliers > 0:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_bin, y_pred_bin)
    else:
        metrics['balanced_accuracy'] = np.nan

    # AUC (if scores available)
    if scores is not None and n_true_outliers > 0 and n_true_inliers > 0:
        try:
            # Negate scores: higher decision_function = more normal,
            # but AUC expects higher score = more positive (outlier)
            metrics['auc'] = roc_auc_score(y_true_bin, -np.asarray(scores))
        except Exception:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan

    return metrics


# ============================================================================
# ONE-CLASS CROSS-VALIDATION
# ============================================================================

def one_class_cv(model, X, y_true, folds=5, random_state=42):
    """Cross-validate a one-class model with proper train/test splitting.

    Training folds contain ONLY inlier samples (+1).
    Test folds contain BOTH inliers and outliers to evaluate detection.

    Parameters
    ----------
    model : estimator
        One-class model with fit/predict/decision_function interface.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (all samples).
    y_true : ndarray of shape (n_samples,)
        True labels: +1 for inlier, -1 for outlier.
    folds : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Averaged metrics across folds plus per-fold details.
    """
    from sklearn.model_selection import KFold
    from sklearn.base import clone

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true)

    # Split inliers and outliers
    inlier_mask = y_true == 1
    outlier_mask = y_true == -1
    inlier_indices = np.where(inlier_mask)[0]
    outlier_indices = np.where(outlier_mask)[0]

    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)

    if n_inliers < folds:
        raise ValueError(
            f"Not enough inlier samples ({n_inliers}) for {folds}-fold CV. "
            f"Need at least {folds} inlier samples."
        )

    # KFold on inlier indices only
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    fold_metrics = []
    for train_inlier_idx, test_inlier_idx in kf.split(inlier_indices):
        # Training set: only inliers from training fold
        train_idx = inlier_indices[train_inlier_idx]

        # Test set: held-out inliers + ALL outliers
        test_inlier = inlier_indices[test_inlier_idx]
        test_idx = np.concatenate([test_inlier, outlier_indices])
        test_labels = np.concatenate([
            np.ones(len(test_inlier), dtype=int),
            -np.ones(n_outliers, dtype=int)
        ])

        # Fit on clean training data only
        model_clone = clone(model)
        model_clone.fit(X[train_idx])

        # Predict on test set
        y_pred = model_clone.predict(X[test_idx])

        # Get decision scores if available
        scores = None
        if hasattr(model_clone, 'decision_function'):
            scores = model_clone.decision_function(X[test_idx])
        elif hasattr(model_clone, 'score_samples'):
            scores = model_clone.score_samples(X[test_idx])

        fold_result = one_class_metrics(test_labels, y_pred, scores)
        fold_metrics.append(fold_result)

    # Average metrics across folds
    avg_metrics = {}
    for key in fold_metrics[0]:
        values = [fm[key] for fm in fold_metrics if not np.isnan(fm[key])]
        if values:
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        else:
            avg_metrics[key] = np.nan
            avg_metrics[f'{key}_std'] = np.nan

    avg_metrics['n_folds'] = folds
    avg_metrics['n_inliers'] = n_inliers
    avg_metrics['n_outliers'] = n_outliers
    avg_metrics['fold_metrics'] = fold_metrics

    return avg_metrics
