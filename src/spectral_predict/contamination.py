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
- PCASIMCA: DD-SIMCA — data-driven chi-squared thresholds with Fisher's
  combined p-value test for joint T²/Q acceptance.

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
- Vanden Branden, K. & Hubert, M. (2005). Robust classification in high
  dimensions based on the SIMCA method.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

logger = logging.getLogger(__name__)


# ============================================================================
# PCA-SIMCA: Classic chemometrics one-class model
# ============================================================================

class PCASIMCA(BaseEstimator, ClassifierMixin):
    """DD-SIMCA one-class classifier for spectral data.

    Implements the Data-Driven SIMCA (DD-SIMCA) method: a PCA model is built
    on clean/inlier samples and scaled chi-squared distributions are fitted
    to the training Hotelling T² and Q-residual (SPE) statistics.

    Acceptance is determined by Fisher's combined p-value test.  For each
    sample the T² and Q p-values are computed from their fitted chi-squared
    distributions and combined via ``-2 * ln(p_T2 * p_Q) ~ chi2(4)``.
    A sample is accepted (inlier) when this combined statistic does not
    exceed the ``chi2(4)`` quantile at level ``1 - alpha``.  This gives
    proper type-I error control at the requested ``alpha``.

    A method-of-moments fallback is used when scipy's MLE chi-squared fit
    fails (common for very small training sets).

    Parameters
    ----------
    n_components : int or float, default=5
        Number of PCA components. If float in (0, 1), selects components
        to explain that fraction of variance.
    alpha : float, default=0.05
        Significance level for the T² and Q thresholds (0.05 = 95% confidence).
    """

    def __init__(self, n_components: int | float = 5, alpha: float = 0.05):
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X, y=None):
        """Fit DD-SIMCA model on clean/inlier samples only.

        Fits a PCA model and then estimates data-driven chi-squared
        distributions for both Hotelling T² and Q-residuals.  The joint
        acceptance test uses Fisher's method to combine the two p-values
        into a single chi-squared(4) statistic, giving proper type-I error
        control at the requested ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (clean samples only).
        y : ignored
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if n_samples < 3:
            raise ValueError(
                f"Need at least 3 clean samples to fit DD-SIMCA, got {n_samples}"
            )

        # Determine n_components
        max_components = min(n_samples - 1, n_features)
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
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

        # Store eigenvalues for diagonal T² computation
        self.eigenvalues_ = self.pca_.explained_variance_

        # Data-driven chi-squared fit to training T² values
        t2_train = self._compute_t2(self.scores_)
        self.t2_train_ = t2_train
        self.t2_dof_, _, self.t2_scale_ = self._fit_chi2(t2_train, "T2")

        # Data-driven chi-squared fit to training Q residuals
        q_train = self._compute_q_residuals(X)
        if np.max(q_train) > 1e-10:
            self.q_dof_, _, self.q_scale_ = self._fit_chi2(q_train, "Q")
            self.q_threshold_method_ = "chi2_fit"
        else:
            # PCA reconstructs perfectly — Q is degenerate
            self.q_dof_ = 2.0
            self.q_scale_ = 1e-10
            self.q_threshold_method_ = "zero_guard"

        # Joint threshold via Fisher's method: -2*ln(p_t2*p_q) ~ chi2(4)
        self.joint_threshold_ = stats.chi2.ppf(1 - self.alpha, 4)

        # Store per-axis thresholds for diagnostics (not used in accept/reject)
        self.t2_threshold_ = stats.chi2.ppf(
            1 - self.alpha, self.t2_dof_, loc=0, scale=self.t2_scale_
        )
        self.q_threshold_ = stats.chi2.ppf(
            1 - self.alpha, self.q_dof_, loc=0, scale=self.q_scale_
        )

        return self

    def _fit_chi2(self, values: np.ndarray, label: str) -> tuple[float, float, float]:
        """Fit a scaled chi-squared distribution to positive sample statistics.

        Tries scipy MLE first; falls back to method-of-moments if MLE raises.

        Parameters
        ----------
        values : ndarray
            Positive statistic values (T² or Q) computed on training data.
        label : str
            Name used in debug log messages.

        Returns
        -------
        dof, loc, scale : float
            Fitted chi-squared degrees of freedom, location (always 0), and scale.
        """
        try:
            dof, loc, scale = stats.chi2.fit(values, floc=0)
            return dof, loc, scale
        except (RuntimeError, ValueError) as exc:
            logger.debug("chi2.fit failed for %s (%s); using method-of-moments", label, exc)

        # Method-of-moments fallback: df = 2*mean²/var, scale = var/(2*mean)
        mean_val = np.mean(values)
        var_val = np.var(values)
        if var_val > 1e-10 and mean_val > 1e-10:
            dof = 2 * mean_val**2 / var_val
            scale = var_val / (2 * mean_val)
        else:
            # Last resort: match mean to chi2(df=2) scaled by mean/2
            dof = 2.0
            scale = max(mean_val / 2.0, 1e-10)
        return dof, 0.0, scale

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
        """Compute anomaly score via Fisher's combined p-value test.

        For each sample, p-values are computed from the fitted T² and Q
        chi-squared distributions.  These are combined using Fisher's
        method: ``-2 * ln(p_T2 * p_Q) ~ chi2(4)``.  The score is the
        difference ``joint_threshold - fisher_stat``; positive means
        inlier, negative means outlier.

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

        t2 = self._compute_t2(pc_scores)
        q = self._compute_q_residuals(X)

        # Per-axis p-values from fitted chi-squared distributions
        p_t2 = 1.0 - stats.chi2.cdf(t2, self.t2_dof_, loc=0, scale=self.t2_scale_)
        p_q = 1.0 - stats.chi2.cdf(q, self.q_dof_, loc=0, scale=self.q_scale_)

        # Clip to avoid log(0); 1e-300 is safely above float64 underflow
        p_t2 = np.clip(p_t2, 1e-300, 1.0)
        p_q = np.clip(p_q, 1e-300, 1.0)

        # Fisher's method: combine independent p-values → chi2(4)
        fisher_stat = -2.0 * (np.log(p_t2) + np.log(p_q))

        # Score: positive = inside acceptance, negative = outside
        return self.joint_threshold_ - fisher_stat

    def score_samples(self, X):
        """Alias for decision_function (sklearn convention)."""
        return self.decision_function(X)

    def _compute_t2(self, scores):
        """Compute Hotelling T² using diagonal formulation.

        Uses T² = sum(t_a² / lambda_a) which is numerically stable
        and exact for PCA scores (which are orthogonal by construction).
        """
        eigenvalues = self.eigenvalues_
        if self.n_components_ == 1:
            lam = max(eigenvalues[0], 1e-10)
            return (scores.ravel() ** 2) / lam
        else:
            # Vectorized: T² = sum_a (score_a² / lambda_a)
            lam = np.maximum(eigenvalues, 1e-10)
            return np.sum(scores ** 2 / lam, axis=1)

    def _compute_q_residuals(self, X):
        """Compute Q-residuals (SPE) for samples."""
        X = np.asarray(X, dtype=np.float64)
        scores = self.pca_.transform(X)
        X_reconstructed = scores @ self.pca_.components_ + self.pca_.mean_
        residuals = X - X_reconstructed
        return np.sum(residuals ** 2, axis=1)


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
        defaults = dict(n_components=5, alpha=0.05)
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
        except (ValueError, TypeError) as exc:
            logger.debug("AUC computation failed: %s", exc)
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan

    return metrics
