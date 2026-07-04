"""One-class detection models for NIR spectroscopic screening.

Implements one-class classification models that learn ONLY from clean/inlier
samples and flag anything outside that distribution as out-of-class/suspect.
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
    - predict(X) returns +1 (inlier/clean) or -1 (outlier/out-of-class)
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

import ast
import logging
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA

from spectral_predict.cv_utils import build_cv_splitter, _is_repeated_cv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, balanced_accuracy_score,
)
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
        # Reject out-of-range alpha at construction time. The chi² quantile
        # used for the joint threshold (fit() at line 161) is undefined or
        # NaN for alpha <= 0 or alpha >= 1, which would silently produce
        # a model that flags everything (or nothing). All current GUI/grid
        # callers pass valid alphas, but this guards the public API.
        if not (isinstance(alpha, (int, float)) and 0 < float(alpha) < 1):
            raise ValueError(
                f"alpha must be in (0, 1), got {alpha!r}"
            )
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

        # Absolute mathematical minimum: need ≥ 3 samples for the chi² moment
        # fit and ≥ n_components + 1 for PCA to have any residual dimension.
        # The previous hardcoded floor of 10 was a conservative stability
        # cushion that silently killed SIMCA on small-fold CV (e.g. 5-fold on
        # ~7 training inliers leaves 5–6 samples per fold, which OCSVM / IF /
        # LOF / EllipticEnvelope all handle via their own fallbacks). With
        # this relaxed guard SIMCA attempts to fit; method-of-moments chi²
        # fallback at _fit_chi2 handles the small-N instability gracefully.
        if n_samples < 3:
            raise ValueError(
                f"Need at least 3 clean samples to fit DD-SIMCA, got {n_samples}"
            )

        # Determine n_components, clamping aggressively for small samples:
        # PCA needs at least one residual dimension, so n_components <= n_samples - 1.
        max_components = min(n_samples - 1, n_features)
        if max_components < 1:
            raise ValueError(
                f"Cannot fit DD-SIMCA with n_samples={n_samples}, n_features={n_features}"
            )
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # sklearn PCA supports float n_components as variance fraction
            pca_probe = PCA(n_components=self.n_components)
            pca_probe.fit(X)
            n_comp = min(pca_probe.n_components_, max_components)
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

        Fits via scipy method-of-moments (``method='mm'``) first; falls back to
        a manual method-of-moments if scipy raises.

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
            # Use method='mm' (method-of-moments) for speed — MLE can hang on
            # certain data distributions due to scipy's optimizer.
            # Fall through to manual method-of-moments if this scipy version
            # doesn't support method='mm'.
            dof, loc, scale = stats.chi2.fit(values, floc=0, method='mm')
            return dof, loc, scale
        except (RuntimeError, ValueError, TypeError) as exc:
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
            +1 for inlier (clean), -1 for outlier (out-of-class).
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
        # Score: positive = inside acceptance, negative = outside
        return self.joint_threshold_ - self._fisher_stat(X)

    def _fisher_stat(self, X):
        """Fisher-combined statistic ``-2*(ln p_T2 + ln p_Q) ~ chi2(4)``.

        Shared by ``decision_function`` (margin) and ``p_joint`` (p-value) so the
        two never diverge.
        """
        X = np.asarray(X, dtype=np.float64)
        pc_scores = self.pca_.transform(X)

        t2 = self._compute_t2(pc_scores)
        q = self._compute_q_residuals_from_scores(X, pc_scores)

        # Per-axis p-values from fitted chi-squared distributions
        p_t2 = 1.0 - stats.chi2.cdf(t2, self.t2_dof_, loc=0, scale=self.t2_scale_)
        p_q = 1.0 - stats.chi2.cdf(q, self.q_dof_, loc=0, scale=self.q_scale_)

        # Clip to avoid log(0); 1e-300 is safely above float64 underflow
        p_t2 = np.clip(p_t2, 1e-300, 1.0)
        p_q = np.clip(p_q, 1e-300, 1.0)

        # Fisher's method: combine independent p-values → chi2(4)
        return -2.0 * (np.log(p_t2) + np.log(p_q))

    def p_joint(self, X):
        """Joint DD-SIMCA p-value ``P(chi2(4) >= fisher_stat)`` per sample.

        The calibrated joint p-value used by the multi-class SIMCA decision
        matrix (T-31). A sample is in-class iff ``p_joint >= alpha``, which is
        exactly equivalent to ``decision_function(X) >= 0``. Returns values in
        [0, 1].
        """
        return stats.chi2.sf(self._fisher_stat(X), 4)

    def score_samples(self, X):
        """Alias for decision_function (sklearn convention)."""
        return self.decision_function(X)

    def _compute_t2(self, scores):
        """Compute Hotelling T² using diagonal formulation.

        Uses T² = sum(t_a² / lambda_a) which is numerically stable
        and exact for PCA scores (which are orthogonal by construction).
        """
        lam = np.maximum(self.eigenvalues_, 1e-10)
        return np.sum(scores ** 2 / lam, axis=1)

    def _compute_q_residuals(self, X):
        """Compute Q-residuals (SPE) for samples."""
        X = np.asarray(X, dtype=np.float64)
        scores = self.pca_.transform(X)
        return self._compute_q_residuals_from_scores(X, scores)

    def _compute_q_residuals_from_scores(self, X, pc_scores):
        """Compute Q-residuals using pre-computed PCA scores (avoids duplicate transform)."""
        X = np.asarray(X, dtype=np.float64)
        X_reconstructed = pc_scores @ self.pca_.components_ + self.pca_.mean_
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
            random_state=42, n_jobs=1
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
            novelty=True, n_jobs=1
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
        return IsolationForest(**{**params, 'random_state': 42, 'n_jobs': 1})
    elif model_name == 'EllipticEnvelope':
        return EllipticEnvelope(**{**params, 'random_state': 42})
    elif model_name == 'LOF':
        return LocalOutlierFactor(**{**params, 'novelty': True, 'n_jobs': 1})
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
            {'n_estimators': 100, 'contamination': 0.01, 'max_features': 1.0, 'max_samples': 'auto'},
            {'n_estimators': 100, 'contamination': 0.05, 'max_features': 1.0, 'max_samples': 'auto'},
            {'n_estimators': 100, 'contamination': 0.05, 'max_features': 0.5, 'max_samples': 'auto'},
            {'n_estimators': 100, 'contamination': 0.1, 'max_features': 1.0, 'max_samples': 'auto'},
            {'n_estimators': 100, 'contamination': 0.1, 'max_features': 0.5, 'max_samples': 'auto'},
            {'n_estimators': 200, 'contamination': 0.05, 'max_features': 1.0, 'max_samples': 'auto'},
            {'n_estimators': 500, 'contamination': 0.05, 'max_features': 1.0, 'max_samples': 'auto'},
            {'n_estimators': 100, 'contamination': 0.05, 'max_features': 1.0, 'max_samples': 256},
            {'n_estimators': 100, 'contamination': 0.05, 'max_features': 1.0, 'max_samples': 512},
        ],
        'EllipticEnvelope': [
            {'contamination': 0.01},
            {'contamination': 0.05},
            {'contamination': 0.1},
            {'contamination': 0.05, 'support_fraction': 0.5},
            {'contamination': 0.05, 'support_fraction': 0.75},
        ],
        'LOF': [
            {'n_neighbors': 10, 'contamination': 0.05, 'metric': 'euclidean'},
            {'n_neighbors': 20, 'contamination': 0.05, 'metric': 'euclidean'},
            {'n_neighbors': 30, 'contamination': 0.05, 'metric': 'euclidean'},
            {'n_neighbors': 10, 'contamination': 0.01, 'metric': 'euclidean'},
            {'n_neighbors': 10, 'contamination': 0.1, 'metric': 'euclidean'},
            {'n_neighbors': 20, 'contamination': 0.01, 'metric': 'euclidean'},
            {'n_neighbors': 20, 'contamination': 0.1, 'metric': 'euclidean'},
            {'n_neighbors': 10, 'contamination': 0.05, 'metric': 'manhattan'},
            {'n_neighbors': 20, 'contamination': 0.05, 'metric': 'manhattan'},
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
        True labels: +1 for inlier (clean), -1 for outlier (out-of-class).
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


def run_one_class_cv(
    X: np.ndarray,
    y_oc: np.ndarray,
    model_name: str,
    params: dict,
    n_folds: int = 5,
    cv_strategy: str = 'kfold',
    cv_n_repeats: int = 5,
    random_state: int = 42,
    y_original: np.ndarray | None = None,
    compute_calibration: bool = True,
) -> dict:
    """Run cross-validation for a one-class model.

    Trains only on inlier samples, tests on inliers + all outliers.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (already preprocessed).
    y_oc : np.ndarray
        Binary labels: +1 for inliers, -1 for outliers.
    model_name : str
        Name of the model (e.g., 'PCA-SIMCA', 'OneClassSVM').
    params : dict
        Hyperparameters for the model.
    n_folds : int
        Number of CV folds (applied to inlier data only).
    cv_strategy : str
        Cross-validation strategy: 'kfold', 'repeated_kfold', or 'loo'.
    cv_n_repeats : int
        Number of repeats for 'repeated_kfold' strategy.
    random_state : int
        Random state for reproducibility.
    y_original : np.ndarray, optional
        Original string labels for per-contaminant sensitivity.
    compute_calibration : bool
        If False, skip the calibration block (fit on all inliers + per-contaminant
        stats). Set to False during Bayesian optimization trials where only
        mean_metrics['balanced_accuracy'] is used, saving ~17% overhead per trial.

    Returns
    -------
    dict
        fold_metrics, mean_metrics, cal_model, cal_scaler, cal_pca_reducer,
        cal_metrics, per_contaminant_sensitivity, skipped.
    """
    inlier_indices = np.where(y_oc == 1)[0]
    outlier_indices = np.where(y_oc == -1)[0]
    n_outliers = len(outlier_indices)

    # Track per-fold failure reasons so the caller (and ultimately the GUI
    # progress tab) can surface WHY a model was skipped instead of silently
    # returning +inf from the Bayesian objective.
    fold_errors: list[str] = []

    # Determine minimum inlier count based on CV strategy AND model.
    # PCA-SIMCA has a hard floor of 3 training samples (PCASIMCA.fit's
    # n_samples<3 guard), so under LOO with 3 inliers every training fold
    # has only 2 samples and every fold fails silently. Guard upfront.
    SIMCA_FIT_FLOOR = 3
    if cv_strategy == 'loo':
        base_min = 2  # Leave one out and still train
        if model_name == 'PCA-SIMCA':
            min_inliers = SIMCA_FIT_FLOOR + 1  # Training fold size = n_inliers - 1
        else:
            min_inliers = base_min
    elif cv_strategy in ('kfold', 'repeated_kfold'):
        # K-fold and Repeated K-fold: each training fold has ~(n_inliers * (k-1)/k)
        # samples. PCA-SIMCA needs SIMCA_FIT_FLOOR in the training fold.
        if model_name == 'PCA-SIMCA':
            min_inliers = max(n_folds, int(np.ceil(SIMCA_FIT_FLOOR * n_folds / (n_folds - 1))))
        else:
            min_inliers = n_folds
    else:
        # build_cv_splitter already rejects unknown strategies, but belt-and-
        # suspenders against future additions that bypass it.
        raise ValueError(f"Unknown cv_strategy: {cv_strategy!r}")

    if len(inlier_indices) < min_inliers:
        strategy_label = {
            'kfold': f'{n_folds}-fold CV',
            'repeated_kfold': f'repeated {n_folds}-fold CV',
            'loo': 'LOO CV',
        }
        if model_name == 'PCA-SIMCA':
            msg = (
                f"Too few inliers ({len(inlier_indices)}) for PCA-SIMCA under "
                f"{strategy_label.get(cv_strategy, cv_strategy)} — needs {min_inliers}+ "
                f"so every training fold has >= {SIMCA_FIT_FLOOR} samples"
            )
        else:
            msg = (
                f"Too few inliers ({len(inlier_indices)}) for "
                f"{strategy_label.get(cv_strategy, cv_strategy)}"
            )
        logger.warning(msg)
        return {'skipped': True, 'fold_metrics': [], 'skip_reason': msg, 'fold_errors': [msg]}

    kf = build_cv_splitter(
        strategy=cv_strategy,
        n_folds=n_folds,
        task_type='one_class',
        n_repeats=cv_n_repeats,
        random_state=random_state,
    )

    # --- Per-fold CV: collect predictions for pooled metric computation ---
    fold_metrics = []
    all_test_labels: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    # Original sample indices for each fold's test rows — enables per-sample
    # reduction under repeated CV (outliers appear in every fold, inliers
    # appear r times across repeats; without per-sample reduction the pooled
    # metrics come from correlated observations).
    all_test_idx: list[np.ndarray] = []

    for fold_i, (train_inlier_idx, test_inlier_idx) in enumerate(kf.split(inlier_indices)):
        train_idx = inlier_indices[train_inlier_idx]
        test_inlier = inlier_indices[test_inlier_idx]
        test_idx = np.concatenate([test_inlier, outlier_indices])
        test_labels = np.concatenate([
            np.ones(len(test_inlier), dtype=int),
            -np.ones(n_outliers, dtype=int),
        ])

        try:
            model = build_one_class_model(model_name, params)

            # Scale data for OCSVM, EllipticEnvelope, LOF
            if model_name in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])

                # EllipticEnvelope requires n_samples > n_features
                if model_name == 'EllipticEnvelope' and X_train.shape[1] > X_train.shape[0]:
                    n_pca = max(2, X_train.shape[0] // 2)
                    pca_reducer = PCA(n_components=n_pca)
                    X_train = pca_reducer.fit_transform(X_train)
                    X_test = pca_reducer.transform(X_test)
            else:
                X_train = X[train_idx]
                X_test = X[test_idx]

            model.fit(X_train)

            # Get decision scores and derive predictions to avoid duplicate calls.
            # For PCASIMCA, predict() calls decision_function() internally,
            # so calling both wastes a full PCA transform + scoring pass.
            scores = None
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                y_pred = np.where(scores >= 0, 1, -1)
            elif hasattr(model, 'score_samples'):
                scores = model.score_samples(X_test)
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)

            # Per-fold metrics retained for debugging/display
            fold_result = one_class_metrics(test_labels, y_pred, scores)
            fold_metrics.append(fold_result)

            # Collect for pooled computation
            all_test_labels.append(test_labels)
            all_y_pred.append(y_pred)
            all_test_idx.append(test_idx)
            if scores is not None:
                all_scores.append(scores)
        except (np.linalg.LinAlgError, ValueError) as e:
            err_msg = f"fold {fold_i} train_n={len(train_idx)}: {type(e).__name__}: {e}"
            fold_errors.append(err_msg)
            logger.warning("Fold failed for %s (%s)", model_name, err_msg)
            continue

    # Skip guard — check how many folds succeeded
    n_successful = len(all_y_pred)
    n_actual_splits = kf.get_n_splits(inlier_indices) if hasattr(kf, 'get_n_splits') else n_folds
    if n_successful < max(2, n_actual_splits // 2):
        reason = (
            f"{n_successful}/{n_actual_splits} folds succeeded; first error: "
            f"{fold_errors[0] if fold_errors else 'unknown'}"
        )
        return {
            'skipped': True,
            'fold_metrics': fold_metrics,
            'skip_reason': reason,
            'fold_errors': fold_errors,
        }

    # Compute metrics from POOLED predictions across all folds.
    # Matches chemometrics convention: ratio-of-sums, not averaging-of-ratios.
    # Under LOO, per-fold sensitivity degenerates to 0/1. Pooling fixes this.
    #
    # Under repeated CV, inliers appear r times and outliers appear k*r times
    # in the flat pool — pooled metrics would come from correlated observations
    # (same sample's r predictions come from related models). Reduce per-sample
    # by majority vote first, matching the regression/classification path's
    # cross_val_predict_pooled semantic. Plain K-Fold and LOO are unchanged.
    # TODO: plain K-Fold one-class has the same correlated-prediction
    # structure (outliers appear k times), but changing it would break
    # user-memorized numbers from existing models — rebaseline in a separate PR.
    if _is_repeated_cv(kf) and all_test_idx:
        pooled_idx_flat = np.concatenate(all_test_idx)
        pooled_labels_flat = np.concatenate(all_test_labels)
        pooled_preds_flat = np.concatenate(all_y_pred)
        # NOTE: pooled_scores is intentionally dropped under repeated CV.
        # decision_function / score_samples outputs from independently fitted
        # OCSVM/IsolationForest/EllipticEnvelope/LOF are NOT on a common scale
        # across folds — averaging them produces a meaningless ranking and
        # destroys AUC semantics. Labels-based metrics pool correctly via
        # per-sample majority vote below; AUC is computed separately as the
        # mean of per-fold AUCs (see pooled_scores=None handling in
        # one_class_metrics — it returns NaN for auc — then we override it
        # with the fold-mean).

        unique_samples = np.unique(pooled_idx_flat)
        pooled_labels = np.empty(len(unique_samples), dtype=pooled_labels_flat.dtype)
        pooled_preds = np.empty(len(unique_samples), dtype=pooled_preds_flat.dtype)
        for i, s in enumerate(unique_samples):
            sample_mask = pooled_idx_flat == s
            # Label is deterministic per sample (same row in y_oc), take first
            pooled_labels[i] = pooled_labels_flat[sample_mask][0]
            # Majority vote via np.unique + argmax. Deterministic but
            # tie-breaks TOWARD THE LOWER-SORTED LABEL (-1 = outlier), which
            # is the conservative default for contamination detection: when
            # the model is ambiguous about a sample, flag it as an outlier.
            # Callers who need a different tie-break policy should use an
            # odd cv_n_repeats to avoid exact ties.
            vals, counts = np.unique(pooled_preds_flat[sample_mask], return_counts=True)
            pooled_preds[i] = vals[np.argmax(counts)]
        # Pass None for scores — AUC overridden below from per-fold AUCs
        pooled_scores = None
    else:
        pooled_labels = np.concatenate(all_test_labels)
        pooled_preds = np.concatenate(all_y_pred)
        pooled_scores = np.concatenate(all_scores) if all_scores else None

    mean_metrics = one_class_metrics(pooled_labels, pooled_preds, pooled_scores)

    # Under repeated CV, override AUC with mean-of-fold-AUCs because
    # decision_function scores are not comparable across independently-fitted
    # folds. Per-fold AUCs are self-contained (scored within the same model).
    if _is_repeated_cv(kf) and fold_metrics:
        fold_aucs = [m.get('auc', np.nan) for m in fold_metrics]
        fold_aucs = [a for a in fold_aucs if a is not None and not np.isnan(a)]
        mean_metrics['auc'] = float(np.mean(fold_aucs)) if fold_aucs else float('nan')

    # NaN guard for balanced_accuracy in zero-outlier case
    if np.isnan(mean_metrics['balanced_accuracy']):
        mean_metrics['balanced_accuracy'] = mean_metrics['specificity']

    # --- Calibration: fit on ALL inliers, evaluate on all data ---
    cal_model = None
    cal_scaler = None
    cal_pca_reducer = None
    cal_metrics = {}
    per_contaminant = {}
    oc_score_stats = None

    if not compute_calibration:
        return {
            'fold_metrics': fold_metrics,
            'mean_metrics': mean_metrics,
            'cal_model': None,
            'cal_scaler': None,
            'cal_pca_reducer': None,
            'cal_metrics': {},
            'per_contaminant_sensitivity': {},
            'oc_score_stats': None,
            'skipped': False,
        }

    try:
        cal_model = build_one_class_model(model_name, params)
        if model_name in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
            cal_scaler = StandardScaler()
            X_inlier_scaled = cal_scaler.fit_transform(X[inlier_indices])
            X_all_scaled = cal_scaler.transform(X)

            if model_name == 'EllipticEnvelope' and X_inlier_scaled.shape[1] > X_inlier_scaled.shape[0]:
                n_pca = max(2, X_inlier_scaled.shape[0] // 2)
                cal_pca_reducer = PCA(n_components=n_pca)
                X_inlier_scaled = cal_pca_reducer.fit_transform(X_inlier_scaled)
                X_all_scaled = cal_pca_reducer.transform(X_all_scaled)

            cal_model.fit(X_inlier_scaled)
            if hasattr(cal_model, 'decision_function'):
                scores_cal = cal_model.decision_function(X_all_scaled)
                y_pred_cal = np.where(scores_cal >= 0, 1, -1)
            else:
                y_pred_cal = cal_model.predict(X_all_scaled)
                scores_cal = None
        else:
            cal_model.fit(X[inlier_indices])
            if hasattr(cal_model, 'decision_function'):
                scores_cal = cal_model.decision_function(X)
                y_pred_cal = np.where(scores_cal >= 0, 1, -1)
            else:
                y_pred_cal = cal_model.predict(X)
                scores_cal = None
        cal_metrics = one_class_metrics(y_oc, y_pred_cal, scores_cal)

        # Per-contaminant sensitivity (calibration-only, not CV)
        if n_outliers > 0 and y_original is not None:
            outlier_mask = y_oc == -1
            outlier_labels_unique = np.unique(y_original[outlier_mask])
            for lbl in outlier_labels_unique:
                lbl_mask = y_original == lbl
                if lbl_mask.sum() > 0:
                    lbl_preds = y_pred_cal[lbl_mask]
                    per_contaminant[str(lbl)] = float(np.mean(lbl_preds == -1))
        cal_metrics['per_contaminant'] = per_contaminant

        # Compute training inlier score statistics for stable AD thresholds at prediction time
        if scores_cal is not None:
            inlier_scores = scores_cal[inlier_indices]
            oc_score_stats = {
                'q10': float(np.percentile(inlier_scores, 10)),
                'q25': float(np.percentile(inlier_scores, 25)),
                'mean': float(np.mean(inlier_scores)),
                'std': float(np.std(inlier_scores)),
            }

    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        logger.warning("Calibration failed for %s: %s", model_name, e)
        cal_model = None  # Ensure unfitted model is not returned
        cal_scaler = None
        cal_pca_reducer = None
        cal_metrics = {
            k: np.nan
            for k in ['sensitivity', 'specificity', 'precision', 'f1',
                       'accuracy', 'balanced_accuracy', 'auc']
        }
        cal_metrics['per_contaminant'] = {}

    return {
        'fold_metrics': fold_metrics,
        'mean_metrics': mean_metrics,
        'cal_model': cal_model,
        'cal_scaler': cal_scaler,
        'cal_pca_reducer': cal_pca_reducer,
        'cal_metrics': cal_metrics,
        'per_contaminant_sensitivity': per_contaminant,
        'oc_score_stats': oc_score_stats,
        'skipped': False,
    }


# Canonical order for external validation columns on one-class results.
# Mirrors the cal/CV metric block (Sensitivity, Specificity, Precision, F1,
# Accuracy, BalancedAcc, AUC) so the Results tab shows a clean contiguous block.
_VAL_OC_COLUMNS = [
    'val_Sensitivity',
    'val_Specificity',
    'val_Precision',
    'val_F1',
    'val_Accuracy',
    'val_BalancedAcc',
    'val_AUC',
]

# Map from one_class_metrics() dict keys → val_* column names.
_VAL_OC_METRIC_KEY_TO_COLUMN = {
    'sensitivity': 'val_Sensitivity',
    'specificity': 'val_Specificity',
    'precision': 'val_Precision',
    'f1': 'val_F1',
    'accuracy': 'val_Accuracy',
    'balanced_accuracy': 'val_BalancedAcc',
    'auc': 'val_AUC',
}


# build_preprocessing_pipeline accepts these names; everything else is a
# display variant that has to be normalized first.
_BUILDABLE_PREPROCESS_NAMES = frozenset({
    'raw', 'snv', 'deriv', 'snv_deriv', 'deriv_snv',
})


def _normalize_preprocess_for_pipeline(name: str) -> str:
    """Map a result-row Preprocess display name to the base name accepted by
    build_preprocessing_pipeline.

    Grid search writes display names like ``snv_deriv1_w11`` (method + window)
    which the pipeline builder rejects with
    ``ValueError("Unknown preprocess: snv_deriv1_w11")``. The proper fix is
    for the producer to write a separate ``PreprocessBase`` column (see
    search.py:5136 / unified_bayesian.py:1995). This function is the
    defense-in-depth fallback for callers that don't.

    Strips ``_w<digits>`` suffixes and digits attached to ``deriv``. Returns
    the input unchanged if it is already a buildable base name."""
    if not name:
        return 'raw'
    base = name.strip()
    if base in _BUILDABLE_PREPROCESS_NAMES:
        return base
    # Drop trailing window suffix like '_w11'.
    base = re.sub(r'_w\d+$', '', base)
    # Drop digits glued to 'deriv' (deriv1 → deriv, snv_deriv2 → snv_deriv).
    base = re.sub(r'deriv\d+', 'deriv', base)
    if base in _BUILDABLE_PREPROCESS_NAMES:
        return base
    # Last resort: strip everything that isn't a known token. Anything not
    # matching falls back to 'raw' so the helper can still produce metrics
    # rather than silently dropping the row.
    return 'raw'


def compute_validation_metrics_for_top_one_class_models(
    df_results: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    inlier_label,
    wavelengths: np.ndarray,
    top_n: int = 700,
    progress_callback=None,
) -> pd.DataFrame:
    """Compute external validation metrics for the top N one-class models.

    Parallel to ``compute_validation_metrics_for_top_models`` in ``search.py``
    but for ``task_type='one_class'``. Adds these columns to ``df_results``
    in this order:

        val_Sensitivity, val_Specificity, val_Precision, val_F1,
        val_Accuracy, val_BalancedAcc, val_AUC

    The column set mirrors the cal/CV one-class metric block so Results tab
    users see validation parity with ``BalancedAcccv``, ``AUCcv``, etc.

    Parameters
    ----------
    df_results : pd.DataFrame
        One-class results (from grid search or Bayesian). Must include
        ``Model`` and ``Params`` columns, plus the preprocessing columns
        stored by the search paths (``PreprocessBase``/``Preprocess``,
        ``Deriv``, ``Window``, ``Poly``, optional ``baseline_method``,
        ``smoothing``, ``smoothing_window``, ``smoothing_polyorder``).
    X_train, y_train : np.ndarray
        Full-spectrum training spectra and raw target labels (strings/ints).
    X_val, y_val : np.ndarray
        Full-spectrum external validation spectra and raw target labels.
    inlier_label : Any
        Label value denoting the inlier/clean class. Compared as str to
        match the convention used elsewhere (Bayesian and grid paths).
    wavelengths : np.ndarray
        Wavelength values aligned with ``X_train``/``X_val`` columns. Used
        to map ``all_vars`` (wavelength values) back to column indices.
    top_n : int
        Number of top models (by Rank / CompositeScore) to compute
        validation for. Default 700 matches the GUI spinbox default.
    progress_callback : callable, optional
        Called with a status dict for each model processed. Signature:
        ``progress_callback({'stage': 'validation', 'current': i, 'total': n})``.

    Returns
    -------
    pd.DataFrame
        The same ``df_results`` with val_* columns added/populated.
    """
    from spectral_predict.preprocess import build_preprocessing_pipeline
    from sklearn.pipeline import Pipeline

    if df_results is None or len(df_results) == 0:
        return df_results

    # Drop training samples with NaN raw labels (safety net — upstream should
    # already filter, but matches classification helper behavior).
    try:
        train_nan_mask = pd.isna(y_train)
    except TypeError:
        train_nan_mask = np.zeros(len(y_train), dtype=bool)
    if np.any(train_nan_mask):
        n_dropped = int(np.sum(train_nan_mask))
        logger.info("[OC Validation] Dropping %d training sample(s) with NaN labels", n_dropped)
        X_train = X_train[~train_nan_mask]
        y_train = y_train[~train_nan_mask]

    try:
        val_nan_mask = pd.isna(y_val)
    except TypeError:
        val_nan_mask = np.zeros(len(y_val), dtype=bool)
    if np.any(val_nan_mask):
        n_dropped = int(np.sum(val_nan_mask))
        logger.info("[OC Validation] Dropping %d validation sample(s) with NaN labels", n_dropped)
        X_val = X_val[~val_nan_mask]
        y_val = y_val[~val_nan_mask]

    # Map raw labels to +1 (inlier) / -1 (outlier). Use str comparison to match
    # the convention used in unified_bayesian and the GUI (gui:25130, 34468).
    inlier_str = str(inlier_label)
    y_train_oc = np.where(np.asarray(y_train, dtype=str) == inlier_str, 1, -1)
    y_val_oc = np.where(np.asarray(y_val, dtype=str) == inlier_str, 1, -1)

    # Initialise val_* columns in canonical order (preserves column ordering).
    for col in _VAL_OC_COLUMNS:
        if col not in df_results.columns:
            df_results[col] = np.nan

    # Select top-N rows. Prefer Rank (both Bayesian and grid set it), then
    # CompositeScore (lower = better), then fall back to insertion order.
    n_to_process = min(top_n, len(df_results))
    if 'Rank' in df_results.columns:
        rank_numeric = pd.to_numeric(df_results['Rank'], errors='coerce')
        top_indices = rank_numeric.sort_values(kind='mergesort').index[:n_to_process]
    elif 'CompositeScore' in df_results.columns:
        df_results['CompositeScore'] = pd.to_numeric(df_results['CompositeScore'], errors='coerce')
        top_indices = df_results.nsmallest(n_to_process, 'CompositeScore').index
    elif 'BalancedAcccv' in df_results.columns:
        bacc = pd.to_numeric(df_results['BalancedAcccv'], errors='coerce')
        top_indices = bacc.sort_values(ascending=False, kind='mergesort').index[:n_to_process]
    else:
        top_indices = df_results.head(n_to_process).index

    logger.info("[OC Validation] Computing validation metrics for top %d one-class models", n_to_process)

    # Cache preprocessed (X_train_prep, X_val_prep) by config key so models
    # sharing preprocessing don't pay the transform cost repeatedly.
    preprocess_cache: dict = {}

    # Pre-compute wavelength → column-index mapping for the all_vars lookup.
    try:
        wl_to_idx = {float(wl): i for i, wl in enumerate(np.asarray(wavelengths))}
    except (TypeError, ValueError):
        wl_to_idx = {}

    for i, idx in enumerate(top_indices):
        # Report progress every 10 models (matches classification helper at
        # search.py:738). Without the 'message' key, _progress_callback() logs
        # an empty line per call — flooding the progress tab with blanks.
        if progress_callback is not None and (i + 1) % 10 == 0:
            try:
                progress_callback({
                    'stage': 'validation_metrics',
                    'message': f'  Computing one-class validation metrics ({i + 1}/{n_to_process})',
                    'current': i + 1,
                    'total': n_to_process,
                })
            except Exception:
                pass

        row = df_results.loc[idx]
        try:
            # === Parse preprocessing config ===
            # PreprocessBase is the clean pipeline name written by both
            # Bayesian (unified_bayesian.py:1995) and grid search
            # (search.py:5136 — added April 2026 to fix the silent NaN
            # val_* bug). Fall back to Preprocess (display name) and
            # normalize, so any caller that forgets PreprocessBase still
            # produces metrics instead of silently bailing.
            preprocess_name = row.get('PreprocessBase')
            if preprocess_name is None or (isinstance(preprocess_name, float) and pd.isna(preprocess_name)):
                preprocess_name = row.get('Preprocess', 'raw')
            if preprocess_name is None or (isinstance(preprocess_name, float) and pd.isna(preprocess_name)):
                preprocess_name = 'raw'

            def _maybe_int(v, default=0):
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return default
                    return int(v)
                except (TypeError, ValueError):
                    return default

            deriv = _maybe_int(row.get('Deriv'), 0)
            window = _maybe_int(row.get('Window'), 0)
            # When the row stores Poly=None (grid search defers to
            # SavgolDerivative's polyorder_map), fall back to that SAME map
            # rather than the older `min(2, window-1)` heuristic. The old
            # fallback gave poly=2 for deriv=2, but SavgolDerivative actually
            # uses poly=3 for deriv=2 — val_* metrics drifted off training
            # by one polynomial order. Source of truth lives in
            # preprocess.SAVGOL_POLYORDER_DEFAULTS.
            from .preprocess import SAVGOL_POLYORDER_DEFAULTS
            poly = _maybe_int(
                row.get('Poly'),
                SAVGOL_POLYORDER_DEFAULTS.get(deriv, deriv + 1),
            )

            baseline_method = row.get('baseline_method', None)
            if isinstance(baseline_method, float) and pd.isna(baseline_method):
                baseline_method = None
            smoothing_raw = row.get('smoothing', False)
            if isinstance(smoothing_raw, float) and pd.isna(smoothing_raw):
                smoothing = False
            else:
                smoothing = bool(smoothing_raw)
            smoothing_window = _maybe_int(row.get('smoothing_window'), 17)
            smoothing_polyorder = _maybe_int(row.get('smoothing_polyorder'), 2)
            # T-36: parse autoscale flag with the same robust handling as search.py
            # (handles bool, numpy.bool_, int 0/1, NaN-float, and string "true"/"false").
            autoscale_raw = row.get('Autoscale', False)
            if isinstance(autoscale_raw, float) and pd.isna(autoscale_raw):
                autoscale = False
            elif isinstance(autoscale_raw, str):
                autoscale = autoscale_raw.strip().lower() in ('true', '1', 'yes')
            else:
                autoscale = bool(autoscale_raw)

            # T-36 fix (post-merge review v2): mirror search.py's display-name
            # fallback so legacy .dasp files without an explicit Autoscale /
            # baseline column still rebuild the right pipeline from the suffixed
            # pipeline name (e.g. "als+sg0+snv+autoscale"). Must run BEFORE
            # _normalize_preprocess_for_pipeline below — that helper collapses
            # any '+'-containing name to 'raw' and would erase the suffixes
            # before this parser ever saw them. Explicit columns still win.
            if '+' in str(preprocess_name):
                parts = str(preprocess_name).split('+')
                core_parts = []
                for part in parts:
                    if part in ('als', 'polynomial', 'rubber_band', 'airpls', 'advanced'):
                        if baseline_method is None:
                            baseline_method = part
                    elif part == 'sg0':
                        smoothing = True
                    elif part == 'autoscale':
                        if not autoscale:
                            autoscale = True
                    else:
                        core_parts.append(part)
                preprocess_name = '_'.join(core_parts) if core_parts else 'raw'

            preprocess_name = _normalize_preprocess_for_pipeline(str(preprocess_name))

            # T-36 fix (post-merge review): persist baseline_params from the row so
            # non-default ALS/polynomial settings survive the validation roundtrip
            # rather than silently snapping back to defaults during rebuild.
            baseline_params_raw = row.get('baseline_params', None)
            baseline_params = None
            if baseline_params_raw is not None:
                if isinstance(baseline_params_raw, dict):
                    baseline_params = baseline_params_raw
                elif isinstance(baseline_params_raw, str) and baseline_params_raw.strip():
                    try:
                        import ast as _ast
                        parsed = _ast.literal_eval(baseline_params_raw)
                        if isinstance(parsed, dict):
                            baseline_params = parsed
                    except (ValueError, SyntaxError):
                        baseline_params = None

            cache_key = (
                preprocess_name, deriv, window, poly,
                baseline_method, smoothing, smoothing_window, smoothing_polyorder,
                autoscale,  # T-36: must vary key — autoscale changes preprocessing output
                # baseline_params is intentionally excluded — keyed by method only since
                # callers don't currently parameterize per-config; safe today.
            )

            # === Preprocess FULL spectrum (matching search.py pattern) ===
            if cache_key in preprocess_cache:
                X_train_prep, X_val_prep = preprocess_cache[cache_key]
            else:
                prep_steps = build_preprocessing_pipeline(
                    preprocess_name,
                    deriv=deriv if deriv > 0 else None,
                    window=window if window > 0 else None,
                    polyorder=poly if poly > 0 else None,
                    task_type='one_class',
                    baseline_method=baseline_method,
                    baseline_params=baseline_params,  # T-36 fix: was hardcoded None
                    smoothing=smoothing,
                    smoothing_window=smoothing_window,
                    smoothing_polyorder=smoothing_polyorder,
                    autoscale=autoscale,  # T-36
                )
                if prep_steps:
                    pipe = Pipeline(list(prep_steps))
                    X_train_prep = pipe.fit_transform(X_train)
                    X_val_prep = pipe.transform(X_val)
                else:
                    X_train_prep = np.asarray(X_train)
                    X_val_prep = np.asarray(X_val)
                preprocess_cache[cache_key] = (X_train_prep, X_val_prep)

            # === Wavelength subset (from all_vars only) ===
            # NOTE: do NOT fall back to 'selected_wavelengths' — Bayesian stores
            # only the first 50 wavelengths there (display-only), while 'all_vars'
            # has the full trained subset. Using selected_wavelengths would silently
            # validate on a truncated feature set.
            col_indices = None
            all_vars_str = row.get('all_vars', None)
            if isinstance(all_vars_str, float) and pd.isna(all_vars_str):
                all_vars_str = None
            if all_vars_str is None or not isinstance(all_vars_str, str) or not all_vars_str.strip() or all_vars_str == 'N/A':
                logger.warning(
                    "[OC Validation] Row %s: 'all_vars' missing or empty, skipping "
                    "(both OC grid and Bayesian always populate it — missing means row metadata is corrupt)",
                    idx,
                )
                continue
            try:
                model_wls = [float(w.strip()) for w in all_vars_str.split(',') if w.strip()]
                if wl_to_idx:
                    missing = [wl for wl in model_wls if wl not in wl_to_idx]
                    if missing:
                        logger.warning(
                            "[OC Validation] Row %s: %d/%d model wavelengths not found in wavelength map, skipping",
                            idx, len(missing), len(model_wls),
                        )
                        continue
                    col_indices = [wl_to_idx[wl] for wl in model_wls]
                else:
                    # No wavelength mapping — nearest-index fallback
                    all_wl_arr = np.asarray(wavelengths, dtype=float)
                    col_indices = [int(np.argmin(np.abs(all_wl_arr - wl))) for wl in model_wls]
                if not col_indices:
                    logger.warning("[OC Validation] Row %s: no valid wavelength indices resolved, skipping", idx)
                    continue
            except Exception as wl_err:
                logger.warning("[OC Validation] all_vars parse failed for row %s: %s, skipping", idx, wl_err)
                continue

            if col_indices is not None:
                max_col = X_train_prep.shape[1] - 1
                col_indices = [c for c in col_indices if 0 <= c <= max_col]
                if not col_indices:
                    logger.warning("[OC Validation] All wavelength indices out of bounds for row %s", idx)
                    continue
                X_train_final = X_train_prep[:, col_indices]
                X_val_final = X_val_prep[:, col_indices]
            else:
                X_train_final = X_train_prep
                X_val_final = X_val_prep

            # === Fit on inliers only (mirrors run_one_class_cv calibration) ===
            inlier_mask = y_train_oc == 1
            if not np.any(inlier_mask):
                logger.warning("[OC Validation] No inliers in training set for row %s", idx)
                continue
            X_inliers = X_train_final[inlier_mask]

            model_name = str(row.get('Model', ''))
            params_raw = row.get('Params', None)
            if params_raw is None:
                logger.warning("[OC Validation] Params missing for row %s, skipping", idx)
                continue
            if isinstance(params_raw, dict):
                params = params_raw
            elif isinstance(params_raw, str) and params_raw.strip():
                try:
                    params = ast.literal_eval(params_raw)
                except (ValueError, SyntaxError):
                    logger.warning("[OC Validation] Params parse failed for row %s, skipping", idx)
                    continue
            else:
                logger.warning("[OC Validation] Params empty for row %s, skipping", idx)
                continue

            # Scale + optional PCA for scale-sensitive models, matching the
            # pattern in run_one_class_cv's calibration block.
            scaler = None
            pca_reducer = None
            if model_name in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
                scaler = StandardScaler()
                X_inliers_scaled = scaler.fit_transform(X_inliers)
                X_val_scaled = scaler.transform(X_val_final)
                if model_name == 'EllipticEnvelope' and X_inliers_scaled.shape[1] > X_inliers_scaled.shape[0]:
                    n_pca = max(2, X_inliers_scaled.shape[0] // 2)
                    pca_reducer = PCA(n_components=n_pca)
                    X_inliers_scaled = pca_reducer.fit_transform(X_inliers_scaled)
                    X_val_scaled = pca_reducer.transform(X_val_scaled)
            else:
                X_inliers_scaled = X_inliers
                X_val_scaled = X_val_final

            model = build_one_class_model(model_name, params)
            model.fit(X_inliers_scaled)

            # === Predict on validation, prefer decision_function for AUC ===
            val_scores = None
            if hasattr(model, 'decision_function'):
                val_scores = model.decision_function(X_val_scaled)
                val_preds = np.where(val_scores >= 0, 1, -1)
            elif hasattr(model, 'score_samples'):
                val_scores = model.score_samples(X_val_scaled)
                val_preds = model.predict(X_val_scaled)
            else:
                val_preds = model.predict(X_val_scaled)

            metrics = one_class_metrics(y_val_oc, val_preds, val_scores)

            for metric_key, col_name in _VAL_OC_METRIC_KEY_TO_COLUMN.items():
                value = metrics.get(metric_key, np.nan)
                df_results.loc[idx, col_name] = float(value) if value is not None and not np.isnan(value) else np.nan

        except Exception as row_err:
            logger.warning("[OC Validation] Row %s failed: %s", idx, row_err)
            continue

    logger.info("[OC Validation] Completed validation metrics for %d models", n_to_process)

    # Reorder columns so val_* sit directly after the cv metric block.
    # This mirrors compute_validation_metrics_for_top_models in search.py
    # (lines 748-789) and matches user expectation that validation columns
    # appear next to calibration/CV columns rather than at the far right.
    # Canonical one-class layout (scoring.py:392-401):
    #   ...cal metrics... (Sensitivity..AUC)
    #   ...cv metrics...  (Sensitivitycv..AUCcv)
    #   val_* columns     (Sensitivity..AUC — injected here)
    #   top_vars, all_vars, CompositeScore, Rank
    cols = list(df_results.columns)
    present_val_cols = [c for c in _VAL_OC_COLUMNS if c in cols]
    if present_val_cols:
        for c in present_val_cols:
            cols.remove(c)

        # Insert after the last cv metric that exists in the frame. Prefer
        # AUCcv (final cv metric in canonical order), fall back to any cv
        # metric, then to Imbalance/SubsetTag, then to the end.
        cv_metric_order = [
            'AUCcv', 'BalancedAcccv', 'Accuracycv', 'F1cv',
            'Precisioncv', 'Specificitycv', 'Sensitivitycv',
        ]
        insert_after = None
        for cv_col in cv_metric_order:
            if cv_col in cols:
                insert_after = cv_col
                break
        if insert_after is None:
            for anchor in ('Imbalance', 'SubsetTag'):
                if anchor in cols:
                    insert_after = anchor
                    break

        if insert_after is not None:
            anchor_idx = cols.index(insert_after) + 1
        else:
            anchor_idx = len(cols)

        for offset, c in enumerate(present_val_cols):
            cols.insert(anchor_idx + offset, c)

        df_results = df_results[cols]

    return df_results


def compute_one_class_importances(
    X: np.ndarray,
    y_oc: np.ndarray,
    method: str = 'lightgbm',
    random_state: int = 42,
) -> np.ndarray:
    """Compute feature importances by treating one-class labels as binary classification.

    Uses class_weight='balanced' to handle typical inlier/outlier imbalance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y_oc : np.ndarray
        Binary labels: +1 for inliers, -1 for outliers.
    method : str
        'lightgbm' (default) or 'random_forest'.
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Feature importance array of shape (n_features,).
    """
    n_features = X.shape[1]
    n_outliers = np.sum(np.asarray(y_oc) == -1)

    # Edge case: too few outliers to learn meaningful importances
    if n_outliers < 2:
        logger.warning(
            "Fewer than 2 outliers (%d) — returning uniform importances.", n_outliers
        )
        return np.ones(n_features, dtype=np.float64) / n_features

    if method == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1,
        )
    else:
        # Default: LightGBM
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )

    # LightGBM binary classification expects 0/1 labels, not +1/-1
    y_for_fit = (np.asarray(y_oc) == -1).astype(int)  # 1=outlier, 0=inlier
    model.fit(X, y_for_fit)
    return model.feature_importances_
