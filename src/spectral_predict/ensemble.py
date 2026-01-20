"""
Intelligent ensemble methods for spectral prediction.

This module implements advanced ensemble strategies that go beyond simple averaging:
1. Region-based model analysis (identify where each model excels)
2. Weighted ensembles with regional specialization
3. Mixture of experts with regional gates
4. Traditional stacking for comparison
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
import warnings

from .preprocessing_wrapper import PreprocessorConfig


class SimpleAverageEnsemble(BaseEstimator, RegressorMixin):
    """
    Simple averaging ensemble.

    Averages predictions from multiple base models with optional
    per-model preprocessing.
    """

    def __init__(self, models, model_names=None, preprocessors=None, preprocessor_configs=None):
        """
        Parameters
        ----------
        models : list of fitted models
            The base models to ensemble
        model_names : list of str, optional
            Names for the models. If None, generates "Model_0", "Model_1", etc.
        preprocessors : list of preprocessors, optional
            Individual fitted preprocessor for each base model. If None, assumes
            models receive raw data directly.
        preprocessor_configs : list of PreprocessorConfig, optional
            Configuration objects for reconstructing preprocessing per model.
            Used when preprocessors aren't available directly.
        """
        self.models = models
        self.model_names = model_names if model_names else [f"Model_{i}" for i in range(len(models))]
        self.preprocessors = preprocessors
        self.preprocessor_configs = preprocessor_configs

    def _get_preprocessor(self, idx):
        """
        Get preprocessor for model at index idx.

        Returns fitted preprocessor object or PreprocessorConfig,
        preferring preprocessor_configs if available.
        """
        if self.preprocessor_configs and idx < len(self.preprocessor_configs):
            return self.preprocessor_configs[idx]
        elif self.preprocessors and idx < len(self.preprocessors):
            return self.preprocessors[idx]
        else:
            return None

    def fit(self, X, y):
        """Fit method (no-op for simple average since models are pre-fitted)."""
        return self

    def predict(self, X):
        """Predict using simple average of all models."""
        # Apply individual preprocessors if provided
        predictions = []
        for i, model in enumerate(self.models):
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            pred = model.predict(X_processed)
            # Ensure 1D predictions (PLSRegression returns (n,1))
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()
            predictions.append(pred)
        return np.mean(predictions, axis=0)


class RegionBasedAnalyzer:
    """
    Analyze model performance across different regions of the target space.

    This identifies which models are "specialists" (excel in specific ranges)
    vs "generalists" (perform consistently across all ranges).
    """

    def __init__(self, n_regions=5, method='quantile'):
        """
        Parameters
        ----------
        n_regions : int, default=5
            Number of regions to divide the target space into
        method : str, default='quantile'
            How to divide regions: 'quantile' or 'uniform'
        """
        self.n_regions = n_regions
        self.method = method
        self.region_boundaries = None

    def fit(self, y_true):
        """Define region boundaries based on true values."""
        if self.method == 'quantile':
            # Divide into quantiles (equal number of samples per region)
            self.region_boundaries = np.percentile(
                y_true,
                np.linspace(0, 100, self.n_regions + 1)
            )
        else:  # uniform
            # Divide into uniform ranges
            self.region_boundaries = np.linspace(
                y_true.min(),
                y_true.max(),
                self.n_regions + 1
            )
        return self

    def assign_regions(self, y_values):
        """Assign each value to a region (0 to n_regions-1)."""
        regions = np.digitize(y_values, self.region_boundaries[1:-1])
        return regions

    def analyze_model_performance(self, y_true, y_pred, metric='rmse'):
        """
        Compute performance metrics for each region.

        Returns
        -------
        dict with keys:
            'overall': float - overall metric
            'by_region': array of shape (n_regions,) - metric per region
            'region_sizes': array of shape (n_regions,) - samples per region
            'specialization_score': float - how specialized vs generalist
        """
        regions = self.assign_regions(y_true)

        # Compute overall metric
        if metric == 'rmse':
            overall_metric = np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == 'mae':
            overall_metric = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            overall_metric = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Compute per-region metrics
        region_metrics = np.zeros(self.n_regions)
        region_sizes = np.zeros(self.n_regions, dtype=int)

        for region_idx in range(self.n_regions):
            mask = regions == region_idx
            region_sizes[region_idx] = np.sum(mask)

            if region_sizes[region_idx] == 0:
                region_metrics[region_idx] = np.nan
                continue

            y_true_region = y_true[mask]
            y_pred_region = y_pred[mask]

            if metric == 'rmse':
                region_metrics[region_idx] = np.sqrt(np.mean((y_true_region - y_pred_region) ** 2))
            elif metric == 'mae':
                region_metrics[region_idx] = np.mean(np.abs(y_true_region - y_pred_region))
            elif metric == 'r2':
                ss_res = np.sum((y_true_region - y_pred_region) ** 2)
                ss_tot = np.sum((y_true_region - np.mean(y_true_region)) ** 2)
                region_metrics[region_idx] = 1 - (ss_res / (ss_tot + 1e-10))

        # Compute specialization score
        # High variance in regional performance = specialist
        # Low variance = generalist
        valid_metrics = region_metrics[~np.isnan(region_metrics)]
        if len(valid_metrics) > 1:
            specialization_score = np.std(valid_metrics) / (np.mean(np.abs(valid_metrics)) + 1e-10)
        else:
            specialization_score = 0.0

        return {
            'overall': overall_metric,
            'by_region': region_metrics,
            'region_sizes': region_sizes,
            'specialization_score': specialization_score,
            'region_boundaries': self.region_boundaries
        }


class RegionAwareWeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Weighted ensemble that assigns different weights to models
    based on their performance in different regions.

    Instead of a single weight per model, each model gets a weight function
    that varies based on the predicted value.
    """

    def __init__(self, models, model_names=None, n_regions=5, cv=5, preprocessors=None, preprocessor_configs=None, y_percentiles=None):
        """
        Parameters
        ----------
        models : list of fitted models
            The base models to ensemble
        model_names : list of str, optional
            Names for the models
        n_regions : int, default=5
            Number of regions to analyze
        cv : int, default=5
            Cross-validation folds for computing weights
        preprocessors : list of preprocessors, optional
            Individual preprocessor for each base model. If None, assumes
            models receive raw data directly.
        preprocessor_configs : list of PreprocessorConfig, optional
            Configuration objects for reconstructing preprocessing per model.
            Used when preprocessors aren't available directly.
        y_percentiles : array-like, optional
            Pre-computed TRUE Y percentile values. If provided, these boundaries
            will be used for region assignment instead of computing from predictions.
            This ensures consistent region assignment between selection (based on
            TRUE Y) and routing (based on these boundaries).
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.n_regions = n_regions
        self.cv = cv
        self.preprocessors = preprocessors
        self.preprocessor_configs = preprocessor_configs
        self.y_percentiles = y_percentiles
        self.regional_weights_ = None
        self.analyzer_ = RegionBasedAnalyzer(n_regions=n_regions)

    @property
    def weights_(self):
        """
        Alias for regional_weights_ so that save/load helpers
        can treat all ensembles consistently.
        """
        return self.regional_weights_

    @weights_.setter
    def weights_(self, value):
        self.regional_weights_ = value

    def _get_preprocessor(self, idx):
        """
        Get preprocessor for model at index idx.

        Returns fitted preprocessor object or PreprocessorConfig,
        preferring preprocessor_configs if available.
        """
        if self.preprocessor_configs and idx < len(self.preprocessor_configs):
            return self.preprocessor_configs[idx]
        elif self.preprocessors and idx < len(self.preprocessors):
            return self.preprocessors[idx]
        else:
            return None

    def fit(self, X, y):
        """
        Fit the ensemble by computing regional performance weights using
        out-of-fold predictions to prevent data leakage.

        Uses KFold cross-validation to generate OOF predictions for computing
        regional weights. This ensures the ensemble weights are computed on
        predictions the models haven't seen during training, providing
        realistic estimates of model performance.

        The original fitted models are preserved for use in predict().
        """
        # Ensure y is numpy array
        y = np.asarray(y).ravel()
        n_samples = len(y)

        # Set up cross-validation using self.cv parameter
        n_splits = min(self.cv, n_samples)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Generate out-of-fold predictions for all models
        predictions = np.zeros((len(self.models), n_samples))

        for i, model in enumerate(self.models):
            try:
                # Get preprocessor for this model
                preprocessor = self._get_preprocessor(i)

                # Generate OOF predictions using CV
                oof_pred = np.zeros(n_samples)
                for train_idx, val_idx in kf.split(X):
                    # Split data
                    if hasattr(X, 'iloc'):  # DataFrame
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    else:  # numpy array
                        X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]

                    # Apply preprocessing
                    if preprocessor is not None:
                        X_train_proc = preprocessor.transform(X_train)
                        X_val_proc = preprocessor.transform(X_val)
                    else:
                        X_train_proc, X_val_proc = X_train, X_val

                    # Clone and fit model on this fold
                    fold_model = clone(model)
                    fold_model.fit(X_train_proc, y_train)

                    # Predict on validation fold
                    pred = fold_model.predict(X_val_proc)
                    oof_pred[val_idx] = np.asarray(pred).ravel()

                predictions[i] = oof_pred
            except Exception as e:
                # More informative error message to help diagnose sklearn.clone() issues
                warnings.warn(
                    f"Model {self.model_names[i]} failed during OOF prediction:\n"
                    f"  Type: {type(model).__name__}\n"
                    f"  Error: {e}\n"
                    f"  This model will be excluded from weight calculation.\n"
                    f"  If this is a wrapper class, ensure it inherits from sklearn.base.BaseEstimator\n"
                    f"  and implements get_params()/set_params() for sklearn.clone() compatibility."
                )
                # Use NaN to mark as failed - allows excluding from weight calculation
                predictions[i] = np.full(n_samples, np.nan)

        # Define region boundaries
        if self.y_percentiles is not None:
            # Use fixed TRUE Y boundaries (ensures selection and routing use same boundaries)
            self.analyzer_.region_boundaries = np.array(self.y_percentiles)
        else:
            # Fallback to prediction-based boundaries (legacy behavior)
            avg_pred = np.mean(predictions, axis=0)
            self.analyzer_.fit(avg_pred)

        # Compute regional performance for each model using OOF predictions
        # When using TRUE Y boundaries, performance is computed on TRUE Y regions
        regional_errors = np.zeros((len(self.models), self.n_regions))

        for model_idx in range(len(self.models)):
            analysis = self.analyzer_.analyze_model_performance(
                y, predictions[model_idx], metric='rmse'
            )
            regional_errors[model_idx] = analysis['by_region']

        # Convert errors to weights (inverse error)
        # Lower error = higher weight
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regional_weights = 1.0 / (regional_errors + 1e-6)

            # Normalize weights per region (sum to 1)
            regional_weights = regional_weights / (
                np.sum(regional_weights, axis=0, keepdims=True) + 1e-10
            )

        # Handle NaN regions
        regional_weights = np.nan_to_num(regional_weights, nan=1.0/len(self.models))

        self.regional_weights_ = regional_weights  # (n_models, n_regions)

        return self

    def predict(self, X):
        """Predict using region-aware weighted averaging."""
        # Get predictions from all models, applying individual preprocessors
        predictions = []
        for i, model in enumerate(self.models):
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            pred = model.predict(X_processed)
            # Ensure 1D predictions (PLSRegression returns (n,1))
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()
            predictions.append(pred)
        predictions = np.array(predictions)  # (n_models, n_samples)

        # For each prediction, determine which region it falls in
        # Use the average prediction to determine region (chicken-and-egg problem)
        avg_pred = np.mean(predictions, axis=0)
        regions = self.analyzer_.assign_regions(avg_pred)

        # Apply regional weights
        weighted_pred = np.zeros(len(X))
        for sample_idx in range(len(X)):
            region_idx = regions[sample_idx]
            weights = self.regional_weights_[:, region_idx]
            weighted_pred[sample_idx] = np.sum(
                predictions[:, sample_idx] * weights
            )

        return weighted_pred

    def get_model_profiles(self):
        """
        Get information about each model's regional strengths.

        Returns
        -------
        dict with model names as keys, containing:
            - 'weights': regional weights
            - 'specialization': whether model is specialist or generalist
            - 'best_regions': regions where this model excels
        """
        profiles = {}

        for model_idx, model_name in enumerate(self.model_names):
            weights = self.regional_weights_[model_idx]

            # Find regions where this model has highest weight
            relative_weight = weights / np.mean(weights)
            best_regions = np.where(relative_weight > 1.2)[0]  # 20% above average

            # Determine if specialist or generalist
            weight_variance = np.std(weights)
            is_specialist = weight_variance > 0.1

            profiles[model_name] = {
                'weights': weights,
                'specialization': 'specialist' if is_specialist else 'generalist',
                'best_regions': best_regions,
                'weight_variance': weight_variance
            }

        return profiles


class MixtureOfExpertsEnsemble(BaseEstimator, RegressorMixin):
    """
    Mixture of Experts ensemble with regional gating.

    Instead of weighting predictions, this selects the best model
    for each region and uses only that model's prediction.
    Optionally uses soft gating (weighted combination).
    """

    def __init__(self, models, model_names=None, n_regions=5, soft_gating=True, cv=5, preprocessors=None, preprocessor_configs=None, y_percentiles=None):
        """
        Parameters
        ----------
        models : list of fitted models
        model_names : list of str, optional
        n_regions : int, default=5
        soft_gating : bool, default=True
            If True, use weighted combination. If False, use hard selection.
        cv : int, default=5
            Cross-validation folds for computing expert assignments using OOF predictions.
        preprocessors : list of preprocessors, optional
            Individual preprocessor for each base model. If None, assumes
            models receive raw data directly.
        preprocessor_configs : list of PreprocessorConfig, optional
            Configuration objects for reconstructing preprocessing per model.
            Used when preprocessors aren't available directly.
        y_percentiles : array-like, optional
            Pre-computed TRUE Y percentile values. If provided, these boundaries
            will be used for region assignment instead of computing from predictions.
            This ensures consistent region assignment between selection (based on
            TRUE Y) and routing (based on these boundaries).
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.n_regions = n_regions
        self.soft_gating = soft_gating
        self.cv = cv
        self.preprocessors = preprocessors
        self.preprocessor_configs = preprocessor_configs
        self.y_percentiles = y_percentiles
        self.expert_assignment_ = None  # Which model is best for each region
        self.expert_weights_ = None  # Soft weights if soft_gating=True
        self.analyzer_ = RegionBasedAnalyzer(n_regions=n_regions)

    @property
    def weights_(self):
        """
        Alias for expert_weights_ so that save/load helpers
        can access ensemble weights in a uniform way.
        """
        return self.expert_weights_

    @weights_.setter
    def weights_(self, value):
        self.expert_weights_ = value

    def _get_preprocessor(self, idx):
        """
        Get preprocessor for model at index idx.

        Returns fitted preprocessor object or PreprocessorConfig,
        preferring preprocessor_configs if available.
        """
        if self.preprocessor_configs and idx < len(self.preprocessor_configs):
            return self.preprocessor_configs[idx]
        elif self.preprocessors and idx < len(self.preprocessors):
            return self.preprocessors[idx]
        else:
            return None

    def fit(self, X, y):
        """
        Fit by determining which expert handles which region using
        out-of-fold predictions to prevent data leakage.

        Uses KFold cross-validation to generate OOF predictions for determining
        expert assignments. This ensures expert selection is based on realistic
        model performance estimates.

        The original fitted models are preserved for use in predict().
        """
        # Ensure y is numpy array
        y = np.asarray(y).ravel()
        n_samples = len(y)

        # Set up cross-validation using self.cv parameter
        n_splits = min(self.cv, n_samples)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Generate out-of-fold predictions for all models
        predictions = np.zeros((len(self.models), n_samples))

        for i, model in enumerate(self.models):
            try:
                # Get preprocessor for this model
                preprocessor = self._get_preprocessor(i)

                # Generate OOF predictions using CV
                oof_pred = np.zeros(n_samples)
                for train_idx, val_idx in kf.split(X):
                    # Split data
                    if hasattr(X, 'iloc'):  # DataFrame
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    else:  # numpy array
                        X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]

                    # Apply preprocessing
                    if preprocessor is not None:
                        X_train_proc = preprocessor.transform(X_train)
                        X_val_proc = preprocessor.transform(X_val)
                    else:
                        X_train_proc, X_val_proc = X_train, X_val

                    # Clone and fit model on this fold
                    fold_model = clone(model)
                    fold_model.fit(X_train_proc, y_train)

                    # Predict on validation fold
                    pred = fold_model.predict(X_val_proc)
                    oof_pred[val_idx] = np.asarray(pred).ravel()

                predictions[i] = oof_pred
            except Exception as e:
                # More informative error message to help diagnose sklearn.clone() issues
                warnings.warn(
                    f"Model {self.model_names[i]} failed during OOF prediction:\n"
                    f"  Type: {type(model).__name__}\n"
                    f"  Error: {e}\n"
                    f"  This model will be excluded from weight calculation.\n"
                    f"  If this is a wrapper class, ensure it inherits from sklearn.base.BaseEstimator\n"
                    f"  and implements get_params()/set_params() for sklearn.clone() compatibility."
                )
                # Use NaN to mark as failed - allows excluding from weight calculation
                predictions[i] = np.full(n_samples, np.nan)

        # Define region boundaries
        if self.y_percentiles is not None:
            # Use fixed TRUE Y boundaries (ensures selection and routing use same boundaries)
            self.analyzer_.region_boundaries = np.array(self.y_percentiles)
            # Assign regions based on TRUE Y values (not predictions)
            regions = self.analyzer_.assign_regions(y)
        else:
            # Fallback to prediction-based boundaries (legacy behavior)
            avg_pred = np.mean(predictions, axis=0)
            self.analyzer_.fit(avg_pred)
            # Assign regions based on average OOF predictions
            regions = self.analyzer_.assign_regions(avg_pred)

        # For each region, find the best model using OOF predictions
        self.expert_assignment_ = np.zeros(self.n_regions, dtype=int)
        self.expert_weights_ = np.zeros((len(self.models), self.n_regions))

        for region_idx in range(self.n_regions):
            mask = regions == region_idx

            if np.sum(mask) == 0:
                # No samples in this region, use first model
                self.expert_assignment_[region_idx] = 0
                self.expert_weights_[0, region_idx] = 1.0
                continue

            y_region = y[mask]

            # Compute error for each model in this region using OOF predictions
            region_errors = []
            for model_idx in range(len(self.models)):
                pred_region = predictions[model_idx][mask]
                error = np.sqrt(np.mean((y_region - pred_region) ** 2))
                region_errors.append(error)

            region_errors = np.array(region_errors)

            # Hard assignment: best model
            self.expert_assignment_[region_idx] = np.argmin(region_errors)

            # Soft weights: inverse error
            if self.soft_gating:
                weights = 1.0 / (region_errors + 1e-6)
                weights = weights / np.sum(weights)
                self.expert_weights_[:, region_idx] = weights
            else:
                self.expert_weights_[self.expert_assignment_[region_idx], region_idx] = 1.0

        return self

    def predict(self, X):
        """Predict using mixture of experts."""
        # Get predictions from all models, applying individual preprocessors
        predictions = []
        for i, model in enumerate(self.models):
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            pred = model.predict(X_processed)
            # Ensure 1D predictions (PLSRegression returns (n,1))
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()
            predictions.append(pred)
        predictions = np.array(predictions)

        # Determine regions for predictions
        avg_pred = np.mean(predictions, axis=0)
        regions = self.analyzer_.assign_regions(avg_pred)

        # Apply expert gating
        final_pred = np.zeros(len(X))
        for sample_idx in range(len(X)):
            region_idx = regions[sample_idx]
            weights = self.expert_weights_[:, region_idx]
            final_pred[sample_idx] = np.sum(
                predictions[:, sample_idx] * weights
            )

        return final_pred

    def get_expert_assignments(self):
        """Return which expert handles which region."""
        assignments = {}
        for region_idx in range(self.n_regions):
            expert_idx = self.expert_assignment_[region_idx]
            assignments[f"Region {region_idx}"] = {
                'primary_expert': self.model_names[expert_idx],
                'weights': dict(zip(self.model_names, self.expert_weights_[:, region_idx]))
            }
        return assignments


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Traditional stacking ensemble with optional region-aware features.

    Trains a meta-model on the predictions of base models.
    Optionally includes region information as additional features.
    """

    def __init__(self, models, model_names=None, meta_model=None,
                 region_aware=True, n_regions=5, cv=5, preprocessors=None, preprocessor_configs=None, y_percentiles=None):
        """
        Parameters
        ----------
        models : list of fitted models
        model_names : list of str, optional
        meta_model : estimator, optional
            Meta-learner (default: Ridge regression)
        region_aware : bool, default=True
            Include region features in meta-model
        n_regions : int, default=5
        cv : int, default=5
            Cross-validation folds
        preprocessors : list of preprocessors, optional
            Individual preprocessor for each base model. If None, assumes
            models receive raw data directly.
        preprocessor_configs : list of PreprocessorConfig, optional
            Configuration objects for reconstructing preprocessing per model.
            Used when preprocessors aren't available directly.
        y_percentiles : array-like, optional
            Pre-computed TRUE Y percentile values. If provided, these boundaries
            will be used for region assignment instead of computing from predictions.
            This ensures consistent region assignment between selection (based on
            TRUE Y) and routing (based on these boundaries).
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.region_aware = region_aware
        self.n_regions = n_regions
        self.cv = cv
        self.preprocessors = preprocessors
        self.preprocessor_configs = preprocessor_configs
        self.y_percentiles = y_percentiles
        self.analyzer_ = RegionBasedAnalyzer(n_regions=n_regions) if region_aware else None

    @property
    def meta_model_(self):
        """
        Alias for meta_model so that save/load helpers can
        persist and restore the stacking meta-learner.
        """
        return self.meta_model

    @meta_model_.setter
    def meta_model_(self, value):
        self.meta_model = value

    def _get_preprocessor(self, idx):
        """
        Get preprocessor for model at index idx.

        Returns fitted preprocessor object or PreprocessorConfig,
        preferring preprocessor_configs if available.
        """
        if self.preprocessor_configs and idx < len(self.preprocessor_configs):
            return self.preprocessor_configs[idx]
        elif self.preprocessors and idx < len(self.preprocessors):
            return self.preprocessors[idx]
        else:
            return None

    def fit(self, X, y):
        """
        Fit the stacking ensemble using out-of-fold predictions to prevent
        data leakage in the meta-model.

        Uses KFold cross-validation to generate OOF predictions from base models.
        The meta-model is trained on these OOF predictions, ensuring it learns
        to combine base model predictions on unseen data.

        The original fitted base models are preserved for use in predict().
        """
        # Ensure y is numpy array
        y = np.asarray(y).ravel()
        n_samples = len(y)

        # Set up cross-validation using self.cv parameter
        n_splits = min(self.cv, n_samples)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Generate out-of-fold predictions for all models
        meta_features = np.zeros((n_samples, len(self.models)))

        for i, model in enumerate(self.models):
            try:
                # Get preprocessor for this model
                preprocessor = self._get_preprocessor(i)

                # Generate OOF predictions using CV
                oof_pred = np.zeros(n_samples)
                for train_idx, val_idx in kf.split(X):
                    # Split data
                    if hasattr(X, 'iloc'):  # DataFrame
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    else:  # numpy array
                        X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]

                    # Apply preprocessing
                    if preprocessor is not None:
                        X_train_proc = preprocessor.transform(X_train)
                        X_val_proc = preprocessor.transform(X_val)
                    else:
                        X_train_proc, X_val_proc = X_train, X_val

                    # Clone and fit model on this fold
                    fold_model = clone(model)
                    fold_model.fit(X_train_proc, y_train)

                    # Predict on validation fold
                    pred = fold_model.predict(X_val_proc)
                    oof_pred[val_idx] = np.asarray(pred).ravel()

                meta_features[:, i] = oof_pred
            except Exception as e:
                # More informative error message to help diagnose sklearn.clone() issues
                warnings.warn(
                    f"Model {self.model_names[i]} failed during OOF prediction:\n"
                    f"  Type: {type(model).__name__}\n"
                    f"  Error: {e}\n"
                    f"  This model will be excluded from weight calculation.\n"
                    f"  If this is a wrapper class, ensure it inherits from sklearn.base.BaseEstimator\n"
                    f"  and implements get_params()/set_params() for sklearn.clone() compatibility."
                )
                # Use NaN to mark as failed - allows excluding from weight calculation
                meta_features[:, i] = np.full(n_samples, np.nan)

        # Add region-aware features if enabled
        if self.region_aware:
            # Define region boundaries
            avg_pred = np.mean(meta_features, axis=1)
            if self.y_percentiles is not None:
                # Use fixed TRUE Y boundaries (ensures selection and routing use same boundaries)
                self.analyzer_.region_boundaries = np.array(self.y_percentiles)
                # Assign regions based on TRUE Y values (not predictions)
                regions = self.analyzer_.assign_regions(y)
            else:
                # Fallback to prediction-based boundaries (legacy behavior)
                self.analyzer_.fit(avg_pred)
                # Assign regions based on average OOF predictions
                regions = self.analyzer_.assign_regions(avg_pred)

            # One-hot encode regions
            region_features = np.zeros((n_samples, self.n_regions))
            for i, region in enumerate(regions):
                region_features[i, region] = 1.0

            # Also add the predicted value itself
            pred_value_feature = avg_pred.reshape(-1, 1)

            meta_features = np.hstack([
                meta_features,
                region_features,
                pred_value_feature
            ])

        # Fit meta-model on OOF predictions (prevents leakage)
        self.meta_model.fit(meta_features, y)

        return self

    def predict(self, X):
        """Predict using stacking ensemble."""
        # Get predictions from base models, applying individual preprocessors
        predictions = []
        for i, model in enumerate(self.models):
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            pred = model.predict(X_processed)
            # Ensure 1D predictions (PLSRegression returns (n,1))
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()
            predictions.append(pred)
        meta_features = np.column_stack(predictions)

        # Add region features if enabled
        if self.region_aware:
            avg_pred = np.mean(meta_features, axis=1)
            regions = self.analyzer_.assign_regions(avg_pred)

            region_features = np.zeros((len(X), self.n_regions))
            for i, region in enumerate(regions):
                region_features[i, region] = 1.0

            pred_value_feature = avg_pred.reshape(-1, 1)

            meta_features = np.hstack([
                meta_features,
                region_features,
                pred_value_feature
            ])

        return self.meta_model.predict(meta_features)

    def get_specialist_info(self):
        """
        Get information about the stacked models.

        Returns
        -------
        dict mapping 'Stacked' -> list of model names
        """
        return {'Stacked': self.model_names.copy() if self.model_names else []}


class RegionSpecialistEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble where each region uses only its specialist model(s).

    For a Q1 sample, only Q1's top model(s) contribute to the prediction.
    For a Q2 sample, only Q2's top model(s) contribute.
    And so on for each quartile.

    This is similar to MixtureOfExpertsEnsemble with hard gating, but:
    - Experts are assigned per quartile based on pre-computed regional rankings
    - Each quartile has its own set of specialist models (not learned from scratch)

    Parameters
    ----------
    models_per_region : dict
        Dictionary mapping region names ('Q1', 'Q2', 'Q3', 'Q4') to lists of
        (model, preprocessor_config) tuples. Each list contains the specialist
        models for that region.
    region_boundaries : array-like
        Percentile boundaries for assigning samples to regions.
        Should be [0, 25, 50, 75, 100] for quartiles.
    model_names_per_region : dict, optional
        Dictionary mapping region names to lists of model names.
    """

    def __init__(self, models_per_region, region_boundaries, model_names_per_region=None,
                 y_percentiles=None):
        """
        Parameters
        ----------
        models_per_region : dict
            Dictionary mapping region names ('Q1', 'Q2', 'Q3', 'Q4') to lists of
            (model, preprocessor_config) tuples.
        region_boundaries : array-like
            Percentile boundaries for assigning samples to regions.
            Should be [0, 25, 50, 75, 100] for quartiles.
        model_names_per_region : dict, optional
            Dictionary mapping region names to lists of model names.
        y_percentiles : array-like, optional
            Pre-computed TRUE Y percentile values. If provided, these boundaries
            will be used for region assignment instead of computing from predictions.
            This ensures consistent region assignment between selection (based on
            TRUE Y) and routing (based on these boundaries).
        """
        self.models_per_region = models_per_region
        self.region_boundaries = np.array(region_boundaries)
        self.model_names_per_region = model_names_per_region or {}
        self.y_percentiles_ = np.array(y_percentiles) if y_percentiles is not None else None

    def fit(self, X, y):
        """
        Fit by computing region boundaries (if not already provided).

        If y_percentiles was provided at construction time, those boundaries
        are used directly (ensuring selection and routing use the same TRUE Y
        quartile boundaries).

        Otherwise, computes boundaries from initial predictions to ensure
        consistency with predict()'s region assignment.
        """
        # If y_percentiles already provided, skip computing
        if self.y_percentiles_ is not None:
            return self

        # Compute initial predictions (same as predict() does)
        all_preds = []
        for region, model_list in self.models_per_region.items():
            for model, preproc_config in model_list:
                try:
                    if preproc_config is not None:
                        X_processed = preproc_config.transform(X)
                    else:
                        X_processed = X
                    pred = model.predict(X_processed)
                    # FIX D: Ensure 1D predictions (PLSRegression returns (n,1))
                    if hasattr(pred, 'ravel'):
                        pred = pred.ravel()
                    all_preds.append(pred)
                except Exception:
                    pass  # Skip models that fail

        if not all_preds:
            # Fallback to y-based boundaries if all models fail
            self.y_percentiles_ = np.percentile(y, self.region_boundaries)
            return self

        # Compute initial prediction (mean of all model predictions)
        initial_pred = np.mean(all_preds, axis=0)

        # Compute boundaries from PREDICTIONS, not y
        # This ensures consistent region assignment with predict()
        self.y_percentiles_ = np.percentile(initial_pred, self.region_boundaries)
        return self

    def _assign_region(self, y_pred):
        """
        Assign a predicted Y-value to a region (Q1, Q2, Q3, Q4).

        Uses the percentile boundaries from training data.
        """
        if self.y_percentiles_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # digitize returns 1-based index, we want 0-based
        region_idx = np.digitize(y_pred, self.y_percentiles_[1:-1])
        region_names = ['Q1', 'Q2', 'Q3', 'Q4']
        # Clip to valid range
        region_idx = np.clip(region_idx, 0, 3)
        return region_names[region_idx]

    def predict(self, X):
        """
        Predict using region-specialist models.

        For each sample:
        1. Get initial prediction (simple average of all models) to determine region
        2. Use only that region's specialist models for final prediction
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        # First pass: get initial predictions from all models to determine regions
        all_preds = []
        all_models = []
        for region, model_list in self.models_per_region.items():
            for model, preproc_config in model_list:
                all_models.append((model, preproc_config))

        # Deduplicate models (same model might be specialist in multiple regions)
        seen = set()
        unique_models = []
        for model, preproc_config in all_models:
            model_id = id(model)
            if model_id not in seen:
                seen.add(model_id)
                unique_models.append((model, preproc_config))

        # Get predictions from unique models
        for model, preproc_config in unique_models:
            if preproc_config is not None:
                X_processed = preproc_config.transform(X)
            else:
                X_processed = X
            pred = model.predict(X_processed)
            # FIX D: Ensure 1D predictions (PLSRegression returns (n,1))
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()
            all_preds.append(pred)

        if not all_preds:
            return np.zeros(n_samples)

        # Use average prediction to determine region
        initial_pred = np.mean(all_preds, axis=0)

        # Second pass: for each sample, use only the region's specialist models
        for i in range(n_samples):
            region = self._assign_region(initial_pred[i])
            region_models = self.models_per_region.get(region, [])

            if not region_models:
                # Fallback to initial prediction if no models for this region
                predictions[i] = initial_pred[i]
                continue

            # Get predictions from region's specialist models
            region_preds = []
            for model, preproc_config in region_models:
                if preproc_config is not None:
                    X_i = preproc_config.transform(X[i:i+1])
                else:
                    X_i = X[i:i+1]
                pred = model.predict(X_i)
                # FIX D: Ensure 1D predictions (PLSRegression returns (n,1))
                if hasattr(pred, 'ravel'):
                    pred = pred.ravel()
                region_preds.append(pred[0])

            # Average the specialist predictions (equal weight within region)
            predictions[i] = np.mean(region_preds)

        return predictions

    def get_specialist_info(self):
        """
        Get information about which models are specialists for each region.

        Returns
        -------
        dict mapping region -> list of model names
        """
        info = {}
        for region, names in self.model_names_per_region.items():
            info[region] = names if names else []
        return info


class ClassSpecialistEnsemble(BaseEstimator, ClassifierMixin):
    """
    Classification ensemble where each class uses its specialist model(s).

    For predicting class A, models that specialize in class A get higher weight.
    Specialists are determined by per-class F1 scores from the search results.

    Uses soft voting with class-specific weighting: each model's vote is weighted
    by its F1 score for the class it predicts.

    Parameters
    ----------
    models_per_class : dict
        Dictionary mapping class labels to lists of (model, preprocessor_config) tuples.
        Each list contains the specialist models for that class.
    model_names_per_class : dict, optional
        Dictionary mapping class labels to lists of model names.
    classes : array-like
        Array of class labels (will be set during fit if not provided).
    """

    def __init__(self, models_per_class, model_names_per_class=None, classes=None):
        self.models_per_class = models_per_class
        self.model_names_per_class = model_names_per_class or {}
        self.classes_ = np.array(classes) if classes is not None else None

    def fit(self, X, y):
        """
        Fit by setting class labels from training data.

        The base models are already fitted. We just need to store the class labels.
        """
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using class-specialist models.

        For each sample:
        1. Get probability predictions from all specialist models
        2. Weight each model's prediction by its specialization for each class
        3. Normalize to get final probabilities
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Collect all unique models
        all_models = []
        for class_label, model_list in self.models_per_class.items():
            for model, preproc_config in model_list:
                all_models.append((model, preproc_config, class_label))

        # Deduplicate models (same model might be specialist in multiple classes)
        seen = set()
        unique_models = []
        for model, preproc_config, class_label in all_models:
            model_id = id(model)
            if model_id not in seen:
                seen.add(model_id)
                unique_models.append((model, preproc_config))

        if not unique_models:
            # Return uniform probabilities if no models
            return np.ones((n_samples, n_classes)) / n_classes

        # Get predictions from all unique models
        all_proba = []
        for model, preproc_config in unique_models:
            if preproc_config is not None:
                X_processed = preproc_config.transform(X)
            else:
                X_processed = X

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_processed)
            else:
                # Fall back to hard predictions converted to one-hot
                preds = model.predict(X_processed)
                proba = np.zeros((len(preds), n_classes))
                for i, p in enumerate(preds):
                    class_idx = np.where(self.classes_ == p)[0]
                    if len(class_idx) > 0:
                        proba[i, class_idx[0]] = 1.0
                    else:
                        proba[i] = 1.0 / n_classes
            all_proba.append(proba)

        # Weight models by their class specialization
        # For now, use equal weighting (simpler approach)
        # More sophisticated: weight by class-specific F1 scores stored during creation
        weighted_proba = np.mean(all_proba, axis=0)

        # Normalize to ensure valid probabilities
        row_sums = weighted_proba.sum(axis=1, keepdims=True)
        weighted_proba = np.divide(
            weighted_proba,
            row_sums,
            out=np.ones_like(weighted_proba) / n_classes,
            where=row_sums > 0
        )

        return weighted_proba

    def predict(self, X):
        """Predict class labels using class-specialist models."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_specialist_info(self):
        """
        Get information about which models are specialists for each class.

        Returns
        -------
        dict mapping class_label -> list of model names
        """
        info = {}
        for class_label, names in self.model_names_per_class.items():
            info[class_label] = names if names else []
        return info


def extract_preprocessor_config(row, all_wavelengths):
    """
    Extract preprocessing configuration from a results DataFrame row.

    This function parses the Preprocess, Deriv, Window, Poly, and all_vars
    columns to create a PreprocessorConfig object that can reconstruct
    the preprocessing pipeline.

    Parameters
    ----------
    row : pd.Series or dict
        Row from results DataFrame containing:
        - 'Preprocess': preprocessing method name
        - 'Deriv': derivative order (0, 1, or 2)
        - 'Window': Savitzky-Golay window size
        - 'Poly': Savitzky-Golay polynomial order
        - 'all_vars': comma-separated wavelengths or 'N/A'
    all_wavelengths : list or array
        Full wavelength array from the dataset

    Returns
    -------
    PreprocessorConfig
        Configuration object for preprocessing reconstruction
    """
    # Extract preprocessing parameters
    preprocess_name = row.get('Preprocess', 'raw')
    deriv = row.get('Deriv', 0)
    window = row.get('Window', 15)
    polyorder = row.get('Poly', 2)

    # Parse wavelength subset from all_vars column
    all_vars_str = row.get('all_vars', 'N/A')
    wavelengths = None

    if all_vars_str and all_vars_str != 'N/A' and not pd.isna(all_vars_str):
        try:
            # Parse comma-separated wavelengths
            wavelengths = [float(w.strip()) for w in str(all_vars_str).split(',')]
        except (ValueError, AttributeError):
            # If parsing fails, use all wavelengths
            wavelengths = None

    # Create PreprocessorConfig
    config = PreprocessorConfig(
        preprocess_name=preprocess_name,
        deriv=deriv,
        window=window,
        polyorder=polyorder,
        wavelengths=wavelengths,
        all_wavelengths=list(all_wavelengths) if wavelengths is not None else None
    )

    return config


def create_ensemble(models, model_names, X, y, ensemble_type='region_weighted',
                    n_regions=5, **kwargs):
    """
    Factory function to create and fit an ensemble.

    Parameters
    ----------
    models : list of fitted models
    model_names : list of str
    X : array-like
        Training features
    y : array-like
        Training targets
    ensemble_type : str
        Type of ensemble:
        - 'simple_average': Simple averaging
        - 'region_weighted': Region-aware weighted ensemble
        - 'mixture_experts': Mixture of experts with regional gates
        - 'stacking': Traditional stacking
        - 'region_stacking': Region-aware stacking
    n_regions : int, default=5
    **kwargs : additional arguments for specific ensemble types
        Common kwargs:
        - y_percentiles : array-like, optional
            Pre-computed TRUE Y percentile values for region boundaries.
            Used by region_weighted, mixture_experts, and region_stacking.
        - preprocessors : list of preprocessors, optional
        - preprocessor_configs : list of PreprocessorConfig, optional
        - cv : int, cross-validation folds
        - soft_gating : bool, for mixture_experts
        - meta_model : estimator, for stacking

    Returns
    -------
    Fitted ensemble model
    """
    if ensemble_type == 'simple_average':
        # Simple averaging ensemble (baseline)
        ensemble = SimpleAverageEnsemble(
            models,
            model_names=model_names,
            preprocessors=kwargs.get('preprocessors'),
            preprocessor_configs=kwargs.get('preprocessor_configs')
        )
        ensemble.fit(X, y)

    elif ensemble_type == 'region_weighted':
        ensemble = RegionAwareWeightedEnsemble(
            models, model_names, n_regions=n_regions, **kwargs
        )
        ensemble.fit(X, y)

    elif ensemble_type == 'mixture_experts':
        ensemble = MixtureOfExpertsEnsemble(
            models, model_names, n_regions=n_regions, **kwargs
        )
        ensemble.fit(X, y)

    elif ensemble_type == 'stacking':
        ensemble = StackingEnsemble(
            models, model_names, region_aware=False, **kwargs
        )
        ensemble.fit(X, y)

    elif ensemble_type == 'region_stacking':
        ensemble = StackingEnsemble(
            models, model_names, region_aware=True, n_regions=n_regions, **kwargs
        )
        ensemble.fit(X, y)

    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    return ensemble


def compute_regional_rankings(results_df, top_n=10):
    """
    Rank models by performance in each Y-value region (quartile).

    Uses the 'regional_rmse' column computed during grid search, which contains
    a dict with keys 'Q1', 'Q2', 'Q3', 'Q4' for each quartile's RMSE.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame containing 'regional_rmse' column
    top_n : int, default=10
        Number of top models to identify per region

    Returns
    -------
    dict with keys:
        'rankings': dict mapping region -> list of (row_idx, rmse, rank) tuples
        'best_region': dict mapping row_idx -> (best_region, rank, other_regions_count)
        'top_models': set of row indices that are in top N of any region
        'y_quartiles': list of quartile boundaries if available
    """
    regions = ['Q1', 'Q2', 'Q3', 'Q4']
    rankings = {r: [] for r in regions}
    best_region = {}
    top_models = set()

    # Check if regional_rmse column exists
    if 'regional_rmse' not in results_df.columns:
        return {
            'rankings': rankings,
            'best_region': best_region,
            'top_models': top_models,
            'y_quartiles': None
        }

    # Extract y_quartiles from first row if available
    y_quartiles = None
    if 'y_quartiles' in results_df.columns:
        first_quartiles = results_df['y_quartiles'].iloc[0]
        if first_quartiles is not None:
            y_quartiles = first_quartiles

    # Collect RMSE values for each region
    for region in regions:
        region_data = []
        for idx, row in results_df.iterrows():
            regional_rmse = row.get('regional_rmse')
            if regional_rmse is not None and isinstance(regional_rmse, dict):
                rmse = regional_rmse.get(region)
                if rmse is not None and not np.isnan(rmse):
                    region_data.append((idx, rmse))

        # Sort by RMSE (ascending - lower is better)
        region_data.sort(key=lambda x: x[1])

        # Assign ranks
        for rank, (idx, rmse) in enumerate(region_data, start=1):
            rankings[region].append((idx, rmse, rank))
            if rank <= top_n:
                top_models.add(idx)

    # Determine best region for each model in top_models
    for idx in top_models:
        model_ranks = {}
        for region in regions:
            for (row_idx, rmse, rank) in rankings[region]:
                if row_idx == idx and rank <= top_n:
                    model_ranks[region] = rank
                    break

        if model_ranks:
            # Find region with best (lowest) rank
            best_r = min(model_ranks.keys(), key=lambda r: model_ranks[r])
            best_rank = model_ranks[best_r]
            other_count = len(model_ranks) - 1
            best_region[idx] = (best_r, best_rank, other_count)

    return {
        'rankings': rankings,
        'best_region': best_region,
        'top_models': top_models,
        'y_quartiles': y_quartiles
    }


def compute_class_rankings(results_df, top_n=10):
    """
    Rank models by performance on each class (for classification tasks).

    Uses the 'per_class_metrics' column computed during grid search, which contains
    a dict with class labels as keys and metrics (F1, Precision, Recall, Support) as values.

    This is analogous to compute_regional_rankings for regression, but uses F1 score
    (higher is better) instead of RMSE (lower is better).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame containing 'per_class_metrics' column
    top_n : int, default=10
        Number of top models to identify per class

    Returns
    -------
    dict with keys:
        'rankings': dict mapping class_label -> list of (row_idx, f1, rank) tuples
        'best_class': dict mapping row_idx -> (best_class, rank, other_classes_count)
        'top_models': set of row indices that are in top N of any class
        'class_labels': list of class labels
    """
    # Check if per_class_metrics column exists
    if 'per_class_metrics' not in results_df.columns:
        return {
            'rankings': {},
            'best_class': {},
            'top_models': set(),
            'class_labels': None
        }

    # Get class labels from first row with per_class_metrics
    class_labels = None
    for idx, row in results_df.iterrows():
        per_class = row.get('per_class_metrics')
        if per_class is not None and isinstance(per_class, dict) and len(per_class) > 0:
            class_labels = sorted(per_class.keys())
            break

    if not class_labels:
        return {
            'rankings': {},
            'best_class': {},
            'top_models': set(),
            'class_labels': None
        }

    rankings = {c: [] for c in class_labels}
    best_class = {}
    top_models = set()

    # Collect F1 values for each class
    for class_label in class_labels:
        class_data = []
        for idx, row in results_df.iterrows():
            per_class = row.get('per_class_metrics')
            if per_class is not None and isinstance(per_class, dict):
                metrics = per_class.get(str(class_label))
                if metrics is not None:
                    f1 = metrics.get('F1')
                    if f1 is not None and not np.isnan(f1):
                        class_data.append((idx, f1))

        # Sort by F1 (descending - higher is better, unlike RMSE)
        class_data.sort(key=lambda x: x[1], reverse=True)

        # Assign ranks
        for rank, (idx, f1) in enumerate(class_data, start=1):
            rankings[class_label].append((idx, f1, rank))
            if rank <= top_n:
                top_models.add(idx)

    # Determine best class for each model in top_models
    for idx in top_models:
        model_ranks = {}
        for class_label in class_labels:
            for (row_idx, f1, rank) in rankings[class_label]:
                if row_idx == idx and rank <= top_n:
                    model_ranks[class_label] = rank
                    break

        if model_ranks:
            # Find class with best (lowest) rank
            best_c = min(model_ranks.keys(), key=lambda c: model_ranks[c])
            best_rank = model_ranks[best_c]
            other_count = len(model_ranks) - 1
            best_class[idx] = (best_c, best_rank, other_count)

    return {
        'rankings': rankings,
        'best_class': best_class,
        'top_models': top_models,
        'class_labels': class_labels
    }


def select_top_models_per_region(results_df, top_n, task_type, reconstruct_func, X_train, y_train, all_wavelengths):
    """
    Select top N models for each quartile (regression) or class (classification).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with model performance metrics
    top_n : int
        Number of top models to select per region/class
    task_type : str
        'regression' or 'classification'
    reconstruct_func : callable
        Function to reconstruct a fitted model from a results row.
        Signature: reconstruct_func(row, X_train, y_train) -> (fitted_model, model_name)
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    all_wavelengths : array-like
        Full wavelength array from the dataset

    Returns
    -------
    dict with keys:
        'models_per_region': dict mapping region/class -> list of (model, preproc_config) tuples
        'model_names_per_region': dict mapping region/class -> list of model names
        'unique_model_count': int - total number of unique models selected
    """
    if task_type == 'regression':
        # Use regional rankings (quartiles)
        rankings_result = compute_regional_rankings(results_df, top_n=top_n)
        rankings = rankings_result['rankings']
        regions = ['Q1', 'Q2', 'Q3', 'Q4']
    else:
        # Use class rankings
        rankings_result = compute_class_rankings(results_df, top_n=top_n)
        rankings = rankings_result['rankings']
        regions = rankings_result.get('class_labels') or []

    if not rankings or not regions:
        return {
            'models_per_region': {},
            'model_names_per_region': {},
            'unique_model_count': 0
        }

    models_per_region = {}
    model_names_per_region = {}
    all_selected_indices = set()

    for region in regions:
        region_rankings = rankings.get(region, [])
        models_per_region[region] = []
        model_names_per_region[region] = []

        for idx, metric_val, rank in region_rankings[:top_n]:
            if idx in results_df.index:
                all_selected_indices.add(idx)
                row = results_df.loc[idx]

                # Reconstruct the model
                try:
                    fitted_model, model_name = reconstruct_func(row, X_train, y_train)
                    # Model already handles preprocessing internally
                    models_per_region[region].append((fitted_model, None))
                    model_names_per_region[region].append(model_name)
                except Exception as e:
                    warnings.warn(f"Failed to reconstruct model for region {region}: {e}")

    return {
        'models_per_region': models_per_region,
        'model_names_per_region': model_names_per_region,
        'unique_model_count': len(all_selected_indices)
    }


def select_top_models_quartile_flat(results_df: pd.DataFrame,
                                     top_n_per_quartile: int,
                                     task_type: str = 'regression') -> dict:
    """
    Select top N models from each quartile, returning a flat unique list.

    This function enables an alternative model selection strategy for ensembles:
    instead of selecting overall top N models by CompositeScore, it selects
    models that specialize in different Y-value ranges (quartiles).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with model performance metrics and regional_rmse column
    top_n_per_quartile : int
        Number of top models to select from each quartile (Q1, Q2, Q3, Q4)
    task_type : str, default='regression'
        Task type - quartile selection only applies to regression

    Returns
    -------
    dict with keys:
        'indices': list of unique row indices for selected models
        'model_count': number of unique models (may be less than 4*top_n due to overlap)
        'coverage': dict mapping quartile -> number of models selected from it
        'quartile_details': dict mapping quartile -> list of (idx, rmse, rank) tuples

    Notes
    -----
    A model may rank highly in multiple quartiles (e.g., a good overall model),
    so the returned indices are deduplicated. This means the total unique model
    count may be less than top_n_per_quartile * 4.

    Example
    -------
    >>> selection = select_top_models_quartile_flat(results_df, top_n_per_quartile=5)
    >>> print(f"Selected {selection['model_count']} unique models")
    >>> print(f"Coverage: {selection['coverage']}")
    # Could show: Selected 15 unique models (some overlap between quartiles)
    # Coverage: {'Q1': 5, 'Q2': 5, 'Q3': 5, 'Q4': 5}
    """
    if task_type != 'regression':
        # For classification, quartile-based selection doesn't apply
        # Return empty result - caller should use overall top N instead
        return {
            'indices': [],
            'model_count': 0,
            'coverage': {},
            'quartile_details': {}
        }

    # Use existing compute_regional_rankings function
    rankings_result = compute_regional_rankings(results_df, top_n=top_n_per_quartile)
    rankings = rankings_result['rankings']

    if not rankings:
        return {
            'indices': [],
            'model_count': 0,
            'coverage': {},
            'quartile_details': {}
        }

    # Collect unique indices from all quartiles
    all_indices = set()
    coverage = {}
    quartile_details = {}

    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        quartile_rankings = rankings.get(quartile, [])
        top_n_for_quartile = quartile_rankings[:top_n_per_quartile]

        quartile_details[quartile] = top_n_for_quartile
        coverage[quartile] = len(top_n_for_quartile)

        for idx, rmse, rank in top_n_for_quartile:
            all_indices.add(idx)

    return {
        'indices': list(all_indices),
        'model_count': len(all_indices),
        'coverage': coverage,
        'quartile_details': quartile_details
    }


def create_auto_ensembles(results_df, X_train, y_train, task_type, reconstruct_func, all_wavelengths):
    """
    Create Ensemble-Top1, Top2, Top3 with region-specialist weighting.

    Generates 3 automatic ensembles at the end of every search run:
    - Ensemble-Top1: Top 1 model from each quartile/class
    - Ensemble-Top2: Top 2 models from each quartile/class
    - Ensemble-Top3: Top 3 models from each quartile/class

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with model performance metrics and regional/class rankings
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    task_type : str
        'regression' or 'classification'
    reconstruct_func : callable
        Function to reconstruct a fitted model from a results row.
        Signature: reconstruct_func(row, X_train, y_train) -> (fitted_model, model_name)
    all_wavelengths : array-like
        Full wavelength array from the dataset

    Returns
    -------
    dict with keys 'Ensemble-Top1', 'Ensemble-Top2', 'Ensemble-Top3', each containing:
        'ensemble': fitted ensemble model (RegionSpecialistEnsemble or ClassSpecialistEnsemble)
        'base_models': list of model names in the ensemble
        'metrics': dict with 'r2', 'rmse' (regression) or 'accuracy', 'f1' (classification)
        'n_models': int - number of unique base models
        'specialist_info': dict mapping region/class -> list of model names
    """
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
    from sklearn.model_selection import KFold

    auto_ensembles = {}
    region_boundaries = [0, 25, 50, 75, 100]  # Quartile percentiles
    n_cv_folds = min(5, len(y_train))

    for top_n, ensemble_name in [(1, 'Ensemble-Top1'), (2, 'Ensemble-Top2'), (3, 'Ensemble-Top3')]:
        # Select top models for each region
        selection = select_top_models_per_region(
            results_df, top_n, task_type, reconstruct_func, X_train, y_train, all_wavelengths
        )

        models_per_region = selection['models_per_region']
        model_names_per_region = selection['model_names_per_region']
        unique_model_count = selection['unique_model_count']

        # Skip if too few unique models selected
        if unique_model_count < 2:
            continue

        # Create appropriate ensemble based on task type
        if task_type == 'regression':
            # Compute TRUE Y quartile boundaries
            # These boundaries are used for BOTH:
            # 1. Selection: Models were ranked by TRUE Y regional performance
            # 2. Routing: Samples are assigned to regions using these same boundaries
            y_percentiles = np.percentile(y_train, region_boundaries)

            # Use RegionSpecialistEnsemble with TRUE Y boundaries
            # Only the region's specialist models contribute to each sample's prediction
            ensemble = RegionSpecialistEnsemble(
                models_per_region=models_per_region,
                region_boundaries=region_boundaries,
                model_names_per_region=model_names_per_region,
                y_percentiles=y_percentiles  # Pass TRUE Y boundaries for consistent routing
            )
            ensemble.fit(X_train, y_train)

            # Use cross-validation to get realistic metrics (comparable to RMSECV)
            # Without CV, metrics on training data would be inflated (high RPD, low RMSE)
            y_std = np.std(y_train)  # Store for proper RPD calculation

            if n_cv_folds >= 2:
                kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
                cv_predictions = np.full(len(y_train), np.nan)

                for train_idx, val_idx in kf.split(X_train):
                    # Use iloc for DataFrame row indexing, direct indexing for numpy arrays
                    if hasattr(X_train, 'iloc'):
                        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    else:
                        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train = y_train[train_idx]

                    # Reconstruct models for this CV fold
                    cv_selection = select_top_models_per_region(
                        results_df, top_n, task_type, reconstruct_func,
                        X_cv_train, y_cv_train, all_wavelengths
                    )

                    if cv_selection['unique_model_count'] >= 2:
                        cv_y_percentiles = np.percentile(y_cv_train, region_boundaries)
                        cv_ensemble = RegionSpecialistEnsemble(
                            models_per_region=cv_selection['models_per_region'],
                            region_boundaries=region_boundaries,
                            model_names_per_region=cv_selection['model_names_per_region'],
                            y_percentiles=cv_y_percentiles
                        )
                        cv_ensemble.fit(X_cv_train, y_cv_train)
                        cv_predictions[val_idx] = cv_ensemble.predict(X_cv_val)
                    else:
                        # Fallback: use main ensemble for this fold
                        cv_predictions[val_idx] = ensemble.predict(X_cv_val)

                # Calculate CV metrics
                r2 = r2_score(y_train, cv_predictions)
                rmse = np.sqrt(mean_squared_error(y_train, cv_predictions))
            else:
                # Fallback for very small datasets
                y_pred = ensemble.predict(X_train)
                r2 = r2_score(y_train, y_pred)
                rmse = np.sqrt(mean_squared_error(y_train, y_pred))

            metrics = {'r2': r2, 'rmse': rmse, 'y_std': y_std}

            # === DIAGNOSTIC: Check if reconstructed models match original performance ===
            # Set SPECTRAL_PREDICT_DEBUG=1 environment variable to enable
            import os
            if os.environ.get('SPECTRAL_PREDICT_DEBUG', '').strip() in ('1', 'true', 'True'):
                print(f"\n=== DIAGNOSTIC: {ensemble_name} ===")
                for region, models in models_per_region.items():
                    for i, (model, preproc) in enumerate(models):
                        try:
                            if preproc is not None:
                                X_proc = preproc.transform(X_train)
                            else:
                                X_proc = X_train
                            y_pred_model = model.predict(X_proc)
                            if hasattr(y_pred_model, 'ravel'):
                                y_pred_model = y_pred_model.ravel()
                            r2_model = r2_score(y_train, y_pred_model)
                            rmse_model = np.sqrt(mean_squared_error(y_train, y_pred_model))
                            model_name = model_names_per_region.get(region, ['unknown'])[i] if i < len(model_names_per_region.get(region, [])) else 'unknown'
                            print(f"  {region} model {i} ({model_name}): R2={r2_model:.4f}, RMSE={rmse_model:.4f}")
                        except Exception as e:
                            print(f"  {region} model {i}: FAILED - {e}")

                print(f"  ENSEMBLE {ensemble_name} (RegionSpecialist): R2={r2:.4f}, RMSE={rmse:.4f}")
                print(f"  Expected: Individual model R2s should match their original R2 from results_df")
                print(f"  If individual R2s are LOW, reconstruction is broken (hyperparams/preprocessing)")
                print(f"  RegionSpecialist uses TRUE Y boundaries for routing")
            # === END DIAGNOSTIC ===
        else:
            # Classification
            classes = np.unique(y_train)
            ensemble = ClassSpecialistEnsemble(
                models_per_class=models_per_region,  # Same structure, different semantics
                model_names_per_class=model_names_per_region,
                classes=classes
            )
            ensemble.fit(X_train, y_train)

            # Use cross-validation to get realistic metrics
            if n_cv_folds >= 2:
                kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
                cv_predictions = np.full(len(y_train), np.nan, dtype=object)

                for train_idx, val_idx in kf.split(X_train):
                    # Use iloc for DataFrame row indexing, direct indexing for numpy arrays
                    if hasattr(X_train, 'iloc'):
                        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    else:
                        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train = y_train[train_idx]

                    # Reconstruct models for this CV fold
                    cv_selection = select_top_models_per_region(
                        results_df, top_n, task_type, reconstruct_func,
                        X_cv_train, y_cv_train, all_wavelengths
                    )

                    if cv_selection['unique_model_count'] >= 2:
                        cv_classes = np.unique(y_cv_train)
                        cv_ensemble = ClassSpecialistEnsemble(
                            models_per_class=cv_selection['models_per_region'],
                            model_names_per_class=cv_selection['model_names_per_region'],
                            classes=cv_classes
                        )
                        cv_ensemble.fit(X_cv_train, y_cv_train)
                        cv_predictions[val_idx] = cv_ensemble.predict(X_cv_val)
                    else:
                        # Fallback: use main ensemble for this fold
                        cv_predictions[val_idx] = ensemble.predict(X_cv_val)

                # Calculate CV metrics
                accuracy = accuracy_score(y_train, cv_predictions)
                f1 = f1_score(y_train, cv_predictions, average='weighted')
            else:
                # Fallback for very small datasets
                y_pred = ensemble.predict(X_train)
                accuracy = accuracy_score(y_train, y_pred)
                f1 = f1_score(y_train, y_pred, average='weighted')

            metrics = {'accuracy': accuracy, 'f1': f1}

        # Collect all unique model names
        all_model_names = []
        seen_names = set()
        for region, names in model_names_per_region.items():
            for name in names:
                if name not in seen_names:
                    seen_names.add(name)
                    all_model_names.append(name)

        auto_ensembles[ensemble_name] = {
            'ensemble': ensemble,
            'base_models': all_model_names,
            'metrics': metrics,
            'n_models': unique_model_count,
            'specialist_info': ensemble.get_specialist_info()
        }

    return auto_ensembles
