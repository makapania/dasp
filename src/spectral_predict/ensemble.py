"""
Intelligent ensemble methods for spectral prediction.

This module implements advanced ensemble strategies that go beyond simple averaging:
1. Region-based model analysis (identify where each model excels)
2. Weighted ensembles with regional specialization
3. Mixture of experts with regional gates
4. Traditional stacking for comparison
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict
import warnings


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

    def __init__(self, models, model_names=None, n_regions=5, cv=5, preprocessors=None):
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
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.n_regions = n_regions
        self.cv = cv
        self.preprocessors = preprocessors
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

    def fit(self, X, y):
        """
        Fit the ensemble by computing regional performance weights.

        Uses cross-validation to avoid overfitting on the same data
        used to compute weights.
        """
        self.analyzer_.fit(y)

        # Get cross-validated predictions for each model
        cv_predictions = []
        for model in self.models:
            try:
                cv_pred = cross_val_predict(model, X, y, cv=self.cv)
                cv_predictions.append(cv_pred)
            except Exception as e:
                warnings.warn(f"Model {self.model_names[len(cv_predictions)]} failed in CV: {e}")
                cv_predictions.append(np.zeros_like(y))

        cv_predictions = np.array(cv_predictions)  # (n_models, n_samples)

        # Compute regional performance for each model
        regional_errors = np.zeros((len(self.models), self.n_regions))

        for model_idx in range(len(self.models)):
            analysis = self.analyzer_.analyze_model_performance(
                y, cv_predictions[model_idx], metric='rmse'
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
            if self.preprocessors and self.preprocessors[i] is not None:
                X_processed = self.preprocessors[i].transform(X)
            else:
                X_processed = X
            predictions.append(model.predict(X_processed))
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

    def __init__(self, models, model_names=None, n_regions=5, soft_gating=True, preprocessors=None):
        """
        Parameters
        ----------
        models : list of fitted models
        model_names : list of str, optional
        n_regions : int, default=5
        soft_gating : bool, default=True
            If True, use weighted combination. If False, use hard selection.
        preprocessors : list of preprocessors, optional
            Individual preprocessor for each base model. If None, assumes
            models receive raw data directly.
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.n_regions = n_regions
        self.soft_gating = soft_gating
        self.preprocessors = preprocessors
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

    def fit(self, X, y):
        """Fit by determining which expert handles which region."""
        self.analyzer_.fit(y)

        # Get cross-validated predictions from all models to avoid data leakage
        # This ensures we evaluate model performance on out-of-fold predictions
        predictions = []
        for model in self.models:
            # Use cross_val_predict to get out-of-fold predictions
            # This prevents overfitting and gives realistic performance estimates
            cv_pred = cross_val_predict(model, X, y, cv=5)
            predictions.append(cv_pred)
        predictions = np.array(predictions)

        # For each region, find the best model
        self.expert_assignment_ = np.zeros(self.n_regions, dtype=int)
        self.expert_weights_ = np.zeros((len(self.models), self.n_regions))

        regions = self.analyzer_.assign_regions(y)

        for region_idx in range(self.n_regions):
            mask = regions == region_idx

            if np.sum(mask) == 0:
                # No samples in this region, use first model
                self.expert_assignment_[region_idx] = 0
                self.expert_weights_[0, region_idx] = 1.0
                continue

            y_region = y[mask]

            # Compute error for each model in this region
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
            if self.preprocessors and self.preprocessors[i] is not None:
                X_processed = self.preprocessors[i].transform(X)
            else:
                X_processed = X
            predictions.append(model.predict(X_processed))
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
                 region_aware=True, n_regions=5, cv=5, preprocessors=None):
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
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.region_aware = region_aware
        self.n_regions = n_regions
        self.cv = cv
        self.preprocessors = preprocessors
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

    def fit(self, X, y):
        """Fit the stacking ensemble."""
        # Get cross-validated predictions for meta-features
        meta_features = []

        for model in self.models:
            try:
                cv_pred = cross_val_predict(model, X, y, cv=self.cv)
                meta_features.append(cv_pred)
            except Exception as e:
                warnings.warn(f"Model failed in CV: {e}")
                meta_features.append(np.zeros_like(y))

        meta_features = np.column_stack(meta_features)  # (n_samples, n_models)

        # Add region-aware features if enabled
        if self.region_aware:
            self.analyzer_.fit(y)

            # Add one-hot encoded region features
            avg_pred = np.mean(meta_features, axis=1)
            regions = self.analyzer_.assign_regions(avg_pred)

            # One-hot encode regions
            region_features = np.zeros((len(y), self.n_regions))
            for i, region in enumerate(regions):
                region_features[i, region] = 1.0

            # Also add the predicted value itself
            pred_value_feature = avg_pred.reshape(-1, 1)

            meta_features = np.hstack([
                meta_features,
                region_features,
                pred_value_feature
            ])

        # Fit meta-model
        self.meta_model.fit(meta_features, y)

        return self

    def predict(self, X):
        """Predict using stacking ensemble."""
        # Get predictions from base models, applying individual preprocessors
        predictions = []
        for i, model in enumerate(self.models):
            if self.preprocessors and self.preprocessors[i] is not None:
                X_processed = self.preprocessors[i].transform(X)
            else:
                X_processed = X
            predictions.append(model.predict(X_processed))
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

    Returns
    -------
    Fitted ensemble model
    """
    if ensemble_type == 'simple_average':
        # Simple averaging ensemble (baseline)
        class SimpleAverage(BaseEstimator, RegressorMixin):
            def __init__(self, models, model_names=None, preprocessors=None):
                self.models = models
                self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
                self.preprocessors = preprocessors
            def fit(self, X, y):
                return self
            def predict(self, X):
                # Apply individual preprocessors if provided
                predictions = []
                for i, model in enumerate(self.models):
                    if self.preprocessors and self.preprocessors[i] is not None:
                        X_processed = self.preprocessors[i].transform(X)
                    else:
                        X_processed = X
                    predictions.append(model.predict(X_processed))
                return np.mean(predictions, axis=0)

        ensemble = SimpleAverage(models, model_names=model_names, preprocessors=kwargs.get('preprocessors'))
        ensemble.fit(X, y)

    elif ensemble_type == 'region_weighted':
        ensemble = RegionAwareWeightedEnsemble(
            models, model_names, n_regions=n_regions, **kwargs
        )
        ensemble.fit(X, y)

    elif ensemble_type == 'mixture_experts':
        # MixtureOfExpertsEnsemble doesn't use cv parameter
        moe_kwargs = {k: v for k, v in kwargs.items() if k != 'cv'}
        ensemble = MixtureOfExpertsEnsemble(
            models, model_names, n_regions=n_regions, **moe_kwargs
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
