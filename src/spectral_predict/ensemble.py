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
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict
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
            predictions.append(model.predict(X_processed))
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

    def __init__(self, models, model_names=None, n_regions=5, cv=5, preprocessors=None, preprocessor_configs=None):
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
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.n_regions = n_regions
        self.cv = cv
        self.preprocessors = preprocessors
        self.preprocessor_configs = preprocessor_configs
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
        Fit the ensemble by computing regional performance weights.

        Uses direct predictions from already-fitted models (not cross_val_predict)
        to ensure weights match what predict() will actually produce.

        Note: Models passed to ensembles are typically already fitted with
        preprocessing wrappers. cross_val_predict would clone these models
        and refit from scratch, creating a mismatch between weight calculation
        and actual prediction behavior.
        """
        # Get direct predictions from fitted models
        # IMPORTANT: Apply preprocessing to X for each model (same as in predict)
        predictions = []
        for i, model in enumerate(self.models):
            try:
                # Apply preprocessing for this model (must match predict behavior)
                preprocessor = self._get_preprocessor(i)
                if preprocessor is not None:
                    X_processed = preprocessor.transform(X)
                else:
                    X_processed = X

                # Use direct prediction from fitted model (NOT cross_val_predict)
                # This ensures weights match what predict() will produce
                pred = model.predict(X_processed)
                pred = np.asarray(pred).ravel()
                predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Model {self.model_names[len(predictions)]} failed: {e}")
                predictions.append(np.zeros_like(y))

        predictions = np.array(predictions)  # (n_models, n_samples)

        # Define region boundaries from average predictions
        # This ensures consistency with predict(), which assigns regions
        # based on average predictions
        avg_pred = np.mean(predictions, axis=0)
        self.analyzer_.fit(avg_pred)

        # Compute regional performance for each model
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

    def __init__(self, models, model_names=None, n_regions=5, soft_gating=True, preprocessors=None, preprocessor_configs=None):
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
        preprocessor_configs : list of PreprocessorConfig, optional
            Configuration objects for reconstructing preprocessing per model.
            Used when preprocessors aren't available directly.
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.n_regions = n_regions
        self.soft_gating = soft_gating
        self.preprocessors = preprocessors
        self.preprocessor_configs = preprocessor_configs
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
        Fit by determining which expert handles which region.

        Uses direct predictions from already-fitted models (not cross_val_predict)
        to ensure weights match what predict() will actually produce.
        """
        # Get direct predictions from all fitted models
        # IMPORTANT: Apply preprocessing to X for each model (same as in predict)
        predictions = []
        for i, model in enumerate(self.models):
            # Apply preprocessing for this model (must match predict behavior)
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X

            # Use direct prediction from fitted model (NOT cross_val_predict)
            # This ensures weights match what predict() will produce
            pred = model.predict(X_processed)
            pred = np.asarray(pred).ravel()
            predictions.append(pred)
        predictions = np.array(predictions)

        # Define region boundaries from average predictions
        # This ensures consistency with predict(), which assigns regions
        # based on average predictions
        avg_pred = np.mean(predictions, axis=0)
        self.analyzer_.fit(avg_pred)

        # For each region, find the best model
        self.expert_assignment_ = np.zeros(self.n_regions, dtype=int)
        self.expert_weights_ = np.zeros((len(self.models), self.n_regions))

        # Assign regions based on average predictions (matches predict behavior)
        regions = self.analyzer_.assign_regions(avg_pred)

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
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
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
                 region_aware=True, n_regions=5, cv=5, preprocessors=None, preprocessor_configs=None):
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
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.region_aware = region_aware
        self.n_regions = n_regions
        self.cv = cv
        self.preprocessors = preprocessors
        self.preprocessor_configs = preprocessor_configs
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
        Fit the stacking ensemble.

        Uses direct predictions from already-fitted models (not cross_val_predict)
        to ensure meta-features match what predict() will actually produce.
        """
        # Get direct predictions for meta-features
        # IMPORTANT: Apply preprocessing to X for each model (same as in predict)
        meta_features = []

        for i, model in enumerate(self.models):
            try:
                # Apply preprocessing for this model (must match predict behavior)
                preprocessor = self._get_preprocessor(i)
                if preprocessor is not None:
                    X_processed = preprocessor.transform(X)
                else:
                    X_processed = X

                # Use direct prediction from fitted model (NOT cross_val_predict)
                # This ensures meta-features match what predict() will produce
                pred = model.predict(X_processed)
                pred = np.asarray(pred).ravel()
                meta_features.append(pred)
            except Exception as e:
                warnings.warn(f"Model failed: {e}")
                meta_features.append(np.zeros_like(y))

        meta_features = np.column_stack(meta_features)  # (n_samples, n_models)

        # Add region-aware features if enabled
        if self.region_aware:
            # Define region boundaries from average predictions
            # This ensures consistency with predict()
            avg_pred = np.mean(meta_features, axis=1)
            self.analyzer_.fit(avg_pred)

            # Add one-hot encoded region features
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
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
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
