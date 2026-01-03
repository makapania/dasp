"""Ensemble preprocessing via stacking multiple preprocessing methods.

This module provides ensemble models that train the same base model on multiple
preprocessing methods and combine predictions using a meta-model (Ridge regression).
This captures complementary information from different preprocessing approaches.

Key Features:
- Stack multiple preprocessing methods (raw, SNV, derivatives, baseline corrections)
- Train same base model on each preprocessed version
- Combine predictions using RidgeCV meta-model (prevents overfitting)
- Works with any sklearn-compatible base model
- Separate classes for regression and classification

Example:
--------
>>> from spectral_predict.ensemble_preprocessing import StackedPreprocessingRegressor
>>> from sklearn.ensemble import RandomForestRegressor
>>>
>>> # Define preprocessing methods to ensemble
>>> preprocessings = [
...     ('raw', []),
...     ('snv', [('snv', SNV())]),
...     ('deriv1', [('savgol', SavgolDerivative(deriv=1, window=11))]),
...     ('snv_deriv1', [('snv', SNV()), ('savgol', SavgolDerivative(deriv=1, window=11))])
... ]
>>>
>>> # Create ensemble model
>>> model = StackedPreprocessingRegressor(
...     base_model=RandomForestRegressor(n_estimators=100, random_state=42),
...     preprocessings=preprocessings
... )
>>>
>>> # Fit and predict
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold

# Import existing transformers
from spectral_predict.preprocess import SNV, SavgolDerivative
from spectral_predict.baseline import BaselineALS, BaselinePolynomial, BaselineAirPLS


class StackedPreprocessingRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble regressor that stacks multiple preprocessing methods.

    Trains the same base model on multiple preprocessed versions of the data,
    then combines predictions using a RidgeCV meta-model. This captures
    complementary information from different preprocessing approaches.

    Parameters
    ----------
    base_model : estimator
        Base regression model (must support fit/predict). Will be cloned for each
        preprocessing method.
    preprocessings : list of tuples
        List of (name, preprocessing_steps) tuples. Each preprocessing_steps is a
        list of (step_name, transformer) tuples suitable for sklearn.pipeline.Pipeline.
        Example: [('raw', []), ('snv', [('snv', SNV())])]
    meta_model : estimator, optional
        Meta-model for combining predictions. If None, uses RidgeCV with default alphas.
    cv_folds : int, default=5
        Number of cross-validation folds for generating out-of-fold predictions
        (prevents overfitting when training meta-model).
    n_jobs : int, default=-1
        Number of parallel jobs for meta-model cross-validation.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    models_ : list
        Fitted base models (one per preprocessing method)
    meta_model_ : estimator
        Fitted meta-model for combining predictions
    preprocessing_names_ : list of str
        Names of preprocessing methods

    Examples
    --------
    >>> from spectral_predict.ensemble_preprocessing import StackedPreprocessingRegressor
    >>> from spectral_predict.preprocess import SNV, SavgolDerivative
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Define preprocessings
    >>> preprocessings = [
    ...     ('raw', []),
    ...     ('snv', [('snv', SNV())]),
    ...     ('deriv1', [('deriv', SavgolDerivative(deriv=1, window=11))])
    ... ]
    >>>
    >>> # Create and fit ensemble
    >>> ensemble = StackedPreprocessingRegressor(
    ...     base_model=RandomForestRegressor(n_estimators=100, random_state=42),
    ...     preprocessings=preprocessings
    ... )
    >>> ensemble.fit(X_train, y_train)
    >>> y_pred = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_model,
        preprocessings: list,
        meta_model=None,
        cv_folds: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.preprocessings = preprocessings
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit ensemble model.

        Steps:
        1. For each preprocessing method, train base model on preprocessed data
        2. Generate out-of-fold predictions for training set (via cross-validation)
        3. Train meta-model on stacked out-of-fold predictions

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training spectral data
        y : array-like, shape (n_samples,)
            Training target values

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Store preprocessing names
        self.preprocessing_names_ = [name for name, _ in self.preprocessings]

        # Train base models on each preprocessing
        self.models_ = []
        meta_features = np.zeros((len(y), len(self.preprocessings)))

        for i, (name, preprocess_steps) in enumerate(self.preprocessings):
            # Create pipeline: preprocessing + base model
            steps = preprocess_steps + [('model', clone(self.base_model))]
            pipeline = Pipeline(steps)

            # Fit pipeline on full training set
            pipeline.fit(X, y)
            self.models_.append(pipeline)

            # Generate out-of-fold predictions for meta-model training
            # (prevents overfitting - meta-model sees predictions on unseen data)
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            oof_preds = cross_val_predict(
                clone(pipeline), X, y, cv=cv, n_jobs=1, method='predict'
            )
            meta_features[:, i] = oof_preds

        # Train meta-model on out-of-fold predictions
        if self.meta_model is None:
            # Default: RidgeCV with automatic alpha selection
            self.meta_model_ = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)
        else:
            self.meta_model_ = clone(self.meta_model)

        self.meta_model_.fit(meta_features, y)

        return self

    def predict(self, X):
        """
        Predict using ensemble model.

        Steps:
        1. Generate predictions from each base model (on its preprocessed data)
        2. Stack predictions into feature matrix
        3. Use meta-model to combine predictions

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test spectral data

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values
        """
        X = np.asarray(X)

        # Generate predictions from each base model
        meta_features = np.zeros((len(X), len(self.models_)))
        for i, model in enumerate(self.models_):
            meta_features[:, i] = model.predict(X)

        # Combine predictions using meta-model
        return self.meta_model_.predict(meta_features)

    def get_feature_importances(self):
        """
        Get feature importances showing contribution of each preprocessing method.

        Returns
        -------
        importances : dict
            Dictionary mapping preprocessing names to their meta-model coefficients
            (higher absolute value = more important)
        """
        if not hasattr(self.meta_model_, 'coef_'):
            raise ValueError("Meta-model does not have coefficients (not a linear model)")

        coefs = self.meta_model_.coef_
        return {name: coef for name, coef in zip(self.preprocessing_names_, coefs)}


class StackedPreprocessingClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier that stacks multiple preprocessing methods.

    Trains the same base model on multiple preprocessed versions of the data,
    then combines predictions using a LogisticRegressionCV meta-model.

    Parameters
    ----------
    base_model : estimator
        Base classification model (must support fit/predict). Will be cloned for each
        preprocessing method.
    preprocessings : list of tuples
        List of (name, preprocessing_steps) tuples. Each preprocessing_steps is a
        list of (step_name, transformer) tuples suitable for sklearn.pipeline.Pipeline.
        Example: [('raw', []), ('snv', [('snv', SNV())])]
    meta_model : estimator, optional
        Meta-model for combining predictions. If None, uses LogisticRegressionCV.
    cv_folds : int, default=5
        Number of cross-validation folds for generating out-of-fold predictions.
    n_jobs : int, default=-1
        Number of parallel jobs for meta-model cross-validation.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    models_ : list
        Fitted base models (one per preprocessing method)
    meta_model_ : estimator
        Fitted meta-model for combining predictions
    preprocessing_names_ : list of str
        Names of preprocessing methods
    classes_ : array
        Class labels

    Examples
    --------
    >>> from spectral_predict.ensemble_preprocessing import StackedPreprocessingClassifier
    >>> from spectral_predict.preprocess import SNV, SavgolDerivative
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Define preprocessings
    >>> preprocessings = [
    ...     ('raw', []),
    ...     ('snv', [('snv', SNV())]),
    ...     ('deriv1', [('deriv', SavgolDerivative(deriv=1, window=11))])
    ... ]
    >>>
    >>> # Create and fit ensemble
    >>> ensemble = StackedPreprocessingClassifier(
    ...     base_model=RandomForestClassifier(n_estimators=100, random_state=42),
    ...     preprocessings=preprocessings
    ... )
    >>> ensemble.fit(X_train, y_train)
    >>> y_pred = ensemble.predict(X_test)
    >>> y_proba = ensemble.predict_proba(X_test)
    """

    def __init__(
        self,
        base_model,
        preprocessings: list,
        meta_model=None,
        cv_folds: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.preprocessings = preprocessings
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit ensemble classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training spectral data
        y : array-like, shape (n_samples,)
            Training class labels

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Store classes
        self.classes_ = np.unique(y)

        # Store preprocessing names
        self.preprocessing_names_ = [name for name, _ in self.preprocessings]

        # Train base models on each preprocessing
        self.models_ = []
        # For classification, we use predicted probabilities as meta-features
        n_classes = len(self.classes_)
        meta_features = np.zeros((len(y), len(self.preprocessings) * n_classes))

        for i, (name, preprocess_steps) in enumerate(self.preprocessings):
            # Create pipeline: preprocessing + base model
            steps = preprocess_steps + [('model', clone(self.base_model))]
            pipeline = Pipeline(steps)

            # Fit pipeline on full training set
            pipeline.fit(X, y)
            self.models_.append(pipeline)

            # Generate out-of-fold probability predictions for meta-model training
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            oof_proba = cross_val_predict(
                clone(pipeline), X, y, cv=cv, n_jobs=1, method='predict_proba'
            )
            # Store probabilities for all classes
            meta_features[:, i * n_classes:(i + 1) * n_classes] = oof_proba

        # Train meta-model on out-of-fold predictions
        if self.meta_model is None:
            # Default: LogisticRegressionCV with automatic regularization
            self.meta_model_ = LogisticRegressionCV(
                cv=3, max_iter=1000, random_state=self.random_state, n_jobs=self.n_jobs
            )
        else:
            self.meta_model_ = clone(self.meta_model)

        self.meta_model_.fit(meta_features, y)

        return self

    def predict(self, X):
        """
        Predict class labels using ensemble model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test spectral data

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        X = np.asarray(X)
        n_classes = len(self.classes_)

        # Generate probability predictions from each base model
        meta_features = np.zeros((len(X), len(self.models_) * n_classes))
        for i, model in enumerate(self.models_):
            proba = model.predict_proba(X)
            meta_features[:, i * n_classes:(i + 1) * n_classes] = proba

        # Combine predictions using meta-model
        return self.meta_model_.predict(meta_features)

    def predict_proba(self, X):
        """
        Predict class probabilities using ensemble model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test spectral data

        Returns
        -------
        y_proba : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X = np.asarray(X)
        n_classes = len(self.classes_)

        # Generate probability predictions from each base model
        meta_features = np.zeros((len(X), len(self.models_) * n_classes))
        for i, model in enumerate(self.models_):
            proba = model.predict_proba(X)
            meta_features[:, i * n_classes:(i + 1) * n_classes] = proba

        # Combine predictions using meta-model
        return self.meta_model_.predict_proba(meta_features)

    def get_feature_importances(self):
        """
        Get feature importances showing contribution of each preprocessing method.

        Returns
        -------
        importances : dict
            Dictionary mapping preprocessing names to their meta-model coefficients
        """
        if not hasattr(self.meta_model_, 'coef_'):
            raise ValueError("Meta-model does not have coefficients (not a linear model)")

        coefs = self.meta_model_.coef_
        n_classes = len(self.classes_)

        # Average absolute coefficients across classes for each preprocessing
        importances = {}
        for i, name in enumerate(self.preprocessing_names_):
            # Extract coefficients for this preprocessing (across all classes)
            prep_coefs = coefs[:, i * n_classes:(i + 1) * n_classes]
            # Average absolute value
            importances[name] = np.mean(np.abs(prep_coefs))

        return importances


# Convenience function to create common preprocessing ensembles
def create_standard_preprocessing_ensemble(
    base_model, task_type: str = 'regression', include_baseline: bool = True
):
    """
    Create a standard ensemble with common preprocessing methods.

    Parameters
    ----------
    base_model : estimator
        Base model to use for all preprocessing methods
    task_type : str, default='regression'
        'regression' or 'classification'
    include_baseline : bool, default=True
        Whether to include baseline-corrected versions

    Returns
    -------
    ensemble : StackedPreprocessingRegressor or StackedPreprocessingClassifier
        Configured ensemble model

    Examples
    --------
    >>> from spectral_predict.ensemble_preprocessing import create_standard_preprocessing_ensemble
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Create ensemble with standard preprocessings
    >>> ensemble = create_standard_preprocessing_ensemble(
    ...     RandomForestRegressor(n_estimators=100, random_state=42),
    ...     task_type='regression'
    ... )
    >>> ensemble.fit(X_train, y_train)
    >>> y_pred = ensemble.predict(X_test)
    """
    preprocessings = [
        ('raw', []),
        ('snv', [('snv', SNV())]),
        ('deriv1', [('deriv', SavgolDerivative(deriv=1, window=11, polyorder=2))]),
        ('deriv2', [('deriv', SavgolDerivative(deriv=2, window=11, polyorder=3))]),
        ('snv_deriv1', [
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=1, window=11, polyorder=2))
        ]),
        ('snv_deriv2', [
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=2, window=11, polyorder=3))
        ]),
    ]

    if include_baseline:
        # Add baseline-corrected versions
        preprocessings.extend([
            ('baseline_als', [('baseline', BaselineALS(lambda_=1e6, p=0.001))]),
            ('baseline_als_snv', [
                ('baseline', BaselineALS(lambda_=1e6, p=0.001)),
                ('snv', SNV())
            ]),
        ])

    if task_type == 'regression':
        return StackedPreprocessingRegressor(base_model=base_model, preprocessings=preprocessings)
    else:
        return StackedPreprocessingClassifier(base_model=base_model, preprocessings=preprocessings)


# Self-test using synthetic data
if __name__ == "__main__":
    print("Testing ensemble_preprocessing.py with synthetic data...")

    # Generate synthetic spectral data
    np.random.seed(42)
    n_samples = 150
    n_wavelengths = 200

    # Simulate spectra with baseline + peaks + noise
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    X = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        # Baseline (polynomial)
        baseline = 0.5 + 0.0001 * wavelengths - 0.00000005 * wavelengths ** 2

        # Add Gaussian peaks
        peak1 = 0.3 * np.exp(-((wavelengths - 1000) ** 2) / (2 * 50 ** 2))
        peak2 = 0.5 * np.exp(-((wavelengths - 1500) ** 2) / (2 * 80 ** 2))

        # Noise
        noise = 0.02 * np.random.randn(n_wavelengths)

        X[i, :] = baseline + peak1 + peak2 + noise

    # Create regression target
    y_regression = (
        X[:, np.argmin(np.abs(wavelengths - 1000))]
        + X[:, np.argmin(np.abs(wavelengths - 1500))]
        + 0.1 * np.random.randn(n_samples)
    )

    # Create classification target
    y_classification = (y_regression > np.median(y_regression)).astype(int)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.3, random_state=42
    )
    _, _, y_train_clf, y_test_clf = train_test_split(
        X, y_classification, test_size=0.3, random_state=42
    )

    # Test 1: Regression ensemble with RandomForest
    print("\n" + "=" * 60)
    print("Test 1: Regression Ensemble")
    print("=" * 60)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    ensemble_reg = create_standard_preprocessing_ensemble(
        RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        task_type='regression',
        include_baseline=False  # Faster test
    )

    print("Fitting ensemble regressor...")
    ensemble_reg.fit(X_train, y_train_reg)

    print("Predicting...")
    y_pred_reg = ensemble_reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)

    print(f"\nResults:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"\nPreprocessing importances:")
    importances = ensemble_reg.get_feature_importances()
    for name, importance in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {importance:.6f}")

    # Test 2: Classification ensemble with RandomForest
    print("\n" + "=" * 60)
    print("Test 2: Classification Ensemble")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    ensemble_clf = create_standard_preprocessing_ensemble(
        RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        task_type='classification',
        include_baseline=False  # Faster test
    )

    print("Fitting ensemble classifier...")
    ensemble_clf.fit(X_train, y_train_clf)

    print("Predicting...")
    y_pred_clf = ensemble_clf.predict(X_test)

    accuracy = accuracy_score(y_test_clf, y_pred_clf)

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"\nPreprocessing importances:")
    importances_clf = ensemble_clf.get_feature_importances()
    for name, importance in sorted(importances_clf.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {importance:.6f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
