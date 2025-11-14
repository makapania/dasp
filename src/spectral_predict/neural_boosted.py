"""Neural Boosted Regression and Classification for spectral analysis.

Implements gradient boosting with small neural networks as weak learners,
following JMP's Neural Boosted methodology.
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


class NeuralBoostedRegressor(BaseEstimator, RegressorMixin):
    """
    Neural Boosted Regression for spectral data.

    Implements gradient boosting with small neural networks as weak learners,
    following JMP's Neural Boosted methodology. The algorithm builds an ensemble
    through stagewise addition: small neural networks fit to scaled residuals,
    with predictions scaled by a learning rate before aggregation.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting rounds (weak learners).

    learning_rate : float, default=0.1
        Shrinkage parameter (0 < ν ≤ 1). Lower values require more estimators
        but can improve generalization. Typical range: 0.05-0.2.

    hidden_layer_size : int, default=3
        Number of nodes in the single hidden layer.
        Should be small (1-5) to maintain weak learner properties.
        Larger values may overfit.

    activation : {'tanh', 'relu', 'identity', 'logistic'}, default='tanh'
        Activation function for hidden layer:
        - 'tanh': Hyperbolic tangent (JMP default, smooth)
        - 'identity': Linear activation (fast, simple)
        - 'relu': Rectified Linear Unit (common in deep learning)
        - 'logistic': Sigmoid function

    alpha : float, default=0.0001
        L2 penalty (weight decay) parameter. Helps prevent overfitting.

    max_iter : int, default=500
        Maximum iterations for training each weak learner.

    early_stopping : bool, default=True
        Whether to use early stopping based on validation score.
        Recommended to prevent overfitting and save computation.

    validation_fraction : float, default=0.15
        Fraction of training data to use for validation (if early_stopping=True).
        Typical range: 0.1-0.2.

    n_iter_no_change : int, default=10
        Stop if validation score doesn't improve for this many rounds.

    loss : {'mse', 'huber'}, default='mse'
        Loss function:
        - 'mse': Mean squared error (standard)
        - 'huber': Huber loss (robust to outliers)

    huber_delta : float, default=1.35
        Delta parameter for Huber loss (ignored if loss='mse').
        Smaller values make it more robust to large outliers.

    random_state : int, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level:
        - 0: Silent
        - 1: Progress updates every 10 iterations
        - 2: Detailed progress

    Attributes
    ----------
    estimators_ : list of MLPRegressor
        The collection of fitted weak learners.

    train_score_ : list of float
        Training score (loss) at each boosting iteration.

    validation_score_ : list of float
        Validation score at each iteration (if early_stopping=True).

    n_estimators_ : int
        Actual number of estimators used (may be less than n_estimators
        if early stopping triggered).

    Examples
    --------
    >>> from neural_boosted import NeuralBoostedRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)  # 100 samples, 50 wavelengths
    >>> y = X[:, 10] + 2*X[:, 20] + np.random.randn(100) * 0.1
    >>> model = NeuralBoostedRegressor(n_estimators=50, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)

    Notes
    -----
    This implementation uses single hidden layer networks as weak learners,
    as specified in JMP's Neural Boosted methodology. For best results:
    - Keep hidden_layer_size small (3-5 nodes)
    - Use learning_rate in range 0.05-0.2
    - Enable early_stopping to prevent overfitting
    - Use Huber loss if data has outliers

    References
    ----------
    .. [1] Friedman, J. H. (2001). "Greedy Function Approximation:
           A Gradient Boosting Machine." Annals of Statistics 29(5): 1189-1232.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        hidden_layer_size=3,
        activation='tanh',
        alpha=0.0001,
        max_iter=100,  # OPTIMIZED: Reduced from 500 (Phase A - evidence shows 15-30 needed)
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        loss='mse',
        huber_delta=1.35,
        random_state=None,
        verbose=0
    ):
        # Parameter validation
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if hidden_layer_size < 1:
            raise ValueError(f"hidden_layer_size must be >= 1, got {hidden_layer_size}")
        if hidden_layer_size > 10:
            warnings.warn(
                f"hidden_layer_size={hidden_layer_size} is large for weak learner. "
                "Consider using 3-5 nodes to maintain weak learner properties.",
                UserWarning
            )
        if activation not in ['tanh', 'relu', 'identity', 'logistic']:
            raise ValueError(f"Unknown activation: {activation}")
        if loss not in ['mse', 'huber']:
            raise ValueError(f"Unknown loss: {loss}. Must be 'mse' or 'huber'")
        if not 0 < validation_fraction < 1:
            raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.loss = loss
        self.huber_delta = huber_delta
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the Neural Boosted ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Initialize storage
        self.estimators_ = []
        self.train_score_ = []
        self.validation_score_ = []

        # Split validation set if early stopping
        if self.early_stopping:
            if len(y) < 20:
                warnings.warn(
                    "Dataset is very small (<20 samples). "
                    "Consider setting early_stopping=False or using more data.",
                    UserWarning
                )
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.validation_fraction,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Initialize ensemble prediction to zero
        F_train = np.zeros(len(y_train))
        if X_val is not None:
            F_val = np.zeros(len(y_val))

        best_val_score = np.inf
        no_improvement_count = 0

        if self.verbose > 0:
            print(f"Neural Boosted Training: {len(y_train)} train samples, "
                  f"{X_train.shape[1]} features")
            if self.early_stopping:
                print(f"Validation: {len(y_val)} samples")

        # Boosting loop
        for i in range(self.n_estimators):
            # Compute residuals (what we got wrong)
            residuals = y_train - F_train

            # Create and fit weak learner
            # Use lbfgs solver for small networks - converges better than adam
            weak_learner = MLPRegressor(
                hidden_layer_sizes=(self.hidden_layer_size,),
                activation=self.activation,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state + i if self.random_state is not None else None,
                warm_start=False,
                solver='lbfgs',  # Better for small networks
                tol=5e-4,  # OPTIMIZED: Relaxed from 1e-4 (Phase A - faster convergence)
                verbose=False
            )

            try:
                weak_learner.fit(X_train, residuals)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Weak learner {i} failed to converge: {e}")
                # If fitting fails, use zero prediction (no contribution)
                self.estimators_.append(None)
                continue

            self.estimators_.append(weak_learner)

            # Update ensemble predictions
            pred_train = weak_learner.predict(X_train)
            F_train += self.learning_rate * pred_train

            # Compute training score
            train_score = self._compute_loss(y_train, F_train)
            self.train_score_.append(train_score)

            # Early stopping check
            if self.early_stopping:
                pred_val = weak_learner.predict(X_val)
                F_val += self.learning_rate * pred_val
                val_score = self._compute_loss(y_val, F_val)
                self.validation_score_.append(val_score)

                if val_score < best_val_score:
                    best_val_score = val_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    if self.verbose > 0:
                        print(f"Early stopping at iteration {i+1}/{self.n_estimators}")
                        print(f"Best validation score: {best_val_score:.6f}")
                    break

            # Progress reporting
            if self.verbose > 0 and (i + 1) % 10 == 0:
                msg = f"Iteration {i+1}/{self.n_estimators}, Train Loss: {train_score:.6f}"
                if self.early_stopping:
                    msg += f", Val Loss: {val_score:.6f}"
                print(msg)

        # Remove None entries (failed weak learners)
        self.estimators_ = [e for e in self.estimators_ if e is not None]
        self.n_estimators_ = len(self.estimators_)

        if self.verbose > 0:
            print(f"Training complete: {self.n_estimators_} weak learners")

        return self

    def predict(self, X):
        """
        Predict using the Neural Boosted ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ['estimators_'])
        X = check_array(X, accept_sparse=False)

        # Initialize predictions to zero
        predictions = np.zeros(X.shape[0])

        # Aggregate predictions from all weak learners
        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)

        return predictions

    def _compute_loss(self, y_true, y_pred):
        """
        Compute loss (MSE or Huber).

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values.

        Returns
        -------
        loss : float
            Computed loss value.
        """
        if self.loss == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss == 'huber':
            residuals = y_true - y_pred
            abs_residuals = np.abs(residuals)
            # Huber loss: quadratic for small errors, linear for large
            quadratic = np.minimum(abs_residuals, self.huber_delta)
            linear = abs_residuals - quadratic
            return np.mean(0.5 * quadratic**2 + self.huber_delta * linear)

    def get_feature_importances(self):
        """
        Compute feature importances by averaging absolute weights
        across all weak learners.

        Feature importance for variable i is computed as:
        importance_i = (1/N) * Σ_n mean(|W_n[i,:]|)

        Where:
        - N = number of weak learners
        - W_n = weight matrix of weak learner n
        - W_n[i,:] = weights from feature i to all hidden nodes

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Feature importance scores (higher = more important).
            All values are non-negative.

        Notes
        -----
        This method aggregates importances across all weak learners in the
        ensemble. Features with consistently high weights across many learners
        will have higher importance scores.
        """
        check_is_fitted(self, ['estimators_'])

        n_features = self.estimators_[0].coefs_[0].shape[0]
        importances = np.zeros(n_features)

        for estimator in self.estimators_:
            # First layer weights: (n_features, n_hidden)
            weights = estimator.coefs_[0]
            # Average absolute weight per feature (across hidden nodes)
            feature_importance = np.mean(np.abs(weights), axis=1)
            importances += feature_importance

        # Normalize by number of estimators
        importances /= len(self.estimators_)

        return importances

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'hidden_layer_size': self.hidden_layer_size,
            'activation': self.activation,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'loss': self.loss,
            'huber_delta': self.huber_delta,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class NeuralBoostedClassifier(BaseEstimator, ClassifierMixin):
    """
    Neural Boosted Classification for spectral data.

    Implements gradient boosting with small neural networks as weak learners for
    classification tasks. Uses log-loss (binary cross-entropy) for gradient computation
    and supports both binary and multiclass classification via one-vs-rest strategy.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting rounds (weak learners).

    learning_rate : float, default=0.1
        Shrinkage parameter (0 < ν ≤ 1). Lower values require more estimators
        but can improve generalization. Typical range: 0.05-0.2.

    hidden_layer_size : int, default=3
        Number of nodes in the single hidden layer.
        Should be small (3-8) to maintain weak learner properties.
        Larger values may overfit.

    activation : {'tanh', 'relu', 'identity', 'logistic'}, default='tanh'
        Activation function for hidden layer:
        - 'tanh': Hyperbolic tangent (smooth, good for small networks)
        - 'relu': Rectified Linear Unit (common in deep learning)
        - 'identity': Linear activation (fast, simple)
        - 'logistic': Sigmoid function

    alpha : float, default=0.0001
        L2 penalty (weight decay) parameter. Helps prevent overfitting.

    max_iter : int, default=100
        Maximum iterations for training each weak learner.

    early_stopping : bool, default=True
        Whether to use early stopping based on validation score.
        Recommended to prevent overfitting and save computation.

    validation_fraction : float, default=0.15
        Fraction of training data to use for validation (if early_stopping=True).
        Typical range: 0.1-0.2.

    n_iter_no_change : int, default=10
        Stop if validation score doesn't improve for this many rounds.

    early_stopping_metric : {'accuracy', 'log_loss'}, default='accuracy'
        Metric to use for early stopping:
        - 'accuracy': Classification accuracy (easier to interpret)
        - 'log_loss': Log-loss / cross-entropy (more principled)

    class_weight : {'balanced', None} or dict, default=None
        Weights associated with classes:
        - None: All classes have weight 1
        - 'balanced': Automatic weighting inversely proportional to class frequencies
        - dict: Manual weights per class {class_label: weight}

    random_state : int, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level:
        - 0: Silent
        - 1: Progress updates every 10 iterations
        - 2: Detailed progress

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.

    n_classes_ : int
        Number of classes.

    estimators_ : list of list of MLPRegressor
        The collection of fitted weak learners. For binary classification,
        estimators_[0] contains all weak learners. For multiclass,
        estimators_[i] contains weak learners for class i vs rest.

    train_score_ : list of float
        Training score at each boosting iteration.

    validation_score_ : list of float
        Validation score at each iteration (if early_stopping=True).

    n_estimators_ : int or list of int
        Actual number of estimators used (may be less than n_estimators
        if early stopping triggered). For multiclass, list of counts per class.

    label_encoder_ : LabelEncoder
        Encoder for class labels.

    Examples
    --------
    >>> from neural_boosted import NeuralBoostedClassifier
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=50, n_classes=2)
    >>> model = NeuralBoostedClassifier(n_estimators=50, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> probabilities = model.predict_proba(X)

    Notes
    -----
    - For binary classification: Uses log-loss gradient boosting
    - For multiclass: Uses one-vs-rest strategy (fits N binary classifiers)
    - Keep hidden_layer_size small (3-8 nodes) for best generalization
    - Use class_weight='balanced' for imbalanced datasets
    - early_stopping_metric='accuracy' is recommended for most cases

    References
    ----------
    .. [1] Friedman, J. H. (2001). "Greedy Function Approximation:
           A Gradient Boosting Machine." Annals of Statistics 29(5): 1189-1232.
    .. [2] Friedman, J., Hastie, T., & Tibshirani, R. (2000). "Additive logistic
           regression: a statistical view of boosting." Annals of Statistics,
           28(2), 337-407.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        hidden_layer_size=5,
        activation='tanh',
        alpha=0.0001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        early_stopping_metric='accuracy',
        class_weight=None,
        random_state=None,
        verbose=0
    ):
        # Parameter validation
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if hidden_layer_size < 1:
            raise ValueError(f"hidden_layer_size must be >= 1, got {hidden_layer_size}")
        if hidden_layer_size > 15:
            warnings.warn(
                f"hidden_layer_size={hidden_layer_size} is large for weak learner. "
                "Consider using 3-8 nodes to maintain weak learner properties.",
                UserWarning
            )
        if activation not in ['tanh', 'relu', 'identity', 'logistic']:
            raise ValueError(f"Unknown activation: {activation}")
        if early_stopping_metric not in ['accuracy', 'log_loss']:
            raise ValueError(f"early_stopping_metric must be 'accuracy' or 'log_loss', got {early_stopping_metric}")
        if not 0 < validation_fraction < 1:
            raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
        if class_weight is not None and class_weight != 'balanced' and not isinstance(class_weight, dict):
            raise ValueError(f"class_weight must be None, 'balanced', or a dict, got {type(class_weight)}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping_metric = early_stopping_metric
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the Neural Boosted classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        if self.verbose > 0:
            print(f"Neural Boosted Classifier Training")
            print(f"Classes: {self.n_classes_} ({self.classes_})")
            print(f"Samples: {len(y)}, Features: {X.shape[1]}")

        # Compute sample weights if requested
        sample_weight = None
        if self.class_weight is not None:
            if self.class_weight == 'balanced':
                sample_weight = compute_sample_weight('balanced', y_encoded)
            elif isinstance(self.class_weight, dict):
                # Convert class labels to weights
                sample_weight = np.array([self.class_weight.get(self.classes_[yi], 1.0)
                                         for yi in y_encoded])

            if self.verbose > 0:
                unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
                print(f"Class distribution: {dict(zip(self.classes_[unique_classes], class_counts))}")
                if sample_weight is not None:
                    print(f"Using class weights (mean={sample_weight.mean():.3f})")

        # Binary vs multiclass
        if self.n_classes_ == 2:
            # Binary classification
            self.estimators_ = self._fit_binary(X, y_encoded, sample_weight)
            self.n_estimators_ = len(self.estimators_)
        else:
            # Multiclass via one-vs-rest
            self.estimators_ = []
            self.n_estimators_ = []

            for class_idx in range(self.n_classes_):
                if self.verbose > 0:
                    print(f"\nTraining classifier for class {self.classes_[class_idx]} vs rest...")

                # Create binary labels (1 for this class, 0 for all others)
                y_binary = (y_encoded == class_idx).astype(int)

                # Fit binary classifier
                estimators = self._fit_binary(X, y_binary, sample_weight)
                self.estimators_.append(estimators)
                self.n_estimators_.append(len(estimators))

        return self

    def _fit_binary(self, X, y_binary, sample_weight=None):
        """
        Fit binary classifier using log-loss gradient boosting.

        Parameters
        ----------
        X : ndarray
            Training features.
        y_binary : ndarray
            Binary labels (0 or 1).
        sample_weight : ndarray or None
            Sample weights.

        Returns
        -------
        estimators : list of MLPRegressor
            Fitted weak learners.
        """
        estimators = []
        train_scores = []
        val_scores = []

        # Split validation set if early stopping
        if self.early_stopping:
            if len(y_binary) < 20:
                warnings.warn(
                    "Dataset is very small (<20 samples). "
                    "Consider setting early_stopping=False or using more data.",
                    UserWarning
                )

            # Split while preserving sample weights
            indices = np.arange(len(y_binary))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=self.validation_fraction,
                stratify=y_binary,
                random_state=self.random_state
            )

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_binary[train_idx], y_binary[val_idx]

            if sample_weight is not None:
                sw_train = sample_weight[train_idx]
                sw_val = sample_weight[val_idx]
            else:
                sw_train = None
                sw_val = None
        else:
            X_train, y_train = X, y_binary
            sw_train = sample_weight
            X_val, y_val, sw_val = None, None, None

        # Initialize log-odds (F = log(p / (1-p)))
        # Start with class proportion
        if sw_train is not None:
            p_init = np.average(y_train, weights=sw_train)
        else:
            p_init = y_train.mean()

        # Clip to avoid log(0)
        p_init = np.clip(p_init, 1e-7, 1 - 1e-7)
        F_init = np.log(p_init / (1 - p_init))

        F_train = np.full(len(y_train), F_init)
        if X_val is not None:
            F_val = np.full(len(y_val), F_init)

        best_val_score = -np.inf if self.early_stopping_metric == 'accuracy' else np.inf
        no_improvement_count = 0

        # Boosting loop
        for i in range(self.n_estimators):
            # Compute current probabilities
            p_train = 1 / (1 + np.exp(-F_train))

            # Compute residuals (gradient of log-loss)
            residuals = y_train - p_train

            # Apply sample weights to residuals
            if sw_train is not None:
                weighted_residuals = residuals * sw_train
            else:
                weighted_residuals = residuals

            # Fit weak learner to residuals
            weak_learner = MLPRegressor(
                hidden_layer_sizes=(self.hidden_layer_size,),
                activation=self.activation,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state + i if self.random_state is not None else None,
                warm_start=False,
                solver='lbfgs',
                tol=5e-4,
                verbose=False
            )

            try:
                weak_learner.fit(X_train, weighted_residuals)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Weak learner {i} failed to converge: {e}")
                continue

            estimators.append(weak_learner)

            # Update log-odds
            pred_train = weak_learner.predict(X_train)
            F_train += self.learning_rate * pred_train

            # Compute training score
            if self.early_stopping_metric == 'accuracy':
                p_train_updated = 1 / (1 + np.exp(-F_train))
                train_score = np.mean((p_train_updated > 0.5) == y_train)
                train_scores.append(train_score)
            else:  # log_loss
                p_train_updated = 1 / (1 + np.exp(-F_train))
                p_train_updated = np.clip(p_train_updated, 1e-7, 1 - 1e-7)
                train_score = -np.mean(y_train * np.log(p_train_updated) +
                                      (1 - y_train) * np.log(1 - p_train_updated))
                train_scores.append(train_score)

            # Early stopping check
            if self.early_stopping:
                pred_val = weak_learner.predict(X_val)
                F_val += self.learning_rate * pred_val
                p_val = 1 / (1 + np.exp(-F_val))

                if self.early_stopping_metric == 'accuracy':
                    val_score = np.mean((p_val > 0.5) == y_val)
                    improved = val_score > best_val_score
                else:  # log_loss
                    p_val_clipped = np.clip(p_val, 1e-7, 1 - 1e-7)
                    val_score = -np.mean(y_val * np.log(p_val_clipped) +
                                        (1 - y_val) * np.log(1 - p_val_clipped))
                    improved = val_score < best_val_score

                val_scores.append(val_score)

                if improved:
                    best_val_score = val_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    if self.verbose > 0:
                        print(f"  Early stopping at iteration {i+1}/{self.n_estimators}")
                        print(f"  Best validation {self.early_stopping_metric}: {best_val_score:.6f}")
                    break

            # Progress reporting
            if self.verbose > 0 and (i + 1) % 10 == 0:
                msg = f"  Iteration {i+1}/{self.n_estimators}"
                if self.early_stopping_metric == 'accuracy':
                    msg += f", Train Acc: {train_score:.4f}"
                else:
                    msg += f", Train Loss: {train_score:.6f}"
                if self.early_stopping:
                    if self.early_stopping_metric == 'accuracy':
                        msg += f", Val Acc: {val_score:.4f}"
                    else:
                        msg += f", Val Loss: {val_score:.6f}"
                print(msg)

        if self.verbose > 0:
            print(f"  Completed: {len(estimators)} weak learners")

        # Store scores for this binary classifier
        self.train_score_ = train_scores
        self.validation_score_ = val_scores

        return estimators

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['estimators_', 'classes_'])
        X = check_array(X, accept_sparse=False)

        if self.n_classes_ == 2:
            # Binary classification
            # Initialize with class prior (log-odds)
            p_init = 0.5  # Neutral prior
            F_init = np.log(p_init / (1 - p_init))
            F = np.full(X.shape[0], F_init)

            # Aggregate weak learners
            for estimator in self.estimators_:
                F += self.learning_rate * estimator.predict(X)

            # Convert log-odds to probability
            p1 = 1 / (1 + np.exp(-F))
            p0 = 1 - p1

            return np.column_stack([p0, p1])

        else:
            # Multiclass: one-vs-rest
            # Get raw scores for each class
            scores = np.zeros((X.shape[0], self.n_classes_))

            for class_idx in range(self.n_classes_):
                F_init = 0.0  # Neutral prior
                F = np.full(X.shape[0], F_init)

                for estimator in self.estimators_[class_idx]:
                    F += self.learning_rate * estimator.predict(X)

                # Store log-odds for this class
                scores[:, class_idx] = F

            # Apply softmax to get probabilities
            # Subtract max for numerical stability
            scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba = scores_exp / scores_exp.sum(axis=1, keepdims=True)

            return proba

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        y_pred_encoded = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log of class probabilities.
        """
        proba = self.predict_proba(X)
        # Clip to avoid log(0)
        return np.log(np.clip(proba, 1e-10, 1.0))

    def get_feature_importances(self):
        """
        Compute feature importances by averaging absolute weights
        across all weak learners.

        For multiclass, importances are averaged across all one-vs-rest classifiers.

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Feature importance scores (higher = more important).
        """
        check_is_fitted(self, ['estimators_'])

        if self.n_classes_ == 2:
            # Binary: single set of estimators
            estimators_list = self.estimators_
        else:
            # Multiclass: flatten all estimators
            estimators_list = [est for class_estimators in self.estimators_
                              for est in class_estimators]

        n_features = estimators_list[0].coefs_[0].shape[0]
        importances = np.zeros(n_features)

        for estimator in estimators_list:
            weights = estimator.coefs_[0]
            feature_importance = np.mean(np.abs(weights), axis=1)
            importances += feature_importance

        # Normalize by number of estimators
        importances /= len(estimators_list)

        return importances

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'hidden_layer_size': self.hidden_layer_size,
            'activation': self.activation,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'early_stopping_metric': self.early_stopping_metric,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
