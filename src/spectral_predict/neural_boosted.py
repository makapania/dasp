"""Neural Boosted Regression for spectral analysis.

Implements gradient boosting with small neural networks as weak learners,
following JMP's Neural Boosted methodology.
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


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
        max_iter=500,
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
                tol=1e-4,  # Tolerance for optimization
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
