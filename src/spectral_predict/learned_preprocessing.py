"""Learned preprocessing using PyTorch neural networks.

This module provides learnable preprocessing layers that replace traditional fixed
transformations (SNV, Savitzky-Golay) with neural network layers optimized end-to-end
with the predictive model.

Key Features:
- InstanceNorm1d: Learnable normalization (replaces SNV)
- Conv1d: Learnable filtering (replaces Savitzky-Golay derivatives)
- Dropout: Regularization to prevent overfitting
- End-to-end training: Preprocessing and prediction optimized jointly
- Extract preprocessed spectra: Can use learned preprocessing with sklearn models
- Graceful fallback: Returns informative error if PyTorch not installed

Example:
--------
>>> # Only works if PyTorch is installed
>>> from spectral_predict.learned_preprocessing import SpectralPreprocessorWithRegressor
>>>
>>> # Create end-to-end model (preprocessing + regression)
>>> model = SpectralPreprocessorWithRegressor(
...     n_wavelengths=200,
...     n_conv_layers=2,
...     n_filters=16,
...     kernel_size=11,
...     hidden_size=64,
...     dropout=0.3,
...     learning_rate=1e-3
... )
>>>
>>> # Fit and predict
>>> model.fit(X_train, y_train, epochs=100)
>>> y_pred = model.predict(X_test)
>>>
>>> # Extract preprocessed spectra for use with sklearn models
>>> X_preprocessed = model.transform(X)

Requirements:
-------------
- PyTorch (optional): pip install torch
- If PyTorch is not installed, module will provide informative error messages
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# Stub classes with helpful error messages if PyTorch not available
if not PYTORCH_AVAILABLE:

    class LearnedSpectralPreprocessing:
        """Stub class - PyTorch not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for learned preprocessing.\n"
                "Install with: pip install torch\n"
                "Or use conda: conda install pytorch -c pytorch"
            )

    class SpectralPreprocessorWithRegressor:
        """Stub class - PyTorch not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for learned preprocessing.\n"
                "Install with: pip install torch\n"
                "Or use conda: conda install pytorch -c pytorch"
            )

else:
    # Full implementation if PyTorch is available

    class LearnedSpectralPreprocessing(nn.Module):
        """
        Learnable 1D CNN preprocessing for spectral data.

        This neural network learns optimal spectral transformations (normalization,
        filtering, feature extraction) end-to-end with the predictive task.

        Architecture:
        - InstanceNorm1d: Learnable per-spectrum normalization (like SNV)
        - Conv1d layers: Learnable filtering (like Savitzky-Golay)
        - Dropout: Regularization

        Parameters
        ----------
        n_wavelengths : int
            Number of input wavelengths (spectral features)
        n_conv_layers : int, default=2
            Number of convolutional layers
        n_filters : int, default=16
            Number of filters per convolutional layer
        kernel_size : int, default=11
            Kernel size for convolutions (like Savitzky-Golay window)
        dropout : float, default=0.3
            Dropout rate for regularization
        """

        def __init__(
            self,
            n_wavelengths: int,
            n_conv_layers: int = 2,
            n_filters: int = 16,
            kernel_size: int = 11,
            dropout: float = 0.3,
        ):
            super().__init__()

            self.n_wavelengths = n_wavelengths
            self.n_conv_layers = n_conv_layers
            self.n_filters = n_filters
            self.kernel_size = kernel_size
            self.dropout = dropout

            # Instance normalization (learnable, per-spectrum)
            # Similar to SNV but with learnable affine parameters
            self.instance_norm = nn.InstanceNorm1d(1, affine=True)

            # Convolutional layers (learnable filtering)
            conv_layers = []
            for i in range(n_conv_layers):
                padding = kernel_size // 2  # Same padding
                if i == 0:
                    # First layer: 1 input channel (spectrum)
                    conv_layers.append(
                        nn.Conv1d(1, n_filters, kernel_size, padding=padding, bias=True)
                    )
                else:
                    # Subsequent layers
                    conv_layers.append(
                        nn.Conv1d(n_filters, n_filters, kernel_size, padding=padding, bias=True)
                    )
                conv_layers.append(nn.ReLU())
                conv_layers.append(nn.Dropout(dropout))

            self.conv_layers = nn.Sequential(*conv_layers)

            # Final projection back to original dimensionality
            self.projection = nn.Conv1d(n_filters, 1, kernel_size=1, bias=True)

        def forward(self, x):
            """
            Apply learned preprocessing.

            Parameters
            ----------
            x : torch.Tensor, shape (batch_size, n_wavelengths)
                Input spectra

            Returns
            -------
            x_preprocessed : torch.Tensor, shape (batch_size, n_wavelengths)
                Preprocessed spectra
            """
            # Add channel dimension: (batch, wavelengths) -> (batch, 1, wavelengths)
            x = x.unsqueeze(1)

            # Instance normalization (learnable SNV-like)
            x = self.instance_norm(x)

            # Convolutional layers (learnable filtering)
            x = self.conv_layers(x)

            # Project back to 1 channel
            x = self.projection(x)

            # Remove channel dimension: (batch, 1, wavelengths) -> (batch, wavelengths)
            x = x.squeeze(1)

            return x

    class SpectralPreprocessorWithRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
        """
        End-to-end learnable preprocessing + regression model.

        This model jointly optimizes spectral preprocessing (normalization, filtering)
        and regression head using gradient descent. The preprocessing network learns
        optimal transformations specific to the prediction task.

        Parameters
        ----------
        n_wavelengths : int, optional
            Number of input wavelengths. If None, inferred from training data.
        n_conv_layers : int, default=2
            Number of convolutional preprocessing layers
        n_filters : int, default=16
            Number of filters in each preprocessing layer
        kernel_size : int, default=11
            Kernel size for convolutions
        hidden_size : int, default=64
            Size of hidden layer in regression head
        dropout : float, default=0.3
            Dropout rate for regularization
        learning_rate : float, default=1e-3
            Learning rate for Adam optimizer
        batch_size : int, default=32
            Batch size for training
        device : str, optional
            Device for computation ('cpu', 'cuda', or 'mps'). If None, auto-detected.

        Attributes
        ----------
        model_ : nn.Module
            Fitted PyTorch model (preprocessing + regression head)
        device_ : torch.device
            Device used for computation
        """

        def __init__(
            self,
            n_wavelengths: int = None,
            n_conv_layers: int = 2,
            n_filters: int = 16,
            kernel_size: int = 11,
            hidden_size: int = 64,
            dropout: float = 0.3,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            device: str = None,
        ):
            self.n_wavelengths = n_wavelengths
            self.n_conv_layers = n_conv_layers
            self.n_filters = n_filters
            self.kernel_size = kernel_size
            self.hidden_size = hidden_size
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.device = device

        def fit(self, X, y, epochs: int = 100, validation_split: float = 0.2, verbose: bool = False):
            """
            Fit the learned preprocessing + regression model.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_wavelengths)
                Training spectral data
            y : array-like, shape (n_samples,)
                Training target values
            epochs : int, default=100
                Number of training epochs
            validation_split : float, default=0.2
                Fraction of training data to use for validation
            verbose : bool, default=False
                Whether to print training progress

            Returns
            -------
            self : object
                Fitted estimator
            """
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

            # Infer n_wavelengths if not provided
            if self.n_wavelengths is None:
                self.n_wavelengths = X.shape[1]

            # Determine device
            if self.device is None:
                if torch.cuda.is_available():
                    self.device_ = torch.device('cuda')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device_ = torch.device('mps')
                else:
                    self.device_ = torch.device('cpu')
            else:
                self.device_ = torch.device(self.device)

            # Split into training and validation sets
            n_samples = len(X)
            n_val = int(n_samples * validation_split)
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # Create PyTorch datasets and dataloaders
            train_dataset = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train)
            )
            val_dataset = TensorDataset(
                torch.from_numpy(X_val), torch.from_numpy(y_val)
            )

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            # Build model: preprocessing + regression head
            class PreprocessorRegressor(nn.Module):
                def __init__(self, preprocessor, n_wavelengths, hidden_size, dropout):
                    super().__init__()
                    self.preprocessor = preprocessor
                    self.fc1 = nn.Linear(n_wavelengths, hidden_size)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(dropout)
                    self.fc2 = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    # Preprocessing
                    x = self.preprocessor(x)
                    # Regression head
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x

            preprocessor = LearnedSpectralPreprocessing(
                n_wavelengths=self.n_wavelengths,
                n_conv_layers=self.n_conv_layers,
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )

            self.model_ = PreprocessorRegressor(
                preprocessor, self.n_wavelengths, self.hidden_size, self.dropout
            ).to(self.device_)

            # Optimizer and loss
            optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.model_.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device_), y_batch.to(self.device_)

                    optimizer.zero_grad()
                    y_pred = self.model_(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * len(X_batch)

                train_loss /= len(train_dataset)

                # Validation phase
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device_), y_batch.to(self.device_)
                        y_pred = self.model_(X_batch)
                        loss = criterion(y_pred, y_batch)
                        val_loss += loss.item() * len(X_batch)

                val_loss /= len(val_dataset)

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )

            return self

        def predict(self, X):
            """
            Predict using learned preprocessing + regression model.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_wavelengths)
                Test spectral data

            Returns
            -------
            y_pred : array, shape (n_samples,)
                Predicted target values
            """
            X = np.asarray(X, dtype=np.float32)
            X_tensor = torch.from_numpy(X).to(self.device_)

            self.model_.eval()
            with torch.no_grad():
                y_pred = self.model_(X_tensor)

            return y_pred.cpu().numpy().flatten()

        def transform(self, X):
            """
            Extract preprocessed spectra (for use with sklearn models).

            Parameters
            ----------
            X : array-like, shape (n_samples, n_wavelengths)
                Input spectral data

            Returns
            -------
            X_preprocessed : array, shape (n_samples, n_wavelengths)
                Preprocessed spectral data
            """
            X = np.asarray(X, dtype=np.float32)
            X_tensor = torch.from_numpy(X).to(self.device_)

            self.model_.eval()
            with torch.no_grad():
                X_preprocessed = self.model_.preprocessor(X_tensor)

            return X_preprocessed.cpu().numpy()

        def fit_transform(self, X, y, **fit_params):
            """Fit and transform in one step."""
            self.fit(X, y, **fit_params)
            return self.transform(X)


# Self-test
if __name__ == "__main__":
    if not PYTORCH_AVAILABLE:
        print("=" * 60)
        print("PyTorch Not Installed")
        print("=" * 60)
        print("PyTorch is required for learned preprocessing.")
        print("Install with: pip install torch")
        print("Or use conda: conda install pytorch -c pytorch")
        print("=" * 60)
        print("\nSkipping tests (PyTorch not available)")
    else:
        print("Testing learned_preprocessing.py with synthetic data...")

        # Generate synthetic spectral data
        np.random.seed(42)
        n_samples = 200
        n_wavelengths = 200

        # Simulate spectra with baseline + peaks + noise
        wavelengths = np.linspace(400, 2500, n_wavelengths)
        X = np.zeros((n_samples, n_wavelengths))

        for i in range(n_samples):
            # Baseline
            baseline = 0.5 + 0.0001 * wavelengths - 0.00000005 * wavelengths ** 2

            # Add Gaussian peaks
            peak1 = 0.3 * np.exp(-((wavelengths - 1000) ** 2) / (2 * 50 ** 2))
            peak2 = 0.5 * np.exp(-((wavelengths - 1500) ** 2) / (2 * 80 ** 2))

            # Noise
            noise = 0.02 * np.random.randn(n_wavelengths)

            X[i, :] = baseline + peak1 + peak2 + noise

        # Create regression target
        y = (
            X[:, np.argmin(np.abs(wavelengths - 1000))]
            + X[:, np.argmin(np.abs(wavelengths - 1500))]
            + 0.1 * np.random.randn(n_samples)
        )

        # Split data
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Test: Learned preprocessing + regression
        print("\n" + "=" * 60)
        print("Test: Learned Preprocessing + Regression")
        print("=" * 60)

        model = SpectralPreprocessorWithRegressor(
            n_conv_layers=2,
            n_filters=16,
            kernel_size=11,
            hidden_size=64,
            dropout=0.3,
            learning_rate=1e-3,
            batch_size=32,
        )

        print("Training model (50 epochs)...")
        model.fit(X_train, y_train, epochs=50, verbose=True)

        print("\nPredicting...")
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\nResults:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")

        # Test transform (extract preprocessed spectra)
        print("\nExtracting preprocessed spectra...")
        X_preprocessed = model.transform(X_test)
        print(f"Preprocessed shape: {X_preprocessed.shape}")
        print(f"Original range: [{X_test.min():.4f}, {X_test.max():.4f}]")
        print(f"Preprocessed range: [{X_preprocessed.min():.4f}, {X_preprocessed.max():.4f}]")

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
