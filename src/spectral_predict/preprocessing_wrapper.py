"""
Preprocessing configuration wrapper for ensemble models.

This module provides PreprocessorConfig, which reconstructs preprocessing
from stored configuration instead of requiring fitted preprocessor objects.
This is essential for ensembles where each base model may use different
preprocessing methods and wavelength subsets.
"""

import numpy as np
from typing import Optional, List, Union
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessorConfig(BaseEstimator, TransformerMixin):
    """
    Reconstructs and applies preprocessing from stored configuration.

    This class allows ensemble models to apply per-model preprocessing
    without storing fitted preprocessor objects. It reconstructs the
    preprocessing pipeline from configuration parameters and applies
    wavelength subsetting if specified.

    Parameters
    ----------
    preprocess_name : str
        Preprocessing method name (e.g., 'raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2')
    deriv : int, optional
        Derivative order (0, 1, or 2). Inferred from preprocess_name if not provided.
    window : int, optional
        Savitzky-Golay window size. Default is 15.
    polyorder : int, optional
        Savitzky-Golay polynomial order. Default is 2.
    wavelengths : list of float, optional
        Selected wavelengths to subset (from all_vars column).
        If None, uses all wavelengths.
    all_wavelengths : list of float, optional
        Full wavelength array (needed for derivative calculation before subsetting).
        If None, assumes wavelengths is already the full range.
    """

    def __init__(
        self,
        preprocess_name: str,
        deriv: Optional[int] = None,
        window: Optional[int] = None,
        polyorder: Optional[int] = None,
        wavelengths: Optional[List[float]] = None,
        all_wavelengths: Optional[List[float]] = None
    ):
        self.preprocess_name = preprocess_name
        self.deriv = deriv
        self.window = window if window is not None else 15
        self.polyorder = polyorder if polyorder is not None else 2
        self.wavelengths = wavelengths
        self.all_wavelengths = all_wavelengths

        # Parse preprocessing name to determine operations
        self._parse_preprocessing_name()

        # Pre-compute wavelength subset indices if applicable
        self.wavelength_indices_ = None
        if self.wavelengths is not None and self.all_wavelengths is not None:
            self._compute_wavelength_indices()

    def _parse_preprocessing_name(self):
        """Parse preprocessing name to determine SNV and derivative settings."""
        name = self.preprocess_name.lower()

        # Determine if SNV should be applied
        self.apply_snv = 'snv' in name

        # Determine derivative order if not explicitly provided
        if self.deriv is None:
            if 'sg2' in name or 'deriv2' in name:
                self.deriv = 2
            elif 'sg1' in name or 'deriv1' in name:
                self.deriv = 1
            else:
                self.deriv = 0

    def _compute_wavelength_indices(self):
        """Pre-compute indices for wavelength subsetting."""
        if self.wavelengths is None or self.all_wavelengths is None:
            self.wavelength_indices_ = None
            return

        all_wl_array = np.array(self.all_wavelengths, dtype=np.float64)
        selected_wl = np.array(self.wavelengths, dtype=np.float64)

        # Find matching indices (with tolerance for float comparison)
        indices = []
        for wl in selected_wl:
            idx = np.argmin(np.abs(all_wl_array - wl))
            if np.abs(all_wl_array[idx] - wl) < 0.5:  # Tolerance
                indices.append(idx)

        self.wavelength_indices_ = np.array(indices) if indices else None

    def fit(self, X, y=None):
        """
        Fit method (no-op since preprocessing is stateless).

        Parameters
        ----------
        X : array-like
            Training data
        y : array-like, optional
            Target values (unused)

        Returns
        -------
        self
        """
        return self

    def transform(self, X):
        """
        Apply preprocessing and wavelength subsetting.

        For derivative preprocessing with wavelength subsetting, the order is:
        1. Apply SNV (if enabled) to full spectrum
        2. Apply derivative to full spectrum
        3. Subset to selected wavelengths

        Parameters
        ----------
        X : array-like of shape (n_samples, n_wavelengths)
            Input spectral data

        Returns
        -------
        X_processed : ndarray of shape (n_samples, n_selected_wavelengths)
            Preprocessed and optionally subsetted data
        """
        X_work = np.asarray(X, dtype=np.float64)

        # Step 1: Apply SNV if enabled
        if self.apply_snv:
            X_work = self._apply_snv(X_work)

        # Step 2: Apply derivative if enabled
        if self.deriv > 0:
            X_work = self._apply_derivative(X_work, self.deriv)

        # Step 3: Apply wavelength subsetting if configured
        if self.wavelength_indices_ is not None:
            X_work = X_work[:, self.wavelength_indices_]

        return X_work

    def _apply_snv(self, X):
        """Apply Standard Normal Variate transformation."""
        from src.spectral_predict.preprocess import SNV
        snv = SNV()
        return snv.fit_transform(X)

    def _apply_derivative(self, X, deriv_order):
        """Apply Savitzky-Golay derivative."""
        from src.spectral_predict.preprocess import SavgolDerivative
        sg = SavgolDerivative(
            deriv=deriv_order,
            window=self.window,
            polyorder=self.polyorder
        )
        return sg.fit_transform(X)

    def __repr__(self):
        """String representation for debugging."""
        parts = [f"PreprocessorConfig(preprocess='{self.preprocess_name}'"]
        if self.apply_snv:
            parts.append("SNV=True")
        if self.deriv > 0:
            parts.append(f"deriv={self.deriv}, window={self.window}")
        if self.wavelengths is not None:
            parts.append(f"n_wavelengths={len(self.wavelengths)}")
        return ", ".join(parts) + ")"

    def get_config(self):
        """
        Get configuration as dictionary for serialization.

        Returns
        -------
        dict
            Configuration dictionary
        """
        return {
            'preprocess_name': self.preprocess_name,
            'deriv': self.deriv,
            'window': self.window,
            'polyorder': self.polyorder,
            'wavelengths': self.wavelengths,
            'all_wavelengths': self.all_wavelengths,
            'apply_snv': self.apply_snv,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create PreprocessorConfig from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from get_config()

        Returns
        -------
        PreprocessorConfig
            Reconstructed preprocessor config
        """
        return cls(
            preprocess_name=config['preprocess_name'],
            deriv=config.get('deriv'),
            window=config.get('window'),
            polyorder=config.get('polyorder'),
            wavelengths=config.get('wavelengths'),
            all_wavelengths=config.get('all_wavelengths')
        )
