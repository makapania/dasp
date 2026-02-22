"""Y-transform preprocessing wrapper using TransformedTargetRegressor.

Wraps a regressor (or pipeline) so that the target variable is transformed
before training and inverse-transformed for predictions. Supports log, log1p,
sqrt, Box-Cox, and Yeo-Johnson transforms.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer


class YTransformWrapper:
    """Static utility for wrapping regressors with target transforms."""

    METHODS = ('none', 'log', 'log1p', 'sqrt', 'boxcox', 'yeo-johnson')

    @staticmethod
    def wrap(regressor: BaseEstimator, method: str) -> BaseEstimator:
        """Wrap a regressor with TransformedTargetRegressor.

        Parameters
        ----------
        regressor : BaseEstimator
            The sklearn regressor or pipeline to wrap.
        method : str
            Transform method: 'none', 'log', 'log1p', 'sqrt', 'boxcox', 'yeo-johnson'.

        Returns
        -------
        BaseEstimator
            The original regressor (if method is 'none') or a
            TransformedTargetRegressor wrapping it.
        """
        method_lower = method.lower().replace('-', '-').strip()

        if method_lower in ('none', ''):
            return regressor

        if method_lower == 'log':
            return TransformedTargetRegressor(
                regressor=regressor,
                func=np.log,
                inverse_func=np.exp,
            )

        if method_lower == 'log1p':
            return TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
            )

        if method_lower == 'sqrt':
            return TransformedTargetRegressor(
                regressor=regressor,
                func=np.sqrt,
                inverse_func=np.square,
            )

        if method_lower == 'boxcox':
            # PowerTransformer handles lambda fitting and serialization
            return TransformedTargetRegressor(
                regressor=regressor,
                transformer=PowerTransformer(method='box-cox', standardize=False),
            )

        if method_lower == 'yeo-johnson':
            return TransformedTargetRegressor(
                regressor=regressor,
                transformer=PowerTransformer(method='yeo-johnson', standardize=False),
            )

        raise ValueError(
            f"Unknown Y-transform method: '{method}'. "
            f"Choose from: {', '.join(YTransformWrapper.METHODS)}"
        )

    @staticmethod
    def _get_transformer(method: str):
        """Get a sklearn-compatible transformer for manual y-transform.

        Used when early stopping requires manual y-transform instead of
        wrapping with TransformedTargetRegressor.

        Returns a transformer with fit_transform/transform/inverse_transform methods.
        """
        from sklearn.preprocessing import FunctionTransformer

        method_lower = method.lower().strip()

        if method_lower == 'log':
            return FunctionTransformer(func=np.log, inverse_func=np.exp)

        if method_lower == 'log1p':
            return FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

        if method_lower == 'sqrt':
            return FunctionTransformer(func=np.sqrt, inverse_func=np.square)

        if method_lower in ('boxcox', 'box-cox'):
            return PowerTransformer(method='box-cox', standardize=False)

        if method_lower == 'yeo-johnson':
            return PowerTransformer(method='yeo-johnson', standardize=False)

        raise ValueError(f"Unknown Y-transform method: '{method}'")

    @staticmethod
    def validate(y: np.ndarray, method: str) -> Optional[str]:
        """Check if y is compatible with the chosen transform.

        Returns
        -------
        str or None
            Error message if incompatible, None if OK.
        """
        y = np.asarray(y, dtype=float).ravel()
        method_lower = method.lower().strip()

        if method_lower in ('none', ''):
            return None

        if len(y) == 0:
            return "No target values to transform."

        if np.any(~np.isfinite(y)):
            return "Target contains NaN or infinite values."

        if method_lower == 'log':
            if np.any(y <= 0):
                return (
                    "Log transform requires all y > 0. "
                    "Try 'Log1p' (handles zeros) or 'Yeo-Johnson' (handles negatives)."
                )

        if method_lower == 'log1p':
            if np.any(y < -1):
                return (
                    "Log1p transform requires all y > -1. "
                    "Try 'Yeo-Johnson' for data with large negative values."
                )

        if method_lower == 'sqrt':
            if np.any(y < 0):
                return (
                    "Sqrt transform requires all y >= 0. "
                    "Try 'Yeo-Johnson' for data with negative values."
                )

        if method_lower == 'boxcox':
            if np.any(y <= 0):
                return (
                    "Box-Cox transform requires all y > 0. "
                    "Try 'Yeo-Johnson' which handles zero and negative values."
                )

        return None
