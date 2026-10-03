"""Y-transform preprocessing wrapper using TransformedTargetRegressor.

Wraps a regressor (or pipeline) so that the target variable is transformed
before training and inverse-transformed for predictions. Supports log, log1p,
sqrt, Box-Cox, and Yeo-Johnson transforms.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

# Accepted spellings -> canonical method name. The GUI combobox shows 'Box-Cox' and
# 'Yeo-Johnson'; lower-casing alone gives 'box-cox', which wrap() used to reject.
_METHOD_ALIASES = {
    "": "none",
    "none": "none",
    "log": "log",
    "log1p": "log1p",
    "sqrt": "sqrt",
    "boxcox": "boxcox",
    "box-cox": "boxcox",
    "box_cox": "boxcox",
    "box cox": "boxcox",
    "yeo-johnson": "yeo-johnson",
    "yeojohnson": "yeo-johnson",
    "yeo_johnson": "yeo-johnson",
    "yeo johnson": "yeo-johnson",
}


def normalize_y_transform_method(method: Optional[str]) -> str:
    """Return the canonical Y-transform name for any accepted spelling.

    Args:
        method: A method name as shown in the GUI ('Box-Cox', 'Yeo-Johnson', 'Log', ...)
            or a canonical name. ``None`` means no transform.

    Returns:
        One of ``YTransformWrapper.METHODS``.

    Raises:
        ValueError: If the name is not a supported transform.
    """
    if method is None:
        return "none"
    key = str(method).strip().lower()
    try:
        return _METHOD_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown Y-transform method: '{method}'. "
            f"Choose from: {', '.join(YTransformWrapper.METHODS)}"
        ) from None


def replace_fitted_regressor(
    ttr: TransformedTargetRegressor, fitted_regressor: BaseEstimator
) -> TransformedTargetRegressor:
    """Return a copy of a fitted TTR whose fitted inner regressor is replaced.

    Used when saving: the TTR was fitted around a training pipeline (which may hold
    resampling steps), but only the prediction part (e.g. ``[scaler, model]`` or the
    bare model) should be persisted. The fitted target transformer is kept as is, so
    the copy predicts exactly like the original whenever ``fitted_regressor`` predicts
    like ``ttr.regressor_``.

    Args:
        ttr: A fitted ``TransformedTargetRegressor``.
        fitted_regressor: Fitted estimator to use as ``regressor_`` of the copy.

    Returns:
        A shallow copy of ``ttr`` with ``regressor_`` replaced and ``regressor`` set to
        an unfitted clone of ``fitted_regressor`` (so ``clone()`` of the copy refits the
        same prediction model).

    Raises:
        TypeError: If ``ttr`` is not a ``TransformedTargetRegressor``.
    """
    if not isinstance(ttr, TransformedTargetRegressor):
        raise TypeError(f"Expected a TransformedTargetRegressor, got {type(ttr).__name__}")
    new = copy.copy(ttr)
    new.regressor = clone(fitted_regressor)
    new.regressor_ = fitted_regressor
    return new


class YTransformWrapper:
    """Static utility for wrapping regressors with target transforms."""

    METHODS = ("none", "log", "log1p", "sqrt", "boxcox", "yeo-johnson")

    @staticmethod
    def wrap(regressor: BaseEstimator, method: str) -> BaseEstimator:
        """Wrap a regressor with TransformedTargetRegressor.

        Parameters
        ----------
        regressor : BaseEstimator
            The sklearn regressor or pipeline to wrap.
        method : str
            Transform method: 'none', 'log', 'log1p', 'sqrt', 'boxcox' (or 'Box-Cox'),
            'yeo-johnson'. Case-insensitive.

        Returns
        -------
        BaseEstimator
            The original regressor (if method is 'none') or a
            TransformedTargetRegressor wrapping it.
        """
        method_lower = normalize_y_transform_method(method)

        if method_lower == "none":
            return regressor

        if method_lower == "log":
            return TransformedTargetRegressor(
                regressor=regressor,
                func=np.log,
                inverse_func=np.exp,
            )

        if method_lower == "log1p":
            return TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
            )

        if method_lower == "sqrt":
            return TransformedTargetRegressor(
                regressor=regressor,
                func=np.sqrt,
                inverse_func=np.square,
            )

        if method_lower == "boxcox":
            # PowerTransformer handles lambda fitting and serialization
            return TransformedTargetRegressor(
                regressor=regressor,
                transformer=PowerTransformer(method="box-cox", standardize=False),
            )

        # 'yeo-johnson' (normalize_y_transform_method admits nothing else)
        return TransformedTargetRegressor(
            regressor=regressor,
            transformer=PowerTransformer(method="yeo-johnson", standardize=False),
        )

    @staticmethod
    def _get_transformer(method: str):
        """Get a sklearn-compatible transformer for manual y-transform.

        Used when early stopping requires manual y-transform instead of
        wrapping with TransformedTargetRegressor.

        Returns a transformer with fit_transform/transform/inverse_transform methods.
        """
        from sklearn.preprocessing import FunctionTransformer

        method_lower = normalize_y_transform_method(method)

        if method_lower == "log":
            return FunctionTransformer(func=np.log, inverse_func=np.exp)

        if method_lower == "log1p":
            return FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

        if method_lower == "sqrt":
            return FunctionTransformer(func=np.sqrt, inverse_func=np.square)

        if method_lower == "boxcox":
            return PowerTransformer(method="box-cox", standardize=False)

        if method_lower == "yeo-johnson":
            return PowerTransformer(method="yeo-johnson", standardize=False)

        raise ValueError(f"Unknown Y-transform method: '{method}'")

    @staticmethod
    def validate(y: np.ndarray, method: str) -> Optional[str]:
        """Check if y is compatible with the chosen transform.

        Returns
        -------
        str or None
            Error message if incompatible, None if OK.
        """
        try:
            method_lower = normalize_y_transform_method(method)
        except ValueError as e:
            return str(e)

        if method_lower == "none":
            return None

        y = np.asarray(y, dtype=float).ravel()

        if len(y) == 0:
            return "No target values to transform."

        if np.any(~np.isfinite(y)):
            return "Target contains NaN or infinite values."

        if method_lower == "log":
            if np.any(y <= 0):
                return (
                    "Log transform requires all y > 0. "
                    "Try 'Log1p' (handles zeros) or 'Yeo-Johnson' (handles negatives)."
                )

        if method_lower == "log1p":
            if np.any(y <= -1):
                return (
                    "Log1p transform requires all y > -1. "
                    "Try 'Yeo-Johnson' for data with large negative values."
                )

        if method_lower == "sqrt":
            if np.any(y < 0):
                return (
                    "Sqrt transform requires all y >= 0. "
                    "Try 'Yeo-Johnson' for data with negative values."
                )

        if method_lower == "boxcox":
            if np.any(y <= 0):
                return (
                    "Box-Cox transform requires all y > 0. "
                    "Try 'Yeo-Johnson' which handles zero and negative values."
                )

        return None
