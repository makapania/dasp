"""Persistent preprocessing wrappers for ensemble base models.

The GUI's ensemble reconstruction wraps base models in these classes, and saved
ensembles (.dasp) pickle them. They live in this importable module, not in the GUI
script, so a saved model refers to ``spectral_predict.model_wrappers.<Class>`` and loads
in any process. Files saved before the move refer to ``__main__.<Class>`` or
``spectral_predict_gui_optimized.<Class>``; ``model_io`` maps those names here
(:data:`LEGACY_PICKLE_NAMES`).

Internal module: not part of the declared composition surface.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _build_transform_from_config(config: dict):
    """
    Build a preprocessing transform function from configuration.

    This is factored out so wrapper classes can recreate transforms after cloning.
    Imports are done inside to avoid circular dependencies.
    """
    from spectral_predict.preprocess import SavgolDerivative, SNV, SavgolSmooth
    from spectral_predict.baseline import BaselinePolynomial, BaselineALS, BaselineAirPLS

    def transform(X):
        import numpy as np

        X_out = np.asarray(X, dtype=np.float64)

        # Apply smoothing first (if enabled)
        if config.get("smooth"):
            smoother = SavgolSmooth(window_length=config["smooth"], polyorder=2)
            X_out = smoother.fit_transform(X_out)

        # Apply baseline correction
        baseline = config.get("baseline")
        bl_params = config.get("baseline_params", {})
        if baseline == "polynomial":
            bl = BaselinePolynomial(degree=bl_params.get("degree", 2))
            X_out = bl.fit_transform(X_out)
        elif baseline == "als":
            bl = BaselineALS(
                lambda_=bl_params.get("lam", 1e5),
                p=bl_params.get("p", 0.01),
                niter=10,
            )
            X_out = bl.fit_transform(X_out)
        elif baseline == "airpls":
            bl = BaselineAirPLS(
                lam=bl_params.get("lam", 1e5),
                max_iter=15,
            )
            X_out = bl.fit_transform(X_out)

        # Apply main preprocessing
        pt = config.get("type", "raw")
        w = config.get("window", 15)

        if pt == "raw":
            pass
        elif pt == "snv":
            X_out = SNV().fit_transform(X_out)
        elif pt == "deriv1":
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
        elif pt == "deriv2":
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
        elif pt == "deriv3":
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
        elif pt == "deriv4":
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
        elif pt == "snv_deriv1":
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
        elif pt == "snv_deriv2":
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
        elif pt == "snv_deriv3":
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
        elif pt == "snv_deriv4":
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
        elif pt == "deriv1_snv":
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == "deriv2_snv":
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == "deriv3_snv":
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == "deriv4_snv":
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)

        return X_out

    return transform


def _match_wavelengths_normalized(requested_cols, available_columns, precision=1):
    """
    Match wavelengths using exact precision matching after normalization.

    This function solves the CARS wavelength matching bug where tolerance-based
    matching (±0.5nm) could match wrong wavelengths or fail silently.

    Instead of tolerance-based matching, this:
    1. Rounds both requested and available wavelengths to the same precision
    2. Matches on the normalized string representation
    3. Tries multiple precision levels if initial match fails

    Args:
        requested_cols: List of wavelength column names (can be strings or floats)
        available_columns: Column names from the new DataFrame
        precision: Decimal places to round to (default 1)

    Returns:
        List of matched column names from available_columns

    Raises:
        KeyError: If any wavelength cannot be matched
    """
    # Build lookup dict: normalized wavelength string -> original column name
    col_lookup = {}
    col_names = list(available_columns)

    for col in col_names:
        try:
            col_float = float(col)
            normalized_key = f"{round(col_float, precision):.{precision}f}"
            if normalized_key not in col_lookup:
                col_lookup[normalized_key] = col
        except ValueError, TypeError:
            continue

    # Match each requested wavelength
    matched_by_index = {}  # index -> matched column
    missing_by_index = {}  # index -> wavelength

    for idx, req_col in enumerate(requested_cols):
        try:
            req_float = float(req_col)
            normalized_key = f"{round(req_float, precision):.{precision}f}"
            if normalized_key in col_lookup:
                matched_by_index[idx] = col_lookup[normalized_key]
            else:
                missing_by_index[idx] = req_col
        except ValueError, TypeError:
            # Non-numeric column - try direct string match
            if req_col in col_names:
                matched_by_index[idx] = req_col
            else:
                missing_by_index[idx] = req_col

    # Try different precision levels for missing wavelengths
    if missing_by_index:
        for alt_precision in [0, 2, 3]:
            if alt_precision == precision:
                continue

            # Rebuild lookup at alternative precision
            col_lookup_alt = {}
            for col in col_names:
                try:
                    col_float = float(col)
                    normalized_key = f"{round(col_float, alt_precision):.{alt_precision}f}"
                    if normalized_key not in col_lookup_alt:
                        col_lookup_alt[normalized_key] = col
                except ValueError, TypeError:
                    continue

            # Try matching still-missing wavelengths
            still_missing = {}
            for idx, wl in missing_by_index.items():
                try:
                    wl_float = float(wl)
                    normalized_key = f"{round(wl_float, alt_precision):.{alt_precision}f}"
                    if normalized_key in col_lookup_alt:
                        matched_by_index[idx] = col_lookup_alt[normalized_key]
                    else:
                        still_missing[idx] = wl
                except ValueError, TypeError:
                    still_missing[idx] = wl

            missing_by_index = still_missing
            if not missing_by_index:
                break

    # Report any still-missing wavelengths
    if missing_by_index:
        missing_wls = list(missing_by_index.values())
        raise KeyError(
            f"Could not match all wavelength columns. "
            f"Missing {len(missing_wls)} wavelengths: {missing_wls[:5]}{'...' if len(missing_wls) > 5 else ''}"
        )

    # Return matched columns in original order
    return [matched_by_index[i] for i in range(len(requested_cols))]


def _subset_wavelength_columns(X, wavelength_cols, all_columns=None):
    """Select ``wavelength_cols`` from X, keeping their order.

    DataFrames are subset by column name (normalised-precision fallback for float vs
    string names). Arrays are subset by position when ``all_columns`` describes their
    full width, passed through when they already have the subset's width, and rejected
    otherwise: silently feeding a full-width array to a subset model is never right.
    """
    if wavelength_cols is None:
        return X
    if hasattr(X, "loc"):
        try:
            return X[wavelength_cols]
        except KeyError:
            matching_cols = _match_wavelengths_normalized(wavelength_cols, X.columns, precision=1)
            return X[matching_cols]
    X_arr = np.asarray(X)
    n_cols = X_arr.shape[1] if X_arr.ndim > 1 else 1
    if n_cols == len(wavelength_cols):
        # Contract: an array exactly as wide as the subset is taken to be ALREADY subset,
        # in wavelength_cols order. Arrays carry no column names, so this cannot be
        # checked; callers holding the full spectrum must pass all of it.
        return X_arr
    if all_columns is not None and n_cols == len(all_columns):
        all_cols = list(all_columns)
        return X_arr[:, [all_cols.index(c) for c in wavelength_cols]]
    raise ValueError(
        f"Array with {n_cols} columns cannot be mapped to the {len(wavelength_cols)} "
        f"selected wavelengths (full spectrum: "
        f"{len(all_columns) if all_columns is not None else 'unknown'} columns)."
    )


class WavelengthSubsetWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper that applies wavelength subsetting during fit and predict.

    This wrapper is clonable via sklearn.clone() because it inherits from BaseEstimator
    and implements get_params/set_params properly.

    FIXED: Uses exact precision matching instead of tolerance-based matching (±0.5nm)
    to solve the CARS wavelength mismatch bug in ensembles.
    """

    def __init__(self, pipeline=None, wavelength_cols=None, all_columns=None):
        self.pipeline = pipeline
        self.wavelength_cols = wavelength_cols
        # Full column list the subset refers to: lets a numpy matrix of the full
        # spectrum be subset by position (ensemble CV, validation arrays).
        self.all_columns = all_columns

    def _subset(self, X):
        """
        Subset X to selected wavelengths using exact precision matching.

        This method solves the CARS wavelength matching bug where tolerance-based
        matching (±0.5nm) could match wrong wavelengths or fail silently.
        """
        return _subset_wavelength_columns(
            X, self.wavelength_cols, getattr(self, "all_columns", None)
        )

    def fit(self, X, y):
        X_subset = self._subset(X)
        self.pipeline.fit(X_subset, y)
        return self

    def predict(self, X):
        X_subset = self._subset(X)
        return self.pipeline.predict(X_subset)

    def get_params(self, deep=True):
        return {
            "pipeline": self.pipeline,
            "wavelength_cols": self.wavelength_cols,
            "all_columns": getattr(self, "all_columns", None),
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class GAPreprocessWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for GA/NSGA preprocessing.

    Stores preprocessing config (not the transform function) so it can be cloned.
    The transform function is recreated from config when needed.
    """

    def __init__(self, pipeline=None, preprocess_config=None):
        self.pipeline = pipeline
        self.preprocess_config = preprocess_config
        self._transform = None

    @property
    def transform(self):
        """Lazily create transform function from config."""
        if getattr(self, "_transform", None) is None and self.preprocess_config:
            self._transform = _build_transform_from_config(self.preprocess_config)
        return self._transform

    def __getstate__(self):
        # The cached transform is a local closure, which cannot be pickled (saving an
        # ensemble with this member failed). It is rebuilt from preprocess_config.
        state = dict(super().__getstate__())  # a copy: never clear the live cache
        state["_transform"] = None
        return state

    def fit(self, X, y):
        X_preproc = self.transform(X.values if hasattr(X, "values") else X)
        self.pipeline.fit(X_preproc, y)
        return self

    def predict(self, X):
        X_preproc = self.transform(X.values if hasattr(X, "values") else X)
        return self.pipeline.predict(X_preproc)

    def get_params(self, deep=True):
        return {"pipeline": self.pipeline, "preprocess_config": self.preprocess_config}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Reset transform cache if config changes
        if "preprocess_config" in params:
            self._transform = None
        return self


class CombinedPreprocessWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for combined preprocessing + wavelength selection (NSGA-II).

    Stores preprocessing config and column info (not the transform function) so it can be cloned.
    The transform function is recreated from config when needed.
    """

    def __init__(
        self, pipeline=None, preprocess_config=None, wavelength_cols=None, all_columns=None
    ):
        self.pipeline = pipeline
        self.preprocess_config = preprocess_config
        self.wavelength_cols = wavelength_cols
        self.all_columns = all_columns
        self._transform = None
        self._col_indices = None

    @property
    def transform(self):
        """Lazily create transform function from config."""
        if getattr(self, "_transform", None) is None and self.preprocess_config:
            self._transform = _build_transform_from_config(self.preprocess_config)
        return self._transform

    def __getstate__(self):
        # The cached transform is a local closure, which cannot be pickled (saving an
        # ensemble with this member failed). It is rebuilt from preprocess_config.
        state = dict(super().__getstate__())  # a copy: never clear the live cache
        state["_transform"] = None
        return state

    @property
    def col_indices(self):
        """Lazily compute column indices."""
        if (
            self._col_indices is None
            and self.all_columns is not None
            and self.wavelength_cols is not None
        ):
            all_cols_list = list(self.all_columns)
            self._col_indices = [all_cols_list.index(c) for c in self.wavelength_cols]
        return self._col_indices

    def _preprocess_and_subset(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        X_preproc = self.transform(X_arr)
        # Subset to selected wavelengths (after preprocessing)
        return X_preproc[:, self.col_indices]

    def fit(self, X, y):
        X_processed = self._preprocess_and_subset(X)
        self.pipeline.fit(X_processed, y)
        return self

    def predict(self, X):
        X_processed = self._preprocess_and_subset(X)
        return self.pipeline.predict(X_processed)

    def get_params(self, deep=True):
        return {
            "pipeline": self.pipeline,
            "preprocess_config": self.preprocess_config,
            "wavelength_cols": self.wavelength_cols,
            "all_columns": self.all_columns,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Reset caches if relevant params change
        if "preprocess_config" in params:
            self._transform = None
        if "all_columns" in params or "wavelength_cols" in params:
            self._col_indices = None
        return self


# ==================== CLASSIFIER WRAPPERS ====================
# These are classification-specific versions of the wrappers above.
# They inherit ClassifierMixin, expose predict_proba(), and have a classes_ property.


class WavelengthSubsetClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible classifier wrapper that applies wavelength subsetting during fit and predict.

    This wrapper is clonable via sklearn.clone() because it inherits from BaseEstimator
    and implements get_params/set_params properly.

    FIXED: Uses exact precision matching instead of tolerance-based matching (±0.5nm)
    to solve the CARS wavelength mismatch bug in ensembles.
    """

    def __init__(self, pipeline=None, wavelength_cols=None, all_columns=None):
        self.pipeline = pipeline
        self.wavelength_cols = wavelength_cols
        # Full column list the subset refers to: lets a numpy matrix of the full
        # spectrum be subset by position (ensemble CV, validation arrays).
        self.all_columns = all_columns

    def _subset(self, X):
        """
        Subset X to selected wavelengths using exact precision matching.

        This method solves the CARS wavelength matching bug where tolerance-based
        matching (±0.5nm) could match wrong wavelengths or fail silently.
        """
        return _subset_wavelength_columns(
            X, self.wavelength_cols, getattr(self, "all_columns", None)
        )

    def fit(self, X, y):
        X_subset = self._subset(X)
        self.pipeline.fit(X_subset, y)
        return self

    def predict(self, X):
        X_subset = self._subset(X)
        return self.pipeline.predict(X_subset)

    def predict_proba(self, X):
        """Return probability predictions for classification."""
        X_subset = self._subset(X)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_subset)
        raise AttributeError(f"{type(self.pipeline).__name__} does not support predict_proba")

    @property
    def classes_(self):
        """Return classes from the underlying classifier."""
        if hasattr(self.pipeline, "classes_"):
            return self.pipeline.classes_
        raise AttributeError(f"{type(self.pipeline).__name__} does not have classes_ attribute")

    def get_params(self, deep=True):
        return {
            "pipeline": self.pipeline,
            "wavelength_cols": self.wavelength_cols,
            "all_columns": getattr(self, "all_columns", None),
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class GAPreprocessClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible classifier wrapper for GA/NSGA preprocessing.

    Stores preprocessing config (not the transform function) so it can be cloned.
    The transform function is recreated from config when needed.
    """

    def __init__(self, pipeline=None, preprocess_config=None):
        self.pipeline = pipeline
        self.preprocess_config = preprocess_config
        self._transform = None

    @property
    def transform(self):
        """Lazily create transform function from config."""
        if getattr(self, "_transform", None) is None and self.preprocess_config:
            self._transform = _build_transform_from_config(self.preprocess_config)
        return self._transform

    def __getstate__(self):
        # The cached transform is a local closure, which cannot be pickled (saving an
        # ensemble with this member failed). It is rebuilt from preprocess_config.
        state = dict(super().__getstate__())  # a copy: never clear the live cache
        state["_transform"] = None
        return state

    def fit(self, X, y):
        X_preproc = self.transform(X.values if hasattr(X, "values") else X)
        self.pipeline.fit(X_preproc, y)
        return self

    def predict(self, X):
        X_preproc = self.transform(X.values if hasattr(X, "values") else X)
        return self.pipeline.predict(X_preproc)

    def predict_proba(self, X):
        """Return probability predictions for classification."""
        X_preproc = self.transform(X.values if hasattr(X, "values") else X)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_preproc)
        raise AttributeError(f"{type(self.pipeline).__name__} does not support predict_proba")

    @property
    def classes_(self):
        """Return classes from the underlying classifier."""
        if hasattr(self.pipeline, "classes_"):
            return self.pipeline.classes_
        raise AttributeError(f"{type(self.pipeline).__name__} does not have classes_ attribute")

    def get_params(self, deep=True):
        return {"pipeline": self.pipeline, "preprocess_config": self.preprocess_config}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Reset transform cache if config changes
        if "preprocess_config" in params:
            self._transform = None
        return self


class CombinedPreprocessClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible classifier wrapper for combined preprocessing + wavelength selection (NSGA-II).

    Stores preprocessing config and column info (not the transform function) so it can be cloned.
    The transform function is recreated from config when needed.
    """

    def __init__(
        self, pipeline=None, preprocess_config=None, wavelength_cols=None, all_columns=None
    ):
        self.pipeline = pipeline
        self.preprocess_config = preprocess_config
        self.wavelength_cols = wavelength_cols
        self.all_columns = all_columns
        self._transform = None
        self._col_indices = None

    @property
    def transform(self):
        """Lazily create transform function from config."""
        if getattr(self, "_transform", None) is None and self.preprocess_config:
            self._transform = _build_transform_from_config(self.preprocess_config)
        return self._transform

    def __getstate__(self):
        # The cached transform is a local closure, which cannot be pickled (saving an
        # ensemble with this member failed). It is rebuilt from preprocess_config.
        state = dict(super().__getstate__())  # a copy: never clear the live cache
        state["_transform"] = None
        return state

    @property
    def col_indices(self):
        """Lazily compute column indices."""
        if (
            self._col_indices is None
            and self.all_columns is not None
            and self.wavelength_cols is not None
        ):
            all_cols_list = list(self.all_columns)
            self._col_indices = [all_cols_list.index(c) for c in self.wavelength_cols]
        return self._col_indices

    def _preprocess_and_subset(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        X_preproc = self.transform(X_arr)
        # Subset to selected wavelengths (after preprocessing)
        return X_preproc[:, self.col_indices]

    def fit(self, X, y):
        X_processed = self._preprocess_and_subset(X)
        self.pipeline.fit(X_processed, y)
        return self

    def predict(self, X):
        X_processed = self._preprocess_and_subset(X)
        return self.pipeline.predict(X_processed)

    def predict_proba(self, X):
        """Return probability predictions for classification."""
        X_processed = self._preprocess_and_subset(X)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_processed)
        raise AttributeError(f"{type(self.pipeline).__name__} does not support predict_proba")

    @property
    def classes_(self):
        """Return classes from the underlying classifier."""
        if hasattr(self.pipeline, "classes_"):
            return self.pipeline.classes_
        raise AttributeError(f"{type(self.pipeline).__name__} does not have classes_ attribute")

    def get_params(self, deep=True):
        return {
            "pipeline": self.pipeline,
            "preprocess_config": self.preprocess_config,
            "wavelength_cols": self.wavelength_cols,
            "all_columns": self.all_columns,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Reset caches if relevant params change
        if "preprocess_config" in params:
            self._transform = None
        if "all_columns" in params or "wavelength_cols" in params:
            self._col_indices = None
        return self


# Class names that pre-move pickles may reference under ``__main__`` (GUI run as a
# script) or ``spectral_predict_gui_optimized`` (GUI imported as a module).
LEGACY_PICKLE_NAMES = (
    "WavelengthSubsetWrapper",
    "GAPreprocessWrapper",
    "CombinedPreprocessWrapper",
    "WavelengthSubsetClassifierWrapper",
    "GAPreprocessClassifierWrapper",
    "CombinedPreprocessClassifierWrapper",
)
