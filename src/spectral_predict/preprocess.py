"""Preprocessing transformers for spectral data."""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SNV(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) transformation.

    Normalizes each spectrum (row) by subtracting its mean and dividing by its standard deviation.
    """

    def fit(self, X, y=None):
        """Fit transformer (no-op for SNV)."""
        return self

    def transform(self, X):
        """
        Apply SNV transformation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_snv : ndarray, shape (n_samples, n_features)
            SNV-transformed spectra
        """
        X = np.asarray(X)
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)

        # Avoid division by zero
        stds[stds == 0] = 1.0

        return (X - means) / stds


class SavgolDerivative(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay derivative transformation.

    Parameters
    ----------
    deriv : int, default=1
        Derivative order (1 or 2)
    window : int, default=7
        Window length (must be odd; if even, will be incremented by 1)
    polyorder : int, optional
        Polynomial order. If None, defaults to 2 for deriv=1, 3 for deriv=2
    """

    def __init__(self, deriv=1, window=7, polyorder=None):
        self.deriv = deriv
        self.window = window
        self.polyorder = polyorder

    def fit(self, X, y=None):
        """Fit transformer (no-op for Savgol)."""
        return self

    def transform(self, X):
        """
        Apply Savitzky-Golay derivative.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_deriv : ndarray, shape (n_samples, n_features)
            Derivative spectra
        """
        X = np.asarray(X)

        # Ensure odd window
        window = self.window
        if window % 2 == 0:
            window = window + 1

        # Default polyorder: deriv + 1 for higher derivatives
        polyorder = self.polyorder
        if polyorder is None:
            polyorder_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
            polyorder = polyorder_map.get(self.deriv, self.deriv + 1)

        # Auto-adjust window if too large for feature count
        n_features = X.shape[1]
        original_window = window

        if window > n_features:
            # Find largest valid odd window <= n_features
            max_valid_window = n_features if n_features % 2 == 1 else n_features - 1
            min_valid_window = polyorder + 2
            if min_valid_window % 2 == 0:
                min_valid_window += 1  # Must be odd

            if max_valid_window >= min_valid_window:
                window = max_valid_window
                print(f"WARNING: Auto-adjusted Savitzky-Golay window from {original_window} to {window} "
                      f"(only {n_features} wavelengths available)")
            else:
                # Cannot satisfy constraints - raise helpful error
                raise ValueError(
                    f"Cannot apply derivative with polyorder={polyorder}: "
                    f"requires minimum {min_valid_window} wavelengths, but only {n_features} available. "
                    f"Consider using a lower derivative order or expanding the wavelength range."
                )

        # Validate polyorder constraint
        if window < polyorder + 2:
            raise ValueError(f"Window length ({window}) must be >= polyorder ({polyorder}) + 2")

        # Apply along axis=1 (features)
        X_deriv = savgol_filter(
            X, window_length=window, polyorder=polyorder, deriv=self.deriv, axis=1
        )

        return X_deriv


class SavgolSmooth(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay smoothing without derivatives.

    Applies smoothing to reduce noise while preserving spectral shape.
    Unlike SavgolDerivative, this uses deriv=0 for pure smoothing.

    Parameters
    ----------
    window_length : int, default=17
        Window length (must be odd; if even, will be incremented by 1)
    polyorder : int, default=2
        Polynomial order for fitting
    """

    def __init__(self, window_length=17, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X, y=None):
        """Fit transformer (no-op for smoothing)."""
        return self

    def transform(self, X):
        """
        Apply Savitzky-Golay smoothing.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_smooth : ndarray, shape (n_samples, n_features)
            Smoothed spectra
        """
        X = np.asarray(X)

        # Ensure odd window
        window = self.window_length
        if window % 2 == 0:
            window = window + 1

        # Auto-adjust window if too large for feature count
        n_features = X.shape[1]
        original_window = window
        polyorder = self.polyorder

        if window > n_features:
            # Find largest valid odd window <= n_features
            max_valid_window = n_features if n_features % 2 == 1 else n_features - 1
            min_valid_window = polyorder + 2
            if min_valid_window % 2 == 0:
                min_valid_window += 1  # Must be odd

            if max_valid_window >= min_valid_window:
                window = max_valid_window
                print(f"WARNING: Auto-adjusted smoothing window from {original_window} to {window} "
                      f"(only {n_features} wavelengths available)")
            else:
                # Cannot satisfy constraints - raise helpful error
                raise ValueError(
                    f"Cannot apply smoothing with polyorder={polyorder}: "
                    f"requires minimum {min_valid_window} wavelengths, but only {n_features} available. "
                    f"Consider reducing smoothing polyorder or expanding the wavelength range."
                )

        # Validate polyorder constraint
        if window < polyorder + 2:
            raise ValueError(
                f"Window length ({window}) must be >= polyorder ({polyorder}) + 2"
            )

        # Apply smoothing (deriv=0) along axis=1 (features)
        X_smooth = savgol_filter(
            X, window_length=window, polyorder=self.polyorder, deriv=0, axis=1
        )

        return X_smooth


def build_preprocessing_pipeline(preprocess_name, deriv=None, window=None, polyorder=None,
                                 imbalance_method=None, imbalance_params=None, task_type=None,
                                 interference=None, wavelengths=None, random_state=42,
                                 baseline_method=None, baseline_params=None,
                                 smoothing=False, smoothing_window=17, smoothing_polyorder=2):
    """
    Build a preprocessing pipeline from a configuration.

    Parameters
    ----------
    preprocess_name : str
        One of: 'raw', 'snv', 'deriv', 'snv_deriv', 'deriv_snv'
    deriv : int, optional
        Derivative order (for deriv-based pipelines)
    window : int, optional
        Window size (for deriv-based pipelines)
    polyorder : int, optional
        Polynomial order (for deriv-based pipelines)
    imbalance_method : str, optional
        Imbalance handling method ('smote', 'adasyn', 'binning', etc.)
        If None, no imbalance handling is applied (default behavior)
    imbalance_params : dict, optional
        Parameters for imbalance method (e.g., {'k_neighbors': 5})
    task_type : str, optional
        'classification' or 'regression' (required if imbalance_method is specified)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    baseline_method : str, optional
        Baseline correction method: 'polynomial', 'asls', or None (default)
    baseline_params : dict, optional
        Parameters for baseline correction (e.g., {'degree': 2} for polynomial,
        {'lam': 1e5, 'p': 0.01} for asls)
    smoothing : bool, default=False
        Whether to apply Savitzky-Golay smoothing before other transformations
    smoothing_window : int, default=17
        Window size for smoothing
    smoothing_polyorder : int, default=2
        Polynomial order for smoothing

    Returns
    -------
    steps : list
        List of (name, transformer) tuples

    Notes
    -----
    Processing order: Baseline → Smoothing → SNV/Derivatives → Imbalance handling
    Imbalance handling is applied AFTER spectral preprocessing (SNV/derivatives)
    but BEFORE the model. This ensures resampling operates on preprocessed spectra.
    """
    from sklearn.pipeline import Pipeline

    steps = []

    # Step 0: Interference removal (Phase 3)
    # Applied BEFORE standard preprocessing (SNV/derivatives)
    if interference is not None:
        try:
            from spectral_predict.interference import WavelengthExcluder, MSC, OSC

            # Wavelength Exclusion
            if interference.get('wavelength_exclusion', {}).get('enabled', False):
                exclude_ranges_str = interference['wavelength_exclusion'].get('exclude_ranges', '')
                if exclude_ranges_str and wavelengths is not None:
                    # Parse ranges: "1400-1500, 1900-2000" -> [(1400, 1500), (1900, 2000)]
                    ranges = []
                    for range_str in exclude_ranges_str.split(','):
                        range_str = range_str.strip()
                        if '-' in range_str:
                            try:
                                start, end = range_str.split('-')
                                ranges.append((float(start.strip()), float(end.strip())))
                            except ValueError:
                                print(f"WARNING: Invalid wavelength range '{range_str}', ignoring")

                    if ranges:
                        steps.append(("wl_exclude", WavelengthExcluder(wavelengths, exclude_ranges=ranges)))

            # MSC (Multiplicative Scatter Correction)
            if interference.get('msc', False):
                steps.append(("msc", MSC(reference='mean')))

            # OSC (Orthogonal Signal Correction)
            if interference.get('osc', {}).get('enabled', False):
                n_components = interference['osc'].get('n_components', 2)
                steps.append(("osc", OSC(n_components=n_components)))

            # Phase 4: Advanced interference removal methods
            advanced = interference.get('advanced', {})

            # EPO (External Parameter Orthogonalization)
            if advanced.get('epo', {}).get('enabled', False):
                from spectral_predict.interference import EPO

                epo_settings = advanced['epo']
                library_name = epo_settings.get('library', '')
                interferent_libraries = interference.get('interferent_libraries', {})

                if library_name and library_name in interferent_libraries:
                    lib = interferent_libraries[library_name]

                    # EPO requires interferent spectra during fit(), not __init__()
                    # We need a wrapper to pass X_interferents
                    class EPOWithLibrary(EPO):
                        """Wrapper for EPO that auto-passes interferent library during fit."""
                        def __init__(self, interferent_library, n_components=2, center=True, svd_tol=1e-8):
                            super().__init__(n_components=n_components, center=center, svd_tol=svd_tol)
                            self.interferent_library = interferent_library

                        def fit(self, X, y=None):
                            # Pass interferent library automatically
                            return super().fit(X, y, X_interferents=self.interferent_library)

                    epo = EPOWithLibrary(
                        interferent_library=lib['X'],
                        n_components=epo_settings.get('n_components', 2),
                        center=epo_settings.get('center', True),
                        svd_tol=epo_settings.get('svd_tol', 1e-8)
                    )
                    steps.append(("epo", epo))
                else:
                    print(f"WARNING: EPO enabled but library '{library_name}' not found, skipping EPO")

            # DOSC (Direct Orthogonal Signal Correction)
            if advanced.get('dosc', {}).get('enabled', False):
                from spectral_predict.interference import DOSC

                dosc_settings = advanced['dosc']

                # Parse n_pls_components (can be 'auto' or integer)
                n_pls_comp = dosc_settings.get('n_pls_components', 'auto')
                # Handle string conversion if needed (GUI passes string)
                if isinstance(n_pls_comp, str):
                    if n_pls_comp.lower() == 'auto':
                        n_pls_comp = 'auto'
                    else:
                        try:
                            n_pls_comp = int(n_pls_comp)
                        except ValueError:
                            print(f"WARNING: Invalid n_pls_components '{n_pls_comp}', using 'auto'")
                            n_pls_comp = 'auto'

                # DOSC parameters: n_components, center, n_pls_components
                dosc = DOSC(
                    n_components=dosc_settings.get('n_components', 1),
                    center=dosc_settings.get('center', True),
                    n_pls_components=n_pls_comp
                )
                steps.append(("dosc", dosc))

            # GLSW (Generalized Least Squares Weighting)
            if advanced.get('glsw', {}).get('enabled', False):
                from spectral_predict.interference import GLSW

                glsw_settings = advanced['glsw']

                # Parse n_components (only used for residual method)
                n_comp = glsw_settings.get('n_components', None)
                if isinstance(n_comp, str):
                    if n_comp.lower() in ['auto', 'none', '']:
                        n_comp = None
                    else:
                        try:
                            n_comp = int(n_comp)
                        except ValueError:
                            print(f"WARNING: Invalid GLSW n_components '{n_comp}', using auto")
                            n_comp = None

                # Parse regularization (convert string to float)
                reg = glsw_settings.get('regularization', 1e-6)
                if isinstance(reg, str):
                    try:
                        reg = float(reg)
                    except ValueError:
                        print(f"WARNING: Invalid GLSW regularization '{reg}', using 1e-6")
                        reg = 1e-6

                # GLSW parameters: method, regularization, n_components
                glsw = GLSW(
                    method=glsw_settings.get('method', 'covariance'),
                    regularization=reg,
                    n_components=n_comp
                )
                steps.append(("glsw", glsw))

        except ImportError as e:
            print(f"WARNING: Interference removal module not available ({e}), skipping interference removal")

    # Step 0.5: Baseline correction (applied before smoothing and transforms)
    if baseline_method is not None:
        try:
            from spectral_predict.baseline import BaselineALS, BaselinePolynomial

            params = baseline_params or {}

            if baseline_method == 'polynomial':
                degree = params.get('degree', 2)
                steps.append(("baseline", BaselinePolynomial(degree=degree)))

            elif baseline_method == 'asls':
                lam = params.get('lam', 1e5)
                p = params.get('p', 0.01)
                niter = params.get('niter', 10)
                steps.append(("baseline", BaselineALS(lambda_=lam, p=p, niter=niter)))

            else:
                print(f"WARNING: Unknown baseline method '{baseline_method}', skipping baseline correction")

        except ImportError as e:
            print(f"WARNING: Baseline module not available ({e}), skipping baseline correction")

    # Step 0.6: Savitzky-Golay smoothing (applied after baseline, before transforms)
    if smoothing:
        steps.append(("smooth", SavgolSmooth(
            window_length=smoothing_window,
            polyorder=smoothing_polyorder
        )))

    # Step 1: Spectral preprocessing
    if preprocess_name == "raw":
        pass  # No preprocessing

    elif preprocess_name == "snv":
        steps.append(("snv", SNV()))

    elif preprocess_name == "deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        steps.append(("savgol", savgol))

    elif preprocess_name == "snv_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        steps.append(("snv", SNV()))
        steps.append(("savgol", savgol))

    elif preprocess_name == "deriv_snv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        steps.append(("savgol", savgol))
        steps.append(("snv", SNV()))

    else:
        raise ValueError(f"Unknown preprocess: {preprocess_name}")

    # Step 2: Imbalance handling (optional, added only if user enables it)
    if imbalance_method is not None:
        if task_type is None:
            raise ValueError("task_type must be specified when using imbalance_method")

        # Import imbalance module
        try:
            from spectral_predict.imbalance import build_imbalance_transformer
        except ImportError:
            raise ImportError(
                "Imbalance handling requires the imbalance module. "
                "Ensure src/spectral_predict/imbalance.py is available."
            )

        # Build imbalance transformer
        if imbalance_params is None:
            imbalance_params = {}

        imbalance_transformer = build_imbalance_transformer(
            method=imbalance_method,
            task_type=task_type,
            random_state=random_state,  # CRITICAL for reproducibility
            **imbalance_params
        )

        steps.append(("imbalance", imbalance_transformer))

    return steps
