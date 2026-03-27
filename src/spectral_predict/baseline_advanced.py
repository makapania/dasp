"""Advanced baseline correction via pybaselines library.

Provides a sklearn-compatible transformer wrapping 15+ pybaselines algorithms,
organized into tiers (Recommended, Specialized, Experimental) with a curated
registry of default parameters and GUI metadata.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.linalg import LinAlgError
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

try:
    from pybaselines import Baseline as _PyBaseline

    HAS_PYBASELINES = True
except ImportError:
    HAS_PYBASELINES = False


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ADVANCED_ALGORITHMS: dict[str, dict] = {
    # --- Tier 1: Recommended ---
    "arpls": {
        "display_name": "arPLS",
        "category": "whittaker",
        "tier": 1,
        "default_params": {"lam": 1e5},
        "param_info": {
            "lam": {
                "range": (1e2, 1e9),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e2", "2e2", "5e2", "1e3", "2e3", "5e3",
                    "1e4", "2e4", "5e4", "1e5", "2e5", "5e5",
                    "1e6", "2e6", "5e6", "1e7", "1e8",
                ],
            },
        },
        "supports_optimize": True,
    },
    "iarpls": {
        "display_name": "iArPLS",
        "category": "whittaker",
        "tier": 1,
        "default_params": {"lam": 1e5},
        "param_info": {
            "lam": {
                "range": (1e2, 1e9),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e2", "2e2", "5e2", "1e3", "2e3", "5e3",
                    "1e4", "2e4", "5e4", "1e5", "2e5", "5e5",
                    "1e6", "2e6", "5e6", "1e7", "1e8",
                ],
            },
        },
        "supports_optimize": True,
    },
    "drpls": {
        "display_name": "drPLS",
        "category": "whittaker",
        "tier": 1,
        "default_params": {"lam": 1e5, "eta": 0.5},
        "param_info": {
            "lam": {
                "range": (1e2, 1e9),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e2", "2e2", "5e2", "1e3", "2e3", "5e3",
                    "1e4", "2e4", "5e4", "1e5", "2e5", "5e5",
                    "1e6", "2e6", "5e6", "1e7", "1e8",
                ],
            },
            "eta": {
                "range": (0.01, 1.0),
                "log_scale": False,
                "type": float,
                "values": [
                    "0.01", "0.05", "0.1", "0.2", "0.3",
                    "0.5", "0.7", "0.8", "0.9", "1.0",
                ],
            },
        },
        "supports_optimize": True,
    },
    "imor": {
        "display_name": "iMor",
        "category": "morphological",
        "tier": 1,
        "default_params": {"half_window": 30},
        "param_info": {
            "half_window": {
                "range": (5, 100),
                "log_scale": False,
                "type": int,
                "values": [
                    "5", "10", "15", "20", "25", "30",
                    "40", "50", "60", "80", "100",
                ],
            },
        },
        "supports_optimize": False,
    },
    "snip": {
        "display_name": "SNIP",
        "category": "morphological",
        "tier": 1,
        "default_params": {"max_half_window": 30},
        "param_info": {
            "max_half_window": {
                "range": (5, 100),
                "log_scale": False,
                "type": int,
                "values": [
                    "5", "10", "15", "20", "25", "30",
                    "40", "50", "60", "80", "100",
                ],
            },
        },
        "supports_optimize": False,
    },
    "modpoly": {
        "display_name": "ModPoly",
        "category": "polynomial",
        "tier": 1,
        "default_params": {"poly_order": 5},
        "param_info": {
            "poly_order": {
                "range": (1, 8),
                "log_scale": False,
                "type": int,
                "values": ["1", "2", "3", "4", "5", "6", "7", "8"],
            },
        },
        "supports_optimize": True,
    },
    "imodpoly": {
        "display_name": "iModPoly",
        "category": "polynomial",
        "tier": 1,
        "default_params": {"poly_order": 5},
        "param_info": {
            "poly_order": {
                "range": (1, 8),
                "log_scale": False,
                "type": int,
                "values": ["1", "2", "3", "4", "5", "6", "7", "8"],
            },
        },
        "supports_optimize": True,
    },
    # --- Tier 2: Specialized ---
    "pspline_arpls": {
        "display_name": "P-Spline arPLS",
        "category": "spline",
        "tier": 2,
        "default_params": {"lam": 1e3},
        "param_info": {
            "lam": {
                "range": (1e1, 1e7),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e1", "5e1", "1e2", "5e2", "1e3", "5e3",
                    "1e4", "5e4", "1e5", "5e5", "1e6", "1e7",
                ],
            },
        },
        "supports_optimize": True,
    },
    "pspline_iarpls": {
        "display_name": "P-Spline iArPLS",
        "category": "spline",
        "tier": 2,
        "default_params": {"lam": 1e3},
        "param_info": {
            "lam": {
                "range": (1e1, 1e7),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e1", "5e1", "1e2", "5e2", "1e3", "5e3",
                    "1e4", "5e4", "1e5", "5e5", "1e6", "1e7",
                ],
            },
        },
        "supports_optimize": True,
    },
    "fabc": {
        "display_name": "FABC",
        "category": "classification",
        "tier": 2,
        "default_params": {"lam": 1e6},
        "param_info": {
            "lam": {
                "range": (1e2, 1e9),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e2", "5e2", "1e3", "5e3", "1e4", "5e4",
                    "1e5", "5e5", "1e6", "5e6", "1e7", "1e8",
                ],
            },
        },
        "supports_optimize": False,
    },
    "dietrich": {
        "display_name": "Dietrich",
        "category": "classification",
        "tier": 2,
        "default_params": {"poly_order": 5},
        "param_info": {
            "poly_order": {
                "range": (1, 8),
                "log_scale": False,
                "type": int,
                "values": ["1", "2", "3", "4", "5", "6", "7", "8"],
            },
        },
        "supports_optimize": False,
    },
    "mpls": {
        "display_name": "MorPLS",
        "category": "morphological",
        "tier": 2,
        "default_params": {"half_window": 30, "lam": 1e5},
        "param_info": {
            "half_window": {
                "range": (5, 100),
                "log_scale": False,
                "type": int,
                "values": [
                    "5", "10", "15", "20", "25", "30",
                    "40", "50", "60", "80", "100",
                ],
            },
            "lam": {
                "range": (1e2, 1e9),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e2", "5e2", "1e3", "5e3", "1e4", "5e4",
                    "1e5", "5e5", "1e6", "5e6", "1e7", "1e8",
                ],
            },
        },
        "supports_optimize": False,
    },
    # --- Tier 3: Experimental ---
    "aspls": {
        "display_name": "asPLS",
        "category": "whittaker",
        "tier": 3,
        "default_params": {"lam": 1e5},
        "param_info": {
            "lam": {
                "range": (1e2, 1e9),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e2", "5e2", "1e3", "5e3", "1e4", "5e4",
                    "1e5", "5e5", "1e6", "5e6", "1e7", "1e8",
                ],
            },
        },
        "supports_optimize": True,
    },
    "psalsa": {
        "display_name": "PSALSA",
        "category": "whittaker",
        "tier": 3,
        "default_params": {"lam": 1e5, "p": 0.01},
        "param_info": {
            "lam": {
                "range": (1e2, 1e9),
                "log_scale": True,
                "type": float,
                "values": [
                    "1e2", "5e2", "1e3", "5e3", "1e4", "5e4",
                    "1e5", "5e5", "1e6", "5e6", "1e7", "1e8",
                ],
            },
            "p": {
                "range": (1e-4, 0.5),
                "log_scale": False,
                "type": float,
                "values": [
                    "0.0001", "0.0005", "0.001", "0.005",
                    "0.01", "0.02", "0.05", "0.1", "0.2", "0.5",
                ],
            },
        },
        "supports_optimize": True,
    },
    "std_distribution": {
        "display_name": "StdDist",
        "category": "classification",
        "tier": 3,
        "default_params": {"half_window": 30},
        "param_info": {
            "half_window": {
                "range": (5, 100),
                "log_scale": False,
                "type": int,
                "values": [
                    "5", "10", "15", "20", "25", "30",
                    "40", "50", "60", "80", "100",
                ],
            },
        },
        "supports_optimize": False,
    },
}

# Tier labels used as separators in dropdowns
TIER_LABELS: list[tuple[int, str]] = [
    (1, "-- Recommended --"),
    (2, "-- Specialized --"),
    (3, "-- Experimental --"),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_display_to_key_map() -> dict[str, str]:
    """Return mapping from display name to algorithm key.

    Example: ``{'arPLS': 'arpls', 'SNIP': 'snip', ...}``
    """
    return {info["display_name"]: key for key, info in ADVANCED_ALGORITHMS.items()}


def is_separator(value: str) -> bool:
    """Return True if *value* is a tier separator label, not a real algorithm."""
    return value.startswith("--") and value.endswith("--")


def get_dropdown_values() -> list[str]:
    """Build the ordered list of display names with tier separators for a Combobox."""
    values: list[str] = []
    for tier, label in TIER_LABELS:
        algos = [
            info["display_name"]
            for info in ADVANCED_ALGORITHMS.values()
            if info["tier"] == tier
        ]
        if algos:
            values.append(label)
            values.extend(algos)
    return values


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class BaselineAdvanced(BaseEstimator, TransformerMixin):
    """Sklearn-compatible baseline correction using pybaselines.

    Parameters
    ----------
    method : str
        Algorithm key from ``ADVANCED_ALGORITHMS`` (e.g. ``'arpls'``).
    wavenumbers : array-like or None
        Wavenumber axis. If None, index-based spacing is used (with a warning).
    **params
        Algorithm-specific parameters (e.g. ``lam=1e5``). Unknown keys are
        silently ignored by pybaselines.
    """

    def __init__(
        self,
        method: str = "arpls",
        wavenumbers: np.ndarray | None = None,
        **params,
    ):
        self.method = method
        self.wavenumbers = wavenumbers
        for key, val in params.items():
            setattr(self, key, val)

    def fit(self, X, y=None):
        """Store number of features (wavelengths)."""
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        if self.wavenumbers is not None and len(self.wavenumbers) != self.n_features_in_:
            raise ValueError(
                f"wavenumbers length ({len(self.wavenumbers)}) does not match "
                f"number of features ({self.n_features_in_})"
            )
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply baseline correction to each spectrum in *X*.

        Returns
        -------
        X_corrected : ndarray, same shape as *X*
        """
        if not HAS_PYBASELINES:
            raise ImportError(
                "pybaselines is required for advanced baseline correction. "
                "Install it with: pip install pybaselines"
            )
        X = np.asarray(X, dtype=float)
        if self.method not in ADVANCED_ALGORITHMS:
            raise ValueError(
                f"Unknown baseline method '{self.method}'. "
                f"Available: {sorted(ADVANCED_ALGORITHMS)}"
            )

        wn = self.wavenumbers
        if wn is None:
            warnings.warn(
                "No wavenumbers provided; using index-based spacing. "
                "Results may differ from wavenumber-aware correction.",
                stacklevel=2,
            )
            wn = np.arange(X.shape[1], dtype=float)

        # Collect algorithm-specific params from instance attributes
        info = ADVANCED_ALGORITHMS[self.method]
        algo_params = {}
        for pname in info["default_params"]:
            if hasattr(self, pname):
                algo_params[pname] = getattr(self, pname)
            else:
                algo_params[pname] = info["default_params"][pname]

        fitter = _PyBaseline(x_data=wn)
        method_func = getattr(fitter, self.method)

        X_out = np.empty_like(X)
        for i in range(X.shape[0]):
            try:
                baseline, _ = method_func(X[i], **algo_params)
                X_out[i] = X[i] - baseline
            except (LinAlgError, ValueError, np.linalg.LinAlgError) as exc:
                logger.warning(
                    "Baseline correction failed for spectrum %d (%s: %s); "
                    "keeping original spectrum.",
                    i, type(exc).__name__, exc,
                )
                X_out[i] = X[i]
        return X_out

    def transform_single_with_baseline(
        self, spectrum: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (corrected_spectrum, baseline) for a single spectrum."""
        if not HAS_PYBASELINES:
            raise ImportError("pybaselines is required")

        spectrum = np.asarray(spectrum, dtype=float)
        wn = self.wavenumbers
        if wn is None:
            wn = np.arange(len(spectrum), dtype=float)

        info = ADVANCED_ALGORITHMS[self.method]
        algo_params = {}
        for pname in info["default_params"]:
            if hasattr(self, pname):
                algo_params[pname] = getattr(self, pname)
            else:
                algo_params[pname] = info["default_params"][pname]

        fitter = _PyBaseline(x_data=wn)
        method_func = getattr(fitter, self.method)
        baseline, _ = method_func(spectrum, **algo_params)
        return spectrum - baseline, baseline

    @classmethod
    def auto_optimize(
        cls,
        X: np.ndarray,
        wavenumbers: np.ndarray | None = None,
        method: str = "arpls",
    ) -> dict:
        """Use pybaselines' optimize_extended_range to find optimal params.

        Only works for Whittaker and polynomial methods. Raises ValueError
        for unsupported methods (morphological, classification).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_features,)
            Spectral data. If 2-D, uses the mean spectrum.
        wavenumbers : array-like or None
            Wavenumber axis.
        method : str
            Algorithm key.

        Returns
        -------
        dict
            Optimized parameters (e.g. ``{'lam': 1e4}``).
        """
        if not HAS_PYBASELINES:
            raise ImportError("pybaselines is required")

        if method not in ADVANCED_ALGORITHMS:
            raise ValueError(f"Unknown method '{method}'")
        if not ADVANCED_ALGORITHMS[method]["supports_optimize"]:
            raise ValueError(
                f"Auto-optimize is not supported for '{method}' "
                f"(category: {ADVANCED_ALGORITHMS[method]['category']}). "
                f"Only Whittaker and polynomial methods are supported."
            )

        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            spectrum = X.mean(axis=0)
        else:
            spectrum = X

        wn = wavenumbers if wavenumbers is not None else np.arange(len(spectrum), dtype=float)
        fitter = _PyBaseline(x_data=wn)
        _, params_dict = fitter.optimize_extended_range(spectrum, method=method)

        # Extract the optimized parameters
        result = {}
        info = ADVANCED_ALGORITHMS[method]
        for pname in info["default_params"]:
            if pname in params_dict:
                result[pname] = params_dict[pname]
        # optimize_extended_range returns optimal lam/poly_order in the params dict
        if "optimal_parameter" in params_dict:
            # Map the optimal parameter back to the right key
            opt_val = params_dict["optimal_parameter"]
            primary_param = next(iter(info["default_params"]))
            result[primary_param] = opt_val
        return result
