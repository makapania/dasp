# ASP Refactor — Implementation Plan

**Date**: 2026-02-07
**Source repo**: `C:\Users\sponheim\git\dasp`
**Target repo**: `C:\Users\sponheim\git\asp`
**Design doc**: `docs/plans/2026-02-07-asp-refactor-design.md`

> This plan was generated from a full audit of every source file in the current codebase.
> Every function signature, line number, and class was verified against the actual code.

---

## How to Use This Plan

Each task has:
- **ID**: `P<phase>-<number>` (e.g., P1-01)
- **Files to create**: Exact path under `C:\Users\sponheim\git\asp\`
- **Port from**: Exact current source file path(s) with line numbers
- **What to copy**: Every function, class, and constant — with **exact signatures**
- **Depends on**: Other task IDs that must be complete first
- **Verification**: How to confirm the task is done correctly
- **Code example**: Skeleton or key snippet showing the expected structure

### Execution Rules
1. Work **one task at a time**, in ID order within each phase
2. A task is not done until its **verification step passes**
3. **Never skip functions** — if the source has it, the target must have it
4. Keep every file **under 500 lines** — split if needed
5. Port from actual source code at `C:\Users\sponheim\git\dasp\src\spectral_predict\`, **not** from old docs

---

## Phase 1: Project Skeleton (7 tasks)

### P1-01: Initialize repository and pyproject.toml

**Create**: `C:\Users\sponheim\git\asp\pyproject.toml`

**Contents**:
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "spectral-predict"
version = "2.0.0"
description = "Automated spectral modeling and prediction"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "openpyxl>=3.1",
    "xlsxwriter>=3.1",
    "optuna>=3.3",
    "pymoo>=0.6",
    "imbalanced-learn>=0.11",
    "lightgbm>=4.0",
    "xgboost>=2.0",
    "PySide6>=6.6",
    "qt-material>=2.14",
    "joblib>=1.3",
    "platformdirs>=4.0",
]

[project.optional-dependencies]
catboost = ["catboost>=1.2"]
torch = ["torch>=2.0"]
asd = ["specdal"]
opus = ["brukeropus"]
spc = ["spc-io"]
jcamp = ["jcamp"]
perkinelmer = ["specio"]
agilent = ["agilent-ir-formats"]
all = [
    "catboost>=1.2",
    "torch>=2.0",
    "specdal",
    "brukeropus",
    "spc-io",
    "jcamp",
    "specio",
    "agilent-ir-formats",
]
dev = ["pytest>=7.4", "black>=23.7", "ruff>=0.1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
```

**Also create**: `C:\Users\sponheim\git\asp\README.md` with one-liner: `# ASP — Spectral Predict`

**Depends on**: None

**Verification**:
```bash
cd C:\Users\sponheim\git\asp
git init
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```
Should install without errors.

---

### P1-02: Create directory structure

**Create all these `__init__.py` files** (each initially empty):

```
src/spectral_predict/__init__.py
src/spectral_predict/core/__init__.py
src/spectral_predict/readers/__init__.py
src/spectral_predict/preprocessing/__init__.py
src/spectral_predict/models/__init__.py
src/spectral_predict/search/__init__.py
src/spectral_predict/selection/__init__.py
src/spectral_predict/ensemble/__init__.py
src/spectral_predict/transfer/__init__.py
src/spectral_predict/analysis/__init__.py
src/spectral_predict/library/__init__.py
src/spectral_predict/export/__init__.py
src/spectral_predict/export/templates/__init__.py
src/spectral_predict/utils/__init__.py
src/spectral_predict/data_management/__init__.py
src/spectral_predict/gui/__init__.py
src/spectral_predict/gui/tabs/__init__.py
src/spectral_predict/gui/widgets/__init__.py
src/spectral_predict/gui/services/__init__.py
tests/__init__.py
tests/core/__init__.py
tests/readers/__init__.py
tests/preprocessing/__init__.py
tests/models/__init__.py
tests/search/__init__.py
tests/selection/__init__.py
tests/ensemble/__init__.py
tests/transfer/__init__.py
tests/analysis/__init__.py
example/
```

**Depends on**: P1-01

**Verification**:
```python
import spectral_predict
import spectral_predict.core
import spectral_predict.readers
import spectral_predict.preprocessing
import spectral_predict.models
import spectral_predict.search
import spectral_predict.selection
import spectral_predict.ensemble
import spectral_predict.transfer
import spectral_predict.analysis
import spectral_predict.library
import spectral_predict.export
import spectral_predict.utils
import spectral_predict.data_management
# All imports succeed without error
```

---

### P1-03: core/exceptions.py

**Create**: `src/spectral_predict/core/exceptions.py`
**Port from**: New (currently errors are scattered with bare `ValueError`/`RuntimeError`)

```python
"""Exception hierarchy for Spectral Predict."""

from __future__ import annotations


class SpectralPredictError(Exception):
    """Base exception for all Spectral Predict errors."""


class DataLoadError(SpectralPredictError):
    """Failed to load or parse a spectral data file."""


class AlignmentError(SpectralPredictError):
    """Failed to align spectral data with reference targets."""


class PreprocessingError(SpectralPredictError):
    """Error during spectral preprocessing."""


class SearchError(SpectralPredictError):
    """Error during model search / optimization."""


class ConfigurationError(SpectralPredictError):
    """Invalid configuration or parameter combination."""


class ValidationError(SpectralPredictError):
    """Data validation failed (e.g., shape mismatch, missing columns)."""


class ExportError(SpectralPredictError):
    """Error during model export or code generation."""
```

**Depends on**: P1-02

**Verification**:
```python
from spectral_predict.core.exceptions import (
    SpectralPredictError, DataLoadError, AlignmentError,
    PreprocessingError, SearchError, ConfigurationError,
    ValidationError, ExportError,
)
assert issubclass(DataLoadError, SpectralPredictError)
assert issubclass(SearchError, SpectralPredictError)
```

---

### P1-04: core/constants.py

**Create**: `src/spectral_predict/core/constants.py`
**Port from**:
- `constants.py` line 7: `RANDOM_STATE = 42`
- `search.py` lines 53-75: Model group sets
- `model_config.py` lines 19-123: Tier definitions
- `model_registry.py` lines 9-49: Model lists

```python
"""Global constants for Spectral Predict."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Model group sets (from search.py lines 53-75)
# ---------------------------------------------------------------------------
PLS_MODELS: set[str] = {"PLS", "PLS-DA", "Ridge", "Lasso", "ElasticNet"}
NEURAL_SVM_MODELS: set[str] = {"MLP", "SVR", "SVC"}
TREE_MODELS: set[str] = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}
NEURALBOOSTED_MODELS: set[str] = {"NeuralBoosted"}
SCALE_SENSITIVE_MODELS: set[str] = {"SVC", "SVR", "MLP", "NeuralBoosted", "Ridge", "Lasso", "ElasticNet"}
MODELS_PREFER_SERIAL_CV: set[str] = {"SVM", "PLS", "PLS-DA", "Ridge", "Lasso", "ElasticNet"}
LINEAR_MODELS: set[str] = PLS_MODELS | NEURAL_SVM_MODELS

# ---------------------------------------------------------------------------
# Model registries (from model_registry.py lines 9-49)
# ---------------------------------------------------------------------------
REGRESSION_MODELS: list[str] = [
    "PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest",
    "MLP", "NeuralBoosted", "SVR", "XGBoost", "LightGBM", "CatBoost",
]
CLASSIFICATION_MODELS: list[str] = [
    "PLS-DA", "RandomForest", "MLP", "NeuralBoosted",
    "SVM", "XGBoost", "LightGBM", "CatBoost",
]
ALL_MODELS: list[str] = sorted(set(REGRESSION_MODELS + CLASSIFICATION_MODELS))
MODELS_WITH_FEATURE_IMPORTANCE: list[str] = [
    "PLS", "PLS-DA", "Ridge", "Lasso", "ElasticNet", "RandomForest",
    "MLP", "NeuralBoosted", "SVR", "XGBoost", "LightGBM", "CatBoost",
]

# ---------------------------------------------------------------------------
# Tier definitions (from model_config.py lines 19-123)
# ---------------------------------------------------------------------------
MODEL_TIERS: dict[str, dict] = {
    "quick": {
        "description": "Fast initial scan (2-3 models, limited hyperparameters)",
        "models": ["PLS", "Ridge", "RandomForest"],
    },
    "standard": {
        "description": "Balanced coverage (6-8 models, standard hyperparameters)",
        "models": ["PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest", "SVR", "LightGBM", "MLP"],
    },
    "comprehensive": {
        "description": "Full sweep (all models, extended hyperparameters)",
        "models": ["PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest",
                    "SVR", "MLP", "XGBoost", "LightGBM", "CatBoost", "NeuralBoosted"],
    },
    "experimental": {
        "description": "Everything including experimental models",
        "models": ["PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest",
                    "SVR", "MLP", "XGBoost", "LightGBM", "CatBoost", "NeuralBoosted"],
    },
}

CLASSIFICATION_TIERS: dict[str, dict] = {
    "quick": {
        "description": "Fast classification scan",
        "models": ["PLS-DA", "RandomForest", "LightGBM"],
    },
    "standard": {
        "description": "Standard classification coverage",
        "models": ["PLS-DA", "RandomForest", "SVM", "LightGBM", "MLP", "XGBoost"],
    },
    "comprehensive": {
        "description": "Full classification sweep",
        "models": ["PLS-DA", "RandomForest", "SVM", "MLP", "XGBoost",
                    "LightGBM", "CatBoost", "NeuralBoosted"],
    },
    "experimental": {
        "description": "All classification models",
        "models": ["PLS-DA", "RandomForest", "SVM", "MLP", "XGBoost",
                    "LightGBM", "CatBoost", "NeuralBoosted"],
    },
}

DEFAULT_TIER: str = "standard"
```

**Depends on**: P1-02

**Verification**:
```python
from spectral_predict.core.constants import (
    RANDOM_STATE, PLS_MODELS, TREE_MODELS, REGRESSION_MODELS,
    CLASSIFICATION_MODELS, MODEL_TIERS, DEFAULT_TIER,
)
assert RANDOM_STATE == 42
assert "PLS" in PLS_MODELS
assert "RandomForest" in TREE_MODELS
assert len(MODEL_TIERS) == 4
assert DEFAULT_TIER == "standard"
```

---

### P1-05: core/types.py

**Create**: `src/spectral_predict/core/types.py`
**Port from**: New (replaces scattered `self.X`, `self.y`, `self.ref`, etc.)

```python
"""Core data types for Spectral Predict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SpectralData:
    """Container for spectral dataset.

    Attributes:
        X: Spectral matrix (samples x wavelengths). Index = sample IDs,
           columns = wavelength values as floats.
        y: Target values. Can be None for spectra-only mode.
        wavelengths: 1-D array of wavelength values (matches X.columns).
        sample_ids: List of sample identifiers (matches X.index).
        metadata: Arbitrary metadata from file readers.
        data_type: "reflectance" or "absorbance".
        ref: Full reference DataFrame (if loaded from CSV/Excel with extra columns).
    """

    X: pd.DataFrame
    y: pd.Series | None = None
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    data_type: str = "reflectance"
    ref: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if len(self.wavelengths) == 0 and self.X is not None:
            self.wavelengths = np.array(self.X.columns, dtype=float)
        if not self.sample_ids and self.X is not None:
            self.sample_ids = list(self.X.index)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0] if self.X is not None else 0

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X is not None else 0

    @property
    def has_target(self) -> bool:
        return self.y is not None and len(self.y) > 0


@dataclass
class SearchResult:
    """Single row in search results."""

    model_name: str
    preprocessing: str
    params: dict[str, Any]
    wavelength_indices: np.ndarray | None = None
    wavelength_tag: str = "full"
    metrics: dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    rank: int = 0


@dataclass
class TransferModel:
    """Calibration transfer model container.

    Port from: calibration_transfer.py lines 26-37
    """

    method: str  # "ds", "pds", "tsr", "ctai", "nspfce", "jypls-inv"
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Depends on**: P1-02

**Verification**:
```python
import numpy as np
import pandas as pd
from spectral_predict.core.types import SpectralData, SearchResult, TransferModel

X = pd.DataFrame(np.random.rand(10, 200), columns=[f"{w}" for w in range(400, 600)])
data = SpectralData(X=X, data_type="reflectance")
assert data.n_samples == 10
assert data.n_features == 200
assert data.has_target is False
assert len(data.wavelengths) == 200

tm = TransferModel(method="ds")
assert tm.method == "ds"
```

---

### P1-06: core/config.py

**Create**: `src/spectral_predict/core/config.py`
**Port from**: New (replaces loose parameter dicts)

```python
"""Configuration dataclasses for Spectral Predict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreprocessConfig:
    """Preprocessing configuration for a single pipeline step.

    Mirrors the `preprocess_cfg` dicts passed around in search.py.
    """

    name: str = "raw"  # e.g. "snv", "deriv1", "snv_deriv1"
    derivative_order: int | None = None
    window: int | None = None
    polyorder: int | None = None
    baseline_method: str | None = None  # "polynomial", "als", "airpls", "rubber_band"
    baseline_params: dict[str, Any] = field(default_factory=dict)
    smoothing: bool = False
    smoothing_window: int = 17
    smoothing_polyorder: int = 2


@dataclass
class SearchConfig:
    """Full search configuration.

    Collects all parameters currently spread across run_search()'s 70+ arguments.
    """

    task_type: str = "regression"
    folds: int = 5
    variable_penalty: float = 0.0
    complexity_penalty: float = 0.0
    max_n_components: int = 10
    max_iter: int = 500
    tier: str = "standard"
    enabled_models: list[str] | None = None
    preprocessing_methods: list[str] | None = None
    window_sizes: list[int] | None = None

    # Variable selection
    enable_variable_subsets: bool = True
    variable_counts: list[int] | None = None
    variable_selection_methods: list[str] | None = None

    # Region analysis
    enable_region_subsets: bool = True
    n_top_regions: int = 10
    region_test_all_individual: bool = False
    region_test_pairwise: bool = False

    # Optimization method
    optimization_method: str = "grid"  # "grid", "bayesian", "nsga2"
    bayesian_trials: int | None = None
    nsga2_population: int = 64
    nsga2_generations: int = 100

    # Imbalance
    imbalance_method: str | None = None
    imbalance_params: dict[str, Any] = field(default_factory=dict)

    # Baseline
    baseline_method: str | None = None
    baseline_params: dict[str, Any] = field(default_factory=dict)

    # Smoothing
    smoothing: bool = False
    smoothing_window: int = 17
    smoothing_polyorder: int = 2

    # GA / Smart preprocessing
    ga_preprocess: bool = False
    ga_preprocess_method: str = "exhaustive"
    smart_preprocess: bool = False
    smart_preprocess_importance: str = "model_specific"
    smart_preprocess_n_top: int = 10

    # Validation
    compute_validation: bool = False
    validation_top_n: int = 100
    early_stopping_rounds: int = 40


@dataclass
class ExportOptions:
    """Options for code generation export.

    Port from: code_generator.py lines 29-51
    """

    include_data_loading: bool = True
    include_preprocessing: bool = True
    include_variable_selection: bool = True
    include_cross_validation: bool = True
    include_visualization: bool = True
    include_comments: bool = True
    include_prediction_template: bool = True
    format: str = "script"  # "script" or "notebook"
    data_path: str = "your_data.csv"
    target_column: str = "target"
    include_data: bool = False
    colab_ready: bool = False
```

**Depends on**: P1-02

**Verification**:
```python
from spectral_predict.core.config import PreprocessConfig, SearchConfig, ExportOptions

pc = PreprocessConfig(name="snv_deriv1", window=17)
assert pc.name == "snv_deriv1"
assert pc.window == 17

sc = SearchConfig(task_type="classification", tier="comprehensive")
assert sc.folds == 5
assert sc.tier == "comprehensive"
```

---

### P1-07: core/__init__.py — Re-export all public names

**Create**: `src/spectral_predict/core/__init__.py`

```python
"""Core module: types, config, constants, exceptions."""

from spectral_predict.core.constants import *  # noqa: F401,F403
from spectral_predict.core.config import ExportOptions, PreprocessConfig, SearchConfig
from spectral_predict.core.exceptions import (
    AlignmentError,
    ConfigurationError,
    DataLoadError,
    ExportError,
    PreprocessingError,
    SearchError,
    SpectralPredictError,
    ValidationError,
)
from spectral_predict.core.types import SearchResult, SpectralData, TransferModel

__all__ = [
    # types
    "SpectralData", "SearchResult", "TransferModel",
    # config
    "PreprocessConfig", "SearchConfig", "ExportOptions",
    # exceptions
    "SpectralPredictError", "DataLoadError", "AlignmentError",
    "PreprocessingError", "SearchError", "ConfigurationError",
    "ValidationError", "ExportError",
]
```

**Depends on**: P1-03, P1-04, P1-05, P1-06

**Verification**:
```python
from spectral_predict.core import (
    SpectralData, RANDOM_STATE, PreprocessConfig,
    DataLoadError, REGRESSION_MODELS, MODEL_TIERS,
)
assert RANDOM_STATE == 42
```

---

## Phase 2: Backend Port

### Phase 2A: Preprocessing (11 tasks)

---

### P2-01: preprocessing/base.py

**Create**: `src/spectral_predict/preprocessing/base.py`
**Port from**: New (establishes interface)

```python
"""Base interface for preprocessing transformers."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all spectral preprocessors.

    All preprocessors must implement fit() and transform() following
    the scikit-learn transformer API. This enables pipeline integration.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError
```

**Depends on**: P1-07

**Verification**: `from spectral_predict.preprocessing.base import BasePreprocessor`

---

### P2-02: preprocessing/snv.py

**Create**: `src/spectral_predict/preprocessing/snv.py`
**Port from**: `preprocess.py` lines 8-44

**What to copy — exact signatures**:
| Source | Function/Class | Signature |
|--------|---------------|-----------|
| preprocess.py:8 | `class SNV(BaseEstimator, TransformerMixin)` | |
| preprocess.py:15 | `SNV.fit(self, X, y=None)` | returns `self` |
| preprocess.py:23 | `SNV.transform(self, X)` | returns `ndarray[n_samples, n_features]` |

**Code skeleton**:
```python
"""Standard Normal Variate (SNV) preprocessing."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate transformation.

    Centers each spectrum to zero mean and unit variance.
    Port from: preprocess.py lines 8-44
    """

    def fit(self, X, y=None):
        # Validate input
        return self

    def transform(self, X):
        # X_array = np.asarray(X, dtype=float)
        # row_means = X_array.mean(axis=1, keepdims=True)
        # row_stds = X_array.std(axis=1, keepdims=True, ddof=0)
        # Handle zero std...
        # return (X_array - row_means) / row_stds
        ...
```

**Depends on**: P2-01

**Verification**:
```python
import numpy as np
from spectral_predict.preprocessing.snv import SNV

X = np.random.rand(10, 200)
snv = SNV()
X_t = snv.fit_transform(X)
assert X_t.shape == (10, 200)
# Each row should have ~0 mean and ~1 std
assert abs(X_t[0].mean()) < 1e-10
assert abs(X_t[0].std() - 1.0) < 1e-10
```

---

### P2-03: preprocessing/derivatives.py

**Create**: `src/spectral_predict/preprocessing/derivatives.py`
**Port from**: `preprocess.py` lines 47-219

**What to copy — exact signatures**:
| Source | Function/Class | Signature |
|--------|---------------|-----------|
| preprocess.py:47 | `class SavgolDerivative(BaseEstimator, TransformerMixin)` | |
| preprocess.py:61 | `__init__(self, deriv=1, window=7, polyorder=None)` | |
| preprocess.py:66 | `fit(self, X, y=None)` | returns `self` |
| preprocess.py:74 | `transform(self, X)` | returns `ndarray` — auto-adjusts window if too large for n_features |
| preprocess.py:136 | `class SavgolSmooth(BaseEstimator, TransformerMixin)` | |
| preprocess.py:151 | `__init__(self, window_length=17, polyorder=2)` | |
| preprocess.py:155 | `fit(self, X, y=None)` | returns `self` |
| preprocess.py:163 | `transform(self, X)` | returns `ndarray` — auto-adjusts window |

**Key logic to preserve**:
- `SavgolDerivative.transform()`: If `polyorder` is None, defaults to `deriv + 1`. Window must be odd and > polyorder. Auto-reduces window if `n_features < window`. Uses `scipy.signal.savgol_filter`.
- `SavgolSmooth.transform()`: Same auto-reduction logic.

**Depends on**: P2-01

**Verification**:
```python
import numpy as np
from spectral_predict.preprocessing.derivatives import SavgolDerivative, SavgolSmooth

X = np.random.rand(10, 200)
d1 = SavgolDerivative(deriv=1, window=17)
X_d1 = d1.fit_transform(X)
assert X_d1.shape == (10, 200)

smooth = SavgolSmooth(window_length=17, polyorder=2)
X_s = smooth.fit_transform(X)
assert X_s.shape == (10, 200)

# Test auto-adjustment with small data
X_small = np.random.rand(5, 8)
d2 = SavgolDerivative(deriv=1, window=17)
X_d2 = d2.fit_transform(X_small)
assert X_d2.shape == (5, 8)
```

---

### P2-04: preprocessing/baseline.py

**Create**: `src/spectral_predict/preprocessing/baseline.py`
**Port from**: `baseline.py` (entire file, 360 lines)

**What to copy — exact signatures**:
| Source | Function/Class | Signature |
|--------|---------------|-----------|
| baseline.py:9 | `rubber_band_baseline(y: np.ndarray) -> np.ndarray` | Andrew's lower convex hull |
| baseline.py:54 | `class BaselineALS(BaseEstimator, TransformerMixin)` | |
| baseline.py:96 | `__init__(self, lambda_=1e5, p=0.001, niter=10)` | |
| baseline.py:101 | `fit(self, X, y=None)` | returns `self` |
| baseline.py:109 | `transform(self, X)` | returns `ndarray` |
| baseline.py:153 | `class BaselinePolynomial(BaseEstimator, TransformerMixin)` | |
| baseline.py:180 | `__init__(self, degree=3)` | |
| baseline.py:183 | `fit(self, X, y=None)` | returns `self` |
| baseline.py:191 | `transform(self, X)` | returns `ndarray` |
| baseline.py:224 | `class BaselineAirPLS(BaseEstimator, TransformerMixin)` | |
| baseline.py:267 | `__init__(self, lam=1e5, max_iter=15, tol=1e-3)` | |
| baseline.py:272 | `fit(self, X, y=None)` | returns `self` |
| baseline.py:279 | `_baseline_airpls_single(self, y)` | internal |
| baseline.py:338 | `transform(self, X)` | returns `ndarray` |

**CRITICAL**: All four methods (rubber_band, ALS, polynomial, airPLS) must be available as analysis preprocessing options in Basic Settings, not just Explore tab visualizations.

**Depends on**: P2-01

**Verification**:
```python
import numpy as np
from spectral_predict.preprocessing.baseline import (
    rubber_band_baseline, BaselineALS, BaselinePolynomial, BaselineAirPLS,
)

# Rubber band
spectrum = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.5
corrected = rubber_band_baseline(spectrum)
assert corrected.shape == (200,)

# ALS
X = np.random.rand(10, 200) + np.linspace(0, 2, 200)
als = BaselineALS(lambda_=1e5, p=0.001, niter=10)
X_t = als.fit_transform(X)
assert X_t.shape == (10, 200)

# Polynomial
poly = BaselinePolynomial(degree=3)
X_t = poly.fit_transform(X)
assert X_t.shape == (10, 200)

# AirPLS
airpls = BaselineAirPLS(lam=1e5, max_iter=15, tol=1e-3)
X_t = airpls.fit_transform(X)
assert X_t.shape == (10, 200)
```

---

### P2-05: preprocessing/msc.py

**Create**: `src/spectral_predict/preprocessing/msc.py`
**Port from**: `interference.py` lines 217-354

**What to copy — exact signatures**:
| Source | Function/Class | Signature |
|--------|---------------|-----------|
| interference.py:217 | `class MSC(BaseEstimator, TransformerMixin)` | Multiplicative Scatter Correction |
| interference.py | `__init__(self, reference=None)` | |
| interference.py | `fit(self, X, y=None)` | computes mean reference spectrum |
| interference.py | `transform(self, X)` | returns corrected spectra |

**Depends on**: P2-01

**Verification**:
```python
import numpy as np
from spectral_predict.preprocessing.msc import MSC

X = np.random.rand(10, 200)
msc = MSC()
X_t = msc.fit_transform(X)
assert X_t.shape == (10, 200)
```

---

### P2-06: preprocessing/transforms.py

**Create**: `src/spectral_predict/preprocessing/transforms.py`
**Port from**: `io.py` lines 2006-2212

**What to copy — exact signatures**:
| Source | Function/Class | Signature |
|--------|---------------|-----------|
| io.py:2006 | `detect_spectral_data_type(X, metadata=None)` | returns `tuple[str, float, str]` — (type, confidence, method) |
| io.py:2182 | `infer_reflectance_scale(X) -> float` | returns 1.0 or 100.0 |
| New | `convert_reflectance_to_absorbance(X, scale=None)` | |
| New | `convert_absorbance_to_reflectance(X, scale=100.0)` | |

**Key logic**:
- `detect_spectral_data_type`: Uses value bounds, mean analysis, peak/valley shape, metadata keywords. Returns confidence 0-100%.
- Conversion: `absorbance = -log10(reflectance)`, `reflectance = 10^(-absorbance)`

**Depends on**: P2-01

**Verification**:
```python
import numpy as np
from spectral_predict.preprocessing.transforms import (
    detect_spectral_data_type, infer_reflectance_scale,
    convert_reflectance_to_absorbance, convert_absorbance_to_reflectance,
)
import pandas as pd

# Reflectance data (0-100 scale)
X_refl = pd.DataFrame(np.random.uniform(10, 90, (5, 100)))
dtype, conf, method = detect_spectral_data_type(X_refl)
assert dtype in ("reflectance", "absorbance")

scale = infer_reflectance_scale(X_refl)
assert scale in (1.0, 100.0)
```

---

### P2-07: preprocessing/pipeline.py

**Create**: `src/spectral_predict/preprocessing/pipeline.py`
**Port from**:
- `preprocess.py` lines 222-491: `build_preprocessing_pipeline()`
- `preprocessing_wrapper.py` lines 15-220: `class PreprocessorConfig`

**What to copy — exact signatures**:
| Source | Function/Class | Signature |
|--------|---------------|-----------|
| preprocess.py:222 | `build_preprocessing_pipeline(preprocess_name, deriv=None, window=None, polyorder=None, imbalance_method=None, imbalance_params=None, task_type=None, interference=None, wavelengths=None, random_state=42, baseline_method=None, baseline_params=None, smoothing=False, smoothing_window=17, smoothing_polyorder=2)` | returns `list[(str, transformer)]` |
| preprocessing_wrapper.py:15 | `class PreprocessorConfig(BaseEstimator, TransformerMixin)` | |
| preprocessing_wrapper.py:42 | `__init__(self, preprocess_name, deriv=None, window=None, polyorder=None, wavelengths=None, all_wavelengths=None)` | |
| preprocessing_wrapper.py:100 | `fit(self, X, y=None)` | returns `self` |
| preprocessing_wrapper.py:117 | `transform(self, X)` | returns `ndarray` — handles wavelength subsetting + preprocessing |
| preprocessing_wrapper.py:179 | `get_config(self)` | returns `dict` |
| preprocessing_wrapper.py:198 | `from_config(cls, config)` | classmethod, returns `PreprocessorConfig` |

**Key logic**: `build_preprocessing_pipeline` constructs a list of `(name, transformer)` tuples that can be passed to `sklearn.pipeline.Pipeline`. It dispatches to SNV, SavgolDerivative, baseline transformers, MSC, smoothing, and optionally imbalance transformers.

**Depends on**: P2-02, P2-03, P2-04, P2-05

**Verification**:
```python
from spectral_predict.preprocessing.pipeline import (
    build_preprocessing_pipeline, PreprocessorConfig,
)
import numpy as np

# Test pipeline building
steps = build_preprocessing_pipeline("snv_deriv1", window=17)
assert len(steps) >= 1
assert steps[0][0]  # has a name

# Test PreprocessorConfig
cfg = PreprocessorConfig(preprocess_name="snv", wavelengths=[400, 500, 600])
X = np.random.rand(5, 3)
X_t = cfg.fit_transform(X)
assert X_t.shape[0] == 5

cfg_dict = cfg.get_config()
cfg2 = PreprocessorConfig.from_config(cfg_dict)
assert cfg2.preprocess_name == "snv"
```

---

### P2-08: preprocessing/discovery.py

**Create**: `src/spectral_predict/preprocessing/discovery.py`
**Port from**: `preprocessing_discovery.py` (987 lines)

**What to copy — exact signatures** (all from preprocessing_discovery.py):

| Line | Function | Signature |
|------|----------|-----------|
| 28 | constant `PREPROCESSING_CANDIDATES` | list of 14 preprocessing types |
| 52 | constant `WINDOW_SIZES` | `[7, 11, 17, 25, 31, 37]` |
| 55 | constant `SUBSET_SIZES` | `[50, 100, 200, 300]` |
| 58 | constant `PREPROCESSING_COMPLEXITY` | dict of complexity scores |
| 80 | constant `IMPORTANCE_METHODS` | dict of 4 methods |
| 100 | `compute_importance(X, y, method='model_specific', model_name=None, task_type='regression')` | returns `ndarray` |
| 162 | `_compute_cars_tree_importance(X, y, task_type)` | returns `ndarray` |
| 190 | `_compute_lightgbm_importance(X, y, task_type)` | returns `ndarray` |
| 224 | `_compute_vip_importance(X, y)` | returns `ndarray` |
| 243 | `_compute_model_specific_importance(X, y, model_name, task_type)` | returns `ndarray` |
| 280 | `_compute_coefficient_importance(X, y, model_name)` | returns `ndarray` |
| 304 | `_compute_tree_importance(X, y, model_name, task_type)` | returns `ndarray` |
| 357 | `_compute_neural_importance(X, y, task_type)` | returns `ndarray` |
| 386 | `_compute_svm_importance(X, y, task_type)` | returns `ndarray` |
| 423 | `apply_preprocessing(X, preproc_name, window=None)` | returns `ndarray` |
| 497 | `get_edge_zone(preproc_name, window)` | returns `int` |
| 514 | `select_wavelengths_by_importance(importance, target_n=200, edge_zone=0)` | returns `ndarray` |
| 569 | `evaluate_preprocessing_config(X, y, preproc_name, window, importance_method, model_name, task_type, cv_folds=5)` | returns `dict` |
| 665 | `_quick_evaluate(X, y, task_type, cv_folds)` | returns `float` |
| 717 | `score_config(config, all_configs, task_type)` | returns `float` |
| 750 | `select_diverse_configs(configs, n_top, task_type)` | returns `list[dict]` |
| 806 | `discover_preprocessing(X, y, models_to_test=None, task_type='regression', importance_method='model_specific', n_top=10, cv_folds=5, progress_callback=None)` | returns `list[dict]` |

**Note**: This file is close to 500 lines. If it exceeds, split importance functions into `preprocessing/importance.py`.

**Depends on**: P2-07

**Verification**:
```python
import numpy as np
from spectral_predict.preprocessing.discovery import (
    compute_importance, apply_preprocessing, discover_preprocessing,
    PREPROCESSING_CANDIDATES, WINDOW_SIZES, IMPORTANCE_METHODS,
)

X = np.random.rand(50, 200)
y = np.random.rand(50)

imp = compute_importance(X, y, method="vip")
assert imp.shape == (200,)

X_proc = apply_preprocessing(X, "snv")
assert X_proc.shape == (50, 200)

assert len(PREPROCESSING_CANDIDATES) >= 14
assert len(WINDOW_SIZES) >= 6
```

---

### P2-09: preprocessing/ga_optimization.py

**Create**: `src/spectral_predict/preprocessing/ga_optimization.py`
**Port from**: `ga_preprocessing.py` lines 51-770

**What to copy — exact signatures** (from ga_preprocessing.py):

| Line | Constant/Function | Signature |
|------|-------------------|-----------|
| 51 | `PREPROC_TYPES` | list of 14 preprocessing type names |
| 69 | `WINDOW_SIZES` | `[5, 7, 9, ..., 51]` (17 values) |
| 74 | `DERIVATIVE_WINDOW_RANGES` | dict of derivative-specific ranges |
| 82 | `MODEL_TO_PROXY` | dict mapping model names to proxy types |
| 92 | `ROBUSTNESS_SEEDS` | `[42, 123, 456, 789, 999]` |
| 95 | `VARIANCE_PENALTY` | `0.1` |
| 98 | `N_GENES` | `2` |
| 102 | `random_chromosome(rng)` | returns `ndarray[2]` |
| 110 | `get_seed_chromosomes()` | returns `list[ndarray]` |
| 145 | `chromosome_to_transform(genes)` | returns `tuple(str, Callable or None)` |
| 224 | `get_config_description(genes)` | returns `str` |
| 239 | `evaluate_fitness(genes, X, y, cv_folds=5, n_components=10, task_type='regression', random_state=42, fitness_model='pls', model_config=None)` | returns `float` |
| 340 | `_evaluate_with_actual_model(X, y, cv, task_type, model_name, model_params, random_state)` | returns `float` |
| 449 | `_evaluate_pls(X, y, cv, n_comp, task_type)` | returns `float` |
| 467 | `_evaluate_lightgbm(X, y, cv, task_type, random_state)` | returns `float` |
| 494 | `_evaluate_mlp(X, y, cv, task_type, random_state)` | returns `float` |
| 523 | `_evaluate_neuralboosted(X, y, cv, n_comp, task_type, random_state)` | returns `float` |
| 554 | `get_proxy_for_model(model_name)` | returns `str` |
| 578 | `get_smart_window_range(preproc_type)` | returns `list[int]` |
| 604 | `get_smart_combinations()` | returns `list[tuple]` |
| 636 | `evaluate_fitness_robust(genes, X, y, cv_folds=5, n_components=10, task_type='regression', fitness_model='pls', model_config=None, n_seeds=5, variance_penalty=0.1)` | returns `tuple(float, float, float)` |
| 718 | `tournament_selection(population, fitness, tournament_size, rng)` | returns `ndarray` |
| 731 | `crossover(parent1, parent2, crossover_rate, rng)` | returns `tuple(ndarray, ndarray)` |
| 752 | `mutate(chromosome, mutation_rate, rng)` | returns `ndarray` |
| 776 | `select_diverse_exhaustive_configs(all_genes, all_fitness, n_top=5, exclude_raw_from_diversity=True)` | returns `list[int]` |

**Depends on**: P2-07

**Verification**:
```python
from spectral_predict.preprocessing.ga_optimization import (
    PREPROC_TYPES, WINDOW_SIZES, random_chromosome,
    chromosome_to_transform, evaluate_fitness,
)
import numpy as np

assert len(PREPROC_TYPES) >= 14
assert len(WINDOW_SIZES) >= 17

rng = np.random.RandomState(42)
chrom = random_chromosome(rng)
assert chrom.shape == (2,)

name, transform_fn = chromosome_to_transform(chrom)
assert isinstance(name, str)
```

---

### P2-10: preprocessing/ga_search.py

**Create**: `src/spectral_predict/preprocessing/ga_search.py`
**Port from**: `ga_preprocessing.py` lines 776-end (1806)

**What to copy — exact signatures**:
| Line | Function | Signature |
|------|----------|-----------|
| 863 | `exhaustive_search(X, y, cv_folds=5, n_components=10, task_type='regression', random_state=42, fitness_model='pls', n_jobs=1, verbose=1, progress_callback=None, top_n=5, model_config=None)` | returns `dict` |
| 1082 | `smart_exhaustive_search(X, y, cv_folds=5, n_components=10, task_type='regression', fitness_model='auto', target_model=None, n_jobs=1, verbose=1, progress_callback=None, top_n=5, model_config=None, stage1_top_k=20, robust_validation=True)` | returns `dict` |
| 1381 | `optimize_preprocessing(X, y, method='ga', population_size=48, n_generations=30, crossover_rate=0.7, mutation_rate=0.15, tournament_size=3, cv_folds=5, n_components=10, elitism=2, task_type='regression', random_state=42, verbose=1, progress_callback=None, fitness_model='pls', n_jobs=1, top_n=5, model_config=None)` | returns `dict` |
| 1751 | `get_optimized_preproc_config(X, y, quick=True, random_state=42, verbose=1)` | returns `tuple(str, Callable or None)` |

**Depends on**: P2-09

**Verification**:
```python
from spectral_predict.preprocessing.ga_search import (
    exhaustive_search, smart_exhaustive_search, optimize_preprocessing,
)
# Import check only — full tests require larger data
```

---

### P2-11: preprocessing/learned.py

**Create**: `src/spectral_predict/preprocessing/learned.py`
**Port from**: `learned_preprocessing.py` (775 lines)

**What to copy — exact signatures**:
| Line | Class/Function | Notes |
|------|---------------|-------|
| 56 | `PYTORCH_AVAILABLE` | bool flag |
| 64-82 | Stub classes when PyTorch unavailable | |
| 87 | `class LearnedSpectralPreprocessing(nn.Module)` | `__init__(n_wavelengths, n_conv_layers=2, n_filters=16, kernel_size=11, dropout=0.3)`, `forward(x)` |
| 186 | `class SpectralPreprocessorWithRegressor(BaseEstimator, RegressorMixin, TransformerMixin)` | `__init__(...)`, `fit(X, y, epochs=100, ...)`, `predict(X)`, `transform(X)` |
| 472 | `class LearnedPreprocessor(BaseEstimator, TransformerMixin)` | `__init__(...)`, `fit(X, y)`, `transform(X)` |

**Key**: Must provide graceful fallback stubs when PyTorch is not installed.

**Depends on**: P2-01

**Verification**:
```python
from spectral_predict.preprocessing.learned import (
    PYTORCH_AVAILABLE, LearnedSpectralPreprocessing,
    SpectralPreprocessorWithRegressor, LearnedPreprocessor,
)
assert isinstance(PYTORCH_AVAILABLE, bool)
# If PyTorch not installed, instantiation should raise ImportError with helpful message
```

---

### Phase 2B: Readers (13 tasks)

---

### P2-12: readers/base.py

**Create**: `src/spectral_predict/readers/base.py`
**Port from**: New

```python
"""Base interface for spectral file readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseReader(ABC):
    """Abstract base for all format readers.

    Convention:
    - read_file() returns Tuple[pd.Series, dict] (single spectrum + metadata)
    - read_dir() returns Tuple[pd.DataFrame, dict] (multiple spectra + metadata)
    """

    @abstractmethod
    def read_file(self, path: str | Path) -> tuple[pd.Series, dict[str, Any]]:
        ...

    @abstractmethod
    def read_dir(self, directory: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        ...

    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """File extensions this reader handles (e.g., ['.csv', '.txt'])."""
        ...
```

**Depends on**: P1-07

---

### P2-13: readers/alignment.py

**Create**: `src/spectral_predict/readers/alignment.py`
**Port from**: `io.py` lines 291-598

**What to copy — exact signatures**:
| Line | Function | Signature |
|------|----------|-----------|
| 291 | `_rename_duplicate_ids(index: pd.Index)` | returns `tuple[pd.Index, int, dict]` |
| 381 | `_normalize_filename_for_matching(filename)` | returns `str` |
| 412 | `align_xy(X, ref, id_column, target, return_alignment_info=False)` | returns `tuple[pd.DataFrame, pd.Series]` or with dict |

**Depends on**: P1-07

**Verification**:
```python
import pandas as pd
import numpy as np
from spectral_predict.readers.alignment import align_xy

X = pd.DataFrame(np.random.rand(3, 10), index=["a", "b", "c"])
ref = pd.DataFrame({"id": ["a", "b", "c"], "target": [1.0, 2.0, 3.0]})
X_aligned, y = align_xy(X, ref, id_column="id", target="target")
assert len(y) == 3
assert list(y) == [1.0, 2.0, 3.0]
```

---

### P2-14: readers/detection.py

**Create**: `src/spectral_predict/readers/detection.py`
**Port from**: `io.py` lines 982-1221, 2220-2315, 2570-2587

**What to copy — exact signatures**:
| Line | Function | Signature |
|------|----------|-----------|
| 982 | `detect_combined_format(directory_path)` | returns `tuple[bool, str or None]` |
| 1022 | `identify_wavelength_columns(df)` | returns `list` |
| 1057 | `auto_detect_specimen_id_column(df, exclude_wavelength_cols)` | returns `str or None` |
| 1168 | `auto_detect_y_column(df, exclude_cols)` | returns `str` |
| 2220 | `detect_format(path: str | Path)` | returns `str` |
| 2570 | `_detect_directory_format(directory: Path)` | returns `str` |

**Depends on**: P1-07

---

### P2-15: readers/csv.py

**Create**: `src/spectral_predict/readers/csv.py`
**Port from**: `io.py` CSV portions

**What to copy — exact signatures**:
| Line | Function | Signature |
|------|----------|-----------|
| 9 | `read_csv_spectra(path)` | returns `tuple[pd.DataFrame, dict]` |
| 182 | `read_csv_dir(csv_dir, exclude_files=None)` | returns `tuple[pd.DataFrame, dict]` |
| 336 | `read_reference_csv(path, id_column)` | returns `pd.DataFrame` |
| 1223 | `read_combined_csv(filepath, specimen_id_col=None, y_col=None, drop_na_y=True)` | returns `tuple[pd.DataFrame, pd.Series, pd.DataFrame or None, dict]` |
| 3341 | `write_csv_spectra(data, path, metadata=None, float_format='%.6f', include_index=True)` | returns `None` |

Also include: `_is_likely_reference_csv(path)` (line 124)

**Depends on**: P2-13, P2-14

---

### P2-16: readers/excel.py

**Create**: `src/spectral_predict/readers/excel.py`
**Port from**: `io.py` Excel portions

**What to copy — exact signatures**:
| Line | Function | Signature |
|------|----------|-----------|
| 2588 | `read_excel_spectra(path, sheet_name=0)` | returns `tuple[pd.DataFrame, dict]` |
| 2682 | `read_combined_excel(filepath, specimen_id_col=None, y_col=None, sheet_name=0, drop_na_y=True)` | returns `tuple[pd.DataFrame, pd.Series, pd.DataFrame or None, dict]` |
| 2949 | `detect_combined_excel_format(directory_path)` | returns `tuple[bool, str or None, str or None]` |
| 3354 | `write_excel_spectra(data, path, metadata=None, sheet_name='Spectra', freeze_panes=(1,1), float_format='0.000000')` | returns `None` |

**Depends on**: P2-13, P2-14

---

### P2-17: readers/asd.py

**Create**: `src/spectral_predict/readers/asd.py`
**Port from**: `io.py` lines 600-841, `readers/asd_native.py` (62 lines), `readers/asd_r_bridge.py` (137 lines)

**What to copy — exact signatures**:
| Source | Function | Signature |
|--------|----------|-----------|
| io.py:600 | `read_asd_dir(asd_dir, reader_mode="auto")` | returns `tuple[pd.DataFrame, dict]` |
| io.py:707 | `_read_single_asd_ascii(asd_file, reader_mode)` | returns `pd.Series` |
| io.py:776 | `_handle_binary_asd(asd_file, reader_mode)` | returns `pd.Series or None` |
| asd_native.py:10 | `read_binary_asd(asd_file)` | stub — raises NotImplementedError |
| asd_r_bridge.py:17 | `check_r_available()` | returns `bool` |
| asd_r_bridge.py:33 | `read_asd_with_r(asd_file, r_package="asdreader")` | stub |

**Depends on**: P2-13

---

### P2-18: readers/opus.py

**Create**: `src/spectral_predict/readers/opus.py`
**Port from**: `readers/opus_reader.py` (383 lines), `io.py` OPUS portions

**What to copy — exact signatures**:
| Source | Function | Signature |
|--------|----------|-----------|
| opus_reader.py:16 | `read_opus_file(filepath)` | returns `tuple[pd.Series, dict]` |
| opus_reader.py:187 | `read_opus_dir(directory, pattern="*.[0-9]*")` | returns `tuple[pd.DataFrame, dict]` |
| opus_reader.py:337 | `convert_wavenumber_to_wavelength(wavenumber_cm: float)` | returns `float` |
| opus_reader.py:361 | `convert_wavelength_to_wavenumber(wavelength_nm: float)` | returns `float` |

**Depends on**: P2-13

---

### P2-19: readers/spc.py

**Create**: `src/spectral_predict/readers/spc.py`
**Port from**: `io.py` SPC portions

**What to copy**:
| Line | Function | Signature |
|------|----------|-----------|
| 842 | `read_spc_dir(spc_dir)` | returns `tuple[pd.DataFrame, dict]` |
| 2987 | `read_spc_file(path)` | returns `tuple[pd.DataFrame, dict]` |
| 3438 | `write_spc_file(data, path, metadata=None)` | returns `None` |

**Depends on**: P2-13

---

### P2-20: readers/jcamp.py

**Create**: `src/spectral_predict/readers/jcamp.py`
**Port from**: `io.py` JCAMP portions

**What to copy**:
| Line | Function | Signature |
|------|----------|-----------|
| 1483 | `read_jcamp_file(path)` | returns `tuple[pd.Series, dict]` |
| 1558 | `read_jcamp_dir(jcamp_dir)` | returns `tuple[pd.DataFrame, dict]` |
| 1666 | `write_jcamp(df, output_dir, title_prefix="spectrum", xunits="1/CM", yunits="ABSORBANCE", metadata=None)` | returns `list` |
| 3495 | `write_jcamp_file(data, path, metadata=None, title=None, data_type='INFRARED SPECTRUM', xunits='NANOMETERS', yunits='REFLECTANCE')` | returns `None` |

**Depends on**: P2-13

---

### P2-21: readers/ascii.py

**Create**: `src/spectral_predict/readers/ascii.py`
**Port from**: `io.py` ASCII portions

**What to copy**:
| Line | Function | Signature |
|------|----------|-----------|
| 1747 | `read_ascii_spectra(path)` | returns `tuple[pd.DataFrame, dict]` |
| 1823 | `_read_ascii_dir(directory)` | returns `tuple[pd.DataFrame, dict]` |
| 1917 | `_parse_ascii_file(filepath)` | returns `tuple[pd.DataFrame, str, str] or None` |
| 3548 | `write_ascii_spectra(data, path, metadata=None, delimiter='\t', include_header=True)` | returns `None` |

**Depends on**: P2-13

---

### P2-22: readers/perkinelmer.py

**Create**: `src/spectral_predict/readers/perkinelmer.py`
**Port from**: `readers/perkinelmer_reader.py` (294 lines)

**What to copy**:
| Line | Function | Signature |
|------|----------|-----------|
| 15 | `read_sp_file(filepath)` | returns `tuple[pd.Series, dict]` |
| 155 | `read_sp_dir(directory)` | returns `tuple[pd.DataFrame, dict]` |

**Depends on**: P2-13

---

### P2-23: readers/agilent.py

**Create**: `src/spectral_predict/readers/agilent.py`
**Port from**: `readers/agilent_reader.py` (426 lines)

**What to copy**:
| Line | Function | Signature |
|------|----------|-----------|
| 21 | `read_agilent_file(filepath, extract_mode='total')` | returns `tuple[pd.Series, dict]` |
| 194 | `read_agilent_dir(directory, extensions=None, extract_mode='total')` | returns `tuple[pd.DataFrame, dict]` |
| 361 | `read_seq_file(filepath, extract_mode='total')` | convenience wrapper |
| 384 | `read_dmt_file(filepath, extract_mode='total')` | convenience wrapper |
| 407 | `read_asp_file(filepath)` | convenience wrapper |

**Depends on**: P2-13

---

### P2-24: readers/dispatcher.py

**Create**: `src/spectral_predict/readers/dispatcher.py`
**Port from**: `io.py` lines 2317-2569, 2460-2569

**What to copy**:
| Line | Function | Signature |
|------|----------|-----------|
| 2317 | `read_spectra(path, format='auto', **kwargs)` | Universal reader dispatcher → returns `tuple[pd.DataFrame, dict]` |
| 2460 | `write_spectra(data, path, format, metadata=None, **kwargs)` | Universal writer dispatcher → returns `None` |

**Depends on**: P2-14 through P2-23 (all format readers)

**Verification**:
```python
from spectral_predict.readers.dispatcher import read_spectra, write_spectra
# These should import cleanly; full functional tests need example data files
```

---

### Phase 2C: Models (10 tasks)

### P2-25 through P2-34

These tasks port `models.py` (1776 lines), `neural_boosted.py` (1006 lines), `model_config.py` (345 lines), and `model_registry.py` (177 lines) into:

| Task | File | Contents | Port from |
|------|------|----------|-----------|
| P2-25 | `models/base.py` | `BaseModelWrapper` interface | New |
| P2-26 | `models/pls.py` | `PLSTransformer` class (lines 22-91) | models.py |
| P2-27 | `models/linear.py` | Ridge, Lasso, ElasticNet factories | models.py |
| P2-28 | `models/trees.py` | RandomForest, XGBoost, LightGBM, CatBoost factories | models.py |
| P2-29 | `models/svm.py` | SVR, SVC factories | models.py |
| P2-30 | `models/neural.py` | MLP + `NeuralBoostedRegressor` (line 17, 170+ params), `NeuralBoostedClassifier` (line 416) | models.py, neural_boosted.py |
| P2-31 | `models/grids.py` | First half of `get_model_grids()` (lines 506-1100): PLS/Ridge/Lasso/ElasticNet/RF grids | models.py |
| P2-32 | `models/grids_advanced.py` | Second half: XGB/LGBM/CatBoost/SVR/MLP/NeuralBoosted grids + `compute_vip()` (line 1660) + `get_feature_importances()` (line 1705) | models.py |
| P2-33 | `models/registry.py` | `get_supported_models()`, `supports_feature_importance()`, `supports_subset_analysis()`, `is_valid_model()`, `get_tier_models()`, `get_hyperparameters()`, `print_tier_summary()` | model_registry.py, model_config.py |
| P2-34 | `models/get_model.py` | `get_model(model_name, task_type, n_components, ...)` (line 94), `build_model(model_name, params, task_type)` (line 340) | models.py |

**Critical signatures from models.py**:
- `get_model(model_name, task_type='regression', n_components=10, max_n_components=10, max_iter=500, n_jobs=-1)` → estimator
- `build_model(model_name, params, task_type='regression')` → estimator
- `get_model_grids(task_type, n_features, max_n_components=10, max_iter=500, ...[60+ keyword args]..., tier='standard', enabled_models=None, n_jobs=-1)` → dict
- `compute_vip(pls_model, X, y)` → ndarray
- `get_feature_importances(model, model_name, X, y)` → ndarray

**Critical signatures from neural_boosted.py**:
- `NeuralBoostedRegressor.__init__(n_estimators=100, learning_rate=0.1, hidden_layer_size=3, activation='tanh', alpha=0.0001, max_iter=100, early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, loss='mse', huber_delta=1.35, random_state=None, verbose=0)`
- `NeuralBoostedClassifier.__init__(n_estimators=100, learning_rate=0.1, hidden_layer_size=5, activation='tanh', alpha=0.0001, max_iter=100, early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, early_stopping_metric='accuracy', class_weight=None, random_state=None, verbose=0)`

**Depends on**: P1-04, P1-07

---

### Phase 2D: Search (15 tasks)

### P2-35 through P2-49

These tasks port the search system into focused modules:

| Task | File | Contents | Port from | Lines |
|------|------|----------|-----------|-------|
| P2-35 | `search/cv.py` | `is_boosting_model()`, `cross_validate_with_early_stopping()`, `cross_val_predict_with_early_stopping()`, `cross_val_score_with_early_stopping()` | cv_utils.py | 523 |
| P2-36 | `search/results.py` | `compute_composite_score()`, `create_results_dataframe()`, `add_result()`, `compute_imbalance_metrics()`, `compute_specificity()` | scoring.py | 577 |
| P2-37 | `search/controller.py` | `SearchController` class (pause/resume/stop/check_and_wait) | search_controller.py | 74 |
| P2-38 | `search/grid.py` | `_run_single_fold()`, `_run_single_config()`, helper functions | search.py:3199-end | ~500 |
| P2-39 | `search/phases.py` | Full spectrum, variable selection, region, iPLS phase runners + edge masking | search.py (extracted) | ~500 |
| P2-40 | `search/orchestrator.py` | `run_search()` main coordinator + `compute_validation_metrics_for_top_models()` | search.py:734-2614 | ~500 |
| P2-41 | `search/bayesian.py` | `suggest_preprocessing()`, `suggest_model_params()`, preprocessing helpers | unified_bayesian.py:88-460 | ~500 |
| P2-42 | `search/bayesian_objective.py` | `create_unified_objective()`, `run_unified_bayesian()`, `convert_study_to_dataframe()` | unified_bayesian.py:543-end | ~500 |
| P2-43 | `search/bayesian_config.py` | `get_bayesian_search_space()` + per-model space functions | bayesian_config.py | 575 |
| P2-44 | `search/bayesian_utils.py` | `create_optuna_study()`, `create_objective_function()` | bayesian_utils.py:34-611 | ~500 |
| P2-45 | `search/bayesian_reporting.py` | `print_optimization_summary()`, `ProgressCallback` | bayesian_utils.py:789-end | ~350 |
| P2-46 | `search/nsga2.py` | `SmartMutation`, `ImportanceTracker`, `SeededWavelengthSampling` | nsga2_search.py:155-544 | ~500 |
| P2-47 | `search/nsga2_problem.py` | `SpectralOptimizationProblem`, decode functions | nsga2_search.py:653-1574 | ~500 |
| P2-48 | `search/nsga2_runner.py` | `run_nsga2_search()`, `decode_solution()`, `pareto_to_dataframe()` | nsga2_search.py:1700-2439 | ~500 |
| P2-49 | `search/nsga2_metrics.py` | Solution metrics, `convert_nsga2_to_v1_format()` | nsga2_search.py:2439-end | ~500 |

**Critical signature — `run_search()`** (search.py line 734): This function has 70+ parameters. The new version should accept a `SearchConfig` dataclass instead, with a compatibility shim for the old signature.

**Depends on**: P2-07 (preprocessing), P2-34 (models), P1-04 (constants)

---

### Phase 2E: Variable Selection (11 tasks)

### P2-50 through P2-60

| Task | File | Contents | Port from |
|------|------|----------|-----------|
| P2-50 | `selection/base.py` | `BaseSelector` interface | New |
| P2-51 | `selection/importance.py` | Re-export `compute_importance()` from `preprocessing/discovery.py` | preprocessing_discovery.py |
| P2-52 | `selection/uve.py` | `uve_selection(X, y, cutoff_multiplier=1.0, n_components=None, cv_folds=5, random_state=42)` → ndarray, `get_uve_threshold()` → tuple | variable_selection.py:21-299 |
| P2-53 | `selection/spa.py` | `spa_selection(X, y, n_features, n_random_starts=10, cv_folds=5, random_state=42)` → ndarray | variable_selection.py:300-500 |
| P2-54 | `selection/cars.py` | `cars_selection(X, y, n_iterations=50, pls_components=5, cv_folds=5, monte_carlo_samples=80, random_state=42, model_type=None, use_hybrid_importance=False, hybrid_importance_weight=0.5, task_type='regression')` → ndarray | variable_selection.py:824-1184 |
| P2-55 | `selection/uve_spa.py` | `uve_spa_selection(X, y, n_features, ...)` → ndarray (hybrid) | variable_selection.py:689-823 |
| P2-56 | `selection/ipls.py` | `ipls_selection()`, `ipls_forward(X, y, wavelengths, n_intervals=20, max_combine=5, cv_folds=5, random_state=42)` → list[dict], `ipls_backward(X, y, wavelengths, n_intervals=20, cv_folds=5, random_state=42, min_intervals=1)` → list[dict], `_create_intervals()`, `_evaluate_interval_pls()`, `_get_combined_indices()`, `_get_wavelength_ranges()` | variable_selection.py:501-688, 1185-end |
| P2-57 | `selection/vcpa.py` | `vcpa_iriv(X, y, n_outer_iterations=10, n_inner_iterations=50, pls_components=5, cv_folds=5, binary_matrix_samples=100, importance_threshold=0.5, model_type=None, random_state=None)` → Dict, `compare_selection_methods()` | wavelength_selection.py |
| P2-58 | `selection/regions.py` | `compute_region_correlations(X, y, wavelengths, region_size=50, overlap=25)`, `get_top_regions()`, `get_region_variable_indices()`, `create_region_subsets()`, `format_region_report()` | regions.py (305 lines) |
| P2-59 | `selection/ga.py` | `FitnessCache`, `ga_pls_selection(X, y, population_size=64, n_generations=100, ...)` → ndarray, `ga_pls_selection_detailed()` | ga_pls.py (763 lines) |
| P2-60 | `selection/ga_lightgbm.py` | `FitnessCache`, `ga_lightgbm_selection(X, y, ...)` → ndarray, `ga_lightgbm_selection_detailed()` | ga_lightgbm.py (754 lines) |

**Depends on**: P1-07, P2-07

---

### Phase 2F: Ensemble, Transfer, Analysis, Library, Export, Utils, Data Management (34 tasks)

### P2-61 through P2-94

| Task | File | Contents | Port from | Lines |
|------|------|----------|-----------|-------|
| **Ensemble** | | | | |
| P2-61 | `ensemble/methods.py` | `SimpleAverageEnsemble`, `RegionBasedAnalyzer`, `RegionAwareWeightedEnsemble`, `MixtureOfExpertsEnsemble`, `StackingEnsemble` | ensemble.py:21-935 | ~500 |
| P2-62 | `ensemble/specialists.py` | `RegionSpecialistEnsemble`, `ClassSpecialistEnsemble` | ensemble.py:936-1264 | ~330 |
| P2-63 | `ensemble/auto.py` | `extract_preprocessor_config()`, `create_auto_ensembles()`, `compute_regional_rankings()`, `compute_class_rankings()`, `select_top_models_per_region()`, `select_top_models_quartile_flat()` | ensemble.py:1265-end | ~500 |
| P2-64 | `ensemble/preprocessing.py` | `StackedPreprocessingRegressor`, `StackedPreprocessingClassifier`, `create_standard_preprocessing_ensemble()` | ensemble_preprocessing.py (701 lines) | ~500 |
| P2-65 | `ensemble/viz.py` | `plot_regional_performance()`, `plot_ensemble_weights()`, `plot_model_specialization_profile()`, `plot_prediction_comparison()`, `create_ensemble_report()` | ensemble_viz.py (779 lines) | ~500 |
| **Transfer** | | | | |
| P2-66 | `transfer/methods.py` | `estimate_ds()`, `apply_ds()`, `estimate_pds()`, `apply_pds()`, `estimate_tsr()`, `apply_tsr()`, `estimate_ctai()`, `apply_ctai()`, `apply_transfer_dispatch()` | calibration_transfer.py:114-970 | ~500 |
| P2-67 | `transfer/methods_advanced.py` | `estimate_nspfce()`, `apply_nspfce()`, `estimate_jypls_inv()`, `apply_jypls_inv()` | calibration_transfer.py:971-1649 | ~500 |
| P2-68 | `transfer/io.py` | `TransferModel` dataclass, `save_transfer_model()`, `load_transfer_model()` | calibration_transfer.py:26-379 | ~130 |
| P2-69 | `transfer/utils.py` | `resample_to_grid()`, `clip_wavelengths_to_region()` | calibration_transfer.py:40-111 | ~80 |
| **Analysis** | | | | |
| P2-70 | `analysis/outliers.py` | `run_pca_outlier_detection()`, `compute_q_residuals()`, `compute_mahalanobis_distance()`, `check_y_data_consistency()`, `generate_outlier_report()` | outlier_detection.py (612 lines) | ~490 |
| P2-71 | `analysis/interference.py` | `WavelengthExcluder`, `OSC`, `EPO` | interference.py:75-820 (excluding MSC) | ~500 |
| P2-72 | `analysis/interference_advanced.py` | `GLSW`, `DOSC` | interference.py:604-1445 | ~500 |
| P2-73 | `analysis/contaminant.py` | `DifferenceAnalyzer`, `EstimatedEPO` | contaminant_analysis.py:80-800 | ~500 |
| P2-74 | `analysis/contaminant_methods.py` | `ContaminantOPLSDA`, `ContaminantGLSW`, `RegionExcluder`, `analyze_contaminant_influence()` | contaminant_analysis.py:801-1837 | ~500 |
| P2-75 | `analysis/contaminant_multi.py` | `MultiContaminantAnalyzer`, `MultiGroupEPO`, `MultiContaminantGLSW`, `analyze_multiple_contaminants()` | contaminant_analysis.py:1838-end | ~490 |
| P2-76 | `analysis/imbalance.py` | `detect_class_imbalance()`, `detect_regression_imbalance()`, `ClassificationResampler`, `RegressionUndersampler`, `RegressionResampler` | imbalance.py:61-779 | ~500 |
| P2-77 | `analysis/imbalance_utils.py` | `RegressionSampleWeighter`, `build_imbalance_transformer()`, `get_available_methods()`, `recommend_imbalance_method()`, `validate_classification_config()`, `validate_imbalance_with_features()` | imbalance.py:780-end | ~500 |
| P2-78 | `analysis/sample_selection.py` | `kennard_stone()`, `duplex()`, `spxy()`, `random_selection()`, `compare_selection_methods()` | sample_selection.py (556 lines) | ~490 |
| P2-79 | `analysis/similarity.py` | `hit_quality_index()`, `spectral_angle_mapper()`, `sam_to_similarity()`, `euclidean_distance()`, `euclidean_to_similarity()`, `cosine_similarity()`, `first_derivative_correlation()`, `second_derivative_correlation()`, `spectral_information_divergence()`, `sid_to_similarity()`, `compute_similarity()`, `compute_batch_similarity()`, `METRICS` registry | similarity_metrics.py (489 lines) | ~488 |
| P2-80 | `analysis/diagnostics.py` | `compute_residuals()`, `compute_leverage()`, `qq_plot_data()`, `jackknife_prediction_intervals()`, `compute_pls_complexity_curve()`, `compute_sklearn_validation_curve()` | diagnostics.py:25-478 | ~500 |
| P2-81 | `analysis/diagnostics_curves.py` | `compute_ensemble_validation_curve()`, `compute_regularization_validation_curve()`, `compute_learning_curve()` | diagnostics.py:481-853 | ~370 |
| **Library** | | | | |
| P2-82 | `library/library.py` | `LibraryEntry` dataclass, `SpectralLibrary` class, `get_library()`, `add_to_library()` | library_search.py:38-632 | ~430 |
| P2-83 | `library/search.py` | `search_library()` | library_search.py:635-664 | ~40 |
| **Export** | | | | |
| P2-84 | `export/model_io.py` | `save_model()`, `load_model()`, `predict_with_model()`, `predict_with_uncertainty()` | model_io.py:51-838 | ~500 |
| P2-85 | `export/model_io_utils.py` | `get_model_info()`, `save_ensemble()`, `load_ensemble()`, serialization helpers | model_io.py:841-end | ~500 |
| P2-86 | `export/code_gen.py` | `ExportOptions` dataclass, `CodeGenerator` (main methods) | code_generator.py first half | ~500 |
| P2-87 | `export/code_gen_render.py` | `CodeGenerator` render methods, `generate_script_from_config()`, `generate_notebook_from_config()` | code_generator.py second half | ~500 |
| P2-88 | `export/r_gen.py` | `RCodeGenerator`, `generate_r_script_from_config()` | r_code_generator.py (149 lines) | ~149 |
| P2-89 | `export/bundle.py` | `ExportBundle`, `create_export_bundle()` | export_bundle.py (444 lines) | ~444 |
| P2-90 | `export/report.py` | `write_markdown_report()` | report.py (145 lines) | ~145 |
| P2-91 | `export/templates/` | Direct copy of all 7 template files | templates/ (~1488 lines) | ~1488 |
| **Utils** | | | | |
| P2-92 | `utils/instruments.py` | `InstrumentProfile`, `characterize_instrument()`, `compute_wavelength_spacing()`, `compute_roughness()`, `detect_interpolation()`, `analyze_peaks()`, `choose_common_grid()`, `build_equalization_mapping_for_instrument()`, `equalize_dataset()` | instrument_profiles.py (459 lines), equalization.py (148 lines) | ~500 |
| P2-93 | `utils/progress.py` | `ProgressState` dataclass (backend only, no Tkinter) — Qt layer observes this | progress_monitor.py (replaced) | ~100 |
| **Data Management** | | | | |
| P2-94 | `data_management/manager.py` | `DataSource` dataclass, `MergedDataset` dataclass, `DataSourceManager` (add/remove/merge/filter/trim) | data_management.py (~750 lines) | ~500 |

---

## Phase 3: GUI Shell (12 tasks)

### P3-01: gui/app.py — Main window

**Create**: `src/spectral_predict/gui/app.py`

```python
"""Main application window with sidebar navigation."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QStackedWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt
from qt_material import apply_stylesheet


class MainWindow(QMainWindow):
    """Main ASP application window.

    Layout: sidebar (QListWidget) + content area (QStackedWidget).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ASP — Spectral Predict")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Sidebar navigation
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.currentRowChanged.connect(self._on_tab_changed)
        layout.addWidget(self.sidebar)

        # Tab container
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self._register_tabs()

    def _register_tabs(self) -> None:
        """Register all 15 tabs."""
        # Will be populated in Phase 4
        pass

    def _on_tab_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)


def run_app() -> None:
    """Entry point for the application."""
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

**Depends on**: P1-07

**Verification**: `python -c "from spectral_predict.gui.app import MainWindow"` (import check only)

---

### P3-02 through P3-12: Remaining GUI shell

| Task | File | Contents | Depends on |
|------|------|----------|------------|
| P3-02 | `gui/state.py` | `AppState(QObject)` with Qt signals | P1-05, P3-01 |
| P3-03 | `gui/tabs/base.py` | `BaseTab(QWidget)` interface | P3-02 |
| P3-04 | `gui/widgets/spectra_plot.py` | Matplotlib canvas + navigation toolbar | P3-01 |
| P3-05 | `gui/widgets/data_table.py` | QTableView + QAbstractTableModel for DataFrames | P3-01 |
| P3-06 | `gui/widgets/collapsible.py` | Expandable/collapsible section | P3-01 |
| P3-07 | `gui/widgets/file_browser.py` | File/dir picker with format detection | P3-01 |
| P3-08 | `gui/widgets/card.py` | Styled card container | P3-01 |
| P3-09 | `gui/widgets/progress_bar.py` | Progress bar + ETA + stage label | P3-01 |
| P3-10 | `gui/services/file_service.py` | Wraps reader dispatch for GUI | P2-24, P3-02 |
| P3-11 | `gui/services/search_service.py` | QThread search worker + signals | P2-40, P2-37, P3-02 |
| P3-12 | `gui/services/export_service.py` | Wraps model I/O, code gen, bundle | P2-84, P2-86, P2-89 |

---

## Phase 4: Wire Tabs (15 tasks)

Each tab is one file under 500 lines. Replicates **every** control from the current GUI.

| Task | File | Tab | Subtabs | Key features | Depends on |
|------|------|-----|---------|-------------|------------|
| P4-01 | `gui/tabs/import_tab.py` | Import | 2 (Data, Plots) | File browsers, format detection, advanced options, Load Data, Append, wavelength range restriction, data type selection | P3-02, P3-04, P3-07, P3-10 |
| P4-02 | `gui/tabs/explore.py` | Explore | 9 | Raw, Deriv1, Deriv2 (SG window spinbox), Target, Screening (VIP/RF, Top-N), Rubber Band BL, Poly BL, ALS BL, Manual BL. Color-by dropdown. Save Corrected / Replace Working Data. | P3-04, P2-02, P2-03, P2-04 |
| P4-03 | `gui/tabs/data_viewer.py` | Data Viewer | 0 | Spreadsheet, Show excluded, Export CSV, Save Changes, Undo, Add/Delete Column, Delete Rows, Exclude/Include | P3-05 |
| P4-04 | `gui/tabs/quality_check.py` | Quality Check | 0 | PCA components spinbox, Y range, Run Outlier Detection, Export Report | P2-70, P3-04 |
| P4-05 | `gui/tabs/analysis_config.py` | Analysis Config | 5 | Basic Settings (preprocessing, baseline: polynomial/als/rubber_band/airpls, smoothing, discovery), Variable Selection (methods, iPLS forward/backward), Model Config (tier, optimization method), Ensemble, Validation | P3-06, P3-08, P1-06 |
| P4-06 | `gui/tabs/analysis_progress.py` | Progress | 0 | Animated indicator, best model, ETA, Pause/Resume/Stop, colored output | P3-09, P3-11 |
| P4-07 | `gui/tabs/results.py` | Results | 0 | Ranked table, quartile highlighting, overfit indicator, double-click loads Model Dev | P3-05, P2-36 |
| P4-08 | `gui/tabs/model_dev.py` | Model Dev | 4 | Selection, Features, Configuration, Diagnostics | P2-80, P3-04 |
| P4-09 | `gui/tabs/prediction.py` | Prediction | 2 | Load models, data source, Run All Models, results | P2-84, P3-05 |
| P4-10 | `gui/tabs/multi_model.py` | Multi-Model | 0 | Multi-model comparison | P2-84 |
| P4-11 | `gui/tabs/calibration.py` | Cal Transfer | 0 | DS, PDS, TSR, CTAI, NSPFCE, JYPLS-INV. **Reflectance↔absorbance conversion**. Region of interest. Save/load transfer models. | P2-66, P2-67, P2-68, P2-69 |
| P4-12 | `gui/tabs/interference.py` | Interference | 4 | Library, Config (OSC/EPO/GLSW/DOSC/WavelengthExcluder), Application, Diagnostics | P2-71, P2-72 |
| P4-13 | `gui/tabs/spectral_library.py` | Spectral Library | 2 | Management (add/remove/clear/batch/categories/export), Similarity Search (HQI/SAM/euclidean/cosine/deriv1/deriv2/SID) | P2-82, P2-83 |
| P4-14 | `gui/tabs/contaminant.py` | Contaminant | 4 | Groups, Difference, Detection (EstimatedEPO/OPLS-DA/GLSW/multi), Validate | P2-73, P2-74, P2-75 |
| P4-15 | `gui/tabs/data_management.py` | Data Mgmt | 4 | Sources (TreeView, Add/Remove/Save Config), Merge (alignment, duplicates), Manipulation (filtering, trimming, spectral conversion), View (spreadsheet) | P2-94, P3-05, P3-07 |

---

## Phase 5: Test and Validate (3 tasks)

### P5-01: Backend unit tests

One test file per backend module. Minimum 3-5 tests each:
- Happy path with known inputs
- Edge cases (empty data, single sample, single feature)
- Output shape and type validation
- Round-trip (e.g., save/load model)

**Depends on**: All Phase 2

### P5-02: Integration tests with example data

Load example data through the new system. Verify:
- Same data in old and new apps produces identical preprocessing results
- Same search config produces same ranked results
- Model save/load round-trips correctly

**Depends on**: P5-01

### P5-03: Feature parity checklist

Manual verification: every button, checkbox, dropdown, and spinbox from the current 15-tab GUI has an equivalent in the new GUI.

**Depends on**: All Phase 4

---

## File Mapping: Current → New

| Current File | Lines | New Location(s) |
|-------------|-------|-----------------|
| `constants.py` | 7 | `core/constants.py` |
| `preprocess.py` | 491 | `preprocessing/snv.py`, `derivatives.py`, `pipeline.py` |
| `baseline.py` | 360 | `preprocessing/baseline.py` |
| `preprocessing_wrapper.py` | 220 | `preprocessing/pipeline.py` |
| `preprocessing_discovery.py` | 987 | `preprocessing/discovery.py` |
| `ga_preprocessing.py` | 1806 | `preprocessing/ga_optimization.py`, `ga_search.py` |
| `learned_preprocessing.py` | 775 | `preprocessing/learned.py` |
| `io.py` | 3708 | `readers/csv.py`, `excel.py`, `asd.py`, `spc.py`, `jcamp.py`, `ascii.py`, `alignment.py`, `detection.py`, `dispatcher.py` |
| `readers/opus_reader.py` | 383 | `readers/opus.py` |
| `readers/perkinelmer_reader.py` | 294 | `readers/perkinelmer.py` |
| `readers/agilent_reader.py` | 426 | `readers/agilent.py` |
| `readers/asd_native.py` | 62 | `readers/asd.py` |
| `readers/asd_r_bridge.py` | 137 | `readers/asd.py` |
| `models.py` | 1776 | `models/pls.py`, `linear.py`, `trees.py`, `svm.py`, `grids.py`, `grids_advanced.py`, `get_model.py` |
| `neural_boosted.py` | 1006 | `models/neural.py` |
| `model_config.py` | 345 | `core/constants.py`, `models/registry.py` |
| `model_registry.py` | 177 | `models/registry.py` |
| `search.py` | 4193 | `search/orchestrator.py`, `grid.py`, `phases.py` |
| `unified_bayesian.py` | 1702 | `search/bayesian.py`, `bayesian_objective.py` |
| `bayesian_config.py` | 575 | `search/bayesian_config.py` |
| `bayesian_utils.py` | 1036 | `search/bayesian_utils.py`, `bayesian_reporting.py` |
| `nsga2_search.py` | 3957 | `search/nsga2.py`, `nsga2_problem.py`, `nsga2_runner.py`, `nsga2_metrics.py` |
| `cv_utils.py` | 523 | `search/cv.py` |
| `scoring.py` | 577 | `search/results.py` |
| `search_controller.py` | 74 | `search/controller.py` |
| `variable_selection.py` | 1697 | `selection/uve.py`, `spa.py`, `cars.py`, `uve_spa.py`, `ipls.py` |
| `wavelength_selection.py` | 801 | `selection/vcpa.py` |
| `regions.py` | 305 | `selection/regions.py` |
| `ga_pls.py` | 763 | `selection/ga.py` |
| `ga_lightgbm.py` | 754 | `selection/ga_lightgbm.py` |
| `ensemble.py` | 1965 | `ensemble/methods.py`, `specialists.py`, `auto.py` |
| `ensemble_preprocessing.py` | 701 | `ensemble/preprocessing.py` |
| `ensemble_viz.py` | 779 | `ensemble/viz.py` |
| `calibration_transfer.py` | 1746 | `transfer/methods.py`, `methods_advanced.py`, `io.py`, `utils.py` |
| `outlier_detection.py` | 612 | `analysis/outliers.py` |
| `interference.py` | 1445 | `preprocessing/msc.py`, `analysis/interference.py`, `interference_advanced.py` |
| `contaminant_analysis.py` | 2727 | `analysis/contaminant.py`, `contaminant_methods.py`, `contaminant_multi.py` |
| `imbalance.py` | 1302 | `analysis/imbalance.py`, `imbalance_utils.py` |
| `sample_selection.py` | 556 | `analysis/sample_selection.py` |
| `similarity_metrics.py` | 489 | `analysis/similarity.py` |
| `diagnostics.py` | 853 | `analysis/diagnostics.py`, `diagnostics_curves.py` |
| `library_search.py` | 665 | `library/library.py`, `search.py` |
| `model_io.py` | 1402 | `export/model_io.py`, `model_io_utils.py` |
| `code_generator.py` | 1680 | `export/code_gen.py`, `code_gen_render.py` |
| `r_code_generator.py` | 149 | `export/r_gen.py` |
| `export_bundle.py` | 444 | `export/bundle.py` |
| `report.py` | 145 | `export/report.py` |
| `templates/` | 1488 | `export/templates/` (direct copy) |
| `instrument_profiles.py` | 459 | `utils/instruments.py` |
| `equalization.py` | 148 | `utils/instruments.py` |
| `progress_monitor.py` | 435 | `utils/progress.py` (backend only) |
| `data_management.py` | 750 | `data_management/manager.py` |

## Files NOT Ported (intentionally excluded)

| File | Reason |
|------|--------|
| `interactive.py` | CLI interactive mode, replaced by GUI |
| `interactive_gui.py` | Old Tkinter interactive GUI, replaced |
| `cli.py` | Will be rewritten for new structure if needed |
| `resource_paths.py` | Tkinter resource paths, not needed in Qt |
| `nsga2_search.py.backup` | Backup file |

---

## Task Count Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1 | 7 | Project skeleton, core module |
| Phase 2A | 11 | Preprocessing |
| Phase 2B | 13 | Readers |
| Phase 2C | 10 | Models |
| Phase 2D | 15 | Search |
| Phase 2E | 11 | Selection |
| Phase 2F | 34 | Ensemble, Transfer, Analysis, Library, Export, Utils, DataMgmt |
| Phase 3 | 12 | GUI shell, widgets, services |
| Phase 4 | 15 | Wire tabs |
| Phase 5 | 3 | Test and validate |
| **Total** | **131** | |

---

## Execution Batches

For use with `superpowers:execute-plan`:

| Batch | Tasks | Review checkpoint |
|-------|-------|-------------------|
| 1 | P1-01 through P1-07 | Core types import correctly |
| 2 | P2-01 through P2-07 | Preprocessing pipeline builds and transforms data |
| 3 | P2-08 through P2-11 | Discovery + GA preprocessing works |
| 4 | P2-12 through P2-24 | All readers import, dispatcher works |
| 5 | P2-25 through P2-34 | Models build and predict |
| 6 | P2-35 through P2-49 | Search orchestrator runs |
| 7 | P2-50 through P2-60 | Variable selection methods work |
| 8 | P2-61 through P2-65 | Ensembles fit/predict |
| 9 | P2-66 through P2-69 | Calibration transfer methods work |
| 10 | P2-70 through P2-81 | Analysis modules work |
| 11 | P2-82 through P2-94 | Library, export, utils, data management work |
| 12 | P3-01 through P3-12 | GUI shell launches with sidebar |
| 13 | P4-01 through P4-08 | Core tabs wired (Import through Model Dev) |
| 14 | P4-09 through P4-15 | Advanced tabs wired |
| 15 | P5-01 through P5-03 | All tests pass, feature parity verified |
