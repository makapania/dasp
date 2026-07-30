"""Contract tests for the composition primitives documented in
``docs/AGENT_COMPOSITION.md``.

Why this file exists
--------------------
Research scripts (including agent-written ones) do not call ``run_search``; they
compose their own search loop from DASP's chemometric primitives because each
project owns its own sampling design, CV splitter, ranking objective, and
reporting. At least one such pipeline lived outside this repo and imported the
*private* ``search._multiclass_varsel_mask`` — so a rename here would have broken
it silently, with no test anywhere catching it.

These tests pin the documented surface so a refactor fails HERE instead of on
someone else's machine.

Deliberately repo-local: nothing below reads a path outside the repository, so it
runs for any user on any machine from a clean clone.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import numpy as np
import pytest


# The documented composition surface. Keep in sync with docs/AGENT_COMPOSITION.md.
# Adding a name here is a promise; removing one is a breaking change.
PRIMITIVES: dict[str, list[str]] = {
    "io": [
        "read_spectra",
        "read_asd_dir",
        "read_csv_spectra",
        "read_reference_csv",
        "align_xy",
    ],
    "preprocess": ["build_preprocessing_pipeline"],
    "unified_bayesian": ["apply_preprocessing", "run_unified_bayesian"],
    "variable_selection": [
        "cars_selection",
        "ipls_selection",
        "spa_selection",
        "uve_selection",
    ],
    "simca": ["MultiClassClassModel"],
    "contamination": ["PCASIMCA"],
    "models": ["PLSTransformer"],
    "model_io": ["save_model", "load_model", "predict_with_model"],
    "search": [
        "run_search",
        "run_one_class_search",
        "run_multiclass_simca_search",
        "multiclass_varsel_mask",
    ],
}


@pytest.mark.parametrize(
    "module_name,symbol",
    [(m, s) for m, symbols in PRIMITIVES.items() for s in symbols],
)
def test_documented_primitive_is_importable(module_name: str, symbol: str) -> None:
    """Every primitive named in the composition guide must exist and be callable."""
    module = importlib.import_module(f"spectral_predict.{module_name}")
    assert hasattr(module, symbol), (
        f"spectral_predict.{module_name}.{symbol} is documented in "
        f"docs/AGENT_COMPOSITION.md but does not exist"
    )
    assert callable(getattr(module, symbol)), (
        f"spectral_predict.{module_name}.{symbol} is documented as callable"
    )


def test_search_declares_its_public_surface() -> None:
    """search.py declares __all__ so 'internal detail' is explicit, not implied."""
    from spectral_predict import search

    assert hasattr(search, "__all__"), "search.py must declare __all__"
    for name in PRIMITIVES["search"]:
        assert name in search.__all__, f"{name} missing from search.__all__"


def _graded_multiclass(n_per_class=(30, 30, 30), n_features=40, seed=0):
    """Synthetic >2-class spectra with a class-dependent gradient.

    Self-contained on purpose — no reliance on repo example data, so this test
    travels to any clone.
    """
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for class_index, n in enumerate(n_per_class):
        base = rng.normal(0.5, 0.02, (n, n_features))
        # Give each class a distinguishable ramp so selection has real signal.
        base += np.linspace(0, 0.1 * (class_index + 1), n_features)
        blocks.append(base)
        labels.extend([f"class{class_index}"] * n)
    return np.vstack(blocks), np.asarray(labels)


class TestMulticlassVarselMaskContract:
    """The one primitive that was private while being depended on externally."""

    def test_public_name_returns_documented_shape(self) -> None:
        from spectral_predict.search import multiclass_varsel_mask

        X, y = _graded_multiclass(n_features=40, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = multiclass_varsel_mask(
                X, y, np.arange(X.shape[1]), "importance", 10
            )

        assert isinstance(mask, np.ndarray), "documented to return an ndarray mask"
        assert mask.dtype == bool, "documented to return a boolean mask"
        assert mask.shape == (X.shape[1],), "mask must be one entry per feature"
        assert mask.sum() == 10, "must keep exactly the requested Top-N"

    def test_none_method_returns_none(self) -> None:
        """'none' is documented to mean 'no selection', not an error."""
        from spectral_predict.search import multiclass_varsel_mask

        X, y = _graded_multiclass(n_features=40, seed=3)
        assert (
            multiclass_varsel_mask(X, y, np.arange(X.shape[1]), "none", 10) is None
        )

    def test_private_alias_still_works_for_external_callers(self) -> None:
        """Back-compat: a pipeline outside this repo imports the underscore name.

        This is the exact regression that motivated the public promotion.
        """
        from spectral_predict.search import (
            _multiclass_varsel_mask,
            multiclass_varsel_mask,
        )

        assert _multiclass_varsel_mask is multiclass_varsel_mask, (
            "the private alias must keep resolving to the public implementation"
        )

    def test_unsupported_method_raises_rather_than_returning_empty(self) -> None:
        """A method with no multi-class implementation must fail loudly.

        Silently returning nothing is the failure mode this suite guards against:
        a clean-looking empty result is indistinguishable from a real negative.
        """
        from spectral_predict.search import (
            MulticlassVarselUnsupported,
            multiclass_varsel_mask,
        )

        X, y = _graded_multiclass(n_features=40, seed=5)
        with pytest.raises(MulticlassVarselUnsupported):
            multiclass_varsel_mask(
                X, y, np.arange(X.shape[1]), "definitely_not_a_method", 10
            )


def test_importing_package_stays_headless() -> None:
    """``import spectral_predict`` must not pull in matplotlib or tkinter.

    This guarantee used to be documented in ``cli.py``'s module docstring (PR #56:
    ``--version``/``--help`` had to work on headless Linux, where importing a GUI
    module calls ``matplotlib.use('TkAgg')`` and fails). The CLI is retired, so the
    guarantee is pinned here instead of being silently dropped with the file.

    Runs in a subprocess because the pytest session has already imported both.
    """
    code = (
        "import sys; import spectral_predict; "
        "print(','.join(m for m in ('matplotlib', 'tkinter') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", (
        f"import spectral_predict leaked GUI/plotting modules: {leaked}. "
        "Keep heavy imports inside the functions that need them."
    )
