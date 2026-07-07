"""Regression: print() string literals in the fiPLS/interval varsel path must be
cp1252-encodable so they don't raise UnicodeEncodeError on Windows cp1252 stdout.

Background: U+2192 (->) arrows inside print() f-strings crashed the fiPLS/interval
variable-selection final-summary prints on cp1252 consoles once a fix made those
methods actually run on derivative configs. Note U+00B2 (superscript 2, as in R2),
U+00F8, U+00E9 etc. ARE cp1252-encodable and are intentionally left alone.
"""

from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "spectral_predict"

# Modules on the multiclass interval/fiPLS varsel path that emit stdout.
MODULES_UNDER_TEST = [SRC / "variable_selection.py"]


def _print_and_logging_string_literals(source: str):
    """Yield (lineno, string_value) for every string literal that is an argument
    to a print(...) or logging call (log/logger/logging.<level>(...))."""
    tree = ast.parse(source)

    def _is_target_call(node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            return True
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in {
                "debug",
                "info",
                "warning",
                "warn",
                "error",
                "exception",
                "critical",
                "log",
            }:
                root = func.value
                if isinstance(root, ast.Name) and root.id in {
                    "log",
                    "logger",
                    "logging",
                    "_log",
                    "_logger",
                }:
                    return True
        return False

    def _iter_str_constants(node):
        # Handles plain string constants and f-strings (JoinedStr), including nested.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    yield value.lineno, value.value

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_target_call(node):
            for arg in node.args:
                yield from _iter_str_constants(arg)


@pytest.mark.parametrize("module_path", MODULES_UNDER_TEST, ids=lambda p: p.name)
def test_print_literals_are_cp1252_encodable(module_path: Path) -> None:
    source = module_path.read_text(encoding="utf-8")
    offenders = []
    for lineno, text in _print_and_logging_string_literals(source):
        try:
            text.encode("cp1252", errors="strict")
        except UnicodeEncodeError:
            offenders.append((lineno, text))

    assert not offenders, (
        f"{module_path.name}: print/log string literal(s) not cp1252-encodable "
        f"(would raise UnicodeEncodeError on Windows cp1252 stdout): "
        + "; ".join(f"line {ln}: {t!r}" for ln, t in offenders)
    )


def _tiny_dataset(n_samples: int = 24, n_features: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    wavelengths = np.linspace(1000.0, 2000.0, n_features)
    # Simple linear target so the varsel routines have real signal to work with.
    coefs = rng.normal(size=n_features)
    y = X @ coefs + rng.normal(scale=0.01, size=n_samples)
    return X, y, wavelengths


def test_fipls_spa_selection_runtime_cp1252_stdout() -> None:
    from spectral_predict.variable_selection import fipls_spa_selection

    X, y, wavelengths = _tiny_dataset()
    sink = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with contextlib.redirect_stdout(sink):
        result = fipls_spa_selection(
            X, y, wavelengths=wavelengths,
            n_intervals=5, max_combine=2, ipls_cv_folds=3, spa_cv_folds=3,
            random_state=0,
        )
    assert result.shape[0] == X.shape[1]


def test_fipls_cars_selection_runtime_cp1252_stdout() -> None:
    from spectral_predict.variable_selection import fipls_cars_selection

    X, y, wavelengths = _tiny_dataset()
    sink = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with contextlib.redirect_stdout(sink):
        result = fipls_cars_selection(
            X, y, wavelengths=wavelengths,
            n_intervals=5, max_combine=2, ipls_cv_folds=3,
            n_iterations=10, pls_components=3, cars_cv_folds=3,
            monte_carlo_samples=20, random_state=0,
        )
    assert result.shape[0] == X.shape[1]
