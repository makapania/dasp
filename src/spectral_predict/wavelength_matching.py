"""One contract for mapping wavelength values onto the columns of a spectral axis.

Wavelengths are chosen from the chemistry, so a model must be trained and applied on
exactly the channels it names. Every site that turns a list of wavelength *values*
into column *indices* (the Tab 7 refit, ``model_io`` prediction, and the validation
rebuild of a results row) goes through :func:`match_wavelengths` or
:func:`resolve_wavelength_list`, so training and prediction cannot disagree about
which column a value means.

The contract, per requested value:

1. An exact match on the axis wins.
2. Otherwise the value must lie within ``tolerance`` of exactly one axis value.
3. No match, or more than one candidate, raises :class:`WavelengthMatchError`.
   Nothing is dropped, nothing falls back to the full spectrum, and no "first hit"
   is taken. A window wide enough to hold two channels therefore fails rather than
   guessing, which ties the usable tolerance to the axis spacing.

Order is preserved (output index ``k`` belongs to requested value ``k``), and the
axis may be ascending, descending or unsorted.

Results rows store wavelength lists as text (``all_vars``, ``top_vars``). Write them
with :func:`format_wavelength_list`, which round-trips exactly. Older rows were
written with ``%g`` (6 significant digits), so :func:`resolve_wavelength_list`
accepts those when the rounding cannot have merged two channels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_TOLERANCE",
    "WavelengthMatchError",
    "format_wavelength_list",
    "match_wavelengths",
    "resolve_wavelength_list",
]

#: Absolute tolerance (axis units) for values that should already be exact axis
#: values but may carry float noise from a file header. It is model_io's historical
#: prediction tolerance; it is deliberately NOT 0.5, which spans neighbouring
#: channels on any grid finer than 0.5 units.
DEFAULT_TOLERANCE = 0.01

# Relative slack for float noise when a list was written round-trip-exactly.
_FLOAT_SLACK = 1e-12


class WavelengthMatchError(ValueError):
    """Requested wavelengths could not be mapped one-to-one onto the axis.

    Attributes:
        missing: Requested values with no axis value inside the tolerance.
        ambiguous: Requested values with more than one candidate column.
    """

    def __init__(
        self,
        message: str,
        *,
        missing: Iterable[float] = (),
        ambiguous: Iterable[float] = (),
    ) -> None:
        super().__init__(message)
        self.missing = [float(v) for v in missing]
        self.ambiguous = [float(v) for v in ambiguous]


def _as_axis(axis: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        ax = np.asarray(axis, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise WavelengthMatchError(f"wavelength axis is not numeric: {exc}") from exc
    if ax.size == 0:
        raise WavelengthMatchError("wavelength axis is empty")
    if not np.all(np.isfinite(ax)):
        raise WavelengthMatchError("wavelength axis contains NaN or infinite values")
    return ax


def _preview(values: list[float], limit: int = 5) -> str:
    shown = ", ".join(repr(v) for v in values[:limit])
    return shown + (f", ... ({len(values)} in total)" if len(values) > limit else "")


def _match(req: np.ndarray, ax: np.ndarray, tol: np.ndarray, *, prefer_exact: bool) -> np.ndarray:
    """Vectorised core: ``tol`` is a per-request absolute half-window."""
    order = np.argsort(ax, kind="stable")
    sorted_ax = ax[order]

    exact_lo = np.searchsorted(sorted_ax, req, side="left")
    exact_hi = np.searchsorted(sorted_ax, req, side="right")
    n_exact = exact_hi - exact_lo

    win_lo = np.searchsorted(sorted_ax, req - tol, side="left")
    win_hi = np.searchsorted(sorted_ax, req + tol, side="right")
    n_window = win_hi - win_lo

    result = np.full(req.shape, -1, dtype=np.intp)
    missing: list[float] = []
    ambiguous: list[float] = []
    for k, value in enumerate(req):
        if n_exact[k] > 1:
            ambiguous.append(float(value))  # duplicated axis value
        elif prefer_exact and n_exact[k] == 1:
            result[k] = order[exact_lo[k]]
        elif n_window[k] == 1:
            result[k] = order[win_lo[k]]
        elif n_window[k] == 0:
            missing.append(float(value))
        else:
            ambiguous.append(float(value))

    if missing or ambiguous:
        parts = []
        if missing:
            parts.append(f"{len(missing)} not on the axis ({_preview(missing)})")
        if ambiguous:
            parts.append(
                f"{len(ambiguous)} match more than one axis column ({_preview(ambiguous)})"
            )
        raise WavelengthMatchError(
            f"Could not map {len(missing) + len(ambiguous)} of {req.size} wavelengths "
            f"to the spectral axis: " + "; ".join(parts) + ".",
            missing=missing,
            ambiguous=ambiguous,
        )

    # Two different requested values must never collapse onto one column.
    seen: dict[int, float] = {}
    for value, col in zip(req.tolist(), result.tolist()):
        other = seen.setdefault(col, value)
        if other != value:
            raise WavelengthMatchError(
                f"Wavelengths {other!r} and {value!r} both map to axis column "
                f"{col} ({float(ax[col])!r}); the list does not describe distinct channels.",
                ambiguous=[other, value],
            )
    return result


def match_wavelengths(
    requested: Iterable[float],
    axis: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Map wavelength values to column indices of ``axis``, in the requested order.

    Args:
        requested: Wavelength values, in feature order.
        axis: The spectral axis (column wavelengths), any order.
        tolerance: Absolute half-window, in axis units, for a value that is not an
            exact axis value. ``0`` demands exact matches.

    Returns:
        Integer array, ``result[k]`` is the column of ``requested[k]``.

    Raises:
        WavelengthMatchError: A value has no column within ``tolerance``, has more
            than one, or two values land on the same column.
    """
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite value >= 0, got {tolerance!r}")
    try:
        req = np.asarray(list(requested), dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise WavelengthMatchError(f"requested wavelengths are not numeric: {exc}") from exc
    if not np.all(np.isfinite(req)):
        raise WavelengthMatchError("requested wavelengths contain NaN or infinite values")
    ax = _as_axis(axis)
    if req.size == 0:
        return np.zeros(0, dtype=np.intp)
    tol = np.maximum(float(tolerance), np.abs(req) * _FLOAT_SLACK)
    return _match(req, ax, tol, prefer_exact=True)


def format_wavelength_list(wavelengths: Iterable[float]) -> str:
    """Serialise wavelengths as comma-separated text that parses back exactly.

    Uses Python's shortest round-trip float repr (``1500.0``, ``7407.407407407408``),
    never ``%g``, whose 6 significant digits merge or shift channels.
    """
    return ",".join(repr(float(w)) for w in wavelengths)


def _could_be_g_format(token: str, value: float) -> bool:
    """True if ``%g`` formatting of ``value`` reproduces ``token`` exactly."""
    return f"{value:g}" == token


def _g_half_unit(values: np.ndarray) -> np.ndarray:
    """Largest distance between a true value and its 6-significant-digit ``%g`` text."""
    mag = np.abs(values)
    safe = np.where(mag > 0, mag, 1.0)
    exponent = np.floor(np.log10(safe))
    half_unit = 0.5 * np.power(10.0, exponent - 5)
    # Float slack so a value exactly on the rounding boundary is still inside.
    return np.where(mag > 0, half_unit * (1 + 1e-9), 0.0)


def resolve_wavelength_list(text: str, axis: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a stored wavelength list (``all_vars`` / ``top_vars`` text) to axis columns.

    A list written by :func:`format_wavelength_list` is matched exactly. A list
    written by the old ``%g`` writer (every token is what ``%g`` would print) is
    matched within each value's 6-significant-digit rounding window, and is accepted
    only when exactly one axis column falls in that window. When two columns do,
    the old text cannot say which channel was meant, and this raises.

    Args:
        text: Comma-separated wavelength values.
        axis: The spectral axis the values refer to.

    Returns:
        Integer column indices in the stored order.

    Raises:
        WavelengthMatchError: Empty or non-numeric text, or any value that is
            missing or ambiguous on ``axis``. A partial match is a failure.
    """
    if not isinstance(text, str):
        raise WavelengthMatchError(f"wavelength list must be text, got {type(text).__name__}")
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        raise WavelengthMatchError("wavelength list is empty")
    try:
        values = np.array([float(t) for t in tokens], dtype=float)
    except ValueError as exc:
        raise WavelengthMatchError(f"wavelength list is not numeric: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise WavelengthMatchError("wavelength list contains NaN or infinite values")
    ax = _as_axis(axis)

    legacy_g = all(_could_be_g_format(t, v) for t, v in zip(tokens, values.tolist()))
    if not legacy_g:
        tol = np.abs(values) * _FLOAT_SLACK
        return _match(values, ax, tol, prefer_exact=True)
    # Every token looks like %g output. An exact hit is NOT trusted on its own here:
    # under %g, 12345.7 may stand for 12345.67, so any second column inside the
    # rounding window makes the value ambiguous.
    return _match(values, ax, _g_half_unit(values), prefer_exact=False)
