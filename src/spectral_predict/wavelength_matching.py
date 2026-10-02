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
with :func:`format_wavelength_list`: every token round-trips exactly and is spelled
so that ``%g`` could never have produced it (its mantissa always ends in a ``0``
after the decimal point, e.g. ``1500.0``, ``10000.10``, ``1.0e-05``). Older rows were
written with ``%g`` (6 significant digits); :func:`resolve_wavelength_list` maps such
a token to the single axis column whose own ``%g`` text is that token, and raises
when two columns print the same.
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


def _as_values(requested: Iterable[float]) -> np.ndarray:
    try:
        req = np.asarray(list(requested), dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise WavelengthMatchError(f"requested wavelengths are not numeric: {exc}") from exc
    if not np.all(np.isfinite(req)):
        raise WavelengthMatchError("requested wavelengths contain NaN or infinite values")
    return req


def _preview(values: list[float], limit: int = 5) -> str:
    shown = ", ".join(repr(v) for v in values[:limit])
    return shown + (f", ... ({len(values)} in total)" if len(values) > limit else "")


def _g_text(value: float) -> str:
    return f"{value:g}"


def _is_g_shaped(value: float) -> bool:
    """True if ``value`` is exactly what parsing its own ``%g`` text gives back."""
    return float(_g_text(value)) == value


def _match(
    req: np.ndarray,
    ax: np.ndarray,
    tol: np.ndarray,
    g_tokens: list[str | None] | None = None,
) -> np.ndarray:
    """Core matcher.

    For value ``k``: when ``g_tokens[k]`` is a string, the value is legacy ``%g``
    text and its candidates are the axis columns whose own ``%g`` text equals it.
    Otherwise an exact hit wins, else the single axis value within ``tol[k]``.
    """
    order = np.argsort(ax, kind="stable")
    sorted_ax = ax[order]

    exact_lo = np.searchsorted(sorted_ax, req, side="left")
    exact_hi = np.searchsorted(sorted_ax, req, side="right")
    n_exact = exact_hi - exact_lo

    win_lo = np.searchsorted(sorted_ax, req - tol, side="left")
    win_hi = np.searchsorted(sorted_ax, req + tol, side="right")
    n_window = win_hi - win_lo

    g_index: dict[str, list[int]] = {}
    if g_tokens is not None and any(t is not None for t in g_tokens):
        for col, value in enumerate(ax.tolist()):
            g_index.setdefault(_g_text(value), []).append(col)

    result = np.full(req.shape, -1, dtype=np.intp)
    missing: list[float] = []
    ambiguous: list[float] = []
    for k, value in enumerate(req):
        token = g_tokens[k] if g_tokens is not None else None
        if token is not None:
            candidates = g_index.get(token, [])
            if len(candidates) == 1:
                result[k] = candidates[0]
            elif candidates:
                ambiguous.append(float(value))
            else:
                missing.append(float(value))
        elif n_exact[k] > 1:
            ambiguous.append(float(value))  # duplicated axis value
        elif n_exact[k] == 1:
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
    legacy_g: bool = False,
) -> np.ndarray:
    """Map wavelength values to column indices of ``axis``, in the requested order.

    Args:
        requested: Wavelength values, in feature order.
        axis: The spectral axis (column wavelengths), any order.
        tolerance: Absolute half-window, in axis units, for a value that is not an
            exact axis value. ``0`` demands exact matches.
        legacy_g: The values may have been parsed from 6-significant-digit ``%g``
            text (wavelength metadata saved before the 2026-10 fix). A value equal
            to its own ``%g`` text is then matched to the single axis column whose
            ``%g`` text is the same; any other value uses the normal rule.

    Returns:
        Integer array, ``result[k]`` is the column of ``requested[k]``.

    Raises:
        WavelengthMatchError: A value has no column within ``tolerance``, has more
            than one, or two values land on the same column.
    """
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite value >= 0, got {tolerance!r}")
    req = _as_values(requested)
    ax = _as_axis(axis)
    if req.size == 0:
        return np.zeros(0, dtype=np.intp)
    tol = np.maximum(float(tolerance), np.abs(req) * _FLOAT_SLACK)
    g_tokens = None
    if legacy_g:
        g_tokens = [_g_text(v) if _is_g_shaped(v) else None for v in req.tolist()]
    return _match(req, ax, tol, g_tokens)


def _exact_token(value: float) -> str:
    """Round-trip-exact text whose mantissa ends in a 0 after the decimal point.

    ``%g`` strips trailing zeros and a bare decimal point, so it never writes such a
    token. That is how :func:`resolve_wavelength_list` tells new rows from old ones.
    """
    text = repr(float(value))
    if not math.isfinite(float(value)):
        return text
    mantissa, sep, exponent = text.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    elif not mantissa.endswith("0"):
        mantissa += "0"
    return mantissa + sep + exponent


def _is_exact_token(token: str) -> bool:
    mantissa = token.lower().partition("e")[0]
    return "." in mantissa and mantissa.endswith("0")


def format_wavelength_list(wavelengths: Iterable[float]) -> str:
    """Serialise wavelengths as comma-separated text that parses back exactly.

    Each token is Python's shortest round-trip float repr, with a trailing ``0``
    added after the decimal point when the repr lacks one (``1500.0``, ``10000.10``,
    ``7407.4074074074080``, ``1.0e-05``). ``float()`` reads every token back to the
    same value, and no token can be mistaken for the old ``%g`` output.
    """
    return ",".join(_exact_token(w) for w in wavelengths)


def resolve_wavelength_list(text: str, axis: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a stored wavelength list (``all_vars`` / ``top_vars`` text) to axis columns.

    Each token is classified on its own:

    - written by :func:`format_wavelength_list` (mantissa ends in ``0`` after a
      decimal point): matched exactly;
    - exactly what ``%g`` prints for its value (the pre-2026-10 writer): matched to
      the single axis column whose own ``%g`` text equals the token; two such
      columns make the token ambiguous, because the old text cannot say which
      channel was meant;
    - anything else (full-precision text from other writers): matched exactly.

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

    g_tokens: list[str | None] = [
        None if _is_exact_token(t) or _g_text(v) != t else t
        for t, v in zip(tokens, values.tolist())
    ]
    return _match(values, ax, np.abs(values) * _FLOAT_SLACK, g_tokens)
