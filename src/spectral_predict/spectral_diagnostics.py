"""Noise maps and detector joins measured from a set of spectra.

Everything here is computed from the spectra handed in: noise on one instrument changes
between sessions (on SPYDER Bone's LabSpec data, one unit was ~4x quieter on another day),
so a noise map belongs to the scans in hand, not to an instrument.

- ``estimate_noise``: per-wavelength noise of the spectra and of their Savitzky-Golay
  second derivative. Noise on a grid interpolated from a coarser native spacing (ASD
  spectra are delivered at 1 nm) is correlated between neighbouring channels; the
  estimator infers that correlation from the high-pass residual and propagates it, so
  the derivative noise is not underestimated ~2x as a white-noise formula would.
- ``noisy_regions``: where noise is far above the quiet part of the same detector block.
- ``noise_comparison``: where one instrument is much noisier than another.
- ``detect_detector_joins`` / ``resolve_joins``: where one detector hands over to the next
  (a level step at one channel boundary), from a file header, the spectra, or the user.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_coeffs

logger = logging.getLogger(__name__)

Space = Literal["reflectance", "absorbance"]
SPACES: tuple[str, ...] = ("reflectance", "absorbance")

REFLECTANCE_FLOOR = 1e-6
"""Reflectance is clipped here before log10(1/R)."""

HIGHPASS_WINDOW_NM = 11.0
D2_WINDOW_NM = 31.0
PROFILE_SMOOTH_NM = 21.0
NOISE_FLOOR = 1e-6
"""Absorbance noise below this is numerical, not instrument noise (ASD spectra sit at
1e-5 and above). Used as the floor of the quiet level that noisy regions are judged by."""


# ----------------------------------------------------------------------------- basics


def check_wavelengths(wavelengths: Sequence[float], p: int | None = None) -> np.ndarray:
    """Return wavelengths as a strictly increasing float array.

    Raises:
        ValueError: Not 1-D, not strictly increasing, or not ``p`` long.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wavelengths must be a 1-D array of at least 2 values")
    if not np.all(np.diff(wl) > 0):
        raise ValueError("wavelengths must be strictly increasing")
    if p is not None and wl.size != p:
        raise ValueError(f"wavelengths has {wl.size} values for spectra with {p} columns")
    return wl


def grid_step(wavelengths: np.ndarray) -> float:
    """Median wavelength spacing."""
    return float(np.median(np.diff(wavelengths)))


def check_space(space: str) -> str:
    """Validate a spectral space name."""
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")
    return space


def to_absorbance(X: np.ndarray, space: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert spectra to absorbance log10(1/R).

    Args:
        X: (m, p) spectra.
        space: ``'reflectance'`` (0-1 scale) or ``'absorbance'``.

    Returns:
        ``(A, valid)``: absorbance and a boolean mask of channels that were usable
        (reflectance > 0 and finite).

    Raises:
        ValueError: Reflectance that looks like percent (values > 1.5 in more than half
            the channels).
    """
    X = np.asarray(X, dtype=np.float64)
    check_space(space)
    if space == "absorbance":
        return X.copy(), np.isfinite(X)
    if np.mean(X > 1.5) > 0.5:
        raise ValueError(
            "Reflectance values are mostly above 1.5: is this percent reflectance? "
            "Divide by 100 first."
        )
    valid = np.isfinite(X) & (X > 0)
    A = np.log10(1.0 / np.clip(np.where(np.isfinite(X), X, 1.0), REFLECTANCE_FLOOR, None))
    return A, valid


def from_absorbance(A: np.ndarray, space: str) -> np.ndarray:
    """Inverse of ``to_absorbance``."""
    return 10.0 ** (-A) if space == "reflectance" else A


def block_slices(wavelengths: np.ndarray, joins: Sequence[float] | None) -> list[slice]:
    """Split the grid into detector blocks.

    A join is the last wavelength of the block to its left (ASD headers store 1000 and
    1800: VNIR ends at 1000 nm, SWIR1 at 1800 nm). Joins outside the grid are ignored.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    js = () if joins is None else [float(j) for j in np.atleast_1d(joins)]
    cuts = sorted({int(np.searchsorted(wl, j, side="right")) for j in js})
    cuts = [c for c in cuts if 0 < c < wl.size]
    edges = [0, *cuts, wl.size]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def block_step(wavelengths: np.ndarray, sl: slice) -> float:
    """Median spacing inside one block (the grid's spacing when the block has one channel)."""
    wl = np.asarray(wavelengths, dtype=np.float64)[sl]
    return float(np.median(np.diff(wl))) if wl.size > 1 else grid_step(np.asarray(wavelengths))


def check_uniform_blocks(
    wavelengths: np.ndarray, joins: Sequence[float] | None, tol: float = 0.02
) -> None:
    """Refuse grids whose spacing varies inside a detector block.

    Savitzky-Golay filters and the nm-wide blur assume even spacing within a block.
    Spacing may change at a join (e.g. 1.4 nm VNIR, 2 nm SWIR before ASD's resampling).

    Raises:
        ValueError: Spacing inside a block varies by more than ``tol`` (relative).
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    for sl in block_slices(wl, joins):
        d = np.diff(wl[sl])
        if d.size and (d.max() - d.min()) > tol * float(np.median(d)):
            raise ValueError(
                f"Wavelength spacing varies inside {wl[sl][0]:g}-{wl[sl][-1]:g} nm "
                f"({d.min():g}-{d.max():g} nm); resample to an even grid (or give the "
                "detector joins where the spacing changes)"
            )


def odd_channels(nm: float, step: float, minimum: int) -> int:
    """Odd number of channels spanning about ``nm`` (at least ``minimum``)."""
    k = int(round(nm / step))
    k = max(k, minimum)
    return k if k % 2 == 1 else k + 1


def _apply_fir(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Centred FIR filter along axis 1, edge-replicate padding (one block)."""
    h = coeffs.size // 2
    Xp = np.pad(X, ((0, 0), (h, h)), mode="edge")
    return sliding_window_view(Xp, coeffs.size, axis=1) @ coeffs[::-1]


def savgol_blocks(
    X: np.ndarray,
    wavelengths: np.ndarray,
    joins: Sequence[float] | None,
    *,
    window_nm: float,
    polyorder: int,
    deriv: int = 0,
) -> np.ndarray:
    """Savitzky-Golay filter applied within each detector block (never across a join).

    The derivative is per nm. A block too short for the window gets a shorter window;
    one shorter than ``polyorder + 2`` channels is returned unchanged (deriv 0) or zero.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    out = np.zeros_like(X) if deriv else X.copy()
    for sl in block_slices(wavelengths, joins):
        nb = sl.stop - sl.start
        step = block_step(wavelengths, sl)
        w = odd_channels(window_nm, step, polyorder + 2)
        if w > nb:
            w = nb if nb % 2 == 1 else nb - 1
        if w < polyorder + 2:
            continue
        c = savgol_coeffs(w, polyorder, deriv=deriv, delta=step, use="dot")
        out[:, sl] = _apply_fir(X[:, sl], c[::-1])
    return out


def second_derivative(
    A: np.ndarray, wavelengths: np.ndarray, joins: Sequence[float] | None = None
) -> np.ndarray:
    """Cubic Savitzky-Golay second derivative over ~31 nm, within detector blocks."""
    return savgol_blocks(A, wavelengths, joins, window_nm=D2_WINDOW_NM, polyorder=3, deriv=2)


def _smooth_profile(v: np.ndarray, wavelengths: np.ndarray, joins, nm: float) -> np.ndarray:
    out = v.copy()
    for sl in block_slices(wavelengths, joins):
        k = min(odd_channels(nm, block_step(wavelengths, sl), 1), (sl.stop - sl.start) // 2 * 2 + 1)
        if k > 1:
            out[sl] = _apply_fir(v[None, sl], np.ones(k) / k)[0]
    return out


# ----------------------------------------------------------------------------- noise


def _noise_kernels() -> dict[str, np.ndarray]:
    """Candidate shapes of channel-to-channel noise correlation."""
    ks: dict[str, np.ndarray] = {"white": np.array([1.0])}
    for k in (2, 3, 4, 5):
        ks[f"box{k}"] = np.ones(k) / k
    for s in (1.5, 2.0, 3.0, 4.0):  # linear interpolation from a native spacing of s channels
        t = np.arange(-int(np.ceil(s)) + 1, int(np.ceil(s)))
        h = np.clip(1.0 - np.abs(t) / s, 0.0, None)
        ks[f"interp{s:g}"] = h / h.sum()
    return ks


def noise_kernel(name: str) -> np.ndarray:
    """The noise-correlation kernel named in ``NoiseEstimate.kernels`` (white if unknown)."""
    return _noise_kernels().get(name, np.array([1.0]))


def _acf(f: np.ndarray, lags: int) -> np.ndarray:
    a = np.correlate(f, f, "full")
    mid = f.size - 1
    return np.array([a[mid + lag] if mid + lag < a.size else 0.0 for lag in range(lags + 1)])


@dataclass
class NoiseEstimate:
    """Per-wavelength noise of a set of spectra.

    Attributes:
        channel_sd: Noise standard deviation per channel, in the units of the spectra
            passed in (absorbance when called through ``noise_profile``).
        d2_sd: Noise standard deviation of one spectrum's second derivative
            (``second_derivative``) per channel. Divide by sqrt(n) for a mean of n.
        kernels: Detected noise-correlation shape per block (``'white'``,
            ``'interp2'`` ...).
    """

    channel_sd: np.ndarray
    d2_sd: np.ndarray
    kernels: list[str] = field(default_factory=list)


def estimate_noise(
    A: np.ndarray, wavelengths: Sequence[float], joins: Sequence[float] | None = None
) -> NoiseEstimate:
    """Estimate channel and second-derivative noise from the spectra themselves.

    The high-pass residual from an ~11 nm quadratic Savitzky-Golay smooth is mostly
    noise. Its autocorrelation at lags 1-4 picks the noise-correlation shape, which
    turns the residual's spread into channel noise and propagates it through the
    second-derivative filter. Robust (MAD) over spectra, smoothed over ~21 nm, per
    detector block. On simulated data this recovers derivative noise within ~10% for
    white noise and for 1.5-2 channel interpolation (a white-noise formula reads ~0.45x).

    Args:
        A: (m, p) spectra (absorbance is what the transfer code passes).
        wavelengths: (p,) grid.
        joins: Detector joins; blocks are treated separately.

    Returns:
        A ``NoiseEstimate``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    wl = check_wavelengths(wavelengths, A.shape[1])
    channel_sd = np.zeros(wl.size)
    d2_sd = np.zeros(wl.size)
    kernels: list[str] = []
    kern = _noise_kernels()
    for sl in block_slices(wl, joins):
        nb = sl.stop - sl.start
        step = block_step(wl, sl)
        w_hp = min(odd_channels(HIGHPASS_WINDOW_NM, step, 5), nb if nb % 2 else nb - 1)
        w_d2 = min(odd_channels(D2_WINDOW_NM, step, 5), nb if nb % 2 else nb - 1)
        if w_hp < 5 or w_d2 < 5:
            kernels.append("too_short")
            continue
        smooth = savgol_coeffs(w_hp, 2, use="dot")
        hp = -smooth
        hp[w_hp // 2] += 1.0
        d2 = savgol_coeffs(w_d2, 3, deriv=2, delta=step, use="dot")
        Ab = A[:, sl]
        r = Ab - _apply_fir(Ab, smooth[::-1])
        core = r[:, w_hp : max(w_hp + 1, nb - w_hp)]
        lags = min(4, max(1, core.shape[1] - 1))
        emp = np.array(
            [np.mean(core[:, lag:] * core[:, : core.shape[1] - lag]) for lag in range(lags + 1)]
        )
        best_name, best_h, best_err = "white", kern["white"], np.inf
        if emp[0] > 0:
            emp = emp / emp[0]
            for name, h in kern.items():
                a = _acf(np.convolve(h, hp), lags)
                err = float(((a[1:] / a[0] - emp[1:]) ** 2).sum())
                if err < best_err:
                    best_name, best_h, best_err = name, h, err
        kernels.append(best_name)
        hp_gain = float(np.sqrt((np.convolve(best_h, hp) ** 2).sum()))
        sd_r = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
        if A.shape[0] < 3:  # MAD over one or two spectra is unstable; use the spread itself
            sd_r = np.sqrt(np.mean(r**2, axis=0))
        sd_white = sd_r / hp_gain
        channel_sd[sl] = sd_white * float(np.sqrt((best_h**2).sum()))
        d2_sd[sl] = sd_white * float(np.sqrt((np.convolve(best_h, d2) ** 2).sum()))
    channel_sd = _smooth_profile(channel_sd, wl, joins, PROFILE_SMOOTH_NM)
    d2_sd = _smooth_profile(d2_sd, wl, joins, PROFILE_SMOOTH_NM)
    return NoiseEstimate(channel_sd=channel_sd, d2_sd=d2_sd, kernels=kernels)


def noise_profile(
    X: np.ndarray,
    wavelengths: Sequence[float],
    *,
    space: str,
    joins: Sequence[float] | None = None,
) -> np.ndarray:
    """Per-wavelength noise in absorbance units, estimated from the spectra.

    Args:
        X: (m, p) spectra.
        wavelengths: (p,) grid.
        space: ``'reflectance'`` or ``'absorbance'``.
        joins: Detector joins.

    Returns:
        (p,) noise standard deviation.
    """
    A, _ = to_absorbance(X, space)
    return estimate_noise(A, wavelengths, joins).channel_sd


def _runs(mask: np.ndarray, wl: np.ndarray, min_width_nm: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    start = None
    for i, b in enumerate(np.append(mask, False)):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if wl[i - 1] - wl[start] >= min_width_nm - 1e-9:
                out.append((float(wl[start]), float(wl[i - 1])))
            start = None
    return out


def block_quiet_level(
    profile: np.ndarray, wavelengths: np.ndarray, joins: Sequence[float] | None
) -> np.ndarray:
    """Median of the quietest half of each detector block, broadcast per channel."""
    out = np.empty_like(profile)
    for sl in block_slices(wavelengths, joins):
        v = np.sort(profile[sl])
        out[sl] = np.median(v[: max(1, v.size // 2)])
    return out


def noisy_regions(
    profile: np.ndarray,
    wavelengths: Sequence[float],
    joins: Sequence[float] | None = None,
    *,
    factor: float = 10.0,
    min_width_nm: float = 10.0,
) -> list[tuple[float, float]]:
    """Regions where noise exceeds ``factor`` x the quiet level of the same block.

    The reference is per detector block because detectors differ in noise: on clean
    LabSpec units the SWIR2 detector is several times noisier than SWIR1 everywhere,
    and an instrument-wide reference flags all of it.

    Args:
        profile: (p,) noise, e.g. from ``noise_profile``.
        wavelengths: (p,) grid.
        joins: Detector joins.
        factor: Multiple of the block's quiet-half median noise.
        min_width_nm: Shortest region reported.

    Returns:
        ``[(lo_nm, hi_nm), ...]``.
    """
    wl = check_wavelengths(wavelengths, np.asarray(profile).size)
    prof = np.asarray(profile, dtype=np.float64)
    quiet = block_quiet_level(prof, wl, joins)
    return _runs(prof > factor * np.maximum(quiet, NOISE_FLOOR), wl, min_width_nm)


def regions_mask(regions: Sequence[tuple[float, float]], wavelengths: np.ndarray) -> np.ndarray:
    """Boolean mask of channels inside any ``(lo, hi)`` region."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    m = np.zeros(wl.size, dtype=bool)
    for lo, hi in regions:
        m |= (wl >= lo) & (wl <= hi)
    return m


@dataclass
class NoiseComparison:
    """Satellite noise relative to primary noise.

    Attributes:
        ratio: (p,) satellite / primary noise; the primary is floored at its block's
            quiet-half median so a near-zero primary cannot blow the ratio up.
        regions: Regions >= ``min_width_nm`` where the ratio exceeds the threshold.
        primary_noisy: ``noisy_regions`` of the primary.
        satellite_noisy: ``noisy_regions`` of the satellite.
    """

    ratio: np.ndarray
    regions: list[tuple[float, float]]
    primary_noisy: list[tuple[float, float]]
    satellite_noisy: list[tuple[float, float]]


def noise_comparison(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    wavelengths: Sequence[float],
    *,
    space: str,
    joins: Sequence[float] | None = None,
    factor: float = 5.0,
    noisy_factor: float = 10.0,
    min_width_nm: float = 10.0,
) -> NoiseComparison:
    """Compare noise of two instruments measured on the same grid."""
    wl = check_wavelengths(wavelengths, np.asarray(X_primary).shape[1])
    n_p = noise_profile(X_primary, wl, space=space, joins=joins)
    n_s = noise_profile(X_satellite, wl, space=space, joins=joins)
    floor = block_quiet_level(n_p, wl, joins)
    ratio = n_s / np.maximum(n_p, np.maximum(floor, NOISE_FLOOR))
    return NoiseComparison(
        ratio=ratio,
        regions=_runs(ratio > factor, wl, min_width_nm),
        primary_noisy=noisy_regions(n_p, wl, joins, factor=noisy_factor, min_width_nm=min_width_nm),
        satellite_noisy=noisy_regions(
            n_s, wl, joins, factor=noisy_factor, min_width_nm=min_width_nm
        ),
    )


# ----------------------------------------------------------------------------- joins


@dataclass
class JoinDetection:
    """Detector joins and where they came from.

    Attributes:
        joins: Join wavelengths (last wavelength of the left block), ascending.
        source: ``'header'``, ``'detected'``, ``'user'`` or ``'none'``.
        confident: For detection: at least one join cleared every test.
        scores: Candidate wavelength -> step score (detection only).
        warnings: Plain-language notes (e.g. header and data disagree).
    """

    joins: tuple[float, ...] = ()
    source: str = "none"
    confident: bool = False
    scores: dict[float, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _line_extrapolators(w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights extrapolating a least-squares line on w points to the boundary, and the
    residual (hat) operator.

    Left window x = -w+1..0 extrapolated to +0.5; right window x = 1..w to +0.5.
    """
    xl = np.arange(-w + 1, 1, dtype=float)
    xr = np.arange(1, w + 1, dtype=float)
    Ml = np.column_stack([np.ones(w), xl])
    Mr = np.column_stack([np.ones(w), xr])
    el = np.array([1.0, 0.5]) @ np.linalg.pinv(Ml)
    er = np.array([1.0, 0.5]) @ np.linalg.pinv(Mr)
    H = np.eye(w) - Ml @ np.linalg.pinv(Ml)  # same residual operator for both (shift-invariant)
    return el, er, H


def _boundary_steps(R: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Per spectrum and boundary i|i+1: step (right minus left extrapolation) and the
    combined residual scale of the two line fits. Boundaries without full windows = nan."""
    m, p = R.shape
    el, er, H = _line_extrapolators(w)
    V = sliding_window_view(R, w, axis=1)  # (m, p-w+1, w); window j covers j..j+w-1
    ext_r = V @ er  # right window starting at j
    ext_l = V @ el  # left window ending at j+w-1
    res = np.sqrt(np.mean((V @ H.T) ** 2, axis=2))
    steps = np.full((m, p - 1), np.nan)
    scale = np.full((m, p - 1), np.nan)
    # boundary i: left window ends at i (start i-w+1), right window starts at i+1
    i = np.arange(w - 1, p - w)
    steps[:, i] = ext_r[:, i + 1] - ext_l[:, i - w + 1]
    scale[:, i] = np.sqrt(res[:, i - w + 1] ** 2 + res[:, i + 1] ** 2)
    return steps, scale


def detect_detector_joins(
    X: np.ndarray,
    wavelengths: Sequence[float],
    *,
    space: str,
    max_joins: int = 3,
    window_nm: float = 8.0,
    min_score: float = 6.0,
    min_separation_nm: float = 60.0,
) -> JoinDetection:
    """Find detector joins from the spectra alone.

    A join shows as a level step between two adjacent channels, present in most spectra
    of a set. For each channel boundary, lines fitted to ~8 nm on each side are
    extrapolated to the boundary; the step is divided by the fits' residual noise and
    the median over spectra is the score. A candidate must also:

    - stand out: score >= max(``min_score``, 3x the 95th percentile of all boundaries);
    - be sharp: at least 3x the scores 2-4 channels away (a steep but smooth absorption
      edge or a narrow band scores over a plateau of boundaries, a splice at one; real
      ASD joins score 10-35x their neighbours);
    - be isolated: both adjacent boundaries below half its score (one bad channel present
      in every spectrum steps at the two boundaries around it).

    A join whose step is small next to the noise (e.g. a well-matched unit) can go
    undetected; that is harmless for the step itself, and a header or the user can
    still supply it.

    Detection works in reflectance (ASD splices are offsets in reflectance). Prototype
    scores on LabSpec 4, FieldSpec 3 and dasp's example ASD files were 4-180 at
    1000/1800 nm against a background of ~3; a Trek-type handheld showed no clear join.

    Args:
        X: (m, p) spectra.
        wavelengths: (p,) grid.
        space: ``'reflectance'`` or ``'absorbance'``.
        max_joins: Most joins returned.
        window_nm: Line-fit window on each side of a boundary.
        min_score: Absolute score floor.
        min_separation_nm: Joins closer than this are not both kept.

    Returns:
        A ``JoinDetection`` with ``source='detected'`` when confident, else ``'none'``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    wl = check_wavelengths(wavelengths, X.shape[1])
    check_space(space)
    R = X if space == "reflectance" else 10.0 ** (-X)
    step = grid_step(wl)
    w = max(3, int(round(window_nm / step)))
    if wl.size < 4 * w + 10:
        return JoinDetection(warnings=["Too few wavelengths to look for detector joins."])
    steps, scale = _boundary_steps(R, w)
    floor = 1e-9 + 1e-6 * float(np.nanmedian(np.abs(R)))
    z = np.abs(steps) / np.maximum(scale, floor)
    score = np.zeros(z.shape[1])
    have = np.isfinite(z).any(axis=0)  # edge boundaries lack a full window on one side
    score[have] = np.nanmedian(z[:, have], axis=0)
    background = float(np.percentile(score[score > 0], 95)) if np.any(score > 0) else 0.0
    threshold = max(min_score, 3.0 * background)

    s = score.copy()
    joins: list[float] = []
    scores: dict[float, float] = {}
    for _ in range(4 * max_joins):
        if len(joins) >= max_joins:
            break
        i = int(np.argmax(s))
        if s[i] < threshold:
            break
        near = np.r_[score[max(0, i - 4) : max(0, i - 1)], score[i + 2 : i + 5]]
        sharp = near.size == 0 or score[i] >= 3.0 * float(np.max(near))
        adjacent = np.r_[score[max(0, i - 1) : i], score[i + 1 : i + 2]]
        isolated = adjacent.size == 0 or float(np.max(adjacent)) < 0.5 * score[i]
        if sharp and isolated:
            joins.append(float(wl[i]))
            scores[float(wl[i])] = float(score[i])
        sep = np.abs(wl[:-1] - wl[i]) < min_separation_nm
        s[sep] = 0.0
    joins.sort()
    if joins:
        return JoinDetection(tuple(joins), "detected", True, scores)
    return JoinDetection((), "none", False, scores)


def _plausible_joins(
    joins: Sequence[float] | None, wl: np.ndarray, margin_nm: float = 100.0
) -> tuple[float, ...]:
    out = []
    for j in () if joins is None else np.atleast_1d(joins):
        j = float(j)
        if np.isfinite(j) and wl[0] + margin_nm <= j <= wl[-1] - margin_nm:
            out.append(j)
    return tuple(sorted(set(out)))


def resolve_joins(
    wavelengths: Sequence[float],
    *,
    user_joins: Sequence[float] | None = None,
    header_joins: Sequence[float] | None = None,
    X: np.ndarray | None = None,
    space: str | None = None,
    agree_nm: float = 2.0,
) -> JoinDetection:
    """Decide the detector joins: user, then a plausible header, then detection, else none.

    Args:
        wavelengths: Grid.
        user_joins: Joins the user gave (taken as they are; ``()`` means "no joins").
        header_joins: Joins from the file header (e.g. ``read_asd_header_joins``).
        X: Spectra to detect joins from (with ``space``).
        space: Space of ``X``.
        agree_nm: Header and detected joins this close are treated as the same join.

    Returns:
        A ``JoinDetection``; ``warnings`` notes a header the data contradict.
    """
    wl = check_wavelengths(wavelengths)
    if user_joins is not None:
        return JoinDetection(tuple(sorted(float(j) for j in user_joins)), "user", True)
    detected = None
    if X is not None and space is not None:
        detected = detect_detector_joins(X, wl, space=space)
    header = _plausible_joins(header_joins, wl)
    if header:
        res = JoinDetection(header, "header", True)
        if detected is not None and detected.confident:
            extra = [d for d in detected.joins if all(abs(d - h) > agree_nm for h in header)]
            if extra:
                res.warnings.append(
                    f"The spectra show detector joins at {', '.join(f'{d:g}' for d in extra)} nm "
                    f"that the file header ({', '.join(f'{h:g}' for h in header)} nm) does not "
                    "list; using the header."
                )
        return res
    if detected is not None and detected.confident:
        return detected
    note = []
    if header_joins is not None and any(float(j) != 0.0 for j in header_joins):
        note.append("The header's detector joins are outside the wavelength range; ignored.")
    return JoinDetection((), "none", False, detected.scores if detected else {}, note)


# ----------------------------------------------------------------------------- ASD headers


_ASD_MAGIC = (b"ASD", b"as2", b"as3", b"as4", b"as5", b"as6", b"as7", b"as8")


def read_asd_header_joins(path: str | Path) -> tuple[float, float] | None:
    """Splice (detector join) wavelengths from an ASD binary header.

    Legacy v1 (``ASD\0``) and as2-as8 headers store them as float32 at bytes 444 and 448
    (checked on LabSpec 4 as8, FieldSpec 3 v1 and dasp's example files: 1000/1800; a
    Trek-type as7 stores 0/0, meaning none). The header is read directly because the
    SpecDAL path used for modern files drops it.

    Returns:
        ``(join1, join2)`` when both are finite, ordered and inside the file's grid with
        100 nm to spare; otherwise ``None``.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = fh.read(484)
    # Binary headers have a NUL right after the magic; an ASCII export can start "ASD ...".
    if len(raw) < 484 or raw[:3] not in _ASD_MAGIC or raw[3] != 0:
        return None
    ch1 = struct.unpack_from("<f", raw, 191)[0]
    step = struct.unpack_from("<f", raw, 195)[0]
    channels = struct.unpack_from("<H", raw, 204)[0]
    if path.stat().st_size < 484 + 4 * channels:  # float32 data at least
        return None
    s1 = struct.unpack_from("<f", raw, 444)[0]
    s2 = struct.unpack_from("<f", raw, 448)[0]
    if not all(np.isfinite(v) for v in (ch1, step, s1, s2)) or step <= 0 or channels < 2:
        return None
    lo, hi = ch1, ch1 + step * (channels - 1)
    if not (lo + 100.0 <= s1 < s2 <= hi - 100.0):
        return None
    return float(s1), float(s2)
