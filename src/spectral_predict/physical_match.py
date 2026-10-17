"""Physical-match calibration transfer: per detector block, a blur, a band-depth gain and an
optional wavelength shift, learned from paired standards.

Model for block b (blocks split at the detector joins):

    satellite --(Gaussian blur sigma_b nm, shift delta_b nm)--> in absorbance A' = g_b * A + c_b

Resolution acts on intensity, so for reflectance input the blur is applied to reflectance;
band depth scales in absorbance (Beer-Lambert), so the gain is applied to log10(1/R). For
absorbance input the blur is applied to absorbance (the small-blur approximation).

Why this form: on SPYDER Bone's 40 bones scanned on LabSpec 4 Hi-Res and Standard-Res units,
the difference was mostly a band-depth gain (hi-res features ~1.6x deeper) plus a little blur.
DS and PDS fit thousands of coefficients from the standards and learn the session: fitted on
one unit pair and applied to the other, PDS gave M2045 RMSD 3.92 (slope 0.38); this
two-to-three-parameter-per-block form, learned from the spectra alone, gave 2.99 (slope 0.88).
See docs/SESSION_LOG.md 2026-10-16.

Fitting, per block, on the MEAN Savitzky-Golay second derivative over the standards:

- Averaging over standards removes most errors-in-variables bias that satellite noise puts
  on a pooled least-squares gain (pooled fits ran 0.6x too low on hi-res SWIR2). The final
  gain also subtracts the estimated noise variance of the satellite mean from the
  denominator (floored at half of it).
- Noisy regions (see ``spectral_diagnostics.noisy_regions``) and channels with
  reflectance <= 0 are left out; the rest count equally.
- sigma is chosen on the residual minus its expected noise floor, so a blur cannot win by
  smoothing satellite noise away, then the smallest sigma within 2% of the best is kept.
- If blurring the PRIMARY fits clearly better (>= 20% lower corrected residual), the
  satellite is the coarser instrument; a transfer cannot sharpen it, so sigma = 0 and a
  note suggests refitting the model on blurred primary spectra.
- A block with no measurable second-derivative signal (RMS signal / RMS noise of the
  primary mean below 10) gets sigma = 0 and gain 1 (or the pooled gain on request).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from scipy.signal import savgol_coeffs

from . import spectral_diagnostics as sd

logger = logging.getLogger(__name__)

PHYSICAL_MATCH_FORMAT_VERSION = 1

SIGMA_GRID_NM: tuple[float, ...] = tuple(np.round(np.arange(0.0, 10.0001, 0.5), 2))
SHIFT_GRID_NM: tuple[float, ...] = tuple(np.round(np.arange(-3.0, 3.0001, 0.25), 2))
FEATURELESS_SNR = 10.0
FIT_MARGIN_NM = 25.0
SIGMA_TOLERANCE = 0.02
DIRECTION_MARGIN = 0.8
KERNEL_TRUNCATE = 4.0
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ----------------------------------------------------------------------------- operator


def blur_shift(X: np.ndarray, wl: np.ndarray, sigma: float, shift: float = 0.0) -> np.ndarray:
    """Gaussian blur (sigma nm) centred at wl + shift, within one block.

    Output channel i is the normalised Gaussian-weighted mean of the input around
    ``wl[i] + shift``; taps outside the block are dropped and the rest renormalised.
    ``sigma = 0`` with a shift is linear interpolation (edge values held).
    """
    X = np.atleast_2d(X)
    if sigma <= 0.0 and shift == 0.0:
        return X.copy()
    if sigma <= 0.0:
        return np.vstack([np.interp(wl + shift, wl, row) for row in X])
    step = sd.grid_step(wl)
    nb = wl.size
    r = int(np.ceil((KERNEL_TRUNCATE * sigma + abs(shift)) / step)) + 1
    offsets = np.arange(-r, r + 1)
    J = np.arange(nb)[:, None] + offsets[None, :]
    inside = (J >= 0) & (J < nb)
    Jc = np.clip(J, 0, nb - 1)
    t = (wl[Jc] - (wl[:, None] + shift)) / sigma
    W = np.where(inside & (np.abs(t) <= KERNEL_TRUNCATE), np.exp(-0.5 * t**2), 0.0)
    tot = W.sum(axis=1)
    empty = tot <= 0
    if np.any(empty):  # no tap in reach: nearest in-block channel
        near = np.clip(np.searchsorted(wl, wl[empty] + shift), 0, nb - 1)
        W[empty] = 0.0
        W[empty, :] = (Jc[empty] == near[:, None]).astype(float)
        tot = W.sum(axis=1)
    W /= tot[:, None]
    Y = np.zeros_like(X, dtype=np.float64)
    for k in range(offsets.size):
        col = Jc[:, k]
        Y += W[:, k] * X[:, col]
    return Y


def _kernel_channels(sigma: float, shift: float, step: float) -> np.ndarray:
    """Interior blur kernel in channel taps (for noise propagation)."""
    if sigma <= 0.0:
        return np.array([1.0])
    r = int(np.ceil(KERNEL_TRUNCATE * sigma / step))
    t = np.arange(-r, r + 1) * step / sigma
    h = np.exp(-0.5 * t**2)
    return h / h.sum()


def _transform_block(
    Xb: np.ndarray, wl_b: np.ndarray, space: str, sigma: float, shift: float
) -> np.ndarray:
    """Blur/shift one block in the physical domain and return absorbance."""
    if space == "reflectance":
        Rb = blur_shift(np.where(np.isfinite(Xb), Xb, 0.0), wl_b, sigma, shift)
        return np.log10(1.0 / np.clip(Rb, sd.REFLECTANCE_FLOOR, None))
    return blur_shift(Xb, wl_b, sigma, shift)


def _block_bounds(params: dict, wl: np.ndarray) -> list[slice]:
    blocks = np.atleast_2d(np.asarray(params["blocks"], dtype=np.float64))
    out = []
    covered = np.zeros(wl.size, dtype=bool)
    for lo, hi in blocks:
        idx = np.flatnonzero((wl >= lo - 1e-6) & (wl <= hi + 1e-6))
        if idx.size == 0 or np.any(np.diff(idx) != 1):
            raise ValueError(
                f"Physical-match block {lo:g}-{hi:g} nm is not on this wavelength grid"
            )
        out.append(slice(int(idx[0]), int(idx[-1]) + 1))
        covered[idx] = True
    if not covered.all():
        raise ValueError(
            "Wavelength grid extends beyond the physical-match blocks; apply on the grid "
            "the transfer was fitted on"
        )
    return out


def apply_physical_match(
    X_satellite_new: np.ndarray, params: dict, wavelengths: Sequence[float]
) -> np.ndarray:
    """Apply a fitted physical match to satellite spectra.

    Args:
        X_satellite_new: (m, p) spectra in the space the transfer was fitted in.
        params: From ``estimate_physical_match``.
        wavelengths: (p,) grid (the transfer's common grid).

    Returns:
        (m, p) spectra in primary-instrument space, same space as the input.

    Raises:
        ValueError: Grid does not match the stored blocks, or reflectance expected but
            the data look like absorbance or percent reflectance.
    """
    X = np.atleast_2d(np.asarray(X_satellite_new, dtype=np.float64))
    wl = sd.check_wavelengths(wavelengths, X.shape[1])
    blocks = np.atleast_2d(np.asarray(params["blocks"], dtype=np.float64))
    sd.check_uniform_blocks(wl, blocks[:-1, 1])
    space = str(params["space"])
    sd.check_space(space)
    if space == "reflectance" and np.mean(X > 1.5) > 0.5:
        raise ValueError(
            "This transfer was fitted on reflectance (0-1); the spectra look like absorbance "
            "or percent reflectance"
        )
    sig = np.atleast_1d(np.asarray(params["sigma_nm"], dtype=np.float64))
    sh = np.atleast_1d(np.asarray(params["shift_nm"], dtype=np.float64))
    g = np.atleast_1d(np.asarray(params["gain"], dtype=np.float64))
    c = np.atleast_1d(np.asarray(params["offset"], dtype=np.float64))
    A, valid = sd.to_absorbance(X, space)
    X, _ = _fill_invalid(X, A, valid, wl, space)  # a dead channel must not spread
    out = np.empty_like(X)
    for b, slc in enumerate(_block_bounds(params, wl)):
        A = _transform_block(X[:, slc], wl[slc], space, float(sig[b]), float(sh[b]))
        out[:, slc] = sd.from_absorbance(g[b] * A + c[b], space)
    return out


# ----------------------------------------------------------------------------- fitting


def _fill_invalid(
    X: np.ndarray, A: np.ndarray, valid: np.ndarray, wl: np.ndarray, space: str
) -> tuple[np.ndarray, np.ndarray]:
    """Replace unusable channels of each spectrum by linear interpolation in absorbance."""
    if valid.all():
        return X, A
    A = A.copy()
    for i in np.flatnonzero(~valid.all(axis=1)):
        ok = valid[i]
        if ok.sum() < 2:
            raise ValueError("A spectrum has fewer than 2 usable channels")
        A[i, ~ok] = np.interp(wl[~ok], wl[ok], A[i, ok])
    return sd.from_absorbance(A, space), A


def _dilate(mask: np.ndarray, wl: np.ndarray, nm: float) -> np.ndarray:
    """Grow a channel mask by ``nm`` on each side."""
    if not mask.any():
        return mask
    out = mask.copy()
    for w in wl[mask]:
        out |= np.abs(wl - w) <= nm
    return out


def _d2_block(A: np.ndarray, wl_b: np.ndarray) -> np.ndarray:
    return sd.second_derivative(A, wl_b, None)


def _noise_factor(kernel: np.ndarray, d2: np.ndarray, blur: np.ndarray) -> float:
    """Variance factor of D2 noise after a blur, relative to no blur."""
    base = float((np.convolve(kernel, d2) ** 2).sum())
    if base <= 0:
        return 1.0
    return float((np.convolve(np.convolve(kernel, blur), d2) ** 2).sum()) / base


class _BlockFit:
    """Mean-D2 fit of one detector block."""

    def __init__(self, Xp, Xs, Ap, wl_b, space, mask, v_p, v_s0, kernel_p, kernel_s, d2c, step):
        self.Xp, self.Xs, self.Ap, self.wl, self.space = Xp, Xs, Ap, wl_b, space
        self.mask, self.v_p, self.v_s0 = mask, v_p, v_s0
        self.kernel_p, self.kernel_s, self.d2c, self.step = kernel_p, kernel_s, d2c, step
        self.dp = _d2_block(Ap, wl_b).mean(axis=0)
        # Equal weights inside the fit mask: the noise map decides which channels take part,
        # not how much each counts. Inverse-noise weights leaned on the quietest channels and
        # moved SPYDER's pooled gain from 0.65 to 0.73 (cross-unit M2045 3.48 vs 3.20).
        self.w = mask.astype(np.float64)
        self._cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    def sat(self, sigma: float, shift: float) -> tuple[np.ndarray, np.ndarray]:
        key = (sigma, shift)
        if key not in self._cache:
            A = _transform_block(self.Xs, self.wl, self.space, sigma, shift)
            ds = _d2_block(A, self.wl).mean(axis=0)
            vs = self.v_s0 * _noise_factor(
                self.kernel_s, self.d2c, _kernel_channels(sigma, shift, self.step)
            )
            self._cache[key] = (ds, vs)
        return self._cache[key]

    def criterion(self, ds: np.ndarray, vs: np.ndarray, dp=None, vp=None):
        dp = self.dp if dp is None else dp
        vp = self.v_p if vp is None else vp
        m, w = self.mask, self.w
        den = float((w[m] * ds[m] ** 2).sum())
        g = float((w[m] * ds[m] * dp[m]).sum()) / den if den > 0 else 1.0
        resid = float((w[m] * (dp[m] - g * ds[m]) ** 2).sum())
        floor = float((w[m] * (vp[m] + g**2 * vs[m])).sum())
        return g, resid - floor

    def signal(self) -> float:
        return float((self.w[self.mask] * self.dp[self.mask] ** 2).sum())

    def snr(self) -> float:
        m = self.mask
        noise = float(self.v_p[m].sum())
        return float(np.sqrt(float((self.dp[m] ** 2).sum()) / noise)) if noise > 0 else np.inf

    def corrected_gain_terms(self, sigma: float, shift: float) -> tuple[float, float]:
        ds, vs = self.sat(sigma, shift)
        m, w = self.mask, self.w
        num = float((w[m] * ds[m] * self.dp[m]).sum())
        raw = float((w[m] * ds[m] ** 2).sum())
        den = max(raw - float((w[m] * vs[m]).sum()), 0.5 * raw)
        return num, den


def _choose_sigma(fit: _BlockFit, sigmas: Sequence[float], shift: float) -> tuple[float, float]:
    """Smallest sigma within tolerance of the best corrected residual; returns (sigma, R)."""
    crit = [(s, fit.criterion(*fit.sat(s, shift))[1]) for s in sigmas]
    best = min(r for _, r in crit)
    tol = SIGMA_TOLERANCE * abs(best) + 1e-4 * fit.signal()
    for s, r in crit:  # sigmas ascending
        if r <= best + tol:
            return float(s), float(r)
    return float(crit[0][0]), float(crit[0][1])


def _choose_shift(fit: _BlockFit, sigma: float) -> tuple[float, float]:
    crit = [(d, fit.criterion(*fit.sat(sigma, d))[1]) for d in SHIFT_GRID_NM]
    best = min(r for _, r in crit)
    tol = SIGMA_TOLERANCE * abs(best) + 1e-4 * fit.signal()
    # smallest |shift| within tolerance
    for d, r in sorted(crit, key=lambda t: (abs(t[0]), t[0])):
        if r <= best + tol:
            return float(d), float(r)
    return 0.0, float(crit[0][1])


def _primary_blur_better(fit: _BlockFit, best_r: float, sigmas: Sequence[float]) -> bool:
    if best_r <= 0:
        return False
    ds0, vs0 = fit.sat(0.0, 0.0)
    best_prim = np.inf
    for s in sigmas:
        if s <= 0:
            continue
        Ap = _transform_block(fit.Xp, fit.wl, fit.space, float(s), 0.0)
        dp = _d2_block(Ap, fit.wl).mean(axis=0)
        vp = fit.v_p * _noise_factor(
            fit.kernel_p, fit.d2c, _kernel_channels(float(s), 0.0, fit.step)
        )
        best_prim = min(best_prim, fit.criterion(ds0, vs0, dp=dp, vp=vp)[1])
    return bool(best_prim <= DIRECTION_MARGIN * best_r)


def _sigma_from_fwhm(fwhm, n_blocks: int) -> list[float]:
    fp, fs = fwhm
    fp = np.broadcast_to(np.asarray(fp, dtype=float), (n_blocks,))
    fs = np.broadcast_to(np.asarray(fs, dtype=float), (n_blocks,))
    return [float(np.sqrt(max(a**2 - b**2, 0.0)) * FWHM_TO_SIGMA) for a, b in zip(fp, fs)]


def estimate_physical_match(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    wavelengths: Sequence[float],
    *,
    space: str,
    joins: Sequence[float] | None = None,
    sigma: str | float | Sequence[float] = "learn",
    fwhm: tuple[Any, Any] | None = None,
    gain: str = "per_block",
    shift: bool = False,
    featureless: str = "unity",
    exclude: Sequence[tuple[float, float]] | None = None,
    noisy_factor: float = 10.0,
) -> dict:
    """Fit a per-block blur + band-depth gain (+ optional shift) from paired standards.

    Args:
        X_primary: (n, p) primary spectra of the standards.
        X_satellite: (n, p) satellite spectra of the same standards, same row order.
        wavelengths: (p,) common grid, strictly increasing.
        space: ``'reflectance'`` (0-1) or ``'absorbance'``.
        joins: Detector joins of the satellite (last wavelength of each left block);
            ``None`` = one block. See ``spectral_diagnostics.resolve_joins``.
        sigma: ``'learn'`` (0-10 nm grid per block), one value in nm for every block,
            or one per block. Ignored when ``fwhm`` is given.
        fwhm: ``(primary_fwhm, satellite_fwhm)`` in nm, scalars or one per block
            (datasheet resolutions); sigma = sqrt(max(Fp^2 - Fs^2, 0)) / 2.3548.
        gain: ``'per_block'`` or ``'single'`` (one gain pooled over blocks with signal).
        shift: Also learn a wavelength shift per block (-3..3 nm, 0.25 nm steps).
        featureless: ``'unity'`` (gain 1) or ``'pooled'`` for blocks without signal.
        exclude: Extra ``(lo, hi)`` nm regions left out of the fit.
        noisy_factor: Regions noisier than this multiple of the block's quiet level, on
            either instrument, are left out of the fit.

    Returns:
        Params for ``apply_physical_match``: ``space``, ``blocks`` (B, 2) nm,
        ``sigma_nm``, ``shift_nm``, ``gain``, ``offset`` (each (B,)), ``featureless`` and
        ``satellite_coarser`` (bool (B,)), ``fit_mask`` (p,) bool, ``snr`` (B,), and
        ``notes`` (one string; empty when there is nothing to report).

    Raises:
        ValueError: Shapes differ, fewer than 2 standards, bad wavelengths or space.
    """
    Xp = np.atleast_2d(np.asarray(X_primary, dtype=np.float64))
    Xs = np.atleast_2d(np.asarray(X_satellite, dtype=np.float64))
    if Xp.shape != Xs.shape:
        raise ValueError(f"X_primary {Xp.shape} and X_satellite {Xs.shape} differ in shape")
    n, p = Xp.shape
    if n < 2:
        raise ValueError(f"The physical match needs at least 2 standards, got {n}")
    wl = sd.check_wavelengths(wavelengths, p)
    sd.check_space(space)
    if gain not in ("per_block", "single"):
        raise ValueError(f"gain must be 'per_block' or 'single', got {gain!r}")
    if featureless not in ("unity", "pooled"):
        raise ValueError(f"featureless must be 'unity' or 'pooled', got {featureless!r}")

    sd.check_uniform_blocks(wl, joins)
    A_p, valid_p = sd.to_absorbance(Xp, space)
    A_s, valid_s = sd.to_absorbance(Xs, space)
    # Unusable channels (reflectance <= 0, non-finite) are filled from their neighbours
    # so the 31 nm derivative and the blur cannot spread them; the fit mask then stays
    # a derivative half-window away from them.
    Xp, A_p = _fill_invalid(Xp, A_p, valid_p, wl, space)
    Xs, A_s = _fill_invalid(Xs, A_s, valid_s, wl, space)
    bad = ~(valid_p.all(axis=0) & valid_s.all(axis=0))
    valid = ~_dilate(bad, wl, 0.5 * sd.D2_WINDOW_NM)
    blocks = sd.block_slices(wl, joins)
    B = len(blocks)
    ne_p = sd.estimate_noise(A_p, wl, joins)
    ne_s = sd.estimate_noise(A_s, wl, joins)
    noisy = sd.regions_mask(
        sd.noisy_regions(ne_p.channel_sd, wl, joins, factor=noisy_factor), wl
    ) | sd.regions_mask(sd.noisy_regions(ne_s.channel_sd, wl, joins, factor=noisy_factor), wl)
    excl = sd.regions_mask(() if exclude is None else exclude, wl)

    if fwhm is not None:
        fixed = _sigma_from_fwhm(fwhm, B)
    elif isinstance(sigma, str):
        if sigma != "learn":
            raise ValueError(f"sigma must be 'learn' or numbers, got {sigma!r}")
        fixed = None
    else:
        fixed = [float(v) for v in np.broadcast_to(np.asarray(sigma, dtype=float), (B,))]
        if any(v < 0 for v in fixed):
            raise ValueError("sigma must be >= 0")

    fit_mask = np.zeros(p, dtype=bool)
    fits: list[_BlockFit | None] = []
    for slc in blocks:
        wl_b = wl[slc]
        interior = (wl_b >= wl_b[0] + FIT_MARGIN_NM) & (wl_b <= wl_b[-1] - FIT_MARGIN_NM)
        m = interior & valid[slc] & ~noisy[slc] & ~excl[slc]
        fit_mask[slc] = m
        if m.sum() < 10:
            fits.append(None)
            continue
        step = sd.block_step(wl, slc)
        w_d2 = sd.odd_channels(sd.D2_WINDOW_NM, step, 5)
        w_d2 = min(w_d2, wl_b.size if wl_b.size % 2 else wl_b.size - 1)
        d2c = savgol_coeffs(w_d2, 3, deriv=2, delta=step, use="dot")
        bi = len(fits)
        kname_p = ne_p.kernels[bi] if bi < len(ne_p.kernels) else "white"
        kname_s = ne_s.kernels[bi] if bi < len(ne_s.kernels) else "white"
        fits.append(
            _BlockFit(
                Xp[:, slc],
                Xs[:, slc],
                A_p[:, slc],
                wl_b,
                space,
                m,
                ne_p.d2_sd[slc] ** 2 / n,
                ne_s.d2_sd[slc] ** 2 / n,
                sd.noise_kernel(kname_p),
                sd.noise_kernel(kname_s),
                d2c,
                step,
            )
        )

    sig = np.zeros(B)
    sh = np.zeros(B)
    snr = np.zeros(B)
    feat = np.zeros(B, dtype=bool)
    coarser = np.zeros(B, dtype=bool)
    notes: list[str] = []
    for b, fit in enumerate(fits):
        lo, hi = float(wl[blocks[b]][0]), float(wl[blocks[b]][-1])
        if fit is None:
            feat[b] = True
            notes.append(f"{lo:g}-{hi:g} nm: too few usable wavelengths to fit; left unscaled.")
            continue
        snr[b] = fit.snr()
        if snr[b] < FEATURELESS_SNR:
            feat[b] = True
            notes.append(
                f"{lo:g}-{hi:g} nm: no measurable band structure (signal/noise {snr[b]:.1f}); "
                "no blur, gain " + ("pooled." if featureless == "pooled" else "1.")
            )
            continue
        if fixed is not None:
            s_b = fixed[b]
            d_b = _choose_shift(fit, s_b)[0] if shift else 0.0
        else:
            s_b, r_b = _choose_sigma(fit, SIGMA_GRID_NM, 0.0)
            d_b = 0.0
            if shift:
                d_b = _choose_shift(fit, s_b)[0]
                s_b, r_b = _choose_sigma(fit, SIGMA_GRID_NM, d_b)
            best_r = min(fit.criterion(*fit.sat(s, d_b))[1] for s in SIGMA_GRID_NM)
            if _primary_blur_better(fit, best_r, SIGMA_GRID_NM):
                coarser[b] = True
                s_b = 0.0
                notes.append(
                    f"{lo:g}-{hi:g} nm: the satellite looks lower-resolution than the primary. "
                    "A transfer cannot sharpen it, so only the gain is corrected; refitting the "
                    "model on primary spectra blurred to the satellite's resolution would match "
                    "better."
                )
        sig[b], sh[b] = s_b, d_b

    gains = np.ones(B)
    terms = {}
    for b, fit in enumerate(fits):
        if fit is not None and not feat[b]:
            terms[b] = fit.corrected_gain_terms(sig[b], sh[b])
            num, den = terms[b]
            gains[b] = num / den if den > 0 else 1.0
    pooled = None
    if terms:
        num = sum(t[0] for t in terms.values())
        den = sum(t[1] for t in terms.values())
        pooled = num / den if den > 0 else None
    if gain == "single" and pooled is not None:
        for b in terms:
            gains[b] = pooled
    if featureless == "pooled" and pooled is not None:
        gains[feat] = pooled

    offsets = np.zeros(B)
    for b, slc in enumerate(blocks):
        A_t = _transform_block(Xs[:, slc], wl[slc], space, float(sig[b]), float(sh[b]))
        m = fit_mask[slc] if fit_mask[slc].any() else valid[slc]
        if not m.any():
            m = np.ones(slc.stop - slc.start, dtype=bool)
        offsets[b] = float(np.mean(A_p[:, slc][:, m] - gains[b] * A_t[:, m]))

    return {
        "form": "physical_match",
        "space": space,
        "blocks": np.array([[wl[s][0], wl[s][-1]] for s in blocks], dtype=np.float64),
        "sigma_nm": sig,
        "shift_nm": sh,
        "gain": gains,
        "offset": offsets,
        "featureless": feat,
        "satellite_coarser": coarser,
        "snr": snr,
        "fit_mask": fit_mask,
        "notes": " ".join(
            notes
            + (
                [
                    "Only 2 standards: the noise estimate is rough, so blur and band-structure "
                    "checks are approximate."
                ]
                if n < 3
                else []
            )
        ),
    }
