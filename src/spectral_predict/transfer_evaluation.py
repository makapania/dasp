"""Leave-one-standard-out validation and comparison of calibration-transfer methods.

A transfer is judged by prediction error on the second (satellite) instrument, in y
units, not by how well it fits the standards it was estimated from (Workman 2018,
*Appl Spectrosc* 72(3):340-365). ``evaluate_transfer`` refits every candidate with
each standard left out in turn, transforms the left-out satellite spectrum, and
predicts it. A "No correction" row is always present, so a leaderboard can say that
nothing helped. With a model the board is ranked by ``RMSD_vs_primary``, the distance
of each satellite prediction from the primary spectrum's prediction: it is in y units
like RMSEP but excludes the model's own error, which on a few standards can cancel a
transfer error by chance.

Every candidate, in every leave-one-out fold, is fitted by ``fit_transfer``, the
function a caller then uses to build the chosen transfer on all standards, so the
evaluated and the deployed transfer are the same code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Union

import numpy as np
import pandas as pd

from . import calibration_transfer as ct
from . import spectral_diagnostics as sd
from .physical_match import PHYSICAL_MATCH_FORMAT_VERSION
from .scoring import regression_figures_of_merit

logger = logging.getLogger(__name__)

PredictFn = Callable[[np.ndarray], np.ndarray]
"""Maps spectra of shape (m, p) on the common grid to 1-D numeric predictions."""

_METHOD_KINDS: dict[str, str] = {
    "none": "none",
    "tsr": "spectral",
    "tsr_bias": "spectral",
    "pds": "spectral",
    "ds": "spectral",
    "pds_legacy": "spectral",
    "ds_legacy": "spectral",
    "pred_slope_bias": "prediction",
    "pred_bias": "prediction",
    "physical": "spectral",
}

_REQUIRED_SETTINGS: dict[str, tuple[str, ...]] = {
    "pds": ("window", "rank"),
    "pds_legacy": ("window",),
    "ds": ("lam_rel",),
    "ds_legacy": ("lam",),
}

DEFAULT_PDS_WINDOWS: tuple[int, ...] = (5, 11, 21, 31)
DEFAULT_PDS_MAX_RANK: int = 4
DEFAULT_DS_LAM_REL: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)

MIN_STANDARDS: int = 3
"""Leave-one-out needs at least two standards in every fit."""

REFERENCE_LABEL = "Primary instrument (reference)"
NONE_LABEL = "No correction"


@dataclass(frozen=True)
class TransferCandidate:
    """One transfer method with fixed settings.

    Attributes:
        method: ``'none'``, ``'tsr'``, ``'tsr_bias'``, ``'pds'`` (centred, low rank),
            ``'ds'`` (centred dual-form ridge), ``'pred_slope_bias'``, ``'pred_bias'``,
            ``'physical'`` (blur + band-depth gain per detector block; settings
            ``sigma`` = ``'learn'`` or nm, ``gain`` = ``'per_block'`` or ``'single'``,
            ``shift`` = bool; needs ``wavelengths`` and ``space``), or ``'pds_legacy'``
            / ``'ds_legacy'`` (the pre-2026-10 estimators, for comparing with existing
            settings).
        params: Sorted ``(name, value)`` pairs; build with ``TransferCandidate.make``.
    """

    method: str
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.method not in _METHOD_KINDS:
            raise ValueError(
                f"Unknown transfer candidate method {self.method!r}; "
                f"expected one of {sorted(_METHOD_KINDS)}"
            )
        missing = [k for k in _REQUIRED_SETTINGS.get(self.method, ()) if k not in dict(self.params)]
        if missing:
            raise ValueError(f"Transfer candidate {self.method!r} needs settings {missing}")
        if self.method == "physical":
            s = dict(self.params)
            sig = s.get("sigma", "learn")
            if not (sig == "learn" or (isinstance(sig, (int, float)) and sig >= 0)):
                raise ValueError(f"physical sigma must be 'learn' or a number >= 0, got {sig!r}")
            if s.get("gain", "per_block") not in ("per_block", "single"):
                raise ValueError(
                    f"physical gain must be 'per_block' or 'single', got {s['gain']!r}"
                )

    @classmethod
    def make(cls, method: str, **params: Any) -> TransferCandidate:
        """Build a candidate from keyword settings."""
        return cls(method, tuple(sorted(params.items())))

    @property
    def kind(self) -> str:
        """``'none'``, ``'spectral'`` (maps spectra) or ``'prediction'`` (corrects ŷ)."""
        return _METHOD_KINDS[self.method]

    @property
    def settings(self) -> dict[str, Any]:
        """The settings as a dict."""
        return dict(self.params)

    @property
    def label(self) -> str:
        """Human-readable name for a leaderboard row."""
        s = self.settings
        if self.method == "none":
            return NONE_LABEL
        if self.method == "tsr":
            return "Slope/bias per wavelength"
        if self.method == "tsr_bias":
            return "Bias per wavelength"
        if self.method == "pds":
            return f"PDS (centred, w={s['window']}, rank {s['rank']})"
        if self.method == "pds_legacy":
            return f"PDS (legacy, w={s['window']})"
        if self.method == "ds":
            return f"DS (centred, ridge {s['lam_rel']:g})"
        if self.method == "ds_legacy":
            return f"DS (legacy, ridge {s['lam']:g})"
        if self.method == "pred_slope_bias":
            return "Prediction slope/bias"
        if self.method == "physical":
            sig = s.get("sigma", "learn")
            if sig == "learn":
                blur = "learned blur"
            elif float(sig) == 0.0:
                blur = "no blur"
            else:
                blur = f"blur {float(sig):g} nm"
            gain = "one gain" if s.get("gain", "per_block") == "single" else "gain per block"
            extra = ", shift" if s.get("shift", False) else ""
            return f"Physical match ({blur}, {gain}{extra})"
        return "Prediction bias"

    @property
    def min_fit_standards(self) -> int:
        """Fewest standards one fit of this candidate needs."""
        if self.method == "pds":
            return int(self.settings["rank"]) + 1
        if self.method in ("tsr", "ds", "physical"):
            return 2
        if self.method == "pred_slope_bias":
            return 3
        return 1


@dataclass
class TransferEvaluation:
    """Result of ``evaluate_transfer``.

    Attributes:
        leaderboard: One row per candidate, best first, plus the unranked reference
            row last when there is one.
        score_column: The column the leaderboard is sorted by (``'RMSEP'``,
            ``'RMSD_vs_primary'``, ``'spectral_RMSE'`` or one of their ``_grouped``
            forms).
        n_standards: Number of paired standards.
        n_candidates: Number of candidates that were scored (including "No
            correction"; failed candidates excluded). The best of many leave-one-out
            scores is mildly optimistic.
        ids: The standard IDs, in the row order of the out-of-fold arrays.
        oof_predictions: Label -> leave-one-standard-out prediction per standard.
        oof_residual_spectra: Label -> (n_standards, p) primary minus transferred
            spectra (spectral candidates and "No correction" only).
        candidates: Label -> the candidate, to pass to ``fit_transfer``.
        warnings: Plain-language cautions about the evaluation.
        oof_predictions_grouped: Label -> leave-one-group-out prediction per standard
            (only with ``groups``).
        groups: The group of each standard, or None.
        diagnostics: Joins (and their source), noisy regions of each instrument, and
            where the satellite is much noisier; filled when ``wavelengths`` and
            ``space`` are given.
    """

    leaderboard: pd.DataFrame
    score_column: str
    n_standards: int
    n_candidates: int
    ids: list = field(default_factory=list)
    oof_predictions: dict[str, np.ndarray] = field(default_factory=dict)
    oof_residual_spectra: dict[str, np.ndarray] = field(default_factory=dict)
    candidates: dict[str, TransferCandidate] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    oof_predictions_grouped: dict[str, np.ndarray] = field(default_factory=dict)
    groups: list | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> TransferCandidate:
        """The top-ranked candidate (may be "No correction")."""
        return self.candidates[self.leaderboard.iloc[0]["label"]]


def default_transfer_candidates(
    n_standards: int,
    *,
    has_y: bool,
    has_predict: bool,
    n_wavelengths: int | None = None,
    physical: bool = False,
) -> list[TransferCandidate]:
    """The default bake-off for ``n_standards`` paired standards.

    Only candidates that every leave-one-out fit (``n_standards - 1`` standards) can
    estimate are returned. Prediction corrections need reference values and a model.

    Args:
        n_standards: Number of paired standards.
        has_y: Reference values for the standards are available.
        has_predict: A prediction function (model) is available.
        n_wavelengths: Drop PDS windows wider than the spectrum.
        physical: Add the physical-match candidates (they need wavelengths and a
            spectral space; ``evaluate_transfer`` sets this when it has both).

    Returns:
        Candidates, "No correction" first.
    """
    n_fit = int(n_standards) - 1
    out = [TransferCandidate.make("none")]
    if n_fit >= 2:
        out.append(TransferCandidate.make("tsr"))
    out.append(TransferCandidate.make("tsr_bias"))
    for window in DEFAULT_PDS_WINDOWS:
        if n_wavelengths is not None and window > n_wavelengths:
            continue
        for rank in range(1, min(n_fit - 1, DEFAULT_PDS_MAX_RANK) + 1):
            out.append(TransferCandidate.make("pds", window=window, rank=rank))
    if n_fit >= 2:
        for lam_rel in DEFAULT_DS_LAM_REL:
            out.append(TransferCandidate.make("ds", lam_rel=lam_rel))
        if physical:
            out.append(TransferCandidate.make("physical", sigma="learn", gain="per_block"))
            out.append(TransferCandidate.make("physical", sigma="learn", gain="single"))
            out.append(TransferCandidate.make("physical", sigma=0.0, gain="per_block"))
    if has_y and has_predict:
        out.append(TransferCandidate.make("pred_bias"))
        if n_fit >= 3:
            out.append(TransferCandidate.make("pred_slope_bias"))
    return out


def _check_predictions(values: Any, m: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or arr.shape[0] != m:
        raise ValueError(
            f"predict must return one numeric value per spectrum (shape ({m},)), "
            f"got shape {arr.shape}"
        )
    try:
        return arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("predict must return numeric values (regression models)") from exc


JoinsArg = Union[str, Sequence[float], None]


def _fold_joins(
    Xp: np.ndarray, Xs: np.ndarray, wl: np.ndarray, space: str, joins: JoinsArg
) -> tuple[tuple[float, ...], str]:
    """Joins for one fit: given ones as they are; ``'auto'`` = detected from these rows
    of both instruments (union, merged within 5 nm)."""
    if joins is None:
        return (), "none"
    if isinstance(joins, str):
        if joins != "auto":
            raise ValueError(f"joins must be 'auto', None or wavelengths, got {joins!r}")
        found: list[float] = []
        for X in (Xs, Xp):
            det = sd.detect_detector_joins(X, wl, space=space)
            for j in det.joins:
                if all(abs(j - f) > 5.0 for f in found):
                    found.append(j)
        return tuple(sorted(found)), ("detected" if found else "none")
    return tuple(sorted(float(j) for j in joins)), "given"


def fit_transfer(
    candidate: TransferCandidate,
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    *,
    y: np.ndarray | None = None,
    predict: PredictFn | None = None,
    wavelengths: np.ndarray | None = None,
    space: str | None = None,
    joins: JoinsArg = "auto",
    primary_id: str = "primary",
    satellite_id: str = "satellite",
) -> ct.TransferModel | dict | None:
    """Fit one candidate on all the given paired standards.

    Args:
        candidate: What to fit.
        X_primary: Primary spectra of the standards, (n, p).
        X_satellite: Satellite spectra of the same standards in the same row order.
        y: Reference values (prediction corrections only).
        predict: Model prediction function (prediction corrections only).
        wavelengths: Common grid stored in the transfer model (default 0..p-1);
            required for the physical match.
        space: ``'reflectance'`` or ``'absorbance'``; required for the physical match.
        joins: Detector joins for the physical match: ``'auto'`` detects them from
            these standards, ``None`` means one block, or give the wavelengths (from a
            file header or the user; see ``spectral_diagnostics.resolve_joins``).
        primary_id: Stored in the transfer model.
        satellite_id: Stored in the transfer model.

    Returns:
        ``None`` for "No correction"; a ``TransferModel`` (apply with
        ``calibration_transfer.apply_transfer_dispatch``) for a spectral method; a
        ``bias_correction``-format dict (``calibration_transfer.apply_prediction_correction``) for a
        prediction correction.

    Raises:
        ValueError: Shapes differ, too few standards, a prediction correction
            without ``y`` and ``predict``, or a physical match without wavelengths
            and space.
    """
    Xp = np.asarray(X_primary, dtype=np.float64)
    Xs = np.asarray(X_satellite, dtype=np.float64)
    if Xp.ndim != 2 or Xp.shape != Xs.shape:
        raise ValueError(
            f"X_primary and X_satellite must be 2-D with the same shape, got "
            f"{Xp.shape} and {Xs.shape}"
        )
    n, p = Xs.shape
    if n < candidate.min_fit_standards:
        raise ValueError(
            f"{candidate.label} needs at least {candidate.min_fit_standards} standards, got {n}"
        )
    s = candidate.settings
    if candidate.kind == "none":
        return None
    if candidate.kind == "prediction":
        if y is None or predict is None:
            raise ValueError(f"{candidate.label} needs reference values y and a predict function")
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.shape[0] != n:
            raise ValueError(f"y has {y_arr.shape[0]} values for {n} standards")
        y_sat = _check_predictions(predict(Xs), n)
        return ct.estimate_prediction_correction(
            y_arr, y_sat, fit_slope=candidate.method == "pred_slope_bias"
        )

    wl = np.arange(p, dtype=np.float64) if wavelengths is None else np.asarray(wavelengths)
    if wl.shape != (p,):
        raise ValueError(f"wavelengths has shape {wl.shape} for spectra with {p} columns")
    meta: dict[str, Any] = {"candidate": candidate.label, "n_standards": int(n)}
    if candidate.method in ("tsr", "tsr_bias"):
        method = "tsr"
        params = ct.estimate_tsr(
            Xp, Xs, np.arange(n), slope_bias_correction=candidate.method == "tsr"
        )
    elif candidate.method == "pds":
        method = "pds"
        params = ct.estimate_pds_lowrank(Xp, Xs, window=int(s["window"]), rank=int(s["rank"]))
    elif candidate.method == "pds_legacy":
        method = "pds"
        window = int(s["window"])
        params = {"B": ct.estimate_pds(Xp, Xs, window=window), "window": window}
    elif candidate.method == "ds":
        method = "ds"
        params = ct.estimate_ds_dual(Xp, Xs, lam_rel=float(s["lam_rel"]))
    elif candidate.method == "physical":
        if wavelengths is None or space is None:
            raise ValueError(f"{candidate.label} needs wavelengths and space")
        method = "physical_match"
        fold_joins, joins_source = _fold_joins(Xp, Xs, wl, space, joins)
        sig = s.get("sigma", "learn")
        params = ct.estimate_physical_match(
            Xp,
            Xs,
            wl,
            space=space,
            joins=fold_joins,
            sigma=sig if sig == "learn" else float(sig),
            gain=s.get("gain", "per_block"),
            shift=bool(s.get("shift", False)),
        )
        meta.update(
            joins=list(fold_joins), joins_source=joins_source, notes=params.get("notes", "")
        )
    else:  # ds_legacy
        method = "ds"
        params = {"A": ct.estimate_ds(Xp, Xs, lam=float(s["lam"]))}
    if candidate.method in ("pds", "ds"):
        meta["format_version"] = ct.TRANSFER_FORMAT_VERSION
    elif candidate.method == "physical":
        meta["format_version"] = PHYSICAL_MATCH_FORMAT_VERSION
    else:
        meta["format_version"] = 1
    return ct.TransferModel(
        primary_id=primary_id,
        satellite_id=satellite_id,
        method=method,
        wavelengths_common=wl,
        params=params,
        meta=meta,
    )


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.std(x)) == 0.0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _check_groups(
    groups: Sequence[Any] | None, n: int, warnings_out: list[str]
) -> np.ndarray | None:
    if groups is None:
        return None
    g = np.asarray([str(v) for v in groups], dtype=object)
    if g.shape[0] != n:
        raise ValueError(f"groups has {g.shape[0]} entries for {n} standards")
    labels, counts = np.unique(g.astype(str), return_counts=True)
    if labels.size < 2:
        warnings_out.append("All standards are in one group; only leave-one-out is scored.")
        return None
    if labels.size == 2:
        warnings_out.append(
            "Grouped scores fit on one group and test on the other (2 groups); each fit "
            "uses only part of the standards."
        )
    elif counts.min() < 3:
        warnings_out.append("Some groups have fewer than 3 standards; grouped scores are noisy.")
    return g


def evaluate_transfer(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    *,
    y: np.ndarray | None = None,
    predict: PredictFn | None = None,
    candidates: Sequence[TransferCandidate] | None = None,
    ids: Sequence[Any] | None = None,
    y_range_reference: tuple[float, float] | None = None,
    rank_by: str = "auto",
    groups: Sequence[Any] | None = None,
    wavelengths: Sequence[float] | None = None,
    space: str | None = None,
    joins: JoinsArg = "auto",
    model_wavelengths: Sequence[float] | None = None,
) -> TransferEvaluation:
    """Compare transfer methods by leave-one-standard-out, and by group when given.

    For each candidate and each standard i, the candidate is fitted on the other
    standards, satellite spectrum i is transformed (or its prediction corrected), and
    the result is scored against standard i. Standard i's spectrum and y are never
    used to fit the transfer that scores it.

    Leave-one-out shares a session's effects (presentation, drift, that day's noise)
    between the fit and the scored standard, so it favours methods that learn the
    session. On SPYDER Bone's paired bones it ranked PDS first (RMSD 2.82, slope 0.70)
    though PDS failed across instrument units (3.92, slope 0.38). Give ``groups`` (unit,
    session or batch) to also score leave-one-group-out; the board is then ranked by
    the grouped score.

    Args:
        X_primary: Primary spectra of the standards, (n, p), on a grid common with
            ``X_satellite``.
        X_satellite: Satellite spectra of the same standards, same row order (see
            ``pair_standards_by_id``).
        y: Reference values of the standards (optional). Enables RMSEP, bias, SEP
            and slope, and the prediction corrections.
        predict: Model prediction function (optional); see
            ``predict_fn_from_model``. Without it only spectral RMSE is reported.
        candidates: Candidates to compare (default ``default_transfer_candidates``).
            "No correction" is added when missing.
        ids: Standard IDs, stored as ``TransferEvaluation.ids`` in the row order of
            the out-of-fold arrays (default 0..n-1).
        y_range_reference: (min, max) of the model's calibration y. A warning is
            raised when the standards span much less of it.
        rank_by: Leaderboard sort column. ``'auto'`` ranks by the grouped
            ``RMSD_vs_primary`` with groups, else ``RMSD_vs_primary`` with a model,
            else (grouped) ``spectral_RMSE``. ``RMSD_vs_primary`` is the transfer's own
            error in y units: RMSEP also carries the model's error on each standard,
            and on a few standards a transfer error can cancel it by chance.
            ``'RMSEP'``, ``'RMSD_vs_primary'``, ``'spectral_RMSE'`` and their
            ``_grouped`` forms are accepted.
        groups: Group of each standard (instrument unit, session, batch).
        wavelengths: The common grid. With ``space`` it enables the physical-match
            candidates and the noise and join diagnostics.
        space: ``'reflectance'`` (0-1) or ``'absorbance'``.
        joins: Detector joins: ``'auto'`` (detected inside each fit from its own
            standards), ``None`` (one block), or wavelengths from a header or the user.
        model_wavelengths: Wavelengths the model reads; a warning names any that fall
            where the satellite is much noisier than the primary.

    Returns:
        A ``TransferEvaluation``.

    Raises:
        ValueError: Fewer than 3 standards, mismatched shapes, non-finite spectra or
            y, ``y`` without ``predict``, ``groups`` of the wrong length, physical
            candidates without wavelengths and space, or a ``rank_by`` the inputs
            cannot score.
    """
    Xp = np.asarray(X_primary, dtype=np.float64)
    Xs = np.asarray(X_satellite, dtype=np.float64)
    if Xp.ndim != 2 or Xp.shape != Xs.shape:
        raise ValueError(
            f"X_primary and X_satellite must be 2-D with the same shape, got "
            f"{Xp.shape} and {Xs.shape}"
        )
    n, p = Xs.shape
    if n < MIN_STANDARDS:
        raise ValueError(
            f"Leave-one-standard-out needs at least {MIN_STANDARDS} paired standards, got {n}"
        )
    if not (np.isfinite(Xp).all() and np.isfinite(Xs).all()):
        raise ValueError("Standard spectra contain NaN or infinite values")
    y_arr = None
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.shape[0] != n:
            raise ValueError(f"y has {y_arr.shape[0]} values for {n} standards")
        if not np.isfinite(y_arr).all():
            raise ValueError("y contains NaN or infinite values")
        if predict is None:
            raise ValueError("y needs a predict function: RMSEP is a prediction error")
    if ids is not None and len(ids) != n:
        raise ValueError(f"ids has {len(ids)} entries for {n} standards")
    wl = None
    if wavelengths is not None:
        wl = sd.check_wavelengths(wavelengths, p)
    if space is not None:
        sd.check_space(space)
    if candidates is not None and any(c.method == "physical" for c in candidates):
        if wl is None or space is None:
            raise ValueError("Physical-match candidates need wavelengths and space")

    warnings_out: list[str] = []
    g = _check_groups(groups, n, warnings_out)

    if candidates is None:
        candidates = default_transfer_candidates(
            n,
            has_y=y_arr is not None,
            has_predict=predict is not None,
            n_wavelengths=p,
            physical=wl is not None and space is not None,
        )
    cand_list = list(dict.fromkeys(candidates))
    if not any(c.method == "none" for c in cand_list):
        cand_list.insert(0, TransferCandidate.make("none"))

    if n < 5:
        warnings_out.append(
            f"Only {n} standards: each leave-one-out score rests on {n} predictions, "
            "so small differences between methods are not meaningful."
        )
    if y_arr is not None and y_range_reference is not None:
        lo, hi = (float(v) for v in y_range_reference)
        if hi > lo and (y_arr.max() - y_arr.min()) < 0.3 * (hi - lo):
            warnings_out.append(
                "The standards span less than 30% of the calibration y range; a slope "
                "fitted on them extrapolates."
            )

    diagnostics: dict[str, Any] = {}
    if wl is not None and space is not None:
        diagnostics = _diagnostics(Xp, Xs, wl, space, joins, model_wavelengths, warnings_out)

    y_primary = y_sat = None
    if predict is not None:
        y_primary = _check_predictions(predict(Xp), n)
        y_sat = _check_predictions(predict(Xs), n)

    fit_kw = {"wavelengths": wl, "space": space, "joins": joins}
    loso_folds = [(np.flatnonzero(np.arange(n) != i), np.array([i])) for i in range(n)]
    group_folds = None
    if g is not None:
        group_folds = [
            (np.flatnonzero(g != lab), np.flatnonzero(g == lab)) for lab in dict.fromkeys(g)
        ]

    rows: list[dict[str, Any]] = []
    oof_pred: dict[str, np.ndarray] = {}
    oof_pred_g: dict[str, np.ndarray] = {}
    oof_res: dict[str, np.ndarray] = {}
    by_label: dict[str, TransferCandidate] = {}

    for cand in cand_list:
        label = cand.label
        if label in by_label:
            # Labels round settings (%g); keep a distinct candidate distinguishable.
            label = f"{label} [{', '.join(f'{k}={v!r}' for k, v in cand.params)}]"
            if label in by_label:
                continue
        by_label[label] = cand
        row: dict[str, Any] = {
            "label": label,
            "method": cand.method,
            "kind": cand.kind,
            "settings": cand.settings,
            "n_standards": n,
            "error": "",
        }
        if cand.kind == "prediction" and (y_arr is None or predict is None):
            row["error"] = "needs y and a model"
            rows.append(row)
            continue
        if cand.min_fit_standards > n - 1:
            row["error"] = f"needs {cand.min_fit_standards} standards per fit"
            rows.append(row)
            continue
        try:
            pred, Z = _out_of_fold(cand, Xp, Xs, y_arr, predict, y_sat, loso_folds, fit_kw)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Transfer candidate %s failed: %s", label, exc)
            row["error"] = str(exc)
            rows.append(row)
            continue
        if Z is not None:
            oof_res[label] = Xp - Z
            row["spectral_RMSE"] = float(np.mean(np.sqrt(np.mean((Z - Xp) ** 2, axis=1))))
        if pred is not None:
            oof_pred[label] = pred
            row["RMSD_vs_primary"] = _rmsd(pred, y_primary)
            row["Slope_vs_primary"] = _slope(y_primary, pred)
            if y_arr is not None:
                row.update(_fom_columns(y_arr, pred))
        if group_folds is not None:
            row["error_grouped"] = _grouped_scores(
                row,
                cand,
                Xp,
                Xs,
                y_arr,
                predict,
                y_sat,
                y_primary,
                group_folds,
                fit_kw,
                oof_pred_g,
            )
        rows.append(row)

    score_column = _resolve_rank_by(
        rank_by,
        has_y=y_arr is not None,
        has_predict=predict is not None,
        has_groups=group_folds is not None,
    )

    board = pd.DataFrame(rows)
    for col in _SCORE_COLUMNS:
        if col not in board:
            board[col] = np.nan
    if group_folds is not None:
        if "error_grouped" in board:
            board["error_grouped"] = board["error_grouped"].fillna("")
        else:
            board["error_grouped"] = ""
    none_score = float(board.loc[board["method"] == "none", score_column].iloc[0])
    if np.isfinite(none_score) and none_score > 0:
        board["improvement"] = 1.0 - board[score_column] / none_score
    else:
        board["improvement"] = np.nan
    board = board.sort_values(score_column, kind="mergesort", na_position="last")
    board = board.reset_index(drop=True)

    ranked = board[board[score_column].notna()]
    n_scored = int(len(ranked))
    if not ranked.empty and ranked.iloc[0]["method"] == "none":
        warnings_out.append("No method beat No correction on these standards.")
    if group_folds is not None and predict is not None:
        warnings_out.extend(_grouped_warnings(board, y_primary))

    if y_arr is not None:
        ref = {
            "label": REFERENCE_LABEL,
            "method": "reference",
            "kind": "reference",
            "settings": {},
            "n_standards": n,
            "error": "",
            "RMSD_vs_primary": 0.0,
            **_fom_columns(y_arr, y_primary),
        }
        oof_pred[REFERENCE_LABEL] = y_primary
        board = pd.concat([board, pd.DataFrame([ref])], ignore_index=True)

    return TransferEvaluation(
        leaderboard=board,
        score_column=score_column,
        n_standards=n,
        n_candidates=n_scored,
        ids=list(ids) if ids is not None else list(range(n)),
        oof_predictions=oof_pred,
        oof_residual_spectra=oof_res,
        candidates=by_label,
        warnings=warnings_out,
        oof_predictions_grouped=oof_pred_g,
        groups=list(g) if g is not None else None,
        diagnostics=diagnostics,
    )


_SCORE_COLUMNS = (
    "RMSEP",
    "Bias",
    "SEP",
    "Slope",
    "Intercept",
    "RMSD_vs_primary",
    "Slope_vs_primary",
    "spectral_RMSE",
    "RMSD_vs_primary_grouped",
    "Slope_vs_primary_grouped",
    "RMSEP_grouped",
    "spectral_RMSE_grouped",
)


def _grouped_scores(
    row: dict[str, Any],
    cand: TransferCandidate,
    Xp: np.ndarray,
    Xs: np.ndarray,
    y_arr: np.ndarray | None,
    predict: PredictFn | None,
    y_sat: np.ndarray | None,
    y_primary: np.ndarray | None,
    group_folds: list[tuple[np.ndarray, np.ndarray]],
    fit_kw: dict[str, Any],
    oof_pred_g: dict[str, np.ndarray],
) -> str:
    """Fill the grouped columns of ``row``; return an error text ('' when scored)."""
    smallest_fit = min(tr.size for tr, _ in group_folds)
    if cand.min_fit_standards > smallest_fit:
        return f"needs {cand.min_fit_standards} standards per fit"
    try:
        pred_g, Z_g = _out_of_fold(cand, Xp, Xs, y_arr, predict, y_sat, group_folds, fit_kw)
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.warning("Transfer candidate %s failed (grouped): %s", row["label"], exc)
        return str(exc)
    if Z_g is not None:
        row["spectral_RMSE_grouped"] = float(np.mean(np.sqrt(np.mean((Z_g - Xp) ** 2, axis=1))))
    if pred_g is not None:
        oof_pred_g[row["label"]] = pred_g
        row["RMSD_vs_primary_grouped"] = _rmsd(pred_g, y_primary)
        row["Slope_vs_primary_grouped"] = _slope(y_primary, pred_g)
        if y_arr is not None:
            row["RMSEP_grouped"] = _rmsd(pred_g, y_arr)
    return ""


def _grouped_warnings(board: pd.DataFrame, y_primary: np.ndarray) -> list[str]:
    out: list[str] = []
    ok = board[board["RMSD_vs_primary"].notna() & board["RMSD_vs_primary_grouped"].notna()]
    if ok.empty:
        return out
    best_loso = ok.loc[ok["RMSD_vs_primary"].idxmin(), "label"]
    best_grp = ok.loc[ok["RMSD_vs_primary_grouped"].idxmin(), "label"]
    if best_loso != best_grp:
        out.append(
            f"Leave-one-out favours {best_loso}, but across groups {best_grp} transfers "
            "better: the leave-one-out score shares each session's effects between fit and "
            "test."
        )
    spread = float(np.ptp(y_primary)) if y_primary.size else 0.0
    worse = []
    for _, r in ok.iterrows():
        if r["method"] == "none":
            continue
        a, b = float(r["RMSD_vs_primary"]), float(r["RMSD_vs_primary_grouped"])
        if b > 1.25 * a and (b - a) > 0.1 * spread:
            worse.append(f"{r['label']} ({b:.3g} vs {a:.3g})")
    if worse:
        shown = "; ".join(worse[:4]) + (f"; and {len(worse) - 4} more" if len(worse) > 4 else "")
        out.append(
            "Much worse across groups than leave-one-out (they learn the session, or have "
            f"too few standards per fit): {shown}."
        )
    return out


def _diagnostics(
    Xp: np.ndarray,
    Xs: np.ndarray,
    wl: np.ndarray,
    space: str,
    joins: JoinsArg,
    model_wavelengths: Sequence[float] | None,
    warnings_out: list[str],
) -> dict[str, Any]:
    """Report-only diagnostics over all standards (no fit uses them)."""
    fold_joins, source = _fold_joins(Xp, Xs, wl, space, joins)
    comp = sd.noise_comparison(Xp, Xs, wl, space=space, joins=fold_joins)
    out: dict[str, Any] = {
        "joins": list(fold_joins),
        "joins_source": source,
        "primary_noisy_regions": comp.primary_noisy,
        "satellite_noisy_regions": comp.satellite_noisy,
        "satellite_much_noisier": comp.regions,
        "noise_ratio": comp.ratio,
    }
    if model_wavelengths is not None:
        mw = np.asarray(model_wavelengths, dtype=np.float64)
        on_model = np.isin(wl, mw) if np.isin(mw, wl).all() else (wl >= mw.min()) & (wl <= mw.max())
        noisy = sd.regions_mask(comp.satellite_noisy, wl) & on_model
        if noisy.any():
            span = ", ".join(
                f"{lo:g}-{hi:g}"
                for lo, hi in comp.satellite_noisy
                if np.any((mw >= lo) & (mw <= hi))
            )
            warnings_out.append(
                f"The model reads {span} nm, where the satellite is far noisier than the "
                "rest of its detector. No transfer removes noise."
            )
        much = sd.regions_mask(comp.regions, wl) & on_model
        if much.any():
            factor = float(np.median(comp.ratio[much]))
            warnings_out.append(
                f"At the wavelengths the model reads, the satellite is ~{factor:.0f}x noisier "
                "than the primary. No transfer removes noise; its predictions will scatter "
                "more than the primary's."
            )
    return out


def _resolve_rank_by(
    rank_by: str, *, has_y: bool, has_predict: bool, has_groups: bool = False
) -> str:
    if rank_by == "auto":
        base = "RMSD_vs_primary" if has_predict else "spectral_RMSE"
        return f"{base}_grouped" if has_groups else base
    available = {
        "RMSEP": has_y and has_predict,
        "RMSD_vs_primary": has_predict,
        "spectral_RMSE": True,
        "RMSEP_grouped": has_y and has_predict and has_groups,
        "RMSD_vs_primary_grouped": has_predict and has_groups,
        "spectral_RMSE_grouped": has_groups,
    }
    if rank_by not in available:
        raise ValueError(f"rank_by must be 'auto' or one of {sorted(available)}, got {rank_by!r}")
    if not available[rank_by]:
        base = rank_by.removesuffix("_grouped")
        need = (
            "y and a model" if base == "RMSEP" else ("a model" if base != "spectral_RMSE" else "")
        )
        if rank_by.endswith("_grouped"):
            need = f"{need} and groups" if need else "groups"
        raise ValueError(f"rank_by={rank_by!r} needs {need}")
    return rank_by


def _fom_columns(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    fom = regression_figures_of_merit(y_true, y_pred, context="cv")
    return {
        "RMSEP": fom["RMSE"],
        "Bias": fom["Bias"],
        "SEP": fom["SEP"],
        "Slope": fom["Slope"],
        "Intercept": fom["Intercept"],
    }


def _out_of_fold(
    cand: TransferCandidate,
    Xp: np.ndarray,
    Xs: np.ndarray,
    y: np.ndarray | None,
    predict: PredictFn | None,
    y_sat: np.ndarray | None,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    fit_kw: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Out-of-fold predictions and transferred spectra for one candidate.

    Every fold is fitted on its training rows only; joins detected with ``'auto'``
    and the physical match's noise maps are computed from those rows too.
    """
    n = Xs.shape[0]
    if cand.kind == "none":
        return (y_sat.copy() if y_sat is not None else None), Xs.copy()

    if cand.kind == "prediction":
        pred = np.empty(n)
        for tr, te in folds:
            corr = fit_transfer(cand, Xp[tr], Xs[tr], y=y[tr], predict=predict)
            pred[te] = ct.apply_prediction_correction(y_sat[te], corr)
        return pred, None

    Z = np.empty_like(Xs)
    for tr, te in folds:
        tm = fit_transfer(cand, Xp[tr], Xs[tr], **fit_kw)
        Z[te] = ct.apply_transfer_dispatch(Xs[te], tm)
    if not np.isfinite(Z).all():
        raise ValueError("transfer produced non-finite spectra")
    pred = _check_predictions(predict(Z), n) if predict is not None else None
    return pred, Z


def predict_fn_from_model(model_dict: dict, wavelengths: Sequence[float]) -> PredictFn:
    """Wrap a loaded ``.dasp`` regression model as a ``predict`` function.

    The model is used as saved, including any bias correction stored in it, so
    prediction corrections fitted by ``evaluate_transfer`` are relative to the model's
    own output.

    Args:
        model_dict: From ``model_io.load_model``.
        wavelengths: Wavelengths of the common-grid columns the spectra will have.

    Returns:
        Function mapping (m, p) spectra to (m,) predictions.

    Raises:
        NotImplementedError: The model is not a regression model.
    """
    from .model_io import predict_with_model

    task = str((model_dict.get("metadata") or {}).get("task_type") or "regression").lower()
    if task == "regression" and (
        model_dict.get("label_encoder") is not None
        or getattr(model_dict.get("model"), "classes_", None) is not None
    ):
        # Files from before task_type was saved default to regression; a fitted
        # classifier still carries its classes.
        task = "classification"
    if task != "regression":
        raise NotImplementedError(
            f"Transfer evaluation by prediction supports regression models; this model "
            f"is {task!r}. Use evaluate_transfer without predict for spectral RMSE."
        )
    cols = [float(w) for w in wavelengths]

    def _predict(Z: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(Z, dtype=np.float64), columns=cols)
        return np.asarray(predict_with_model(model_dict, frame), dtype=np.float64).ravel()

    return _predict


@dataclass(frozen=True)
class PairedStandards:
    """Standards matched by sample ID across two instruments.

    Attributes:
        X_primary: (n, p) primary spectra, in primary-file order.
        X_satellite: (n, p) satellite spectra, same specimens and order.
        ids: Primary IDs of the matched specimens.
        satellite_ids: The matching satellite IDs.
        wavelengths: The primary wavelengths (column order of both arrays).
        unmatched_primary: Primary IDs with no satellite match.
        unmatched_satellite: Satellite IDs with no primary match.
    """

    X_primary: np.ndarray
    X_satellite: np.ndarray
    ids: list
    satellite_ids: list
    wavelengths: np.ndarray
    unmatched_primary: list
    unmatched_satellite: list


def _normalised_index(df: pd.DataFrame, which: str) -> dict[str, Any]:
    from .io import _normalize_filename_for_matching

    out: dict[str, Any] = {}
    dupes: dict[str, list] = {}
    for raw in df.index:
        key = _normalize_filename_for_matching(raw)
        if key in out:
            dupes.setdefault(key, [out[key]]).append(raw)
        else:
            out[key] = raw
    if dupes:
        shown = "; ".join(f"{', '.join(map(str, v))}" for v in list(dupes.values())[:5])
        raise ValueError(f"{which} spectra have IDs that match each other: {shown}")
    return out


def pair_standards_by_id(primary: pd.DataFrame, satellite: pd.DataFrame) -> PairedStandards:
    """Pair standards measured on two instruments by sample ID, not row order.

    IDs are compared as ``io.align_xy`` compares them: file extensions, spaces and
    case are ignored.

    Args:
        primary: Primary spectra; index = sample IDs, columns = wavelengths.
        satellite: Satellite spectra in the same layout, any row order.

    Returns:
        A ``PairedStandards``.

    Raises:
        ValueError: Duplicate IDs, fewer than 2 matches, or wavelength columns that
            are not the same grid (resample both to one grid first).
    """
    from .wavelength_matching import WavelengthMatchError, match_wavelengths

    wl_p = np.asarray(primary.columns, dtype=np.float64)
    wl_s = np.asarray(satellite.columns, dtype=np.float64)
    if wl_p.shape != wl_s.shape:
        raise ValueError(
            f"Primary has {wl_p.size} wavelengths and satellite {wl_s.size}; "
            "put both on one common grid before pairing"
        )
    try:
        cols = match_wavelengths(wl_p, wl_s)
    except WavelengthMatchError as exc:
        raise ValueError(
            f"Primary and satellite wavelength grids differ ({exc}); "
            "put both on one common grid before pairing"
        ) from exc

    p_keys = _normalised_index(primary, "Primary")
    s_keys = _normalised_index(satellite, "Satellite")
    matched = [k for k in p_keys if k in s_keys]
    if len(matched) < 2:
        raise ValueError(
            f"Only {len(matched)} sample ID(s) appear on both instruments; need at least 2"
        )
    ids = [p_keys[k] for k in matched]
    sat_ids = [s_keys[k] for k in matched]
    Xp = primary.loc[ids].to_numpy(dtype=np.float64)
    Xs = satellite.loc[sat_ids].to_numpy(dtype=np.float64)[:, cols]
    return PairedStandards(
        X_primary=Xp,
        X_satellite=Xs,
        ids=ids,
        satellite_ids=sat_ids,
        wavelengths=wl_p,
        unmatched_primary=[v for k, v in p_keys.items() if k not in s_keys],
        unmatched_satellite=[v for k, v in s_keys.items() if k not in p_keys],
    )
