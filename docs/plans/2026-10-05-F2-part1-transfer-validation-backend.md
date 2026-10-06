# F2 part 1: calibration transfer that validates itself (backend)

Branch `feat/ct-evaluate-transfer`. Roadmap: `docs/reviews/2026-09-28-improvement-roadmap.md`
F2 items 1-2 (backend half), CT1, CT2, CT3. GUI work (validate/compare panel, ID pairing in
the tab, "Plan transfer standards") is part 2; `.dasp` per-instrument adapters are part 3.

> **As built (after GLM 5.3 + DeepSeek plan review, 2026-10-05).** Differences from the text below:
> - The evaluation API lives in a new module `transfer_evaluation`, which has its own surface row. The estimators
>   are in `calibration_transfer`.
> - Centred PDS stores `B_centred` + `offset`, never `B`, so pre-2026-10 builds raise a KeyError instead of
>   silently dropping the offset. `apply_pds_centred` and `apply_ds_dual` are the apply functions.
> - The board is ranked by `RMSD_vs_primary` whenever a model is given (`rank_by=` overrides). RMSEP includes model
>   error that a transfer can cancel by chance on few standards; see SESSION_LOG 2026-10-05.
> - The rank is capped per window, and a numerical-rank cut on the data-set scale guards constant windows. The
>   resolved integer rank is stored.
> - `estimate_prediction_correction` returns no fit metrics.
> - Part 3 must decide whether a satellite slope/bias replaces or chains with a correction already stored in the
>   `.dasp`. It is fitted on the model's output with that correction applied.

## Goal

A script, and later the GUI, can answer the question "did the transfer work, and which method
should I use?" with prediction error on held-out standards in y units. A "no correction" row
is always shown. Leave-one-standard-out (LOSO) is the validation: refit the transfer on the
other n-1 standards, transform the left-out satellite spectrum, and predict it.

## 1. New and fixed methods (calibration_transfer.py)

Existing functions keep their signatures and outputs. Saved transfer files from before this PR
load and apply unchanged.

**CT1: centred low-rank PDS.** `estimate_pds_lowrank(X_primary, X_satellite, window=11,
rank=None, center=True) -> dict`. For wavelength i, the satellite window and the primary
channel are mean-centred over the standards. b is fitted by truncated SVD of the centred window
(rank capped at `min(n-1, window)`; `None` = that cap). It stores
`offset[i] = mean_p[i] - mean_s[window] @ b`. It returns params
`{'B', 'window', 'offset', 'rank', 'centred': True}` (method key stays `'pds'`).
`apply_pds(X, B, window=None, offset=None)` adds the offset when one is given. Edge windows are
truncated as today. Wang, Veltkamp & Kowalski 1991 *Anal Chem* 63(23):2750-2756.

**CT2: dual-form centred ridge DS.** `estimate_ds_dual(X_primary, X_satellite, lam_rel=1e-2,
center=True) -> dict`. Xc and Pc are the centred standards (n×p). It computes
`W = (Xc Xcᵀ + λ I)⁻¹ Pc` with `λ = lam_rel · trace(Xc Xcᵀ)/n`, so lam_rel is scale-free. It
stores `basis = Xc` and `W`, both n×p, rather than the p×p A. That is milliseconds and KB,
against 1.5 s and 37 MB today. Apply: `mean_p + ((X - mean_s) @ basisᵀ) @ W`. Params are
`{'ds_form': 'dual', 'basis', 'W', 'mean_satellite', 'mean_primary', 'lam', 'lam_rel'}`
(method key stays `'ds'`). Mathematically this equals primal ridge DS on centred data with the
same λ, and a test asserts that.

**CT3: prediction slope/bias from satellite standards.**
`estimate_prediction_correction(y_ref, y_pred_satellite, fit_slope=True) -> dict`. It returns
a `bias_correction`-format dict (`method='linear'`, `bias`, `slope`, so
`bias_correction.apply_correction` and `model_io` already apply it). Bias-only fixes slope = 1.
It adds `source='satellite_standards'`, `n_standards`, and `warnings` (fewer than 3 standards;
a y range narrower than `y_range_reference`, when given). This PR does not write it into a
`.dasp`; that is part 3. Bouveresse et al. 1996 *Anal Chem* 68(6):982-990.

**One apply path.** `apply_transfer_dispatch` learns the new param shapes: DS `ds_form` dual vs
legacy `A`, and the PDS `offset` (absent = 0). The three duplicate apply sites route through it
so a new-style transfer file works everywhere: GUI `_plot_transfer_quality` (~:49605), GUI
`_apply_transfer_model` (~:51772) and `equalization.py:84-88`. `meta['format_version'] = 2` is
written for the new forms. Older dasp builds cannot apply a new-form file and fail with a
KeyError. Forward compatibility is not promised.

## 2. evaluate_transfer (LOSO bake-off)

```python
@dataclass(frozen=True)
class TransferCandidate:
    method: str          # 'none' | 'tsr' | 'tsr_bias' | 'pds' | 'ds' | 'pred_slope_bias' | 'pred_bias'
                         # (+ 'pds_legacy', 'ds_legacy' for comparing with existing settings)
    params: tuple        # sorted (key, value) pairs, hashable
    kind: str            # 'none' | 'spectral' | 'prediction'
    label: str           # 'PDS (centred, w=11, rank 2)'

def default_transfer_candidates(n_standards, *, has_y, has_predict) -> list[TransferCandidate]
def evaluate_transfer(X_primary, X_satellite, *, y=None, predict=None, candidates=None,
                      ids=None, y_range_reference=None) -> TransferEvaluation
def fit_transfer(candidate, X_primary, X_satellite, *, y=None, predict=None,
                 wavelengths=None) -> TransferModel | dict
```

- **Inputs.** Paired standards on one common grid; row i is the same specimen on both
  instruments. `predict` is any callable mapping an (m, p) array to 1-D numeric predictions.
  `predict_fn_from_model(model_dict, wavelengths)` builds one from a loaded `.dasp` through
  `model_io.predict_with_model`. It is regression only in this PR and raises
  `NotImplementedError` for other task types. Spectral metrics still work for any model.
- **Per candidate, per left-out standard i:** fit on the other n-1, transform satellite i, and
  record the transferred spectrum, its residual against primary i, and predict(transferred i).
  Prediction-kind candidates fit the slope/bias on the other standards' (y_j, ŷ_sat_j), which
  needs y.
- **Rows always present:** `No correction` (predict the raw satellite spectra). When y and
  predict are both given, also `Primary instrument (reference)`, which predicts the primary
  spectra. It is the floor a transfer can approach and is not ranked.
- **Leaderboard columns** (DataFrame): label, method, kind, n_standards; RMSEP, Bias, SEP,
  Slope, Intercept from `scoring.regression_figures_of_merit(context="cv")`, the out-of-fold
  convention, against y; `RMSD_vs_primary` (ŷ transferred vs ŷ of the primary spectrum, no y
  needed); `spectral_RMSE` (mean over standards of the RMS over wavelengths); `improvement` =
  1 - RMSEP/RMSEP_none.
- **Sort key:** RMSEP if y is given, else RMSD_vs_primary if predict is given, else
  spectral_RMSE. `TransferEvaluation` carries `leaderboard`, `score_column`, the per-candidate
  out-of-fold arrays (predictions and residual spectra, for the part 2 plots), `warnings` and
  `n_candidates`.
- **Default grid** (only candidates every LOSO fold can fit, with n_train = n-1):
  - TSR (slope/bias per wavelength) and TSR bias-only.
  - Centred PDS: window ∈ {5, 11, 21, 31}, rank ∈ {1 … min(n-2, 4)}.
  - Dual DS: lam_rel ∈ {1e-3, 1e-2, 1e-1, 1}.
  - Prediction slope/bias (n ≥ 4) and prediction bias-only (n ≥ 2), when y is given.
  - About 25 candidates in all. At 2151 bands with 12 standards that is about 12 fits each,
    so seconds.
- **n < 3** raises ValueError: LOSO on two standards fits on one.

**Selection optimism.** Reporting the best of about 25 LOSO scores is mildly optimistic, as with
picking the PLS LV count from CV (accepted chemometrics practice; see CLAUDE.md validation
conventions). No test fold's y is used to fit that fold's transfer. The leaderboard records
`n_candidates`. Nested LOSO for an "auto" choice is out of scope; part 2 can add it if wanted.

## 3. Pairing by sample ID

`pair_standards_by_id(primary_df, satellite_df) -> PairedStandards(X_primary, X_satellite, ids,
unmatched_primary, unmatched_satellite)`. Each DataFrame has rows = specimens (index = IDs) and
wavelength columns. IDs are normalised as `io.align_xy` does (extension, spaces, case). It
raises ValueError on duplicate normalised IDs, on fewer than 2 matches, or when the wavelength
columns differ (match within `wavelength_matching` tolerance; different grids are CT5, not
here). The GUI tab adopts it in part 2.

## 4. Public surface, docs, tests

- **Public surface.** Add a `calibration_transfer` row to AGENT_COMPOSITION's declared surface:
  `evaluate_transfer`, `default_transfer_candidates`, `fit_transfer`, `TransferCandidate`,
  `TransferEvaluation`, `pair_standards_by_id`, `predict_fn_from_model`,
  `estimate_pds_lowrank`, `estimate_ds_dual`, `estimate_prediction_correction`,
  `apply_transfer_dispatch`, `save_transfer_model`, `load_transfer_model`. Add a short worked
  section run against `example/`, and extend the contract test.
- **Tests** (`tests/test_ct_evaluate_transfer.py`):
  - Dual DS equals primal centred ridge.
  - Centred PDS recovers an exact local affine map, and the offset is applied.
  - Legacy B-only and A-only files behave as before.
  - Save/load round trip and dispatch for the new forms.
  - LOSO hand-check on a tiny prediction slope/bias case.
  - n<3 and mismatched shapes raise.
  - Pairing: shuffled order, an extension or case difference, unmatched lists, duplicates.
  - Simulated second instrument built from the 49 example ASD spectra, re-creating the
    2026-09-28 spot-check (shift, blur, gain tilt, curved baseline, noise; PLS on the primary).
    The ranking is asserted, not the exact values: the best transfer beats No correction;
    centred PDS beats legacy PDS at 8+ standards; dual DS at λ_rel = 1e-2 beats No correction.
    The fast variant runs on every-4-nm bands; it is not marked slow.
  - The GUI apply sites call `apply_transfer_dispatch` (behavioural test of
    `_apply_transfer_model` with a dual-DS model on a stub).
- **Not in this PR:** the GUI panel, `.dasp` adapters, TOP/EPO and calibration augmentation
  (CT4, they need refits), classification/one-class agreement metrics, grid intersection (CT5),
  deleting the dead `_build_ct_transfer_model` (part 2).
