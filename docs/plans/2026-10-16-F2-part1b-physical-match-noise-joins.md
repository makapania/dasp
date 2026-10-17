# F2 part 1b: physical-match transfer, noise map, detector joins, grouped validation (backend)

Branch `feat/ct-physical-match`. Follows F2 part 1 (PR #97, `transfer_evaluation`). GUI work stays in part 2,
which will then show these candidates, the noise map and the joins from the start. Evidence: SESSION_LOG
2026-10-16 (SPYDER Bone's 40 real instrument pairs) and SPYDER `planning/08_transfer.md`.

## Why

On 40 bones scanned on LabSpec 4 Hi-Res and Standard-Res (two independent unit pairs), dasp's part-1 bake-off
picks centred PDS (LOSO M2045 RMSD 2.82, slope 0.70). Fitted on one unit pair and applied to the other, PDS gives
3.92 with slope 0.38, and DS gives 4.6-5.7 with slope about 0. LOSO shares session and presentation effects between
the fit and the scored standard, so it rewards methods that learn them.

The instrument difference is mostly a band-depth gain (hi-res absorbance features ~1.6x deeper) plus a little
blur. A two- or three-parameter "physical match" per detector block, learned from the paired spectra alone (no
datasheet), gives cross-unit M2045 2.99 / slope 0.88, water-removed 3.18, M1500 3.27. It is fitted on the MEAN
second-derivative spectrum over the standards. Pooled per-spectrum least squares lets satellite noise bias the gain
low (errors in variables) and push the blur to the top of its grid.

Noise rises 15-50x from 2045 to 2450 nm on every instrument seen, is 5-30x higher on hi-res, and changes ~4x between
sessions on one unit. Detector joins show as a level step (1000/1800 nm found from reflectance alone on 7 of 8
sets; nothing clear on a Trek handheld, whose crossover is invisible in reflectance).

## 1. Physical-match transfer (calibration_transfer.py)

`estimate_physical_match(X_primary, X_satellite, wavelengths, *, space, joins=None, sigma="learn",
fwhm=None, gain="per_block", shift=False, noise_weights=None, exclude=None) -> dict`

- **Model, per block b** (blocks = the spectrum split at `joins`; no joins = one block):
  satellite → blur with a Gaussian of width `sigma_b` nm → optional shift `delta_b` nm (linear interpolation) →
  in absorbance `A' = g_b · A + c_b` → back to the input space.
- **Space.** `space='reflectance'` converts R → A = log10(1/max(R, 1e-6)) for the gain and back; the blur is applied
  in reflectance (resolution acts on intensity). `space='absorbance'` blurs and scales A directly.
  `space='linear'` (already-derivative or otherwise preprocessed data) applies `g_b · X` with no offset. Stored in
  params; apply refuses a mismatched space.
- **Blur.** Gaussian in nm built from the actual wavelengths (non-uniform grids allowed), taps within
  `ceil(4σ)` nm, never crossing a join; at block edges the truncated kernel is renormalised over in-block taps.
  σ = 0 is identity.
- **σ choice.** `sigma="learn"`: per block, grid 0-10 nm in 0.5 nm steps (scaled to the grid spacing), each with its
  closed-form gain; pick the lowest weighted residual on the mean D2 (below). `sigma=float` or per-block tuple:
  fixed. `fwhm=(primary_fwhm_per_block, satellite_fwhm_per_block)`: σ = sqrt(max(Fp² − Fs², 0))/2.3548.
- **Direction.** Also fit "blur the primary instead". If that fits clearly better (the satellite is the
  lower-resolution instrument), a transfer cannot sharpen it: use σ = 0, keep the gain, and record a
  `satellite_coarser` warning that suggests refitting the model on blurred primary spectra (roadmap CT5).
- **Gain fit, noise-robust.** D2 = Savitzky-Golay 2nd derivative, window ~31 nm (in channels from the grid),
  cubic. `g_b = Σ w·d̄_s·d̄_p / Σ w·d̄_s²` on the MEAN D2 over the standards (d̄_s after blur and shift), over
  fit wavelengths = block interior (margin = SG half-window + kernel radius) minus `exclude` and minus
  noise-flagged regions. `w` = 1/(noise² of the mean D2 at that wavelength), from §2. `gain="single"` pools
  every block that has features into one g.
- **Offset.** `c_b` = mean over standards and fit wavelengths of `A_p − g_b·A'_s` (derivative models are
  exactly invariant to it; it restores displayed levels).
- **Featureless guard.** A block whose primary mean D2 has weighted signal-to-noise below a threshold (default
  10) gets σ = 0 and the pooled gain of the blocks that have features (none → g = 1), with a
  `featureless_block` note. Bone VNIR gave gains of 0.05-0.45 without it.
- **Shift (optional).** `shift=True` adds a −3..3 nm grid (0.25 nm) jointly with σ. Off by default (SPYDER found
  none above ±1 nm; the joint grid triples the cost).
- Returns `{'form': 'physical_match', 'space', 'blocks': [(lo_nm, hi_nm)], 'sigma_nm', 'shift_nm', 'gain',
  'offset', 'fit_wavelength_mask', 'notes', 'format_version': 1}`. `apply_physical_match(X, params, wavelengths)`.
  `apply_transfer_dispatch` learns `method='physical_match'`.

## 2. Spectral diagnostics (new module `spectral_diagnostics.py`)

- `noise_profile(X, wavelengths, *, space) -> np.ndarray`. Per-wavelength noise in absorbance units: residual from
  an 11-point quadratic SG smooth, robust (1.4826·MAD) over spectra, smoothed over ~21 nm. Also
  `d2_noise_profile` (the same residual passed through the fit's SG2) for the gain weights. Computed from the
  scans in hand, never stored per instrument (noise is session-dependent).
- `noisy_regions(profile, wavelengths, joins, factor=10) -> list[(lo, hi)]`. Regions ≥ 10 nm where noise exceeds
  `factor` × the median of the quietest half of the same detector block. Block-relative because SWIR2 is
  inherently noisier than SWIR1: an instrument-wide reference flagged the whole of SWIR2 on clean units.
- `noise_comparison(primary, satellite, wavelengths)`: ratio profile and regions where the satellite is > 5x the
  primary.
- `detect_detector_joins(X, wavelengths, *, space) -> JoinDetection(joins, scores, confident, source)`. Level-step
  score per channel boundary: left and right 8-channel linear fits extrapolated to the boundary, |step| over the
  fits' residual noise, median over spectra. A boundary is a join when its score ≥ max(6, 3× the 95th percentile
  of all boundary scores), peaks ≥ 60 nm apart, at most 3. `confident=False` and no joins otherwise. Prototype
  scores: 4-180 at 1000/1800 on LabSpec, FieldSpec 3 and dasp example data vs background ≈ 3; Trek: no clear
  join.
- `resolve_joins(header_joins, X, wavelengths)`: header (when plausible: inside the grid, ≥ 100 nm from the ends)
  → detected (confident) → none. Records `source` ∈ {'header', 'detected', 'user', 'none'}. User override is a
  plain argument everywhere joins are taken.
- **ASD header joins.** `readers.asd_native.read_asd_header_joins(path) -> tuple[float, float] | None`: float32 at
  offsets 444/448 (modern as7/as8 and legacy v1 headers share them; zero = absent). `io.read_asd_dir` adds
  `metadata['detector_joins']` when every file agrees, else none plus a warning. No change to returned spectra.

## 3. Validation changes (transfer_evaluation.py)

- **Grouped score.** `evaluate_transfer(..., groups=None)`. With ≥ 2 groups, each candidate is also scored
  leave-one-group-out (fit on the other groups' standards, apply to this group's). Columns
  `RMSD_vs_primary_grouped`, `Slope_vs_primary_grouped` (+ `RMSEP_grouped` with y). When groups are given,
  `rank_by='auto'` ranks by the grouped RMSD; the LOSO columns stay. A warning names a candidate whose grouped
  RMSD is > 1.25x its LOSO RMSD ("learns this session, not the instrument"). Groups = unit, session or batch; a
  pair of units here.
- **Slope vs primary.** `Slope_vs_primary`: slope of out-of-fold satellite predictions on primary predictions.
  The board shows compression (PDS 0.70) that RMSD hides.
- **Candidates.** `TransferCandidate` methods `physical` (settings: `sigma` 'learn' | float, `gain` 'per_block'
  | 'single', `shift` bool). `default_transfer_candidates(..., wavelengths=..., space=...)` adds
  `physical(learn, per_block)`, `physical(learn, single)` and `physical(0, per_block)` (gain only) when
  wavelengths and space are given. `fit_transfer` and `evaluate_transfer` take `wavelengths`, `space`, `joins`.
- **Noise and join warnings** on the evaluation: noisy regions of each instrument, satellite/primary noise
  ratio, and, when `model_wavelengths` (the bands the model reads; optional argument) overlap a noisy or
  satellite-much-noisier region, "the model reads X-Y nm where the satellite is N× noisier; no transfer
  removes noise". Joins and their source are reported in `TransferEvaluation.diagnostics`.

## 4. Clean-up

- `instrument_profiles.estimate_smoothing_between_instruments` fits σ in channels with no gain and blurs across
  joins: it mistakes band depth for blur. Reimplement as a thin wrapper over the §1 σ/gain fit (returns σ in nm
  and the gain), keep the name and import working, drop "skeleton" from the module docstring for that function.
- Roadmap CT5: replace "use estimate_smoothing_between_instruments" with the physical match and the
  `satellite_coarser` path.
- AGENT_COMPOSITION §9: physical match, groups, diagnostics; new surface rows.

## 5. Tests

- Simulated pairs on the example spectra (`simulated_instruments` fixture extended): known per-block σ, gain,
  small shift, joins with steps, a noise ramp above 2300 nm. Recover σ within 1 nm, gain within 3%, shift within
  0.5 nm; fit unaffected by the noise ramp (LS on pooled spectra fails the same test: guards the mean-D2 choice).
- Featureless block → pooled gain + note. Satellite coarser → σ = 0 + warning. Kernel never crosses a join.
  Non-uniform grid. Absorbance/reflectance/linear spaces round-trip; mismatched space refused.
- Joins: header plausible/implausible; detection on a synthetic step (found) and on smooth spectra (none,
  `confident=False`); the example ASD files give 1000/1800.
- Noise: block-relative flags on a synthetic ramp; satellite-noisier warning fires on model wavelengths.
- Grouped evaluation: a candidate that fits a per-group artefact wins LOSO and loses grouped; ranking switches.
- Dispatch: a physical-match `TransferModel` saves, loads and applies through `apply_transfer_dispatch`.

## 6. Real-data benchmark (not committed data)

`tools/benchmarks/transfer_pairs_benchmark.py`: takes two spectra folders, a pair table and optional groups,
runs `evaluate_transfer` with groups and prints the board, noise regions and joins. The SPYDER run (paths
outside the repo, results summarised in SESSION_LOG, never committed) is the acceptance check:
- grouped (cross-unit) M2045 RMSD ≤ 3.1 with slope ≥ 0.85 for the default physical candidate;
- the grouped ranking puts a physical candidate above every DS/PDS;
- joins 1000/1800 detected on all LabSpec sets, none confident on Trek;
- noisy regions include ≥ ~2400 nm on every set and not the 2030-2060 nm M2045 window on standard-res.

## Not in this PR

GUI (part 2). `.dasp` per-instrument adapters (part 3). Calibration from satellite scans with reference values
but no primary partner (SPYDER DECISIONS 93-97). Splice correction. Instrument registries by serial.
