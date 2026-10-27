# F2 part 2b: physical-match fit regions and joins from raw readings

> **As built (2026-10-26, branch `feat/f2-part2b`).** Both options are in; nothing was tuned to a dataset.
> - Option 1: `estimate_physical_match(fit_regions=)` next to `exclude=`; `physical_match.WATER_BANDS_NM`
>   `((1380, 1500), (1880, 1990))` and `water_bands_covered(wl)`; `evaluate_transfer(exclude=, fit_regions=)`;
>   `TransferCandidate` settings `exclude` / `fit_regions` (tuples; `transfer_evaluation.regions_tuple`); the default
>   board adds "Physical match (learned blur, gain per block, water bands excluded)" when the grid spans a water
>   band completely. Regions are stored in `params['fit_regions']` / `params['exclude_regions']` ((R, 2) arrays,
>   saved in the NPZ) and `meta[...]` (JSON). CT Validate panel: "Fit regions" row (All usable channels /
>   Exclude water bands / Custom (fit only on) + nm ranges); the diagnostics box names the regions used.
> - Decision the plan left open: left-out regions are widened (and fit regions narrowed) by half the 31 nm
>   derivative window plus one channel, for `exclude` as well as `fit_regions`, so no fitted channel's derivative
>   reads excluded data. This also tightens the existing `exclude=` slightly (it used to mask only the listed
>   channels). Custom regions are an include list ("fit only on"). The widening reaches the largest in-block
>   step, never a gap at a join. A block the regions leave out entirely gets no correction (gain 1, no blur,
>   offset 0), also under `featureless='pooled'`; an empty `exclude` is a no-op.
> - Option 2: `readers.asd_native.read_asd_white_reference` (as7/as8 raw DN), `spectral_diagnostics.
>   detect_joins_from_white_reference` / `read_asd_dn_joins` (largest relative white-reference DN jump in the
>   950-1050 and 1700-1900 nm windows between two live readings, at least 3x the window median and at least 0.01), `resolve_joins(dn_joins=)`
>   (user > header > dn > detected), `union_joins`. `read_asd_dir` reports `metadata['detector_joins_source']`:
>   `'header'` (plausible header: each join inside its window), `'dn'`, or `'mixed'`; files must agree (header
>   exactly, DN within 2 nm) or nothing is recorded. The GUI's "ASD file header" option uses the union of the two
>   instruments' joins; `'auto'` already unioned. `read_asd_header_joins` no longer needs a NUL after as2-as8 magic
>   (real headers carry comment text there; only legacy `ASD` needs the NUL to tell it from an ASCII export).
> - Acceptance numbers are in SESSION_LOG 2026-10-26 (Beasley: oh1545 improves 1.70 to 1.27 but stays above No
>   correction 1.18; FS3 to Trek oh2045 loses most of its gain with the water bands out; LabSpec default rows
>   unchanged to the digit).

Comes after F2 part 2 (the CT GUI) merges. Two general options found on SPYDER Bone's second real instrument
pair. Neither is bone-specific: water state at scan time varies for soils, grain and plant material as well.

**Generality rule (user, 2026-10-25):** dasp gets general features only, never code for one instrument, one
sample type or one dataset. No instrument model names in code paths, no "Trek mode", no bone mode. The SPYDER
datasets below are evidence and acceptance checks, not design inputs: when a choice is open, decide it on
principle and use the real pairs to confirm it. Every behaviour also gets a synthetic test in `tests/`.

## Evidence (2026-10-16)

Beasley FieldSpec 3 <-> Trek, 16 clean scan pairs over 15 bones, scored leave-one-bone-out. Report:
`C:/Users/sponheim/git/SPYDER Bone/planning/work/transfer_dasp/REPORT_beasley_transfer.md` (data stay there).
SESSION_LOG 2026-10-16, "Second real pair".

- The physical match equalizes the two instruments: cons3 inter-instrument RMSD 1.40 -> 0.87 / 1.02. DS is worse
  than no correction.
- The collagen bands already nearly agree (Trek/FS3 mean-D2 ratio 1.05-1.16), but the water bands do not
  (1450: 1.89, 1930: 1.27). The whole-block gain is therefore mostly a water correction. It helps the 2045 models
  and hurts oh1545 (RMSD 1.18 -> 1.70); the board's "No method beat No correction" caught it.
- SPYDER's own per-detector correction measures gain only on the collagen bands. Its implied SWIR1 ratio, 0.83-0.86,
  is what dasp's collagen-band D2 ratios give too. dasp's whole-block SWIR1 gain is 0.795-0.82.
- The Trek as7 header stores joins 0/0, and Trek reflectance shows no step, because the sample/reference ratio
  cancels it. The raw DN (sample and white reference) show 1000/1785. SPYDER's reader
  (`crates/spyder-core/src/read/asd.rs`) already finds them: header joins inside the windows 950-1050 / 1700-1900 nm,
  else the largest clear white-reference DN jump in each window, else 1000/1800.

## Option 1: choose the regions the gain is fitted on

- Backend: `estimate_physical_match(exclude=...)` already exists. Add `fit_regions=` (include list, nm), so the
  gain and σ are fitted only there, and a named preset `WATER_BANDS_NM = ((1380, 1500), (1880, 1990))` for
  `exclude`. Check the exact edges on real data; the 31 nm D2 window needs the existing dilation.
- Board: add a candidate "Physical match (water bands excluded)" when wavelengths cover them, so the comparison
  shows whether it helps. Grouped scoring decides; nothing is switched on silently.
- GUI (CT Validate panel from part 2): a "Fit regions" control with "All usable channels" (default),
  "Exclude water bands" and "Custom" (nm ranges). Record the regions in the transfer model's meta.
- Acceptance: on the Beasley pairs, oh1545 no longer gets worse than no correction, and the 2045 models keep
  most of their gain. On SPYDER's 40 LabSpec pairs, results stay within noise of the current ones (water and
  collagen bands scaled together there).

## Option 2: detector joins from raw readings

- Reader: when an ASD file holds raw DN and a white reference (as7, as8), detect joins from the DN jump in each join
  window, as SPYDER's reader does, and report them as `metadata['detector_joins']` with source "dn". Order: user >
  header (when plausible) > DN > detected from reflectance.
- Pairs whose instruments have different joins (e.g. 1785 on one, 1800 on the other): blocks follow the union of
  both instruments' joins, because the blur and gain must not cross a join of either instrument. This is decided on
  principle; Beasley is a check that it does no harm, not a bake-off between rules.
- The join windows and fallbacks describe the ASD file format's detector layout and apply to every ASD file
  with raw DN; nothing branches on the instrument model.
- Acceptance: Trek files give 1000/1785 from `read_asd_dir`; LabSpec and FieldSpec 3 files are unchanged.

## Not planned

No bone or collagen mode. dasp scores each transfer through the user's own model, so the model-specific part is
already handled by the board.
