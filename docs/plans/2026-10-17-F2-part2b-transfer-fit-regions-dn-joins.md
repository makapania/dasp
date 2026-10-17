# F2 part 2b: physical-match fit regions and joins from raw readings (follow-up, not started)

Comes after F2 part 2 (the CT GUI) merges. Two general options found on SPYDER Bone's second real instrument
pair. Neither is bone-specific: water state at scan time varies for soils, grain and plant material as well.

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
- Pairs whose instruments have different joins (Trek 1785, FS3 1800): decide whether blocks follow the satellite's
  joins (the blur must not cross the satellite's joins) or the union of both. Test both on Beasley.
- Acceptance: Trek files give 1000/1785 from `read_asd_dir`; LabSpec and FieldSpec 3 files are unchanged.

## Not planned

No bone or collagen mode. dasp scores each transfer through the user's own model, so the model-specific part is
already handled by the board.
