# T-31 Multi-Class SIMCA — UX Parity with Sibling Methods

**Date:** 2026-07-06
**Branch:** `feat/T31-multiclass-simca`
**Status:** Design approved, ready for implementation plan

## Problem

The shipped multi-class SIMCA (Phase D) works but does not *feel* like the
other task types. Three concrete divergences the user identified:

1. **Config lives on the Import page.** The whole multi-class control panel
   (`mc_hyperparams_frame`: alpha, n_components, min-class-n, variable-selection
   paths + top-N) is grid'd into the import `config_frame`. Every other method
   is configured in the Analysis Configuration subtabs — Model Config (4A) for
   hyperparameters, Variable Selection (4B) for varsel. One-class put a *couple*
   of hyperparameters at import; multi-class put its *entire* configuration
   there. (The per-class engine picker `mc_models_frame` is the one piece
   already correctly placed, in the Model Config card.)

2. **Variable selection runs a single size, not a sweep.** Standard methods
   expose the "Top-N Variable Counts" checkbox row (N=10/20/50/100/250/500/1000)
   and sweep every checked size. Multi-class has one `mc_varsel_n_select`
   spinbox, and `run_multiclass_simca_search` takes `variable_selection_n_select`
   as a scalar — so multiple varsel *methods* can be picked but each runs at
   exactly one size. This is the "why aren't multiple variable sizes being run"
   complaint.

3. **A parallel, inconsistent control surface.** In multi-class mode the code
   hides the shared Variable Selection card and substitutes its own mini-picker
   on import — two different UIs for "pick variable selection."

Additionally: alpha and n_components are single values per run (other methods,
and one-class SIMCA specifically, sweep them); most variable-selection methods
available elsewhere were never wired through for multi-class; and the results
schema does not report the swept dimensions clearly enough to compare rows.

## Guiding Principle

**Multi-class SIMCA's closest sibling is one-class SIMCA.** Mirror one-class's
configuration pattern — sweepable lists (preset checkboxes + custom field) in
the Analysis Configuration subtabs — while keeping multi-class's unique
decision-matrix / novelty identity. One-class's `_collect_simca_overrides`
already solves n_components and alpha sweeps correctly; reuse that template and
its `_parse_oc_n_components_list` parser rather than inventing a new scheme.

## Design

### A. Placement — move all config off the Import page
- Remove `mc_hyperparams_frame` from the import `config_frame`. Import keeps only
  the task-type radio (parallel to one-class keeping just its inlier picker).
- Multi-class hyperparameters live in **4A Model Config**; variable selection
  lives in **4B Variable Selection** — shown/hidden by task type exactly like
  one-class's cards (the `_on_task_type_changed` show/hide logic already has the
  `multiclass_simca` branch that swaps model panels; extend it to reveal the new
  subtab cards instead of the import panel).

### B. n_components — swept list (mirror one-class)
- Preset checkbox row + custom comma-list in Model Config, reusing
  `_parse_oc_n_components_list` (mixes integer component counts and variance
  fractions freely). Default: `0.99` checked only, so a default run costs the
  same as today.
- `per_class_cv` remains a **separate multi-class-only toggle** (its unique take;
  not part of the checkbox sweep, since mixing "auto" into a numeric grid axis
  is confusing).

### C. alpha — swept list (mirror one-class)
- Preset checkboxes (`0.01 / 0.05`) + custom field, replacing the single
  spinbox. Swept as a grid axis.

### D. Variable-count — reuse shared Top-N checkboxes
- Multi-class stops hiding the 4B card; it reuses the existing "Top-N Variable
  Counts" row so multiple sizes sweep.

### E. Variable-selection methods — full set, grouped and labeled
Multi-class has a **genuine categorical label**, so supervised methods correctly
disabled for one-class (which had only inlier/outlier) are valid here. The
backend `MultiClassClassModel` already has a **precomputed boolean-mask hook**
("the hook for the C search layer [to] wire ANY supervised method by computing
the mask externally"), and `compute_importances` / the standard varsel dispatch
already run on classification targets. So wiring these is mostly plumbing.

4B presents two groups for multi-class:
- **SIMCA-native / novelty-safe:** Wold modeling power, Wold balanced, Wold
  discriminating. Describe each class's own variance structure; do not erode the
  "none of the above" novelty detection.
- **Discrimination-based / novelty caveat:** importance, CARS, CARS-Tree, SPA,
  UVE, iPLS family (iPLS / forward / backward / MC-siPLS / MWPLS), GA, VCPA-IRIV,
  UVE-CARS hybrids. Computed on the real multi-class label. Sharpen accuracy
  among *known* classes but can weaken novelty detection; carry a short
  "discrimination-oriented — confirm novelty on a true external class" note.

Wired through the mask hook, computing masks via the standard varsel dispatch on
the multi-class label. Methods that cannot handle >2 classes (binary-only PLS
coefficient paths) get a **graceful skip with a logged note** rather than a
crash; PLS-coefficient methods handle >2 classes via dummy-encoding where the
existing classification path already does.

### F. Backend — `run_multiclass_simca_search` accepts lists
- `alpha`, `n_components`, `variable_selection_n_select` become **list-capable**.
- Grid expands to `preprocessing × engines × varsel_paths × sizes × alphas ×
  n_components`, each combination a leaderboard row.
- **Scalar-or-list normalization** at the top of the function keeps existing
  callers (tests, saved-model reload, `_run_selected_multiclass_result`,
  `_fit_and_save_multiclass_model`) byte-identical when a single value is passed:
  a scalar is wrapped to a one-element list, producing exactly the current grid.

### G. min-class-n
- Single spinbox in Model Config (a data-eligibility floor, not swept).

### H. Reporting — every swept dimension is visible and self-describing
With alpha, n_components, and size all sweeping, each row must be
self-describing or the "dig in and compare" workflow fails.
- **Add a stored `NComponents` column** to the multi-class schema in
  `scoring.create_results_dataframe`, emitted per-row by
  `run_multiclass_simca_search` (the actual per-row value: `0.95`, `5`, or
  `per_class_cv`). `Alpha` and `varsel_path` already exist in the schema.
- **Surface the swept knobs in the visible leaderboard**: multi-class
  `display_cols` gains `Alpha`, `NComponents`, `VarSelMethod` (from
  `varsel_path`), and `Engine` (from `engine_family`) alongside the existing
  `n_vars` — so the table alone tells you which row is the "simpler
  5-component, 50-variable" model vs. the algorithmic best.
- **Column help/tooltips** for the new columns, matching the existing
  `TOOLTIP_CONTENT` column-help system.
- **Decision-matrix view + reproduction script/notebook** gain a config header
  stating the exact row identity (engine, alpha, n_components, varsel method +
  size), so a saved/selected row is unambiguous.
- All additive columns; consumers that don't reference them are unaffected.

### I. Holdout Validation tab (4E) — wire multi-class in
Currently multi-class is not linked into the Validation tab. The splitter UI is
task-agnostic (it only picks indices), but
`compute_validation_metrics_for_top_models` (`search.py`) has branches only for
`regression` (RMSEP/R²pred) and `classification` (val_Accuracy) — there is **no
`multiclass_simca` branch**, so a holdout set produces no metrics. That is the
missing link.

- **Splitters:** Kennard-Stone (X-only), Random, and Stratified apply directly.
  **Exclude SPXY** for multi-class — its `d_SPXY = d_X + d_y` term is Euclidean
  on the target, undefined for a categorical class label (same reason it is the
  wrong choice for classification). Disable/hide the SPXY radio when task type is
  `multiclass_simca`.
- **Metrics:** add a `multiclass_simca` branch to
  `compute_validation_metrics_for_top_models` that refits the
  `MultiClassClassModel` on the calibration split and computes **decision-matrix
  metrics** on the holdout (per-class sensitivity/specificity, novelty rate,
  ambiguity, exact-set rate) via the same `multiclass_simca_metrics` backend the
  search uses — NOT RMSEP or plain accuracy. Emit them as `val_*` columns
  parallel to the existing validation columns.
- **Honesty label (required):** a same-known-class holdout validates known-class
  performance only. It does NOT test the "none of the above" novelty capability,
  because every held-out sample belongs to a trained class. The tab must state
  this plainly and point to LOCO / a true external class for novelty validation
  (mirroring the existing optimistic-proxy note).

## Scope

- **Multi-class only** this pass. One-class's import-page hyperparameters
  (`oc_hyperparams_frame`) are minor creep and one-class already has its rich
  sweepable config in the subtab via `_collect_simca_overrides`; relocating them
  is a clean follow-up, not part of this work.

## Scientific Rationale (why this matters here specifically)

Multi-class class-modeling is the right tool for contaminant *identification*
(not quantification): its per-class, non-exclusive decision matrix lets a
specimen be accepted by several class models at once (multiple contaminants) or
none (novel/unknown) — capabilities a single-label discriminant classifier
cannot express. Novelty performance is highly configuration-sensitive (real
data showed n_components swinging held-out novelty 17%→100%), so the
sweep-and-compare leaderboard is what lets a user find and defend a robust
configuration — the UX parity is a correctness enabler, not cosmetics.

## Testing

- **Backend:** unit tests for scalar-or-list normalization (scalar in →
  identical single-row-per-combination grid) and grid expansion (N alphas × M
  n_components × K sizes → N·M·K× the rows).
- **Varsel:** a per-method >2-class smoke test for the newly-wired methods;
  assert binary-only methods skip-and-log rather than crash; assert Wold-family
  results unchanged from Phase B.
- **Reporting:** assert `NComponents` is populated per row and the new
  `display_cols` render; column tooltips resolve.
- **GUI state:** multi-class shows the 4A/4B cards, import panel is gone, model
  swap still works; toggling task types leaves no orphaned/duplicated widgets.
- **Validation tab:** a `multiclass_simca` holdout run reports `val_*`
  decision-matrix metrics (not RMSEP/accuracy); SPXY is unavailable for
  multi-class; KS/Random/Stratified splits produce a valid calibration/holdout
  partition and the metrics populate the results table.
- **Real-data e2e:** ORAU Excel (`Site`) — a multi-size / multi-alpha /
  multi-component leaderboard renders with self-describing rows; selecting a
  non-top row still saves → loads → predicts and reproduces its decision matrix.
- Zero regression across `test_simca`, `test_multiclass_search`, `model_io`,
  `contamination`; single-Y (regression/classification/one-class) paths
  byte-identical.
