# T-49: Persist external validation set indices for resume

**Status:** PLANNED — moderate fix, ~50 LOC.
**Filed:** 2026-05-02.
**Source:** User question during T-43 review wrap-up: "for the pause function, when it restarts does it also maintain the external validation set?"
**Priority:** MEDIUM — silent-leakage risk on resume when the validation algorithm is non-deterministic.

## Problem

T-43 captures the five validation-related Tk vars (`validation_enabled`, `validation_percentage`, `validation_algorithm`, `show_validation_metrics`, `validation_top_n`) so the user's resume banner restores them. For deterministic algorithms — SPXY, Kennard-Stone, Stratified — the same loaded data + same algorithm + same percentage produces the same partition; capturing the config is sufficient.

For two algorithms that's NOT enough:

- **`Random`** — the validation partition is drawn from a non-deterministic seed at "Create Validation Set" click time. On resume, if the user clicks the button again, they get a DIFFERENT random partition. Samples that the resumed Bayesian trials trained on (calibration partition from the prior session) could land in the new "validation" partition. The post-search RMSEP / val_Accuracy column would be polluted by samples the model has seen — silent leakage.

- **`Manual`** — the user picked specific samples by hand. There is no way to reproduce the partition without persisting the indices.

The actual partition state lives in three instance attributes on the GUI, not Tk vars:

```python
self.validation_indices = set()   # int indices
self.validation_X = None          # DataFrame slice
self.validation_y = None          # Series slice
```

These are not capturable via the existing `CAPTURABLE_SETTINGS` whitelist (which only covers Tk vars).

## Failure mode

1. User loads dataset, clicks "Create Validation Set" with algorithm=Random, %=20, seed not exposed.
2. Validation indices = {3, 17, 42, 88, ...} (random draw). Cal partition = remaining indices.
3. Bayesian search runs ~100 trials on cal partition; SQLite study persists.
4. Process killed before search completes. Sidecar persists with `validation_enabled=True, validation_algorithm=Random, validation_percentage=20`.
5. User relaunches. Resume banner says "Restored 90 settings" (including validation toggles + algorithm).
6. User clicks "Create Validation Set" again with the restored settings. **Different** random partition, e.g. {5, 17, 42, 91, ...}. Index 88 is no longer in validation; index 91 is.
7. User clicks Run Analysis. Resumed trials skip up to where they left off; new trials run on the NEW cal partition (which excludes 91 — but the OLD trials trained on data that included 91).
8. Post-search RMSEP / val_Accuracy is computed against the new validation partition. Index 91 (which the old trials trained on) is now in the validation set. Leakage.

## Fix sketch

Persist `validation_indices` in the sidecar alongside `gui_settings`:

```python
# run_state.py:RunMetadata
@dataclasses.dataclass
class RunMetadata:
    ...
    gui_settings: dict[str, Any] | None = None
    validation_indices: list[int] | None = None  # T-49
```

`start_run` accepts `validation_indices` kwarg, the GUI passes `sorted(self.validation_indices)` (set → sorted list for JSON).

On resume, the GUI's `_check_for_incomplete_run` resume-yes path:

1. Restore Tk vars (existing T-43 path).
2. If `resumed.validation_indices` is non-empty AND the dataset_fingerprint matches:
   - `self.validation_indices = set(resumed.validation_indices)`
   - Re-slice `self.X` to populate `self.validation_X`/`self.validation_y` from the same indices.
   - Update the GUI's "Validation set: N samples" status label.
3. If fingerprint mismatch (different data loaded), DON'T blindly apply the old indices — surface a warning.

## Why surgical-fold-in was rejected

Adding indices to `RunMetadata` is straightforward but the GUI-side restoration involves:

- Re-slicing `self.X`/`self.y` based on indices (must run AFTER the user re-loads the same dataset).
- Updating the validation status label.
- Handling the dataset_fingerprint mismatch case (current data ≠ resumed data) — skip restoration safely.

This is more than a one-line whitelist addition and would expand the scope of the T-43 PR. Filed as a separate ticket so the consolidated PR stays focused.

## Verification

1. Random algorithm: capture indices, kill process, relaunch, confirm `self.validation_indices` matches the captured set.
2. Manual algorithm: same as above.
3. Fingerprint mismatch: load different data on resume; confirm validation indices NOT applied; confirm warning surfaced.
4. Round-trip test: serialize indices to sidecar, deserialize, slice DataFrame, compare values.

## Out of scope

- Storing the full `validation_X` / `validation_y` content in the sidecar — too large; indices + the loaded DataFrame is the right contract.
- Auto-persisting indices on every `_create_validation_set` click independent of resume. Could be useful but separate UX concern.

## Dependencies

- Merge of `fix/bayesian-resume-and-cleanup` (T-42 + T-43 + T-38 consolidated PR) first.
