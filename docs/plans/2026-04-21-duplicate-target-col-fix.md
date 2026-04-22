# 2026-04-21 — Fix duplicate target column in data viewer and exports

## Context

In the combined-format CSV/Excel load path, the target column appears twice in
the Data Management viewer header row and in CSV/Excel exports. Root cause:
the low-level readers correctly strip the target from `metadata_df`, but the
GUI then **re-adds** it to `self.combined_metadata_df` at
`spectral_predict_gui_optimized.py:17985-17990` so that the target dropdown can
list it for target-switching. Downstream, `_populate_data_viewer` and three
export helpers iterate `self.combined_metadata_df.columns` AND append a
dedicated `[target_col]` entry, producing a duplicate.

Additionally, `_apply_data_viewer_edits` rebuilds `self.combined_metadata_df`
from metadata-indexed headers only (excluding the target), which silently
removes the target from `combined_metadata_df` after any cell edit. That breaks
the target-switch dropdown (`_get_available_target_columns` reads
`combined_metadata_df.columns`).

Two independent investigations (Codex + Kimi K2.6) agreed on the same fix
shape. See their reports in the session log and the original root-cause doc.

## Goal

Fix the duplicate in all four places that render or export, **without
mutating `self.combined_metadata_df`**, and add a guard in
`_apply_data_viewer_edits` that preserves the target after an edit.

All changes are **view-layer / export-layer projections only**. The
authoritative state (`self.combined_metadata_df`, `self.y`, `self.X`) is
untouched.

## Non-Goals

- Do **not** change how `self.combined_metadata_df` is populated at load time
  (`spectral_predict_gui_optimized.py:17985-17990`). That re-insertion is
  intentional — the target dropdown depends on it.
- Do **not** change `_populate_data_viewer`'s separate-ref branch
  (line ~30291). It already works correctly.
- Do **not** change the detection-only combined read path (~15915) or the
  Data Management combined load (~17061-17075, ~17122-17136) — they do not
  duplicate today.
- Do **not** change the calibration-transfer replacement path (~46834-46849) —
  it already strips `y_col`.
- Do **not** touch `_delete_data_viewer_column` or `_on_data_viewer_right_click`
  unless necessary. They currently name-match `target_col` to block deletion,
  which is safe behavior regardless of the duplicate.
- Do **not** refactor `_apply_data_viewer_edits`'s header-scanning logic
  beyond the minimum target-preservation guard specified below.
- Do **not** modify any test.

## Sites to change (5 edits in one file)

All line numbers are relative to **current `spectral_predict_gui_optimized.py`
HEAD at `2301287`** (the highlight-fix commit). GLM must re-read the file to
confirm line numbers have not drifted before editing.

The file is `spectral_predict_gui_optimized.py` in the worktree root
(`C:\Users\sponheim\git\dasp\.worktrees\fix-duplicate-target-col\`).

---

### Edit 1 — `_populate_data_viewer` combined branch (~line 30283)

**Current code** (`_populate_data_viewer`, ~line 30277-30283):

```python
# For combined files, metadata is in combined_metadata_df
if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None and len(self.combined_metadata_df.columns) > 0:
    # Filter metadata to match displayed samples
    if show_excluded:
        display_metadata = self.combined_metadata_df.copy()
    else:
        display_metadata = self.combined_metadata_df[mask].copy()
    metadata_cols = list(self.combined_metadata_df.columns)
```

**Change** — replace the `metadata_cols = list(...)` line with:

```python
    metadata_cols = [c for c in self.combined_metadata_df.columns if c != target_col]
```

**Rationale:** mirrors the already-correct separate-ref branch at the
immediately-following `elif` (line ~30291). The local `display_metadata` data
frame keeps the target column so we can still read its values from it for the
dedicated `[target_col]` position later (line ~30302). But `metadata_cols`
(the list that feeds the header row and the per-row metadata-values loop)
excludes the target, so it's no longer rendered twice.

**Warning — do NOT change** the `display_metadata` assignment above it. The
per-row loop at line ~30316-30325 iterates `metadata_cols` and reads from
`display_metadata`; because `target_col` is excluded from `metadata_cols`, the
loop won't read target from `display_metadata`. Leaving `display_metadata`
unfiltered is harmless and preserves the data source if future code needs it.

### Edit 2 — `_export_data_viewer_to_csv` combined branch (~line 30485-30488)

**Current code** (`_export_data_viewer_to_csv`, ~line 30482-30494):

```python
# Add metadata columns if available (insert before target column)
# Check combined file metadata first, then reference file metadata
insert_position = 0
if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None and len(self.combined_metadata_df.columns) > 0:
    for col in self.combined_metadata_df.columns:
        export_df.insert(insert_position, col, self.combined_metadata_df[col])
        insert_position += 1
elif self.ref is not None and len(self.ref.columns) > 0:
    for col in self.ref.columns:
        if col == target_col:
            continue
        export_df.insert(insert_position, col, self.ref[col])
        insert_position += 1
```

**Change** — add `if col == target_col: continue` to the combined branch so
both branches treat the target consistently:

```python
if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None and len(self.combined_metadata_df.columns) > 0:
    for col in self.combined_metadata_df.columns:
        if col == target_col:
            continue
        export_df.insert(insert_position, col, self.combined_metadata_df[col])
        insert_position += 1
```

**Rationale:** the dedicated target column is inserted at line ~30502-30503
(`export_df.insert(insert_position, target_col, self.y)`). Without this skip,
pandas raises `ValueError: cannot insert <col>, already exists` at the
dedicated insert, OR the target column ends up in the CSV twice.

### Edit 3 — `_export_to_csv` combined branch (~line 17836-17838)

**Current code** (`_export_to_csv`, ~line 17830-17844):

```python
export_df = self.X.copy()
target_col = self.target_column.get() or 'Target'
if self.y is not None:
    export_df.insert(0, target_col, self.y)
if self.sample_sets is not None and self.sample_sets.notna().any():
    export_df.insert(0, 'Set', self.sample_sets)
if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None:
    for col in self.combined_metadata_df.columns:
        export_df.insert(0, col, self.combined_metadata_df[col])
elif self.ref is not None:
    for col in self.ref.columns:
        if col == target_col:
            continue
        export_df.insert(0, col, self.ref[col])
export_df.to_csv(filename)
```

**Change** — add the same skip to the combined branch:

```python
if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None:
    for col in self.combined_metadata_df.columns:
        if col == target_col:
            continue
        export_df.insert(0, col, self.combined_metadata_df[col])
```

**Rationale:** target has already been inserted at position 0 earlier
(line 17833). Iterating the combined metadata without skipping target raises
`ValueError: cannot insert <col>, already exists`.

### Edit 4 — `_export_to_excel` combined branch (~line 17876-17878)

**Current code** (`_export_to_excel`, ~line 17869-17885):

```python
export_df = self.X.copy()
target_col = self.target_column.get() or 'Target'
if self.y is not None:
    export_df.insert(0, target_col, self.y)
if self.sample_sets is not None and self.sample_sets.notna().any():
    export_df.insert(0, 'Set', self.sample_sets)
# Add metadata columns (from combined file or reference file)
if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None:
    for col in self.combined_metadata_df.columns:
        export_df.insert(0, col, self.combined_metadata_df[col])
elif self.ref is not None:
    for col in self.ref.columns:
        if col == target_col:
            continue
        export_df.insert(0, col, self.ref[col])

export_df.to_excel(writer, sheet_name='Spectral_Data')
```

**Change** — add the same skip to the combined branch:

```python
if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None:
    for col in self.combined_metadata_df.columns:
        if col == target_col:
            continue
        export_df.insert(0, col, self.combined_metadata_df[col])
```

**Rationale:** same as Edit 3 but for Excel export.

### Edit 5 — `_apply_data_viewer_edits` target-preservation guard (~line 30623-30629)

**Current code** (`_apply_data_viewer_edits`, ~line 30623-30629):

```python
# Update metadata (ref or combined_metadata_df)
if new_metadata:
    metadata_df = pd.DataFrame(new_metadata, index=new_indices)
    if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None:
        self.combined_metadata_df = metadata_df
    else:
        self.ref = metadata_df
```

**Change** — after replacing `combined_metadata_df`, re-attach the current
target column so `_get_available_target_columns` still finds it:

```python
# Update metadata (ref or combined_metadata_df)
if new_metadata:
    metadata_df = pd.DataFrame(new_metadata, index=new_indices)
    if hasattr(self, 'combined_metadata_df') and self.combined_metadata_df is not None:
        # Re-attach target column — `_get_available_target_columns` reads
        # `combined_metadata_df.columns` to populate the target dropdown.
        # The sheet-rebuild loop above excludes target-named headers from
        # `metadata_indices`, so without this re-attachment the target would
        # disappear from the dropdown after the first edit.
        self.combined_metadata_df = metadata_df
        if target_idx is not None and self.y is not None and target_col_name:
            self.combined_metadata_df[target_col_name] = self.y
    else:
        self.ref = metadata_df
```

**Rationale:** self.y was just rebuilt above (line ~30613-30614 with
`new_y_values` indexed by `new_indices`), so it's index-aligned with the new
`metadata_df`. Re-attaching `self.y` under `target_col_name` brings the target
back into `combined_metadata_df.columns` so target-switching continues to
work. `target_idx is not None` guards against the case where the sheet had
no target column at all; `self.y is not None` guards against a spectra-only
run.

**Do NOT** change how `target_idx` is detected (lines 30552-30563). The
current header scan overwrites `target_idx` when duplicates exist, but after
Edits 1-4 land the duplicate no longer renders, so the scan behavior is moot
for future edits. We're leaving the scan alone because changing it risks
breaking the existing heuristic for the "Target" fallback name (line 30555).

---

## Verification steps (must pass before declaring complete)

Run in this order inside the worktree
(`C:\Users\sponheim\git\dasp\.worktrees\fix-duplicate-target-col\`). Report
each command's exit code and tail output.

### V1 — Syntax check

```bash
python -c "import ast; ast.parse(open('spectral_predict_gui_optimized.py', encoding='utf-8').read()); print('parse OK')"
```

Must print `parse OK`.

### V2 — Test suite

```bash
python -m pytest tests/test_refine_task_type_preservation.py tests/test_cv_strategy.py tests/test_contamination_detection.py -q
```

Expected: `144 passed`. Baseline (before your changes) was 144/144 on this
worktree. Zero regressions.

### V3 — Manual grep proof that all five sites changed

Run each command below and confirm the `target_col` guard appears:

```bash
# Edit 1: _populate_data_viewer
grep -n "for c in self.combined_metadata_df.columns if c != target_col" spectral_predict_gui_optimized.py

# Edits 2, 3, 4: the three export helpers — each should have a `if col == target_col: continue` near a combined_metadata_df iteration
grep -n "if col == target_col" spectral_predict_gui_optimized.py
# Expect at least 6 matches total (3 new combined-branch skips + 3 pre-existing ref-branch skips)

# Edit 5: target-preservation guard
grep -n "_get_available_target_columns" spectral_predict_gui_optimized.py
# No new matches expected here, just confirming context
grep -n "self.combined_metadata_df\[target_col_name\]" spectral_predict_gui_optimized.py
# Must have at least 1 match (the new guard in _apply_data_viewer_edits)
```

### V4 — Import smoke test

```bash
python -c "import spectral_predict_gui_optimized; print('import OK')"
```

Must print `import OK` without errors. This catches NameError / attribute
issues the parser misses.

### V5 — Targeted behavior sanity check

Write a minimal Python snippet that simulates the combined-load state and
calls `_populate_data_viewer`-style logic. (Optional — if this is too
awkward to set up headlessly because of the Tkinter dependency, skip V5 and
rely on V1-V4 plus user manual verification.)

## Acceptance criteria

- V1 passes (parse OK).
- V2 passes (144/144 tests, zero regressions).
- V3 shows all five edits present.
- V4 passes (import OK).
- No file outside `spectral_predict_gui_optimized.py` is modified.
- No lines in `spectral_predict_gui_optimized.py` are changed except the five
  hunks specified above.
- `git diff --stat` shows exactly one file changed, with a reasonable line
  count (~15-25 insertions, ~3-5 deletions).

## What to do if something breaks

- If any baseline test that was passing now fails — STOP. Do not try to fix.
  Write what broke to a file `GLM_FAILURE.md` at the worktree root, describe
  the symptom and what you tried, and halt.
- If the syntax check fails — re-read your edit, verify indentation matches
  surrounding code (tabs vs spaces), and retry.
- If a line number in this plan does not match what you find — re-grep for
  the exact surrounding context I quoted and edit that location. Line numbers
  may have drifted by 1-2 due to git or editor normalization.

## User manual verification (post-GLM, not GLM's responsibility)

The user will manually verify after merge:
1. Load `C:\Users\sponheim\Desktop\_DeskSync\Arctic\Divided Habitat\Valley_contactProbe_dry.csv`.
2. Confirm target defaults to `group.cov.valley` and the Data Management
   viewer shows the target column only ONCE (header row).
3. Change target dropdown to `group.lab.valley`. Confirm the viewer refreshes
   and still shows only ONE target column, now named `group.lab.valley`.
4. Edit a cell in the viewer, confirm the target dropdown still lists both
   `group.cov.valley` and `group.lab.valley` options.
5. Click "Export to CSV" → confirm the exported file has exactly ONE target
   column.
6. Repeat load with a file that uses a separate reference CSV (e.g. BoneCollagen
   if available) → confirm no regression in the separate-ref path.

## Commit message template (for post-implementation commit)

```
fix(data-viewer): eliminate duplicate target column in combined-CSV viewer and exports

In the combined-format CSV/Excel load path, `self.combined_metadata_df`
contains the target column (re-added by the GUI loader at
spectral_predict_gui_optimized.py:17985-17990 so the target-switch dropdown
can list it). The data viewer and three export helpers iterated
`combined_metadata_df.columns` AND appended a dedicated `[target_col]`,
producing a duplicate — visible as a double-highlighted column header, a
duplicate column in CSV/Excel exports, and (in `_apply_data_viewer_edits`)
silent removal of the target from `combined_metadata_df` after any cell
edit, which broke the target-switch dropdown.

Fix: filter `target_col` out of the combined-metadata iteration at four sites
(one viewer, three exports), mirroring the already-correct separate-ref
branch. Add a target-preservation guard in `_apply_data_viewer_edits` so
`_get_available_target_columns` continues to find the target after cell edits.

All changes are view/export-layer projections. `self.combined_metadata_df`,
`self.y`, and `self.X` state are untouched. Separate-ref workflows are
unchanged.

Investigation: Codex + Kimi K2.6 independently agreed on the fix shape.
Plan: docs/plans/2026-04-21-duplicate-target-col-fix.md.

Co-Authored-By: GLM-5.1 <noreply@z.ai>
```
