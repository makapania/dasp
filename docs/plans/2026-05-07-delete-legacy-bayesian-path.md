# Delete Legacy Bayesian Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the dead `run_bayesian_search` (test-only) function and all its now-unreachable helpers in `bayesian_utils.py`, while proving via before/after snapshot tests that `run_unified_bayesian` and adjacent search paths produce byte-identical output.

**Architecture:** Two-phase removal with a temporary snapshot harness. **Phase A** captures a JSON fixture of `run_unified_bayesian` outputs (best trial value, top-5 leaderboard rows, sorted fingerprint set) on three configs (PLS regression, LightGBM regression, PLS-DA classification) with fixed seeds. **Phase B** deletes the legacy code and verifies the harness still produces identical fixtures. The harness is committed (so review can run it) and removed in the same PR's final task once green.

**Tech Stack:** Python 3.12, pytest, Optuna 4.8, sklearn, numpy, pandas. Test data from `example/BoneCollagen.csv` (already used by `tools/bench_dedup_real.py`).

---

## Pre-flight context for the next agent

**What's being deleted:**

| Location | Symbol | Reason |
|---|---|---|
| `src/spectral_predict/search.py:4018-4736` | `def run_bayesian_search(...)` | Test-only legacy path. Comment at `:4455` already says `NOTE (T-36): run_bayesian_search is test-only — no GUI caller`. |
| `src/spectral_predict/bayesian_utils.py:86` | `_warn_mixed_regime_once` | Only used internally to deleted code (verified via grep). |
| `src/spectral_predict/bayesian_utils.py:113` | `create_optuna_study` | Only callers are `search.py:4124, 4526` inside `run_bayesian_search`. |
| `src/spectral_predict/bayesian_utils.py:203` | `create_objective_function` | Only caller is `search.py:4125` inside `run_bayesian_search`. |
| `src/spectral_predict/bayesian_utils.py:815` | `convert_optuna_result_to_dasp_format` | Only caller is `search.py:4599` inside `run_bayesian_search`. |
| `src/spectral_predict/bayesian_utils.py:1035` | `print_optimization_summary` | No external grep matches. |
| `src/spectral_predict/bayesian_utils.py:1068` | `get_param_importance` | No external grep matches. |
| `src/spectral_predict/bayesian_utils.py:1098` | `save_optimization_plots` | No external grep matches. |
| `src/spectral_predict/bayesian_utils.py:1141` | `class ProgressCallback` | Only used at `search.py:4127, 4573` inside `run_bayesian_search`. (Note: `nsga2_search.py:2022` defines its own local `ProgressCallback` — different symbol, do not confuse.) |
| `src/spectral_predict/bayesian_utils.py:1218` | `handle_failed_trial` | No external grep matches. |

**What survives in `bayesian_utils.py`:** `_extract_fitted_n_components` (line 32). Used by `nsga2_search.py:61, 3981` and `tests/test_cv_pls_clamp.py:15, 426`. **Do not touch.**

**Test files affected:**

| File | Action |
|---|---|
| `tests/test_bayesian_utils.py` | DELETE entirely. All test classes import `create_optuna_study`, `create_objective_function`, or `convert_optuna_result_to_dasp_format` (see lines 22-26). No test in this file targets a surviving symbol. |
| `tests/test_class_weight_validation_rebuild.py:334` | Drop `"run_bayesian_search"` from the parametrize list at the cited line. The grid-path assertion via `run_search` stays. |
| `tests/test_cv_pls_clamp.py:266-355` (approx — verify boundary) | Delete `TestRunBayesianSearchPLSGridClamping` class entirely (legacy black-box test). Keep `test_extract_fitted_n_components_handles_pipeline_keys` at line 424 — that pins a surviving symbol. |
| `tests/test_golden_standard_performance.py:213, 236` | Delete the legacy-path test entirely. Verify no unique assertion exists; the existing `tests/test_unified_bayesian_baseline.py` already pins golden-standard performance via `run_unified_bayesian`. |

**Comment-only references (low priority — clean up if convenient, do not block on them):**
- `src/spectral_predict/models.py:587` — docstring mentions `run_bayesian_search`. Replace with `run_unified_bayesian` reference.
- `src/spectral_predict/unified_bayesian.py:112, 310` — comments referencing `bayesian_utils.convert_optuna_result_to_dasp_format` and "legacy bayesian_utils path." Reword to drop `bayesian_utils` mention.

**Out of scope:**
- `src/spectral_predict/nsga2_search.py.backup` (1546 lines) — pre-existing repo cruft. Flag in the PR description; do not delete here.
- The 8 user-decision items from `docs/CONTINUATION_PROMPT_2026-05-07_pr54_followups.md` Section 8.

---

## File Structure

**Files created (temporary, removed in Task 8):**
- `tests/test_legacy_deletion_snapshot.py` — three snapshot tests pinning `run_unified_bayesian` byte-identical behavior across the deletion. Removed at end of plan.
- `tests/snapshots/unified_bayesian_pls_regression.json` — committed fixture.
- `tests/snapshots/unified_bayesian_lgbm_regression.json` — committed fixture.
- `tests/snapshots/unified_bayesian_plsda_classification.json` — committed fixture.

**Files modified:**
- `src/spectral_predict/search.py` — delete `run_bayesian_search` body (lines 4018-4736).
- `src/spectral_predict/bayesian_utils.py` — delete the 9 unused helpers; keep `_extract_fitted_n_components`.
- `src/spectral_predict/unified_bayesian.py` — reword 2 stale comments at `:112, :310`.
- `src/spectral_predict/models.py` — fix docstring at `:587`.
- `tests/test_class_weight_validation_rebuild.py` — drop one parametrize entry at `:334`.
- `tests/test_cv_pls_clamp.py` — delete the legacy-path test class.
- `docs/PROJECT_STATUS.md` — update with deletion record.
- `docs/SESSION_LOG.md` — append entry covering the deletion + snapshot strategy.

**Files deleted:**
- `tests/test_bayesian_utils.py` (entire file).
- `tests/test_golden_standard_performance.py` (entire file — verify in Task 4 before deletion).

---

### Task 1: Build the snapshot harness (BEFORE state)

**Files:**
- Create: `tests/test_legacy_deletion_snapshot.py`
- Create: `tests/snapshots/unified_bayesian_pls_regression.json`
- Create: `tests/snapshots/unified_bayesian_lgbm_regression.json`
- Create: `tests/snapshots/unified_bayesian_plsda_classification.json`

This task captures the current behavior of `run_unified_bayesian` on three representative configurations as JSON fixtures. The fixtures will be the regression oracle for Phase B.

- [ ] **Step 1: Create snapshots directory**

```bash
mkdir tests/snapshots
```

- [ ] **Step 2: Write the snapshot harness file**

Create `tests/test_legacy_deletion_snapshot.py` with this exact content:

```python
"""Before/after snapshot tests for the legacy Bayesian path deletion.

Pins ``run_unified_bayesian`` outputs on three representative configurations
with fixed RNG seeds. Generated with the legacy path present; asserted
byte-identical after the legacy path is deleted. This file is removed
once the deletion PR merges.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectral_predict.unified_bayesian import run_unified_bayesian

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _load_bone_collagen_xy():
    """Load the standard test dataset used by tools/bench_dedup_real.py."""
    df = pd.read_csv(Path(__file__).parent.parent / "example" / "BoneCollagen.csv")
    y = df["delta13C"].to_numpy()
    spectra_cols = [c for c in df.columns if c not in ("Sample", "delta13C", "delta15N")]
    X = df[spectra_cols].to_numpy()
    wl = np.array([float(c) for c in spectra_cols])
    return X, y, wl


def _load_bone_collagen_classification():
    """Synthesize a 3-class label from delta13C tertiles for PLS-DA testing."""
    X, y_continuous, wl = _load_bone_collagen_xy()
    q33, q67 = np.quantile(y_continuous, [0.333, 0.667])
    y_class = np.where(y_continuous < q33, 0, np.where(y_continuous < q67, 1, 2))
    return X, y_class, wl


def _serialize_results(df: pd.DataFrame, study, top_n: int = 5) -> dict:
    """Extract a stable, committable summary of the run.

    Captures three orthogonal signals so any drift surfaces:
    - DataFrame top-N rows (the leaderboard surface, GUI-visible)
    - study.best_value + study.best_params (Optuna's internal best)
    - Sorted fingerprint set (the dedup mechanism's shape — most sensitive
      to subtle changes in trial-suggest order or param-resolution logic)
    """
    df_sorted = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    top = df_sorted.head(top_n)
    fingerprints = sorted(
        str(t.user_attrs.get("fingerprint", ""))
        for t in study.trials
        if t.user_attrs.get("fingerprint")
    )
    return {
        "n_rows": len(df_sorted),
        "columns": sorted(df_sorted.columns.tolist()),
        "top_n": top_n,
        "top_rows": [
            {col: _coerce_scalar(row[col]) for col in sorted(df_sorted.columns)}
            for _, row in top.iterrows()
        ],
        "best_value": _coerce_scalar(study.best_value),
        "best_params": {
            k: _coerce_scalar(v) for k, v in sorted(study.best_params.items())
        },
        "n_fingerprints": len(fingerprints),
        "sorted_fingerprints": fingerprints,
    }


def _coerce_scalar(value):
    """Coerce numpy/pandas scalars to JSON-serializable types."""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "__nan__"
        if np.isinf(value):
            return "__pos_inf__" if value > 0 else "__neg_inf__"
        return round(float(value), 10)
    if isinstance(value, (np.integer, int, bool)):
        return int(value) if not isinstance(value, bool) else bool(value)
    return str(value)


def _assert_matches_snapshot(payload: dict, snapshot_name: str):
    """Compare payload against the committed JSON fixture."""
    snapshot_path = SNAPSHOT_DIR / snapshot_name
    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        pytest.fail(
            f"Snapshot {snapshot_name} did not exist; created from current run. "
            f"Re-run to verify it matches."
        )
    expected = json.loads(snapshot_path.read_text())
    assert payload == expected, (
        f"Snapshot {snapshot_name} drifted. "
        f"If intentional, delete the file and re-run to regenerate."
    )


@pytest.mark.integration
class TestUnifiedBayesianDeletionSnapshot:
    """Snapshot the public surface of run_unified_bayesian for three configs."""

    def test_pls_regression_snapshot(self):
        X, y, wl = _load_bone_collagen_xy()
        df, study = run_unified_bayesian(
            X=X, y=y, wavelengths=wl,
            model_name="PLS", task_type="regression",
            n_trials=20, random_state=42,
            cv_strategy="kfold", folds=5,
        )
        _assert_matches_snapshot(
            _serialize_results(df, study), "unified_bayesian_pls_regression.json"
        )

    def test_lgbm_regression_snapshot(self):
        X, y, wl = _load_bone_collagen_xy()
        df, study = run_unified_bayesian(
            X=X, y=y, wavelengths=wl,
            model_name="LightGBM", task_type="regression",
            n_trials=15, random_state=42,
            cv_strategy="kfold", folds=5,
        )
        _assert_matches_snapshot(
            _serialize_results(df, study), "unified_bayesian_lgbm_regression.json"
        )

    def test_plsda_classification_snapshot(self):
        X, y, wl = _load_bone_collagen_classification()
        df, study = run_unified_bayesian(
            X=X, y=y, wavelengths=wl,
            model_name="PLS-DA", task_type="classification",
            n_trials=15, random_state=42,
            cv_strategy="kfold", folds=5,
        )
        _assert_matches_snapshot(
            _serialize_results(df, study), "unified_bayesian_plsda_classification.json"
        )
```

- [ ] **Step 3: Verify the harness's exact `run_unified_bayesian` signature**

Run: `pytest tests/test_legacy_deletion_snapshot.py -v --collect-only`
Expected: 3 tests collected. If any fail to collect, inspect `src/spectral_predict/unified_bayesian.py::run_unified_bayesian` signature and adjust kwargs in the test (most likely culprits: `cv_strategy`, `folds`, `wavelengths` arg name).

- [ ] **Step 4: Run the harness once to generate the snapshots**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_legacy_deletion_snapshot.py -v`
Expected: 3 FAIL with "Snapshot X did not exist; created from current run. Re-run to verify it matches."

- [ ] **Step 5: Run the harness a second time to verify the snapshots are deterministic**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_legacy_deletion_snapshot.py -v`
Expected: 3 PASS. If any FAIL with "drifted," the harness is non-deterministic — investigate before proceeding (likely missing `random_state` somewhere downstream of `run_unified_bayesian`).

- [ ] **Step 6: Commit the harness + fixtures**

```bash
git add tests/test_legacy_deletion_snapshot.py tests/snapshots/
git commit -m "test(legacy-deletion): pin run_unified_bayesian behavior on 3 configs

Snapshot harness for the legacy run_bayesian_search deletion. Three
configurations (PLS regression, LightGBM regression, PLS-DA classification)
on example/BoneCollagen.csv with fixed seeds. Snapshots committed as JSON
fixtures so the post-deletion state can be asserted byte-identical.

Harness file is removed at the end of the deletion PR; the assertion was
load-bearing only across the deletion."
```

---

### Task 2: Delete `run_bayesian_search` from `search.py`

**Files:**
- Modify: `src/spectral_predict/search.py:4018-4736`

- [ ] **Step 1: Verify the function boundaries before deleting**

Run: `grep -n "^def \|^class " src/spectral_predict/search.py | grep -A1 "run_bayesian_search"`
Expected output:
```
4018:def run_bayesian_search(
4737:def _run_single_fold(
```

If line numbers have drifted (e.g. due to recent commits), re-grep and adjust the deletion range below accordingly.

- [ ] **Step 2: Delete the function body via line-number splice**

Regex-based deletion is fragile here — if `run_bayesian_search`'s body contains a comment or string literal matching `def _run_single_fold(`, a regex with a non-greedy lookahead truncates early and silently leaves orphan code. Use a line-number splice instead, with explicit boundary checks.

Run this Python recipe verbatim:

```python
import subprocess

# Step A: re-read the actual line numbers (do not trust stale values from the plan).
src_path = "src/spectral_predict/search.py"
with open(src_path, encoding="utf-8") as f:
    lines = f.readlines()

# Find the start of run_bayesian_search and the start of the NEXT top-level def/class.
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.startswith("def run_bayesian_search("):
        start_idx = i
        continue
    if start_idx is not None and (line.startswith("def ") or line.startswith("class ")):
        end_idx = i
        break

assert start_idx is not None, "run_bayesian_search not found"
assert end_idx is not None, "no following top-level def/class found"
assert end_idx > start_idx, "computed empty range"

# The next top-level def MUST be _run_single_fold per our static analysis.
assert lines[end_idx].startswith("def _run_single_fold("), (
    f"expected _run_single_fold at line {end_idx + 1}, got {lines[end_idx]!r}. "
    f"Refusing to delete — investigate."
)

deleted_count = end_idx - start_idx
print(f"Deleting lines {start_idx + 1}..{end_idx} ({deleted_count} lines)")

# Splice: keep lines before run_bayesian_search, then everything from _run_single_fold onward.
new_lines = lines[:start_idx] + lines[end_idx:]

with open(src_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

# Verify the file shrunk by exactly the expected count.
result = subprocess.run(["wc", "-l", src_path], capture_output=True, text=True)
print(f"After deletion: {result.stdout.strip()}")
print(f"Expected reduction: {deleted_count} lines")
```

Save this as a one-off script (e.g. `tools/_delete_run_bayesian_search.py`), run it once, then delete the script.

Expected: prints `Deleting lines 4019..4737 (719 lines)` (line numbers ±a few depending on recent commits), and the file shrinks by 719 lines.

If any of the asserts fail, STOP. The repo state diverged from the plan's assumptions — investigate before proceeding.

- [ ] **Step 3: Verify the file compiles**

Run: `.venv312/Scripts/python.exe -m py_compile src/spectral_predict/search.py`
Expected: no output (success).

- [ ] **Step 4: Verify no orphan references survive**

Run: `grep -n "run_bayesian_search\|create_objective_function\|convert_optuna_result_to_dasp_format" src/spectral_predict/search.py`
Expected: no matches.

- [ ] **Step 5: Verify the snapshot harness still passes (legacy path gone, but unified path untouched)**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_legacy_deletion_snapshot.py -v`
Expected: 3 PASS. If any FAIL, the deletion accidentally affected something `run_unified_bayesian` depends on — investigate before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/spectral_predict/search.py
git commit -m "refactor(bayesian): remove dead run_bayesian_search

Test-only legacy path with no GUI caller. Production Bayesian is
run_unified_bayesian; production preprocessing-discovery is the TPE
path in tpe_preprocessing_discovery.py.

Snapshot harness in tests/test_legacy_deletion_snapshot.py confirms
run_unified_bayesian produces byte-identical output on PLS regression,
LightGBM regression, and PLS-DA classification."
```

---

### Task 3: Migrate `tests/test_class_weight_validation_rebuild.py`

**Files:**
- Modify: `tests/test_class_weight_validation_rebuild.py:334` (drop one parametrize entry)

- [ ] **Step 1: Read the existing parametrize block**

Run: `pytest tests/test_class_weight_validation_rebuild.py -v --collect-only | grep -i bayesian`
Confirm which test IDs use the `run_bayesian_search` parametrize variant.

- [ ] **Step 2: Drop the legacy entry**

Use Edit to change:
```python
["run_search", "run_bayesian_search"],
```
to:
```python
["run_search"],
```

(The exact context around line 334 will tell you whether this is a `@pytest.mark.parametrize` decorator or a list literal in a fixture; the change is the same either way.)

- [ ] **Step 3: Verify the file compiles and the surviving tests still pass**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_class_weight_validation_rebuild.py -v`
Expected: all surviving tests PASS, no collection errors. The number of tests should be ~12 → ~11 (or whichever is the parametrize count minus one).

- [ ] **Step 4: Commit**

```bash
git add tests/test_class_weight_validation_rebuild.py
git commit -m "test(class-weight-rebuild): drop run_bayesian_search parametrize entry

Surviving run_search variant retains the grid-path assertion. Bayesian-side
contract is pinned by tests/test_unified_bayesian_baseline.py."
```

---

### Task 4: Delete `tests/test_bayesian_utils.py` and `tests/test_golden_standard_performance.py`

**Files:**
- Delete: `tests/test_bayesian_utils.py`
- Delete: `tests/test_golden_standard_performance.py` (after verification — see Step 2)

- [ ] **Step 1: Verify `tests/test_bayesian_utils.py` only tests deleted symbols**

Run: `grep "^def test\|^class Test\|from spectral_predict.bayesian_utils" tests/test_bayesian_utils.py`
Confirm every test class/function is testing one of: `create_optuna_study`, `create_objective_function`, `convert_optuna_result_to_dasp_format`. If any test targets a surviving symbol (e.g. `_extract_fitted_n_components`), STOP and migrate that test to a kept file before deleting.

Expected: all imports from `bayesian_utils` are the three doomed symbols.

- [ ] **Step 2: Verify `tests/test_golden_standard_performance.py` has no unique assertion**

Run: `grep "^def test\|assert " tests/test_golden_standard_performance.py | head -40`

Then check whether equivalent performance pinning exists in `tests/test_unified_bayesian_baseline.py`:

Run: `grep "^def test\|assert " tests/test_unified_bayesian_baseline.py | head -40`

If `test_golden_standard_performance.py` asserts a metric value (e.g. RMSE ≤ 0.5 on BoneCollagen) that `test_unified_bayesian_baseline.py` does NOT pin, port the assertion to `test_unified_bayesian_baseline.py` first using `run_unified_bayesian`. Then delete the file. If both files pin equivalent contracts, delete outright.

- [ ] **Step 3: Delete the two files**

```bash
git rm tests/test_bayesian_utils.py tests/test_golden_standard_performance.py
```

- [ ] **Step 4: Verify the test suite still collects and passes around the deletion**

Run: `.venv312/Scripts/python.exe -m pytest tests/ -v --collect-only 2>&1 | tail -20`
Expected: collection succeeds; no `ImportError` or `ModuleNotFoundError`.

Run: `.venv312/Scripts/python.exe -m pytest tests/test_legacy_deletion_snapshot.py tests/test_class_weight_validation_rebuild.py tests/test_cv_pls_clamp.py tests/test_unified_bayesian_baseline.py tests/test_bayesian_dedup.py tests/test_t44_autoscale_wiring.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "test: remove tests for deleted legacy Bayesian helpers

test_bayesian_utils.py exclusively tested create_optuna_study,
create_objective_function, and convert_optuna_result_to_dasp_format —
all removed in the previous commit.

test_golden_standard_performance.py duplicated coverage already provided
by test_unified_bayesian_baseline.py [or: ported assertion X to that file
in this commit]."
```

---

### Task 5: Delete the legacy-only test class in `tests/test_cv_pls_clamp.py`

**Files:**
- Modify: `tests/test_cv_pls_clamp.py` — delete the legacy-path test class around lines 266-355

- [ ] **Step 1: Identify the exact boundaries of the legacy class**

Run: `grep -n "^class \|^def " tests/test_cv_pls_clamp.py`

Find `class TestRunBayesianSearchPLSGridClamping` (or similar — verify the class name). Note its line number and the line of the next `class` or top-level `def`. The class body is the deletion target.

- [ ] **Step 2: Delete the class body**

Use Edit to replace `class TestRunBayesianSearchPLSGridClamping:\n<full body>` with an empty string, ensuring the next class/def's preceding blank line is preserved.

CRITICAL: do NOT touch `test_extract_fitted_n_components_handles_pipeline_keys` at line 424 — that pins a surviving symbol.

- [ ] **Step 3: Verify**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_cv_pls_clamp.py -v`
Expected: PASS, with the test count reduced by the size of the deleted class. `_extract_fitted_n_components_handles_pipeline_keys` and surrounding tests still PASS.

- [ ] **Step 4: Confirm no `run_bayesian_search` references survive in tests/**

Run: `grep -rn "run_bayesian_search" tests/`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cv_pls_clamp.py
git commit -m "test(cv-pls-clamp): drop legacy run_bayesian_search black-box test

Surviving _extract_fitted_n_components tests retain the contract.
Production-path equivalent (TestRunBayesianSearchPLSGridClamping
under run_search, not the legacy run_bayesian_search) was already
present and unaffected."
```

---

### Task 6: Delete unreachable helpers from `bayesian_utils.py`

**Files:**
- Modify: `src/spectral_predict/bayesian_utils.py` — delete 9 symbols, keep `_extract_fitted_n_components`

- [ ] **Step 1: Re-verify the import surface**

Run: `grep -rn "from .bayesian_utils\|from spectral_predict.bayesian_utils\|import bayesian_utils" src/ tests/ spectral_predict_gui_optimized.py`

Expected (after Tasks 2-5):
```
src/spectral_predict/nsga2_search.py:61:from .bayesian_utils import _extract_fitted_n_components
tests/test_cv_pls_clamp.py:15:from spectral_predict.bayesian_utils import _extract_fitted_n_components
tests/test_cv_pls_clamp.py:426:from spectral_predict.bayesian_utils import _extract_fitted_n_components
```

If any other import surfaces, STOP and audit before deleting helpers.

- [ ] **Step 2: Delete the 9 doomed symbols + the module-level state + the `__main__` example block**

Targets (verify line numbers via grep first — they may have shifted):

| Symbol | Why deletable |
|---|---|
| `_warn_mixed_regime_once` (function, ~line 86) | Only caller was inside the legacy path. Verified: `grep -rn _warn_mixed_regime_once src/ tests/` returns no external matches. |
| `_mixed_regime_warned = False` (module-level, ~line 83) | The global toggle that `_warn_mixed_regime_once` flips. Verified no external readers. Must delete with the function. |
| `create_optuna_study` (function, ~line 113) | Only callers were `search.py:4124, 4526` inside the now-deleted `run_bayesian_search`. |
| `create_objective_function` (function, ~line 203) | Only caller was `search.py:4125` (deleted in Task 2). |
| `convert_optuna_result_to_dasp_format` (function, ~line 815) | Only caller was `search.py:4599` (deleted in Task 2). |
| `print_optimization_summary` (function, ~line 1035) | No external grep matches. |
| `get_param_importance` (function, ~line 1068) | No external grep matches. |
| `save_optimization_plots` (function, ~line 1098) | No external grep matches. |
| `class ProgressCallback` (~line 1141) | Only used at `search.py:4127, 4573` (deleted). NOTE: `nsga2_search.py:2022` defines its OWN local `ProgressCallback` — different symbol, do not touch. |
| `handle_failed_trial` (function, ~line 1218) | Pure function (logs + returns 1e10), no global state, no external matches. Verified by reading body. |
| `if __name__ == '__main__':` block (~line 1247 to EOF) | Example code that calls `create_optuna_study`. Becomes broken once the helpers are deleted. Delete the whole block. |

For each deletion, use the same line-number splice pattern from Task 2 Step 2:
1. Re-read the file's lines.
2. Find the start (`def NAME(` or `class NAME:` or `if __name__ ==`).
3. Find the next top-level boundary (`^def `, `^class `, `^if __name__`, or EOF).
4. Splice out the range.
5. Assert the boundary is what you expect before writing.

Practical recipe to do all deletions in one pass:

```python
src_path = "src/spectral_predict/bayesian_utils.py"
with open(src_path, encoding="utf-8") as f:
    lines = f.readlines()

# Symbols to delete and the marker that starts each block.
DELETE_MARKERS = [
    "_mixed_regime_warned = False",
    "def _warn_mixed_regime_once(",
    "def create_optuna_study(",
    "def create_objective_function(",
    "def convert_optuna_result_to_dasp_format(",
    "def print_optimization_summary(",
    "def get_param_importance(",
    "def save_optimization_plots(",
    "class ProgressCallback",
    "def handle_failed_trial(",
    "if __name__ ==",
]

KEEP_MARKERS = ("def _extract_fitted_n_components",)

def find_block_end(lines, start, top_level_prefixes):
    """Find the first line after `start` that begins a new top-level block."""
    for i in range(start + 1, len(lines)):
        if any(lines[i].startswith(p) for p in top_level_prefixes):
            return i
    return len(lines)

TOP_LEVEL = ("def ", "class ", "if __name__", "_mixed_regime_warned")

# Delete from bottom to top so earlier line numbers stay stable.
spans_to_delete = []
for marker in DELETE_MARKERS:
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(marker)),
        None,
    )
    assert start is not None, f"marker {marker!r} not found"
    end = find_block_end(lines, start, TOP_LEVEL)
    spans_to_delete.append((start, end, marker))

# Sort descending by start so we splice from end backwards.
spans_to_delete.sort(key=lambda s: s[0], reverse=True)

for start, end, marker in spans_to_delete:
    # Sanity: confirm we're not about to delete _extract_fitted_n_components.
    assert not any(
        lines[i].startswith(KEEP_MARKERS) for i in range(start, end)
    ), f"deletion span for {marker!r} overlaps a KEEP marker — refusing"
    print(f"Deleting {marker!r} at lines {start + 1}..{end} ({end - start} lines)")
    del lines[start:end]

with open(src_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Done.")
```

After running, the file should contain (top-to-bottom): module docstring, imports, `_extract_fitted_n_components`, and nothing else of substance. Module-level constants used only by deleted code (e.g. result-format string templates referenced only inside `convert_optuna_result_to_dasp_format`) should also be deleted — manual pass over the surviving file recommended.

- [ ] **Step 3: Verify the file compiles**

Run: `.venv312/Scripts/python.exe -m py_compile src/spectral_predict/bayesian_utils.py`
Expected: no output.

- [ ] **Step 4: Verify the snapshot harness still passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_legacy_deletion_snapshot.py -v`
Expected: 3 PASS. If any FAIL, a deleted helper was actually being used by `run_unified_bayesian` via some indirect path — investigate immediately. The most likely culprit would be an import inside a function body (not at module top) that I missed in the static grep. Restore that helper before continuing.

- [ ] **Step 5: Verify nsga2 still works**

Run: `.venv312/Scripts/python.exe -c "from spectral_predict.nsga2_search import *; print('nsga2 imports OK')"`
Expected: `nsga2 imports OK`.

- [ ] **Step 6: Commit**

```bash
git add src/spectral_predict/bayesian_utils.py
git commit -m "refactor(bayesian-utils): remove helpers exclusive to legacy path

Deleted: _warn_mixed_regime_once, create_optuna_study,
create_objective_function, convert_optuna_result_to_dasp_format,
print_optimization_summary, get_param_importance,
save_optimization_plots, ProgressCallback, handle_failed_trial.

Retained: _extract_fitted_n_components — used by nsga2_search.py
and tests/test_cv_pls_clamp.py."
```

---

### Task 7: Clean up stale comment references

**Files:**
- Modify: `src/spectral_predict/unified_bayesian.py:112, 310`
- Modify: `src/spectral_predict/models.py:587`

- [ ] **Step 1: Reword `unified_bayesian.py:112` comment**

Read 5 lines of context around line 112. The current text says:
```
# bayesian_utils.convert_optuna_result_to_dasp_format. Extracting this as a
```

Replace `bayesian_utils.convert_optuna_result_to_dasp_format` with a description that doesn't reference the deleted module. Likely: `the result-conversion logic. Extracting this as a` (verify the surrounding sentence reads naturally).

- [ ] **Step 2: Reword `unified_bayesian.py:310` comment**

Read 5 lines of context. Current text says:
```
Sub-fits (legacy bayesian_utils path) store ``(trial_number, None)`` for
```

Replace `(legacy bayesian_utils path)` with `(deduplication placeholder)` or similar. Verify the surrounding sentence still parses.

- [ ] **Step 3: Update `models.py:587` docstring**

Read 5 lines around line 587. Current text mentions both `run_search` and `run_bayesian_search` in a docstring. Replace `run_bayesian_search` with `run_unified_bayesian`.

- [ ] **Step 4: Verify all three files compile**

Run: `.venv312/Scripts/python.exe -m py_compile src/spectral_predict/unified_bayesian.py src/spectral_predict/models.py`
Expected: no output.

- [ ] **Step 5: Verify the snapshot harness still passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_legacy_deletion_snapshot.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/spectral_predict/unified_bayesian.py src/spectral_predict/models.py
git commit -m "docs(comments): drop stale references to deleted bayesian_utils helpers"
```

---

### Task 8: Final verification + remove the snapshot harness + update docs

**Files:**
- Delete: `tests/test_legacy_deletion_snapshot.py`
- Delete: `tests/snapshots/unified_bayesian_pls_regression.json`
- Delete: `tests/snapshots/unified_bayesian_lgbm_regression.json`
- Delete: `tests/snapshots/unified_bayesian_plsda_classification.json`
- Modify: `docs/PROJECT_STATUS.md` (header)
- Modify: `docs/SESSION_LOG.md` (append entry)
- Delete (optional): `docs/CONTINUATION_PROMPT_2026-05-07_pr54_followups.md` (Item 7 now closed)

- [ ] **Step 1: Final dependency sweep — confirm no orphan references**

Run: `grep -rn "run_bayesian_search\|create_objective_function\|convert_optuna_result_to_dasp_format\|create_optuna_study\|print_optimization_summary\|get_param_importance\|save_optimization_plots\|handle_failed_trial\|_warn_mixed_regime_once" src/ tests/ spectral_predict_gui_optimized.py 2>&1 | grep -v "\.pyc:"`

Expected: NO MATCHES. The only remaining `ProgressCallback` reference should be the local class in `nsga2_search.py:2022` — that's a different symbol.

If anything else surfaces, fix it before continuing.

- [ ] **Step 2: Run the full targeted regression battery**

```bash
.venv312/Scripts/python.exe -m pytest \
  tests/test_legacy_deletion_snapshot.py \
  tests/test_bayesian_dedup.py \
  tests/test_t44_autoscale_wiring.py \
  tests/test_cv_pls_clamp.py \
  tests/test_class_weight_validation_rebuild.py \
  tests/test_unified_bayesian_baseline.py \
  -v
```

Expected: all PASS (snapshot harness still green = byte-identical behavior preserved across the deletion).

- [ ] **Step 3: Run the broader suite to catch indirect breakage**

```bash
.venv312/Scripts/python.exe -m pytest tests/ -x --ignore=tests/test_legacy_deletion_snapshot.py 2>&1 | tail -30
```

Expected: PASS (or only pre-existing failures unrelated to this PR — verify any failures by checking against pre-PR state).

- [ ] **Step 4: Actually import all production modules (catches runtime-only failures `py_compile` misses)**

`py_compile` only validates syntax — it does NOT execute module-level code. A missing import (e.g. an `ImportError` from a deleted helper that some module references at module scope) only surfaces at actual import time. Run a real import:

```bash
.venv312/Scripts/python.exe -c "
import spectral_predict.search
import spectral_predict.bayesian_utils
import spectral_predict.unified_bayesian
import spectral_predict.nsga2_search
import spectral_predict.models
print('production modules import OK')
"
```

Expected: `production modules import OK`.

Then verify the GUI module imports cleanly. The GUI module is huge (~70K lines) and may have lazy/dynamic imports; a real top-level import is the only way to surface module-scope failures:

```bash
.venv312/Scripts/python.exe -c "
import importlib, sys
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('gui_module', 'spectral_predict_gui_optimized.py')
mod = importlib.util.module_from_spec(spec)
# Don't actually run the Tk mainloop — just import the module top-level.
import os
os.environ['SPECTRAL_PREDICT_HEADLESS'] = '1'  # if the module honors this flag
try:
    spec.loader.exec_module(mod)
    print('gui module imports OK')
except SystemExit:
    print('gui module imports OK (module called sys.exit on load)')
"
```

Expected: `gui module imports OK` (with or without the SystemExit fallback).

If the GUI module imports cleanly only when `__name__ == '__main__'` is bypassed, that's still sufficient evidence — the deletion didn't break any module-scope reference. If it raises `ImportError`, `AttributeError`, or `NameError` referencing a deleted symbol, STOP and investigate — that's the exact failure mode this step exists to catch.

- [ ] **Step 5: Delete the snapshot harness + fixtures**

```bash
git rm tests/test_legacy_deletion_snapshot.py
git rm -r tests/snapshots/
```

- [ ] **Step 6: Verify the deletion didn't break anything**

```bash
.venv312/Scripts/python.exe -m pytest tests/ --collect-only 2>&1 | tail -5
```

Expected: collection clean. Run a short subset to be sure:

```bash
.venv312/Scripts/python.exe -m pytest tests/test_bayesian_dedup.py tests/test_t44_autoscale_wiring.py -v
```

Expected: 28 PASS.

- [ ] **Step 7: Update `docs/PROJECT_STATUS.md` header**

Read the top 5 lines, then update the `> **Last updated:**` line to record this session's work. Format following the existing pattern. Mention: deleted ~720 lines of test-only code in `search.py`, deleted 9 helpers from `bayesian_utils.py`, migrated 3 test files, deleted 2 test files, snapshot harness pinned byte-identical `run_unified_bayesian` behavior across the deletion.

- [ ] **Step 8: Append entry to `docs/SESSION_LOG.md`**

Add a new top-level entry (under the existing "PR #54 follow-ups" entry, before the "2026-05-06 late evening" entry) following the project's narrative style. Cover:
- The risk model (Class A/B/C from the plan);
- Why the snapshot harness was the right approach (assert byte-identical, no need to reason about every helper individually);
- The deletion scope was BIGGER than the continuation prompt suggested (9 helpers vs 3);
- `nsga2_search.py.backup` flagged as pre-existing repo cruft (out of scope here).

- [ ] **Step 9: (Optional) Mark continuation prompt as closed**

If keeping the continuation prompt for historical reference, prepend a SUPERSEDED banner. If deleting, `git rm docs/CONTINUATION_PROMPT_2026-05-07_pr54_followups.md`.

- [ ] **Step 10: Final commit**

```bash
git add docs/PROJECT_STATUS.md docs/SESSION_LOG.md
git rm tests/test_legacy_deletion_snapshot.py tests/snapshots/
# (optional) git rm docs/CONTINUATION_PROMPT_2026-05-07_pr54_followups.md
git commit -m "docs+test: close T-36 legacy Bayesian deletion

Snapshot harness validated byte-identical run_unified_bayesian behavior
across the deletion (PLS regression, LightGBM regression, PLS-DA
classification on example/BoneCollagen.csv, fixed seed=42, 15-20 trials
each). Harness removed now that the deletion has merged.

Closes Item 7 of CONTINUATION_PROMPT_2026-05-07_pr54_followups.md."
```

- [ ] **Step 11: Push and open PR**

Push the branch and open a PR per the project's review protocol.

```bash
git push -u origin <branch-name>
gh pr create --title "Delete legacy run_bayesian_search + bayesian_utils helpers" --body "$(cat <<'EOF'
## Summary
- Delete `run_bayesian_search` (~720 lines, test-only, no GUI caller per existing `NOTE (T-36)` comment).
- Delete 9 unused helpers from `bayesian_utils.py`; retain `_extract_fitted_n_components` (used by `nsga2_search.py` + 1 test).
- Migrate/delete affected test files: drop one parametrize entry in `test_class_weight_validation_rebuild.py`, delete `test_bayesian_utils.py` and `test_golden_standard_performance.py`, drop one test class in `test_cv_pls_clamp.py`.

## Behavioral safety
A temporary snapshot harness (`tests/test_legacy_deletion_snapshot.py`) pinned `run_unified_bayesian` outputs on three configurations (PLS regression, LightGBM regression, PLS-DA classification) on `example/BoneCollagen.csv` with fixed seed 42. The harness was green before the deletion and remained green at every commit through the deletion sequence — byte-identical top-5 leaderboard rows, identical best-trial values, identical fingerprint sets. Harness removed in the final commit.

## Test plan
- [x] `pytest tests/test_bayesian_dedup.py tests/test_t44_autoscale_wiring.py -v` — 28/28
- [x] Snapshot harness green at every commit
- [x] `py_compile` clean on all modified production modules
- [x] No grep matches for any deleted symbol in `src/`, `tests/`, or `spectral_predict_gui_optimized.py`

## Out of scope
- `src/spectral_predict/nsga2_search.py.backup` (1546-line pre-existing repo cruft) — flag for future cleanup PR.
- Item 8 (eight methodology/behavior changes) from the continuation prompt — needs explicit user approval.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review (already applied)

Walked through the spec one more time:

1. **Spec coverage:** Each item from the continuation prompt's Section 7 is covered: function deletion (Task 2), helper deletion (Task 6), test migration (Tasks 3-5), verification recipe (Task 8). The continuation prompt's stale-comment cleanup (its Section 5) is folded into Task 7. The continuation prompt under-scoped the helper-deletion list — this plan corrects that based on the actual grep audit.
2. **Placeholder scan:** No "TBD" or "implement appropriately" placeholders. Every step has either an exact command or a code recipe.
3. **Type consistency:** No new types introduced. The snapshot harness uses only standard library + numpy/pandas + the existing `run_unified_bayesian` API.
4. **Order safety:** Test deletion (Tasks 3-5) precedes helper deletion (Task 6) so the test suite never references a missing symbol via a doomed-but-not-yet-deleted test file. Snapshot harness (Task 1) is built FIRST so it's green BEFORE any production code changes — this is the load-bearing safety check.

## Notes on what's deliberately NOT in this plan

- **No coverage extension during this PR.** The temporary snapshot harness is sufficient for "deletion-doesn't-change-behavior" — it's not the right place to add new permanent regression tests. Permanent test additions should be a separate ticket.
- **No `nsga2_search.py.backup` cleanup.** Pre-existing cruft, not load-bearing for this deletion. Flag in PR description, defer.
- **No methodology changes from Item 8.** Those need explicit user approval before any agent acts.
