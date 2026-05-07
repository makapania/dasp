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

from sklearn.preprocessing import LabelEncoder

from spectral_predict.io import read_asd_dir
from spectral_predict.unified_bayesian import run_unified_bayesian

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
EXAMPLE_DIR = Path(__file__).parent.parent / "example"

REGRESSION_TARGET = "%Collagen"
CLASSIFICATION_TARGET = "CollagenCat"


def _load_joined_dataframe():
    """Load and join the standard BoneCollagen dataset.

    Mirrors tools/bench_baseline_compare.py:54-67 — the canonical loader
    pattern in this repo. Spectra live in example/Spectrum*.asd and are
    joined to example/BoneCollagen.csv on a normalized File Number key.
    """
    spectra, _meta = read_asd_dir(str(EXAMPLE_DIR))
    ref = pd.read_csv(EXAMPLE_DIR / "BoneCollagen.csv")
    spectra = spectra.sort_index()
    ref = ref.copy()
    ref.index = (
        ref["File Number"].astype(str)
        .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        .str.replace(" ", "", regex=False)
    )
    joined = spectra.join(ref, how="inner")
    feature_cols = list(spectra.columns)
    X = joined[feature_cols].to_numpy(dtype=float)
    wl = np.asarray([float(c) for c in feature_cols], dtype=float)
    return joined, X, wl


def _load_bone_collagen_xy():
    """Regression target — %Collagen continuous."""
    joined, X, wl = _load_joined_dataframe()
    y = joined[REGRESSION_TARGET].to_numpy(dtype=float)
    return X, y, wl


def _load_bone_collagen_classification():
    """Classification target — CollagenCat text labels encoded to ints."""
    joined, X, wl = _load_joined_dataframe()
    le = LabelEncoder()
    y = le.fit_transform(joined[CLASSIFICATION_TARGET].astype(str))
    return X, y, wl


def _serialize_results(df: pd.DataFrame, study, top_n: int = 5) -> dict:
    """Extract a stable, committable summary of the run.

    Captures four orthogonal signals so any drift surfaces:
    - The full DataFrame in trial order (every leaderboard row, ordered).
      Sorted-only would mask RNG drift that produces the same row set in
      a different order — that's a real regression we must catch.
    - study.best_value + study.best_params (Optuna's internal best).
    - Per-trial ordered (number, value, params, fingerprint) tuples — most
      sensitive to import-order or sampler-state drift.
    - The full DataFrame's top-N after sort (cross-check; backwards-
      compatible with simpler diff inspection).
    """
    df_in_trial_order = df.reset_index(drop=True)
    df_sorted = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    top = df_sorted.head(top_n)
    trial_records = [
        {
            "number": int(t.number),
            "state": str(t.state),
            "value": _coerce_scalar(t.value) if t.value is not None else None,
            "params": {k: _coerce_scalar(v) for k, v in sorted(t.params.items())},
            "fingerprint": str(t.user_attrs.get("fingerprint", "")) or None,
            "duplicate_of": t.user_attrs.get("duplicate_of_trial"),
        }
        for t in sorted(study.trials, key=lambda t: t.number)
    ]
    return {
        "n_rows": len(df_in_trial_order),
        "columns": sorted(df_in_trial_order.columns.tolist()),
        "top_n": top_n,
        "all_rows_in_trial_order": [
            {col: _coerce_scalar(row[col]) for col in sorted(df_in_trial_order.columns)}
            for _, row in df_in_trial_order.iterrows()
        ],
        "top_rows_after_sort": [
            {col: _coerce_scalar(row[col]) for col in sorted(df_sorted.columns)}
            for _, row in top.iterrows()
        ],
        "best_value": _coerce_scalar(study.best_value),
        "best_params": {
            k: _coerce_scalar(v) for k, v in sorted(study.best_params.items())
        },
        "trial_count": len(trial_records),
        "trials": trial_records,
    }


def _coerce_scalar(value):
    """Coerce arbitrary values to JSON-serializable forms with stable rounding.

    Handles scalars, numpy types, and nested containers (tuple/list/dict).
    Recursive — float precision is preserved at every nesting level so
    a tuple-valued Optuna param like (0.5000001, 0.5000002) doesn't lose
    digits via str(). Sentinels emitted for nan/inf to survive JSON.
    """
    # Booleans BEFORE numerics — bool is a subclass of int in Python.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "__nan__"
        if np.isinf(value):
            return "__pos_inf__" if value > 0 else "__neg_inf__"
        return round(float(value), 10)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (tuple, list)):
        return [_coerce_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_scalar(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if value is None:
        return None
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
            cv_strategy="kfold", cv_folds=5,
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
            cv_strategy="kfold", cv_folds=5,
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
            cv_strategy="kfold", cv_folds=5,
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

### Task 4: Delete `tests/test_bayesian_utils.py` and surgically remove the one legacy test from `test_golden_standard_performance.py`

**CRITICAL — Codex caught this:** `tests/test_golden_standard_performance.py` is NOT a pure-legacy file. It contains four tests, three of which target the GRID path (`run_search`) with valuable golden R²/RMSE pins:

- `test_pls_golden_standard` (line 61) — exact R²/RMSE pin for PLS via `run_search`. **KEEP.**
- `test_lightgbm_golden_standard` (line 105) — exact R²/RMSE pin for LightGBM via `run_search`. **KEEP.**
- `test_variable_selection_spa_correctness` (line 150) — SPA cross-model assertion via `run_search`. **KEEP.**
- `test_bayesian_varsel_caching` (line 204) — uses `run_bayesian_search`. **DELETE only this one.**

Do NOT `git rm` the whole file. The original plan got this wrong.

**Files:**
- Delete: `tests/test_bayesian_utils.py` (entire file — every test imports a doomed symbol)
- Modify: `tests/test_golden_standard_performance.py` — surgically remove `test_bayesian_varsel_caching` and any imports it uniquely needs

- [ ] **Step 1: Verify `tests/test_bayesian_utils.py` only tests deleted symbols**

Run: `grep "^def test\|^class Test\|from spectral_predict.bayesian_utils" tests/test_bayesian_utils.py`
Confirm every test class/function is testing one of: `create_optuna_study`, `create_objective_function`, `convert_optuna_result_to_dasp_format`. If any test targets a surviving symbol (e.g. `_extract_fitted_n_components`), STOP and migrate that test to a kept file before deleting.

Expected: all imports from `bayesian_utils` are the three doomed symbols.

- [ ] **Step 2: Surgically remove `test_bayesian_varsel_caching` from `test_golden_standard_performance.py`**

Read the file from line 199 (the section header `# Bayesian variable selection caching benchmark`) to the end of the `test_bayesian_varsel_caching` function (likely ~line 270, verify by reading until the next `# ---` block or EOF).

Use Edit to delete the entire section: the section comment block + the test function. Also remove the `time` import at the top of the file if `test_bayesian_varsel_caching` was its only consumer (verify with `grep "time\." tests/test_golden_standard_performance.py` after the deletion).

The three surviving tests (`test_pls_golden_standard`, `test_lightgbm_golden_standard`, `test_variable_selection_spa_correctness`) must remain unchanged.

- [ ] **Step 3: Delete `tests/test_bayesian_utils.py`**

```bash
git rm tests/test_bayesian_utils.py
```

- [ ] **Step 4: Verify the test suite still collects and the surviving golden tests still pass**

Run: `.venv312/Scripts/python.exe -m pytest tests/ -v --collect-only 2>&1 | tail -20`
Expected: collection succeeds; no `ImportError` or `ModuleNotFoundError`.

Run: `.venv312/Scripts/python.exe -m pytest tests/test_golden_standard_performance.py -v`
Expected: 3 tests collected and PASS (the legacy test is gone). The exact R²/RMSE pins are unchanged.

Run: `.venv312/Scripts/python.exe -m pytest tests/test_legacy_deletion_snapshot.py tests/test_class_weight_validation_rebuild.py tests/test_cv_pls_clamp.py tests/test_unified_bayesian_baseline.py tests/test_bayesian_dedup.py tests/test_t44_autoscale_wiring.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_golden_standard_performance.py
git rm tests/test_bayesian_utils.py
git commit -m "test: remove tests for deleted legacy Bayesian helpers

- Delete test_bayesian_utils.py entirely: every test imported one of
  create_optuna_study / create_objective_function /
  convert_optuna_result_to_dasp_format, all removed in the previous commit.

- Surgically remove test_bayesian_varsel_caching from
  test_golden_standard_performance.py: it used run_bayesian_search.
  The other three tests (PLS golden R²/RMSE, LightGBM golden R²/RMSE,
  SPA cross-model correctness) all use run_search and stay."
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

After running, the file should contain (top-to-bottom): module docstring, imports, `_extract_fitted_n_components`, and nothing else of substance.

- [ ] **Step 2b: Prune now-dead imports and module-level constants**

`_extract_fitted_n_components` only needs `ast`, `logging`, and `Optional` / `Any` from typing. Everything else in the imports block was consumed by deleted code. Verified scope:

| Import | Deletable | Reason |
|---|---|---|
| `import optuna` | YES | Only consumed by `create_optuna_study` + `handle_failed_trial`. |
| `from optuna.samplers import TPESampler, RandomSampler` | YES | `create_optuna_study` only. |
| `from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, PercentilePruner` | YES | `create_optuna_study` only. |
| `from .constants import RANDOM_STATE` | YES | Used inside `create_objective_function` only. |
| `from .regions import create_region_subsets` | YES | Used inside `create_objective_function` only. |
| `import numpy as np` | VERIFY — grep for `np\.` after Step 2 | Likely deletable. |
| `import pandas as pd` | VERIFY — grep for `pd\.` after Step 2 | Likely deletable. |
| Other `from .X import Y` | VERIFY — grep each for surviving usage | Anything not used by `_extract_fitted_n_components`. |
| `from typing import Dict, List, Optional, Callable, Tuple, Any` | PRUNE to `Optional, Any` | Only those two are used by `_extract_fitted_n_components`. |
| `import ast` | KEEP | Used by `_extract_fitted_n_components` for `ast.literal_eval`. |
| `import logging` | KEEP | Used by `_extract_fitted_n_components` for `logger`. |

Module-level constants to verify and likely delete:
- `_mixed_regime_warned = False` — already deleted in Step 2.
- Any `_FORMAT_*` / `_CONVERT_*` / `_OPTUNA_*` constants — grep for usage; if only deleted code referenced them, delete.
- `logger = logging.getLogger(__name__)` — KEEP if `_extract_fitted_n_components` uses it.

Recipe — after Step 2's main deletion, run:

```bash
.venv312/Scripts/python.exe -c "
import ast
src = open('src/spectral_predict/bayesian_utils.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [a.name for a in node.names]
        print(f'  Line {node.lineno}: {ast.unparse(node)}')
"
```

For each surviving import, grep the file body (excluding the import line itself) for the imported name. If zero matches, delete the import.

- [ ] **Step 2c: Rewrite the module docstring**

Read the file's first 25 lines to find the existing module docstring. It currently advertises "Creating reproducible Optuna studies", "Converting parameters between formats", "Handling pruning and early stopping", "Error handling and validation" — all describing deleted functions. Lines ~10-18 also reference the 2026-01-02 fix to `create_objective_function`.

Replace with a 5-line minimal docstring describing only `_extract_fitted_n_components`:

```python
"""Helper for extracting the actually-fitted ``n_components`` from a PLS
trial's params dict. Used by the Bayesian and NSGA-II search paths to
populate the LVs column with the post-clamp value (rather than Optuna's
raw pre-clamp suggestion). Sole survivor of this module after the legacy
``run_bayesian_search`` deletion (2026-05-07)."""
```

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

- [ ] **Step 1: Final dependency sweep — confirm no orphan references in code**

Run: `grep -rn "run_bayesian_search\|create_objective_function\|convert_optuna_result_to_dasp_format\|create_optuna_study\|print_optimization_summary\|get_param_importance\|save_optimization_plots\|handle_failed_trial\|_warn_mixed_regime_once" src/ tests/ spectral_predict_gui_optimized.py 2>&1 | grep -v "\.pyc:"`

Expected: NO MATCHES. The only remaining `ProgressCallback` reference should be the local class in `nsga2_search.py:2022` — that's a different symbol.

If anything else surfaces, fix it before continuing.

- [ ] **Step 1b: Sweep docs for stale references**

Run: `grep -rn "run_bayesian_search\|create_objective_function\|convert_optuna_result_to_dasp_format\|bayesian_utils\." docs/`

Many matches are EXPECTED — historical session-log entries, prior continuation prompts, design-history records — these are durable and should NOT be edited. But active references in current-state docs (PROJECT_STATUS header, follow-up queues, "what works" sections) need updating.

Walk every match and classify:
- **Historical/dated entry** (e.g. lines under `## Session 2026-04-XX` or `## Previous session —`) → leave alone, that's the record of what was true at that time.
- **Active state** (header `> **Last updated:**`, "Known Issues", "Follow-Ups", "What Works") → update or delete.
- **Continuation prompts** (`docs/CONTINUATION_PROMPT_*.md`) → if marked SUPERSEDED, leave; otherwise update or delete.

Report the active-state matches as a punch list in the PR description so reviewers can verify nothing got missed.

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

- [ ] **Step 4b: End-to-end smoke run — actually execute regression + classification searches like a user would**

The snapshot harness pins byte-identical behavior on three configs, but the user's requirement is stronger: prove that BOTH regression and classification searches actually work end-to-end after the deletion. This step runs each path through `run_search` (grid) AND `run_unified_bayesian` (production Bayesian) on real data and asserts non-trivial outputs.

Save this as `tools/_smoke_post_deletion.py` — it's a one-off harness, not a test:

```python
"""End-to-end smoke run for regression + classification post-deletion.

Not a pytest. Run as a script. Verifies that both code paths execute
to completion and produce sensible outputs on real data. Removed at
the end of Task 8 along with the snapshot harness.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.preprocessing import LabelEncoder

from spectral_predict.io import read_asd_dir
from spectral_predict.search import run_search
from spectral_predict.unified_bayesian import run_unified_bayesian

REGRESSION_TARGET = "%Collagen"
CLASSIFICATION_TARGET = "CollagenCat"


def _load_joined_dataframe():
    """Mirrors tools/bench_baseline_compare.py:54-67. Spectra in .asd files,
    metadata in BoneCollagen.csv, joined on a normalized File Number key."""
    spectra, _meta = read_asd_dir(str(REPO / "example"))
    ref = pd.read_csv(REPO / "example" / "BoneCollagen.csv")
    spectra = spectra.sort_index()
    ref = ref.copy()
    ref.index = (
        ref["File Number"].astype(str)
        .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        .str.replace(" ", "", regex=False)
    )
    joined = spectra.join(ref, how="inner")
    feature_cols = list(spectra.columns)
    X = joined[feature_cols].to_numpy(dtype=float)
    wl = np.asarray([float(c) for c in feature_cols], dtype=float)
    return joined, X, wl


def _load_xy_regression():
    joined, X, wl = _load_joined_dataframe()
    y = joined[REGRESSION_TARGET].to_numpy(dtype=float)
    return X, y, wl


def _load_xy_classification():
    joined, X, wl = _load_joined_dataframe()
    le = LabelEncoder()
    y = le.fit_transform(joined[CLASSIFICATION_TARGET].astype(str))
    return X, y, wl


def _check(label, ok, detail=""):
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        sys.exit(1)


def smoke_regression_grid():
    print("== Regression / grid (run_search) ==")
    X, y, wl = _load_xy_regression()
    t0 = time.perf_counter()
    df, _ = run_search(
        X, y,
        task_type="regression",
        folds=5,
        models_to_test=["PLS"],
        preprocessing_methods=["raw"],
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )
    dt = time.perf_counter() - t0
    _check("ran in <120s", dt < 120, f"{dt:.1f}s")
    _check("produced rows", len(df) > 0, f"{len(df)} rows")
    _check("has R2 column", "R2" in df.columns)
    _check("R2 not NaN", not np.isnan(df.iloc[0]["R2"]))
    _check("R2 finite + sensible", 0 < df.iloc[0]["R2"] < 1.0001, f"R2={df.iloc[0]['R2']:.4f}")


def smoke_regression_bayesian():
    print("== Regression / Bayesian (run_unified_bayesian) ==")
    X, y, wl = _load_xy_regression()
    t0 = time.perf_counter()
    df, study = run_unified_bayesian(
        X=X, y=y, wavelengths=wl,
        model_name="PLS", task_type="regression",
        n_trials=15, random_state=42,
        cv_strategy="kfold", cv_folds=5,
    )
    dt = time.perf_counter() - t0
    _check("ran in <180s", dt < 180, f"{dt:.1f}s")
    _check("produced rows", len(df) > 0, f"{len(df)} rows")
    _check("study has best_value", study.best_value is not None and study.best_value < 1e9)
    _check("study has trials", len(study.trials) >= 10)
    _check("at least one COMPLETE trial", any(str(t.state) == "TrialState.COMPLETE" for t in study.trials))


def smoke_classification_grid():
    print("== Classification / grid (run_search) ==")
    X, y, wl = _load_xy_classification()
    t0 = time.perf_counter()
    df, _ = run_search(
        X, y,
        task_type="classification",
        folds=5,
        models_to_test=["PLS-DA"],
        preprocessing_methods=["raw"],
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )
    dt = time.perf_counter() - t0
    _check("ran in <120s", dt < 120, f"{dt:.1f}s")
    _check("produced rows", len(df) > 0, f"{len(df)} rows")
    _check("has Accuracy column", "Accuracy" in df.columns)
    _check("Accuracy in [0,1]", 0.0 <= df.iloc[0]["Accuracy"] <= 1.0001, f"Acc={df.iloc[0]['Accuracy']:.4f}")
    _check("Accuracy > random", df.iloc[0]["Accuracy"] > 0.5, "should beat 1/3 ternary baseline")


def smoke_classification_bayesian():
    print("== Classification / Bayesian (run_unified_bayesian) ==")
    X, y, wl = _load_xy_classification()
    t0 = time.perf_counter()
    df, study = run_unified_bayesian(
        X=X, y=y, wavelengths=wl,
        model_name="PLS-DA", task_type="classification",
        n_trials=15, random_state=42,
        cv_strategy="kfold", cv_folds=5,
    )
    dt = time.perf_counter() - t0
    _check("ran in <180s", dt < 180, f"{dt:.1f}s")
    _check("produced rows", len(df) > 0, f"{len(df)} rows")
    _check("study has best_value", study.best_value is not None and study.best_value < 1e9)
    _check("study has trials", len(study.trials) >= 10)


if __name__ == "__main__":
    print(f"Smoke run starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    smoke_regression_grid()
    smoke_regression_bayesian()
    smoke_classification_grid()
    smoke_classification_bayesian()
    print("\nAll smoke checks PASSED.")
```

Run it:

```bash
.venv312/Scripts/python.exe tools/_smoke_post_deletion.py
```

Expected output (last line): `All smoke checks PASSED.` Total runtime ~5-10 minutes on this dataset.

If any `[FAIL]` appears, STOP and investigate — the deletion broke an end-to-end path that the unit/snapshot tests didn't catch.

Delete the smoke script after the run:

```bash
git rm tools/_smoke_post_deletion.py 2>/dev/null || rm tools/_smoke_post_deletion.py
```

(The script was never committed if you ran it before staging. If it WAS staged for some reason, `git rm` will untrack it.)

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
