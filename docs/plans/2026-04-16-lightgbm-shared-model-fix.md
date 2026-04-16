# LightGBM Shared-Model-State Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the shared-`model` state leak in `search.py` that causes `LGBMRegressor` to raise `"X has N features, but LGBMRegressor is expecting 2135 features as input"` during grid search with variable + region subsets enabled on sklearn 1.5.2 (bundled `.venv311`).

**Architecture:** Two-site minimal fix — wrap the shared `model` in `sklearn.base.clone()` at the importance-capture pipe construction (`search.py:2191`) and at the `_run_single_config` pipe construction (`search.py:4161`, `:4163`). `clone` is already imported at `search.py:19`. Verify with a before/after harness that runs LightGBM + PLS grid searches on the BoneCollagen dataset (`example/`) against **both** `.venv311` (sklearn 1.5.2, where bug triggers) and `.venv312` (sklearn 1.7.2, where bug is latent).

**Tech Stack:** Python 3.11/3.12, scikit-learn 1.5.2/1.7.2, LightGBM 4.6.0, existing `spectral_predict.search.run_search`, `spectral_predict.io.read_asd_dir`, BoneCollagen example data.

---

## Background

- **Root cause:** at `search.py:2191`, `model` (a single instance) is put into a Pipeline and fit on the full preprocessed X (e.g. 2135 features). The fit leaves `model.n_features_in_ = 2135`. Later, `_run_single_config` reuses that **same** `model` reference in a new pipeline and calls `pipe.fit(X_subset, y)` on a 10/20/50/… feature slice. sklearn 1.5.2's pre-fit validation runs `_check_n_features(reset=False)` **before** the fit body would reset `n_features_in_`, so it raises. sklearn 1.7.2 relaxed that pre-fit check and the bug is silently tolerated.
- **Scope of impact:** confirmed LightGBM (user reproed). Any estimator with sklearn's strict `_check_n_features` path is vulnerable in principle (LightGBM's sklearn wrapper inherits it). PLS is historically robust — included in verification as a control to confirm we don't perturb its metrics.
- **Why shipping without fix is unsafe:** the bundled PyInstaller app runs `.venv311`. Users would see NaN calibration metrics + stdout warnings for LightGBM whenever variable subsets + region subsets are both enabled (GUI default).
- **Evidence / prior artifacts:** `docs/SESSION_LOG.md` entry `2026-04-16`, `docs/pr4_parity/repro_lightgbm_regression_v2.py` (on `cv-strategy-overhaul` worktree).

---

## Fix sites (pre-verified by reading file)

| Location | Current code | Fix |
|---|---|---|
| `search.py:2191` (inside importance-capture block) | `pipe_steps.append(("model", model))` | `pipe_steps.append(("model", clone(model)))` |
| `search.py:4161` (scale-sensitive branch in `_run_single_config`) | `pipe_steps.append(("model", model))` | `pipe_steps.append(("model", clone(model)))` |
| `search.py:4163` (default branch in `_run_single_config`) | `pipe_steps.append(("model", model))` | `pipe_steps.append(("model", clone(model)))` |

`clone` is imported at `search.py:19` (`from sklearn.base import clone`). No new imports needed.

**Deliberately NOT touching:**
- `search.py:4136` (`pipe_steps.append(("pls", model))` for PLS-DA) — different code path, not involved in the reported failure. Left alone to keep diff minimal and reviewable. Can be follow-up.
- Any other `pipe.fit(model)` site — not in the pathway to the observed bug.

---

## Verification matrix (4 runs)

Each run executes the same harness script against the same dataset (`example/` BoneCollagen + ASD files), capturing a deterministic JSON summary.

| # | Venv | sklearn | Stage | Expectation |
|---|---|---|---|---|
| 1 | `.venv311` | 1.5.2 | **before** fix | LightGBM has NaN calibration rows + feature-count warnings. PLS clean. |
| 2 | `.venv312` | 1.7.2 | **before** fix | Both models clean (bug is latent). |
| 3 | `.venv311` | 1.5.2 | **after** fix | Both models clean. Zero feature-count warnings. |
| 4 | `.venv312` | 1.7.2 | **after** fix | Both models clean. Metrics match run #2 within tolerance. |

Tolerance rule: CV RMSE / R² should match to 1e-9 across before/after runs within the same venv (since fit() already resets `n_features_in_`, `clone()` shouldn't change numerics unless there was an actual stale-state effect). We'll sanity-check this in the report.

---

## Task 1: Create fresh branch off main

**Files:** none modified yet.

**Step 1:** Verify clean baseline.
```bash
git status
git log --oneline -3
```
Expected: on `main`, HEAD at `b2cc3ed`, only `.claude/settings.local.json` + `tests/test_cv_strategy.py` unstaged (as per session start).

**Step 2:** Create and switch to new branch.
```bash
git checkout -b fix/lightgbm-shared-model-state
```
Expected: "Switched to a new branch 'fix/lightgbm-shared-model-state'".

No commit yet.

---

## Task 2: Write the verification harness

**Files:**
- Create: `scripts/verify_shared_model_fix.py`

**Step 1:** Write the harness.

The harness must:
1. Load the BoneCollagen dataset using `read_asd_dir(example/)` and join to `example/BoneCollagen.csv` on `File Number`. Target = `%Collagen` (regression, 49 samples).
2. Call `run_search(X, y, task_type='regression', ...)` twice — once with `models_to_test=['LightGBM']`, once with `models_to_test=['PLS']`. Keep all other kwargs at the values that correspond to GUI defaults:
   - `folds=5`
   - `preprocessing_methods={'raw': False, 'snv': True, 'sg1': True, 'sg2': True, 'sg3': False, 'sg4': False, 'deriv_snv': True}`  ← matches **GUI** defaults at `gui:2833-2839` (not `run_search`'s internal defaults, which differ)
   - `window_sizes=[17]`  ← GUI default at `gui:2867` (only window_17 checked)
   - `enable_variable_subsets=True`
   - `enable_region_subsets=True`
   - `variable_counts=[10, 20, 50, 100, 250]`  ← GUI default at `gui:2856-2862` (500/1000 off by default)
   - `variable_selection_methods=['importance']`
3. Redirect stdout through `contextlib.redirect_stdout(io.StringIO())` so the harness can count `"parameter capture"` warning lines.
4. Emit a JSON record to stdout with, per model:
   - `n_rows`: total result rows
   - `n_nan_cal_rmse`: count of rows where `RMSE` is NaN / non-finite
   - `n_nan_cv_rmse`: count of rows where `RMSEcv` is NaN / non-finite
   - `warning_feature_mismatch_count`: count of `"parameter capture"` lines in stdout
   - `best_cv_rmse`: min finite `RMSEcv`
   - `best_cal_rmse`: min finite `RMSE`
   - `median_cv_rmse`, `median_cal_rmse` (for parity tolerance checks)
5. Also record: `sklearn_version`, `lightgbm_version`, `python_version`, `git_sha`.
6. Exit 1 if any of the following are true for any model — makes it usable as a gate:
   - `error is not None` (run crashed, caught by outer try)
   - `n_rows == 0` (no result rows — silent total failure)
   - `warning_feature_mismatch_count > 0`
   - `n_nan_cal_rmse > 0`
   - `n_nan_cv_rmse > 0` (a broken run can manifest here even if calibration stays finite)

Seed everything: `random_state=42` passed through `run_search`.

Full code:
```python
"""
Verify the LightGBM shared-model-state fix.

Runs run_search with GUI defaults on BoneCollagen (example/) for both
LightGBM and PLS. Captures calibration / CV metrics and counts
"parameter capture" warnings. Emits JSON to stdout.

Use via:
    .venv311/Scripts/python.exe scripts/verify_shared_model_fix.py > out.json
    .venv312/Scripts/python.exe scripts/verify_shared_model_fix.py > out.json

Exit 0 if clean (no NaN cal, no feature-mismatch warnings). Exit 1 if bug present.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXAMPLE_DIR = REPO_ROOT / "example"


def load_bone_collagen():
    from spectral_predict.io import read_asd_dir

    spectra, meta = read_asd_dir(EXAMPLE_DIR)
    ref = pd.read_csv(EXAMPLE_DIR / "BoneCollagen.csv", encoding="utf-8-sig")

    # Join spectra (index = filename stems like "Spectrum00001") to ref ("File Number" = "Spectrum 00001")
    ref["__key__"] = ref["File Number"].str.replace(" ", "", regex=False)
    spectra.index = spectra.index.str.replace(".asd", "", regex=False).str.replace(".spc", "", regex=False)
    joined = spectra.join(ref.set_index("__key__")[["%Collagen"]], how="inner")
    joined = joined.dropna(subset=["%Collagen"])

    y = joined["%Collagen"].astype(float)
    X = joined.drop(columns=["%Collagen"]).astype(float)
    return X, y


def _finite_count(series):
    if series is None or len(series) == 0:
        return 0
    return int(series.apply(lambda v: isinstance(v, (int, float, np.integer, np.floating))
                                     and not (np.isnan(float(v)) or np.isinf(float(v)))).sum())


def _nan_count(series):
    return int(len(series) - _finite_count(series))


def run_one_model(model_name, X, y):
    from spectral_predict.search import run_search

    kwargs = dict(
        folds=5,
        models_to_test=[model_name],
        # GUI defaults — spectral_predict_gui_optimized.py:2833-2839
        preprocessing_methods={
            "raw": False,
            "snv": True,
            "sg1": True,
            "sg2": True,
            "sg3": False,
            "sg4": False,
            "deriv_snv": True,
        },
        window_sizes=[17],  # GUI default: only window_17 checked (gui:2867)
        enable_variable_subsets=True,
        enable_region_subsets=True,
        variable_counts=[10, 20, 50, 100, 250],  # GUI default (gui:2856-2862)
        variable_selection_methods=["importance"],
        random_state=42,
    )

    captured = io.StringIO()
    error = None
    df_out = None
    try:
        with redirect_stdout(captured):
            df_out, _ = run_search(X, y, task_type="regression", **kwargs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    stdout_text = captured.getvalue()
    warning_lines = [L for L in stdout_text.splitlines() if "parameter capture" in L]

    record = {
        "model": model_name,
        "error": error,
        "warning_feature_mismatch_count": len(warning_lines),
        "warning_sample": warning_lines[:5],
    }

    if df_out is not None and len(df_out):
        rmse = df_out.get("RMSE")
        rmsecv = df_out.get("RMSEcv")
        record["n_rows"] = int(len(df_out))
        record["n_nan_cal_rmse"] = _nan_count(rmse) if rmse is not None else None
        record["n_nan_cv_rmse"] = _nan_count(rmsecv) if rmsecv is not None else None

        if rmse is not None:
            finite_rmse = rmse[rmse.apply(
                lambda v: isinstance(v, (int, float, np.integer, np.floating))
                          and not (np.isnan(float(v)) or np.isinf(float(v))))]
            record["best_cal_rmse"] = float(finite_rmse.min()) if len(finite_rmse) else None
            record["median_cal_rmse"] = float(finite_rmse.median()) if len(finite_rmse) else None
        if rmsecv is not None:
            finite_cv = rmsecv[rmsecv.apply(
                lambda v: isinstance(v, (int, float, np.integer, np.floating))
                          and not (np.isnan(float(v)) or np.isinf(float(v))))]
            record["best_cv_rmse"] = float(finite_cv.min()) if len(finite_cv) else None
            record["median_cv_rmse"] = float(finite_cv.median()) if len(finite_cv) else None

    return record


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def main():
    import sklearn, lightgbm, platform

    X, y = load_bone_collagen()
    print(f"[verify] dataset X={X.shape}, y={y.shape}", file=sys.stderr)

    lgbm_rec = run_one_model("LightGBM", X, y)
    pls_rec = run_one_model("PLS", X, y)

    result = {
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "git_sha": git_sha(),
        },
        "LightGBM": lgbm_rec,
        "PLS": pls_rec,
    }

    bug_present = any(
        (r.get("error") is not None)
        or ((r.get("n_rows") or 0) == 0)
        or ((r.get("warning_feature_mismatch_count") or 0) > 0)
        or ((r.get("n_nan_cal_rmse") or 0) > 0)
        or ((r.get("n_nan_cv_rmse") or 0) > 0)
        for r in (lgbm_rec, pls_rec)
    )

    print(json.dumps(result, indent=2, default=str))
    sys.exit(1 if bug_present else 0)


if __name__ == "__main__":
    main()
```

**Step 2:** Sanity-check the dataset loader only (don't run the full harness yet).

Run:
```bash
.venv311/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); from verify_shared_model_fix import load_bone_collagen; X, y = load_bone_collagen(); print('X:', X.shape, 'y:', y.shape, 'y.dtype:', y.dtype, 'n_nan_y:', y.isna().sum())"
```
Expected: `X: (49, N)` where N is the wavelength count, `y: (49,)`, `y.dtype: float64`, `n_nan_y: 0`.

**Step 3:** Commit the harness on its own (so we can run it at this exact SHA before the fix).
```bash
git add scripts/verify_shared_model_fix.py
git commit -m "test: add shared-model-state verification harness"
```

---

## Task 3: Baseline run — `.venv311` before fix

**Files:**
- Create: `docs/plans/artifacts/2026-04-16/baseline_venv311.json`

**Step 1:** Make the artifacts dir.
```bash
mkdir -p docs/plans/artifacts/2026-04-16
```

**Step 2:** Run baseline.
```bash
.venv311/Scripts/python.exe scripts/verify_shared_model_fix.py > docs/plans/artifacts/2026-04-16/baseline_venv311.json
echo "exit=$?"
```
Expected: `exit=1` (bug present). JSON shows LightGBM has `warning_feature_mismatch_count > 0` AND/OR `n_nan_cal_rmse > 0`. PLS should be clean (no warnings, no NaN cal). Env shows sklearn 1.5.2, python 3.11.

**Step 3:** If the bug did NOT appear on `.venv311`, **STOP**. This means our GUI-defaults harness didn't reproduce the user's failure mode and we need to tighten it before proceeding. Report to user and investigate.

---

## Task 4: Baseline run — `.venv312` before fix

**Files:**
- Create: `docs/plans/artifacts/2026-04-16/baseline_venv312.json`

**Step 1:** Run.
```bash
.venv312/Scripts/python.exe scripts/verify_shared_model_fix.py > docs/plans/artifacts/2026-04-16/baseline_venv312.json
echo "exit=$?"
```
Expected: `exit=0` (clean). LightGBM and PLS both have `warning_feature_mismatch_count == 0` and `n_nan_cal_rmse == 0`. Env shows sklearn 1.7.2, python 3.12.

---

## Task 5: Apply the clone fixes

**Files:**
- Modify: `src/spectral_predict/search.py:2191`, `:4161`, `:4163`

**Step 1:** Three exact edits. `old_string` must match the current line context exactly — include 2–3 surrounding lines to guarantee uniqueness.

Edit 1 (line 2191, inside importance-capture):
```python
# Build model-only pipeline (data is already preprocessed and filtered)
pipe_steps = []
pipe_steps.append(("model", model))
pipe = Pipeline(pipe_steps)
```
→
```python
# Build model-only pipeline (data is already preprocessed and filtered)
pipe_steps = []
pipe_steps.append(("model", clone(model)))
pipe = Pipeline(pipe_steps)
```

Edit 2 (line 4161, scale-sensitive branch in `_run_single_config`):
```python
    elif model_name in SCALE_SENSITIVE_MODELS:
        pipe_steps.append(("scaler", StandardScaler()))
        pipe_steps.append(("model", model))
    else:
        pipe_steps.append(("model", model))
```
→
```python
    elif model_name in SCALE_SENSITIVE_MODELS:
        pipe_steps.append(("scaler", StandardScaler()))
        pipe_steps.append(("model", clone(model)))
    else:
        pipe_steps.append(("model", clone(model)))
```

(Single Edit with enough context covers both lines 4161 and 4163 at once.)

**Step 2:** Verify the change compiles / imports.
```bash
.venv311/Scripts/python.exe -c "from spectral_predict import search; print('ok')"
```
Expected: `ok`.

**Step 3:** Confirm no other shared-model occurrences slipped in.
```bash
```
Use Grep tool for `pipe_steps\.append\(\(\"model\", model\)\)` in `src/spectral_predict/search.py`.
Expected: zero matches. Only `clone(model)` variants should remain.

**Step 4:** Commit.
```bash
git add src/spectral_predict/search.py
git commit -m "fix: clone shared model estimator in search pipelines

LGBMRegressor raised 'X has N features, but LGBMRegressor is expecting 2135'
on sklearn 1.5.2 (bundled .venv311) when variable + region subsets were both
enabled. Root cause: the importance-capture pipe at search.py:2191 fit a
shared model instance on the full preprocessed X (leaving n_features_in_=2135),
and _run_single_config reused that same instance in subset-feature fits.
sklearn 1.5.2's pre-fit _check_n_features(reset=False) fires before the fit
body resets n_features_in_, raising; sklearn 1.7.2 relaxed this.

Wrap model in clone() at all three pipe construction sites. Verified on
.venv311 (1.5.2) + .venv312 (1.7.2) before/after via
scripts/verify_shared_model_fix.py."
```

---

## Task 6: Post-fix run — `.venv311` after fix

**Files:**
- Create: `docs/plans/artifacts/2026-04-16/postfix_venv311.json`

**Step 1:** Run.
```bash
.venv311/Scripts/python.exe scripts/verify_shared_model_fix.py > docs/plans/artifacts/2026-04-16/postfix_venv311.json
echo "exit=$?"
```
Expected: `exit=0`. LightGBM `warning_feature_mismatch_count == 0`, `n_nan_cal_rmse == 0`.

**Step 2:** If still failing, **STOP** and investigate (unexpected — means there's another shared-state site we missed).

---

## Task 7: Post-fix run — `.venv312` after fix

**Files:**
- Create: `docs/plans/artifacts/2026-04-16/postfix_venv312.json`

**Step 1:** Run.
```bash
.venv312/Scripts/python.exe scripts/verify_shared_model_fix.py > docs/plans/artifacts/2026-04-16/postfix_venv312.json
echo "exit=$?"
```
Expected: `exit=0`.

---

## Task 8: Compare the 4-run matrix + commit verification artifacts

**Files:**
- Create: `docs/plans/artifacts/2026-04-16/COMPARISON.md`

**Step 1:** Write a short comparison document.

For each venv, compare `baseline_*.json` vs `postfix_*.json`:
- `.venv311`:
  - LightGBM: `warning_feature_mismatch_count` dropped from >0 to 0. `n_nan_cal_rmse` dropped from >0 to 0. `best_cv_rmse` / `median_cv_rmse` should be **identical or very close** to baseline (CV path doesn't touch the broken importance refit, so it shouldn't move).
  - PLS: metrics should be identical before/after (no change expected).
- `.venv312`:
  - LightGBM + PLS: all numeric metrics should match baseline to within floating-point tolerance (since bug was latent, `clone()` shouldn't change behavior).

Flag anything that shifts more than 1e-6 on CV RMSE or 1e-6 on median cal RMSE for PLS — that would be an unexpected regression worth investigating.

**Step 2:** Commit.
```bash
git add docs/plans/artifacts/2026-04-16/
git commit -m "test: add before/after verification artifacts for shared-model fix"
```

---

## Task 9: Update living docs

**Files:**
- Modify: `docs/PROJECT_STATUS.md` — remove/collapse the 🔴 PRIORITY FOR NEXT SESSION section; add a one-liner to "What Works" acknowledging the fix.
- Modify: `docs/SESSION_LOG.md` — append a 2026-04-16 resolution entry pointing to the commit SHA and the artifacts dir.
- Modify: `CLAUDE.md` — remove the 🔴 TOP PRIORITY block at top.

**Step 1:** Edit PROJECT_STATUS.md top section: replace the "🔴 PRIORITY FOR NEXT SESSION" block with a dated "Recently resolved" note pointing to the fix SHA and the verification artifacts.

**Step 2:** Append to SESSION_LOG.md under the 2026-04-16 entry:
```
**Fix shipped 2026-04-16 (post-investigation):** two sites wrapped in `clone()` on branch `fix/lightgbm-shared-model-state`. Verified before/after on both venvs. See `docs/plans/artifacts/2026-04-16/COMPARISON.md`.
```

**Step 3:** Remove the 🔴 TOP PRIORITY block at the top of `CLAUDE.md` (the whole "added 2026-04-16" section).

**Step 4:** Commit.
```bash
git add docs/PROJECT_STATUS.md docs/SESSION_LOG.md CLAUDE.md
git commit -m "docs: close LightGBM shared-model fix — verified on both venvs"
```

---

## Non-goals / explicitly out of scope

- PLS-DA pipeline site (`search.py:4136`) — not involved in the reported bug.
- `run_one_class_search` / `unified_bayesian` — not involved; one-class flow already uses fresh estimators in its CV loop.
- Any refactor of `_run_single_config`'s `try/except Exception` swallow at `~:4437` — that's a correctness amplifier, not the root cause. Flagged as a follow-up in `docs/SESSION_LOG.md` if we ever want to surface swallowed exceptions to the GUI progress tab.

---

## Open questions for reviewer (Codex)

1. Am I missing any additional pipe-construction sites in `search.py` where the same shared-model mutation could occur (e.g. during ensemble fits, transfer-learning paths, or secondary refinement)? A grep of `pipe_steps.append(("model", model))` shows only the three sites above — is that exhaustive given actual call graphs?
2. Is `clone(model)` safe for every estimator we register in `models_to_test`? In particular: does any custom estimator in `src/spectral_predict/models.py` break under `sklearn.base.clone` (non-standard `__init__`, non-estimator members, etc.)?
3. Is the harness's "GUI defaults" set actually equivalent to what a user sees when they click Run with no config changes? Specifically: does `run_search` default `tier` (Quick/Standard/Comprehensive) matter here, and should the harness pin it?
4. Is the harness's pass/fail gate strict enough? It checks NaN cal RMSE and stdout warnings — is there another silent-failure mode (e.g. CV metrics also NaN, or exit without raising) worth adding?
