# Validation Prompt: PR #4 Numeric Parity vs `main` (Overnight / Autonomous)

> Hand this to a fresh Claude Code agent in the `dasp` repo. **This runs overnight without user supervision — do not ask the user any clarifying questions, do not request new permissions, and do not push to remote. Commit locally only. The user will review the report in the morning.**

---

## Operating constraints

1. **Do not push to remote.** `git commit` locally is fine; the user will `git push` after reviewing.
2. **Do not modify source code in `src/` or `spectral_predict_gui_optimized.py`.** This is a validation pass, not a fix pass. You may add tests under `tests/` and a script/report under `docs/`.
3. **Do not run the GUI or anything requiring a display.** All comparisons go through `spectral_predict.search.run_search` (programmatic API).
4. **Avoid unusual shell tools.** Use only: `python`, `python -m pytest`, `git`, `Read`, `Write`, `Edit`, `Bash` for `python`/`git`. No `pip install` of new packages — everything you need is already installed.
5. **If a step fails and you cannot proceed, write a partial report to `docs/pr4_parity_report.md` documenting exactly where you stopped and why.** Don't skip silently.
6. **Budget:** spend up to ~60 minutes of wall time on fits. LightGBM on 49 samples with 5-fold CV is under a second per fit; if anything hangs for more than a few minutes, kill it and note in the report.

---

## Your task

Validate that commit `c091c93` on branch `claude/cv-strategy-overhaul` is numerically safe to merge into `main`. Three comparisons:

| Test | Expectation |
|---|---|
| K-Fold (folds=5) on `main` vs same on branch | CV metrics nearly identical (regression: bit-identical; classification: ≤1e-6). **Calibration metrics must be bit-identical** — calibration doesn't touch CV plumbing. |
| LOO on branch (main doesn't have LOO) | Numbers must be sensible given the dataset. Not expected to match K-Fold. |
| All four (model, task) combinations | PLS-regression, PLS-DA-classification, LightGBM-regression, LightGBM-classification |

---

## Setup

- Main repo: `C:\Users\sponheim\git\dasp` (currently on `main`)
- Branch worktree: `C:\Users\sponheim\git\dasp\.worktrees\cv-strategy-overhaul` (on `claude/cv-strategy-overhaul` @ `c091c93`)
- Dataset: `example/BoneCollagen.csv` + `example/Spectrum00001.asd` through `Spectrum00049.asd` (49 samples). Exists in both checkouts.

**Critical gotcha:** The editable pip install roots at the MAIN repo's `src/`. A plain `python -c "from spectral_predict import ..."` from inside the worktree still imports MAIN's code. To run the worktree's code, use one of:
- `cd <worktree> && python -m pytest <test-file>` (pytest respects `pyproject.toml`'s `pythonpath=["src"]`)
- `cd <worktree> && PYTHONPATH=./src python <script>` (explicit override)

**Verify which code you're running before every test run** with this one-liner (prints `BRANCH` if running branch code, `MAIN` otherwise):
```bash
cd <location> && PYTHONPATH=./src python -c "
try:
    from spectral_predict.cv_utils import _is_repeated_cv, reduce_repeated_cv_predictions
    print('BRANCH')
except ImportError:
    print('MAIN')
"
```

---

## Step 1 — Write the reproducer script

Create `docs/pr4_parity/run_parity.py` in the branch worktree. Single script, no CLI args, produces one JSON output file. Pseudocode:

```python
"""
Run 4 (model, task) combinations with kfold=5 and dump headline metrics.
Output: docs/pr4_parity/results_{location}_{strategy}.json
where location is 'main' or 'branch' (inferred from import probe) and
strategy is 'kfold' or 'loo'.
"""
import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Import probe — establishes whether we're running main or branch code
REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from spectral_predict.cv_utils import _is_repeated_cv  # noqa: F401
    LOCATION = 'branch'
    # Branch signature:
    RUN_KWARGS_EXTRA = {'cv_strategy': os.environ.get('CV_STRATEGY', 'kfold'),
                        'cv_n_repeats': 1}
except ImportError:
    LOCATION = 'main'
    RUN_KWARGS_EXTRA = {}  # main signature doesn't accept these

CV_STRATEGY = RUN_KWARGS_EXTRA.get('cv_strategy', 'kfold')

from spectral_predict.io import read_asd_dir
from spectral_predict.search import run_search

# Load dataset (mirrors tests/gui/conftest.py:_load_example_data)
data_path = REPO_ROOT / 'example'
ref_df = pd.read_csv(data_path / 'BoneCollagen.csv', encoding='utf-8-sig')
result = read_asd_dir(str(data_path))
X = result[0] if isinstance(result, tuple) else result
X.index = [idx.replace('Spectrum', 'Spectrum ') if idx.startswith('Spectrum') else idx
           for idx in X.index]
ref_df['File Number'] = ref_df['File Number'].str.strip()
X.index = X.index.str.strip()
common = X.index.intersection(ref_df.set_index('File Number').index)
X = X.loc[common]
ref = ref_df.set_index('File Number').loc[common]
y_reg = ref['%Collagen'].astype(float)
y_cls = ref['CollagenCat'].astype(str)

def run_one(model, task):
    y = y_reg if task == 'regression' else y_cls
    kwargs = dict(
        folds=5, models_to_test=[model],
        preprocessing_methods={'raw': True},
        enable_variable_subsets=False, enable_region_subsets=False,
    )
    kwargs.update(RUN_KWARGS_EXTRA)
    df, _ = run_search(X, y, task_type=task, **kwargs)
    if len(df) == 0:
        return {'_error': 'empty results'}
    row = df.iloc[0].to_dict()
    # Keep only finite numeric values; stringify others
    out = {}
    for k, v in row.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v) if not np.isnan(v) else None
    return out

combos = [('PLS', 'regression'), ('PLS-DA', 'classification'),
          ('LightGBM', 'regression'), ('LightGBM', 'classification')]
results = {f'{m}__{t}': run_one(m, t) for m, t in combos}

out_dir = REPO_ROOT / 'docs' / 'pr4_parity'
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f'results_{LOCATION}_{CV_STRATEGY}.json'
out_file.write_text(json.dumps({'location': LOCATION, 'strategy': CV_STRATEGY,
                                 'results': results}, indent=2))
print(f'Wrote {out_file}')
```

**Place it in `docs/pr4_parity/run_parity.py`** (both branches so it works when run from either).

---

## Step 2 — Run on main

The main repo at `C:\Users\sponheim\git\dasp` is on `main`. Copy the script there too (or use absolute path):

```bash
cd C:/Users/sponheim/git/dasp
# Copy the script into main
mkdir -p docs/pr4_parity
cp .worktrees/cv-strategy-overhaul/docs/pr4_parity/run_parity.py docs/pr4_parity/

# Run it (main code path — no PYTHONPATH override needed since editable install points here)
python docs/pr4_parity/run_parity.py
# Should print: Wrote .../docs/pr4_parity/results_main_kfold.json

# Copy the result back to the branch worktree for comparison
cp docs/pr4_parity/results_main_kfold.json .worktrees/cv-strategy-overhaul/docs/pr4_parity/

# Clean up main (don't commit anything there)
rm -rf docs/pr4_parity
```

---

## Step 3 — Run on branch (K-Fold)

```bash
cd C:/Users/sponheim/git/dasp/.worktrees/cv-strategy-overhaul
CV_STRATEGY=kfold PYTHONPATH=./src python docs/pr4_parity/run_parity.py
# Should print: Wrote .../docs/pr4_parity/results_branch_kfold.json
```

---

## Step 4 — Run on branch (LOO)

```bash
cd C:/Users/sponheim/git/dasp/.worktrees/cv-strategy-overhaul
CV_STRATEGY=loo PYTHONPATH=./src python docs/pr4_parity/run_parity.py
# Should print: Wrote .../docs/pr4_parity/results_branch_loo.json
```

Note: LOO with 49 samples × 4 models × 1 preprocessing = 196 fits. Should be fast for PLS, slow-ish for LightGBM (~30-60s total). If LightGBM hangs, it's fine to skip it for LOO and note in the report.

---

## Step 5 — Write the comparison report

Create `docs/pr4_parity_report.md`. Structure:

```markdown
# PR #4 Parity Validation — `claude/cv-strategy-overhaul` vs `main`

**Branch SHA:** <c091c93 at time of test>
**Main SHA:** <main SHA at time of test — get via `cd C:/Users/sponheim/git/dasp && git rev-parse main`>
**Date:** <UTC timestamp>
**Python:** <from `python --version`>
**Key deps:** <sklearn, numpy, lightgbm versions from `python -c "import sklearn,numpy,lightgbm; print(sklearn.__version__, numpy.__version__, lightgbm.__version__)"`>

## K-Fold parity (main vs branch)

| Model × Task | Metric | main | branch | abs diff | status |
|---|---|---|---|---|---|
| PLS × regression | RMSE (cal) | ... | ... | ... | PASS/FAIL |
| PLS × regression | RMSEcv | ... | ... | ... | PASS/FAIL |
| ... |

**Calibration drift (MUST be 0):** <summary — any nonzero diff in cal metrics is a BLOCKER>

**CV drift (regression expect 0, classification expect ≤1e-6):** <summary>

## LOO sanity (branch only)

| Model × Task | RMSEcv / Acccv | Plausible? | Notes |
|---|---|---|---|
| PLS × regression | ... | YES/NO | ... |
| ... |

## Verdict

- [ ] PASS — cal identical, CV within tolerance, LOO sensible → merge
- [ ] PARTIAL — diffs found but explainable by documented semantic changes (BER/pooled specificity under repeated-CV only); plain K-Fold unaffected → merge with release note
- [ ] FAIL — unexplained drift or LOO produces nonsense → do not merge, report specifics below

### If FAIL, specifics:
- Exact metric, model, task
- Numeric values from both branches
- Hypothesis for root cause
- Attached raw JSON excerpts
```

---

## Step 6 — Commit (NO push)

```bash
cd C:/Users/sponheim/git/dasp/.worktrees/cv-strategy-overhaul
git add docs/pr4_parity/ docs/pr4_parity_report.md
git status  # confirm only added files, nothing modified in src/
git commit -m "docs: PR #4 numeric parity validation report vs main

See docs/pr4_parity_report.md for full comparison table and verdict.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
# Do NOT push — user will review first.
```

---

## Known acceptable / expected diffs

- **Plain K-Fold regression headline metrics** should be bit-identical. The round-2 commit's pooled-per-sample reduction is a no-op under non-repeated CV (each sample appears in exactly one test fold).
- **Plain K-Fold classification `Accuracycv` / `Kappacv` / `MCCcv`** should be bit-identical for the same reason.
- **Plain K-Fold classification `F1cv` / `Precisioncv` / `Recallcv`** may differ at ~1e-6 because the round-2 commit switched the pooled-repeated-CV path to use explicit `average='binary' if is_binary else 'macro'` (for 3-class CollagenCat, macro). If main was using a different average mode (check main's `_run_single_fold` for the F1 call), diffs could be larger. Document whatever you find.
- **Plain K-Fold classification `Specificitycv`** may differ at ~1e-6 because the round-3 commit added `labels=np.unique(y)` to the repeated-CV pooled-specificity path (but not the plain K-Fold path). Expected equal.
- **BER under plain K-Fold** should be mean-of-fold-BERs on both branches (branch only changed repeated-CV BER). Expect equal.

## If you find non-determinism

- Check LightGBM: add `n_jobs=1, random_state=42` via the `run_search` kwargs (look at `run_search` signature for `lightgbm_*` params).
- Check Python/numpy/sklearn versions match between main and branch runs (they should, same machine, same venv).
- Re-run once; if results differ on re-run, non-determinism is real and is itself a concern to report (not specific to this PR).

## Failure-mode fallbacks

- Script crashes on main: `run_search` on main may not accept `cv_n_repeats` kwarg. Script's `RUN_KWARGS_EXTRA = {}` on main should handle this. If it still crashes, inspect `run_search` signature on main.
- Script crashes on branch only: likely a bug introduced by PR #4 — document and stop. Do not attempt to fix.
- Script crashes on LOO only: LOO needs `cv_strategy='loo'` routed correctly. If `cv_n_repeats=1` is rejected under LOO, try `cv_n_repeats=0` or omit it (some signatures make it conditional).
- All four (model, task) combos empty results from main: `enable_variable_subsets=False, enable_region_subsets=False` may produce zero rows if main uses a different subset convention. Try `variable_counts=[]` or remove those kwargs entirely on main.

**If you hit any of these:** document in the report under a "Blockers encountered" section, commit what you have, and stop. Don't spin your wheels past 60 minutes.

---

## Success criteria

You're done when `docs/pr4_parity_report.md` exists on the branch, is committed (but not pushed), and contains a clear verdict backed by a side-by-side comparison table. User picks up in the morning.
