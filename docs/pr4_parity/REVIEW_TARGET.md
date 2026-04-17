# PR #4 Numeric Parity Validation — Plan + Reproducer for Review

## Background

PR #4 on branch `claude/cv-strategy-overhaul` (HEAD `c091c93`) is a cross-validation strategy overhaul that adds:
- Leave-One-Out CV as an alternative to K-Fold
- Repeated K-Fold CV (controlled by `cv_n_repeats`)
- Pooled per-sample prediction reduction for repeated CV (via new `spectral_predict.cv_utils` module)

Main (`fa39504`) has neither LOO nor repeated K-Fold. It runs plain K-Fold always.

The goal of this validation is to prove that with `cv_strategy='kfold'` + `cv_n_repeats=1`, branch produces numerically identical results to main for the 4 (model, task) combinations: PLS-regression, PLS-DA-classification, LightGBM-regression, LightGBM-classification. LOO is a sanity check on branch only.

Dataset is `example/BoneCollagen.csv` + `Spectrum00001.asd..Spectrum00049.asd` (49 samples, `%Collagen` continuous target, `CollagenCat` 3-class string target).

## Expected diffs (per the prompt)

- **Plain K-Fold regression** headline metrics: bit-identical.
- **Plain K-Fold classification** `Accuracycv`, `Kappacv`, `MCCcv`: bit-identical.
- **Plain K-Fold classification** `F1cv`, `Precisioncv`, `Recallcv`: may differ ~1e-6 because round-2 commit switched the pooled-repeated-CV path to `average='binary' if is_binary else 'macro'`. For 3-class CollagenCat, macro. The non-repeated path should be unchanged, so expect equal.
- **Plain K-Fold classification** `Specificitycv`: may differ ~1e-6 because round-3 commit added `labels=np.unique(y)` to the pooled-repeated-CV specificity path (not to plain K-Fold). Expected equal.
- **BER under plain K-Fold**: mean-of-fold-BERs on both branches (branch only changed the repeated-CV BER). Expected equal.
- **Calibration metrics**: MUST be bit-identical. Calibration doesn't go through the CV reduction plumbing at all.

## Reproducer script

File: `docs/pr4_parity/run_parity.py` (identical copy placed in both main and branch checkouts).

```python
"""
Run 4 (model, task) combinations with kfold=5 (or LOO) and dump headline metrics.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from spectral_predict.cv_utils import _is_repeated_cv  # noqa: F401
    LOCATION = 'branch'
    CV_STRATEGY = os.environ.get('CV_STRATEGY', 'kfold')
    if CV_STRATEGY == 'loo':
        RUN_KWARGS_EXTRA = {'cv_strategy': 'loo'}
    else:
        RUN_KWARGS_EXTRA = {'cv_strategy': 'kfold', 'cv_n_repeats': 1}
except ImportError:
    LOCATION = 'main'
    CV_STRATEGY = 'kfold'
    RUN_KWARGS_EXTRA = {}

from spectral_predict.io import read_asd_dir
from spectral_predict.search import run_search


def load_data():
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
    return X, y_reg, y_cls


def run_one(X, y_reg, y_cls, model, task):
    y = y_reg if task == 'regression' else y_cls
    kwargs = dict(
        folds=5,
        models_to_test=[model],
        preprocessing_methods={'raw': True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
    )
    kwargs.update(RUN_KWARGS_EXTRA)
    try:
        df, _ = run_search(X, y, task_type=task, **kwargs)
    except Exception as exc:
        return {'_error': f'{type(exc).__name__}: {exc}'}
    if df is None or len(df) == 0:
        return {'_error': 'empty results'}
    row = df.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            out[k] = None if (np.isnan(f) or np.isinf(f)) else f
        elif isinstance(v, (str, bool)):
            out[k] = v
    return out


def main():
    X, y_reg, y_cls = load_data()
    combos = [
        ('PLS', 'regression'),
        ('PLS-DA', 'classification'),
        ('LightGBM', 'regression'),
        ('LightGBM', 'classification'),
    ]
    results = {f'{m}__{t}': run_one(X, y_reg, y_cls, m, t) for m, t in combos}
    out_dir = REPO_ROOT / 'docs' / 'pr4_parity'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'results_{LOCATION}_{CV_STRATEGY}.json').write_text(json.dumps({
        'location': LOCATION, 'strategy': CV_STRATEGY,
        'extra_kwargs': RUN_KWARGS_EXTRA, 'results': results,
    }, indent=2))
```

The **import probe** is the only thing determining which code path gets tested: branch has `spectral_predict.cv_utils` (new module), main does not. `PYTHONPATH=./src` in the worktree forces Python to import the worktree's `src/`, overriding the editable install that points at main.

## Reviewer questions

1. **Does `df.iloc[0]` give a stable comparison point?** The search produces a results DataFrame that is then scored/ranked. If main and branch rank identically when given identical inputs, row 0 is the same model+preprocessing combo. If scoring order differs, row 0 could correspond to different combos on each side and the comparison is meaningless. The prompt assumes `preprocessing_methods={'raw': True}` + `models_to_test=[model]` + `enable_variable_subsets=False` produces exactly one combo, but search still emits multiple rows (main produced 96 rows for LightGBM classification in one test). Is that a problem?

2. **Is the bit-identical expectation for calibration actually achievable?** `run_search` internally fits models on full data for calibration. Branch and main should use identical seeds (`random_state=42`) and identical fit paths for calibration. Any reason LightGBM calibration could drift between branches?

3. **Is the ≤1e-6 tolerance for classification CV reasonable?** The prompt claims round-2 commit switched only the pooled-repeated-CV path to macro averaging, not the plain K-Fold path. Is this claim actually true? What's the plain K-Fold path's F1 average mode on main vs branch?

4. **Does `cv_n_repeats=1` on branch actually reproduce main's plain-K-Fold semantics?** The new cv_utils module defines `_is_repeated_cv`. Does `n_repeats=1` take the non-repeated path entirely, or does it still route through the pooling reducer (which should be a no-op but could numerically differ due to float accumulation order)?

5. **LOO with LightGBM on 49 samples**: is there anything structural that makes LightGBM misbehave with LOO (e.g., single-sample test folds)? If LightGBM is written to skip tiny test folds or produce NaN for them, LOO could silently produce meaningless metrics.

6. **Any determinism hazards?** Both runs happen on the same machine, same Python 3.11.9, sklearn 1.8.0, numpy 2.3.5, lightgbm 4.6.0. But are there any OMP thread-count-dependent code paths, float-reduction-order issues, or timestamp-seeded randomness anywhere in `run_search`?

7. **If results mismatch in a way not covered by the "expected diffs" list, is there a plausible innocent explanation?** For example: main's row 0 at LightGBM classification has 96 rows — why? (Presumably 96 comes from a Bayesian trial count.) If branch produces a different count, the ranking could pick a different top row.

Please flag any concern that could make the validation report falsely conclude PASS when the branch actually has a numerical bug.
