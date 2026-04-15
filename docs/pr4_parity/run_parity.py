"""
Run 4 (model, task) combinations with kfold=5 (or LOO) and dump headline metrics.

Output: docs/pr4_parity/results_{location}_{strategy}.json
  location = 'main' or 'branch' (inferred from import probe)
  strategy = 'kfold' or 'loo' (from CV_STRATEGY env var; only branch honors it)
"""
from __future__ import annotations

import json
import os
import sys
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
    print(f'[{LOCATION}/{CV_STRATEGY}] Data loaded: X={X.shape}, y_reg={y_reg.shape}, y_cls={y_cls.shape}')
    combos = [
        ('PLS', 'regression'),
        ('PLS-DA', 'classification'),
        ('LightGBM', 'regression'),
        ('LightGBM', 'classification'),
    ]
    results = {}
    for model, task in combos:
        key = f'{model}__{task}'
        print(f'[{LOCATION}/{CV_STRATEGY}] running {key} ...', flush=True)
        results[key] = run_one(X, y_reg, y_cls, model, task)
        err = results[key].get('_error') if isinstance(results[key], dict) else None
        print(f'[{LOCATION}/{CV_STRATEGY}] {key}: {"ERROR " + err if err else "ok (" + str(len(results[key])) + " cols)"}')

    out_dir = REPO_ROOT / 'docs' / 'pr4_parity'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'results_{LOCATION}_{CV_STRATEGY}.json'
    out_file.write_text(json.dumps({
        'location': LOCATION,
        'strategy': CV_STRATEGY,
        'extra_kwargs': RUN_KWARGS_EXTRA,
        'results': results,
    }, indent=2))
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
