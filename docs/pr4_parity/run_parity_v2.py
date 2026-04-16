"""
Parity v2: runs on the larger Yield/CollagenCat dataset and adds one-class.

Output: docs/pr4_parity/resultsv2_{location}_{strategy}.json

Combos (7):
  PLS__regression, PLS-DA__classification,
  LightGBM__regression, LightGBM__classification,
  OneClassSVM__one_class, IsolationForest__one_class, PCA-SIMCA__one_class

Binary target: YieldCat = 'High' if Yield > 3 else 'Low'.
One-class inlier label: 'High'.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_CSV = Path(r'C:/Users/sponheim/Desktop/2025 Model Samples/spectral_data_20260203_170711abs.csv')

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

from spectral_predict.search import run_search, run_one_class_search


def load_data():
    df = pd.read_csv(DATASET_CSV)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    wl_cols = [c for c in df.columns if c.isdigit()]
    X = df[wl_cols].astype(float)
    X.columns = [float(c) for c in X.columns]
    X.index = df['Sample_ID'].astype(str).str.strip()
    y_reg = df['Yield'].astype(float).values
    y_cls = np.where(df['Yield'].values > 3, 'High', 'Low')
    y_reg = pd.Series(y_reg, index=X.index, name='Yield')
    y_cls = pd.Series(y_cls, index=X.index, name='YieldCat')
    return X, y_reg, y_cls


def run_reg_or_cls(X, y, model, task):
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
    return _serialize_row(row)


def run_oc(X, y, model):
    kwargs = dict(
        folds=5,
        inlier_class_label='High',
        enabled_models=[model],
        preprocessing_methods={'raw': True},
    )
    kwargs.update(RUN_KWARGS_EXTRA)
    try:
        df = run_one_class_search(X, y, **kwargs)
    except Exception as exc:
        return {'_error': f'{type(exc).__name__}: {exc}'}
    if df is None or len(df) == 0:
        return {'_error': 'empty results'}
    row = df.iloc[0].to_dict()
    return _serialize_row(row)


def _serialize_row(row):
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
    print(f'[{LOCATION}/{CV_STRATEGY}] Data loaded: X={X.shape}, y_reg n={len(y_reg)} (mean={y_reg.mean():.2f}), y_cls counts={dict(pd.Series(y_cls).value_counts())}')

    combos = [
        ('PLS', 'regression', 'reg'),
        ('PLS-DA', 'classification', 'reg'),
        ('LightGBM', 'regression', 'reg'),
        ('LightGBM', 'classification', 'reg'),
        ('OneClassSVM', 'one_class', 'oc'),
        ('IsolationForest', 'one_class', 'oc'),
        ('PCA-SIMCA', 'one_class', 'oc'),
    ]
    results = {}
    for model, task, kind in combos:
        key = f'{model}__{task}'
        print(f'[{LOCATION}/{CV_STRATEGY}] running {key} ...', flush=True)
        if kind == 'oc':
            results[key] = run_oc(X, y_cls, model)
        else:
            y = y_reg if task == 'regression' else y_cls
            results[key] = run_reg_or_cls(X, y, model, task)
        err = results[key].get('_error') if isinstance(results[key], dict) else None
        print(f'[{LOCATION}/{CV_STRATEGY}] {key}: {"ERROR " + err if err else "ok (" + str(len(results[key])) + " cols)"}', flush=True)

    out_dir = REPO_ROOT / 'docs' / 'pr4_parity'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'resultsv2_{LOCATION}_{CV_STRATEGY}.json'
    out_file.write_text(json.dumps({
        'location': LOCATION,
        'strategy': CV_STRATEGY,
        'dataset': str(DATASET_CSV),
        'extra_kwargs': RUN_KWARGS_EXTRA,
        'results': results,
    }, indent=2))
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
