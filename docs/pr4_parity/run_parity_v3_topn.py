"""
Parity v3: top-N variable selection on the 159-sample Yield dataset.

Verifies that the LightGBM cv R^2 dip on raw 2151-feature inputs is an
overfitting artifact, not a CV plumbing issue. Also serves as a 2nd
parity comparison with non-trivial preprocessing-discovery + var-selection.

Output: docs/pr4_parity/resultsv3_topn_{location}_{strategy}.json
"""
from __future__ import annotations

import json
import os
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

from spectral_predict.search import run_search


def load_data():
    df = pd.read_csv(DATASET_CSV)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    wl_cols = [c for c in df.columns if c.isdigit()]
    X = df[wl_cols].astype(float)
    X.columns = [float(c) for c in X.columns]
    X.index = df['Sample_ID'].astype(str).str.strip()
    y_reg = pd.Series(df['Yield'].astype(float).values, index=X.index, name='Yield')
    y_cls = pd.Series(np.where(df['Yield'].values > 3, 'High', 'Low'), index=X.index, name='YieldCat')
    return X, y_reg, y_cls


PREPROCESS_VARIANTS = {
    'raw': {'preprocessing_methods': {'raw': True}, 'window_sizes': None},
    'sg2_w31': {'preprocessing_methods': {'sg2': True}, 'window_sizes': [31]},
}


def run_one(X, y, model, task, top_n, prep_name):
    prep = PREPROCESS_VARIANTS[prep_name]
    kwargs = dict(
        folds=5,
        models_to_test=[model],
        preprocessing_methods=prep['preprocessing_methods'],
        enable_variable_subsets=True,
        variable_selection_methods=['importance'],
        variable_counts=[top_n],
        enable_region_subsets=False,
    )
    if prep['window_sizes'] is not None:
        kwargs['window_sizes'] = prep['window_sizes']
    kwargs.update(RUN_KWARGS_EXTRA)
    try:
        df, _ = run_search(X, y, task_type=task, **kwargs)
    except Exception as exc:
        return {'_error': f'{type(exc).__name__}: {exc}'}
    if df is None or len(df) == 0:
        return {'_error': 'empty results'}
    # Prefer rows whose SubsetTag includes the requested top_n; fall back to iloc[0]
    sub = df[df['SubsetTag'].astype(str).str.contains(str(top_n))] if 'SubsetTag' in df.columns else df.iloc[:0]
    row = (sub.iloc[0] if len(sub) else df.iloc[0]).to_dict()
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
    print(f'[{LOCATION}/{CV_STRATEGY}] X={X.shape}')
    combos = [
        ('PLS', 'regression'),
        ('LightGBM', 'regression'),
        ('PLS-DA', 'classification'),
        ('LightGBM', 'classification'),
    ]
    top_ns = [100, 50]
    preps = list(PREPROCESS_VARIANTS.keys())
    results = {}
    for prep_name in preps:
        for top_n in top_ns:
            for model, task in combos:
                key = f'{model}__{task}__{prep_name}__top{top_n}'
                print(f'[{LOCATION}/{CV_STRATEGY}] {key} ...', flush=True)
                results[key] = run_one(X, y_reg if task == 'regression' else y_cls, model, task, top_n, prep_name)
                err = results[key].get('_error') if isinstance(results[key], dict) else None
                print(f'[{LOCATION}/{CV_STRATEGY}] {key}: {"ERROR " + err if err else "ok " + str(len(results[key])) + "cols, SubsetTag=" + str(results[key].get("SubsetTag"))}', flush=True)

    out_dir = REPO_ROOT / 'docs' / 'pr4_parity'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'resultsv3_topn_{LOCATION}_{CV_STRATEGY}.json'
    out_file.write_text(json.dumps({
        'location': LOCATION,
        'strategy': CV_STRATEGY,
        'dataset': str(DATASET_CSV),
        'top_ns': top_ns,
        'preps': preps,
        'extra_kwargs': RUN_KWARGS_EXTRA,
        'results': results,
    }, indent=2))
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
