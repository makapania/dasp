"""
Broader repro attempt — matches GUI defaults more faithfully than v1.

v1 did not trigger the bug on branch HEAD. This version adds:
  - Multiple variable_counts (the GUI default sweep)
  - Region subsets enabled
  - (still SG2 only, window=17)
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_CSV = Path(r'C:/Users/sponheim/Desktop/2025 Model Samples/spectral_data_20260203_170711abs.csv')


def load_data():
    df = pd.read_csv(DATASET_CSV)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    wl_cols = [c for c in df.columns if c.isdigit()]
    X = df[wl_cols].astype(float)
    X.columns = [float(c) for c in X.columns]
    X.index = df['Sample_ID'].astype(str).str.strip()
    y = pd.Series(df['Yield'].astype(float).values, index=X.index, name='Yield')
    return X, y


def main():
    from spectral_predict.search import run_search

    X, y = load_data()
    print(f'[REPRO v2] dataset X={X.shape}, y={y.shape}', file=sys.stderr)

    kwargs = dict(
        folds=5,
        models_to_test=['LightGBM'],
        preprocessing_methods={'sg2': True},
        window_sizes=[17],
        enable_variable_subsets=True,
        variable_selection_methods=['importance'],
        variable_counts=[10, 20, 50, 100, 250, 500, 1000],  # GUI default sweep
        enable_region_subsets=True,  # GUI default
    )
    try:
        from spectral_predict.cv_utils import _is_repeated_cv  # noqa
        kwargs['cv_strategy'] = 'kfold'
        kwargs['cv_n_repeats'] = 1
        location = 'branch'
    except ImportError:
        location = 'main'

    captured = io.StringIO()
    error = None
    df_out = None
    try:
        with redirect_stdout(captured):
            df_out, _ = run_search(X, y, task_type='regression', **kwargs)
    except Exception as e:
        error = f'{type(e).__name__}: {e}'

    stdout_text = captured.getvalue()

    evidence = {
        'location': location,
        'error': error,
        'stdout_mentions_feature_mismatch': 'parameter capture' in stdout_text or ('2135' in stdout_text and '2151' in stdout_text),
        'stdout_mentions_LOO': ' LOO' in stdout_text or ' loo' in stdout_text,
        'warning_lines': [L for L in stdout_text.splitlines() if 'parameter capture' in L or 'Warning' in L][:15],
    }

    if df_out is not None and len(df_out):
        def _finite(v):
            if not isinstance(v, (int, float, np.integer, np.floating)):
                return False
            f = float(v)
            return not (np.isnan(f) or np.isinf(f))

        evidence['n_rows'] = int(len(df_out))

        lgbm_rows = df_out[df_out['Model'] == 'LightGBM'] if 'Model' in df_out.columns else df_out.iloc[:0]
        if len(lgbm_rows):
            nan_cal = (~lgbm_rows['RMSE'].apply(_finite)).sum() if 'RMSE' in lgbm_rows.columns else 0
            evidence['lightgbm_rows_total'] = int(len(lgbm_rows))
            evidence['lightgbm_rows_with_nan_calibration'] = int(nan_cal)

            nan_subset_tags = []
            if nan_cal > 0:
                bad = lgbm_rows[~lgbm_rows['RMSE'].apply(_finite)]
                nan_subset_tags = bad.get('SubsetTag', pd.Series([])).astype(str).tolist()[:20]
            evidence['nan_subset_tags_sample'] = nan_subset_tags

    bug_present = (
        evidence['stdout_mentions_feature_mismatch']
        or evidence.get('lightgbm_rows_with_nan_calibration', 0) > 0
    )
    evidence['BUG_PRESENT'] = bug_present

    if evidence['warning_lines']:
        print('--- warning lines ---', file=sys.stderr)
        for L in evidence['warning_lines']:
            print(L, file=sys.stderr)

    print(json.dumps(evidence, indent=2, default=str))
    sys.exit(1 if bug_present else 0)


if __name__ == '__main__':
    main()
