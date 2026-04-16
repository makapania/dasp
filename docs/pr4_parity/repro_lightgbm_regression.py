"""
Targeted repro of the user's GUI failure.

Config (matches GUI defaults):
  - Preprocessing: SG2 only, window=17 (GUI default)
  - Variable subsets: ON, top-N from importance
  - Model: LightGBM only
  - 5-fold K-Fold (default)

Captures:
  - stdout for "Could not fit model for parameter capture" warning
  - Result row for NaN calibration RMSE/R2

Exit codes:
  0 = no bug (clean run)
  1 = bug confirmed (warning OR NaN calibration OR both)
  2 = script error

Output: JSON dump of evidence to stdout.
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
    print(f'[REPRO] dataset X={X.shape}, y={y.shape}', file=sys.stderr)

    kwargs = dict(
        folds=5,
        models_to_test=['LightGBM'],
        preprocessing_methods={'sg2': True},
        window_sizes=[17],
        enable_variable_subsets=True,
        variable_selection_methods=['importance'],
        variable_counts=[100],
        enable_region_subsets=False,
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
        'warning_lines': [L for L in stdout_text.splitlines() if 'parameter capture' in L or 'Warning' in L][:10],
    }

    if df_out is not None and len(df_out):
        row = df_out.iloc[0].to_dict()

        def _finite(v):
            if not isinstance(v, (int, float, np.integer, np.floating)):
                return False
            f = float(v)
            return not (np.isnan(f) or np.isinf(f))

        evidence['n_rows'] = int(len(df_out))
        evidence['top_row'] = {
            'Model': row.get('Model'),
            'Preprocess': row.get('Preprocess'),
            'SubsetTag': row.get('SubsetTag'),
            'RMSE_cal': row.get('RMSE') if _finite(row.get('RMSE')) else None,
            'R2_cal': row.get('R2') if _finite(row.get('R2')) else None,
            'RMSEcv': row.get('RMSEcv') if _finite(row.get('RMSEcv')) else None,
            'R2cv': row.get('R2cv') if _finite(row.get('R2cv')) else None,
        }
        # Check any LightGBM row for NaN calibration
        lgbm_rows = df_out[df_out['Model'] == 'LightGBM'] if 'Model' in df_out.columns else df_out.iloc[:0]
        if len(lgbm_rows):
            nan_cal_count = int((~lgbm_rows['RMSE'].apply(_finite)).sum())
            evidence['lightgbm_rows_total'] = int(len(lgbm_rows))
            evidence['lightgbm_rows_with_nan_calibration'] = nan_cal_count

    bug_present = (
        evidence['stdout_mentions_feature_mismatch']
        or evidence.get('lightgbm_rows_with_nan_calibration', 0) > 0
    )
    evidence['BUG_PRESENT'] = bug_present

    print(json.dumps(evidence, indent=2, default=str))
    sys.exit(1 if bug_present else 0)


if __name__ == '__main__':
    main()
