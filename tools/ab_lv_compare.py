"""A/B comparison: PLS Bayesian search before vs after the LVs reporting fix.

Run this BEFORE the edits with --tag old, then AFTER the edits with --tag new.
Then run with --diff to compare. Model-fit columns must be byte-identical;
LVs column may differ only on previously-buggy clamped rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)


def run_search() -> pd.DataFrame:
    from spectral_predict.unified_bayesian import run_unified_bayesian

    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = 12  # < 20 so the n_components clamp at unified_bayesian.py:462 fires
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.05 * rng.standard_normal(n_samples)
    wavelengths = np.arange(1.0, n_features + 1.0)

    df, _ = run_unified_bayesian(
        X, y, wavelengths,
        model_name="PLS",
        task_type="regression",
        n_trials=25,
        cv_folds=5,
        cv_strategy="kfold",
        random_state=42,
    )
    return df


_FIT_COLS = (
    "Task", "Model", "Params", "Preprocess", "PreprocessBase",
    "Deriv", "Window", "Poly", "n_vars", "full_vars", "SubsetTag",
    "RMSE", "R2", "RMSEcv", "R2cv", "MAEcv",
    "trial_number", "Folds", "all_vars", "top_vars",
)


def diff_runs(old: pd.DataFrame, new: pd.DataFrame) -> int:
    print(f"OLD rows: {len(old)}  NEW rows: {len(new)}")
    if len(old) != len(new):
        print("ROW COUNT MISMATCH — aborting deeper diff.")
        return 2

    old_s = old.sort_values("trial_number").reset_index(drop=True)
    new_s = new.sort_values("trial_number").reset_index(drop=True)

    fit_cols = [c for c in _FIT_COLS if c in old_s.columns and c in new_s.columns]
    print(f"Comparing {len(fit_cols)} fit columns: {fit_cols}")

    rc = 0
    for c in fit_cols:
        a = old_s[c].astype(str).values
        b = new_s[c].astype(str).values
        diff_mask = a != b
        n_diff = int(diff_mask.sum())
        if n_diff:
            print(f"  [DIFF] column '{c}': {n_diff} rows differ")
            for idx in np.where(diff_mask)[0][:3]:
                print(f"    row {idx}: OLD={a[idx]!r}  NEW={b[idx]!r}")
            rc = 1
        else:
            print(f"  [OK]   column '{c}': identical across {len(a)} rows")

    if "LVs" in old_s.columns and "LVs" in new_s.columns:
        old_lv = old_s["LVs"].astype(str).values
        new_lv = new_s["LVs"].astype(str).values
        lv_diff = old_lv != new_lv
        n = int(lv_diff.sum())
        print(f"\nLVs column: {n} rows differ (expected: 0 if no clamp fired, >0 if it did).")
        for idx in np.where(lv_diff)[0][:5]:
            old_p = old_s.iloc[idx].get("Params", "")
            print(
                f"  row {idx} trial={old_s.iloc[idx].get('trial_number')}: "
                f"OLD LVs={old_lv[idx]} NEW LVs={new_lv[idx]} | Params={old_p[:140]}"
            )

    return rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", choices=["old", "new"], help="Tag this run as old or new.")
    p.add_argument("--diff", action="store_true", help="Diff the saved old vs new runs.")
    args = p.parse_args()

    if args.diff:
        old = pd.read_csv(OUT / "ab_compare_OLD.csv")
        new = pd.read_csv(OUT / "ab_compare_NEW.csv")
        return diff_runs(old, new)

    if not args.tag:
        p.error("--tag is required (or --diff)")

    df = run_search()
    target = OUT / f"ab_compare_{args.tag.upper()}.csv"
    df.to_csv(target, index=False)
    print(f"Wrote {target} ({len(df)} rows, {len(df.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
