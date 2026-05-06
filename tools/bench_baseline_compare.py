"""Cross-checkout comparison: pre-session baseline (727077f) vs current HEAD.

Spawns two Python subprocesses with PYTHONPATH pointing at different source
trees so they actually load the OLD code from C:/Users/mspon/dasp-baseline
and the NEW code from the current repo. Each subprocess runs unified
Bayesian search with identical inputs (same seed, same data, same
n_trials) and dumps the result CSV + study fingerprints.

The driver compares:
- Model fit columns (Params, PreprocessBase, Deriv, Window, Poly,
  Autoscale, n_vars, SubsetTag) and metrics (RMSEcv / Accuracycv / etc.)
  for every fingerprint that appears in BOTH runs.
- Pre's best fingerprint reached by post? Post's best by pre?
- Wall-clock difference.

Usage:
    .venv312/Scripts/python.exe tools/bench_baseline_compare.py [--scenario pls|lightgbm] [--n-trials 100]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_NEW = Path(r"C:\Users\mspon\git\dasp")
REPO_OLD = Path(r"C:\Users\mspon\dasp-baseline")
PYTHON = REPO_NEW / ".venv312" / "Scripts" / "python.exe"


WORKER_SCRIPT = r'''
import json, sys, time
import numpy as np
import pandas as pd

from spectral_predict.io import read_asd_dir
from spectral_predict import unified_bayesian as ub

scenario = sys.argv[1]   # 'pls' | 'lightgbm'
n_trials = int(sys.argv[2])
output_path = sys.argv[3]
max_features = int(sys.argv[4])

REGRESSION_TARGET = "%Collagen"
CLASSIFICATION_TARGET = "CollagenCat"
if scenario == "pls":
    model_name, task_type, target = "PLS", "regression", REGRESSION_TARGET
else:
    model_name, task_type, target = "LightGBM", "classification", CLASSIFICATION_TARGET

spectra, _meta = read_asd_dir("example")
ref = pd.read_csv("example/BoneCollagen.csv")
spectra = spectra.sort_index()
ref = ref.copy()
ref.index = (
    ref["File Number"].astype(str)
    .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    .str.replace(" ", "", regex=False)
)
joined = spectra.join(ref, how="inner")
feature_cols = list(spectra.columns)[:max_features]
X = joined[feature_cols].to_numpy(dtype=float)
wavelengths = np.asarray([float(c) for c in feature_cols], dtype=float)
y = joined[target].to_numpy(dtype=float if task_type == "regression" else None)

t0 = time.perf_counter()
df, study = ub.run_unified_bayesian(
    X, y, wavelengths,
    model_name=model_name, task_type=task_type,
    n_trials=n_trials, cv_folds=3, cv_strategy="kfold",
    random_state=42, n_top_regions=3,
    enable_sqlite_persistence="never",
    early_stopping_rounds=None, enable_uve=False,
    verbose=False,
)
elapsed = time.perf_counter() - t0

# Dump structured payload
import optuna
trials = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    trials.append({
        "trial_number": t.number,
        "value": float(t.value) if t.value is not None else None,
        "fingerprint": t.user_attrs.get("fingerprint"),
        # Note: hardcoded string here (not the DUPLICATE_OF_TRIAL_ATTR constant)
        # because this tool runs in a subprocess across two different repo
        # checkouts; importing from spectral_predict.unified_bayesian would
        # vary by sys.path. The diagnostic string is forward-compatible:
        # absent in old (727077f) trials, set on new-mechanism dups.
        "duplicate_of_trial": t.user_attrs.get("duplicate_of_trial"),
    })

# Filter df to dedup-relevant columns; serialize to JSON-safe form
out_cols = [c for c in df.columns if c in {
    "Model","Preprocess","PreprocessBase","Deriv","Window","Poly","Autoscale",
    "n_vars","SubsetTag","Params","trial_number",
    "RMSEcv","R2cv","MAEcv","RMSE","R2",
    "Accuracycv","Accuracy","ROC_AUCcv","F1cv","BalancedAcccv",
    "imbalance_method"
}]
csv_payload = df[out_cols].to_dict(orient="records") if out_cols else []

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "elapsed": elapsed,
        "scenario": scenario,
        "n_trials": n_trials,
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "trials": trials,
        "csv_rows": csv_payload,
    }, f)
'''


def run_subprocess_in(repo: Path, label: str, scenario: str, n_trials: int, max_features: int):
    """Run worker script with sys.path/working-dir set to ``repo``."""
    output = REPO_NEW / "tools" / f"_bench_baseline_{label}_output.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    print(f"\n[{label}] PYTHONPATH={repo}/src — running {scenario} n_trials={n_trials}...")
    t0 = time.perf_counter()
    subprocess.run(
        [str(PYTHON), "-c", WORKER_SCRIPT, scenario, str(n_trials), str(output), str(max_features)],
        cwd=str(repo),
        env=env,
        check=True,
    )
    wall = time.perf_counter() - t0
    with open(output, "r", encoding="utf-8") as f:
        result = json.load(f)
    print(f"[{label}] outer wall={wall:.1f}s  inner wall={result['elapsed']:.1f}s  "
          f"COMPLETE={result['n_complete']}  PRUNED={result['n_pruned']}")
    return result


def csv_row_key(row):
    """Build a tuple key from CSV columns that survive both old and new code.
    Old code (727077f) doesn't have stored fingerprints, so we match on the
    user-visible CSV-row identity instead."""
    return (
        row.get("Params"),
        row.get("PreprocessBase"),
        row.get("Deriv"),
        row.get("Window"),
        row.get("Poly"),
        row.get("Autoscale"),
        row.get("n_vars"),
        row.get("SubsetTag"),
    )


def compare_payloads(old_result: dict, new_result: dict, scenario: str):
    """Compare baseline vs current HEAD outputs by CSV-row identity."""
    metric = "RMSEcv" if scenario == "pls" else "Accuracycv"
    higher_is_better = scenario != "pls"

    # Build map: csv_key -> metric value (first occurrence wins)
    old_value_by_key = {}
    for r in old_result["csv_rows"]:
        k = csv_row_key(r)
        if r.get(metric) is not None and k not in old_value_by_key:
            old_value_by_key[k] = r[metric]
    new_value_by_key = {}
    for r in new_result["csv_rows"]:
        k = csv_row_key(r)
        if r.get(metric) is not None and k not in new_value_by_key:
            new_value_by_key[k] = r[metric]

    old_keys = set(old_value_by_key)
    new_keys = set(new_value_by_key)
    common = old_keys & new_keys
    old_only = old_keys - new_keys
    new_only = new_keys - old_keys

    def best_kv(value_by_key):
        if not value_by_key:
            return None, None
        items = list(value_by_key.items())
        if higher_is_better:
            best = max(items, key=lambda kv: kv[1])
        else:
            best = min(items, key=lambda kv: kv[1])
        return best[1], best[0]

    old_best_v, old_best_key = best_kv(old_value_by_key)
    new_best_v, new_best_key = best_kv(new_value_by_key)

    matched = 0
    mismatches = []
    for k in common:
        ov = old_value_by_key.get(k)
        nv = new_value_by_key.get(k)
        if ov is None or nv is None:
            continue
        if abs(ov - nv) < 1e-10:
            matched += 1
        else:
            mismatches.append((str(k)[:120], ov, nv, abs(ov - nv)))

    print(f"\n=== Cross-checkout comparison: {scenario} ===")
    print(f"  OLD (727077f, pre-session): wall={old_result['elapsed']:.1f}s  "
          f"COMPLETE={old_result['n_complete']}  unique_fps={len(old_fp_set)}  "
          f"CSV rows={len(old_result['csv_rows'])}")
    print(f"  NEW (current HEAD):         wall={new_result['elapsed']:.1f}s  "
          f"COMPLETE={new_result['n_complete']}  PRUNED={new_result['n_pruned']}  "
          f"unique_fps={len(new_fp_set)}  CSV rows={len(new_result['csv_rows'])}")
    print(f"\n  Wall-clock delta: {new_result['elapsed'] - old_result['elapsed']:+.1f} s "
          f"({(new_result['elapsed']/old_result['elapsed'] - 1)*100:+.1f}%)")
    print(f"\n  Fingerprint sets:")
    print(f"    |OLD|={len(old_fp_set)}  |NEW|={len(new_fp_set)}  |OLD AND NEW|={len(common)}")
    print(f"    Old-only (not explored by new): {len(old_only)}")
    print(f"    New-only (not explored by old): {len(new_only)}")
    print(f"    Old's best fingerprint reached by NEW: "
          f"{'YES' if old_best_fp in new_fp_set else 'NO'}")
    print(f"    New's best fingerprint reached by OLD: "
          f"{'YES' if new_best_fp in old_fp_set else 'NO'}")
    print(f"\n  Best metric ({metric}):")
    print(f"    OLD={old_best_v:.6f}")
    print(f"    NEW={new_best_v:.6f}")
    print(f"    Delta: {new_best_v - old_best_v:+.6f}")
    print(f"\n  Per-fingerprint metric agreement (shared fingerprints, abs delta < 1e-10):")
    print(f"    Matched bit-identically: {matched}/{len(common)}")
    if mismatches:
        print(f"    Mismatches (first 5):")
        for fp, ov, nv, delta in mismatches[:5]:
            print(f"      fp={fp!r}  old={ov:.10g}  new={nv:.10g}  |delta|={delta:.2e}")
    return {
        "common_count": len(common),
        "old_only_count": len(old_only),
        "new_only_count": len(new_only),
        "matched_bit_identical": matched,
        "old_best_in_new": old_best_fp in new_fp_set,
        "new_best_in_old": new_best_fp in old_fp_set,
        "old_best_value": old_best_v,
        "new_best_value": new_best_v,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["pls", "lightgbm"], default="pls")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=120)
    args = parser.parse_args()

    if not REPO_OLD.exists():
        raise SystemExit(
            f"Baseline worktree not found at {REPO_OLD}. "
            f"Create with: git worktree add {REPO_OLD} 727077f"
        )

    old = run_subprocess_in(REPO_OLD, "old-727077f", args.scenario, args.n_trials, args.max_features)
    new = run_subprocess_in(REPO_NEW, "new-HEAD", args.scenario, args.n_trials, args.max_features)
    compare_payloads(old, new, args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
