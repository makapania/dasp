"""Bayesian top-K stability across seeds — empirical companion to exhaustive_seed_compare.

Question: when run_unified_bayesian uses single-seed CV per trial (the current
default), do its top-7/10 configs across runs with different random_state
look like the same set, or does seed lottery dominate as it does for
single-seed exhaustive?

Design:
  * Run run_unified_bayesian on BoneCollagen 3 times with different random_state
    on the same task (PLS-DA classification — the noisy case from the exhaustive
    test) and on regression (PLS — the stable case).
  * For each run, extract the results_df sorted by CV metric.
  * For top-K (K in {5, 7, 10}), compute Jaccard across the 3 runs.
  * For each run's top-K set, look up where each member ranks in the OTHER 2
    runs' rankings — the inverse of "deep infiltration" diagnostic.

Cost: ~3 × Bayesian-run wall time. With n_trials=100 on PLS/PLS-DA on n=49,
this is roughly 60-180s total.
"""

from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectral_predict.io import read_asd_dir  # noqa: E402
from spectral_predict.unified_bayesian import run_unified_bayesian  # noqa: E402

EXAMPLE_DIR = REPO_ROOT / "example"
LABEL_CSV = EXAMPLE_DIR / "BoneCollagen.csv"

SEEDS = [42, 0, 7]
N_TRIALS = 100

warnings.filterwarnings("ignore")


def load_paired_data():
    from sklearn.preprocessing import LabelEncoder
    asd_df, _meta = read_asd_dir(EXAMPLE_DIR)
    asd_df.index = asd_df.index.astype(str).map(lambda s: re.sub(r"\s+", "", s))
    labels = pd.read_csv(LABEL_CSV)
    labels["join_key"] = labels["File Number"].astype(str).map(
        lambda s: re.sub(r"\s+", "", s)
    )
    common = asd_df.index.intersection(labels["join_key"])
    asd_df = asd_df.loc[common].sort_index()
    labels = labels.set_index("join_key").loc[common].sort_index()
    X = asd_df.to_numpy(dtype=float)
    wavelengths = asd_df.columns.to_numpy(dtype=float)
    y_reg = labels["%Collagen"].to_numpy(dtype=float)
    le = LabelEncoder()
    y_cls = le.fit_transform(labels["CollagenCat"].to_numpy())
    return X, y_reg, y_cls, wavelengths


def config_key(row: pd.Series) -> tuple:
    """Fingerprint by DISCRETE dimensions only.

    n_components / LVs is a continuous-ish int (2-30) that TPE samples
    differently each seed, so including it makes every config look unique.
    The discrete identity (preproc, deriv, window, autoscale, baseline,
    smoothing) is what we want to compare across runs — "did the same
    preprocessing family land in top-K in run 2 as in run 1?"
    """
    return (
        str(row.get("Preprocess", "")),
        int(row.get("Deriv", 0) or 0),
        int(row.get("Window", 0) or 0),
        bool(row.get("Autoscale", False)),
        str(row.get("baseline_method", "") or ""),
        bool(row.get("smoothing", False)),
    )


def run_one_bayesian(X, y, wavelengths, model_name, task_type, seed, n_trials):
    t0 = time.time()
    df, _study = run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=wavelengths,
        model_name=model_name,
        task_type=task_type,
        n_trials=n_trials,
        cv_folds=5,
        cv_strategy="kfold",
        cv_n_repeats=1,
        random_state=seed,
        verbose=False,
        enable_autoscale=True,
        enable_uve=False,
        enable_sqlite_persistence="never",
    )
    elapsed = time.time() - t0
    if "Model" not in df.columns:
        df["Model"] = model_name
    return df, elapsed


def analyze(task: str, model_name: str, X, y, wavelengths) -> None:
    print(f"\n{'='*78}")
    print(f"BAYESIAN TOP-K STABILITY — {task.upper()} ({model_name})")
    print(f"{'='*78}\n")

    metric_col = "RMSEcv" if task == "regression" else "BalancedAcccv"
    metric_ascending = (task == "regression")

    runs: list[pd.DataFrame] = []
    for seed in SEEDS:
        print(f"  Run with seed={seed}...", end=" ", flush=True)
        df, elapsed = run_one_bayesian(X, y, wavelengths, model_name, task, seed, N_TRIALS)
        df_sorted = df.sort_values(metric_col, ascending=metric_ascending, na_position="last").reset_index(drop=True)
        df_sorted["rank_in_run"] = np.arange(1, len(df_sorted) + 1)
        df_sorted["config_key"] = df_sorted.apply(config_key, axis=1)
        runs.append(df_sorted)
        best = df_sorted.iloc[0]
        print(f"done ({elapsed:.1f}s, n_trials={len(df_sorted)}, best {metric_col}={best[metric_col]:.4f})")

    # Debug: check what's actually evaluated in each run
    print()
    print("Diagnostic: how many configs are evaluated per run, how do they overlap?")
    keys_per_run = [set(r["config_key"]) for r in runs]
    for i, ks in enumerate(keys_per_run):
        print(f"  Run {i+1} (seed={SEEDS[i]}): {len(runs[i])} trials, {len(ks)} unique config_keys")
    union_all = set().union(*keys_per_run)
    intersect_all = set.intersection(*keys_per_run) if keys_per_run else set()
    print(f"  Union across runs: {len(union_all)}")
    print(f"  Intersection across runs (configs ALL 3 evaluated): {len(intersect_all)}")
    print(f"  Pairwise overlap on EVALUATED set:")
    for i in range(len(keys_per_run)):
        for j in range(i + 1, len(keys_per_run)):
            inter = len(keys_per_run[i] & keys_per_run[j])
            print(f"    runs {i+1} & {j+1}: {inter} keys in common")

    # Also dump a few sample fingerprints to verify they look right
    print()
    print("Sample fingerprints (first 3 from run 1):")
    for _, r in runs[0].head(3).iterrows():
        print(f"  rank={int(r['rank_in_run'])}: {r['config_key']}")
    print("Sample fingerprints (first 3 from run 2):")
    for _, r in runs[1].head(3).iterrows():
        print(f"  rank={int(r['rank_in_run'])}: {r['config_key']}")

    print()
    print("Top-K stability across the 3 seeded Bayesian runs:")
    for k in (5, 7, 10):
        sets = [set(r.head(k)["config_key"]) for r in runs]
        pairwise = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                pairwise.append(inter / max(union, 1))
        union_all = set().union(*sets)
        intersect_all = set.intersection(*sets) if sets else set()
        print(
            f"  top-{k}: pairwise jaccard mean = {np.mean(pairwise):.2f}  "
            f"min = {np.min(pairwise):.2f}  "
            f"max = {np.max(pairwise):.2f}  "
            f"intersect-all = {len(intersect_all)}/{k}  "
            f"union = {len(union_all)}"
        )

    print()
    print("Where do top-K from run 1 (seed=42) land in runs 2 and 3?")
    for k in (5, 7, 10):
        run1_topk = runs[0].head(k)
        run2 = runs[1]
        run3 = runs[2]
        worst_rank2 = 0
        worst_rank3 = 0
        n_deep2 = 0
        n_deep3 = 0
        rows = []
        for _, r in run1_topk.iterrows():
            key = r["config_key"]
            r2 = run2[run2["config_key"] == key]
            r3 = run3[run3["config_key"] == key]
            rank2 = int(r2["rank_in_run"].iloc[0]) if len(r2) else 999
            rank3 = int(r3["rank_in_run"].iloc[0]) if len(r3) else 999
            worst_rank2 = max(worst_rank2, rank2 if rank2 != 999 else 0)
            worst_rank3 = max(worst_rank3, rank3 if rank3 != 999 else 0)
            if rank2 > 20 and rank2 != 999: n_deep2 += 1
            if rank3 > 20 and rank3 != 999: n_deep3 += 1
            rows.append((int(r["rank_in_run"]), rank2, rank3))
        print(f"\n  top-{k}: configs from run 1 (seed=42) viewed by other runs:")
        print(f"    worst rank in run 2 (seed=0):   {worst_rank2}    (deep > 20: {n_deep2}/{k})")
        print(f"    worst rank in run 3 (seed=7):   {worst_rank3}    (deep > 20: {n_deep3}/{k})")
        print(f"    rank_run1 | rank_run2 | rank_run3")
        for r1, r2, r3 in rows:
            r2s = "absent" if r2 == 999 else f"{r2}"
            r3s = "absent" if r3 == 999 else f"{r3}"
            flag = "  <-- DEEP" if (r2 != 999 and r2 > 20) or (r3 != 999 and r3 > 20) else ""
            print(f"    {r1:>9} | {r2s:>9} | {r3s:>9}{flag}")


def main():
    print("Loading BoneCollagen ...")
    X, y_reg, y_cls, wavelengths = load_paired_data()
    print(f"  X: {X.shape}, n_classes: {len(np.unique(y_cls))}")

    analyze("regression", "PLS", X, y_reg, wavelengths)
    analyze("classification", "PLS-DA", X, y_cls, wavelengths)


if __name__ == "__main__":
    main()
