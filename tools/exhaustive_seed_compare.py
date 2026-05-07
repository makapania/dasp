"""Empirical test: does multi-seed variance penalty change exhaustive_search rankings?

Question: GA's `evaluate_fitness_robust` runs 5 seeds and picks by `mean - 0.1*std`.
Exhaustive uses single-seed `evaluate_fitness`. If we ported multi-seed to
exhaustive, would the chosen top-K configs actually change in practice?

Design:
  * For all 14 × 17 = 238 (preproc, window) combos, evaluate fitness at 5 seeds
    with the same call exhaustive_search uses (PLS proxy, model_config=PLS,
    5-fold CV).
  * Compare three rankings:
      - single-seed (seed=42 only) — current exhaustive behavior
      - multi-seed mean — same as single-seed mean of the 5 seeds
      - multi-seed robust — mean - 0.1*std (what the GA path uses)
  * Report top-1 / top-3 / top-5 stability between single-seed and robust.

Usage:
  python tools/exhaustive_seed_compare.py                       # PLS regression
  python tools/exhaustive_seed_compare.py --task classification # PLS-DA
  python tools/exhaustive_seed_compare.py --quick               # 1st-deriv only (35 combos)
"""

from __future__ import annotations

import argparse
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
from spectral_predict.ga_preprocessing import (  # noqa: E402
    PREPROC_TYPES,
    WINDOW_SIZES,
    evaluate_fitness,
)

EXAMPLE_DIR = REPO_ROOT / "example"
LABEL_CSV = EXAMPLE_DIR / "BoneCollagen.csv"

SEEDS = [42, 0, 7, 100, 31]
VARIANCE_PENALTY = 0.1

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_paired_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load BoneCollagen labels + ASD spectra. Same loader as autoscale_bayesian_compare."""
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
    y_reg = labels["%Collagen"].to_numpy(dtype=float)

    le = LabelEncoder()
    y_cls = le.fit_transform(labels["CollagenCat"].to_numpy())

    return X, y_reg, y_cls


def evaluate_one(genes: np.ndarray, X: np.ndarray, y: np.ndarray, task_type: str, model_name: str, seed: int) -> float:
    """Wrap evaluate_fitness with the call shape exhaustive_search uses."""
    return evaluate_fitness(
        genes, X, y,
        cv_folds=5,
        n_components=10,
        task_type=task_type,
        random_state=seed,
        fitness_model="pls",
        model_config={"name": model_name, "params": {}},
    )


def run_sweep(
    X: np.ndarray, y: np.ndarray, task_type: str, model_name: str, combos: list[tuple[int, int]]
) -> pd.DataFrame:
    """Evaluate every combo at every seed; return long-format DataFrame."""
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, (p_idx, w_idx) in enumerate(combos):
        genes = np.array([p_idx, w_idx], dtype=np.int32)
        for seed in SEEDS:
            score = evaluate_one(genes, X, y, task_type, model_name, seed)
            rows.append({
                "preproc": PREPROC_TYPES[p_idx],
                "window": WINDOW_SIZES[w_idx],
                "p_idx": p_idx,
                "w_idx": w_idx,
                "seed": seed,
                "score": score,
            })
        if (i + 1) % 25 == 0 or (i + 1) == len(combos):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(combos)}] {PREPROC_TYPES[p_idx]:>14} w={WINDOW_SIZES[w_idx]:>2}  ({elapsed:.1f}s elapsed)")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Reduce long-format to one row per (preproc, window) with single-seed/mean/robust."""
    df_valid = df[df["score"] > -np.inf].copy()

    grouped = df_valid.groupby(["preproc", "window"])["score"].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)

    single_seed = df_valid[df_valid["seed"] == SEEDS[0]][["preproc", "window", "score"]].rename(
        columns={"score": "single_seed_score"}
    )
    grouped = grouped.merge(single_seed, on=["preproc", "window"], how="left")

    grouped["robust"] = grouped["mean"] - VARIANCE_PENALTY * grouped["std"]

    if task_type == "regression":
        grouped["single_seed_rmse"] = -grouped["single_seed_score"]
        grouped["mean_rmse"] = -grouped["mean"]
        grouped["robust_rmse"] = -grouped["robust"]
        grouped["rank_single"] = grouped["single_seed_rmse"].rank(method="min")
        grouped["rank_mean"] = grouped["mean_rmse"].rank(method="min")
        grouped["rank_robust"] = grouped["robust_rmse"].rank(method="min")
    else:
        grouped["rank_single"] = (-grouped["single_seed_score"]).rank(method="min")
        grouped["rank_mean"] = (-grouped["mean"]).rank(method="min")
        grouped["rank_robust"] = (-grouped["robust"]).rank(method="min")

    return grouped


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def report(grouped: pd.DataFrame, task_type: str) -> None:
    print()
    print("=" * 78)
    print("RANKING COMPARISON")
    print("=" * 78)

    rank_col_to_score = {
        "rank_single": "single_seed_score",
        "rank_mean": "mean",
        "rank_robust": "robust",
    }

    # Top-K stability between single-seed and robust
    for k in (1, 3, 5, 10):
        single_top = set(
            grouped.nsmallest(k, "rank_single").apply(lambda r: (r["preproc"], r["window"]), axis=1)
        )
        mean_top = set(
            grouped.nsmallest(k, "rank_mean").apply(lambda r: (r["preproc"], r["window"]), axis=1)
        )
        robust_top = set(
            grouped.nsmallest(k, "rank_robust").apply(lambda r: (r["preproc"], r["window"]), axis=1)
        )
        print(
            f"  top-{k:>2}: "
            f"jaccard(single,mean)={jaccard(single_top, mean_top):.2f}  "
            f"jaccard(single,robust)={jaccard(single_top, robust_top):.2f}  "
            f"jaccard(mean,robust)={jaccard(mean_top, robust_top):.2f}"
        )

    print()
    print("Top-1 picks per ranking:")
    for label, col in [("single-seed", "rank_single"), ("multi-seed mean", "rank_mean"), ("robust (mean-0.1*std)", "rank_robust")]:
        row = grouped.nsmallest(1, col).iloc[0]
        if task_type == "regression":
            print(f"  {label:>22}: {row['preproc']:>14} w={row['window']:>2}  "
                  f"RMSEcv mean={row['mean_rmse']:.4f}  std={row['std']:.4f}  "
                  f"robust={row['robust_rmse']:.4f}")
        else:
            print(f"  {label:>22}: {row['preproc']:>14} w={row['window']:>2}  "
                  f"acc mean={row['mean']:.4f}  std={row['std']:.4f}  "
                  f"robust={row['robust']:.4f}")

    print()
    print("Top-5 by ROBUST (mean - 0.1*std):")
    top5_robust = grouped.nsmallest(5, "rank_robust")
    for _, row in top5_robust.iterrows():
        if task_type == "regression":
            print(f"  {row['preproc']:>14} w={row['window']:>2}  "
                  f"single={row['single_seed_rmse']:.4f}  "
                  f"mean={row['mean_rmse']:.4f}  std={row['std']:.4f}  "
                  f"robust={row['robust_rmse']:.4f}  "
                  f"rank_single={int(row['rank_single']):>3}")
        else:
            print(f"  {row['preproc']:>14} w={row['window']:>2}  "
                  f"single={row['single_seed_score']:.4f}  "
                  f"mean={row['mean']:.4f}  std={row['std']:.4f}  "
                  f"robust={row['robust']:.4f}  "
                  f"rank_single={int(row['rank_single']):>3}")

    # ===== KEY DIAGNOSTIC =====
    # Practical question: does single-seed exhaustive sneak deep-in-mean configs
    # into its top-7 or top-10? The user picks top-N in practice; order within
    # that group doesn't matter, but a "rank 20" config masquerading as
    # "rank 7" is a real problem.
    print()
    print("=" * 78)
    print("SINGLE-SEED TOP-N vs MEAN RANK")
    print("=" * 78)
    for top_n in (7, 10):
        single_topn = grouped.nsmallest(top_n, "rank_single").sort_values("rank_single").copy()
        worst_mean_rank = int(single_topn["rank_mean"].max())
        n_worse_than_20 = int((single_topn["rank_mean"] > 20).sum())
        n_worse_than_15 = int((single_topn["rank_mean"] > 15).sum())
        print(f"\nSingle-seed top-{top_n}:")
        print(f"  worst mean-rank in this set: {worst_mean_rank}")
        print(f"  members with mean-rank > 15: {n_worse_than_15}")
        print(f"  members with mean-rank > 20: {n_worse_than_20}")
        print(f"  rank_single | rank_mean | preproc        | window | single  | mean    | std")
        for _, row in single_topn.iterrows():
            if task_type == "regression":
                single_v = row["single_seed_rmse"]
                mean_v = row["mean_rmse"]
            else:
                single_v = row["single_seed_score"]
                mean_v = row["mean"]
            print(
                f"  {int(row['rank_single']):>11} | {int(row['rank_mean']):>9} | "
                f"{row['preproc']:>14} | w={row['window']:>3} | "
                f"{single_v:.4f} | {mean_v:.4f} | {row['std']:.4f}"
            )

    # ===== INVERSE DIAGNOSTIC =====
    # Two-phase question: if phase-1 takes single-seed top-K and phase-2
    # rescores with multi-seed, how big does K have to be to guarantee
    # capturing the legit top-N (by mean)?
    print()
    print("=" * 78)
    print("MEAN TOP-N -> WORST SINGLE-SEED RANK (sets phase-1 pool size)")
    print("=" * 78)
    for top_n in (5, 7, 10, 15):
        mean_topn = grouped.nsmallest(top_n, "rank_mean").sort_values("rank_mean").copy()
        worst_single = int(mean_topn["rank_single"].max())
        n_above_20 = int((mean_topn["rank_single"] > 20).sum())
        n_above_30 = int((mean_topn["rank_single"] > 30).sum())
        n_above_50 = int((mean_topn["rank_single"] > 50).sum())
        print(f"\nMean top-{top_n}:")
        print(f"  worst single-seed rank in this set: {worst_single}")
        print(f"  -> phase-1 pool must be >= {worst_single} to catch all of mean top-{top_n}")
        print(f"  members with single-rank > 20: {n_above_20}")
        print(f"  members with single-rank > 30: {n_above_30}")
        print(f"  members with single-rank > 50: {n_above_50}")
        print(f"  rank_mean | rank_single | preproc        | window | mean    | single  | std")
        for _, row in mean_topn.iterrows():
            if task_type == "regression":
                single_v = row["single_seed_rmse"]
                mean_v = row["mean_rmse"]
            else:
                single_v = row["single_seed_score"]
                mean_v = row["mean"]
            flag = "  <-- DEEP" if int(row["rank_single"]) > 20 else ""
            print(
                f"  {int(row['rank_mean']):>9} | {int(row['rank_single']):>11} | "
                f"{row['preproc']:>14} | w={row['window']:>3} | "
                f"{mean_v:.4f} | {single_v:.4f} | {row['std']:.4f}{flag}"
            )

    print()
    print("Std distribution across all valid configs:")
    print(f"  median std: {grouped['std'].median():.4f}")
    print(f"  90th pct std: {grouped['std'].quantile(0.9):.4f}")
    print(f"  max std: {grouped['std'].max():.4f}")
    if task_type == "regression":
        rmse_range = grouped["mean_rmse"].max() - grouped["mean_rmse"].min()
        median_gap = grouped["mean_rmse"].nsmallest(2).diff().iloc[-1]
        print(f"  RMSEcv range across all configs: {rmse_range:.4f}")
        print(f"  RMSEcv gap between top-1 and top-2 (by mean): {median_gap:.4f}")
        print(f"  -> variance penalty (0.1*std) at median std ~= {0.1 * grouped['std'].median():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--quick", action="store_true", help="Restrict to 1st-derivative combos for fast smoke test")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path for raw long-format output")
    args = parser.parse_args()

    print(f"Loading BoneCollagen ...")
    X, y_reg, y_cls = load_paired_data()
    print(f"  X: {X.shape}, y_reg: {y_reg.shape}, y_cls classes: {len(np.unique(y_cls))}")

    if args.task == "regression":
        y = y_reg
        model_name = "PLS"
    else:
        y = y_cls
        model_name = "PLS-DA"

    if args.quick:
        combos = [(p, w) for p in [PREPROC_TYPES.index("deriv1"),
                                    PREPROC_TYPES.index("snv_deriv1"),
                                    PREPROC_TYPES.index("deriv1_snv"),
                                    PREPROC_TYPES.index("snv"),
                                    PREPROC_TYPES.index("raw")]
                  for w in range(len(WINDOW_SIZES))]
    else:
        combos = [(p, w) for p in range(len(PREPROC_TYPES)) for w in range(len(WINDOW_SIZES))]

    print(f"Sweeping {len(combos)} (preproc, window) combos x {len(SEEDS)} seeds = {len(combos) * len(SEEDS)} fits")
    print(f"Task: {args.task}, model: {model_name}")
    print()

    df = run_sweep(X, y, args.task, model_name, combos)
    grouped = aggregate(df, args.task)

    n_failed = int((df["score"] == -np.inf).sum())
    print(f"\nValid evaluations: {len(df) - n_failed}/{len(df)} (failures: {n_failed})")

    report(grouped, args.task)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nLong-format output -> {args.out}")


if __name__ == "__main__":
    main()
