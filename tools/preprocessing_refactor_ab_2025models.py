"""Phase 2 multi-seed A/B on the '2025 Model Samples' dataset.

Reruns the 2026-05-07 BoneCollagen A/B (legacy `ga_preprocess_phase2_n_seeds=0`
vs refactor `=5`) on a different dataset to test whether the "neutral on
quality" verdict replicates — and crucially, whether the ~5-6x wall-clock
cost is justified.

Per feedback_neutral_means_user_facing.md: "neutral" requires parity on
BOTH quality AND wall-clock. A feature that ties on R2pred but costs 5-6x
compute is a regression, not neutral.

This script runs regression only because the 2025 Model Samples dataset
provides a continuous Collagen Yield (no class column). That's the right
target anyway — Phase 2's previous BoneCollagen verdict was bit-identical
on regression, so this dataset stress-tests that finding on different data.

Usage:
  python tools/preprocessing_refactor_ab_2025models.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from spectral_predict.io import read_asd_dir
from preprocessing_refactor_ab import (
    cell_summary,
    score_full_leaderboard,
    run_one_cell,
    task_config,
)

DATASET_DIR = Path(r"C:\Users\mspon\Desktop\2025 Model Samples")
LABELS_XLSX = DATASET_DIR / "2025 Publication Sample IDs and Collagen Yields 2.xlsx"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def load_paired_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    asd_df, _meta = read_asd_dir(DATASET_DIR)
    asd_df.index = asd_df.index.astype(str).map(_norm)

    labels = pd.read_excel(LABELS_XLSX)
    # "File Name" column contains values like "10274_1.asd"; strip extension.
    labels["join_key"] = labels["File Name"].astype(str).map(
        lambda s: _norm(Path(s).stem)
    )

    common = asd_df.index.intersection(labels["join_key"])
    print(f"Join: ASD index n={len(asd_df.index)}, labels n={len(labels)}, "
          f"matched n={len(common)}")
    if len(common) < 30:
        # This script presumes a healthy join. If it's that small, dump
        # diagnostics instead of silently running on a tiny set.
        only_asd = sorted(set(asd_df.index) - set(labels["join_key"]))[:5]
        only_lbl = sorted(set(labels["join_key"]) - set(asd_df.index))[:5]
        raise RuntimeError(
            f"Insufficient matched samples ({len(common)}). "
            f"Only-asd sample: {only_asd!r}; only-label sample: {only_lbl!r}"
        )

    asd_df = asd_df.loc[common].sort_index()
    labels = labels.set_index("join_key").loc[common].sort_index()

    wavelengths = asd_df.columns.to_numpy(dtype=float)
    y_reg = labels["Collagen Yield"].to_numpy(dtype=float)
    return asd_df, y_reg, wavelengths


def quantile_stratified_split(y: np.ndarray, test_frac: float = 0.30,
                              seed: int = 0, n_bins: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Split into train/external using quantile bins of y as strata.
    Keeps the y distribution comparable across train/external for regression."""
    from sklearn.model_selection import train_test_split

    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_frac, stratify=bins, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--score-cap", type=int, default=300)
    parser.add_argument("--gap-thresh", type=float, default=0.10)
    parser.add_argument("--print-top", type=int, default=15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-top", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 2 multi-seed A/B - 2025 Model Samples (regression, single trajectory)")
    print("=" * 72)

    X_df, y_reg, wavelengths = load_paired_data()
    print(f"Loaded n={len(y_reg)} samples x {X_df.shape[1]} wavelengths")
    print(f"y range: [{y_reg.min():.2f}, {y_reg.max():.2f}], mean={y_reg.mean():.2f}, std={y_reg.std():.2f}")

    train_idx, val_idx = quantile_stratified_split(y_reg, seed=args.split_seed)
    X_train_df = X_df.iloc[train_idx]
    X_val = X_df.iloc[val_idx].to_numpy(dtype=float)
    y_train = y_reg[train_idx]
    y_val = y_reg[val_idx]
    print(f"Split (quantile-stratified): train n={len(train_idx)}, external n={len(val_idx)}")
    print(f"y_train mean={y_train.mean():.2f} std={y_train.std():.2f}; "
          f"y_val mean={y_val.mean():.2f} std={y_val.std():.2f}")
    print(f"Sweep: phase=exhaustive task=regression score_cap={args.score_cap} "
          f"gap_thresh={args.gap_thresh}")
    print()

    model_name, task_type = task_config("regression")
    cells: list[dict[str, Any]] = []
    arms = [("off", False), ("on", True)]
    for cell_i, (arm_label, arm_on) in enumerate(arms, start=1):
        tag = f"[{cell_i}/{len(arms)}] phase=exhaustive task=regression arm={arm_label}"
        print(tag + " ...", flush=True)
        try:
            summary = run_one_cell(
                X_train_df=X_train_df,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                wavelengths=wavelengths,
                model_name=model_name,
                task_type=task_type,
                phase="exhaustive",
                arm_on=arm_on,
                n_trials=0,    # not used for exhaustive
                n_starts=0,    # not used for exhaustive
                n_top=args.n_top,
                cv_folds=args.cv_folds,
                score_cap=args.score_cap,
                gap_thresh=args.gap_thresh,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        summary.update({
            "phase": "exhaustive", "task": "regression", "arm": arm_label,
            "arm_on": arm_on, "model": model_name,
        })
        cells.append(summary)
        if summary.get("ok"):
            n_pass = summary.get("n_passing", 0)
            passing = summary.get("passing", [])
            best_pass_str = (f"{passing[0]['R2pred']:.3f}@R2cv{passing[0]['R2cv']:.3f}"
                             if passing else "-")
            print(f"    ok (search {summary['elapsed_s_search']:.0f}s, "
                  f"score {summary['elapsed_s_score']:.0f}s) "
                  f"n_results={summary['n_results_total']} "
                  f"n_scored={summary['n_scored']} | "
                  f"n_passing={n_pass}  best passing R2pred={best_pass_str}")
        else:
            print(f"    FAILED: {summary.get('error')}")

    # Quality + wall-clock side-by-side report
    off = next((c for c in cells if c.get("ok") and c["arm"] == "off"), None)
    on = next((c for c in cells if c.get("ok") and c["arm"] == "on"), None)
    print()
    print("=" * 72)
    print(f"PASSING-SET REPORT (gap filter: |R2cv - R2pred| <= {args.gap_thresh:.2f})")
    print("=" * 72)
    if off is None or on is None:
        print(f"Cannot compare — off={'ok' if off else 'failed'}, on={'ok' if on else 'failed'}")
        return 1

    n_off, n_on = off["n_passing"], on["n_passing"]
    print(f"\n  n passing gap filter: off (legacy)={n_off}  on (refactor 5-seed)={n_on}")

    for arm_label, cell in [("OFF (legacy n_seeds=0)", off), ("ON (refactor n_seeds=5)", on)]:
        rows = cell["passing"][: args.print_top]
        print(f"\n  arm={arm_label} - top {len(rows)} passing models by R2pred:")
        print(f"    {'rank':>4}  {'R2pred':>7}  {'R2cv':>7}  {'gap':>6}  {'n_vars':>6}  preprocess")
        for i, r in enumerate(rows):
            print(f"    {i+1:>4}  {r['R2pred']:>7.4f}  {r['R2cv']:>7.4f}  "
                  f"{r['gap']:>6.4f}  {r['n_vars']:>6}  {r['preprocess']}")

    off_set = {r["fp"] for r in off["passing"]}
    on_set = {r["fp"] for r in on["passing"]}
    shared = off_set & on_set
    only_off = off_set - on_set
    only_on = on_set - off_set
    print(f"\n  full passing-set diff (row fingerprint):")
    print(f"    shared (passing in BOTH arms):    {len(shared)}")
    print(f"    unique to OFF (legacy only):      {len(only_off)}")
    print(f"    unique to ON  (refactor only):    {len(only_on)}")
    only_on_rows = [r for r in on["passing"] if r["fp"] in only_on][:5]
    only_off_rows = [r for r in off["passing"] if r["fp"] in only_off][:5]
    if only_on_rows:
        print(f"    *** top passing models UNIQUE to refactor (would be lost without it):")
        for r in only_on_rows:
            print(f"        R2pred={r['R2pred']:.4f} R2cv={r['R2cv']:.4f} gap={r['gap']:.4f}  {r['preprocess']}")
    if only_off_rows:
        print(f"    *** top passing models UNIQUE to legacy (would be lost WITH refactor):")
        for r in only_off_rows:
            print(f"        R2pred={r['R2pred']:.4f} R2cv={r['R2cv']:.4f} gap={r['gap']:.4f}  {r['preprocess']}")

    off_best = off["passing"][0]["R2pred"] if off["passing"] else float("nan")
    on_best = on["passing"][0]["R2pred"] if on["passing"] else float("nan")
    off_search = off["elapsed_s_search"]
    on_search = on["elapsed_s_search"]

    print(f"\n  BEST passing R2pred:  off={off_best:.4f}  on={on_best:.4f}  delta={on_best-off_best:+.4f}")
    print(f"  WALL TIME (search):   off={off_search:.0f}s  on={on_search:.0f}s  "
          f"ratio={on_search/off_search:.2f}x  delta={on_search-off_search:+.0f}s")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    delta_q = on_best - off_best
    ratio_t = on_search / off_search if off_search > 0 else float("inf")
    if abs(delta_q) < 0.01 and ratio_t > 1.5:
        print(f"  REGRESSION: tied on R2pred (delta={delta_q:+.4f}) but {ratio_t:.1f}x slower "
              f"(+{on_search-off_search:.0f}s). Per neutral-means-user-facing rule, "
              f"this is net-negative -- the 5-seed rescore costs wall-clock for zero "
              f"quality gain.")
    elif delta_q > 0.01:
        print(f"  WIN-FOR-REFACTOR: R2pred +{delta_q:.4f} on this dataset, costs "
              f"{ratio_t:.1f}x wall time. Worth keeping as opt-in if the lift "
              f"replicates on a third dataset.")
    elif delta_q < -0.01:
        print(f"  WIN-FOR-LEGACY: refactor LOSES R2pred {delta_q:+.4f} AND costs "
              f"{ratio_t:.1f}x wall time. Confirms net-negative; rip-out warranted.")
    else:
        print(f"  delta R2pred={delta_q:+.4f}, wall-time ratio {ratio_t:.2f}x. "
              f"Tied on quality AND tied on wall-time -- genuinely neutral on both axes "
              f"by the neutral-means-user-facing rule. See passing-set diff above for "
              f"whether the arms find different models.")

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w") as f:
            json.dump({"cells": cells, "args": vars(args)}, f, indent=2, default=str)
        print(f"\n  results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
