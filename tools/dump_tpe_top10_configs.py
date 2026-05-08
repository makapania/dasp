"""Print the top-10 preprocessing configs from BOTH refactor phases on
BoneCollagen — Phase 4 (TPE single-start vs multistart) AND Phase 2
(exhaustive single-seed vs multi-seed rescore).

Direct calls into the discovery functions — no full `run_search` overhead.
This shows what the user actually sees as the 10 distinct preprocessing
configs that get fed downstream into model × varsel × hyperparam expansion.

Usage:
  python tools/dump_tpe_top10_configs.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from spectral_predict.io import read_asd_dir
from spectral_predict.tpe_preprocessing_discovery import (
    run_tpe_preprocessing_discovery,
    run_tpe_multistart_preprocessing_discovery,
)
from spectral_predict.ga_preprocessing import exhaustive_search

EXAMPLE_DIR = REPO_ROOT / "example"
LABEL_CSV = EXAMPLE_DIR / "BoneCollagen.csv"

warnings.filterwarnings("ignore")


def load_data():
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


def fmt_tpe_cfg(cfg: dict) -> str:
    pre = cfg.get("preprocessing", "?")
    window = cfg.get("window")
    deriv = cfg.get("deriv")
    extras = []
    if cfg.get("_tpe_baseline_method"):
        extras.append(cfg["_tpe_baseline_method"])
    if cfg.get("_tpe_smoothing"):
        extras.append("sg0")
    if cfg.get("_tpe_autoscale"):
        extras.append("autoscale")
    bits = [pre]
    if deriv:
        bits.append(f"d{deriv}")
    if window:
        bits.append(f"w={window}")
    if extras:
        bits.append("[" + "+".join(extras) + "]")
    return " ".join(bits)


def fmt_exh_cfg(cfg: dict) -> str:
    name = cfg.get("name", "?")
    deriv = cfg.get("deriv")
    window = cfg.get("window")
    bits = [str(name)]
    if deriv:
        bits.append(f"d{deriv}")
    if window:
        bits.append(f"w={window}")
    # autoscale gene
    genes = cfg.get("genes")
    if genes is not None and len(genes) >= 3 and bool(genes[2]):
        bits.append("[autoscale]")
    return " ".join(bits)


def print_table(label: str, configs: list[dict], task_type: str, fmt_fn) -> None:
    score_label = "RMSE" if task_type == "regression" else "score"
    print(f"\n--- {label} (task={task_type}) ---")
    print(f"{'rank':>4}  {'score':>10}  config")
    score_field = "score" if "score" in (configs[0] if configs else {}) else "rmsecv"
    for i, cfg in enumerate(configs[:10]):
        sc = cfg.get(score_field, cfg.get("rmsecv", float("nan")))
        print(f"{i+1:>4}  {sc:>10.4f}  {fmt_fn(cfg)}")


def tpe_keyset(cfgs):
    return {(c.get("preprocessing"), c.get("window"), c.get("deriv"),
             c.get("_tpe_autoscale"), c.get("_tpe_baseline_method"),
             c.get("_tpe_smoothing")) for c in cfgs[:10]}


def exh_keyset(cfgs):
    out = set()
    for c in cfgs[:10]:
        genes = c.get("genes")
        autoscale = bool(genes[2]) if genes is not None and len(genes) >= 3 else False
        out.add((c.get("name"), c.get("window"), c.get("deriv"), autoscale))
    return out


def main() -> int:
    X, y_reg, y_cls = load_data()
    print(f"Loaded n={X.shape[0]} samples × {X.shape[1]} wavelengths")

    for task, y in [("regression", y_reg), ("classification", y_cls)]:
        print(f"\n{'='*72}")
        print(f"TASK: {task}")
        print('='*72)

        # PHASE 4: TPE single vs multistart
        print(f"\n##### PHASE 4: TPE preprocessing discovery #####")
        single = run_tpe_preprocessing_discovery(
            X, y, task_type=task,
            n_trials=75, n_top=10, cv_folds=5,
            enable_autoscale=True, enable_baseline=False, enable_smoothing=False,
            progress_callback=None,
        )
        print_table("TPE single-start (legacy)", single, task, fmt_tpe_cfg)

        multi = run_tpe_multistart_preprocessing_discovery(
            X, y, task_type=task,
            n_trials=75, n_top=10, cv_folds=5,
            enable_autoscale=True, enable_baseline=False, enable_smoothing=False,
            n_starts=5, progress_callback=None,
        )
        print_table("TPE multistart (refactor)", multi, task, fmt_tpe_cfg)

        s = tpe_keyset(single)
        m = tpe_keyset(multi)
        print(f"\n  TPE set diff (top-10):")
        print(f"    shared:           {len(s & m)}/{min(len(s), len(m))}")
        print(f"    only single:      {len(s - m)}")
        print(f"    only multistart:  {len(m - s)}")
        for k in sorted(m - s, key=str):
            print(f"      ONLY-multi: pre={k[0]} window={k[1]} deriv={k[2]} "
                  f"auto={k[3]} baseline={k[4]} smooth={k[5]}")
        for k in sorted(s - m, key=str):
            print(f"      ONLY-single: pre={k[0]} window={k[1]} deriv={k[2]} "
                  f"auto={k[3]} baseline={k[4]} smooth={k[5]}")

        # PHASE 2: exhaustive single-seed vs multi-seed rescore
        print(f"\n##### PHASE 2: exhaustive preprocessing search #####")
        exh_off = exhaustive_search(
            X, y, task_type=task, cv_folds=5, n_components=10,
            apply_autoscale=True, top_n=10, phase2_n_seeds=0, verbose=0,
        )
        print_table("Exhaustive phase2 OFF (legacy)", exh_off["configs"], task, fmt_exh_cfg)

        exh_on = exhaustive_search(
            X, y, task_type=task, cv_folds=5, n_components=10,
            apply_autoscale=True, top_n=10, phase2_n_seeds=5, verbose=0,
        )
        print_table("Exhaustive phase2 ON (refactor)", exh_on["configs"], task, fmt_exh_cfg)

        s_e = exh_keyset(exh_off["configs"])
        m_e = exh_keyset(exh_on["configs"])
        print(f"\n  Exhaustive set diff (top-10):")
        print(f"    shared:           {len(s_e & m_e)}/{min(len(s_e), len(m_e))}")
        print(f"    only phase2 OFF:  {len(s_e - m_e)}")
        print(f"    only phase2 ON:   {len(m_e - s_e)}")
        for k in sorted(m_e - s_e, key=str):
            print(f"      ONLY-on: name={k[0]} window={k[1]} deriv={k[2]} auto={k[3]}")
        for k in sorted(s_e - m_e, key=str):
            print(f"      ONLY-off: name={k[0]} window={k[1]} deriv={k[2]} auto={k[3]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
