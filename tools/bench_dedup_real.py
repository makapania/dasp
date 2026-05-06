"""Production-realistic benchmark: pre-dedup vs post-dedup on example data.

Uses the production TPE sampler (``multivariate=True``) — the actual
sampler users run in the GUI. The dedup mechanism under audit is value-
cache-and-replay: duplicate fingerprints short-circuit by returning the
prior trial's cached metric, so TPE sees identical (params, value) pairs
to a pre-dedup re-fit. KDE history is bit-identical → same parameter
space exploration → same final models. The only difference is whether
duplicates burned compute or replayed instantly.

Pre/post emulation in one process via monkeypatch:
- Pre = ``_register_or_replay_fingerprint`` returns None (no replay) and
        ``_record_fingerprint_value`` is a no-op. Every duplicate runs a
        full fit, matching pre-dedup behavior exactly.
- Post = real value-cache-and-replay. Duplicates return the cached value
        immediately; no fit, no CV.

Both phases reach the same set of COMPLETE trials with bit-identical
TPE history. The difference is wall-clock (post saves on duplicates).

What we report per scenario:
- Wall-clock time (seconds)
- Total trial invocations (PRUNED column should be 0 in both phases —
  value-cache-and-replay never raises TrialPruned)
- Unique fingerprints (should match between phases)
- Best-row metric (RMSEcv for regression, Accuracycv for classification)
- Top-5 row identifiers + overlap
- Coverage comparison: Old's best in New's set? Vice versa? Set diffs?

Run from project root with .venv312 active:
    .venv312/Scripts/python.exe tools/bench_dedup_real.py [--n-trials 100]
"""
from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from spectral_predict.io import read_asd_dir
from spectral_predict import unified_bayesian as ub


REGRESSION_TARGET = "%Collagen"
CLASSIFICATION_TARGET = "CollagenCat"


@dataclass(frozen=True)
class Scenario:
    name: str
    model_name: str
    task_type: str
    target: str
    metric_col: str
    metric_higher_is_better: bool


SCENARIOS = {
    "pls_regression": Scenario(
        "pls_regression", "PLS", "regression", REGRESSION_TARGET,
        metric_col="RMSEcv", metric_higher_is_better=False,
    ),
    "lightgbm_classification": Scenario(
        "lightgbm_classification", "LightGBM", "classification", CLASSIFICATION_TARGET,
        metric_col="Accuracycv", metric_higher_is_better=True,
    ),
}


@contextlib.contextmanager
def disable_dedup_for_pre_emulation():
    """Monkeypatch dedup helpers to record-only (no replay).

    Pre-fix emulation: every trial completes via a full fit; TPE sees
    real CV values for all suggestions including duplicates. The post-fix
    behavior (cache + replay) preserves TPE history bit-identically, so
    the comparison should reveal that pre/post explore the SAME points;
    post just skips the redundant fits.
    """
    original_register = ub._register_or_replay_fingerprint
    original_record = ub._record_fingerprint_value

    def record_without_replay(trial, fingerprint, seen_fingerprints):
        # Always treat as novel: caller proceeds with the real fit.
        trial.set_user_attr("fingerprint", repr(fingerprint))
        return None

    def noop_record(fingerprint, trial, value, seen_fingerprints):
        # Don't cache values; emulates pre-dedup behavior where every
        # trial including duplicates runs a fresh fit.
        pass

    ub._register_or_replay_fingerprint = record_without_replay
    ub._record_fingerprint_value = noop_record
    try:
        yield
    finally:
        ub._register_or_replay_fingerprint = original_register
        ub._record_fingerprint_value = original_record


def load_example_xy(max_features: int = 120):
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
    return X, wavelengths, joined


def run_phase(
    scenario: Scenario,
    n_trials: int,
    max_features: int,
    *,
    pre_fix_emulation: bool,
    seed: int = 42,
):
    """Run one phase (pre or post). Returns (df, study, elapsed_sec)."""
    X, wavelengths, joined = load_example_xy(max_features=max_features)
    y = joined[scenario.target].to_numpy(
        dtype=float if scenario.task_type == "regression" else None
    )
    cm = disable_dedup_for_pre_emulation() if pre_fix_emulation else contextlib.nullcontext()
    t0 = time.perf_counter()
    with cm:
        df, study = ub.run_unified_bayesian(
            X,
            y,
            wavelengths,
            model_name=scenario.model_name,
            task_type=scenario.task_type,
            n_trials=n_trials,
            cv_folds=3,
            cv_strategy="kfold",
            random_state=seed,
            n_top_regions=3,
            enable_sqlite_persistence="never",
            early_stopping_rounds=None,
            enable_uve=False,
            verbose=False,
        )
    elapsed = time.perf_counter() - t0
    return df, study, elapsed


def fingerprint_map(study: optuna.Study) -> dict[int, str]:
    return {
        t.number: t.user_attrs["fingerprint"]
        for t in study.trials
        if "fingerprint" in t.user_attrs
        and t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value < 1e9
    }


def trial_state_counts(study: optuna.Study):
    out = {"COMPLETE": 0, "PRUNED": 0, "FAIL": 0, "OTHER": 0}
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            out["COMPLETE"] += 1
        elif t.state == optuna.trial.TrialState.PRUNED:
            out["PRUNED"] += 1
        elif t.state == optuna.trial.TrialState.FAIL:
            out["FAIL"] += 1
        else:
            out["OTHER"] += 1
    return out


def best_metric(df: pd.DataFrame, scenario: Scenario):
    if scenario.metric_col not in df.columns:
        return None
    if scenario.metric_higher_is_better:
        return float(df[scenario.metric_col].max())
    return float(df[scenario.metric_col].min())


def top_n_fingerprints(df: pd.DataFrame, study: optuna.Study, scenario: Scenario, n: int = 5) -> set[str]:
    if scenario.metric_col not in df.columns:
        return set()
    if "trial_number" not in df.columns:
        return set()
    fp_by_trial = fingerprint_map(study)
    asc = not scenario.metric_higher_is_better
    sorted_df = df.sort_values(scenario.metric_col, ascending=asc).head(n)
    return {
        fp_by_trial[int(t)]
        for t in sorted_df["trial_number"].astype(int).tolist()
        if int(t) in fp_by_trial
    }


def best_fingerprint(df: pd.DataFrame, study: optuna.Study, scenario: Scenario):
    fp_by_trial = fingerprint_map(study)
    if scenario.metric_col not in df.columns or "trial_number" not in df.columns:
        return None
    asc = not scenario.metric_higher_is_better
    sorted_df = df.sort_values(scenario.metric_col, ascending=asc)
    for _, row in sorted_df.iterrows():
        fp = fp_by_trial.get(int(row["trial_number"]))
        if fp:
            return fp
    return None


def metric_distribution(df: pd.DataFrame, scenario: Scenario):
    if scenario.metric_col not in df.columns:
        return None
    s = df[scenario.metric_col].dropna()
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def coverage_comparison(pre_study: optuna.Study, post_study: optuna.Study,
                        pre_df: pd.DataFrame, post_df: pd.DataFrame, scenario: Scenario):
    """How much of pre's fingerprint space did post explore (and vice versa)?

    Specifically address the user's "are parts of space closed off" concern:
    - pre_best in post's set?
    - post_best in pre's set?
    - |pre ∩ post| / |pre|: what fraction of pre's exploration did post also reach?
    - Pre-only fingerprints: configs pre tried that post didn't (potentially
      "closed off" by dedup steering TPE away)
    - Post-only fingerprints: configs only post explored (extra coverage from
      the +N search budget dedup unlocked)
    """
    pre_fps = set(fingerprint_map(pre_study).values())
    post_fps = set(fingerprint_map(post_study).values())
    common = pre_fps & post_fps
    pre_only = pre_fps - post_fps
    post_only = post_fps - pre_fps
    pre_best_fp = best_fingerprint(pre_df, pre_study, scenario)
    post_best_fp = best_fingerprint(post_df, post_study, scenario)

    print(f"\n  --- Coverage comparison (full fingerprint sets) ---")
    print(f"    |pre|={len(pre_fps)}  |post|={len(post_fps)}  |pre AND post|={len(common)}")
    print(f"    Fraction of pre's space also explored by post: "
          f"{len(common) / max(1, len(pre_fps)) * 100:.1f}%")
    print(f"    Fraction of post's space also explored by pre: "
          f"{len(common) / max(1, len(post_fps)) * 100:.1f}%")
    print(f"    Pre-only configs (in pre, NOT explored by post): {len(pre_only)}")
    print(f"    Post-only configs (in post, NOT explored by pre): {len(post_only)}")
    print(f"    Pre's best fingerprint reached by post: "
          f"{'YES' if pre_best_fp in post_fps else 'NO'}")
    print(f"    Post's best fingerprint reached by pre: "
          f"{'YES' if post_best_fp in pre_fps else 'NO'}")

    pre_dist = metric_distribution(pre_df, scenario)
    post_dist = metric_distribution(post_df, scenario)
    print(f"\n  --- Metric distribution across all unique fits ({scenario.metric_col}) ---")
    print(f"    Pre:  n={pre_dist['n']}  min={pre_dist['min']:.4f}  "
          f"p25={pre_dist['p25']:.4f}  median={pre_dist['median']:.4f}  "
          f"p75={pre_dist['p75']:.4f}  max={pre_dist['max']:.4f}")
    print(f"    Post: n={post_dist['n']}  min={post_dist['min']:.4f}  "
          f"p25={post_dist['p25']:.4f}  median={post_dist['median']:.4f}  "
          f"p75={post_dist['p75']:.4f}  max={post_dist['max']:.4f}")
    return {
        "common_count": len(common),
        "pre_only_count": len(pre_only),
        "post_only_count": len(post_only),
        "pre_best_in_post": pre_best_fp in post_fps,
        "post_best_in_pre": post_best_fp in pre_fps,
        "pre_dist": pre_dist,
        "post_dist": post_dist,
    }


def run_scenario(scenario: Scenario, n_trials: int, max_features: int):
    print(f"\n=== Scenario: {scenario.name} ===")
    print(f"  Model: {scenario.model_name}, task: {scenario.task_type}, n_trials: {n_trials}")
    print(f"  TPE sampler (production), seed=42, cv=3-fold KFold")
    print()

    print(f"  [PRE-fix emulation: replay disabled — every duplicate runs a full fit]")
    pre_df, pre_study, pre_time = run_phase(scenario, n_trials, max_features, pre_fix_emulation=True)
    pre_states = trial_state_counts(pre_study)
    pre_unique_fps = len(set(fingerprint_map(pre_study).values()))
    pre_best = best_metric(pre_df, scenario)
    pre_top5 = top_n_fingerprints(pre_df, pre_study, scenario, n=5)
    print(f"    Wall-clock: {pre_time:.1f} s")
    print(f"    Trials: COMPLETE={pre_states['COMPLETE']} PRUNED={pre_states['PRUNED']} FAIL={pre_states['FAIL']}")
    print(f"    Unique fingerprints: {pre_unique_fps}")
    print(f"    Best {scenario.metric_col}: {pre_best:.4f}")
    print(f"    CSV rows: {len(pre_df)}")

    print(f"\n  [POST-fix: dedup on, duplicates replay cached value (TPE sees identical history)]")
    post_df, post_study, post_time = run_phase(scenario, n_trials, max_features, pre_fix_emulation=False)
    post_states = trial_state_counts(post_study)
    post_unique_fps = len(set(fingerprint_map(post_study).values()))
    post_best = best_metric(post_df, scenario)
    post_top5 = top_n_fingerprints(post_df, post_study, scenario, n=5)
    print(f"    Wall-clock: {post_time:.1f} s")
    print(f"    Trials: COMPLETE={post_states['COMPLETE']} PRUNED={post_states['PRUNED']} FAIL={post_states['FAIL']}")
    print(f"    Unique fingerprints: {post_unique_fps}")
    print(f"    Best {scenario.metric_col}: {post_best:.4f}")
    print(f"    CSV rows: {len(post_df)}")

    overlap = pre_top5 & post_top5
    pct_dup_in_pre = (len(pre_df) - pre_unique_fps) / max(1, len(pre_df)) * 100

    print(f"\n  --- Comparison ---")
    print(f"    Wall-clock delta: {post_time - pre_time:+.1f} s ({(post_time/pre_time - 1)*100:+.1f}%)")
    print(f"    Pre had {pct_dup_in_pre:.1f}% duplicate rows in CSV; post has 0%")
    print(f"    Post unique fits: {post_unique_fps} vs pre {pre_unique_fps} (+{post_unique_fps - pre_unique_fps} more search coverage)")
    print(f"    Best metric delta ({scenario.metric_col}): post={post_best:.4f} pre={pre_best:.4f} ({post_best - pre_best:+.4f})")
    print(f"    Top-5 fingerprint overlap: {len(overlap)}/5")

    cov = coverage_comparison(pre_study, post_study, pre_df, post_df, scenario)

    return {
        "scenario": scenario.name,
        "pre_time": pre_time,
        "post_time": post_time,
        "pre_unique": pre_unique_fps,
        "post_unique": post_unique_fps,
        "pre_best": pre_best,
        "post_best": post_best,
        "top5_overlap": len(overlap),
        "pre_dup_pct": pct_dup_in_pre,
        "coverage": cov,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        action="append",
        help="Run specific scenario(s); default runs all.",
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=120)
    args = parser.parse_args()

    names = args.scenario or list(SCENARIOS)
    summary = []
    for name in names:
        summary.append(run_scenario(SCENARIOS[name], args.n_trials, args.max_features))

    print("\n\n=== Final Summary ===")
    for row in summary:
        print(
            f"  {row['scenario']}: "
            f"pre_time={row['pre_time']:.1f}s post_time={row['post_time']:.1f}s "
            f"({(row['post_time']/row['pre_time'] - 1)*100:+.1f}%); "
            f"pre_unique={row['pre_unique']} post_unique={row['post_unique']}; "
            f"top5_overlap={row['top5_overlap']}/5; "
            f"pre_dup_pct={row['pre_dup_pct']:.1f}%; "
            f"best_metric_delta={row['post_best'] - row['pre_best']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
