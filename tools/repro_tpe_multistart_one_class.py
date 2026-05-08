"""
Reproduction harness for the 2026-05-07 TPE multi-start one_class crash.

Symptom: GUI Tk main loop dies ~2 seconds after multi-start TPE begins on a
one_class Quick analysis. The Python process keeps running but the GUI is
gone. The crash log contains zombie tk.Variable destructors but no real
traceback.

Two hypotheses to distinguish:

  H1 — backend crash. Optuna / scipy KDE multivariate / LightGBM segfault
       or hard-error on the user's data shape. If true, this script will
       crash the same way (or print a real traceback).

  H2 — GUI thread-safety. The backend is fine; the GUI worker-thread
       interaction with Tk is what dies. If true, this script will run
       cleanly to completion.

Usage (full repro matching production GUI defaults):
    python tools/repro_tpe_multistart_one_class.py 75 5

Fast smoke (verifies the wrapper boots and the first study completes):
    python tools/repro_tpe_multistart_one_class.py

The script generates synthetic data sized to match the user's run (40
calibration samples × 2151 features, ~5 outliers) and exercises
``run_tpe_multistart_preprocessing_discovery`` with these settings:

  - n_trials  -> argv[1] (default 10; production GUI uses 75)
  - n_starts  -> argv[2] (default 3;  production GUI uses 5)
  - per_start_pool=7
  - n_seeds=5
  - task_type='one_class'

If you want to mirror the user's actual data, swap ``_make_synth_data``
for a call that loads ``example/BoneCollagen.csv`` (or whatever the user
was running) and binarises the target on inlier_class='High'.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np


def _make_synth_data(n_samples: int = 40, n_features: int = 2151, n_outliers: int = 5,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Match the user's data shape: 40 calibration samples, ~5 outliers, 2151 wavelengths."""
    rng = np.random.default_rng(seed)
    inlier_count = n_samples - n_outliers
    inlier_X = rng.normal(0, 1, size=(inlier_count, n_features))
    outlier_X = rng.normal(3, 1, size=(n_outliers, n_features))
    X = np.vstack([inlier_X, outlier_X])
    # +1 inlier, -1 outlier per dasp's one_class label convention.
    y = np.concatenate([np.ones(inlier_count), -np.ones(n_outliers)]).astype(int)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def main() -> int:
    print("=" * 72)
    print("REPRO: TPE multi-start one_class crash")
    print(f"Python: {sys.version.split()[0]} on {sys.platform}")
    print("=" * 72)

    X, y = _make_synth_data()
    print(f"Data: X{X.shape}, y inliers={int((y == 1).sum())}, outliers={int((y == -1).sum())}")

    from spectral_predict.tpe_preprocessing_discovery import (
        run_tpe_multistart_preprocessing_discovery,
    )

    callback_calls: list[tuple[int, int, str]] = []

    def progress_callback(current: int, total: int, message: str) -> None:
        callback_calls.append((current, total, message))
        print(f"  [{current}/{total}] {message}")

    # Fast settings for repro — full search would take 5-10 minutes. The bug
    # surfaces ~2s after the wrapper begins (per the user's crash log), so a
    # truncated search is sufficient to discriminate H1 from H2.
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    t0 = time.time()
    try:
        configs = run_tpe_multistart_preprocessing_discovery(
            X,
            y,
            task_type="one_class",
            n_trials=n_trials,
            n_top=10,
            cv_folds=5,
            enable_autoscale=True,
            enable_baseline=True,
            enable_smoothing=True,
            smoothing_window=17,
            smoothing_polyorder=2,
            n_starts=n_starts,
            per_start_pool=7,
            n_seeds=5,
            progress_callback=progress_callback,
        )
    except BaseException:
        elapsed = time.time() - t0
        print(f"\nFAIL at {elapsed:.1f}s — backend raised:")
        traceback.print_exc()
        print(f"\ncallback fires before crash: {len(callback_calls)}")
        return 1

    elapsed = time.time() - t0
    print(f"\nOK — completed in {elapsed:.1f}s, {len(configs)} configs returned.")
    print(f"callback fires: {len(callback_calls)}")
    print(
        "Verdict: backend ran cleanly. The GUI crash is NOT in the multi-start "
        "wrapper itself — it is the GUI/Tk worker-thread interaction. Look for "
        "messagebox / tk.Variable.get() / widget.config() calls reachable from "
        "the analysis thread that don't go through root.after."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
