# LightGBM Shared-Model Fix — Before/After Verification

**Dataset:** BoneCollagen (49 ASD files + `%Collagen` target, 49 samples × 2151 wavelengths, 350–2500 nm).
**Harness:** `scripts/verify_shared_model_fix.py` — runs `run_search` with GUI-default kwargs for LightGBM and PLS. See script header for the defaults.
**Fix SHA:** `129bf46` (clone(model) at `search.py:2191,:4161,:4163`).

## Results matrix

| Run | git SHA | python | sklearn | LightGBM rows | LightGBM best_cv | LightGBM best_cal | PLS rows | PLS best_cv | PLS best_cal | bug_present |
|---|---|---|---|---|---|---|---|---|---|---|
| `.venv311` baseline | `085ef5c` | 3.11.9 | 1.5.2 | **0** (crash) | — | — | 973 | 1.5053957426145248 | 0.5242968634854136 | **TRUE** |
| `.venv312` baseline | `085ef5c` | 3.12.0 | 1.7.2 | 9408 | 0.9702327793086989 | 0.025748032302456404 | 973 | 1.5053957426145248 | 0.5242968634854136 | FALSE |
| `.venv311` post-fix | `129bf46` | 3.11.9 | 1.5.2 | 9408 | 0.9702327793086989 | 0.025748032302456404 | 973 | 1.5053957426145248 | 0.5242968634854136 | FALSE |
| `.venv312` post-fix | `129bf46` | 3.12.0 | 1.7.2 | 9408 | 0.9702327793086989 | 0.025748032302456404 | 973 | 1.5053957426145248 | 0.5242968634854136 | FALSE |

## What this proves

1. **Bug reproduced on pre-fix `.venv311`:** LightGBM raised `ValueError: X has 2135 features, but LGBMRegressor is expecting 2151 features as input` after emitting 1249 `"Could not fit model for parameter capture"` warnings. Whole LightGBM search bailed → `n_rows=0` (worse than the plan expected; actual symptom on main is total wipeout, not NaN rows).
2. **Bug latent on pre-fix `.venv312`:** same code, no exception (sklearn 1.7.2 tolerant). Produced 9408 LightGBM rows. Stderr was flooded with sklearn's `"X does not have valid feature names, but LGBMRegressor was fitted with feature names"` UserWarnings (~200k+) — the *silent* version of the shared-state mismatch.
3. **Fix eliminates the bug on `.venv311`:** post-fix LightGBM produces 9408 rows, zero warnings, zero NaN. Numerics **bit-identical** to `.venv312` baseline (the environment where the bug was latent).
4. **Fix is a no-op on `.venv312`:** post-fix numerics bit-identical to pre-fix. `clone()` doesn't perturb the result because fit() already resets `n_features_in_` — it only prevents the stale state from being observed between fit() calls.
5. **PLS is deterministic and unaffected** across all four runs — same best_cv/best_cal RMSE (1.5053957426145248 / 0.5242968634854136) everywhere. Good control.
6. **Side benefit on `.venv311`:** the feature-name UserWarnings (which sklearn 1.7.2 emits) don't exist as a warning path on 1.5.2, but more importantly clone() removes the shared-state mutation that was their upstream cause. Post-fix `.venv311` stderr is 40 bytes (one line) vs ~6MB / 100k lines on post-fix `.venv312`.

## Metric-level parity check (for the paranoid)

All three passing runs report:
- `LightGBM.best_cv_rmse = 0.9702327793086989` (identical to 16 significant figures)
- `LightGBM.best_cal_rmse = 0.025748032302456404`
- `LightGBM.median_cv_rmse = 2.1131579119266095`
- `LightGBM.median_cal_rmse = 0.39028291920559643`
- `PLS.*` — identical to all four runs (PLS was never affected).

Parity tolerance: exact equality (diff = 0.0 across all metrics). The fix changes only how often `n_features_in_` is stored — not any numeric computation.

## Runtime notes

- `.venv311` baseline: ~3 min (LightGBM crashed early, only PLS ran to completion).
- `.venv312` baseline: ~62 min (full LightGBM grid + heavy UserWarning overhead).
- `.venv311` post-fix: ~33 min (full LightGBM grid, zero warning overhead).
- `.venv312` post-fix: ~75 min (full grid, still has UserWarning overhead from sklearn 1.7.2 DataFrame→ndarray transitions — unrelated to our fix).

## Raw artifacts

- `baseline_venv311.json` + `.stderr`
- `baseline_venv312.json` + `.stderr` (7 MB)
- `postfix_venv311.json` + `.stderr` (40 bytes)
- `postfix_venv312.json` + `.stderr` (10 MB)
