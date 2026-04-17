# LightGBM Shared-Model Fix — Before/After Verification

**Dataset:** BoneCollagen (49 ASD files + `%Collagen` target, 49 samples × 2151 wavelengths, 350–2500 nm).
**Harness:** `scripts/verify_shared_model_fix.py` — runs `run_search` with GUI-default kwargs for LightGBM and PLS. See script header for the defaults.
**Initial regression-fix SHA:** `129bf46` (three `clone(model)` sites at `search.py:2191,:4161,:4163`). The PLS-DA site at `:4139` was added later in `1fd222c` after pr-review feedback — see the Classification matrix section below.

## Results matrix

| Run | git SHA | python | sklearn | LightGBM rows | LightGBM best_cv | LightGBM best_cal | PLS rows | PLS best_cv | PLS best_cal | bug_present |
|---|---|---|---|---|---|---|---|---|---|---|
| `.venv311` baseline | `085ef5c` | 3.11.9 | 1.5.2 | **0** (crash) | — | — | 973 | 1.5053957426145248 | 0.5242968634854136 | **TRUE** |
| `.venv312` baseline | `085ef5c` | 3.12.0 | 1.7.2 | 9408 | 0.9702327793086989 | 0.025748032302456404 | 973 | 1.5053957426145248 | 0.5242968634854136 | FALSE |
| `.venv311` post-fix | `129bf46` | 3.11.9 | 1.5.2 | 9408 | 0.9702327793086989 | 0.025748032302456404 | 973 | 1.5053957426145248 | 0.5242968634854136 | FALSE |
| `.venv312` post-fix | `129bf46` | 3.12.0 | 1.7.2 | 9408 | 0.9702327793086989 | 0.025748032302456404 | 973 | 1.5053957426145248 | 0.5242968634854136 | FALSE |

## What this proves

1. **Bug reproduced on pre-fix `.venv311`:** LightGBM raised `ValueError: X has 2135 features, but LGBMRegressor is expecting 2151 features as input` after emitting 1249 `"Could not fit model for parameter capture"` warnings. Whole LightGBM search bailed → `n_rows=0` (worse than the plan expected; actual symptom on main is total wipeout, not NaN rows).
2. **No crash on pre-fix `.venv312`:** same code, no exception (sklearn 1.7.2's pre-fit check is more lenient). Produced 9408 LightGBM rows. Stderr also contained ~112k `"X does not have valid feature names"` UserWarnings from sklearn 1.7.2 — these come from the DataFrame-fit / ndarray-predict transition in the pipeline and are **unrelated** to the shared-state bug. They appear in the same quantity before and after the fix (`baseline_venv312.stderr.gz` and `postfix_venv312.stderr.gz` decompress to byte-identical content, md5 `7064c68d141c876da088fa1df85e0d3f`). The fix does NOT address them.
3. **Fix eliminates the crash on `.venv311`:** post-fix LightGBM produces 9408 rows, zero errors. Numerics **bit-identical** to `.venv312` baseline (the environment where the bug never crashed).
4. **Fix is a numeric no-op on `.venv312`:** post-fix metrics bit-identical to pre-fix. `clone()` doesn't change computation because `fit()` already resets `n_features_in_` — it only prevents the stale state from being observed between fit() calls.
5. **PLS is deterministic and unaffected** across all four runs — same best_cv/best_cal RMSE (1.5053957426145248 / 0.5242968634854136) everywhere. Good control.
6. **Why `.venv311` stderr is quiet and `.venv312` isn't:** the sklearn 1.7.2 feature-name check that produces the flood of UserWarnings on `.venv312` simply doesn't exist as a warning path in sklearn 1.5.2 — no relation to the fix. On `.venv311` pre-fix we never reached the predict path for LightGBM (crash bailed early), so stderr was also quiet; on `.venv311` post-fix the whole run succeeds silently because 1.5.2 lacks that warning entirely.

## Metric-level parity check (for the paranoid)

All three passing runs report:
- `LightGBM.best_cv_rmse = 0.9702327793086989` (identical to 16 significant figures)
- `LightGBM.best_cal_rmse = 0.025748032302456404`
- `LightGBM.median_cv_rmse = 2.1131579119266095`
- `LightGBM.median_cal_rmse = 0.39028291920559643`
- `PLS.*` — identical to all four runs (PLS was never affected).

Parity tolerance: exact equality (diff = 0.0 across all metrics). The fix changes only how often `n_features_in_` is stored — not any numeric computation.

## Classification matrix (added after Claude pr-reviewer flagged PLS-DA branch)

The original 4-run matrix only covered regression. After the reviewer flagged `search.py:4139` (the PLS-DA branch in `_run_single_config`) as having the same shared-model pattern, I added a `--task classification` mode to the harness and ran a 3-run mini-matrix on the BoneCollagen classification target (`CollagenCat` — Low/Medium/High).

| Run | git SHA | python / sklearn | PLS-DA cloned? | PLS-DA rows | PLS-DA best_cv_acc | LGBM_cls rows | LGBM_cls best_cv_acc | bug_present |
|---|---|---|---|---|---|---|---|---|
| `.venv311` cls baseline | `19d9346` | 3.11 / 1.5.2 | **NO** (PLS-DA edit reverted via stash) | 973 | 0.9377777777777776 | 9408 | 0.9800000000000001 | FALSE |
| `.venv311` cls post-fix | `19d9346 + WT` | 3.11 / 1.5.2 | YES | 973 | 0.9377777777777776 | 9408 | 0.9800000000000001 | FALSE |
| `.venv312` cls post-fix | `19d9346 + WT` | 3.12 / 1.7.2 | YES | 973 | 0.9377777777777776 | 9408 | 0.9800000000000001 | FALSE |

**All three classification runs are bit-identical to 16 sig figs.** Important conclusion: the **`.venv311` baseline (PLS-DA un-cloned) ran cleanly with no crash and no warnings**. PLS-DA does NOT actually hit the shared-state bug on sklearn 1.5.2. The reviewer's "PLS-DA is latently vulnerable" concern was speculative; the importance-capture pre-fit at `search.py:2191` doesn't fire for PLS-DA because PLS-DA uses its own coefficient-based importance method, so `n_features_in_` never gets pre-set on the PLS-DA estimator and there's no later collision.

The PLS-DA `clone()` is therefore a **defensive no-op**, kept for symmetry with the other three sites. Microbenchmark of clone(PLSRegression) overhead: 41 μs each × 973 PLS-DA configs = 40 ms total per run — negligible. Wall-clock runtime baseline 70 min vs postfix 69 min — within noise.

## Runtime notes (from file mtimes)

- `.venv311` regression baseline: ~3 min (LightGBM crashed early, only PLS ran to completion).
- `.venv312` regression baseline: 62 min (full LightGBM grid + heavy UserWarning overhead).
- `.venv311` regression post-fix: 47 min (full LightGBM grid, zero warning overhead).
- `.venv312` regression post-fix: 79 min (full grid, still has UserWarning overhead from sklearn 1.7.2 DataFrame→ndarray transitions — unrelated to our fix).
- `.venv311` regression rerun (post-PLS-DA-clone): 43 min — within noise of original.
- `.venv312` regression rerun (post-PLS-DA-clone): 55 min — within noise of original.
- `.venv311` classification baseline (PLS-DA un-cloned): 70 min.
- `.venv311` classification post-fix: 69 min.
- `.venv312` classification post-fix: 70 min.

## Raw artifacts

Regression:
- `baseline_venv311.json` + `baseline_venv311.stderr` (40 bytes)
- `baseline_venv312.json` + `baseline_venv312.stderr.gz` (11.5 MB decompressed, 45 KB compressed)
- `postfix_venv311.json` + `postfix_venv311.stderr` (40 bytes)
- `postfix_venv312.json` + `postfix_venv312.stderr.gz` (11.5 MB decompressed, 45 KB compressed — byte-identical to baseline_venv312.stderr.gz contents)
- `postfix_venv311_rerun.json` + `postfix_venv311_rerun.stderr` (post-PLS-DA-clone parity — zero diff vs original postfix)
- `postfix_venv312_rerun.json` + `postfix_venv312_rerun.stderr.gz` (post-PLS-DA-clone parity — zero diff vs original postfix)

Classification (added after reviewer flagged PLS-DA):
- `baseline_cls_venv311.json` + `baseline_cls_venv311.stderr` (PLS-DA un-cloned — proves PLS-DA doesn't hit the bug)
- `postfix_cls_venv311.json` + `postfix_cls_venv311.stderr`
- `postfix_cls_venv312.json` + `postfix_cls_venv312.stderr.gz` (24 MB decompressed, 96 KB compressed)
