# Plan: Exhaustive multi-seed + autoscale; TPE multi-seed; remove GA

**Date:** 2026-05-06
**Status:** DRAFT — awaiting Codex review
**Prior work:** Empirical investigation in `tools/exhaustive_seed_compare.py` and `tools/bayesian_topk_stability.py` (committed in `8e03dc0`).

## Summary

Three coordinated changes to dasp's preprocessing-discovery infrastructure:

1. **Remove GA mode** from the GA/Exhaustive section. It's vestigial — same 238-cell space as exhaustive, slower (no parallelism by default), no offsetting feature. The dropdown becomes a single-mode toggle.
2. **Add multi-seed phase-2 rescore to exhaustive** (n_seeds=5 over single-seed top-K=45) to fix top-N infiltration on small-n / classification tasks. Empirical data shows single-seed top-7 contains 6/7 mean-rank-deep fakes on PLS-DA classification; rescore is the cheapest fix.
3. **Add an autoscale dimension to exhaustive** with a GUI checkbox (default ON). Mirrors what TPE already does. Doubles search space to 14×17×2 = 476 cells.
4. **Add multi-seed mechanism to TPE preprocessing discovery** to fix the same fold-randomness issue, with the additional benefit of denoising TPE's adaptive sampling.

**Out of scope** (deferred):
- `run_unified_bayesian` (the main Bayesian search) shows the same TPE drift problem in `tools/bayesian_topk_stability.py` — pairwise Jaccard ≈ 0 across top-K of seeded runs. Same fix conceptually applies. Not addressed here; would be a separate ticket.
- "Basic Preprocessing Discovery" stays as-is. Its LightGBM-on-full-wavelengths proxy + diversity selection are useful when downstream variable-selection methods (UVE/SPA/iPLS/CARS/GA-PLS) will pick a different feature subset from the one used to score preprocessing — preserving preprocessing-type diversity in that case is more useful than picking the single best (preproc, window) per actual model.
- The dead `smart_selected_wavelengths` metadata column stays metadata (separate methodology question).

## Empirical evidence

From `tools/exhaustive_seed_compare.py` on BoneCollagen (n=49, 2151 wavelengths):

**Regression (PLS):**
- Single-seed top-7: worst mean-rank = 23, 1/7 fakes (rank > 20).
- Pool size 40 catches all of mean top-15.
- 3-seed vs 5-seed top-10: Jaccard 0.43 (only 6/10 match). 3-seed pool needed = 22-24, 5-seed = 40 — different rankings.

**Classification (PLS-DA):**
- Single-seed top-7: worst mean-rank = **65**, 6/7 fakes.
- Single-seed top-10: 9/10 fakes.
- Pool size 41 catches all of mean top-10.
- 3-seed vs 5-seed top-10: Jaccard **0.18** (only 3/10 match). 3-seed pool needed = 51, 5-seed = 41 — 3-seed actually requires a *bigger* pool.
- Root cause: discrete-tied accuracy values create artificial tie clusters at single seed.

**Conclusion**: 5 seeds is the floor (3 is not a cheap substitute), and pool size **45** has ~10% margin over the empirical 41 floor for both tasks.

From `tools/bayesian_topk_stability.py`:
- Pairwise Jaccard ≈ 0 across all top-K (5/7/10) for both tasks across 3 seeded Bayesian runs.
- Of ~99 trials per run, only 5 (regression) or 3 (classification) discrete config-keys are evaluated by ALL 3 seeded runs.
- Most of run-1's top-K is never sampled in runs 2-3 — TPE drifts to different regions per seed.

## Phase 1 — Remove GA mode

### Code changes

`src/spectral_predict/ga_preprocessing.py`:
- Delete the `if method == 'ga'` branch in `optimize_preprocessing` (function is at `:1390+`, the GA branch starts around `:1500+`).
- Delete the GA-specific helpers: `random_chromosome` (`:103`), `get_seed_chromosomes` (`:111`), GA mutation/crossover/selection operators in the genetic-operators block (`:725+`), the population/generation loop in `optimize_preprocessing` for `method='ga'`.
- Keep `evaluate_fitness_robust` (`:647`) — it'll be reused by Phase 2.
- Keep `smart_exhaustive_search` (`:1092`) for now — it's a separate mode that's not in the GUI dropdown but may have callers.

`src/spectral_predict/search.py`:
- `:2052-2068`: drop `population_size` and `n_generations` arguments to `optimize_preprocessing` (they were only used by GA path). Drop the `n_jobs=-1 if ga_preprocess_method == "exhaustive" else 1` conditional — always parallel.
- Drop the `ga_preprocess_method` parameter and any of its plumbing in the function signature.

`spectral_predict_gui_optimized.py`:
- `:13755-13800`: rename "GA and Exhaustive Preprocessing" section to "Exhaustive Preprocessing Discovery". Replace the `["exhaustive", "ga"]` ttk.Combobox dropdown with a single Enable checkbox.
- `:3278`: drop `self.ga_preprocess_method` StringVar.
- Remove all references to the dropdown control and its callbacks.

### Tests

- Remove `tests/` GA-specific tests (grep for `method='ga'` or `optimize_preprocessing.*ga`).
- Verify exhaustive tests still pass.

### Risk

Low. GA was never the recommended path. If any saved-config files reference `ga_preprocess_method='ga'`, they'd fall back to exhaustive — acceptable because the user explicitly approved removal.

### Estimated LOC

- Deletions: ~200-300 (most of the GA evolution machinery in `ga_preprocessing.py`)
- Additions: ~10 (GUI checkbox replacement)

---

## Phase 2 — Multi-seed phase-2 rescore on exhaustive

### Concept

Two-phase ranking:
1. Phase 1: `exhaustive_search` runs single-seed `evaluate_fitness` over all 238 (or 476 with autoscale, see Phase 3) cells with `n_jobs=-1` parallelism. Same as today.
2. Phase 2: take top-K (default K=45) by single-seed score. For each, run `evaluate_fitness_robust` with `n_seeds=5`. Re-rank by **mean** (not `mean - 0.1*std` — empirical Q1 finding shows variance penalty does nothing).
3. Apply diversity selection to the rescored top-K, return final top-N (default 5 per model).

### Why K=45 specifically

Empirical floor is 41 (worst single-seed rank in mean top-10 on BoneCollagen classification); 45 has ~10% margin. Should be revalidated on a second dataset before final commit (see Open Questions).

### Why drop the variance penalty

`tools/exhaustive_seed_compare.py` showed `jaccard(mean_top_K, robust_top_K)` ≥ 0.67 for K=10 and = 1.0 for top-1. The `0.1 * std` term shifts scores by ≈0.02 RMSE units while the mean-based ranking already reflects the signal. Variance penalty is theoretical noise, not signal.

### Code changes

`src/spectral_predict/ga_preprocessing.py`:
- Add `phase2_rescore` function that takes top-K configs + raw X/y/cv params + n_seeds, runs `evaluate_fitness_robust` for each, returns reordered list.
- Modify `exhaustive_search` to call `phase2_rescore` after the diversity selection step, controlled by a new `phase2_n_seeds` parameter (default 5; 0 disables).
- Diversity selection runs again post-rescore (same `select_diverse_exhaustive_configs`) to ensure the final top-N still has variety after re-ranking.

`src/spectral_predict/search.py`:
- Pass through new `phase2_n_seeds` and `phase2_pool_size` parameters from GUI to `optimize_preprocessing`.

`spectral_predict_gui_optimized.py`:
- Add checkbox "Robust ranking (5-seed phase-2 rescore)" in the Exhaustive Preprocessing section. Default ON.
- Add a tooltip noting expected ~1.5-1.8× wall-time cost.
- (Optional, advanced): expose `phase2_n_seeds` (default 5) and `phase2_pool_size` (default 45) in an advanced collapsible.

### Tests

- `tests/test_exhaustive_phase2.py`:
  - Test that `phase2_rescore` reorders configs when scores differ across seeds.
  - Test that disabling phase 2 (`phase2_n_seeds=0`) gives identical output to current single-seed exhaustive.
  - Behavioral test: on a synthetic noisy classification dataset, phase 2 promotes mean-top configs out of the rank-21+ zone of single-seed.

### Estimated cost

- Phase 2 wall time = K × n_seeds / 238 × phase-1 wall time = 45 × 5 / 238 ≈ 95% extra. Total ~1.95× current.
- Combined with Phase 3 autoscale (Phase 1 doubles to 476 cells): phase-2 wall time stays at K × n_seeds = 225 evaluations regardless of phase-1 size.

### Estimated LOC

- ga_preprocessing.py: ~80-120 additions (new function + integration).
- search.py: ~15 additions (parameter passthrough).
- GUI: ~30 additions (checkbox + tooltip).

---

## Phase 3 — Add autoscale dimension to exhaustive

### Concept

Mirror what TPE already does (`tpe_preprocessing_discovery.py:127-129`): add autoscale (StandardScaler) as a final step after the SNV/derivative core preprocessing. The chromosome grows from 2 genes to 3.

### Code changes

`src/spectral_predict/ga_preprocessing.py`:
- Expand chromosome encoding: `genes = [preproc_idx, window_idx, autoscale]` where `autoscale ∈ {0, 1}`.
- `chromosome_to_transform` (`:146`): when `autoscale=1`, append `StandardScaler().fit_transform(X)` to the returned transform closure.
- `get_config_description` (`:225`): append `+autoscale` to description string when autoscale gene is set.
- `exhaustive_search`: expand `all_genes` triple-loop: `for p in ... for w in ... for a in [0, 1]`.
- `select_diverse_exhaustive_configs` (`:787`): key diversity off `(preproc_type, autoscale)` tuple instead of just `preproc_type` to avoid filling top-N with autoscale-on/off siblings of the same preproc.

`src/spectral_predict/search.py`:
- `:2120-2151`: in the GA→preprocess_configs conversion loop, read `autoscale = bool(genes[2])` and emit `"autoscale": autoscale` on the config dict. Rebuild path already consumes this field — see `search.py:2628, 4483` (Phase 3 just adds a new producer, no consumer changes).

`spectral_predict_gui_optimized.py`:
- Add checkbox "Test with/without autoscale" in the Exhaustive Preprocessing section. Default ON.
- When OFF: pass autoscale=False, restrict enumeration to autoscale=0 only (preserves current behavior).

### Tree-model duplication note

For tree models (LightGBM/XGBoost/CatBoost/RandomForest), autoscale on/off is mathematically equivalent (per PROJECT_STATUS analysis 2026-05-07). When ALL enabled models are tree-based, the autoscale dimension is wasted compute. Optional optimization: `if all(m in TREE_MODELS for m in models_to_test): autoscale_choices = [False]` — saves 50% on the autoscale-relevant runs. ~5 LOC. Defer as polish unless empirical wall-time on tree-heavy runs is a problem.

### Tests

- `tests/test_exhaustive_autoscale.py`:
  - When checkbox is on, exhaustive emits both `autoscale=True` and `autoscale=False` configs in its top-N output.
  - When off, all output configs have `autoscale=False`.
  - End-to-end smoke: exhaustive + autoscale on PLS-DA classification produces meaningfully different top-N than autoscale off.

### Estimated cost

- 2× current wall time when ON (476 cells vs 238).
- Combined with Phase 2 multi-seed: 2 × 1.95 ≈ ~3.9× current. Should still complete in single-digit minutes for typical workflows.

### Estimated LOC

- ga_preprocessing.py: ~50-80 (chromosome expansion + transform + diversity).
- search.py: ~10 (config dict population).
- GUI: ~30 (checkbox + tooltip).

---

## Phase 4 — Add multi-seed mechanism to TPE preprocessing discovery

### The choice between two options

**Option A — Phase-2 rescore on TPE top-K**
- Take TPE's top-K outputs, rescore each with multi-seed.
- ~1.5× cost.
- Doesn't fix TPE drift (the deeper problem from `tools/bayesian_topk_stability.py`); only the per-trial CV noise.

**Option B — Multi-seed CV inside each TPE trial**
- Modify `_quick_evaluate` to optionally average over n_seeds CV runs.
- ~5× cost on TPE wall time (still small in absolute terms — ~30-60s on BoneCollagen at 75 trials).
- Fixes both problems: trial scores denoised AND TPE's adaptive sampling no longer follows lottery winners.

Empirical evidence from `tools/bayesian_topk_stability.py` shows TPE drift is a real failure mode (Jaccard ≈ 0 across seeded runs). Option B addresses it; Option A doesn't.

**Recommendation: Option B**. The 5× cost on TPE is small in absolute wall-time terms compared to Bayesian search overall, and the methodological correctness is meaningful.

### Code changes (Option B)

`src/spectral_predict/tpe_preprocessing_discovery.py`:
- Add `n_seeds` parameter to `_quick_evaluate` (default 1 for backwards compat).
- When `n_seeds > 1`, run the inner CV loop n_seeds times with seeds `[42, 0, 7, 100, 31][:n_seeds]`, return mean.
- Add `n_seeds` parameter to `run_tpe_preprocessing_discovery` signature; thread it through to `_objective` → `_quick_evaluate`.

`src/spectral_predict/search.py`:
- `:1854-1880`: pass new `tpe_n_seeds` parameter through to `run_tpe_preprocessing_discovery`.

`spectral_predict_gui_optimized.py`:
- Add checkbox "Robust ranking (5-seed CV per trial)" in the TPE section. Default OFF (since cost is real and only matters for small-n / classification).
- Tooltip noting it's recommended for small-n classification (cite the empirical evidence).

### Tests

- `tests/test_tpe_multiseed.py`:
  - When `n_seeds=1`, output is identical to current behavior (regression pin).
  - When `n_seeds=5`, top-K stability across seeded TPE runs is meaningfully better than `n_seeds=1`.

### Estimated cost

- ~5× TPE wall time when on (75 trials × 5-seed × 5-fold CV ≈ 1875 fits vs current 375).
- In absolute terms on small-n: 30s → 150s.

### Estimated LOC

- tpe_preprocessing_discovery.py: ~30-50 (parameter threading + multi-seed loop).
- search.py: ~10 (parameter passthrough).
- GUI: ~30 (checkbox + tooltip).

---

## Implementation order

1. ✅ Commit empirical investigation tools (done in `8e03dc0`).
2. **Phase 1**: GA removal (clean baseline, smallest risk).
3. **Phase 2**: Exhaustive multi-seed rescore (orthogonal to Phase 3; can ship independently).
4. **Phase 3**: Exhaustive autoscale (depends on chromosome shape changes; safer to ship after Phase 1 GA removal so we don't have to maintain GA-compat shims for the 3-gene chromosome).
5. **Phase 4**: TPE multi-seed (independent of all above).

Each phase ships as its own commit/PR with tests + a smoke check via GUI on BoneCollagen, both regression and classification.

## Risks / open questions

1. **K=45 generalization**: validated on BoneCollagen only. Should run `tools/exhaustive_seed_compare.py` on 1-2 more datasets (synthetic NIR or another labeled dataset) before final commit. If the floor is consistently ≤41 across datasets, K=45 is fine. If it's higher (60+) on some datasets, may need K=55 or expose K as a parameter.

2. **Combined Phase 2 + 3 cost**: ~3.9× current. For PLS-heavy workflows that's still 5-30s. For MLP/SVM-heavy: 12-40 min. Acceptable?

3. **GA removal artifacts**: any saved-config files, regression-test fixtures, or documentation that references `method='ga'` need migration paths. Will scan during Phase 1 implementation.

4. **TPE multi-seed default-OFF on classification**: classification users have to know to turn it on. Possible alternative: default ON for classification, OFF for regression (auto-detect from `task_type`). More magic, but matches the empirical asymmetry.

5. **Bayesian (`run_unified_bayesian`) parallel ticket**: same TPE drift problem, same Option B fix. Should this be batched with Phase 4 or filed as a separate follow-up? Architecturally similar but `unified_bayesian.py` is much larger and has more dimensions in scope.

6. **Phase 4 Option A vs B**: Codex review may have a different recommendation. Will defer final choice to that review.

## Testing plan

- Per-phase pytest as listed in each phase's "Tests" section.
- After each phase merge: smoke test by running through GUI on BoneCollagen, both regression and classification, verifying:
  - Top-K configs in results CSV are stable run-to-run (within reasonable variance).
  - Wall time matches estimates within ~30%.
  - GUI controls behave correctly (toggle on/off produces expected behavior).
- Final cumulative regression: full `pytest tests/` after all 4 phases land.
- E2E smoke check follows the project's `feedback_e2e_smoke_after_refactor.md` rule — actual classification + regression through `run_search` + `run_unified_bayesian` after each merge.

## Open question for Codex

Is Option B (multi-seed inside TPE trials) the right call, or is there a TPE-architectural reason Option A would be preferred? Specifically: does Optuna's TPESampler with `multivariate=True` + multi-seed scoring per trial actually improve KDE convergence, or is that wishful thinking — and if so, what would the right fix be (more startup trials, different sampler, etc.)?
