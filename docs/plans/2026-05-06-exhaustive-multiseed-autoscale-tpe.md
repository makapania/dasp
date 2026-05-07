# Plan: Exhaustive multi-seed + autoscale; TPE multi-start + multi-seed; remove GA

**Date:** 2026-05-06
**Status:** REVISED v2 — incorporates Codex review of v1 (3 BLOCKERs + 3 WEAKs)
**Prior work:** Empirical investigation in `tools/exhaustive_seed_compare.py` and `tools/bayesian_topk_stability.py` (committed in `8e03dc0`).

---

## Revisions in v2 (relative to v1, which was filed in `739740e`)

| # | Codex finding | Severity | Resolution in v2 |
|---|---|---|---|
| 1 | Fixed K=45 won't generalize across datasets | BLOCKER | Adaptive K with stability-halting loop (Phase 2) |
| 2 | Chromosome shape change breaks 2-gene callers and saved CSVs | BLOCKER | Backward-compat decode in `chromosome_to_transform` (Phase 3) |
| 3 | Phase 2 ordering self-contradicts (rescore-then-diversity vs diversity-then-rescore) | BLOCKER | Explicit pipeline: single-seed rank → top-K (adaptive) → multi-seed → diversity → top-N (Phase 2) |
| 4 | TPE Option B alone doesn't fix early random-startup divergence | WEAK | Multi-start TPE + union + multi-seed rescore replaces Option B (Phase 4) |
| 5 | Phase 2 tie-breaking unspecified; `evaluate_fitness_robust` is shared | WEAK | Explicit mean → std → stable-key tie-break; phase-2 consumes mean from tuple, not function-level change (Phase 2) |
| 6 | "Keep Basic" rationale unverified at merger boundary | WEAK | Clarified — no merger added in this plan; Basic stands alone (out of scope section) |

---

## Summary

Four coordinated changes to dasp's preprocessing-discovery infrastructure:

1. **Remove GA mode** from the GA/Exhaustive section.
2. **Add multi-seed phase-2 rescore to exhaustive** with **adaptive pool size** that grows until top-N identity stabilizes.
3. **Add an autoscale dimension to exhaustive** with a GUI checkbox (default ON), with **backward-compatible chromosome decoding** so 2-gene callers and saved CSVs keep working.
4. **Add multi-start + multi-seed mechanism to TPE preprocessing discovery** — run M independent TPE studies with different seeds, union their top candidates, then multi-seed rescore the union. Per-trial multi-seed CV is a complementary option but not the primary fix.

**Out of scope** (deferred):
- `run_unified_bayesian` (the main Bayesian search). Same TPE-drift problem; same fix conceptually applies. Separate ticket.
- "Basic Preprocessing Discovery" stays as-is, **completely separate from the new phase-2 rescore path**. Its LightGBM-on-full-wavelengths proxy + diversity selection are useful for downstream-varsel scenarios. **No multi-path merger is added in this plan** — Basic does not feed into the phase-2 rescore on exhaustive/TPE outputs, and exhaustive/TPE outputs do not feed into Basic. They remain alternative top-level discovery paths chosen by GUI checkbox.
- The dead `smart_selected_wavelengths` metadata column — separate methodology question.

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
- 3-seed vs 5-seed top-10: Jaccard **0.18** (only 3/10 match). 3-seed pool needed = 51, 5-seed = 41.
- Root cause: discrete-tied accuracy values create artificial tie clusters at single seed.

**Conclusion**: 5 seeds is the floor (3 is not a cheap substitute). Pool floor is dataset-specific (41 here, may be higher elsewhere) — hence the v2 shift to **adaptive K** instead of fixed K=45.

From `tools/bayesian_topk_stability.py`:
- Pairwise Jaccard ≈ 0 across all top-K (5/7/10) for both tasks across 3 seeded Bayesian runs.
- Of ~99 trials per run, only 5 (regression) or 3 (classification) discrete config-keys are evaluated by ALL 3 seeded runs.
- TPE drifts to different regions per seed — drift starts at the random-startup phase (n_startup_trials=20) where each seed picks different initial categorical configs, and the KDEs diverge from there.

---

## Phase 1 — Remove GA mode

### Code changes

`src/spectral_predict/ga_preprocessing.py`:
- Delete the `if method == 'ga'` branch in `optimize_preprocessing` (function at `:1390+`, GA branch around `:1500+`).
- Delete GA-specific helpers: GA mutation/crossover/selection operators in the genetic-operators block (`:725+`), the population/generation loop in `optimize_preprocessing` for `method='ga'`.
- **Keep** `evaluate_fitness_robust` (`:647`) — Phase 2 reuses it (consuming `mean_fitness` from the returned tuple).
- **Keep** `random_chromosome` (`:103`), `get_seed_chromosomes` (`:111`) for now — these are used by `smart_exhaustive_search` (`:1092+`) which is not in the GUI dropdown but may have programmatic callers. Phase 1 does not touch smart.
- **Keep** `N_GENES=2` constant (`:99`). The 2-gene chromosome shape is the legacy interface; Phase 3 will introduce 3-gene as a superset, with backward-compat decoding.

`src/spectral_predict/search.py`:
- `:2052-2068`: drop `population_size` and `n_generations` arguments to `optimize_preprocessing` (they were only used by GA path). Drop the `n_jobs=-1 if ga_preprocess_method == "exhaustive" else 1` conditional — always parallel.
- Drop the `ga_preprocess_method` parameter and any of its plumbing in the function signature.

`spectral_predict_gui_optimized.py`:
- `:13755-13800`: rename "GA and Exhaustive Preprocessing" section to "Exhaustive Preprocessing Discovery". Replace the `["exhaustive", "ga"]` ttk.Combobox dropdown with a single Enable checkbox.
- `:3278`: drop `self.ga_preprocess_method` StringVar.

### Tests

- Remove `tests/` GA-specific tests (grep for `method='ga'` or `optimize_preprocessing.*ga`).
- Verify exhaustive tests still pass.

### Risk

Low. GA was never the recommended path. **Saved-config artifacts** (result CSVs) that store `ga_genes` arrays will remain decodable — Phase 3's backward-compat decode (`chromosome_to_transform` accepts `len(genes) == 2`) preserves rebuild compatibility.

### Estimated LOC

- Deletions: ~150-250 (GA evolution machinery)
- Additions: ~10 (GUI checkbox replacement)

---

## Phase 2 — Multi-seed phase-2 rescore on exhaustive (with adaptive K)

### Pipeline (explicit, post-Codex revision)

```
1. exhaustive_search runs single-seed evaluate_fitness over all 238 (or 476 with autoscale)
   cells with n_jobs=-1 parallelism. → all_fitness array.

2. ADAPTIVE K LOOP:
   K = 3 * top_n         # initial pool (e.g., 15 for top_n=5)
   prev_topN = None
   while True:
       candidates = top-K configs by single-seed score (rank=min)
       rescored = phase2_multiseed_rescore(candidates, n_seeds=5)  # below
       current_topN = diversity_select(rescored, n=top_n, key=(preproc, autoscale))
       if prev_topN == current_topN:
           break  # halt: top-N stable between expansions
       if K >= min(8 * top_n, len(all_fitness)):
           break  # halt: pool reached cap
       prev_topN = current_topN
       K = next_step(K)   # K progression: 3*top_n → 5*top_n → 8*top_n

3. Return current_topN.
```

### `phase2_multiseed_rescore` details

For each candidate config in the pool:
- Call `evaluate_fitness_robust(genes, X, y, ..., n_seeds=5)` — returns `(robust_fitness, mean_fitness, std_fitness)`.
- **Consume `mean_fitness` only**. The `0.1 * std` variance penalty inside `robust_fitness` is dropped (empirical Q1 finding: Jaccard(mean, robust) ≥ 0.67 on top-10).
- **Tie-break**: rank by `mean_fitness` desc, then by `std_fitness` asc (lower variance wins on ties), then by stable config key `(preproc_idx, window_idx, autoscale)` to ensure deterministic output.

### Why adaptive K instead of fixed K=45 (Codex BLOCKER #1)

Empirical pool-size floors are dataset-specific — 41 on BoneCollagen classification, 22-24 on BoneCollagen regression with 3 seeds, unknown on other datasets. A fixed K is brittle. The adaptive loop:
- Starts cheap (3 × top_n) for fast convergence on stable rankings.
- Expands only when top-N changes between K-iterations — i.e., the pool was undersized.
- Caps at `8 × top_n` (or search-space size) to bound worst-case cost.
- Halts on stability — exact identity match of the rescored top-N set.

**Cost estimate:**
- Stable rankings (most regression cases): K=15 only, ~75 multi-seed evaluations. Phase-2 wall time ≈ 75 / 238 ≈ 32% of phase-1.
- Unstable rankings (PLS-DA on small n): K grows to 25 then 40 before halting. ~200 multi-seed evaluations. Phase-2 wall time ≈ 200 / 238 ≈ 84% of phase-1.
- Worst case (search space cap reached): full multi-seed on 8 × top_n. For top_n=5, ~200 evaluations. Same upper bound as fixed K=45 in v1.

### `evaluate_fitness_robust` is NOT modified globally (Codex WEAK #5)

- GA / smart paths (e.g., `ga_preprocessing.py:1268, 1540`) still consume `robust_fitness` (= mean - 0.1*std) — leave them alone.
- Phase-2 rescore consumes `mean_fitness` directly from the tuple return. No function signature change to `evaluate_fitness_robust`.

### Code changes

`src/spectral_predict/ga_preprocessing.py`:
- Add `phase2_adaptive_rescore(all_genes, all_fitness, X, y, top_n, n_seeds, max_K_multiplier, ...)` function implementing the loop above.
- Modify `exhaustive_search` to call the adaptive rescore after the existing single-seed enumeration, controlled by a new `phase2_n_seeds` parameter (default 5; 0 disables = current behavior).
- Diversity selection inside the loop uses existing `select_diverse_exhaustive_configs`, but with the diversity key updated to include autoscale (Phase 3 dependency — see Phase 3).

`src/spectral_predict/search.py`:
- Pass through new `phase2_n_seeds` parameter from GUI to `optimize_preprocessing`.

`spectral_predict_gui_optimized.py`:
- Add checkbox "Robust ranking (5-seed phase-2 rescore)" in the Exhaustive Preprocessing section. Default ON.
- Tooltip: "Re-evaluates top configs with 5 random_state values to detect lottery winners. Recommended for classification on small datasets. Cost: 1.3-1.8x current."

### Tests

- `tests/test_exhaustive_phase2.py`:
  - `test_phase2_disabled_matches_legacy`: `phase2_n_seeds=0` produces identical output to current single-seed exhaustive. Regression pin.
  - `test_phase2_reorders_on_unstable_data`: synthetic dataset with high CV-fold variability — assert phase-2 changes the top-N from single-seed top-N.
  - `test_adaptive_k_halts_on_stability`: synthetic dataset where rankings are stable — assert K stops at `3 * top_n`.
  - `test_adaptive_k_expands_when_unstable`: synthetic dataset where they're not stable — assert K reaches `8 * top_n` cap.
  - `test_tie_breaking_deterministic`: configs with identical mean — assert std is the tie-breaker, then config key.

### Estimated LOC

- ga_preprocessing.py: ~120-150 additions (adaptive loop + diversity-after-rescore wiring).
- search.py: ~15 additions (parameter passthrough).
- GUI: ~30 additions (checkbox + tooltip).

---

## Phase 3 — Add autoscale dimension to exhaustive (backward-compat)

### Concept

Mirror what TPE already does (`tpe_preprocessing_discovery.py:127-129`): add autoscale (StandardScaler) as a final step after SNV/derivative core preprocessing. The chromosome grows from 2 genes to 3.

### Backward-compatible chromosome decoding (Codex BLOCKER #2)

`chromosome_to_transform` (`:146-200`) must accept BOTH 2-gene and 3-gene chromosomes:
```python
def chromosome_to_transform(genes):
    preproc_idx = genes[0]
    window_idx = genes[1]
    autoscale = bool(genes[2]) if len(genes) >= 3 else False  # backward-compat
    ...
```

Surviving 2-gene callers (Phase 1 keeps these alive):
- `random_chromosome` (`:103`), `get_seed_chromosomes` (`:111`) — used by `smart_exhaustive_search`.
- `smart_exhaustive_search` (`:1092+`) — its `stage1_genes` arrays at `:1194+` and `:1526+` construct 2-gene chromosomes.
- `search.py:792-800` — rebuilds saved `ga_genes` from result-CSV rows.
- Any saved `.dasp` model files containing `ga_genes` arrays.

These all continue to work because the 2-gene chromosomes decode as `autoscale=False`, matching their existing behavior exactly.

### Code changes

`src/spectral_predict/ga_preprocessing.py`:
- `chromosome_to_transform` (`:146`): backward-compat decode (above). When `autoscale=True`, append `StandardScaler().fit_transform(X)` to the returned transform closure.
- `get_config_description` (`:225`): backward-compat decode. Append `+autoscale` to description when autoscale=1.
- `exhaustive_search` (`:874+`): expand `all_genes` triple-loop: `for p in ... for w in ... for a in [0, 1]`. Generates 3-gene chromosomes for the new path.
- `select_diverse_exhaustive_configs` (`:787`): key diversity off `(preproc_type, autoscale)` tuple. Diversity-key extraction uses backward-compat decode.

`src/spectral_predict/search.py`:
- `:2120-2151`: in the GA→preprocess_configs conversion loop, read `autoscale = bool(genes[2]) if len(genes) >= 3 else False` and emit `"autoscale": autoscale` on the config dict. Rebuild path already consumes this field at `search.py:2628, 4483`.

`spectral_predict_gui_optimized.py`:
- Add checkbox "Test with/without autoscale" in the Exhaustive Preprocessing section. Default ON.
- When OFF: pass autoscale_choices=[False] to exhaustive_search, restricting enumeration to autoscale=False (preserves legacy 238-cell behavior even with the new 3-gene chromosome shape).

### Tree-model duplication note (deferred polish)

For tree models (LightGBM/XGBoost/CatBoost/RandomForest), autoscale on/off is mathematically equivalent (per PROJECT_STATUS analysis 2026-05-07). When ALL enabled models are tree-based, the autoscale dimension wastes 50% compute. Optional optimization: `if all(m in TREE_MODELS for m in models_to_test): autoscale_choices = [False]` — saves 50% on the autoscale-relevant runs. Defer as polish unless empirical wall-time on tree-heavy runs is a problem.

### Tests

- `tests/test_exhaustive_autoscale.py`:
  - `test_2gene_chromosome_decodes_no_autoscale`: pass `genes = np.array([3, 5])` to `chromosome_to_transform`, assert no StandardScaler in resulting transform.
  - `test_3gene_chromosome_with_autoscale_true`: pass `genes = np.array([3, 5, 1])`, assert StandardScaler appears.
  - `test_3gene_chromosome_with_autoscale_false`: pass `genes = np.array([3, 5, 0])`, assert no StandardScaler.
  - `test_saved_csv_rebuild_compatible`: simulate loading an old result CSV with 2-gene `ga_genes` — assert `chromosome_to_transform` succeeds and pipeline reconstructs.
  - `test_exhaustive_with_autoscale_emits_both`: exhaustive run with autoscale dimension on emits both `autoscale=True` and `autoscale=False` configs in top-N.

### Estimated cost

- 2× current wall time when ON (476 cells vs 238).
- Combined with Phase 2 multi-seed: 2 × current Phase-1 + adaptive multi-seed rescore. Worst case ~3-3.5× current.

### Estimated LOC

- ga_preprocessing.py: ~50-80 (chromosome backward-compat + transform + diversity key).
- search.py: ~10 (config dict population with backward-compat decode).
- GUI: ~30 (checkbox + tooltip).

---

## Phase 4 — Multi-start + multi-seed for TPE preprocessing discovery

### Why multi-start instead of just multi-seed-per-trial (Codex WEAK #4)

Empirical finding from `tools/bayesian_topk_stability.py`: TPE seeds produce essentially disjoint top-K sets (Jaccard ≈ 0). Codex's analysis: multi-seed CV per trial denoises objective scores but doesn't fix early-trial random-startup divergence — different `random_state` values see different categorical configs in the n_startup_trials=20 random phase, and the KDEs diverge from there onward.

**The fix that addresses both failure modes:**

1. **Multi-start TPE**: run M independent TPE studies with different `random_state` seeds. Each does its own random-startup + TPE-guided sampling. Different seeds explore different regions.
2. **Union top-K' candidates** across the M studies. M × K' configs (with possible overlap).
3. **Phase-2 rescore the union** with multi-seed CV (n_seeds=5) — same mechanism as exhaustive's Phase 2.
4. **Apply diversity selection + return top-N**.

This preserves the per-study TPE adaptive-sampling advantage (don't blindly enumerate) while adding cross-study coverage (multi-start) and per-trial denoising (multi-seed rescore at phase 2).

**Per-trial multi-seed CV** (the original "Option B") is a complementary but lower-priority improvement. It would tighten each study's KDE convergence but doesn't address the cross-study divergence — which is the dominant failure mode per the empirical data. Out of scope for v1 implementation; revisit if multi-start alone proves insufficient.

### Pipeline

```
1. M = 3 (default, exposed as parameter)
2. K' = max(top_n, 7)  # per-study candidate pool, default 7
3. Run M independent run_tpe_preprocessing_discovery calls with seeds [42, 0, 7]
   (current default n_trials=75 each, single-seed CV per trial — same as today).
4. Union top-K' candidates across M studies (deduplicate by discrete config key).
5. Phase-2 rescore the union with n_seeds=5 (re-using exhaustive's phase2 mechanism
   adapted for the TPE config dict shape).
6. Re-rank union by mean fitness, tie-break by std then stable key.
7. Apply select_diverse_configs (existing, in tpe_preprocessing_discovery.py:467).
8. Return top-N.
```

### Cost

- M=3 sequential TPE studies: ~3 × current TPE wall time. On BoneCollagen: 3 × 30s ≈ 90s (regression), 3 × 10s ≈ 30s (classification).
- Phase-2 rescore on |union| candidates (≤ M × K' = 21 if no overlap; typically 12-18): ~|union| × 5 / 75 × current TPE phase-1. Negligible.
- Total wall-time multiplier: **~3-3.5× current TPE**.
- In absolute terms on small-n: 30s → 100-110s.

### Code changes

`src/spectral_predict/tpe_preprocessing_discovery.py`:
- Add `run_tpe_multistart_preprocessing_discovery(X, y, ..., n_starts=3, per_start_pool=7, n_seeds=5)`:
  - Loop M times, each call to existing `run_tpe_preprocessing_discovery` with different `random_state`.
  - Collect `top_configs[:per_start_pool]` from each.
  - Union by discrete config key (preproc, window, autoscale, baseline_method, smoothing).
  - Phase-2 rescore (call into ga_preprocessing.phase2_multiseed_rescore or a duplicated helper — TBD during implementation, prefer shared helper).
  - Re-rank, apply existing `select_diverse_configs`, return top-N.

`src/spectral_predict/search.py`:
- `:1854-1880`: gate on a new `tpe_multistart` flag. When ON, call the multistart wrapper; when OFF, fall through to current single-start behavior.

`spectral_predict_gui_optimized.py`:
- Add checkbox "Multi-start TPE (3 seeds + rescore)" in the TPE section. Default ON for classification, OFF for regression (or just "OFF default with tooltip" — see Open Questions).
- Tooltip: "Runs TPE 3 times with different seeds and rescores the union with 5-seed CV. Strongly recommended for classification on small datasets. Cost: ~3x current."

### Tests

- `tests/test_tpe_multistart.py`:
  - `test_multistart_disabled_matches_legacy`: `tpe_multistart=False` produces identical output to current single-start. Regression pin.
  - `test_multistart_enabled_increases_top_K_stability`: run multistart twice with different overall seeds, assert top-K Jaccard is > 0.5 (vs ~0 currently).
  - `test_union_dedup`: assert configs that appear in multiple per-start pools are deduplicated by discrete key.

### Estimated LOC

- tpe_preprocessing_discovery.py: ~80-120 (new wrapper + union + rescore plumbing).
- search.py: ~15 (parameter passthrough + flag-gated branch).
- GUI: ~30 (checkbox + tooltip).

---

## Implementation order

1. ✅ Commit empirical investigation tools (done in `8e03dc0`).
2. ✅ File this revised plan (this document, replacing v1 in `739740e`).
3. **Phase 1**: GA removal (clean baseline, smallest risk; preserves 2-gene callers).
4. **Phase 3**: Exhaustive autoscale with backward-compat decode (3-gene chromosome). Ship this BEFORE Phase 2 because Phase 2's diversity-key uses `(preproc, autoscale)` and benefits from the autoscale field already being present.
5. **Phase 2**: Exhaustive multi-seed adaptive rescore.
6. **Phase 4**: TPE multi-start + rescore.

Each phase ships as its own commit/PR with tests + a smoke check via GUI on BoneCollagen, both regression and classification.

## Risks / open questions

1. **Adaptive K halt criterion**: exact set-equality of top-N. Could be too strict — one config swap between K-iterations forces another expansion. Alternative: Jaccard > 0.9 threshold. Choose during implementation; default to exact match and revisit if stability tests show too many expansions.

2. **Combined Phase 2 + 3 cost**: ~3× current. For PLS-heavy: 5-30s. For MLP/SVM-heavy: 10-30 min. Acceptable given the user's stated workflow (top-7 to top-10 selection, where infiltration matters)?

3. **Phase 4 M=3 vs M=5**: 3 starts may be insufficient for very noisy datasets. Ship with M=3, expose as advanced parameter, revisit if empirical top-K stability across overall seeds is still poor.

4. **TPE multistart default ON/OFF**: classification users have to know to turn it on. Auto-default-ON-for-classification adds magic but matches the empirical asymmetry. Defer to user preference; ship default OFF with strong tooltip recommendation, document the asymmetry in user docs.

5. **Bayesian (`run_unified_bayesian`) parallel ticket**: same TPE drift problem; same multi-start fix conceptually applies. Should be a follow-up after Phase 4 ships, since the architectural pattern will be proven.

6. **Saved CSV ga_genes column**: existing CSVs have 2-gene arrays. Phase 3's backward-compat decode handles rebuild. New CSVs (post-Phase 3) will have 3-gene arrays. Document this in CHANGELOG / docstring on the column. Consider migration helper.

## Testing plan

- Per-phase pytest as listed in each phase's "Tests" section.
- After each phase merge: smoke test by running through GUI on BoneCollagen, both regression and classification, verifying:
  - Top-K configs in results CSV are stable run-to-run (within reasonable variance — for Phase 2 onward, top-K should be more stable than legacy single-seed).
  - Wall time matches estimates within ~30%.
  - GUI controls behave correctly.
- After Phase 3: explicit backward-compat test — open an old `.dasp` file (or simulated equivalent) with 2-gene `ga_genes` and verify model rebuild + prediction works identically.
- Final cumulative regression: full `pytest tests/` after all 4 phases land.
- E2E smoke check follows the project's `feedback_e2e_smoke_after_refactor.md` rule — actual classification + regression through `run_search` + `run_unified_bayesian` after each merge.

## Cross-references

- Empirical investigation tools: `tools/exhaustive_seed_compare.py`, `tools/bayesian_topk_stability.py` (committed `8e03dc0`).
- Codex review of v1: see chat transcript leading to this v2 revision. Three BLOCKERs all addressed; three WEAKs all addressed.
- Project-policy memos consulted: `feedback_chemometrics_relevance_per_ticket.md` (this is methodology change, user has approved); `feedback_e2e_smoke_after_refactor.md` (testing plan complies); `feedback_review_method_signal.md` (Codex correctly used for cross-file dispatcher review).
