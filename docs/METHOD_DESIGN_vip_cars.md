# Method-Design Note: Alternative Importance Signals for CARS (incl. VIP-CARS)

**Date:** 2026-07-07
**Status:** Design note / not yet implemented
**Scope:** `src/spectral_predict/variable_selection.py` — `cars_selection` and the multi-target
`_cars_multi_cell` (T-17 worktree, commit `bc750d9`)
**Context:** Arose from reviewing the recovered overnight T-17 multi-target UVE/CARS work.

---

## 1. The core insight — CARS is two separable ideas

CARS (Competitive Adaptive Reweighted Sampling, Li et al. 2009) factors cleanly into:

1. **A per-variable importance/ranking signal** — canonically the absolute PLS regression
   coefficient `|b_i|`.
2. **An adaptive reweighted sampling (ARS) loop** — an exponentially decaying retention
   function culls variables over ~50 Monte Carlo iterations; each candidate subset is scored by
   RMSECV; the subset from the lowest-RMSECV iteration wins.

**The ARS loop is model-agnostic — it only needs *some* per-variable importance number.** The PLS
coefficient is just *one choice* of signal. This is exactly the door dasp already walked through
with **CARS-Tree** (tree models have no PLS coefficients, so it swaps in LightGBM split/gain
importances).

## 2. How the loop actually consumes the signal (grounded in code)

The critical structural fact — **the importance signal does not perform the elimination:**

- **Elimination is the EDF, not the signal.** `r = 0.8*exp(-2*iteration/n_iterations)`
  (`variable_selection.py:1386`) deterministically shrinks the retained count each iteration,
  regardless of the signal.
- **The signal only sets sampling *preference*.** `probabilities = weights/weights.sum()` then
  `rng.choice(..., p=probabilities)` (`:1394-1400`) — the importance is a weighted-sampling
  probability, not a threshold. Non-selected variables retain their stale weight, so the
  competition is soft and reversible.
- **Write-back:** PLS path `weights[selected] = np.abs(coef)` (`:1531`); tree path normalizes
  split+gain and floors at `1e-6` (`:1504`) precisely because sparse tree importances (many exact
  zeros) would otherwise make variables unrecoverable in `rng.choice`.
- **Winner = `argmin(rmsecv_history)`** (`:1556`). RMSECV is a *separate* signal from the
  importance, currently from the *same* model that produced the importance.

Three constraints this imposes on any alternative signal:

1. **Must be non-negative** (it becomes a probability). Signed signals (permutation importance, raw
   correlation, raw linear coefficients) need `|·|`/clipping and must not collapse to all-zeros.
2. **Re-estimated on the shrinking subset every iteration** — 50 iters x CV folds x grid cells, so
   per-iteration cost compounds hard.
3. **Coupling risk** — the winner is chosen by RMSECV (still PLS/LGBM). Swap only the *importance*
   and you select-by-X but score-by-PLS: a chemometrics-defensibility mismatch, not just
   inefficiency.

There is an existing, unused-by-CARS dispatcher `get_feature_importances()` (`models.py:1790`) that
already maps PLS->VIP (`compute_vip`, `models.py:1717`), Ridge/Lasso/ElasticNet->`|coef|`,
trees->`feature_importances_`. CARS reimplements its signal inline instead of calling it — relevant
if a VIP path is added (route through the dispatcher).

## 3. Current state of CARS signals in the codebase

| Path | Signal | Status |
|------|--------|--------|
| PLS1 (single-target regression) | `\|PLS coef\|` | **Canonical CARS (Li 2009)** — do not change |
| PLS2 (multi-target regression) | `\|PLS-2 coef\|`, Y column-scaled, aggregated across targets (`cars_scaled_coef` l2 rule) | dasp extension (T-17, unreleased WIP) |
| CARS-Tree (any tree model) | LightGBM split/gain hybrid importance | dasp invention (already shipped) |
| Multi-target classification | — | Rejected (no coherent joint PLS-DA criterion) |

**T-17 note:** the recovered multi-target work extends **both** the PLS-coef path *and* the
tree-importance path to multi-Y (`_cars_multi_cell` fits `MultiOutputRegressor` of per-target
LightGBM and aggregates via the same rule). CARS-Tree is a first-class multi-Y citizen, not
PLS-coef-only.

## 4. Alternative-signal analysis (ranked)

### Worth prototyping

1. **VIP as the PLS-path signal — the clear win.** Replace `np.abs(coef)` (`:1531`) with
   `compute_vip()` on the **already-fitted** PLS (`:1526`). Zero extra model fits, non-negative by
   construction, canonical chemometrics, and theoretically better on collinear NIR: VIP's
   SSY-weighting spreads credit across a correlated absorption band instead of letting raw `|coef|`
   arbitrarily spike one channel and starve its neighbors. **The only candidate immune to the
   RMSECV coupling problem — VIP *is* the PLS fit.** Multi-Y aggregation rule already exists.
2. **Ridge `|coef|` — conditional.** Only for *model-consistency* when the downstream model is Ridge
   (mirrors the CARS-Tree "match the importance to the model family" logic). Cheap, dense, stable;
   modest expected gain since it's a shrunk PLS-coef. Lower priority.

### Traps — do not build

- **Permutation importance & SHAP** — 10-100x the per-iteration cost inside a compute-sensitive
  loop, *and* they systematically under-credit collinear NIR bands (permuting one channel while its
  neighbors carry the same info scores it ~0 — exactly wrong for spectra). TreeSHAP is redundant
  with the split+gain importance CARS-Tree already has for free.
- **Lasso/ElasticNet `|coef|`** — embedded L1 sparsity *inside* an elimination schedule
  double-selects and re-triggers the all-zero-subset failure the `1e-6` floor was added to mask.
  Mechanically insertable, methodologically incoherent.
- **Mutual information / univariate correlation** — univariate and band-blind; they fight the loop's
  core job of finding a *jointly*-compact subset, and aren't recognized chemometrics selection
  signals. (Correlation has a minor secondary use as a non-uniform *initializer* for `weights`
  `:1373`, not as the iteration signal.)

## 5. Novelty finding — is VIP-CARS already published?

Literature search (2026-07-07):

- **Canonical CARS = `|regression coefficients|`.** Confirmed across all sources ("variable
  significance assessed using absolute values of regression coefficients"). The PLS1 path as-shipped
  IS canonical CARS.
- **VIP-CARS (VIP swapped *inside* the reweighting loop) does NOT appear to be an established, named
  published method.** In the literature VIP and CARS relate in three ways, none of which is this:
  (a) two-stage hybrid (VIP>1 pre-filter, then CARS/SPA); (b) alternatives compared side-by-side;
  (c) VIP coupled with a *different* wrapper (VIP-GA). No paper found driving the CARS ARS loop with
  VIP.
- **But the *approach* of swapping the CARS internal signal has clear precedent: SCARS — Stability
  CARS** (Zheng et al. 2012, *Chemom. Intell. Lab. Syst.*) replaces raw `|coef|` with a stability
  measure (coefficient mean / std). That is exactly the "keep the ARS loop, swap the signal" move,
  and it was published as a named method. So VIP-CARS would belong to a legitimate, citable-by-
  analogy family — novel in the specific choice of VIP, not fringe in approach.
- **Caveat:** a web search can't prove a negative. A proper Scopus/Web-of-Science review is the
  due-diligence step before any publication novelty claim.

Sources:
- SCARS (Zheng 2012): https://www.sciencedirect.com/science/article/abs/pii/S0169743912000032
- CARS-SPA (two-stage): https://pubmed.ncbi.nlm.nih.gov/25078711/
- Double-CARS (calibration transfer): https://www.sciencedirect.com/science/article/abs/pii/S0169743918305872
- Varsel comparison incl. CARS vs VIP: https://www.sciencedirect.com/science/article/abs/pii/S0003267021012162
- Novel importance-scores varsel (2025, MIR/NIR): https://www.sciencedirect.com/science/article/abs/pii/S1386142525000071

## 6. Design decision

**VIP-CARS is to be added as a NEW, separately-named method — additive only. It must never change
the behavior of the existing canonical methods.**

- **PLS1 canonical CARS (`|coef|`) stays exactly as-is.** It is the published Li 2009 method and
  carries that citation. Untouched.
- **CARS-Tree stays as-is.** Untouched.
- The VIP variant is offered *alongside* canonical CARS as a distinct selectable method (dasp's own),
  legitimately available for **PLS1, PLS-DA (classification), and PLS2** — because it is presented as
  a separate method, not a silent modification of CARS. On PLS1/PLS-DA it must be named and validated
  as dasp's method, not shipped as "CARS."
- Naming TBD (e.g. "VIP-CARS" / "CARS-VIP"); must be distinct in the UI, exports, and any citations
  so users never confuse it with canonical CARS.

Rationale: canonical CARS on PLS1 is a published method with a novelty/citation claim we must not
break. Presenting VIP-CARS as its own method (the way CARS-Tree already is) lets us offer it across
all PLS paths without touching the canon.

## 7. Open questions / next steps

1. **Naming** — settle the method name (UI label, export string, docstring).
2. **A/B validation incl. wall-clock** (project rule: "neutral must include wall-clock") — VIP-CARS
   vs canonical CARS on real data; VIP should be ~flat wall-clock (reuses the existing PLS fit).
   Report joint-Q2 / F1 *and* wall-time; a tie-and-slower is a regression.
3. **Lit due-diligence** — Scopus/WoS review before any novelty claim; check the 2025 "importance-
   scores based variable selection" paper (S1386142525000071) for overlap.
4. **Implementation shape** — add a signal selector to `cars_selection`; route VIP through the
   existing `get_feature_importances()` dispatcher (`models.py:1790`) rather than reimplementing
   inline; reuse the multi-Y `cars_scaled_coef` aggregation for the PLS2 case.
5. **PLS-DA VIP** — VIP is defined for PLS-DA too; confirm `compute_vip` handles the classification
   PLS fit, or add the DA path.
