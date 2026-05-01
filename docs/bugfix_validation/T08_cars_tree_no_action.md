# T-08 CARS tree-mode weight-update bias — DROP / WONT_FIX

**Branch:** none (no implementation)
**Status:** DROPPED 2026-04-30 — false alarm on framing's specific claims
**Date:** 2026-04-30
**Investigation:** [T08_findings.md](T08_findings.md)

## 1. Roadmap framing

> "T-08: `variable_selection.py:1519-1522, 1549` gives selected variables tiny
> weights (~0.01) while unselected stay at ~1.0, biasing sampling toward
> unselected variables. Li 2009 CARS uses exponentially decreasing function
> (EDF) + adaptive reweighted sampling (ARS); the tree-mode variant should
> follow the same convergence logic. The current implementation oscillates
> instead of converging."

## 2. What the gate actually found

**The framing was wrong on three specific claims and on the cited line numbers.**

### 2a. Cited lines reference the wrong branch

Lines 1519-1522 are inside the **PLS-mode** branch (`pls.fit/predict/mse`),
not tree-mode. Line 1549 is post-loop bookkeeping. The actual tree-mode
weight update is at lines 1499-1507. Both PLS-mode and tree-mode are
followed by global sum-to-1 renormalization at line 1534 — which the
framing missed.

### 2b. The three bug claims are empirically disproved

Empirical reproducer at `tests/_t08_empirical.py` (deleted post-investigation;
output preserved in [T08_findings.md](T08_findings.md)):

| Claim | Reality |
|---|---|
| "Selected get tiny weights ~0.01, unselected stay at ~1.0" | Only true for **iteration 1** (startup transient from `weights = np.ones`); renormalization at line 1534 fixes it immediately. |
| "Biases sampling toward unselected variables" | True only at iteration 2; polarity flips by iterations 5-6. |
| "Oscillates instead of converging" | **Empirically false.** Last-10-iteration weight std: 0.0038 (tree-hybrid), 0.0007 (tree-plain), 0.0012 (PLS). All three modes converge tightly. Tree-hybrid recovers 5/5 of the true informative wavelengths. |

### 2c. Reachability check

Tree-mode CARS IS fully GUI-reachable (gui:12005-12006 checkbox, NSGA-II
hardcoded path at `nsga2_search.py:1884`, Smart Preprocessing default,
Bayesian search). Reachability is not the escape valve — the algorithm is
genuinely run by users. It just isn't broken.

## 3. Field alignment — orthogonal finding worth noting

The agent surfaced a real but separate fact: **dasp's *entire* CARS
algorithm is non-canonical, not just tree-mode.** libPLS (Li's own R
package) uses deterministic top-K elimination. auswahl uses
with-replacement roulette-wheel sampling with weights recomputed per
iteration. dasp uses without-replacement weighted sampling with
persistent across-iteration weights. Empirically dasp's variant works,
but it is not Li 2009 and not what the original framing implied.

**CARS-Tree specifically is a dasp-original 2024 invention** (commit
`c865e70`). The user confirmed this and the rationale: canonical CARS uses
PLS regression coefficients to drive its weight updates; tree models
don't expose comparable per-variable regression coefficients (they expose
feature importances), so canonical CARS literally cannot run for tree
models. The choice was either "no CARS for tree models" or "invent a
CARS-like variant using tree importances instead." dasp chose the latter
and called it CARS-Tree.

This makes the standard master-rule frame inapplicable: dasp invented
this for a real reason. The verdict frame becomes "does our intentional
invention work?" not "does it match canon?" Empirical answer: yes, it
converges and recovers informative wavelengths.

## 4. Verdict

**DROP / WONT_FIX.**

Reasoning:
1. **The cited bug doesn't fire.** Lines 1519-1522 are PLS-mode, not
   tree-mode. The renormalization at line 1534 fixes the iter-1 transient
   immediately.
2. **The algorithm converges.** Empirical std on last 10 iterations: 0.0007
   to 0.0038. That's tight, not oscillating.
3. **Variable recovery works.** Tree-hybrid recovered 5/5 informative
   wavelengths in the synthetic test.
4. **CARS-Tree is dasp-invented for a real reason.** Canonical CARS doesn't
   work with tree models because they don't expose PLS coefficients. The
   "should match Li 2009" frame doesn't apply.
5. **T-26 precedent applies in spirit:** real finding can warrant zero
   action. Here the underlying observation (iter-1 transient) is real but
   produces no practical effect.

## 5. Cheap optional cleanup (not required)

The iter-1 transient could be smoothed away by initializing
`weights = np.ones(n_vars) / n_vars` instead of `np.ones(n_vars)`. About
30 minutes of work, zero practical impact since renormalization happens
immediately anyway. Filed as informal nice-to-have; not part of the T-08
disposition.

## 6. Future re-evaluation criteria

Re-open T-08 only if:
- A user reports CARS-Tree producing visibly bad selections in real bone-FTIR
  workflows (none currently reported).
- A future session decides to align dasp's whole CARS implementation with
  libPLS / auswahl canonical patterns. That's a separate scope from "fix
  the bug" — methodology realignment, not bugfix. Estimated effort if
  pursued: ~2-3 days.

## 7. Lesson reinforced

The framing's specific line-number references didn't survive direct code
inspection. This is the **third gate-caught false alarm** with the same
shape (T-26 SNV, search.py:2855 top_n_vars hardcoding, T-08 tree-mode):
prior agents misread the code or applied the wrong literature lens, and
the framing didn't survive step-1 of the gate methodology (verify reality
in the codebase, including line numbers and control flow).

Future ticket framings citing "buggy" behavior at specific line numbers
should be expected to have this kind of error. The gate's reality check
is the only mechanism catching it. Without it, dasp would have shipped
~30 min to ~2-3 days of work for zero behavioral improvement.
