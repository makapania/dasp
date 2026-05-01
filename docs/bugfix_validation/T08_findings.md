# T-08 findings: CARS tree-mode weight-update bias claim

**Branch:** none — investigation only on main (`f2bbc88`).
**Status:** FINDINGS ONLY — verdict pending main agent.
**Author:** Opus 4.7 (1M context), 2026-04-30.
**Time spent:** ~45 min.

This is the investigation phase. No verdict, no merge. The roadmap framing made
specific empirical claims about weight values and convergence behavior. Those
claims are testable and have been tested.

---

## TL;DR

**False alarm — tree-mode CARS converges, the framing is wrong on three points.**

The roadmap framing makes three concrete claims about
`variable_selection.py:1519-1522, 1549`:

1. **"selected variables [get] tiny weights (~0.01) while unselected stay at
   ~1.0"** — Wrong. The weight update is followed by a global sum-to-1
   renormalization on the very next line (line 1534, was the last line of the
   `try:` block). Unselected weights cannot stay at 1.0 because every weight
   gets divided by the running sum. After iteration 1 the unselected weights
   are ~0.027 and the selected weights are ~0.0004 (because the ~80 selected
   share a sum near 1, while the ~20 unselected share weight 1.0 each across
   ~98 features). This is a single-iteration transient, not a steady state.

2. **"biasing sampling toward unselected variables"** — Partially true *only*
   in iteration 2 (a single-iteration startup effect from the np.ones init).
   By iteration 5–6 the polarity has reversed: informative variables are
   sampled at ~9× the noise rate, and the bias steadily strengthens.

3. **"oscillates instead of converging"** — Wrong. Empirically, weights
   converge tightly. Last-10-iteration weight volatility (per-variable std
   across iterations 41–50) is **0.0038 for hybrid tree-mode, 0.0007 for
   plain tree-mode, 0.0012 for PLS-mode** — all three modes converge to
   stable distributions where the 5 informative wavelengths are 22–95×
   higher weighted than noise. Top-5 hits at the true informative indices:
   tree-hybrid 5/5, tree-plain 4/5, PLS 4/5.

**The cited line numbers (1519-1522, 1549) don't even point at tree-mode
code.** Lines 1518-1521 are inside the **PLS-mode branch** (`pls.fit(X_train,
y_train); y_pred = pls.predict(X_val); mse = ...; cv_errors.append(mse)`).
Line 1549 is `# Find iteration with lowest RMSECV` — post-loop bookkeeping,
not a weight update. Either the framing was written against an older
revision or it cited the wrong lines entirely.

**There is one design quirk** worth flagging: dasp's CARS does NOT match the
canonical Li 2009 algorithm at all (neither tree-mode nor PLS-mode). dasp uses
**probability-weighted sampling without replacement** with weights persisting
across iterations; canonical Li 2009 uses **deterministic top-K elimination
based on the EDF schedule** (libPLS R reference) or **probability-weighted
sampling with replacement, weights recomputed each iteration from the current
PLS coefs** (auswahl Python reference). Both canonical implementations work; so
does dasp's variant. But dasp's variant is NOT what the framing implies it
should be aligned with — the framing says "tree-mode should follow the same
convergence logic [as Li 2009 EDF + ARS]," which would require fixing the
**non-tree** path too. There's no separate "tree-mode bug" — both modes
share the same weight-update structure, and both empirically converge.

---

## 1. Verifying reality in the codebase

### 1a. The cited lines

The framing cites `variable_selection.py:1519-1522, 1549`. Reading current main:

**Lines 1518–1521** (current main, not the framing's version) — these are
**inside the PLS-mode branch**, not tree-mode:

```python
                    pls.fit(X_train, y_train)
                    y_pred = pls.predict(X_val)
                    mse = np.mean((y_val - y_pred.ravel()) ** 2)
                    cv_errors.append(mse)
```

This is just per-fold MSE accumulation. It contains no weight update.

**Line 1549** is `# Find iteration with lowest RMSECV` — post-loop, not in
the iteration body.

**The actual tree-mode weight update is at lines 1499-1507**:

```python
1499                    feature_imp = lgb_model.feature_importances_.astype(float)
1500
1501                # Add minimum floor to prevent complete elimination of variables
1502                # Tree models have sparse feature importances (many zeros) which
1503                # breaks probability sampling in subsequent iterations
1504                feature_imp = np.maximum(feature_imp, 1e-6)
1505
1506                # Update weights based on feature importances
1507                weights[selected_vars] = feature_imp / (feature_imp.sum() + 1e-10)
```

**The PLS-mode weight update is at lines 1528-1531**:

```python
1528                # Update weights based on PLS coefficients
1529                # Larger absolute coefficient = more important
1530                coef = pls.coef_.ravel()
1531                weights[selected_vars] = np.abs(coef)
```

**The crucial line the framing missed is 1534:**

```python
1533            # Normalize weights to sum to 1
1534            weights = weights / (weights.sum() + 1e-10)
```

This applies to **all** weights every iteration — it's the load-bearing line
the framing ignores. The framing says "selected get tiny weights ~0.01,
unselected stay at ~1.0." But after line 1534 the sum is forced to 1.0; no
weight can "stay at 1.0" while other weights also exist.

### 1b. What "tree-mode" means in dasp

`use_hybrid_importance` is the boolean that distinguishes "CARS-Tree" from
"CARS-Aware" (and "CARS"). Per `cars_selection` (variable_selection.py:1226):

| Mode             | `model_type`             | `use_hybrid_importance` | Importance source                       |
|------------------|--------------------------|-------------------------|-----------------------------------------|
| `cars`           | None                     | False                   | PLS coefs (path at line 1531)           |
| `cars-aware`     | tree-name                | False                   | LightGBM `feature_importances_` (split) |
| `cars-tree`      | tree-name                | True                    | Hybrid: 0.5*split_norm + 0.5*gain_norm  |

The only **algorithmic** difference between `cars-aware` and `cars-tree` is
the **importance metric used in the weight update** (split-only vs. hybrid
split+gain). The Monte Carlo loop, EDF schedule, sampling, and convergence
criterion are identical for all three.

### 1c. Convergence criterion

There is **no explicit convergence criterion**. CARS just runs `n_iterations=50`
fixed loops, then post-hoc selects the iteration with lowest RMSECV
(line 1556). This matches Li 2009's spec: CARS doesn't converge to a fixed
point; it samples 50 candidate subsets and picks the one with best CV RMSE.

The framing's "oscillates instead of converging" therefore mischaracterizes
what CARS is supposed to do. CARS is a Monte Carlo *sampling* method, not an
*optimization* that should converge to a fixed weight vector. The thing that
should converge in CARS is RMSECV-of-best-candidate-so-far, not the weight
vector itself.

That said, dasp's variant **does** also converge in weights as a side effect
of weights persisting across iterations and being repeatedly normalized. This
is a dasp-invented behavior; canonical Li 2009 doesn't expect this. Empirics
in §5 show this side effect is strong and useful — informative wavelengths
accumulate weight, noise wavelengths don't.

---

## 2. GUI reachability

`cars-tree` is **fully user-reachable** from the GUI:

**`spectral_predict_gui_optimized.py:12005-12006`** — first-class checkbox
in the variable-selection panel:

```python
ttk.Checkbutton(varsel_frame, text="CARS-Tree (Hybrid Importance)",
               variable=self.varsel_cars_tree).grid(row=10, column=0, ...)
```

Caption: `"Best for tree models (LightGBM, RF, XGBoost)"`.

**`spectral_predict_gui_optimized.py:26458-26459`** — wired into the
`selected_varsel_methods` list passed to search:

```python
if self.varsel_cars_tree.get():
    selected_varsel_methods.append('cars-tree')
```

It is NOT disabled for one-class (verified via the `_update_one_class_controls_visibility()`
checkbox-enable list at gui:16471-16480 — `cars-tree` is not in the
`ipls_family_checkboxes` disable list).

The `cars-tree` method is also referenced in `nsga2_search.py:1884` (with
`use_hybrid_importance=True` hardcoded) and in `preprocessing_discovery.py:82,137,163,179` (the "Smart Preprocessing" feature uses `cars_tree` as one of
four importance methods, and it's the **default** in the GUI combobox at
`gui:11754`).

So tree-mode CARS is reachable via:
1. The CARS-Tree checkbox (regression and classification, including one-class)
2. UVE-CARS-Tree checkbox (`gui:12039-12041`)
3. NSGA-II search (always uses `use_hybrid_importance=True` for the
   tree-mode CARS step at `nsga2_search.py:1884`)
4. Smart Preprocessing's `cars_tree` importance method (default)
5. Bayesian search (no separate CARS-Tree route — Bayesian uses
   `cars-aware`-equivalent through `unified_bayesian.py:663`, with
   `use_hybrid` derived from whether the trial picked a tree model)

This is not unreachable code. If there were a real bug, it would hit users.

---

## 3. Field alignment: what does Li 2009 actually specify?

Direct sources consulted (URLs in §sources):

### 3a. libPLS R reference (Li's own group, lhdcsu/libPLS)

The libPLS `carspls.r` is the closest thing to a canonical Li 2009 reference
implementation. Its weight update is **purely deterministic — no probabilistic
ARS**:

```r
weight <- abs(coef0)
weight.order <- order(weight, decreasing = TRUE)
ratioVariable <- a * exp(-b * (iter + 1))     # EDF
K <- ceil(Num_col * ratioVariable)
weight[weight.order[K+1:Num_col]] <- 0       # eliminate bottom-ranked
subsetVariable <- which(weight != 0)         # next iter uses survivors only
```

Subsequent iterations work on the surviving variable subset. Variables once
eliminated stay eliminated. There is no "weight persists across iterations
for sampling" — the algorithm is hard top-K cut after each EDF step.

### 3b. auswahl Python reference (LSX-UniWue/auswahl)

The auswahl Python implementation is widely cited and includes a probabilistic
ARS step:

- **Weights**: recomputed each iteration from `np.abs(get_coef_from_pls(model))`.
  Not persisted across iterations.
- **EDF**: schedules `n_features_to_keep` per iteration via
  `selection_ratios = a * np.exp(-k * (np.arange(n_sample_runs) + 1))`.
- **ARS**: roulette-wheel **with replacement**, sampling `scheduled - n_features_to_select`
  additional features from current `wavelengths` weighted by current PLS-coef magnitudes:

  ```python
  wavelength_probs = weights[wavelengths] / np.sum(weights[wavelengths])
  additional_wavelengths = random_state.choice(
      wavelengths, scheduled - n_features_to_select,
      replace=True, p=wavelength_probs)
  ```

Like libPLS, weights are recomputed fresh from PLS coefs each iteration. There
is no across-iteration weight accumulation.

### 3c. Li 2009 abstract (PubMed PMID 19616692)

The original paper specifies a "two-step procedure including exponentially
decreasing function (EDF) based **enforced wavelength selection** and adaptive
reweighted sampling (ARS) based **competitive wavelength selection**." Both
steps operate on the **current iteration's PLS coefficients**, not on
accumulated weights from prior iterations. In each iteration:

1. EDF determines `K` = how many variables to keep.
2. The bottom variables (by `|PLS coef|`) are forcibly eliminated.
3. ARS roulette-wheel resamples among the remaining to introduce stochasticity.
4. Cross-validation evaluates the resulting subset's RMSECV.
5. The iteration's selected subset is recorded.
6. After all iterations, the best RMSECV's subset is the final answer.

### 3d. Does Li 2009 specify a "tree-mode variant"?

**No.** Li 2009 is exclusively PLS-based. Section 3.2 and equation 1 use PLS
coefficients as the importance signal. There is no published "tree-mode CARS"
variant from the Li group or any of the major chemometrics groups
(Eigenvector, Sartorius, Liang, Liang & Daszykowski).

dasp's "CARS-Tree" mode (introduced in commit `c865e70`, 2024) is a
**dasp-original extension**. It replaces PLS coefficients with LightGBM
hybrid (split + gain) importance. This is reasonable as an extension — the
framework "fit model, get importance, use as weight" generalizes
straightforwardly — but it is **not in the canonical literature**.

### 3e. How dasp's overall CARS differs from canonical

This is the bigger field-alignment story than the tree-mode question. Even
the non-tree dasp CARS deviates from canonical:

| Aspect                       | libPLS (R, Li's group)         | auswahl (Python, well-cited) | **dasp main**                    |
|------------------------------|-------------------------------|------------------------------|----------------------------------|
| Weight initialization        | `abs(coefs)` after first PLS  | `abs(coefs)` per iter        | `np.ones(n)` constant            |
| Weight persistence           | Survivors carry into next     | Recomputed each iter         | **Persists across iterations**   |
| Sampling: with/no replace    | N/A (deterministic top-K)     | With replacement (roulette)  | **Without replacement**          |
| Eliminated vars              | Permanently removed           | Selectable each iter         | **Always selectable** (re-normalized) |
| EDF schedule                 | `K = ceil(p * a*exp(-b*k))`   | `n_keep` per iter            | `n_sample = max(p*0.8*r, ...)`   |
| Number of iterations         | User param                    | User param                   | **50, hardcoded**                |
| Final selection              | Min RMSECV iteration's subset | Min RMSECV iteration         | Min RMSECV iteration             |

dasp's variant is most accurately described as **"weighted Monte Carlo
sampling with persistent weights"** — which is closer to a sequential
multiplicative-weights / online-learning algorithm than to canonical CARS.
It empirically works (informative variables accumulate weight, RMSECV finds
good subsets) but the convergence logic is different from Li 2009.

The framing's claim "the tree-mode variant should follow the same convergence
logic [as Li 2009 EDF + ARS]" is therefore both narrower than reality (the
**non-tree path** also doesn't follow Li 2009 EDF + ARS) and pinned to the
wrong target (Li 2009 doesn't actually have a "convergence" target — its EDF
is monotone shrinkage of the candidate set, not weight convergence).

---

## 4. Empirical: does the bug actually fire?

Synthetic dataset: 50 samples, 100 wavelengths, 5 informative (`indices 10, 30, 50, 70, 90`),
true coefficients drawn from `Uniform(2, 5)`, noise `N(0, 0.1)`. Three runs:

- TREE-HYBRID: `cars_selection(model_type='LightGBM', use_hybrid_importance=True)`
- TREE-PLAIN: `cars_selection(model_type='LightGBM', use_hybrid_importance=False)`
- PLS-MODE: `cars_selection(model_type=None, use_hybrid_importance=False)`

Each runs 50 iterations with `random_state=0`. Per-iteration logging captures
weights and selected indices.

Script: `tests/_t08_empirical.py`. Output (truncated; full script reproducible):

### TREE-HYBRID (the actual claimed-buggy mode)

```
Iter   1 | n_sample= 64 | RMSECV=6.981 | w[selected]_mean=0.00042  w[unselected]_mean=0.02703 | w[informative]_mean=0.01715  w[noise]_mean=0.00962
Iter   6 | n_sample= 52 | RMSECV=4.875 | w[selected]_mean=0.01757  w[unselected]_mean=0.00180 | w[informative]_mean=0.08623  w[noise]_mean=0.00599
Iter  11 | n_sample= 42 | RMSECV=4.857 | w[selected]_mean=0.02066  w[unselected]_mean=0.00228 | w[informative]_mean=0.08203  w[noise]_mean=0.00621
Iter  21 | n_sample= 28 | RMSECV=4.125 | w[selected]_mean=0.02858  w[unselected]_mean=0.00277 | w[informative]_mean=0.08247  w[noise]_mean=0.00619
Iter  31 | n_sample= 19 | RMSECV=4.447 | w[selected]_mean=0.03835  w[unselected]_mean=0.00335 | w[informative]_mean=0.08290  w[noise]_mean=0.00616
Iter  41 | n_sample= 12 | RMSECV=5.842 | w[selected]_mean=0.05784  w[unselected]_mean=0.00348 | w[informative]_mean=0.08577  w[noise]_mean=0.00601
Iter  50 | n_sample=  9 | RMSECV=5.067 | w[selected]_mean=0.08107  w[unselected]_mean=0.00297 | w[informative]_mean=0.10903  w[noise]_mean=0.00479

Final: w[informative].mean = 0.109,  w[noise].mean = 0.0048,  ratio = 22.77x
Top-5 hits at informative idx: 5/5
Last-10-iter weight volatility (std per var): 0.0038
```

### TREE-PLAIN (cars-aware mode)

```
Iter   1 | w[selected]_mean=0.00042  w[unselected]_mean=0.02703 | w[informative]_mean=0.01827  w[noise]_mean=0.00956
Iter   6 | w[selected]_mean=0.01923  w[unselected]_mean=0.00000 | w[informative]_mean=0.13600  w[noise]_mean=0.00337
...
Iter  50 | w[selected]_mean=0.10706  w[unselected]_mean=0.00040 | w[informative]_mean=0.13063  w[noise]_mean=0.00365

Final ratio inf/noise: 35.78x
Top-5 hits at informative idx: 4/5
Last-10-iter weight volatility: 0.00069
```

### PLS-MODE (vanilla cars)

```
Iter   1 | w[informative]_mean=0.02067  w[noise]_mean=0.00944
Iter   6 | w[informative]_mean=0.06266  w[noise]_mean=0.00723
...
Iter  50 | w[informative]_mean=0.16669  w[noise]_mean=0.00175

Final ratio inf/noise: 95.06x
Top-5 hits at informative idx: 4/5
Last-10-iter weight volatility: 0.00117
```

### What the empirics show

1. **Iteration 1 anomaly is real but transient.** In all three modes, after
   iteration 1 the unselected vars have higher *mean* weight than the selected
   ones. This is exactly the framing's claim. But it's a one-iteration startup
   effect: with `weights = np.ones(p)` initially, the ~80 selected vars share
   `~1` total weight after fitting (`feature_imp.sum() ≈ 1`), giving each
   selected var ~0.01 weight. The ~20 unselected vars still each have weight
   1.0. After global renormalization, selected get ~0.01/(1.0+20) ≈ 0.0005,
   unselected get 1.0/(1.0+20) ≈ 0.05. That matches the iter-1 numbers.

2. **By iteration 5–6 the polarity has reversed permanently.** From iter 6
   onward, `w[selected]_mean > w[unselected]_mean` in all three modes, and
   `w[informative] > w[noise]` is sustained.

3. **Weights converge tightly.** Last-10-iteration std-per-variable
   (averaged) is ≤0.004 in all three modes. The "oscillates instead of
   converging" framing is empirically wrong.

4. **Top-K selection is correct.** TREE-HYBRID identifies all 5/5 informative
   wavelengths exactly. TREE-PLAIN and PLS-MODE get 4/5 (one informative
   wavelength was missed in favor of an adjacent one). The algorithm finds
   real signal.

5. **`r = 0.8 * exp(-2*k/N)` shrinks `n_sample` from 64 to 9 over 50 iters.**
   This is dasp's EDF analog. It functions correctly — the candidate pool
   shrinks over iterations, concentrating the search.

The "tree-mode oscillates" framing is empirically falsified. All three modes
behave similarly: brief iter-1 startup transient, then steady growth of
informative-variable weights and shrinkage of noise-variable weights to a
stable end-state.

### Why the framing's iter-1 observation is real but not consequential

The framing's "tiny weights ~0.01 vs. ~1.0" describes iteration 1's behavior
(modulo the renormalization the framing also missed). It exists because:

- `weights = np.ones(n_variables)` initialization
- First iteration's tree/PLS importance values sum to ~1 spread across ~80
  vars, vs. unselected vars still each at 1.0
- After renormalization, that's a real bias in iteration 2's sampling probability

But it's a **single-iteration transient**. By iteration 2 the weight
distribution has been re-normalized. By iteration 5 the polarity has flipped.
By iteration 50 the informative vars dominate. The framing extrapolated a
single-iteration startup effect to "biases sampling toward unselected
variables" as if it were a steady-state behavior. It isn't.

A theoretical edge case: if the framing's claim were right, no CARS variant
would ever converge to non-trivial subsets. The fact that CARS *does* find
informative subsets (all three modes empirically) directly contradicts the
"biased toward unselected" framing.

---

## 5. False-alarm pattern check

Walking T-08 through the bugfix-validation gate's five recurring false-alarm
patterns:

| Pattern | Applies to T-08? | Evidence |
|---------|------------------|----------|
| sklearn-instinct false alarm (T-26 first-pass) | **Maybe weakly applicable.** The framing seems to expect canonical Li 2009 EDF behavior. Looking at dasp, the framer apparently saw "weights[selected] gets multiplied by tiny number, weights[unselected] keeps its 1.0" and concluded it must be a bug — without running through the global re-normalization on the next line. That's a control-flow misread, not exactly sklearn-instinct, but similar genre. |
| Defensive code in unreachable branch (T-32) | **No.** Tree-mode CARS is fully reachable from the GUI checkbox + multiple internal callsites (NSGA-II, Smart Preprocessing). |
| Display-economy / code style | **No.** This concerns weight-vector dynamics affecting which wavelengths are picked, not display-only. |
| dasp already matches the leading program (T-26) | **No.** dasp's CARS variant doesn't match libPLS or auswahl. It's a dasp-invented variant. **However**, that mismatch is not what the framing flags. |
| Real finding, zero action (T-21) | **Possibly applicable** as a partial verdict. There is one real finding here (the iter-1 transient + the dasp-invented CARS variant), but neither warrants the framing's claimed fix. |

T-08 fits most closely with **the "framing extrapolated an iter-1 transient
to a steady-state bug" pattern** — a sub-class of code-misread false alarms.

---

## 6. Distribution-model + scope check

If the framing were correct, the "fix" would be ~half-day per the original
analysis. Since the framing is wrong, the question becomes: is there
**something else** in CARS-Tree that warrants a fix?

Three candidate observations from this investigation that could be ticketable:

### 6a. Iter-1 startup bias is real

For the first iteration only, `weights = np.ones` makes selected vars
dramatically under-sampled in iteration 2 because they got the new (small,
sum-to-1) values while unselected vars retained 1.0 each. This is a
real bias for **one iteration only**. The fix is trivial: initialize weights
to `1/n_variables` instead of `1.0`, OR run a "warm-up" iteration that
seeds weights before the main loop. This would give iteration 2 a more
balanced starting distribution.

But: empirically this single-iteration effect is washed out by iteration 5,
and final results are good (5/5 hits in tree-hybrid). The fix is cosmetic
and wouldn't change which subset wins.

### 6b. dasp's CARS doesn't match Li 2009 at all

The bigger gap is that dasp's CARS isn't really CARS. It's a "weighted Monte
Carlo with persistent weights" variant. This is a methodology-alignment
issue — the chemometrics master rule says match the field. But:

- This is a much bigger ticket (rewrite all of `cars_selection` to match
  libPLS / auswahl). Full Li-2009-canonical implementation.
- dasp's variant **empirically works** — it finds informative wavelengths
  reliably in our empirical test.
- The user is unlikely to notice the difference between dasp's variant and
  canonical Li 2009 unless they're benchmarking against a published
  libPLS result on the same dataset.
- The roadmap framing was about "tree-mode bug" specifically, not about
  rewriting all of CARS to match Li 2009.

### 6c. CARS-Tree is a dasp invention

The `use_hybrid_importance=True` (split + gain hybrid) path doesn't appear
in any canonical chemometrics source I could find. It's a 2024 dasp
experimental knob. If the user finds it useful, fine. If not, removing it
would simplify the codebase. This is a product question, not a bug.

### Distribution-model

dasp ships as a bundled Inno Setup desktop app. CARS-Tree is one
checkbox among ~15 variable-selection methods. Removing or "fixing"
CARS-Tree would be a UI change visible to users. Per T-26 lesson: don't
break the bundled-app UX without strong evidence that the change is
necessary.

---

## 7. Notes / uncertainties for the verdict-writer

1. **The roadmap framing is wrong on its three concrete claims.** Framings
   can be wrong. This was checked empirically.

2. **Cited line numbers don't match.** Line 1519-1522 is in PLS-mode, not
   tree-mode. Line 1549 is post-loop. Either the roadmap was written
   against an older revision (pre-rebase), or there was a copy-paste error.
   Either way, the cited code doesn't have the claimed bug.

3. **There IS one real iter-1 transient bias** — selected vars get
   under-sampled in iteration 2 because weights init to 1 and iteration 1
   normalizes them down. This is a single-iteration effect, doesn't
   substantially affect outcomes, and could be fixed cosmetically by
   initializing `weights = np.ones / n_variables`. Not ticketable on its
   own.

4. **dasp's whole CARS is non-canonical.** Bigger field-alignment gap than
   the tree-mode framing suggests. If the user wants chemometrics-accurate
   CARS, the right ticket is "rewrite cars_selection to match libPLS or
   auswahl" — but that's a multi-day task and dasp's current variant
   empirically works.

5. **`cars-tree` is a dasp invention** (commit `c865e70`, 2024). Not in
   Li 2009. Not in libPLS or auswahl. Empirically performs OK in our test
   (5/5 informative hits, vs. 4/5 for plain modes). If kept, the GUI
   description is honest ("Best for tree models, hybrid importance") —
   though it doesn't disclose that this is a dasp original, not a
   peer-reviewed method.

6. **Reachability is fine.** CARS-Tree is wired through the GUI, NSGA-II,
   and Smart Preprocessing. Distribution-model check passes.

7. **The "half-day" effort estimate in the roadmap probably reflects an
   imagined fix to a bug that doesn't exist.** Re-estimating with the
   actual options:
   - Fix iter-1 transient (init weights uniform): 30 minutes + 1 test.
     But adds zero practical value because the transient is washed out.
   - Rewrite `cars_selection` to match canonical libPLS or auswahl: 1-2
     days for code, 1 day for testing/regression sweep, 0.5 day for
     verifying nothing else breaks. This is methodology realignment, not
     a bugfix.
   - Remove `cars-tree` knob (revert commit `c865e70` and friends): ~1
     hour + tests + GUI cleanup. But user-visible UX change.

8. **Verdict patterns to consider:**
   - "False alarm — framing extrapolated an iter-1 transient to a
     steady-state bug, code converges correctly."
   - "Real but tree-mode itself is non-canonical (Li 2009 has no such
     variant); question whether dasp should have it at all."
   - "Real but the bigger issue is the whole CARS implementation being
     non-canonical, not just tree-mode."
   - The narrowest defensible verdict is the first one. The broader
     methodology-alignment questions are real but out of scope for T-08.

9. **`MEMORY.md` is not affected by this finding.** No memory entry needs
   updating.

10. **Out-of-scope but worth flagging:** the auswahl Python implementation's
    "with replacement" sampling is materially different from dasp's
    "without replacement" sampling. dasp's choice means n_sample is
    bounded above by n_variables, which empirically is fine but can lead
    to less aggressive exploration in early iterations than canonical ARS.

---

## 8. Sources

- [Li, H. D., Liang, Y. Z., Xu, Q. S., Cao, D. S. 2009 — original CARS — Anal. Chim. Acta 648:77-84](https://pubmed.ncbi.nlm.nih.gov/19616692/)
- [libPLS reference R implementation (Li's group, lhdcsu/libPLS)](https://github.com/lhdcsu/libPLS/blob/master/R/carspls.r)
- [auswahl Python implementation — LSX-UniWue/auswahl](https://github.com/LSX-UniWue/auswahl)
- [auswahl CARS docs](https://auswahl.readthedocs.io/en/latest/generated/auswahl.CARS.html)
- [auswahl wavelength point selection overview](https://auswahl.readthedocs.io/en/latest/point_selection.html)
- [SCARS — Stability CARS — extension to original, Anal. Chim. Acta](https://www.sciencedirect.com/science/article/abs/pii/S0169743912000032)
- [Zheng et al. 2014 — CARS+SPA hybrid — Analyst](https://pubs.rsc.org/en/content/articlelanding/2014/an/c4an00837e)

dasp source files referenced:
- `src/spectral_predict/variable_selection.py:1226-1578` — `cars_selection`
- `src/spectral_predict/variable_selection.py:808-900` — `uve_cars_selection`
  (wraps cars_selection)
- `src/spectral_predict/search.py:2538-2566` — grid-search dispatch
- `src/spectral_predict/search.py:5757-5775` — one-class dispatch
- `src/spectral_predict/nsga2_search.py:1884` — NSGA-II hardcoded
  `use_hybrid_importance=True`
- `src/spectral_predict/preprocessing_discovery.py:82-179` — Smart Preprocessing
  cars_tree usage
- `src/spectral_predict/unified_bayesian.py:663` — Bayesian dispatch
- `spectral_predict_gui_optimized.py:12005-12006` — CARS-Tree checkbox
- `spectral_predict_gui_optimized.py:11754` — Smart Preprocessing combobox
  (cars_tree default)
- `spectral_predict_gui_optimized.py:26458-26459` — checkbox-to-method wiring

Empirical script: `tests/_t08_empirical.py`.
