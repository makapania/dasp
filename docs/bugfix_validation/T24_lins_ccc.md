# T-24 validation: Lin's CCC metric addition

**Branch:** `fix/T24-lins-ccc` (rebased to current main + GUI plumbing commit `0e851bb`)
**Status:** APPROVED for merge (after rebase + GUI fix + strict-test pass)
**Author:** Opus 4.7, 2026-04-30
**Verdict:** APPROVE — formula bit-correct vs Lin 1989, user's explicit call to merge despite
the field-alignment soft-flag, GUI plumbing gap closed before merge.

Investigation findings: [T24_findings.md](T24_findings.md).

---

## TL;DR

T-24 is different from the other four validations: it's an **addition**, not a fix. dasp main
does not have Lin's CCC at all; the branch adds it as a regression metric (`CCC` for
calibration, `CCCcv` for CV-pooled) wired through all four result-construction paths
(grid/Bayesian-1/Bayesian-2/NSGA-II) plus the exported-script template.

Three things to validate:
1. **Formula correctness vs Lin 1989** — bit-correct, verified empirically.
2. **Field alignment** — soft-flagged. CCC is canonical in clinical method-comparison since Lin 1989
   (5,000+ citations) and adopted in R `yardstick`/`metrica`/`keras3` and NIRPY Research's NIR
   tutorial, but dedicated chemometrics packages (`mdatools`, `rchemo`, `pls`) don't include it
   by default. Bone FTIR / paleoanthropology / isotope domain papers tend to report
   R²/RMSE/Bias rather than CCC. The user explicitly called to add it anyway: "EVEN IF ccc IS
   NOT SUPER common in my domain and not used by the big programs, it is clearly relevant and
   a small addition so i would say go ahead."
3. **Distribution-model check** — Originally failed (CCC column would auto-discover but tooltip +
   sort-direction were missing). Closed by commit `0e851bb` adding `CCC`/`CCCcv` to
   `higher_is_better_cols` and the metrics tooltip dict per the user's explicit pre-merge
   requirement: "we must fix the gui gaps first."

## 1. Canonical Lin 1989 CCC formula

**Lin, L. I.-K.** (1989). _A concordance correlation coefficient to evaluate reproducibility._
Biometrics 45(1), 255–268.

```
CCC = 2 · cov(X, Y) / (var(X) + var(Y) + (mean(X) - mean(Y))²)
```

Decomposes into precision (Pearson r) × accuracy (bias correction factor C_b):

```
CCC = r · C_b
C_b = 2 / (v + 1/v + u²)        where v = sd_X / sd_Y,  u = (μ_X - μ_Y) / sqrt(sd_X · sd_Y)
```

ddof=0 (population variance) per Lin's closed-form. Range: [-1, 1], 1 = perfect agreement.

## 2. What the branch adds

Source changes (verified by the investigation agent + `git diff`):

| File                                              | What                                                 |
|---------------------------------------------------|------------------------------------------------------|
| `src/spectral_predict/scoring.py`                 | New `lins_ccc()` helper, lines ~369-410              |
| `src/spectral_predict/search.py`                  | Emit `CCC`/`CCCcv` in grid regression result dict    |
| `src/spectral_predict/unified_bayesian.py`        | Emit in both Bayesian regression paths               |
| `src/spectral_predict/nsga2_search.py`            | Emit in NSGA-II regression result rows               |
| `src/spectral_predict/templates/validation.py`    | Inline `_lins_ccc()` for exported standalone scripts |
| `tests/test_scoring.py`                           | New `TestLinsCCC` class, 14 tests                    |
| **GUI plumbing (added 2026-04-30 per user req):** |                                                      |
| `spectral_predict_gui_optimized.py:1447-1448`     | Tooltip dict entries for CCC/CCCcv columns           |
| `spectral_predict_gui_optimized.py:27964-27968`   | `higher_is_better_cols` += `'CCC', 'CCCcv'`          |

Display convention: `0.0` for one-sided degeneracy (constant predictions or constant truth);
`1.0` for both-equal-constants. Lin (1989) leaves these undefined; the implementation's
choices are documented and tested.

## 3. Empirical correctness check (from agent investigation)

| Input                                       | Expected CCC | Result    |
|---------------------------------------------|--------------|-----------|
| Identity (`y_pred = y_true`)                | 1.0          | 1.0 ✓     |
| Anti-correlation (`y_pred = -y_true`)       | -1.0         | -1.0 ✓    |
| `y_pred = 2·y_true` (Pearson r=1, bias=0)   | 0.8 (closed) | 0.8 ✓     |
| `y=[0,1,2], pred=[1,2,3]` (closed-form)     | 4/7          | 4/7 ✓     |
| Random uncorrelated                          | ~0           | ~0 ✓     |
| Constant predictions                         | 0.0          | 0.0 ✓     |
| Constant truth                               | 0.0          | 0.0 ✓     |
| Both equal constants                         | 1.0          | 1.0 ✓     |

The `test_ccc_finite_sample_ddof_zero_exact` test (added per Codex suggestion) is the
discriminator that locks ddof=0 in — it would change to a different value if anyone tried
ddof=1.

## 4. Field-alignment check (soft-flagged but accepted)

**For CCC adoption:**
- Lin 1989 is canonical for method-comparison since the original paper (5,000+ citations)
- R `yardstick`, `metrica`, `keras3` include CCC
- NIRPY Research (NIR tutorial site) recommends CCC alongside R²/RMSE
- It's a real metric, well-defined, with no controversy on the formula

**Soft-flagged by the agent:**
- Chemometrics-specific R packages (`mdatools`, `rchemo`, `pls`) don't include CCC by default
- The agent searched but **could not verify** the original plan's claim that "the FTIR Bone
  PLS paper reports CCC alongside RMSE/R²"
- Bone FTIR / paleoanthropology / isotope domain papers the agent could find report
  R²/RMSE/Bias, not CCC
- PLS_Toolbox / SIMCA / Unscrambler likely don't report CCC by default (vendor docs partly
  behind license walls — "weak no")
- Soft-flagged per master rule's caveat 4 ("if expensive software doesn't do it, treat with
  suspicion")

**User's explicit decision:** "EVEN IF ccc IS NOT SUPER common in my domain and not used by
the big programs, it is clearly relevant and a small addition so i would say go ahead."

This satisfies the master rule's escape hatch — the user authoritatively decides what
metrics belong in their tool, especially when the metric is correctly defined per Lin 1989,
backed by 5,000+ citations in the broader scientific literature, and adds an information-only
column with no behavior change to existing modeling.

## 5. Distribution-model check (closed before merge)

**Original gap (flagged by investigation agent):**
- The CCC column auto-discovers in the results table (column discovery from dataframe schema)
- But hovering shows no tooltip (CCC not in metrics tooltip dict)
- And clicking the CCC column header sorts ascending (worst-first) instead of descending
  (best-first) because CCC was missing from `higher_is_better_cols`

**Closed by commit `0e851bb`** per the user's explicit requirement ("we must fix the gui
gaps first"):
- Added `CCC`/`CCCcv` to `higher_is_better_cols` set at line 27964-27968 (descending sort
  default, matching R²/RPD/RER behavior)
- Added tooltip dict entries at line 1447-1448 citing Lin 1989 Biometrics 45:255-268,
  explaining CCC penalizes both correlation departures AND bias/scale shifts (vs. Pearson r)

This is the bundled-app-distribution principle from T-26: backend-only knobs in a GUI app
are dead code. CCC is now a fully-surfaced GUI column.

## 6. Test results

**Branch (post-rebase):** `pytest tests/test_scoring.py -v` (full file, not just CCC):
```
30 passed in 0.90s
```

The 14 `TestLinsCCC` tests + 16 other scoring tests all pass after the GUI plumbing commit.
GUI changes don't touch scoring logic, so the CCC tests are unaffected.

**Main (pre-merge):** Zero CCC tests because the metric doesn't exist. Discrimination is
trivial — main can't even run the test file.

## 7. Risks weighed

- **Back-compat:** Purely additive columns. Old CSVs load fine. Existing metric
  computations untouched. ✓
- **Performance:** CCC adds one O(n) computation per fold and per result row. Negligible
  vs. PLS fits. ✓
- **GUI clutter:** Two more columns in the results table. The user has flexibility to
  hide columns via the existing column-visibility UI; no forced behavior change. ✓
- **Field-alignment:** Soft-flagged but explicitly accepted by the user. ✓
- **Internal consistency:** No competing implementations to reconcile (unlike T-05 where
  `contaminant_analysis.py` already had canonical VIP). ✓

## 8. Verdict

**APPROVE for merge.**

Reasoning:

1. **Formula bit-correct vs Lin 1989.** Verified empirically across 8 test cases including
   the closed-form `4/7` ddof-discriminator.
2. **User's explicit decision overrides the soft-flag.** "EVEN IF ccc IS NOT SUPER common in
   my domain and not used by the big programs, it is clearly relevant and a small addition
   so i would say go ahead." Master-rule caveat 6 (a real finding can warrant zero action,
   inverse: a soft flag can be overridden by user judgment for additions).
3. **Distribution-model gap closed.** GUI tooltip + sort-direction added per user's explicit
   pre-merge requirement, before merge attempt.
4. **No back-compat or performance risk.** Purely additive metric column.
5. **No commercial-software false-alarm risk on the formula itself.** Lin 1989 is canonical
   and the implementation matches.
6. **Tests cover the discriminating behavior.** 14 unit tests including ddof=0 lock-in.

This is the only validation in the gate so far that's an addition rather than a fix, and
the only one with a soft-flagged field-alignment that the user explicitly accepted. Both
points worth recording in the `bugfix_validation/README.md` lessons list.

## 9. Pre-merge checklist

- [x] User reads + approves this validation note (user decision recorded above)
- [x] Rebase `fix/T24-lins-ccc` onto current main (done — same conflict pattern as T-05/T-07/T-10)
- [x] GUI plumbing fix committed on the T-24 branch (commit `0e851bb`)
- [x] Re-run `pytest tests/test_scoring.py -v` post-rebase + GUI fix (30/30 pass)
- [ ] Run regression sweep on T-24-relevant tests (running in background as of 2026-04-30)
- [ ] If green, fast-forward merge into main
- [ ] Push
- [ ] Update `docs/bugfix_validation/README.md` status table to MERGED
