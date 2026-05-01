# T-15 LeaveOneGroupOut / GroupKFold — DROPPED

**Branch:** none (no implementation)
**Status:** DROPPED 2026-04-30 — user decision after gate investigation
**Date:** 2026-04-30
**Investigation:** [T15_findings.md](T15_findings.md)

## 1. Roadmap framing

> "T-15: Add LeaveOneGroupOut / GroupKFold + group-column concept. The
> FTIR Bone PLS paper's headline finding (MCC = 0.636 for LDA on PLS
> scores via pooled LOGO ≥5 sites) depends on holding out whole sites
> during CV. Standard 5-fold CV across pooled spectra is optimistically
> biased when within-site spectral correlations exist. LOGO is standard
> practice in multi-site bone-FTIR studies and in the broader chemometrics
> transferability literature. Without LOGO, transfer claims aren't
> defensible to reviewers."

## 2. User's pre-investigation skepticism

The user pushed back on this framing before any work began:
> "I'm not sure what we should do — but useful certainly. Since group
> composition can be so variable, [LOGO] seems complicated to do well
> in practice."

The gate's job: validate that skepticism against actual chemometrics
literature + competitor parity + the user's own data regime.

## 3. What the gate found

### 3a. Competitor parity: mixed, not universal

| Tool | Group-aware CV exposed? |
|---|---|
| PLS_Toolbox / Solo (Eigenvector) | YES (Custom vector + Contiguous Blocks) |
| OPUS / Bruker QUANT | NO (only "Cross Validation" vs "Test Set Validation") |
| SIMCA / Sartorius | Ambiguous public docs |
| CAMO Aspen Unscrambler | Ambiguous public docs |
| mdatools (R) | NO group splitter |

Adding LOGO to dasp would match PLS_Toolbox's most-explicit capability,
not invention — but it is NOT universal across competitors.

### 3b. Chemometrics literature: split, NOT mandating LOGO

- **Westad & Marini 2015** (canonical chemometrics validation tutorial):
  recommends **external test sets for n > 50**, with segmented CV as
  fallback.
- **Workman 2018** (calibration transfer review): assumes
  master/child instrument with **paired holdout test sets** — external
  sets, not LOGO.
- **Filzmoser 2009 rdCV**: for iid bias correction, not transferability.
- **Soneson 2014 PLOS One**: confirms k-fold confounding bias but
  recommends **prevention at design time**, not LOGO.

**No single canonical chemometrics citation mandates LOGO** over external
test sets for transferability claims.

### 3c. Practical viability — the user's empirical concern is correct

The user's own paper data (`paper/results/cross_dataset_unified_min5.meta.json`)
after the ≥5-per-site filter retains **15 sites with N per site ranging
5 to 100 (20× ratio)**. MEC alone is 35% of the data.

Archive notes from the paper itself document the exact failure mode the
user predicted: "K sites that fail (MAR, THA) are dominated by severely
degraded bone; when held out, the training set loses most examples of
extreme degradation."

LOGO without uncertainty quantification (T-16 block bootstrap) produces
high-variance fold estimates in this regime. The user described it
correctly: complicated to do well in practice.

### 3d. Scope estimate

Roadmap claimed 3-5 days; agent estimated **1.5-2 weeks for T-15 alone**
(GUI dropdowns, cost-estimator updates, propagation through `search.py`,
`bayesian_utils.py`, `unified_bayesian.py`, `contamination.py`,
`preprocessing_discovery.py`, plus new tests).

Paired T-15 + T-16 (the agent's recommended unit, since LOGO point
estimates without uncertainty bands are a footgun): 3-4 weeks.

## 4. Verdict

**DROP** by user decision 2026-04-30.

### Reasoning

The single load-bearing reason to keep T-15 in scope was that the user's
own FTIR Bone PLS paper §2.6 explicitly defends LOGO over random k-fold
for transferability. Since the user is not actively shipping that paper
*now* and the chemometrics literature does NOT mandate LOGO (canonical
solution is external test sets per Westad & Marini 2015 / Workman 2018),
the cost-benefit doesn't favor implementation:

- 1.5-2 weeks of engineering for a feature that produces footgun results
  in the user's primary data regime (uneven group sizes, confounded
  batches).
- LOGO alone without T-16 uncertainty quantification is publishing-grade
  misleading.
- T-15 + T-16 paired = 3-4 weeks; better spent on items that deliver more
  user value.
- External test sets (the canonical chemometrics solution) are already
  partially supported in dasp's existing validation workflow — could be
  enhanced if the user later wants stronger transferability claims.

### What this leaves unaddressed

- The original T-01 reframe ("external-test-set workflow") — still
  pending user decision. T-15 being dropped doesn't remove the need for
  cleaner external-test-set UX if/when it becomes the right priority.
- The user's paper §2.6 LOGO defense — if the user revives the paper
  push, T-15 would need re-opening as the paired T-15+T-16 unit.

## 5. Lesson reinforced

The user's domain skepticism was empirically validated by the agent's
investigation. The "field doesn't actually mandate this" framing is the
**second gate-caught case** of a roadmap ticket overstating chemometrics
canonical practice (T-26 SNV was the first; this is the second).

Pattern: prior agents framed sklearn/AutoML/biostatistics conventions as
"chemometrics standard practice." The gate's documentation lookup
(Westad & Marini, Workman, Filzmoser, Soneson) is what catches it.
Future "for publication-defensible claims you must do X" framings need
to be tested against actual chemometrics journal practice, not against
ML-conference bias-correction patterns.

## 6. Future re-evaluation criteria

Re-open T-15 only if:
- The user revives the FTIR Bone PLS paper push that depends on LOGO.
- A new transferability project demands site-aware CV with sufficient
  per-site N to make LOGO statistically meaningful (20+ samples per group
  per Westad & Marini's segmented-CV guidance).
- T-16 paired implementation becomes a priority — at which point T-15
  becomes the cheaper half of a paired unit rather than a standalone
  footgun.
