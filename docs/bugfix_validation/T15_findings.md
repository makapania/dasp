# T-15 findings: LeaveOneGroupOut / GroupKFold + group-column concept

**Branch:** none yet — T-15 is **not implemented**. Investigation of main + competitor docs + literature.
**Status:** FINDINGS ONLY — verdict pending main agent
**Author:** Opus 4.7 (1M context), 2026-04-30

This investigation directly addresses the user's master-rule challenge: *"is the
group really that important relative to competitors? useful certainly, but
since group composition can be so variable [it] seems complicated to do well
in practice."* No verdict, no merge recommendation. Just answers to the four
empirical questions the user posed.

---

## TL;DR

**Mixed — and the user is half-right.**

1. **Competitor parity:** Of the leading commercial chemometrics packages,
   **only PLS_Toolbox / Solo (Eigenvector)** exposes a first-class
   group-aware CV mechanism — and it does so via a generic "Custom" CV
   vector, not a named `LeaveOneGroupOut` button. **OPUS / Bruker QUANT
   exposes only "Cross Validation" vs. "Test Set Validation" with no group
   primitive.** **CAMO/Aspen Unscrambler** exposes "category variables" and
   segmented validation but has no documented "leave-one-category-out"
   button visible in the marketing or tutorial pages I could verify.
   **SIMCA/Sartorius** documents segmented CV (random/systematic, n
   segments, repeats) but the user-guide section discoverable via web
   search does not surface a "group/site/batch" hold-out as a one-click
   option. **Conclusion:** group-aware CV in commercial spectroscopy GUIs is
   **achievable in PLS_Toolbox via a custom vector**, present-but-implicit
   in CAMO/Sartorius (workaround via category-variable + segmentation), and
   **not exposed in OPUS at all.** dasp adding it would put dasp on parity
   with PLS_Toolbox's most explicit capability — not invent a new pattern.

2. **Field-alignment for transferability claims:** The chemometrics
   literature is **split on default practice but converging on group-aware
   methods specifically when the claim is transferability**. Westad &
   Marini 2015 (*ACA* 893:14–24) — the canonical chemometrics-validation
   tutorial — discusses "segmented" / "category" CV and recommends external
   test sets for n > 50; CV is the small-dataset fallback. Filzmoser
   et al. 2009 rdCV doesn't use site/group structure (the focus is
   bias-correction via repeated double CV on iid samples). Transferability-
   specific literature (Workman 2018 review *Appl. Spectrosc.*, Feudale
   et al. 2002) explicitly assumes a **master/child instrument structure**
   with separate test sets — i.e. *external test sets* are the default
   transferability proof, not LOGO. **The PLOS One 2014 paper by Nygård
   et al.** ("Batch Effect Confounding Leads to Strong Bias in Performance
   Estimates Obtained by Cross-Validation") shows that standard k-fold CV
   with confounded batch effects gives over-optimistic estimates. Their
   recommendation is *prevent confounding at design time*, not "use LOGO."
   **So: LOGO is one tool; external test sets are the more frequently
   recommended canonical solution.**

3. **The user's own paper uses LOGO and explicitly defends the choice
   against random k-fold.** `paper_short_rewrite4.md` §2.6 says: "We chose
   LOGO over a random K-fold split for the cross-dataset analysis because
   random splits would let the model train on neighbors of the held-out
   specimens and inflate the apparent transfer; LOGO simulates the
   deployment case in which an investigator screens a new collection
   without help from training specimens at the same site." This is
   field-aligned reasoning, and the paper ships it.

4. **Practical viability — the user's variable-group-composition
   concern is empirically warranted.** The user's own paper after the ≥5
   filter retains 15 sites with N per site ranging from **5 to 100 (20×
   ratio)**. MEC alone is N=100, making it a leverage point — when MEC is
   held out, the training set loses 35% of specimens. Sites with N=5–9
   (DEN, Andranosoa, Andolonomby, Saint-Martin-Bassin_Tortues, Laurion,
   Ampoza, Tampolove, Lamboara) provide noisy fold-level estimates. The
   site-size sensitivity table (S7) compares ≥1 vs ≥5 protocols and shows
   the headline qualitative ranking is robust, but the absolute numbers
   shift, confirming LOGO is sensitive to group-size choices. The bone-
   FTIR community's practical reality (per the user's own data and
   archive notes — "The K sites that fail (MAR, THA) are dominated by
   severely degraded bone; when they are held out, the training set
   loses most of its examples of extreme degradation") is exactly the
   pathology the user predicted: **single-site holdouts can remove a
   chemistry mode entirely from training**, producing misleading transfer
   estimates.

5. **Alternatives canonical literature recommends when LOGO is fragile:**
   external test sets (Workman 2018, Westad & Marini 2015 for n > 50);
   stratified random splits with site provenance preserved (the paper's
   "random-split sensitivity test"); site-level **block bootstrap** for
   uncertainty around LOGO point estimates (the paper does this too —
   1000 iterations resampling at site level). T-15 + T-16 together
   (LOGO + block bootstrap) is the canonical pairing, not LOGO alone.

**Bottom line for the verdict-writer:** The user is correct that LOGO is
"complicated to do well in practice" with variable group sizes. The user
is also correct that competitors don't make it easy. But **LOGO is
necessary for transferability claims in the user's specific publication
workflow (their paper defends the choice explicitly), and PLS_Toolbox
exposes it as a first-class option** — so dasp adding it is parity, not
invention. The right scope is probably **LOGO + GroupKFold + site-level
block bootstrap (T-16) bundled together**, with explicit GUI warnings
about minimum group sizes and group-size imbalance, plus an alternative
external-test-set workflow path (T-01 reframe) for users who have a clean
holdout. Backing T-15 alone without T-16 leaves the LOGO point estimates
without uncertainty quantification, which is exactly the regime where
the user's "fragile metric" concern bites hardest.

---

## 1. Reality in the codebase (Step 1)

### 1a. `cv_utils.build_cv_splitter` supports only iid splitters

**`src/spectral_predict/cv_utils.py:275-339`** — `build_cv_splitter()`:

```python
if strategy == 'loo':
    from sklearn.model_selection import LeaveOneOut
    return LeaveOneOut()
if strategy == 'repeated_kfold':
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
    if use_stratified:
        return RepeatedStratifiedKFold(...)
    return RepeatedKFold(...)
if strategy == 'kfold':
    if use_stratified:
        return StratifiedKFold(...)
    return KFold(...)
raise ValueError(f"Unknown CV strategy: {strategy!r}. ...")
```

Only `'kfold'`, `'repeated_kfold'`, `'loo'` are accepted. No
`GroupKFold` / `LeaveOneGroupOut` branches.

### 1b. The companion size-estimator already has T-15 placeholders

**`src/spectral_predict/cv_utils.py:222-226`** —
`compute_min_train_fold_size()` already documents the gap:

```python
if cv_strategy in ('group_kfold', 'leave_one_group_out'):
    raise NotImplementedError(
        f"compute_min_train_fold_size: {cv_strategy!r} not supported yet "
        "(T-15 will add group-aware sizing)."
    )
```

### 1c. `search.py` already references T-15 as a follow-up

**`src/spectral_predict/search.py:1100-1115`** — the PLS components-clamp
path already has a comment placeholder:

```python
# T-10: cv_strategy-aware. K-fold/RepeatedKFold use the exact
# floor `n_samples * (folds - 1) // folds`. LOO has train-fold size n-1.
# Group splitters are not yet supported (T-15 follow-up).
```

So T-10 anticipated T-15.

### 1d. Other group-related infrastructure that exists today

- `src/spectral_predict/ga_pls.py` imports `GroupKFold` — but it's an
  internal use for one specific model's inner CV, not exposed to users.
- `src/spectral_predict/contaminant_analysis.py` and
  `src/spectral_predict/analysis_subset.py` use the word "group" but
  for metadata categorical filtering, not CV groups.
- `src/spectral_predict/scoring.py` — **zero matches for "group"**.
  No group-aware metric infrastructure exists.

### 1e. Per-fold downstream paths that would need group plumbing

If T-15 lands, the group vector has to flow through:

- `cv_utils.build_cv_splitter` (entry) — needs new strategies + a
  `groups` param.
- `cv_utils.compute_min_train_fold_size` — needs group-aware sizing
  (the placeholder is already there).
- `cv_utils.cross_validate_with_early_stopping`,
  `cross_val_predict_pooled`, `cross_val_predict_with_early_stopping`,
  `cross_val_score_with_early_stopping` — all currently call
  `cv.split(X, y)` without groups; need `cv.split(X, y, groups=...)`.
- `search.py:run_search`, `run_one_class_search` — need group plumbing
  through every per-fold path.
- `bayesian_utils.py`, `unified_bayesian.py` — Bayesian objective
  needs to receive groups + propagate.
- `contamination.py:run_one_class_cv` — same.
- `preprocessing_discovery.py:_quick_evaluate` — same.

The plumbing is non-trivial. Probably 1–2 weeks scope per the original
roadmap estimate.

---

## 2. GUI surface for groups (Step 2)

### 2a. CV strategy combobox (Analysis tab)

**`spectral_predict_gui_optimized.py:11296-11305`** — the existing CV
strategy widget:

```python
cv_strategy_label = ttk.Label(options_frame, text="CV Strategy:")
...
self.cv_strategy_combo = ttk.Combobox(
    options_frame, textvariable=self.cv_strategy, width=18,
    values=['kfold', 'repeated_kfold', 'loo'], state='readonly'
)
```

To extend: add `'group_kfold'` and `'leave_one_group_out'` to the
`values` tuple, plus a conditional "Group column:" combobox that lists
metadata categorical columns when one of the group strategies is selected.

A second instance lives at **`gui:15178-15186`** (`refine_cv_strategy`)
in the model-refinement tab; same change required there.

### 2b. Group-column concept does not exist anywhere in the GUI today

**`spectral_predict_gui_optimized.py`** — the word "group" appears 30+
times but every match is for: contamination model groups
(`contam_groups`, `source_group_names`), explore-tab visualization
groupings (`explore_n_groups`, color-bin grouping), checkbox card
layout helpers (`_create_checkbox_group`), and operator-grouping for
manual expressions. **None of these connect to CV-fold groups.**

### 2c. Metadata categorical infrastructure that COULD wire to a group dropdown

The Analysis Subset V1 work (`docs/plans/2026-04-19-analysis-subset-v1-glm-worktree-plan.md`,
`src/spectral_predict/analysis_subset.py`) added metadata-categorical
reading into dasp. T-15 GUI exposure could reuse that infrastructure:
the user picks one already-loaded categorical metadata column as the
"group column" via a dropdown next to CV strategy. No new metadata
plumbing needed; the data is already loaded.

### 2d. Distribution-model check

dasp ships as a bundled Inno Setup desktop GUI for non-technical
bone-FTIR / paleoanthropology / archaeology / isotope users. A
backend-only T-15 (e.g., "expose `groups=` parameter on `run_search`")
would be dead code per the T-26 lesson. **The fix has to surface as a
GUI dropdown to deliver value.**

The dropdown design is straightforward (1 combobox + 1 conditional
combobox), but the per-fold backend plumbing is the hard part (Step 6
estimate below).

---

## 3. Competitor coverage (Step 3)

### 3a. PLS_Toolbox / Solo (Eigenvector Research)

**Verdict: yes, exposes group-aware CV via the "Custom" mechanism.**

From the Eigenvector cross-validation docs ([Using Cross-Validation](https://www.eigenvectordocs.com/index.php?title=Using_Cross-Validation),
[Custom CV FAQ](https://www.eigenvectordocs.com/index.php?title=Faq_use_custom_cross_validation_option)):

> "Five different cross-validation methods are available, and these
> methods vary with respect to how the different sample subsets are
> selected for these subvalidation steps. **Custom**: Each of the test
> sets is manually defined by the user."
>
> "In Solo and PLS_Toolbox, the user can also manually specify the
> subsets of samples to be used as tests sets in the cross-validation
> procedure, using the 'Custom' cross-validation method. ... the user
> will be asked to specify a 'custom cross-validation' vector that
> specifies the subsets of objects to be placed in each test subset
> for each sub-validation."

The "Custom" vector is a per-sample integer assigning each spectrum to
a fold. To implement leave-one-site-out, the user provides a vector
where each unique integer = one site. PLS_Toolbox does **not** ship a
named "Leave-one-group-out" button, but the Custom mechanism
**fully covers the use case**.

PLS_Toolbox also documents "Contiguous Blocks" as suitable for
"datasets with grouped or hierarchical structures, where samples are
organised into distinct groups based on shared characteristics, such
as replicates, batches or sampling sites" — a structurally similar
mechanism that requires the user to *order* their samples by site, but
delivers leave-one-site-out behavior when the segment count = number
of sites.

So the field's leading commercial program treats group-aware CV as a
**user-controlled vector**, not a magic dropdown.

### 3b. SIMCA / Sartorius

**Verdict: ambiguous — segmented CV exposed, but "Class ID-based" CV
as a one-click option is not surfaced in publicly searchable docs.**

The SIMCA 15 user guide ([PDF](https://www.sartorius.com/download/544940/simca-15-user-guide-en-b-00076-sartorius-data.pdf))
exists but I could not extract its full text via WebFetch (10 MB cap).
What is searchable:

- SIMCA exposes a `crossval` parameter — the [mdatools R
  port](https://rdrr.io/cran/mdatools/man/simca.html) (Kucheryavskiy,
  who is a co-author of the LOVE paper cited in T-04) documents:
  `cv` can be `1` (full / leave-one-out), an integer (random N segments),
  a list with `'rand'` / `'ven'` / `'loo'` and `nseg`, `nrep`. **No
  group-aware option in the mdatools port.**
- A `classid` (M×1 vector of integer class identifiers) exists for
  classification, but I couldn't verify whether SIMCA exposes a "CV by
  classid" button.

So SIMCA exposes segmented CV (analogous to PLS_Toolbox's venetian
blinds / contiguous blocks), but a clean "leave-one-group-out by site"
button is **not visible in public docs**.

### 3c. CAMO / Aspen Unscrambler

**Verdict: category-variable infrastructure exists, but explicit
"leave-one-category-out" is not in the publicly searchable docs.**

[Unscrambler tutorials](https://www.yumpu.com/en/document/view/7708461/the-unscrambler-methods-camo-software)
and the [data-analysis blog](http://dataanalysis.blog.camo.com/index.php/user-tip-create-category-variables/)
document "Category variables" as a first-class data type (right-click
column → Change Data Type → Category). Cross-validation is exposed
with "Full cross-validation" + "Sample grouping" controls.

Searchable phrasing: "Sample grouping" appears, but I couldn't pin down
whether picking a category variable as the group key gives clean LOGO
behavior, or whether it's only used for segmentation/visualization. The
paid-software paywall blocks deeper verification. **Inconclusive.**

### 3d. OPUS / Bruker QUANT

**Verdict: no group-aware CV exposed.**

The OPUS QUANT manual ([PDF](https://files.nocnt.ru/hardware/science/senterra/opus65-doc-en/quant.pdf)
and Bruker [Quantification page](https://www.bruker.com/en/products-and-solutions/infrared-and-raman/opus-spectroscopy-software/quantification.html))
documents two and only two validation types:

> "There are two validation types: 'Cross Validation' and 'Test Set
> Validation'. The Cross Validation uses the same set of samples for
> calibration and validation. ... The test set validation uses two
> independent sets of samples, one for calibrating the system and the
> other for validating the model."

Cross-validation in OPUS/QUANT is leave-one-out / random-segment based.
**No group, category, site, or batch primitive in the validation UI.**
OPUS is an FTIR-instrument-vendor tool focused on calibration
deployment, not multi-site research workflows. The "Test Set
Validation" path is what bone-FTIR researchers using OPUS would use
for transferability.

### 3e. mdatools R package (Kucheryavskiy)

**Verdict: no group-aware CV.**

[mdatools `simca` docs](https://rdrr.io/cran/mdatools/man/simca.html) +
[crossval.simca](https://rdrr.io/cran/mdatools/man/crossval.simca.html):
`cv` parameter accepts integer, `'loo'`, `'rand'`, `'ven'` —
random/venetian/leave-one-out. **No GroupKFold or LeaveOneGroupOut.**

This is the open-source SIMCA reference and it doesn't ship the
feature.

### 3f. OpenSpecy

[OpenSpecy](https://openspecy.org) is a polymer/spectral library tool,
not a calibration-modeling tool. It doesn't ship CV at all in the
search-and-match workflow; it's a different problem class.
**Not applicable.**

### 3g. GRAMS

GRAMS (Thermo) is an instrument-software / spectral-database tool with
plugin-based PLS via the IQ family. The IQ Predict / IQ Calibration
docs are not freely searchable. **Inconclusive but unlikely** to expose
group-aware CV given the same focus as OPUS (vendor instrument-tied).

### 3h. Competitor summary table

| Tool | Group-aware CV? | How surfaced |
|------|-----------------|--------------|
| PLS_Toolbox / Solo | YES (via Custom vector) | User-supplied per-sample integer fold-assignment vector. Equivalent to LOGO. Also "Contiguous Blocks" with site-ordered data. |
| SIMCA / Sartorius | Ambiguous | Segmented CV (random / venetian / N-fold) is documented; class-ID-based CV not visible in public docs. |
| Aspen Unscrambler (CAMO) | Ambiguous | "Category variables" + "Sample grouping" exist; explicit leave-one-category-out not verifiable in public tutorials. |
| OPUS / Bruker QUANT | NO | Only "Cross Validation" (full/random) vs. "Test Set Validation". No group primitive. |
| mdatools R | NO | `cv = 'loo' / 'rand' / 'ven' / N`. No group splitter. |
| OpenSpecy | N/A | Polymer search/match; no calibration CV at all. |
| GRAMS / Thermo | Inconclusive | Vendor-tied, docs not freely searchable. |
| sklearn (Python) | YES | `GroupKFold`, `LeaveOneGroupOut` first-class. Ships in dasp's stack; just not wired through the GUI yet. |

**Reading:** dasp adding LOGO/GroupKFold puts dasp **on parity with
PLS_Toolbox's most explicit capability** while exceeding OPUS / mdatools
/ SIMCA-as-documented. It does NOT invent a pattern outside the
field — sklearn's primitives are field-standard for the underlying
math, and PLS_Toolbox's Custom vector mechanism is field-standard for
the workflow.

The novelty isn't the math; it's that dasp would be the first GUI
chemometrics tool I can verify with a *named, documented*
"Leave-one-site-out" dropdown rather than a generic Custom-vector
mechanism. That's UX value, not new methodology.

---

## 4. Field alignment in the chemometrics literature (Step 4)

### 4a. Westad & Marini 2015 — the canonical chemometrics-validation tutorial

**Westad, F. & Marini, F. (2015).** *Validation of chemometric models —
A tutorial.* Anal. Chim. Acta 893:14–24.

I could not extract the full PDF via WebFetch (binary content). What I
can verify from the abstract / search results / references in the
literature:

- The paper's central argument: for n > 50, **test set validation is
  preferred**; CV is the small-dataset fallback.
- "Segmented" CV is documented as a CV variant where the user defines
  segments — the same primitive PLS_Toolbox exposes as "Contiguous
  Blocks" and "Custom."
- The paper does NOT, per the search-result summaries, explicitly
  recommend LOGO/GroupKFold as a default. It recommends
  category-stratified test-set validation when site/batch structure
  exists.

So the canonical tutorial's first preference for transferability is
**external test sets**, with segmented CV as a structured-data
fallback — close to LOGO in spirit but described as "user-defined
segments," not "leave-one-group-out."

Sources: [PubMed 26398418](https://pubmed.ncbi.nlm.nih.gov/26398418/),
[ResearchGate](https://www.researchgate.net/publication/281855322_Validation_of_chemometric_models_-_a_Tutorial).

### 4b. Filzmoser, Liebmann, Varmuza 2009 — repeated double CV

**Filzmoser, P., Liebmann, B., Varmuza, K. (2009).** *Repeated double
cross validation.* Journal of Chemometrics 23(4):160–171.

rdCV is positioned as "a strategy for (a) optimizing the complexity of
regression models and (b) for a realistic estimation of prediction
errors when the model is applied to new cases. This strategy is suited
for small data sets and is a complementary method to bootstrap methods."

**rdCV does not use group/site structure.** The outer CV is repeated
random splits; the inner is repeated random splits for hyperparameter
tuning. Filzmoser's R package
[`prm_dcv`](https://rdrr.io/cran/chemometrics/man/prm_dcv.html) is for
robust PLS, again on iid samples.

So Filzmoser et al. is about **bias correction within iid CV**, not
about transferability across groups. It addresses a different problem
than T-15 — it's the small-dataset bias-correction story, not the
multi-site transferability story.

[DOI 10.1002/cem.1225](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.1225)

### 4c. Krstajic et al. 2014 — CV pitfalls (cited in dasp tooltips)

**Krstajic, D.; Buturovic, L. J.; Leahy, D. E.; Thomas, S. (2014).**
*Cross-validation pitfalls when selecting and assessing classification
and regression models.* J. Cheminformatics 6:10.

The paper recommends:
- Repeated grid-search CV for parameter tuning.
- Nested (double) CV for performance assessment.
- Repeated runs to characterize variance.

**It does not specifically recommend group-aware CV either.** The
focus is performance-estimate variance under iid splits. Like
Filzmoser, this is about iid-CV bias, not group structure.

[Open access](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10)

### 4d. Brereton chemometrics textbooks

Brereton's *Chemometrics: Data Driven Extraction for Science* (2018)
and *Chemometrics for Pattern Recognition* — the discussions I can
verify focus on classical iid CV (LOO, k-fold, bootstrapping, test
set / training set splits). Group-aware CV is not the central
recommendation. **The default Brereton recipe for transferability is
the external test set.**

I could not get a verbatim quote without paywall access, so this is a
soft signal not a hard citation.

### 4e. Workman 2018 — calibration transfer review (Appl. Spectrosc.)

**Workman, J. J. (2018).** *A Review of Calibration Transfer Practices
and Instrument Differences in Spectroscopy.* Applied Spectroscopy
72(3):340–365.

The structural assumption throughout: **master/parent instrument →
child/secondary instrument with paired samples + transfer math (DS, PDS,
canonical correlation, FIR filtering)**. The validation construct is
"RMSEP on the child instrument's test set" — i.e. *external test set
validation specifically scoped to the transfer problem*, not LOGO over
mixed-instrument calibration data.

[Sage DOI](https://journals.sagepub.com/doi/full/10.1177/0003702817736064)

Reading: in pure calibration-transfer literature, the canonical proof
is "build on instrument A, test on instrument B's holdout set" — an
external test set with explicit group provenance, not LOGO over a
pooled dataset.

### 4f. Nygård / Soneson / PLOS One 2014 — batch confounding paper

**(authorship via PMC)** Soneson, C.; Gerster, S.; Delorenzi, M.
(2014). *Batch Effect Confounding Leads to Strong Bias in Performance
Estimates Obtained by Cross-Validation.* PLOS One 9(6):e100335.

Key finding (verified via WebFetch summary):

> "When there is no confounding between the group and the batch, the
> cross-validation estimate is almost unbiased ... as the degree of
> confounding increases, the cross-validation performance estimate
> becomes increasingly over-optimistic."

> "Batch effect removal methods should not be trusted blindly as a
> 'post-experimental' way of rescuing a badly designed experiment."

Their recommendation is **prevent confounding at experimental design
time**, not "use LOGO." The paper focuses on the omics/microarray
genre but the math generalizes.

This is a citation in *favor* of group-aware methods being a problem
domain — but the recommendation is design, not LOGO. LOGO is one
defensive measure when design is already done.

[DOI 10.1371/journal.pone.0100335](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0100335)

### 4g. Sponheimer et al. (the user's paper) — explicit LOGO defense

**`paper_short_rewrite4.md` §2.6** —

> "Within-dataset analyses used five-fold stratified cross-validation
> (random_state = 42; Supplementary Table S1). Cross-dataset analyses
> used leave-one-group-out by site, with each site held out in turn.
> **We chose LOGO over a random K-fold split for the cross-dataset
> analysis because random splits would let the model train on
> neighbors of the held-out specimens and inflate the apparent
> transfer; LOGO simulates the deployment case in which an
> investigator screens a new collection without help from training
> specimens at the same site.**"

This is **the user's own published rationale for LOGO**, in the user's
own bone-FTIR domain. The framing matches the chemometrics-literature
spirit (transferability needs group structure preserved) without
citing a single paper that explicitly mandates LOGO — because there
isn't one canonical mandate. The user's paper is making the
*scientifically defensible argument*, and the choice of LOGO follows.

### 4h. Field-alignment summary

The chemometrics literature is **split** but for *transferability
specifically* leans toward:

1. **External test sets** (Workman 2018 calibration transfer review;
   Westad & Marini 2015 for n > 50; Brereton's textbook recipe). This is
   the *most-frequently-recommended* canonical solution.
2. **LOGO / segmented CV / group-aware splitting** when external test
   sets aren't available and within-group structure exists (the user's
   paper, Soneson 2014's negative case study, PLS_Toolbox's Contiguous
   Blocks default for site-organized data).
3. **rdCV / repeated CV** for bias correction on iid samples
   (Filzmoser 2009, Krstajic 2014). **Not for transferability.**

So when the user asks "Is LOGO actually how the chemometrics literature
validates transferability?" the answer is:

- Not exclusively. External test sets are at least as canonical, often
  more so.
- LOGO is a recognized fallback / honest-CV variant for transferability
  when external test sets aren't feasible.
- "Standard k-fold CV inflates transferability claims" — yes, this is
  documented (Soneson 2014, the user's own paper §2.6) but the
  recommended remedy is "external test set" first, "group-aware CV"
  second, not LOGO as the only canonical answer.

---

## 5. Practical viability — the user's variable-group-composition concern (Step 5)

### 5a. The user's own paper's site-size distribution

From `paper/results/cross_dataset_unified_min5.meta.json` (the v4
headline analysis after the ≥5 specimen filter):

| Site | N |
|------|---|
| MEC | 100 |
| Greece_Kotzia_Square | 31 |
| MAN | 26 |
| SAR | 21 |
| KAS | 19 |
| Greece_Kerameikos | 17 |
| MAR | 15 |
| DEN | 9 |
| Madagascar_Andranosoa | 9 |
| Madagascar_Andolonomby | 7 |
| Monaco_Saint-Martin-Bassin_Tortues | 7 |
| Greece_Laurion | 6 |
| Madagascar_Ampoza | 6 |
| Madagascar_Itampolove_Tampolove_-_TAMP | 6 |
| Madagascar_Lamboharana_Lamboara | 5 |

**N per site ranges from 5 to 100, a 20× ratio.** The largest site (MEC)
contains 35% of all specimens. **This is exactly the imbalanced regime
the user is asking about.** Mean N = 19; median = 9; the distribution is
heavily right-skewed.

Pre-filter (≥1 protocol, 37 sites, n=317), the imbalance is even more
extreme: 17 of 37 sites have N=1 (single-specimen sites that are
nonsensical for LOGO — leaving one specimen out is leaving a single
sample out, not a site).

### 5b. Does LOGO work in this regime?

**Mechanically:** Yes — the user's paper ran it and got publishable
numbers. MCC = 0.636 (LDA on PLS scores) headline, 95% CI excludes zero
on the gap-vs-baseline metric.

**Statistically reliable in the way the user worries about:** No, not
without caveats. Three documented failure modes:

1. **Single-site dominance.** When MEC is held out, the training set
   loses 35% of specimens (100 of 284). The fold-level metric for the
   MEC-held-out fold is essentially "model trained on 65% of the data
   that lacks the dominant site" — it's not measuring transferability,
   it's measuring "small-data robustness." When MEC is in the training
   set (14 of 15 folds), the test-fold metric is dominated by
   smaller-N sites. **The pooled LOGO metric is not measuring the same
   thing across folds.**

2. **Single-chemistry-mode loss.** From the archive notes (`archive/
   notes/cross_dataset_report.md:160`): *"PC sites generally do better
   (F1 = 0.80) than K sites (F1 = 0.56). The K sites that fail (MAR,
   THA) are dominated by severely degraded bone; when they are held
   out, the training set loses most of its examples of extreme
   degradation from the K portion of the data."* This is the user's
   "complicated to do well in practice" concern, empirically
   demonstrated in the user's own data.

3. **Confounded operator/instrument/sample-prep effects.** Kontopoulos
   and Hixon were collected on different instruments, by different
   operators, with different sample-prep protocols. LOGO by site
   doesn't disentangle "the model can't transfer to new sites" from
   "the model can't transfer to a new operator + instrument +
   protocol." The user's paper §2.6 acknowledges this explicitly:
   *"Kontopoulos and Hixon were collected on different instruments at
   different institutions ... a model that does well on one is not
   guaranteed to do well on the other."*

### 5c. Statistical literature on LOGO / unequal-group-size reliability

[Distributional bias in LOO CV (Science Advances 2024)](https://www.science.org/doi/10.1126/sciadv.adx6976):

> "Distributional bias has a statistically significant impact across
> different sample sizes (20 to 100) and feature spaces. ... while the
> size of the validation set affects the magnitude of the bias, one
> cannot make the bias vanish by simply changing the size of the
> validation set."

[Leave-one-out cross-validation, penalization, and differential bias
(PMC 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10152625/):

> "Leave-one-out cross-validated c-statistics can be strongly biased
> towards zero, and this bias is even more pronounced for model
> estimators shrinking estimated probabilities towards the observed
> event fraction."

The math doesn't say "don't use LOGO." It says "LOGO with small/
imbalanced groups produces high-variance, potentially-biased point
estimates." **The remedy is uncertainty quantification** (block
bootstrap CI, confidence intervals on the fold-pooled metric — exactly
what T-16 covers), not avoiding LOGO.

### 5d. The site-size sensitivity test in the user's paper

`paper_short_rewrite4_supplementary.md` §S7 compares ≥1 vs ≥5
filtering protocols and concludes:

> "The ≥5 site-size filter (used in the headline) reduces noise from
> one- and two-specimen sites at the cost of dropping ~33 specimens.
> The ≥1 protocol shows the same qualitative pattern with slightly
> lower numbers, confirming the headline is not an artifact of the
> site-size choice."

So the user's own pipeline already demonstrates that **LOGO IS
sensitive to the group-size cutoff**, but the qualitative ranking is
robust. The cutoff (≥5) is a hyperparameter the analyst has to think
about. dasp's GUI exposing LOGO would have to surface this hyperparameter
("Minimum specimens per group") — not just a binary "use LOGO" toggle.

### 5e. Practical viability conclusion

**LOGO is viable but fragile in the user's regime.** The user's "complicated
to do well in practice" intuition is correct. The mitigations the
literature recommends:

1. **Pair with site-level block bootstrap (T-16).** The user's own paper
   does this — 1000 iterations resampling at site level to put a CI
   around the LOGO point estimate. Without T-16, LOGO ships a single
   number that's high-variance and potentially misleading.
2. **Filter to a minimum group size before LOGO runs.** The user's paper
   uses ≥5 as a default; dasp's GUI would need to expose this as a
   user-controlled parameter.
3. **Run a sensitivity analysis comparing protocols (≥1, ≥3, ≥5).**
   The user's S7 table is the format. dasp could either auto-run multiple
   thresholds or the user does it manually with multiple runs.
4. **Surface single-site-dominance warnings.** When one site contributes
   >25% of the data (MEC in this case), that site's hold-out fold
   essentially isn't measuring transferability. dasp could compute the
   imbalance ratio at run-start and warn.
5. **Provide an external-test-set workflow as an alternative** (T-01
   reframe). For users who have a clean held-out site-stratified test
   set, that's the canonical-er solution per Westad & Marini /
   Workman.

---

## 6. Distribution-model + scope estimate (Step 6)

### 6a. If T-15 is delivered

GUI work (~2–3 days):

- Add `'group_kfold'` and `'leave_one_group_out'` to the CV strategy
  combobox (`gui:11300-11304`) plus the refinement-tab clone
  (`gui:15184-15186`).
- Add a conditional "Group column:" combobox below CV strategy when
  one of the group strategies is selected. List metadata categorical
  columns (reuse Analysis Subset V1 metadata-loading infrastructure).
- Add a "Minimum specimens per group:" spinbox (default 5, range 2–N,
  with explanatory tooltip).
- Add an info banner / dialog when the imbalance ratio exceeds 5×
  (e.g., "Largest group N=100, smallest N=5 — LOGO with this
  imbalance produces high-variance per-fold estimates. Consider
  increasing the minimum-group-size filter or pairing with bootstrap
  CIs (T-16).").
- Update the cost-estimator dialog to handle variable-fold-count
  strategies (LOGO has N folds = N groups, not user-controlled).
- Update tooltip strings to explain LOGO/GroupKFold semantics
  (probably 1-2 paragraphs each).

Backend work (~5–7 days):

- `cv_utils.build_cv_splitter` — new branches for `'group_kfold'`
  (GroupKFold(n_splits=...)) and `'leave_one_group_out'`
  (LeaveOneGroupOut()). Both need a `groups=` parameter.
- `cv_utils.compute_min_train_fold_size` — replace the
  NotImplementedError at line 222 with actual group-aware sizing.
  For LOGO: min train fold = total - max(group_size). For GroupKFold:
  same logic across the K-most-balanced grouping.
- `cv_utils.validate_cv_strategy_for_task` — add group-validity checks
  (groups vector length matches n; min group size; n_groups >=
  n_folds for GroupKFold).
- `cv_utils.cross_validate_with_early_stopping`, all its peers —
  thread `groups` through to `cv.split(X, y, groups=groups)`.
- `search.py:run_search`, `run_one_class_search` — accept `groups`
  parameter, propagate.
- `bayesian_utils.py`, `unified_bayesian.py` — same.
- `contamination.py:run_one_class_cv` — same.
- `preprocessing_discovery.py:_quick_evaluate` — same.
- New tests: `tests/test_cv_strategy_groups.py` covering group-aware
  splitter behavior, minimum group size validation, sample weight
  alignment under group splits, downstream propagation.

Documentation:

- Tooltips, help text, methods description in code-export templates.
- Update `docs/PROJECT_STATUS.md`, `docs/SESSION_LOG.md`.

**Total estimate: 1.5–2 weeks** for a single ticket scope. The roadmap
estimate of 3–5 days was probably optimistic.

### 6b. T-15 alone vs. T-15 + T-16 paired

Per the practical-viability analysis (§5), **T-15 without T-16 ships
LOGO point estimates that are high-variance in the user's regime**.
The user's own paper pairs them. Shipping only T-15 ships an
incomplete tool that delivers LOGO numbers without the uncertainty
quantification that makes them publishable.

If scope allows, **T-15 + T-16 together is the right unit of work** —
3–4 weeks combined, but ships a coherent transferability+inference
workflow rather than a half-feature.

### 6c. Alternative: T-15 reframe to external-test-set workflow

If the verdict-writer concludes the field default is external test
sets (per Westad & Marini 2015 + Workman 2018), T-15 could be reframed
as: "Add an external-test-set workflow with site-stratified
calibration/test split + RMSEP/MCC reporting on the held-out site
group, plus optional N-random-split sensitivity (random_state varies)
inside the calibration set."

This would deliver Workman's canonical "build on parent, test on
child" pattern directly. It would NOT cover the user's bone-FTIR
multi-site pooled-LOGO use case (Sponheimer's paper §2.6) — that
still needs LOGO + block bootstrap for the cross-dataset
chemometric panel.

So **T-15 reframe to test set is not a substitute** for the LOGO need;
it's a *complementary* feature for the calibration-transfer use case.
Probably both are needed; they serve different scientific questions.

---

## 7. Notes / uncertainties for the verdict-writer

1. **Competitor parity is real but partial.** Only PLS_Toolbox surfaces
   group-aware CV explicitly (via Custom vector / Contiguous Blocks).
   OPUS doesn't have it. SIMCA / Unscrambler are ambiguous in public
   docs. dasp adding it puts dasp on parity with PLS_Toolbox's most
   advanced capability.

2. **Field-alignment is mixed.** The chemometrics literature does not
   have a single canonical LOGO recommendation. Westad & Marini 2015
   and Workman 2018 lean toward external test sets. Soneson 2014
   confirms the underlying bias problem. The user's own paper uses
   LOGO and defends the choice on transferability grounds. **There is
   no single citation that mandates LOGO over external test sets.**

3. **The user's variable-group-composition concern is empirically
   correct.** Site sizes range from 5 to 100 (20× ratio) in the user's
   own bone-FTIR data. This produces high-variance LOGO point
   estimates that need uncertainty quantification (T-16) to be
   defensible.

4. **LOGO without T-16 ships an incomplete tool.** The user's paper
   pairs them; the literature recommends pairing them; statistical
   reasoning requires CI quantification when groups are unequal.
   T-15-alone is a footgun.

5. **External test sets (T-01 reframe) cover a different but
   complementary use case.** For calibration transfer in the Workman
   2018 sense (master/child instruments with a paired holdout), the
   external-test-set workflow is more canonical. For multi-site pooled
   transferability claims (the user's paper), LOGO + block bootstrap is
   what the user actually needs. Both are valid; they serve different
   scientific questions.

6. **Distribution-model: GUI-required.** Per the T-26 lesson, a
   backend-only T-15 is dead code for dasp's bundled-app users. The
   GUI surface (CV strategy combobox + group column dropdown +
   minimum-group-size spinbox + imbalance warning) is required for
   T-15 to deliver value.

7. **Validation gate filter check** — running T-15 through the five
   recurring false-alarm patterns:

   | Pattern | Applies to T-15? |
   |---------|------------------|
   | Sklearn-instinct false alarm | **No.** sklearn ships GroupKFold/LOGO; this is iso-canonical with sklearn. The chemometrics literature has segmented CV / Custom-vector CV that's mathematically equivalent. |
   | Defensive code in unreachable branch | **No.** The code doesn't exist yet; the question is whether to add it. |
   | Display-economy / code-style | **No.** Adding LOGO changes which folds are used, not just what's displayed. |
   | dasp already matches the leading program | **Partial.** PLS_Toolbox offers Custom-vector CV; dasp doesn't. Adding T-15 would close the gap. |
   | Real finding can warrant zero action | **Possible verdict path.** The verdict-writer could conclude "LOGO is real but the canonical solution is external test sets (T-01 reframe), so route effort there instead." That's defensible per §4 + Workman 2018. |

8. **Scope decision needed.**
   - **Narrow (T-15 alone):** LOGO/GroupKFold splitters, group-column
     dropdown, minimum-group-size spinbox, propagation through all
     per-fold paths. ~1.5–2 weeks. Ships a half-feature without
     uncertainty quantification.
   - **Paired (T-15 + T-16):** Above plus site-level block bootstrap
     and paired permutation tests. ~3–4 weeks. Ships a coherent
     publication-defensible transferability workflow.
   - **Reframe to external test set:** Site-stratified calibration/
     test split + RMSEP on holdout. ~1 week. Doesn't cover the user's
     paper §2.6 multi-site LOGO use case.
   - **Hybrid:** Paired (T-15 + T-16) plus the external-test-set
     workflow as a separate Analysis tab option. ~4–5 weeks. Most
     comprehensive; covers both transferability frameworks.

9. **The roadmap-estimate underspecified the scope.** The 3–5 day
   estimate in `RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` doesn't
   account for the GUI imbalance-warning UX, the cost-estimator
   updates, the propagation through Bayesian/contamination paths, or
   the test-coverage-for-each-path work. Realistic estimate is closer
   to 1.5–2 weeks for T-15 alone, 3–4 weeks for T-15 + T-16.

10. **Real finding warrants zero action — possible but unlikely path
    here.** Unlike T-26, dasp main does NOT match any leading program's
    behavior on this. PLS_Toolbox has Custom-vector CV that covers the
    use case; dasp has no equivalent. So T-26's "do nothing because
    we already match the field" verdict doesn't apply. The closest
    "zero action" path is "the field default is external test sets,
    not LOGO; route effort to T-01 reframe instead" — but that
    abandons the user's own paper's LOGO workflow, which is the
    primary use case driving T-15.

---

## Sources

- [scikit-learn LeaveOneGroupOut docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)
- [scikit-learn GroupKFold docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Eigenvector — Using Cross-Validation in PLS_Toolbox / Solo](https://www.eigenvectordocs.com/index.php?title=Using_Cross-Validation)
- [Eigenvector — Custom CV vector FAQ](https://www.eigenvectordocs.com/index.php?title=Faq_use_custom_cross_validation_option)
- [Eigenvector — Crossval function reference](https://www.eigenvectordocs.com/index.php?title=Crossval)
- [Westad & Marini 2015 — Validation of chemometric models, a tutorial — ACA 893:14–24](https://pubmed.ncbi.nlm.nih.gov/26398418/)
- [Filzmoser, Liebmann, Varmuza 2009 — Repeated double cross validation — J. Chemom. 23(4):160–171](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.1225)
- [Krstajic et al. 2014 — Cross-validation pitfalls — J. Cheminformatics 6:10](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10)
- [Workman 2018 — Calibration Transfer Practices and Instrument Differences — Appl. Spectrosc. 72(3):340–365](https://journals.sagepub.com/doi/full/10.1177/0003702817736064)
- [Soneson, Gerster, Delorenzi 2014 — Batch Effect Confounding — PLOS One 9(6):e100335](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0100335)
- [Distributional bias compromises LOO CV — Science Advances 2024](https://www.science.org/doi/10.1126/sciadv.adx6976)
- [Leave-one-out cross-validation differential bias — PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10152625/)
- [SIMCA 15 User Guide — Sartorius PDF](https://www.sartorius.com/download/544940/simca-15-user-guide-en-b-00076-sartorius-data.pdf)
- [mdatools simca() reference — Kucheryavskiy](https://rdrr.io/cran/mdatools/man/simca.html)
- [mdatools crossval.simca reference](https://rdrr.io/cran/mdatools/man/crossval.simca.html)
- [OPUS QUANT manual — Bruker PDF](https://files.nocnt.ru/hardware/science/senterra/opus65-doc-en/quant.pdf)
- [Bruker — OPUS Quantification page](https://www.bruker.com/en/products-and-solutions/infrared-and-raman/opus-spectroscopy-software/quantification.html)
- [CAMO Unscrambler Tutorials](https://www.yumpu.com/en/document/view/7708386/the-unscrambler-tutorials-camo)
- [CAMO blog — Category variables](http://dataanalysis.blog.camo.com/index.php/user-tip-create-category-variables/)
- [Sponheimer et al. (in prep) — `paper/manuscript/paper_short_rewrite4.md` §2.6 — LOGO defense](file://C:/Users/mspon/Desktop/_DeskSync/FTIR%20Bone%20PLS/paper/manuscript/paper_short_rewrite4.md)
- [Sponheimer et al. (in prep) — `paper/results/cross_dataset_unified_min5.meta.json` — site-size distribution](file://C:/Users/mspon/Desktop/_DeskSync/FTIR%20Bone%20PLS/paper/results/cross_dataset_unified_min5.meta.json)
- [dasp `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` §1 — LOGO gap analysis](file://C:/Users/mspon/git/dasp/docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md)
