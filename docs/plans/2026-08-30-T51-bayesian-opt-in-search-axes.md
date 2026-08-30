# T-51 — Opt-in Bayesian search-space axes

> **STATUS: ticket drafted 2026-08-30, not started. Picking up tomorrow.**
>
> **First action next session:** copy this content to
> `docs/plans/2026-08-30-T51-bayesian-opt-in-search-axes.md` (repo convention
> `docs/plans/YYYY-MM-DD-T<NN>-<slug>.md`; highest existing is T-50, so T-51), match the
> structure of `2026-05-01-T41-bayesian-sqlite-auto-calculator.md`, and add a T-51 line
> to `docs/PROJECT_STATUS.md`. It must live in the repo, not in a local plan file —
> per CLAUDE.md's Session Protocol, project knowledge is git-tracked so other machines
> get it on pull.
>
> No code has been written. Nothing has been committed. Everything below is design.

**One-line summary:** supervised Bayesian works well; add opt-in knobs for the
hyperparameters that currently take exactly one value, curated per model family,
defaults untouched.

## Context

A downstream contamination/one-class project driving DASP from Python reported that
DASP's Bayesian search "isn't exploring much of the potential space", and had to
monkey-patch `unified_bayesian.suggest_model_params` in memory to widen it
(`HANDOFF_2026-08-29_DASP_SEARCH_SPACE.md`).

Investigation confirmed the underlying facts but **not** the handoff's headline framing.

**The premise of this ticket is added value, not a defect.** Supervised Bayesian search
performs well today and nothing here assumes otherwise. Opening every axis at once was
measured downstream to give only a small improvement, which is consistent with the
current configuration being good. What the ticket adds is a few more knobs, available
when a user has a specific reason to reach for them.

- **Defaults stay bit-identical.** Nothing changes for a user who touches nothing.
- **The target is hyperparameters that currently take exactly one value** and can never
  be tried at any other. Not the ranges already being searched.
- **Curated per model family** — what is worth exposing differs by family (PLS should
  search latent variables and nothing else; SVM's `gamma` is worth making reachable).
- **GUI is the primary audience**, but the Python primitives path used by agents must be
  served by the same mechanism, not a parallel one.

Ranges that are already searched are **left alone**. They would only be revisited on a
strong, specific argument that a current range is wrong, and this ticket makes no such
argument.

### Review history

- **Codex (gpt-5.5, medium, full repo access).** Two factual corrections incorporated
  (`PREPROCESSING_OPTIONS` is 14 not 17; supervised PLS already fingerprints post-clamp).
  Its `enabled_extra_axes` bundle design was adopted over the raw-dict API first sketched.
  *(`gpt-5.6` is not reachable on a ChatGPT-account Codex login — it hard-errors with
  "not supported when using Codex with a ChatGPT account". `gpt-5.5` is the only reachable
  model on that auth mode.)*
- **Peer panel (DeepSeek V4 Pro + GLM-5.1; Kimi K2.6 failed to return), no repo access.**
  Two blocking findings, both accepted: the duplicate-`suggest_*` constraint that killed
  `linear_alpha_wide` and reshaped the bundle menu, and the observation that non-bundle
  behaviour changes are unprotected by the conditional hash. Their proposed fix for the
  latter was superseded — `_dasp_version` is already in the hash, which neither could see.
  Also accepted: NeuralBoosted must be a bundle not a base change; `n_startup_trials` must
  be exposed; `search_space`/`enabled_extra_axes` precedence must be specified; bundle
  axes must be suggested uniformly; the `gamma='scale'`-is-a-defect framing is too strong;
  the `min_child_samples` "unlocks num_leaves" claim was backwards.

### Where the opportunity is

Each trial already searches ~9–12 dimensions jointly: preprocessing (`preprocessing`,
14 categorical; `savgol_window`), variable selection (`subset_type`, `n_vars`,
`region_id`), and ~4 model hyperparameters. At `n_trials=300` with
`n_startup_trials=20` that is ~25 trials per dimension — a reasonable budget, and it is
working.

The opportunity is narrower and specific: **a number of model hyperparameters are fixed
at exactly one value in `suggest_model_params` and can never be tried at any other.**
LightGBM's `reg_alpha=0.1`, `subsample=0.8`, `min_child_samples=5`; XGBoost's
`colsample_bytree=0.8` with `gamma` and `min_child_weight` absent entirely;
RandomForest's `max_features='sqrt'`; SVM's `gamma='scale'`; PLS-DA's logistic head
sitting at `C=1.0`. Several of these are also not the underlying library's own defaults,
so a user cannot infer them without reading DASP's source.

No claim is made that these values are wrong — several are well chosen for spectra. The
claim is only that a user with a reason to explore one currently has no way to.

**A useful coincidence:** because Optuna refuses a second `suggest_*` on the same
parameter name, the additive design in this ticket can *only* open axes that are pinned
to a constant — widening an already-searched range is not additively possible at all. The
technical constraint and the intended scope are therefore the same set. That is why this
design stays safe without needing discipline to hold it back.

---

## PRIME DIRECTIVE — do not disturb the working path

**Supervised Bayesian works well today and must not regress.** This ticket adds function
for cases where it is appropriate; it does not improve, tidy, or restructure what already
works. Any change that cannot be shown to leave the default path untouched is out of
scope, however attractive.

Concretely, this rules out the obvious implementation. Rewriting `suggest_model_params`
into a table-driven executor would rewrite the exact code that currently works, and
"equivalence tests pass" is a weaker guarantee than "the code did not change".

**Therefore: the default path is not refactored.**

- `suggest_model_params` (`:652-812`) and `suggest_one_class_params` (`:815-863`) stay
  **byte-for-byte as they are**, including every pinned constant and both clamps.
- Extra axes are applied by a **separate, additive** step that runs *after* them:

  ```python
  params = suggest_model_params(trial, model_name, n_features, task_type)
  params = apply_extra_axes(trial, model_name, params, enabled_extra_axes, ctx)
  ```

- With `enabled_extra_axes=()` — the default, and what every existing GUI user gets —
  `apply_extra_axes` returns `params` unchanged without issuing a single
  `trial.suggest_*` call. The executing code path is identical to today's.

The declarative table still gets written, but as a **descriptive** artifact only: it
feeds the GUI checkbox menu, the documentation of what the defaults actually are, and
the study-identity hash. It is never the thing that executes the default space. That
keeps the documentation benefit (pinned constants become discoverable) at effectively
zero risk to a working search.

## Design

### 1. Extra-axis bundles (`search_spaces.py`, new module)

The table describes bundles and documents defaults. A plain dict-of-ranges cannot express
conditional or clamped axes, so use typed specs:

```python
@dataclass(frozen=True)
class AxisSpec:
    name: str
    kind: Literal["int", "float", "categorical", "constant", "derived"]
    default: Any
    suggest: Callable[[Trial, SearchContext], Any] | None = None
    resolve: Callable[[Any, SearchContext], Any] | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ModelSpace:
    base: tuple[AxisSpec, ...]
    extras: Mapping[str, tuple[AxisSpec, ...]]   # bundle id -> axes
```

- `constant` records today's pinned values (`reg_alpha=0.1`, `subsample=0.8`,
  `min_child_samples=5`) as **documentation** — answering the handoff's "the defaults
  are opinions and should be documented as such" — without those entries driving
  execution.
- `derived` + `suggest` lets a bundle axis depend on an already-sampled value.
- `resolve` lets a bundle axis be clamped by data (`SearchContext`).
- `SearchContext` carries `model_name, task_type, n_samples, n_features_final,
  cv_strategy, cv_folds, already-sampled params`.

**Requirement:** `apply_extra_axes` with an empty bundle set is a no-op — it must not
call `trial.suggest_*` at all, so Optuna's parameter space for a default run is
unchanged and existing persisted studies stay resumable.

### HARD CONSTRAINT — additive bundles can only open *pinned* axes

Optuna raises if the same parameter name is suggested twice in one trial with a different
distribution. So `apply_extra_axes` can only add axes the base sampler leaves as
**constants**. It cannot widen an axis the base already suggests. Auditing the base:

| Safe — base pins a constant, bundle may open it | Blocked — base already suggests it |
|---|---|
| XGBoost `reg_alpha`, `reg_lambda`, `colsample_bytree` (`:765-767`); `gamma`, `min_child_weight` (absent entirely) | Ridge/Lasso/ElasticNet `alpha` (`:698`, `:706`, `:713`) |
| LightGBM `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` (`:748-753`) | MLP `alpha` (`:804`) |
| RandomForest `max_features='sqrt'` (`:724`) | OneClassSVM `gamma` — `suggest_categorical(['scale','auto'])` (`:843`) |
| SVM/SVR `gamma='scale'` — assigned, not suggested (`:788`) | PCA-SIMCA `n_components` (`:837`) |
| CatBoost `subsample`, `rsm` (absent) | |
| PLS-DA logistic head `C` (absent) | |
| LOF `metric`, IsolationForest `max_samples` (absent) | |

**Consequence:** `linear_alpha_wide`, `mlp_activation`'s wider alpha, and
`ocsvm_gamma_float` as originally written are **not implementable additively**. Options
per case: (a) drop them from v1; (b) suggest under an alias name (`gamma_log`) and
`resolve` it onto the real key, accepting a second parameter in `trial.params`; or
(c) accept a base-sampler edit for that family under its own approval gate. **Default to
(a) for supervised** — the prime directive outranks completeness. One-class may use (c).

Add a test that enables every bundle against a mock trial and asserts no duplicate-name
`suggest_*`.

**Why this is TPE-safe.** Bundles are fixed for the whole run, so every trial in a given
study suggests the same parameter names — no *within-study* dynamic space is introduced.
That matters because TPE with `multivariate=True` (`:2123`, `:2499`) builds its KDE only
from trials containing all parameters of the current space. The codebase already has one
genuine dynamic space here: LightGBM's `num_leaves` is set **without** a `suggest_*` call
when `max_valid < 15` (`:738-739`), so some trials legitimately lack that parameter —
which is what `warn_independent_sampling=False` (`:2125`, `:2501`) exists to silence.
Bundles add no new instance of that pattern. Cross-study mixing is prevented by the
identity hash below.

### 2. Curated per-family bundles

All bundles default **off**. Menu is fixed in code, not user-invented.

| Family | Bundle id | Opens |
|---|---|---|
| PLS (regression) | — | *nothing.* Selecting LVs by CV with all else fixed is the chemometric standard. Deliberate. |
| PLS-DA | `plsda_head` | inner LogisticRegression `C` (log 1e-3–1e3) — currently pinned at 1.0 in every PLS-DA candidate |
| Ridge/Lasso/ElasticNet | ~~`linear_alpha_wide`~~ | **dropped from v1** — `alpha` is already suggested, so it cannot be widened additively |
| RandomForest | `rf_features` | `max_features` (sqrt/log2/0.1/0.3/0.5/1.0) — currently pinned `'sqrt'` |
| XGBoost | `xgb_regularization` | `reg_alpha`, `reg_lambda` (log) |
| | `xgb_child` | `min_child_weight`, `gamma` |
| | `xgb_sampling` | `colsample_bytree`, `colsample_bylevel` |
| LightGBM | `lgbm_regularization` | `reg_alpha`, `reg_lambda` |
| | `lgbm_sampling` | `subsample`, `colsample_bytree` |
| | `lgbm_child` | `min_child_samples` (2–50), pinned at 5. Note the direction: *lowering* it makes more leaves reachable, raising it fewer. |
| CatBoost | `catboost_sampling` | `subsample`, `rsm` |
| SVM/SVR | `svm_gamma` | `gamma` as log float instead of the pinned `'scale'` |
| | `svm_kernels` | `poly`/`sigmoid` + `degree`, `coef0` — **suggest `degree`/`coef0` on every trial** even when the kernel ignores them, so the parameter-name set stays uniform (see below) |
| MLP | `mlp_activation` | `activation` only — the wider `alpha` is dropped (already suggested) |
| PCA-SIMCA | `simca_ncomp_wide` | raise the `n_components` ceiling above the data-derived default — requires the base-sampler edit, see one-class section |
| OneClassSVM | `ocsvm_gamma_float` | `gamma` as log float. Base already suggests `gamma` categorically, so this needs an alias or a base edit — one-class may take the base edit |
| | `ocsvm_poly` | `degree`, `coef0` — same uniform-suggestion rule as `svm_kernels` |
| IsolationForest | `if_max_samples` | `max_samples` |
| LOF | `lof_metric` | `euclidean`/`manhattan`/`minkowski`/`cosine` |

One-class is in the table from the start (user decision) — it is currently a second
bespoke code path and this is the opportunity to stop that. Note `lof_metric` and
`if_max_samples` already exist in the **grid** path (PR #58 round-2), so these bundles
bring Bayesian to parity rather than inventing anything.

**Uniform parameter names within a bundle.** A bundle must issue the *same*
`trial.suggest_*` calls on every trial — never conditionally, e.g. "only suggest `degree`
when `kernel == 'poly'`". Conditional suggestion splits TPE's multivariate KDE into
subgroups with different parameter sets, each trained on a fraction of the trials. Suggest
all of a bundle's axes always and let the model ignore the inapplicable ones.

`NeuralBoosted` currently falls through to `return {}` (`:810-812`) and is tuned on
nothing. **Expose it as an opt-in bundle (`neuralboosted_base`), not as a base change** —
`bayesian_config.py:458` has a usable space to lift. Making it a base change would alter
default results for anyone who selects that model, which the prime directive forbids.

### 3. API surface

```python
run_unified_bayesian(
    ...,
    enabled_extra_axes: Sequence[str] = (),   # bundle ids, e.g. ("svm_gamma",)
    search_space: ModelSpace | None = None,   # advanced full override
    n_startup_trials: int | None = None,      # None -> today's 20
)
```

**Precedence (must be specified, not left implicit):** when `search_space` is given it
*replaces* the default `ModelSpace` for that family, and `enabled_extra_axes` then selects
bundles from the supplied space. When it is `None`, bundles are selected from the curated
default space. Test both paths.

**`n_startup_trials` must be exposed**, or the GUI's advice is unactionable: the GUI is
specified below to suggest `max(20, 3 × effective_dim)`, but the sampler currently
hardcodes 20 at both construction sites (`:2121`, `:2497`). Default `None` → 20, so
default runs are unchanged. Thread it to both sites.

GUI checkboxes emit the **same bundle ids**, so GUI and script callers drive one
mechanism with one set of documentation and tests. Thread through
`create_unified_objective` (`:991-1018`, wired at `:2465-2492`) as well.

Add `run_unified_bayesian` bundle support to the declared stable surface in
`docs/AGENT_COMPOSITION.md` (it is already listed there and contract-tested in
`tests/test_agent_composition_api.py`).

### 4. Study identity — blocking correctness item

`unified_bayesian.py:2553-2582` builds `config_hash` → `study_name`, used with
`load_if_exists=True` for SQLite persistence. **A canonical serialization of the
effective space must enter that hash** — schema version, enabled bundle ids, resolved
distributions, and a resolver version. Without it, resuming a persisted study silently
mixes trials sampled from different spaces.

**Per the prime directive, the hash must be unchanged when no bundles are enabled.**
Appending even an empty `space=...` segment to `config_components` would change every
existing study name and orphan every persisted study on disk. So the segment is appended
**only** when `enabled_extra_axes` is non-empty or `search_space` is given. Default runs
keep today's hash exactly. Add a test pinning that default `enabled_extra_axes=()`
produces byte-for-byte today's `study_name`.

**Canonicalise before hashing:** sort and de-duplicate bundle ids, so
`("svm_gamma","svm_kernels")` and `("svm_kernels","svm_gamma")` hash identically. Test
with reordered and duplicated ids.

#### The gap both reviewers found — and what actually closes it

DeepSeek and GLM independently raised the same blocker: the conditional hash protects the
*bundle* path, but PRs #1 (SVM scaler), #5 (one-class clamp) and #6 (PLS clamp) change the
effective space **with no bundles enabled**, so a persisted study resumed across those PRs
would mix trials from two different spaces. That is correct and it is the sharpest finding
of the review.

Their proposed fix — add a new unconditional schema version to `config_components` — is
**not needed**, because neither reviewer had repo access and both missed that
`config_components` already begins with `f"version={_dasp_version}|"` (`:2563`). A dasp
version bump already orphans every study.

So the requirement is narrower and cheaper: **each of PRs #1, #5 and #6 must ship with a
`__version__` bump in `src/spectral_predict/__init__.py`.** Add that to each PR's
checklist explicitly. The real hazard is not the missing mechanism but merging a
behaviour-changing PR *without* bumping — including intermediate states on a development
branch where several PRs share one version. If those PRs are to be merged before a
release bump, add a `space_behaviour=N` segment bumped per behaviour change instead.

### 5. GUI

New collapsible card in Tab 4C Model Config, following the existing one-class card
idiom (`spectral_predict_gui_optimized.py:14072`). Per-model checkboxes, all off.

When any bundle is enabled, show a live suggestion (user decision — suggest, don't
auto-apply): *"4 extra axes enabled (13 search dimensions). Consider raising Bayesian
trials from 300 to ~600."* Codex's heuristic: ~500–800 trials with extras on, or
startup trials `max(20, 3 × effective_dim)`.

Note the GUI's existing "Advanced Model Options" (~60 editable lists) feeds **only**
`run_search`; it has never affected the Bayesian path. Place the new card so that
distinction is visible, and soften `CLAUDE.md:90` ("All hyperparameters are exposed and
user-editable" — false for Bayesian today).

---

## Prerequisite: the SVM scaler bug

`SCALE_SENSITIVE_MODELS` contains `'SVC'` but the registered classifier family is
`'SVM'` (`models.py:281,498`; `model_registry.py:32`), so **no StandardScaler is
attached to classification SVM**. Sites: `search.py:156` (used `:464`, `:4962`),
`unified_bayesian.py:1534` (used `:1605`), `nsga2_search.py:1388` (omits both), and
`spectral_predict_gui_optimized.py:40481`.

This must be fixed **before** the `svm_gamma` bundle ships. `gamma` and `C` are both
scale-dependent; tuning `gamma` on unscaled 2000-column spectra tunes it against an
artefact. Both reviewers independently endorsed this ordering. `'SVC'` in the set is dead
weight — nothing ever matches it.

**Framing correction (both reviewers).** Calling the pinned `gamma='scale'` a *defect* is
too strong. Once the scaler bug is fixed, `'scale'` is sklearn's default and a defensible
heuristic for high-dimensional spectra. The honest justification for the bundle is "an
opt-in axis for users who need to tune it", not "this should never have been pinned" —
and the GUI should note that tuning `C` and `gamma` jointly on small `n_train` can overfit.

**Verification must cover more than the ticket originally said:** add an SVR test and an
NSGA-II test alongside the classification-SVM one, and confirm the registry strings are
really `'SVM'`/`'SVR'` rather than trusting this ticket. Because this PR changes default
SVM results, add an SVM-specific baseline snapshot on a fixed small dataset and re-bless
it deliberately as the intended change — the existing baseline suite pins no numbers and
will not catch it.

## Also in scope: one-class `n_components` clamp + record

`suggest_one_class_params` (`:837`) samples `n_components` 2–20 with no clamp, while
`contamination.py:132-147` clamps internally to `min(n-1, n_features)`. For a 16-row
class, requests 16–20 all fit the *same* model but are recorded as distinct points, so
TPE is told they are different. The objective declines to mirror the clamp by explicit
comment (`:1308-1312`, "missing that rare duplicate is acceptable") — the downstream
measurements are what make that judgment worth revisiting: 5 of 19 values on that axis,
not "rare".

**Correction from review (DeepSeek).** An earlier draft of this ticket claimed that
clamping *after* `suggest_int` would stop TPE treating 16–20 as distinct points. **That is
wrong.** Optuna stores whatever `trial.suggest_*` returned in `trial.params`; mutating the
value afterwards changes the fingerprint and the reported value but not what TPE learns
from. Post-suggest clamping alone does not fix the search trajectory.

Two separable pieces, therefore:

**(a) Clamp before fingerprinting + record the resolved value — additive, safe.**
Buys dedup (identical fits stop being re-fitted) and an honest `LVs` column. Does *not*
change what TPE sees. `run_one_class_cv` already returns `cal_model` and `PCASIMCA.fit`
sets `self.n_components_` (`contamination.py:147`), so the resolved value is a one-line
read with no coupling to internals.

**(b) Derive the ceiling and pass it into `suggest_int` — the only thing that actually
fixes TPE's view.** This edits the one-class sampler body, so it is an explicit exception
to the byte-for-byte rule and needs its own approval gate. It is safe within a study: the
ceiling depends on `n_samples` and the CV config, both constant for a given run, so every
trial in a study shares one distribution. It differs only across datasets, which are
different studies anyway.

The prime directive is about **supervised**; (b) touches only `suggest_one_class_params`.
Recommend (a) unconditionally and (b) with the user's explicit approval.

Precedent: `_build_fit_fingerprint`'s own docstring (`:199-266`) says it captures
"resolved fit identity, not user-suggested intent" and already excludes `subset_size`
for exactly this reason. Supervised PLS already does clamp-then-record
(`:686-689`, `:1874-1880`).

## PLS clamp asymmetry — approved, ships last and separately

**Decision (2026-08-29):** include it, as the final PR, with its own approval gate.

Bayesian clamps `n_components` to `min(20, n_features - 1)` —
features, not samples. On spectra `n_features` is thousands, so it never binds. The grid
path uses `cv_utils.compute_min_train_fold_size()` and clamps regression PLS to
`min(max_n, min_train_samples, n_features)` (`search.py:1832-1844`), which
`unified_bayesian.py` never imports. On a 30-sample dataset at 5-fold (24 training rows)
Bayesian will fit 20 latent variables; grid refuses.

This is a **correctness divergence, not a default anyone chose** — but fixing it *does*
change Bayesian PLS results on small datasets, which is the one place the "defaults stay
frozen" decision collides with a bug fix.

Fix by matching the grid path, **including its deliberate PLS-DA relaxation**
(`search.py:1841-1855` — the `if task_type == "regression": … else: …` split): regression
PLS gets the strict
`min(max_n, min_train_samples, n_features)`; PLS-DA keeps the looser feature-only bound
because the latent scores feed a logistic head that tolerates more components than
samples. Do not collapse the two.

Ships as the last PR so it can be reverted alone without unpicking anything else.

## Out of scope — recorded, not done

From the handoff, deliberately deferred: library version recording in
`model_io.py:176` (small, recommended as a companion); `cv_n_repeats` silently ignored
under `cv_strategy='kfold'` (`cv_utils.py:333-336`) while still written into
`training_config` and the study hash as if honoured; early-stopping default 40 applied
invisibly with the persisted refit not reproducing the scored model; per-family trial
budget; alpha re-grading (one fit graded at many alphas — `PCASIMCA.p_joint` is
alpha-free, so this is exact and cheap).

---

## Two kinds of work in this ticket — keep them distinct

Most of this ticket is the **enhancement** above: new optional knobs, nothing claimed to
be wrong. Two items are **genuine bug fixes** that happen to sit next to it, and they are
the only places where anything is asserted to be incorrect:

1. **The SVM scaler string mismatch** — classification SVM is fit unscaled because a set
   contains `'SVC'` while the family is named `'SVM'`. This is a defect by any reading,
   and it is a prerequisite for the `svm_gamma` knob rather than an optional extra.
2. **The PLS clamp asymmetry** — Bayesian and grid disagree about how many latent
   variables are valid on a small dataset. Approved to ship last and separately.

Everything else adds capability without asserting that current behaviour is wrong. Do not
let the bug-fix framing leak into the enhancement PRs.

## Sequencing — ordered by risk to the working supervised path

Separate commits/PRs so anything that changes behaviour can be reverted alone.

| # | Work | Risk to supervised |
|---|---|---|
| 0 | Enumerate every `run_unified_bayesian` / `create_unified_objective` caller (src, GUI, tests, docs examples, notebooks) so no call site silently ignores the new kwargs | None |
| 1 | SVM scaler fix (`'SVC'` → `'SVM'`) + `__version__` bump | **Changes supervised SVM results** — but SVM is currently fit unscaled, which is the bug. Own PR, own approval. |
| 2 | `search_spaces.py` + `apply_extra_axes` + kwargs + conditional study hash | **None.** No-op when no bundles enabled; default path byte-for-byte unchanged. |
| 3 | Bundle definitions per family (supervised, then one-class) | **None.** All off by default. |
| 4 | GUI card + trial-count suggestion | **None.** |
| 5a | One-class clamp-before-fingerprint + record resolved value | None. Fits are unchanged; buys dedup and honest reporting. |
| 5b | One-class data-derived ceiling passed into `suggest_int` (+ `__version__` bump) | None to supervised. Edits the one-class sampler — explicit exception to byte-for-byte, own approval. |
| 6 | PLS clamp asymmetry (+ `__version__` bump) | **The only item that changes default supervised Bayesian behaviour.** See below. |

**Item 6 needs its own decision.** Under the prime directive it is the one change that
alters supervised results without a bundle being switched on — it would reduce the LV
ceiling on small datasets. It is a genuine correctness divergence from the grid path, not
a chosen default, but "don't screw up what works" argues for shipping it separately, last,
and only with explicit approval. It can also be deferred indefinitely without blocking
anything else here.

## Implementation detail

### `src/spectral_predict/search_spaces.py` (new)

```python
"""Opt-in extra hyperparameter axes for the unified Bayesian search.

This module NEVER defines the default search space. Defaults live where they
always have, in unified_bayesian.suggest_model_params / suggest_one_class_params,
and are not touched. Everything here is additive and off unless a caller names a
bundle id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

AxisKind = Literal["int", "float", "categorical", "constant", "derived"]


@dataclass(frozen=True)
class SearchContext:
    model_name: str
    task_type: str
    n_samples: int
    n_features: int
    cv_strategy: str
    cv_folds: int
    params: Mapping[str, Any]          # already-sampled values this trial


@dataclass(frozen=True)
class AxisSpec:
    name: str                           # key written into model params
    kind: AxisKind
    low: Any = None
    high: Any = None
    choices: tuple[Any, ...] | None = None
    log: bool = False
    step: int | None = None
    suggest: Callable[[Any, SearchContext], Any] | None = None   # kind="derived"
    resolve: Callable[[Any, SearchContext], Any] | None = None
    param_name: str | None = None       # Optuna name if aliased; defaults to `name`


@dataclass(frozen=True)
class ModelSpace:
    documented_constants: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, tuple[AxisSpec, ...]] = field(default_factory=dict)


def apply_extra_axes(trial, model_name, params, enabled_extra_axes, ctx_factory):
    """Additively open extra axes. No-op when nothing is enabled.

    MUST return before doing any work when `enabled_extra_axes` is empty --
    including before building a SearchContext. See T-51 prime directive.
    """
    if not enabled_extra_axes:
        return params                     # zero suggest_* calls, zero work
    ...
```

Notes for the implementer:

- `documented_constants` exists so the pinned values are discoverable and can be
  rendered in docs/GUI tooltips. It is **not** consulted at sample time.
- `ctx_factory` is a callable, not a built `SearchContext`, so the no-bundle path
  never pays for constructing one (peer-review finding).
- `param_name` supports the alias route if a one-class axis ever needs to shadow a
  name the base already suggests.
- Resolve bundle ids case-sensitively against a registry and **raise on an unknown
  id** — silently ignoring a typo'd bundle would look like "the knob did nothing".

### Insertion points in `unified_bayesian.py`

Two call sites, both one added line. Nothing above or below changes.

```python
# supervised, after the existing call at :1508-1510
model_params = suggest_model_params(trial, model_name, n_features_final, task_type)
model_params = apply_extra_axes(trial, model_name, model_params,
                                enabled_extra_axes, _make_ctx)

# one-class, after the existing call at :1306
oc_params = suggest_one_class_params(trial, model_name)
oc_params = apply_extra_axes(trial, model_name, oc_params,
                             enabled_extra_axes, _make_ctx)
```

Signature threading: `run_unified_bayesian` (`:2211-2240`) →
`create_unified_objective` (`:991-1018`), wired at `:2465-2492`. Both need the three
new kwargs.

Sampler: `_make_tpe_sampler` (`:2117-2126`) and the inline construction (`:2494-2502`)
both hardcode `n_startup_trials=20` — thread the new kwarg to both, defaulting to 20.
These two sites are duplicates of each other; do not "fix" that here.

### Study identity (`:2553-2582`)

```python
if enabled_extra_axes or search_space is not None:
    _space_id = canonical_space_identity(model_name, enabled_extra_axes, search_space)
    config_components += f"|space={_space_id}"
```

`canonical_space_identity` must sort + de-duplicate bundle ids and include a resolver
schema version. Do **not** append anything when the condition is false.

### GUI wiring

Follow the existing one-class card idiom at `spectral_predict_gui_optimized.py:14072`
(`_create_collapsible_section`, BooleanVar + Checkbutton, a `_collect_*` method that
diffs widget state against defaults).

- New collapsible card in Tab 4C, e.g. "Bayesian — Extra Hyperparameter Axes",
  positioned so it is visibly distinct from "Advanced Model Options" (`:12826`), which
  feeds `run_search` only and has never affected the Bayesian path.
- One `tk.BooleanVar` per bundle id, all `value=False`.
- `_collect_enabled_extra_axes()` returns a sorted tuple of enabled ids.
- Pass at both Bayesian call sites: `:28894` (supervised) and `:28336` (one-class).
- Trial suggestion label recomputed on any checkbox toggle: count effective dimensions
  (base model axes + enabled bundle axes + the ~5 shared preprocessing/varsel axes) and
  display the advisory. Advisory only — never write `n_unified_trials` automatically.
- Add the bundle vars to `run_gui_settings.CAPTURABLE_SETTINGS` so a resumed run
  restores them. Note the existing gap: no per-model hyperparameter vars are captured
  today, so a resumed run loses the grid path's Advanced Model Options — do not
  replicate that gap for bundles.

### Task checklist

- [ ] Write `docs/plans/2026-08-30-T51-bayesian-opt-in-search-axes.md` from this content
- [ ] Enumerate every `run_unified_bayesian` / `create_unified_objective` caller
- [ ] `search_spaces.py` with `AxisSpec` / `ModelSpace` / `apply_extra_axes` / registry
- [ ] Bundle definitions, supervised (pinned-constant axes only)
- [ ] Bundle definitions, one-class
- [ ] Thread kwargs through both entry points + both sampler sites
- [ ] Conditional study-identity segment + canonical serializer
- [ ] Tests: no-op, collision, study-name pin, bundle-order canonicalisation
- [ ] GUI card + collector + call sites + advisory label + settings capture
- [ ] Docs: `AGENT_COMPOSITION.md` stable surface, soften `CLAUDE.md:90`
- [ ] Separate PR: SVM scaler fix (+ version bump, + SVM baseline snapshot)
- [ ] Separate PR: one-class clamp/record (5a), then ceiling-into-suggest (5b)
- [ ] Separate PR, last, own approval: PLS clamp asymmetry
- [ ] Per repo Session Protocol: update `docs/PROJECT_STATUS.md`, append any non-obvious
      findings to `docs/SESSION_LOG.md`, commit and push

## Files

| File | Change |
|---|---|
| `src/spectral_predict/search_spaces.py` | **new** — `AxisSpec`, `ModelSpace`, bundle registry, `apply_extra_axes()`, canonical serializer for the study hash. All new code; nothing existing moves into it. |
| `src/spectral_predict/unified_bayesian.py` | **additive only.** `suggest_model_params` / `suggest_one_class_params` bodies unchanged. Add the `apply_extra_axes` call after each (`:1508`, `:1306`); `enabled_extra_axes` + `search_space` kwargs on `run_unified_bayesian` (`:2211`) and `create_unified_objective` (`:991`); conditional space identity in `config_components` (`:2562`). Separately (own commits): one-class clamp + record; PLS clamp via `compute_min_train_fold_size` |
| `src/spectral_predict/search.py` | `SCALE_SENSITIVE_MODELS` `'SVC'` → `'SVM'` (+ `'SVR'`) |
| `src/spectral_predict/nsga2_search.py`, `spectral_predict_gui_optimized.py:40481` | same scaler fix |
| `spectral_predict_gui_optimized.py` | Tab 4C bundle card; pass `enabled_extra_axes` at the Bayesian call sites (`:28894`, `:28336`); trial-count suggestion |
| `src/spectral_predict/bayesian_config.py` | delete after lifting its NeuralBoosted space — **but first** grep for *string* references (dynamic imports, docs, notebooks), not just `import` statements, and record the evidence in the PR |
| `docs/AGENT_COMPOSITION.md`, `CLAUDE.md` | document bundles; soften the "all hyperparameters exposed" claim |

## Verification

1. **Default path untouched (the critical check).** Two parts:
   a. `git diff` on `suggest_model_params` / `suggest_one_class_params` must be **empty**.
      This is the guarantee the prime directive asks for, and it is stronger than any test.
   b. Assert `apply_extra_axes` with an empty bundle set issues **zero** `trial.suggest_*`
      calls — use a mock trial that raises on any suggest call. This is the semantically
      meaningful guarantee. Also assert it returns `params` before constructing
      `SearchContext` or doing any other work. (An `is params` identity assertion is a
      useful proxy but is fragile — a later `params = {**params}` would break it without
      breaking correctness, so don't let downstream code depend on identity.)
   c. **Bundle/base collision test:** enable every bundle in turn against a mock trial and
      assert no parameter name is suggested twice. This is what catches the class of bug
      that killed `linear_alpha_wide`.
2. `pytest tests/test_unified_bayesian_baseline.py` — shape/sanity plus same-seed
   reproducibility. Note it does **not** pin numeric values, so it cannot by itself prove
   the default space is unchanged; check 1 is what does.
3. `pytest tests/test_agent_composition_api.py tests/test_contamination_detection.py
   tests/test_simca.py tests/test_bayesian_dedup.py tests/test_cv_pls_clamp.py`
4. Study-identity test: same data + different `enabled_extra_axes` must yield different
   `study_name`; identical settings must yield the same one.
5. One-class clamp: on a class with fewer rows than the ceiling, assert requests above
   the ceiling collapse to one fingerprint and that the recorded value equals
   `cal_model.n_components_`.
6. SVM scaler: assert a classification `SVM` pipeline contains a `StandardScaler`.
7. GUI smoke: run a short Bayesian search with `svm_gamma` on and confirm the leaderboard
   `Params` column shows varying `gamma`.
