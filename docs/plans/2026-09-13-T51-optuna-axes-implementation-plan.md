# T-51 steps 2–5 — Opt-in Optuna search axes: implementation plan

> **STATUS (2026-09-14): revised after review rounds 1 and 2, and after the contamination
> project's evidence (§10). No code written.** Round 2 (`gpt-6-astra`) found one
> blocker (B0's version bump) and five other issues, all folded in (§9b).
>
> This plan turns the design ticket
> [`2026-08-30-T51-bayesian-opt-in-search-axes.md`](2026-08-30-T51-bayesian-opt-in-search-axes.md)
> into ordered, checkable work. The ticket stays the authority on *why*. This file covers
> *how*, and where the code has moved since the ticket was written. **Where the two
> disagree, this file wins.**
>
> **Out of scope here:** step 1, the SVM scaler fix (`fix/T51-svm-scaler`, `4f2b995`);
> step 6, the PLS clamp asymmetry (last PR, own approval).
>
> **Review round 1 (2026-09-14):**
> - **Reviewers:** Fable (full repo access, library probes in `.venv314`) and Codex
>   `gpt-6-astra`. `gpt-6-alpha` is rejected on a ChatGPT-account Codex login ("not
>   supported when using Codex with a ChatGPT account"); `gpt-6-astra` is the only gpt-6
>   id in the catalog.
> - **Result:** both returned "not ready as written". Both independently found the
>   `plsda_head` construction failure. Every finding below is folded in, and the decisions
>   are recorded in §9.
> - **One conflict, resolved on evidence.** On CatBoost `bootstrap_type`, Fable probed
>   binary and regression, and both fit. Codex probed multiclass, and it **fails**
>   (`default bootstrap type (bayesian) doesn't support 'subsample' option`). The failing
>   probe wins.

## 0. What changed since the ticket (verified against `main` @ `696fce9`)

Line numbers are `src/spectral_predict/unified_bayesian.py` unless another file is named.
**The ticket's numbers are stale by about 95 lines.**

| Anchor | Ticket | Now |
|---|---|---|
| `suggest_model_params` | `:652-812` | `:748-908` |
| `suggest_one_class_params` | `:815-863` | `:911-…` |
| `create_unified_objective` def | `:991` | `:1087` |
| one-class suggest call / fingerprint | `:1306` | `:1402` / `:1409` |
| supervised suggest call / `build_model` / fingerprint | `:1508` | `:1604` / `:1625` / `:1747` |
| `run_unified_bayesian` def / name normalisation | `:2211` | `:2306` / `:2419-2439` |
| objective wiring | `:2465-2492` | `:2560-2587` |
| `config_components` / study name | `:2553-2582` | `:2656-2675` / `:2684-2685` |
| auto-persistence constants `_AUTO_WARMUP`, `_AUTO_THRESHOLD_S` | — | locals, `:2856-2857` |

**Facts the design did not have:**

1. **There are three sampler paths, built at two constructor sites** (verified by both
   reviewers).

   | Path | Constructed at |
   |---|---|
   | Inline sampler for 'never' and the 'auto' warmup | `:2590`, attached `:2796` |
   | 'always' reattach | `:2791` → `_make_tpe_sampler` (`:2213`) |
   | T-41 auto-migration | `:2951` → `_migrate_study_to_sqlite` → `:2302` → `_make_tpe_sampler` |

   Threading `n_startup_trials` to only two of them resets it to 20 after auto-migration.
   Optuna's throwaway default sampler in `create_study` (`:2767`) never drives optimisation.
2. **The study name carries a numerical-environment digest**
   (`unified_bayesian_<model>_<confighash>_env1_<envhash>`). The `space=` segment goes into
   `<confighash>`; the env suffix and its notice (`:2707-2751`) are untouched.
3. **The fit fingerprint includes `model_params`** (`:352`), and both insertion points come
   before both fingerprint calls. That is automatic dedup separation, and also a
   **dedup-exactness hazard for inapplicable axes** (§3.0).
4. **The `model_params` user attr is overwritten** at `:1994-1996` by
   `_capture_serializable_params(model)`, which is deep `Pipeline.get_params()`. Supervised
   `Params` therefore shows real estimator params (`model__gamma`, `max_features`, `lr__C`)
   and **never alias keys**. One-class `Params` is the raw `oc_params` (`:1493`).
5. **Settings capture already misses `bayes_enable_autoscale`**, which feeds both Bayesian
   calls and the study hash (`run_gui_settings.py:279-283`, GUI `:28913`). Restore only
   writes keys present in the snapshot.
6. **NeuralBoosted:**
   - It is absent from `model_name_map`.
   - Its `bayesian_config.py:458` space is tier-dependent.
   - The code exporter has no constructor for it (`code_generator.py:916,1186`,
     `templates/models.py:441`). That gap predates T-51.

## 1. PR split

| PR | Step | Content | Default-path risk |
|---|---|---|---|
| **A** | 2 | Mechanism, per §2. Registry with no bundles, plus one test fixture. Kwargs through both entry points and all three sampler paths. Conditional hash segment and readable study attrs. Hoist `_AUTO_WARMUP`/`_AUTO_THRESHOLD_S` to module level (value-identical) for testability. `ExtraAxesConfigError` with pre-flight validation. Tests T1–T7, T2b, T10, T12, T13, T15. | none (T2/T2b/T3 prove it) |
| **B0** | prerequisite | **PLS-DA head-param plumbing.** It is a latent bug today, and it blocks `plsda_head`. Details in §3.2. Its own approval, because it changes grid-path Tab 7 refit for users whose grid `plsda_lr_C_list` differs from 1.0. **No version bump.** | **fixes existing refit divergence**; default Bayesian search and study names unchanged |
| **B** | 3a | Supervised bundles (§3.1), **needs step 1 merged**. `plsda_head` is included only if B0 is merged; otherwise it moves to a follow-up. | none, all off |
| **C** | 3b | One-class bundles: `if_max_samples`, `lof_metric`, `ocsvm_poly`. | none to supervised |
| **D** | 4 | GUI card, collector, both call sites, advisory, settings capture (plus the existing `bayes_enable_autoscale` gap), docs. | none |
| **E** | 5a | One-class clamp-before-fingerprint and record. PCA-SIMCA `LVs` reporting (§5). | none to supervised |
| **F** | 5b | One-class ceiling into `suggest_int`. Own approval. | none to supervised |

**Deferred out of T-51 v1:**
- `svm_kernels`: both reviewers said to drop it.
- `ocsvm_gamma_float` and `simca_ncomp_wide`: each needs a base-sampler edit.
- `neuralboosted_base`: tier choice plus no export support. Its own follow-up must ship
  exporter support, or document that NeuralBoosted export is unsupported.

**Version / identity bumps.** `__version__` is in every study's config hash (`:2657`), so
**any** bump renames every default study.
- **Step 1 is the one accepted exception.** It changes SVM fits, and the ticket requires
  the bump.
- A–D: none. Their identity is fully captured by the conditional segment.
- B0: **no bump** (round 2 blocker). It changes refit only, never search-time scores or
  study contents. It gets a CHANGELOG entry. If B0 happens to ship in a release that bumps
  for other reasons, that bump is the release's, not B0's.
- T2b pins a literal default-name prefix per model that **does not interpolate
  `__version__`**. A bump therefore fails it on purpose, and re-blessing it is the recorded
  approval of the exception.
- E: **no global bump**, because that would rename every supervised default study.
  Instead add a PCA-SIMCA-only `oc_revision=1` into that model's `config_components`,
  inside a `model_name == 'PCA-SIMCA'` guard.
- F: same mechanism, `oc_revision=2`.

## 2. PR A — mechanism

### 2.1 `src/spectral_predict/search_spaces.py` (new)

```python
SPACE_SCHEMA_VERSION = 1

@dataclass(frozen=True)
class AxisSpec:
    key: str                         # key written into model params
    kind: Literal["int", "float", "categorical"]
    low: Any = None
    high: Any = None
    choices: tuple[Any, ...] | None = None
    log: bool = False
    step: int | None = None
    param_name: str | None = None    # Optuna name; defaults to key
    applies_when_id: str | None = None  # key into PREDICATES (§3.0); no callables in the spec

@dataclass(frozen=True)
class BundleSpec:
    id: str
    families: frozenset[str]         # canonical names AFTER model_name_map
    task_types: frozenset[str]
    axes: tuple[AxisSpec, ...]
    constants: Mapping[str, Any]     # fixed values the bundle also writes (e.g. bootstrap_type)
    label: str
    help: str
    revision: int = 1                # bump on any semantic change to this bundle
```

**`resolve_bundles(model_name, task_type, enabled_extra_axes, search_space) -> tuple[BundleSpec, ...]`.**
This is the single canonicalisation step, and both identity and application consume its
output (Codex M2).
1. `registry = search_space if search_space is not None else BUNDLES`.
2. Unknown id (not in `registry`) → `ValueError` listing valid ids.
3. De-duplicate, then sort by id.
4. Filter to `model_name in families and task_type in task_types`.
5. **Predicate check (round 2 M3):** every `applies_when_id` must be a key of the
   module-level `PREDICATES` registry; an unknown id raises `ExtraAxesConfigError`. The spec
   holds no callables, so identical serialised ids imply identical write semantics.
6. **Static collision check (round 2 M5):** run the real `suggest_model_params` /
   `suggest_one_class_params` for this model against a recording stub trial
   (representative `n_features`, every categorical branch enumerated where the base branches
   on a suggested value, e.g. LightGBM `max_depth`). Collect the base Optuna names. Any
   bundle axis name in that set, or repeated across bundles, raises `ExtraAxesConfigError`.
   This runs **before any storage access or trial**.
7. Return `tuple(copy.deepcopy(b) for b in applicable)`. The snapshot keeps later registry
   mutation from diverging from the hash.

`ExtraAxesConfigError(ValueError)` is a new exception type. The objective's broad handler
(`:2137`) **re-raises** it, so a configuration error aborts the run instead of spending the
trial budget on `1e10` penalty trials. Unknown bundle ids also raise it (step 2).

**`canonical_space_identity(resolved, search_space_given) -> str | None`.**
- Return `None` if `resolved` is empty and `search_space` was not given.
- Otherwise hash sorted-key JSON of:
  - `SPACE_SCHEMA_VERSION`
  - for each bundle: `id`, `revision`, `constants`
  - for each axis: `key`, `param_name or key`, `kind`, `low`, `high`, `log`, `step`,
    `choices` (in order), `applies_when_id`
- Return `sha256[:12]`.
- Choices serialise with a type tag (`["str","sqrt"]`, `["float",0.1]`) so `1` and `1.0`
  stay distinct.

**`apply_extra_axes(trial, params, resolved) -> dict`.**
- Signature matches both call sites (Codex m1). The first statement is
  `if not resolved: return params`.
- **Runtime collision guard (defence in depth):** if any axis's Optuna name is already in
  `trial.params`, raise `ExtraAxesConfigError` (re-raised by the objective, see above).
- **Suggest every axis on every trial**, so the parameter-name set stays uniform for TPE.
  Then **write** `params[key] = value` only if
  `applies_when_id is None or PREDICATES[applies_when_id](params)` (§3.0).
- Then write `constants`.
- Return a new dict.

### 2.2 `unified_bayesian.py` edits (additive)

1. `run_unified_bayesian(..., enabled_extra_axes: Sequence[str] = (), search_space: Mapping[str, BundleSpec] | None = None, n_startup_trials: int | None = None)`.
2. **After** name normalisation (`:2439`), before any storage access:
   `_resolved = resolve_bundles(model_name, task_type, enabled_extra_axes, search_space)`.
   - Enabled ids with nothing applicable → one `logger.warning` plus a `progress_callback`
     message.
   - Some applicable → DEBUG only.
   - Add `'neuralboosted': 'NeuralBoosted'` to `model_name_map`. It is additive: the
     `.get` fallback already returned `'NeuralBoosted'` for the exact spelling.
3. `create_unified_objective(..., resolved_extra_axes: tuple[BundleSpec, ...] = ())`. The
   objective receives the resolved tuple, not raw ids, so it cannot re-canonicalise
   differently. Direct callers of `create_unified_objective` (tests only) get `()`.
4. Insertion:
   - after `:1604-1606`: `model_params = apply_extra_axes(trial, model_params, resolved_extra_axes)`
   - after `:1402`: `oc_params = apply_extra_axes(trial, oc_params, resolved_extra_axes)`
5. Startup trials: `_startup = 20 if n_startup_trials is None else int(n_startup_trials)`,
   and `< 1` raises `ValueError`.
   - Pass it to the inline sampler.
   - Pass it to `_make_tpe_sampler(random_state, n_startup_trials=20)`, a new keyword with
     default 20.
   - Pass it to `_migrate_study_to_sqlite(..., n_startup_trials=20)` and its call at `:2951`.
6. **Identity:** after `config_components` (`:2674`):
   ```python
   _space_id = canonical_space_identity(_resolved, search_space is not None)
   if _space_id is not None:
       config_components += f"|space={_space_id}"
   ```
7. **Readable study attrs, written only when `_space_id is not None`:** add
   `extra_axes_bundles` (sorted applicable ids plus revisions) and `extra_axes_space_id`
   through the existing hoist loop (`:2826-2846`). It carries across `copy_study`
   migration (T12). Record `n_startup_trials_session` **only when the caller passed
   non-None**. It is session metadata, not an audit trail across concurrent sessions.
8. **Hoist** `_AUTO_WARMUP = 10` and `_AUTO_THRESHOLD_S = 1.0` to module scope with
   identical values. Their use sites read the module names.
9. **Not changed:**
   - the bodies of `suggest_model_params` / `suggest_one_class_params` (`git diff` must
     show no hunk inside them)
   - `_build_fit_fingerprint`
   - the env suffix
   - `tools/bench_baseline_compare.py` `WORKER_SCRIPT`, which runs against an old checkout

## 3. PR B0 / B — supervised

### 3.0 Suggest uniformly, write only when applicable (Fable M5, Codex §7)

Writing an inapplicable value gives identical fits distinct fingerprints and silently
defeats dedup. `applies_when` gates the **write**, never the suggest:

| Axis | `applies_when_id` | Predicate |
|---|---|---|
| `svm_gamma.gamma` | `kernel_is_rbf` | `params.get('kernel') == 'rbf'` |
| `ocsvm_poly.degree` | `oc_kernel_is_poly` | `params.get('kernel') == 'poly'` |
| `ocsvm_poly.coef0` | `oc_kernel_poly_or_sigmoid` | `params.get('kernel') in ('poly','sigmoid')` |

Predicates live only in the module-level `PREDICATES` dict keyed by `applies_when_id`.
Custom `search_space` bundles may reference existing ids but cannot supply callables, so
the hashed id fully determines behaviour. Adding or changing a predicate means a new id. TPE still sees the suggested value in `trial.params` even when it is not
written. That is the accepted cost of uniformity: TPE learns that the axis does not matter
under that kernel.

### 3.1 Bundle table

| id | family / task | axes (Optuna name: distribution) | constants / notes |
|---|---|---|---|
| `rf_features` | RandomForest / both | `max_features`: cat[`sqrt`,`log2`,0.1,0.3,0.5,1.0] | overrides base constant `'sqrt'` (`:821`); no warning on mixed types (probed, Optuna 5.0.0) |
| `xgb_regularization` | XGBoost / both | `reg_alpha`: float[1e-4,10] log; `reg_lambda`: float[1e-3,100] log | overrides `:861-862` |
| `xgb_child` | XGBoost / both | `min_child_weight`: float[0.5,20] log; `gamma`: float[1e-4,5] log | |
| `xgb_sampling` | XGBoost / both | `colsample_bytree`: float[0.3,1.0]; `colsample_bylevel`: float[0.3,1.0] | base `subsample` suggest untouched |
| `lgbm_regularization` | LightGBM / both | `reg_alpha`: float[1e-4,10] log; `reg_lambda`: float[1e-3,100] log | |
| `lgbm_sampling` | LightGBM / both | `subsample`: float[0.5,1.0]; `colsample_bytree`: float[0.3,1.0] | `bagging_freq=1` already set (`:846`) |
| `lgbm_child` | LightGBM / both | `min_child_samples`: int[2,50]; `min_split_gain`: float[1e-4,1] log (user-approved 2026-09-14, §10) | constant range per study; large values stop splitting, documented in `help` |
| `catboost_sampling` | CatBoost / both | `subsample`: float[0.5,1.0]; `rsm`: float[0.1,1.0] | **constant `bootstrap_type='Bernoulli'`**. Multiclass defaults to Bayesian, which rejects `subsample` (Codex probe). T8b fits regression, binary and multiclass. |
| `svm_gamma` | SVM, SVR / matching task | `gamma`: float[1e-5,10] log, `applies_when=kernel_is_rbf` | `help`: tuning C and gamma jointly on small n overfits |
| `mlp_activation` | MLP / both | `activation`: cat[`relu`,`tanh`,`logistic`] | |
| `plsda_head` | PLS-DA / classification | `lr_C`: float[1e-3,1e3] log | **only after B0** |

`svm_kernels` and `neuralboosted_base` are deferred (§1).

### 3.2 PR B0 — PLS-DA head params (both reviewers, independently)

The chain as verified today, with Fable's and Codex's probes:

| Hop | State | Fix in B0 |
|---|---|---|
| `build_model('PLS-DA', params)` → `PLSTransformer(scale=False, **params)` (`models.py:468-469`) | **`TypeError` on `lr_C`** | build the transformer from params without `lr_`-prefixed keys. It is a no-op for current callers; T-B0a proves it. |
| objective LR construction (`unified_bayesian.py:1692`) | reads `lr_C`; correct once construction is fixed | none |
| `_capture_serializable_params` | stores `lr__C` | none; `lr__C` is the stored canonical form |
| `search._rebuild_model_from_row` (`search.py:432-452`) | **skips `lr__*`, then reads `lr_C` → C=1.0** | map `lr__C/lr__solver/lr__max_iter` → head kwargs; accept legacy `lr_C` |
| GUI Tab 7 refit (`:38804`, `_lr_kwargs` `:39469` used `:39624/39752/39808`) | **no C at all**; also drops the grid path's `plsda_lr_C_list` today | read `lr__C` / `lr_C` from the row into `_lr_kwargs` |
| `model_io` save/load (joblib, `model_io.py:216,433`) | keeps whatever the refit used | none |
| `code_generator._split_pls_da_params` (`:1305-1306`) | correct for `lr__C`; a raw `lr_C` passes through at `:1314` as an invalid kwarg | accept legacy `lr_C` too |

**The B0 mechanism (round 2 M2).** One shared helper,
`split_plsda_params(params) -> (transformer_params, head_params)`, runs **before any
`set_params()`** in all three consumers:
- It accepts `lr__C|solver|max_iter` (canonical) and `lr_C|solver|max_iter` (legacy).
  Canonical wins when both are present.
- It strips `pls__` from transformer keys, and never passes any head key to the transformer.
- Rebuild (`search.py:425-452`) and Tab 7 (`GUI :38806-38810`) currently apply unprefixed
  `lr_C` to the transformer. Both change to call the helper.
- Tab 7 `_lr_kwargs` (`:39469`) takes C, **solver and max_iter** from `head_params`, which
  the grid already uses (`search.py:4952`), defaulting to today's values.

**B0 acceptance.** Test a matrix of {canonical `lr__*` row, legacy `lr_*` row} ×
{default, non-default transformer settings (`n_components`)} × {non-default C, solver,
max_iter}, run through rebuild, Tab 7 refit and export. Each case must yield those exact
head/transformer params and predicted probabilities equal to the search-time pipeline
(`np.allclose`).

### 3.3 Downstream consumers

- **Flat bundles:** keys arrive as real estimator params (`model__gamma`, …). They follow
  the existing refit/export paths, and T9 covers one per family.
- **One-class:** raw `oc_params` flow to `build_one_class_model` (`contamination.py:400,406`),
  which passes them through.

## 4. PR D — GUI and docs

1. The card "Bayesian — extra hyperparameter axes" goes in Tab 4C after the Bayesian trials
   controls, visually separate from "Advanced Model Options" (grid-only). **It is built
   from the registry** (`label`, `help`).
2. **Stable var names:** `self.bayes_axis_<bundle_id>` (`tk.BooleanVar`). Capture uses
   `getattr`, so names must not be generated nondeterministically.
3. `_collect_enabled_extra_axes()` returns a sorted tuple. Pass it at GUI `:28332` (one-class)
   and `:28890` (supervised). The GUI does not filter by model; the backend resolves.
4. `n_startup_trials`: a `tk.StringVar` spinbox (blank → `None`; an `IntVar` cannot be blank).
5. **Settings capture (Codex M4):**
   - Add every bundle var, `bayes_n_startup_trials`, and the existing gap
     `bayes_enable_autoscale` to `CAPTURABLE_SETTINGS`.
   - **Legacy defaults on restore:** a snapshot without these keys resets bundles to off
     and startup to blank, never "leave the current widget state".
   - **Missing `bayes_enable_autoscale` (round 2 m6):** the historical value is
     unrecoverable. Restore sets it to the backend default `False` **and lists it in the
     restore report as "not recorded in this snapshot — assumed off"**, so a user who ran
     with autoscale on sees why resume picked a fresh study. Legacy defaults apply before
     the empty-settings early return.
   - T11 restores an old snapshot onto controls that were already toggled.
6. Advisory:
   - `effective_dim = base_axes[model] + applicable bundle axes + shared axes`.
   - Show "~N trials; startup max(20, 3×dim)".
   - Never write `n_unified_trials`.
   - **Evidence caveat (§10):** in the contamination project, the widened supervised space
     reached its best result *earlier* (trial 58 vs 120) and was flat after trial 150. The
     advisory is an upper-bound suggestion and must say so.
   - **Runtime note** (§10): show it when IsolationForest or boosting bundles are on.
     IsolationForest dominated wall time downstream.
   - **Help-text wording** (§10): "Opening axes improves the best candidates but lowers
     the average one. Validate externally." Downstream: 4 vs 1 models at full holdout, but
     a pass rate of 6.8% vs 16.8%.
   - The `base_axes` table sits in `search_spaces.py` with a test that counts real suggests
     per family on a recording trial.
7. Resume banner help text: "changing extra axes starts a new study."
8. Docs:
   - `AGENT_COMPOSITION.md`: the three kwargs, precedence rule, and one executed example.
   - Update the `tests/test_agent_composition_api.py` contract.
   - Soften the `CLAUDE.md` "all hyperparameters exposed" claim.

## 5. PR C / E / F — one-class

- **C:**
  - `if_max_samples`: `max_samples` cat[`auto`,0.5,0.8,1.0].
  - `lof_metric`: cat[`euclidean`,`manhattan`,`cosine`]. `minkowski` is dropped because
    with default `p=2` it duplicates euclidean. `algorithm='auto'` already picks brute for
    cosine (probed by both reviewers).
  - `ocsvm_poly`: `degree` int[2,3], `coef0` float[-1,1], gated per §3.0. The base kernel
    set `['rbf','poly','sigmoid']` (`:939`) makes it valid. Degree is capped at 3 because
    that is the only range with downstream evidence (§10).
- **E (Codex M6):**
  - Record the resolved components as `n_components_actual` for PCA-SIMCA.
  - **Extend the dataframe converter** (`:3402`), which today reads `n_components_actual`
    only for PLS/PLS-DA.
  - Calibration ceiling: `min(total_inliers - 1, selected_features)`. Each fold keeps its
    existing internal clamp (`contamination.py:132-147`), so fits are unchanged.
  - A missing `cal_model` falls back to `LVs=None`.
  - Old studies are unaffected, because `oc_revision` renames PCA-SIMCA studies only.
- **F (round 2 M4, §10).** Specified so the distribution is **fixed within a study**:
  - The ceiling is `min(20, min_train_fold_inliers - 1)`, from `cv_utils` fold arithmetic
    on the inlier count. It is **not** the selected feature count, which varies per trial
    and would make the distribution dynamic, splitting TPE's multivariate KDE.
  - Feature-count clamping stays inside `PCASIMCA.fit` and PR E's recorded value.
  - If the ceiling is below 2, PCA-SIMCA is skipped for the run with a `progress_callback`
    notice. There is no `suggest_int(…, 2, 1)`.
  - Uses `oc_revision=2`, and needs its own approval.
  - **Explicit exception** to the no-hunk rule: F edits `suggest_one_class_params` by
    passing the ceiling in through a new keyword with default 20, so other callers are
    unchanged.
  - The no-hunk check applies to A–E only.
  - Downstream evidence supports a fold-based ceiling: 20 inliers → about 16 per fold, and
    a 16-row class collapsed requests of 16–20 into one model.
  - It also shows that more components raised CV while making external transfer worse, so
    F is about honesty and dedup, not performance.

## 6. Tests

| id | PR | Test |
|---|---|---|
| T1 | A | `apply_extra_axes(trial, params, ())` with a trial whose every attribute access raises → returns `params` |
| T2 | A | **Pinned default study name as a template:** build the expected `config_components` in the test from `__version__` and the call's settings, then assert `study.study_name == f"unified_bayesian_PLS_{sha256(expected)[:8]}_{ENV_FINGERPRINT_VERSION}_{_environment_digest(_numerical_environment())}"`. No extraction refactor. Captured after step 1 merges. |
| T2b | A | Literal default study-name prefix per model (`unified_bayesian_PLS_<hash>`) **without** interpolating `__version__`; deliberately fails on any version bump (§1) |
| T3 | A | **TPE trajectory, not just startup:** PLS regression, `n_trials=30`, fixed inputs, `enable_sqlite_persistence='never'`. Store `[(t.params, t.value) for t in study.trials]` from **`main` after step 1 merges** as a JSON fixture that records the reference commit, the full `_numerical_environment()` dict, the data recipe and the call settings. **Skip when the current env digest differs**, and emit an explicit "fixture needs re-blessing" message. Re-blessing is a human-reviewed commit. Assert equality for the omitted-kwarg run **and** the `enabled_extra_axes=()` run. |
| T4 | A | Sampler startup on all three paths, end to end. Patch `unified_bayesian.get_storage_url` (`:2690`) to a temp SQLite URL, set the hoisted `_AUTO_THRESHOLD_S=0.0` (consumed `:2919`), spy on constructors at `:2590` and `:2215`, run ≥11 trials, and assert migration (`:2951`) **actually happened** and later trials persist. Default gives 20 on the initial, 'always' and migrated samplers. (`test_t41_...:164` defines a duration patch but never installs it, so it is not a template.) |
| T5 | A | Unknown id raises `ValueError` and the temp SQLite file is never created |
| T6 | A | Reordered and duplicated ids give the same study name **and the same `trial.params` sequence**. Different applicable sets give different names. A non-applicable-only set gives the default name, and the warning is emitted once. |
| T7 | A | Fingerprints differ when only a written bundle value differs; replay works when identical; **identical when only an unwritten (`applies_when` false) value differs** |
| T10 | A | `n_startup_trials=5` asserted on **each** of the initial, 'always'-reattached and migrated samplers (hooks as in T4) |
| T12 | A | Resume in 'always' with the same bundles continues the same study: the name matches and the trial count grows. **The second call's `n_trials` must be larger, because it is a total target (`:3075`).** A different set starts a new study. Turning bundles off again selects the default study. `extra_axes_*` attrs survive migration, checked after reopening SQLite. |
| T13 | A | Fixture `search_space` whose bundles differ only in `param_name`, `revision`, `applies_when_id` or constant → distinct identities; unknown `applies_when_id` → `ExtraAxesConfigError`; precedence (`search_space` replaces the registry) tested both ways |
| T15 | A | A fixture bundle colliding with a base name (e.g. PLS `n_components`) raises `ExtraAxesConfigError` from `run_unified_bayesian` **before** storage or any trial. It is not a penalty-only study. |
| T8a | B | Every bundle × applicable family × task_type: the real `suggest_model_params` + `apply_extra_axes` on a recording trial yields no duplicate name, and `build_model(**params)` does not raise |
| T8b | B | Real `fit` on tiny data for every bundle × family × {regression, binary, multiclass} as applicable. Catches CatBoost-style fit-time failures. |
| T9 | B | Short real run per bundle: bundle keys appear in `ast.literal_eval(results_df['Params'])`. One flat bundle per family round-trips rebuild and Tab 7 → identical predictions. **LightGBM must include the base `bagging_freq=1`** (§10): it is an alias key that `get_params` reports alongside `subsample_freq=0`, and a rebuild that drops it silently disables bagging. |
| T-B0a/b | B0 | `build_model('PLS-DA', {'n_components': 5})` unchanged; `lr__C` round-trip through rebuild, Tab 7 refit and export (§3.2 acceptance) |
| T11 | D | Collector, registry-built card, stable var names, legacy-snapshot restore, `bayes_enable_autoscale` captured |
| T14 | E | PCA-SIMCA `LVs` equals `cal_model.n_components_`; above-ceiling requests collapse to one fingerprint |
| — | all | Existing: `test_unified_bayesian_baseline`, `test_agent_composition_api`, `test_bayesian_dedup`, `test_contamination_detection`, `test_simca`, `test_cv_pls_clamp`, `test_t41_*`, `test_bayesian_study_lookup`, `test_bayesian_environment_fingerprint`, plus the literal no-hunk `git diff` check on both sampler bodies for **A–E** (F is the documented exception, §5) |

## 7. Step-0 caller enumeration

Swept 2026-09-13 across all Python, the docs code examples and notebooks (there are no
`.ipynb` files). **55 real calls; no production wrapper forwards arbitrary `**kwargs`.**
`tests/test_bayesian_study_lookup.py` expands a **fixed-key** `search_options` dict five
times. That is not a forwarding wrapper, but new kwargs reach it only if they are added to
the fixture. Neither function is monkeypatched, fetched with `getattr`, or imported with
`import *`.

| Category | Site | Action for T-51 |
|---|---|---|
| src | `unified_bayesian.py:2560` RUB → CUO, explicit 26 kwargs | **PR A: pass `resolved_extra_axes`** |
| src | three sampler paths (§0.1) | **PR A: thread `n_startup_trials`** |
| src | `unified_bayesian.py:3547` `__main__` demo; module docstring `:23` | none |
| GUI | `spectral_predict_gui_optimized.py:28332` (one-class), `:28890` (supervised), both in `_run_analysis_thread` (`:26064`) | **PR D** |
| tests | CUO: `test_bayesian_dedup.py:258,317`, `test_one_class_varsel_filtering.py:190`. RUB: `test_bayesian_study_lookup.py` (fixed-key splat ×5), `test_class_weight_sister_sites.py`, `test_cv_anova.py`, `test_cv_pls_clamp.py`, `test_cv_strategy.py`, `test_t19_auto_mode_entry_points.py`, `test_t41_bayesian_sqlite_auto_calculator.py` (8), `test_t42_write_path_plumbing.py` (5), `test_unified_bayesian_baseline.py` (5), `test_varsel_caching_correctness.py` (6) | none. Defaults preserve behaviour (T2/T3). |
| tests (introspection) | `test_autoscale_bayesian.py:169`, `test_bayesian_environment_fingerprint.py:152-158`, `test_t44_autoscale_wiring.py:35-44` | substring/signature presence only; re-run |
| tools | `ab_dedup_compare.run_case`, `ab_lv_compare.run_search`, `autoscale_bayesian_compare.run_one_cell`, `bayesian_topk_stability.run_one_bayesian`, `bench_dedup_real.run_phase` | none (default-path benchmarks) |
| tools | `bench_baseline_compare.py:70` `WORKER_SCRIPT` string, run against an **old checkout** | **must NOT gain new kwargs** |
| docs | historical plans and continuation prompts | none |
| docs | `docs/AGENT_COMPOSITION.md:385`, `tests/test_agent_composition_api.py:42` | **PR D** |

The GUI's supervised call already omits `early_stopping_rounds` and `n_top_regions`, so
both silently take their defaults. PR D must not copy that pattern for bundles.

## 8. Open questions (remaining)

1. **B0 approval — APPROVED by the user 2026-09-14** ("if it was legit wrong before then
   fixing it we should do"). Ship it as a separate correctness fix, with no version bump.
   It changes Tab 7 refit for PLS-DA rows whose C, solver or max_iter differ from the
   defaults. Saved models keep their fitted objects; only new refits change.
2. **T3 fixture maintenance.** *Round 2 answer, adopted:* never auto-bless. The upgrade
   check may generate a candidate fixture and diff report; a human commit accepts it.
3. **`min_split_gain` for LightGBM — APPROVED by the user 2026-09-14.** It is added to
   `lgbm_child` (§3.1) as float[1e-4,1] log. It was absent from the base (so it cannot
   collide), and it is the downstream-measured regulariser (winner 0.058).
4. **`colsample_bytree` floor.** The downstream winner sat at 0.32, near the plan's 0.3
   floor. The evidence is one model, too thin to lower the floor. Keep 0.3 and revisit with
   data.

## 9. Review round 1 decisions

| Finding | Source | Decision |
|---|---|---|
| `plsda_head` raises in `build_model`; rebuild and Tab 7 drop C | Fable B1, Codex B1 + M1 | new PR B0; `plsda_head` gated on it |
| CatBoost multiclass rejects `subsample` | Codex B2 (Fable probed binary only) | constant `bootstrap_type='Bernoulli'` in identity; T8b multiclass fit |
| T4/T10 unwritable (locals) | Fable B2, Codex §5 | hoist the two constants; T-41 duration patching |
| §2.2.6 "Params shows bundle values" is wrong | Fable M1 | §0.4 rewritten; aliases never surface |
| T2 literal stale after the version bump | Fable M2 | template pin |
| T3 startup-only / Optuna-bound | Fable M3, Codex M5 | 30 trials, fixture tagged by Optuna version, captured after step 1 |
| validate after normalisation; NeuralBoosted map | Fable M4 | §2.2.2 |
| inapplicable axes break dedup exactness | Fable M5, Codex §7 | §3.0 suggest-uniform / write-when-applicable |
| user_attrs on default runs; no readable bundle record | Fable M6 | write only when non-default; `extra_axes_*` attrs |
| NeuralBoosted tier + no export | Fable M7, Codex M7 + m3 | defer bundle |
| identity vs application canonicalisation mismatch | Codex M2 | single `resolve_bundles` |
| `param_name` missing from identity | Codex M3 | full axis serialisation plus `revision`, `applies_when_id`, constants |
| settings restore leaves bundles on; `bayes_enable_autoscale` missing | Codex M4 | legacy defaults on restore; capture autoscale |
| PR E's `LVs` never reach the dataframe | Codex M6 | extend converter; PCA-SIMCA-only `oc_revision` |
| signature mismatch between §2.1 and §2.2 | Codex m1 | one signature over the resolved tuple |
| "no `**kwargs`" overstated; LOF brute; mixed-type warning | Codex m2, Fable minors | corrected; `minkowski` dropped |
| drop `svm_kernels` | both | dropped |
| skip vs raise | both | raise on unknown, skip non-applicable, warn once when none apply |
| `n_startup_trials` excluded from identity | both agree | kept excluded; session attr only when passed |
| PR E version bump | Fable: none; Codex: model-scoped revision | model-scoped `oc_revision` (no global bump) |

## 9b. Review round 2 decisions (`gpt-6-astra`, 2026-09-14)

| Finding | Severity | Decision |
|---|---|---|
| B0's `__version__` bump renames every default study; T2's template would hide it | BLOCKER | B0 drops the bump; new T2b literal prefix without version; step 1 is the one recorded exception (§1) |
| B0 still routes legacy `lr_C` into the transformer via `set_params`; Tab 7 drops solver/max_iter | MAJOR | shared `split_plsda_params` before any `set_params`; canonical `lr__*` wins; full acceptance matrix (§3.2) |
| callable `applies_when` is an identity loophole | MAJOR | callables removed from the spec; `PREDICATES` registry authoritative; unknown id raises (§2.1, §3.0) |
| PR F's feature-count ceiling makes the distribution dynamic; ceiling <2; no-hunk conflict | MAJOR | fold-inlier ceiling, fixed per study; skip PCA-SIMCA below 2; F is the documented no-hunk exception (§5) |
| collision error inside the objective becomes 1e10 penalty trials | MAJOR | static pre-flight collision check and `ExtraAxesConfigError` re-raised by the objective; T15 (§2.1) |
| legacy restore leaves autoscale unspecified | MINOR | default False plus an explicit restore-report line (§4.5) |
| T3 fixture tagged by Optuna only | MINOR | full numerical-env record; skip with re-bless message; human-reviewed refresh (§6) |
| T4/T10 cited a T-41 patch that is never installed; T12 `n_trials` is a total target | test spec | hooks rewritten (`get_storage_url`, hoisted threshold, constructor spies); T12 raises the target (§6) |

Round 2 confirmed as correct: suggest-uniform/write-when-applicable against the fingerprint
and replay, single canonical resolution, the hoisted constants leaving defaults identical,
model-scoped `oc_revision`, the B0 construction fix, and all three sampler paths.

## 10. Evidence from the contamination project (2026-09-14)

The downstream project at `Desktop\_DeskSync\contamination` ran an expanded search. Its
supervised probe monkeypatched `suggest_model_params`; its one-class searches used its own
Optuna harness. It was surveyed read-only, from code, handoffs and aggregate results only.
**Nothing is copied; these are lessons.**

| Observation (source) | Plan consequence |
|---|---|
| **Widened LightGBM/XGBoost, 300 trials:** pass rate 16.8% → 6.8%, but models at full holdout 1 → 4. Best result at trial 58 vs 120; flat after trial 150. (`HANDOFF_2026-08-29_TUNING_PROBES.md:183-224`) | GUI help and docs: "improves best, lowers average"; advisory is an upper bound (§4.6) |
| **Their `subsample` axis was inert.** Their rebuild dropped `bagging_freq` ("captured params not settable"). | **Verified in DASP (probe, 2026-09-14):** `build_model` with `bagging_freq=1` *does* activate bagging (predictions equal `subsample_freq=1` and differ from `subsample=1.0`). `get_params` reports `bagging_freq=1` alongside `subsample_freq=0`. Risk is refit parity only → T9 LightGBM clause. `lgbm_sampling` stays. |
| **Winner:** `colsample_bytree` 0.32, `min_child_samples` 35, `min_split_gain` 0.058, all inside plan ranges except `min_split_gain` (absent) | §8 Q3 (add `min_split_gain`), Q4 (keep 0.3 floor) |
| **PCA-SIMCA ceiling 10 → 14:** CV and holdout up, external false positives 79 → 105; best-external model at CV rank 214 | supports deferring `simca_ncomp_wide`; if revived, a categorical including mode values (`per_class_cv`, variance fraction), not a wider int |
| **16-row class:** requests 16–20 collapse to one model; fold arithmetic gives ≤15 | PR F fold-based ceiling (§5) |
| **Floating one-class continuous params:** 2× compute for +0.008 CV; IsolationForest and LOF holdout fell | one-class bundles are for legitimacy/exploration, not expected gains; say so in help |
| **IsolationForest** dominated wall time (21,255 of 24,041 s) | runtime note in the advisory (§4.6) |
| **OCSVM `degree`** measured only over 2–3 | `ocsvm_poly.degree` int[2,3] (§5) |
| **Run-id collisions** between probe and published candidates | supports T12 / space identity in the study name |
| **Discrete spaces saturate:** 1,419 trials gave 697 unique configs | GUI advisory also warns when the enabled space is mostly categorical and the trial count exceeds its cardinality; dedup replay already avoids refits |
