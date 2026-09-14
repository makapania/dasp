# T-51 steps 2–5 — Opt-in Optuna search axes: implementation plan

> **STATUS (2026-09-13): plan for review, no code written.** It turns the design ticket
> [`2026-08-30-T51-bayesian-opt-in-search-axes.md`](2026-08-30-T51-bayesian-opt-in-search-axes.md)
> into ordered, checkable work. The ticket is still the authority on *why*. This file covers
> *how*, and it records where the code has moved since the ticket was written.
>
> **Not covered here:** step 1, the SVM scaler fix (its own PR, `fix/T51-svm-scaler`), and
> step 6, the PLS clamp asymmetry (last PR, own approval).
>
> Reviewers: Fable, GPT-6 alpha.

## 0. What changed since the ticket (verified against `main` @ `696fce9`)

Line numbers below are for `src/spectral_predict/unified_bayesian.py` unless another file
is named. **The ticket's line numbers are stale by roughly 90–100 lines. Do not use them.**

| Anchor | Ticket said | Now | Consequence |
|---|---|---|---|
| `suggest_model_params` | `:652-812` | `:748-908` | none |
| `suggest_one_class_params` | `:815-863` | `:911-…` | none |
| `create_unified_objective` def | `:991` | `:1087` | none |
| one-class suggest call | `:1306` | `:1402` | insertion point |
| supervised suggest call | `:1508` | `:1604-1606` | insertion point |
| fit fingerprint call (supervised) | — | `:1747` | **must stay after** the insertion point (it does) |
| `run_unified_bayesian` def | `:2211` | `:2306` | none |
| objective wiring | `:2465-2492` | `:2560-2587` | kwarg threading |
| inline `TPESampler` | `:2494` | `:2590-2596` | sampler site 1 |
| `_make_tpe_sampler` | `:2117` | `:2213-2221` | **now has two callers, see below** |
| `config_components` | `:2553-2582` | `:2656-2675` | hash segment |

**New since the ticket (not in its design):**

1. **There are three sampler construction paths, not two.** `_make_tpe_sampler(random_state)`
   is called from `_migrate_study_to_sqlite` (`:2302`, the T-41 'auto' in-memory→SQLite
   migration after warmup) **and** from the 'always' resume path (`:2791`). The inline
   sampler at `:2590` serves 'never' and the 'auto' warmup. A custom `n_startup_trials`
   threaded only to the two sites the ticket names would be **silently reset to 20 after
   the T-41 auto-migration**, which is exactly the case where trials are slow and the
   startup budget matters most. `_make_tpe_sampler` and `_migrate_study_to_sqlite` both
   need the kwarg.
2. **The study name now carries a numerical-environment digest**
   (`unified_bayesian_<model>_<confighash>_env1_<envhash>`, `:2684-2685`) plus a legacy and
   incompatible-study notice (`:2707-2751`). The conditional `space=` segment goes into
   `config_components`, so it lands in `<confighash>`. The env suffix and notice logic are
   untouched. A resumed bundle study whose bundle set changed shows up as "no matching
   study", not "incompatible environment". That is correct but not self-explanatory. See
   §4.3.
3. **The fit fingerprint already includes `model_params`** (`_build_fit_fingerprint`,
   `:295-360`, `('model_params', _freeze_for_fingerprint(model_params))`). Bundle-sampled
   values therefore separate fingerprints automatically **as long as `apply_extra_axes` runs
   before `:1747`**, which it will. No fingerprint change is needed. Test it anyway (§6, T7).
4. **The calibration refit overwrites the `model_params` user attr** with
   `_capture_serializable_params(model)` (`:1994-1996`), i.e. `Pipeline.get_params()`
   filtered. For flat estimators, bundle keys such as `gamma` and `max_features` survive
   because they are real estimator params. For **PLS-DA `lr_C`** the value lives on the
   pipeline's `lr` step, and a captured dict has to carry it in a form that Tab 7 refit and
   `model_io` can consume. **This is unverified. It is the highest-risk item in the plan.**
   See §3.2.

## 1. Deliverables and PR split

| PR | Ticket step | Content | Default-path risk |
|---|---|---|---|
| A | 2 | `search_spaces.py` (registry, `AxisSpec`, `apply_extra_axes`, `canonical_space_identity`), kwargs threaded through both entry points and **all three** sampler paths, conditional hash segment, invariant tests. **No bundles registered** except one test-only fixture. | none (proved by §6 T1–T4) |
| B | 3a | Supervised bundles: `rf_features`, `xgb_regularization`, `xgb_child`, `xgb_sampling`, `lgbm_regularization`, `lgbm_sampling`, `lgbm_child`, `catboost_sampling`, `svm_gamma`, `svm_kernels`, `mlp_activation`, `plsda_head`, `neuralboosted_base`. **Requires PR for step 1 merged first** (`svm_gamma` on unscaled SVM is meaningless). | none, all off |
| C | 3b | One-class bundles: `if_max_samples`, `lof_metric`, `ocsvm_poly`. `ocsvm_gamma_float` and `simca_ncomp_wide` are **deferred** (they need a base-sampler edit, which is gated by step 5b). | none to supervised |
| D | 4 | GUI card, collector, both call sites, advisory label, `CAPTURABLE_SETTINGS`. Docs: `AGENT_COMPOSITION.md`, soften `CLAUDE.md` "all hyperparameters exposed". | none |
| E | 5a | One-class clamp-before-fingerprint + record resolved `n_components_`. | none |
| F | 5b | One-class ceiling into `suggest_int` + `__version__` bump. Own approval. | none to supervised |

PRs A–D carry **no** `__version__` bump. None of them changes default behaviour, and a bump
would needlessly orphan every persisted study. E is borderline: it changes dedup, not fits.
Bump only if the review says the recorded `LVs` change counts as a trial-value change for
resume. F bumps.

## 2. PR A — mechanism

### 2.1 `src/spectral_predict/search_spaces.py` (new)

Follow the ticket's sketch, with these concrete decisions:

- **Registry shape:** `BUNDLES: dict[str, BundleSpec]`, keyed by bundle id.
  `BundleSpec(id, families: frozenset[str], task_types: frozenset[str], axes: tuple[AxisSpec, ...], label: str, help: str)`.
  `families` uses the **canonical names after `model_name_map` normalisation** (`:2419-2439`):
  `'SVM'` and `'SVR'`, not `'svm'`.
- **`AxisSpec`** as in the ticket, minus `derived`/`resolve` for PR A. Add them only when a
  PR-B/C bundle needs them (YAGNI; the only candidate is `svm_kernels.degree`, and it does
  not need them).
- **`apply_extra_axes(trial, model_name, task_type, params, enabled_extra_axes) -> dict`**
  - Empty or `None` `enabled_extra_axes` → `return params` **as the first statement**. No
    registry lookup, no copy.
  - Otherwise: validate every id (unknown → `ValueError` naming the valid ids for this
    family). An id valid for a *different* family or task is **skipped silently** for the
    current model, because one GUI selection is shared across a multi-model run (the GUI
    runs `run_unified_bayesian` once per model). Log it at DEBUG.
  - Before any suggest call, check each axis's Optuna name against `trial.params` and
    against names already suggested in this call. On a collision → `RuntimeError`. This is
    the runtime twin of the static collision test.
  - Returns `{**params, **extra}`, a new dict. Callers must not rely on identity.
- **`canonical_space_identity(model_name, task_type, enabled_extra_axes, search_space) -> str`**
  - Filter to bundles that apply to this `(model_name, task_type)`, then sort and de-duplicate.
  - Serialise each applicable bundle's axes (name, kind, low, high, log, step, choices) as
    JSON with sorted keys, and prefix `SPACE_SCHEMA_VERSION = 1`.
  - Return `sha256(...)[:12]`.
  - **An enabled set with no applicable bundles returns `None`**, and the caller appends
    nothing. Otherwise enabling `svm_gamma` would change the PLS study name in a multi-model
    run, orphaning PLS studies for no reason. This deviates from the ticket's literal "append
    when non-empty" rule, and the deviation is deliberate.
- **`search_space` override:** ship the kwarg in PR A as the ticket requires, typed
  `Mapping[str, BundleSpec] | None`, replacing the registry for lookup when given. It always
  appends a segment, even when empty, because the caller asserted a custom space.

### 2.2 `unified_bayesian.py` edits (additive only)

1. `create_unified_objective(..., enabled_extra_axes: Sequence[str] = (), search_space=None)`.
2. After `:1604-1606`:
   `model_params = apply_extra_axes(trial, model_name, task_type, model_params, enabled_extra_axes, search_space)`.
3. After `:1402`: the same for `oc_params`, with `task_type='one_class'`.
4. `run_unified_bayesian(..., enabled_extra_axes=(), search_space=None, n_startup_trials: int | None = None)`.
   - Validate ids **once up front**, before any study is created. A typo must raise before
     SQLite is touched.
   - Forward to `create_unified_objective` (`:2560-2587`).
   - `_startup = 20 if n_startup_trials is None else int(n_startup_trials)`, with
     `ValueError` if `< 1`. Pass it to the inline sampler (`:2592`), to
     `_make_tpe_sampler(random_state, n_startup_trials=20)` (new keyword with default 20, so
     other callers are unchanged), and through
     `_migrate_study_to_sqlite(..., n_startup_trials=20)` and its call site.
   - **`n_startup_trials` is not part of the study identity.** It changes future sampling,
     not the validity of cached scores. Resuming with a different startup count is legitimate
     and matches how `n_trials` is already excluded (`:2620-2626`). Record the value in
     `study.user_attrs` as an audit field. The existing "don't overwrite on resume" rule
     (`:2803+`) does **not** apply to it, so store it per session as
     `n_startup_trials_last`.
5. After `config_components` is built (`:2674`), and before hashing:
   ```python
   _space_id = canonical_space_identity(model_name, task_type, enabled_extra_axes, search_space)
   if _space_id is not None:
       config_components += f"|space={_space_id}"
   ```
6. **Results/leaderboard:** the `Params` column already comes from `model_params`, so bundle
   values appear with no change. Verify it (T8).

**`git diff` on the bodies of `suggest_model_params` and `suggest_one_class_params` must be
empty in every PR A–E.** The review checks this literally.

### 2.3 Other callers

Fill from the step-0 enumeration (§7). Rule: any wrapper that forwards a **fixed subset** of
kwargs to either entry point either forwards the three new kwargs or is documented as not
supporting bundles. A wrapper that silently drops `enabled_extra_axes` is a bug.

## 3. PR B — supervised bundles

### 3.1 Bundle table (all axes suggested unconditionally on every trial)

| id | family | axes (Optuna name: distribution) | notes |
|---|---|---|---|
| `rf_features` | RandomForest | `max_features`: cat[`sqrt`,`log2`,0.1,0.3,0.5,1.0] | check that the Optuna categorical accepts a mixed str/float list (it does, with a warning; confirm no warning escalation in tests) |
| `xgb_regularization` | XGBoost | `reg_alpha`: float[1e-4,10] log; `reg_lambda`: float[1e-3,100] log | replaces 0.1 / 1.0 |
| `xgb_child` | XGBoost | `min_child_weight`: float[0.5,20] log; `gamma`: float[1e-4,5] log | `gamma` name collides with nothing in XGBoost's base |
| `xgb_sampling` | XGBoost | `colsample_bytree`: float[0.3,1.0]; `colsample_bylevel`: float[0.3,1.0] | base already suggests `subsample`; not touched |
| `lgbm_regularization` | LightGBM | `reg_alpha`: float[1e-4,10] log; `reg_lambda`: float[1e-3,100] log | |
| `lgbm_sampling` | LightGBM | `subsample`: float[0.5,1.0]; `colsample_bytree`: float[0.3,1.0] | `bagging_freq=1` stays; fine at `subsample=1.0` |
| `lgbm_child` | LightGBM | `min_child_samples`: int[2,50] | clamp high to `< min_train_fold_size` via `resolve`? **No:** the range must stay constant per study, and LightGBM tolerates a large value (just no splits). Document it. |
| `catboost_sampling` | CatBoost | `subsample`: float[0.5,1.0]; `rsm`: float[0.1,1.0] | `subsample` needs `bootstrap_type` ≠ `Bayesian`; **verify `build_model`'s CatBoost bootstrap type**, else set `bootstrap_type='Bernoulli'` inside the bundle and include that constant in the identity |
| `svm_gamma` | SVM, SVR | `gamma`: float[1e-5,1e1] log | **the base sets `gamma='scale'` only when kernel==rbf.** The bundle suggests `gamma` on every trial (uniformity rule) and writes it into params for every kernel; sklearn ignores it for linear. GUI note: tuning C and gamma jointly on small n overfits. |
| `svm_kernels` | SVM, SVR | `kernel_ext`: cat[`base`,`poly`,`sigmoid`]; `degree`: int[2,5]; `coef0`: float[-1,1] | **The base already suggests `kernel`.** It cannot be widened, so use an **alias** `kernel_ext` that overrides `params['kernel']` when ≠ `base`. Record in the plan that this doubles the kernel dimension. Reviewers: is this worth it, or drop it from v1 like `linear_alpha_wide`? **Recommendation: drop it from v1.** |
| `mlp_activation` | MLP | `activation`: cat[`relu`,`tanh`,`logistic`] | |
| `plsda_head` | PLS-DA (classification only) | `lr_C`: float[1e-3,1e3] log | consumed at `:1692`; see §3.2 |
| `neuralboosted_base` | NeuralBoosted | lift from `bayesian_config.py:458` | the base returns `{}`; no collision possible. Do **not** delete `bayesian_config.py` in this PR. |

### 3.2 `plsda_head` persistence risk (verify before implementing)

Trace `lr_C` end to end: `apply_extra_axes` → `model_params['lr_C']` → `build_model('PLS-DA', params)`
(does it tolerate or strip the unknown key?) → pipeline `lr` step (`:1692`) → `_capture_serializable_params`
(`:1994`, on a Pipeline this yields `lr__C`, not `lr_C`) → leaderboard `Params` → Tab 7 refit /
`_rebuild_model_from_row` in `search.py` (reads `lr_C`, `search.py:441`) → `model_io` save/load.
If the chain loses the value anywhere, a refit silently uses `C=1.0` and does not reproduce the
leaderboard score. **Acceptance:** a test fits `plsda_head` with `lr_C=1e-2`, rebuilds from the
leaderboard row, and asserts `lr.C == 1e-2` and identical predictions. The same round-trip test
applies to one representative flat bundle (`xgb_child`) (T9).

### 3.3 Downstream consumers of `Params`

For each PR-B bundle key, check that the three consumers accept it: Tab 7 refit,
`model_io` save/load, and the code exporter (`code_generator.py`). The exporter may emit
`gamma=1e-3` into an SVC constructor fine, but `kernel_ext` would break it, which is one
more reason to drop `svm_kernels`.

## 4. PR D — GUI and docs

1. Card "Bayesian — extra hyperparameter axes" in Tab 4C, placed **after** the Bayesian
   trials controls and visually separate from "Advanced Model Options" (which feeds
   `run_search` only). Build it from the registry (`label`, `help`), never a hand-written
   list, so the GUI cannot drift from the backend.
2. `_collect_enabled_extra_axes() -> tuple[str, ...]`, sorted. Pass it at both Bayesian call
   sites (step-0 table has the current lines). The GUI does **not** filter by model; the
   backend skips non-applicable ids (§2.1).
3. Advisory label: `effective_dim = base_axes(model) + Σ applicable bundle axes + shared axes`.
   Show "Consider ~N trials / startup max(20, 3×dim)". Never write to `n_unified_trials`.
   Base-axis counts come from a small hand-maintained table in `search_spaces.py`
   (documentation-only, like `documented_constants`), plus a test that counts actual
   suggests on a recording mock trial per family so the table cannot rot.
4. Expose `n_startup_trials` as an optional spinbox (blank → None).
5. `CAPTURABLE_SETTINGS`: add bundle vars and `n_startup_trials`. Test that a settings
   round-trip restores them.
6. §4.3 **Resume notice:** when `_persistence_mode == 'always'` and a study with the same
   `unified_bayesian_<model>_` prefix exists but with a different config hash, the current
   code says nothing. Out of scope to add a general notice, but PR D should add a line to the
   GUI's resume banner help text: "changing extra axes starts a new study."
7. Docs: add `enabled_extra_axes`, `search_space` and `n_startup_trials` to the
   `run_unified_bayesian` entry in `docs/AGENT_COMPOSITION.md`, with one executed example.
   Update `tests/test_agent_composition_api.py` contract. Soften the `CLAUDE.md` claim.

## 5. PR C / E / F — one-class

- **C:** `if_max_samples` (`max_samples`: cat[`auto`,0.5,0.8,1.0]), `lof_metric`
  (cat[euclidean, manhattan, minkowski, cosine]; **LOF with `cosine` requires
  `algorithm='brute'`; verify `build_one_class_model`**), `ocsvm_poly` (`degree` int[2,5],
  `coef0` float[-1,1]; only meaningful if the base's kernel choices include `poly`, so check
  `:911+`. If they don't, drop the bundle rather than alias the kernel).
- **E, F:** as the ticket describes, §"one-class n_components clamp". E reads
  `cal_model.n_components_`.

## 6. Tests (PR A unless noted)

| id | Test | Guards |
|---|---|---|
| T1 | `apply_extra_axes(..., ())` and `(..., None)` with a trial whose every `suggest_*` and `params` access raises → returns the input unchanged | prime directive |
| T2 | Pinned study name: fixed tiny dataset, `enable_sqlite_persistence='never'`, default kwargs → compute `config_components` via a small extracted pure helper **or** capture `study.study_name` and assert the `<confighash>` segment equals the value captured on `main` before the PR (store the literal) | default identity unchanged |
| T3 | Same-seed default run: `results_df` equal (`assert_frame_equal`, excluding timing columns) between `enabled_extra_axes=()` and omitting the kwarg; also compare to a pickle-free literal of the first 5 `trial.params` captured on `main` | default trajectory unchanged |
| T4 | `n_startup_trials=None` → all three sampler paths get 20 (inspect `study.sampler._n_startup_trials` after a forced 'auto' migration using a monkeypatched fit-time median) | T-41 migration reset |
| T5 | Unknown id → `ValueError` before any storage is touched (the storage URL points to a tmp path; assert the file does not exist) | typo safety |
| T6 | Canonicalisation: reordered/duplicated ids give the same study name; different applicable sets give different names; enabling only a non-applicable bundle leaves the name identical to default | hash |
| T7 | Two trials identical except a bundle value → different fingerprints; identical including it → dedup replay | dedup |
| T8 (B) | Every registered bundle, applied to a recording mock trial after the real `suggest_model_params` for each applicable family and both task types, **never suggests a name twice**; every bundle key reaches `build_model` without raising; one short real run per family shows varying values in `Params` | collision + plumbing |
| T9 (B) | Leaderboard→rebuild round-trip for `plsda_head` and `xgb_child`: identical predictions | persistence (§3.2) |
| T10 (B) | `n_startup_trials=5` honoured after auto-migration | T-41 |
| T11 (D) | GUI collector + settings capture round-trip; the card is built from the registry | GUI |
| — | Existing: `test_unified_bayesian_baseline.py`, `test_agent_composition_api.py`, `test_bayesian_dedup.py`, `test_contamination_detection.py`, `test_simca.py`, `test_cv_pls_clamp.py`, T-41 persistence tests | regression |

Plus the literal check in every PR: `git diff main -- src/spectral_predict/unified_bayesian.py`
shows no hunk inside `suggest_model_params` / `suggest_one_class_params`.

## 7. Step-0 caller enumeration

Swept 2026-09-13 by parsing all Python, the docs code examples and notebooks (there are no
`.ipynb` files). **55 real calls, and none of them goes through `**kwargs`.** Every caller
lists its kwargs explicitly, so a new kwarg only reaches a call site that is edited to pass it.
Neither function is ever monkeypatched, looked up with `getattr`, or pulled in with
`import *`.

| Category | Site | Action for T-51 |
|---|---|---|
| src | `unified_bayesian.py:2560` `run_unified_bayesian` → `create_unified_objective`, explicit 26 kwargs | **PR A: thread `enabled_extra_axes`, `search_space`** |
| src | `_make_tpe_sampler(random_state)` `:2213`, used by `_migrate_study_to_sqlite` (`:2224`, called `:2951`) and the SQLite reattach (`:2791`); inline sampler `:2590` | **PR A: thread `n_startup_trials` to all three** |
| src | `unified_bayesian.py:3547` `__main__` demo; module docstring example `:23` | none (defaults) |
| GUI | `spectral_predict_gui_optimized.py:28332` (one-class), `:28890` (supervised), both in `_run_analysis_thread` (`:26064`), hand-listed kwargs | **PR D: pass bundle ids + `n_startup_trials`** |
| tests | `test_bayesian_dedup.py:258,317`, `test_one_class_varsel_filtering.py:190` (CUO); `test_bayesian_study_lookup.py` (fixed-key `**search_options` fixture), `test_class_weight_sister_sites.py`, `test_cv_anova.py`, `test_cv_pls_clamp.py`, `test_cv_strategy.py`, `test_t19_auto_mode_entry_points.py`, `test_t41_bayesian_sqlite_auto_calculator.py` (8), `test_t42_write_path_plumbing.py` (5), `test_unified_bayesian_baseline.py` (5), `test_varsel_caching_correctness.py` (6) | none. Defaults preserve behaviour, which T2/T3 prove. |
| tests (source/signature introspection) | `test_autoscale_bayesian.py:169` (`getsource(create_unified_objective)` looks for `apply_autoscale`), `test_bayesian_environment_fingerprint.py:152` (`getsource(run_unified_bayesian)`), `test_t44_autoscale_wiring.py:37,44` (signature) | **PR A: re-run. They are brittle to source edits, not to semantics.** |
| tools | `ab_dedup_compare.run_case`, `ab_lv_compare.run_search`, `autoscale_bayesian_compare.run_one_cell`, `bayesian_topk_stability.run_one_bayesian`, `bench_dedup_real.run_phase`: fixed-subset wrappers | none. They are benchmarks of the default path. Leave them. |
| tools | `bench_baseline_compare.py:70` `WORKER_SCRIPT` string, run **against an old checkout** in a subprocess | **must NOT gain new kwargs.** They would raise `TypeError` on the old checkout. |
| docs | `docs/plans/2026-05-07-delete-legacy-bayesian-path.md` (historical harness), T-36/T-41 plan checklists, continuation prompts | none (historical) |
| docs | `docs/AGENT_COMPOSITION.md:385` stable-surface table; `tests/test_agent_composition_api.py:42` import check | **PR D: document the new kwargs** |

**Precedent worth knowing:** the GUI's supervised call already omits `early_stopping_rounds`
and `n_top_regions`, so they silently take their defaults. PR D should not copy that pattern
for bundles.

## 8. Open questions for reviewers

1. **Skip non-applicable ids silently (§2.1)** vs. raising. Silent skipping is needed for
   the GUI's shared selection across a multi-model run, but it hides a scripted caller who
   passes `svm_gamma` for PLS. Middle ground: raise on ids unknown to the registry, and warn
   once per run when *no* enabled id applies to the model.
2. **Drop `svm_kernels` from v1 (§3.1)?** The recommendation is yes.
3. **`n_startup_trials` excluded from identity (§2.2.4).** Agree?
4. **Should PR E bump `__version__`?**
5. **Is a pure `_build_config_components(...)` extraction acceptable for T2?** It touches
   `run_unified_bayesian` (not the sampler bodies), which is a small refactor of working
   code. The alternative is to assert only the captured study name, which needs a
   deterministic env hash in tests (it is deterministic within one venv).
6. Anything in the three-sampler-path finding (§0.1) or the `plsda_head` persistence chain
   (§3.2) that is wrong or incomplete.
