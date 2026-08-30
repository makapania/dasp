# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

## 2026-08-30 - T-51 design: two non-obvious constraints on widening the Bayesian search space

**Context**: a downstream contamination project asked for a way to widen DASP's Optuna
search space (`HANDOFF_2026-08-29_DASP_SEARCH_SPACE.md`). Design work only - no code.
Ticket at `docs/plans/2026-08-30-T51-bayesian-opt-in-search-axes.md`.

**1. You cannot additively widen an axis the base sampler already suggests.**
Optuna raises if the same parameter name is suggested twice in one trial with a
different distribution. So an "add axes after the untouched sampler" design can ONLY
open hyperparameters that `suggest_model_params` pins as constants. Audit:
- Openable (pinned constants): XGBoost `reg_alpha`/`reg_lambda`/`colsample_bytree`
  (`unified_bayesian.py:765-767`), `gamma`/`min_child_weight` (absent); LightGBM
  `min_child_samples`/`subsample`/`colsample_bytree`/`reg_alpha`/`reg_lambda`
  (`:748-753`); RandomForest `max_features` (`:724`); SVM `gamma='scale'` - assigned,
  not suggested (`:788`); PLS-DA logistic head `C` (absent).
- NOT openable (already suggested): Ridge/Lasso/ElasticNet `alpha` (`:698`, `:706`,
  `:713`); MLP `alpha` (`:804`); OneClassSVM `gamma` - `suggest_categorical` (`:843`);
  PCA-SIMCA `n_components` (`:837`).
This killed a planned `linear_alpha_wide` bundle. Found by DeepSeek in peer review.

**2. Clamping a value AFTER `trial.suggest_*` does not change what TPE learns.**
Optuna stores whatever `suggest_*` returned in `trial.params`. Mutating the value
afterwards changes the fingerprint and the reported value, but TPE's KDE is still built
from the original suggestion. So the one-class `n_components` fix splits in two:
- clamp-before-fingerprint + record resolved value -> buys dedup and an honest `LVs`
  column, but does NOT change the search trajectory;
- deriving the ceiling and passing it INTO `suggest_int` is the only thing that changes
  what TPE sees, and that requires editing the sampler body.
An earlier draft of the ticket asserted the first would fix the trajectory. It would not.

**3. `_dasp_version` is already in the Optuna study-identity hash** (`:2563`), so a
version bump already orphans persisted studies. Two reviewers independently proposed
adding a new unconditional schema version for resume safety; unnecessary. The real
requirement is just that behaviour-changing PRs bump `__version__`.

**Also confirmed (not yet fixed)**: `SCALE_SENSITIVE_MODELS` contains `'SVC'` but the
registered classifier family is `'SVM'` (`models.py:281,498`; `model_registry.py:32`),
so classification SVM is fit with no StandardScaler. Sites: `search.py:156` (used
`:464`, `:4962`), `unified_bayesian.py:1534` (used `:1605`), `nsga2_search.py:1388`
(omits both), GUI `:40481`. Prerequisite for any SVM `gamma` tuning.

**Tooling note**: `gpt-5.6` is not reachable from a ChatGPT-account Codex login - it
hard-errors "not supported when using Codex with a ChatGPT account". `gpt-5.5` is the
only model that auth mode can reach. Switching would require API-key auth.

---

## 2026-07-30 - Second-round review of feat/agent-composition-guide: doc examples were wrong

**Context**: the branch's own commit log claimed a Codex + GLM 5.2 round had already
passed. A fresh independent round (Codex + GLM 5.2 via the Alibaba Token Plan route)
found the *guide itself* - the branch's headline deliverable - contained examples that
fail on first use. Do not treat a prior in-branch review claim as sufficient.

**Real, introduced by the branch** (both fixed here):
- `docs/AGENT_COMPOSITION.md` listed `get_uve_threshold` in the score-array family,
  whose documented return is a single `(n_features,)` array. It actually returns a
  3-tuple `(importances, threshold, selected_mask)` (`variable_selection.py:306`).
- `AGENT_COMPOSITION.md` section 8 saved the `MultiClassClassModel` bound in section 6
  under `"model_name": "PLS", "task_type": "regression"`, and never defined `X_new`.
  A false schema loads fine and then predicts down the wrong dispatch path.

**Real but PRE-EXISTING on main** (fixed opportunistically, not regressions):
- `README.md` `SubsetTag` claimed a fixed enum `all, top-20, top-5, top-3`. Actual
  tags are method-dependent: `full`, `top{n}_{method}`, `{method}_top{n}`, interval
  tags. Match by prefix, not equality.
- `docs/MACHINE_LEARNING_MODELS.md` had `get_model('NeuralBoosted', learning_rate=0.2)`
  - `get_model` takes no per-model hyperparameters; this raises `TypeError`.
- `README.md` clone/issues/citation URLs said `yourusername/deepspec`; repo is
  `makapania/dasp`.

**Review-method note**: Codex labelled 8 findings MERGE BLOCKER including items it
itself filed under MEDIUM/LOW, and flagged two pre-existing README/doc errors as
blockers introduced by the PR. GLM declared `AGENT_COMPOSITION.md` clean and missed
both real bugs in it, and flagged its own pre-existing find as a blocker without
checking it against main. **Always diff a claimed blocker against `origin/main`
before accepting it as introduced by the branch.** Both reviewers independently
confirmed the genuinely load-bearing facts: no live reference to the deleted CLI in
installer/.spec/CI/GUI, and the `run_search` 2-tuple docstring is true
(`search.py:4405`).

**Verification**: both fixed examples executed end-to-end against the real 49-sample
bone collagen dataset (UVE kept 142/2151 vars; save/load round-tripped 49
predictions). `tests/test_agent_composition_api.py` 53 passed.

## 2026-07-30 - GOTCHA: verifying from a git worktree silently tests main's src

`.venv312` has an editable install of `spectral-predict` that resolves the package
to a FIXED path: `C:\Users\mspon\git\dasp\src`. Running `python script.py` from a
git worktree therefore imports the MAIN checkout's source, not the worktree's -
silently, with no error.

This produced a false negative during the agent-composition review: a check that
exported a bundle and scanned it for a placeholder URL "failed" after the fix was
applied, because it was exercising main's `export_bundle.py`, not the worktree's.

Confusingly, `pytest` run from the worktree DOES pick up the worktree source (the
rootdir/conftest path insertion puts `src/` first), so tests can pass against branch
code while a plain script in the same directory silently tests main. Do not infer
from "pytest passed" that an ad-hoc script tested the same tree.

**When verifying branch code from a worktree, do one of:**
- `sys.path.insert(0, os.path.join(os.getcwd(), "src"))` at the top of the script, and
  assert the resolved module path contains the worktree directory before trusting the
  result - e.g. `assert "wt-name" in spectral_predict.__file__`
- or set `PYTHONPATH` to the worktree `src/`
- or re-run `pip install -e .` from the worktree (pollutes the shared venv - avoid)

Cheap habit that would have caught it immediately: print `module.__file__` and assert
on it as the first line of any verification script.

---

## 2026-07-30 - Agent-composition branch reviewed, fixed, and MERGED to main (`763c4ed`)

**What happened this session.** Reviewed `feat/agent-composition-guide` (CLI retirement +
`docs/AGENT_COMPOSITION.md` + public `multiclass_varsel_mask`), found and fixed real
defects, merged to `main`, pushed.

**Sequence:**
1. Independent second-round review dispatched: Codex, GLM 5.2, Qwen 3.8. The branch's own
   commit log already claimed a passing Codex + GLM round - it was not sufficient.
   GLM's first dispatch stalled (opencode-go subscription at limit) and was surgically
   re-routed to `glm-alibaba` (Alibaba Token Plan); Qwen ran on the same plan.
2. Fixed 2 real regressions the branch introduced, both in the new guide (`29adaf3`).
3. Fixed 2 Qwen nits (`2a170bb`) - one of which was under-rated as cosmetic and was
   actually consumer-visible (the `"preprocessing"` metadata key).
4. Fixed pre-existing placeholder repo URLs incl. live code in `export_bundle.py`
   (`e36a579`), closing the tracked QUICK_WINS P2.
5. Recorded the worktree/editable-install gotcha (`2c52a48`).
6. Merged `--no-ff` to `main`, pushed, re-ran `pip install -e .` to clear the stale shim.

**CLI: abandoned, user decision 2026-07-30.** Codex recommended keeping `cli.py` for one
release as a deprecation stub printing a migration path; the user declined outright
("we are abandoning cli for now"). Clean removal stands. Do not re-add a console script.

**ACTION REQUIRED ON EVERY OTHER MACHINE** on first pull of `763c4ed` or later: run
`pip install -e .`. `git pull` deletes `cli.py` but leaves
`.venv312/Scripts/spectral-predict.exe` behind, which then raises
`ModuleNotFoundError: No module named 'spectral_predict.cli'`. Full first-pull checklist
is now at the TOP of `docs/PROJECT_STATUS.md`.

**Merge-safety basis.** `main`'s CI is red and has been since ~June 2026 (T-CI-1 rot), so
a green check was not available. Merged on failure-set-diff instead: ran
`test_export_code.py` on the branch AND on the untouched `main` checkout and got an
IDENTICAL 2-failure set. Targeted suites 75 passed. Both corrected doc examples executed
end-to-end against the 49-sample bone collagen dataset; export bundle generated and
scanned for the placeholder URL (zero hits).

**Post-merge check that caught my own sloppiness:** I first reported a matplotlib leak on
`import spectral_predict`. Wrong - my check had also imported `spectral_predict.search`,
which legitimately pulls matplotlib. The bare-import guarantee is intact. Test the exact
claim, not a superset of it.
## 2026-07-29 — Agent-facing API: CLI retired, composition guide added

**Context:** user asked whether the repo is in a good state for AI agents to run analyses headlessly. Investigation reframed the question twice; the plan that survived was collapsed on the user's own argument plus a Codex review.

**The `spectral-predict` CLI was dead, with FOUR independent defects — not the "3-line fix" it looked like.** Verified by running it: (1) `read_csv_spectra`/`read_asd_dir` return `(df, metadata)` tuples while `cli.py:190,196` treated the result as a DataFrame → `AttributeError`; (2) `import sys` inside an `except` block shadowed the module-level import, so the top-level handler raised `UnboundLocalError` and **masked every real error** — the single most turn-wasting defect, hit on three unrelated failures; (3) `lambda_penalty=` passed to `run_search`, which takes `variable_penalty` and has no `**kwargs` → `TypeError` (found by Codex); (4) `run_search` returns a **2-tuple** `(df_ranked, label_encoder)` but the CLI assigned it to `df_ranked` and called `.to_csv()`. Only `--help`/`--version` ever worked, which is exactly what `tests/test_cli_help.py` covered — hence years of silent rot. **Retired rather than repaired**, on the user's argument that a CLI can only encode a fixed analysis shape and *there is no such thing as a standard analysis*.

**Agents deliberately bypass the orchestrators, and that is correct.** An active agent-driven research program (528-combination DD-SIMCA search, Bayesian PLS-DA, 32-page manuscript validated on 948 external objects) never imports `run_search`/`run_one_class_search`. It imports primitives (`simca.MultiClassClassModel`, `contamination.PCASIMCA`, `variable_selection.cars_selection`, `preprocess.build_preprocessing_pipeline`) and writes its own orchestration + `StratifiedGroupKFold` CV. This is the scikit-learn pattern and it is right for research: each project owns its sampling design, splitter, ranking objective, and reporting. **Conclusion: do NOT build an orchestrator or a config-file runner.** An earlier draft plan proposed declaring `__all__` across all modules + `__init__.py` re-exports + a full private-function audit; the user and Codex both rejected it as work that would not change what agents can do. Codex additionally noted top-level re-exports would pull heavy optional deps into `import spectral_predict` and erode its headless-safety.

**Root cause of the fragility that DID need fixing: no declared public API.** `search.py` is 5 public / 24 private; only `model_io.py` (plus `readers/` and `templates/` package inits) declared `__all__`; `__init__.py` exported only `__version__`. So composing a novel analysis *required* reaching into internals — the external pipeline imported the private `search._multiclass_varsel_mask`, which a rename here would have broken silently on another machine. Fix: promoted to public `multiclass_varsel_mask` with `_multiclass_varsel_mask` retained as a delegating alias, plus `__all__` on `search.py` only (selective, not a sweep) and a repo-local contract test.

**Undocumented traps found by actually executing the doc examples (5 of 10 first drafts failed).** Do not write API docs from inspection: (a) `run_search`'s `preprocessing_methods` is a **dict of bools** (`{"raw": True}`), not a list of strings — a list raises `AttributeError: 'list' object has no attribute 'get'`; (b) valid `build_preprocessing_pipeline` names are only `raw|snv|deriv|snv_deriv|deriv_snv` — `snv_deriv1` raises; (c) variable selectors return **importance score arrays** of shape `(n_features,)`, not boolean masks; (d) `save_model` requires metadata keys `model_name`/`task_type`/`wavelengths`/**`n_vars`**; (e) `run_search`'s docstring claimed a bare DataFrame while returning a 2-tuple — docstring corrected. Also: `example/` holds **49** ASD files, not the 37 the README claimed.

**Grouped CV remains a real backend gap (T-15), deliberately not closed.** No search entry point accepts `groups`, and `cv_utils.py` raises `NotImplementedError` for `group_kfold`/`leave_one_group_out`. Agents compose around it with their own splitter, so it is documented in `AGENT_COMPOSITION.md` rather than plumbed — closing it would make `run_search` usable for grouped designs but buys no flexibility that scripts don't already have.

**`interactive.py` / `interactive_gui.py` are now orphaned** — they were imported only by the retired `cli.py`. Left in place (deleting needs explicit permission); their docstrings now say so, since they otherwise read as live API.

**Review lesson — executing examples and reading source catch DIFFERENT classes of doc error. Do both.** Every example in `AGENT_COMPOSITION.md` was executed (17/17 green) and two reviewers still found real inaccuracies that execution structurally could not catch:
- **Codex:** the guide said "all selectors take `(X, y, ...)` and return an `(n_features,)` importance array", then listed `ipls_forward`/`ipls_backward`/`mc_sipls`/`mwpls` among them. Those are a *different family* — `wavelengths` is a required THIRD POSITIONAL arg and they return a **list of subset dicts**. Cause: I verified the four selectors I exercised and generalised the claim to the ones I had only checked were importable. Codex found it by reading `variable_selection.py`.
- **GLM 5.2:** the guide promised `n_select` could be omitted while the signature made it a required positional — following the doc raised `TypeError`. Fixed by making the signature match the doc (`n_select=None`); the body already handled `None`.
- **GLM 5.2 (subtler):** the guide listed `rank` as an always-present key on interval-subset dicts. It is absent on `ipls_forward`'s *combined-interval* entries (`variable_selection.py` ~1855-1862) — my test run stopped early and produced only single-interval entries, so **execution reported the key as always present**. Also `tag` was undocumented. Only source-reading catches a conditional key that a given run happens not to exercise.
- **I also reintroduced the very trap the guide exists to prevent:** my rewrite of `docs/MACHINE_LEARNING_MODELS.md` wrote `df = run_search(...)` three times, right after discovering `run_search` returns a 2-tuple. Caught by Codex.

**Reviewers disagreed on the back-compat alias.** GLM wanted a `DeprecationWarning` wrapper on `_multiclass_varsel_mask`; Codex explicitly argued against (would break warning-strict callers, adds noise). Kept it a silent alias — its whole purpose is to not disturb a live off-repo research pipeline. Revisit only if the private name is actually being retired.

**Pre-existing unrelated failure:** `tests/test_cv_strategy.py::TestPostMergeReviewFixes::test_classification_metrics_template_has_no_nameerror` fails with `NameError: name '_fit_fold' is not defined` — verified identical on `main`, so not from this work.

**GUI tests spawn Tk windows and closing them kills the run.** A full `pytest tests/` background run died at 36% with exit 127 when the user manually closed stuck analysis windows. Run `pytest tests/ --ignore=tests/gui` for background/unattended verification (this is also what the repo's Linux CI does); run `tests/gui` only when someone is expecting windows to appear. Also: don't run the GUI suite at all for a change that touches no GUI code — that was needless here and cost the user manual cleanup.

**"We didn't touch the GUI" is a claim to CHECK, not assume — the GUI imports PRIVATE backend names.** `spectral_predict_gui_optimized.py:30550` does `from spectral_predict.search import _WOLD_METHODS, _multiclass_preprocess_matrix, _multiclass_varsel_mask, build_multiclass_decision_view` and calls `_multiclass_varsel_mask` at `gui:30573` in the decision-view rebuild. A hard rename of that private function would have broken the GUI silently — the back-compat alias added for the off-repo research pipeline is what saved it. Before claiming a backend change is GUI-safe, grep the GUI for the symbol; it reaches past the public surface.

## 2026-07-07 — T-31 PR #64 review fold-ins (GPT-5.5 F1/F2 + Kimi M1/M2)

**Context:** PR review of #64 + Codex GPT-5.5 (medium) independent review, then Kimi K2.7 cross-family review of the resulting fixes.

**GPT-5.5 F1 — malformed multiclass labels abort the whole run (root cause).** Every multiclass entry point derives its class set from `np.unique(y)` on the RAW target. An object-dtype target that mixes types (`1` and `"2"`) or contains `NaN`/`None` makes `np.unique` raise an opaque `TypeError` (unorderable / mixed) BEFORE any per-row failure guard runs — so one messy column kills the entire grid instead of failing one row. Fix: `simca.check_multiclass_labels(y)` — `pd.isna(y_arr).any()` → clear ValueError on missing; `try np.unique except TypeError` → clear ValueError on mixed types. Wired into `MultiClassClassModel.fit` / `.cross_validate` / `.evaluate_novelty` AND `run_multiclass_simca_search` (the last does its own `np.unique(y_np)` before any fit, so it needs its own guard, not just fit's).

**GPT-5.5 F2 — importance varsel with omitted n_select returns an EMPTY leaderboard (root cause).** `variable_selection_n_select` is documented-optional and defaults to `None` → `n_select_list=[None]`. For a mask path (`importance`/`spa`/`uve`/…) `_multiclass_varsel_mask` does `int(n_select)` → `TypeError` → caught as `MulticlassVarselUnsupported` → the loop `continue`s → EVERY row skipped → empty DataFrame with a clean-looking return. GUI never hit this because `_collect_mc_sizes()` floors to `[100]`; it's a programmatic-API-only latent bug. Fix: default `n_select` to `min(100, n_features)` at the top of `_multiclass_varsel_mask` (covers the closure `_mask_from_scores` + the spa/uve_spa branches that also `int(n_select)` — all defined after the reassignment, so they capture the default).

**Kimi M1 — the F2 default must also catch NaN.** The top-decision-view rebuild reads `top["NSelect"]` back from the DataFrame, where pandas has coerced a mixed int/missing column to float64 → an omitted Top-N is `NaN`, not `None`. `if n_select is None` skipped it → `int(NaN)` → ValueError. Fix: `_multiclass_varsel_mask` defaults on `None` OR `(isinstance(float) and np.isnan)`; the top-view path also coerces `NaN→None` before use (matching the run-selected / holdout rebuilds, which already did). NOTE for future: NSelect round-trips through float64 — always guard NaN, not just None, when reading it back.

**Kimi M2 — second unguarded public entry point.** `compute_validation_metrics_for_top_models` (holdout metrics) is separate from `run_multiclass_simca_search` and its multiclass branch reaches `np.unique(y)` unvalidated. Added `check_multiclass_labels(y_train)`/`(y_val)` at its multiclass init.

**GPT-5.5 F3 (deferred, not a bug):** `predict_with_model` + `build_multiclass_decision_view` call `decision_matrix()` then `predict()`, and `predict()` recomputes `decision_matrix()` — doubles scoring cost for wide spectra × multiple engines. Efficiency-only; deferred as a perf ticket (a `labels_from_acceptance(A, classes)` helper reused across `predict`/`predict_with_model`/decision-view).

**Result:** 6 new regression tests in `test_multiclass_search.py`; targeted multiclass suites **116 passed**. Pre-existing unrelated failure `tests/gui/test_multiclass_gui.py::test_tab9_rejects_multiclass_primary` (patches `zipfile.ZipFile` but `load_model` raises `FileNotFoundError` first) — NOT in this diff. User confirmed the live GUI visual pass looked fine → merge cleared.

---

## 2026-07-07 — cp1252 UnicodeEncodeError in fiPLS/interval varsel prints (Fable + Opus impl/review)

Commit `e57b8f4` on `feat/T31-multiclass-simca`. The axis fix below made
fiPLS/interval selectors actually RUN on derivative configs, which exposed latent
U+2192 arrows in `print()` f-strings in `variable_selection.py` lines 1028
(`uve_cars_spa_selection`), 1119 (`fipls_spa_selection`), 1219
(`fipls_cars_selection`) — `UnicodeEncodeError: 'charmap' codec can't encode
character '→'` on cp1252 stdout (Windows console default). Same recurring bug
class as the `.sco`/`.asd` import emoji prints.

- **Encoding gotcha:** R² (U+00B2), ø, ×, • ARE cp1252-encodable — do not "fix"
  those. →, ✓, ✗, ⚠ are NOT. Test with `'ch'.encode('cp1252')` before touching.
- Fix: arrows -> ASCII `->`; display strings only, zero logic/index changes.
- Sweep: search.py multiclass fiPLS dispatch already ASCII; preprocess.py arrows are
  docstring-only; `wavelength_selection.py` ✓/✗ (lines 746/754) live in a
  never-called benchmark helper and `io.py` ⚠️ prints are off the varsel path —
  both left alone (same bug class if those paths ever go live on Windows).
- Regression guard `tests/test_cp1252_prints.py`: AST scan of print/logging string
  literals (incl. `ast.JoinedStr` f-string parts) asserting strict-cp1252
  encodability, PLUS runtime fipls_spa/fipls_cars calls on a 24x50 synthetic set
  under `redirect_stdout` to a strict-cp1252 `TextIOWrapper`. Confirmed RED
  pre-fix (both prongs), green post-fix. cp1252 codec ships with CPython
  everywhere, so it runs on Linux CI too.
- Targeted `test_multiclass_search.py + test_simca.py + new` = 100 passed on a
  FRESH `--basetemp` (reused basetemp gave 2 spurious `TestPersistenceA8` tmp-dir
  collision errors — Windows testing gotcha, not a product bug).

Detail: `.superpowers/sdd/cp1252-print-fix-report.md`.

## 2026-07-07 — T-31 derivative-varsel axis bug FIXED (Fable + Opus impl/review)

Closed the open review finding from the Codex entry below. Commits `4c8ba73`
(fix) + `b9e0c40` (test) on `feat/T31-multiclass-simca`.

- **Root cause:** `_multiclass_preprocess_matrix` returns `(X_pp, wl_trimmed, ...)`;
  after an SG derivative the wavelength axis is edge-trimmed (window 7 → trim
  [3:-3] → 40 cols become 34). Four call sites discarded the trimmed axis and
  passed the full `wavelengths_full` into `_multiclass_varsel_mask` alongside the
  trimmed `X_pp`. Interval selectors that map wavelength indices to columns
  (ipls_forward/backward, mc_sipls, mwpls, fipls_spa, fipls_cars) then produced
  indices up to the full width and indexed past the trimmed matrix →
  `IndexError: index 34 is out of bounds for axis 1 with size 34`. In the main
  search this is caught as `MulticlassVarselUnsupported -> continue`, so the
  derivative+interval rows were SILENTLY SKIPPED (never appeared on the
  leaderboard); rebuild paths could raise. Non-interval methods
  (importance/cars/spa/uve) never take the axis, which is why the existing
  (non-trimming) tests missed it.
- **Fix:** capture the 2nd return value of the SAME preprocess call at each site
  and pass it instead of `wavelengths_full`: `_multiclass_holdout_metrics`
  (`_wl_tr`, was captured then discarded), `run_multiclass_simca_search` main
  loop (`wavelengths_current`, already captured per preprocessing config),
  top-decision-view build (`_top_wl`, was `_`), GUI
  `_run_selected_multiclass_result` (`wl_trimmed`, was `_`). Grep confirmed these
  are the only 4 production `_multiclass_varsel_mask(` call sites. Mask width was
  always `X.shape[1]` = trimmed; only the interval-index axis was wrong, so
  save/reload/predict + `build_multiclass_decision_view` alignment are unchanged.
- **Regression test gotcha (key):** the old failure mode is a caught exception →
  silent skip, so a "does not raise" test PASSES against buggy code. The new
  tests instead assert a leaderboard row IS produced for a deriv=1/window-7 +
  interval-method config and that `full_vars == 34` (trimmed, not full). A third
  test drives the `_multiclass_holdout_metrics` rebuild path. All three confirmed
  RED against pre-fix code before the fix was applied.
- **Review:** Opus implementer (TDD) + separate Opus reviewer (read-only,
  APPROVE, no blocking findings; verified same-call axis/X pairing, no loop
  staleness on `wavelengths_current`, downstream mask consistency). Targeted
  suites `test_multiclass_search/simca/model_io/multiclass_export/
  multiclass_decision_view/gui parity`: **174 passed**. Detail in
  `.superpowers/sdd/varsel-axis-fix-report.md` (gitignored).
- **Non-blocking observation (pre-existing, separate ticket):**
  `variable_selection.py:1119` `fipls_spa_selection` uses a `print(... "→")`
  that would raise `UnicodeEncodeError` on cp1252 Windows console; only surfaced
  on the FULL-axis failure path, harmless on the correct trimmed path.

---

## 2026-07-07 — T-31 review/live-test + merge-gate check (Codex)

Reviewed `f4158d0..HEAD` on `feat/T31-multiclass-simca`, ran the requested
targeted multiclass suites, drove the real ORAU contamination workbook through
the multiclass run paths via the app/backend integration surface, and repeated
the ex-GUI branch-vs-`origin/main` failure-set merge gate. Branch is NOT merged.

- **Real code-review finding:** derivative-preprocessed multiclass rows can hand
  the *full* wavelength axis to `_multiclass_varsel_mask` even after
  `_multiclass_preprocess_matrix` has SG-edge-trimmed `X`. This affects the
  search loop (`run_multiclass_simca_search`), holdout rebuild
  (`_multiclass_holdout_metrics`), and double-click run-selected path
  (`_run_selected_multiclass_result`). Interval/fiPLS selectors that use
  wavelength-derived indices can then raise out-of-bounds on derivative rows
  (`ipls_forward`, `ipls_backward`, `fipls_spa`, `fipls_cars` reproduced on a
  40->34-column derivative probe). Raw rows and score-only methods like CARS/SPA
  are unaffected. Fix should pass the trimmed `wavelengths_current`/`wlc` axis
  into `_multiclass_varsel_mask` everywhere the preprocessed matrix is trimmed.
- **Targeted tests:** requested multiclass command needed `--basetemp
  .pytest-tmp-targeted` because Windows temp roots were access-denied; with that
  equivalent scratch-dir override: **171 passed**.
- **Live workbook exercise:** loaded `Contaminated Samples Raw_ORAU Added.xlsx`
  with target `contamination` (757 rows, 2151 spectral columns; class counts:
  Bone 328, Glyptol 286, PB72 53, PVA 34, Paraffin 29, Animal Glue 27). The
  interactive screenshot API was unavailable in this execution session
  (`CopyFromScreen` handle invalid), so screenshots could not be captured; the
  app/backend paths were exercised with the real workbook and artifacts written
  under `live_gui_artifacts/`. Minimal run produced 1 row; sweep produced 40
  rows across 2 alphas x 2 n_components x 2 engines x CARS/SPA/Wold with Top-N
  collapse; `per_class_cv` produced rows; CARS/SPA/Wold decision matrices,
  non-top `.dasp` save->predict, validation `val_*` columns, CSV/script/notebook
  exports, and Tab-9 rejection were verified.
- **Merge gate:** branch ex-GUI = **5 failed / 2789 passed / 26 skipped**;
  `origin/main` ex-GUI = **5 failed / 2673 passed / 26 skipped**. Failure sets
  are identical (the known T-CI-3/T-CI-4 tests), so branch-minus-main is empty:
  **PASS, zero new failures**. `gh pr view feat/T31-multiclass-simca` found no
  PR for this branch.

---

## 2026-07-07 — Design note: alternative importance signals for CARS (VIP-CARS)

Reviewing the recovered overnight T-17 multi-target CARS work led to a design question: CARS
factors into (1) a per-variable importance signal and (2) a model-agnostic ARS loop; the PLS
coefficient is only *one* choice of signal (CARS-Tree already swaps in LightGBM importances).

Analysis + lit search captured in **[METHOD_DESIGN_vip_cars.md](METHOD_DESIGN_vip_cars.md)**. Key
outcomes:
- **VIP is the standout alternative signal** — near-zero extra cost (reuses the PLS fit already
  computed for RMSECV), non-negative, better on collinear NIR, and the only candidate immune to the
  select-by-X/score-by-PLS coupling problem. Permutation/SHAP/Lasso/MI are traps (cost, or
  methodologically incoherent inside an elimination schedule).
- **Novelty:** canonical CARS = `|PLS coef|` (Li 2009); VIP-CARS (VIP *inside* the reweighting loop)
  appears unpublished, but the "swap the CARS signal" approach has clear precedent in **SCARS**
  (Stability CARS, Zheng 2012).
- **Design decision:** VIP-CARS to be added as a NEW, separately-named method (like CARS-Tree),
  additive only — canonical CARS on PLS1 and CARS-Tree stay untouched. The VIP variant may be
  offered for PLS1 / PLS-DA / PLS2 *as dasp's own named method*, never as a silent change to CARS.
  Requires A/B incl. wall-clock + naming + lit due-diligence before shipping.

## 2026-07-06 — Run-Selected multiclass: mask-based varsel rows opened NO decision view (Fable; Opus review of primary fix)

User report: double-clicking a multi-class leaderboard row to "Run Selected
Result" opened a decision view for Wold/none rows but importance rows "did not
run at all"; suspected broader. **Confirmed broader — ALL discrimination
(mask-based) methods** (importance/cars/spa/uve/uve_*/ipls/fipls_*/ipls_forward/
ipls_backward/mc_sipls/mwpls).

- **Localization was a red herring.** Headless repro on synthetic AND the real
  ORAU workbook (raw + snv_deriv1) showed the mask branch — `_multiclass_varsel_mask`
  + `build_multiclass_decision_view` — SUCCEEDS for every path. The bug is in the
  GUI RENDER, not the resolution.
- **Root cause:** `build_multiclass_decision_view` stores the *resolved*
  `variable_selection` on `view['config']` — a **boolean ndarray** for
  discrimination methods (None for `none`, a string for `wold_*`).
  `_show_multiclass_decision_view` fed that into the header row, and
  `_multiclass_decision_header` did `f"{row.get('varsel_path') or 'none'}"` →
  `ndarray or 'none'` raises `ValueError: ambiguous truth value`. The broad
  render-`except` swallowed it and DESTROYED the half-built Toplevel → user saw
  nothing. Wold/none survive because their config value is truthy-safe.
- **`_mc_worker_running` is NOT stuck** — reset in the success `root.after` lambda
  (setattr before the show call), so the flag is already False when the render
  fails; repeated double-clicks aren't blocked, they just fail identically.
- **Gotcha for future:** any `<value> or default` where `<value>` might be a numpy
  array raises. The config carries the resolved selection (array), not the method
  name — stash the readable name separately if a label needs it.
- **Second defect (Opus review, newly reachable):** the now-openable window hosts
  "Export Repro Script/Notebook", which embed `repr(view['config'])`. For mask
  methods that leaks a bare `array([...])` literal — an undefined name in the
  generated file (imports only `numpy as np`) → `NameError` when the user runs it.
  Latent-but-unreachable before (window never opened for mask methods).
- **Fix:** (GUI) header renders an ndarray as `"mask (N vars)"`; worker stamps the
  method name onto `view['config']['varsel_path']`; header-builder prefers it.
  (code_generator) `_reprsafe_multiclass_config` `.tolist()`s the mask; shared BODY
  template coerces the list back to a bool ndarray (a plain list would be treated
  as an unknown string method by `MultiClassClassModel`). Both script + notebook
  covered. Detail: `.superpowers/sdd/run-selected-varsel-investigation.md`.
- Tests (all red-first): crash-site header test, importance/cars end-to-end
  view+header, generated script/notebook mask reproduction (no `array([`, executes,
  bit-exact). 134 targeted passed.

## 2026-07-06 — T-31 test-flake fix + cosmetic minors + Tab-9 scope (commits `ce96db5`, `afa02ea`)

Fable session, branch `feat/T31-multiclass-simca` (NOT merged). Detail in
`.superpowers/sdd/outstanding-report.md`.

- **A — combined-run flake FIXED (`ce96db5`). The prior diagnosis was WRONG.**
  It is NOT a `src.spectral_predict` dual-import class-identity clash. Real root
  cause: `test_multiclass_gui_parity.py::_install_recording_thread` globally
  monkeypatches `threading.Thread`. The `spa` run-selected path runs
  `joblib.Parallel(backend="threading")`, whose `multiprocessing.dummy.ThreadPool`
  calls the *unbound* `threading.Thread.start(self)` on its own `DummyProcess`
  instances. With the patch active that resolves to `_RecordingThread.start` with
  a non-`_RecordingThread` self, so `super().start()` raised the `super(type,
  obj)` TypeError. Order dependency = joblib's cached ThreadPool (fresh pool only
  re-created under the patch when the search file ran first). Fix: `_RecordingThread`
  records + `super()`s only for genuine instances, delegates foreign DummyProcess
  to the real bound `Thread.start`. **Also unified the namespace as requested**
  (9 GUI + 3 backend `src.spectral_predict` sites → unprefixed): the launcher's own
  comment already documented `src.spectral_predict` "silently failed in dev and
  bundle", so these were latent frozen-bundle bugs; removing them kills the
  dual-module hazard for good, but it was NOT what fixed the flake. Combined run
  now 54 passed both orders (55 with the new collapse test), 132 across adjacent
  multiclass suites.
- **B — cosmetic minors DONE (`afa02ea`).** (1) decision header `varsel: None`
  → `none`; (2) `_on_task_type_changed` guards `mc_varsel_group_frame.pack()` with
  `winfo_manager()`; (3) discrimination-group honesty parenthetical split off the
  Subheading into a Caption label; (4) `run_multiclass_simca_search` collapses the
  Top-N size axis for paths that ignore n_select (`none` + 3 Wold modes) to one
  representative size — new `test_n_select_axis_collapses_for_paths_that_ignore_it`
  pins Wold/none×3 sizes → 1 row, discrimination×3 → 3 rows.
- **C — Tab-9 multiclass side-by-side comparison: SCOPED, not built.** Exclusion
  (`_comparison_reject_multiclass`) stays. A real view needs a new verdict-grid
  data model (per-sample K-vector of p-values + accepted/multiple/novel verdict —
  no scalar to average, consensus-by-R² spine inapplicable), decision-agreement
  metrics (verdict concordance, Cohen's κ, per-class novelty deltas, class-set
  alignment), and a new Tk sub-view (side-by-side verdict table + disagreement
  filter). Backend material exists (`predict_with_model` returns the matrix). Effort
  ≈ Phase D2. First settle whether the verdict-concordance use-case is real; else
  keep the exclusion. Full note in the outstanding report.

**Still owed:** live GUI `run`/`screenshot` pass (needs a display); user merge
greenlight (do NOT auto-merge).

---

## 2026-07-06 — T-31 saved-model consumers wired/verified (commit `cc0f486`)

Closed the "owed next" from the parity feature: verify/wire the 3 saved-model
consumers for a saved multiclass `.dasp`. Fable session, branch
`feat/T31-multiclass-simca`, NOT merged/pushed. Full detail in
`.superpowers/sdd/consumers-report.md`.

- **Predict tab: worked already.** Backend `predict_with_uncertainty` multiclass
  branch (model_io:915) + GUI `_display_uncertainty` `'p_values'` branch
  (gui:43410) + `_display_predictions` string-label path already render the
  decision matrix. Verified e2e via a real save→load→`predict_with_uncertainty`
  round-trip. No code change.
- **Tab 9 (Multi-Model comparison): scoped-and-excluded.** Tab 9 is entirely
  scalar-prediction machinery (consensus by R², numeric/label flag rules,
  applicability-domain distances). A decision matrix has no representation there,
  and letting it flow mislabels the model "(Reg)" and silently drops per-class
  p-values / the novelty decision. New `_comparison_reject_multiclass()` refuses
  a `multiclass_simca` model at load (primary + auxiliary) with an explicit
  "run it on the Predict tab" message. Prefer-exclusion-over-half-build per the
  task guidance.
- **Notebook export: worked already; test hardened.** `generate_multiclass_
  reproduction_notebook` builds a valid v4 notebook; the GUI writes it via
  `json.dumps`. Verified: JSON round-trips, nbformat-v4 structure valid, and the
  concatenated code cells EXECUTE (subprocess) → `decision_matrix.csv`. Added a
  test that executes the cells (prior test only checked the backend call string).
  nbformat/jupyter are NOT installed and not a dep — validated structurally + by
  execution, did not add a dependency.

**Non-obvious findings:**
- **Pre-existing dual-import test flake (NOT a product bug).** In the combined
  pytest session, when `test_multiclass_search.py` runs before the parity file,
  `test_run_selected_accepts_discrimination_varsel` fails with `TypeError:
  super(type, obj): obj must be an instance or subtype of type` on the `spa`
  run-selected path. Root cause = the GUI mixes `from src.spectral_predict ...`
  (9 sites) and unprefixed `from spectral_predict ...` → two module objects, two
  class definitions; test ordering poisons `sys.modules` so an instance from one
  namespace hits `super()` referencing the other. Passes standalone (parity
  20/20) AND in a fresh production-like process (imported `src.spectral_predict`,
  ran `_multiclass_varsel_mask('spa',...)` → 5 vars, no error). Recommend a
  separate test-infra ticket to unify the import namespace. Do NOT chase it as a
  spa bug.
- **Stale test from the parity refactor.** `test_multiclass_task_vars_exist`
  still asserted scalar `mc_alpha`/`mc_n_components`, which `1359d22` replaced
  with preset-checkbox + custom-list collectors → pre-existing red. Fixed to the
  new `mc_alpha_005`/`mc_ncomp_099` vars + `_collect_mc_alpha_list`/
  `_collect_mc_ncomp_list` (same 0.05/0.99/min-10 defaults).

**Tests:** 194 passed (+3 new: Tab9 primary/aux exclusion, notebook exec-repro),
1 pre-existing dual-import flake as above. **Still owed:** live GUI
`run`/`screenshot` pass; user merge greenlight (do NOT auto-merge).

---

## 2026-07-06 — T-31 Multi-Class SIMCA UX-parity feature (Tasks 1-12 + final-review fix)

**User flagged that multi-class didn't behave like the other methods** (config on the Import page instead of the 4A/4B subtabs; variable-count a single spinbox not the multi-size sweep; most varsel methods missing; an auto-popup on run completion). Brainstormed → spec (A-J) → 13-task plan → subagent-driven execution (fresh Opus implementer + Opus reviewer per task, Fable final whole-feature review). Branch `feat/T31-multiclass-simca`, commits `4230fa6..4d0115d`, NOT merged.

**Non-obvious discoveries / decisions:**
- **Variable selection: select-on-calibration is the chemometrics convention, NOT ML leakage.** The initial Task-4 review (my own ML reflex) flagged "compute the varsel mask once on the calibration set → leakage." User pushed back: wavelengths carry fixed chemical meaning, and the major packages (PLS_Toolbox GA, mdatools iPLS, CARS) all select from the full calibration set using each method's own internal CV, then validate — "selection bias" (Andersen & Bro, J. Chemometrics 2010) is handled by an external test set / double-CV, NOT per-fold nesting. So per-fold selection refit is the wrong "fix." Recorded in `feedback_varsel_select_on_calibration.md`. External Validation tab is the honesty check.
- **`compute_importances` only implements importance/cars/uve** — the "full set" required calling `variable_selection.py`'s real selectors directly (`uve_selection`/`spa_selection`/`cars_selection`/`ipls_selection`/`ipls_forward`/`ipls_backward`/`mc_sipls`/`mwpls`/`uve_spa`/`uve_cars`/`uve_cars_spa`/`fipls_spa`/`fipls_cars`), mirroring the standard dispatch at `search.py:3138-3510` for each function's signature (importance-array vs interval-subset return shapes). `cars_tree`/`uve_cars_tree` are no-ops for PLS-based SIMCA (`model_type=None`) → dropped from the GUI list; `vcpa`/`ga` live in other modules → deferred.
- **SPXY is invalid for multi-class** (its `d_SPXY = d_X + d_y` term is Euclidean on the target — undefined for a categorical class label). Disabled for `multiclass_simca`; KS/Random/Stratified are the valid splitters.
- **A same-known-class holdout validates KNOWN-CLASS performance, not novelty** — every held-out sample belongs to a trained class, so the "none of the above" capability needs a held-out class (LOCO) or a true external contaminant. UI carries this honesty note.
- **Cross-task blocker the per-task reviews missed (Fable final review caught it):** Task 8 broadened the GUI's offered varsel set and Task 4 broadened the backend resolver, but the double-click **Run Selected / Save Model** path (`_run_selected_multiclass_result`) still validated `varsel_path` against only the old 5-method set → a user who swept `cars`, saw it rank #1, and double-clicked to save was BLOCKED. Fix `4d0115d`: widened the guard to `set(_MULTICLASS_VARSEL_PATHS) | set(_MULTICLASS_MASK_METHODS)` and resolved the row's mask via `_multiclass_varsel_mask` (mirrors the holdout rebuild), so the double-click model is identical to the ranked row. Lesson: when two tasks broaden opposite ends of a set (UI offers / backend resolves), a THIRD consumer (save/rebuild) can silently drift — the whole-feature review is what catches it.
- **Three rebuilders must agree:** the search row-builder, the holdout-validation rebuild (`_multiclass_holdout_metrics`), and the double-click save path all reconstruct a row's model from its columns — they now all route through `_multiclass_varsel_mask` + `_multiclass_row_to_preprocess_cfg` so they can't drift.

**Owed next (new session):** verify/wire the 3 saved-model consumers for multiclass — Predict tab (backend `predict_with_model` handles it; confirm GUI consumer), Multi-Model comparison Tab 9 (R²/ensemble-oriented — likely needs a multiclass branch or clean exclusion), Colab notebook export (`generate_multiclass_reproduction_notebook` exists — verify runnable). Plus live GUI `run`/`screenshot` (headless only this session), user merge greenlight (do NOT auto-merge), deferred cosmetic minors. See PROJECT_STATUS ACTIVE DIRECTION.

---

## 2026-07-05 — T-31 "run a selected multi-class leaderboard result" + Save Model (.dasp)

**Follow-up gap the user hit on real use, now shipped on `feat/T31-multiclass-simca` (commits `e83f6f9` feature, `e8d2b46` review fold-in; pushed, NOT merged).** Double-clicking a `multiclass_simca` leaderboard row previously routed to the regression/classification Refine tab, where `get_model('pca-simca')` fails — "run selected result" was unimplemented for class-modeling.

- **Interception point:** `_on_result_double_click` early-returns on `model_config["Task"] == "multiclass_simca"` (the ROW's own Task, NOT the live `task_type` radio — see the HIGH below) into a new `_run_selected_multiclass_result(row)`. Single-Y (regression/classification/one-class) Refine path is byte-identical below the branch.
- **Run-selected handler** rebuilds THAT row's exact config: engine/alpha/varsel from the row; preprocessing from the row's Preprocess/Deriv/Window/Poly via `_reconstruct_mc_preprocess_cfg` (mirrors the search's config builder — `method` from the name split on `_w` + deriv-digit strip); n_components policy / floors / baseline / smoothing from a new `_mc_run_config` stash (set atomically with `_mc_export_data` at search dispatch). Builds the in-sample decision view on a worker thread (all Tk via `root.after`; no bare Tk off-thread — both reviewers confirmed clean) and reuses `_show_multiclass_decision_view`.
- **Save Model (.dasp)** button added to the decision-view window → `_fit_and_save_multiclass_model(config, X, y, path)`: fits a `MultiClassClassModel` on the preprocessed matrix and `model_io.save_model(model, preprocessor, metadata, path)`. **Key correctness choice:** the per-spectrum preprocessing pipeline is stored as the model's `preprocessor`, and when an SG-derivative edge mask trims the axis the existing `use_full_spectrum_preprocessing` + `full_wavelengths` handshake is recorded — so `predict_with_model` reproduces preprocessing + edge mask on RAW new specimens (the user's stated goal), not just on pre-preprocessed input. Verified: `_apply_edge_mask_to_data` is a pure symmetric column-trim, so preprocess-full-then-subset is bit-exact vs the trimmed training matrix.
- **Real-data e2e smoke (ORAU Excel, `Site`, 716×269 subsampled):** search → selected the NON-top row (snv) → rebuilt config → decision view (single=410, multiple=271, novel=35) → save→load→`predict_with_model` on RAW spectra reproduced the decision matrix + labels EXACTLY.

**Review fold-in (Codex 5.5 + Kimi K2.7 cross-family, commit `e8d2b46`) — all findings verified before folding (this ticket's panels have produced empirically-wrong findings before):**
- **HIGH (Codex + Kimi convergent):** the router keyed on the live `task_type` radio (`or self.task_type.get()==...`), hijacking a stale regression/classification/one-class row into the multiclass handler after a radio flip. Fixed: route on the row's Task only.
- **HIGH (Codex):** numeric-STRING wavelength labels (common from CSV/Excel headers) were stored verbatim; `predict_with_model` matches wavelengths as floats, so `"1000" != 1000.0` → the saved model failed to predict on its own training data. Fixed: coerce numeric wavelengths + `full_wavelengths` to float on save. Regression test uses `X.columns=["0.0",...]`.
- **MEDIUM (Kimi):** stash-absent silent fallback — reconstruction read baseline/smoothing only from `_mc_run_config`; if absent it would silently reconstruct a no-baseline pipeline. Fixed by enforcing the atomic invariant: require BOTH `_mc_export_data` and `_mc_run_config` (set together by a live search) or bail with a clear message. NOTE: not a backend schema change — the data + run-config are the atomic session artifacts; a future "reload leaderboard CSV then double-click" path would need baseline/smoothing persisted in the row schema (deferred, no such reload path exists today).
- **MEDIUM (Kimi):** unknown `varsel_path` was silently coerced to `None` via `.get()`; backend indexes strictly. Fixed: validate up front + strict index.
- **LOW (Kimi):** `_mc_worker_running` busy flag prevents overlapping workers/windows on repeated double-clicks.
- **Both reviewers confirmed sound:** thread-safety, the save/predict preprocessing handshake, and the config reconstruction field-match vs `run_multiclass_simca_search`.

**Tests (all green):** 23 multiclass-GUI (`tests/gui/test_multiclass_gui.py` — routing both ways, radio-flip untouched, run-selected builds view, missing-stash + unknown-varsel guards, save→load→predict roundtrip float+str × raw/snv/deriv/snv_deriv) + 120 adjacent (simca/model_io/decision_view/export/foldins). LF-clean, single-Y paths byte-identical. **STILL OWED: a live GUI `run`/`screenshot` pass (headless-verified only this session).**

**POST-REVIEW BUG CAUGHT IN INDEPENDENT VERIFICATION (Fable, commit `294d65f`):** the delegated round-trip test (line above) only exercised `predict_with_model(validate_wavelengths=True)`, so it MISSED that `predict_with_model`'s DataFrame branch skips the full-spectrum edge-mask handshake when `validate_wavelengths=False` (the common path): it did `X_new.values` + `preprocessor.transform` with NO subset to the trained wavelengths. For a DERIVATIVE config (SG edge mask trims the axis, e.g. 40→34 cols) the model's per-class StandardScaler then got the full-width matrix → `ValueError: X has 40 features, but StandardScaler is expecting 34`. So the "bit-exact" claim above held ONLY for `validate_wavelengths=True`. **Fix:** the `validate_wavelengths=False` DataFrame branch now honors `use_full_spectrum_preprocessing` (preprocess full → subset to `required_wl`), mirroring the ndarray branch; guarded by the handshake metadata so single-Y paths are untouched (`test_model_io` green). Extended `test_save_multiclass_model_roundtrip` to assert BOTH validate modes AND ndarray input reproduce the decision matrix on raw spectra. **Lesson: a save→load→predict test MUST exercise `validate_wavelengths=False` (and array input) — the True-only path hid a shared-function regression. This is why independent verification of delegated work matters even when the sub-agent reports "roundtrip works."**

---

## 2026-07-05 — T-31 Phase D (GUI + export) built; deferred fold-ins closed; merge-gate pending

**Phase D COMPLETE on `feat/T31-multiclass-simca` (commits `2819090` D1, `1c7c326` D2, `ff8875c` D3, `5af8ec8` fold-ins; pushed, NOT merged).** Built by Fable-5 directly (not delegated — the reconnaissance needed to write contract tests + review a delegate's GUI diff across a 40k-line drift-prone file exceeded the cost of implementing directly; single-Y paths verified untouched at each step).

- **D1 — 5th task radio + controls.** `multiclass_simca` radio; a control panel (global α, per-class `n_components` with the `0.99` novelty-oriented default + int/`per_class_cv` advanced, Wold/importance varsel-path picker, top-N, min class n) in `config_frame`; a per-class engine picker (`mc_models_frame`) mirroring `MULTICLASS_ENGINES` in the models card. Visibility via a new `multiclass_simca` branch in `_update_one_class_controls_visibility` (hides one-class/standard/imbalance widgets, swaps in the engine picker; switching away fully restores standard). Dedicated `mc_*` tk vars — NO reuse of the one-class vars (avoids state bleed).
- **D2 — decision-matrix + Wold results view.** Backend: factored the loop's per-config preprocessing into `_multiclass_preprocess_matrix` (shared) + added `build_multiclass_decision_view()` (fits one config on full data → classes/p-values/accept/labels/resolved-nc/unmodelable/Wold-aggregates/config). `run_multiclass_simca_search` gained `compute_top_decision_view=False`; when True it attaches the top config's view via `df.attrs["top_decision_view"]` (return type stays DataFrame — every existing caller unaffected). GUI: `multiclass_simca` dispatch branch in `_run_analysis_thread` (worker thread, forwards progress_callback + controller), leaderboard via the generic table (`all_vars` hidden), `_show_multiclass_decision_view` = decision-matrix Treeview (novel/multiple tinted) + embedded Wold MPOW/DPOW plots; leaderboard + decision-matrix CSVs auto-saved.
- **D3 — reproduction export.** The class-modeling paradigm (per-class engines → decision matrix, not one estimator + CV score) does NOT fit the generic section pipeline, so instead of threading `multiclass_simca` through header/model/CV/metrics/prediction templates, added dedicated `generate_multiclass_reproduction_script`/`_notebook` that call `build_multiclass_decision_view` with the exact config → decision matrix bit-identical by construction. The view echoes its full `config` so the export is self-describing. Data embedded (base64 float64 + JSON labels) when available. Export buttons wired into the decision-view window; the Refine-based single-Y export path is untouched.
- **Fold-ins closed** (commit `5af8ec8`): (1) `predict_with_uncertainty` multiclass branch (delegated to `predict_with_model` but then treated the dict as an ndarray → now a proper decision-matrix envelope). (2) `_cross_fit_null` bare `except: continue` silently swallowed EE covariance failures on wide spectra → empty null, broken calibration, no signal; now counts failures + warns (all-failed vs partial) with cause + remedy. **A PCA-reduce was REJECTED**: it would desync the null from the actual per-class engine fit. (3) `_multiclass_loco_novelty_auc(oof_cv=...)` reuses the loop's already-computed OOF CV (halves per-config OOF cost; behavior-preserving).
- **Non-obvious audits.** (a) `model_config.get_tier_models("multiclass_simca")` RAISES, but its only multiclass caller (`_on_tier_changed`) wraps it in try/except and multiclass uses the engine picker not tiers → no change needed. (b) `cv_utils.build_cv_splitter` falls through to plain KFold for `multiclass_simca` (only classification stratifies) AND the model uses its own internal CV → not reached. (c) `templates/validation.py` lives only in the generic export pipeline, which the D3 generator bypasses. So the "remaining task-type sites" fold-in was a no-op after verification — do NOT thread `multiclass_simca` into them.
- **GUI verification is headless-only this session** (user asleep, so `run`/`screenshot` live-launch could not be visually confirmed): the app constructs headless, all 5 task-type toggles switch correctly and restore, the decision-view window opens, and the dispatch branch's every referenced attr/method exists. Real-data e2e through the D2 provider: the 757×2151 ORAU set builds a 10-class decision matrix (single/multiple/novel labels, 2151-length Wold aggregates); with 3 sites trained, a held-out Arcy site is 62% novel at ~5% in-sample false-novel. **A live GUI-launch pass is still owed at the merge gate.**
- **Tests added:** `tests/gui/test_multiclass_gui.py` (7), `tests/test_multiclass_decision_view.py` (5), `tests/test_multiclass_export.py` (3, incl. subprocess exec reproducing the decision matrix to 1e-6), `tests/test_multiclass_foldins.py` (4). All green; zero regression across simca/model_io/contamination suites.

**MERGE-GATE RUN (2026-07-05, commit `c2ade23`) — passed, awaiting user greenlight.** Panel: Codex 5.5 + Kimi K2.7 + MiniMax M3 (cross-family) + pr-review-toolkit silent-failure-hunter + pr-test-analyzer, all on `6d6c1a7..HEAD`. **No CRITICAL/HIGH survived; all verified findings folded into `c2ade23`:**
- **Wold diagnostic collapsed to 1 PC (found independently by pr-test, Kimi, MiniMax — triple-converged, REAL):** `build_multiclass_decision_view` passed the model's `n_components` (a variance fraction / `per_class_cv` sentinel) straight into the INT-ONLY `wold_diagnostic_plot_data`; `int(0.99)=0 -> max(1,0)=1` so the default Wold plots used a 1-component subspace, and `per_class_cv` raised -> swallowed -> no plots. Fixed with `_resolve_wold_n_components` (uses the model's resolved per-class components, or a PCA-resolved fraction).
- **Empty-null silent all-reject (silent-failure-hunter, REAL):** a non-SIMCA class whose every calibration fold failed got an empty null -> all-NaN p -> `NaN>=alpha` False -> silently rejected every sample while NOT being in `unmodelable_` (corrupting sensitivity/novelty). Now marked unmodelable + dropped from `models_`. My earlier fold-in only WARNED (stderr, invisible under pythonw); the state change is the real fix.
- **predict_with_uncertainty had no GUI consumer (MiniMax, REAL):** the predict tab's `_display_uncertainty` fell through to the regression formatter and raised on string labels. Added a multiclass branch (decision + accepted classes + per-class p-values). The model_io branch alone did NOT complete the predict path.
- **Export lost sample index + coerced labels to str (Codex, REAL):** now embeds the index + preserves int/float/bool label dtypes.
- **UX silent-failure surfacing:** crash no longer plays the success chime (`mc_failed` flag); no-decision-view logs a warning; n_components 1.0-boundary guarded (was silently 1 component) + `<=0` rejected; decision-view Save/Export callbacks wrapped (Tk swallowed them — file-locked-by-Excel was invisible) in try/except + `messagebox`; Wold-unavailable label; half-built window destroyed on render failure; string wavelengths coerced numeric; validation-set-loaded note; LOCO `oof_cv` docstring.
- **REFUTED (verified false positive, per the "verify before agreeing" rule):** Kimi's "tier-change falls through to regression and mutates model_checkboxes" — `get_tier_models("multiclass_simca")` RAISES `ValueError`, caught before any mutation, so it already no-ops. (Consistent with prior gates producing empirically-wrong findings.)
- **KNOWN-DEFERRED (out of Phase-D scope, noted):** MiniMax LOW-2 (embedded reproduction script ~16MB for the 757×2151 real set — consider a CSV-sidecar option, separate ticket); Refine-tab predict consumer (`_run_refined_model_thread`) likely also lacks a multiclass branch (Refine tab was deferred at D1).
- **DIFF-FAILURE-SET vs `origin/main`:** full ex-GUI suite on HEAD = **5 failed / 2768 passed**; the 5 are the pre-existing T-CI-3/T-CI-4 set (`test_cv_strategy::test_classification_metrics_template_has_no_nameerror`, `test_export_code` ×2, `test_t19_class_weight_per_library` ×2), confirmed failing IDENTICALLY on `origin/main`. **Phase D adds ZERO new failures.** Gate fixes touch only multiclass code + tests (103 green), which cannot affect those 5.
- **STILL OWED:** a live GUI `run`/`screenshot` pass (deferred — user was asleep; verified headless only). **Do NOT auto-merge — await explicit user greenlight.**

**POST-GATE BUG (commit `80e70d7`) — found on the user's first real run: "please select a model" no matter how many engines chosen.** Root cause: the pre-run guard in `_run_analysis` (gui:~23906) collected `selected_models` from the one-class checkboxes (`task=one_class`) or the standard model checkboxes (`else`); `multiclass_simca` hit the `else`, found no standard models checked, and always warned "Please select at least one model to test". The engine picker lives in `mc_engine_vars`, which the guard never consulted (the dispatch in `_run_analysis_thread` re-collects engines itself, so this guard was the ONLY thing reading them for validation). Fix: add a `multiclass_simca` branch collecting from `mc_engine_vars`. **Lesson (reinforces the pr-test-analyzer gate finding): headless widget-toggle tests + backend e2e do NOT cover the `_run_analysis` button-handler guard — the one integration path that runs first. Added a regression test that drives `_run_analysis` with a stubbed worker + mocked messagebox.** The other task-type gates (inlier-label resolution, LOO+classification) correctly skip multiclass, verified.

---

## 2026-07-04 — T-31 Multi-Class SIMCA Phase A (A1–A8) built + 3-family review-gate hardening

**Branch `feat/T31-multiclass-simca` (off origin/main). Phase A backend complete, pushed, not merged.** Execution model: Opus orchestrator wrote each task's contract tests (TDD, confirmed-red first), GLM-5.2 write-mode workers (opencode-call, HALT-OR-BLOCK) implemented to the tests, Opus reviewed every diff + committed per task. New module `src/spectral_predict/simca.py` (`MultiClassClassModel` + `multiclass_simca_metrics`/`wilson_ci`/`novelty_tradeoff_auc`), `PCASIMCA.p_joint` in contamination.py (A1), `predict_with_model` multiclass branch + `_SUPPORTED_TASK_TYPES` gate in model_io.py (A8).

**Non-obvious findings (empirically verified, worth not re-discovering):**
- **DD-SIMCA small-n over-rejection is severe and MoM-driven.** Held-out false-rejection at α=0.05: n=100→~5%, n=30→~8–12%, n=15→~39% (nc=3). `PCASIMCA`'s `var/(2·mean)` method-of-moments χ² fit is high-variance below ~n=30. `min_class_samples=10` was only a *crash* floor (PCASIMCA needs n≥3), NOT a calibration floor. Empirically confirmed in A2; the naive `[0.02,0.10]` false-rejection test band only holds for well-sampled (n≥100) classes.
- **Empirical p-value (conformal, add-one) mathematically cannot reject below m=20.** For non-SIMCA engines the per-class p = `(1+#{null≤s})/(m+1)`, so min p = `1/(m+1)`; at m=10 that's 0.091 > 0.05 → nothing ever rejected regardless of anomaly. DeepSeek's catch; verified live. Rejection first possible at m≥20. Fix: non-SIMCA classes with n<20 marked unmodelable.
- **User-approved layered floor policy:** hard block n<min_class_samples(10) unmodelable; non-SIMCA n<20 unmodelable+warn; SIMCA warns at n<max(20,5·n_comp) but still models; per-class calibration surfaced via A7 metrics + Wilson CIs. Grounded in Rodionova & Pomerantsev 2018 (20–30 samples/class).
- **Scaler-prefit-before-inner-CV is a (mild) leakage.** The per-class StandardScaler was fit on all class rows before `_cross_fit_null`'s KFold, so held-out null rows influenced their own scaler. Fixed: fit a fresh scaler per null-fold for `scaling="per_class"`. (Only Codex flagged; DeepSeek/Kimi considered the engine cross-fit already leakage-free.)
- **NaN (unmodelable) columns silently broke two metrics:** `novelty_tradeoff_auc` collapsed to 0 (NaN comparisons are False, so novel samples never counted as novel) and `efficiency` propagated NaN. Fixed: treat NaN as never-accept (`np.isnan(P) | (P<alpha)`), use `nanmean`.
- **IsolationForest score direction:** `score_samples` is already higher=more-normal — do NOT negate (a spec bug GPT-5.5 caught pre-build; pinned by `test_isolationforest_direction_not_inverted`).

**3-family gate verdict:** core sound (p_joint byte-identical vs pre-A1, all engine signs correct, nested-CV leakage-safe, forward-compat gate preserves legacy None/absent task_type). All findings were small-n/NaN/edge; folded into `cbb69bf` with 6 discriminating tests (`TestPhaseAHardening`). Deferred to merge-readiness: tuning-scaler leakage (negligible, affects only discrete nc choice), `_cross_fit_null` EE-without-PCA-wrapper, `predict_with_uncertainty` multiclass branch (needed before Phase D GUI), AUC threshold downsampling (production-scale only).

**Real-data smoke** (`Contaminated Samples Raw_ORAU Added.xlsx`, 757×2151 FTIR, `Site` = 10 classes): train on Calibrate+Colby, hold out other sites → SIMCA labels 53% (2 known) to 86% (1 known) of held-out-site samples "novel" vs LDA/PLS-DA forcing 100% into a trained site; in-site false-novel ~9%. Raw spectra; Phase C preprocessing will improve it. This is the exact fossil-bone-site-doesn't-generalize use case from the spec.

**NEXT: Phase B** (Wold varsel + diagnostics + supervised prefilter). Then C (search/wiring/e2e), D (GUI/export), merge-gate vs origin/main.

---

## 2026-07-04 — T-31 Phase B (B1–B3): Wold variable selection + supervised prefilter

**Branch `feat/T31-multiclass-simca`, commits 72e6c1c (B1) / 265d687 (B2) / 93d43ea (B3), pushed.** Same execution model (Opus TDD contract tests + review + per-task commit; GLM-5.2 wrote B1 & B3, Opus wrote B2 directly as trivial glue). All additions in `src/spectral_predict/simca.py`; 49 test_simca green (43 → +6 B3), 145 adjacent green, zero regression.

**Non-obvious findings (worth not re-discovering):**
- **Wold discriminating power MUST use RMS-about-zero, not std.** First DPOW implementation used residual *standard deviation* (`resid.std(axis=0)`) for both the cross-class (class-c on model-j) and own-class residuals. That collapsed DPOW to ≈1.0 for EVERY variable — including strongly class-separated ones — and the ranking-stability test (Spearman) fell to ρ≈−0.05 (worse than random). Root cause: a wrong-class PCA model reconstructs class-c rows with the WRONG mean (model_j.mean_ = class-j mean), so the residual carries a large *constant* mean offset per variable; `std` centers that offset out, leaving only within-class scatter (≈ own model's) → ratio ≈1. Classical Wold discriminating power uses the RMS residual **about zero** (`sqrt(mean(resid**2))`) precisely so the mean offset survives. Switching to RMS-about-zero: relevant-feature DPOW jumped to 30–55 vs noise ≈1.5, and consensus-ranking stability rose to ρ≈0.93. (MPOW residuals are naturally zero-mean — PCA reconstruction preserves the column mean exactly — so std==RMS there and either works.)
- **A pure-noise-feature synthetic design makes a stability test meaningless.** With K classes separated on a few features and the rest iid noise, the noise features have NO true DPOW ranking, so their order is random across resamples and dominates a full-vector Spearman (got ρ≈0.53 even with correct RMS DPOW). Fix: use a GRADED between-class separation (`np.geomspace(sep, sep/8, n_features)`, feature 0 strongest, decreasing) so every variable has a distinct, recoverable rank; then consensus-vs-per-resample Spearman median ≈0.93. Empirically probed BEFORE pinning the ≥0.8 band (per continuation-prompt discipline).
- **DD-SIMCA flags Gaussian-blob novels at ~100% regardless — a trivial novelty guard can't catch a broken mask.** For the §9.9 supervised-varsel novelty guard, well-separated synthetic novels give full=supervised=1.0 (gap 0), which passes but wouldn't detect a mask that selected noise. Made the guard discriminating by building the external novel set as a MIXTURE (56 far-novel + 24 genuine class-2-like inliers) → non-trivial baseline novelty ≈0.73; supervised (RF-importance top-12) came out ≈0.74 (|gap|<0.03), so no degradation, pinned at tolerance 0.10.

**Scope decision (surface at gate / to user):** the model-layer `variable_selection` supervised path ships **`"importance"` only** (RandomForest importances on the genuine multi-class label, `task_type="classification"`). The plan's fuller list (`spa`/`cars`/`cars-tree`/`ga`/`vcpa-iriv`) raises `NotImplementedError` at the model layer — those PLS-based methods treat integer multi-class labels as regression targets and belong to the C search-layer enumeration, not a per-model prefilter. `importance` is sufficient to satisfy the novelty gate (the actual B3 deliverable).

**Phase-B multi-family gate:** Codex 5.5 (high) + Kimi K2.7 + MiniMax M3 (rotated away from GLM which wrote the code). **Verdict: no BLOCKER; core Wold math + leakage discipline sound.** Fold-in commit `eb72c05` (307/-49). Convergent real bugs fixed with red-first discriminating tests:
- **Wold varsel crashed on a below-floor class** (Codex HIGH + Kimi M3, both reproduced): power estimation ran on ALL classes before the unmodelable floor → a small class hit `KFold(n_splits=1)`. Fix: `_varsel_modelable_subset` restricts varsel to classes with ≥ `max(min_class_samples, n_components+1)` rows; the small class is still preserved as an unmodelable decision-matrix column. + n<2 guard in `_wold_cross_fit_own_rms`.
- **Empty/invalid `n_select`** (Kimi H1/L8): `n_select≤0` and empty masks now raise a clear ValueError at fit, not a confusing 0-feature DD-SIMCA error deep downstream.
- **Non-finite supervised importances** (Kimi #7): raise instead of an arbitrary `imp≥mean(imp)` mask.
- **Wold PCA `random_state` pinned** (Kimi H2): deterministic mask even under randomized-SVD (n>500).
- **Precomputed boolean-mask hook** added: `variable_selection` accepts an `(n_features,)` bool array (`varsel_path_="precomputed"`) — the C search layer can wire ANY supervised method by computing the mask externally, cleanly closing the importance-only scope gap.
- Test hardening: novelty guard now multi-seed(10) PAIRED across `n_select∈{5,12,20}` (deterministic — RF importances seeded via `build_model` `random_state=42`; mean-gap ≥ −0.05); leakage pin spies `wold_variable_selection` (one call/fold, fold-train rows only).

**Pushed back after verification (receiving-code-review discipline — verify, don't performatively agree):**
- **MiniMax H1 "switch DPOW to std":** my probe empirically REFUTES its core claim. On mean-separated data std → Spearman ρ≈−0.05 with DPOW≈1 everywhere (the between-class mean offset is centered out); RMS-about-zero → ρ≈0.93. "Residual standard deviation" in the SIMCA sense IS RMS-of-residuals. Kept RMS; reworded the docstring (the old "std would collapse" line was loosely worded). Surfaced as a methodology decision.
- **MiniMax H3 "make own/cross symmetric (full model for both)":** contradicts spec §5.6 (own_rms is pinned cross-fit) and would REINTRODUCE the larger in-sample optimism. Kept per spec; documented the deliberate asymmetry.
- **MiniMax L2 / Kimi "supervised RF `random_state` unpinned":** FALSE POSITIVE — `models.build_model` hardcodes `random_state=42` for `RandomForestClassifier`, so `compute_importances('importance','RandomForest')` is already reproducible (verified).

**Surfaced to user (methodology, not unilaterally changed):** (1) DPOW RMS-vs-std (recommend keep RMS); (2) MPOW `ddof=0` upward bias at small n — documented, dof-correction deferred; (3) `balanced = MPOW·DPOW` is DPOW-scale-dominated (M2) — normalization deferred pending user pick of min-max vs rank vs percentile-cap; (4) **supervised-varsel adversarial-novelty limitation** — a novel class distinctive ONLY on low-importance (discarded) features can be missed by the supervised prefilter; the strengthened guard covers representative (not adversarial) novels; (5) importance-only model-layer scope (precomputed-mask hook is the extensibility answer for C). **[UPDATE post-user 2026-07-04: (1) keep RMS confirmed; (3) min-max normalization implemented (commit 748289c); (2)/(4)/(5) documented/accepted.]**

---

## 2026-07-04 — T-31 Phase C (C1 + C2/C3): search + task-type wiring; 3-family gate + real-data e2e

**Commits `69a4750`(C1) / `04e4e04`(C2/C3), pushed.** C1: `multiclass_simca` threaded through `scoring.create_results_dataframe` (dedicated NoveltyAUC/Efficiency/… schema + `engine_family`/`varsel_path` tags) + `compute_composite_score` (ranks by α-sweep NoveltyAUC, tie-break MinClassN) + `model_registry.MULTICLASS_ENGINES`. C2 (Opus subagent, Opus-orchestrator reviewed): `run_multiclass_simca_search` (+506/−0, existing paths byte-identical) — grid = preprocessing × engines × varsel_paths (NO G^K; per-class n_components auto-tuned inside each row); per-row OOF metrics via `cross_validate` + LOCO `NoveltyAUC` ranking. 66 tests green.

**HEADLINE real-data finding (user's `Contaminated Samples Raw_ORAU Added.xlsx`, `Site` classes — MUST NOT re-discover):** `n_components="per_class_cv"` (the A5 default, tuned by one-vs-rest **balanced accuracy** on known classes) is **misaligned with novelty detection** and cripples it on real held-out sites. Held-out Arcy site flagged novel: **n_components 3→6.9%, 5→69%, 10→100%, 20→100%, but per_class_cv (tuned {Calibrate:7,Colby:3,Shanidar:3})→17.2%** — all at ~6% in-sample false-novel (well-calibrated). Root cause: one-vs-rest balanced-accuracy tuning optimizes within-known *discrimination*, picks too-few components → loose per-class models → poor novelty at α=0.05. The spec §5.4 caveat guessed this trade-off went the *other* way; real data shows the opposite, severely. **Options surfaced to user:** (1 recommended) novelty-oriented n_components selection (tune to max LOCO novelty/false-rejection AUC); (2) fixed n_components / variance-explained default; (3) keep per_class_cv + document. **Awaiting user decision (this contradicts a LOCKED A5 decision, so not changed unilaterally).**

**e2e functional gate PASSED:** search runs (2 configs/18s), NoveltyAUC ranks pca-simca(0.90) > ocsvm(0.74); **save→load reproduces the decision matrix exactly (max|Δ|=0)**; LDA baseline forces **100%** of held-out Arcy into a trained site (the flagship "can't abstain" contrast).

**3-family gate (Codex 5.5 + Kimi K2.7 + MiniMax M3, rotated off GLM/subagent) — NO BLOCKER; MiniMax proved the degenerate flag-everything config scores AUC=0.5 and cannot win.** Consolidated findings → **fold-in queue (bugs, unambiguous):**
- **LOCO AUC endpoint collapse** (Codex H1, empirically repro'd): perfect separation (own-p all 1.0) → false-rejection axis has no range → trapezoid 0.0. Fix: dedupe duplicate false_rej x by max novelty + span [0,1].
- **NaN / additive-tiebreak ranking** (Codex H2/M1 + Kimi M5, repro'd): a NaN-NoveltyAUC row with large MinClassN ranks #1 via the `−1e-9·MinClassN` additive term; and the term can flip a real sub-1e-9 AUC gap. Fix: lexicographic (NaN last → AUC desc → MinClassN desc), not additive.
- **All-NaN leaderboard no guard** (Kimi H1/H2): every-config-fails returns a plausible `Rank=1`-everywhere frame; add a warning/sentinel.
- **Vacuously-novel inflation** (MiniMax M4): a held-out row with all-NaN foreign p (all K−1 unmodelable) → `-inf` → counted novel at any α, inflating novelty for configs with many unmodelable classes. Fix: `np.nan` + exclude from num+denom.
- **Off-schema columns** (Kimi M3): C2 emits `unmodelable_classes`/`reason` not in the C1 schema → declare them.
- **Malformed `preprocess_configs` aborts whole search** (Codex M2): pipeline build sits outside the NaN-guard.
- **`PCASIMCA` PCA unseeded** (Kimi M6): add `random_state` (extends the Phase-B Wold pinning; matters >500-sample classes).
- Docstring x-axis relabel (MiniMax M1 — trapz invariant, ranking unaffected), threshold cap 500→2000+log (MiniMax M3), verbose multiclass display (Kimi L7), + adversarial-edge tests (Codex L1, MiniMax M5).

**Methodology decisions surfaced to user (interlock — NOT changed unilaterally):** (A) **LOCO is a within-dataset novelty PROXY that OVER-estimates the §1 held-out-4th-class target** (MiniMax H1: K−1 ruling models + in-distribution held-out class vs §1's K ruling + foreign class) — document + K=4 quantifying test; explains why NoveltyAUC=0.90 coexists with 17% operating-point Arcy novelty. (B) **novelty_rate is sample-weighted, not class-balanced** (MiniMax H2) — undermines the spec's "small-class-robust" claim on imbalanced Site data; recommend class-balanced default. (C) **composite is single-objective on NoveltyAUC** (MiniMax M2) — a config that catches all novel but destroys known-class discrimination outranks a balanced one; option `NoveltyAUC·Efficiency^0.5`. (D) the per_class_cv novelty finding above. (E) `variable_penalty` uses full-fit `n_vars` (Kimi M4, minor at default 0). **Leakage clean per all three (only the M4/variable_penalty note); no fall-through.**

**FOLD-IN DONE (commit `4bf39db`, user-approved A/B/D; C unchanged).** 13 red-first tests, 180 green, existing paths byte-identical. Bugs B1–B9 all fixed (LOCO endpoint anchor+dedup, lexicographic NaN-last ranking, all-NaN guard, vacuous-novel exclusion, off-schema cols declared, preprocess-abort guard, PCASIMCA `random_state=0`, docstring/threshold/verbose). **Decision B:** novelty_rate now CLASS-BALANCED. **Decision D (the fix that matters):** `n_components` accepts a float in (0,1) = per-class variance fraction (passed through to PCASIMCA, resolved int recorded); `run_multiclass_simca_search` defaults to **0.99** — **real-data e2e re-verified: held-out Arcy novelty 17%→100%** at ~7% in-sample false-novel, per-class resolved nc {Calibrate:7,Colby:9,Shanidar:7}; also faster (no one-vs-rest CV tuning). per_class_cv kept, relabeled discrimination-oriented. **Decision A:** LOCO documented as an optimistic within-dataset proxy. **Decision C:** unchanged (NoveltyAUC-primary; `NoveltyAUC·Efficiency^0.5` noted as the undecided alternative).

**Phase C COMPLETE.** NEXT: Phase D (5th GUI radio + engine/α/n_components/varsel controls; decision-matrix + Wold-diagnostic results view; code export) → merge-gate (whole-diff multi-family + pr-review-toolkit + local diff-failure-set vs origin/main; user greenlight only). Deferred fold-ins still open for D (predict_with_uncertainty multiclass branch; EE PCA-wrapper in `_cross_fit_null`; remaining task-type sites model_config/cv_utils/code_generator/templates; the double-CV perf optimization in the LOCO helper for real-data scale).

---

## 2026-07-01 — Legacy `.sco` review fold-in: broad-except swallowed corruption guards; GUI detection globs drift; centralized ASD extensions

**Context.** High-effort code review of the `feat/legacy-asd-sco-import` branch (the native legacy
float32 ASD/`.sco` reader) surfaced 4 real findings, folded in this session and cleared by Codex 5.5.

**1 — Broad `except Exception` defeated the reader's "raise loudly" contract.** `read_legacy_asd`
deliberately raises `ValueError` for a file that carries the legacy `ASD\x00` magic but has an
inconsistent header (implausible channel count, bad wavelength axis, or a size matching neither the
float32 nor float64 layout — i.e. corrupt/truncated). The point is to *not* let real corruption be
misread as "modern format, hand to SpecDAL". But `read_asd_dir`'s per-file loop caught that
`ValueError` in a blanket `except Exception as e: print(warning)` and skipped the file — so a corrupt
legacy folder imported as "No valid spectra could be read" (no hint of corruption), or dropped
samples silently from a partially-corrupt folder. **Fix:** capture `binary = _is_binary_asd(asd_file)`
*before* the `try`; in the handler `if binary: raise` (propagate loudly) else keep printing a warning.
ASCII stays tolerant on purpose — one junk `.sig`/text file shouldn't abort a whole folder — but a
binary/legacy decode failure is a real signal the user must see. `io.py:~715-737`.

**2 — GUI directory-detection globs drift from backend support.** The GUI gates on its *own* glob of a
folder before it ever calls `read_asd_dir`. The original `.sco` PR updated 6 such globs but missed
the **Calibration Transfer tab** (4 sites) plus a few other tabs and the export-naming `folder_ext_map`
registry — all still `*.asd`/`*.sig` only. Net effect: legacy `.sco` folders showed "No spectral files
detected" on those tabs despite the backend handling them fine. This is the *second* time an ASD
extension addition missed a GUI site (Kimi caught one in the original PR). Root lesson: the extension
list was duplicated ~13 times across the GUI + `io.py`, so every addition is a find-every-copy hazard.

**5 — Centralized to kill the recurrence.** Added `io.ASD_EXTENSIONS = (".asd", ".sig", ".sco")` and
`io.list_asd_files(directory) -> list[Path]` (case-insensitive via `suffix.lower()`, sorted,
non-recursive `Path.iterdir`) as the single source of truth. Replaced all ~13 hand-maintained globs:
folder-enumeration sites use `list_asd_files`; suffix-membership checks and the registry use
`ASD_EXTENSIONS`. **Path-vs-string gotcha:** some original sites used `glob.glob` (returns `str`) and
some `Path.glob` (returns `Path`); `list_asd_files` always returns `Path`. Verified (and Codex
re-verified) every converted site only uses the result for truthiness/`len`/iteration/suffix-membership
or explicitly `str()`-wraps it (e.g. `str(asd_files[0])`, `str(f)` in the registry loop) — nothing
relied on `str` elements. GUI imports the two symbols once at module top (`spectral_predict.io` is
already in the top-level dependency graph via `search_controller`, so no new circular-import risk).

**3 — Reworded error message broke a skipped test on CI only.** `read_binary_asd`'s
`NotImplementedError` had been reworded and lost the substring `not yet implemented` that
`tests/test_optional_r_bridge.py::test_native_reader_not_implemented` asserts via
`match="Native Python.*not yet implemented"`. That whole test file is `skipif(not check_r_available())`,
so it's silently skipped on the dev Windows box (no Rscript) but FAILS on any CI runner / machine with
R in PATH — a red test the branch never observed. **Fix:** reworded to "Native Python binary ASD
reader: modern (as5-as8) float64 files are not yet implemented. Only the legacy float32 ASD-v1 format
… is supported." Contains both anchors in order; verified the regex matches live.

**Follow-up (same day) — fault-tolerance policy reversed after a PR-review pass.** A medium-effort
PR review of #61 flagged that finding #1's `if binary: raise` was a footgun: one corrupt `.sco` in a
folder of 30 aborted the *entire* import (loaded 0), and `_file_equalize_batch` amplified it to
dropping a whole instrument. Per user direction ("one corrupt file should not stop the whole folder;
just mention skipped files"), `read_asd_dir` now **skips** an unreadable/corrupt file (binary OR
ASCII), collects the reasons, and prints a `[!] Skipped N ...` summary after the loop; it hard-raises
only when *zero* files load, and that error now carries the specific reason(s) instead of the generic
"No valid spectra could be read". The `UnicodeDecodeError` fallback is now nested around just the
ASCII read so a fallback failure also lands in the skip path. **Windows gotcha:** the summary `print`
must be ASCII (`[!]`, not `⚠️`) — a `⚠️` on a cp1252 stdout raises `UnicodeEncodeError` and would abort
`read_asd_dir` in the very corrupt-file path this hardens (the pre-existing duplicate-stem `⚠️` prints
have the same latent bug). Not folded in (out of the narrowed scope): the truncated-below-484-bytes and
all-NaN-payload files still route to SpecDAL / enter the matrix silently rather than being skipped —
noted as optional future hardening. Tests: `test_read_asd_dir_skips_corrupt_and_continues` +
`test_read_asd_dir_all_corrupt_raises_with_reason` added.

**Second review round (Codex 5.5 — note: 5.3/5.2 are rejected by the ChatGPT-account codex auth, so it fell back to 5.5).** Four more findings folded in: (1) HIGH — `_normalize_filename_for_matching` (io.py:460) stripped `.asd/.sig/.csv/.txt/.spc` but not `.sco`, so a reference table keyed on `sample.000.sco` failed alignment against spectra indexed `sample.000`; fixed by deriving the ASD part of the list from `ASD_EXTENSIONS` (can't drift again). (2) The new skip-and-report loop only recorded a skip when an exception *escaped* — a `None` return from `_handle_binary_asd` (SpecDAL failed/unavailable, or a sub-484-byte legacy file that returns `None` then SpecDAL-fails) was silently dropped and omitted from the `[!] Skipped` summary; `read_asd_dir` now treats `spectrum is None` as a reported skip. (3) all-NaN payloads: `read_legacy_asd` now raises `ValueError` when a decoded spectrum is *entirely* non-finite (partial NaNs still tolerated — real spectra can have isolated bad channels), so a dead `.sco` is skipped+reported instead of entering `df` as an all-NaN row. (4) LOW — the converter's "No spectra could be decoded" now lists the collected per-file reasons. Tests: renamed `test_nan_payload_does_not_crash` → `test_partial_nan_payload_still_decodes`; added `test_all_nan_payload_rejected` + `test_read_asd_dir_skips_all_nan_spectrum` (23 ASD tests pass; 136 align/match tests pass).

**Review trail.** All 4 fixes locally verified (19 ASD tests pass; corrupt-`.sco` re-raise and the
regex match confirmed by a direct repro script). Handed the uncommitted diff to Codex 5.5 (gpt-5.5,
high reasoning, read-only) — **no findings**; it statically checked every converted site for the
Path-vs-string hazard and confirmed no circular import. Caveat: Codex review was static + import-level;
it did not launch the GUI or run the R-gated test (out of read-only scope), both covered directly.

**Housekeeping note.** This file is ~1450 lines — well over the ~200-line archive threshold in
CLAUDE.md. Older entries should be moved to `SESSION_LOG_ARCHIVE.md` in a dedicated housekeeping pass.

---

## 2026-06-19 — Radio-button data-type toggle silently log-transforms plots (stale `use_absorbance`)

**Symptom.** User imports a (CSV/XLS) reflectance file; the Import & Preview plots look like
**absorbance** even though the radio reads **Reflectance**. Clicking the **Absorbance** radio makes
the plot snap to a reflectance shape — i.e. the radio appears to *convert* the data, which it must
never do (radios are relabel-only).

**Root cause.** Not the importer — `read_combined_csv`/`read_combined_excel` and
`detect_spectral_data_type` never touch the values; detection only sets a *label*. The transform
lives in the hidden legacy `use_absorbance` flag. Every plot generator runs the data through
`_apply_transformation()` (`spectral_predict_gui_optimized.py:20820`), which applies
`A = log10(1/R)` iff `use_absorbance` is True AND `data_has_been_converted` is False AND
`current_data_type != "absorbance"`. `use_absorbance` is set True only by clicking **Convert to
Absorbance** (`:19588`), but it was **never reset on a new file load** — `_apply_data_type_metadata`
reset `data_has_been_converted` but not `use_absorbance`, and the `_update_data_type_status_ui`
reflectance branch (`:19829`) enabled the legacy checkbox without resetting the flag (only the
`else` branch reset it — an asymmetry). So after any prior conversion, the next import kept
`use_absorbance=True` → phantom log transform in plots while the radio still says Reflectance.
Toggling to the Absorbance radio short-circuits at `:20831` (`current_data_type=="absorbance"` →
return raw), revealing the true reflectance shape. The shape-change-on-toggle is the tell that it's
this flag, not a detection mislabel (a mislabel changes only the y-axis text, not the curve).

**Fix.** Reset `self.use_absorbance.set(False)` in `_apply_data_type_metadata` (canonical
load-reset point, alongside `data_has_been_converted = False`), and made the `:19829` reflectance
branch reset the flag too so the invariant "fresh unconverted load ⇒ `use_absorbance` False" holds
locally. The legacy checkbox is created but never `.pack`ed (hidden), so this only clears stale
cross-load state — no user-facing workflow changes.

---

## 2026-06-19 — Legacy float32 ASD-v1 (.sco) files read as all-NaN because SpecDAL assumes float64

**Root cause.** A user's `.sco` / numbered `.000` FieldSpec files wouldn't import; renaming to
`.asd` made them load but every value came back NaN. These are the *oldest* ASD binary format —
version string `b"ASD\x00"` — which stores the spectrum as **float32** at offset 484
(file size == `484 + channels*4`; 9088 bytes for 2151 bands). The app reads binary ASD via
**SpecDAL**, which assumes the modern layout (float64 at the same offset), so it reads past the
data into garbage → all-NaN. Two independent failures stacked: (1) `read_asd_dir()` only globbed
`*.sig`/`*.asd`, so `.sco` was never picked up at all; (2) even renamed, SpecDAL mis-decoded it.

**Gotcha — `raw[:3] == b"ASD"` is NOT a safe binary discriminator.** ASCII `.asd`/`.sig` files
start with literal text `ASD Field Spec Pro`, so their first 3 bytes are also `ASD`. The binary
magic is 4 bytes `ASD\x00` (`_is_binary_asd` checks 4). Header fields (little-endian): first
wavelength `float@191`, nm/channel `float@195`, channel count `uint16@204`, dataType `byte@186`
(1 = reflectance).

**Fix.** New `readers/asd_native.py::read_legacy_asd()` decodes the float32 layout natively and
is tried *before* SpecDAL in `_handle_binary_asd()`. Discriminator is exact file size:
`484 + channels*4` → decode float32; `484 + channels*8` → return `None` (modern, hand to SpecDAL);
neither → raise `ValueError` (corrupt/truncated, so real corruption isn't silently treated as
"modern, try SpecDAL"). `.sco` added to the glob, `format_map`, `detect_format` magic branch, and
`_detect_directory_format`. One-off converter at `scripts/convert_old_asd.py` reuses the decoder.

**GUI gate gotcha (caught by Kimi cross-family review).** The backend fix alone is NOT enough:
the Tkinter GUI does its *own* directory globbing to decide "Detected N ASD files" at six sites
(`spectral_predict_gui_optimized.py` ~16072 Import-tab auto-detect, ~17663 load path, ~41276
prediction tab, ~44994 sample-ID ext list, ~45612 + ~45685 dir-load helpers) and gates on that
*before* `read_asd_dir` runs. All six globbed `*.asd`/`*.sig` only, so a `.sco` folder showed
"No supported spectral files found" and never reached the backend. Added `.sco` to all six.
Lesson: format support in `io.py` is necessary but not sufficient — the GUI has parallel
detection logic that must be updated in lockstep.

**Variant decision.** `.sco` and the bare `.000` companions decode to near-identical reflectance
(differ ~0.001–0.003) — duplicate measurements of the same scan, not distinct types. Standardized
on `.sco`-only import: stems (`italy.000`, `italy.001`…) are unique, whereas every bare
`italy.000…029` has stem `italy` and would collapse 30 spectra into 1 row. Bare numeric files
still classify as OPUS in `detect_format` (intentionally untouched). Real-folder result:
30 files × 2151 bands, 0 NaN, reflectance 0.06–0.97, 100% reflectance confidence.

---

## 2026-06-16 — X-unit radio is relabel-only, so it now relabels plots in place instead of full-regenerating them

**Perf fix.** The Import-tab nm/cm⁻¹ radios are *declarative* (`_on_x_unit_override`) — they
correct a mis-detected unit and never convert values (the 1e7/x converter lives behind the
hidden `Convert to…` button, disabled in T-21 because reciprocal regrid breaks SG derivatives).
The handler nonetheless called `_generate_plots()` + `_generate_explore_plots()`, tearing down
and rebuilding ~11 figures (re-running SG 1st/2nd-deriv transforms + one Line2D per spectrum
across 3 Import tabs + 8 Explore plots) on every click. Since a relabel changes no data/geometry,
all that was wasted — only the x-axis label text needs to change.

**Mechanism.** Added `self._spectral_x_canvases` registry; spectral creators register their
canvas (`_create_plot_tab`, `_create_explore_plot_in_frame`, `_init_manual_baseline_plot`).
`_relabel_spectral_x_axes()` swaps the xlabel in place on every registered live canvas (axes
self-identify by current label ∈ {"Wavelength (nm)","Wavenumber (cm⁻¹)"}), then `draw_idle()`.
`_on_x_unit_override` calls that instead of the two `_generate_*`.

**Gotchas / lessons.**
- `ax.set_xlabel(text)` with no fontdict/kwargs preserves existing font size/color — so the
  in-place relabel keeps per-axis styling (Import/Explore 12pt, manual baseline 10pt). Verified
  against mpl 3.10.8.
- **Registry liveness (Codex gpt-5.5-high MEDIUM, fixed before commit):** pruning only inside
  the relabel method leaks destroyed `FigureCanvasTkAgg`/figures when the user regenerates plots
  *without* toggling units (filter/exclude/reload register fresh canvases, dead ones linger).
  Fix: `_register_spectral_canvas()` prunes dead canvases (via `winfo_exists()`) and dedups on
  every call, keeping the registry bounded to live canvases. `_canvas_is_alive()` shared helper.
- Exact-string label matching is internally consistent with `_get_spectral_xlabel()` today but is
  fragile if the wording/superscript ever changes — a future axis-metadata flag would be sturdier.
- Scope matches the old behavior exactly: only Import + Explore plots refresh on toggle. Other
  tabs (Model Dev / Contaminant / CT) were never refreshed by the override and still aren't —
  widening registration app-wide is a noted optional follow-up.

---

## 2026-06-16 — Wavelength Importance click-popup hardcoded "nm" unit (ignored cm⁻¹ x-unit)

**Bug.** Model Development → Results → Wavelength Importance figure: clicking a point opens an
annotation whose first line was `f"Wavelength: {wl:.1f} nm"` (hardcoded). The figure axis itself
already routes through the unit-aware helpers, so in cm⁻¹ mode the axis read "Wavenumber (cm⁻¹)"
while the popup still said "Wavelength … nm". Value was correct (the `wavelengths` array is stored
in display units and feeds both `ax1.stem(...)` and the popup `wl`); only the label was wrong.

**Fix.** `spectral_predict_gui_optimized.py`, `on_importance_click` in `_plot_wavelength_importance()`
(~line 34987): use `self._get_x_axis_name()` + `self._get_x_unit_short()` instead of the literal
"Wavelength … nm". Helpers defined at lines 19628–19647, driven by `self.current_x_unit`.

**Audit — same bug class found in 3 more places (all fixed same commit).** Grepped the GUI for
`nm` and cross-checked each against whether its plot's `set_xlabel` already routes through
`_get_spectral_xlabel()` (i.e. is unit-toggle-aware). Confirmed siblings, all now using the helpers:
- **Predictor-screening listbox** (`_update_screening_info_panel`, ~line 22388/22390): rows read
  `{wl} nm -> r/imp`; the screening plot axis (22350) is already unit-aware. → use `_get_x_unit_short()`.
- **Contaminant plot click-popup** (`_contam_create_wavelength_click_handler`, ~line 56452): shared by
  4 contaminant plots (group spectra / clean overview / influence / exclusion — all use
  `_get_spectral_xlabel()`). One fix corrects all four popups. → `_get_x_axis_name()` + `_get_x_unit_short()`.
- **Diagnostics "Wavelength Contribution" plot** (~line 58499/58501/58540): main axis (58477) is
  unit-aware, but the top-20 barh y-tick labels, the `ax2` ylabel, and the "Top 3 wavelengths" metrics
  label hardcoded "nm". → helpers.

**Lesson / pin candidate.** The unit helpers (`_get_spectral_xlabel` / `_get_x_unit_short` /
`_get_x_axis_name`) are the single source of truth for nm-vs-cm⁻¹, but they were added after some
click-handler annotations / listboxes / tick labels were already written with literal "nm". The
file's 40+ `set_xlabel` call sites were all swept to the helper; the *satellite* text (popups,
tooltips, listbox rows, barh tick labels, summary labels) was not. **Audit rule:** when a plot's
axis is unit-aware, every other piece of text in/around that plot that prints a wavelength value
must also use the helper. The remaining literal-"nm" hits in the file are legitimate (input-field
labels, file-I/O metadata, calibration-transfer which always operates in nm, synthetic-data
generation) and were intentionally left alone.

---


---

> **Older entries archived to [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md)** — third batch on 2026-07-30 moved entries dated before 2026-06-01 (the 2026-05-08 TPE/CI cluster and everything older). Second batch (2026-05-02) moved 2026-05-01 and earlier. First batch (2026-04-29) moved entries before 2026-04-15. Active log keeps roughly the last two months. Grep the archive when you need historical context on a closed bug, decision, or PR.
