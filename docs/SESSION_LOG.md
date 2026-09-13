# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

Older entries are in [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md); batch 5 on 2026-09-12 moved entries before 2026-07-12, following the two-month retention rule.

## 2026-09-12 - Python 3.14 migration: what the analysis got wrong, and what only building could tell us

**An analysis document that was never executed had nine errors in it.**
`docs/PYTHON_UPGRADE_DECISION.md` was careful, cited file:line throughout, and
was still wrong in ways that would each have cost a turn. Its own Appendix B
admits nothing was installed, built or run. Recorded because the failure mode
generalizes: *verified by inspection* and *verified* are different claims.

- It reported the installed jcamp as 1.2.1 drifting from a 1.2.2 lockfile pin.
  It had read `jcamp.__version__`, which the package ships stale. pip metadata
  said 1.2.2 and matched. **Reading a `__version__` attribute is not a version
  check** - use `importlib.metadata`.
- It said the build path hardcodes 3.12 "in four places". It was ~25 across
  three files, several user-visible.
- It called source-only `jcamp==1.2.2` the one blocker for 3.14. It builds from
  source on 3.14 without complaint.
- It carried the Python 3.12 float `sum()` change in as a migration risk. That
  landed in 3.11->3.12; the project already started at 3.12.
- Both build files still deferred to a "production 3.11 spec" and a
  `build_installer.py` that no longer exist.

**The pandas TOC collision fires on essentially every build.** The post-COLLECT
repair in `build_installer_py312.py` triggered on all three 3.14 builds. It is
load-bearing, not a historical workaround awaiting cleanup.

**PyInstaller's manual DLL globs never matched OpenBLAS.** The spec collects
`*/lib/*.dll`, `*/*.libs/*.dll`, `*/.libs/*.dll`, `*/libs/*.dll` - every pattern
requires a parent directory. But `numpy.libs/`, `scipy.libs/`, `pandas.libs/`
and `llvmlite.libs/` sit at the TOP level of site-packages. The bundle works
only because PyInstaller's hooks collect them. **That manual list is a false
safety net**; if hook behavior changes, it will not catch the fall.

**A build step that could not fail was failing.** Every path in
`run_inno_setup()` returned True and `main()` discarded the result, so a build
producing no installer printed "Build Complete" and exited 0. Reproduced
directly. Compounding it, `find_inno_setup()` never checked
`%LOCALAPPDATA%\Programs`, which is where winget installs Inno Setup - so a
machine with it correctly installed reported "not found". For a project whose
only distribution channel is that installer, this was the most expensive
possible thing to fail quietly.

**Aggregate metrics hide per-sample changes.** The first baseline compared only
ranked tables. Adding per-sample out-of-fold predictions changed what the
comparison could see: after numpy 2.4.4->2.5.3, per-sample predictions were
byte-identical while every aggregate metric column moved by 1e-13 to 1e-15.
Comparing only one of the two would have given a misleading answer either way.

**Ranking ties are decided by floating-point noise.** That same numpy bump
swapped ranks 225/226 between two PLS models whose `CompositeScore` agreed to 15
significant figures. Harmless here, but near-tied rows can reorder for reasons
unrelated to model quality.

**Upgrade ordering can be forced by a cap you cannot see.** `numba 0.66`
required `numpy<2.5`, so numpy could not advance until numba did - the resolver
does not explain this, it just refuses. Conversely `alive-progress 3.3.0` pins
`about-time` and `graphemeu` to exact versions, so pip installs the newer one
and *then* reports the conflict, leaving the environment broken.
`scripts/upgrade_check.py` now detects both classes before you try.

**Do not modify a venv while its test suite is running.** Upgrading a package
mid-run invalidated a 1-hour suite and it had to be redone.

---

## 2026-09-10 - Repo hygiene: a .gitignore rule that never matched, and AGENTS.md drift

**1. `git status` output is not valid `.gitignore` syntax.** The rule added to
suppress the mangled Windows tempfile names was written as
`C\357\200\272Users*` - copied straight out of `git status`, which C-quotes
non-ASCII bytes in paths. Git does **not** decode those octal escapes in a
`.gitignore` pattern, so the rule matched nothing and all five files kept showing up
as untracked for months. Replaced with `C*Users*AppData*Temp*`, verified with
`git check-ignore -v`. Lesson: always confirm a new ignore rule with
`git check-ignore -v <path>` rather than assuming a pasted path works as a pattern.

**2. `AGENTS.md` was an untracked, stale copy of `CLAUDE.md`.** It was 17 lines
behind - missing the "there is no CLI" section and the mandate to read
`docs/AGENT_COMPOSITION.md` - so Codex was reading a guide that still implied a CLI
existed, while Claude read the current one. Because it was untracked it also existed
on only one machine. Replaced with a short **pointer** to `CLAUDE.md` and committed.
Do not re-copy the contents: a copy is what drifted. One guide, one file.

**3. ~40 untracked scratch files were masking the real answer to "is everything
committed?"** `.pytest-tmp*/` trees, `tools/_*` A/B JSONs and repro scripts,
`*_fails.txt`, `merge_gate_diff.json`, `live_gui_*`, timestamped `example/colab_*.ipynb`.
All now ignored, with `!tools/_autoscale_bayes_compare_full.json` negated because it
is tracked on purpose. A noisy `git status` hid one genuinely unpushed branch
(`feat/T16-phase2-permutation`, local-only, now pushed).

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

## 2026-09-12 - Both bugs fixed, and two verification failures worth remembering

Implemented the Codex review below. Notes on the parts that were not obvious.

**An unreadable version must be fatal, not a placeholder.** The Optuna fingerprint
raises `EnvironmentFingerprintError` when a tracked distribution's version cannot
be read. Degrading to `"unknown"` would make two DIFFERENT broken environments
hash identically and therefore resume-compatible - the exact bug the fingerprint
exists to prevent. A genuinely ABSENT package is different: that is a definite
fact, recorded as `"absent"` and hashed.

**Enumerating Optuna studies CREATES the SQLite file.** The 'previous results were
computed elsewhere' notice originally ran whenever a storage URL existed, which
broke the 'never' and 'auto'-warmup promise of staying purely in memory. Two T-41
tests caught it (`test_auto_picks_in_memory_no_db_file`,
`test_never_mode_in_memory_no_db`). Now gated on `always`, which is also the only
mode that actually resumes before a trial runs.

**A revert that silently does not apply produces a fake proof.** Verifying that the
new MultiGroupEPO tests actually FAIL against the old code, the revert was written
with `
` line endings against a CRLF file, so `str.replace` matched nothing. The
tests 'passed against the buggy code' because the buggy code was never restored.
**Assert the mutation happened** (count occurrences before/after) rather than
trusting a replace, and do byte-level edits on CRLF files.

**A test can pass for the wrong reason.** `test_group_labels_are_sorted` used the
probe's forward group order, which happened to ALREADY be sorted - so it passed
with the bug present. It only distinguishes sorted-vs-insertion order when the
input is deliberately unsorted.

**`run_unified_bayesian`'s third positional is `wavelengths`, not `task_type`.**
Passing `'regression'` there makes every trial fail with `IndexError: too many
indices for array: array is 0-dimensional`, and the run returns an empty
leaderboard rather than raising. That looks exactly like an upgrade regression.
Signature: `run_unified_bayesian(X, y, wavelengths, model_name, task_type=...)`.

**Not fixed, deliberately:** the GUI 'Apply EPO' path builds `EstimatedEPO`
(GUI:58521) with `random_state=None` (`contaminant_analysis.py:462`) and is still
nondeterministic. The full cross-version SQLite replay matrix is also not built;
current coverage is the digest's sensitivity plus the existing T-41 storage tests.

---

## 2026-09-12 - Codex review: numerical-environment resume and MultiGroupEPO seeds

Evaluation only; no application source edits. Both pre-existing bugs are real.
Recommendation: FIX NOW for both, but Optuna needs explicit old-cache retirement,
not merely an extra version string silently changing the study name.

- Optuna persistence defaults to auto in both GUI and backend; after 10 trials it
  migrates if median completed-trial duration exceeds 1 second (or fewer than 3
  completed). GUI crash recovery restores the storage URL and forces always mode.
  A bare backend call without active run_state has no disk persistence.
- Reproduced on a synthetic temporary SQLite: Python 3.12.10 / numpy 2.4.4 /
  sklearn 1.8.0 / Optuna 4.8.0 created one completed PLS trial; Python 3.14.7 /
  numpy 2.5.3 / sklearn 1.9.1 / Optuna 5.0.0 reopened the same named study and
  replayed its exact score as trial 1 with ZERO CV calls. This also affects TPE
  history, trial-budget accounting, and old leaderboard rows, so changing only
  trial fingerprints is insufficient. Source: unified_bayesian.py:2562, 2685,
  1672, 2912, 3074; GUI:23902.
- Small fix design: keep the existing config-only name as a base; append a stable
  environment digest, persist the unhashed environment in study.user_attrs before
  trials, and warn through logging plus progress_callback if resuming instead
  starts a fresh environment-specific study. Never reuse legacy scores with
  unknown provenance. Leave old study rows/databases intact. Old studies cannot
  continue under the new identity even when their actual environment happens to
  match: this is intentional one-time cache invalidation, not database corruption.
- MultiGroupEPO's hash(label) at contaminant_analysis.py:2323 drives the library,
  SVD, projection at :2376, and transform return at :2405. Two subprocesses with
  identical synthetic input and PYTHONHASHSEED=1/2 differed by max abs 1.976 in
  transformed values. An in-memory stable blake2b replacement produced identical
  library/projection/transformed data across those processes. Reversing dict
  insertion order still produced 5.3e-15 transform differences; sort string keys
  during validation to fix numeric assembly order too. No second hash() call was
  found in src/spectral_predict.
- Blast-radius nuance: analyze_multiple_contaminants returns this transformer
  under results['epo'], but computes combined_influence/exclusion_regions through
  a SEPARATE MultiContaminantAnalyzer(random_state=42), at :2721. The GUI displays
  that combined result (:57677) and Apply EPO uses EstimatedEPO (:58531), not the
  MultiGroupEPO object. EstimatedEPO defaults random_state=None (:469), including
  that GUI call, and remains a separate randomness issue after the hash fix.
- Existing focused suite: 110 passed (test_bayesian_dedup,
  test_t41_bayesian_sqlite_auto_calculator, test_t42_write_path_plumbing,
  test_contaminant_analysis). Existing MultiGroupEPO tests mostly assert shapes;
  add cross-process transformed-output and dict-order tests. Add end-to-end
  SQLite resume tests for same environment, changed environment, legacy study,
  and auto-migration preserving environment metadata.

## 2026-09-12 - PR #65 performance/safety review (Codex)

Review target is e13393f against main 8de7445; Python 3.14.7 in .venv314
passes pip check. The 125 focused environment/EPO/dedup/SQLite/contaminant tests
pass, as do 9 JCAMP tests (1 skipped).

Two non-obvious review findings (subsequently verified and quantified below):
- The new compatibility notice asks Optuna for every study SUMMARY but only uses
  names. Installed Optuna 5.0.0 study/study.py:1594 loads ALL trials for EACH
  study, including their arrays and user attributes. This adds a whole-database
  scan before every always-persistent model search, including unrelated models.
  get_all_study_names avoids loading trial history entirely.
- _query_build_python splits stdout on all whitespace, truncating a site-packages
  path containing a space. The pandas post-build repair then silently skips
  because its source path does not exist, and falsely reports a match. The
  existing log records this repair as necessary on all three 3.14 builds.

PR #65 verification update:
- Build helper replay with `C:\Users\Jane Doe\dasp\.venv314\Lib\site-packages`
  returned `C:\Users\Jane`. This confirms the whitespace parsing regression.
- Three-study SQLite with 240 realistic payload-bearing trials (11.4 MiB):
  median study-summary scan 0.205 s versus name listing 0.0197 s (3 repetitions).
- Real PLS SQLite calls: the same numerical environment resumed the same study;
  mocking numpy's installed version created a different study and emitted the
  incompatibility notice. No prior database was modified by these probes.
- An attempted pybaselines fingerprint probe was NOT valid evidence: Bayesian
  apply_preprocessing only implements polynomial/als/rubber_band/airpls and
  silently ignores `asls`. Missing pybaselines is not a proven fingerprint defect
  on this execution path. Do not promote an unexercised dependency to a finding.
- The existing dist bundle predates dcde845: bundled unified_bayesian.py contains
  no ENV_FINGERPRINT definitions and its contaminant_analysis.py also differs
  from HEAD. Its executable timestamp is 15:46; the bug-fix commit is 17:23.
  A --test run on that artifact cannot validate the final PR's two bug fixes.

PR #65 review complete: three P2 findings, no application edits. The third is
RUN_SPECTRAL_PREDICT.bat:25-27: a failed lock install is masked by a successful
editable --no-deps install. A temporary copy of the actual launcher, with only
Python calls replaced by a controlled batch stub, reached gui_launched and
returned 0 after the lock install returned 1. install.bat already handles this
correctly. Fix both command error checks in the launcher.

Two fixed-workload timing rounds (3.12 then 3.14, then reversed), six measurements
per environment after warmup, show RandomForest 2.633 -> 2.014 seconds (24% faster)
and XGBoost 1.887 -> 2.160 seconds (14% slower). Single-thread 240 x 512 synthetic
regression, 3-fold CV; all five tested models' RMSEs agree across stacks. These
are combined-stack workload measurements, not general GUI performance claims.
Full evidence, boundaries and next steps are in
[the review](reviews/2026-09-12-pr65-performance-safety.md).

## 2026-09-12 - PR #65 name-only lookup audit and Fable review

The summaries really do populate _CachedStorage's trial cache, but for the URL
string passed here Optuna constructs a NEW temporary storage instance. That cache
is discarded after the summary call; _existing retains only strings. Actual
resume uses separately constructed create_study/load_study storage objects, and
rehydrates fingerprints, sampler history, result rows and arrays from study.trials.
Both enumeration APIs use the same RDBStorage constructor and get_all_studies;
keep the existing always-mode gate because both can initialize a new database.

Requested Fable review completed through the read-only wrapper; modelUsage
confirms claude-fable-5-1. Fable independently confirms the replacement is safe
and all three prior findings are real; it characterizes the scan as avoidable
cost rather than a correctness bug. Fable did not rerun the timing or SQLite
probes, and requests a regression check for the incompatible-environment notice.

Probe gotcha: Optuna 5.0 removes Study.set_system_attr (not just deprecates it).
The first preservation probe stopped at fixture setup for that reason. Use the
public RDBStorage system-attribute methods for fixture metadata. The subsequent
Windows cleanup error was an open SQLite handle, not evidence of data loss.

Name-only lookup verification PASSED using the exact replacement compiled only
in memory (no application source edit):
- Complete SQLite dumps identical before/after either enumeration. SQL trace:
  summaries read two studies' trial tables; names read none; neither wrote data.
- Original and proposed functions resumed the same 24-trial study with all
  stored fields unchanged, including arrays, parameters, distributions, dates,
  user/system attrs and 23 fingerprints; identical 23-row leaderboards.
- Continued original/proposed database copies to 26 trials: identical new TPE
  suggestions, scores, attrs, arrays, leaderboards and progress callbacks.
- Same warning for a retained legacy study; its original rows/metadata preserved.
- auto warmup and never mode performed no lookup and created no SQLite database.

Fable's full returned opinion is saved verbatim in
[the Fable review](reviews/2026-09-12-pr65-fable.md). Codex agrees with its safety
conclusion. Its claim that the benchmark differences are specifically library
version effects is too strong: our benchmark changed Python and dependencies
together, so it cannot isolate those causes. All checks used disposable SQLite
fixtures; no existing studies or environments were changed. The proposed source
change remains unapplied, consistent with this verification/review request.

## 2026-09-12 - Implementing the three PR #65 review fixes

Added ten focused cases before touching application code: five failed against
HEAD, reproducing the whitespace-path pandas corruption, missing repair inputs
being called a successful verification, lock-install failure reaching the GUI,
and resume using the full-summary API. The other five protect existing behavior.

First launcher attempt added a nested failure/exit block immediately after the
lock command. It prevented GUI startup but the controlled cmd.exe batch probe
still returned 0 on that first failure path. Replaced both pip error paths with
a shared failure label outside the conditional, matching install.bat's pattern;
keep the exit-code assertion, not just the 'GUI did not start' assertion.

All ten new cases now pass, and the combined focused suite is 77 passed
(environment fingerprint, Bayesian dedup, T-41/T-42 persistence and build version
checks included). Black with target py314 and flake8 pass for the new test files.
The production fix keeps the always-mode gate and uses only study-name lookup;
resume still reloads the selected study. Build-path lines are split with
splitlines(), missing repair source or bundle now fails verification, and both
launcher pip commands branch to the shared nonzero failure exit.

A fresh standalone/installer build is running to replace the artifact that
predated dcde845. Existing installed application and environments are untouched.

The fresh PyInstaller/Inno build completed successfully and actually repaired
the pandas.util TOC collision. All 75 bundled project Python files match the
working source byte-for-byte, and the repaired pandas module matches .venv314.
The hidden executable --test returned 0 after 16.84 seconds. The capture helper
then hit a cp1252 UnicodeEncodeError while printing the saved log, after the
executable had already completed; inspect the saved output with UTF-8 console
encoding instead of rerunning the completed test.

Saved smoke-test output confirms ALL TESTS PASSED: 42/42 imports, functional
XGBoost/LightGBM/CatBoost fits, active frozen threading fallback and a completed
99-row PLS/LightGBM cross-validated search. Installer size is 230,106,755 bytes
(219.4 MiB), SHA256
171742e9f918ee776416d12cc25998a5a64d1b4f86e597ecc17d70039f021c9a.
No clean install, installed-app upgrade or uninstall was performed. The full
suite was not repeated; the 77 focused tests and new bundle smoke test passed.
