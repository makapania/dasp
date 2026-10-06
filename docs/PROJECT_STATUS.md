# Project Status

> Historical material (completed-work narratives, old hand-offs, per-PR details, superseded sections) was moved verbatim to [PROJECT_STATUS_ARCHIVE.md](PROJECT_STATUS_ARCHIVE.md) on 2026-09-15. Grep it for history.

## ▶ NEXT SESSION — START HERE (hand-off updated 2026-10-06)

### 0. NOW (2026-10-06): Wave 3 — F2 part 1 MERGED (PR #97, `d405299`); next is F2 part 2 (GUI)
Wave 1 + 2 (PRs #83-#96) are merged; their summary is in the archive (Batch 2). Wave 3 order (§0a): F2 part 1 ✔ →
F2 part 2 (GUI) → F1 part 1 (CVPlan) → F4/CS1/F3/F5/MW1/SP1-2. No open branches.
**F2 part 1** (plan `docs/plans/2026-10-05-F2-part1-transfer-validation-backend.md`; plan, code and confirm rounds
reviewed by GLM 5.3 + DeepSeek Flash, both MERGE; full suite 5029 passed / 26 skipped / 0 failed):
`transfer_evaluation.evaluate_transfer` (leave-one-standard-out
bake-off, "No correction" + primary reference rows), `fit_transfer`, `pair_standards_by_id`,
`predict_fn_from_model`; `calibration_transfer.estimate_pds_lowrank` (centred low-rank PDS, `B_centred` key),
`estimate_ds_dual`, `estimate_prediction_correction`/`apply_prediction_correction`; all apply sites use
`apply_transfer_dispatch`. AGENT_COMPOSITION §9. Simulated second instrument from the example data in
`tests/test_ct_evaluate_transfer.py`.
**Decision for the user (my call, revertible):** the board ranks by `RMSD_vs_primary` (transfer error alone, y
units), not RMSEP, whenever a model is given; `rank_by="RMSEP"` restores the other. Why: SESSION_LOG 2026-10-05.
**Part 2 next (GUI):** Validate/Compare panel on the CT tab, ID pairing in the loaders, "Plan transfer standards",
delete the dead `_build_ct_transfer_model`/`_load_ct_paired_spectra`. **Part 3:** `.dasp` per-instrument adapters;
must decide replace vs chain for a satellite slope/bias over a stored `bias_correction` (CT3 is fitted on the
model's corrected output), and the GUI Model Development display reads `metrics_original` from a correction dict.
**Reviewers:** Codex for MAJOR checks only; routine confirms GLM 5.3 + DeepSeek Flash via opencode (forbid shell
redirection, writes, reads outside the repo; demand a verdict). Merge rule: well-reviewed PR → merge origin/main
into branch, test, `gh pr create`, `gh pr merge N --merge --match-head-commit <full sha>`.
**Open follow-ups from Wave 1/2:** decimal-comma CSV/ASD readers; PLS + regression imbalance sample_weight;
regression sample weighters not applied to the final refit; QW4 distances on preprocessed/PCA spectra; #86 LOWs.
**Open questions for the user:** Import rounds wavelengths to integers and refuses sub-unit spacing (FTIR?); comma
ASCII files default to dot-decimal with a warning (my call); advisory EPO count (my call); delete stray GLM temp files
in %TEMP% (diff.txt, gui_f359708.py, opencode\repo_*); many stale `.claude/worktrees/agent-*` checkouts (delete?).
**Crash-resume design (PR #79, details in archive):** `_confirm_resume_before_launch` is the single main-thread launch
gate and freezes every worker input; a new Bayesian input must join `BAYESIAN_REQUIRED_SETTINGS` and
`CAPTURABLE_SETTINGS`.
**Python 3.14:** the project is 3.14 only (`.venv314`). A machine still on `.venv312`, or a first pull on a new
machine: follow archive Batch 2 "MOVING A 3.12 MACHINE TO 3.14" and "FIRST PULL ON A NEW MACHINE" first.
### 0a. The 2026-09-28 review results and the combined order
Two whole-codebase reviews ran on `main` `449dfb1` (PR D merged as `85790dd`):
- **Correctness:** `docs/reviews/2026-09-28-adversarial-review.md`. 133 findings kept (129 confirmed by an
  independent refuter; 2 critical, 30 high). IDs are R001-R133; themes in SESSION_LOG 2026-09-28.
- **Improvement roadmap:** `docs/reviews/2026-09-28-improvement-roadmap.md` (7 lenses: calibration transfer,
  contamination, modelling workflow, speed, GUI usability, visuals, structure). IDs QW*/F*/CT*/CS*/MW*/SP*/LF*/ST*.
- **Selector test:** the MC-PLS selector is not adopted; its generic form is MW1 (see §4 item 3).

**Combined order (Waves 1-2 done; user started Wave 3 on 2026-10-05).** Wave 1 fixes the numbers users report
and deploy.
1. **CV leakage:** booster early stopping on the CV test fold (R028, the root cause in cv_utils; R003, R022, R126),
   and ensemble CV/weights fitted in-sample (R002 critical, R021, R018, R105). Reported scores will drop.
2. **Saved model ≠ validated model:** Y-transform save paths (R001 critical, R020, R014/R019, R048), stale bias
   correction (R010), Tab 7 wavelength matching (R009, R112), `all_vars` %g (R031, R078), numeric label encoder (R016).
3. **QW1 + QW10, thread budget and test split.** Measured 60x per booster config and 50x for LOF; the test suite
   should drop from ~38 min to under 10. Can go first, since it speeds up testing every later PR.
4. **Data in:** OPUS reader returns the background, not absorbance (R017); duplicate `read_ascii_spectra` (R062);
   GUI exclusion and dataset-switch bugs (R004-R007, R037).
Wave 2 stops the app misleading:
5. **QW2 + QW6:** calibration-transfer relabel, TSR default, resubstitution R² label, JYPLS-inv centring (R091), dead
   interference controls.
6. **QW3:** contaminant maths. EPO `pca_diff` removes noise (R024), OSC removes the predictive direction (R025), the
   Interference tab crashes (R075), plus R113/R114. Add behavioural tests.
7. **QW4 + QW5:** holdout direction (KS/SPXY must pick CALIBRATION; R085 starting pair), figures of merit; classification
   metrics R029/R030.
8. **QW7:** DPI awareness and fonts (the cheapest visible upgrade). **Merged (#85).** It adds system DPI awareness before `tk.Tk()` plus a DPI-aware manifest in the spec (the frozen build is
   untested), the `_px`/`_px_geometry` scale helpers, and six named fonts (`self.fonts`) wired into every ttk style.
   The literal font-tuple sweep (101 tuples) and the per-tab pixel padding are queued in
   `docs/plans/2026-10-02-font-tuple-sweep.md`; do them after the concurrent GUI branches merge.
Wave 3 (flagships): F2 calibration transfer that validates itself (backend, then GUI), F1 CVPlan/grouped CV, F4 real
DD-SIMCA, CS1 EMSC-with-interferent then F3 in-fold saved contaminant correction, F5 publication output, MW1 stability
selection, SP1/SP2 PLS kernel and SPA. Structural enablers ST1a/ST2/ST4 whenever a flagship touches that area.

**After PR D: T-51 PR E, then F** (old plan `docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md` §1, §5).
- **PR E:** the one-class clamp before the fingerprint, with a record. PCA-SIMCA `LVs` reporting. Use the
  PCA-SIMCA-only `oc_revision=1` in `config_components`, **not** a global version bump.
- **PR F:** the one-class ceiling moves into `suggest_int`. **It needs its own approval from the user**, and it edits
  `suggest_one_class_params`. That is the one planned exception to the no-sampler-edits rule, and it changes the
  pinned sampler hashes in `tests/test_t51_extra_axes_mechanism.py`.


### 3. Open PRs / decisions for the user
- **#63** T-17 multi-target regression (+16k lines, stale since 2026-07-08): user leaning
  toward not using it. Leave open; close only on the user's word.
- **Repo-wide black/flake8 pass?** ~212 files would be reformatted; CI lint steps are
  informational. Do it between feature PRs if at all.
- **Settled 2026-09-15:** `plsda_head` stays PLS-DA-only. The GUI cannot tick PLS in
  classification (`CLASSIFICATION_MODELS` excludes it, and `_on_task_type_changed` unticks
  and disables it on every data load), so only direct backend callers can pass `PLS` +
  classification; `AGENT_COMPOSITION.md` §7b already tells them to spell it `PLS-DA`.

### 4. Queued work
1. **T-51 next:** PR E, then F (one-class clamp; F needs its own approval). PR D is merged.
2. **Smaller follow-ups:**
   - **Already failing on `main`:**
     `tests/gui/test_multiclass_gui.py::test_run_analysis_accepts_multiclass_engine_selection`
     (the worker never starts). Its `_FakeThread` accepts no `kwargs`; it probably predates #79's
     launch changes, but this is unverified. Found by the full GUI suite on 2026-09-27.
   - The PR D GUI tests leave the result-filter values and traces a real run creates on the shared
     session app. Existing real-run tests do the same. Codex round 2 flagged it; it is harmless today.
   - `split_plsda_params` passes `C` through uncoerced (hand-edited CSV `'0.05'` would crash).
   - Tab 7 re-applies stored params over a user's `n_components` edit — intended?
   - Tab 7 refit of Bayesian XGBoost rows prints XGBoost "params not used" warning (harmless).
   - `apply_extra_axes` constant-clash message wording.
   - #73: surface CatBoost/model failures that search paths used to swallow silently;
     `docs/AGENT_COMPOSITION.md` §7 wrongly says `models_to_test` overrides tier.
   - `tests/test_baseline_advanced.py` (~L90-106) clears/restores all of `sys.modules`
     (possible order-leak; not observed failing).
   - Pre-existing from T-51 step 1: unscaled Bayesian importance proxy; unscaled NSGA-II
     display metrics; NSGA-II 'SVM' chromosomes always 1e10; `MODELS_WITH_FEATURE_IMPORTANCE`
     lacks 'SVM'; GUI refit double-scaling under autoscale.
3. **MC-PLS stability selector (from leaf_phys_nir): tested 2026-09-28, NOT adopting.** It did not predict better
   than dasp CARS (dasp was equal or higher on 4/4 targets), was more seed-dependent, and gave less reproducible
   regions. The spin-off worth doing is **seed-frequency reporting for any selector**: run it over N seeds and report
   per-band and per-region selection frequency. See SESSION_LOG 2026-09-28.
4. **`SESSION_LOG.md` housekeeping done 2026-09-15** (1705 → 521 lines): batches 6 and 7 in
   `docs/SESSION_LOG_ARCHIVE.md` hold everything before 2026-09-14 plus the full #79
   round-by-round history. Keep it short the same way: archive verbatim, and condense a
   finished PR's narrative down to its durable lessons.

### 5. Tooling notes (2026-09-14/15)
- **Codex:** on this ChatGPT-account login only `gpt-6-astra` works for the "astra"
  model. `gpt-5.6-astra`, `gpt-5.6-alpha` and `gpt-6-alpha` are all rejected.
- **opencode/GLM write mode** worked this session when pointed at a pre-created worktree
  inside the repo (`.claude/worktrees/...`) — used for #75. Read-only opencode can't read
  outside the repo root; give it refs and `git show`, and forbid `gh`/`git fetch` (hangs).
- **DeepSeek via opencode:** tell it to read files only via `git show <sha>:<relpath>` — it
  once mistyped an absolute path and aborted.
- **Parallel agents:** give each a unique PR-body filename (a shared `pr_body.md` got clobbered).
- **GLM 5.3 (opencode, read-only) earns its keep on search-space work:** on #80 it probed the
  real estimators in `.venv314` and caught that `max_samples=1.0` duplicates `'auto'` below
  256 samples. Point it at a commit sha and tell it to use `git show`; forbid `gh`/`git fetch`.
- **Implementation agents often end their turn while a background test run is still going** —
  tell them to block on it, and check.
- **opencode read-only** can't read outside the repo root (scratchpad worktrees, temp
  files). Have it read branch code via `git show origin/<branch>:<path>`; run tests yourself.
- **Rewriting docs from PowerShell:** use `[IO.File]::WriteAllText(..., UTF8Encoding
  $false)` and check `git diff --stat`. `Set-Content` once rewrote the whole of
  PROJECT_STATUS with new line endings.

### Keeping machines in sync: `requirements-lock.txt` (added 2026-09-10)

`pyproject.toml` declares **floors** (`>=`), so two machines installing from it can end
up on different versions of numpy/pandas/sklearn and disagree about results. The pinned
set actually verified on the primary machine lives in **`requirements-lock.txt`** at the
repo root (Python **3.14.7**).

On a new or drifting machine (or just run `install.bat` / `install.sh`, which do
exactly this):

```bash
py -3.14 -m venv .venv314
.venv314\Scripts\activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Check for and apply updates with the standing process rather than improvising:

```bash
python scripts\upgrade_check.py   # what is outdated, risky, blocked, order-forced
```

then follow `docs/upgrade/UPGRADE_RUNBOOK.md`. After any intentional upgrade,
regenerate and commit the lock:

```bash
.venv314\Scripts\python -m pip freeze --exclude-editable > requirements-lock.txt
```

(then restore the comment header at the top of the file).

**Verified on a second machine 2026-09-11.** The recreate procedure above was run
end-to-end on a clean Windows box with no prior Python: all pins resolved with no
conflicts, `spectral_predict` imports from `src/`, and no stale `spectral-predict.exe`
shim appears in a fresh venv. Two fixes came out of that run:

- `pytest-timeout` was **missing from the lock** — `pyproject.toml` declares it in the
  dev extra and `.github/workflows/ci.yml` needs it for the T-CI-1 timeout flags, but
  the original `pip freeze` did not capture it, so a machine following this procedure
  got a venv that could not run CI's test invocation (`--no-deps` backfills nothing).
  Now pinned at `pytest-timeout==2.4.0`; it adds no transitive deps.
- The activate line above contained a raw `0x07` (BEL) byte instead of `\a`, so it read
  `.venv312\Scriptsctivate` and could not be copy-pasted.

> **That verification covered the Python 3.12 procedure**, which the 3.14 section at the
> top of this file supersedes. Both fixes it produced still stand — `pytest-timeout` is
> pinned and the BEL byte is gone. The **3.14** recreate procedure has been run
> end-to-end on the primary machine only; a second-machine run is still outstanding.

**Superseded 2026-09-12.** This previously read: *"Python 3.12 only" is a convention
enforced only in docs — `pyproject.toml` still declares `requires-python = ">=3.10"`
and advertises 3.10/3.11/3.12 classifiers, so nothing stops an install on 3.10. Left
as-is deliberately; tighten it only if the packaging metadata is meant to match the
rule.* That tightening has now happened: the version rule is **enforced by packaging
metadata**, not convention. `requires-python = ">=3.14"`, the classifiers list 3.14
alone, and CI tests 3.14 only, so pip refuses to install on anything older.

### If you are verifying branch code from a git worktree

The editable install pins `spectral_predict` to a **fixed path under the main
checkout**, so `python script.py` run from a worktree silently imports `main`'s source.
`pytest` from a worktree does NOT have this problem, which makes the mismatch easy to
miss. Assert on the resolved path first:

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import spectral_predict
assert "wt-" in spectral_predict.__file__, spectral_predict.__file__
```

---

## Distribution path — bundle-only as of 0.5.0b1 (updated 2026-04-21)

**Decision:** the PyInstaller 3.12 bundle is now the only supported distribution path. Nobody is expected to clone and `pip install -e .`; the source-install scaffolding (`install.bat` / `install.sh` / `INSTALL.md`) stays in-repo as a developer convenience but is no longer marketed to end users. Beta version `0.5.0b1` ships exclusively as the bundled installer.

**Implication for parallelism:** the 3.12 bundle still uses the threading-backend fallback (see `src/spectral_predict/search.py:_frozen_needs_threading_fallback` — frozen-state-only, NOT version-gated; the original 3.12 plan to recover loky was wrong). Practical impact: numpy/sklearn/lightgbm/xgboost get thread-parallel speedup (those C extensions release the GIL), but pure-Python parallel loops (pymoo NSGA-II, GA-PLS evaluation) are single-core in the bundle. There is no longer a "use the source install for full multiprocessing" escape hatch for users — what the bundle does is what they get.

**Still in-repo from the source-install era (kept, not deleted):**
- `install.bat` / `install.sh`: detects Python 3.14, creates `.venv314`, installs `requirements-lock.txt` then `pip install -e . --no-deps`. Idempotent. Useful for developer setup.
- `INSTALL.md`: GUI-focused walkthrough — now developer-facing, not user-facing.
- `pyproject.toml` deps audited via AST scan. Added: `Pillow>=10.0.0`, `shap>=0.44.0`. Re-enabled: `jcamp>=1.2.1`. Floors bumped: `numpy>=2.0`, `pandas>=2.0`, `scikit-learn>=1.5`, `scipy>=1.11`.

**Intentionally NOT declared as required:**
- **`torch`** — no in-tree module imports it. T-38 deleted the last importer (`learned_preprocessing.py`); the dead `HAS_ENSEMBLE_PREPROCESSING` flag was removed at the same time. The build still excludes torch defensively in case a transitive PyInstaller import sneaks it in.
- **`agilent-ir-formats`** — no Python 3.12 wheel on PyPI. Stays in optional `[agilent]` extra. .seq file loading raises a clear ImportError from `agilent_reader.py:62` until upstream ships a 3.12 wheel.

**Open follow-up if bundle parallelism becomes a real bottleneck:** fix the PyInstaller spawned-child runtime hook (the argv-parse crash in `multiprocessing.freeze_support()`) so loky can be used in the bundle. Tractable but non-trivial — would need a custom runtime hook + verification across the supported Windows targets.

---

## Known follow-ups (deferred from PR #4 reviews, non-blocking)

- ~~**`run_bayesian_search()` does not call `validate_cv_strategy_for_task()` upfront.**~~ — CLOSED 2026-05-07 by T-36 / Item 7 deletion. The function no longer exists; `run_search()` and `run_unified_bayesian()` (the two surviving entry points) both call `validate_cv_strategy_for_task()` upfront, so the gap is moot.
- **Style nits in CV code**: `cv_utils.py` duplicates the `RepeatedKFold` import (module-level + inside `build_cv_splitter`); `templates/validation.py` prints `({cv_folds}-fold)` even for `loo`/`repeated_kfold` (misleading once strategy-specific exports are working); `_majority_vote` should default to NaN in the empty-votes else-branch for safety; LOO `splits = list(...)` materialization at `search.py:~4228` is O(n²) memory for very large datasets (fine for n<5000 spectral data but worth a comment).
- **Suppress sklearn 1.7.2 `"X does not have valid feature names"` UserWarning flood** on `.venv312` — cosmetic noise, ~12MB stderr per grid run, unrelated to any bug. Consider `warnings.filterwarnings(...)` in the GUI entry point.

---

## What Works

- [x] **Analysis Subset V1** (branch `glm/analysis-subset-v1`, uncommitted): Pure-logic module `src/spectral_predict/analysis_subset.py` with 41 tests. Analysis tab card, metadata-only dialog with categorical multi-select, subset provenance in training config, mismatch warnings, one-class guardrail, C2 missing-column safe handling.
- [x] One-class model implementations in `src/spectral_predict/contamination.py`
- [x] Bayesian optimization for one-class (`unified_bayesian.py` handles `task_type='one_class'`)
- [x] Grid search for one-class (`run_one_class_search` in `search.py`)
- [x] GUI task type selection: "One-Class" radio button shows one-class model checkboxes
- [x] Inlier class selection UI with auto-detection
- [x] Model save/load with scaler + PCA reducer persistence (`model_io.py`)
- [x] Prediction tab: save→load→predict round-trip verified manually 2026-04-11. Training inliers come back as "Inlier (X)" as expected. Regression test: `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`.
- [x] External validation: label mapping, confusion matrix, balanced accuracy/sensitivity/specificity
- [x] External validation metrics in Results tab — top N respects `validation_top_n` (default 700), matching classification/regression
- [x] External validation produces full 7-metric set (Sensitivity, Specificity, Precision, F1, Accuracy, BalancedAcc, AUC) via `compute_validation_metrics_for_top_one_class_models()` in `contamination.py` — parity with cal/CV
- [x] External validation metrics in Model Development results text
- [x] Validation checkbox preserved when loading results into Model Development
- [x] Dancing man animation stops on completion
- [x] Model Development: runs complete, buttons re-enable, cursor resets
- [x] Wavelength importance: surrogate LightGBM via `compute_one_class_importances()`
- [x] Code export: one-class models export as standalone Python scripts and Jupyter notebooks with full CV reproduction
- [x] SHAP: permutation importance for most models, TreeExplainer for IsolationForest
- [x] Diagnostics: decision score distribution + sample classification plots
- [x] Residual/leverage diagnostics route to one-class-specific plots (not regression fallback)
- [x] Basic preprocessing discovery works for one-class grid search (callback wrapper fixed)
- [x] Variable selection: 'importance' method works for one-class
- [x] Results Treeview tooltips: all one-class metrics (Sensitivity/AUC/cv) + validation metrics (RMSEP/R2pred/val_*) now covered; jargon-heavy tooltips (ROC_AUC, Kappa, MCC, BER, LogLoss) rewritten for non-technical audience
- [x] CV strategy support: LOO, Repeated K-Fold, and standard K-Fold via `build_cv_splitter()` factory in `cv_utils.py`. Pooled RMSEcv (regression) and pooled sensitivity/specificity (one-class). GUI controls in Analysis tab + Model Development. Cost estimator with LOO/Repeated warnings. training_config stores cv_strategy for model save/load. Post-review fixes (2026-04-12): RepeatedKFold crash via `cross_val_predict_pooled`, one-class Bayesian cv_strategy forwarding, GUI LOO folds metadata, LOO classification minority-class guard, differentiated mixed-regime warnings.

## Known Issues

- [ ] **One-class grid search inherently slower than classification** — Expected due to two-phase CV+calibration design and LOF O(n²) complexity. Optimized but still slower by nature.
- [ ] **Preprocessing importance dropdown has no effect for one-class** — All methods resolve to LightGBM. Per-model refinement never triggers because `models_to_test` isn't passed.
- [ ] **Variable selection limited** — Only 'importance' method works. UVE/SPA/iPLS/CARS are PLS-specific and incompatible. This is by design but could use a UI hint.
- [ ] **Residual correlation overlay** — Disabled for one-class (no continuous residuals). Shows zeros.
- [ ] **LOO / Repeated K-Fold results may not auto-populate the Results tab** — observed 2026-04-12 during manual testing of PR #4 (cv-strategy-overhaul). User ran LOO on a 49-sample regression dataset; analysis completed without errors, but the Results tab did not display the completed runs as expected. Repeated K-Fold uncertain — not explicitly tested for the same behavior. Save/load/predict round-trip from a refined LOO model works. Possible causes to investigate: (a) Results tab's Treeview populate path is gated on `Folds` matching a specific integer range or on `training_config['cv_strategy'] == 'kfold'`; (b) progress callback's final "completion" signal isn't wired for non-kfold strategies; (c) sort/filter logic in `_populate_results_treeview` (or equivalent) drops rows where `Folds == n_samples` (LOO case). Needs repro + trace before fixing.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Surrogate LightGBM for wavelength importance | Model-agnostic, fast, already implemented in `compute_one_class_importances()` |
| Permutation importance instead of SHAP KernelExplainer | KernelExplainer is impossibly slow for spectral data (hundreds of wavelengths × thousands of model evals per sample) |
| TreeExplainer only for IsolationForest without scaler/PCA | Only case where SHAP TreeExplainer works directly |
| Validation metrics: balanced accuracy, sensitivity, specificity | Standard one-class metrics. Sensitivity = outlier detection rate, Specificity = inlier retention rate |
| String comparison for inlier labels | Both Bayesian and grid search convert y and inlier_label to strings before comparison to handle numeric/text label mismatches |
| Early return pattern in GUI threads | One-class has separate pipeline from regression/classification. Every early return MUST include cleanup (animation stop, button reset, etc.) |

## Key Files

| File | Role |
|------|------|
| `spectral_predict_gui_optimized.py` | Main GUI (~45K lines). One-class paths scattered throughout. |
| `src/spectral_predict/contamination.py` | One-class model implementations, `run_one_class_cv()`, `compute_one_class_importances()` |
| `src/spectral_predict/search.py` | `run_one_class_search()` for grid search |
| `src/spectral_predict/unified_bayesian.py` | Bayesian optimization, handles `task_type='one_class'` |
| `src/spectral_predict/scoring.py` | Scoring functions with one-class metrics |
| `src/spectral_predict/model_io.py` | Save/load with scaler/PCA persistence |
| `src/spectral_predict/preprocessing_discovery.py` | Smart preprocessing, has one-class path at line ~680 |

## Follow-Ups (unclaimed)

- **Decimal-comma misreads in the CSV, reference and ASD-text readers (deferred from fix/readers review round 5, 2026-10-02).** These predate fix/readers and are not made worse by it (the ASCII reader's delimiter/decimal policy is not applied to these readers). Neither case gives a Python warning or an `import_warnings` entry.
  1. **CSV spectra and reference readers shift columns** (`read_csv_spectra` ~io.py:81, `read_combined_csv` ~io.py:421, `read_reference_csv` ~io.py:1474). When decimal-comma values are written unquoted in a comma-delimited file, data rows have more fields than the header, and pandas silently turns the extra leading field into an implicit index. Codex repro: header `id,1000,1001,...,1099`, one data row `s1` followed by 100 unquoted `0,123` values → 100 wavelengths, alternating values `0` and `123`, sample ID `123`. Reference repro: `id,y\n1,12,34\n2,13,45` → IDs `12,13`, targets `34,45`. Fix: compare each row's field count with the header before pandas can infer an index; apply the ASCII delimiter/decimal policy (decimal-point default, warn or refuse when a decimal-comma split competes); pass diagnostics to the GUI's `import_warnings` dialog.
  2. **ASD text parsing reports detector counts instead of a decimal-comma reflectance** (~io.py:849, ~858). For 100 rows shaped like `1000 12345 12000 0,123`, `read_asd_dir` drops the unparseable last field and returns `12000` rather than `0.123`. Fix: keep column positions; recognise decimal-comma fields, or reject/report a non-numeric expected ordinate instead of falling back to an earlier numeric column.

- **T-51 — Opt-in Bayesian search-space axes (ticket written 2026-08-30; design complete, no code).** Full ticket: `docs/plans/2026-08-30-T51-bayesian-opt-in-search-axes.md`. **Premise is added value, not a defect** — supervised Bayesian performs well and the ticket does not assume otherwise. It adds opt-in knobs for hyperparameters that currently take exactly *one* value (LightGBM `reg_alpha=0.1`/`subsample=0.8`/`min_child_samples=5`, XGBoost `colsample_bytree=0.8` with `gamma`/`min_child_weight` absent, RandomForest `max_features='sqrt'`, SVM `gamma='scale'`, PLS-DA logistic head `C=1.0`), curated per model family, all off by default. **Design:** `suggest_model_params` / `suggest_one_class_params` stay byte-for-byte; a new `search_spaces.py` supplies `apply_extra_axes()` that runs after them and is a literal no-op when no bundle is enabled. GUI checkboxes and the Python API drive the same bundle ids (`enabled_extra_axes=(...)`). One-class included from the start, closing the PR #58 deferral. **Two hard constraints discovered in review (see SESSION_LOG 2026-08-30):** (a) Optuna forbids re-suggesting a parameter name, so only *pinned-constant* axes can be opened additively — ranges already searched (Ridge/Lasso/ElasticNet `alpha`, MLP `alpha`, OneClassSVM `gamma`, PCA-SIMCA `n_components`) cannot be widened this way, which is fine since widening them is explicitly out of scope; (b) clamping after `trial.suggest_*` does not change what TPE learns. **Two genuine bugs ride alongside, sequenced separately:** the `'SVC'`/`'SVM'` string mismatch leaving classification SVM unscaled (prerequisite for any SVM `gamma` knob), and the PLS clamp asymmetry where Bayesian bounds `n_components` by `n_features` while the grid path uses `compute_min_train_fold_size` (approved to ship last, own gate). Reviewed by Codex gpt-5.5 and a DeepSeek+GLM peer panel.

- **Non-balanced imbalance ratios + ElasticNet for PLS-DA inner LR — niche reproducibility gap (revised 2026-05-07; supersedes 2026-04-29 framing).** The original bullet claimed boosters and PLS-DA's inner LR don't receive any class_weight handling. That's stale: PR #38 added a dispatcher in `_apply_class_weight_discriminator_for_rebuilt_model` (`search.py` ~4555-4611) that injects balanced reweighting for every classifier when the user picks `class_weight` from the Data Quality dropdown. CatBoost gets `auto_class_weights='Balanced'`; sklearn estimators with a `class_weight` attribute get `class_weight='balanced'`; XGBoost / LightGBM / RidgeClassifier get per-fold `compute_sample_weight('balanced', y)` threaded as `sample_weight` (cv_utils.py `cross_val_boosting_rounds` / `_fit_fold_full_rounds`; the grid final refit uses balanced weights from the full calibration y); PLS-DA's inner LR gets `class_weight='balanced'` at the Pipeline build site; MLP warns and falls back. Codegen parity in exported scripts via runtime `IMBALANCE_METHOD == 'class_weight'` conditional (code_generator.py). **Remaining gaps:** (a) non-balanced ratios — XGBoost `scale_pos_weight=N` for a specific positive ratio, CatBoost `auto_class_weights='SqrtBalanced'` for softer correction at extreme imbalance, custom `class_weight={0: w0, 1: w1}` dicts; (b) ElasticNet penalty for PLS-DA inner LR (`penalty='elasticnet'` + `l1_ratio`) — separate from imbalance, surfaced by the same FTIR Bone PLS paper reference; (c) per-model imbalance override — the dropdown is global, which is arguably the *correct* design for a fair model-comparison search (per-model would confound rankings). Effort if ever needed: ~half-day for non-balanced ratios via one branch per model class in the existing dispatcher + GUI numeric entry + codegen mirror. User judgment 2026-05-07: niche enough to leave open until a specific paper-reproduction asks for it. Original proposal at `docs/plans/2026-04-29-model-native-loss-reweighting.md` is preserved as record but its framing is superseded by this revision.

- **One-class hyperparameters — round-2 shipping in PR #58 (2026-05-07); residual deferrals listed.** Round-1 (2026-04-19) added Tab 4C cards for the curated `get_one_class_model_grids()` defaults: OCSVM `kernel`/`gamma`/`nu`/`degree`, IF `n_estimators`/`contamination`/`max_features`, EllipticEnvelope `contamination`, LOF `n_neighbors`/`contamination`, PCA-SIMCA `n_components`/`alpha`. **Round-2 (PR #58) closes:** LOF `metric` (euclidean/manhattan/minkowski/cosine + custom-string with whitelist validation), LOF contamination presets 0.01/0.1, IsolationForest `max_samples` (auto/256/512), IsolationForest `n_estimators` presets 200/500, EllipticEnvelope `support_fraction` (None/0.5/0.75). Plus a post-review hardening pass: `_parse_oc_int_list` / `_parse_oc_float_list` / `_parse_oc_metric_list` replace the scalar parsers so comma-separated Custom: input ("256, 512") parses correctly instead of being silently dropped, and bad tokens surface in a single `messagebox.showwarning` summary rather than vanishing. **Still deferred:** OCSVM `coef0` and additional kernels (`sigmoid`, `linear`) — RBF/poly dominate spectroscopy per user judgment 2026-05-07; OCSVM `nu=0.2` / LOF `n_neighbors=50` / PCA-SIMCA `alpha=0.10` predefined checkboxes (custom-string already reaches them); the one-class **Bayesian** path (`unified_bayesian.py::suggest_one_class_params()`) which still uses hardcoded ranges and would need a separate design to honor user-customized grids.

- **`predict_with_uncertainty` swallows one-class decision_function failures silently.** `model_io.py:854-860`: after `predict_with_model()` returns, the bare `except Exception: decision_scores = None` catches any score-extraction error and returns predictions with empty `uncertainty`/`applicability_domain` payloads. The GUI has no way to surface "predictions worked, uncertainty broke" — exactly the failure mode this PR has already hit once with the order-of-operations bug. Fix: log the exception with `logger.warning(...)` at minimum; ideally return a structured `decision_score_error` flag that the GUI can display alongside the predictions.

- **OC validation helper can't recover the right `polyorder` for 2nd-derivative grid-search rows.** Grid search writes `polyorder=None` (delegating to `polyorder_map` inside `SavgolDerivative` which picks 3 for `deriv=2`), but the validation helper at `contamination.py:922-927` falls back to `min(2, window-1) if window > 2 else 0`, which gives `poly=2` for `deriv=2`. The pipeline then runs with the wrong polyorder and the resulting `val_*` metrics are slightly off vs. what training actually used. Fix: either store the resolved polyorder in the grid-search result dict (most direct), or import the same `polyorder_map` lookup into the validation helper, or have `_maybe_int` fall back to `polyorder_map[deriv]` when poly isn't set.

- **CV Strategy Phase 2: Propagate outer CV strategy into variable selection inner loops.** Currently inner loops (UVE, SPA, iPLS, CARS, GA-PLS) always use hardcoded 5-fold K-fold regardless of the outer CV strategy. A GUI warning is displayed when outer != kfold. Phase 2 would thread `cv_strategy`/`cv_n_repeats` into the variable selection internals.

- **CV Strategy Phase 2: Propagate CV strategy into predictor screening and smart preprocessing.** These internal CV sites are currently hardcoded 5-fold and don't respect the user's chosen strategy.

- **CV Strategy Phase 2: NSGA-II multi-objective search CV strategy support.** Currently falls back to K-fold with a logged warning when non-kfold strategy is selected. Needs native LOO/Repeated K-fold support.

- **One-class search: Add `training_config` to result rows.** Regression/classification grid search writes `training_config` (with `cv_strategy`, `folds`, `cv_n_repeats`) but one-class search does not. This means one-class saved models don't preserve the CV strategy used during training.

- **One-class export lacks automated regression tests.** The existing 33 export tests (`tests/test_code_generator.py`, etc.) do not exercise the `task_type='one_class'` path. Manual verification confirms all 5 models compile, but a future refactor of `templates/validation.py` or `code_generator.py` could silently break one-class export. A single parameterized test generating scripts for all 5 OC models would be cheap insurance.

- **Template string formatting fragility in one-class CV block.** `CROSS_VALIDATION_ONE_CLASS_TEMPLATE` uses `.format()` with `{model_name}` and `{x_var}` variables inside a large multi-line template string. This is correct today but slightly fragile: a future edit that introduces an additional brace pair (e.g. a dictionary literal `{` `}`) would cause a `KeyError` at generation time. This is pre-existing technical debt in the template system, not introduced by this commit, but worth noting if the template engine is ever refactored.

- **One-class export lacks per-fold CV statistics printout (parity bug).** Regression export prints pooled metrics **and** per-fold breakdown (`print(f"\\nPer-fold RMSE: {{[f'{{x:.4f}}' for x in fold_rmse]}}")`). One-class CV template computes `fold_metrics` list but `METRICS_ONE_CLASS_TEMPLATE` prints only pooled metrics, not the per-fold details. This is a **parity mismatch** with the regression export experience. Classification also lacks per-fold printout but gives detailed confusion matrix + classification report instead, which one-class already does via its own confusion matrix plot. Still, for parity with regression's explicit per-fold reporting, one-class should print per-fold sensitivity/specificity/AUC if available.

- **Pause/resume hardening — mostly shipped (revised 2026-05-07; supersedes 2026-04-23 framing).** Of the five gaps surfaced in the original bullet:
  1. **Coarse Bayesian checkpoint granularity** — STILL TRUE but mitigated. Pause only takes effect between Optuna trials, so a 30-minute CatBoost trial keeps burning CPU after the click. The UI now honestly transitions to a `'pausing'` state and emits "[PAUSE PENDING] Trial still running. Pause takes effect between trials; current trial may be 20+ minutes." after 30s (`_poll_actually_paused`, gui ~23100-23125). Fundamental third-party constraint — can't wedge a check inside a CatBoost / XGBoost / LightGBM `fit()` call.
  2. **UI lies about pause state** — CLOSED (T-11 A). `_pause_search` (gui ~23083) transitions to `'pausing'`, polls `search_controller.is_actually_paused`, and only flips to `'paused'` when the worker acks via `check_and_wait()`.
  3. **No thread-alive check on Resume** — CLOSED (T-11 B). `_resume_search` (gui ~23131-23138) calls `thread.is_alive()` and refuses with a clear log message if the worker died.
  4. **Zero persistence** — CLOSED (T-41). `unified_bayesian.run_unified_bayesian` accepts `enable_sqlite_persistence: 'auto' | 'always' | 'never'`. Default `'auto'` runs the first 10 trials in-memory, then decides based on median fit time; `'always'` writes to SQLite from trial 0. WAL pragmas via `_apply_wal_pragmas` (T-46) for sync-friendly performance. Crash-recovery sidecar at `<user_data_dir>/dasp/optuna/active_run.json`; on app startup `_check_for_incomplete_run` (gui ~23150) offers Resume / Discard / Decide-later (T-11 D).
  5. **Bonus: disk logging** — CLOSED (T-45). `run_logging.py::setup_app_logger` wires `_SafeRotatingFileHandler` (50 MB × 3 backups). Post-mortem now possible after a crashed run.

  Net: only gap #1 remains, and it's mitigated by honest UI messaging. No further work warranted unless a specific failure mode surfaces.

## Analysis Subset V1 — Known Limitations / Risk (2026-04-19)

Blocking bugs (stale `active_indices` after dataset replacement or row deletion) fixed in this session. Remaining deferrals:

- **Integration-level GUI tests not yet written:** reload-with-subset, row-delete-with-subset, revert-after-column-delete, and mismatch-warning text. Pure-logic coverage (41 tests in `test_analysis_subset.py`) is solid, but no automated test drives the GUI through these multi-step scenarios.
- **Manual GUI verification checklist still pending:** end-to-end validation that the Analysis-tab card, data-viewer highlighting, provenance in training config, and one-class guardrail all behave correctly after each dataset-replacement path.
- **`_use_for_analysis()` does not carry `combined_metadata_df`** from data sources. If a user loads data via Data Management → Use for Analysis, metadata columns may be absent, causing the subset dialog to show no columns. This is a pre-existing limitation of the data-source path, not introduced by Analysis Subset V1.
