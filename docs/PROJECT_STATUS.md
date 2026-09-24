# Project Status

> Historical material (completed-work narratives, old hand-offs, per-PR details, superseded sections) was moved verbatim to [PROJECT_STATUS_ARCHIVE.md](PROJECT_STATUS_ARCHIVE.md) on 2026-09-15. Grep it for history.

## ▶ NEXT SESSION — START HERE (hand-off updated 2026-09-16)

### Pending decision (2026-09-23): sparse-selector top-N cap, branch `fix/sparse-selector-topn-cap`
CARS-family top-N subsets padded with zero-importance long wavelengths whenever N > the number CARS kept (default grid
top-250 and Bayesian N=500/1000 on BoneCollagen). The branch caps N for named sparse methods, dedupes capped counts, and
fixes the one-class `n_vars` column and the Bayesian dedup fingerprint. New tests pass; the non-GUI suite's only failures are the 4
pre-existing `test_bayesian_environment_fingerprint` cases. **No PR opened, awaiting the user.** Details: SESSION_LOG
2026-09-23 "CARS top-N padding".

### 0. First job: T-51 PR D (GUI card to enable bundles). Nothing is half-finished.
`main` is clean, no open work branches, no worktrees. PR #80 (T-51 PR C) merged as
`f401c29`; PR #79 (crash-resume) merged as `a5f9a70`.

**PR D** adds the GUI card that turns bundles on, which is what makes PR B (11 supervised
bundles) and PR C (3 one-class bundles) reachable without writing Python. Plan:
`docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md` §4. After it: E/F
(one-class clamp; F needs its own approval and edits `suggest_one_class_params`, the one
planned exception to the no-sampler-edits rule).

**Before writing PR D, read these two — both cost a full review round in PR C:**
1. **Adding to `BUNDLES` breaks registry-wide assertions** in
   `tests/test_t51_supervised_bundles.py` (they iterate the whole registry and fit
   supervised models). PR C scoped them to a `SUPERVISED_BUNDLES` subset; PR D will hit
   the same tests if it adds or changes bundles.
2. **A new GUI input that feeds the Bayesian study-name hash must go in
   `CAPTURABLE_SETTINGS` AND `BAYESIAN_REQUIRED_SETTINGS`** (see §1). A bundle-enable
   control is exactly such an input: miss it and a changed control silently starts a
   different study while the finished run releases the saved one's record.

**PR C's known limitation, if PR D surfaces these bundles in the GUI:** `if_max_samples`'
fractions need >= 2 inliers per training fold. Too few successful folds → the trial scores
+inf and never reaches the leaderboard; but under repeated CV half the folds suffice, so
the row IS scored from those alone, unmarked, possibly omitting inliers. The GUI should not
promise more than that. Wording is in the bundle `help` text — reuse it, don't reinvent it.

### 1. PR #79 (crash-resume) is MERGED — `a5f9a70`. Nothing pending on it.
13 review rounds. Rounds 1-8 were reviewed by Codex + DeepSeek Flash + GLM 5.3; rounds 9-13
by Codex alone (it blocked 8-12 and returned **MERGE** on 13 with no findings). Branches and
worktrees are deleted. Full non-GUI suite 3405 passed / 26 skipped; `tests/gui/test_resume_*.py`
91 passed; F821 clean.

**The design, worth knowing before touching the GUI's Run Analysis path:**
`_confirm_resume_before_launch` is the single main-thread launch gate. It decides
resume/delete/fresh, claims the run, and freezes everything the worker needs — models, trial
count, data, optimization method + task type, a `capture_gui_settings` snapshot, and the
calibration rows (holdout / excluded / active). The worker reads those arguments, never live
Tk state. **Nearly every Codex block in rounds 9-12 was a place where the worker still
re-read something the gate had decided.** Add a new Bayesian input → add it to the snapshot
(`BAYESIAN_REQUIRED_SETTINGS`) and to `CAPTURABLE_SETTINGS`, or a changed control silently
starts a different study and the finished run releases the saved one's record.

**Binding user decisions (unchanged):** a data mismatch never deletes anything by itself; a
saved paused/failed/crashed run is offered until it is resumed to completion or deleted;
resume never changes the persistence radio.

**Known limitations (in CHANGELOG under 0.5.0b3):** two dasp windows share one
`active_run.json` with no file locking; a finished run whose record couldn't be deleted is
offered again; a run whose store is missing/empty is not offered; resume restores only
whitelisted GUI settings, not hyperparameter grids. Also out of scope and still live: grid,
one-class grid, NSGA-II and post-search paths read live Tk state (they don't touch run_state).

### 2. Merged this session (2026-09-14/15)
| PR | What |
|---|---|
| #69 | T-41 'auto' re-run no longer deletes an earlier study |
| #68 | T-51 PR A: opt-in extra Optuna axes mechanism |
| #70 | Test-order leak fix (`reimport_modules` fixture) |
| #71 | CI: 5 drifted tests fixed, actions v7, docs paths-ignore, PR concurrency, timeouts |
| #72 | T-51 PR B0: PLS-DA head params survive rebuild/refit/ensemble/export |
| #73 | CatBoost never writes `catboost_info/` (failed fits under Program Files) |
| #74 | T-51 PR B: 11 opt-in supervised bundles (Python-only until PR D) |
| #75 | Removed Linux Xvfb GUI CI job (T-CI-2 was a too-short timeout, not a hang) |
| #76 | CI: per-sha push concurrency; **blocking flake8 F821/F822/F823 gate** |
| #77 | Ensembles: tuned `model__*` params + correct row preprocessing (incl. chromosome rows, float64), PLS-DA class_weight/seed, legacy CatBoost refit, GUI NameErrors; shared row helpers on declared surface §8b. **Ensemble scores change.** 5 review rounds. |
| #78 | Bayesian: no automatic study deletion after failed migration; baseline_params NameError; svm_gamma pair rejection; 'pls-da'; equal categorical choices; numpy scalars; legacy-study warning. 3 review rounds. |
| #79 | Crash-resume: persistence setting kept; saved run offered until resumed or deleted; damaged record reported not replaced; everything the Bayesian worker uses frozen at the click. 13 rounds. See §1. |
| #80 | T-51 PR C: 3 opt-in one-class bundles (`if_max_samples`, `lof_metric`, `ocsvm_poly`). Python-only until PR D. GLM 5.3 + 4 Codex rounds; see SESSION_LOG 2026-09-16. |

Review process used (user preference): **Codex + DeepSeek Flash + GLM 5.3** on every PR,
re-review each round until clean. A post-merge round on #68-#75 found real pre-existing bugs
(fixed in #76-#78). See SESSION_LOG 2026-09-14 "Post-merge review round".

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
1. **T-51 next:** PR D (GUI card to enable bundles) — see §0 — then E/F (one-class clamp;
   F needs its own approval).
2. **Smaller follow-ups:**
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
3. **`SESSION_LOG.md` housekeeping done 2026-09-15** (1705 → 521 lines): batches 6 and 7 in
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

---

## ⚠ MOVING A 3.12 MACHINE TO 3.14 (added 2026-09-12) — DO THIS FIRST

If this machine is still on `.venv312`, do this **before** pulling and before trying
to run anything. The project is now Python 3.14 only.

> **The trap:** `pyproject.toml` now sets `requires-python = ">=3.14"`. The moment you
> pull, `pip install -e .` **fails inside `.venv312`** with a `requires-python` error.
> That is expected, not a broken checkout. Build the new environment instead of trying
> to repair the old one.

```bash
# 1. Install Python 3.14 (ordinary GIL build, NOT the free-threaded "t" variant)
winget install Python.Python.3.14
py -3.14 -V                       # expect 3.14.x

# 2. Pull
cd <repo>
git checkout main                       # PR #65 merged 2026-09-13 (6b956c8)
git pull

# 3. Build the new environment alongside the old one. Do NOT delete .venv312 yet.
py -3.14 -m venv .venv314
.venv314\Scripts\python -m pip install --upgrade pip
.venv314\Scripts\python -m pip install -r requirements-lock.txt
.venv314\Scripts\python -m pip install -e . --no-deps
.venv314\Scripts\python -m pip check          # expect: No broken requirements found.

# 3b. Existing .venv314? Re-apply the lock after every pull. The launchers now do
#     this automatically when the venv differs from requirements-lock.txt;
#     this is the manual equivalent. A venv built from the 09-10 lock still has
#     jcamp 1.2.2, and JCAMP-DX import is broken until it is resynced.
.venv314\Scripts\python -m pip install -r requirements-lock.txt
.venv314\Scripts\python -m pip install -e . --no-deps
.venv314\Scripts\python scripts\check_env_lock.py   # expect: Environment matches ...
#     (pip check only verifies that installed packages satisfy each other's
#     declared requirements, so it misses drift from the exact lock pins.
#     check_env_lock.py compares against the lock.)

# 4. Verify before trusting it
.venv314\Scripts\python -m pytest -q -p no:randomly --tb=no -rf
#    Compare failing node IDs with the known baseline below. Counts vary with
#    platform and optional dependencies. The bar is ZERO NEW failures.

# 5. Launch the GUI
.venv314\Scripts\python spectral_predict_gui_optimized.py
```

**The launchers are already updated on this branch.** `install.bat`, `install.sh`,
`RUN_SPECTRAL_PREDICT.bat` and `run_gui.sh` all target 3.14 / `.venv314`, and the
installers now install `requirements-lock.txt` instead of resolving pyproject floors.
Steps 3–4 above are exactly what `install.bat` does, so you can just run that.

**Note that `py` now defaults to 3.14** once installed, so a bare `py` or `python` no
longer means what it did. Always use the explicit `.venv314\Scripts\python` path.

**Machine progress (2026-09-13):** the primary Windows machine is on `.venv314`
(Python 3.14.7, jcamp 1.3.2, `pip check` clean). Full suite: all 3,042 tests ran and
only the five baseline IDs failed (listed in the installation validation report
below). But Python crashed with an access violation in the GUI `session_app`
teardown after the last test, so pytest printed no summary. It did not recur in a
later `tests/gui` run: 132 passed, 2 baseline failed. The GUI launches and runs an
analysis. Its first launch exposed a Model Development CV-row `TclError` on the
default kfold, fixed in `a746d5c`. **Other machines:** pull, then launch with
`RUN_SPECTRAL_PREDICT.bat` (it resyncs the venv to the lock) or do step 3b by hand.
JCAMP-DX import is broken in any `.venv314` still on jcamp 1.2.2. **Automatic lock
sync (PR #66, merged 2026-09-13 as `77bb4ad`):** the launchers run
`scripts/check_env_lock.py` and reinstall from the lock on drift, the installer build
refuses a drifted build venv (`DASP_ALLOW_LOCK_DRIFT=1` overrides), the frozen
self-test checks `jcamp.readfile`, and `read_jcamp_file` raises an actionable
`ImportError` on jcamp <1.3. **Still
untested on 3.14:** importing a JCAMP-DX file through the GUI (the backend round
trip passed), and a real installer build with the new lock-drift gate (logic is
covered by `tests/test_build_and_launcher_safety.py`).

**Keep `.venv312` for now.** It is the rollback lever: the build path is parameterized,
so `DASP_BUILD_PYTHON=312 DASP_ALLOW_LOCK_DRIFT=1 python build_installer_py312.py` rebuilds on 3.12. A true
rollback would also mean lowering `requires-python` again. Delete `.venv312` only once
you are confident, and reclaim the disk then.

**Installer validation complete locally (2026-09-12, application `db975e2`):** a
real installer built from base `main` (`8de7445`) and retained `.venv312` was
installed, then upgraded in place. The first attempt exposed 3,309 obsolete
runtime files and mismatched package metadata; a GUI analysis also exposed the
missing `logging.handlers` dependency. Both are fixed. The installer replaces
only its app-owned `_internal` directory, and PyInstaller now analyzes `src`.

The corrected upgrade and a fresh installation each match all 20,384 build
runtime files by SHA256. Both pass the expanded executable smoke test (44/44
imports, numerical metadata, booster fits, 99-row search and model round trip)
and programmatic installed-GUI analysis/model loading. A model saved under 3.12
reproduces all 30 reference predictions after upgrade. Both uninstall checks
preserve user files and remove the application registration. The test installs
were removed; the source shortcut and retained `.venv312` are unchanged.

These were current-user installations on this development machine, not a fresh
Windows VM or manual desktop test. The native automation bridge was unavailable.
The final CI comparison also passes: Windows has 3,008 passed and the same five
baseline failures; Linux and optional dependencies each have 2,880 passed and
the same three baseline failures. Build passes; the informational GUI timeout
matches base. **PR #65 was merged into `main` on 2026-09-13 as `6b956c8`** (merge
commit, user-authorized, guarded to head `8bd3f8f`). See the
[installation validation report](reviews/2026-09-12-pr65-installation-validation.md)
for exact commits, artifact hash, review scope and remaining limitations.

---

## ⚠ FIRST PULL ON A NEW MACHINE (added 2026-07-30) — DO THIS BEFORE ANYTHING ELSE

If this checkout is the first one on this machine to see commit `763c4ed` or later,
**run this before running or testing anything:**

```bash
# from the repo root, in the project venv (.venv314 — see the 3.14 section above)
pip install -e . --no-deps
```

> Superseded in practice by the 3.14 section above: building a fresh `.venv314` cannot
> inherit a stale shim, so this only matters for an environment that predates the
> migration. Kept because the failure mode below is confusing if you hit it.

**Why it is mandatory, not housekeeping.** The `spectral-predict` console script was
retired and `src/spectral_predict/cli.py` deleted. A `git pull` removes the source but
**leaves the installed launcher behind**: `.venv312/Scripts/spectral-predict.exe`
survives and fails with

```
ModuleNotFoundError: No module named 'spectral_predict.cli'
```

Re-running `pip install -e .` removes the stale shim. Verified on the primary machine
2026-07-30: present and broken before, gone after.

### The other three things to know on first pull

1. **There is no CLI. It is not coming back.** Retired 2026-07-29, abandonment
   confirmed by the user 2026-07-30 — Codex proposed a one-release deprecation stub and
   the user declined. Do not add one, and do not re-add a console script "for
   convenience". Humans use the GUI (`python spectral_predict_gui_optimized.py`);
   scripts and agents compose backend primitives directly.

2. **Read `docs/AGENT_COMPOSITION.md` before writing any script that calls the
   backend.** `CLAUDE.md` makes this a MUST. It documents the stable surface and the
   traps that otherwise cost a turn each — readers and `run_search` both return
   **tuples**, selectors return **score arrays not masks**, `preprocessing_methods` is a
   **dict of bools not a list of strings**, and the saved-model metadata key is
   **`"preprocessing"`, not `"preprocess"`**. Every example in it has been executed
   against the repo's own `example/` data.

3. **`main`'s CI is red, and has been since ≈June 2026.** That is the pre-existing
   T-CI-1 rot, NOT this merge. Do not treat a red check as a signal about your own
   branch — until T-CI-1 closes, diff your failure set against `origin/main` and
   confirm you add zero NEW failures. Current known-red on `main`:
   `test_export_code.py` (2), `test_cv_strategy.py` (1), `test_t19_class_weight_per_library.py` (2),
   **plus two GUI tests** (`tests/gui/test_comprehensive.py::test_catboost_via_gui`,
   `tests/gui/test_multiclass_gui.py::test_tab9_rejects_multiclass_primary`). That is
   **7 total** — the list above previously said 5 and was stale. Verified 2026-09-12
   by full runs on 3.12, on 3.14, and on 3.14 with every dependency upgraded: the
   same seven fail in all three, 2971 pass, 33 skip.

> ## ▶ ACTIVE DIRECTION (2026-09-12) — **Python 3.14 migration COMPLETE and MERGED to `main`** (PR #65, `6b956c8`, 2026-09-13), all dependencies current
>
> **The project is Python 3.14 only.** `requires-python = ">=3.14"`, CI matrix is
> `['3.14']`, classifiers list 3.14 alone. Earlier versions are not supported and
> pip refuses to install on them. Use the ordinary GIL build, **not** free-threaded.
> Recreate an environment with `py -3.14 -m venv .venv314` +
> `pip install -r requirements-lock.txt` + `pip install -e . --no-deps`.
>
> **What was verified, not assumed.** The analysis in `docs/PYTHON_UPGRADE_DECISION.md`
> was read-only — its own Appendix B says nothing was ever installed, built or run.
> All of it has now been executed:
> - All 114 pinned distributions install on CPython 3.14.7; whole stack imports.
> - Full suite: **7 failed / 2971 passed / 33 skipped, identical on 3.12, on 3.14,
>   and on 3.14 with every dependency upgraded. Zero new failures.**
> - **The frozen bundle builds AND runs** — 42/42 imports, all three booster DLLs,
>   threading fallback engaged, a real LightGBM CV job completing (the historical
>   fork-bomb scenario), GUI launching, and a 243.9 MB Inno installer produced.
>
> **Dependencies are all current except two**, held back by an upstream pin, not by
> oversight: `alive-progress 3.3.0` requires `about-time==4.2.1` and
> `graphemeu==0.7.2` exactly and is itself the latest release. Majors that landed:
> optuna 5.0, plotly 7.0, moocore 0.3.2, xgboost 3.4.1, numpy 2.5.3, sklearn 1.9.1.
>
> **Rollback is one variable.** The build path no longer hardcodes an interpreter;
> `DASP_BUILD_PYTHON=312 DASP_ALLOW_LOCK_DRIFT=1 python build_installer_py312.py` rebuilds on 3.12. The
> user-visible artifact names still say `py312` **deliberately** — they are a stable
> identity so existing installs upgrade in place. Do not "fix" them.
>
> **Keeping current is now a routine:** `python scripts/upgrade_check.py` reports
> what is outdated, what is risky (numerical vs infrastructure), what cannot move
> and why, and what ordering version caps force. `docs/upgrade/UPGRADE_RUNBOOK.md`
> is the process; run it quarterly or when a new Python minor ships.
>
> **Two pre-existing issues found en route, NOT fixed here** (separate tickets):
> Optuna study identity omits Python/dependency versions, so a *resumed* study can
> return cached scores computed under a different numerical stack; and `MultiGroupEPO`
> seeds off `hash(label)`, which is `PYTHONHASHSEED`-dependent. See
> `docs/upgrade/PYTHON_UPGRADE_PLAN.md`.
>
> **UPDATE — BOTH ARE NOW FIXED (2026-09-12).** Codex's design was followed.
> Optuna study names carry a numerical-environment digest
> (`unified_bayesian_<model>_<confighash>_env1_<envhash>`), the readable
> environment is stored in `study.user_attrs["numerical_environment"]`, an
> incompatible prior study produces an explicit notice instead of a silent fresh
> start, and an unreadable package version raises rather than degrading to a
> placeholder. MultiGroupEPO uses a stable blake2b label digest and sorted group
> assembly. Existing pre-fix studies are intentionally no longer auto-resumable;
> their databases stay intact. **Still open:** the separate GUI
> `EstimatedEPO(random_state=None)` path remains nondeterministic, and a full
> cross-version SQLite replay matrix is not implemented.
>
> The original review recommendation follows.
>
> **Review recommendation (2026-09-12, Codex): FIX NOW for both pre-existing bugs.**
> Optuna needs environment-specific study names,
> stored environment metadata, and a visible fresh-study notice for incompatible
> or legacy caches; preserve existing databases, but do not resume their unknown
> scores. MultiGroupEPO needs a stable label digest and sorted group assembly.
> The latter does not fix the separate GUI EstimatedEPO(random_state=None) path.
> Cross-version SQLite replay and cross-process EPO output drift reproduced;
> 110 focused existing tests passed. Details are in SESSION_LOG.md.
>
> **Installation follow-up:** fresh-directory installation and a real in-place
> upgrade now pass on this development machine; a fresh-OS check remains untested.
> See the final validation report linked above.

> **Review findings fixed in `f60cfa5` (2026-09-12).** PR 65 was reviewed by Claude,
> GLM 5.3 and Codex (`gpt-6-astra`), and Codex confirmed each fix before it was applied.
> Fixed: fingerprint now rejects missing/empty package version metadata; the resume
> notice separates legacy (pre-fingerprint) study names from other-environment ones;
> MultiGroupEPO sorts mixed int/str group labels with a type-tagged key (string-label
> output bit-identical); deprecated `warn_independent_sampling` removed; launcher
> points to `install.bat` when `.venv314` is missing; `run_gui.sh` repairs from the
> lockfile; launcher tests use explicit paths (`NoDefaultCurrentDirectoryInExePath=1`);
> installer label, harness hash-seed claim and README scipy floor corrected.
> **Deferred:** (1) a fingerprint read failure still aborts `never`-mode in-memory
> runs; this is deliberate, and relaxing it must keep persistent-study strictness.
> (2) Optuna's `consider_endpoints` is also deprecated (removal in 6.0), but dropping
> it changes its effective value, so it needs a numerical A/B before removal.

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

- **T-51 — Opt-in Bayesian search-space axes (ticket written 2026-08-30; design complete, no code).** Full ticket: `docs/plans/2026-08-30-T51-bayesian-opt-in-search-axes.md`. **Premise is added value, not a defect** — supervised Bayesian performs well and the ticket does not assume otherwise. It adds opt-in knobs for hyperparameters that currently take exactly *one* value (LightGBM `reg_alpha=0.1`/`subsample=0.8`/`min_child_samples=5`, XGBoost `colsample_bytree=0.8` with `gamma`/`min_child_weight` absent, RandomForest `max_features='sqrt'`, SVM `gamma='scale'`, PLS-DA logistic head `C=1.0`), curated per model family, all off by default. **Design:** `suggest_model_params` / `suggest_one_class_params` stay byte-for-byte; a new `search_spaces.py` supplies `apply_extra_axes()` that runs after them and is a literal no-op when no bundle is enabled. GUI checkboxes and the Python API drive the same bundle ids (`enabled_extra_axes=(...)`). One-class included from the start, closing the PR #58 deferral. **Two hard constraints discovered in review (see SESSION_LOG 2026-08-30):** (a) Optuna forbids re-suggesting a parameter name, so only *pinned-constant* axes can be opened additively — ranges already searched (Ridge/Lasso/ElasticNet `alpha`, MLP `alpha`, OneClassSVM `gamma`, PCA-SIMCA `n_components`) cannot be widened this way, which is fine since widening them is explicitly out of scope; (b) clamping after `trial.suggest_*` does not change what TPE learns. **Two genuine bugs ride alongside, sequenced separately:** the `'SVC'`/`'SVM'` string mismatch leaving classification SVM unscaled (prerequisite for any SVM `gamma` knob), and the PLS clamp asymmetry where Bayesian bounds `n_components` by `n_features` while the grid path uses `compute_min_train_fold_size` (approved to ship last, own gate). Reviewed by Codex gpt-5.5 and a DeepSeek+GLM peer panel.

- **Non-balanced imbalance ratios + ElasticNet for PLS-DA inner LR — niche reproducibility gap (revised 2026-05-07; supersedes 2026-04-29 framing).** The original bullet claimed boosters and PLS-DA's inner LR don't receive any class_weight handling. That's stale: PR #38 added a dispatcher in `_apply_class_weight_discriminator_for_rebuilt_model` (`search.py` ~4555-4611) that injects balanced reweighting for every classifier when the user picks `class_weight` from the Data Quality dropdown. CatBoost gets `auto_class_weights='Balanced'`; sklearn estimators with a `class_weight` attribute get `class_weight='balanced'`; XGBoost / LightGBM / RidgeClassifier get per-fold `compute_sample_weight('balanced', y)` threaded as `sample_weight` (cv_utils.py `_fit_with_early_stopping`); PLS-DA's inner LR gets `class_weight='balanced'` at the Pipeline build site; MLP warns and falls back. Codegen parity in exported scripts via runtime `IMBALANCE_METHOD == 'class_weight'` conditional (code_generator.py). **Remaining gaps:** (a) non-balanced ratios — XGBoost `scale_pos_weight=N` for a specific positive ratio, CatBoost `auto_class_weights='SqrtBalanced'` for softer correction at extreme imbalance, custom `class_weight={0: w0, 1: w1}` dicts; (b) ElasticNet penalty for PLS-DA inner LR (`penalty='elasticnet'` + `l1_ratio`) — separate from imbalance, surfaced by the same FTIR Bone PLS paper reference; (c) per-model imbalance override — the dropdown is global, which is arguably the *correct* design for a fair model-comparison search (per-model would confound rankings). Effort if ever needed: ~half-day for non-balanced ratios via one branch per model class in the existing dispatcher + GUI numeric entry + codegen mirror. User judgment 2026-05-07: niche enough to leave open until a specific paper-reproduction asks for it. Original proposal at `docs/plans/2026-04-29-model-native-loss-reweighting.md` is preserved as record but its framing is superseded by this revision.

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
