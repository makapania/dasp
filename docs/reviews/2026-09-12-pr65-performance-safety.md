# PR #65: performance and safety review

Reviewed 2026-09-12: [PR #65](https://github.com/makapania/dasp/pull/65),
head `e13393fbab3f676bc3d23a382728a2f4a43723a8`, base
`8de744595e17a6317a43153b2b7364a22973bd74`. Both remote refs were verified.
The initial review and follow-up audit made no application changes. The user
subsequently authorized all three fixes; see the implementation record below.

**The three fixes below are now implemented.** The new environment passes focused
checks, but the upgrade is not a uniform performance improvement. Installation
testing remains outstanding; the implementation record distinguishes the new
checks from the original review's evidence.

## Findings in the reviewed head (now fixed)

### P2: Preserve spaces in the build interpreter's site-packages path

`build_installer_py312.py:157-158` parses two lines of subprocess output with
`stdout.split()`. A checkout under `C:\Users\Jane Doe\dasp` consequently returns
`C:\Users\Jane` as the site-packages directory. The pandas repair at lines
277-282 then skips its nonexistent source and reports that the checked files
match, leaving a corrupt bundled `pandas/util/__init__.py` unrepaired.

This is introduced by the PR: previously the repair path was built directly from
`PROJECT_ROOT`. The existing session log records the pandas collision in every
one of the three migration builds; it is a practical startup risk, conditional on
the collision and a build path containing spaces.

**Reproduction:** execute the unmodified helper extracted from the current AST,
with subprocess stdout `314\nC:\Users\Jane Doe\dasp\.venv314\Lib\site-packages\n`.
The actual returned path is `C:\Users\Jane`.

**Fix:** use `splitlines()` or JSON for the subprocess result; report a missing
repair source as a failed verification rather than a match.

### P2: Stop the launcher immediately if lockfile installation fails

`RUN_SPECTRAL_PREDICT.bat:25-27` now performs two pip commands but checks
`errorlevel` only after the second. If installing the lock fails, for example
because a wheel cannot be downloaded, the subsequent editable install with
`--no-deps` can succeed. That replaces the error status, bypasses the failure
message, and starts the GUI with incomplete or stale dependencies.

**Reproduction:** run a temporary copy of the actual batch file with only its
Python executable calls replaced by a batch stub. Return 1 for the missing-package
check, 1 for the lock install, and 0 for the editable install. The trace reaches
`gui_launched`, omits the installation error, and exits 0. No real environment was
modified. `install.bat` already checks both commands separately.

**Fix:** check the lockfile command's result before running the editable install.

### P2: List study names without loading every trial in the database

`src/spectral_predict/unified_bayesian.py:2704-2707` calls
`get_all_study_summaries()` only to extract names. The installed Optuna 5.0.0
implementation (`optuna/study/study.py:1594`) loads all trials for each study,
including their user attributes and numerical arrays. The new advisory check
therefore scans unrelated models' histories before each `always`-persistent
search, including crash recovery. Its cost grows with accumulated trial payloads,
even when the requested study has no compatible history to resume.

**Measurement:** three studies, 80 completed trials each, with a 2,151-value
importance vector and 240-value prediction vector per trial, produced an 11.4 MiB
SQLite database. Across three calls, median summary enumeration took **0.205 s**;
name enumeration took **0.0197 s**, approximately **10.4 times faster**. Absolute
cost is modest for this small fixture; the avoidable history load is the concern.

**Fix:** use [`optuna.study.get_all_study_names()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.get_all_study_names.html).
Setting `include_best_trial=False` is insufficient: the installed implementation
still loads all trials to calculate counts and dates.

## Runtime comparison

Compared the existing `.venv312` against the PR's `.venv314`, without changing
either. This measures the interpreter and dependency upgrade together, not the
interpreter alone. The older environment has numpy 2.4.4 / sklearn 1.8.0 /
LightGBM 4.6.0 / XGBoost 3.2.0; the new one has numpy 2.5.3 / sklearn 1.9.1 /
LightGBM 4.7.0 / XGBoost 3.4.1.

The workload was a fixed synthetic regression matrix, 240 samples by 512
features, three shuffled folds, seed 42, and one native/model thread. Each model
had one warmup followed by three measured repetitions. Two rounds were run in
opposite environment order; the table uses the median of six measurements per
environment. Each measurement includes fitting and prediction across all folds.

| Model | 3.12 stack | 3.14 stack | Change in elapsed time |
|---|---:|---:|---:|
| PLS, 8 components | 0.00468 s | 0.00506 s | +8%; less than 1 ms |
| Ridge, alpha 1 | 0.00272 s | 0.00302 s | +11%; less than 1 ms |
| RandomForest, 75 trees, depth 8 | 2.633 s | 2.014 s | **24% faster** |
| LightGBM, 75 trees, 15 leaves | 0.256 s | 0.265 s | +4% |
| XGBoost, 75 trees, depth 4 | 1.887 s | 2.160 s | **14% slower** |

Reported RMSEs matched across stacks for all five models. These timings are a
bounded workload check, not a GUI search benchmark or a guarantee for larger
datasets. In particular, the XGBoost result warrants a representative application
benchmark if that model dominates a user's workload; it does not by itself prove
an implementation bug or justify rejecting the whole migration.

## Verification and limits

- `.venv314` is CPython 3.14.7; all 119 lockfile pins match the installed
  distribution metadata, and `pip check` reports no broken requirements.
- **125 passed** across environment fingerprint, cross-process EPO determinism,
  Bayesian deduplication, T-41 persistence, T-42 write paths, and contaminant
  analysis. **9 passed, 1 skipped** in the JCAMP reader suite.
- A real PLS SQLite search resumed the same study under the same environment.
  Mocking numpy's installed version created a different study, completed a new
  trial, and emitted the incompatibility notice. This exercises persistence
  integration; it is not an actual cross-interpreter replay matrix.
- The existing frozen artifact predates `dcde845`: its bundled
  `unified_bayesian.py` has no environment fingerprint definitions, and its
  `contaminant_analysis.py` differs from HEAD. The executable timestamp is 15:46;
  the bug-fix commit is 17:23 on the same day. Running that artifact cannot verify
  the final PR. Rebuild and test the final head before distributing it.
- No clean installation, in-place installed-app upgrade, or uninstall was
  performed. The overlay installer still has the previously documented obsolete
  payload risk. This review does not claim those paths are safe.
- The full suite was not rerun in this review. The PR reports seven pre-existing
  failures with zero new failures; those historical runs are distinguished from
  the focused checks executed here.

The stable artifact names containing `py312` are intentional and are not a
finding. The GUI's pre-existing unseeded `EstimatedEPO` path is also outside these
changes. No missing-dependency fingerprint finding is asserted without proving
that dependency participates in the Bayesian calculation.

## Follow-up: could name-only enumeration lose anything?

**No, for this exact replacement and the pinned Optuna 5.0.0 implementation.**
The user specifically asked whether the summary load saves state needed later.
It does populate an in-memory trial cache, but `get_storage(storage_url)` creates
a fresh temporary `_CachedStorage` for that call. The application retains only
the study-name strings. Its later `create_study` and `load_study` calls create
separate storage objects; fingerprints, history, parameters, importances and
leaderboard rows are reloaded from the selected `study.trials`. The discarded
summary cache supplies none of these.

Both enumeration APIs call the same storage constructor and `get_all_studies`.
Both can create/initialize a missing database; the existing `always`-mode gate
must remain in place. Changing only the enumeration inside that gate preserves
this behavior. It also preserves the incompatible-environment notice.

**Executed comparison:** the proposed one-line replacement was compiled in
memory, without editing the application. An actual 24-trial PLS study and an
unrelated study with array payloads and completed/pruned/failed/running trials
were created in a temporary SQLite database. Two copies were then exercised by
the original function and the proposed function.

- Both enumeration calls returned the same names and left the complete SQLite
  schema/data dump unchanged. SQL tracing found zero data writes for either;
  summaries issued two trial SELECTs, names issued zero.
- Resuming at the same 24-trial budget left the entire database unchanged in both
  cases. All trial records, dates, distributions, parameters, user/system attrs,
  stored arrays and 23 saved fingerprints were preserved. The 23-row leaderboard
  matched exactly.
- Continuing both copies to 26 trials produced identical new TPE suggestions,
  scores, parameters, distributions, arrays, attrs, leaderboards and progress
  callbacks. Existing trials were preserved exactly; new trial timestamps were
  excluded from comparisons because the runs occurred at different times.
- A retained legacy study triggered the identical incompatibility callback in
  both versions, while its original trial and metadata stayed intact.
- `auto` warmup and `never` mode made no name lookup and created no database.

[Fable's independent opinion](2026-09-12-pr65-fable.md), requested by the user,
agrees the swap loses nothing and confirms the other two PR findings. Fable
reviewed source; the executable database comparisons above were done by Codex.
The timing is an avoidable performance cost, not a data-corruption defect. Fable's
attribution of all benchmark differences to library versions is stronger than
the experiment supports: it changed interpreter and dependencies together.

That follow-up was verification only: at that point the proposed source change
was not applied, and no production study or virtual environment was modified.

## Implementation after user authorization

All three review fixes are applied:

- The build interpreter's output is parsed with `splitlines()` to preserve
  spaces. Missing pandas repair source or destination now fails verification
  instead of reporting a match.
- The launcher checks each pip command immediately and sends either failure to
  a shared error label that exits nonzero. Successful startup exits zero.
- The compatibility notice uses `optuna.study.get_all_study_names()` inside the
  existing `always`-mode gate. Actual study loading and resume are unchanged.

Ten permanent regression cases were added in
`tests/test_build_and_launcher_safety.py` and
`tests/test_bayesian_study_lookup.py`. Running them before the source edits
produced five expected failures; all ten pass after the fixes. They exercise a
corrupt pandas module under a path containing spaces, missing verification files,
both launcher install failures and success paths, preservation of stored trials
and leaderboard rows on resume, continued trial budgeting, and both legacy and
different-environment warning paths.

**77 focused tests passed**, including the existing environment fingerprint,
deduplication, T-41/T-42 persistence and build-version checks. The two new test
files pass Black (target Python 3.14) and flake8. The full suite was not rerun for
these small changes.

A fresh standalone and installer build completed successfully with Python
3.14.7, replacing the stale artifact described in the original review. The build
encountered and repaired the pandas.util collision; the repaired module matches
the virtual environment byte-for-byte. All 75 bundled project Python files match
the current source, including the environment fingerprint and stable EPO digest.

The packaged executable's hidden `--test` run returned 0 in 16.84 seconds. Its
saved output confirms 42/42 imports, all three booster fits, active frozen
threading fallback and a completed PLS/LightGBM cross-validated search with 99
ranked rows. A console-encoding failure in the capture helper occurred only when
displaying that completed run's log; inspecting the saved output confirmed the
success without repeating the test.

Local artifact: `dist/installer/SpectralPredict_Setup_py312_0.5.0b2.exe`,
230,106,755 bytes (219.4 MiB), SHA256
`171742e9f918ee776416d12cc25998a5a64d1b4f86e597ecc17d70039f021c9a`.
Clean installation, in-place installed-app upgrade and uninstall were not
tested; those release checks remain outstanding.

## Final installation follow-up

The initial artifact and outstanding-installation statement above are historical.
The [final installation validation](2026-09-12-pr65-installation-validation.md)
records the newer `f60cfa5` and `db975e2` commits, two additional packaging fixes,
the corrected artifact, and completed fresh-installation, real upgrade, installed
GUI/model and uninstall checks. The final full CI comparison also passes with
zero new failures: Windows has the same five failing node IDs as base, and both
Linux jobs have the same three.
