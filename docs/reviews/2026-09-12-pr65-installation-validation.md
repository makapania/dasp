# PR #65: installation and final validation

Application commit: `db975e2a4551e3e521ca75d2fb8a3fe82324718d` on
`feat/python-314-upgrade`. Base: `8de744595e17a6317a43153b2b7364a22973bd74`.
This includes the additional review fixes in `f60cfa5`, which Codex explicitly
reviewed and tested after noticing the concurrent commit in the branch history.

**Local installation checks and the final CI comparison pass. Every failing CI
node matches the existing baseline. PR #65 remains open and ready to merge.**

The work is committed and pushed (`fbe8e95`; the subsequent `ecad576` adds review
notes only). An attempt to merge the reviewed head was stopped before execution
by automatic approval review, which did not find explicit authorization for the
merge into `main`. Only that explicit user confirmation remains; no further code
fixes or validation runs are pending.

## Review scope

Codex performed the source review and executable checks recorded here. The
user-requested final Fable call reviewed `89f9479`, before `f60cfa5` and `db975e2`.
It found no material merge blocker and identified an Optuna minimum-version
mismatch; `pyproject.toml` now requires `optuna>=3.4.0`.
[Fable's complete returned opinion and model provenance](2026-09-12-pr65-fable-final.md)
are preserved separately. Fable did not review the two later commits or execute
their installation checks.

## Installation findings and fixes

1. **The frozen GUI lacked `logging.handlers`.** The original executable passed
   its 42-import smoke test, but a real GUI analysis callback failed when importing
   `spectral_predict.run_logging`. The file was absent from the packaged modules,
   and PyInstaller reported backend modules as unresolved. Its analysis search
   path included the repository root but omitted the `src` directory. Adding
   `src` to `Analysis.pathex` fixes dependency discovery. The older baseline bundle
   also had this omission.
2. **Overlay upgrades retained obsolete runtime files.** Installing the original
   PR artifact over the baseline 3.12 installer retained 3,309 obsolete files,
   including `python312.dll` and old package metadata. NumPy imported as 2.5.3
   while `importlib.metadata.version("numpy")` returned 2.4.4. Seven numerical
   packages had mismatched metadata, undermining numerical-environment identity.
   The installer now replaces only its app-owned `{app}\_internal` directory.
   Normal run state, SQLite studies and logs use the separate user-data directory
   (`%LOCALAPPDATA%\dasp` on Windows); files elsewhere under `{app}` are retained.

The executable's smoke test now imports `run_logging` and `run_state`, compares
runtime and distribution versions for eight numerical packages, and saves,
reloads and predicts with a `.dasp` model. As a negative control, this extended
test function was executed against the original upgraded installation: it failed
exactly eight checks, detecting the missing module and seven metadata mismatches.
That control used the new function in an isolated host; it was not the original
executable's unchanged `--test`.

## Executed checks

The host was Windows 11 Home. It had a source-launcher shortcut but no registered
prior installation. The baseline installer was therefore built from the exact
base `main` commit using the retained `.venv312`, then installed in an isolated
workspace directory. The corrected installer used CPython 3.14.7 and the current
119-entry lockfile. Both installation paths contained spaces.

| Check | Result |
|---|---|
| Focused build, launcher, Optuna identity/resume/dedup and persistence suites | 81 passed |
| Latest `MultiGroupEPO` determinism suite | 5 passed, including subprocess and mixed-label cases |
| Unix launcher control flow in Git Bash with a stub Python | 6 cases passed; lockfile/editable failures stop startup |
| Corrected standalone executable `--test` | Exit 0; 44/44 imports |
| Actual in-place upgrade and fresh installation | Both exit 0; no reboot |
| Installed runtime versus corrected build, before adding test-host files | All 20,384 files match by SHA256 in both installations; zero extra/missing/different files |
| Corrected installed executable `--test` | Both exit 0; all version checks, booster fits, 99-row PLS/LightGBM CV search and 30-prediction model round trip pass |
| Installed GUI integration | Both complete a three-row PLS analysis and load a saved model through the GUI callback, without error dialogs |
| Model saved with baseline Python 3.12 stack | Reloads under the upgraded runtime; all 30 predictions agree within `rtol=atol=1e-12` |
| Uninstall after upgrade and after fresh installation | Both exit 0; executable, runtime DLL and application registration removed; user model and note retained |

The GUI integration check executed the packaged GUI bytecode with the installed
Python DLL and installed libraries, asserted that imports came from the installed
directory, invoked the actual Run Analysis and Load Model File(s) buttons, and
ran Tk's event loop and background worker. It used isolated profile directories
and controlled file dialogs. This was programmatic GUI integration, not manual
desktop interaction; the native automation bridge was unavailable.

The retained model's SHA256 before and after upgrade/uninstall is
`4bd9b402eb0e1faa15adabbaf9b140e69d89e29e36fabd79325d9b2267931cb3`.
The fresh-install uninstall also compared both model and note hashes before and
after; both match. The first upgrade-uninstall note check incorrectly expected
LF instead of the original Windows CRLF; inspecting its bytes resolved that
checker error. Scikit-learn emitted its cross-version warning while loading the
baseline model. This representative PLS check does not establish compatibility
for every previously serialized estimator.

The test installations were removed. The source-launcher shortcut and the
retained `.venv312` were not changed. Only this project's editable metadata was
refreshed in `.venv314` to reflect the Optuna floor; dependencies were not changed
during these final checks, and `pip check` passes.

## CI comparison

The completed run for `89f9479`
([34729233237](https://github.com/makapania/dasp/actions/runs/34729233237))
has the same five Windows failure node IDs as base `main`
([34644787150](https://github.com/makapania/dasp/actions/runs/34644787150)).
Its Linux and optional-dependency jobs have the same three non-GUI failures.
The Windows job passed 3,002 tests, versus 2,977 on base. Package build passed.

Known failing node IDs:

- `tests/gui/test_comprehensive.py::TestAllModelsViaGUI::test_catboost_via_gui`
- `tests/gui/test_multiclass_gui.py::test_tab9_rejects_multiclass_primary`
- `tests/test_cv_strategy.py::TestPostMergeReviewFixes::test_classification_metrics_template_has_no_nameerror`
- `tests/test_t19_class_weight_per_library.py::test_xgboost_threads_sample_weight_via_fit_kwargs`
- `tests/test_t19_class_weight_per_library.py::test_non_xgboost_classification_does_not_emit_fit_kwargs_plumbing`

The latest application commit's run
([34735172751](https://github.com/makapania/dasp/actions/runs/34735172751))
completed on 2026-09-13 at approximately 05:14 UTC. Its build job passed. The
informational Linux GUI job timed out at the same `test_xgboost_via_gui` case as
base and the earlier PR run.

| Latest full-suite job | Passed | Failed | Skipped | Comparison with base |
|---|---:|---:|---:|---|
| Windows | 3,008 | 5 | 29 | Identical five failing node IDs; zero new failures |
| Linux | 2,880 | 3 | 33 | Identical three failing node IDs; zero new failures |
| Optional dependencies | 2,880 | 3 | 33 | Identical three failing node IDs; zero new failures |

No collection errors were reported. The Windows and Linux test steps took
1:53:28 and 1:54:23 respectively; the optional-dependency step took 1:52:52.
The merge gate uses this completed run for application commit `db975e2`.
Subsequent validation-document commits do not change that application code.

## Artifact and limits

`dist/installer/SpectralPredict_Setup_py312_0.5.0b2.exe`

- Size: 230,295,124 bytes (219.6 MiB).
- SHA256: `3ef6ed77e422a8ee0792c21b3a88c24ba3365469fbd10703969707921f321913`.
- The `py312` filename tokens and AppId retain installer identity; the visible
  application label no longer incorrectly claims Python 3.12.

These checks cover current-user installation on the existing development
machine. A fresh Windows VM, all-users installation, manual visual interaction,
and every historic saved-model type were not tested. The separate GUI
`EstimatedEPO(random_state=None)` issue and existing CI failures remain outside
this PR. The [earlier performance and SQLite-preservation evidence](2026-09-12-pr65-performance-safety.md)
still applies: the name-only Optuna lookup preserves trial data, while the overall
stack upgrade has model-dependent timing changes rather than a uniform speedup.
