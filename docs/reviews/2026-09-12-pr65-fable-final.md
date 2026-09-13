# Fable's final implementation review of PR #65

Requested by the user on 2026-09-12. One read-mode Claude Code invocation of
commit `89f9479804d1809596a5b6ea9b5a50c626ade49f`. The returned `modelUsage`
identifies `claude-fable-5-1` as the substantive reviewer. It also lists a small
internal `claude-haiku-4-5-20251001` call (20 output tokens); that was not a
separately requested independent review. No permission denials were reported.
Fable did not run the tests or edit the repository. Its returned opinion follows
verbatim.

---

The plan file cannot be written in this session, so the review report below is the deliverable.

**Verdict: no material issue blocks merging commit 89f9479.** One low-severity finding and two informational notes.

**Low: the Optuna floor in pyproject no longer matches the API used.** `src/spectral_predict/unified_bayesian.py:2706` now calls `optuna.study.get_all_study_names`, but `pyproject.toml:60` still declares `optuna>=3.0.0`. I fetched the tagged Optuna sources and confirmed the function is absent through v3.3.0 and first appears in v3.4.0. On 3.0 through 3.3 the call raises AttributeError, which the `except Exception` at line 2735 swallows at debug level, so the environment-changed notice is silently lost. The old summaries call worked on every 3.x. Practical exposure is small because the lockfile pins optuna 5.0.0 and an unpinned editable install resolves to a current release. Suggested follow-up: bump the bound to `optuna>=3.4.0`, or fall back to the summaries call inside the try block.

**Informational: the launcher test does not exercise the checked-in byte layout.** `git ls-files --eol` reports `RUN_SPECTRAL_PREDICT.bat` as index LF, working tree mixed. The old lines are CRLF and the new goto/label lines are LF. The test at `tests/test_build_and_launcher_safety.py:97-99` round-trips the file through text mode, which normalizes to CRLF on Windows, so it validates a clean file rather than the mixed one. This matches repo policy (no `.gitattributes`, `core.autocrlf=true`, and `install.bat` shows the same index-LF pattern), so it is hygiene rather than a regression. Renormalizing the working copy to CRLF would be tidy.

**Informational: build test side effects.** `tests/test_build_and_launcher_safety.py:53` patches the real stdlib `subprocess.run` for the test duration, and each parametrized case re-executes the build script, which inserts `src` onto `sys.path` again at line 47. Both are harmless; monkeypatch restores the first and pytest already sets `pythonpath = ["src"]`.

What checked out, by area:

- **Build path fix.** `build_installer_py312.py:157` uses `splitlines()`, which preserves spaces in the purelib path since the child prints two lines and text mode strips CR. The repair block at lines 279-285 now fails the build when either the venv source or the bundled pandas file is missing instead of silently skipping. I confirmed the existing bundle in `dist/` has that file as a loose file and it is byte-identical to the venv copy, so the hard-fail path only fires on genuine breakage. macOS returns before this block at line 207.
- **Launcher.** Each pip step branches to a shared label outside the parenthesized block and exits 1, and success exits 0, mirroring `install.bat:73-75`. The test stub maps positional arguments correctly to the four Python invocations, and the count-equals-4 guard ensures pip is never actually run.
- **Optuna lookup.** I inspected the installed Optuna 5.0.0 source. The names call only runs `storage.get_all_studies()`, while the summaries call additionally fetches every study's full trial list. The prefix filter and the always-mode gate are unchanged, and resume still goes through `create_study(load_if_exists=True)` at line 2749.
- **Study-lookup tests.** `get_storage_url` is imported inside the function at lines 2608-2612, so the monkeypatch takes effect. The test asserts the new API is called once, the old API is not called, the SQLite dump is unchanged, trials and user attributes and leaderboard survive resume, and the trial-budget clamp at line 3054 still counts resumed trials. Legacy and different-environment prior names are both covered. `ENV_FINGERPRINT_VERSION` is `"env1"` at line 126, matching the test.
- **Docs.** The session log, project status, and review record are consistent with the diff. The ten-new-cases claim matches the parametrization (3 + 4 + 1 + 2). Pyc timestamps for both new test files predate the commit by about fourteen minutes, consistent with the claim that they were run.

Executed here: git inspection, Optuna source inspection in `.venv314`, curl of tagged Optuna sources, and a byte comparison of the bundled pandas file. Not executed: the new tests themselves, since this session forbids filesystem changes. Conclusions about test behavior are from reading the source. The pending items remain clean install, in-place upgrade, and uninstall, which the primary agent is covering.
