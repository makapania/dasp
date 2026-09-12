# Upgrade runbook — Python and dependencies

The repeatable process for keeping this project current. Derived from the
3.12 → 3.14 migration (see `PYTHON_UPGRADE_PLAN.md` for that specific record).

**Run this quarterly, or whenever a new Python minor ships.** Letting it drift is
what turns a routine afternoon into a multi-day archaeology project.

---

## TL;DR

```bash
.venv314\Scripts\activate
python scripts\upgrade_check.py          # what changed, what is risky, what is blocked
# ... upgrade in the order the report gives ...
python scripts\upgrade_check.py --gate   # baseline + test suite
python build_installer_py312.py          # THE bundle gate - see step 5
dist\SpectralPredict-py312\SpectralPredict-py312.exe --test
```

---

## The two rules

**1. One variable at a time.** The interpreter and the numerical dependencies
both move results. Move them in the same step and a divergence is
undiagnosable. Interpreter first (pins held), dependencies after.

**2. Source passing does NOT predict the bundle working.** This has bitten this
project before: the windowed bundle fork-bombed on LightGBM worker spawn while
every test passed (`SESSION_LOG_ARCHIVE.md:2159`). The frozen bundle is a
separate gate and nothing substitutes for it.

A third thing that is *not* a rule: **bit-parity with previous results is not
required.** The software is unreleased. The baseline harness is a change
*detector* — an unexplained diff means investigate, not revert. Last-bit
floating-point noise is not a defect.

---

## Step 1 — See what changed

```bash
python scripts\upgrade_check.py
```

The report splits packages into:

- **TIER 1 — infrastructure.** Cannot change numerical results (HTTP, packaging,
  linting, docs, plot rendering). Upgrade as one batch.
- **TIER 2 — numerical.** Can change results. Upgrade in small groups, each with
  its own before/after comparison.
- **BUILD TOOLCHAIN.** PyInstaller and friends. Needs the full bundle gate,
  because it changes what ships rather than what computes.
- **BLOCKED.** Pinned to an exact version by something else you depend on. These
  cannot move until the pinning package does. The report names the pinner.

It also prints an **ORDER** note when an outdated package caps another one.
Respect it: during the 3.14 migration, `numba 0.66` required `numpy<2.5`, so
numpy could not advance until numba did.

## Step 2 — A new Python minor?

Check [python.org](https://www.python.org/downloads/), then:

```bash
winget install Python.Python.3.XX          # ordinary GIL build, NOT free-threaded
py -3.XX -m venv .venvXXX
.venvXXX\Scripts\python -m pip install -r requirements-lock.txt   # pins HELD
.venvXXX\Scripts\python -m pip install -e . --no-deps
.venvXXX\Scripts\python -m pip check
```

Pins held is the point: this isolates the interpreter. Then run step 4 and
compare against the previous interpreter before touching any dependency.

Update in this order once it passes: `pyproject.toml` (`requires-python` +
classifiers), `.github/workflows/ci.yml`, `README.md`, the `requirements-lock.txt`
header, and `BUILD_PYTHON_VERSION` in `build_installer_py312.py`.

Rolling back is one variable: `DASP_BUILD_PYTHON=312 python build_installer_py312.py`.

## Step 3 — Upgrade, in the reported order

TIER 1 as one batch. TIER 2 in groups, comparing after each:

```bash
pip install --upgrade <group>
pip check                                  # catches exact-pin conflicts
python docs\upgrade\baseline_harness.py <label> --outdir <dir>
```

If `pip check` complains that something you just upgraded conflicts with an
existing pin, revert that package — the environment is already wrong, and pip
will not undo it for you.

## Step 4 — The source gate

```bash
python scripts\upgrade_check.py --gate
```

Runs the baseline harness and the test suite.

**The bar is ZERO NEW test failures, not a green run.** `main` has carried
pre-existing failures since ~June 2026 (T-CI-1). Diff your failure set against
the known-red list in `docs/PROJECT_STATUS.md`; a red run that adds nothing new
is a pass.

`--gate` writes its baseline to `docs/upgrade/.baselines/`. **That directory is
committed on purpose — do not add it to `.gitignore`.** Committing it is what
makes `git diff docs/upgrade/.baselines` a meaningful before/after across
upgrades rather than a comparison against whatever happens to be on this
machine. The files are small (a few hundred rows).

For the baseline, compare against the previous run. Per-sample predictions are
the sharper signal — aggregate CV metrics can round two different prediction
vectors to the same RMSE. Expect metric deltas around 1e-13 after any numpy or
scipy change, and expect near-tied rows to occasionally swap rank; neither is a
defect.

## Step 5 — The bundle gate (do not skip)

```bash
python build_installer_py312.py
dist\SpectralPredict-py312\SpectralPredict-py312.exe --test
```

`--test` checks 42 imports (including SHAP, tksheet and all six spectroscopy
readers), fits all three boosters to exercise their native DLLs, and runs a real
cross-validated `run_search` with LightGBM — the historical fork-bomb scenario.
It fails if a frozen build has the threading fallback inactive.

Then **launch the GUI** and confirm it renders. Import tests do not catch a
windowed-startup failure.

Finally, install from the generated installer and confirm an in-place upgrade
over an existing installation still works.

## Step 6 — Record it

```bash
.venv314\Scripts\python -m pip freeze --exclude-editable > requirements-lock.txt
# re-add the header comment block
```

Update `docs/PROJECT_STATUS.md` (what works, what is red) and append anything
non-obvious to `docs/SESSION_LOG.md`. Commit.

---

## Things that will bite you

| Symptom | Cause |
|---|---|
| `pip install --upgrade X` "succeeds" then `pip check` complains | Another package pins X exactly. Revert X. `upgrade_check.py` predicts this. |
| A TIER 2 package refuses to advance | Something outdated caps it. Upgrade the capper first. |
| Build says "Complete" but there is no installer | Was silently non-fatal before 2026-09-12; now fatal. If you see it again, something regressed. |
| "Inno Setup not found" but it is installed | winget installs per-user under `%LOCALAPPDATA%\Programs`. That path is checked now. |
| Tests pass from a git worktree but a script disagrees | The editable install pins the package to the main checkout. `pytest` uses worktree src; ad-hoc scripts do not. |
| Bundle launches, then spawns endless windows | The frozen threading fallback stopped engaging. See `_frozen_needs_threading_fallback()`. |
| `pandas.util` ImportError at bundle launch | The TOC collision. The builder's post-build repair handles it — it still fires on roughly every build, so do not remove it. |
