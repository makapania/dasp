<!-- MIRROR COPY.
     Canonical source: C:\Users\mspon\git\forage\docs\PYTHON_UPGRADE_DECISION.md
     Copied 2026-09-12. This file covers BOTH repos; agents working in dasp
     should read Part 0 (shared context) and Part B (dasp), plus the appendices.
     If you change this file, change the canonical copy too, or they will drift.
-->

# Python 3.14 Upgrade — Decision Record and Work Plan

**Date:** 2026-09-12
**Scope:** Two repositories — `forage` (Forage ABM) and `dasp` (spectral-predict)
**Decision:** Standardize both projects on ordinary GIL-enabled **CPython 3.14**
**Status:** Analysis complete. No code has been changed. Nothing has been built or run under 3.14.

---

## How to use this document

This document is written to be handed to an implementing agent. It is split into two
independent parts:

- **Part A — forage.** Give to an agent working in `C:\Users\mspon\git\forage`.
- **Part B — dasp.** Give to an agent working in `C:\Users\mspon\git\dasp`.
  Copy this file into that repo, or point the agent at this path.

Part 0 is shared context both agents should read. The appendices record what was
verified and — importantly — what was not.

**Every file:line reference in this document was confirmed by direct inspection on
2026-09-12.** Where a claim is inferred rather than verified, it is marked
**[UNVERIFIED]**. Do not treat unverified claims as established; check them.

---

## Part 0 — Shared context

### 0.1 The decision and why it reversed

The initial recommendation was to split the projects: dasp to 3.14, forage held at 3.13.
That rested on treating forage's PyO3 version as fixed. It is not — PyO3 *is* the thing
gating forage's Python version, so it belongs inside the decision, not outside it.

Once the PyO3 migration surface was actually measured (see A.3), it turned out to be
small and confined to one file. Holding forage at 3.13 to avoid an afternoon of binding
work would mean maintaining two Python versions across two machines indefinitely. That
trade is not worth it.

**Both projects go to 3.14.**

### 0.2 What is no longer a constraint

**PyInstaller** — the historical gate on dasp — has supported Python 3.14 since
**6.15.0, released 2025-08-03**. Current stable is **6.22.2** (2026-08-17), which supports
3.13, 3.14, and in-development 3.15. dasp's documented build tool is already in the 6.18.0
series, i.e. already in a range that supports 3.14.

Notably, **nothing in the dasp repository establishes that a dependency ever forced the
original 3.11 → 3.12 move.** It was a general build-environment improvement (newer wheels,
cleaner dependency collection, resolution of booster DLL failures caused by mismatched
Python/compiled-extension versions). Its stated goal of recovering real multiprocessing
*failed* — the same spawn failure reproduced on 3.12 and the version condition was removed
(`docs/SESSION_LOG_ARCHIVE.md:2157`).

### 0.3 The one behavioral change that can move numerical results

**Python 3.12 changed built-in `sum()` to a more accurate float algorithm.** Verified on
the actual interpreters on this machine:

```
3.11.9  -> sum([1e16, 1.0, -1e16]) = 0.0
3.12.10 -> sum([1e16, 1.0, -1e16]) = 1.0
3.13.9  -> sum([1e16, 1.0, -1e16]) = 1.0
```

This is a real mechanism by which simulation trajectories can diverge across the upgrade.
It is **not** proof that they will. It is the reason the before/after seeded comparison in
A.7 is mandatory rather than optional.

What does *not* change: float representation generally, dict insertion ordering, set
unorderedness. forage's main RNG is NumPy `RandomState` seeded from the scenario
(`forage/core/simulation.py:440`, `:811`), not stdlib `random`, so the interpreter bump
does not perturb the primary RNG stream.

### 0.4 The thing that matters more than the Python version

Three findings surfaced during this analysis that are **more consequential than which
Python you run**, and none of them is fixed by choosing a version:

1. A seeded-reproducibility bug in forage's social pathway (A.1).
2. Packaging gaps that break fresh installs on a second machine (A.2).
3. A resume test that tolerates ~1 kg of body-mass drift while being named
   "bit-identical" (A.6) — i.e. the existing validation harness cannot detect the
   class of drift this upgrade risks introducing.

Fix 1 and 2 before the upgrade. Fix 3 before trusting the upgrade's validation.

### 0.5 Global ordering across both repos

The two repos are independent and can proceed in parallel, with one exception: do not
adopt 3.14 as the production interpreter on either until its own validation gate passes.

```
forage:  A.1 → A.2 → A.6 → A.4 (baseline) → A.3 (PyO3, on 3.11) → A.5 (3.14) → A.7 (validate)
dasp:    B.1 → B.2 → B.3 → B.4 (3.14 env) → B.5 (build) → B.6 (validate)
```

Note that in forage, the PyO3 upgrade (A.3) happens **on Python 3.11**, before the
interpreter changes. One variable at a time.

---

# Part A — forage

Repository: `C:\Users\mspon\git\forage`

## A.1 — Fix the social RNG reproducibility bug

**Priority: highest. Do this first. It is independent of the upgrade.**

### Problem

`forage/core/rust_bridge.py:1098` draws the social RNG buffer from the **global** NumPy
random state rather than the simulation's seeded RNG:

```python
n_rng_per_agent = 50  # enough for spatial bucketing with many neighbors
social_rng = np.random.random(n_total * n_rng_per_agent)
```

Consequences:

- Seeded runs are **not reproducible** through the social pathway. Two runs with the same
  scenario seed diverge in social bond formation.
- These draws control social bond formation in `rust/src/social.rs:632`.
- Checkpoints save only the simulation RNG (`forage/core/checkpoint.py:1073`), so
  checkpoint/resume does not restore this stream either.
- No global seeding was found anywhere in the codebase, so the stream is not even
  incidentally deterministic.

This is a correctness bug in a model heading toward publication. It is unrelated to the
Python version — staying on 3.11 does not solve it.

### Fix

Route the draw through the simulation's RNG, the same way the other buffers are generated.
Confirm how `simulation.py:440` and `:811` construct and pass seeded buffers and match
that pattern.

### Verification

- A fixed-seed scenario run twice must produce identical social bond structure.
- A checkpoint/resume run must match a straight-through run on the social pathway.
- Add a regression test asserting seeded reproducibility of social bonds specifically.

### Note

Because this changes the RNG stream, **it will change simulation outputs**. Do it *before*
capturing the pre-upgrade baseline (A.4), so the baseline is taken against corrected code.
Otherwise you will be comparing two changes at once.

---

## A.2 — Close the packaging gaps

These are why a second machine is painful. All three verified.

### A.2.1 — `forage_engine` is imported unconditionally but never declared

`forage/__init__.py:7`:

```python
# Rust extension is required
import forage_engine  # noqa: F401
```

`pyproject.toml` does **not** list `forage_engine` in `dependencies` (lines 12–21).
A fresh `pip install -e .` therefore under-installs, and `import forage` fails.

**Fix:** represent the local Rust package in the dependency graph. If moving to `uv`, note
that default exact syncing can *remove* undeclared manually-installed packages — the Rust
extension must be declared, not merely present.

### A.2.2 — `requirements-lock.txt` is untracked in git

Confirmed: `git ls-files --error-unmatch requirements-lock.txt` → `did not match any file(s)
known to git`.

The entire cross-machine reproducibility story depends on a file that is not in the
repository. It is also incomplete: the export/dev section omits `et-xmlfile`, `coverage`,
`pluggy`, and Black's dependencies, and **`maturin` is unpinned**
(`requirements-lock.txt:51`). These can resolve differently on machine two.

**Fix:** track the file. Complete the closure. Pin `maturin`.

### A.2.3 — Launchers depend on ambient PATH

- `launch_gui.bat:3` → `python gui_launcher.py` (bare `python`)
- `forage/batch/slurm_executor.py:28` → `python_executable: str = "python"`

On this machine `python` resolves to 3.11 while `py` defaults to 3.13, and **neither 3.12
nor 3.13 has `forage_engine` built**. After the upgrade this becomes an active foot-gun.

**Fix:** bind launchers to the project environment explicitly rather than to PATH.

---

## A.3 — Upgrade PyO3 0.23.5 → 0.29.2 (perform this on Python 3.11)

### Current state (verified)

`rust/Cargo.toml:10-14`:

```toml
[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
rand = "0.8"
rand_chacha = "0.3"
```

`rust/Cargo.lock:175` locks PyO3 to **0.23.5**; rust-numpy to **0.23.0**.

### Why 0.23.5 is the ceiling

`C:\Users\mspon\.cargo\registry\src\index.crates.io-.../pyo3-ffi-0.23.5/build.rs:16`:

```rust
max: PythonVersion {
    major: 3,
    minor: 13,
},
```

PyO3 0.23.5 hard-rejects CPython newer than 3.13. This is the gate. Confirmed against the
upstream v0.23.5 tag.

### Target

PyO3 **0.29.2** with matching rust-numpy **0.29.0** (the crates version-pair).

### Migration surface (measured)

| Metric | Value |
|---|---|
| Files in `rust/src/` touching PyO3/numpy | **1** — `lib.rs` only (81 refs; other 10 files: 0) |
| `#[pyfunction]` | 9 |
| `#[pymodule]` | 1 |
| `#[pyclass]` | **2** |
| `FromPyObject` impls/derives | **0** |
| `.extract()` calls | 95 — all callers |
| `Python::with_gil` / `GILGuard` / `PyCell` / `IntoPy` / `ToPyObject` | **0 each** |

The two pyclasses are `#[pyclass(name = "AgentSoA")]` at `lib.rs:157` and
`#[pyclass(name = "SocialState")]` at `lib.rs:607`. Neither derives `Clone`; their `Sync`
requirement already applies under 0.23, so this is not new work.

Because there are **zero `FromPyObject` implementations**, PyO3 0.27's `FromPyObject`
rework — the largest breaking change in the 0.23→0.29 range — does not apply. The 95
`.extract()` calls are all *callers*, and caller-side `.extract()` is unchanged across the
range.

### Required edits (all in `rust/src/lib.rs`)

1. **`PyObject` → `Py<PyAny>`** at `lib.rs:161` and `lib.rs:386`.
   The `PyObject` alias is absent in 0.29.2.
   - `:161` — `food_id_order: Vec<PyObject>` (field of `AgentSoaPy`)
   - `:386` — `let food_id_pyobjects: Vec<PyObject> = food_id_order`

2. **`.as_ref()` → `.as_any()`** at `lib.rs:190` and `lib.rs:412`.
   Both sites are `let agent = agent_obj.as_ref();`. PyO3 added another `AsRef` impl,
   making the bare call ambiguous.

3. **`#[pymodule(gil_used = true)]`** at `lib.rs:1349`.
   PyO3 0.28 changed the default. The code assumes it holds the GIL
   (`lib.rs:899` calls the Rust loop while holding it), so this must be made explicit.

4. **Add the `abi3-py311` feature** — see A.3.1.

### Corrections to earlier framing

- The `pyo3_build_config` change requires a **direct PyO3 dependency**, which forage
  already has. It does **not** mean adding `pyo3-build-config`.
- That change and `raw-dylib` linking on Windows are **0.29** changes, not 0.28.

### rust-numpy 0.23 → 0.29

Little churn. `PyReadonlyArray`, `Bound<PyArray>`, `from_vec`, `from_slice`, and the slice
methods all retain the interfaces forage uses. Slice errors still convert through `?`.
Real changes are alignment checking (0.28), expanded `ndarray` compatibility, and internal
NumPy ABI-v2 handling (0.29). **None of the nine pyfunction signatures needs redesign**,
including the mutable pyclass arguments.

### A.3.1 — Enable `abi3` (strongly recommended)

rust-numpy supports `Py_LIMITED_API` and obtains NumPy's C API through **runtime capsules**,
so it needs no Python-minor-specific linking. Add PyO3's **`abi3-py311`** feature to match
forage's existing declared minimum (`rust/pyproject.toml:8` → `requires-python = ">=3.11"`).

Payoff: **one Windows x64 wheel serves both machines across all supported Python minors.**
This permanently removes the rebuild-per-machine-per-upgrade problem that motivated much
of this exercise.

Caveats:
- OS/architecture compatibility still applies (this is a Windows x64 wheel).
- NumPy compatibility still applies.
- Ordinary `abi3` **excludes free-threaded Python** — which is fine, see A.8.

`rust/pyproject.toml` currently reads:

```toml
[tool.maturin]
features = ["pyo3/extension-module"]
```

This will need the abi3 feature added alongside.

### A.3.2 — Address the two `unsafe` blocks (recommended, not required)

`lib.rs:918` (world-grid writeback) and `lib.rs:1229` (food regrowth) will still compile
after the upgrade. However, **holding the GIL establishes neither writability nor
non-aliasing** — this is a pre-existing soundness gap, not migration churn.

Recommended: replace with `try_readwrite()` guards and safe `as_slice_mut()`. Since the
PyO3 migration already touches only this file, this is the cheapest time to do it. Treat
as a separate commit so it can be reverted independently.

### A.3.3 — RNG crates: leave alone

`rand 0.8` and `rand_chacha 0.3` (`rust/Cargo.toml:13-14`) are **independent of the PyO3
upgrade and currently unused**. Actual draws come from Python's seeded `RandomState`
(`simulation.py:440`) into the Rust buffer reader (`rust/src/helpers.rs:320`).

Declared version ranges cannot silently jump to a newer incompatible minor line. Keep
`Cargo.lock` and build with `--locked`. **Extra exact pins are unnecessary.** Cargo pins
cannot fix the real reproducibility gap, which is A.1.

### Verification gate for A.3

Run **on Python 3.11**, before any interpreter change:

```bash
cd rust && cargo build --release --locked
cargo test
```

Then rebuild the extension and run the full Python suite:

```bash
python -m maturin develop --release --locked --manifest-path rust/Cargo.toml
pytest tests/
```

Seeded simulation outputs must be **unchanged** from the A.4 baseline. The PyO3 upgrade
alone must not move any number. If it does, stop and investigate — do not proceed to 3.14.

---

## A.4 — Capture and isolate the pre-upgrade baseline

Do this after A.1 (so the baseline reflects corrected RNG) and before A.3.

1. **Preserve the current baseline completely:** revision, working changes, configs, seeds,
   exact interpreter (3.11.9), dependency versions, Rust build settings, and the built
   extension wheel. Keep paper results and checkpoints tied to it.
2. **Reproduce it in an isolated environment on the same 3.11.9.** The project is currently
   installed into **global** Python 3.11 site-packages shared with jupyter, supabase,
   openai, and an editable `spectral-predict` — so a plain `pip freeze` there is unusable.
3. **Compare seeded runs between the isolated env and the global install.** They must match.
   If they do not, the global environment has drifted in a way that matters, and that must
   be understood before anything else changes.
4. Preserve this environment permanently for reproducing historical results.

The current extension is a `cp311-cp311-win_amd64` wheel with `abi3=false`. It must be
rebuilt for any new interpreter — which A.3.1 fixes going forward.

---

## A.5 — Move to Python 3.14

Only after A.3's gate passes on 3.11.

1. Create a **separate** environment on 3.14, holding Python package versions, Cargo
   dependencies, Rust compiler version, and release settings fixed.
2. Rebuild explicitly:
   ```bash
   python -m maturin develop --release --locked --manifest-path rust/Cargo.toml
   ```
3. Do **not** let a resolver silently update the numerical stack (NumPy, SciPy, pandas)
   during this step. `uv` is a sensible destination, but a resolver must not move the
   numerical stack during either comparison.

### Tooling to update after validation passes

- `pyproject.toml:12` — `requires-python = ">=3.11"` (a *minimum*, not a pin)
- `pyproject.toml:54` — `[tool.black] target-version = ['py311']`
- `pyproject.toml:58` — `[tool.ruff] target-version = "py311"`
- `pyproject.toml:61` — `[tool.mypy] python_version = "3.11"`
- `rust/pyproject.toml:8` — `requires-python = ">=3.11"` (keep at 3.11 if using `abi3-py311`)

Note these tooling settings do **not** prevent running under a newer interpreter; they
control lint/type targets only. Align both pyproject files, and **separately pin the exact
runtime patch version**.

forage has **no CI configuration and no interpreter matrix.** Consider adding one as part
of this work — it is the only durable defense against the drift this document describes.

---

## A.6 — Tighten the validation harness before trusting it

`tests/test_checkpoint.py`, class `TestBitIdenticalResume`:

```python
assert abs(ma - mc) < 1.0, (
```

Despite the class name, the resume test permits body-mass differences approaching **1 kg**.
The docstring is candid that strict bit-identity is hard because the `Simulation`
constructor re-runs some initialisation — but as written, this test **cannot detect the
class of drift a `sum()` change would introduce.**

Tighten this tolerance before using the test suite as an upgrade gate, or the gate is
decorative.

---

## A.7 — Validation gate before adopting 3.14 in production

Run on **both machines**:

- Full test suite.
- Headless scenario runs.
- A small process-parallel sweep.
- Checkpoint/resume comparisons.
- **Seeded before/after comparison:** compare initial states and early trajectories first,
  then biological summaries across seeds.

**Investigate any differences rather than loosening tolerances.** Given §0.3, a difference
is plausible and must be explained, not accommodated.

Then: adopt one production environment, keep a complete sweep under it, and preserve the
old environment for historical results.

---

## A.8 — Free-threading: not applicable, do not pursue

Recorded so it is not revisited:

- Sweeps already use stdlib `multiprocessing` / `ProcessPoolExecutor` with explicit
  `spawn` (`forage/batch/runner.py:780`). Processes already run concurrently.
  (`multiprocess` appears in the lockfile only as a SALib transitive dependency.)
- The Rust agent loop is **deliberately sequential** — each agent observes earlier agents'
  modifications (`rust/src/step.rs`). Parallelizing it would require preserving those model
  semantics.
- `lib.rs:899` calls that loop **while holding the GIL** and relies on holding it for array
  access.
- Ordinary `abi3` (A.3.1) excludes free-threaded builds anyway.

**Do not expect a speedup from the interpreter upgrade.** 3.12/3.13 optimizations help
Python orchestration but do not make an already-compiled Rust loop faster. Illustrative
arithmetic: making a Python portion that is 10% of runtime 20% faster improves total
throughput by ~1.7%.

If you want real performance, profile this instead: **every Rust step copies food grids
into Rust vectors and back, plus habitat multiplier grids** (`rust/src/lib.rs:851`, `:911`).
That is a far more promising target than the interpreter version.

---

## A.9 — Stale references to clean up

- `rust/README.md:29` still describes an optional Python fallback. The fallback was removed;
  Rust is a hard dependency.
- `scripts/benchmark_rust.py:48` and `scripts/profile_rust_path.py:21` import the removed
  `HAS_RUST`.
- `CLAUDE.md` describes a **PyQt6** dashboard; the GUI is actually **tkinter**
  (`forage/viz/dashboard_window.py:7`). The `viz` extra (`pyqt6`, `pyqtgraph`) is declared
  in `pyproject.toml` but not installed and appears dead. Consider removing it.

forage has no Numba and no PyInstaller constraint.

---

## A.10 — `sum()` exposure sites (from §0.3)

Built-in `sum()` on float sequences, which changed behavior at 3.12. Audit these during
A.7 if seeded outputs diverge:

- `forage/ecology/foraging.py:127` — `pop_digestibility[food_id] = sum(dig_values) / len(dig_values)`
- `forage/ecology/foraging.py:129` — `pop_compatibility[food_id] = sum(compat_values) / len(compat_values)`
- `forage/ecology/foraging.py:134` — `pop_digestibility[food_id] = sum(dig_values) / len(dig_values)`
- `forage/core/world.py:992` — `total = sum(active.values())` (habitat-proportion normalization)
- `forage/ecology/foraging.py:1036` — `sum(agent.recent_feeding_gains) / len(...)`
- `forage/ecology/foraging.py:1046` — `sum(agent.recent_protein_gains) / len(...)`
- `forage/ecology/foraging.py:1385` — `giving_up_biomass = sum(...)`

The first four feed the landscape foraging threshold and habitat normalization — the ones
most likely to propagate. `np.sum` call sites are unaffected by this change.

---

# Part B — dasp (spectral-predict)

Repository: `C:\Users\mspon\git\dasp`

Current: Python **3.12** (`.venv312`), shipped to non-technical end users as a PyInstaller
frozen bundle wrapped in Inno Setup.

## B.1 — Dependency situation (verified: no blockers)

All **114 pinned distributions** in `requirements-lock.txt` were checked against Windows
x64, regular CPython 3.13 and 3.14 wheel tags and `Requires-Python`. **113 have compatible
published wheels for both targets.** Published runtime dependency metadata showed no
version conflicts within the pinned set for either target.

The single exception was source-only `jcamp==1.2.2` — resolved in B.2.

Highlights (versions from the 2026-09-10 lockfile, **not** the older floors in
`pyproject.toml`):

| Dependency (exact pin) | Status for 3.13 and 3.14 |
|---|---|
| `numba 0.66.0`, `llvmlite 0.48.0` | Native wheels for both. Independently confirmed: numba 0.66.0 ships `cp310–cp314` win_amd64. Numba's NumPy constraint admits the pinned 2.4.4. **This was the highest-risk item and it is clear.** |
| `catboost 1.2.10` | Explicit Windows `cp313` and `cp314` wheels |
| `shap 0.52.0` | `cp312-abi3` wheel — compatible with 3.12 and everything after |
| `xgboost 3.2.0`, `lightgbm 4.6.0` | `py3-none-win_amd64` wheels; no CPython-minor-specific wheel needed. Native libraries still need frozen DLL-loading tests. |
| `scikit-learn 1.8.0`, `imbalanced-learn 0.14.1` | Native sklearn wheels; portable imbalanced-learn wheel |
| `pymoo 0.6.1.6`, `moocore 0.2.0` | pymoo native wheels for both; moocore Windows stable-ABI wheel covers both |
| `optuna 4.8.0` | Portable wheel, no interpreter ceiling |
| `tksheet 7.6.0` | Pure Python ≥3.8. Remaining qualification is actual Tk/widget behavior. |
| `specdal 0.2.1` | Pure-Python wheel, Python ≥3 |
| `spc-io 0.2.1` | Pure Python. `~=3.8` means **≥3.8, <4.0** — admits both targets. |
| `specio-py310 0.1.0.post2` | Pure-Python `py3-none-any`, no upper bound. **The name does not impose a 3.10 restriction.** |
| `spectrochempy-omnic 0.2.1` | Pure Python ≥3.10; runtime dep is NumPy. Full SpectroChemPy is an extra, not mandatory. |
| `brukeropus 1.4.3` | Pure Python ≥3.6, depends on NumPy |
| `numpy 2.4.4`, `pandas 3.0.2`, `scipy 1.17.1` | Windows wheels for both |

**Mitigating factor for the niche readers:** `specdal`, `spc-io`, `specio-py310`,
`spectrochempy-omnic`, `brukeropus`, and `jcamp` are all **lazy imports inside `try/except`
blocks** in per-format reader functions (`src/spectral_predict/io.py:1004`, `:3195`,
`:3713`; `src/spectral_predict/readers/omnic_reader.py:37`;
`src/spectral_predict/readers/perkinelmer_reader.py:47`;
`src/spectral_predict/readers/opus_reader.py:49`). If one breaks on 3.14, you lose that
file format with a clean error — not the application.

## B.2 — jcamp: bump the pin AND change the call site

**These must happen together. Doing only one breaks the JCAMP reader.**

### Current state

`pyproject.toml:40`:

```
"jcamp>=1.2.1,<1.3",  # used for READ only; 1.3.0 setup.py is broken (lists stdlib modules as deps); writes go through vendored _build_jcamp_dx_string in io.py
```

Lockfile pins `jcamp==1.2.2`. **The installed `.venv312` actually has 1.2.1** — live drift
from the lockfile, independently confirmed. This is itself evidence for the lockfile
discipline in B.3.

### What changed upstream

`jcamp 1.3.2` (published 2026-07-15) is **Flit-built**, ships `jcamp-1.3.2-py3-none-any.whl`,
and declares `requires_dist: ['numpy']`, `requires_python: >=3.8`. The broken setup.py
problem no longer applies to normal installs (the legacy `setup.py` still incorrectly lists
`datetime`, but the Flit build bypasses it). **No 3.14 source build is needed.**

### The catch — verified

The 1.3.2 wheel exposes **`readfile`** with **no `jcamp_readfile` alias**. Confirmed against
the installed 1.2.1: `hasattr(jcamp, 'jcamp_readfile')` → `True`,
`hasattr(jcamp, 'readfile')` → `False`. The API was renamed.

`src/spectral_predict/io.py:3304` called (before `be7963c`):

```python
jcamp_data = jcamp.jcamp_readfile(str(path))
```

### Upgrading an existing `.venv314`

Use jcamp **1.3.2** in `.venv314`. JCAMP-DX import does not work on 1.2.2
because the current code calls `jcamp.readfile`. On any machine:

```bash
.venv314\Scripts\python -m pip install "jcamp==1.3.2"
```

Confirm with `pip show jcamp` (expect 1.3.2). Don't use `jcamp.__version__`,
which still says `1.2.2` in the 1.3.2 release.

### Required change

> **Done in `be7963c` (2026-09-12).** Kept as the record of what changed. Existing
> venvs pick up the new pin automatically: the launchers resync to
> `requirements-lock.txt` via `scripts/check_env_lock.py`.

1. `pyproject.toml:40` — change the pin to `jcamp>=1.3.2` and remove the now-obsolete
   comment about the broken setup.py.
2. `src/spectral_predict/io.py:3304` — change to `jcamp.readfile(str(path))`.
3. Update `requirements-lock.txt`.
4. **Keep the vendored `_build_jcamp_dx_string` writer** at `io.py:3750` during this
   migration. Removing it is separate work with its own validation.

With this done, dasp is **114/114** on published wheels for 3.14.

## B.3 — Pin the build toolchain

**`pyinstaller` and `pyinstaller-hooks-contrib` are absent from `requirements-lock.txt`
and from the inspected `.venv312`.**

For a project whose entire distribution path is a frozen bundle, the build tool is the
single most important thing to pin, and it is the one thing that is not pinned. Fix this
as part of the migration.

## B.4 — The build path hardcodes 3.12 in four places

**Changing a version string is not sufficient.** Running the existing builder under 3.14
would still select its configured 3.12 interpreter. All four confirmed:

| Location | What it does |
|---|---|
| `build_installer_py312.py:108` | `_find_build_python()` → `PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"` |
| `build_installer_py312.py:181` | Post-build verification checks for `_internal/python312.dll` |
| `spectral_predict_py312.spec:29` | `venv_path = project_root / '.venv312'` — selects the env independently of the builder |
| `installer/spectral_predict_py312.iss:14` | Embeds `MyAppExeName "SpectralPredict-py312.exe"` and `MyAppBundleDir "SpectralPredict-py312"` |

Decide deliberately whether to rename these to `py314` or make them version-agnostic.
Renaming affects T-14b — see B.6.

Also extend the CI matrix: `.github/workflows/ci.yml:18` is currently
`['3.10', '3.11', '3.12']`.

## B.5 — Bundle landmines: retain all of them

A Python bump **disturbs these but neither invalidates nor fixes them.** Keep every
workaround through the migration and re-qualify afterward.

- **Torch cleanup** (`build_installer_py312.py:203`) — retain the post-COLLECT
  `shutil.rmtree`. What matters is its *position in the build sequence*, not 3.12-specific
  behavior. The code documents why filtering `a.binaries`/`a.datas` inside the spec was
  abandoned (TOC corruption).
- **TOC corruption / pandas repair** (`build_installer_py312.py:226`,
  `spectral_predict_py312.spec:110`) — retain and re-check while qualifying the new
  toolchain. Note the current implementation checks **one specific file**; it is not
  general bundle-integrity verification. Because the spec collects every `.pyd` and every
  distribution's metadata, the build environment's contents are significant.
- **`_frozen_needs_threading_fallback`** (`src/spectral_predict/search.py:10`, used at
  `:1673` and `:5008`) — returns frozen status with **no Python-version condition**, so a
  3.14 bump leaves the fallback active. Keep it until a frozen-process test shows removal
  works. The repo's claim that loky is broken in *all* windowed bundles is the basis for
  the conservative workaround, not current proof about upstream.
- **Windowed startup** (`spectral_predict_gui_optimized.py:19`) — also exercise the early
  `freeze_support()`, the CPU-count env setting, and the Windows subprocess monkey-patch.
- **moocore collection workaround** — still matters; its Windows stable-ABI wheel covers
  both targets.

## B.6 — Validation, and what the existing tests do not cover

### T-14b tests version *strings*, not compatibility

`tests/test_t14b_pyinstaller_and_gui_version_drift.py` asserts **application-version
consistency only** — GUI labels, executable metadata, installer version, package version.
It does **not** test Python compatibility and does **not** build an executable. Its
hardcoded build-file paths (`:153`) would need updating if the files in B.4 are renamed.

### Coverage gaps to be aware of

- CI installs from dependency **floors** on 3.10/3.11/3.12, and its build job builds the
  **Python package**, not the frozen installer (`.github/workflows/ci.yml:18`, `:130`).
  So CI has never exercised the thing you actually ship.
- The exe's `--test` list has **no explicit checks for SHAP, tksheet, or the six
  spectroscopy readers** (`spectral_predict_gui_optimized.py:60179`). Consider adding them
  — they are exactly the components most at risk from an interpreter change.

### Validation gate

Wheel metadata being clean is **necessary, not sufficient**. For software shipped to
non-technical users through an installer, the bundle is where the risk lives. Budget a
full build-and-test cycle:

1. Build the frozen bundle on 3.14.
2. Verify DLL loading for xgboost, lightgbm, catboost native libraries.
3. Exercise every one of the six spectroscopy readers against real files.
4. Exercise the GUI: tksheet widgets, SHAP diagnostics, windowed startup.
5. Run a CV job to confirm the threading fallback still behaves.
6. Install from the Inno Setup installer on a clean machine and repeat.

## B.7 — Correct the stale spec comment

`spectral_predict_py312.spec:5` still states as a goal:

```
  - Recover real multiprocessing (loky backend instead of threading fallback)
```

**This goal failed.** The migration log records reproducing the same spawning failure on
3.12 and removing the Python-version condition (`docs/SESSION_LOG_ARCHIVE.md:2157`). The
comment is misleading to anyone reading the spec. Correct it.

---

# Appendix A — Evidence summary

| Claim | Status | Evidence |
|---|---|---|
| PyInstaller supports 3.14 since 6.15.0 (2025-08-03); current 6.22.2 | **Verified** | PyInstaller changelog |
| PyO3 0.23.5 max Python = 3.13 | **Verified** | `pyo3-ffi-0.23.5/build.rs:16` in local cargo registry; upstream v0.23.5 tag |
| PyO3 latest 0.29.2, rust-numpy latest 0.29.0 | **Verified** | crates.io API |
| Python 3.12 changed float `sum()` | **Verified** | Executed on 3.11.9 / 3.12.10 / 3.13.9 on this machine |
| forage: 2 `#[pyclass]` at `lib.rs:157`, `:607` | **Verified** | Direct grep |
| forage: 0 `FromPyObject` impls; 95 `.extract()` all callers | **Verified** | Direct grep |
| forage: bindings confined to `lib.rs` | **Verified** | Per-file grep; other 10 files return 0 |
| forage: `forage_engine` imported at `__init__.py:7`, undeclared in pyproject | **Verified** | Direct read |
| forage: `requirements-lock.txt` untracked | **Verified** | `git ls-files --error-unmatch` |
| forage: resume test tolerance `abs(ma-mc) < 1.0` | **Verified** | `tests/test_checkpoint.py` |
| forage: `rust_bridge.py:1098` uses global `np.random.random()` | **Verified** | Direct read |
| dasp: 113/114 pinned dists have wheels for 3.13 and 3.14 | **Verified** | Published PyPI metadata, all 114 checked |
| dasp: numba 0.66.0 ships cp314 win_amd64 | **Verified** | PyPI API, independently confirmed |
| dasp: jcamp 1.3.2 ships `py3-none-any.whl`, deps `['numpy']` | **Verified** | PyPI API |
| dasp: jcamp 1.3.2 exposes `readfile`, no `jcamp_readfile` alias | **Verified** | Inspection of published wheel + installed 1.2.1 |
| dasp: installed jcamp is 1.2.1, lockfile says 1.2.2 | **Verified** | Executed in `.venv312` |
| dasp: build path hardcodes 3.12 in 4 places | **Verified** | Direct read of all four |
| dasp: pyinstaller absent from lockfile and `.venv312` | **Verified** | Direct inspection |
| dasp: nothing shows a dependency forced 3.11→3.12 | **Verified** (absence of evidence) | Build guides, `SESSION_LOG_ARCHIVE.md:2157` |

# Appendix B — What was NOT done

This analysis was **read-only source and published-metadata inspection.** Specifically,
none of the following was performed:

- The upgraded PyO3 extension was **not built**.
- Neither project was **installed or run** under Python 3.13 or 3.14.
- The dasp frozen bundle was **not built** on any new interpreter.
- No seeded simulation comparison was run.
- No dependency was actually installed on 3.14.

**Wheel availability is not the same as working software.** Every recommendation here
still requires the actual build and the validation gates in A.7 and B.6.

# Appendix C — Open items deliberately deferred

- Removing dasp's vendored `_build_jcamp_dx_string` writer (B.2 item 4).
- forage's `viz` extra (PyQt6/pyqtgraph) appears dead — `CLAUDE.md` describes a PyQt6
  dashboard but the GUI is tkinter (A.9).
- Migration to `uv` — sensible, but must not let a resolver move the numerical stack
  during upgrade validation (A.5).
- Adding CI with an interpreter matrix to forage, which currently has none (A.5).
- Profiling the per-step grid copying in `rust/src/lib.rs:851`, `:911` (A.8) — a more
  promising performance target than any interpreter change.
