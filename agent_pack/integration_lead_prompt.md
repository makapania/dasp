# Integration Lead Playbook & Prompt (Senior Developer)

This document gives a **single prompt** you can hand to a senior developer agent who will take everything that exists, integrate it, and ship a production‑ready v0.1. It also includes a **recommended execution order** so you can choose between sequential and parallel work safely.

---

## Should we run agents sequentially or in parallel? (Short answer)

**Hybrid is best.** Freeze the **interfaces first**, then run most work **in parallel**, and finish with a tight **integration pass**.

- **Freeze contracts (very short, first):** define function signatures and CLI flags that all agents must respect (e.g., `read_asd_binary(path) -> (wl, vals, meta)`, `--asd-reader {auto,python,rs-prospectr,rs-asdreader}`, metadata sidecars path).
- **Parallel phase:** native reader, adapter chain, R bridge, CSV‑dir ingest, interactive CLI, docs, CI can proceed concurrently because they target those frozen contracts.
- **Integration phase:** one senior integrator validates cross‑module behavior, resolves edge cases, and prepares release packaging.

If you prefer minimal coordination overhead, you **can** do it sequentially (A→B→C→…), but it will be slower without materially reducing risk if the contracts are already fixed.

---

## Recommended execution order (hybrid)

1) **A0. Contract Freeze (30–60 min)**
   - Confirm/lock:
     - `read_asd_binary(path) -> (np.ndarray wavelengths_nm, np.ndarray values, dict metadata)`
     - Adapter order: **SpecDAL → Python asdreader → Native → (optional) R bridge on explicit flag**
     - CLI: `--asd-reader {auto,python,rs-prospectr,rs-asdreader}`, `--csv-dir`, `--interactive`
     - Metadata sidecars: `outputs/metadata/<stem>.json`
     - Reporting unchanged (CSV+Markdown)

2) **Parallel:** A (Native), B (Adapter+CLI flag+metadata), C (R bridge), D (CSV‑dir), E (Interactive), F (Docs), G (CI)

3) **H. Integration & Release:** Senior developer performs end‑to‑end tests, security/packaging review, and version cut.

---

## PROMPT — Senior Developer (Integration Lead)

> **You are the Integration Lead engineer.** Your mandate is to take the existing Spectral Predict scaffold and agent outputs, integrate them, eliminate cross‑module issues, and deliver a production‑ready **v0.1.0**. You must enforce contracts, tests, and packaging quality. Follow the Definition of Done and PR checklist below.
>
> ### Context
> - Python package with CLI: preprocessing (SNV, Savitzky‑Golay derivs w=7/19; poly=2 for 1st, **3 for 2nd**), models (PLS/PLS‑DA, RF, MLP), CV, composite score w/ simplicity penalty, CSV+Markdown reporting.
> - ASD ingestion strategy:
>   1) **ASCII `.sig`/ASCII `.asd`** works out‑of‑the‑box.
>   2) **SpecDAL** if installed (preferred for binary `.asd`).
>   3) **Python native** (`read_asd_binary`) fallback.
>   4) **R bridge** optional on explicit `--asd-reader` mode.
> - Repo includes: `agent_pack/` (surveys/specs/prompts), reader stubs, and working CSV paths.
>
> ### Objectives
> 1. **Lock and implement contracts**:
>    - Confirm `read_asd_binary(path) -> (wl_nm, vals, meta)` and wire into `read_asd_dir()` via adapter chain.
>    - Add CLI flag `--asd-reader {auto,python,rs-prospectr,rs-asdreader}` with help text.
>    - Ensure metadata JSON sidecars emitted to `outputs/metadata/<stem>.json` when using ASD inputs.
> 2. **Stabilize inputs**:
>    - `--spectra` (CSV wide/long), `--asd-dir` (ASCII/binary per adapter chain), **`--csv-dir`** (2‑col CSVs → wide), **`--interactive`** (paste mapping as fallback).
>    - Disallow mixing different wavelength grids in one run; raise clear error.
> 3. **Quality gates**:
>    - Determinism: fixed seeds, stable CV splits.
>    - Unit tests for readers, adapters, CSV‑dir, interactive mode. Skip gracefully when optional tools are missing (SpecDAL/Rscript).
>    - CI on Linux/Windows (Python 3.10–3.12).
> 4. **Packaging & release**:
>    - Convert to **pyproject.toml**; build wheels; pin minimal versions.
>    - Entry point `spectral-predict` script for the CLI.
>    - Bump version to **0.1.0**; add CHANGELOG; tag the release.
> 5. **Docs & UX**:
>    - README “ASD Options” and troubleshooting; examples for all modes; crisp error messages and hints.
>
> ### Files to review/change (not exhaustive)
> - `src/spectral_predict/cli.py` — new flags; improved help; interactive mode.
> - `src/spectral_predict/io.py` — adapter chain & metadata sidecars; `read_csv_dir()` helper.
> - `src/spectral_predict/readers/asd_native.py` — ensure final API and tests (if implemented by Agent A).
> - `src/spectral_predict/readers/asd_r_bridge.py` — integrate under explicit modes.
> - `src/spectral_predict/preprocess.py`, `models.py`, `search.py`, `scoring.py`, `report.py` — no API breaks; ensure imports/types OK.
> - `tests/` — add/adjust tests: `test_native.py`, `test_adapters.py`, `test_csv_dir.py`, `test_interactive.py`.
> - `pyproject.toml`, `README.md`, `CHANGELOG.md`, `.github/workflows/ci.yml`.
>
> ### Non‑functional requirements
> - **Performance:** A 5‑sample demo should complete under 90s on a laptop; default grids modest.
> - **Memory:** Fit in < 2 GB RAM for typical 2151‑band spectra.
> - **Resilience:** Clear, actionable errors for: binary ASD without adapters; Rscript missing; mixed wavelength grids; empty targets; polyorder/window mismatches.
> - **Security & Licensing:** Avoid bundling proprietary code; confirm licenses (SpecDAL MIT/BSD‑like; verify). Respect user privacy (no data exfil). 
>
> ### Definition of Done (DoD)
> - `pip install .` (editable and wheel) works on Linux/Windows; `spectral-predict -h` runs.
> - All tests passing locally and in CI; optional paths skipped cleanly.
> - End‑to‑end run succeeds via at least **two** ingestion modes (e.g., ASCII SIG and CSV‑dir).
> - Outputs: `outputs/results.csv` and `reports/<target>.md` created; metadata JSON files written for ASD inputs.
> - Version set to **0.1.0**, CHANGELOG updated, tag created (if in a git repo).
>
> ### PR Checklist (use for each merge)
> - [ ] Public API unchanged or documented (CLI flags, function signatures).
> - [ ] New flags documented in README with examples.
> - [ ] Tests added/updated; coverage not reduced.
> - [ ] Errors are actionable (contain cause + remedy).
> - [ ] No hard dependency on SpecDAL/R (optional only); auto‑detects when present.
> - [ ] No large binaries in repo; fixtures < 200 KB.
> - [ ] Lint/format pass (black/isort/flake8).
>
> ### Final validation commands
> ```bash
> # Build & install
> pipx run build && pipx run twine check dist/*
> pip install .
> spectral-predict -h
>
> # CSV wide demo
> spectral-predict --spectra data/spectra.csv --reference data/ref.csv --id-column sample_id --target "%N"
>
> # ASCII SIG demo
> spectral-predict --asd-dir data/asd_sig --reference data/ref.csv --id-column filename --target "ADF"
>
> # CSV-dir demo (2-col files)
> spectral-predict --csv-dir data/csv_long_one_per_file --reference data/ref.csv --id-column filename --target "%NDF_fiber"
>
> # Interactive demo (no ref.csv)
> spectral-predict --asd-dir data/asd_sig --interactive --target "%collagen"
> # (paste mapping, e.g., "filename,%collagen\nsample1,12.3\nsample2,7.9")
> ```
>
> ### Risk Register (mitigations)
> - **Binary ASD variability:** Detect version/endianness; fallback paths; guided errors; accept ASCII exports.
> - **R dependency:** Only used when explicitly requested; skip tests if Rscript missing.
> - **Windows paths:** Normalize slashes in R bridge; test in CI on Windows.
> - **Splice/jump:** Not in v0.1; document; add TODO to preprocessing roadmap.
> - **Wavelength precision:** Round to 0.01 nm for stable column names; prevent float drift across merges.
>
> ### Deliverables
> - Passing CI (Linux+Windows), wheel artifacts, updated README/CHANGELOG, tag `v0.1.0`.
> - A short RELEASE_NOTES.md summarizing features and known limitations.
>
> Please proceed with cautious refactors, small PRs, and green builds at each step.
