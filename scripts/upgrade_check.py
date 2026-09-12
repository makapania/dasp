"""One-command check for Python and dependency upgrades.

    <venv>\\Scripts\\python scripts\\upgrade_check.py             # report only
    <venv>\\Scripts\\python scripts\\upgrade_check.py --gate      # report + full gate

The report answers three questions that otherwise cost an afternoon each:

  1. What is out of date, and is it risky? Packages are split into TIER 1
     (infrastructure: cannot change numerical results) and TIER 2 (numerical:
     can). That split is the whole reason an upgrade is cheap or expensive.
  2. What CANNOT be upgraded, and why? A package pinned to an exact version by
     something else you depend on will silently refuse to move, and pip's error
     only shows up after you have already broken the environment.
  3. What ordering is forced? When an outdated package caps another one, the
     capped package must be upgraded FIRST or the resolver cannot proceed.

`--gate` then runs the acceptance checks: the numerical baseline and the test
suite. See docs/upgrade/UPGRADE_RUNBOOK.md for the full process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from importlib.metadata import distributions
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# TIER 2 = can change numerical results, so each needs its own before/after
# comparison. Everything not listed here is TIER 1 (infrastructure: build tools,
# HTTP, packaging, docs, linting, GUI chrome) and can be upgraded as one batch.
#
# Judgement calls worth stating: matplotlib/pillow/fonttools render plots but do
# not touch model fitting, so they are TIER 1. joblib IS TIER 2 despite looking
# like plumbing, because it controls parallel execution and therefore reduction
# order.
TIER2 = {
    "numpy", "scipy", "pandas", "scikit-learn", "sklearn-compat",
    "numba", "llvmlite", "joblib", "threadpoolctl",
    "xgboost", "lightgbm", "catboost", "shap", "slicer",
    "optuna", "pymoo", "moocore", "autograd", "cma",
    "imbalanced-learn", "pybaselines",
    # Readers: pure Python, but they decide what numbers enter the pipeline.
    "jcamp", "specdal", "spc-io", "specio-py310", "spectrochempy-omnic",
    "brukeropus",
}

# Upgrading these changes what ships to users, so they get the full bundle gate
# (build + --test + GUI launch) rather than just the test suite.
BUILD_TOOLCHAIN = {"pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile"}


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _outdated() -> list[dict]:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out or "[]")


def _exact_pins() -> dict[str, list[tuple[str, str]]]:
    """Map package -> [(who pins it, the requirement string)] for '==' pins.

    This is what makes an upgrade quietly impossible: pip installs the newer
    version, then reports a conflict, and the environment is already wrong.
    """
    pinned: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for dist in distributions():
        meta = dist.metadata
        if not meta or not meta["Name"]:
            continue
        for req in dist.requires or []:
            if "==" not in req or ";" in req:  # skip environment-conditional reqs
                continue
            target = _norm(req.split("==")[0].split("[")[0].strip())
            pinned[target].append((_norm(meta["Name"]), req.strip()))
    return pinned


def _caps() -> dict[str, list[tuple[str, str]]]:
    """Map package -> [(who caps it, requirement)] for '<' upper bounds.

    An outdated capper forces ordering: upgrade the capper first, or the capped
    package cannot advance. This is exactly how numba 0.66 (numpy<2.5) blocked
    numpy 2.5 during the 3.14 migration.
    """
    capped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for dist in distributions():
        meta = dist.metadata
        if not meta or not meta["Name"]:
            continue
        for req in dist.requires or []:
            if "<" not in req or ";" in req:
                continue
            target = _norm(req.split("<")[0].split(">")[0].split("[")[0].split("!")[0].strip())
            capped[target].append((_norm(meta["Name"]), req.strip()))
    return capped


def report() -> int:
    print(f"Python  {sys.version.split()[0]}   ({sys.executable})")
    print(f"Repo    {REPO}\n")

    outdated = _outdated()
    if not outdated:
        print("Everything is current.")
        return 0

    names = {_norm(p["name"]) for p in outdated}
    pins, caps = _exact_pins(), _caps()

    tier1, tier2, build, blocked = [], [], [], []
    for pkg in sorted(outdated, key=lambda p: _norm(p["name"])):
        n = _norm(pkg["name"])
        line = f"  {pkg['name']:<24} {pkg['version']:>12} -> {pkg['latest_version']}"
        if n in pins:
            who = ", ".join(f"{w} requires {r}" for w, r in pins[n])
            blocked.append(f"{line}\n      BLOCKED: {who}")
        elif n in BUILD_TOOLCHAIN:
            build.append(line)
        elif n in TIER2:
            # Flag ordering: is this capped by something ALSO out of date?
            note = ""
            stale_cappers = [(w, r) for w, r in caps.get(n, []) if w in names]
            if stale_cappers:
                who = ", ".join(f"{w} requires {r}" for w, r in stale_cappers)
                note = f"\n      ORDER: upgrade {stale_cappers[0][0]} first ({who})"
            tier2.append(line + note)
        else:
            tier1.append(line)

    if tier1:
        print(f"TIER 1 - infrastructure, upgrade as one batch ({len(tier1)}):")
        print("\n".join(tier1), "\n")
    if tier2:
        print(f"TIER 2 - numerical, upgrade in groups with a baseline each ({len(tier2)}):")
        print("\n".join(tier2), "\n")
    if build:
        print(f"BUILD TOOLCHAIN - needs the full bundle gate ({len(build)}):")
        print("\n".join(build), "\n")
    if blocked:
        print(f"BLOCKED - pinned by another package, cannot move ({len(blocked)}):")
        print("\n".join(blocked), "\n")

    print("Next: docs/upgrade/UPGRADE_RUNBOOK.md")
    return 0


def gate() -> int:
    """Run the acceptance checks. Non-zero if anything fails."""
    baseline_dir = REPO / "docs" / "upgrade" / ".baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Numerical baseline ===")
    rc = subprocess.run(
        [sys.executable, str(REPO / "docs" / "upgrade" / "baseline_harness.py"),
         "current", "--outdir", str(baseline_dir)],
        cwd=str(REPO),
    ).returncode
    if rc != 0:
        print("FAIL: baseline harness errored")
        return rc
    print(f"Baseline written to {baseline_dir}. Diff it against the previous run:")
    print(f"  git diff --stat {baseline_dir}")

    print("\n=== Test suite ===")
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "--tb=no", "-rf"],
        cwd=str(REPO),
    ).returncode
    print(
        "\nCompare the failure set against the known-red list in "
        "docs/PROJECT_STATUS.md - the bar is ZERO NEW failures, not a green run."
    )

    print("\n=== Not covered by this gate ===")
    print("  Build the bundle and run its self-test - the source passing has")
    print("  historically NOT predicted the standalone working:")
    print("    python build_installer_py312.py")
    print("    dist\\SpectralPredict-py312\\SpectralPredict-py312.exe --test")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", action="store_true",
                    help="also run the baseline and test suite")
    args = ap.parse_args()
    rc = report()
    if args.gate:
        rc = gate() or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
