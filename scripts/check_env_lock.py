"""Report whether the running environment matches requirements-lock.txt.

    <venv>\\Scripts\\python scripts\\check_env_lock.py          # exit 0 in sync, 1 drifted
    <venv>\\Scripts\\python scripts\\check_env_lock.py --quiet  # print nothing when in sync

`pip install -r requirements-lock.txt` only runs when someone remembers to run it.
An environment built before a pin changed keeps the old version indefinitely: a
.venv314 left on jcamp 1.2.2 after the lock moved to 1.3.2 broke JCAMP-DX import
with no warning. The launchers and the installer build call this so a changed
lockfile is applied automatically rather than by a manual checklist step.

Standard library only: it must run in an environment that is itself broken.
Packages installed but absent from the lock are ignored, because pip install -r
does not remove them either.
"""

from __future__ import annotations

import argparse
import codecs
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO / "requirements-lock.txt"
PROJECT_DISTRIBUTION = "spectral-predict"

EXIT_IN_SYNC = 0
EXIT_DRIFTED = 1
EXIT_BAD_LOCK = 2

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock_text(path: Path) -> str:
    """Decode a lockfile, honoring the BOM that PowerShell `>` redirection writes."""
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def parse_lock(text: str) -> dict[str, str]:
    """Return {normalized name: pinned version} for every `name==version` line.

    Raises:
        ValueError: A requirement line is not an exact `==` pin.
    """
    pins: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _PIN.match(line)
        if match is None:
            raise ValueError(f"line {line_number} is not an exact pin: {raw.strip()!r}")
        pins[normalize_name(match.group(1))] = match.group(2)
    return pins


def find_drift(
    pins: dict[str, str], installed_version=version
) -> list[tuple[str, str, str | None]]:
    """Return (name, pinned, installed-or-None) for every pin the environment does not match."""
    drift = []
    for name, pinned in sorted(pins.items()):
        try:
            installed = installed_version(name)
        except PackageNotFoundError:
            installed = None
        if installed != pinned:
            drift.append((name, pinned, installed))
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--quiet", action="store_true", help="print nothing when in sync")
    args = parser.parse_args(argv)

    try:
        pins = parse_lock(read_lock_text(args.lock))
    except (OSError, ValueError) as exc:
        print(f"Cannot read {args.lock}: {exc}", file=sys.stderr)
        return EXIT_BAD_LOCK

    drift = find_drift(pins)
    try:
        version(PROJECT_DISTRIBUTION)
    except PackageNotFoundError:
        drift.append((PROJECT_DISTRIBUTION, "editable install", None))

    if not drift:
        if not args.quiet:
            print(f"Environment matches {args.lock.name} ({len(pins)} pins).")
        return EXIT_IN_SYNC

    print(f"Environment differs from {args.lock.name}:")
    for name, pinned, installed in drift:
        print(f"  {name}: installed {installed or 'MISSING'}, lock {pinned}")
    return EXIT_DRIFTED


if __name__ == "__main__":
    sys.exit(main())
