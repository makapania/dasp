#!/usr/bin/env python
"""Convert a folder of legacy float32 ASD-v1 binary spectra to a single wide CSV.

These are the oldest ASD binary files (version ``ASD\\0``, float32 spectrum at offset
484) that SpecDAL misreads as all-NaN. They typically carry instrument extensions like
``.sco`` or numbered ``.000``/``.001`` rather than ``.asd``. This script reuses the same
native decoder the app uses (`spectral_predict.readers.asd_native.read_legacy_asd`).

Usage:
    python scripts/convert_old_asd.py INPUT_DIR [-o OUTPUT.csv] [--variant {sco,raw,both}]

Output CSV is wide: rows = samples (filename), columns = wavelengths (nm).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spectral_predict.readers.asd_native import read_legacy_asd  # noqa: E402


def _discover(input_dir: Path, variant: str) -> list[Path]:
    """Return candidate legacy-ASD files for the chosen variant.

    ``sco``  -> ``*.sco`` (processed reflectance; default, unique stems)
    ``raw``  -> numbered files like ``name.000`` (bare numeric extension)
    ``both`` -> union of the two
    """
    files: list[Path] = []
    if variant in ("sco", "both"):
        files += sorted(input_dir.glob("*.sco"))
    if variant in ("raw", "both"):
        files += sorted(
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix[1:].isdigit() and not p.name.endswith(".sco")
        )
    return files


def convert(input_dir: Path, output_csv: Path, variant: str) -> pd.DataFrame:
    candidates = _discover(input_dir, variant)
    if not candidates:
        raise SystemExit(f"No '{variant}' legacy-ASD files found in {input_dir}")

    spectra: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for path in candidates:
        try:
            series = read_legacy_asd(path)
        except ValueError as exc:  # corrupt/truncated legacy file
            skipped.append(f"{path.name}: {exc}")
            continue
        if series is None:  # not the legacy float32 layout
            skipped.append(f"{path.name}: not a legacy float32 ASD file")
            continue
        # Use the full filename except for 'sco', where the stem (e.g. "italy.000")
        # is already unique. Bare numbered files (italy.000, italy.001) all share the
        # stem "italy", so 'raw'/'both' must keep the extension to avoid silent
        # dict overwrites that would collapse many spectra into one row.
        sample_id = path.stem if variant == "sco" else path.name
        if sample_id in spectra:
            print(f"Warning: duplicate sample id '{sample_id}' - overwriting earlier file")
        spectra[sample_id] = series

    if not spectra:
        raise SystemExit("No spectra could be decoded.")

    df = pd.DataFrame(spectra).T
    df = df[sorted(df.columns)]
    df.index.name = "sample"
    df.to_csv(output_csv)

    print(f"Read {len(spectra)} spectra ({df.shape[1]} wavelengths) -> {output_csv}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for line in skipped[:10]:
            print(f"  - {line}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="Folder of legacy ASD files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_dir>/converted_spectra.csv)",
    )
    parser.add_argument(
        "--variant",
        choices=("sco", "raw", "both"),
        default="sco",
        help="Which file variant to import (default: sco)",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.input_dir}")
    output = args.output or (args.input_dir / "converted_spectra.csv")

    convert(args.input_dir, output, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
