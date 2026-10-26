"""Compare calibration-transfer methods on paired scans from two instruments.

Reads two folders of spectra (any format ``io.read_spectra`` handles), pairs them by a
table, and runs ``evaluate_transfer`` with the physical-match candidates, grouped
scoring when the table has a group column, and the noise and join diagnostics.

    python tools/transfer_pairs_benchmark.py PRIMARY_DIR SATELLITE_DIR pairs.csv \
        [--model model.dasp] [--space reflectance] [--joins auto|header|none]
        [--fit-regions all|water|1100-1350,1550-1800]

``pairs.csv`` has columns ``primary`` and ``satellite`` (file stems or names) and
optionally ``group`` (instrument unit, session or batch). Data stay where they are;
nothing is written except an optional ``--out`` CSV of the leaderboard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral_predict.io import read_spectra  # noqa: E402
from spectral_predict.physical_match import WATER_BANDS_NM  # noqa: E402
from spectral_predict.spectral_diagnostics import union_joins  # noqa: E402
from spectral_predict.transfer_evaluation import (  # noqa: E402
    evaluate_transfer,
    predict_fn_from_model,
)


def _stem(name: str) -> str:
    return Path(str(name)).stem.strip().lower()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("primary_dir")
    ap.add_argument("satellite_dir")
    ap.add_argument("pairs_csv")
    ap.add_argument("--model", help="a saved regression .dasp model")
    ap.add_argument("--space", choices=["reflectance", "absorbance"])
    ap.add_argument("--joins", default="auto", choices=["auto", "header", "none"])
    ap.add_argument(
        "--fit-regions",
        default="all",
        help="physical-match fit regions: 'all' (default; the board adds a water-bands-excluded "
        "row by itself), 'water' (leave out the water bands) or nm ranges to fit only on",
    )
    ap.add_argument("--out", help="write the leaderboard to this CSV")
    args = ap.parse_args(argv)

    Xp, meta_p = read_spectra(args.primary_dir)
    Xs, meta_s = read_spectra(args.satellite_dir)
    pairs = pd.read_csv(args.pairs_csv)
    p_idx = {_stem(i): i for i in Xp.index}
    s_idx = {_stem(i): i for i in Xs.index}
    rows = [
        (p_idx[_stem(r.primary)], s_idx[_stem(r.satellite)], getattr(r, "group", None))
        for r in pairs.itertuples()
        if _stem(r.primary) in p_idx and _stem(r.satellite) in s_idx
    ]
    if len(rows) < 3:
        print(f"Only {len(rows)} pairs found in both folders; need at least 3.")
        return 1
    wl = Xp.columns.astype(float).to_numpy()
    if not np.array_equal(wl, Xs.columns.astype(float).to_numpy()):
        print("The two folders are on different wavelength grids; resample first.")
        return 1
    A = Xp.loc[[r[0] for r in rows]].to_numpy(float)
    B = Xs.loc[[r[1] for r in rows]].to_numpy(float)
    groups = [r[2] for r in rows] if "group" in pairs else None
    space = args.space or meta_p.get("data_type", "reflectance")
    scale = float(meta_p.get("value_scale", 1.0) or 1.0) if space == "reflectance" else 1.0
    A, B = A / scale, B / scale
    joins: object = "auto"
    if args.joins == "none":
        joins = None
    elif args.joins == "header":
        joins = (
            union_joins(meta_s.get("detector_joins"), meta_p.get("detector_joins")) or "auto"
        )
    exclude = fit_regions = None
    if args.fit_regions == "water":
        exclude = WATER_BANDS_NM
    elif args.fit_regions != "all":
        fit_regions = [
            tuple(float(v) for v in part.split("-")) for part in args.fit_regions.split(",")
        ]

    predict = model_wl = None
    if args.model:
        from spectral_predict.model_io import load_model

        model = load_model(args.model)
        predict = predict_fn_from_model(model, wl)
        model_wl = (model.get("metadata") or {}).get("wavelengths")

    ev = evaluate_transfer(
        A,
        B,
        predict=predict,
        ids=[r[0] for r in rows],
        groups=groups,
        wavelengths=wl,
        space=space,
        joins=joins,
        model_wavelengths=model_wl,
        exclude=exclude,
        fit_regions=fit_regions,
    )
    cols = [
        c
        for c in (
            "label",
            ev.score_column,
            "Slope_vs_primary_grouped",
            "RMSD_vs_primary",
            "Slope_vs_primary",
            "spectral_RMSE",
            "error",
        )
        if c in ev.leaderboard
    ]
    pd.set_option("display.width", 200)
    print(f"{len(rows)} pairs; ranked by {ev.score_column}\n")
    print(ev.leaderboard[list(dict.fromkeys(cols))].head(15).round(3).to_string())
    d = ev.diagnostics
    print(f"\njoins: {d['joins']} ({d['joins_source']})")
    print(f"primary noisy: {d['primary_noisy_regions']}")
    print(f"satellite noisy: {d['satellite_noisy_regions']}")
    for w in ev.warnings:
        print(f"- {w}")
    if args.out:
        ev.leaderboard.drop(columns=["settings"]).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
