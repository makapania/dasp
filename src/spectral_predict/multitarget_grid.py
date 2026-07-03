"""T-17 multi-target grid orchestrator: preprocessing × varsel × model × hp.

New orchestration layer ABOVE the F2 seed evaluator. Reuses pure primitives
from search.py / preprocess.py / models.py / variable_selection.py / ga_pls.py /
multi_y.py and the F2 cell helper from multitarget_search.py. Never touches
run_search's single-Y path. Grid engine ONLY.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

_SG_SPEC = {"sg1": (1, 2), "sg2": (2, 3), "sg3": (3, 4), "sg4": (4, 5)}


def _base_config(name: str, deriv, window, polyorder, *, interference_to_add,
                 baseline_method, baseline_params, smoothing,
                 smoothing_window, smoothing_polyorder) -> dict[str, Any]:
    return {
        "name": name, "deriv": deriv, "window": window, "polyorder": polyorder,
        "interference": interference_to_add,
        "baseline_method": baseline_method, "baseline_params": baseline_params,
        "smoothing": smoothing, "smoothing_window": smoothing_window,
        "smoothing_polyorder": smoothing_polyorder,
    }


def build_multitarget_preprocess_configs(
    preprocessing_methods: dict[str, bool],
    *,
    window_sizes: Optional[Sequence[int]] = None,
    autoscale: bool = False,
    baseline_method: Optional[str] = None,
    baseline_params: Optional[dict] = None,
    smoothing: bool = False,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    interference_to_add: Any = None,
) -> list[dict[str, Any]]:
    """Enumerate the basic preprocessing grid (mirrors search.py:2288-2616)."""
    pm = preprocessing_methods or {}
    if window_sizes is None:
        window_sizes = [11]
    common = dict(
        interference_to_add=interference_to_add,
        baseline_method=baseline_method, baseline_params=baseline_params,
        smoothing=smoothing, smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder,
    )
    configs: list[dict[str, Any]] = []
    if pm.get("raw", False):
        configs.append(_base_config("raw", None, None, None, **common))
    if pm.get("snv", False):
        configs.append(_base_config("snv", None, None, None, **common))
    for sg, (deriv, poly) in _SG_SPEC.items():
        if not pm.get(sg, False):
            continue
        for w in window_sizes:
            configs.append(_base_config("deriv", deriv, w, poly, **common))
            if pm.get("snv", False):
                configs.append(_base_config("snv_deriv", deriv, w, poly, **common))
            if pm.get("deriv_snv", False):
                configs.append(_base_config("deriv_snv", deriv, w, poly, **common))
    if not configs:
        configs.append(_base_config("raw", None, None, None, **common))

    # Baseline doubling (search.py:2565).
    if baseline_method is not None and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["baseline_method"] = None; no["baseline_params"] = None
            without.append(no)
            bl = dict(cfg); bl["base_name"] = cfg.get("base_name", cfg["name"])
            bl["name"] = f"{baseline_method}+{cfg['name']}"
            with_.append(bl)
        configs = without + with_

    # Smoothing doubling (search.py:2582).
    if smoothing and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["smoothing"] = False
            without.append(no)
            sm = dict(cfg); sm["base_name"] = cfg.get("base_name", cfg["name"])
            nm = cfg["name"]
            sm["name"] = (f"{nm.split('+', 1)[0]}+sg0+{nm.split('+', 1)[1]}"
                          if "+" in nm else f"sg0+{nm}")
            with_.append(sm)
        configs = without + with_

    # Autoscale doubling (search.py:2603).
    if autoscale and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["autoscale"] = False
            without.append(no)
            sc = dict(cfg); sc["autoscale"] = True
            sc["base_name"] = cfg.get("base_name", cfg["name"])
            sc["name"] = cfg["name"] + "+autoscale"
            with_.append(sc)
        configs = without + with_

    return configs
