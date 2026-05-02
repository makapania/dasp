"""GUI-settings capture/restore for resume-on-restart (T-43).

`RunMetadata.gui_settings` (T-43) holds a flat ``dict[str, JSON-scalar]``
snapshot of the GUI's analysis-defining state at ``start_run`` time —
preprocessing toggles, model selection, variable-selection methods,
Bayesian/NSGA-II configuration, CV strategy, etc. On resume, the GUI
reads the dict back and writes it onto the live Tk variables so the user
can click Run Analysis without manually recreating each setting.

Why a curated whitelist instead of introspecting all Tk vars on the GUI
object: many of the ~700 Tk vars on the GUI describe display state
(filter dropdowns, chart colors, exploratory-tab UI) that should not be
mutated by a resume action. Restoring them silently would surprise the
user. The whitelist pins the contract to the analysis-defining surface.

Drift handling:
- New settings: add to ``CAPTURABLE_SETTINGS``. Older sidecars without
  the new key are treated as "no value to restore" (no crash, no warning).
- Removed settings: stale keys in older sidecars are silently dropped at
  restore time (returned in ``RestoreReport.skipped``).
- Renamed settings: handled by adding the new name to the whitelist; old
  name is dropped silently.

Dataset paths (``spectral_data_path``, ``reference_file``,
``combined_data_file``) are deliberately omitted: the dataset can move
or be renamed between sessions, and silently re-loading from a stale
path could corrupt the resume flow if the file is now different content
at the same path. The GUI surfaces those values for the user to confirm
manually instead of restoring them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Curated whitelist of analysis-defining Tk variable names. Only these are
# captured at start_run time and restored on resume. Adding a new setting
# requires adding it here.
CAPTURABLE_SETTINGS: tuple[str, ...] = (
    # --- preprocessing toggles ---
    "use_raw",
    "use_snv",
    "use_sg1",
    "use_sg2",
    "use_sg3",
    "use_sg4",
    "use_deriv_snv",
    "use_autoscale",
    "use_msc",
    "use_osc",
    "osc_n_components",
    # --- wavelength exclusion / analysis-window restriction ---
    "enable_wavelength_exclusion",
    "wavelength_exclude_ranges",
    "enable_analysis_wl_restriction",
    "analysis_wl_min",
    "analysis_wl_max",
    "analysis_wl_custom",
    # --- baseline correction ---
    "enable_baseline",
    "baseline_method",
    "baseline_poly_degree",
    "baseline_asls_lambda",
    "baseline_asls_p",
    "baseline_airpls_lambda",
    "baseline_advanced_algorithm",
    "baseline_advanced_lam",
    # --- smoothing ---
    "enable_smoothing",
    "smoothing_window",
    "smoothing_polyorder",
    # --- SG derivative window choices (BooleanVars) ---
    "window_7",
    "window_11",
    "window_17",
    "window_23",
    "window_31",
    # --- preprocessing-discovery search modes ---
    "enable_smart_preprocessing",
    "smart_preprocess_importance",
    "smart_preprocess_n_top",
    "enable_tpe_preprocessing",
    "tpe_preprocess_n_trials",
    "tpe_preprocess_n_top",
    "enable_ga_preprocessing",
    "ga_preprocess_method",
    "ga_preprocess_population",
    "ga_preprocess_generations",
    "ga_preprocess_cv_folds",
    # --- variable-subset toggles ---
    "enable_variable_subsets",
    "enable_region_subsets",
    "n_top_regions",
    "region_test_all_individual",
    "region_test_pairwise",
    "var_10",
    "var_20",
    "var_50",
    "var_100",
    "var_250",
    "var_500",
    "var_1000",
    # --- variable-selection methods ---
    "varsel_importance",
    "varsel_spa",
    "varsel_uve",
    "varsel_uve_spa",
    "varsel_ipls",
    "varsel_ipls_forward",
    "varsel_ipls_backward",
    "varsel_mc_sipls",
    "varsel_mwpls",
    "varsel_cars",
    "varsel_cars_tree",
    "varsel_vcpa",
    "varsel_ga",
    "varsel_uve_cars",
    "varsel_uve_cars_tree",
    "varsel_uve_cars_spa",
    "varsel_fipls_spa",
    "varsel_fipls_cars",
    "uve_cutoff_multiplier",
    "uve_n_components",
    "ipls_n_intervals",
    "ipls_max_combine",
    "ipls_subset_limit",
    # --- GA-PLS params ---
    "ga_population_size",
    "ga_generations",
    "ga_n_runs",
    "ga_quick_mode",
    # --- model selection ---
    "use_pls",
    "use_plsda",
    "use_ridge",
    "use_lasso",
    "use_elasticnet",
    "use_randomforest",
    "use_lightgbm",
    "use_xgboost",
    "use_catboost",
    "use_neuralboosted",
    "use_svr",
    "use_svm",
    "use_mlp",
    "use_ocsvm",
    "use_isolation_forest",
    "use_elliptic_envelope",
    "use_lof",
    "use_pca_simca",
    "inlier_class_label",
    # --- one-class hyperparameters that change Bayesian search space ---
    "oc_nu",
    "oc_contamination",
    "oc_alpha",
    "oc_n_components",
    # --- tier / optimization method ---
    "model_tier",
    "optimization_method",
    "n_unified_trials",
    "nsga2_population",
    "nsga2_generations",
    "nsga2_selection_method",
    "nsga2_mode",
    "bayesian_persistence_mode",
    # --- task type / CV ---
    "task_type",
    "folds",
    "cv_strategy",
    "cv_n_repeats",
    "max_n_components",
    "max_iter",
    # --- imbalance handling ---
    "enable_imbalance_handling",
    "imbalance_method",
    # --- ensembles ---
    "enable_ensembles",
)


@dataclass
class RestoreReport:
    """Outcome of ``restore_gui_settings`` — surfaced to the GUI banner."""

    restored: list[str] = field(default_factory=list)
    skipped_unknown: list[str] = field(default_factory=list)
    skipped_no_var: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_restored(self) -> int:
        return len(self.restored)


def capture_gui_settings(gui_obj: Any) -> dict[str, Any]:
    """Snapshot the analysis-defining Tk vars from ``gui_obj``.

    Iterates :data:`CAPTURABLE_SETTINGS` and reads each Tk variable's
    current value via ``.get()``. Missing attributes are silently
    skipped (allows partial GUI initialization in tests / headless
    callers). Tk-var ``.get()`` failures are logged and the key is
    omitted from the snapshot.
    """
    captured: dict[str, Any] = {}
    for name in CAPTURABLE_SETTINGS:
        var = getattr(gui_obj, name, None)
        if var is None:
            continue
        getter = getattr(var, "get", None)
        if not callable(getter):
            continue
        try:
            captured[name] = getter()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "T-43: capture skipped %s (Tk var read failed: %s)", name, exc
            )
    return captured


def restore_gui_settings(
    gui_obj: Any, settings: dict[str, Any] | None
) -> RestoreReport:
    """Apply ``settings`` onto the GUI's Tk vars; return a per-key report.

    Each key:value pair is written via ``gui_obj.<key>.set(value)``. The
    function never raises — failures are accumulated into
    :class:`RestoreReport`. This keeps the GUI's resume flow robust to
    legacy sidecars, renamed settings, or schema drift.

    Categories of skip:
    - ``skipped_unknown``: key is not in :data:`CAPTURABLE_SETTINGS` (stale
      sidecar from a future build, or a removed setting).
    - ``skipped_no_var``: key is whitelisted but the GUI doesn't have a
      Tk var by that name (partially-initialized GUI in tests).
    - ``errors``: ``.set()`` raised (type coercion failure, bad value).
    """
    report = RestoreReport()
    if not settings:
        return report

    whitelisted = set(CAPTURABLE_SETTINGS)
    for name, value in settings.items():
        if name not in whitelisted:
            report.skipped_unknown.append(name)
            continue
        var = getattr(gui_obj, name, None)
        if var is None or not callable(getattr(var, "set", None)):
            report.skipped_no_var.append(name)
            continue
        try:
            var.set(value)
        except Exception as exc:
            report.errors.append(f"{name}: {exc}")
            continue

        # Tcl quirk: tk.IntVar.set("not-a-number") does NOT raise — Tcl stores
        # the string verbatim and the TclError surfaces on the next .get().
        # Read back and compare so a poisoned value is reported as an error
        # instead of being silently counted as restored.
        try:
            actual = var.get()
        except Exception as exc:
            report.errors.append(f"{name}: set succeeded but get raised: {exc}")
            continue
        if actual != value:
            report.errors.append(
                f"{name}: set succeeded but value mismatch "
                f"(expected {value!r}, got {actual!r})"
            )
            continue
        report.restored.append(name)
    return report


def summarize_gui_settings(settings: dict[str, Any] | None) -> str:
    """Produce a short human-readable summary for the resume banner.

    Highlights settings that most often differ between sessions —
    optimization method, n_trials, model selection, preprocessing
    toggles, variable-selection toggles. Returns an empty string when
    ``settings`` is None or empty so the caller can append conditionally.
    """
    if not settings:
        return ""

    def _bool(key: str) -> bool:
        return bool(settings.get(key))

    def _val(key: str, default: Any = "?") -> Any:
        v = settings.get(key)
        return default if v is None else v

    enabled_models = sorted(
        name.removeprefix("use_")
        for name in settings
        if name.startswith("use_") and bool(settings[name])
    )
    enabled_preproc = sorted(
        name.removeprefix("use_")
        for name in (
            "use_raw",
            "use_snv",
            "use_sg1",
            "use_sg2",
            "use_sg3",
            "use_sg4",
            "use_deriv_snv",
            "use_msc",
            "use_osc",
        )
        if _bool(name)
    )
    enabled_varsel = sorted(
        name.removeprefix("varsel_")
        for name in settings
        if name.startswith("varsel_") and bool(settings[name])
    )

    lines = [
        f"Optimization: {_val('optimization_method')}"
        f" ({_val('n_unified_trials')} trials/model)"
        if _val("optimization_method") == "unified"
        else f"Optimization: {_val('optimization_method')}",
        f"Tier: {_val('model_tier')}    Task: {_val('task_type')}",
        f"CV: {_val('cv_strategy')}, folds={_val('folds')}, repeats={_val('cv_n_repeats')}",
        f"Models ({len(enabled_models)}): {', '.join(enabled_models) or '(none)'}",
        f"Preprocessing ({len(enabled_preproc)}): {', '.join(enabled_preproc) or '(none)'}"
        + (f"  +autoscale" if _bool("use_autoscale") else ""),
        f"Variable selection ({len(enabled_varsel)}): "
        f"{', '.join(enabled_varsel) or '(none)'}",
    ]

    extras: list[str] = []
    if _bool("enable_baseline"):
        extras.append(f"baseline={_val('baseline_method')}")
    if _bool("enable_smoothing"):
        extras.append(
            f"smoothing(window={_val('smoothing_window')}, "
            f"poly={_val('smoothing_polyorder')})"
        )
    if _bool("enable_tpe_preprocessing"):
        extras.append(
            f"TPE-preprocessing({_val('tpe_preprocess_n_trials')} trials)"
        )
    if _bool("enable_smart_preprocessing"):
        extras.append("smart-preprocessing=on")
    if _bool("enable_ga_preprocessing"):
        extras.append(f"GA-preprocessing={_val('ga_preprocess_method')}")
    if _bool("enable_imbalance_handling"):
        extras.append(f"imbalance={_val('imbalance_method')}")
    if extras:
        lines.append("Extras: " + ", ".join(extras))

    return "\n".join(lines)
