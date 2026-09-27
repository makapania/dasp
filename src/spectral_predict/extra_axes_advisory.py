"""Search-dimension advisory for the GUI's Bayesian extra-axes card (T-51 PR D).

Counts the Optuna dimensions one model's unified Bayesian study searches: the model's
base hyperparameters, the shared preprocessing and subset axes, and the axes of the
enabled bundles that apply to that model. Names are discovered by running the real
samplers on a recording trial, so the count follows the backend instead of a table.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from spectral_predict.search_spaces import (
    ExtraAxesConfigError,
    discover_derived_keys,
    discover_suggested_names,
    resolve_bundles,
)
from spectral_predict.unified_bayesian import (
    DEFAULT_N_STARTUP_TRIALS,
    suggest_model_params,
    suggest_one_class_params,
    suggest_preprocessing,
)

# Suggested inline in the unified objective (both the supervised and the one-class
# branch), so there is no sampler to record them from. A drift test compares this
# with the params of a real study.
SHARED_SUBSET_AXES: tuple[str, ...] = ("subset_type", "n_vars", "region_id")

# Names do not depend on the feature count (it only clamps ranges), so a nominal one
# stands in when no data is loaded.
NOMINAL_N_FEATURES = 100

# Models whose studies dominated wall time downstream (§10 of the T-51 plan).
_SLOW_MODELS = frozenset({"IsolationForest", "XGBoost", "LightGBM", "CatBoost"})


@dataclass(frozen=True)
class ModelDimensions:
    """Dimension count for one model's study."""

    model_name: str
    base: int
    shared: int
    extra: int
    bundles: tuple[str, ...]

    @property
    def total(self) -> int:
        return self.base + self.shared + self.extra


@lru_cache(maxsize=256)
def _base_names(model_name: str, task_type: str) -> frozenset[str]:
    if task_type == "one_class":
        sampler = lambda t: suggest_one_class_params(t, model_name)  # noqa: E731
    else:
        sampler = lambda t: suggest_model_params(  # noqa: E731
            t, model_name, NOMINAL_N_FEATURES, task_type
        )
    return discover_suggested_names(sampler)


@lru_cache(maxsize=256)
def _base_param_names(model_name: str, task_type: str) -> frozenset[str]:
    """What the backend passes to ``resolve_bundles`` (suggested names + derived keys)."""
    if task_type == "one_class":
        sampler = lambda t: suggest_one_class_params(t, model_name)  # noqa: E731
    else:
        sampler = lambda t: suggest_model_params(  # noqa: E731
            t, model_name, NOMINAL_N_FEATURES, task_type
        )
    return _base_names(model_name, task_type) | discover_derived_keys(sampler)


@lru_cache(maxsize=64)
def shared_axis_names(baseline: bool, smoothing: bool, autoscale: bool) -> frozenset[str]:
    """Preprocessing axes for the current Bayesian options, plus the subset axes."""
    preprocessing = discover_suggested_names(
        lambda t: suggest_preprocessing(
            t,
            NOMINAL_N_FEATURES,
            baseline_method="als" if baseline else None,
            smoothing=smoothing,
            enable_autoscale=autoscale,
        )
    )
    return preprocessing | frozenset(SHARED_SUBSET_AXES)


def model_dimensions(
    model_name: str,
    task_type: str,
    enabled_extra_axes: Iterable[str],
    *,
    baseline: bool = False,
    smoothing: bool = False,
    autoscale: bool = False,
) -> ModelDimensions:
    """Count the dimensions of ``model_name``'s study with the given bundles enabled.

    Bundles that do not apply to the model (or that the backend would reject for it)
    add nothing, matching what the run does.
    """
    enabled = tuple(sorted(set(enabled_extra_axes)))
    applicable: tuple = ()
    if enabled:
        try:
            applicable = resolve_bundles(
                model_name,
                task_type,
                enabled,
                base_param_names=_base_param_names(model_name, task_type),
            )
        except ExtraAxesConfigError:
            applicable = ()
    return ModelDimensions(
        model_name=model_name,
        base=len(_base_names(model_name, task_type)),
        shared=len(shared_axis_names(baseline, smoothing, autoscale)),
        extra=sum(len(bundle.axes) for bundle in applicable),
        bundles=tuple(bundle.id for bundle in applicable),
    )


def advisory_text(
    models: Sequence[str],
    task_type: str | None,
    enabled_extra_axes: Iterable[str],
    *,
    baseline: bool = False,
    smoothing: bool = False,
    autoscale: bool = False,
) -> str:
    """One caption for the card: the largest study's dimension and the caveats."""
    if task_type not in ("regression", "classification", "one_class"):
        return "Load data (or pick a task type) to see search dimensions."
    if not models:
        return "Select models to see search dimensions."
    enabled = tuple(enabled_extra_axes)
    dims = [
        model_dimensions(
            m, task_type, enabled, baseline=baseline, smoothing=smoothing, autoscale=autoscale
        )
        for m in models
    ]
    largest = max(dims, key=lambda d: d.total)
    opened = [d for d in dims if d.extra]
    if opened:
        lead = (
            f"Extra axes apply to {', '.join(d.model_name for d in opened)}. "
            f"Up to {largest.total} search dimensions ({largest.model_name})."
        )
    elif enabled:
        lead = (
            "None of the ticked bundles apply to the selected models. "
            f"Up to {largest.total} search dimensions ({largest.model_name})."
        )
    else:
        lead = f"Up to {largest.total} search dimensions ({largest.model_name})."
    text = (
        f"{lead} Startup trials: {DEFAULT_N_STARTUP_TRIALS} unless set; a common rule of "
        "thumb is at least 3 × dimensions. An upper-bound guide: opening axes improves "
        "the best candidates but lowers the average one, and widened spaces often peak "
        "earlier. Validate externally."
    )
    if any(d.model_name in _SLOW_MODELS for d in opened):
        text += " Boosting and IsolationForest trials are the slowest; expect longer runs."
    return text
