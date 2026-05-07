"""Helper for extracting the actually-fitted ``n_components`` from a PLS
trial's params dict. Used by the Bayesian and NSGA-II search paths to
populate the LVs column with the post-clamp value (rather than Optuna's
raw pre-clamp suggestion). Sole survivor of this module after the legacy
``run_bayesian_search`` deletion (2026-05-07)."""

import ast
import logging
from typing import Any, Optional


def _extract_fitted_n_components(params_value: Any) -> Optional[int]:
    """Pull the post-clamp n_components from a stored model-params value.

    Handles three shapes the codebase emits:
      - bare 'n_components' (pre-fit dict from suggest_model_params)
      - 'model__n_components' (Pipeline-prefixed, regression PLS captured params)
      - 'pls__n_components' (PLS-DA classifier captured params)

    Returns None on:
      - unrecognized input type (None, empty string, non-dict literal)
      - parse failure (logged at WARNING level — bug signal)
      - no recognized key in the dict (legitimate non-PLS or missing field)
      - present-but-non-numeric value (logged at WARNING — bug signal)
    """
    if isinstance(params_value, dict):
        parsed = params_value
    elif isinstance(params_value, str) and params_value.strip():
        try:
            parsed = ast.literal_eval(params_value)
        except (ValueError, SyntaxError) as e:
            logging.getLogger(__name__).warning(
                "Failed to parse model_params string %r: %s", params_value[:200], e,
            )
            return None
        if not isinstance(parsed, dict):
            return None
    else:
        return None

    for key in ('n_components', 'model__n_components', 'pls__n_components'):
        if key in parsed:
            try:
                return int(parsed[key])
            except (TypeError, ValueError) as e:
                logging.getLogger(__name__).warning(
                    "Found %s=%r in model_params but cannot coerce to int: %s",
                    key, parsed[key], e,
                )
                return None
    return None
