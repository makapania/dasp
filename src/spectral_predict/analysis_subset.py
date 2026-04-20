"""Pure matching/summary logic for the Analysis Subset feature.

This module contains all testable matching and formatting logic so that
tests can validate behaviour without spinning up Tkinter.

GUI code imports these functions and delegates; the GUI file holds only
thin UI glue.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_matches(df: pd.DataFrame, filter_def: dict[str, Any]) -> set:
    """Return set of df index values that match *filter_def*.

    Parameters
    ----------
    df
        A DataFrame whose index corresponds to sample identifiers (may be
        ``combined_metadata_df`` or a reference DataFrame).
    filter_def
        The dict stored at ``self.active_group_filter``.  Expected keys:
        ``column``, ``condition``, and condition-specific keys (``value``,
        ``value2``, ``values``).

    Returns
    -------
    set
        Matching index values.  Returns an empty set when the column is
        missing from *df* or the filter dict is malformed.
    """
    if not filter_def or not isinstance(filter_def, dict):
        return set()

    col: str = filter_def.get("column", "")
    cond: str = filter_def.get("condition", "")

    if not col or not cond:
        return set()

    if col not in df.columns:
        return set()

    series = df[col]

    if cond == "has value":
        mask = series.notna()
        if pd.api.types.is_string_dtype(series.dtype):
            mask = mask & (series.astype(str).str.strip() != "")
        return set(series.index[mask])

    if cond == "contains":
        val = str(filter_def.get("value", ""))
        mask = series.astype(str).str.contains(val, case=False, na=False, regex=False)
        return set(series.index[mask])

    if cond == "in":
        values: list = filter_def.get("values", [])
        if not values:
            return set()
        str_values = [str(v) for v in values]
        mask = series.astype(str).isin(str_values)
        return set(series.index[mask])

    # Numeric / string comparisons
    val = filter_def.get("value", "")
    val2 = filter_def.get("value2", "")

    numeric = pd.to_numeric(series, errors="coerce")
    try:
        val_num = float(val) if val not in (None, "") else None
    except (ValueError, TypeError):
        val_num = None

    if cond in ("==", "!=") and val_num is None:
        str_series = series.astype(str)
        if cond == "==":
            mask = str_series == str(val)
        else:
            mask = str_series != str(val)
        return set(series.index[mask])

    if val_num is None:
        return set()

    if cond == "==":
        mask = numeric == val_num
    elif cond == "!=":
        mask = numeric != val_num
    elif cond == ">":
        mask = numeric > val_num
    elif cond == "<":
        mask = numeric < val_num
    elif cond == ">=":
        mask = numeric >= val_num
    elif cond == "<=":
        mask = numeric <= val_num
    elif cond == "between":
        try:
            val2_num = float(val2) if val2 not in (None, "") else None
        except (ValueError, TypeError):
            val2_num = None
        if val2_num is None:
            return set()
        lo, hi = min(val_num, val2_num), max(val_num, val2_num)
        mask = (numeric >= lo) & (numeric <= hi)
    else:
        return set()

    return set(series.index[mask.fillna(False)])


def format_summary(filter_def: dict[str, Any] | None) -> str:
    """Return a human-readable one-liner for the filter.

    Parameters
    ----------
    filter_def
        The active-group filter dict, or ``None`` / falsy.

    Returns
    -------
    str
        ``"All samples"`` when *filter_def* is falsy; otherwise a formatted
        summary string.
    """
    if not filter_def:
        return "All samples"

    col = filter_def.get("column", "?")
    cond = filter_def.get("condition", "")

    if cond == "has value":
        return f"{col} has value"
    if cond == "between":
        val = filter_def.get("value", "")
        val2 = filter_def.get("value2", "")
        return f"{col} between {val} and {val2}"
    if cond == "contains":
        val = filter_def.get("value", "")
        return f"{col} contains {val}"
    if cond == "in":
        values = filter_def.get("values", [])
        joined = ", ".join(str(v) for v in values)
        return f"{col} in [{joined}]"

    val = filter_def.get("value", "")
    return f"{col} {cond} {val}"


def is_categorical_column(series: pd.Series, max_unique: int = 20) -> bool:
    """Decide whether a column should be offered as exact-value multi-select.

    Returns ``True`` when *series* has a non-numeric dtype, **or** is numeric
    with ``<= max_unique`` unique non-null values.
    """
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return True
    n_unique = series.dropna().nunique()
    return n_unique <= max_unique


def classify_column_type(series: pd.Series, numeric_threshold: float = 0.9) -> str:
    """Return 'numeric' or 'categorical' for subset-UI purposes.

    Rule (explicit — do not substitute anything else):
    1. Drop nulls. If the remaining series is empty, return 'categorical'.
    2. If dtype is already numeric (pd.api.types.is_numeric_dtype), return 'numeric'.
    3. Otherwise, for each non-null value: strip surrounding whitespace, then if
       the value ends with a single trailing '%' strip that too. Attempt
       pd.to_numeric on the cleaned string.
    4. If >= numeric_threshold (default 0.9) of non-null values parse
       cleanly as numeric, return 'numeric'. Otherwise return 'categorical'.

    Examples:
    - [1, 2, 3] -> 'numeric' (native dtype)
    - ['1', '2', '3'] -> 'numeric' (all strings parse)
    - ['12.5%', '7.1%', '3.0%'] -> 'numeric' (trailing %, all parse)
    - ['2021', '2022', 'unknown'] -> 'categorical' (only 2/3 = 0.67 < 0.9)
    - ['grass', 'tree', 'sedge'] -> 'categorical' (none parse)
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical"
    if pd.api.types.is_numeric_dtype(non_null.dtype):
        return "numeric"
    parsed = 0
    for val in non_null:
        s = str(val).strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        try:
            pd.to_numeric(s)
            parsed += 1
        except (ValueError, TypeError):
            pass
    if parsed / len(non_null) >= numeric_threshold:
        return "numeric"
    return "categorical"


def get_unique_non_null_values(series: pd.Series, max_count: int = 200) -> list:
    """Return sorted unique non-null values from a pandas Series.

    For the subset dialog's categorical dropdown / multi-select population.
    Caps at *max_count* for UI responsiveness; if there are more unique values,
    returns the first *max_count* in sort order.

    Sort order:
    - If all surviving values are numeric-coercible (pd.to_numeric succeeds),
      sort numerically.
    - Otherwise, case-insensitive string sort on the string representation.

    This avoids the "2021 sorts before 999" surprise for year-like object
    columns.
    """
    non_null = series.dropna()
    unique_vals = list(non_null.unique())
    if not unique_vals:
        return []
    coerced = pd.to_numeric(pd.Series(unique_vals), errors="coerce")
    if coerced.notna().all():
        order = coerced.argsort()
        sorted_vals = [unique_vals[i] for i in order]
    else:
        sorted_vals = sorted(unique_vals, key=lambda x: str(x).lower())
    return list(sorted_vals[:max_count])


def build_filter_dict(
    column: str, column_kind: str, operator: str, raw_values: dict
) -> dict:
    """Construct the filter dict that ``compute_matches`` consumes.

    Parameters
    ----------
    column
        Column name.
    column_kind
        ``'numeric'`` or ``'categorical'`` (from ``classify_column_type``).
    operator
        One of the supported conditions.
    raw_values
        Dict with keys ``'value'``, ``'value2'``, ``'values'`` (list),
        depending on operator — dialog populates only the relevant keys.

    Returns
    -------
    dict
        Shape matching what ``compute_matches`` handles.

    Raises
    ------
    ValueError
        If *operator* is unknown.
    """
    if operator == "in":
        vals = raw_values.get("values")
        if vals is None or vals == []:
            raise ValueError("'in' requires a non-empty 'values' list in raw_values")
        return {"column": column, "condition": "in", "values": vals}
    if operator == "between":
        v1 = raw_values.get("value")
        v2 = raw_values.get("value2")
        if not v1 or (isinstance(v1, str) and not v1.strip()):
            raise ValueError("'between' requires both 'value' and 'value2' in raw_values")
        if not v2 or (isinstance(v2, str) and not v2.strip()):
            raise ValueError("'between' requires both 'value' and 'value2' in raw_values")
        return {
            "column": column,
            "condition": "between",
            "value": v1,
            "value2": v2,
        }
    if operator == "has value":
        return {"column": column, "condition": "has value"}
    if operator in ("==", "!=", "<", "<=", ">", ">=", "contains"):
        v = raw_values.get("value")
        if not v or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"'{operator}' requires 'value' in raw_values")
        return {
            "column": column,
            "condition": operator,
            "value": v,
            "value2": "",
        }
    raise ValueError(f"Unknown operator: {operator}")


def build_training_subset_metadata(
    active: bool,
    filter_def: dict[str, Any] | None,
    n_samples: int | None,
) -> dict[str, Any]:
    """Build the four ``analysis_subset_*`` keys for ``last_training_config``.

    Parameters
    ----------
    active
        Whether a subset filter is currently active.
    filter_def
        The active filter dict (``None`` when *active* is ``False``).
    n_samples
        Count of samples matched by the subset **before** exclusions and
        validation removal.  ``None`` when inactive (C3).

    Returns
    -------
    dict
        Keys: ``analysis_subset_active``, ``analysis_subset_filter``,
        ``analysis_subset_summary``, ``analysis_subset_n_samples``.
    """
    if not active or not filter_def:
        return {
            "analysis_subset_active": False,
            "analysis_subset_filter": None,
            "analysis_subset_summary": "All samples",
            "analysis_subset_n_samples": None,
        }

    return {
        "analysis_subset_active": True,
        "analysis_subset_filter": dict(filter_def),
        "analysis_subset_summary": format_summary(filter_def),
        "analysis_subset_n_samples": n_samples,
    }


def check_one_class_inlier_guard(
    y_filtered: pd.Series,
    inlier_label: str | int | float,
) -> str | None:
    """Return a blocking message if zero inlier samples remain, else ``None``.

    Parameters
    ----------
    y_filtered
        Target series **after** subset, exclusion, and validation filtering.
    inlier_label
        The intended inlier class label.

    Returns
    -------
    str or None
        A human-readable error message when zero inlier samples remain, or
        ``None`` when at least one inlier sample is present.
    """
    y_str = y_filtered.astype(str)
    n_inliers = (y_str == str(inlier_label)).sum()
    if n_inliers == 0:
        return (
            f"The active Analysis Subset excludes all samples for inlier class "
            f"'{inlier_label}'. Clear or change the subset, or choose a different "
            f"inlier class."
        )
    return None
