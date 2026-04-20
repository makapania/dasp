"""Tests for the Analysis Subset pure-logic module.

Covers compute_matches, format_summary, is_categorical_column,
build_training_subset_metadata, check_one_class_inlier_guard,
classify_column_type, get_unique_non_null_values, and build_filter_dict.
No Tkinter fixtures required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.analysis_subset import (
    build_filter_dict,
    build_training_subset_metadata,
    check_one_class_inlier_guard,
    classify_column_type,
    compute_matches,
    format_summary,
    get_unique_non_null_values,
    is_categorical_column,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small metadata DataFrame for testing."""
    return pd.DataFrame(
        {
            "PlantType": ["grass", "tree", "grass", "sedge", "tree", "grass"],
            "HarvestYear": [2021, 2022, 2021, 2023, 2022, 2023],
            "Collector": ["Alice", "Bob", "alice", "Carol", "bob", "Carol"],
            "Score": [1.5, 2.0, 1.5, 3.0, np.nan, 2.5],
            "Note": ["ok", "  ", np.nan, "good", "ok", "bad"],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )


# ========================================================================
# compute_matches — categorical "in" exact matching
# ========================================================================

class TestComputeMatchesIn:
    def test_in_single_value(self, sample_df: pd.DataFrame):
        filt = {"column": "PlantType", "condition": "in", "values": ["grass"]}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s3", "s6"}

    def test_in_multiple_values(self, sample_df: pd.DataFrame):
        filt = {"column": "PlantType", "condition": "in", "values": ["grass", "sedge"]}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s3", "s4", "s6"}

    def test_in_empty_values_list(self, sample_df: pd.DataFrame):
        filt = {"column": "PlantType", "condition": "in", "values": []}
        result = compute_matches(sample_df, filt)
        assert result == set()

    def test_in_no_match(self, sample_df: pd.DataFrame):
        filt = {"column": "PlantType", "condition": "in", "values": ["shrub"]}
        result = compute_matches(sample_df, filt)
        assert result == set()

    def test_in_numeric_coerced_to_string(self, sample_df: pd.DataFrame):
        """Values in the 'values' list are string-coerced for comparison."""
        filt = {"column": "HarvestYear", "condition": "in", "values": ["2021", "2023"]}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s3", "s4", "s6"}


# ========================================================================
# compute_matches — "contains" uses literal substring (not regex)
# ========================================================================

class TestComputeMatchesContains:
    def test_contains_literal_substring(self, sample_df: pd.DataFrame):
        filt = {"column": "Collector", "condition": "contains", "value": "bob"}
        result = compute_matches(sample_df, filt)
        assert "s2" in result
        assert "s5" in result

    def test_contains_dot_is_literal(self):
        """Confirm 'a.b' matches literal 'a.b' and does NOT regex-match 'axb'."""
        df = pd.DataFrame({"col": ["a.b", "axb", "aXb"]}, index=["r1", "r2", "r3"])
        filt = {"column": "col", "condition": "contains", "value": "a.b"}
        result = compute_matches(df, filt)
        assert result == {"r1"}

    def test_contains_case_insensitive(self, sample_df: pd.DataFrame):
        filt = {"column": "Collector", "condition": "contains", "value": "alice"}
        result = compute_matches(sample_df, filt)
        assert "s1" in result
        assert "s3" in result


# ========================================================================
# compute_matches — "has value" excludes blank / whitespace-only strings
# ========================================================================

class TestComputeMatchesHasValue:
    def test_has_value_excludes_nan(self, sample_df: pd.DataFrame):
        filt = {"column": "Score", "condition": "has value"}
        result = compute_matches(sample_df, filt)
        assert "s5" not in result

    def test_has_value_excludes_blank_strings(self, sample_df: pd.DataFrame):
        filt = {"column": "Note", "condition": "has value"}
        result = compute_matches(sample_df, filt)
        assert "s2" not in result
        assert "s3" not in result

    def test_has_value_includes_real_strings(self, sample_df: pd.DataFrame):
        filt = {"column": "Note", "condition": "has value"}
        result = compute_matches(sample_df, filt)
        assert "s1" in result
        assert "s4" in result
        assert "s5" in result
        assert "s6" in result


# ========================================================================
# compute_matches — numeric "between" (inclusive both ends)
# ========================================================================

class TestComputeMatchesBetween:
    def test_between_inclusive(self, sample_df: pd.DataFrame):
        filt = {"column": "HarvestYear", "condition": "between", "value": "2021", "value2": "2022"}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s2", "s3", "s5"}

    def test_between_boundary_values_included(self, sample_df: pd.DataFrame):
        filt = {"column": "HarvestYear", "condition": "between", "value": "2021", "value2": "2021"}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s3"}

    def test_between_missing_value2(self, sample_df: pd.DataFrame):
        filt = {"column": "HarvestYear", "condition": "between", "value": "2021", "value2": ""}
        result = compute_matches(sample_df, filt)
        assert result == set()


# ========================================================================
# compute_matches — missing-column case returns empty set
# ========================================================================

class TestComputeMatchesMissingColumn:
    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame):
        filt = {"column": "Nonexistent", "condition": "==", "value": "x"}
        result = compute_matches(sample_df, filt)
        assert result == set()

    def test_empty_filter_returns_empty(self, sample_df: pd.DataFrame):
        result = compute_matches(sample_df, {})
        assert result == set()

    def test_none_filter_returns_empty(self, sample_df: pd.DataFrame):
        result = compute_matches(sample_df, None)
        assert result == set()

    def test_malformed_filter_returns_empty(self, sample_df: pd.DataFrame):
        result = compute_matches(sample_df, {"column": "", "condition": ""})
        assert result == set()


# ========================================================================
# compute_matches — numeric and string == / !=
# ========================================================================

class TestComputeMatchesEquality:
    def test_string_equality(self, sample_df: pd.DataFrame):
        filt = {"column": "PlantType", "condition": "==", "value": "tree"}
        result = compute_matches(sample_df, filt)
        assert result == {"s2", "s5"}

    def test_string_inequality(self, sample_df: pd.DataFrame):
        filt = {"column": "PlantType", "condition": "!=", "value": "tree"}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s3", "s4", "s6"}

    def test_numeric_equality(self, sample_df: pd.DataFrame):
        filt = {"column": "HarvestYear", "condition": "==", "value": "2021"}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s3"}


# ========================================================================
# format_summary
# ========================================================================

class TestFormatSummary:
    def test_none_returns_all_samples(self):
        assert format_summary(None) == "All samples"

    def test_empty_dict_returns_all_samples(self):
        assert format_summary({}) == "All samples"

    def test_in_summary(self):
        filt = {"column": "PlantType", "condition": "in", "values": ["grass", "sedge"]}
        assert format_summary(filt) == "PlantType in [grass, sedge]"

    def test_equality_summary(self):
        filt = {"column": "Species", "condition": "==", "value": "tree"}
        assert format_summary(filt) == "Species == tree"

    def test_between_summary(self):
        filt = {"column": "HarvestYear", "condition": "between", "value": "2021", "value2": "2023"}
        assert format_summary(filt) == "HarvestYear between 2021 and 2023"

    def test_contains_summary(self):
        filt = {"column": "Collector", "condition": "contains", "value": "smith"}
        assert format_summary(filt) == "Collector contains smith"

    def test_has_value_summary(self):
        filt = {"column": "Score", "condition": "has value"}
        assert format_summary(filt) == "Score has value"


# ========================================================================
# is_categorical_column
# ========================================================================

class TestIsCategoricalColumn:
    def test_object_dtype_is_categorical(self):
        s = pd.Series(["a", "b", "c"])
        assert is_categorical_column(s) is True

    def test_string_dtype_is_categorical(self):
        s = pd.Series(["grass", "tree"], dtype=str)
        assert is_categorical_column(s) is True

    def test_numeric_few_uniques_is_categorical(self):
        s = pd.Series([1, 2, 3, 1, 2, 3])
        assert is_categorical_column(s) is True

    def test_numeric_many_uniques_is_not_categorical(self):
        s = pd.Series(range(100))
        assert is_categorical_column(s) is False

    def test_custom_max_unique(self):
        s = pd.Series(range(15))
        assert is_categorical_column(s, max_unique=20) is True
        assert is_categorical_column(s, max_unique=10) is False


# ========================================================================
# build_training_subset_metadata
# ========================================================================

class TestBuildTrainingSubsetMetadata:
    def test_inactive_state(self):
        meta = build_training_subset_metadata(active=False, filter_def=None, n_samples=None)
        assert meta["analysis_subset_active"] is False
        assert meta["analysis_subset_filter"] is None
        assert meta["analysis_subset_summary"] == "All samples"
        assert meta["analysis_subset_n_samples"] is None

    def test_active_state(self):
        filt = {"column": "PlantType", "condition": "in", "values": ["grass"]}
        meta = build_training_subset_metadata(active=True, filter_def=filt, n_samples=3)
        assert meta["analysis_subset_active"] is True
        assert meta["analysis_subset_filter"] == filt
        assert meta["analysis_subset_summary"] == "PlantType in [grass]"
        assert meta["analysis_subset_n_samples"] == 3

    def test_inactive_ignores_n_samples(self):
        meta = build_training_subset_metadata(active=False, filter_def=None, n_samples=50)
        assert meta["analysis_subset_n_samples"] is None

    def test_filter_is_copied(self):
        filt = {"column": "X", "condition": "==", "value": "1"}
        meta = build_training_subset_metadata(active=True, filter_def=filt, n_samples=1)
        assert meta["analysis_subset_filter"] is not filt


# ========================================================================
# check_one_class_inlier_guard
# ========================================================================

class TestCheckOneClassInlierGuard:
    def test_inliers_present_returns_none(self):
        y = pd.Series(["grass", "grass", "tree", "sedge"], index=["s1", "s2", "s3", "s4"])
        assert check_one_class_inlier_guard(y, "grass") is None

    def test_zero_inliers_returns_message(self):
        y = pd.Series(["tree", "tree", "sedge"], index=["s1", "s2", "s3"])
        msg = check_one_class_inlier_guard(y, "grass")
        assert msg is not None
        assert "grass" in msg
        assert "excludes all samples" in msg

    def test_numeric_label_coerced(self):
        y = pd.Series([1, 1, 2, 3], index=["s1", "s2", "s3", "s4"])
        msg = check_one_class_inlier_guard(y, 1)
        assert msg is None

    def test_numeric_label_zero_inliers(self):
        y = pd.Series([2, 2, 3], index=["s1", "s2", "s3"])
        msg = check_one_class_inlier_guard(y, 1)
        assert msg is not None


# ========================================================================
# classify_column_type (T3a)
# ========================================================================

class TestClassifyColumnType:
    def test_native_numeric_dtype(self):
        s = pd.Series([1, 2, 3])
        assert classify_column_type(s) == "numeric"

    def test_all_string_numeric(self):
        s = pd.Series(["1", "2", "3"])
        assert classify_column_type(s) == "numeric"

    def test_percentages(self):
        s = pd.Series(["12.5%", "7.1%", "3.0%"])
        assert classify_column_type(s) == "numeric"

    def test_mixed_numeric_garbage(self):
        s = pd.Series(["2021", "2022", "unknown"])
        assert classify_column_type(s) == "categorical"

    def test_plain_text(self):
        s = pd.Series(["grass", "tree", "sedge"])
        assert classify_column_type(s) == "categorical"

    def test_empty_all_null(self):
        s = pd.Series([None, np.nan])
        assert classify_column_type(s) == "categorical"

    def test_empty_series(self):
        s = pd.Series([], dtype=object)
        assert classify_column_type(s) == "categorical"

    def test_threshold_override(self):
        s = pd.Series(["2021", "2022", "unknown"])
        assert classify_column_type(s, numeric_threshold=0.5) == "numeric"

    def test_year_like_object_column(self):
        s = pd.Series(["2021", "2022", "2023"], dtype=object)
        assert classify_column_type(s) == "numeric"

    def test_whitespace_stripped_before_check(self):
        s = pd.Series(["  1.5  ", " 2.0 ", " 3.5 "])
        assert classify_column_type(s) == "numeric"


# ========================================================================
# get_unique_non_null_values (T3b)
# ========================================================================

class TestGetUniqueNonNullValues:
    def test_basic_dedup(self):
        s = pd.Series(["grass", "tree", "grass", "sedge"])
        result = get_unique_non_null_values(s)
        assert set(result) == {"grass", "tree", "sedge"}

    def test_nan_excluded(self):
        s = pd.Series(["a", "b", np.nan, "a"])
        result = get_unique_non_null_values(s)
        assert None not in result
        assert set(result) == {"a", "b"}

    def test_numeric_sort_when_all_coercible(self):
        s = pd.Series(["999", "2021", "7"])
        result = get_unique_non_null_values(s)
        assert result == ["7", "999", "2021"]

    def test_case_insensitive_string_sort(self):
        s = pd.Series(["banana", "Apple", "cherry", "apple"])
        result = get_unique_non_null_values(s)
        assert result == ["Apple", "apple", "banana", "cherry"]

    def test_max_count_cap(self):
        s = pd.Series([str(i) for i in range(100)])
        result = get_unique_non_null_values(s, max_count=10)
        assert len(result) == 10

    def test_empty_series(self):
        s = pd.Series([], dtype=object)
        result = get_unique_non_null_values(s)
        assert result == []

    def test_all_nan(self):
        s = pd.Series([np.nan, None])
        result = get_unique_non_null_values(s)
        assert result == []

    def test_preserves_original_case(self):
        s = pd.Series(["Alice", "alice", "Bob"])
        result = get_unique_non_null_values(s)
        assert "Alice" in result
        assert "alice" in result


# ========================================================================
# build_filter_dict (T3c)
# ========================================================================

class TestBuildFilterDict:
    def test_numeric_eq(self):
        result = build_filter_dict("Carbon", "numeric", "==", {"value": "7.0"})
        assert result == {"column": "Carbon", "condition": "==", "value": "7.0", "value2": ""}

    def test_numeric_ne(self):
        result = build_filter_dict("Carbon", "numeric", "!=", {"value": "7.0"})
        assert result == {"column": "Carbon", "condition": "!=", "value": "7.0", "value2": ""}

    def test_numeric_lt(self):
        result = build_filter_dict("Carbon", "numeric", "<", {"value": "7.0"})
        assert result == {"column": "Carbon", "condition": "<", "value": "7.0", "value2": ""}

    def test_numeric_le(self):
        result = build_filter_dict("Carbon", "numeric", "<=", {"value": "7.0"})
        assert result == {"column": "Carbon", "condition": "<=", "value": "7.0", "value2": ""}

    def test_numeric_gt(self):
        result = build_filter_dict("Carbon", "numeric", ">", {"value": "7.0"})
        assert result == {"column": "Carbon", "condition": ">", "value": "7.0", "value2": ""}

    def test_numeric_ge(self):
        result = build_filter_dict("Carbon", "numeric", ">=", {"value": "7.0"})
        assert result == {"column": "Carbon", "condition": ">=", "value": "7.0", "value2": ""}

    def test_numeric_between(self):
        result = build_filter_dict("Year", "numeric", "between", {"value": "2021", "value2": "2023"})
        assert result == {"column": "Year", "condition": "between", "value": "2021", "value2": "2023"}

    def test_numeric_has_value(self):
        result = build_filter_dict("Carbon", "numeric", "has value", {})
        assert result == {"column": "Carbon", "condition": "has value"}

    def test_categorical_eq(self):
        result = build_filter_dict("PlantType", "categorical", "==", {"value": "grass"})
        assert result == {"column": "PlantType", "condition": "==", "value": "grass", "value2": ""}

    def test_categorical_ne(self):
        result = build_filter_dict("PlantType", "categorical", "!=", {"value": "tree"})
        assert result == {"column": "PlantType", "condition": "!=", "value": "tree", "value2": ""}

    def test_categorical_in(self):
        result = build_filter_dict("PlantType", "categorical", "in", {"values": ["grass", "sedge"]})
        assert result == {"column": "PlantType", "condition": "in", "values": ["grass", "sedge"]}

    def test_categorical_contains(self):
        result = build_filter_dict("Collector", "categorical", "contains", {"value": "ali"})
        assert result == {"column": "Collector", "condition": "contains", "value": "ali", "value2": ""}

    def test_categorical_has_value(self):
        result = build_filter_dict("Note", "categorical", "has value", {})
        assert result == {"column": "Note", "condition": "has value"}

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            build_filter_dict("Col", "numeric", "xor", {"value": "1"})


# ========================================================================
# compute_matches — <, <=, >, >= operators (T5 coverage gap)
# ========================================================================

class TestComputeMatchesComparison:
    def test_less_than(self, sample_df: pd.DataFrame):
        filt = {"column": "Score", "condition": "<", "value": "2.0"}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s3"}

    def test_less_than_or_equal(self, sample_df: pd.DataFrame):
        filt = {"column": "Score", "condition": "<=", "value": "2.0"}
        result = compute_matches(sample_df, filt)
        assert result == {"s1", "s2", "s3"}

    def test_greater_than(self, sample_df: pd.DataFrame):
        filt = {"column": "Score", "condition": ">", "value": "2.0"}
        result = compute_matches(sample_df, filt)
        assert result == {"s4", "s6"}

    def test_greater_than_or_equal(self, sample_df: pd.DataFrame):
        filt = {"column": "Score", "condition": ">=", "value": "2.0"}
        result = compute_matches(sample_df, filt)
        assert result == {"s2", "s4", "s6"}


# ========================================================================
# build_filter_dict — fail-fast on missing inputs (V1.1 review)
# ========================================================================

class TestBuildFilterDictFailFast:
    def test_between_missing_value2_raises(self):
        with pytest.raises(ValueError, match="'between' requires both"):
            build_filter_dict("Year", "numeric", "between", {"value": "2021"})

    def test_between_empty_value2_raises(self):
        with pytest.raises(ValueError, match="'between' requires both"):
            build_filter_dict("Year", "numeric", "between", {"value": "2021", "value2": ""})

    def test_between_whitespace_value2_raises(self):
        with pytest.raises(ValueError, match="'between' requires both"):
            build_filter_dict("Year", "numeric", "between", {"value": "2021", "value2": "  "})

    def test_in_missing_values_raises(self):
        with pytest.raises(ValueError, match="'in' requires a non-empty"):
            build_filter_dict("PlantType", "categorical", "in", {})

    def test_in_none_values_raises(self):
        with pytest.raises(ValueError, match="'in' requires a non-empty"):
            build_filter_dict("PlantType", "categorical", "in", {"values": None})

    def test_in_empty_list_raises(self):
        with pytest.raises(ValueError, match="'in' requires a non-empty"):
            build_filter_dict("PlantType", "categorical", "in", {"values": []})

    def test_eq_missing_value_raises(self):
        with pytest.raises(ValueError, match="'==' requires 'value'"):
            build_filter_dict("Col", "numeric", "==", {})

    def test_lt_empty_value_raises(self):
        with pytest.raises(ValueError, match="'<' requires 'value'"):
            build_filter_dict("Col", "numeric", "<", {"value": ""})

    def test_contains_whitespace_value_raises(self):
        with pytest.raises(ValueError, match="'contains' requires 'value'"):
            build_filter_dict("Col", "categorical", "contains", {"value": "   "})

    def test_has_value_never_raises(self):
        result = build_filter_dict("Col", "numeric", "has value", {})
        assert result == {"column": "Col", "condition": "has value"}


# ========================================================================
# classify_column_type — 0.9 threshold boundary (V1.1 review)
# ========================================================================

class TestClassifyColumnTypeThreshold:
    def test_below_threshold_is_categorical(self):
        vals = ["1", "2", "3", "4", "5", "6", "7", "8", "bad", "bad"]
        assert classify_column_type(pd.Series(vals)) == "categorical"

    def test_at_threshold_is_numeric(self):
        vals = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "bad"]
        assert classify_column_type(pd.Series(vals)) == "numeric"

    def test_list_input_coerced(self):
        assert classify_column_type([1, 2, 3]) == "numeric"

    def test_ndarray_input_coerced(self):
        assert classify_column_type(np.array([1.0, 2.0, 3.0])) == "numeric"


# ========================================================================
# get_unique_non_null_values — sort-then-cap (V1.1 review)
# ========================================================================

class TestGetUniqueNonNullValuesSortThenCap:
    def test_sort_then_cap_numeric(self):
        vals = [
            "50", "2", "100", "7", "9", "20",
            "1", "15", "3", "8", "4", "10",
        ]
        result = get_unique_non_null_values(pd.Series(vals), max_count=5)
        expected = ["1", "2", "3", "4", "7"]
        assert result == expected
