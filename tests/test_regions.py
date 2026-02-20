"""Comprehensive tests for spectral_predict.regions module."""

import numpy as np
import pandas as pd
import pytest

from spectral_predict.regions import (
    compute_region_correlations,
    create_region_subsets,
    format_region_report,
    get_region_variable_indices,
    get_top_regions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_spectral_data():
    """Synthetic spectral data with a known correlated region.

    100 samples, 200 wavelengths (1000-1199 nm).
    Target y correlates strongly with the region around 1050-1100.
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    wavelengths = np.arange(1000, 1200, dtype=float)
    n_features = len(wavelengths)

    X = rng.normal(0, 1, (n_samples, n_features))

    # Inject strong correlation in the 1050-1100 nm region
    signal = rng.normal(0, 1, n_samples)
    corr_start = 50  # index for wavelength 1050
    corr_end = 100   # index for wavelength 1100
    for j in range(corr_start, corr_end):
        X[:, j] = signal + rng.normal(0, 0.1, n_samples)

    y = signal + rng.normal(0, 0.05, n_samples)
    return X, y, wavelengths


@pytest.fixture
def simple_regions():
    """Pre-built list of region dicts for testing get_top_regions and helpers."""
    return [
        {
            "start": 1000.0,
            "end": 1050.0,
            "indices": np.array([0, 1, 2, 3, 4]),
            "mean_corr": 0.30,
            "max_corr": 0.45,
            "n_features": 5,
        },
        {
            "start": 1025.0,
            "end": 1075.0,
            "indices": np.array([5, 6, 7, 8]),
            "mean_corr": 0.80,
            "max_corr": 0.92,
            "n_features": 4,
        },
        {
            "start": 1050.0,
            "end": 1100.0,
            "indices": np.array([8, 9, 10, 11, 12]),
            "mean_corr": 0.75,
            "max_corr": 0.88,
            "n_features": 5,
        },
        {
            "start": 1075.0,
            "end": 1125.0,
            "indices": np.array([13, 14]),
            "mean_corr": 0.20,
            "max_corr": 0.25,
            "n_features": 2,
        },
        {
            "start": 1100.0,
            "end": 1150.0,
            "indices": np.array([15, 16, 17]),
            "mean_corr": 0.50,
            "max_corr": 0.60,
            "n_features": 3,
        },
    ]


# ---------------------------------------------------------------------------
# compute_region_correlations tests
# ---------------------------------------------------------------------------

def test_compute_region_correlations_returns_list(synthetic_spectral_data):
    """Should return a non-empty list of dicts."""
    X, y, wavelengths = synthetic_spectral_data
    regions = compute_region_correlations(X, y, wavelengths, region_size=50, overlap=25)
    assert isinstance(regions, list)
    assert len(regions) > 0


def test_compute_region_correlations_dict_keys(synthetic_spectral_data):
    """Each region dict should have the documented keys."""
    X, y, wavelengths = synthetic_spectral_data
    regions = compute_region_correlations(X, y, wavelengths)
    expected_keys = {"start", "end", "indices", "mean_corr", "max_corr", "n_features"}
    for r in regions:
        assert set(r.keys()) == expected_keys


def test_compute_region_correlations_high_corr_region(synthetic_spectral_data):
    """The injected correlated region (1050-1100) should have high correlation."""
    X, y, wavelengths = synthetic_spectral_data
    regions = compute_region_correlations(X, y, wavelengths, region_size=50, overlap=25)

    # Find the region that covers 1050-1100
    target_region = None
    for r in regions:
        if r["start"] >= 1050 and r["end"] <= 1110:
            if target_region is None or r["mean_corr"] > target_region["mean_corr"]:
                target_region = r

    assert target_region is not None
    assert target_region["mean_corr"] > 0.5


def test_compute_region_correlations_small_region_size(synthetic_spectral_data):
    """Smaller region_size should produce more regions."""
    X, y, wavelengths = synthetic_spectral_data
    large = compute_region_correlations(X, y, wavelengths, region_size=100, overlap=0)
    small = compute_region_correlations(X, y, wavelengths, region_size=25, overlap=0)
    assert len(small) >= len(large)


# ---------------------------------------------------------------------------
# get_top_regions tests
# ---------------------------------------------------------------------------

def test_get_top_regions_ordering(simple_regions):
    """Top regions should be sorted by mean_corr descending."""
    top = get_top_regions(simple_regions, n_top=3, criterion="mean_corr")
    assert len(top) == 3
    for i in range(len(top) - 1):
        assert top[i]["mean_corr"] >= top[i + 1]["mean_corr"]


def test_get_top_regions_by_max_corr(simple_regions):
    """Sorting by max_corr should rank the second region first."""
    top = get_top_regions(simple_regions, n_top=1, criterion="max_corr")
    assert top[0]["max_corr"] == 0.92


def test_get_top_regions_count_capped(simple_regions):
    """Requesting more regions than available should return all of them."""
    top = get_top_regions(simple_regions, n_top=100)
    assert len(top) == len(simple_regions)


# ---------------------------------------------------------------------------
# get_region_variable_indices tests
# ---------------------------------------------------------------------------

def test_get_region_variable_indices_combined(simple_regions):
    """Combined mode should return unique, sorted indices from all regions."""
    indices = get_region_variable_indices(simple_regions[:2], return_combined=True)
    assert isinstance(indices, np.ndarray)
    # Regions 0 and 1 have indices [0..4] and [5..8]
    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(indices, expected)


def test_get_region_variable_indices_combined_deduplicates(simple_regions):
    """Overlapping indices across regions should be deduplicated."""
    # Regions 1 and 2 share index 8
    indices = get_region_variable_indices(simple_regions[1:3], return_combined=True)
    assert len(indices) == len(np.unique(indices))
    assert 8 in indices


def test_get_region_variable_indices_separate(simple_regions):
    """Non-combined mode should return a list of arrays, one per region."""
    separate = get_region_variable_indices(simple_regions[:2], return_combined=False)
    assert isinstance(separate, list)
    assert len(separate) == 2
    np.testing.assert_array_equal(separate[0], simple_regions[0]["indices"])


# ---------------------------------------------------------------------------
# create_region_subsets tests
# ---------------------------------------------------------------------------

def test_create_region_subsets_returns_list(synthetic_spectral_data):
    """Should return a list of subset dicts."""
    X, y, wavelengths = synthetic_spectral_data
    subsets = create_region_subsets(X, y, wavelengths, n_top_regions=3)
    assert isinstance(subsets, list)
    assert len(subsets) > 0


def test_create_region_subsets_dict_keys(synthetic_spectral_data):
    """Each subset should have indices, tag, and description."""
    X, y, wavelengths = synthetic_spectral_data
    subsets = create_region_subsets(X, y, wavelengths, n_top_regions=3)
    for s in subsets:
        assert "indices" in s
        assert "tag" in s
        assert "description" in s


def test_create_region_subsets_indices_within_bounds(synthetic_spectral_data):
    """All indices in subsets should be valid column indices for X."""
    X, y, wavelengths = synthetic_spectral_data
    subsets = create_region_subsets(X, y, wavelengths, n_top_regions=5)
    n_features = X.shape[1]
    for s in subsets:
        assert np.all(s["indices"] >= 0)
        assert np.all(s["indices"] < n_features)


# ---------------------------------------------------------------------------
# format_region_report tests
# ---------------------------------------------------------------------------

def test_format_region_report_is_string(simple_regions):
    """Report should be a non-empty string."""
    wavelengths = np.arange(1000, 1200, dtype=float)
    report = format_region_report(simple_regions, wavelengths, n_top=3)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_region_report_contains_header(simple_regions):
    """Report should contain the standard header line."""
    wavelengths = np.arange(1000, 1200, dtype=float)
    report = format_region_report(simple_regions, wavelengths, n_top=3)
    assert "Top Spectral Regions" in report


def test_format_region_report_contains_region_ranges(simple_regions):
    """Report should include the wavelength range strings for top regions."""
    wavelengths = np.arange(1000, 1200, dtype=float)
    report = format_region_report(simple_regions, wavelengths, n_top=3)
    # The top 3 by mean_corr are regions with starts 1025, 1050, 1100
    assert "1025-1075" in report
    assert "1050-1100" in report


def test_format_region_report_n_top_limits_rows(simple_regions):
    """Requesting n_top=2 should show only 2 ranked rows."""
    wavelengths = np.arange(1000, 1200, dtype=float)
    report = format_region_report(simple_regions, wavelengths, n_top=2)
    # Count lines that start with a rank number (1 or 2, left-aligned)
    data_lines = [line for line in report.splitlines() if line.strip()[:1].isdigit()]
    assert len(data_lines) == 2
