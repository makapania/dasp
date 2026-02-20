"""Comprehensive tests for spectral_predict.similarity_metrics module."""

import numpy as np
import pytest

from spectral_predict.similarity_metrics import (
    METRICS,
    compute_similarity,
    cosine_similarity,
    euclidean_distance,
    euclidean_to_similarity,
    first_derivative_correlation,
    hit_quality_index,
    sam_to_similarity,
    second_derivative_correlation,
    sid_to_similarity,
    spectral_angle_mapper,
    spectral_information_divergence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identical_spectra():
    """Pair of identical spectra (50-point sine curve)."""
    x = np.linspace(0, 2 * np.pi, 50)
    s = np.sin(x) + 2.0  # shift up so all values are positive
    return s.copy(), s.copy()


@pytest.fixture
def orthogonal_spectra():
    """Two spectra that are as uncorrelated as possible."""
    n = 100
    a = np.zeros(n)
    b = np.zeros(n)
    a[:50] = 1.0
    b[50:] = 1.0
    return a, b


@pytest.fixture
def known_distance_pair():
    """Spectra pair with a known Euclidean distance of 5.0."""
    a = np.array([3.0, 0.0])
    b = np.array([0.0, 4.0])
    return a, b  # distance = sqrt(9 + 16) = 5.0


@pytest.fixture
def long_smooth_spectra():
    """Long, smooth spectra suitable for derivative tests (200 points)."""
    x = np.linspace(0, 4 * np.pi, 200)
    a = np.sin(x) + 2.0
    b = np.sin(x + 0.1) + 2.0  # slightly shifted version
    return a, b


# ---------------------------------------------------------------------------
# hit_quality_index tests
# ---------------------------------------------------------------------------

def test_hqi_identical_spectra(identical_spectra):
    """HQI of identical spectra should be 1.0."""
    a, b = identical_spectra
    assert pytest.approx(hit_quality_index(a, b), abs=1e-10) == 1.0


def test_hqi_normal_case():
    """HQI of correlated but non-identical spectra should be between 0 and 1."""
    rng = np.random.default_rng(0)
    a = rng.random(50)
    b = a + rng.normal(0, 0.1, 50)
    hqi = hit_quality_index(a, b)
    assert 0.0 < hqi < 1.0


def test_hqi_constant_spectrum_returns_zero():
    """A constant spectrum (zero variance) should yield HQI of 0."""
    a = np.ones(20)
    b = np.arange(20, dtype=float)
    assert hit_quality_index(a, b) == 0.0


def test_hqi_length_mismatch_raises():
    """Spectra of different lengths should raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        hit_quality_index(np.array([1, 2, 3]), np.array([1, 2]))


# ---------------------------------------------------------------------------
# spectral_angle_mapper tests
# ---------------------------------------------------------------------------

def test_sam_identical_vectors():
    """SAM between identical vectors should be 0."""
    v = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(spectral_angle_mapper(v, v), abs=1e-10) == 0.0


def test_sam_orthogonal_vectors():
    """SAM between orthogonal vectors should be pi/2."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert pytest.approx(spectral_angle_mapper(a, b), abs=1e-10) == np.pi / 2


def test_sam_zero_vector_returns_pi_over_2():
    """Zero vector should yield maximum dissimilarity (pi/2)."""
    a = np.zeros(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spectral_angle_mapper(a, b) == np.pi / 2


def test_sam_scaled_vectors_same_angle():
    """Scaling a vector should not change the angle."""
    v = np.array([1.0, 2.0, 3.0])
    angle_1x = spectral_angle_mapper(v, v * 5.0)
    assert pytest.approx(angle_1x, abs=1e-10) == 0.0


def test_sam_length_mismatch_raises():
    """Spectra of different lengths should raise ValueError."""
    with pytest.raises(ValueError):
        spectral_angle_mapper(np.array([1, 2]), np.array([1]))


# ---------------------------------------------------------------------------
# euclidean_distance tests
# ---------------------------------------------------------------------------

def test_euclidean_known_distance(known_distance_pair):
    """Euclidean distance of (3,0)-(0,4) triangle should be 5.0."""
    a, b = known_distance_pair
    assert pytest.approx(euclidean_distance(a, b)) == 5.0


def test_euclidean_identical_spectra_zero(identical_spectra):
    """Distance between identical spectra should be 0."""
    a, b = identical_spectra
    assert pytest.approx(euclidean_distance(a, b), abs=1e-12) == 0.0


def test_euclidean_length_mismatch_raises():
    """Spectra of different lengths should raise ValueError."""
    with pytest.raises(ValueError):
        euclidean_distance(np.array([1.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# cosine_similarity tests
# ---------------------------------------------------------------------------

def test_cosine_identical_direction():
    """Cosine similarity of vectors in the same direction should be 1."""
    v = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(cosine_similarity(v, v * 10)) == 1.0


def test_cosine_orthogonal():
    """Cosine similarity of orthogonal vectors should be 0."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert pytest.approx(cosine_similarity(a, b), abs=1e-10) == 0.0


def test_cosine_opposite_direction():
    """Cosine similarity of opposite vectors should be -1."""
    v = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(cosine_similarity(v, -v)) == -1.0


def test_cosine_zero_vector():
    """Cosine similarity with a zero vector should return 0."""
    assert cosine_similarity(np.zeros(5), np.ones(5)) == 0.0


# ---------------------------------------------------------------------------
# derivative correlation tests
# ---------------------------------------------------------------------------

def test_first_derivative_correlation_identical(identical_spectra):
    """First derivative correlation of identical spectra should be 1.0."""
    a, b = identical_spectra
    result = first_derivative_correlation(a, b)
    assert pytest.approx(result, abs=0.01) == 1.0


def test_first_derivative_correlation_similar(long_smooth_spectra):
    """Slightly shifted smooth spectra should still have high first-derivative correlation."""
    a, b = long_smooth_spectra
    result = first_derivative_correlation(a, b)
    assert result > 0.9


def test_first_derivative_correlation_short_spectrum():
    """Very short spectrum should not crash and should return a valid float."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.1, 2.1, 3.1, 4.1])
    result = first_derivative_correlation(a, b)
    assert isinstance(result, float)


def test_second_derivative_correlation_identical(identical_spectra):
    """Second derivative correlation of identical spectra should be 1.0."""
    a, b = identical_spectra
    result = second_derivative_correlation(a, b)
    assert pytest.approx(result, abs=0.01) == 1.0


def test_second_derivative_correlation_similar(long_smooth_spectra):
    """Slightly shifted smooth spectra should have high second-derivative correlation."""
    a, b = long_smooth_spectra
    result = second_derivative_correlation(a, b)
    assert result > 0.85


# ---------------------------------------------------------------------------
# spectral_information_divergence tests
# ---------------------------------------------------------------------------

def test_sid_identical_spectra():
    """SID of identical positive spectra should be 0 (or very close)."""
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pytest.approx(spectral_information_divergence(s, s), abs=1e-10) == 0.0


def test_sid_different_spectra_positive():
    """SID of different spectra should be a positive value."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    sid = spectral_information_divergence(a, b)
    assert sid > 0


def test_sid_length_mismatch_raises():
    """SID should raise ValueError for mismatched lengths."""
    with pytest.raises(ValueError):
        spectral_information_divergence(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Conversion helper tests
# ---------------------------------------------------------------------------

def test_sam_to_similarity_zero_angle():
    """Zero angle (identical) should map to similarity 1.0."""
    assert pytest.approx(sam_to_similarity(0.0)) == 1.0


def test_sam_to_similarity_pi_over_2():
    """pi/2 angle (orthogonal) should map to similarity 0.0."""
    assert pytest.approx(sam_to_similarity(np.pi / 2)) == 0.0


def test_euclidean_to_similarity_zero_distance():
    """Zero distance should map to similarity 1.0."""
    assert pytest.approx(euclidean_to_similarity(0.0)) == 1.0


def test_euclidean_to_similarity_large_distance():
    """Large distance should yield a similarity near 0."""
    sim = euclidean_to_similarity(1e6)
    assert sim < 0.01


def test_euclidean_to_similarity_scale_parameter():
    """Scale parameter should control the decay rate."""
    sim_default = euclidean_to_similarity(1.0, scale=1.0)
    sim_wide = euclidean_to_similarity(1.0, scale=10.0)
    assert sim_wide > sim_default  # wider scale decays slower


def test_sid_to_similarity_zero():
    """SID of 0 should map to similarity 1.0."""
    assert pytest.approx(sid_to_similarity(0.0)) == 1.0


def test_sid_to_similarity_large():
    """Large SID should yield similarity near 0."""
    assert sid_to_similarity(1e6) < 0.01


# ---------------------------------------------------------------------------
# compute_similarity dispatch tests
# ---------------------------------------------------------------------------

def test_compute_similarity_hqi_dispatch(identical_spectra):
    """compute_similarity with metric='hqi' should return 1.0 for identical spectra."""
    a, b = identical_spectra
    assert pytest.approx(compute_similarity(a, b, metric="hqi"), abs=0.01) == 1.0


def test_compute_similarity_sam_normalized(identical_spectra):
    """Normalized SAM of identical spectra should be 1.0."""
    a, b = identical_spectra
    sim = compute_similarity(a, b, metric="sam", normalize=True)
    assert pytest.approx(sim, abs=0.01) == 1.0


def test_compute_similarity_euclidean_normalized(identical_spectra):
    """Normalized Euclidean of identical spectra should be 1.0."""
    a, b = identical_spectra
    sim = compute_similarity(a, b, metric="euclidean", normalize=True)
    assert pytest.approx(sim, abs=0.01) == 1.0


def test_compute_similarity_unknown_metric_raises():
    """Unknown metric name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_similarity(np.ones(5), np.ones(5), metric="bogus")


def test_compute_similarity_cosine_dispatch():
    """compute_similarity with metric='cosine' should match cosine_similarity."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    expected = cosine_similarity(a, b)
    actual = compute_similarity(a, b, metric="cosine", normalize=True)
    assert pytest.approx(actual) == expected


def test_metrics_registry_keys():
    """METRICS dict should contain all documented metric keys."""
    expected_keys = {"hqi", "sam", "euclidean", "cosine", "deriv1_corr", "deriv2_corr", "sid"}
    assert set(METRICS.keys()) == expected_keys
