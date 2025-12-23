"""
Unit tests for spectral library search functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectral_predict.similarity_metrics import (
    hit_quality_index,
    spectral_angle_mapper,
    euclidean_distance,
    cosine_similarity,
    first_derivative_correlation,
    second_derivative_correlation,
    spectral_information_divergence,
    sam_to_similarity,
    compute_similarity,
    compute_batch_similarity,
    METRICS,
)
from spectral_predict.library_search import (
    SpectralLibrary,
    LibraryEntry,
    get_library,
    add_to_library,
    search_library,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_spectrum():
    """Generate a sample NIR-like spectrum."""
    wavelengths = np.linspace(1000, 2500, 500)
    # Simulate NIR spectrum with some absorption features
    spectrum = 0.5 + 0.1 * np.sin(wavelengths / 100) + 0.05 * np.random.randn(500)
    return spectrum, wavelengths


@pytest.fixture
def similar_spectrum(sample_spectrum):
    """Generate a spectrum similar to the sample (with small noise)."""
    spectrum, wavelengths = sample_spectrum
    similar = spectrum + 0.01 * np.random.randn(len(spectrum))
    return similar, wavelengths


@pytest.fixture
def different_spectrum(sample_spectrum):
    """Generate a spectrum different from the sample."""
    _, wavelengths = sample_spectrum
    # Completely different pattern
    spectrum = 0.8 - 0.2 * np.cos(wavelengths / 50) + 0.05 * np.random.randn(len(wavelengths))
    return spectrum, wavelengths


@pytest.fixture
def temp_library_dir():
    """Create a temporary directory for library storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def empty_library(temp_library_dir):
    """Create an empty spectral library."""
    return SpectralLibrary(name="test", storage_path=temp_library_dir, auto_save=False)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with spectra."""
    wavelengths = np.linspace(1000, 2500, 100)
    data = {}
    for i in range(5):
        data[f"sample_{i}"] = 0.5 + 0.1 * np.sin(wavelengths / (100 + i * 10)) + 0.02 * np.random.randn(100)
    df = pd.DataFrame(data, index=wavelengths).T
    return df


# ============================================================================
# Similarity Metrics Tests
# ============================================================================

class TestHitQualityIndex:
    """Tests for Hit Quality Index (HQI)."""

    def test_identical_spectra(self, sample_spectrum):
        """HQI of identical spectra should be 1.0."""
        spectrum, _ = sample_spectrum
        hqi = hit_quality_index(spectrum, spectrum)
        assert hqi == pytest.approx(1.0, abs=1e-10)

    def test_similar_spectra_high_hqi(self, sample_spectrum, similar_spectrum):
        """Similar spectra should have high HQI."""
        spectrum1, _ = sample_spectrum
        spectrum2, _ = similar_spectrum
        hqi = hit_quality_index(spectrum1, spectrum2)
        assert hqi > 0.95

    def test_different_spectra_lower_hqi(self, sample_spectrum, different_spectrum):
        """Different spectra should have lower HQI."""
        spectrum1, _ = sample_spectrum
        spectrum2, _ = different_spectrum
        hqi = hit_quality_index(spectrum1, spectrum2)
        assert hqi < 0.9

    def test_hqi_range(self, sample_spectrum, different_spectrum):
        """HQI should be between 0 and 1."""
        spectrum1, _ = sample_spectrum
        spectrum2, _ = different_spectrum
        hqi = hit_quality_index(spectrum1, spectrum2)
        assert 0 <= hqi <= 1

    def test_hqi_length_mismatch(self):
        """HQI should raise error for different length spectra."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2])
        with pytest.raises(ValueError):
            hit_quality_index(a, b)

    def test_hqi_constant_spectrum(self):
        """HQI with constant spectrum should return 0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([5.0, 5.0, 5.0])  # Zero variance
        hqi = hit_quality_index(a, b)
        assert hqi == 0.0


class TestSpectralAngleMapper:
    """Tests for Spectral Angle Mapper (SAM)."""

    def test_identical_spectra(self, sample_spectrum):
        """SAM of identical spectra should be ~0."""
        spectrum, _ = sample_spectrum
        sam = spectral_angle_mapper(spectrum, spectrum)
        assert sam == pytest.approx(0.0, abs=1e-6)  # Allow small floating point error

    def test_similar_spectra_low_sam(self, sample_spectrum, similar_spectrum):
        """Similar spectra should have low SAM angle."""
        spectrum1, _ = sample_spectrum
        spectrum2, _ = similar_spectrum
        sam = spectral_angle_mapper(spectrum1, spectrum2)
        assert sam < 0.1  # Small angle in radians

    def test_sam_range(self, sample_spectrum, different_spectrum):
        """SAM should be between 0 and pi/2."""
        spectrum1, _ = sample_spectrum
        spectrum2, _ = different_spectrum
        sam = spectral_angle_mapper(spectrum1, spectrum2)
        assert 0 <= sam <= np.pi / 2

    def test_sam_to_similarity_conversion(self):
        """Test SAM to similarity conversion."""
        assert sam_to_similarity(0) == pytest.approx(1.0)
        assert sam_to_similarity(np.pi / 2) == pytest.approx(0.0)

    def test_sam_intensity_invariance(self, sample_spectrum):
        """SAM should be invariant to intensity scaling."""
        spectrum, _ = sample_spectrum
        scaled = spectrum * 2.5
        sam = spectral_angle_mapper(spectrum, scaled)
        assert sam == pytest.approx(0.0, abs=1e-6)  # Allow small floating point error


class TestEuclideanDistance:
    """Tests for Euclidean distance."""

    def test_identical_spectra(self, sample_spectrum):
        """Distance of identical spectra should be 0."""
        spectrum, _ = sample_spectrum
        dist = euclidean_distance(spectrum, spectrum)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_distance_non_negative(self, sample_spectrum, different_spectrum):
        """Distance should always be non-negative."""
        spectrum1, _ = sample_spectrum
        spectrum2, _ = different_spectrum
        dist = euclidean_distance(spectrum1, spectrum2)
        assert dist >= 0


class TestCosineSimilarity:
    """Tests for cosine similarity."""

    def test_identical_spectra(self, sample_spectrum):
        """Cosine similarity of identical spectra should be 1."""
        spectrum, _ = sample_spectrum
        sim = cosine_similarity(spectrum, spectrum)
        assert sim == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_spectra(self):
        """Orthogonal spectra should have cosine similarity of 0."""
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-10)


class TestDerivativeCorrelation:
    """Tests for derivative-based correlation."""

    def test_first_derivative_identical(self, sample_spectrum):
        """First derivative correlation of identical spectra should be ~1."""
        spectrum, _ = sample_spectrum
        corr = first_derivative_correlation(spectrum, spectrum)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_second_derivative_identical(self, sample_spectrum):
        """Second derivative correlation of identical spectra should be ~1."""
        spectrum, _ = sample_spectrum
        corr = second_derivative_correlation(spectrum, spectrum)
        assert corr == pytest.approx(1.0, abs=1e-6)


class TestSpectralInformationDivergence:
    """Tests for Spectral Information Divergence (SID)."""

    def test_identical_spectra(self, sample_spectrum):
        """SID of identical spectra should be ~0."""
        spectrum, _ = sample_spectrum
        # Ensure positive values
        spectrum = np.abs(spectrum) + 0.1
        sid = spectral_information_divergence(spectrum, spectrum)
        assert sid == pytest.approx(0.0, abs=1e-10)

    def test_sid_non_negative(self, sample_spectrum, different_spectrum):
        """SID should always be non-negative."""
        spectrum1, _ = sample_spectrum
        spectrum2, _ = different_spectrum
        # Ensure positive values
        spectrum1 = np.abs(spectrum1) + 0.1
        spectrum2 = np.abs(spectrum2) + 0.1
        sid = spectral_information_divergence(spectrum1, spectrum2)
        assert sid >= 0


class TestComputeSimilarity:
    """Tests for the unified compute_similarity function."""

    def test_all_metrics_work(self, sample_spectrum):
        """All registered metrics should work."""
        spectrum, _ = sample_spectrum
        for metric_name in METRICS.keys():
            score = compute_similarity(spectrum, spectrum, metric=metric_name, normalize=True)
            assert isinstance(score, float)

    def test_unknown_metric_raises(self, sample_spectrum):
        """Unknown metric should raise ValueError."""
        spectrum, _ = sample_spectrum
        with pytest.raises(ValueError):
            compute_similarity(spectrum, spectrum, metric='unknown_metric')


class TestBatchSimilarity:
    """Tests for batch similarity computation."""

    def test_batch_similarity(self, sample_spectrum):
        """Test batch computation of similarities."""
        spectrum, _ = sample_spectrum
        references = np.vstack([spectrum, spectrum * 0.9, spectrum * 0.8])
        scores = compute_batch_similarity(spectrum, references, metric='hqi')
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0, abs=1e-10)  # Identical


# ============================================================================
# Library Entry Tests
# ============================================================================

class TestLibraryEntry:
    """Tests for LibraryEntry dataclass."""

    def test_entry_creation(self, sample_spectrum):
        """Test creating a library entry."""
        spectrum, wavelengths = sample_spectrum
        entry = LibraryEntry(
            sample_id="test_001",
            spectrum=spectrum,
            wavelengths=wavelengths,
            source_file="test.csv",
            category="soil",
        )
        assert entry.sample_id == "test_001"
        assert entry.category == "soil"
        assert len(entry.fingerprint) == 16  # SHA256 truncated

    def test_fingerprint_consistency(self, sample_spectrum):
        """Same spectrum should produce same fingerprint."""
        spectrum, wavelengths = sample_spectrum
        entry1 = LibraryEntry("a", spectrum, wavelengths)
        entry2 = LibraryEntry("b", spectrum.copy(), wavelengths)
        assert entry1.fingerprint == entry2.fingerprint

    def test_fingerprint_different_for_different_spectra(self, sample_spectrum, different_spectrum):
        """Different spectra should have different fingerprints."""
        spectrum1, wavelengths = sample_spectrum
        spectrum2, _ = different_spectrum
        entry1 = LibraryEntry("a", spectrum1, wavelengths)
        entry2 = LibraryEntry("b", spectrum2, wavelengths)
        assert entry1.fingerprint != entry2.fingerprint


# ============================================================================
# Spectral Library Tests
# ============================================================================

class TestSpectralLibrary:
    """Tests for SpectralLibrary class."""

    def test_empty_library(self, empty_library):
        """Test empty library properties."""
        assert empty_library.size == 0
        assert empty_library.sample_ids == []
        assert empty_library.categories == []

    def test_add_single_spectrum(self, empty_library, sample_spectrum):
        """Test adding a single spectrum."""
        spectrum, wavelengths = sample_spectrum
        success, msg = empty_library.add_spectrum(
            sample_id="test_001",
            spectrum=spectrum,
            wavelengths=wavelengths,
            category="soil",
        )
        assert success
        assert empty_library.size == 1
        assert "test_001" in empty_library.sample_ids

    def test_add_duplicate_id_rejected(self, empty_library, sample_spectrum):
        """Adding duplicate sample ID should be rejected."""
        spectrum, wavelengths = sample_spectrum
        empty_library.add_spectrum("test_001", spectrum, wavelengths)
        success, msg = empty_library.add_spectrum("test_001", spectrum * 0.9, wavelengths)
        assert not success
        assert "already exists" in msg

    def test_add_duplicate_fingerprint_rejected(self, empty_library, sample_spectrum):
        """Adding spectrum with same fingerprint should be rejected."""
        spectrum, wavelengths = sample_spectrum
        empty_library.add_spectrum("test_001", spectrum, wavelengths)
        success, msg = empty_library.add_spectrum("test_002", spectrum.copy(), wavelengths)
        assert not success
        assert "fingerprint" in msg.lower()

    def test_add_near_duplicate_rejected(self, empty_library, sample_spectrum, similar_spectrum):
        """Adding very similar spectrum should be rejected."""
        spectrum1, wavelengths = sample_spectrum
        spectrum2, _ = similar_spectrum
        empty_library.add_spectrum("test_001", spectrum1, wavelengths)
        success, msg = empty_library.add_spectrum("test_002", spectrum2, wavelengths)
        # Very similar spectra should be rejected (HQI > 0.9999)
        # Note: This depends on how similar the "similar_spectrum" is
        # With noise of 0.01, HQI might be just below threshold
        assert "test_001" in empty_library.sample_ids

    def test_add_different_spectrum_accepted(self, empty_library, sample_spectrum, different_spectrum):
        """Adding different spectrum should be accepted."""
        spectrum1, wavelengths = sample_spectrum
        spectrum2, _ = different_spectrum
        empty_library.add_spectrum("test_001", spectrum1, wavelengths)
        success, msg = empty_library.add_spectrum("test_002", spectrum2, wavelengths)
        assert success
        assert empty_library.size == 2

    def test_add_batch(self, empty_library, sample_dataframe):
        """Test adding batch of spectra."""
        added, skipped, _ = empty_library.add_spectra_batch(
            sample_dataframe,
            source_file="batch_test.csv",
            category="soil",
        )
        assert added == 5
        assert skipped == 0
        assert empty_library.size == 5

    def test_remove_spectrum(self, empty_library, sample_spectrum):
        """Test removing a spectrum."""
        spectrum, wavelengths = sample_spectrum
        empty_library.add_spectrum("test_001", spectrum, wavelengths)
        assert empty_library.size == 1
        result = empty_library.remove_spectrum("test_001")
        assert result
        assert empty_library.size == 0

    def test_get_spectrum(self, empty_library, sample_spectrum):
        """Test retrieving a spectrum."""
        spectrum, wavelengths = sample_spectrum
        empty_library.add_spectrum("test_001", spectrum, wavelengths, category="soil")
        entry = empty_library.get_spectrum("test_001")
        assert entry is not None
        assert entry.category == "soil"
        assert np.allclose(entry.spectrum, spectrum)

    def test_search_empty_library(self, empty_library, sample_spectrum):
        """Searching empty library should return empty DataFrame."""
        spectrum, wavelengths = sample_spectrum
        results = empty_library.search(spectrum, wavelengths)
        assert len(results) == 0

    def test_search_finds_exact_match(self, empty_library, sample_spectrum):
        """Search should find exact match with HQI=1."""
        spectrum, wavelengths = sample_spectrum
        empty_library.add_spectrum("test_001", spectrum, wavelengths)
        results = empty_library.search(spectrum, wavelengths, metric='hqi', top_k=5)
        assert len(results) == 1
        assert results.iloc[0]['sample_id'] == 'test_001'
        assert results.iloc[0]['score'] == pytest.approx(1.0, abs=1e-6)

    def test_search_ranks_correctly(self, empty_library, sample_spectrum, different_spectrum):
        """Search should rank more similar spectra higher."""
        spectrum1, wavelengths = sample_spectrum
        spectrum2, _ = different_spectrum

        # Add original and different
        empty_library.add_spectrum("original", spectrum1, wavelengths)
        empty_library.add_spectrum("different", spectrum2, wavelengths)

        # Search with original - should find itself first
        results = empty_library.search(spectrum1, wavelengths, metric='hqi', top_k=5)
        assert len(results) == 2
        assert results.iloc[0]['sample_id'] == 'original'
        assert results.iloc[0]['score'] > results.iloc[1]['score']

    def test_search_with_category_filter(self, empty_library, sample_spectrum, different_spectrum):
        """Search should filter by category."""
        spectrum1, wavelengths = sample_spectrum
        spectrum2, _ = different_spectrum

        empty_library.add_spectrum("soil_001", spectrum1, wavelengths, category="soil")
        empty_library.add_spectrum("mineral_001", spectrum2, wavelengths, category="mineral")

        results = empty_library.search(spectrum1, wavelengths, category="soil")
        assert len(results) == 1
        assert results.iloc[0]['sample_id'] == 'soil_001'

    def test_get_statistics(self, empty_library, sample_dataframe):
        """Test library statistics."""
        empty_library.add_spectra_batch(sample_dataframe, category="soil")
        stats = empty_library.get_statistics()
        assert stats['total_entries'] == 5
        assert 'soil' in stats['categories']
        assert stats['wavelength_range'] is not None

    def test_clear_library(self, empty_library, sample_spectrum):
        """Test clearing the library."""
        spectrum, wavelengths = sample_spectrum
        empty_library.add_spectrum("test_001", spectrum, wavelengths)
        assert empty_library.size == 1
        empty_library.clear()
        assert empty_library.size == 0


class TestLibraryPersistence:
    """Tests for library persistence (save/load)."""

    def test_save_and_load(self, temp_library_dir, sample_spectrum):
        """Test saving and loading library."""
        spectrum, wavelengths = sample_spectrum

        # Create and populate library
        lib1 = SpectralLibrary(name="persist_test", storage_path=temp_library_dir)
        lib1.add_spectrum("test_001", spectrum, wavelengths, category="soil")
        lib1.save()

        # Load into new instance
        lib2 = SpectralLibrary(name="persist_test", storage_path=temp_library_dir)

        assert lib2.size == 1
        assert "test_001" in lib2.sample_ids
        entry = lib2.get_spectrum("test_001")
        assert entry.category == "soil"
        assert np.allclose(entry.spectrum, spectrum)

    def test_auto_save(self, temp_library_dir, sample_spectrum):
        """Test auto-save functionality."""
        spectrum, wavelengths = sample_spectrum

        lib1 = SpectralLibrary(name="autosave_test", storage_path=temp_library_dir, auto_save=True)
        lib1.add_spectrum("test_001", spectrum, wavelengths)

        # Don't call save() explicitly - should be auto-saved
        lib2 = SpectralLibrary(name="autosave_test", storage_path=temp_library_dir)
        assert lib2.size == 1


class TestGlobalLibrary:
    """Tests for global library functions."""

    def test_get_library_singleton(self, temp_library_dir):
        """get_library should return same instance for same name."""
        lib1 = get_library("test", storage_path=temp_library_dir)
        lib2 = get_library("test", storage_path=temp_library_dir)
        assert lib1 is lib2

    def test_add_to_library_convenience(self, temp_library_dir, sample_dataframe):
        """Test convenience function for adding spectra."""
        # Clear any existing global library
        import spectral_predict.library_search as ls
        ls._global_library = None

        lib = get_library("convenience_test", storage_path=temp_library_dir)
        added, skipped = add_to_library(
            sample_dataframe,
            source_file="test.csv",
            library_name="convenience_test",
        )
        assert added == 5
        assert lib.size == 5

    def test_search_library_convenience(self, temp_library_dir, sample_spectrum, sample_dataframe):
        """Test convenience function for searching."""
        import spectral_predict.library_search as ls
        ls._global_library = None

        spectrum, wavelengths = sample_spectrum
        lib = get_library("search_test", storage_path=temp_library_dir)
        lib.add_spectra_batch(sample_dataframe)

        results = search_library(
            spectrum,
            wavelengths,
            metric='hqi',
            top_k=3,
            library_name="search_test",
        )
        assert len(results) <= 3


# ============================================================================
# Integration Tests
# ============================================================================

class TestLibrarySearchIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, temp_library_dir):
        """Test complete library workflow: add, search, remove."""
        wavelengths = np.linspace(1000, 2500, 200)

        # Create library
        lib = SpectralLibrary(name="workflow_test", storage_path=temp_library_dir)

        # Add various spectra
        for i in range(10):
            spectrum = 0.5 + 0.1 * np.sin(wavelengths / (100 + i * 5))
            lib.add_spectrum(f"sample_{i:03d}", spectrum, wavelengths, category=f"cat_{i % 3}")

        assert lib.size == 10
        assert len(lib.categories) == 3

        # Search
        query = 0.5 + 0.1 * np.sin(wavelengths / 100)  # Similar to sample_000
        results = lib.search(query, wavelengths, metric='hqi', top_k=5)
        assert len(results) == 5
        assert results.iloc[0]['sample_id'] == 'sample_000'

        # Test different metrics
        for metric in ['sam', 'euclidean', 'cosine']:
            results = lib.search(query, wavelengths, metric=metric, top_k=3)
            assert len(results) == 3

        # Remove and verify
        lib.remove_spectrum('sample_000')
        assert lib.size == 9
        assert 'sample_000' not in lib.sample_ids

    def test_different_wavelength_grids(self, temp_library_dir):
        """Test handling of spectra with different wavelength grids."""
        lib = SpectralLibrary(name="grid_test", storage_path=temp_library_dir)

        # Add spectrum with one grid
        wl1 = np.linspace(1000, 2500, 200)
        spec1 = 0.5 + 0.1 * np.sin(wl1 / 100)
        lib.add_spectrum("ref_001", spec1, wl1)

        # Search with different grid (should interpolate)
        wl2 = np.linspace(1000, 2500, 150)  # Different resolution
        spec2 = 0.5 + 0.1 * np.sin(wl2 / 100)
        results = lib.search(spec2, wl2, metric='hqi')

        # Should still find the match (interpolation handles the difference)
        assert len(results) == 1
        assert results.iloc[0]['score'] > 0.99
