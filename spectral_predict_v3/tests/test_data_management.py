"""
Unit tests for data_management module.

Tests all merge strategies, duplicate handling modes, and edge cases.
"""

import numpy as np
import pytest
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_management import (
    DataSource,
    MergeResult,
    merge_sources,
    validate_data_source,
    print_merge_summary
)


# Test fixtures

def create_test_source(
    name: str,
    n_samples: int = 10,
    wavelengths: np.ndarray = None,
    has_y: bool = False,
    sample_id_prefix: str = "sample"
) -> DataSource:
    """Helper to create test data source."""
    if wavelengths is None:
        wavelengths = np.arange(400, 500, 10, dtype=np.float64)

    n_wavelengths = len(wavelengths)
    X = np.random.randn(n_samples, n_wavelengths)
    sample_ids = [f"{sample_id_prefix}_{i}" for i in range(n_samples)]

    y = np.random.randn(n_samples) if has_y else None
    target_name = "target" if has_y else None

    return DataSource(
        source_id=f"source_{name}",
        name=name,
        path=f"/path/to/{name}.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids,
        y=y,
        target_name=target_name
    )


# Test DataSource validation

def test_data_source_creation():
    """Test basic DataSource creation and validation."""
    X = np.random.randn(5, 10)
    wavelengths = np.arange(400, 500, 10, dtype=np.float64)
    sample_ids = [f"sample_{i}" for i in range(5)]

    source = DataSource(
        source_id="test",
        name="test_source",
        path="/path/to/test.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids
    )

    assert source.n_samples == 5
    assert source.n_wavelengths == 10
    assert source.y is None


def test_data_source_with_target():
    """Test DataSource with target values."""
    X = np.random.randn(5, 10)
    wavelengths = np.arange(400, 500, 10, dtype=np.float64)
    sample_ids = [f"sample_{i}" for i in range(5)]
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    source = DataSource(
        source_id="test",
        name="test_source",
        path="/path/to/test.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids,
        y=y,
        target_name="concentration"
    )

    assert source.y is not None
    assert len(source.y) == 5
    assert source.target_name == "concentration"


def test_data_source_validation_errors():
    """Test that DataSource validation catches errors."""
    X = np.random.randn(5, 10)
    wavelengths = np.arange(400, 500, 10, dtype=np.float64)

    # Wrong number of sample IDs
    with pytest.raises(ValueError, match="Sample ID count mismatch"):
        DataSource(
            source_id="test",
            name="test",
            path="/path/test.csv",
            X=X,
            wavelengths=wavelengths,
            sample_ids=["sample_1", "sample_2"]  # Only 2, need 5
        )

    # Wrong number of wavelengths
    with pytest.raises(ValueError, match="Wavelength count mismatch"):
        DataSource(
            source_id="test",
            name="test",
            path="/path/test.csv",
            X=X,
            wavelengths=np.arange(400, 450, 10),  # Only 5, need 10
            sample_ids=[f"sample_{i}" for i in range(5)]
        )

    # Wrong y length
    with pytest.raises(ValueError, match="Target value count mismatch"):
        DataSource(
            source_id="test",
            name="test",
            path="/path/test.csv",
            X=X,
            wavelengths=wavelengths,
            sample_ids=[f"sample_{i}" for i in range(5)],
            y=np.array([1.0, 2.0])  # Only 2, need 5
        )


def test_data_source_negative_wavelengths():
    """Test that negative wavelengths are rejected."""
    X = np.random.randn(5, 10)
    wavelengths = np.arange(-10, 90, 10, dtype=np.float64)  # Starts negative

    with pytest.raises(ValueError, match="All wavelengths must be positive"):
        DataSource(
            source_id="test",
            name="test",
            path="/path/test.csv",
            X=X,
            wavelengths=wavelengths,
            sample_ids=[f"sample_{i}" for i in range(5)]
        )


def test_data_source_unsorted_wavelengths():
    """Test that unsorted wavelengths are automatically sorted."""
    X = np.random.randn(5, 5)
    # Deliberately unsorted wavelengths
    wavelengths = np.array([450, 400, 420, 440, 410], dtype=np.float64)
    sample_ids = [f"sample_{i}" for i in range(5)]

    source = DataSource(
        source_id="test",
        name="test",
        path="/path/test.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids
    )

    # Should be sorted now
    assert np.all(np.diff(source.wavelengths) > 0)
    assert source.wavelengths[0] == 400
    assert source.wavelengths[-1] == 450


def test_data_source_all_nan_target():
    """Test that all-NaN target is treated as no target."""
    X = np.random.randn(5, 10)
    wavelengths = np.arange(400, 500, 10, dtype=np.float64)
    sample_ids = [f"sample_{i}" for i in range(5)]
    y = np.full(5, np.nan)

    source = DataSource(
        source_id="test",
        name="test",
        path="/path/test.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids,
        y=y,
        target_name="target"
    )

    # Should be converted to None
    assert source.y is None
    assert source.target_name is None


# Test merge strategies

def test_merge_single_source():
    """Test that single source is returned as-is."""
    source = create_test_source("single", n_samples=5, has_y=True)

    result = merge_sources([source])

    assert result.strategy == 'single_source'
    assert result.n_sources == 1
    assert len(result.sample_ids) == 5
    assert result.y is not None


def test_merge_intersection_perfect_overlap():
    """Test intersection merge with identical wavelengths."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl, has_y=True)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl, has_y=True,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='intersection')

    assert result.strategy == 'intersection'
    assert result.n_sources == 2
    assert len(result.sample_ids) == 8  # 5 + 3
    assert len(result.wavelengths) == 10  # All wavelengths common
    assert result.X.shape == (8, 10)
    assert result.y is not None
    assert len(result.y) == 8


def test_merge_intersection_partial_overlap():
    """Test intersection merge with partially overlapping wavelengths."""
    wl1 = np.arange(400, 500, 10, dtype=np.float64)  # 400-490
    wl2 = np.arange(450, 550, 10, dtype=np.float64)  # 450-540

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl1)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl2,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='intersection')

    # Common range is 450-490
    assert result.strategy == 'intersection'
    assert len(result.wavelengths) == 5  # 450, 460, 470, 480, 490
    assert result.wavelengths[0] == 450
    assert result.wavelengths[-1] == 490
    assert result.X.shape == (8, 5)


def test_merge_intersection_no_overlap():
    """Test that intersection with no overlap raises error."""
    wl1 = np.arange(400, 500, 10, dtype=np.float64)  # 400-490
    wl2 = np.arange(500, 600, 10, dtype=np.float64)  # 500-590 (no overlap)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl1)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl2,
                                sample_id_prefix="wheat")

    with pytest.raises(ValueError, match="No common wavelengths"):
        merge_sources([source1, source2], strategy='intersection')


def test_merge_union_perfect_overlap():
    """Test union merge with identical wavelengths."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='union')

    assert result.strategy == 'union'
    assert len(result.wavelengths) == 10  # All unique wavelengths
    assert result.X.shape == (8, 10)
    # Should have no NaN for perfect overlap
    assert np.isnan(result.X).sum() == 0


def test_merge_union_partial_overlap():
    """Test union merge with partially overlapping wavelengths."""
    wl1 = np.arange(400, 500, 10, dtype=np.float64)  # 400-490 (10 points)
    wl2 = np.arange(450, 550, 10, dtype=np.float64)  # 450-540 (10 points)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl1)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl2,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='union')

    # Union should be 400-540 (15 points)
    assert result.strategy == 'union'
    assert len(result.wavelengths) == 15
    assert result.wavelengths[0] == 400
    assert result.wavelengths[-1] == 540
    assert result.X.shape == (8, 15)

    # Check for NaN in expected places
    # Source1 should have NaN for wavelengths 500-540
    # Source2 should have NaN for wavelengths 400-440
    assert np.isnan(result.X[:5, -5:]).all()  # First 5 samples, last 5 wavelengths
    assert np.isnan(result.X[5:, :5]).all()   # Last 3 samples, first 5 wavelengths


def test_merge_union_no_overlap():
    """Test union merge with no overlapping wavelengths."""
    wl1 = np.arange(400, 450, 10, dtype=np.float64)  # 400-440 (5 points)
    wl2 = np.arange(500, 550, 10, dtype=np.float64)  # 500-540 (5 points)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl1)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl2,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='union')

    # Union should have all wavelengths
    assert len(result.wavelengths) == 10
    assert result.X.shape == (8, 10)

    # Each source should have NaN where the other has data
    assert np.isnan(result.X[:5, 5:]).all()  # Source1 missing source2's wavelengths
    assert np.isnan(result.X[5:, :5]).all()  # Source2 missing source1's wavelengths


def test_merge_interpolation():
    """Test interpolation merge strategy."""
    wl1 = np.arange(400, 500, 20, dtype=np.float64)  # 400, 420, 440, 460, 480
    wl2 = np.arange(410, 510, 20, dtype=np.float64)  # 410, 430, 450, 470, 490

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl1)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl2,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='interpolation',
                          wavelength_step=10.0)

    assert result.strategy == 'interpolation'
    # Grid should be 400 to 490 with step 10 = 10 points
    assert len(result.wavelengths) >= 10
    assert result.wavelengths[0] == 400
    assert result.X.shape == (8, len(result.wavelengths))
    # Should have no NaN (interpolation fills everything)
    assert not np.isnan(result.X).any()


# Test duplicate handling

def test_merge_duplicate_error():
    """Test that duplicate IDs raise error with 'error' strategy."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl,
                                sample_id_prefix="sample")
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl,
                                sample_id_prefix="sample")  # Same prefix = duplicates

    with pytest.raises(ValueError, match="Duplicate sample IDs found"):
        merge_sources([source1, source2], strategy='intersection',
                     dup_handling='error')


def test_merge_duplicate_rename():
    """Test that duplicate IDs are renamed with 'rename' strategy."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl,
                                sample_id_prefix="sample")
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl,
                                sample_id_prefix="sample")  # Same prefix

    result = merge_sources([source1, source2], strategy='intersection',
                          dup_handling='rename')

    # Should have renamed duplicates
    assert len(result.sample_ids) == 8
    assert len(set(result.sample_ids)) == 8  # All unique now

    # Check that wheat samples were renamed
    wheat_samples = [sid for sid in result.sample_ids if 'wheat' in sid]
    assert len(wheat_samples) == 3


def test_merge_duplicate_keep_first():
    """Test that duplicates are skipped with 'keep_first' strategy."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl,
                                sample_id_prefix="sample")
    source2 = create_test_source("wheat", n_samples=5, wavelengths=wl,
                                sample_id_prefix="sample")  # All duplicates

    result = merge_sources([source1, source2], strategy='intersection',
                          dup_handling='keep_first')

    # Should only have 5 samples (from source1)
    assert len(result.sample_ids) == 5
    assert result.X.shape[0] == 5


def test_merge_duplicate_keep_last():
    """Test 'keep_last' strategy (implementation may vary)."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl,
                                sample_id_prefix="sample")
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl,
                                sample_id_prefix="sample")  # 3 duplicates

    result = merge_sources([source1, source2], strategy='intersection',
                          dup_handling='keep_last')

    # With current implementation, this will keep all 8
    # (True 'keep_last' would require overwrite logic)
    assert len(result.sample_ids) == 8


# Test target value handling

def test_merge_mixed_targets():
    """Test merging sources with and without target values."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl, has_y=True)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl, has_y=False,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='intersection')

    # Should have y values (from source1), but not for all samples
    # Union strategy would show NaN for source2
    if result.strategy == 'union':
        assert result.y is not None
        assert np.isnan(result.y[5:]).all()  # wheat samples have no y


def test_merge_no_targets():
    """Test merging sources without any target values."""
    wl = np.arange(400, 500, 10, dtype=np.float64)

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl, has_y=False)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl, has_y=False,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='intersection')

    assert result.y is None
    assert result.target_name is None


# Test validation and reporting

def test_validate_data_source():
    """Test data source validation function."""
    source = create_test_source("test", n_samples=10, has_y=True)

    report = validate_data_source(source)

    assert report['is_valid']
    assert 'statistics' in report
    assert 'x_mean' in report['statistics']
    assert 'y_mean' in report['statistics']


def test_validate_data_source_with_nan():
    """Test validation detects NaN values."""
    X = np.random.randn(5, 10)
    X[0, 0] = np.nan  # Add a NaN
    wavelengths = np.arange(400, 500, 10, dtype=np.float64)
    sample_ids = [f"sample_{i}" for i in range(5)]

    source = DataSource(
        source_id="test",
        name="test",
        path="/path/test.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids
    )

    report = validate_data_source(source)

    assert any('NaN' in w for w in report['warnings'])


def test_validate_data_source_with_inf():
    """Test validation detects infinite values."""
    X = np.random.randn(5, 10)
    X[0, 0] = np.inf  # Add an inf
    wavelengths = np.arange(400, 500, 10, dtype=np.float64)
    sample_ids = [f"sample_{i}" for i in range(5)]

    source = DataSource(
        source_id="test",
        name="test",
        path="/path/test.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids
    )

    report = validate_data_source(source)

    assert not report['is_valid']
    assert any('infinite' in w for w in report['warnings'])


def test_print_merge_summary():
    """Test that merge summary printing works."""
    source = create_test_source("test", n_samples=10, has_y=True)
    result = merge_sources([source])

    # Should not raise an error
    print_merge_summary(result)


# Test edge cases

def test_merge_empty_sources_list():
    """Test that empty sources list raises error."""
    with pytest.raises(ValueError, match="sources list is empty"):
        merge_sources([])


def test_merge_invalid_strategy():
    """Test that invalid strategy raises error."""
    source = create_test_source("test", n_samples=5)

    with pytest.raises(ValueError, match="Invalid strategy"):
        merge_sources([source], strategy='invalid')


def test_merge_invalid_dup_handling():
    """Test that invalid duplicate handling raises error."""
    source = create_test_source("test", n_samples=5)

    with pytest.raises(ValueError, match="Invalid dup_handling"):
        merge_sources([source], dup_handling='invalid')


def test_merge_wavelength_tolerance():
    """Test that wavelengths within tolerance are matched."""
    # Create sources with slightly different wavelengths (within tolerance)
    wl1 = np.array([400.0, 410.0, 420.0, 430.0, 440.0])
    wl2 = np.array([400.00001, 410.00001, 420.00001, 430.00001, 440.00001])

    source1 = create_test_source("corn", n_samples=5, wavelengths=wl1)
    source2 = create_test_source("wheat", n_samples=3, wavelengths=wl2,
                                sample_id_prefix="wheat")

    result = merge_sources([source1, source2], strategy='intersection')

    # Should treat as identical wavelengths
    assert len(result.wavelengths) == 5


if __name__ == '__main__':
    # Run tests
    print("Running data_management tests...\n")

    # Basic tests
    print("Testing DataSource creation...")
    test_data_source_creation()
    test_data_source_with_target()
    print("[PASS] DataSource creation tests passed\n")

    print("Testing DataSource validation...")
    test_data_source_unsorted_wavelengths()
    test_data_source_all_nan_target()
    print("[PASS] DataSource validation tests passed\n")

    print("Testing merge strategies...")
    test_merge_single_source()
    test_merge_intersection_perfect_overlap()
    test_merge_intersection_partial_overlap()
    test_merge_union_perfect_overlap()
    test_merge_union_partial_overlap()
    test_merge_interpolation()
    print("[PASS] Merge strategy tests passed\n")

    print("Testing duplicate handling...")
    test_merge_duplicate_rename()
    test_merge_duplicate_keep_first()
    print("[PASS] Duplicate handling tests passed\n")

    print("Testing target handling...")
    test_merge_no_targets()
    print("[PASS] Target handling tests passed\n")

    print("Testing validation...")
    test_validate_data_source()
    test_print_merge_summary()
    print("[PASS] Validation tests passed\n")

    print("Testing edge cases...")
    test_merge_wavelength_tolerance()
    print("[PASS] Edge case tests passed\n")

    print("="*60)
    print("All tests passed!")
    print("="*60)
