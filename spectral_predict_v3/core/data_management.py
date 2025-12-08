"""
Data Source Management and Merging for Spectral Predict V3

This module provides functionality for managing and merging multiple spectral data sources.
Forked from V1's proven logic, adapted for numpy arrays (V3's convention).

Key Features:
- Multiple merge strategies (intersection, union, interpolation)
- Robust duplicate sample handling
- Wavelength alignment with tolerance
- Comprehensive validation and error handling
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from scipy.interpolate import interp1d


@dataclass
class DataSource:
    """
    Container for a single spectral data source.

    Uses numpy arrays for V3 compatibility.
    """
    source_id: str
    name: str
    path: str
    X: np.ndarray  # Shape: (n_samples, n_wavelengths)
    wavelengths: np.ndarray  # Shape: (n_wavelengths,)
    sample_ids: List[str]
    y: Optional[np.ndarray] = None  # Shape: (n_samples,) if present
    target_name: Optional[str] = None
    metadata_columns: Dict[str, List[Any]] = field(default_factory=dict)  # Additional metadata columns
    n_samples: int = 0
    n_wavelengths: int = 0

    def __post_init__(self):
        """Validate and initialize computed fields."""
        # Validate array shapes
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {self.X.shape}")

        if self.wavelengths.ndim != 1:
            raise ValueError(f"wavelengths must be 1D array, got shape {self.wavelengths.shape}")

        # Set dimensions
        self.n_samples, self.n_wavelengths = self.X.shape

        # Validate consistency
        if len(self.wavelengths) != self.n_wavelengths:
            raise ValueError(
                f"Wavelength count mismatch: X has {self.n_wavelengths} columns, "
                f"but wavelengths has {len(self.wavelengths)} elements"
            )

        if len(self.sample_ids) != self.n_samples:
            raise ValueError(
                f"Sample ID count mismatch: X has {self.n_samples} rows, "
                f"but sample_ids has {len(self.sample_ids)} elements"
            )

        # Validate wavelengths
        if np.any(self.wavelengths <= 0):
            raise ValueError("All wavelengths must be positive")

        # Sort wavelengths if not monotonic
        if not np.all(np.diff(self.wavelengths) > 0):
            print(f"Warning: Wavelengths in source '{self.name}' are not sorted. Sorting now.")
            sort_idx = np.argsort(self.wavelengths)
            self.wavelengths = self.wavelengths[sort_idx]
            self.X = self.X[:, sort_idx]

        # Validate y if present
        if self.y is not None:
            if self.y.ndim != 1:
                raise ValueError(f"y must be 1D array, got shape {self.y.shape}")

            if len(self.y) != self.n_samples:
                raise ValueError(
                    f"Target value count mismatch: X has {self.n_samples} rows, "
                    f"but y has {len(self.y)} elements"
                )

            # Check if all NaN (treat as no y) - handle any dtype safely
            try:
                if self.y.dtype.kind == 'f' and np.all(np.isnan(self.y)):
                    print(f"Warning: All y values are NaN in source '{self.name}'. Treating as no target.")
                    self.y = None
                    self.target_name = None
            except (TypeError, ValueError):
                pass  # Non-numeric y values are fine (categorical)


@dataclass
class MergeResult:
    """
    Container for merged spectral data from multiple sources.

    Includes metadata about the merge operation.
    """
    X: np.ndarray  # Shape: (n_samples, n_wavelengths)
    wavelengths: np.ndarray  # Shape: (n_wavelengths,)
    sample_ids: List[str]
    y: Optional[np.ndarray] = None  # Shape: (n_samples,) if present
    target_name: Optional[str] = None
    datasource: Optional[List[str]] = None  # Source label for each sample (e.g., "A_bone", "B_corn")
    metadata_columns: Dict[str, List[Any]] = field(default_factory=dict)  # Merged metadata columns
    strategy: str = ""
    n_sources: int = 0
    report: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate merged data consistency."""
        if self.X.ndim != 2:
            raise ValueError(f"Merged X must be 2D array, got shape {self.X.shape}")

        n_samples, n_wavelengths = self.X.shape

        if len(self.wavelengths) != n_wavelengths:
            raise ValueError(
                f"Merged wavelength count mismatch: X has {n_wavelengths} columns, "
                f"but wavelengths has {len(self.wavelengths)} elements"
            )

        if len(self.sample_ids) != n_samples:
            raise ValueError(
                f"Merged sample ID count mismatch: X has {n_samples} rows, "
                f"but sample_ids has {len(self.sample_ids)} elements"
            )

        if self.y is not None:
            if len(self.y) != n_samples:
                raise ValueError(
                    f"Merged target count mismatch: X has {n_samples} rows, "
                    f"but y has {len(self.y)} elements"
                )

        if self.datasource is not None:
            if len(self.datasource) != n_samples:
                raise ValueError(
                    f"Datasource count mismatch: X has {n_samples} rows, "
                    f"but datasource has {len(self.datasource)} elements"
                )


def merge_sources(
    sources: List[DataSource],
    strategy: str = 'intersection',
    dup_handling: str = 'rename',
    wavelength_step: float = 1.0  # for interpolation strategy
) -> MergeResult:
    """
    Merge multiple spectral data sources using specified strategy.

    Parameters
    ----------
    sources : List[DataSource]
        List of data sources to merge. Must contain at least one source.

    strategy : str, default='intersection'
        Merge strategy to use:
        - 'intersection': Keep only common wavelengths across all sources
        - 'union': Keep all wavelengths, fill missing with NaN
        - 'interpolation': Interpolate all sources to a common wavelength grid

    dup_handling : str, default='rename'
        How to handle duplicate sample IDs:
        - 'error': Raise ValueError if duplicates found
        - 'keep_first': Keep first occurrence, skip later duplicates
        - 'keep_last': Keep last occurrence, overwrite earlier duplicates
        - 'rename': Make unique by appending source name (e.g., 'sample1_corn')

    wavelength_step : float, default=1.0
        Step size for interpolation grid (only used with 'interpolation' strategy).
        Grid will span min to max wavelength across all sources.

    Returns
    -------
    MergeResult
        Merged data with metadata about the merge operation.

    Raises
    ------
    ValueError
        If sources list is empty, strategy is invalid, or merge fails.

    Examples
    --------
    >>> # Merge two sources keeping common wavelengths
    >>> result = merge_sources([source1, source2], strategy='intersection')

    >>> # Merge with all wavelengths, filling gaps
    >>> result = merge_sources([source1, source2], strategy='union')

    >>> # Merge with interpolation to uniform grid
    >>> result = merge_sources([source1, source2], strategy='interpolation', wavelength_step=2.0)
    """
    # Validate inputs
    if not sources:
        raise ValueError("Cannot merge: sources list is empty")

    if strategy not in ['intersection', 'union', 'interpolation']:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be 'intersection', 'union', or 'interpolation'"
        )

    if dup_handling not in ['error', 'keep_first', 'keep_last', 'rename']:
        raise ValueError(
            f"Invalid dup_handling '{dup_handling}'. "
            f"Must be 'error', 'keep_first', 'keep_last', or 'rename'"
        )

    if len(sources) == 1:
        print("Single source provided - returning as-is without merge")
        source = sources[0]
        # Create datasource labels even for single source (for consistency)
        datasource_label = f"A_{source.name}"
        datasource_list = [datasource_label] * source.n_samples
        # Copy metadata_columns
        merged_metadata = {k: list(v) for k, v in source.metadata_columns.items()}
        return MergeResult(
            X=source.X.copy(),
            wavelengths=source.wavelengths.copy(),
            sample_ids=source.sample_ids.copy(),
            y=source.y.copy() if source.y is not None else None,
            target_name=source.target_name,
            datasource=datasource_list,
            metadata_columns=merged_metadata,
            strategy='single_source',
            n_sources=1,
            report={
                'n_samples': source.n_samples,
                'n_wavelengths': source.n_wavelengths,
                'has_target': source.y is not None
            }
        )

    print(f"\n{'='*60}")
    print(f"Merging {len(sources)} data sources")
    print(f"Strategy: {strategy}")
    print(f"Duplicate handling: {dup_handling}")
    print(f"{'='*60}\n")

    # Log source details
    for i, source in enumerate(sources, 1):
        print(f"Source {i}: {source.name}")
        print(f"  - Samples: {source.n_samples}")
        print(f"  - Wavelengths: {source.n_wavelengths} "
              f"({source.wavelengths[0]:.2f} to {source.wavelengths[-1]:.2f} nm)")
        print(f"  - Has target: {source.y is not None}")
    print()

    # Route to appropriate merge strategy
    if strategy == 'intersection':
        result = _merge_intersection(sources, dup_handling)
    elif strategy == 'union':
        result = _merge_union(sources, dup_handling)
    elif strategy == 'interpolation':
        result = _merge_interpolation(sources, dup_handling, wavelength_step)

    # Log results
    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"  - Total samples: {len(result.sample_ids)}")
    print(f"  - Total wavelengths: {len(result.wavelengths)}")
    print(f"  - Has target: {result.y is not None}")
    print(f"{'='*60}\n")

    return result


def _merge_intersection(
    sources: List[DataSource],
    dup_handling: str
) -> MergeResult:
    """
    Merge sources using wavelength intersection (only common wavelengths).

    This strategy keeps only wavelengths that are present in ALL sources.
    Uses tolerance-based matching (np.isclose with rtol=1e-5) to account for
    floating-point precision issues.

    Parameters
    ----------
    sources : List[DataSource]
        Sources to merge
    dup_handling : str
        Duplicate sample handling mode

    Returns
    -------
    MergeResult
        Merged data with common wavelengths only

    Raises
    ------
    ValueError
        If no common wavelengths found or no samples remain after merge
    """
    print("Using intersection strategy (common wavelengths only)...")

    # Find common wavelengths across all sources
    # Start with first source's wavelengths
    common_wl = set(np.round(sources[0].wavelengths, decimals=2))

    for source in sources[1:]:
        source_wl = set(np.round(source.wavelengths, decimals=2))
        common_wl = common_wl.intersection(source_wl)

    if not common_wl:
        raise ValueError(
            "No common wavelengths found across all sources. "
            "Consider using 'union' or 'interpolation' strategy instead."
        )

    # Sort common wavelengths
    common_wl = np.array(sorted(common_wl))
    print(f"Found {len(common_wl)} common wavelengths "
          f"({common_wl[0]:.2f} to {common_wl[-1]:.2f} nm)")

    # Extract data at common wavelengths from each source
    all_X_list = []
    all_y_list = []
    all_sample_ids = []
    all_datasource = []  # Track which source each sample comes from
    all_metadata: Dict[str, List[Any]] = {}  # Merged metadata columns
    target_names = []

    # Create source labels (A, B, C, etc.)
    source_labels = [chr(65 + i) for i in range(len(sources))]  # A, B, C, ...

    for source_idx, source in enumerate(sources):
        # Find indices of common wavelengths in this source
        # Use tolerance-based matching
        indices = []
        for wl in common_wl:
            idx = np.where(np.isclose(source.wavelengths, wl, rtol=1e-5, atol=0.01))[0]
            if len(idx) > 0:
                indices.append(idx[0])
            else:
                # This shouldn't happen if our intersection logic is correct
                raise ValueError(f"Wavelength {wl:.2f} not found in source '{source.name}'")

        # Extract X data at common wavelengths
        X_subset = source.X[:, indices]

        # Handle sample IDs and duplicates - returns sample_ids and indices to keep
        sample_ids, keep_indices = _handle_duplicate_ids(
            source.sample_ids,
            all_sample_ids,
            dup_handling,
            source.name
        )

        # Filter X data based on keep_indices
        if keep_indices is not None:
            X_subset = X_subset[keep_indices, :]

        all_X_list.append(X_subset)
        all_sample_ids.extend(sample_ids)

        # Track datasource for each sample
        datasource_label = f"{source_labels[source_idx]}_{source.name}"
        all_datasource.extend([datasource_label] * len(sample_ids))

        # Collect y values if present (also filter by keep_indices)
        # Add NaN for sources without y to keep alignment
        if source.y is not None:
            y_subset = source.y[keep_indices] if keep_indices is not None else source.y
            all_y_list.append(y_subset)
            if source.target_name:
                target_names.append(source.target_name)
        else:
            # Add NaN values for this source
            n_kept = len(sample_ids)
            all_y_list.append(np.full(n_kept, np.nan))

        # Collect metadata columns (filter by keep_indices)
        for col_name, col_values in source.metadata_columns.items():
            if col_name not in all_metadata:
                # Initialize with None for all previous samples
                all_metadata[col_name] = [None] * (len(all_sample_ids) - len(sample_ids))
            # Add values for current source (filtered by keep_indices)
            if keep_indices is not None:
                all_metadata[col_name].extend([col_values[i] for i in keep_indices])
            else:
                all_metadata[col_name].extend(col_values)

        # Fill None for columns not in this source
        for col_name in all_metadata:
            if col_name not in source.metadata_columns:
                all_metadata[col_name].extend([None] * len(sample_ids))

    # Concatenate all data
    X_merged = np.vstack(all_X_list)

    # Handle y values
    y_merged = None
    target_name = None
    if all_y_list:
        y_merged = np.concatenate(all_y_list)
        # If all NaN, treat as no target (only check for float dtype)
        try:
            if y_merged.dtype.kind == 'f' and np.all(np.isnan(y_merged)):
                y_merged = None
                target_name = None
        except (TypeError, ValueError):
            pass  # Non-numeric y values are fine
        if y_merged is not None:
            # Use first non-None target name
            target_name = target_names[0] if target_names else None

            # Warn if target names differ
            if len(set(target_names)) > 1:
                print(f"Warning: Multiple target names found: {set(target_names)}. Using '{target_name}'")

    # Validate result
    if X_merged.shape[0] == 0:
        raise ValueError("Merge resulted in zero samples")

    # Create report
    report = {
        'n_samples_per_source': [s.n_samples for s in sources],
        'n_samples_total': X_merged.shape[0],
        'n_wavelengths_original': [s.n_wavelengths for s in sources],
        'n_wavelengths_common': len(common_wl),
        'wavelength_range': (float(common_wl[0]), float(common_wl[-1])),
        'has_target': y_merged is not None
    }

    return MergeResult(
        X=X_merged,
        wavelengths=common_wl,
        sample_ids=all_sample_ids,
        y=y_merged,
        target_name=target_name,
        datasource=all_datasource,
        metadata_columns=all_metadata,
        strategy='intersection',
        n_sources=len(sources),
        report=report
    )


def _merge_union(
    sources: List[DataSource],
    dup_handling: str
) -> MergeResult:
    """
    Merge sources using wavelength union (all wavelengths, fill missing with NaN).

    This strategy keeps all wavelengths from all sources. For wavelengths not
    present in a particular source, the data is filled with NaN values.

    Parameters
    ----------
    sources : List[DataSource]
        Sources to merge
    dup_handling : str
        Duplicate sample handling mode

    Returns
    -------
    MergeResult
        Merged data with all wavelengths, NaN where data is missing

    Raises
    ------
    ValueError
        If no samples remain after merge
    """
    print("Using union strategy (all wavelengths with NaN fill)...")

    # Collect all unique wavelengths across all sources
    all_wl = set()
    for source in sources:
        all_wl.update(np.round(source.wavelengths, decimals=2))

    # Sort wavelengths
    union_wl = np.array(sorted(all_wl))
    print(f"Found {len(union_wl)} unique wavelengths "
          f"({union_wl[0]:.2f} to {union_wl[-1]:.2f} nm)")

    # Collect data from all sources (will filter duplicates later)
    all_X_parts = []
    all_sample_ids = []
    all_datasource = []  # Track which source each sample comes from
    all_metadata: Dict[str, List[Any]] = {}  # Merged metadata columns
    all_y_list = []
    target_names = []

    # Create source labels (A, B, C, etc.)
    source_labels = [chr(65 + i) for i in range(len(sources))]  # A, B, C, ...

    # Fill data from each source
    for source_idx, source in enumerate(sources):
        # Handle sample IDs and duplicates
        sample_ids, keep_indices = _handle_duplicate_ids(
            source.sample_ids,
            all_sample_ids,
            dup_handling,
            source.name
        )

        # Skip this source if no samples to keep
        if keep_indices is not None and len(keep_indices) == 0:
            continue

        # Filter X data if needed
        X_source = source.X[keep_indices, :] if keep_indices is not None else source.X
        n_samples_kept = X_source.shape[0]

        # Create X matrix for this source with union wavelengths (fill with NaN)
        X_union = np.full((n_samples_kept, len(union_wl)), np.nan, dtype=np.float64)

        # Find where this source's wavelengths fit in union
        for i, wl in enumerate(source.wavelengths):
            wl_rounded = round(wl, 2)
            union_idx = np.where(np.isclose(union_wl, wl_rounded, rtol=1e-5, atol=0.01))[0]
            if len(union_idx) > 0:
                X_union[:, union_idx[0]] = X_source[:, i]

        all_X_parts.append(X_union)
        all_sample_ids.extend(sample_ids)

        # Track datasource for each sample
        datasource_label = f"{source_labels[source_idx]}_{source.name}"
        all_datasource.extend([datasource_label] * len(sample_ids))

        # Collect y values (also filter by keep_indices)
        if source.y is not None:
            y_subset = source.y[keep_indices] if keep_indices is not None else source.y
            all_y_list.append(y_subset)
            if source.target_name:
                target_names.append(source.target_name)
        else:
            # Fill with NaN for sources without y
            all_y_list.append(np.full(n_samples_kept, np.nan))

        # Collect metadata columns (filter by keep_indices)
        for col_name, col_values in source.metadata_columns.items():
            if col_name not in all_metadata:
                # Initialize with None for all previous samples
                all_metadata[col_name] = [None] * (len(all_sample_ids) - len(sample_ids))
            # Add values for current source (filtered by keep_indices)
            if keep_indices is not None:
                all_metadata[col_name].extend([col_values[i] for i in keep_indices])
            else:
                all_metadata[col_name].extend(col_values)

        # Fill None for columns not in this source
        for col_name in all_metadata:
            if col_name not in source.metadata_columns:
                all_metadata[col_name].extend([None] * len(sample_ids))

    # Stack all X parts
    X_merged = np.vstack(all_X_parts)

    # Handle y values
    y_merged = None
    target_name = None
    if all_y_list:
        y_merged = np.concatenate(all_y_list)
        # If all NaN, treat as no target (only check for float dtype)
        try:
            if y_merged.dtype.kind == 'f' and np.all(np.isnan(y_merged)):
                y_merged = None
        except (TypeError, ValueError):
            pass  # Non-numeric y values are fine
        if y_merged is not None:
            target_name = target_names[0] if target_names else None
            if len(set(target_names)) > 1:
                print(f"Warning: Multiple target names found: {set(target_names)}. Using '{target_name}'")

    # Validate result
    if X_merged.shape[0] == 0:
        raise ValueError("Merge resulted in zero samples")

    # Report on NaN content
    nan_percent = 100 * np.isnan(X_merged).sum() / X_merged.size
    print(f"Union merge created {nan_percent:.1f}% NaN values (expected for non-overlapping wavelengths)")

    # Create report
    report = {
        'n_samples_per_source': [s.n_samples for s in sources],
        'n_samples_total': X_merged.shape[0],
        'n_wavelengths_per_source': [s.n_wavelengths for s in sources],
        'n_wavelengths_union': len(union_wl),
        'wavelength_range': (float(union_wl[0]), float(union_wl[-1])),
        'nan_percent': float(nan_percent),
        'has_target': y_merged is not None
    }

    return MergeResult(
        X=X_merged,
        wavelengths=union_wl,
        sample_ids=all_sample_ids,
        y=y_merged,
        target_name=target_name,
        datasource=all_datasource,
        metadata_columns=all_metadata,
        strategy='union',
        n_sources=len(sources),
        report=report
    )


def _merge_interpolation(
    sources: List[DataSource],
    dup_handling: str,
    wavelength_step: float
) -> MergeResult:
    """
    Merge sources by interpolating all to a common wavelength grid.

    This strategy creates a uniform wavelength grid spanning the min to max
    wavelength across all sources, then interpolates each spectrum onto this grid.
    Uses linear interpolation via scipy.interpolate.interp1d.

    Parameters
    ----------
    sources : List[DataSource]
        Sources to merge
    dup_handling : str
        Duplicate sample handling mode
    wavelength_step : float
        Step size for interpolation grid (nm)

    Returns
    -------
    MergeResult
        Merged data interpolated to common wavelength grid

    Raises
    ------
    ValueError
        If wavelength_step is invalid or no samples remain after merge
    """
    print(f"Using interpolation strategy (common grid with {wavelength_step} nm step)...")

    if wavelength_step <= 0:
        raise ValueError(f"wavelength_step must be positive, got {wavelength_step}")

    # Find global wavelength range
    min_wl = min(s.wavelengths.min() for s in sources)
    max_wl = max(s.wavelengths.max() for s in sources)

    # Create common wavelength grid
    common_wl = np.arange(min_wl, max_wl + wavelength_step, wavelength_step)
    print(f"Created interpolation grid: {len(common_wl)} points "
          f"({common_wl[0]:.2f} to {common_wl[-1]:.2f} nm)")

    # Interpolate each source to common grid
    all_X_list = []
    all_y_list = []
    all_sample_ids = []
    all_datasource = []  # Track which source each sample comes from
    all_metadata: Dict[str, List[Any]] = {}  # Merged metadata columns
    target_names = []

    # Create source labels (A, B, C, etc.)
    source_labels = [chr(65 + i) for i in range(len(sources))]  # A, B, C, ...

    for source_idx, source in enumerate(sources):
        print(f"  Interpolating {source.name} ({source.n_samples} samples)...")

        # Handle sample IDs and duplicates first
        sample_ids, keep_indices = _handle_duplicate_ids(
            source.sample_ids,
            all_sample_ids,
            dup_handling,
            source.name
        )

        # Skip this source if no samples to keep
        if keep_indices is not None and len(keep_indices) == 0:
            continue

        # Determine which samples to interpolate
        if keep_indices is not None:
            samples_to_process = keep_indices
            n_samples_kept = len(keep_indices)
        else:
            samples_to_process = range(source.n_samples)
            n_samples_kept = source.n_samples

        # Interpolate each spectrum
        X_interp = np.zeros((n_samples_kept, len(common_wl)))

        for out_idx, in_idx in enumerate(samples_to_process):
            # Create interpolation function
            # Use bounds_error=False and fill_value='extrapolate' to handle edge cases
            interp_func = interp1d(
                source.wavelengths,
                source.X[in_idx, :],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            X_interp[out_idx, :] = interp_func(common_wl)

        all_X_list.append(X_interp)
        all_sample_ids.extend(sample_ids)

        # Track datasource for each sample
        datasource_label = f"{source_labels[source_idx]}_{source.name}"
        all_datasource.extend([datasource_label] * len(sample_ids))

        # Collect y values (also filter by keep_indices)
        if source.y is not None:
            y_subset = source.y[keep_indices] if keep_indices is not None else source.y
            all_y_list.append(y_subset)
            if source.target_name:
                target_names.append(source.target_name)

        # Collect metadata columns (filter by keep_indices)
        for col_name, col_values in source.metadata_columns.items():
            if col_name not in all_metadata:
                # Initialize with None for all previous samples
                all_metadata[col_name] = [None] * (len(all_sample_ids) - len(sample_ids))
            # Add values for current source (filtered by keep_indices)
            if keep_indices is not None:
                all_metadata[col_name].extend([col_values[i] for i in keep_indices])
            else:
                all_metadata[col_name].extend(col_values)

        # Fill None for columns not in this source
        for col_name in all_metadata:
            if col_name not in source.metadata_columns:
                all_metadata[col_name].extend([None] * len(sample_ids))

    # Concatenate all data
    X_merged = np.vstack(all_X_list)

    # Handle y values
    y_merged = None
    target_name = None
    if all_y_list:
        y_merged = np.concatenate(all_y_list)
        target_name = target_names[0] if target_names else None
        if len(set(target_names)) > 1:
            print(f"Warning: Multiple target names found: {set(target_names)}. Using '{target_name}'")

    # Validate result
    if X_merged.shape[0] == 0:
        raise ValueError("Merge resulted in zero samples")

    # Create report
    report = {
        'n_samples_per_source': [s.n_samples for s in sources],
        'n_samples_total': X_merged.shape[0],
        'n_wavelengths_original': [s.n_wavelengths for s in sources],
        'n_wavelengths_interpolated': len(common_wl),
        'wavelength_range': (float(common_wl[0]), float(common_wl[-1])),
        'wavelength_step': wavelength_step,
        'has_target': y_merged is not None
    }

    return MergeResult(
        X=X_merged,
        wavelengths=common_wl,
        sample_ids=all_sample_ids,
        y=y_merged,
        target_name=target_name,
        datasource=all_datasource,
        metadata_columns=all_metadata,
        strategy='interpolation',
        n_sources=len(sources),
        report=report
    )


def _handle_duplicate_ids(
    new_ids: List[str],
    existing_ids: List[str],
    dup_handling: str,
    source_name: str
) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Handle duplicate sample IDs according to specified strategy.

    Parameters
    ----------
    new_ids : List[str]
        New sample IDs to add
    existing_ids : List[str]
        Already existing sample IDs
    dup_handling : str
        Strategy: 'error', 'keep_first', 'keep_last', 'rename'
    source_name : str
        Name of source (used for rename strategy)

    Returns
    -------
    Tuple[List[str], Optional[np.ndarray]]
        - Processed sample IDs according to strategy
        - Array of indices to keep (None if all samples should be kept)

    Raises
    ------
    ValueError
        If dup_handling is 'error' and duplicates are found

    Notes
    -----
    The keep_indices array indicates which samples from new_ids to include.
    This allows the caller to filter both X and y data appropriately.
    """
    if dup_handling == 'error':
        # Check for duplicates
        duplicates = set(new_ids).intersection(set(existing_ids))
        if duplicates:
            raise ValueError(
                f"Duplicate sample IDs found: {duplicates}. "
                f"Use a different dup_handling strategy or rename samples."
            )
        return new_ids, None

    elif dup_handling == 'keep_first':
        # Skip new IDs that already exist
        result = []
        keep_indices = []
        for i, sample_id in enumerate(new_ids):
            if sample_id not in existing_ids:
                result.append(sample_id)
                keep_indices.append(i)
            else:
                print(f"  Skipping duplicate sample ID: {sample_id} (keeping first occurrence)")
        return result, np.array(keep_indices) if keep_indices else np.array([], dtype=int)

    elif dup_handling == 'keep_last':
        # Keep new IDs even if they exist (will overwrite)
        # This is complex to implement properly with overwriting
        # For now, just keep all and warn
        duplicates = set(new_ids).intersection(set(existing_ids))
        if duplicates:
            print(f"  Warning: {len(duplicates)} duplicate sample IDs found. "
                  f"'keep_last' will add all samples (no true overwrite implemented)")
        return new_ids, None

    elif dup_handling == 'rename':
        # Append source name to make unique
        result = []
        for sample_id in new_ids:
            if sample_id in existing_ids or sample_id in result:
                new_id = f"{sample_id}_{source_name}"
                # Handle case where even renamed ID exists
                counter = 1
                while new_id in existing_ids or new_id in result:
                    new_id = f"{sample_id}_{source_name}_{counter}"
                    counter += 1
                print(f"  Renamed duplicate sample ID: {sample_id} -> {new_id}")
                result.append(new_id)
            else:
                result.append(sample_id)
        return result, None

    else:
        raise ValueError(f"Unknown dup_handling strategy: {dup_handling}")


# Utility functions for validation and reporting

def validate_data_source(source: DataSource) -> Dict[str, Any]:
    """
    Validate a data source and return diagnostic information.

    Parameters
    ----------
    source : DataSource
        Data source to validate

    Returns
    -------
    Dict[str, Any]
        Validation report with warnings and statistics
    """
    report = {
        'is_valid': True,
        'warnings': [],
        'statistics': {}
    }

    # Check for NaN values
    nan_count = np.isnan(source.X).sum()
    if nan_count > 0:
        nan_percent = 100 * nan_count / source.X.size
        report['warnings'].append(f"X contains {nan_count} NaN values ({nan_percent:.1f}%)")

    # Check for infinite values
    inf_count = np.isinf(source.X).sum()
    if inf_count > 0:
        report['warnings'].append(f"X contains {inf_count} infinite values")
        report['is_valid'] = False

    # Check wavelength spacing
    wl_diffs = np.diff(source.wavelengths)
    wl_spacing_std = np.std(wl_diffs)
    if wl_spacing_std > 1.0:
        report['warnings'].append(
            f"Irregular wavelength spacing (std={wl_spacing_std:.2f} nm)"
        )

    # Check for duplicate sample IDs
    unique_ids = len(set(source.sample_ids))
    if unique_ids < len(source.sample_ids):
        n_dups = len(source.sample_ids) - unique_ids
        report['warnings'].append(f"Found {n_dups} duplicate sample IDs")

    # Statistics
    report['statistics'] = {
        'x_mean': float(np.nanmean(source.X)),
        'x_std': float(np.nanstd(source.X)),
        'x_min': float(np.nanmin(source.X)),
        'x_max': float(np.nanmax(source.X)),
        'wavelength_spacing_mean': float(np.mean(wl_diffs)),
        'wavelength_spacing_std': float(wl_spacing_std)
    }

    if source.y is not None:
        report['statistics']['y_mean'] = float(np.nanmean(source.y))
        report['statistics']['y_std'] = float(np.nanstd(source.y))
        report['statistics']['y_min'] = float(np.nanmin(source.y))
        report['statistics']['y_max'] = float(np.nanmax(source.y))

    return report


def print_merge_summary(result: MergeResult) -> None:
    """
    Print a formatted summary of merge results.

    Parameters
    ----------
    result : MergeResult
        Merge result to summarize
    """
    print(f"\n{'='*60}")
    print("MERGE SUMMARY")
    print(f"{'='*60}")
    print(f"Strategy: {result.strategy}")
    print(f"Sources merged: {result.n_sources}")
    print(f"Total samples: {len(result.sample_ids)}")
    print(f"Total wavelengths: {len(result.wavelengths)}")
    print(f"Wavelength range: {result.wavelengths[0]:.2f} to {result.wavelengths[-1]:.2f} nm")
    print(f"Has target values: {result.y is not None}")
    if result.target_name:
        print(f"Target name: {result.target_name}")

    # Print strategy-specific details
    if 'nan_percent' in result.report:
        print(f"NaN content: {result.report['nan_percent']:.1f}%")

    if 'n_wavelengths_common' in result.report:
        original = result.report['n_wavelengths_original']
        common = result.report['n_wavelengths_common']
        print(f"Wavelengths retained: {common} of {max(original)} max")

    print(f"{'='*60}\n")
