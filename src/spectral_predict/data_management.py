"""
Data Management Module for Spectral Analysis

This module provides functionality for managing multiple spectral data sources,
including loading, merging, and manipulation operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import json
import warnings
from scipy.interpolate import interp1d


@dataclass
class DataSource:
    """Container for a single spectral data source."""
    source_id: str
    name: str
    path: str
    format_type: str  # 'asd', 'csv', 'spc', 'combined', 'excel'
    X: pd.DataFrame  # Spectral data (samples x wavelengths)
    y: Optional[pd.Series] = None  # Target values
    ref: Optional[pd.DataFrame] = None  # Reference/metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([]))
    n_samples: int = 0
    wavelength_range: Tuple[float, float] = (0.0, 0.0)
    load_timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        if self.X is not None and not self.X.empty:
            self.n_samples = len(self.X)
            if len(self.X.columns) > 0:
                try:
                    # Try to convert column names to wavelengths
                    self.wavelengths = np.array([float(col) for col in self.X.columns])
                    self.wavelength_range = (self.wavelengths.min(), self.wavelengths.max())
                except (ValueError, TypeError):
                    # Column names are not numeric wavelengths
                    self.wavelengths = np.arange(len(self.X.columns))
                    self.wavelength_range = (0, len(self.X.columns) - 1)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this data source."""
        return {
            'source_id': self.source_id,
            'name': self.name,
            'format': self.format_type,
            'n_samples': self.n_samples,
            'n_wavelengths': len(self.wavelengths),
            'wavelength_range': self.wavelength_range,
            'has_targets': self.y is not None,
            'has_metadata': self.ref is not None,
            'load_time': self.load_timestamp.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excludes data)."""
        return {
            'source_id': self.source_id,
            'name': self.name,
            'path': self.path,
            'format_type': self.format_type,
            'n_samples': self.n_samples,
            'n_wavelengths': len(self.wavelengths),
            'wavelength_range': list(self.wavelength_range),
            'has_targets': self.y is not None,
            'has_metadata': self.ref is not None,
            'metadata': self.metadata,
            'notes': self.notes,
            'load_timestamp': self.load_timestamp.isoformat(),
        }


@dataclass
class MergedDataset:
    """Container for merged spectral datasets."""
    sources: List[DataSource]
    X: pd.DataFrame
    y: Optional[pd.Series] = None
    ref: Optional[pd.DataFrame] = None
    merge_strategy: str = 'intersection'  # 'union', 'intersection', 'interpolation'
    merge_report: Dict[str, Any] = field(default_factory=dict)
    target_wavelengths: Optional[np.ndarray] = None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of merged dataset."""
        return {
            'n_sources': len(self.sources),
            'source_names': [s.name for s in self.sources],
            'merge_strategy': self.merge_strategy,
            'n_samples': len(self.X) if self.X is not None else 0,
            'n_wavelengths': len(self.X.columns) if self.X is not None else 0,
            'has_targets': self.y is not None,
            'has_metadata': self.ref is not None,
            'merge_report': self.merge_report,
        }


class DataSourceManager:
    """Manager class for handling multiple data sources and merging operations."""

    def __init__(self):
        """Initialize the data source manager."""
        self.sources: List[DataSource] = []
        self.merged_dataset: Optional[MergedDataset] = None
        self._source_counter = 0

    def add_source(self,
                   X: pd.DataFrame,
                   y: Optional[pd.Series] = None,
                   ref: Optional[pd.DataFrame] = None,
                   path: str = "",
                   format_type: str = "unknown",
                   name: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> DataSource:
        """
        Add a new data source to the manager.

        Parameters
        ----------
        X : pd.DataFrame
            Spectral data (samples x wavelengths)
        y : pd.Series, optional
            Target values
        ref : pd.DataFrame, optional
            Reference/metadata
        path : str
            File/directory path
        format_type : str
            Format type ('asd', 'csv', 'spc', etc.)
        name : str, optional
            User-friendly name for the source
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        DataSource
            The created data source object
        """
        self._source_counter += 1
        source_id = f"source_{self._source_counter:03d}"

        if name is None:
            name = f"Source {self._source_counter}"
            if path:
                name = os.path.basename(path)

        source = DataSource(
            source_id=source_id,
            name=name,
            path=path,
            format_type=format_type,
            X=X,
            y=y,
            ref=ref,
            metadata=metadata or {},
        )

        self.sources.append(source)
        return source

    def remove_source(self, source_id: str) -> bool:
        """
        Remove a data source by ID.

        Parameters
        ----------
        source_id : str
            The ID of the source to remove

        Returns
        -------
        bool
            True if source was removed, False if not found
        """
        for i, source in enumerate(self.sources):
            if source.source_id == source_id:
                self.sources.pop(i)
                return True
        return False

    def get_source(self, source_id: str) -> Optional[DataSource]:
        """Get a data source by ID."""
        for source in self.sources:
            if source.source_id == source_id:
                return source
        return None

    def clear_sources(self):
        """Clear all data sources."""
        self.sources = []
        self.merged_dataset = None
        self._source_counter = 0

    def merge_sources(self,
                     source_ids: Optional[List[str]] = None,
                     strategy: str = 'intersection',
                     handle_duplicates: str = 'error',
                     target_wavelengths: Optional[np.ndarray] = None) -> MergedDataset:
        """
        Merge multiple data sources.

        Parameters
        ----------
        source_ids : list of str, optional
            IDs of sources to merge. If None, merge all sources.
        strategy : str
            Merge strategy: 'intersection', 'union', 'interpolation'
        handle_duplicates : str
            How to handle duplicate sample IDs: 'error', 'keep_first', 'keep_last', 'rename'
        target_wavelengths : np.ndarray, optional
            Target wavelength grid for interpolation (required if strategy='interpolation')

        Returns
        -------
        MergedDataset
            The merged dataset
        """
        # Select sources to merge
        if source_ids is None:
            sources_to_merge = self.sources
        else:
            sources_to_merge = [s for s in self.sources if s.source_id in source_ids]

        if not sources_to_merge:
            raise ValueError("No sources to merge")

        if len(sources_to_merge) == 1:
            # Single source, just copy
            source = sources_to_merge[0]

            # Add datasource column even for single source
            ref_copy = source.ref.copy() if source.ref is not None else pd.DataFrame(index=source.X.index)
            datasource_label = f"A_{source.name}"
            ref_copy.insert(0, 'datasource', datasource_label)

            merged = MergedDataset(
                sources=[source],
                X=source.X.copy(),
                y=source.y.copy() if source.y is not None else None,
                ref=ref_copy,
                merge_strategy='single',
                merge_report={'message': 'Single source, no merge needed'},
            )
            self.merged_dataset = merged
            return merged

        # Determine wavelength grid based on strategy
        if strategy == 'intersection':
            merged_X, merged_y, merged_ref, wavelengths, report = self._merge_intersection(sources_to_merge, handle_duplicates)
        elif strategy == 'union':
            merged_X, merged_y, merged_ref, wavelengths, report = self._merge_union(sources_to_merge, handle_duplicates)
        elif strategy == 'interpolation':
            if target_wavelengths is None:
                # Use union of all wavelengths as target
                all_wavelengths = set()
                for source in sources_to_merge:
                    all_wavelengths.update(source.wavelengths)
                target_wavelengths = np.sort(list(all_wavelengths))
            merged_X, merged_y, merged_ref, wavelengths, report = self._merge_interpolation(
                sources_to_merge, target_wavelengths, handle_duplicates
            )
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        # Create merged dataset
        merged = MergedDataset(
            sources=sources_to_merge,
            X=merged_X,
            y=merged_y,
            ref=merged_ref,
            merge_strategy=strategy,
            merge_report=report,
            target_wavelengths=wavelengths,
        )

        self.merged_dataset = merged
        return merged

    def _merge_intersection(self, sources: List[DataSource], handle_duplicates: str) -> Tuple:
        """Merge using wavelength intersection (only common wavelengths)."""
        # Find common wavelengths
        common_wavelengths = set(sources[0].wavelengths)
        for source in sources[1:]:
            common_wavelengths = common_wavelengths.intersection(set(source.wavelengths))

        if not common_wavelengths:
            raise ValueError("No common wavelengths found across sources")

        common_wavelengths = np.sort(list(common_wavelengths))

        # Collect data from all sources
        all_X = []
        all_y = []
        all_ref = []
        sample_ids = []
        duplicates = []

        # Create datasource labels (A, B, C, etc.)
        source_labels = [chr(65 + i) for i in range(len(sources))]  # A, B, C, ...

        for source_idx, source in enumerate(sources):
            # Select only common wavelengths with flexible column matching
            wl_cols = []
            for w in common_wavelengths:
                # Try multiple formats to find matching column
                col_name = None
                candidates = [
                    str(w),                           # "350.0"
                    str(float(w)),                    # "350.0"
                    str(int(w)) if w == int(w) else None,  # "350" for whole numbers
                    w,                                # Direct (if already right type)
                ]

                for candidate in candidates:
                    if candidate is not None and candidate in source.X.columns:
                        col_name = candidate
                        break

                if col_name is not None:
                    wl_cols.append(col_name)
                else:
                    # Wavelength not found in this source - shouldn't happen for intersection
                    raise ValueError(f"Wavelength {w} not found in source {source.name} columns")

            X_subset = source.X[wl_cols]
            # Rename columns to standardized format
            X_subset = X_subset.copy()
            X_subset.columns = [str(w) for w in common_wavelengths]

            # Check for duplicate sample IDs
            for idx in X_subset.index:
                if idx in sample_ids:
                    duplicates.append((idx, source.name))
                    if handle_duplicates == 'error':
                        raise ValueError(f"Duplicate sample ID '{idx}' found in source '{source.name}'")
                    elif handle_duplicates == 'rename':
                        new_idx = f"{source.name}_{idx}"
                        X_subset.rename(index={idx: new_idx}, inplace=True)
                        if source.y is not None and idx in source.y.index:
                            source.y.rename(index={idx: new_idx}, inplace=True)
                        if source.ref is not None and idx in source.ref.index:
                            source.ref.rename(index={idx: new_idx}, inplace=True)
                    elif handle_duplicates == 'keep_first':
                        continue  # Skip this sample
                    # handle_duplicates == 'keep_last' will overwrite

                sample_ids.append(X_subset.index[0] if len(X_subset.index) > 0 else idx)

            all_X.append(X_subset)

            # Handle y - use NaN for sources without y
            if source.y is not None:
                y_subset = source.y.reindex(X_subset.index)
                all_y.append(y_subset)
            else:
                # Create empty y with NaN for this source
                y_subset = pd.Series(np.nan, index=X_subset.index)
                all_y.append(y_subset)

            # Handle ref - use empty columns for sources without ref
            if source.ref is not None:
                ref_subset = source.ref.reindex(X_subset.index)
            else:
                # Create empty ref for this source (will be filled with NaN when concatenated)
                ref_subset = pd.DataFrame(index=X_subset.index)

            # Add datasource column to track origin
            datasource_label = f"{source_labels[source_idx]}_{source.name}"
            ref_subset = ref_subset.copy()
            ref_subset.insert(0, 'datasource', datasource_label)
            all_ref.append(ref_subset)

        # Combine all data - pandas will align columns and fill missing with NaN
        merged_X = pd.concat(all_X, axis=0)
        merged_y = pd.concat(all_y, axis=0) if all_y else None
        # Use sort=False to preserve column order, fill missing columns with NaN
        merged_ref = pd.concat(all_ref, axis=0, sort=False) if all_ref else None

        report = {
            'strategy': 'intersection',
            'n_common_wavelengths': len(common_wavelengths),
            'n_total_samples': len(merged_X),
            'n_duplicates': len(duplicates),
            'duplicate_handling': handle_duplicates,
        }

        return merged_X, merged_y, merged_ref, common_wavelengths, report

    def _merge_union(self, sources: List[DataSource], handle_duplicates: str) -> Tuple:
        """Merge using wavelength union (all wavelengths, NaN for missing)."""
        # Collect all unique wavelengths
        all_wavelengths = set()
        for source in sources:
            all_wavelengths.update(source.wavelengths)
        all_wavelengths = np.sort(list(all_wavelengths))

        # Prepare data structures
        all_X = []
        all_y = []
        all_ref = []
        sample_ids = []
        duplicates = []

        # Create datasource labels (A, B, C, etc.)
        source_labels = [chr(65 + i) for i in range(len(sources))]  # A, B, C, ...

        for source_idx, source in enumerate(sources):
            # Create DataFrame with all wavelengths
            X_full = pd.DataFrame(index=source.X.index, columns=[str(w) for w in all_wavelengths])

            # Fill in available wavelengths with flexible column matching
            for w in source.wavelengths:
                if w in all_wavelengths:
                    # Try multiple formats to find matching column
                    col_name = None
                    candidates = [
                        str(w),
                        str(float(w)),
                        str(int(w)) if w == int(w) else None,
                        w,
                    ]

                    for candidate in candidates:
                        if candidate is not None and candidate in source.X.columns:
                            col_name = candidate
                            break

                    if col_name is not None:
                        X_full[str(w)] = source.X[col_name]

            # Handle duplicate sample IDs
            for idx in X_full.index:
                if idx in sample_ids:
                    duplicates.append((idx, source.name))
                    if handle_duplicates == 'error':
                        raise ValueError(f"Duplicate sample ID '{idx}' found in source '{source.name}'")
                    elif handle_duplicates == 'rename':
                        new_idx = f"{source.name}_{idx}"
                        X_full.rename(index={idx: new_idx}, inplace=True)
                        if source.y is not None and idx in source.y.index:
                            source.y.rename(index={idx: new_idx}, inplace=True)
                        if source.ref is not None and idx in source.ref.index:
                            source.ref.rename(index={idx: new_idx}, inplace=True)
                    elif handle_duplicates == 'keep_first':
                        continue

                sample_ids.append(X_full.index[0] if len(X_full.index) > 0 else idx)

            all_X.append(X_full)

            # Handle y - use NaN for sources without y
            if source.y is not None:
                y_subset = source.y.reindex(X_full.index)
                all_y.append(y_subset)
            else:
                y_subset = pd.Series(np.nan, index=X_full.index)
                all_y.append(y_subset)

            # Handle ref - use empty columns for sources without ref
            if source.ref is not None:
                ref_subset = source.ref.reindex(X_full.index)
            else:
                ref_subset = pd.DataFrame(index=X_full.index)

            # Add datasource column to track origin
            datasource_label = f"{source_labels[source_idx]}_{source.name}"
            ref_subset = ref_subset.copy()
            ref_subset.insert(0, 'datasource', datasource_label)
            all_ref.append(ref_subset)

        # Combine all data
        merged_X = pd.concat(all_X, axis=0)
        merged_y = pd.concat(all_y, axis=0) if all_y else None
        merged_ref = pd.concat(all_ref, axis=0, sort=False) if all_ref else None

        # Count NaN values
        nan_count = merged_X.isna().sum().sum()
        total_values = merged_X.shape[0] * merged_X.shape[1]

        report = {
            'strategy': 'union',
            'n_total_wavelengths': len(all_wavelengths),
            'n_total_samples': len(merged_X),
            'n_duplicates': len(duplicates),
            'duplicate_handling': handle_duplicates,
            'n_missing_values': nan_count,
            'percent_missing': (nan_count / total_values * 100) if total_values > 0 else 0,
        }

        return merged_X, merged_y, merged_ref, all_wavelengths, report

    def _merge_interpolation(self, sources: List[DataSource], target_wavelengths: np.ndarray,
                           handle_duplicates: str) -> Tuple:
        """Merge using interpolation to common wavelength grid."""
        # Prepare data structures
        all_X = []
        all_y = []
        all_ref = []
        sample_ids = []
        duplicates = []
        interpolation_warnings = []

        # Create datasource labels (A, B, C, etc.)
        source_labels = [chr(65 + i) for i in range(len(sources))]  # A, B, C, ...

        for source_idx, source in enumerate(sources):
            # Create interpolated DataFrame
            X_interp = pd.DataFrame(index=source.X.index, columns=[str(w) for w in target_wavelengths])

            # Interpolate each sample
            for idx in source.X.index:
                spectrum = source.X.loc[idx].values

                # Check if we need to extrapolate
                source_min, source_max = source.wavelengths.min(), source.wavelengths.max()
                target_min, target_max = target_wavelengths.min(), target_wavelengths.max()

                if target_min < source_min or target_max > source_max:
                    interpolation_warnings.append(
                        f"Extrapolation required for {source.name} sample {idx}: "
                        f"source range [{source_min:.1f}, {source_max:.1f}], "
                        f"target range [{target_min:.1f}, {target_max:.1f}]"
                    )

                # Perform interpolation
                try:
                    # Use linear interpolation (conservative, avoids artifacts)
                    f_interp = interp1d(source.wavelengths, spectrum,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value='extrapolate')
                    X_interp.loc[idx] = f_interp(target_wavelengths)
                except Exception as e:
                    warnings.warn(f"Interpolation failed for {source.name} sample {idx}: {e}")
                    X_interp.loc[idx] = np.nan

            # Handle duplicate sample IDs
            for idx in X_interp.index:
                if idx in sample_ids:
                    duplicates.append((idx, source.name))
                    if handle_duplicates == 'error':
                        raise ValueError(f"Duplicate sample ID '{idx}' found in source '{source.name}'")
                    elif handle_duplicates == 'rename':
                        new_idx = f"{source.name}_{idx}"
                        X_interp.rename(index={idx: new_idx}, inplace=True)
                        if source.y is not None and idx in source.y.index:
                            source.y.rename(index={idx: new_idx}, inplace=True)
                        if source.ref is not None and idx in source.ref.index:
                            source.ref.rename(index={idx: new_idx}, inplace=True)
                    elif handle_duplicates == 'keep_first':
                        continue

                sample_ids.append(X_interp.index[0] if len(X_interp.index) > 0 else idx)

            all_X.append(X_interp)

            # Handle y - use NaN for sources without y
            if source.y is not None:
                y_subset = source.y.reindex(X_interp.index)
                all_y.append(y_subset)
            else:
                y_subset = pd.Series(np.nan, index=X_interp.index)
                all_y.append(y_subset)

            # Handle ref - use empty columns for sources without ref
            if source.ref is not None:
                ref_subset = source.ref.reindex(X_interp.index)
            else:
                ref_subset = pd.DataFrame(index=X_interp.index)

            # Add datasource column to track origin
            datasource_label = f"{source_labels[source_idx]}_{source.name}"
            ref_subset = ref_subset.copy()
            ref_subset.insert(0, 'datasource', datasource_label)
            all_ref.append(ref_subset)

        # Combine all data
        merged_X = pd.concat(all_X, axis=0)
        merged_y = pd.concat(all_y, axis=0) if all_y else None
        merged_ref = pd.concat(all_ref, axis=0, sort=False) if all_ref else None

        report = {
            'strategy': 'interpolation',
            'n_target_wavelengths': len(target_wavelengths),
            'n_total_samples': len(merged_X),
            'n_duplicates': len(duplicates),
            'duplicate_handling': handle_duplicates,
            'n_interpolation_warnings': len(interpolation_warnings),
            'warnings': interpolation_warnings[:10] if interpolation_warnings else [],  # First 10 warnings
        }

        return merged_X, merged_y, merged_ref, target_wavelengths, report

    def filter_samples(self,
                       X: pd.DataFrame,
                       y: Optional[pd.Series] = None,
                       ref: Optional[pd.DataFrame] = None,
                       filter_type: str = 'regex',
                       filter_value: Any = None,
                       column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.DataFrame]]:
        """
        Filter samples based on criteria.

        Parameters
        ----------
        X : pd.DataFrame
            Spectral data to filter
        y : pd.Series, optional
            Target values to filter
        ref : pd.DataFrame, optional
            Reference data to filter
        filter_type : str
            Type of filter: 'regex', 'range', 'list'
        filter_value : Any
            Filter value (regex pattern, range tuple, or list)
        column : str, optional
            Column to filter on (for ref DataFrame)

        Returns
        -------
        tuple
            Filtered (X, y, ref)
        """
        if filter_type == 'regex':
            import re
            pattern = re.compile(filter_value)
            if column and ref is not None and column in ref.columns:
                # Filter by reference column
                mask = ref[column].astype(str).apply(lambda x: bool(pattern.search(x)))
                filtered_indices = ref[mask].index
            else:
                # Filter by sample ID
                mask = X.index.to_series().apply(lambda x: bool(pattern.search(str(x))))
                filtered_indices = X[mask].index

        elif filter_type == 'range' and column and ref is not None:
            # Filter by numeric range
            min_val, max_val = filter_value
            mask = (ref[column] >= min_val) & (ref[column] <= max_val)
            filtered_indices = ref[mask].index

        elif filter_type == 'list':
            # Filter by list of sample IDs
            filtered_indices = [idx for idx in filter_value if idx in X.index]

        else:
            # No filter, return original
            return X, y, ref

        # Apply filter
        X_filtered = X.loc[filtered_indices]
        y_filtered = y.loc[filtered_indices] if y is not None else None
        ref_filtered = ref.loc[filtered_indices] if ref is not None else None

        return X_filtered, y_filtered, ref_filtered

    def trim_wavelengths(self,
                        X: pd.DataFrame,
                        min_wavelength: Optional[float] = None,
                        max_wavelength: Optional[float] = None) -> pd.DataFrame:
        """
        Trim wavelength range.

        Parameters
        ----------
        X : pd.DataFrame
            Spectral data
        min_wavelength : float, optional
            Minimum wavelength to keep
        max_wavelength : float, optional
            Maximum wavelength to keep

        Returns
        -------
        pd.DataFrame
            Trimmed spectral data
        """
        # Get wavelengths from columns
        wavelengths = np.array([float(col) for col in X.columns])

        # Create mask for wavelengths to keep
        mask = np.ones(len(wavelengths), dtype=bool)
        if min_wavelength is not None:
            mask &= wavelengths >= min_wavelength
        if max_wavelength is not None:
            mask &= wavelengths <= max_wavelength

        # Select columns
        selected_columns = X.columns[mask]
        return X[selected_columns]

    def export_to_csv(self,
                     X: pd.DataFrame,
                     y: Optional[pd.Series] = None,
                     ref: Optional[pd.DataFrame] = None,
                     filepath: str = "merged_data.csv",
                     include_targets: bool = True,
                     include_metadata: bool = True):
        """Export merged data to CSV."""
        # Combine data
        export_df = X.copy()

        if include_targets and y is not None:
            export_df.insert(0, 'Target', y)

        if include_metadata and ref is not None:
            # Add metadata columns at the beginning
            for col in ref.columns:
                export_df.insert(0, col, ref[col])

        # Export
        export_df.to_csv(filepath)
        print(f"Data exported to {filepath}")

    def save_configuration(self, filepath: str = "data_sources.json"):
        """Save data source configuration to JSON."""
        config = {
            'sources': [source.to_dict() for source in self.sources],
            'n_sources': len(self.sources),
            'has_merged': self.merged_dataset is not None,
        }

        if self.merged_dataset:
            config['merged_summary'] = self.merged_dataset.get_summary()

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"Configuration saved to {filepath}")

    def load_configuration(self, filepath: str = "data_sources.json"):
        """Load data source configuration from JSON."""
        with open(filepath, 'r') as f:
            config = json.load(f)

        print(f"Loaded configuration with {config['n_sources']} sources")
        if config.get('has_merged'):
            print("Previous merge detected:", config['merged_summary'])

        return config