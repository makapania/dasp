"""
Data Management Panel for Spectral Predict v3.

Multi-source data management with merge strategies for combining
spectral datasets from different sources.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..theme import COLORS
from ...core.data_management import merge_sources, DataSource as CoreDataSource, MergeResult as CoreMergeResult


@dataclass
class DataSource:
    """Represents a single loaded data source."""
    source_id: str
    name: str
    path: str
    X: np.ndarray  # (n_samples, n_wavelengths)
    wavelengths: np.ndarray
    sample_ids: List[str]
    y: Optional[np.ndarray] = None
    target_name: Optional[str] = None
    metadata_columns: Dict[str, List[Any]] = field(default_factory=dict)  # Additional metadata columns
    n_samples: int = 0
    n_wavelengths: int = 0
    load_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.n_samples = self.X.shape[0]
        self.n_wavelengths = self.X.shape[1]

    def get_wavelength_range(self) -> Tuple[float, float]:
        """Return min and max wavelength."""
        return (float(np.min(self.wavelengths)), float(np.max(self.wavelengths)))


@dataclass
class MergeResult:
    """Result of merging multiple data sources."""
    X: np.ndarray
    wavelengths: np.ndarray
    sample_ids: List[str]
    y: Optional[np.ndarray] = None
    target_name: Optional[str] = None
    datasource: Optional[List[str]] = None  # Source label for each sample (e.g., "A_bone", "B_corn")
    metadata_columns: Dict[str, List[Any]] = field(default_factory=dict)  # Merged metadata columns
    strategy: str = ""
    n_sources: int = 0
    report: Dict[str, Any] = field(default_factory=dict)


class DataManagementPanel:
    """
    Panel for managing multiple data sources and merging them.

    Features:
    - Add/remove multiple data sources
    - Preview source details
    - Merge strategies: intersection, union, interpolation
    - Duplicate handling: error, keep_first, keep_last, rename
    - Preview merged results
    """

    def __init__(
        self,
        parent_tag: str,
        on_merge_complete: Optional[Callable[[MergeResult], None]] = None,
        on_load_source: Optional[Callable[[], None]] = None,
        on_load_folder: Optional[Callable[[], None]] = None,
        on_load_file: Optional[Callable[[], None]] = None
    ):
        """
        Initialize data management panel.

        Parameters
        ----------
        parent_tag : str
            DearPyGui tag of parent container
        on_merge_complete : callable, optional
            Callback when merge is complete, receives MergeResult
        on_load_source : callable, optional
            Legacy callback (unused)
        on_load_folder : callable, optional
            Callback to open folder dialog (for spectral files + Y file)
        on_load_file : callable, optional
            Callback to open file dialog (for combined CSV/XLS)
        """
        self.parent_tag = parent_tag
        self.on_merge_complete = on_merge_complete
        self.on_load_folder = on_load_folder
        self.on_load_file = on_load_file

        # Data sources
        self.sources: List[DataSource] = []
        self._source_counter = 0

        # Selected source for preview
        self._selected_source_id: Optional[str] = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the data management panel UI."""
        with dpg.child_window(parent=self.parent_tag, tag=f"{self.parent_tag}_datamgmt_main"):
            # Header
            dpg.add_text("Data Management")
            dpg.add_text(
                "Import multiple data sources and merge them with various strategies.",
                color=(150, 150, 150)
            )
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Two-column layout
            with dpg.group(horizontal=True):
                # Left: Source list and controls
                with dpg.child_window(width=350, height=-80, border=True):
                    self._build_sources_panel()

                dpg.add_spacer(width=10)

                # Right: Merge options and preview
                with dpg.child_window(width=-1, height=-80, border=True):
                    self._build_merge_panel()

            # Bottom: Action buttons
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Merge Sources",
                    callback=self._on_merge,
                    tag=f"{self.parent_tag}_merge_btn",
                    enabled=False
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="Clear All",
                    callback=self._on_clear_all,
                    tag=f"{self.parent_tag}_clear_btn"
                )
                dpg.add_spacer(width=30)
                dpg.add_text("", tag=f"{self.parent_tag}_datamgmt_status", color=(100, 200, 100))

    def _build_sources_panel(self):
        """Build the sources list panel on the left."""
        dpg.add_text("Data Sources", color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Add source buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="+ Add Folder",
                callback=self._on_add_folder,
                tag=f"{self.parent_tag}_add_folder_btn"
            )
            dpg.add_spacer(width=5)
            dpg.add_button(
                label="+ Add File",
                callback=self._on_add_file,
                tag=f"{self.parent_tag}_add_file_btn"
            )
            dpg.add_spacer(width=10)
            dpg.add_button(
                label="Remove",
                callback=self._on_remove_source,
                tag=f"{self.parent_tag}_remove_source_btn",
                enabled=False
            )

        dpg.add_spacer(height=10)

        # Sources list
        dpg.add_text("Loaded Sources:", color=(180, 180, 180))
        with dpg.child_window(
            tag=f"{self.parent_tag}_sources_list",
            height=200,
            border=True
        ):
            dpg.add_text(
                "(No sources loaded)",
                tag=f"{self.parent_tag}_no_sources_text",
                color=(120, 120, 120)
            )

        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Source details
        dpg.add_text("Source Details:", color=(200, 200, 255))
        with dpg.child_window(
            tag=f"{self.parent_tag}_source_details",
            height=-1,
            border=True
        ):
            dpg.add_text(
                "Select a source to view details",
                tag=f"{self.parent_tag}_source_details_text",
                color=(120, 120, 120),
                wrap=320
            )

    def _build_merge_panel(self):
        """Build the merge options and preview panel on the right."""
        # Merge options
        dpg.add_text("Merge Options", color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            # Merge strategy
            with dpg.group():
                dpg.add_text("Merge Strategy:")
                dpg.add_combo(
                    items=["Intersection", "Union", "Interpolation"],
                    default_value="Intersection",
                    tag=f"{self.parent_tag}_merge_strategy",
                    width=180,
                    callback=self._on_strategy_change
                )
                dpg.add_spacer(height=5)
                dpg.add_text(
                    "Uses only common wavelengths",
                    tag=f"{self.parent_tag}_strategy_desc",
                    color=(120, 120, 120),
                    wrap=180
                )

            dpg.add_spacer(width=30)

            # Duplicate handling
            with dpg.group():
                dpg.add_text("Duplicate Sample IDs:")
                dpg.add_combo(
                    items=["Error", "Keep First", "Keep Last", "Rename"],
                    default_value="Rename",
                    tag=f"{self.parent_tag}_duplicate_handling",
                    width=150
                )

        dpg.add_spacer(height=10)

        # Interpolation options (hidden by default)
        with dpg.group(tag=f"{self.parent_tag}_interp_options", show=False):
            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_text("Interpolation Settings:", color=(180, 180, 180))
            with dpg.group(horizontal=True):
                dpg.add_text("Target wavelength step:")
                dpg.add_input_float(
                    default_value=1.0,
                    tag=f"{self.parent_tag}_interp_step",
                    width=100,
                    min_value=0.1,
                    max_value=100.0
                )
                dpg.add_text("nm")

        dpg.add_spacer(height=15)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Preview section
        dpg.add_text("Merge Preview", color=(200, 200, 255))
        dpg.add_spacer(height=5)

        # Preview stats
        with dpg.child_window(
            tag=f"{self.parent_tag}_merge_preview",
            height=-1,
            border=True
        ):
            dpg.add_text(
                "Add at least 2 sources to preview merge",
                tag=f"{self.parent_tag}_preview_text",
                color=(120, 120, 120),
                wrap=400
            )

    def _on_strategy_change(self, sender=None, app_data=None):
        """Handle merge strategy selection change."""
        strategy = dpg.get_value(f"{self.parent_tag}_merge_strategy")

        # Update description
        descriptions = {
            "Intersection": "Uses only common wavelengths shared by all sources",
            "Union": "Includes all wavelengths, fills missing with NaN",
            "Interpolation": "Interpolates all sources to a common wavelength grid"
        }
        dpg.set_value(
            f"{self.parent_tag}_strategy_desc",
            descriptions.get(strategy, "")
        )

        # Show/hide interpolation options
        dpg.configure_item(
            f"{self.parent_tag}_interp_options",
            show=(strategy == "Interpolation")
        )

        # Update preview
        self._update_preview()

    def _on_add_folder(self, sender=None, app_data=None):
        """Handle add folder button click."""
        if self.on_load_folder:
            self.on_load_folder()
        else:
            self._set_status("No folder load callback configured", error=True)

    def _on_add_file(self, sender=None, app_data=None):
        """Handle add file button click."""
        if self.on_load_file:
            self.on_load_file()
        else:
            self._set_status("No file load callback configured", error=True)

    def _on_remove_source(self, sender=None, app_data=None):
        """Handle remove source button click."""
        if self._selected_source_id:
            self.remove_source(self._selected_source_id)

    def _on_clear_all(self, sender=None, app_data=None):
        """Clear all sources."""
        self.sources.clear()
        self._selected_source_id = None
        self._source_counter = 0
        self._refresh_sources_list()
        self._update_source_details()
        self._update_preview()
        self._update_button_states()
        self._set_status("All sources cleared")

    def add_source(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
        sample_ids: List[str],
        name: str,
        path: str = "",
        y: Optional[np.ndarray] = None,
        target_name: Optional[str] = None,
        metadata_columns: Optional[Dict[str, List[Any]]] = None
    ) -> str:
        """
        Add a new data source.

        Parameters
        ----------
        X : ndarray
            Spectral data (n_samples, n_wavelengths)
        wavelengths : ndarray
            Wavelength values
        sample_ids : list
            Sample identifiers
        name : str
            Display name for the source
        path : str, optional
            File path
        y : ndarray, optional
            Target values
        target_name : str, optional
            Name of target variable
        metadata_columns : dict, optional
            Additional metadata columns {column_name: [values]}

        Returns
        -------
        str
            Source ID
        """
        self._source_counter += 1
        source_id = f"source_{self._source_counter:03d}"

        source = DataSource(
            source_id=source_id,
            name=name,
            path=path,
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            y=y,
            target_name=target_name,
            metadata_columns=metadata_columns or {}
        )

        self.sources.append(source)
        self._refresh_sources_list()
        self._update_preview()
        self._update_button_states()
        self._set_status(f"Added: {name}")

        return source_id

    def remove_source(self, source_id: str) -> bool:
        """Remove a source by ID."""
        for i, source in enumerate(self.sources):
            if source.source_id == source_id:
                removed_name = source.name
                self.sources.pop(i)

                if self._selected_source_id == source_id:
                    self._selected_source_id = None

                self._refresh_sources_list()
                self._update_source_details()
                self._update_preview()
                self._update_button_states()
                self._set_status(f"Removed: {removed_name}")
                return True

        return False

    def _refresh_sources_list(self):
        """Refresh the sources list display."""
        list_tag = f"{self.parent_tag}_sources_list"

        # Clear existing items
        children = dpg.get_item_children(list_tag, 1)
        if children:
            for child in children:
                dpg.delete_item(child)

        if not self.sources:
            dpg.add_text(
                "(No sources loaded)",
                parent=list_tag,
                tag=f"{self.parent_tag}_no_sources_text",
                color=(120, 120, 120)
            )
            return

        # Add source items
        for source in self.sources:
            wl_range = source.get_wavelength_range()
            with dpg.group(parent=list_tag, horizontal=False):
                dpg.add_selectable(
                    label=f"{source.name}",
                    callback=self._on_source_select,
                    user_data=source.source_id,
                    tag=f"{self.parent_tag}_sel_{source.source_id}"
                )
                dpg.add_text(
                    f"   {source.n_samples} samples, {source.n_wavelengths} wavelengths",
                    color=(140, 140, 140)
                )
                dpg.add_text(
                    f"   Range: {wl_range[0]:.1f} - {wl_range[1]:.1f} nm",
                    color=(140, 140, 140)
                )
                dpg.add_spacer(height=5)

    def _on_source_select(self, sender, app_data, user_data):
        """Handle source selection."""
        source_id = user_data
        self._selected_source_id = source_id

        # Update selection visual
        for source in self.sources:
            sel_tag = f"{self.parent_tag}_sel_{source.source_id}"
            if dpg.does_item_exist(sel_tag):
                dpg.set_value(sel_tag, source.source_id == source_id)

        self._update_source_details()
        self._update_button_states()

    def _update_source_details(self):
        """Update source details display."""
        details_tag = f"{self.parent_tag}_source_details_text"

        if not self._selected_source_id:
            dpg.set_value(details_tag, "Select a source to view details")
            return

        # Find selected source
        source = None
        for s in self.sources:
            if s.source_id == self._selected_source_id:
                source = s
                break

        if not source:
            dpg.set_value(details_tag, "Source not found")
            return

        wl_range = source.get_wavelength_range()
        details = [
            f"Name: {source.name}",
            f"Path: {source.path or 'N/A'}",
            f"",
            f"Samples: {source.n_samples}",
            f"Wavelengths: {source.n_wavelengths}",
            f"Range: {wl_range[0]:.1f} - {wl_range[1]:.1f} nm",
            f"",
            f"Has target: {'Yes' if source.y is not None else 'No'}",
        ]
        if source.target_name:
            details.append(f"Target: {source.target_name}")

        details.append(f"")
        details.append(f"Loaded: {source.load_timestamp.strftime('%H:%M:%S')}")

        dpg.set_value(details_tag, "\n".join(details))

    def _update_preview(self):
        """Update merge preview."""
        preview_tag = f"{self.parent_tag}_preview_text"

        if len(self.sources) < 2:
            dpg.set_value(
                preview_tag,
                "Add at least 2 sources to preview merge\n\n"
                f"Current sources: {len(self.sources)}"
            )
            return

        strategy = dpg.get_value(f"{self.parent_tag}_merge_strategy")

        # Calculate preview stats
        all_wavelengths = [set(s.wavelengths.astype(float)) for s in self.sources]
        common_wl = all_wavelengths[0]
        union_wl = all_wavelengths[0]

        for wl_set in all_wavelengths[1:]:
            common_wl = common_wl.intersection(wl_set)
            union_wl = union_wl.union(wl_set)

        total_samples = sum(s.n_samples for s in self.sources)

        # Check for duplicate sample IDs
        all_ids = []
        for s in self.sources:
            all_ids.extend(s.sample_ids)
        unique_ids = set(all_ids)
        n_duplicates = len(all_ids) - len(unique_ids)

        # Build preview text
        lines = [
            f"Sources: {len(self.sources)}",
            f"Total samples: {total_samples}",
            f"Duplicate IDs: {n_duplicates}",
            "",
            f"Wavelength analysis:",
            f"  Common: {len(common_wl)}",
            f"  Union: {len(union_wl)}",
            ""
        ]

        if strategy == "Intersection":
            lines.append(f"Result will have {len(common_wl)} wavelengths")
            if len(common_wl) == 0:
                lines.append("WARNING: No common wavelengths!")
        elif strategy == "Union":
            lines.append(f"Result will have {len(union_wl)} wavelengths")
            missing_pct = (1 - len(common_wl) / len(union_wl)) * 100 if union_wl else 0
            lines.append(f"~{missing_pct:.1f}% missing values expected")
        else:  # Interpolation
            step = dpg.get_value(f"{self.parent_tag}_interp_step")
            min_wl = min(min(wl) for wl in all_wavelengths)
            max_wl = max(max(wl) for wl in all_wavelengths)
            n_target = int((max_wl - min_wl) / step) + 1
            lines.append(f"Target grid: {min_wl:.1f} - {max_wl:.1f} nm")
            lines.append(f"Result will have ~{n_target} wavelengths")

        dpg.set_value(preview_tag, "\n".join(lines))

    def _update_button_states(self):
        """Update button enabled states."""
        has_sources = len(self.sources) > 0
        has_selection = self._selected_source_id is not None
        can_merge = len(self.sources) >= 2

        dpg.configure_item(f"{self.parent_tag}_remove_source_btn", enabled=has_selection)
        dpg.configure_item(f"{self.parent_tag}_merge_btn", enabled=can_merge)

    def _on_merge(self, sender=None, app_data=None):
        """Execute merge operation using core data_management module."""
        if len(self.sources) < 2:
            self._set_status("Need at least 2 sources to merge", error=True)
            return

        strategy = dpg.get_value(f"{self.parent_tag}_merge_strategy").lower()
        dup_handling = dpg.get_value(f"{self.parent_tag}_duplicate_handling").lower().replace(" ", "_")
        step = dpg.get_value(f"{self.parent_tag}_interp_step")

        try:
            # Convert panel DataSource objects to core DataSource objects
            core_sources = []
            for source in self.sources:
                core_src = CoreDataSource(
                    source_id=source.source_id,
                    name=source.name,
                    path=source.path,
                    X=source.X,
                    wavelengths=source.wavelengths,
                    sample_ids=source.sample_ids,
                    y=source.y,
                    target_name=source.target_name,
                    metadata_columns=source.metadata_columns
                )
                core_sources.append(core_src)

            # Call core merge_sources function
            core_result = merge_sources(
                sources=core_sources,
                strategy=strategy,
                dup_handling=dup_handling,
                wavelength_step=step
            )

            # Convert core MergeResult to panel MergeResult
            result = MergeResult(
                X=core_result.X,
                wavelengths=core_result.wavelengths,
                sample_ids=core_result.sample_ids,
                y=core_result.y,
                target_name=core_result.target_name,
                datasource=core_result.datasource,
                metadata_columns=core_result.metadata_columns,
                strategy=core_result.strategy,
                n_sources=core_result.n_sources,
                report=core_result.report
            )

            self._set_status(
                f"Merged: {result.X.shape[0]} samples, {result.X.shape[1]} wavelengths"
            )

            if self.on_merge_complete:
                self.on_merge_complete(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set_status(f"Merge failed: {str(e)}", error=True)

    def _merge_intersection(self, dup_handling: str) -> MergeResult:
        """Merge sources using intersection of wavelengths."""
        # Find common wavelengths
        common_wl = set(self.sources[0].wavelengths.astype(float))
        for source in self.sources[1:]:
            common_wl = common_wl.intersection(set(source.wavelengths.astype(float)))

        if not common_wl:
            raise ValueError("No common wavelengths found across sources")

        common_wl = np.array(sorted(common_wl))

        # Collect data
        X_parts = []
        y_parts = []
        id_parts = []
        seen_ids = set()
        n_duplicates = 0

        for source in self.sources:
            # Find indices for common wavelengths (vectorized for speed)
            source_wl = source.wavelengths.astype(float)
            # Use searchsorted for O(n log n) instead of O(n*m) loop
            sorted_indices = np.argsort(source_wl)
            sorted_wl = source_wl[sorted_indices]
            insert_positions = np.searchsorted(sorted_wl, common_wl)
            # Clip to valid range and find closest
            insert_positions = np.clip(insert_positions, 0, len(sorted_wl) - 1)
            # Map back to original indices
            wl_indices = sorted_indices[insert_positions]

            X_subset = source.X[:, wl_indices]

            # Check Y array validity
            has_valid_y = (source.y is not None and
                          len(source.y) == len(source.sample_ids))

            for i, sample_id in enumerate(source.sample_ids):
                if sample_id in seen_ids:
                    n_duplicates += 1
                    if dup_handling == "error":
                        raise ValueError(f"Duplicate sample ID: {sample_id}")
                    elif dup_handling == "keep_first":
                        continue
                    elif dup_handling == "rename":
                        sample_id = f"{source.name}_{sample_id}"
                    # keep_last: will overwrite

                X_parts.append(X_subset[i:i+1, :])
                id_parts.append(sample_id)
                seen_ids.add(sample_id)

                if has_valid_y:
                    y_parts.append(source.y[i])

        X_merged = np.vstack(X_parts)
        y_merged = np.array(y_parts) if y_parts else None

        return MergeResult(
            X=X_merged,
            wavelengths=common_wl,
            sample_ids=id_parts,
            y=y_merged,
            target_name=self.sources[0].target_name,
            strategy="intersection",
            n_sources=len(self.sources),
            report={
                "n_wavelengths": len(common_wl),
                "n_samples": len(id_parts),
                "n_duplicates": n_duplicates,
                "duplicate_handling": dup_handling
            }
        )

    def _merge_union(self, dup_handling: str) -> MergeResult:
        """Merge sources using union of wavelengths (NaN for missing)."""
        # Collect all wavelengths
        all_wl = set()
        for source in self.sources:
            all_wl.update(source.wavelengths.astype(float))

        all_wl = np.array(sorted(all_wl))

        # Collect data
        X_parts = []
        y_parts = []
        id_parts = []
        seen_ids = set()
        n_duplicates = 0
        n_missing = 0

        for source in self.sources:
            source_wl_arr = source.wavelengths.astype(float)

            # Create mapping from all wavelengths to source wavelengths (vectorized)
            wl_map = {wl: i for i, wl in enumerate(source_wl_arr)}

            # Pre-compute which columns in all_wl exist in this source
            col_mapping = np.array([wl_map.get(wl, -1) for wl in all_wl])
            valid_cols = col_mapping >= 0
            n_missing_per_sample = np.sum(~valid_cols)

            # Check Y array validity
            has_valid_y = (source.y is not None and
                          len(source.y) == len(source.sample_ids))

            for i, sample_id in enumerate(source.sample_ids):
                if sample_id in seen_ids:
                    n_duplicates += 1
                    if dup_handling == "error":
                        raise ValueError(f"Duplicate sample ID: {sample_id}")
                    elif dup_handling == "keep_first":
                        continue
                    elif dup_handling == "rename":
                        sample_id = f"{source.name}_{sample_id}"

                # Build row with NaN for missing wavelengths (vectorized)
                row = np.full(len(all_wl), np.nan)
                row[valid_cols] = source.X[i, col_mapping[valid_cols]]
                n_missing += n_missing_per_sample

                X_parts.append(row.reshape(1, -1))
                id_parts.append(sample_id)
                seen_ids.add(sample_id)

                if has_valid_y:
                    y_parts.append(source.y[i])

        X_merged = np.vstack(X_parts)
        y_merged = np.array(y_parts) if y_parts else None

        total_values = X_merged.size
        pct_missing = (n_missing / total_values * 100) if total_values > 0 else 0

        return MergeResult(
            X=X_merged,
            wavelengths=all_wl,
            sample_ids=id_parts,
            y=y_merged,
            target_name=self.sources[0].target_name,
            strategy="union",
            n_sources=len(self.sources),
            report={
                "n_wavelengths": len(all_wl),
                "n_samples": len(id_parts),
                "n_duplicates": n_duplicates,
                "n_missing_values": n_missing,
                "percent_missing": pct_missing,
                "duplicate_handling": dup_handling
            }
        )

    def _merge_interpolation(self, dup_handling: str, step: float) -> MergeResult:
        """Merge sources by interpolating to common wavelength grid."""
        from scipy.interpolate import interp1d

        # Determine target wavelength range
        min_wl = min(s.wavelengths.min() for s in self.sources)
        max_wl = max(s.wavelengths.max() for s in self.sources)
        target_wl = np.arange(min_wl, max_wl + step, step)

        # Collect data
        X_parts = []
        y_parts = []
        id_parts = []
        seen_ids = set()
        n_duplicates = 0
        n_warnings = 0

        for source in self.sources:
            # Check Y array validity
            has_valid_y = (source.y is not None and
                          len(source.y) == len(source.sample_ids))

            for i, sample_id in enumerate(source.sample_ids):
                if sample_id in seen_ids:
                    n_duplicates += 1
                    if dup_handling == "error":
                        raise ValueError(f"Duplicate sample ID: {sample_id}")
                    elif dup_handling == "keep_first":
                        continue
                    elif dup_handling == "rename":
                        sample_id = f"{source.name}_{sample_id}"

                # Interpolate this spectrum
                try:
                    f = interp1d(
                        source.wavelengths,
                        source.X[i, :],
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    row = f(target_wl)

                    # Count extrapolated values
                    if np.any(np.isnan(row)):
                        n_warnings += 1

                except Exception:
                    row = np.full(len(target_wl), np.nan)
                    n_warnings += 1

                X_parts.append(row.reshape(1, -1))
                id_parts.append(sample_id)
                seen_ids.add(sample_id)

                if has_valid_y:
                    y_parts.append(source.y[i])

        X_merged = np.vstack(X_parts)
        y_merged = np.array(y_parts) if y_parts else None

        return MergeResult(
            X=X_merged,
            wavelengths=target_wl,
            sample_ids=id_parts,
            y=y_merged,
            target_name=self.sources[0].target_name,
            strategy="interpolation",
            n_sources=len(self.sources),
            report={
                "n_wavelengths": len(target_wl),
                "n_samples": len(id_parts),
                "n_duplicates": n_duplicates,
                "n_interpolation_warnings": n_warnings,
                "wavelength_step": step,
                "duplicate_handling": dup_handling
            }
        )

    def _set_status(self, message: str, error: bool = False):
        """Set status message."""
        tag = f"{self.parent_tag}_datamgmt_status"
        if dpg.does_item_exist(tag):
            color = (255, 100, 100) if error else (100, 200, 100)
            dpg.set_value(tag, message)
            dpg.configure_item(tag, color=color)

    def get_sources(self) -> List[DataSource]:
        """Return list of current sources."""
        return self.sources.copy()

    def get_source_count(self) -> int:
        """Return number of loaded sources."""
        return len(self.sources)
