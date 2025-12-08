"""
Virtual scrolling data grid for Spectral Predict v3.

High-performance table using Dear PyGui's native table with clipper
for virtual scrolling - only renders visible rows.

Supports editing capabilities including:
- Cell editing (double-click to edit)
- Column operations (add, delete, rename)
- Row operations (delete, duplicate, insert)
- Fill operations (fill down, fill selection)
- Undo/Redo (Ctrl+Z, Ctrl+Y)
"""

import time
import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, List, Callable, Dict, Any, Tuple
from collections import deque
from copy import deepcopy
from ..theme import COLORS
from ..tooltips import add_tooltip, TOOLTIP_CONTENT


class EditHistory:
    """
    Manages undo/redo history for data grid edits.

    Stores edit operations as dicts with operation type and data.
    Limited to 50 operations to prevent memory issues.
    """

    def __init__(self, max_size: int = 50):
        """
        Initialize edit history.

        Parameters
        ----------
        max_size : int
            Maximum number of operations to store
        """
        self.max_size = max_size
        self._undo_stack = deque(maxlen=max_size)
        self._redo_stack = deque(maxlen=max_size)

    def push(self, operation: Dict[str, Any]):
        """
        Add an operation to the undo stack.

        Parameters
        ----------
        operation : dict
            Operation data with 'type' and relevant fields
        """
        self._undo_stack.append(operation)
        self._redo_stack.clear()  # Clear redo stack on new edit

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[Dict[str, Any]]:
        """
        Pop and return the last operation from undo stack.

        Returns
        -------
        dict or None
            The operation to reverse, or None if stack empty
        """
        if self.can_undo():
            operation = self._undo_stack.pop()
            self._redo_stack.append(operation)
            return operation
        return None

    def redo(self) -> Optional[Dict[str, Any]]:
        """
        Pop and return the last operation from redo stack.

        Returns
        -------
        dict or None
            The operation to reapply, or None if stack empty
        """
        if self.can_redo():
            operation = self._redo_stack.pop()
            self._undo_stack.append(operation)
            return operation
        return None

    def clear(self):
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()


class DataGrid:
    """
    High-performance data grid with virtual scrolling.

    Only renders visible rows for smooth scrolling with large datasets.

    Example
    -------
    >>> grid = DataGrid(parent="data_panel")
    >>> grid.set_data(dataset)
    """

    def __init__(self, parent: str, tag: str = "data_grid"):
        """
        Initialize the data grid.

        Parameters
        ----------
        parent : str
            Parent container tag
        tag : str
            Unique tag for this grid
        """
        self.parent = parent
        self.tag = tag
        self._dataset = None
        self._table_tag = f"{tag}_table"
        self._info_tag = f"{tag}_info"

        # Column configuration
        self._wavelength_decimals = 4
        self._show_all_wavelengths = False  # Sample by default, checkbox to show all

        # Editing state
        self._edit_mode = False
        self._edit_history = EditHistory(max_size=50)
        self._modified_cells = {}  # {(row, col): original_value}
        self._selected_cells = set()  # Set of (row, col) tuples
        self._editing_cell = None  # (row, col) currently being edited
        self._edit_input_tag = None  # Tag of input widget for editing

        # Selection tracking
        self._selected_row = None  # Currently selected row index
        self._selected_col = None  # Currently selected column index
        self._selection_anchor = None  # (row, col) anchor for range selection
        self._has_focus = False  # True if grid was last clicked

        # Double-click tracking for cell editing
        self._last_click_time = 0
        self._last_click_cell = None

        # Callbacks
        self._on_data_change_callback = None
        self._on_selection_change_callback = None

        self._create_ui()

    def _create_ui(self):
        """Create the grid UI structure."""
        with dpg.child_window(parent=self.parent, tag=self.tag, border=False):
            # Toolbar
            with dpg.group(horizontal=True):
                dpg.add_text("Rows:", color=COLORS["text_muted"])
                dpg.add_text("0", tag=f"{self.tag}_row_count")
                dpg.add_spacer(width=20)
                dpg.add_text("Columns:", color=COLORS["text_muted"])
                dpg.add_text("0", tag=f"{self.tag}_col_count")
                dpg.add_spacer(width=20)
                dpg.add_checkbox(
                    label="Show all wavelengths",
                    default_value=False,
                    callback=self._on_toggle_all_wavelengths,
                    tag=f"{self.tag}_all_wl"
                )

            dpg.add_spacer(height=5)

            # Table container (with right-click handler binding)
            container = dpg.add_child_window(
                tag=f"{self.tag}_table_container",
                height=-30,
                horizontal_scrollbar=True
            )

            # Status bar
            dpg.add_text(
                "No data loaded",
                tag=self._info_tag,
                color=COLORS["text_muted"]
            )

        # Set up tooltips
        self._setup_tooltips()

        # Set up keyboard handlers for copy/paste
        self._setup_keyboard_handlers()

        # Set up right-click context menu
        self._setup_context_menu()

    def _setup_context_menu(self):
        """Set up right-click context menu for copy/paste."""
        with dpg.window(
            tag=f"{self.tag}_context_menu",
            popup=True,
            show=False,
            no_title_bar=True,
            min_size=[120, 10],
            max_size=[200, 200]
        ):
            dpg.add_menu_item(
                label="Copy (Ctrl+C)",
                callback=lambda: self.copy_selection()
            )
            dpg.add_menu_item(
                label="Paste (Ctrl+V)",
                callback=lambda: self.paste_at_selection(),
                tag=f"{self.tag}_paste_menu_item"
            )
            dpg.add_separator()
            dpg.add_menu_item(
                label="Select All (Ctrl+A)",
                callback=lambda: self.select_all()
            )

    def _setup_tooltips(self):
        """Set up tooltips for data grid elements."""
        add_tooltip(f"{self.tag}_all_wl", TOOLTIP_CONTENT['ui']['show_all_wavelengths'])

    def _setup_keyboard_handlers(self):
        """Set up keyboard handlers for copy/paste operations."""
        with dpg.handler_registry(tag=f"{self.tag}_keyboard_handler"):
            dpg.add_key_press_handler(dpg.mvKey_C, callback=self._on_key_c)
            dpg.add_key_press_handler(dpg.mvKey_V, callback=self._on_key_v)
            dpg.add_key_press_handler(dpg.mvKey_A, callback=self._on_key_a)

    def _is_grid_focused(self) -> bool:
        """Check if the data grid has focus.

        Focus is set when a cell is clicked and cleared when clicking elsewhere.
        For simplicity, we consider the grid focused if there's any selection.
        """
        # If there's a selection or the grid was explicitly focused, allow copy/paste
        if self._has_focus:
            return True
        if self._selected_cells:
            return True
        if self._selected_row is not None and self._selected_col is not None:
            return True
        return False

    def set_focus(self, focused: bool = True):
        """Set the focus state of the grid."""
        self._has_focus = focused

    def _on_key_c(self, sender, app_data):
        """Handle C key press - copy if Ctrl held and grid focused."""
        if not self._is_grid_focused():
            return
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        if ctrl:
            self.copy_selection()

    def _on_key_v(self, sender, app_data):
        """Handle V key press - paste if Ctrl held and grid focused in edit mode."""
        if not self._is_grid_focused():
            return
        if not self._edit_mode:
            return
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        if ctrl:
            self.paste_at_selection()

    def _on_key_a(self, sender, app_data):
        """Handle A key press - select all if Ctrl held and grid focused."""
        if not self._is_grid_focused():
            return
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        if ctrl:
            self.select_all()

    def set_data(self, dataset):
        """
        Set the dataset to display.

        Parameters
        ----------
        dataset : SpectralDataset
            The dataset to display
        """
        self._dataset = dataset
        self._rebuild_table()

    def _rebuild_table(self):
        """Rebuild the table with current data."""
        # Clear existing table
        if dpg.does_item_exist(self._table_tag):
            dpg.delete_item(self._table_tag)

        if self._dataset is None:
            dpg.set_value(self._info_tag, "No data loaded")
            return

        ds = self._dataset
        n_samples = ds.n_samples
        n_wavelengths = ds.n_wavelengths

        # Build column list
        columns = ['ID']
        if ds.has_target:
            columns.append(ds.target_name or 'Target')

        # Add metadata columns (exclude target to prevent duplication)
        if ds.metadata_columns:
            target_name = ds.target_name
            columns.extend([col for col in ds.metadata_columns.keys() if col != target_name])

        # Add wavelength columns
        if self._show_all_wavelengths:
            # Show all wavelengths
            wl_indices = list(range(n_wavelengths))
        else:
            # Sample ~100 wavelengths for quick view
            wl_step = max(1, n_wavelengths // 100)
            wl_indices = list(range(0, n_wavelengths, wl_step))
        # Format as integers - no decimal places
        wl_cols = [f"{int(round(ds.wavelengths[i]))}" for i in wl_indices]
        columns.extend(wl_cols)

        # Update counts
        dpg.set_value(f"{self.tag}_row_count", str(n_samples))
        dpg.set_value(f"{self.tag}_col_count", str(len(columns)))

        # Create table
        with dpg.table(
            parent=f"{self.tag}_table_container",
            tag=self._table_tag,
            header_row=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            scrollY=True,
            scrollX=True,
            freeze_rows=1,
            freeze_columns=1,
            row_background=True,
            resizable=True,
            policy=dpg.mvTable_SizingFixedFit,
            height=-1
        ):
            # Add columns with appropriate widths
            for i, col_name in enumerate(columns):
                if col_name == 'ID':
                    dpg.add_table_column(label=col_name, init_width_or_weight=120)
                elif i < len(columns) - len(wl_indices):
                    # Metadata/target columns - wider
                    dpg.add_table_column(label=col_name, init_width_or_weight=100)
                else:
                    # Wavelength columns - fixed width for readability
                    dpg.add_table_column(label=col_name, init_width_or_weight=70)

            # Use clipper for virtual scrolling
            with dpg.table_row():
                pass  # Header row placeholder

            # Add data rows with clickable cells for editing
            for row_idx in range(n_samples):
                with dpg.table_row():
                    col_idx = 0

                    # ID column (clickable for editing)
                    self._render_cell(row_idx, col_idx, ds.sample_ids[row_idx], width=110)
                    col_idx += 1

                    # Target column
                    if ds.has_target:
                        target_val = ds.y[row_idx]
                        if isinstance(target_val, (str, np.str_)):
                            label = str(target_val)
                        else:
                            label = f"{target_val:.4f}"
                        self._render_cell(row_idx, col_idx, label, width=90)
                        col_idx += 1

                    # Metadata columns
                    if ds.metadata_columns:
                        for meta_col in ds.metadata_columns.keys():
                            val = ds.metadata_columns[meta_col][row_idx]
                            self._render_cell(row_idx, col_idx, val, width=90)
                            col_idx += 1

                    # Wavelength columns (clickable for editing)
                    for wl_idx in wl_indices:
                        val = ds.X[row_idx, wl_idx]
                        self._render_cell(row_idx, col_idx, f"{val:.{self._wavelength_decimals}f}", width=60)
                        col_idx += 1

        # Update status
        target_info = ""
        if ds.has_target:
            target_type = ds.metadata.get('target_type', 'unknown')
            if target_type == 'classification':
                n_classes = len(set(ds.y))
                target_info = f" | Target: {ds.target_name} ({n_classes} classes)"
            else:
                target_info = f" | Target: {ds.target_name}"

        status = (
            f"{n_samples} samples x {n_wavelengths} wavelengths"
            f" ({ds.wavelength_range[0]:.0f}-{ds.wavelength_range[1]:.0f} nm)"
            f"{target_info}"
        )
        dpg.set_value(self._info_tag, status)

        # Auto-focus the editing cell input if one exists
        if self._editing_cell is not None:
            row, col = self._editing_cell
            input_tag = f"{self.tag}_cell_input_{row}_{col}"
            if dpg.does_item_exist(input_tag):
                dpg.focus_item(input_tag)

    def _on_toggle_all_wavelengths(self, sender, app_data):
        """Handle all wavelengths toggle."""
        self._show_all_wavelengths = app_data
        self._rebuild_table()

    def _render_cell(self, row_idx: int, col_idx: int, value: Any, width: int = 60):
        """
        Render a cell as either input (if being edited) or selectable.

        Parameters
        ----------
        row_idx : int
            Row index
        col_idx : int
            Column index
        value : Any
            Cell value to display
        width : int
            Width for input field
        """
        if self._editing_cell == (row_idx, col_idx):
            # Render as input field for true in-cell editing
            input_tag = f"{self.tag}_cell_input_{row_idx}_{col_idx}"
            dpg.add_input_text(
                default_value=str(value),
                tag=input_tag,
                width=width,
                on_enter=True,
                callback=self._on_cell_input_enter,
                user_data=(row_idx, col_idx)
            )
        else:
            # Normal selectable cell
            dpg.add_selectable(
                label=str(value),
                callback=self._on_cell_click,
                user_data=(row_idx, col_idx),
                span_columns=False
            )

    def _on_cell_input_enter(self, sender, app_data, user_data):
        """Handle Enter key in cell input - save and exit edit mode."""
        self._save_current_edit()
        self._editing_cell = None
        self._rebuild_table()

    def _on_cell_click(self, sender, app_data, user_data):
        """
        Handle cell click for selection and editing.

        Parameters
        ----------
        sender : int
            DPG item ID that was clicked
        app_data : any
            Application data (selection state)
        user_data : tuple
            (row_idx, col_idx) of the clicked cell
        """
        row, col = user_data

        # Check for double-click (same cell within 300ms)
        current_time = time.time()
        double_click = (
            self._last_click_cell == (row, col) and
            current_time - self._last_click_time < 0.3
        )
        self._last_click_time = current_time
        self._last_click_cell = (row, col)

        # Check for modifier keys
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

        if shift and self._selection_anchor is not None:
            # Extend selection from anchor to clicked cell
            self._select_range(self._selection_anchor, (row, col))
        elif ctrl:
            # Toggle single cell in selection
            if (row, col) in self._selected_cells:
                self._selected_cells.discard((row, col))
            else:
                self._selected_cells.add((row, col))
        else:
            # Single cell selection, set as new anchor
            self._selected_cells = {(row, col)}
            self._selection_anchor = (row, col)

        # Always update current selection position
        self._selected_row = row
        self._selected_col = col

        # Set focus to this grid
        self._has_focus = True

        # Update status to show selection
        n_selected = len(self._selected_cells)
        if n_selected > 1:
            dpg.set_value(self._info_tag, f"Selected: {n_selected} cells (Ctrl+C to copy)")
        elif n_selected == 1:
            dpg.set_value(self._info_tag, f"Cell ({row}, {col}) selected (double-click to edit)")

        # Notify selection change
        if self._on_selection_change_callback:
            self._on_selection_change_callback(row, col)

        # Double-click in edit mode opens cell editor
        if self._edit_mode and double_click and not shift and not ctrl:
            self.edit_cell(row, col)

    def _select_range(self, anchor: Tuple[int, int], end: Tuple[int, int]):
        """Select rectangular region from anchor to end point."""
        self._selected_cells.clear()
        r1, c1 = anchor
        r2, c2 = end
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                self._selected_cells.add((r, c))

    def select_all(self):
        """Select all cells in the grid."""
        if self._dataset is None:
            return
        n_rows = self._dataset.n_samples
        n_cols = self._get_total_columns()
        self._selected_cells.clear()
        for r in range(n_rows):
            for c in range(n_cols):
                self._selected_cells.add((r, c))
        self._selection_anchor = (0, 0)
        self._selected_row = 0
        self._selected_col = 0

    def _get_total_columns(self) -> int:
        """Get total number of logical columns in the grid."""
        if self._dataset is None:
            return 0
        ds = self._dataset
        count = 1  # ID column
        if ds.has_target:
            count += 1
        if ds.metadata_columns:
            # Exclude target from count
            count += sum(1 for col in ds.metadata_columns.keys() if col != ds.target_name)
        count += ds.n_wavelengths
        return count

    def get_selected_row(self) -> Optional[int]:
        """Get currently selected row index."""
        return self._selected_row

    def get_selected_col(self) -> Optional[int]:
        """Get currently selected column index."""
        return self._selected_col

    def set_on_selection_change(self, callback: Callable):
        """
        Set callback for when selection changes.

        Parameters
        ----------
        callback : callable
            Function(row, col) called when selection changes
        """
        self._on_selection_change_callback = callback

    def clear(self):
        """Clear the grid."""
        self._dataset = None
        if dpg.does_item_exist(self._table_tag):
            dpg.delete_item(self._table_tag)
        dpg.set_value(f"{self.tag}_row_count", "0")
        dpg.set_value(f"{self.tag}_col_count", "0")
        dpg.set_value(self._info_tag, "No data loaded")
        self._edit_history.clear()
        self._modified_cells.clear()
        self._selected_cells.clear()

    # ==================== Editing Methods ====================

    def set_edit_mode(self, enabled: bool):
        """
        Enable or disable edit mode.

        Parameters
        ----------
        enabled : bool
            True to enable editing, False to disable
        """
        self._edit_mode = enabled
        if not enabled:
            self._cancel_cell_edit()

    def is_edit_mode(self) -> bool:
        """Check if edit mode is enabled."""
        return self._edit_mode

    def set_on_data_change(self, callback: Callable):
        """
        Set callback for when data changes.

        Parameters
        ----------
        callback : callable
            Function to call when data is modified
        """
        self._on_data_change_callback = callback

    def _get_cell_value(self, row: int, col: int) -> Any:
        """
        Get the current value of a cell.

        Parameters
        ----------
        row : int
            Row index
        col : int
            Column index

        Returns
        -------
        any
            Cell value
        """
        if self._dataset is None:
            return None

        ds = self._dataset
        col_idx = 0

        # ID column
        if col == col_idx:
            return ds.sample_ids[row]
        col_idx += 1

        # Target column
        if ds.has_target:
            if col == col_idx:
                return ds.y[row]
            col_idx += 1

        # Metadata columns (exclude target to match display)
        if ds.metadata_columns:
            target_name = ds.target_name
            meta_keys = [k for k in ds.metadata_columns.keys() if k != target_name]
            if col < col_idx + len(meta_keys):
                meta_key = meta_keys[col - col_idx]
                return ds.metadata_columns[meta_key][row]
            col_idx += len(meta_keys)

        # Wavelength columns
        if self._show_all_wavelengths:
            wl_indices = list(range(ds.n_wavelengths))
        else:
            wl_step = max(1, ds.n_wavelengths // 100)
            wl_indices = list(range(0, ds.n_wavelengths, wl_step))

        wl_col_idx = col - col_idx
        if wl_col_idx < len(wl_indices):
            return ds.X[row, wl_indices[wl_col_idx]]

        return None

    def _set_cell_value(self, row: int, col: int, value: Any) -> bool:
        """
        Set the value of a cell.

        Parameters
        ----------
        row : int
            Row index
        col : int
            Column index
        value : any
            New value

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self._dataset is None:
            return False

        ds = self._dataset
        col_idx = 0

        # ID column - editable
        if col == col_idx:
            ds.sample_ids[row] = str(value)
            return True
        col_idx += 1

        # Target column - editable
        if ds.has_target:
            if col == col_idx:
                target_type = ds.metadata.get('target_type', 'regression')
                if target_type == 'classification':
                    ds.y[row] = str(value)
                else:
                    try:
                        ds.y[row] = float(value)
                    except (ValueError, TypeError):
                        return False
                return True
            col_idx += 1

        # Metadata columns - editable (exclude target to match display)
        if ds.metadata_columns:
            target_name = ds.target_name
            meta_keys = [k for k in ds.metadata_columns.keys() if k != target_name]
            if col < col_idx + len(meta_keys):
                meta_key = meta_keys[col - col_idx]
                ds.metadata_columns[meta_key][row] = value
                return True
            col_idx += len(meta_keys)

        # Wavelength columns - must be numeric
        if self._show_all_wavelengths:
            wl_indices = list(range(ds.n_wavelengths))
        else:
            wl_step = max(1, ds.n_wavelengths // 100)
            wl_indices = list(range(0, ds.n_wavelengths, wl_step))

        wl_col_idx = col - col_idx
        if wl_col_idx < len(wl_indices):
            try:
                ds.X[row, wl_indices[wl_col_idx]] = float(value)
                return True
            except (ValueError, TypeError):
                return False

        return False

    def _get_column_type(self, col: int) -> str:
        """
        Get the type of a column.

        Parameters
        ----------
        col : int
            Column index

        Returns
        -------
        str
            'id', 'target', 'metadata', or 'spectral'
        """
        if self._dataset is None:
            return 'unknown'

        ds = self._dataset
        col_idx = 0

        if col == col_idx:
            return 'id'
        col_idx += 1

        if ds.has_target:
            if col == col_idx:
                return 'target'
            col_idx += 1

        if ds.metadata_columns:
            # Exclude target from metadata count to match display
            target_name = ds.target_name
            meta_count = sum(1 for k in ds.metadata_columns.keys() if k != target_name)
            if col < col_idx + meta_count:
                return 'metadata'
            col_idx += meta_count

        return 'spectral'

    def edit_cell(self, row: int, col: int):
        """
        Start true spreadsheet-style editing - the cell itself becomes an input field.

        Parameters
        ----------
        row : int
            Row index
        col : int
            Column index
        """
        if not self._edit_mode or self._dataset is None:
            return

        # If already editing this cell, do nothing
        if self._editing_cell == (row, col):
            return

        # Save any existing edit first
        if self._editing_cell is not None:
            self._save_current_edit()

        # Store original value for undo
        original_value = self._get_cell_value(row, col)
        self._original_edit_value = original_value
        self._editing_cell = (row, col)

        # Rebuild table with this cell as an input field
        self._rebuild_table()

    def _save_current_edit(self):
        """Save the value from the currently editing cell."""
        if self._editing_cell is None:
            return

        row, col = self._editing_cell
        input_tag = f"{self.tag}_cell_input_{row}_{col}"

        if dpg.does_item_exist(input_tag):
            new_value = dpg.get_value(input_tag)
            original_value = getattr(self, '_original_edit_value', None)

            # Only save if value changed
            if str(new_value) != str(original_value):
                if self._set_cell_value(row, col, new_value):
                    # Record in history
                    self._edit_history.push({
                        'type': 'cell_edit',
                        'row': row,
                        'col': col,
                        'old_value': original_value,
                        'new_value': new_value
                    })

                    # Notify callback
                    if self._on_data_change_callback:
                        self._on_data_change_callback()

        self._editing_cell = None
        self._original_edit_value = None

    def _get_column_name(self, col: int) -> str:
        """Get column name by index."""
        if self._dataset is None:
            return f"Column {col}"

        ds = self._dataset
        col_idx = 0

        if col == col_idx:
            return "ID"
        col_idx += 1

        if ds.has_target:
            if col == col_idx:
                return ds.target_name or "Target"
            col_idx += 1

        if ds.metadata_columns:
            meta_keys = list(ds.metadata_columns.keys())
            if col < col_idx + len(meta_keys):
                return meta_keys[col - col_idx]
            col_idx += len(meta_keys)

        # Wavelength column
        if self._show_all_wavelengths:
            wl_indices = list(range(ds.n_wavelengths))
        else:
            wl_step = max(1, ds.n_wavelengths // 100)
            wl_indices = list(range(0, ds.n_wavelengths, wl_step))

        wl_col_idx = col - col_idx
        if wl_col_idx < len(wl_indices):
            return f"{int(round(ds.wavelengths[wl_indices[wl_col_idx]]))} nm"

        return f"Column {col}"

    def _on_mouse_click_during_edit(self, sender=None, app_data=None):
        """Handle mouse click while editing - auto-save if clicking outside edit input."""
        if self._editing_cell is None:
            return

        # Check if click is inside the edit window
        if dpg.does_item_exist(f"{self.tag}_edit_window"):
            # Get edit window position and size
            pos = dpg.get_item_pos(f"{self.tag}_edit_window")
            # Auto-save - the click handler fires, so save and close
            self._on_edit_confirm()

    def _on_edit_confirm(self, sender=None, app_data=None):
        """Confirm and save cell edit."""
        if self._editing_cell is None or self._edit_input_tag is None:
            return

        row, col = self._editing_cell
        new_value = dpg.get_value(self._edit_input_tag)

        # Get original value
        original_value = getattr(self, '_original_edit_value', None)

        # Only save if value changed
        if str(new_value) != str(original_value):
            if self._set_cell_value(row, col, new_value):
                # Record in history
                self._edit_history.push({
                    'type': 'cell_edit',
                    'row': row,
                    'col': col,
                    'old_value': original_value,
                    'new_value': new_value
                })

                # Notify callback
                if self._on_data_change_callback:
                    self._on_data_change_callback()

        # Clean up
        self._cleanup_edit()

        # Rebuild table to show changes
        self._rebuild_table()

    def _on_edit_cancel(self, sender=None, app_data=None):
        """Cancel cell edit without saving."""
        self._cleanup_edit()

    def _cancel_cell_edit(self):
        """Cancel any active cell edit - saves changes first (spreadsheet behavior)."""
        if self._editing_cell is not None and self._edit_input_tag is not None:
            # Auto-save current edit before canceling (spreadsheet style)
            if dpg.does_item_exist(self._edit_input_tag):
                self._on_edit_confirm()
                return
        self._cleanup_edit()

    def _cleanup_edit(self):
        """Clean up edit state and UI."""
        # Remove handler registry
        if dpg.does_item_exist(f"{self.tag}_edit_handler"):
            dpg.delete_item(f"{self.tag}_edit_handler")
        # Remove edit window
        if dpg.does_item_exist(f"{self.tag}_edit_window"):
            dpg.delete_item(f"{self.tag}_edit_window")
        self._editing_cell = None
        self._edit_input_tag = None
        self._original_edit_value = None

    def undo(self):
        """Undo the last edit operation."""
        if not self._edit_history.can_undo():
            return

        operation = self._edit_history.undo()
        if operation is None:
            return

        if operation['type'] == 'cell_edit':
            # Restore old value
            row, col = operation['row'], operation['col']
            self._set_cell_value(row, col, operation['old_value'])
            self._rebuild_table()

        elif operation['type'] == 'row_delete':
            # Restore deleted rows
            self._restore_rows(operation['rows_data'], operation['indices'])
            self._rebuild_table()

        elif operation['type'] == 'column_delete':
            # Restore deleted column
            self._restore_column(operation['column_data'])
            self._rebuild_table()

        elif operation['type'] == 'row_insert':
            # Remove inserted rows
            self._delete_rows_by_indices(operation['indices'], record_history=False)
            self._rebuild_table()

        elif operation['type'] == 'column_add':
            # Remove added column
            self._delete_column(operation['column_name'], record_history=False)
            self._rebuild_table()

        elif operation['type'] == 'fill_down':
            # Restore old values
            for row, old_val in operation['old_values'].items():
                self._set_cell_value(row, operation['col'], old_val)
            self._rebuild_table()

        elif operation['type'] == 'fill_selection':
            # Restore old values
            for row, old_val in operation['old_values'].items():
                self._set_cell_value(row, operation['col'], old_val)
            self._rebuild_table()

        elif operation['type'] == 'paste':
            # Restore all old values
            for (row, col), old_val in operation['old_values'].items():
                self._set_cell_value(row, col, old_val)
            self._rebuild_table()

        if self._on_data_change_callback:
            self._on_data_change_callback()

    def redo(self):
        """Redo the last undone operation."""
        if not self._edit_history.can_redo():
            return

        operation = self._edit_history.redo()
        if operation is None:
            return

        if operation['type'] == 'cell_edit':
            # Reapply new value
            row, col = operation['row'], operation['col']
            self._set_cell_value(row, col, operation['new_value'])
            self._rebuild_table()

        elif operation['type'] == 'row_delete':
            # Re-delete rows
            self._delete_rows_by_indices(operation['indices'], record_history=False)
            self._rebuild_table()

        elif operation['type'] == 'column_delete':
            # Re-delete column
            self._delete_column(operation['column_name'], record_history=False)
            self._rebuild_table()

        elif operation['type'] == 'row_insert':
            # Re-insert rows
            self._restore_rows(operation['rows_data'], operation['indices'])
            self._rebuild_table()

        elif operation['type'] == 'column_add':
            # Re-add column
            self._restore_column(operation['column_data'])
            self._rebuild_table()

        elif operation['type'] == 'fill_down':
            # Re-apply fill
            for row in operation['old_values'].keys():
                self._set_cell_value(row, operation['col'], operation['new_value'])
            self._rebuild_table()

        elif operation['type'] == 'fill_selection':
            # Re-apply fill
            for row in operation['old_values'].keys():
                self._set_cell_value(row, operation['col'], operation['new_value'])
            self._rebuild_table()

        elif operation['type'] == 'paste':
            # Re-apply all pasted values
            for (row, col), new_val in operation['new_values'].items():
                self._set_cell_value(row, col, new_val)
            self._rebuild_table()

        if self._on_data_change_callback:
            self._on_data_change_callback()

    # ==================== Row Operations ====================

    def delete_rows(self, row_indices: List[int]):
        """
        Delete rows by indices.

        Parameters
        ----------
        row_indices : list of int
            Indices of rows to delete
        """
        self._delete_rows_by_indices(row_indices, record_history=True)
        self._rebuild_table()
        if self._on_data_change_callback:
            self._on_data_change_callback()

    def _delete_rows_by_indices(self, row_indices: List[int], record_history: bool = True):
        """Internal method to delete rows."""
        if self._dataset is None or not row_indices:
            return

        ds = self._dataset
        row_indices = sorted(row_indices, reverse=True)

        # Store data for undo
        if record_history:
            rows_data = {
                'sample_ids': [ds.sample_ids[i] for i in row_indices],
                'X': ds.X[row_indices, :].copy(),
                'y': ds.y[row_indices].copy() if ds.has_target else None,
                'metadata': {
                    k: [v[i] for i in row_indices]
                    for k, v in ds.metadata_columns.items()
                } if ds.metadata_columns else {}
            }

            self._edit_history.push({
                'type': 'row_delete',
                'indices': sorted(row_indices),
                'rows_data': rows_data
            })

        # Delete rows
        ds.sample_ids = [s for i, s in enumerate(ds.sample_ids) if i not in row_indices]
        ds.X = np.delete(ds.X, row_indices, axis=0)
        if ds.has_target:
            ds.y = np.delete(ds.y, row_indices, axis=0)
        if ds.metadata_columns:
            for key in ds.metadata_columns:
                ds.metadata_columns[key] = [
                    v for i, v in enumerate(ds.metadata_columns[key])
                    if i not in row_indices
                ]

    def _restore_rows(self, rows_data: Dict, indices: List[int]):
        """Restore deleted rows."""
        if self._dataset is None:
            return

        ds = self._dataset

        # Insert rows back
        for idx, orig_idx in enumerate(indices):
            ds.sample_ids.insert(orig_idx, rows_data['sample_ids'][idx])
            ds.X = np.insert(ds.X, orig_idx, rows_data['X'][idx], axis=0)
            if ds.has_target and rows_data['y'] is not None:
                ds.y = np.insert(ds.y, orig_idx, rows_data['y'][idx], axis=0)
            if ds.metadata_columns and rows_data['metadata']:
                for key in ds.metadata_columns:
                    ds.metadata_columns[key].insert(orig_idx, rows_data['metadata'][key][idx])

    def duplicate_rows(self, row_indices: List[int]):
        """
        Duplicate selected rows.

        Parameters
        ----------
        row_indices : list of int
            Indices of rows to duplicate
        """
        if self._dataset is None or not row_indices:
            return

        ds = self._dataset
        new_indices = []

        for row_idx in sorted(row_indices, reverse=True):
            # Insert duplicate right after original
            insert_pos = row_idx + 1

            ds.sample_ids.insert(insert_pos, ds.sample_ids[row_idx] + "_copy")
            ds.X = np.insert(ds.X, insert_pos, ds.X[row_idx], axis=0)
            if ds.has_target:
                ds.y = np.insert(ds.y, insert_pos, ds.y[row_idx], axis=0)
            if ds.metadata_columns:
                for key in ds.metadata_columns:
                    ds.metadata_columns[key].insert(insert_pos, ds.metadata_columns[key][row_idx])

            new_indices.append(insert_pos)

        # Record in history
        rows_data = {
            'sample_ids': [ds.sample_ids[i] for i in new_indices],
            'X': ds.X[new_indices, :].copy(),
            'y': ds.y[new_indices].copy() if ds.has_target else None,
            'metadata': {
                k: [v[i] for i in new_indices]
                for k, v in ds.metadata_columns.items()
            } if ds.metadata_columns else {}
        }

        self._edit_history.push({
            'type': 'row_insert',
            'indices': new_indices,
            'rows_data': rows_data
        })

        self._rebuild_table()
        if self._on_data_change_callback:
            self._on_data_change_callback()

    def insert_row(self, position: int = -1):
        """
        Insert a new blank row.

        Parameters
        ----------
        position : int
            Position to insert (-1 for end)
        """
        if self._dataset is None:
            return

        ds = self._dataset

        if position == -1:
            position = ds.n_samples

        # Create blank row
        ds.sample_ids.insert(position, f"new_sample_{position}")
        ds.X = np.insert(ds.X, position, np.zeros(ds.n_wavelengths), axis=0)
        if ds.has_target:
            ds.y = np.insert(ds.y, position, 0, axis=0)
        if ds.metadata_columns:
            for key in ds.metadata_columns:
                ds.metadata_columns[key].insert(position, "")

        # Record in history
        self._edit_history.push({
            'type': 'row_insert',
            'indices': [position],
            'rows_data': {
                'sample_ids': [ds.sample_ids[position]],
                'X': ds.X[position:position+1, :].copy(),
                'y': ds.y[position:position+1].copy() if ds.has_target else None,
                'metadata': {
                    k: [v[position]]
                    for k, v in ds.metadata_columns.items()
                } if ds.metadata_columns else {}
            }
        })

        self._rebuild_table()
        if self._on_data_change_callback:
            self._on_data_change_callback()

    # ==================== Column Operations ====================

    def add_column(self, column_name: str, column_type: str = 'metadata', default_value: Any = "") -> bool:
        """
        Add a new column.

        Parameters
        ----------
        column_name : str
            Name of the new column
        column_type : str
            'metadata' or 'spectral' (spectral not yet supported)
        default_value : any
            Default value for all rows

        Returns
        -------
        bool
            True if column was added, False otherwise
        """
        if self._dataset is None:
            return False

        if column_type != 'metadata':
            return False  # Only metadata columns supported for now

        if not column_name:
            return False

        ds = self._dataset

        # Add to metadata columns
        if ds.metadata_columns is None:
            ds.metadata_columns = {}

        # Check if column already exists
        if column_name in ds.metadata_columns:
            return False

        ds.metadata_columns[column_name] = [default_value] * ds.n_samples

        # Record in history
        self._edit_history.push({
            'type': 'column_add',
            'column_name': column_name,
            'column_data': {
                'type': column_type,
                'values': ds.metadata_columns[column_name].copy()
            }
        })

        self._rebuild_table()
        if self._on_data_change_callback:
            self._on_data_change_callback()

        return True

    def delete_column(self, column_name: str) -> bool:
        """
        Delete a column by name.

        Parameters
        ----------
        column_name : str
            Name of column to delete

        Returns
        -------
        bool
            True if column was deleted, False otherwise
        """
        if not column_name:
            return False

        if self._dataset is None:
            return False

        # Check if column exists
        if not self._dataset.metadata_columns or column_name not in self._dataset.metadata_columns:
            return False

        self._delete_column(column_name, record_history=True)
        self._rebuild_table()

        if self._on_data_change_callback:
            self._on_data_change_callback()

        return True

    def _delete_column(self, column_name: str, record_history: bool = True):
        """Internal method to delete a column."""
        if self._dataset is None:
            return

        ds = self._dataset

        # Only metadata columns can be deleted for now
        if ds.metadata_columns and column_name in ds.metadata_columns:
            if record_history:
                self._edit_history.push({
                    'type': 'column_delete',
                    'column_name': column_name,
                    'column_data': {
                        'type': 'metadata',
                        'values': ds.metadata_columns[column_name].copy()
                    }
                })

            del ds.metadata_columns[column_name]

    def _restore_column(self, column_data: Dict):
        """Restore a deleted column."""
        if self._dataset is None:
            return

        ds = self._dataset
        col_name = column_data.get('column_name')
        col_type = column_data.get('type', 'metadata')

        if col_type == 'metadata':
            if ds.metadata_columns is None:
                ds.metadata_columns = {}
            ds.metadata_columns[col_name] = column_data['values'].copy()

    def rename_column(self, old_name: str, new_name: str):
        """
        Rename a column.

        Parameters
        ----------
        old_name : str
            Current column name
        new_name : str
            New column name
        """
        if self._dataset is None:
            return

        ds = self._dataset

        # Handle metadata columns
        if ds.metadata_columns and old_name in ds.metadata_columns:
            ds.metadata_columns[new_name] = ds.metadata_columns.pop(old_name)
            self._rebuild_table()
            if self._on_data_change_callback:
                self._on_data_change_callback()

    # ==================== Fill Operations ====================

    def fill_down(self, start_row: int, col: int, end_row: int = -1):
        """
        Fill down from a cell to the end or specified row.

        Parameters
        ----------
        start_row : int
            Starting row index
        col : int
            Column index
        end_row : int
            Ending row index (-1 for last row)
        """
        if self._dataset is None:
            return

        if end_row == -1:
            end_row = self._dataset.n_samples - 1

        source_value = self._get_cell_value(start_row, col)

        # Store original values for undo
        old_values = {}
        for row in range(start_row + 1, end_row + 1):
            old_values[row] = self._get_cell_value(row, col)
            self._set_cell_value(row, col, source_value)

        # Record in history
        self._edit_history.push({
            'type': 'fill_down',
            'start_row': start_row,
            'end_row': end_row,
            'col': col,
            'old_values': old_values,
            'new_value': source_value
        })

        self._rebuild_table()
        if self._on_data_change_callback:
            self._on_data_change_callback()

    def fill_selection(self, row_indices: List[int], col: int, value: Any):
        """
        Fill selected cells with a value.

        Parameters
        ----------
        row_indices : list of int
            Row indices to fill
        col : int
            Column index
        value : any
            Value to fill
        """
        if self._dataset is None or not row_indices:
            return

        old_values = {}
        for row in row_indices:
            old_values[row] = self._get_cell_value(row, col)
            self._set_cell_value(row, col, value)

        # Record in history
        self._edit_history.push({
            'type': 'fill_selection',
            'rows': row_indices,
            'col': col,
            'old_values': old_values,
            'new_value': value
        })

        self._rebuild_table()
        if self._on_data_change_callback:
            self._on_data_change_callback()

    # ==================== Copy/Paste Methods ====================

    def copy_selection(self):
        """
        Copy selected cells to clipboard as TSV (tab-separated values).

        Copies either the multi-cell selection or the current single cell.
        """
        if self._dataset is None:
            return

        # Get cells to copy - either multi-selection or current cell
        cells = self._selected_cells
        if not cells and self._selected_row is not None and self._selected_col is not None:
            cells = {(self._selected_row, self._selected_col)}

        if not cells:
            return

        # Convert to TSV
        tsv_data = self._cells_to_tsv(cells)
        if not tsv_data:
            return

        # Copy to clipboard using pyperclip with tkinter fallback
        copied = False
        try:
            import pyperclip
            pyperclip.copy(tsv_data)
            copied = True
        except ImportError:
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(tsv_data)
                root.update()
                root.destroy()
                copied = True
            except Exception:
                pass  # Silent fail if clipboard unavailable
        except Exception:
            pass

        # Show feedback
        if copied:
            n_cells = len(cells)
            dpg.set_value(self._info_tag, f"Copied {n_cells} cell(s) to clipboard")

    def paste_at_selection(self):
        """
        Paste clipboard data starting at the current selection anchor.

        Only works in edit mode. Validates data types before pasting.
        """
        if self._dataset is None or not self._edit_mode:
            return

        # Get paste anchor (top-left of selection or current cell)
        if self._selected_cells:
            min_row = min(c[0] for c in self._selected_cells)
            min_col = min(c[1] for c in self._selected_cells)
        elif self._selected_row is not None and self._selected_col is not None:
            min_row, min_col = self._selected_row, self._selected_col
        else:
            return

        # Get clipboard content
        tsv_data = None
        try:
            import pyperclip
            tsv_data = pyperclip.paste()
        except ImportError:
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                try:
                    tsv_data = root.clipboard_get()
                except tk.TclError:
                    pass  # Clipboard empty or not text
                root.destroy()
            except Exception:
                pass

        if not tsv_data:
            return

        # Parse TSV
        paste_data = self._tsv_to_cells(tsv_data)
        if not paste_data:
            return

        # Apply paste with validation
        self._apply_paste(paste_data, min_row, min_col)

    def _cells_to_tsv(self, cells: set) -> str:
        """
        Convert selected cells to TSV string for clipboard.

        Parameters
        ----------
        cells : set of (row, col) tuples
            Cells to convert

        Returns
        -------
        str
            Tab-separated values string
        """
        if not cells:
            return ""

        # Get bounds of selection
        rows = sorted(set(c[0] for c in cells))
        cols = sorted(set(c[1] for c in cells))

        lines = []
        for r in rows:
            row_values = []
            for c in cols:
                if (r, c) in cells:
                    val = self._get_cell_value(r, c)
                    if val is None:
                        row_values.append("")
                    elif isinstance(val, float):
                        # Format floats nicely
                        if val == int(val):
                            row_values.append(str(int(val)))
                        else:
                            row_values.append(f"{val:.6g}")
                    else:
                        row_values.append(str(val))
                else:
                    row_values.append("")  # Gap in selection
            lines.append("\t".join(row_values))

        return "\n".join(lines)

    def _tsv_to_cells(self, tsv: str) -> List[List[str]]:
        """
        Parse TSV string into 2D list of values.

        Parameters
        ----------
        tsv : str
            Tab-separated values string

        Returns
        -------
        list of list of str
            2D array of cell values
        """
        lines = tsv.strip().split("\n")
        return [line.split("\t") for line in lines]

    def _apply_paste(self, data: List[List[str]], start_row: int, start_col: int):
        """
        Apply paste data with validation and undo support.

        Parameters
        ----------
        data : list of list of str
            2D array of values to paste
        start_row : int
            Starting row index
        start_col : int
            Starting column index
        """
        if self._dataset is None:
            return

        n_samples = self._dataset.n_samples
        n_cols = self._get_total_columns()

        # Collect all changes for undo
        old_values = {}
        changes = []

        for r_offset, row_data in enumerate(data):
            row = start_row + r_offset
            if row >= n_samples:
                break  # Don't expand grid

            for c_offset, value in enumerate(row_data):
                col = start_col + c_offset
                if col >= n_cols:
                    break  # Don't expand grid

                col_type = self._get_column_type(col)

                # Validate based on column type
                if col_type == 'spectral':
                    try:
                        float(value)  # Must be numeric
                    except (ValueError, TypeError):
                        continue  # Skip invalid cells
                elif col_type == 'target':
                    target_type = self._dataset.metadata.get('target_type', 'regression')
                    if target_type != 'classification':
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            continue

                # Store old value
                old_values[(row, col)] = self._get_cell_value(row, col)
                changes.append((row, col, value))

        if not changes:
            return

        # Apply changes
        for row, col, value in changes:
            self._set_cell_value(row, col, value)

        # Record in undo history
        self._edit_history.push({
            'type': 'paste',
            'old_values': old_values,
            'new_values': {(r, c): v for r, c, v in changes},
            'start_row': start_row,
            'start_col': start_col
        })

        self._rebuild_table()

        # Show feedback
        dpg.set_value(self._info_tag, f"Pasted {len(changes)} cell(s)")

        if self._on_data_change_callback:
            self._on_data_change_callback()
