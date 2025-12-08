"""
Column configuration dialog for Spectral Predict v3.

Shows after file load to let user configure:
- ID column selection
- Target column selection (optional)
- Review auto-detected wavelength and metadata columns
"""

import dearpygui.dearpygui as dpg
from typing import Callable, Optional, List, Dict, Any
from ..theme import COLORS


class ColumnConfigDialog:
    """
    Modal dialog for configuring column assignments after file load.

    Example
    -------
    >>> def on_confirm(config):
    ...     print(f"ID: {config['id_column']}, Target: {config['target_column']}")
    >>> dialog = ColumnConfigDialog(on_confirm=on_confirm)
    >>> dialog.show(column_info)
    """

    def __init__(self, on_confirm: Callable[[Dict[str, Any]], None], on_cancel: Optional[Callable] = None):
        """
        Initialize the dialog.

        Parameters
        ----------
        on_confirm : callable
            Called with config dict when user clicks Load Data
        on_cancel : callable, optional
            Called when user clicks Cancel
        """
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self._dialog_id = "column_config_dialog"
        self._column_info = None

    def show(self, column_info: Dict[str, Any]):
        """
        Show the dialog with column information.

        Parameters
        ----------
        column_info : dict
            Dictionary with keys:
            - id_candidates: list of potential ID columns
            - target_candidates: list of potential target columns
            - wavelength_columns: list of wavelength column names
            - metadata_columns: list of other metadata columns
            - all_columns: list of all column names
        """
        self._column_info = column_info

        # Clean up existing dialog if any
        if dpg.does_item_exist(self._dialog_id):
            dpg.delete_item(self._dialog_id)

        # Build dropdown options
        id_options = column_info.get('id_candidates', [])
        if not id_options:
            id_options = column_info.get('all_columns', ['(none)'])[:10]

        target_options = ['(none)'] + column_info.get('target_candidates', [])
        if len(target_options) == 1:  # Only "(none)"
            # Add metadata columns as potential targets
            target_options += column_info.get('metadata_columns', [])

        # Wavelength info
        wl_cols = column_info.get('wavelength_columns', [])
        if wl_cols:
            wl_range = f"{float(wl_cols[0]):.1f} - {float(wl_cols[-1]):.1f} nm"
            wl_count = len(wl_cols)
        else:
            wl_range = "None detected"
            wl_count = 0

        # Metadata info
        meta_cols = column_info.get('metadata_columns', [])
        meta_text = ', '.join(meta_cols[:5])
        if len(meta_cols) > 5:
            meta_text += f" (+{len(meta_cols) - 5} more)"
        elif not meta_cols:
            meta_text = "None detected"

        # Create modal dialog - autosize to fit all content without scrolling
        with dpg.window(
            label="Configure Columns",
            tag=self._dialog_id,
            modal=True,
            width=520,
            autosize=True,
            pos=[500, 200],
            no_resize=False,
            no_scrollbar=True,
            on_close=self._on_cancel
        ):
            dpg.add_text("Configure how your data should be interpreted:")
            dpg.add_spacer(height=15)

            # File Name Column selector (matches spectral filenames)
            dpg.add_text("File Name Column:", color=COLORS["text_secondary"])
            dpg.add_combo(
                items=id_options,
                default_value=id_options[0] if id_options else "(none)",
                tag=f"{self._dialog_id}_id_combo",
                width=-1
            )
            dpg.add_spacer(height=10)

            # Target Column selector
            dpg.add_text("Target Column (optional):", color=COLORS["text_secondary"])
            dpg.add_combo(
                items=target_options,
                default_value=target_options[0],
                tag=f"{self._dialog_id}_target_combo",
                width=-1
            )
            dpg.add_spacer(height=20)

            # Auto-detected info section
            dpg.add_separator()
            dpg.add_text("Auto-detected:", color=COLORS["text_secondary"])
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_text("Wavelength columns:", color=COLORS["text_muted"])
                dpg.add_text(f"{wl_count} ({wl_range})")

            with dpg.group(horizontal=True):
                dpg.add_text("Metadata columns:", color=COLORS["text_muted"])
                dpg.add_text(meta_text, wrap=350)

            dpg.add_spacer(height=40)

            # Buttons - ensure they have enough space
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Cancel",
                    callback=self._on_cancel,
                    width=120
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="Load Data",
                    callback=self._on_confirm,
                    width=140
                )

            dpg.add_spacer(height=20)

    def _on_confirm(self, sender=None, app_data=None):
        """Handle Load Data button click."""
        # Get selected values
        id_col = dpg.get_value(f"{self._dialog_id}_id_combo")
        target_col = dpg.get_value(f"{self._dialog_id}_target_combo")

        config = {
            'id_column': id_col if id_col != "(none)" else None,
            'target_column': target_col if target_col != "(none)" else None,
            'column_info': self._column_info
        }

        # Close dialog
        if dpg.does_item_exist(self._dialog_id):
            dpg.delete_item(self._dialog_id)

        # Call callback
        if self.on_confirm:
            self.on_confirm(config)

    def _on_cancel(self, sender=None, app_data=None):
        """Handle Cancel button click."""
        # Close dialog
        if dpg.does_item_exist(self._dialog_id):
            dpg.delete_item(self._dialog_id)

        # Call callback
        if self.on_cancel:
            self.on_cancel()

    def close(self):
        """Close the dialog if open."""
        if dpg.does_item_exist(self._dialog_id):
            dpg.delete_item(self._dialog_id)


def show_column_config(
    column_info: Dict[str, Any],
    on_confirm: Callable[[Dict[str, Any]], None],
    on_cancel: Optional[Callable] = None
) -> ColumnConfigDialog:
    """
    Convenience function to show column config dialog.

    Parameters
    ----------
    column_info : dict
        Column detection results from io_utils.detect_columns()
    on_confirm : callable
        Called with config when user confirms
    on_cancel : callable, optional
        Called when user cancels

    Returns
    -------
    ColumnConfigDialog
        The dialog instance (can be used to close it programmatically)
    """
    dialog = ColumnConfigDialog(on_confirm=on_confirm, on_cancel=on_cancel)
    dialog.show(column_info)
    return dialog
