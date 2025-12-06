"""
Group assignment dialog for Spectral Predict v3.

Allows assigning selected samples to a group/category
by setting a metadata column value.
"""

import dearpygui.dearpygui as dpg
from typing import Callable, Optional, List, Dict, Any
from ..theme import COLORS


class GroupAssignDialog:
    """
    Modal dialog for assigning selected samples to a group.

    Example
    -------
    >>> def on_assign(column, value):
    ...     print(f"Assign '{value}' to column '{column}'")
    >>> dialog = GroupAssignDialog(on_assign=on_assign)
    >>> dialog.show(existing_columns=["Group", "Site"], n_selected=12)
    """

    def __init__(self, on_assign: Callable[[str, str], None], on_cancel: Optional[Callable] = None):
        """
        Initialize the dialog.

        Parameters
        ----------
        on_assign : callable
            Called with (column_name, value) when user confirms
        on_cancel : callable, optional
            Called when user clicks Cancel
        """
        self.on_assign = on_assign
        self.on_cancel = on_cancel
        self._dialog_id = "group_assign_dialog"
        self._existing_columns: List[str] = []
        self._n_selected: int = 0

    def show(self, existing_columns: List[str] = None, n_selected: int = 0):
        """
        Show the dialog.

        Parameters
        ----------
        existing_columns : list of str
            Existing metadata column names
        n_selected : int
            Number of selected samples
        """
        self._existing_columns = existing_columns or []
        self._n_selected = n_selected

        # Clean up existing dialog if any
        if dpg.does_item_exist(self._dialog_id):
            dpg.delete_item(self._dialog_id)

        # Column options: existing columns + option to create new
        column_options = self._existing_columns.copy()
        if "(New Column)" not in column_options:
            column_options.append("(New Column)")

        # Create modal dialog
        with dpg.window(
            label="Assign to Group",
            tag=self._dialog_id,
            modal=True,
            width=400,
            height=280,
            pos=[600, 300],
            no_resize=True,
            on_close=self._on_cancel
        ):
            dpg.add_text(f"Assign {n_selected} selected sample(s) to a group:", color=COLORS["text_secondary"])
            dpg.add_spacer(height=15)

            # Column selector
            dpg.add_text("Column:", color=COLORS["text_muted"])
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=column_options,
                    default_value=column_options[0] if column_options else "(New Column)",
                    callback=self._on_column_change,
                    tag=f"{self._dialog_id}_column_combo",
                    width=200
                )
                dpg.add_spacer(width=10)
                # New column name input (hidden by default)
                dpg.add_input_text(
                    hint="New column name",
                    tag=f"{self._dialog_id}_new_column",
                    width=150,
                    show=len(column_options) <= 1  # Show if no existing columns
                )

            dpg.add_spacer(height=15)

            # Value input
            dpg.add_text("Value:", color=COLORS["text_muted"])
            dpg.add_input_text(
                hint="e.g., Outlier, Training, Site_A",
                tag=f"{self._dialog_id}_value",
                width=-1
            )

            dpg.add_spacer(height=20)

            # Preview text
            dpg.add_text(
                "",
                tag=f"{self._dialog_id}_preview",
                color=COLORS["text_muted"]
            )
            self._update_preview()

            dpg.add_spacer(height=15)

            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Cancel",
                    callback=self._on_cancel,
                    width=100
                )
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="Assign",
                    callback=self._on_assign,
                    width=100
                )

    def _on_column_change(self, sender, app_data):
        """Handle column selection change."""
        # Show/hide new column input
        is_new = app_data == "(New Column)"
        dpg.configure_item(f"{self._dialog_id}_new_column", show=is_new)
        self._update_preview()

    def _update_preview(self):
        """Update the preview text."""
        column = dpg.get_value(f"{self._dialog_id}_column_combo")
        if column == "(New Column)":
            column = dpg.get_value(f"{self._dialog_id}_new_column") or "?"
        value = dpg.get_value(f"{self._dialog_id}_value") or "?"

        preview = f"Preview: Set '{column}' = '{value}' for {self._n_selected} samples"
        dpg.set_value(f"{self._dialog_id}_preview", preview)

    def _on_assign(self, sender=None, app_data=None):
        """Handle Assign button click."""
        # Get column name
        column = dpg.get_value(f"{self._dialog_id}_column_combo")
        if column == "(New Column)":
            column = dpg.get_value(f"{self._dialog_id}_new_column")
            if not column:
                dpg.set_value(f"{self._dialog_id}_preview", "Error: Please enter a column name")
                return

        # Get value
        value = dpg.get_value(f"{self._dialog_id}_value")
        if not value:
            dpg.set_value(f"{self._dialog_id}_preview", "Error: Please enter a value")
            return

        # Close dialog
        if dpg.does_item_exist(self._dialog_id):
            dpg.delete_item(self._dialog_id)

        # Call callback
        if self.on_assign:
            self.on_assign(column, value)

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


def show_group_assign(
    existing_columns: List[str],
    n_selected: int,
    on_assign: Callable[[str, str], None],
    on_cancel: Optional[Callable] = None
) -> GroupAssignDialog:
    """
    Convenience function to show group assignment dialog.

    Parameters
    ----------
    existing_columns : list of str
        Existing metadata column names
    n_selected : int
        Number of selected samples
    on_assign : callable
        Called with (column, value) when user confirms
    on_cancel : callable, optional
        Called when user cancels

    Returns
    -------
    GroupAssignDialog
        The dialog instance
    """
    dialog = GroupAssignDialog(on_assign=on_assign, on_cancel=on_cancel)
    dialog.show(existing_columns=existing_columns, n_selected=n_selected)
    return dialog
