"""
Main application window for Spectral Predict v3.

Built on Dear PyGui for GPU-accelerated performance.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import os
import threading
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any

from .theme import create_theme, COLORS, on_theme_change
from .components import show_column_config, DataGrid, SpectraPlot, PCAPlot, show_group_assign, DataQualityPanel, ModelDiagnosticsPanel, ParetoPlot
from .panels import CalibrationTransferPanel, ExportPanel
from .tooltips import add_tooltip, TOOLTIP_CONTENT, reset_tooltip_theme
from ..core import Engine
from ..core import io_utils
from ..core.ensemble import create_ensemble
from ..core.data_utils import align_xy
from ..core import export as export_utils
from ..state import SelectionManager


def get_desktop_path() -> Path:
    """Get the user's Desktop path."""
    # Try Windows-style first
    desktop = Path.home() / "Desktop"
    if desktop.exists():
        return desktop
    # Try OneDrive Desktop (common on Windows)
    onedrive_desktop = Path.home() / "OneDrive" / "Desktop"
    if onedrive_desktop.exists():
        return onedrive_desktop
    # Fallback to home
    return Path.home()


class SpectralPredictApp:
    """
    Main application class for Spectral Predict v3.

    Creates the Dear PyGui viewport, applies theme, and manages
    the main application layout.

    Example
    -------
    >>> app = SpectralPredictApp()
    >>> app.run()
    """

    def __init__(self):
        """Initialize the application."""
        self.engine = Engine()
        self._current_dataset = None
        self._theme_id = None
        self._data_grid = None
        self._spectra_plot = None
        self._pca_plot = None
        self._diagnostics_panel = None
        self._pending_path = None
        self._pending_df = None

        # Selection manager for plot-grid sync
        self._selection_manager = SelectionManager()
        self._selection_manager.on_selection_change(self._on_selection_change)

        # Build tab state
        self._selected_result_index = None  # Track selected row in results table

        # Validation holdout state
        self._validation_set = None  # Dict with X_val, y_val, sample_ids, indices
        self._calibration_indices = None
        self._validation_indices = None

        # Predict tab state
        self._loaded_models = []  # List of loaded model bundles (multi-model support)
        self._predict_spectra = None  # Spectra for prediction
        self._predict_wavelengths = None  # Wavelengths for prediction spectra
        self._predict_sample_ids = None  # Sample IDs for prediction spectra
        self._predictions = None  # Prediction results (dict with model names as keys)

        # Settings
        self._sound_notification_enabled = True  # Play sound on search completion
        self._current_theme = "Classic"  # Current color theme

        # Calibration Transfer state
        self._cal_transfer_panel = None

        # Data Management state
        self._data_management_panel = None

        # File dialog callback state (for DearPyGui native dialogs)
        self._file_dialog_callback = None
        self._file_dialog_default_name = None

        # Project state
        self._project_path = None  # Current project file path
        self._project_dirty = False  # True if unsaved changes
        self._project_name = "Untitled"
        self._pending_project_load = None  # For file association

        # Initialize Dear PyGui
        self._setup_dpg()

    def _setup_dpg(self):
        """Set up Dear PyGui context and viewport."""
        dpg.create_context()

        # Configure viewport (main window)
        dpg.create_viewport(
            title="Spectral Predict v3",
            width=1600,
            height=1000,
            min_width=1200,
            min_height=700,
        )

        # Create and bind theme
        self._theme_id = create_theme()
        dpg.bind_theme(self._theme_id)

        # Setup file dialogs (DearPyGui native)
        self._setup_file_dialogs()

        # Create the main window
        self._create_main_window()

    def _setup_file_dialogs(self):
        """Create reusable DearPyGui file dialogs for the application."""
        # Open file dialog (for data import)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_open",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            # Combined filter FIRST = default (spaces in display, commas in braces)
            dpg.add_file_extension("Data Files (*.csv *.xlsx *.xls *.asd *.sig *.spc){.csv,.xlsx,.xls,.asd,.sig,.spc}", color=(0, 255, 0))
            # Individual filters
            dpg.add_file_extension(".csv", color=(0, 255, 0))
            dpg.add_file_extension(".xlsx", color=(0, 255, 0))
            dpg.add_file_extension(".xls", color=(0, 255, 0))
            dpg.add_file_extension(".asd", color=(150, 150, 255))
            dpg.add_file_extension(".sig", color=(150, 150, 255))
            dpg.add_file_extension(".spc", color=(150, 150, 255))
            dpg.add_file_extension(".*")

        # Folder dialog (for batch import)
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self._on_file_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_folder",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            pass

        # Save dialog for model files (.pkl)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_save_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_save_model",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            dpg.add_file_extension(".pkl", color=(0, 255, 0))
            dpg.add_file_extension(".*")

        # Save dialog for CSV files
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_save_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_save_csv",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            dpg.add_file_extension(".csv", color=(0, 255, 0))
            dpg.add_file_extension(".*")

        # Save dialog for Excel files
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_save_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_save_xlsx",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            dpg.add_file_extension(".xlsx", color=(0, 255, 0))
            dpg.add_file_extension(".*")

        # Save dialog for Python code files
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_save_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_save_py",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            dpg.add_file_extension(".py", color=(0, 255, 0))
            dpg.add_file_extension(".*")

        # Open dialog for model files (.pkl)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_open_model",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            dpg.add_file_extension(".pkl", color=(0, 255, 0))
            dpg.add_file_extension(".*")

        # Open dialog for project files (.sproject)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_open_project",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            dpg.add_file_extension(".sproject", color=(0, 200, 255))
            dpg.add_file_extension(".*")

        # Save dialog for project files (.sproject)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_save_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            tag="file_dialog_save_project",
            width=700,
            height=400,
            default_path=str(get_desktop_path()),
        ):
            dpg.add_file_extension(".sproject", color=(0, 200, 255))

    def _on_file_dialog_ok(self, sender, app_data):
        """Handle file selection from open/folder dialog."""
        if app_data and self._file_dialog_callback:
            filepath = app_data.get('file_path_name', '')
            if filepath:
                self._file_dialog_callback(filepath)
        self._file_dialog_callback = None
        self._file_dialog_default_name = None

    def _on_save_dialog_ok(self, sender, app_data):
        """Handle file selection from save dialog."""
        if app_data and self._file_dialog_callback:
            # For save dialogs, construct path from directory + filename
            filepath = app_data.get('file_path_name', '')
            if filepath:
                self._file_dialog_callback(filepath)
        self._file_dialog_callback = None
        self._file_dialog_default_name = None

    def _on_file_dialog_cancel(self, sender, app_data):
        """Handle dialog cancellation."""
        self._file_dialog_callback = None
        self._file_dialog_default_name = None

    def show_open_dialog(self, callback, initial_dir: Path = None):
        """Show open file dialog with callback."""
        self._file_dialog_callback = callback
        if initial_dir:
            dpg.configure_item("file_dialog_open", default_path=str(initial_dir))
        dpg.show_item("file_dialog_open")

    def show_folder_dialog(self, callback, initial_dir: Path = None):
        """Show folder selection dialog with callback."""
        self._file_dialog_callback = callback
        if initial_dir:
            dpg.configure_item("file_dialog_folder", default_path=str(initial_dir))
        dpg.show_item("file_dialog_folder")

    def show_save_dialog(self, callback, dialog_type: str = "model", initial_dir: Path = None):
        """Show save file dialog with callback.

        Parameters
        ----------
        callback : callable
            Function to call with selected filepath
        dialog_type : str
            Type of file: "model" (.pkl), "csv", "xlsx", "py"
        initial_dir : Path, optional
            Initial directory to open
        """
        self._file_dialog_callback = callback
        tag = f"file_dialog_save_{dialog_type}"
        if dpg.does_item_exist(tag):
            if initial_dir:
                dpg.configure_item(tag, default_path=str(initial_dir))
            dpg.show_item(tag)
        else:
            # Fallback to model dialog
            if initial_dir:
                dpg.configure_item("file_dialog_save_model", default_path=str(initial_dir))
            dpg.show_item("file_dialog_save_model")

    def show_open_model_dialog(self, callback, initial_dir: Path = None):
        """Show open file dialog for model files (.pkl)."""
        self._file_dialog_callback = callback
        if initial_dir:
            dpg.configure_item("file_dialog_open_model", default_path=str(initial_dir))
        dpg.show_item("file_dialog_open_model")

    def _create_main_window(self):
        """Create the main application window content."""
        with dpg.window(
            tag="main_window",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_close=True,
        ):
            # Menu bar
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(
                        label="New Project",
                        callback=self._on_new_project,
                        tag="menu_new_project"
                    )
                    dpg.add_menu_item(
                        label="Open Project...",
                        callback=self._on_open_project,
                        tag="menu_open_project"
                    )
                    dpg.add_menu_item(
                        label="Save Project",
                        callback=self._on_save_project,
                        tag="menu_save_project",
                        enabled=False
                    )
                    dpg.add_menu_item(
                        label="Save Project As...",
                        callback=self._on_save_project_as,
                        tag="menu_save_project_as",
                        enabled=False
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Open File...",
                        callback=self._on_open_file
                    )
                    dpg.add_menu_item(
                        label="Open Folder...",
                        callback=self._on_open_folder
                    )
                    dpg.add_separator()
                    with dpg.menu(label="Export"):
                        dpg.add_menu_item(
                            label="Results to CSV...",
                            callback=self._on_export_results_csv,
                            tag="menu_export_results_csv",
                            enabled=False
                        )
                        dpg.add_menu_item(
                            label="Results to Excel...",
                            callback=self._on_export_results_excel,
                            tag="menu_export_results_excel",
                            enabled=False
                        )
                        dpg.add_separator()
                        dpg.add_menu_item(
                            label="Predictions to CSV...",
                            callback=self._on_export_predictions,
                            tag="menu_export_predictions",
                            enabled=False
                        )
                        dpg.add_menu_item(
                            label="Preprocessed Data to CSV...",
                            callback=self._on_export_preprocessed,
                            tag="menu_export_preprocessed",
                            enabled=False
                        )
                        dpg.add_separator()
                        dpg.add_menu_item(
                            label="Code for Publication...",
                            callback=lambda: self._switch_panel("export")
                        )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Exit",
                        callback=self._on_exit
                    )

                with dpg.menu(label="View"):
                    dpg.add_menu_item(
                        label="Import",
                        callback=lambda: self._switch_panel("import")
                    )
                    dpg.add_menu_item(
                        label="Data",
                        callback=lambda: self._switch_panel("data")
                    )
                    dpg.add_menu_item(
                        label="Explore",
                        callback=lambda: self._switch_panel("explore")
                    )
                    dpg.add_menu_item(
                        label="Build",
                        callback=lambda: self._switch_panel("build")
                    )
                    dpg.add_menu_item(
                        label="Predict",
                        callback=lambda: self._switch_panel("predict")
                    )
                    dpg.add_menu_item(
                        label="Cal Transfer",
                        callback=lambda: self._switch_panel("cal_transfer")
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Sound on Completion",
                        callback=self._toggle_sound_notification,
                        check=True,
                        default_value=True,
                        tag="menu_sound_notification"
                    )
                    dpg.add_separator()
                    with dpg.menu(label="Theme"):
                        from .theme import get_available_themes
                        for theme_name in get_available_themes():
                            dpg.add_menu_item(
                                label=theme_name,
                                callback=self._on_theme_menu_click,
                                user_data=theme_name,
                                tag=f"menu_theme_{theme_name.lower()}"
                            )

                with dpg.menu(label="Help"):
                    dpg.add_menu_item(
                        label="About",
                        callback=self._show_about
                    )

            # Main content area - tabbed panels
            with dpg.tab_bar(tag="main_tabs"):
                with dpg.tab(label="Import", tag="tab_import"):
                    self._create_import_panel()

                with dpg.tab(label="Data", tag="tab_data"):
                    self._create_data_panel()

                with dpg.tab(label="Explore", tag="tab_explore"):
                    self._create_explore_panel()

                with dpg.tab(label="Build", tag="tab_build"):
                    self._create_build_panel()

                with dpg.tab(label="Predict", tag="tab_predict"):
                    self._create_predict_panel()

                with dpg.tab(label="Cal Transfer", tag="tab_cal_transfer"):
                    self._create_cal_transfer_panel()

                with dpg.tab(label="Export", tag="tab_export"):
                    self._create_export_panel()

            # Status bar - use table for proper layout
            dpg.add_separator()
            with dpg.table(header_row=False, borders_innerH=False,
                          borders_outerH=False, borders_innerV=False,
                          borders_outerV=False):
                dpg.add_table_column(width_stretch=True)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=200)

                with dpg.table_row():
                    dpg.add_text("Ready", tag="status_text")
                    dpg.add_text("No data loaded", tag="data_status")

    def _create_import_panel(self):
        """Create the Import panel content with data management."""
        from .panels import DataManagementPanel

        with dpg.child_window(height=-30, border=False, tag="import_panel_container"):
            # Tab bar for simple import vs multi-source management
            with dpg.tab_bar(tag="import_tab_bar"):
                # Simple Import Tab
                with dpg.tab(label="Quick Import", tag="import_simple_tab"):
                    dpg.add_spacer(height=10)
                    dpg.add_text("Import Data", color=COLORS["text_secondary"])
                    dpg.add_separator()
                    dpg.add_spacer(height=20)

                    # File browser section
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Browse File...",
                            callback=self._on_open_file,
                            width=150
                        )
                        dpg.add_button(
                            label="Browse Folder...",
                            callback=self._on_open_folder,
                            width=150
                        )

                    dpg.add_spacer(height=20)
                    dpg.add_text(
                        "Or drag and drop files onto this window",
                        color=COLORS["text_muted"]
                    )

                    dpg.add_spacer(height=40)

                    # Status area
                    with dpg.child_window(
                        tag="import_status_area",
                        height=200,
                        border=True
                    ):
                        dpg.add_text(
                            "No files loaded",
                            tag="import_status",
                            color=COLORS["text_muted"]
                        )

                # Multi-Source Management Tab
                with dpg.tab(label="Multi-Source", tag="import_multisource_tab"):
                    dpg.add_spacer(height=10)
                    # Create Data Management Panel
                    self._data_management_panel = DataManagementPanel(
                        parent_tag="import_multisource_tab",
                        on_merge_complete=self._on_data_merge_complete,
                        on_load_folder=self._on_add_folder_source,
                        on_load_file=self._on_add_file_source
                    )

    def _create_data_panel(self):
        """Create the Data panel content with data grid."""
        with dpg.child_window(height=-30, border=False, tag="data_panel_container"):
            # Header with edit controls
            with dpg.group(horizontal=True):
                dpg.add_text("Data Editor", color=COLORS["text_secondary"])
                dpg.add_spacer(width=30)

                # Edit mode toggle
                dpg.add_checkbox(
                    label="Edit Mode",
                    default_value=False,
                    callback=self._on_edit_mode_toggle,
                    tag="edit_mode_checkbox"
                )
                dpg.add_spacer(width=20)

                # Row operations
                dpg.add_button(
                    label="Delete Row",
                    callback=self._on_delete_row,
                    tag="delete_row_btn",
                    enabled=False,
                    width=100
                )
                dpg.add_button(
                    label="Duplicate Row",
                    callback=self._on_duplicate_row,
                    tag="duplicate_row_btn",
                    enabled=False,
                    width=115
                )
                dpg.add_button(
                    label="Insert Row",
                    callback=self._on_insert_row,
                    tag="insert_row_btn",
                    enabled=False,
                    width=100
                )
                dpg.add_spacer(width=20)

                # Column operations
                dpg.add_button(
                    label="Add Column",
                    callback=self._on_add_column,
                    tag="add_column_btn",
                    enabled=False,
                    width=110
                )
                dpg.add_button(
                    label="Delete Column",
                    callback=self._on_delete_column,
                    tag="delete_column_btn",
                    enabled=False,
                    width=115
                )
                dpg.add_spacer(width=20)

                # Undo/Redo
                dpg.add_button(
                    label="Undo",
                    callback=self._on_undo,
                    tag="undo_btn",
                    enabled=False,
                    width=80
                )
                dpg.add_button(
                    label="Redo",
                    callback=self._on_redo,
                    tag="redo_btn",
                    enabled=False,
                    width=80
                )
                dpg.add_spacer(width=20)

                # Fill operations
                dpg.add_button(
                    label="Fill Down",
                    callback=self._on_fill_down,
                    tag="fill_down_btn",
                    enabled=False,
                    width=90
                )

            dpg.add_separator()
            dpg.add_spacer(height=5)

            # Row selection input (for operations)
            with dpg.group(horizontal=True, tag="row_selection_group", show=False):
                dpg.add_text("Row indices (comma-separated):", color=COLORS["text_muted"])
                dpg.add_input_text(
                    tag="row_indices_input",
                    width=200,
                    hint="e.g., 0,1,2,5"
                )
                dpg.add_spacer(width=10)
                dpg.add_text("(0-indexed)", color=COLORS["text_muted"])

            dpg.add_spacer(height=5)

            # Create the data grid
            self._data_grid = DataGrid(
                parent="data_panel_container",
                tag="main_data_grid"
            )
            # Register callback to sync data changes back to app
            self._data_grid.set_on_data_change(self._on_grid_data_changed)

    def _on_grid_data_changed(self):
        """Handle data changes from the grid (column add/delete, cell edits)."""
        # Sync dataset reference - grid and app share the same dataset object
        # but this ensures they stay in sync after structural changes
        if self._data_grid and self._data_grid._dataset:
            self._current_dataset = self._data_grid._dataset
        # Note: Don't call _refresh_data_views() here to avoid infinite loop
        # The grid already rebuilt itself, we just sync the reference

    def _create_explore_panel(self):
        """Create the Explore panel content with spectra, PCA, and data quality visualization."""
        with dpg.child_window(height=-30, border=False, tag="explore_panel_container"):
            # Header row: Title, panel toggles, and controls
            with dpg.group(horizontal=True):
                dpg.add_text("Data Exploration", color=COLORS["text_secondary"])
                dpg.add_spacer(width=20)

                # Panel visibility toggles
                dpg.add_text("Show:", color=COLORS["text_muted"])
                dpg.add_checkbox(
                    label="Spectra",
                    default_value=True,
                    tag="explore_show_spectra",
                    callback=self._on_panel_visibility_change
                )
                dpg.add_checkbox(
                    label="PCA",
                    default_value=True,
                    tag="explore_show_pca",
                    callback=self._on_panel_visibility_change
                )
                dpg.add_checkbox(
                    label="Quality",
                    default_value=True,
                    tag="explore_show_quality",
                    callback=self._on_panel_visibility_change
                )

                dpg.add_spacer(width=20)

                # Color bins control
                dpg.add_text("Color bins:", color=COLORS["text_muted"])
                dpg.add_combo(
                    items=["1", "2", "3", "4", "5", "6", "7", "8"],
                    default_value="4",
                    callback=self._on_bins_change,
                    tag="shared_bins_combo",
                    width=50
                )

                dpg.add_spacer(width=20)

                # Selection info and assign button
                dpg.add_text("0 selected", tag="selection_count_text", color=COLORS["text_muted"])
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Assign to Group",
                    callback=self._on_assign_to_group,
                    tag="assign_group_btn",
                    enabled=False
                )

            dpg.add_separator()
            dpg.add_spacer(height=5)

            # Three panels side by side - equal width (33% each when all visible)
            # We use a table for equal column sizing
            with dpg.table(
                tag="explore_panels_table",
                header_row=False,
                borders_innerV=False,
                borders_outerV=False,
                borders_innerH=False,
                borders_outerH=False,
                resizable=False,
                policy=dpg.mvTable_SizingStretchProp
            ):
                # Add columns for each panel
                dpg.add_table_column(tag="explore_col_spectra", init_width_or_weight=1.0)
                dpg.add_table_column(tag="explore_col_pca", init_width_or_weight=1.0)
                dpg.add_table_column(tag="explore_col_quality", init_width_or_weight=1.0)

                with dpg.table_row():
                    # Spectra plot
                    with dpg.child_window(height=-1, border=True, tag="spectra_container"):
                        self._spectra_plot = SpectraPlot(
                            parent="spectra_container",
                            tag="main_spectra_plot",
                            on_select=self._on_spectra_select
                        )

                    # PCA plot
                    with dpg.child_window(height=-1, border=True, tag="pca_container"):
                        self._pca_plot = PCAPlot(
                            parent="pca_container",
                            tag="main_pca_plot",
                            on_select=self._on_pca_select
                        )

                    # Data Quality panel
                    with dpg.child_window(height=-1, border=True, tag="dataquality_container"):
                        self._data_quality_panel = DataQualityPanel(
                            parent="dataquality_container",
                        tag="main_dq_panel",
                        on_flag=self._on_samples_flagged,
                        on_remove=self._on_samples_removed
                    )

    def _on_samples_flagged(self, indices: List[int]):
        """Handle samples being flagged as outliers."""
        # Highlight flagged samples on PCA plot
        if self._pca_plot and indices:
            self._pca_plot.set_outliers(indices)

    def _on_samples_removed(self, indices: List[int]):
        """Handle samples being removed from the dataset."""
        if self._current_dataset is None or not indices:
            return

        # Remove samples from dataset
        ds = self._current_dataset
        keep_mask = np.ones(ds.n_samples, dtype=bool)
        for idx in indices:
            if idx < len(keep_mask):
                keep_mask[idx] = False

        # Create new dataset without removed samples
        new_X = ds.X[keep_mask]
        new_y = ds.y[keep_mask] if ds.has_target else None
        new_ids = [sid for i, sid in enumerate(ds.sample_ids) if keep_mask[i]]

        # Update the dataset
        ds.X = new_X
        if new_y is not None:
            ds.y = new_y
        ds.sample_ids = new_ids

        # Refresh all views
        self._refresh_data_views()

    def _refresh_data_views(self):
        """Refresh all data views after modification (e.g., sample removal)."""
        ds = self._current_dataset
        if ds is None:
            return

        # Refresh data grid
        if self._data_grid:
            self._data_grid.set_data(ds)

        # Refresh spectra plot
        if self._spectra_plot:
            self._spectra_plot.set_data(ds)

        # Refresh PCA plot
        if self._pca_plot:
            self._pca_plot.set_data(ds)

        # Refresh data quality panel
        if hasattr(self, '_data_quality_panel') and self._data_quality_panel:
            self._data_quality_panel.set_data(ds)

        # Enable export menu for preprocessed data
        if dpg.does_item_exist("menu_export_preprocessed"):
            dpg.configure_item("menu_export_preprocessed", enabled=True)

    def _on_panel_visibility_change(self, sender, app_data):
        """Handle panel visibility toggle changes - auto-sizes panels equally."""
        show_spectra = dpg.get_value("explore_show_spectra")
        show_pca = dpg.get_value("explore_show_pca")
        show_dq = dpg.get_value("explore_show_quality")

        # Count visible panels
        n_visible = sum([show_spectra, show_pca, show_dq])

        if n_visible == 0:
            # Don't allow hiding all panels - re-enable spectra
            dpg.set_value("explore_show_spectra", True)
            show_spectra = True
            n_visible = 1

        # Update column weights based on visibility
        # Hidden columns get weight 0, visible ones share equally
        weight = 1.0 if n_visible > 0 else 0.0

        dpg.configure_item("explore_col_spectra", init_width_or_weight=weight if show_spectra else 0.0)
        dpg.configure_item("explore_col_pca", init_width_or_weight=weight if show_pca else 0.0)
        dpg.configure_item("explore_col_quality", init_width_or_weight=weight if show_dq else 0.0)

        # Also show/hide the containers
        dpg.configure_item("spectra_container", show=show_spectra)
        dpg.configure_item("pca_container", show=show_pca)
        dpg.configure_item("dataquality_container", show=show_dq)

    def _on_bins_change(self, sender, app_data):
        """Handle shared color bins change - update both plots."""
        n_bins = int(app_data)
        if self._spectra_plot:
            self._spectra_plot.set_bins(n_bins)
        if self._pca_plot:
            self._pca_plot.set_bins(n_bins)

    def _on_window_size_change(self, sender=None, app_data=None):
        """Validate window sizes are odd numbers (required for Savitzky-Golay)."""
        primary = dpg.get_value("build_window_primary")
        secondary = dpg.get_value("build_window_secondary")

        # Ensure odd (SG filter requirement)
        if primary % 2 == 0:
            dpg.set_value("build_window_primary", primary + 1)
        if secondary != 0 and secondary % 2 == 0:
            dpg.set_value("build_window_secondary", secondary + 1)

        # Ensure minimum (SG requires window >= 5)
        if primary < 5:
            dpg.set_value("build_window_primary", 5)
        if secondary != 0 and secondary < 5:
            dpg.set_value("build_window_secondary", 5)

    def _on_varsel_checkbox_change(self, sender=None, app_data=None):
        """Show/hide parameter groups based on variable selection checkbox state."""
        # Note: iPLS is now handled separately with its own Phase 4 UI
        if dpg.does_item_exist("uve_params_group"):
            dpg.configure_item("uve_params_group", show=dpg.get_value("build_varsel_uve"))
        if dpg.does_item_exist("spa_params_group"):
            dpg.configure_item("spa_params_group", show=dpg.get_value("build_varsel_spa"))
        if dpg.does_item_exist("uve_spa_params_group"):
            dpg.configure_item("uve_spa_params_group", show=dpg.get_value("build_varsel_uve_spa"))
        if dpg.does_item_exist("ga_pls_params_group"):
            dpg.configure_item("ga_pls_params_group", show=dpg.get_value("build_varsel_ga_pls"))

    def _create_build_panel(self):
        """Create the Build panel content with Manual/Auto mode and model training configuration."""
        self._search_running = False
        self._search_thread = None
        self._search_results = None
        self._pareto_plot = None  # ParetoPlot instance for NSGA-II results
        self._build_mode = 'auto'  # 'manual' or 'auto'

        with dpg.child_window(height=-30, border=False, tag="build_panel_container"):
            dpg.add_text("Model Builder", color=COLORS["text_secondary"])
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Main layout: Settings on left, Results on right
            with dpg.group(horizontal=True):
                # Settings panel (wider to accommodate both modes)
                with dpg.child_window(width=400, height=-1, border=True):
                    # Mode Selection - Manual vs Auto
                    dpg.add_text("Build Mode", color=COLORS["text_secondary"])
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_radio_button(
                            items=["Manual Build", "Auto Build"],
                            default_value="Auto Build",
                            tag="build_mode_radio",
                            horizontal=True,
                            callback=self._on_build_mode_change
                        )
                    dpg.add_text(
                        "Auto: Automated search with preprocessing combos, variable selection",
                        tag="build_mode_desc",
                        color=COLORS["text_muted"],
                        wrap=380
                    )

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    # Task Type (auto-detected from data)
                    dpg.add_text("Task Type", color=COLORS["text_muted"])
                    dpg.add_combo(
                        items=["Regression", "Classification"],
                        default_value="Regression",
                        tag="build_task_type",
                        width=-1,
                        callback=self._on_tier_change
                    )

                    dpg.add_spacer(height=10)

                    # CV Folds (common to both modes)
                    dpg.add_text("CV Folds", color=COLORS["text_muted"])
                    dpg.add_slider_int(
                        default_value=5,
                        min_value=3,
                        max_value=10,
                        tag="build_cv_folds",
                        width=-1
                    )

                    dpg.add_spacer(height=10)

                    # Target Variable selector
                    dpg.add_text("Target Variable", color=COLORS["text_muted"])
                    dpg.add_combo(
                        items=[],
                        default_value="",
                        tag="build_target_variable",
                        width=-1,
                        callback=self._on_target_variable_change
                    )
                    dpg.add_text(
                        "",
                        tag="build_target_info",
                        color=COLORS["text_muted"],
                        wrap=380
                    )

                    dpg.add_spacer(height=15)
                    dpg.add_separator()

                    # ========================================
                    # AUTO BUILD OPTIONS (shown by default)
                    # ========================================
                    with dpg.group(tag="auto_build_options", show=True):
                        dpg.add_spacer(height=10)
                        dpg.add_text("AUTO BUILD SETTINGS", color=COLORS["text_secondary"])
                        dpg.add_spacer(height=10)

                        # Tier Selection (affects models and hyperparameter depth)
                        dpg.add_text("Model Tier", color=COLORS["text_muted"])
                        dpg.add_combo(
                            items=["Quick", "Standard", "Comprehensive", "Custom"],
                            default_value="Standard",
                            tag="build_tier",
                            width=-1,
                            callback=self._on_tier_change
                        )
                        dpg.add_text("", tag="build_tier_info", color=COLORS["text_muted"], wrap=380)

                        dpg.add_spacer(height=10)

                        # Search Mode Selection (Grid Search vs Bayesian vs NSGA-II)
                        dpg.add_text("Search Mode", color=COLORS["text_muted"])
                        dpg.add_combo(
                            items=["Grid Search", "Bayesian (Optuna)", "NSGA-II (Multi-Objective)"],
                            default_value="Grid Search",
                            tag="build_search_mode",
                            width=-1,
                            callback=self._on_search_mode_change
                        )
                        dpg.add_text("", tag="build_search_mode_info", color=COLORS["text_muted"], wrap=380)

                        # Bayesian-specific options (shown only when Bayesian selected)
                        with dpg.group(tag="bayesian_options", show=False):
                            dpg.add_spacer(height=5)
                            dpg.add_text("Optimization Trials:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=50,
                                min_value=20,
                                max_value=200,
                                tag="build_bayesian_trials",
                                width=-1
                            )
                            dpg.add_text(
                                "More trials = better optimization but longer time",
                                color=COLORS["text_muted"],
                                wrap=350
                            )

                        # NSGA-II specific options (shown only when NSGA-II selected)
                        with dpg.group(tag="nsga2_options", show=False):
                            dpg.add_spacer(height=5)
                            dpg.add_text("Population Size:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=50,
                                min_value=20,
                                max_value=100,
                                tag="build_nsga2_population",
                                width=-1
                            )
                            dpg.add_text("Generations:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=100,
                                min_value=30,
                                max_value=300,
                                tag="build_nsga2_generations",
                                width=-1
                            )
                            dpg.add_text("Min Wavelengths:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=10,
                                min_value=5,
                                max_value=100,
                                tag="build_nsga2_min_wavelengths",
                                width=-1
                            )
                            dpg.add_text(
                                "NSGA-II finds Pareto-optimal trade-offs between\n"
                                "prediction error, wavelength count, and complexity.",
                                color=COLORS["text_muted"],
                                wrap=350
                            )

                        dpg.add_spacer(height=10)

                        # Custom model selection (shown only when Custom tier selected)
                        with dpg.group(tag="custom_model_selection", show=False):
                            dpg.add_spacer(height=5)
                            dpg.add_text("Select Models:", color=COLORS["text_muted"])
                            # Regression models
                            with dpg.group(tag="custom_regression_models"):
                                dpg.add_checkbox(label="PLS", tag="custom_model_pls", default_value=True)
                                dpg.add_checkbox(label="Ridge", tag="custom_model_ridge", default_value=True)
                                dpg.add_checkbox(label="Lasso", tag="custom_model_lasso", default_value=False)
                                dpg.add_checkbox(label="ElasticNet", tag="custom_model_elasticnet", default_value=True)
                                dpg.add_checkbox(label="RandomForest", tag="custom_model_rf", default_value=True)
                                dpg.add_checkbox(label="LightGBM", tag="custom_model_lgbm", default_value=True)
                                dpg.add_checkbox(label="XGBoost", tag="custom_model_xgb", default_value=False)
                                dpg.add_checkbox(label="CatBoost", tag="custom_model_catboost", default_value=False)
                                dpg.add_checkbox(label="SVR", tag="custom_model_svr", default_value=False)
                                dpg.add_checkbox(label="NeuralBoosted", tag="custom_model_neuralboosted", default_value=False)
                            # Classification models (hidden when regression)
                            with dpg.group(tag="custom_classification_models", show=False):
                                dpg.add_checkbox(label="PLS-DA", tag="custom_model_plsda", default_value=True)
                                dpg.add_checkbox(label="RandomForest", tag="custom_model_rf_cls", default_value=True)
                                dpg.add_checkbox(label="LightGBM", tag="custom_model_lgbm_cls", default_value=True)
                                dpg.add_checkbox(label="XGBoost", tag="custom_model_xgb_cls", default_value=False)
                                dpg.add_checkbox(label="CatBoost", tag="custom_model_catboost_cls", default_value=False)
                                dpg.add_checkbox(label="SVM", tag="custom_model_svm", default_value=False)
                                dpg.add_checkbox(label="NeuralBoosted", tag="custom_model_neuralboosted_cls", default_value=False)

                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=10)

                        # Data Conversion (applied before preprocessing)
                        dpg.add_text("Data Conversion", color=COLORS["text_muted"])
                        dpg.add_spacer(height=5)
                        dpg.add_checkbox(
                            label="Convert to Absorbance (log10(1/R))",
                            tag="build_convert_absorbance",
                            default_value=False
                        )
                        dpg.add_text(
                            "Enable if your data is in reflectance and you want to work in absorbance space",
                            color=COLORS["text_muted"],
                            wrap=350
                        )

                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=10)

                        # Preprocessing Options (user-selectable, not tied to tier)
                        dpg.add_text("Preprocessing Methods", color=COLORS["text_muted"])
                        dpg.add_spacer(height=5)

                        dpg.add_checkbox(label="Raw (no preprocessing)", tag="build_preproc_raw", default_value=False)
                        dpg.add_checkbox(label="SNV", tag="build_preproc_snv", default_value=True)
                        dpg.add_checkbox(label="1st Derivative", tag="build_preproc_deriv1", default_value=True)
                        dpg.add_checkbox(label="2nd Derivative", tag="build_preproc_deriv2", default_value=True)
                        dpg.add_checkbox(label="3rd Derivative", tag="build_preproc_deriv3", default_value=False)
                        dpg.add_checkbox(label="4th Derivative", tag="build_preproc_deriv4", default_value=False)
                        dpg.add_checkbox(label="SNV + 1st Derivative", tag="build_preproc_snv_deriv1", default_value=True)
                        dpg.add_checkbox(label="SNV + 2nd Derivative", tag="build_preproc_snv_deriv2", default_value=True)
                        dpg.add_checkbox(label="SNV + 3rd Derivative", tag="build_preproc_snv_deriv3", default_value=False)
                        dpg.add_checkbox(label="SNV + 4th Derivative", tag="build_preproc_snv_deriv4", default_value=False)
                        dpg.add_checkbox(label="1st Derivative + SNV", tag="build_preproc_deriv1_snv", default_value=True)
                        dpg.add_checkbox(label="2nd Derivative + SNV", tag="build_preproc_deriv2_snv", default_value=True)
                        dpg.add_checkbox(label="3rd Derivative + SNV", tag="build_preproc_deriv3_snv", default_value=False)
                        dpg.add_checkbox(label="4th Derivative + SNV", tag="build_preproc_deriv4_snv", default_value=False)

                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=10)

                        # Advanced Preprocessing Options (Sprint 3)
                        with dpg.collapsing_header(label="Advanced Preprocessing (MSC, Scatter Correction)", default_open=False):
                            dpg.add_spacer(height=5)
                            dpg.add_checkbox(
                                label="Enable MSC (Multiplicative Scatter Correction)",
                                tag="build_enable_msc",
                                default_value=False
                            )
                            dpg.add_text(
                                "MSC removes scatter effects using reference spectrum",
                                color=COLORS["text_muted"],
                                wrap=350
                            )
                            dpg.add_spacer(height=5)
                            dpg.add_text("MSC Reference:", color=COLORS["text_muted"])
                            dpg.add_combo(
                                items=["mean", "median"],
                                default_value="mean",
                                tag="build_msc_reference",
                                width=-1
                            )

                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=10)

                        # GA Preprocessing Optimization
                        with dpg.collapsing_header(label="GA Preprocessing Optimization", default_open=False):
                            dpg.add_spacer(height=5)
                            dpg.add_checkbox(
                                label="Optimize preprocessing with Genetic Algorithm",
                                tag="build_ga_preprocess",
                                default_value=False,
                                callback=self._on_ga_preprocess_toggle
                            )
                            dpg.add_text(
                                "Uses GA to find optimal preprocessing combination (derivative, window, baseline, smoothing). Replaces manual preprocessing selection.",
                                color=COLORS["text_muted"],
                                wrap=350
                            )
                            dpg.add_text(
                                "",
                                tag="ga_preprocess_info",
                                color=COLORS["accent_primary"],
                                wrap=350
                            )

                            with dpg.group(tag="ga_preprocess_params_group", show=False):
                                dpg.add_spacer(height=8)
                                dpg.add_text("GA Population:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=32,
                                    min_value=16,
                                    max_value=64,
                                    tag="build_ga_preprocess_population",
                                    width=200
                                )
                                dpg.add_text("GA Generations:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=50,
                                    min_value=20,
                                    max_value=100,
                                    tag="build_ga_preprocess_generations",
                                    width=200
                                )
                                dpg.add_text("CV Folds:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=5,
                                    min_value=3,
                                    max_value=10,
                                    tag="build_ga_preprocess_cv_folds",
                                    width=200
                                )

                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=10)

                        # Wavelength Range Selection
                        with dpg.collapsing_header(label="Wavelength Range Selection", default_open=False):
                            dpg.add_spacer(height=5)
                            dpg.add_checkbox(
                                label="Restrict wavelength range for model training",
                                tag="build_enable_wl_range",
                                default_value=False,
                                callback=self._on_wl_range_toggle
                            )
                            dpg.add_text(
                                "Limit analysis to a specific spectral region. Preprocessing is applied to the full spectrum first.",
                                color=COLORS["text_muted"],
                                wrap=350
                            )

                            dpg.add_spacer(height=8)
                            dpg.add_text("Presets:", color=COLORS["text_muted"])
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Full", callback=lambda: self._set_wl_range_preset("full"), width=50)
                                dpg.add_button(label="UV", callback=lambda: self._set_wl_range_preset("uv"), width=40)
                                dpg.add_button(label="VIS", callback=lambda: self._set_wl_range_preset("vis"), width=40)
                                dpg.add_button(label="NIR", callback=lambda: self._set_wl_range_preset("nir"), width=40)
                                dpg.add_button(label="SWIR", callback=lambda: self._set_wl_range_preset("swir"), width=45)
                                dpg.add_button(label="MIR", callback=lambda: self._set_wl_range_preset("mir"), width=40)

                            dpg.add_spacer(height=8)
                            with dpg.group(horizontal=True):
                                dpg.add_text("From:", color=COLORS["text_muted"])
                                dpg.add_input_float(
                                    tag="build_wl_range_min",
                                    default_value=0.0,
                                    width=100,
                                    format="%.0f"
                                )
                                dpg.add_text("nm", color=COLORS["text_muted"])
                                dpg.add_spacer(width=20)
                                dpg.add_text("To:", color=COLORS["text_muted"])
                                dpg.add_input_float(
                                    tag="build_wl_range_max",
                                    default_value=0.0,
                                    width=100,
                                    format="%.0f"
                                )
                                dpg.add_text("nm", color=COLORS["text_muted"])

                            dpg.add_spacer(height=5)
                            dpg.add_text(
                                "",
                                tag="build_wl_range_info",
                                color=COLORS["text_muted"],
                                wrap=350
                            )

                        dpg.add_spacer(height=10)

                        # Interference Removal Options (Sprint 3)
                        with dpg.collapsing_header(label="Interference Removal (Advanced)", default_open=False):
                            dpg.add_spacer(height=5)

                            # Exclude Noisy Regions (moisture bands, etc.)
                            dpg.add_text("Exclude Noisy Regions", color=COLORS["text_primary"])
                            dpg.add_text(
                                "Remove wavelengths with high noise (e.g., water absorption bands)",
                                color=COLORS["text_muted"],
                                wrap=350
                            )
                            dpg.add_spacer(height=5)
                            dpg.add_checkbox(
                                label="Exclude moisture bands (1400-1500, 1900-2000 nm)",
                                tag="build_wavelength_exclude_moisture",
                                default_value=False
                            )
                            dpg.add_text("Additional exclusion ranges:", color=COLORS["text_muted"])
                            dpg.add_input_text(
                                tag="build_wavelength_exclude_custom",
                                default_value="",
                                width=-1,
                                hint="Leave empty for moisture bands only"
                            )

                            dpg.add_spacer(height=10)

                            # OSC (Orthogonal Signal Correction)
                            dpg.add_checkbox(
                                label="Enable OSC (Orthogonal Signal Correction)",
                                tag="build_enable_osc",
                                default_value=False
                            )
                            dpg.add_text(
                                "Remove Y-orthogonal variation (moisture, temperature)",
                                color=COLORS["text_muted"],
                                wrap=350
                            )
                            dpg.add_text("OSC Components:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=1,
                                min_value=1,
                                max_value=5,
                                tag="build_osc_n_components",
                                width=-1
                            )

                            dpg.add_spacer(height=10)

                            # EPO (External Parameter Orthogonalization)
                            dpg.add_checkbox(
                                label="Enable EPO (External Parameter Orthogonalization)",
                                tag="build_enable_epo",
                                default_value=False,
                                enabled=False
                            )
                            dpg.add_text(
                                "Requires interferent library (not implemented in Auto mode)",
                                color=COLORS["text_muted"],
                                wrap=350
                            )

                            dpg.add_spacer(height=10)

                            # GLSW (Generalized Least Squares Weighting)
                            dpg.add_checkbox(
                                label="Enable GLSW (Wavelength Weighting)",
                                tag="build_enable_glsw",
                                default_value=False
                            )
                            dpg.add_text(
                                "Down-weight noisy wavelengths, up-weight informative ones",
                                color=COLORS["text_muted"],
                                wrap=350
                            )
                            dpg.add_text("GLSW Method:", color=COLORS["text_muted"])
                            dpg.add_combo(
                                items=["covariance", "residual"],
                                default_value="covariance",
                                tag="build_glsw_method",
                                width=-1
                            )

                        dpg.add_spacer(height=10)

                        # Baseline Correction & Smoothing (for Raman, fluorescence - hidden by default)
                        with dpg.collapsing_header(label="Baseline Correction & Smoothing (Raman/Fluorescence)", default_open=False):
                            dpg.add_spacer(height=5)
                            dpg.add_text(
                                "For Raman spectroscopy or data with fluorescence background. "
                                "Not typically needed for NIR (derivatives handle baseline).",
                                color=COLORS["text_muted"],
                                wrap=350
                            )

                            dpg.add_spacer(height=10)

                            # Baseline Correction Method
                            dpg.add_text("Baseline Correction:", color=COLORS["text_primary"])
                            dpg.add_combo(
                                items=["None", "Polynomial", "AsLS", "airPLS"],
                                default_value="None",
                                tag="build_baseline_method",
                                width=-1,
                                callback=self._on_baseline_method_change
                            )

                            # Polynomial baseline parameters
                            with dpg.group(tag="baseline_poly_params", show=False):
                                dpg.add_spacer(height=5)
                                dpg.add_text("Polynomial Degree:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=2,
                                    min_value=1,
                                    max_value=5,
                                    tag="build_baseline_poly_degree",
                                    width=-1
                                )
                                dpg.add_text("Number of Segments:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=20,
                                    min_value=5,
                                    max_value=50,
                                    tag="build_baseline_poly_segments",
                                    width=-1
                                )

                            # AsLS/airPLS parameters
                            with dpg.group(tag="baseline_als_params", show=False):
                                dpg.add_spacer(height=5)
                                dpg.add_text("Smoothness (lambda):", color=COLORS["text_muted"])
                                dpg.add_combo(
                                    items=["1e4 (flexible)", "1e5 (moderate)", "1e6 (smooth)", "1e7 (very smooth)"],
                                    default_value="1e5 (moderate)",
                                    tag="build_baseline_lambda",
                                    width=-1
                                )
                                dpg.add_text("Asymmetry (p) - lower fits tighter under peaks:", color=COLORS["text_muted"])
                                dpg.add_slider_float(
                                    default_value=0.01,
                                    min_value=0.001,
                                    max_value=0.1,
                                    tag="build_baseline_p",
                                    width=-1,
                                    format="%.3f"
                                )

                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_spacer(height=5)

                            # Smoothing
                            dpg.add_text("Smoothing:", color=COLORS["text_primary"])
                            dpg.add_checkbox(
                                label="Enable Savitzky-Golay Smoothing",
                                tag="build_enable_smoothing",
                                default_value=False,
                                callback=self._on_smoothing_toggle
                            )

                            with dpg.group(tag="smoothing_params", show=False):
                                dpg.add_spacer(height=5)
                                dpg.add_text("Window Length (odd):", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=11,
                                    min_value=5,
                                    max_value=31,
                                    tag="build_smooth_window",
                                    width=-1
                                )
                                dpg.add_text("Polynomial Order:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=2,
                                    min_value=1,
                                    max_value=4,
                                    tag="build_smooth_polyorder",
                                    width=-1
                                )

                        dpg.add_spacer(height=10)

                        # Advanced Options (collapsible)
                        with dpg.collapsing_header(label="Advanced Options", default_open=False):
                            dpg.add_spacer(height=5)

                            # SG Window sizes
                            dpg.add_text("SG Window Sizes (must be odd):", color=COLORS["text_muted"])
                            with dpg.group(horizontal=True):
                                dpg.add_text("Primary:", color=COLORS["text_muted"])
                                dpg.add_input_int(
                                    default_value=17,
                                    min_value=5,
                                    max_value=51,
                                    tag="build_window_primary",
                                    width=50,
                                    step=0,
                                    callback=self._on_window_size_change
                                )
                                dpg.add_spacer(width=10)
                                dpg.add_text("Secondary:", color=COLORS["text_muted"])
                                dpg.add_input_int(
                                    default_value=0,
                                    min_value=0,
                                    max_value=51,
                                    tag="build_window_secondary",
                                    width=50,
                                    step=0,
                                    callback=self._on_window_size_change
                                )
                            dpg.add_text("(Secondary 0 = unused)", color=COLORS["text_muted"])

                            dpg.add_spacer(height=10)

                            # Variable Selection
                            dpg.add_text("Variable Selection Methods:", color=COLORS["text_muted"])
                            dpg.add_checkbox(label="Feature Importance (RF)", tag="build_varsel_importance", default_value=True)
                            dpg.add_checkbox(label="VIP (PLS-based)", tag="build_varsel_vip", default_value=False)
                            dpg.add_checkbox(label="SPA (Successive Projections)", tag="build_varsel_spa", default_value=False, callback=self._on_varsel_checkbox_change)
                            dpg.add_checkbox(label="UVE (Uninformative Variable Elim.)", tag="build_varsel_uve", default_value=False, callback=self._on_varsel_checkbox_change)
                            dpg.add_checkbox(label="UVE-SPA (Hybrid)", tag="build_varsel_uve_spa", default_value=False, callback=self._on_varsel_checkbox_change)
                            dpg.add_checkbox(label="GA-PLS (Genetic Algorithm)", tag="build_varsel_ga_pls", default_value=False, callback=self._on_varsel_checkbox_change)
                            # Note: iPLS is now its own separate phase with dedicated UI below

                            dpg.add_spacer(height=5)

                            # SPA Parameters (shown when SPA checked)
                            with dpg.group(tag="spa_params_group", show=False):
                                dpg.add_text("  SPA: n_features to select:", color=COLORS["text_muted"])
                                dpg.add_input_int(
                                    default_value=50,
                                    min_value=10,
                                    max_value=500,
                                    tag="build_spa_n_features",
                                    width=150
                                )
                                dpg.add_text("  SPA: random starts:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=10,
                                    min_value=5,
                                    max_value=50,
                                    tag="build_spa_n_random_starts",
                                    width=150
                                )

                            # UVE Parameters (shown when UVE checked)
                            with dpg.group(tag="uve_params_group", show=False):
                                dpg.add_text("  UVE: cutoff multiplier:", color=COLORS["text_muted"])
                                dpg.add_slider_float(
                                    default_value=1.0,
                                    min_value=0.5,
                                    max_value=2.0,
                                    tag="build_uve_cutoff",
                                    width=150,
                                    format="%.2f"
                                )

                            # UVE-SPA Parameters (shown when UVE-SPA checked)
                            with dpg.group(tag="uve_spa_params_group", show=False):
                                dpg.add_text("  UVE-SPA: cutoff multiplier:", color=COLORS["text_muted"])
                                dpg.add_slider_float(
                                    default_value=1.0,
                                    min_value=0.5,
                                    max_value=2.0,
                                    tag="build_uve_spa_cutoff",
                                    width=150,
                                    format="%.2f"
                                )
                                dpg.add_text("  UVE-SPA: n_features to select:", color=COLORS["text_muted"])
                                dpg.add_input_int(
                                    default_value=50,
                                    min_value=10,
                                    max_value=500,
                                    tag="build_uve_spa_n_features",
                                    width=150
                                )

                            # GA-PLS Parameters (shown when GA-PLS checked)
                            with dpg.group(tag="ga_pls_params_group", show=False):
                                dpg.add_text("  GA-PLS: Population size:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=64,
                                    min_value=16,
                                    max_value=128,
                                    tag="build_ga_pls_population",
                                    width=150
                                )
                                dpg.add_text("  GA-PLS: Generations:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=100,
                                    min_value=25,
                                    max_value=300,
                                    tag="build_ga_pls_generations",
                                    width=150
                                )
                                dpg.add_text("  GA-PLS: Multi-runs:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=5,
                                    min_value=1,
                                    max_value=10,
                                    tag="build_ga_pls_n_runs",
                                    width=150
                                )
                                dpg.add_text("  GA-PLS: Selection threshold:", color=COLORS["text_muted"])
                                dpg.add_slider_float(
                                    default_value=0.6,
                                    min_value=0.3,
                                    max_value=0.9,
                                    tag="build_ga_pls_threshold",
                                    width=150,
                                    format="%.2f"
                                )

                            dpg.add_spacer(height=10)

                            # Variable Counts
                            dpg.add_text("Variable Counts to Test:", color=COLORS["text_muted"])
                            dpg.add_input_text(
                                tag="build_var_counts",
                                default_value="20, 50, 100, 250",
                                width=-1,
                                hint="Comma-separated (e.g., 20, 50, 100, 250)"
                            )

                            dpg.add_spacer(height=10)

                            # Region Analysis
                            dpg.add_checkbox(
                                label="Enable Region Analysis",
                                tag="build_enable_regions",
                                default_value=True
                            )
                            dpg.add_text("Top Regions:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=5,
                                min_value=3,
                                max_value=20,
                                tag="build_n_regions",
                                width=-1
                            )

                            dpg.add_spacer(height=10)

                            # iPLS Interval Analysis
                            dpg.add_checkbox(
                                label="Enable iPLS Interval Analysis",
                                tag="build_enable_ipls",
                                default_value=False
                            )
                            dpg.add_text("iPLS Number of Intervals:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=20,
                                min_value=10,
                                max_value=40,
                                tag="build_ipls_n_intervals",
                                width=-1
                            )
                            dpg.add_checkbox(
                                label="Forward iPLS (add best intervals)",
                                tag="build_ipls_forward",
                                default_value=True
                            )
                            dpg.add_checkbox(
                                label="Backward iPLS (remove worst intervals)",
                                tag="build_ipls_backward",
                                default_value=True
                            )
                            dpg.add_text(
                                "Note: iPLS analyzes spectral intervals, not individual variables.",
                                color=COLORS["text_muted"],
                                wrap=350
                            )

                            dpg.add_spacer(height=15)
                            dpg.add_separator()
                            dpg.add_spacer(height=10)

                            # Hyperparameter Grids - show all for transparency
                            dpg.add_text("Hyperparameter Grids", color=COLORS["text_muted"])
                            dpg.add_text(
                                "Values tested for each model (same for all tiers):",
                                color=COLORS["text_muted"],
                                wrap=350
                            )
                            dpg.add_spacer(height=5)

                            # PLS hyperparameters
                            with dpg.collapsing_header(label="PLS / PLS-DA", default_open=False):
                                dpg.add_text("Max Latent Variables:", color=COLORS["text_muted"])
                                dpg.add_slider_int(
                                    default_value=8,
                                    min_value=1,
                                    max_value=20,
                                    tag="hyperparam_pls_max_lv",
                                    width=-1
                                )
                                dpg.add_text("(Tests all values from 1 to max)", color=COLORS["text_muted"])
                                dpg.add_text("scale (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_pls_scale",
                                    default_value="True",
                                    width=-1
                                )
                                dpg.add_text("max_iter:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_pls_maxiter",
                                    default_value="500",
                                    width=-1
                                )
                                dpg.add_text("tol:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_pls_tol",
                                    default_value="1e-06",
                                    width=-1
                                )

                            # Ridge hyperparameters
                            with dpg.collapsing_header(label="Ridge", default_open=False):
                                dpg.add_text("alpha:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_ridge_alpha",
                                    default_value="0.001, 0.01, 0.1, 1.0, 10.0, 100.0",
                                    width=-1
                                )
                                dpg.add_text("fit_intercept (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_ridge_intercept",
                                    default_value="True",
                                    width=-1
                                )
                                dpg.add_text("solver:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_ridge_solver",
                                    default_value="auto",
                                    width=-1,
                                    hint="auto, svd, cholesky, lsqr, sparse_cg, sag, saga"
                                )
                                dpg.add_text("tol:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_ridge_tol",
                                    default_value="0.0001",
                                    width=-1
                                )

                            # Lasso hyperparameters
                            with dpg.collapsing_header(label="Lasso", default_open=False):
                                dpg.add_text("alpha:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lasso_alpha",
                                    default_value="0.001, 0.01, 0.1, 0.5, 1.0, 5.0",
                                    width=-1
                                )
                                dpg.add_text("max_iter:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lasso_maxiter",
                                    default_value="5000",
                                    width=-1
                                )
                                dpg.add_text("tol:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lasso_tol",
                                    default_value="0.0001",
                                    width=-1
                                )
                                dpg.add_text("fit_intercept (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lasso_intercept",
                                    default_value="True",
                                    width=-1
                                )
                                dpg.add_text("selection:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lasso_selection",
                                    default_value="cyclic",
                                    width=-1,
                                    hint="cyclic, random"
                                )

                            # ElasticNet hyperparameters
                            with dpg.collapsing_header(label="ElasticNet", default_open=False):
                                dpg.add_text("alpha:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_elasticnet_alpha",
                                    default_value="0.01, 0.1, 0.5, 1.0",
                                    width=-1
                                )
                                dpg.add_text("l1_ratio:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_elasticnet_l1",
                                    default_value="0.2, 0.5, 0.8",
                                    width=-1
                                )
                                dpg.add_text("max_iter:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_elasticnet_maxiter",
                                    default_value="5000",
                                    width=-1
                                )
                                dpg.add_text("tol:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_elasticnet_tol",
                                    default_value="0.0001",
                                    width=-1
                                )
                                dpg.add_text("fit_intercept (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_elasticnet_intercept",
                                    default_value="True",
                                    width=-1
                                )
                                dpg.add_text("selection:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_elasticnet_selection",
                                    default_value="cyclic",
                                    width=-1,
                                    hint="cyclic, random"
                                )

                            # RandomForest hyperparameters
                            with dpg.collapsing_header(label="RandomForest", default_open=False):
                                dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_estimators",
                                    default_value="100, 200",
                                    width=-1
                                )
                                dpg.add_text("max_depth (0=None):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_depth",
                                    default_value="0, 20, 30",
                                    width=-1
                                )
                                dpg.add_text("min_samples_split:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_min_split",
                                    default_value="2",
                                    width=-1
                                )
                                dpg.add_text("min_samples_leaf:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_min_leaf",
                                    default_value="1",
                                    width=-1
                                )
                                dpg.add_text("max_features:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_max_features",
                                    default_value="sqrt",
                                    width=-1,
                                    hint="sqrt, log2, None, or float (0.0-1.0)"
                                )
                                dpg.add_text("bootstrap (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_bootstrap",
                                    default_value="True",
                                    width=-1
                                )
                                dpg.add_text("max_leaf_nodes (0=None):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_max_leaf",
                                    default_value="0",
                                    width=-1,
                                    hint="0=None, or integer"
                                )
                                dpg.add_text("min_impurity_decrease:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_rf_min_impurity",
                                    default_value="0.0",
                                    width=-1
                                )

                            # LightGBM hyperparameters
                            with dpg.collapsing_header(label="LightGBM", default_open=False):
                                dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_estimators",
                                    default_value="100, 200",
                                    width=-1
                                )
                                dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_lr",
                                    default_value="0.05, 0.1",
                                    width=-1
                                )
                                dpg.add_text("num_leaves:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_leaves",
                                    default_value="31, 63",
                                    width=-1
                                )
                                dpg.add_text("max_depth (-1=no limit):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_depth",
                                    default_value="-1",
                                    width=-1
                                )
                                dpg.add_text("min_child_samples:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_min_child",
                                    default_value="20",
                                    width=-1
                                )
                                dpg.add_text("subsample:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_subsample",
                                    default_value="1.0",
                                    width=-1
                                )
                                dpg.add_text("colsample_bytree:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_colsample",
                                    default_value="1.0",
                                    width=-1
                                )
                                dpg.add_text("reg_alpha (L1):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_reg_alpha",
                                    default_value="0.0",
                                    width=-1
                                )
                                dpg.add_text("reg_lambda (L2):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_lgbm_reg_lambda",
                                    default_value="0.0",
                                    width=-1
                                )

                            # XGBoost hyperparameters
                            with dpg.collapsing_header(label="XGBoost", default_open=False):
                                dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_estimators",
                                    default_value="100, 200",
                                    width=-1
                                )
                                dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_lr",
                                    default_value="0.05, 0.1",
                                    width=-1
                                )
                                dpg.add_text("max_depth:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_depth",
                                    default_value="3, 6",
                                    width=-1
                                )
                                dpg.add_text("min_child_weight:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_min_child",
                                    default_value="1",
                                    width=-1
                                )
                                dpg.add_text("subsample:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_subsample",
                                    default_value="1.0",
                                    width=-1
                                )
                                dpg.add_text("colsample_bytree:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_colsample",
                                    default_value="1.0",
                                    width=-1
                                )
                                dpg.add_text("gamma (min_split_loss):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_gamma",
                                    default_value="0",
                                    width=-1
                                )
                                dpg.add_text("reg_alpha (L1):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_reg_alpha",
                                    default_value="0",
                                    width=-1
                                )
                                dpg.add_text("reg_lambda (L2):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_xgb_reg_lambda",
                                    default_value="1",
                                    width=-1
                                )

                            # CatBoost hyperparameters
                            with dpg.collapsing_header(label="CatBoost", default_open=False):
                                dpg.add_text("iterations:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_catboost_iters",
                                    default_value="100, 200",
                                    width=-1
                                )
                                dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_catboost_lr",
                                    default_value="0.05, 0.1",
                                    width=-1
                                )
                                dpg.add_text("depth:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_catboost_depth",
                                    default_value="4, 6",
                                    width=-1
                                )
                                dpg.add_text("l2_leaf_reg:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_catboost_l2",
                                    default_value="3.0",
                                    width=-1
                                )
                                dpg.add_text("border_count:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_catboost_border",
                                    default_value="254",
                                    width=-1
                                )
                                dpg.add_text("bagging_temperature:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_catboost_bagging",
                                    default_value="1.0",
                                    width=-1
                                )
                                dpg.add_text("random_strength:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_catboost_randstrength",
                                    default_value="1.0",
                                    width=-1
                                )

                            # SVR/SVM hyperparameters
                            with dpg.collapsing_header(label="SVR / SVM", default_open=False):
                                dpg.add_text("kernel:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_kernel",
                                    default_value="rbf, linear",
                                    width=-1,
                                    hint="rbf, linear, poly, sigmoid"
                                )
                                dpg.add_text("C:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_c",
                                    default_value="0.1, 1.0, 10.0, 100.0",
                                    width=-1
                                )
                                dpg.add_text("gamma:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_gamma",
                                    default_value="scale",
                                    width=-1,
                                    hint="scale, auto, or float"
                                )
                                dpg.add_text("epsilon (SVR only):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_epsilon",
                                    default_value="0.1",
                                    width=-1
                                )
                                dpg.add_text("degree (poly kernel):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_degree",
                                    default_value="3",
                                    width=-1
                                )
                                dpg.add_text("coef0 (poly/sigmoid):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_coef0",
                                    default_value="0.0",
                                    width=-1
                                )
                                dpg.add_text("max_iter (-1=no limit):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_maxiter",
                                    default_value="5000",
                                    width=-1
                                )
                                dpg.add_text("shrinking (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_svm_shrinking",
                                    default_value="True",
                                    width=-1
                                )

                            # NeuralBoosted hyperparameters
                            with dpg.collapsing_header(label="NeuralBoosted", default_open=False):
                                dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_nb_estimators",
                                    default_value="50, 100",
                                    width=-1
                                )
                                dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_nb_lr",
                                    default_value="0.05, 0.1",
                                    width=-1
                                )
                                dpg.add_text("hidden_layer_size:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_nb_hidden",
                                    default_value="3, 5",
                                    width=-1
                                )
                                dpg.add_text("max_iter (per estimator):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_nb_maxiter",
                                    default_value="200",
                                    width=-1
                                )
                                dpg.add_text("early_stopping (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_nb_earlystop",
                                    default_value="True",
                                    width=-1
                                )
                                dpg.add_text("tol:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_nb_tol",
                                    default_value="0.01",
                                    width=-1
                                )
                                dpg.add_text("activation:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_nb_activation",
                                    default_value="tanh",
                                    width=-1,
                                    hint="tanh, relu, logistic, identity"
                                )

                            # MLP hyperparameters
                            with dpg.collapsing_header(label="MLP (Multi-Layer Perceptron)", default_open=False):
                                dpg.add_text("hidden_layer_sizes:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_hidden",
                                    default_value="(50,), (100,), (50, 50), (100, 50)",
                                    width=-1,
                                    hint="Tuple format: (50,), (100,), (50, 50)"
                                )
                                dpg.add_text("activation:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_activation",
                                    default_value="relu, tanh",
                                    width=-1,
                                    hint="relu, tanh, logistic, identity"
                                )
                                dpg.add_text("solver:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_solver",
                                    default_value="adam",
                                    width=-1,
                                    hint="adam, sgd, lbfgs"
                                )
                                dpg.add_text("alpha (L2 regularization):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_alpha",
                                    default_value="0.0001, 0.001, 0.01",
                                    width=-1
                                )
                                dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_lr_schedule",
                                    default_value="constant",
                                    width=-1,
                                    hint="constant, invscaling, adaptive"
                                )
                                dpg.add_text("learning_rate_init:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_lr_init",
                                    default_value="0.001",
                                    width=-1
                                )
                                dpg.add_text("max_iter:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_maxiter",
                                    default_value="500",
                                    width=-1
                                )
                                dpg.add_text("batch_size:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_batch",
                                    default_value="auto",
                                    width=-1,
                                    hint="auto, 32, 64, 128, 256"
                                )
                                dpg.add_text("early_stopping (True/False):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_earlystop",
                                    default_value="False",
                                    width=-1
                                )
                                dpg.add_text("momentum (SGD only):", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_momentum",
                                    default_value="0.9",
                                    width=-1
                                )
                                dpg.add_text("tol:", color=COLORS["text_muted"])
                                dpg.add_input_text(
                                    tag="hyperparam_mlp_tol",
                                    default_value="0.0001",
                                    width=-1
                                )

                        dpg.add_spacer(height=10)

                        # Configuration estimate
                        dpg.add_text("", tag="build_config_estimate", color=COLORS["accent_primary"])

                    # ========================================
                    # MANUAL BUILD OPTIONS (hidden by default)
                    # ========================================
                    with dpg.group(tag="manual_build_options", show=False):
                        dpg.add_spacer(height=10)
                        dpg.add_text("MANUAL BUILD SETTINGS", color=COLORS["text_secondary"])
                        dpg.add_spacer(height=10)

                        # Model Selection
                        dpg.add_text("Model", color=COLORS["text_muted"])
                        dpg.add_combo(
                            items=["PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest",
                                   "LightGBM", "XGBoost", "CatBoost", "SVR", "NeuralBoosted"],
                            default_value="PLS",
                            tag="build_manual_model",
                            width=-1,
                            callback=self._on_manual_model_change
                        )

                        dpg.add_spacer(height=10)

                        # Preprocessing
                        dpg.add_text("Preprocessing", color=COLORS["text_muted"])
                        dpg.add_combo(
                            items=["raw", "snv", "deriv1", "deriv2", "snv_deriv1", "snv_deriv2"],
                            default_value="raw",
                            tag="build_manual_preproc",
                            width=-1
                        )

                        dpg.add_spacer(height=5)
                        dpg.add_text("SG Window Size:", color=COLORS["text_muted"])
                        dpg.add_slider_int(
                            default_value=11,
                            min_value=5,
                            max_value=25,
                            tag="build_manual_window",
                            width=-1
                        )

                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=5)

                        # Model-specific hyperparameters
                        dpg.add_text("Hyperparameters", color=COLORS["text_muted"])
                        dpg.add_spacer(height=5)

                        # PLS hyperparameters (shown by default)
                        with dpg.group(tag="manual_params_pls", show=True):
                            dpg.add_text("n_components:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=10,
                                min_value=1,
                                max_value=20,
                                tag="build_pls_components",
                                width=-1
                            )
                            dpg.add_text("scale (True/False):", color=COLORS["text_muted"])
                            dpg.add_checkbox(
                                label="",
                                tag="build_pls_scale",
                                default_value=True
                            )

                        # Ridge hyperparameters
                        with dpg.group(tag="manual_params_ridge", show=False):
                            dpg.add_text("alpha:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=1.0,
                                tag="build_ridge_alpha",
                                width=-1,
                                format="%.6f"
                            )
                            dpg.add_text("fit_intercept:", color=COLORS["text_muted"])
                            dpg.add_checkbox(
                                label="",
                                tag="build_ridge_intercept",
                                default_value=True
                            )
                            dpg.add_text("solver:", color=COLORS["text_muted"])
                            dpg.add_combo(
                                items=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                                default_value="auto",
                                tag="build_ridge_solver",
                                width=-1
                            )

                        # Lasso hyperparameters
                        with dpg.group(tag="manual_params_lasso", show=False):
                            dpg.add_text("alpha:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.1,
                                tag="build_lasso_alpha",
                                width=-1,
                                format="%.6f"
                            )
                            dpg.add_text("max_iter:", color=COLORS["text_muted"])
                            dpg.add_input_int(
                                default_value=5000,
                                tag="build_lasso_maxiter",
                                width=-1
                            )
                            dpg.add_text("tol:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.0001,
                                tag="build_lasso_tol",
                                width=-1,
                                format="%.6f"
                            )
                            dpg.add_text("fit_intercept:", color=COLORS["text_muted"])
                            dpg.add_checkbox(
                                label="",
                                tag="build_lasso_intercept",
                                default_value=True
                            )

                        # ElasticNet hyperparameters
                        with dpg.group(tag="manual_params_elasticnet", show=False):
                            dpg.add_text("alpha:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.1,
                                tag="build_elasticnet_alpha",
                                width=-1,
                                format="%.6f"
                            )
                            dpg.add_text("l1_ratio:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=0.5,
                                min_value=0.0,
                                max_value=1.0,
                                tag="build_elasticnet_l1",
                                width=-1
                            )
                            dpg.add_text("max_iter:", color=COLORS["text_muted"])
                            dpg.add_input_int(
                                default_value=5000,
                                tag="build_elasticnet_maxiter",
                                width=-1
                            )
                            dpg.add_text("tol:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.0001,
                                tag="build_elasticnet_tol",
                                width=-1,
                                format="%.6f"
                            )
                            dpg.add_text("fit_intercept:", color=COLORS["text_muted"])
                            dpg.add_checkbox(
                                label="",
                                tag="build_elasticnet_intercept",
                                default_value=True
                            )

                        # RandomForest hyperparameters
                        with dpg.group(tag="manual_params_rf", show=False):
                            dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=100,
                                min_value=10,
                                max_value=500,
                                tag="build_rf_estimators",
                                width=-1
                            )
                            dpg.add_text("max_depth (0=None):", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=0,
                                min_value=0,
                                max_value=50,
                                tag="build_rf_depth",
                                width=-1
                            )
                            dpg.add_text("min_samples_split:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=2,
                                min_value=2,
                                max_value=20,
                                tag="build_rf_min_split",
                                width=-1
                            )
                            dpg.add_text("min_samples_leaf:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=1,
                                min_value=1,
                                max_value=20,
                                tag="build_rf_min_leaf",
                                width=-1
                            )
                            dpg.add_text("max_features:", color=COLORS["text_muted"])
                            dpg.add_combo(
                                items=["sqrt", "log2", "None", "0.5", "0.75", "1.0"],
                                default_value="sqrt",
                                tag="build_rf_max_features",
                                width=-1
                            )
                            dpg.add_text("bootstrap:", color=COLORS["text_muted"])
                            dpg.add_checkbox(
                                label="",
                                tag="build_rf_bootstrap",
                                default_value=True
                            )

                        # LightGBM hyperparameters
                        with dpg.group(tag="manual_params_lgbm", show=False):
                            dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=100,
                                min_value=10,
                                max_value=500,
                                tag="build_lgbm_estimators",
                                width=-1
                            )
                            dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=0.1,
                                min_value=0.01,
                                max_value=0.5,
                                tag="build_lgbm_lr",
                                width=-1,
                                format="%.3f"
                            )
                            dpg.add_text("num_leaves:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=31,
                                min_value=8,
                                max_value=128,
                                tag="build_lgbm_leaves",
                                width=-1
                            )
                            dpg.add_text("max_depth (-1=no limit):", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=-1,
                                min_value=-1,
                                max_value=30,
                                tag="build_lgbm_depth",
                                width=-1
                            )
                            dpg.add_text("min_child_samples:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=20,
                                min_value=1,
                                max_value=100,
                                tag="build_lgbm_min_child",
                                width=-1
                            )
                            dpg.add_text("subsample:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=1.0,
                                min_value=0.5,
                                max_value=1.0,
                                tag="build_lgbm_subsample",
                                width=-1,
                                format="%.2f"
                            )
                            dpg.add_text("colsample_bytree:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=1.0,
                                min_value=0.5,
                                max_value=1.0,
                                tag="build_lgbm_colsample",
                                width=-1,
                                format="%.2f"
                            )
                            dpg.add_text("reg_alpha (L1):", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.0,
                                tag="build_lgbm_reg_alpha",
                                width=-1,
                                format="%.4f"
                            )
                            dpg.add_text("reg_lambda (L2):", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.0,
                                tag="build_lgbm_reg_lambda",
                                width=-1,
                                format="%.4f"
                            )

                        # XGBoost hyperparameters
                        with dpg.group(tag="manual_params_xgb", show=False):
                            dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=100,
                                min_value=10,
                                max_value=500,
                                tag="build_xgb_estimators",
                                width=-1
                            )
                            dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=0.1,
                                min_value=0.01,
                                max_value=0.5,
                                tag="build_xgb_lr",
                                width=-1,
                                format="%.3f"
                            )
                            dpg.add_text("max_depth:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=6,
                                min_value=1,
                                max_value=15,
                                tag="build_xgb_depth",
                                width=-1
                            )
                            dpg.add_text("min_child_weight:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=1,
                                min_value=1,
                                max_value=20,
                                tag="build_xgb_min_child",
                                width=-1
                            )
                            dpg.add_text("subsample:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=1.0,
                                min_value=0.5,
                                max_value=1.0,
                                tag="build_xgb_subsample",
                                width=-1,
                                format="%.2f"
                            )
                            dpg.add_text("colsample_bytree:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=1.0,
                                min_value=0.5,
                                max_value=1.0,
                                tag="build_xgb_colsample",
                                width=-1,
                                format="%.2f"
                            )
                            dpg.add_text("gamma (min_split_loss):", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.0,
                                tag="build_xgb_gamma",
                                width=-1,
                                format="%.4f"
                            )
                            dpg.add_text("reg_alpha (L1):", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.0,
                                tag="build_xgb_reg_alpha",
                                width=-1,
                                format="%.4f"
                            )
                            dpg.add_text("reg_lambda (L2):", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=1.0,
                                tag="build_xgb_reg_lambda",
                                width=-1,
                                format="%.4f"
                            )

                        # CatBoost hyperparameters
                        with dpg.group(tag="manual_params_catboost", show=False):
                            dpg.add_text("iterations:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=100,
                                min_value=10,
                                max_value=500,
                                tag="build_catboost_iters",
                                width=-1
                            )
                            dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=0.1,
                                min_value=0.01,
                                max_value=0.5,
                                tag="build_catboost_lr",
                                width=-1,
                                format="%.3f"
                            )
                            dpg.add_text("depth:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=6,
                                min_value=1,
                                max_value=16,
                                tag="build_catboost_depth",
                                width=-1
                            )
                            dpg.add_text("l2_leaf_reg:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=3.0,
                                tag="build_catboost_l2",
                                width=-1,
                                format="%.2f"
                            )
                            dpg.add_text("border_count:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=254,
                                min_value=32,
                                max_value=255,
                                tag="build_catboost_border",
                                width=-1
                            )
                            dpg.add_text("bagging_temperature:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=1.0,
                                min_value=0.0,
                                max_value=10.0,
                                tag="build_catboost_bagging",
                                width=-1,
                                format="%.2f"
                            )
                            dpg.add_text("random_strength:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=1.0,
                                min_value=0.0,
                                max_value=10.0,
                                tag="build_catboost_randstrength",
                                width=-1,
                                format="%.2f"
                            )

                        # SVM/SVR hyperparameters
                        with dpg.group(tag="manual_params_svm", show=False):
                            dpg.add_text("kernel:", color=COLORS["text_muted"])
                            dpg.add_combo(
                                items=["rbf", "linear", "poly", "sigmoid"],
                                default_value="rbf",
                                tag="build_svm_kernel",
                                width=-1
                            )
                            dpg.add_text("C:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=1.0,
                                tag="build_svm_c",
                                width=-1,
                                format="%.4f"
                            )
                            dpg.add_text("gamma:", color=COLORS["text_muted"])
                            dpg.add_combo(
                                items=["scale", "auto"],
                                default_value="scale",
                                tag="build_svm_gamma",
                                width=-1
                            )
                            dpg.add_text("epsilon (SVR only):", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.1,
                                tag="build_svm_epsilon",
                                width=-1,
                                format="%.4f"
                            )
                            dpg.add_text("degree (poly kernel):", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=3,
                                min_value=2,
                                max_value=5,
                                tag="build_svm_degree",
                                width=-1
                            )
                            dpg.add_text("coef0 (poly/sigmoid):", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.0,
                                tag="build_svm_coef0",
                                width=-1,
                                format="%.4f"
                            )
                            dpg.add_text("max_iter (-1=no limit):", color=COLORS["text_muted"])
                            dpg.add_input_int(
                                default_value=5000,
                                tag="build_svm_maxiter",
                                width=-1
                            )

                        # NeuralBoosted hyperparameters
                        with dpg.group(tag="manual_params_neuralboosted", show=False):
                            dpg.add_text("n_estimators:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=100,
                                min_value=10,
                                max_value=200,
                                tag="build_nb_estimators",
                                width=-1
                            )
                            dpg.add_text("learning_rate:", color=COLORS["text_muted"])
                            dpg.add_slider_float(
                                default_value=0.1,
                                min_value=0.01,
                                max_value=0.5,
                                tag="build_nb_lr",
                                width=-1,
                                format="%.3f"
                            )
                            dpg.add_text("hidden_layer_size:", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=5,
                                min_value=2,
                                max_value=20,
                                tag="build_nb_hidden",
                                width=-1
                            )
                            dpg.add_text("max_iter (per estimator):", color=COLORS["text_muted"])
                            dpg.add_slider_int(
                                default_value=200,
                                min_value=50,
                                max_value=500,
                                tag="build_nb_maxiter",
                                width=-1
                            )
                            dpg.add_text("early_stopping:", color=COLORS["text_muted"])
                            dpg.add_checkbox(
                                label="",
                                tag="build_nb_earlystop",
                                default_value=True
                            )
                            dpg.add_text("tol:", color=COLORS["text_muted"])
                            dpg.add_input_float(
                                default_value=0.01,
                                tag="build_nb_tol",
                                width=-1,
                                format="%.4f"
                            )

                    dpg.add_spacer(height=15)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    # ========================================
                    # VALIDATION HOLDOUT SECTION
                    # ========================================
                    dpg.add_text("Validation Holdout", color=COLORS["text_secondary"])
                    dpg.add_spacer(height=5)

                    dpg.add_checkbox(
                        label="Hold out validation samples",
                        tag="build_validation_enable",
                        default_value=False,
                        callback=self._on_validation_holdout_change
                    )

                    # Validation options (shown when enabled)
                    with dpg.group(tag="build_validation_options", show=False):
                        dpg.add_spacer(height=5)

                        dpg.add_text("Holdout percentage:", color=COLORS["text_muted"])
                        dpg.add_combo(
                            items=["10%", "15%", "20%", "25%"],
                            default_value="20%",
                            tag="build_validation_percent",
                            width=-1,
                            callback=lambda s, a: self._update_validation_status()
                        )

                        dpg.add_spacer(height=5)
                        dpg.add_text("Selection method:", color=COLORS["text_muted"])
                        dpg.add_combo(
                            items=["SPXY (recommended)", "DUPLEX", "Kennard-Stone", "Random"],
                            default_value="SPXY (recommended)",
                            tag="build_validation_method",
                            width=-1,
                            callback=lambda s, a: self._update_validation_status()
                        )

                        dpg.add_spacer(height=5)
                        dpg.add_text(
                            "",
                            tag="build_validation_status",
                            color=COLORS["accent_primary"],
                            wrap=380
                        )

                    dpg.add_spacer(height=15)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    # Run button
                    dpg.add_button(
                        label="Run Model Search",
                        tag="build_run_btn",
                        callback=self._on_run_search,
                        width=-1,
                        height=40,
                        enabled=False
                    )

                    dpg.add_spacer(height=10)

                    # Progress
                    dpg.add_text("", tag="build_progress_text", color=COLORS["text_muted"], wrap=380)
                    dpg.add_progress_bar(
                        tag="build_progress_bar",
                        default_value=0.0,
                        width=-1,
                        show=False
                    )

                # Results panel
                with dpg.child_window(width=-1, height=-1, border=True, tag="build_results_container"):
                    dpg.add_text("Results", color=COLORS["text_secondary"])
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    dpg.add_text(
                        "Load data and run a search to see model results",
                        tag="build_results_placeholder",
                        color=COLORS["text_muted"]
                    )

                    # Pareto plot container (shown only for NSGA-II results)
                    with dpg.group(tag="build_pareto_container", show=False):
                        dpg.add_text("Pareto Front (Multi-Objective Results)", color=COLORS["text_secondary"])
                        dpg.add_separator()
                        dpg.add_text("Select axis objectives to explore trade-offs", color=COLORS["text_muted"])
                        # The ParetoPlot will be created dynamically here
                        with dpg.group(tag="build_pareto_plot_parent"):
                            pass  # Placeholder for dynamic ParetoPlot
                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=5)

                    # Results table (hidden initially) - show ~15 rows with scroll
                    with dpg.table(
                        tag="build_results_table",
                        header_row=True,
                        freeze_rows=1,  # Keep header visible when scrolling
                        borders_innerH=True,
                        borders_innerV=True,
                        borders_outerH=True,
                        borders_outerV=True,
                        resizable=True,
                        sortable=True,
                        scrollY=True,
                        height=360,  # ~15 rows at 24px each
                        show=False,
                        callback=self._on_results_sort,
                        row_background=True,  # Alternate row colors for readability
                    ):
                        dpg.add_table_column(label="", width_fixed=True, init_width_or_weight=30, no_sort=True)  # Checkbox column
                        dpg.add_table_column(label="#", width_fixed=True, init_width_or_weight=30, no_sort=True)
                        dpg.add_table_column(label="Model", width_fixed=True, init_width_or_weight=100, tag="col_model")
                        dpg.add_table_column(label="Preprocessing", width_fixed=True, init_width_or_weight=110, tag="col_preproc")
                        dpg.add_table_column(label="RMSE", width_fixed=True, init_width_or_weight=65, tag="col_metric1", default_sort=True, prefer_sort_descending=False)
                        dpg.add_table_column(label="R²", width_fixed=True, init_width_or_weight=50, tag="col_metric2")
                        dpg.add_table_column(label="Vars", width_fixed=True, init_width_or_weight=45, tag="col_vars")
                        dpg.add_table_column(label="VarSel", width_fixed=True, init_width_or_weight=55, tag="col_varsel")
                        dpg.add_table_column(label="Params", width_fixed=True, init_width_or_weight=140, tag="col_params")
                        dpg.add_table_column(label="Top Variables (wavelengths)", width_stretch=True, tag="col_topvars")

                    # Save Model section (shown when results exist)
                    dpg.add_spacer(height=5, show=False, tag="build_save_spacer")
                    with dpg.group(horizontal=True, show=False, tag="build_save_group"):
                        dpg.add_button(
                            label="Save Selected Model",
                            callback=self._on_save_model,
                            width=150,
                            enabled=False,
                            tag="build_save_btn"
                        )
                        dpg.add_button(
                            label="Test on Validation Set",
                            callback=self._on_test_validation,
                            width=180,
                            enabled=False,
                            show=False,
                            tag="build_validation_test_btn"
                        )
                        dpg.add_button(
                            label="Show Diagnostics",
                            callback=self._on_show_diagnostics,
                            width=130,
                            enabled=False,
                            tag="build_diagnostics_btn"
                        )
                        dpg.add_text(
                            "Select models using checkboxes",
                            color=COLORS["text_muted"],
                            tag="build_save_status"
                        )

                    # Validation Results (shown after validation test)
                    with dpg.group(show=False, tag="build_validation_results_group"):
                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=5)
                        dpg.add_text("Validation Results", color=COLORS["text_secondary"])
                        dpg.add_text("", tag="build_validation_results_text", color=COLORS["accent_primary"], wrap=-1)

                        # Validation results table
                        with dpg.table(
                            tag="build_validation_table",
                            header_row=True,
                            borders_innerH=True,
                            borders_innerV=True,
                            borders_outerH=True,
                            borders_outerV=True,
                            scrollY=True,
                            height=200,
                            show=False
                        ):
                            dpg.add_table_column(label="Sample", width_fixed=True, init_width_or_weight=100)
                            dpg.add_table_column(label="Actual", width_fixed=True, init_width_or_weight=80)
                            dpg.add_table_column(label="Predicted", width_fixed=True, init_width_or_weight=80)
                            dpg.add_table_column(label="Residual", width_fixed=True, init_width_or_weight=80)
                            dpg.add_table_column(label="Status", width_stretch=True)

                    # Ensemble section (shown when results exist) - collapsible, above diagnostics
                    # When collapsed, takes minimal space; when expanded, pushes diagnostics down
                    with dpg.group(show=False, tag="build_ensemble_group"):
                        dpg.add_spacer(height=5)
                        # Collapsible header for ensemble - starts collapsed
                        with dpg.collapsing_header(label="Create Ensemble", default_open=False, tag="ensemble_header"):
                            dpg.add_spacer(height=5)

                            with dpg.group(horizontal=True):
                                dpg.add_text("Method:", color=COLORS["text_muted"])
                                dpg.add_combo(
                                    items=["Simple Average", "Region Weighted", "Mixture of Experts", "Stacking"],
                                    default_value="Region Weighted",
                                    tag="ensemble_method_combo",
                                    width=150
                                )
                                dpg.add_text("Regions:", color=COLORS["text_muted"])
                                dpg.add_input_int(
                                    default_value=5,
                                    min_value=2,
                                    max_value=10,
                                    min_clamped=True,
                                    max_clamped=True,
                                    tag="ensemble_n_regions",
                                    width=60
                                )
                                dpg.add_button(
                                    label="Create Ensemble",
                                    callback=self._on_create_ensemble,
                                    tag="create_ensemble_btn",
                                    enabled=False
                                )

                            dpg.add_spacer(height=5)
                            dpg.add_text(
                                "Select 2+ models using checkboxes above",
                                tag="ensemble_status_text",
                                color=COLORS["text_muted"]
                            )

                            # Ensemble results display
                            with dpg.group(show=False, tag="ensemble_results_group"):
                                dpg.add_spacer(height=5)
                                dpg.add_text("", tag="ensemble_results_text", color=COLORS["accent_primary"], wrap=-1)

                                # Ensemble comparison table
                                with dpg.table(
                                    tag="ensemble_comparison_table",
                                    header_row=True,
                                    borders_innerH=True,
                                    borders_innerV=True,
                                    borders_outerH=True,
                                    borders_outerV=True,
                                    scrollY=True,
                                    height=120,
                                    show=False
                                ):
                                    dpg.add_table_column(label="Model", width_fixed=True, init_width_or_weight=150)
                                    dpg.add_table_column(label="CV RMSE", width_fixed=True, init_width_or_weight=80)
                                    dpg.add_table_column(label="CV R²", width_fixed=True, init_width_or_weight=60)
                                    dpg.add_table_column(label="Notes", width_stretch=True)

                                dpg.add_spacer(height=5)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(
                                        label="Save Ensemble Model",
                                        callback=self._on_save_ensemble,
                                        tag="save_ensemble_btn",
                                        enabled=False
                                    )
                                    dpg.add_button(
                                        label="Test on Validation",
                                        callback=self._on_test_ensemble_validation,
                                        tag="test_ensemble_validation_btn",
                                        enabled=False,
                                        show=False
                                    )

                    # Model Diagnostics Panel - fills remaining space
                    with dpg.child_window(height=-1, border=False, tag="build_diagnostics_parent"):
                        pass  # ModelDiagnosticsPanel will populate this

        # Create diagnostics panel instance
        self._diagnostics_panel = ModelDiagnosticsPanel(
            parent_tag="build_diagnostics_parent",
            tag="model_diag"
        )

        # Initialize UI state
        self._update_tier_info()

    def _create_predict_panel(self):
        """Create the Predict panel content for model application."""
        with dpg.child_window(height=-30, border=False):
            dpg.add_text("Predict", color=COLORS["text_secondary"])
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Main content in two columns
            with dpg.group(horizontal=True):
                # Left column: Controls
                with dpg.child_window(width=400, border=True):
                    # Load Model section
                    dpg.add_text("1. Load Model(s)", color=COLORS["text_secondary"])
                    dpg.add_separator()
                    dpg.add_spacer(height=5)

                    dpg.add_button(
                        label="Load Model File...",
                        callback=self._on_load_model,
                        width=-1
                    )
                    dpg.add_spacer(height=5)

                    # "Add Additional Model" button (shown after first model loaded)
                    dpg.add_button(
                        label="Add Additional Model...",
                        callback=self._on_load_model,
                        width=-1,
                        show=False,
                        tag="predict_add_model_btn"
                    )
                    dpg.add_spacer(height=5)

                    # Loaded models list
                    with dpg.child_window(height=150, border=True, tag="predict_models_list", show=False):
                        dpg.add_text("Loaded Models:", color=COLORS["text_muted"])
                        dpg.add_spacer(height=5)
                        # Models will be added dynamically here
                        with dpg.group(tag="predict_models_container"):
                            pass

                    dpg.add_spacer(height=5)
                    dpg.add_text("", tag="predict_model_count", color=COLORS["text_muted"], show=False)

                    dpg.add_spacer(height=15)

                    # Load Spectra section
                    dpg.add_text("2. Load Spectra", color=COLORS["text_secondary"])
                    dpg.add_separator()
                    dpg.add_spacer(height=5)

                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Load File...",
                            callback=self._on_load_predict_spectra_file_btn,
                            width=190,
                            enabled=False,
                            tag="predict_load_file_btn"
                        )
                        dpg.add_button(
                            label="Load Folder...",
                            callback=self._on_load_predict_spectra_folder_btn,
                            width=-1,
                            enabled=False,
                            tag="predict_load_folder_btn"
                        )
                    dpg.add_spacer(height=5)

                    # Spectra info display
                    with dpg.group(tag="predict_spectra_info", show=False):
                        dpg.add_text("Spectra loaded:", color=COLORS["text_muted"])
                        dpg.add_text("", tag="predict_spectra_count", color=COLORS["text_primary"])
                        dpg.add_text("Wavelength range:", color=COLORS["text_muted"])
                        dpg.add_text("", tag="predict_spectra_range", color=COLORS["text_primary"])
                        dpg.add_text("", tag="predict_wavelength_warning", color=COLORS["accent_warning"], show=False)

                    dpg.add_spacer(height=15)

                    # Run Prediction section
                    dpg.add_text("3. Run Prediction", color=COLORS["text_secondary"])
                    dpg.add_separator()
                    dpg.add_spacer(height=5)

                    dpg.add_button(
                        label="Predict",
                        callback=self._on_run_prediction,
                        width=-1,
                        enabled=False,
                        tag="predict_run_btn"
                    )
                    dpg.add_spacer(height=5)
                    dpg.add_text("", tag="predict_status", color=COLORS["text_muted"], wrap=370)

                    dpg.add_spacer(height=15)

                    # Export section
                    dpg.add_text("4. Export Results", color=COLORS["text_secondary"])
                    dpg.add_separator()
                    dpg.add_spacer(height=5)

                    dpg.add_button(
                        label="Export to CSV...",
                        callback=self._on_export_predictions,
                        width=-1,
                        enabled=False,
                        tag="predict_export_btn"
                    )

                # Right column: Results table
                with dpg.child_window(width=-1, border=True):
                    dpg.add_text("Prediction Results", color=COLORS["text_secondary"])
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    dpg.add_text(
                        "Load model(s) and spectra to see predictions",
                        tag="predict_results_placeholder",
                        color=COLORS["text_muted"]
                    )

                    # Parent container for results table (will be rebuilt dynamically)
                    with dpg.group(tag="predict_results_parent"):
                        pass

    def _create_cal_transfer_panel(self):
        """Create the Calibration Transfer panel content."""
        with dpg.child_window(height=-30, border=False, tag="cal_transfer_container"):
            # Instantiate the CalibrationTransferPanel with dialog callbacks
            self._cal_transfer_panel = CalibrationTransferPanel(
                parent_tag="cal_transfer_container",
                show_open_dialog_callback=self.show_open_dialog,
                show_folder_dialog_callback=self.show_folder_dialog,
                show_save_dialog_callback=self.show_save_dialog
            )

    def _create_export_panel(self):
        """Create the Export for Publication panel content."""
        with dpg.child_window(height=-30, border=False, tag="export_container"):
            # Instantiate the ExportPanel with callbacks
            self._export_panel = ExportPanel(
                parent_tag="export_container",
                get_model_bundle_callback=self._get_selected_model_bundle,
                show_save_dialog_callback=self.show_save_dialog
            )

    def _get_selected_model_bundle(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected model bundle for export."""
        # Check if we have search results and a selection
        if not hasattr(self, '_search_results') or self._search_results is None:
            return None

        if not hasattr(self, '_selected_result_indices') or not self._selected_result_indices:
            return None

        # Get the first selected result
        idx = list(self._selected_result_indices)[0]
        if idx >= len(self._search_results):
            return None

        row = self._search_results.iloc[idx]

        # Build model bundle from search result
        model_bundle = {
            'model_name': row.get('model', 'Unknown'),
            'preprocessing': row.get('preprocessing', 'raw'),
            'target_name': getattr(self, '_target_name', 'target'),
            'task_type': getattr(self, '_task_type', 'regression'),
            'params': row.get('params', {}),
            'metrics': {
                'RMSE': row.get('rmse', None),
                'R2': row.get('r2', None),
            },
            'variable_indices': row.get('selected_vars', None),
            'variable_selection_method': row.get('varsel', None),
            'cv_folds': getattr(self, '_cv_folds', 5),
            'wavelengths': self._current_dataset.wavelengths if hasattr(self, '_current_dataset') and self._current_dataset else None,
        }

        return model_bundle

    def _switch_panel(self, panel_name: str):
        """Switch to a specific panel tab."""
        tab_map = {
            "import": "tab_import",
            "data": "tab_data",
            "explore": "tab_explore",
            "build": "tab_build",
            "predict": "tab_predict",
            "cal_transfer": "tab_cal_transfer",
            "export": "tab_export",
        }
        if panel_name in tab_map:
            dpg.set_value("main_tabs", tab_map[panel_name])

    # ==================== Project Methods ====================

    def _on_new_project(self, sender=None, app_data=None):
        """Create a new project (clears current data)."""
        # TODO: Check for unsaved changes and prompt
        self._current_dataset = None
        self._project_path = None
        self._project_dirty = False
        self._project_name = "Untitled"

        # Clear data grid
        if self._data_grid:
            self._data_grid.clear()

        # Clear data management panel
        if self._data_management_panel:
            self._data_management_panel.clear_sources()

        # Disable save menu
        dpg.configure_item("menu_save_project", enabled=False)
        dpg.configure_item("menu_save_project_as", enabled=False)

        # Update title
        dpg.set_viewport_title("Spectral Predict v3 - Untitled")
        self._update_status("New project created")

    def _on_open_project(self, sender=None, app_data=None):
        """Open an existing project file."""
        self._file_dialog_callback = self._load_project_file
        dpg.show_item("file_dialog_open_project")

    def _load_project_file(self, filepath: str):
        """Load a .sproject file."""
        from ..core.project_io import load_project, ProjectData
        from ..core.types import SpectralDataset

        try:
            self._update_status(f"Loading project: {filepath}...")

            # Load project data
            project = load_project(filepath)

            # Create SpectralDataset from project data
            dataset = SpectralDataset(
                X=project.X,
                wavelengths=project.wavelengths,
                sample_ids=project.sample_ids,
                y=project.y,
                target_name=project.target_name,
                metadata_columns=project.metadata_columns
            )

            # Set as current dataset
            self._current_dataset = dataset
            self._project_path = filepath
            self._project_dirty = False
            self._project_name = project.project_name

            # Update UI
            if self._data_grid:
                self._data_grid.set_data(dataset)

            # Enable save menu
            dpg.configure_item("menu_save_project", enabled=True)
            dpg.configure_item("menu_save_project_as", enabled=True)

            # Enable export menus if we have results
            # (would need to restore search results from project)

            # Update title
            dpg.set_viewport_title(f"Spectral Predict v3 - {project.project_name}")
            self._update_status(f"Project loaded: {project.project_name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_status(f"Error loading project: {str(e)}")

    def _on_save_project(self, sender=None, app_data=None):
        """Save current project to existing path."""
        if self._project_path:
            self._save_project_to_file(self._project_path)
        else:
            self._on_save_project_as()

    def _on_save_project_as(self, sender=None, app_data=None):
        """Save project to a new path."""
        self._file_dialog_callback = self._save_project_to_file
        if self._project_name:
            dpg.configure_item("file_dialog_save_project",
                               default_filename=f"{self._project_name}.sproject")
        dpg.show_item("file_dialog_save_project")

    def _save_project_to_file(self, filepath: str):
        """Save project state to a .sproject file."""
        from ..core.project_io import save_project, ProjectData, ColumnConfig

        if self._current_dataset is None:
            self._update_status("No data to save")
            return

        try:
            self._update_status(f"Saving project: {filepath}...")

            ds = self._current_dataset

            # Build project data
            project = ProjectData(
                X=ds.X,
                wavelengths=ds.wavelengths,
                sample_ids=ds.sample_ids,
                y=ds.y,
                target_name=ds.target_name,
                metadata_columns=ds.metadata_columns if ds.metadata_columns else {},
                project_name=self._project_name or "Untitled"
            )

            # Add column config if available
            task_type = ds.metadata.get('target_type', 'regression')
            project.column_config = ColumnConfig(
                target_column=ds.target_name,
                task_type=task_type
            )

            # Save to file
            save_project(project, filepath)

            # Update state
            self._project_path = filepath
            self._project_dirty = False

            # Extract project name from filename
            from pathlib import Path
            self._project_name = Path(filepath).stem

            # Update title and status
            dpg.set_viewport_title(f"Spectral Predict v3 - {self._project_name}")
            self._update_status(f"Project saved: {filepath}")

            # Enable save menu
            dpg.configure_item("menu_save_project", enabled=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_status(f"Error saving project: {str(e)}")

    def schedule_project_load(self, filepath: str):
        """Schedule a project to be loaded after UI initialization.

        Used for file association - when app is launched with a .sproject file.
        """
        self._pending_project_load = filepath

    def _enable_project_save(self):
        """Enable save menu items when data is loaded."""
        dpg.configure_item("menu_save_project_as", enabled=True)
        # Only enable Save if we have a project path
        if self._project_path:
            dpg.configure_item("menu_save_project", enabled=True)

    def _on_open_file(self, sender=None, app_data=None):
        """Handle open file action using DearPyGui dialog."""
        self.show_open_dialog(self._load_file, get_desktop_path())

    def _on_open_folder(self, sender=None, app_data=None):
        """Handle open folder action using DearPyGui dialog."""
        self.show_folder_dialog(self._load_file, get_desktop_path())

    def _on_add_folder_source(self):
        """Handle add folder request - opens folder dialog."""
        self.show_folder_dialog(self._add_source_from_folder, get_desktop_path())

    def _on_add_file_source(self):
        """Handle add file request - opens file dialog."""
        self.show_open_dialog(self._add_source_from_file, get_desktop_path())

    def _add_source_from_folder(self, path: str):
        """Load a folder as a data source - auto-finds spectral files and Y values."""
        from pathlib import Path

        path_obj = Path(path).resolve()
        print(f"DEBUG [_add_source_from_folder]: Loading folder {path_obj}")
        self._update_status(f"Loading folder: {path_obj.name}...")

        # Find potential Y value files (CSV/Excel) in the directory
        y_files = io_utils.find_reference_files(path_obj)
        print(f"DEBUG [_add_source_from_folder]: Found {len(y_files)} potential Y files")

        if len(y_files) > 1:
            # Multiple Y files - show selection dialog
            self._pending_source_path = path_obj
            self._show_source_y_file_selection(y_files)
        elif len(y_files) == 1:
            # Single Y file - use it directly
            self._load_source_with_y_file(path_obj, y_files[0])
        else:
            # No Y files - load spectral data only
            self._load_source_simple(path_obj)

    def _add_source_from_file(self, path: str):
        """Load a single file (CSV/Excel) as a data source."""
        from pathlib import Path

        path_obj = Path(path).resolve()

        # Reject directories - user should use "Add Folder" for those
        if path_obj.is_dir():
            self._update_status("Please use 'Add Folder' for directories")
            return

        print(f"DEBUG [_add_source_from_file]: Loading file {path_obj}")
        self._update_status(f"Loading file: {path_obj.name}...")

        # Simple file loading
        self._load_source_simple(path_obj)

    def _load_source_simple(self, path_obj: Path):
        """Load a source without Y file merging."""
        path = str(path_obj)

        def load_thread():
            """Load file in background thread."""
            try:
                print(f"DEBUG [load_thread]: Thread started, loading {path}")
                result = self.engine.load_file(path)
                print(f"DEBUG [load_thread]: engine.load_file returned, samples={result.dataset.n_samples if result.dataset else 0}")
                dataset = result.dataset

                if dataset is None:
                    print("DEBUG [load_thread]: dataset is None")
                    self._update_status("Failed to load data source")
                    return

                # Add to data management panel
                print(f"DEBUG [load_thread]: Adding to panel, X shape={dataset.X.shape}")
                if self._data_management_panel:
                    self._data_management_panel.add_source(
                        X=dataset.X,
                        wavelengths=dataset.wavelengths,
                        sample_ids=dataset.sample_ids,
                        name=path_obj.stem,
                        path=str(path),
                        y=dataset.y,
                        target_name=dataset.target_name,
                        metadata_columns=dataset.metadata_columns
                    )
                    print("DEBUG [load_thread]: add_source completed")

                self._update_status(f"Added source: {path_obj.name}")
                print("DEBUG [load_thread]: Done")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._update_status(f"Error loading source: {str(e)}")

        # Run file loading in background thread
        print("DEBUG [_add_source_from_file]: Starting thread...")
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
        print("DEBUG [_add_source_from_file]: Thread started, returning")

    def _show_source_y_file_selection(self, y_files: list):
        """Show dialog to select which file contains Y values for the source."""
        dialog_id = "source_y_file_select_dialog"

        # Clean up existing dialog if any
        if dpg.does_item_exist(dialog_id):
            dpg.delete_item(dialog_id)

        self._source_y_file_options = y_files

        with dpg.window(
            label="Select Y Value File",
            tag=dialog_id,
            modal=True,
            width=500,
            height=380,
            pos=[550, 300],
            no_resize=True,
        ):
            dpg.add_text("Multiple CSV/Excel files found in folder.")
            dpg.add_text("Select which file contains the Y (target) values:")
            dpg.add_spacer(height=15)

            # List files as radio buttons
            file_names = [f.name for f in y_files]
            dpg.add_radio_button(
                items=file_names,
                tag=f"{dialog_id}_selection",
                default_value=file_names[0]
            )

            dpg.add_spacer(height=15)

            # Option to skip Y file
            dpg.add_checkbox(
                label="Skip Y file (load spectra only)",
                tag=f"{dialog_id}_skip",
                default_value=False
            )

            dpg.add_spacer(height=20)

            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: self._on_source_y_file_cancel(dialog_id),
                    width=100
                )
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="Continue",
                    callback=lambda: self._on_source_y_file_confirm(dialog_id),
                    width=120
                )

    def _on_source_y_file_confirm(self, dialog_id: str):
        """Handle Y file selection confirmation."""
        skip = dpg.get_value(f"{dialog_id}_skip")
        path_obj = self._pending_source_path

        if skip:
            # Load without Y file
            dpg.delete_item(dialog_id)
            self._load_source_simple(path_obj)
            return

        # Get selected file
        selected_name = dpg.get_value(f"{dialog_id}_selection")
        selected_file = None
        for f in self._source_y_file_options:
            if f.name == selected_name:
                selected_file = f
                break

        dpg.delete_item(dialog_id)

        if selected_file:
            self._load_source_with_y_file(path_obj, selected_file)

    def _on_source_y_file_cancel(self, dialog_id: str):
        """Handle Y file selection cancellation."""
        dpg.delete_item(dialog_id)
        self._update_status("Source loading cancelled")

    def _load_source_with_y_file(self, source_path: Path, y_file: Path):
        """Load a source directory and merge Y values from the specified file."""
        def load_thread():
            try:
                print(f"DEBUG [load_with_y]: Loading spectra from {source_path}")
                result = self.engine.load_file(str(source_path))
                dataset = result.dataset

                if dataset is None:
                    self._update_status("Failed to load spectral data")
                    return

                print(f"DEBUG [load_with_y]: Loading Y values from {y_file}")
                # Load Y file and try to merge
                y_df = pd.read_csv(y_file) if y_file.suffix.lower() == '.csv' else pd.read_excel(y_file)

                # Detect columns in Y file
                col_info = io_utils.detect_columns(y_df)
                print(f"DEBUG [load_with_y]: Y file columns: {col_info}")

                # Find ID column and target column
                id_col = col_info['id_candidates'][0] if col_info['id_candidates'] else y_df.columns[0]
                target_cols = col_info['target_candidates']

                if not target_cols:
                    print("DEBUG [load_with_y]: No target columns found in Y file")
                    self._update_status(f"No numeric target columns in {y_file.name}, extracting metadata only")

                    # Still extract metadata columns even if no numeric target
                    metadata_columns = dict(dataset.metadata_columns)
                    all_ref_cols = [c for c in y_df.columns if c != id_col and c not in col_info['wavelength_columns']]
                    print(f"DEBUG [load_with_y]: Extracting {len(all_ref_cols)} metadata columns: {all_ref_cols}")

                    for col_name in all_ref_cols:
                        try:
                            col_values, _ = align_xy(
                                sample_ids=dataset.sample_ids,
                                y_df=y_df,
                                id_column=id_col,
                                target_column=col_name,
                                return_alignment_info=True
                            )
                            col_list = []
                            for val in col_values:
                                try:
                                    if val is None or (isinstance(val, float) and np.isnan(val)):
                                        col_list.append(None)
                                    else:
                                        col_list.append(val)
                                except (TypeError, ValueError):
                                    col_list.append(val)
                            metadata_columns[col_name] = col_list
                            n_valid = sum(1 for v in col_list if v is not None)
                            print(f"DEBUG [load_with_y]: Aligned column '{col_name}' with {n_valid}/{len(col_list)} values")
                        except Exception as e:
                            print(f"DEBUG [load_with_y]: Could not align column '{col_name}': {e}")

                    if self._data_management_panel:
                        self._data_management_panel.add_source(
                            X=dataset.X,
                            wavelengths=dataset.wavelengths,
                            sample_ids=dataset.sample_ids,
                            name=source_path.stem,
                            path=str(source_path),
                            y=None,
                            target_name=None,
                            metadata_columns=metadata_columns
                        )
                    return

                # Use first target column
                target_col = target_cols[0]
                print(f"DEBUG [load_with_y]: Using ID col={id_col}, target col={target_col}")

                # Use align_xy from data_utils for robust multi-strategy matching
                y_array, align_info = align_xy(
                    sample_ids=dataset.sample_ids,
                    y_df=y_df,
                    id_column=id_col,
                    target_column=target_col,
                    return_alignment_info=True
                )

                matched = align_info['n_matched']
                print(f"DEBUG [load_with_y]: Matched {matched}/{len(dataset.sample_ids)} samples")
                print(f"DEBUG [load_with_y]: Strategy used: {align_info['match_strategy_used']}")

                # Set y_array to None if no matches (to indicate no Y values)
                if matched == 0:
                    y_array = None

                # Extract ALL metadata columns from the reference file
                # User can choose which column to use as target later in Build
                metadata_columns = dict(dataset.metadata_columns)  # Start with existing metadata

                # Get all columns that are not the ID column, not wavelengths, and not the target
                # The target is stored separately in y_array/target_name
                # Other columns become metadata so user can switch targets later if needed
                all_ref_cols = [c for c in y_df.columns
                               if c != id_col
                               and c != target_col
                               and c not in col_info['wavelength_columns']]
                print(f"DEBUG [load_with_y]: ID col={id_col}")
                print(f"DEBUG [load_with_y]: Extracting {len(all_ref_cols)} metadata columns: {all_ref_cols}")

                # Align each column with sample_ids (all columns become metadata, including multiple Y values)
                for col_name in all_ref_cols:
                    try:
                        col_values, _ = align_xy(
                            sample_ids=dataset.sample_ids,
                            y_df=y_df,
                            id_column=id_col,
                            target_column=col_name,
                            return_alignment_info=True
                        )
                        # Convert to list, handling NaN/None appropriately for any type
                        col_list = []
                        for val in col_values:
                            try:
                                if val is None or (isinstance(val, float) and np.isnan(val)):
                                    col_list.append(None)
                                else:
                                    col_list.append(val)
                            except (TypeError, ValueError):
                                col_list.append(val)  # Keep as-is if can't check for NaN
                        metadata_columns[col_name] = col_list
                        n_valid = sum(1 for v in col_list if v is not None)
                        print(f"DEBUG [load_with_y]: Aligned column '{col_name}' with {n_valid}/{len(col_list)} values")
                    except Exception as e:
                        print(f"DEBUG [load_with_y]: Could not align column '{col_name}': {e}")

                # Add source to data management panel
                if self._data_management_panel:
                    self._data_management_panel.add_source(
                        X=dataset.X,
                        wavelengths=dataset.wavelengths,
                        sample_ids=dataset.sample_ids,
                        name=source_path.stem,
                        path=str(source_path),
                        y=y_array,
                        target_name=target_col if matched > 0 else None,
                        metadata_columns=metadata_columns
                    )
                self._update_status(f"Added source: {source_path.name} ({matched} Y values matched)")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._update_status(f"Error loading source: {str(e)}")

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def _on_data_merge_complete(self, merge_result):
        """Handle completion of data merge from data management panel."""
        from .panels import MergeResult
        from ..core.types import SpectralDataset

        try:
            # Build metadata_columns dict: start with merged metadata, add datasource
            metadata_columns = dict(merge_result.metadata_columns) if merge_result.metadata_columns else {}
            if merge_result.datasource is not None:
                metadata_columns['datasource'] = merge_result.datasource

            # Convert MergeResult to SpectralDataset
            dataset = SpectralDataset(
                X=merge_result.X,
                wavelengths=merge_result.wavelengths,
                sample_ids=merge_result.sample_ids,
                y=merge_result.y,
                target_name=merge_result.target_name,
                metadata={
                    'merge_strategy': merge_result.strategy,
                    'n_sources': merge_result.n_sources,
                    'merge_report': merge_result.report
                },
                metadata_columns=metadata_columns
            )

            self._current_dataset = dataset

            # Update UI
            self._data_grid.set_data(dataset)
            self._update_explore_plots()
            self._populate_target_variable_combo()  # Populate target dropdown with metadata columns

            # Auto-select first numeric column as target if none was set
            if not dataset.has_target and metadata_columns:
                numeric_target_found = False
                for col_name, col_values in metadata_columns.items():
                    if col_name == 'datasource':
                        continue  # Skip datasource tracking column
                    # Check if column is numeric
                    try:
                        numeric_values = [float(v) for v in col_values if v is not None]
                        if len(numeric_values) > 0:
                            # Found a numeric column - set as target
                            dpg.set_value("build_target_variable", col_name)
                            self._on_target_variable_change()
                            numeric_target_found = True
                            self._update_status(
                                f"Merged: {dataset.X.shape[0]} samples, {dataset.X.shape[1]} wavelengths. "
                                f"Auto-selected '{col_name}' as target."
                            )
                            break
                    except (ValueError, TypeError):
                        continue  # Not numeric, try next column

                if not numeric_target_found:
                    # No numeric column found - prompt user to select target
                    self._update_status(
                        f"Merged: {dataset.X.shape[0]} samples, {dataset.X.shape[1]} wavelengths. "
                        f"Please select a target variable in the Build panel."
                    )
                    # Switch to Build tab so user sees the target selector
                    dpg.set_value("main_tabs", "tab_build")
                    dpg.set_value("build_target_info", "Please select a target variable to run models")
                    # Enable project save
                    self._enable_project_save()
                    return
            else:
                self._update_status(
                    f"Merged: {dataset.X.shape[0]} samples, {dataset.X.shape[1]} wavelengths"
                )

            # Enable project save
            self._enable_project_save()

            # Switch to Data tab
            dpg.set_value("main_tabs", "tab_data")

        except Exception as e:
            self._update_status(f"Error applying merge: {str(e)}")

    def _load_file(self, path: str):
        """Preview a file and show column configuration dialog."""
        import pandas as pd

        self._pending_path = path
        # Normalize path to handle Windows path issues from file dialogs
        path_obj = Path(path).resolve()
        path = str(path_obj)  # Use normalized path string

        try:
            self._update_status(f"Analyzing {path_obj.name}...")

            # Detect format
            format_type = io_utils.detect_format(path)
            print(f"DEBUG: path={path}, suffix={path_obj.suffix}, format_type={format_type}")

            # For directories, check what files are inside
            if path_obj.is_dir():
                # Load spectral data via engine
                result = self.engine.load_file(path)
                self._current_dataset = result.dataset
                self._pending_load_result = result
                self._pending_path = path

                # Look for reference files in the directory
                ref_files = io_utils.find_reference_files(path)

                if len(ref_files) == 1:
                    # Single reference file - use it directly
                    self._load_reference_file(ref_files[0])
                    return
                elif len(ref_files) > 1:
                    # Multiple files - show selection dialog
                    self._show_reference_file_selection(ref_files)
                    return
                else:
                    # No reference file - just show the spectral data
                    self._finalize_load(result)
                    return

            # For files, read and detect columns
            if format_type == 'csv':
                df = pd.read_csv(path)
            elif format_type == 'excel':
                df = pd.read_excel(path)
            else:
                # Other formats - load directly, no column config
                result = self.engine.load_file(path)
                self._current_dataset = result.dataset
                self._finalize_load(result)
                return

            # Detect columns
            column_info = io_utils.detect_columns(df)
            column_info['all_columns'] = list(df.columns)
            column_info['format'] = format_type

            # Store DataFrame for later use
            self._pending_df = df

            # Show column configuration dialog
            show_column_config(
                column_info=column_info,
                on_confirm=self._on_column_config_confirm,
                on_cancel=self._on_column_config_cancel
            )

            self._update_status("Configure columns...")

        except Exception as e:
            dpg.set_value("import_status", f"Error: {str(e)}")
            self._update_status(f"Error: {str(e)}")

    def _on_column_config_confirm(self, config: dict):
        """Handle column configuration confirmation."""
        try:
            import numpy as np
            import pandas as pd
            from ..core.types import SpectralDataset, LoadResult

            path = self._pending_path
            df = self._pending_df
            column_info = config['column_info']

            id_column = config.get('id_column')
            target_column = config.get('target_column')

            # Extract spectral data using configured columns
            wl_cols = column_info['wavelength_columns']

            if not wl_cols:
                raise ValueError("No wavelength columns detected in file")

            # Convert to numeric, coercing errors to NaN
            X_df = df[wl_cols].apply(pd.to_numeric, errors='coerce')

            # Check for non-numeric values that were coerced
            nan_mask = X_df.isna()
            if nan_mask.any().any():
                # Find which columns have issues
                problem_cols = [col for col in wl_cols if nan_mask[col].any()]
                n_problems = nan_mask.sum().sum()
                print(f"Warning: {n_problems} non-numeric values converted to NaN in columns: {problem_cols[:5]}...")
                # Fill NaN with column mean or 0
                X_df = X_df.fillna(X_df.mean())
                X_df = X_df.fillna(0)  # In case entire column was NaN

            X = X_df.values.astype(float)

            # Parse wavelength column names to floats
            wavelengths = []
            for c in wl_cols:
                try:
                    wavelengths.append(float(c))
                except ValueError:
                    # Try extracting number from string like "350nm" or "wavelength_350"
                    import re
                    nums = re.findall(r'[\d.]+', str(c))
                    if nums:
                        wavelengths.append(float(nums[0]))
                    else:
                        raise ValueError(f"Cannot parse wavelength from column name: {c}")
            wavelengths = np.array(wavelengths)

            # Get sample IDs
            if id_column and id_column in df.columns:
                sample_ids = list(df[id_column].astype(str))
            else:
                sample_ids = [f"sample_{i}" for i in range(len(df))]

            # Get target values if specified
            y = None
            target_type = None
            if target_column and target_column in df.columns:
                target_series = df[target_column]
                # Check if numeric or categorical
                if pd.api.types.is_numeric_dtype(target_series):
                    y = target_series.values.astype(float)
                    target_type = 'regression'
                else:
                    # Categorical target - keep as strings for now
                    y = target_series.values
                    target_type = 'classification'

            # Gather ALL non-wavelength columns as metadata
            # User can choose which column to use as target later in Build
            metadata_columns = {}
            wl_set = set(wl_cols)  # wavelength columns to exclude
            all_cols = column_info.get('all_columns', list(df.columns))

            for col in all_cols:
                # Skip only wavelength columns and ID column
                # Include ALL other columns (even auto-selected target) so user can change target later
                if col in wl_set:
                    continue
                if col == id_column:
                    continue
                metadata_columns[col] = list(df[col])

            print(f"DEBUG [column_config]: Retained {len(metadata_columns)} metadata columns: {list(metadata_columns.keys())}")

            dataset = SpectralDataset(
                X=X,
                wavelengths=wavelengths,
                sample_ids=sample_ids,
                y=y,
                target_name=target_column,
                metadata={
                    'file_format': column_info.get('format', 'unknown'),
                    'target_type': target_type
                },
                metadata_columns=metadata_columns
            )

            result = LoadResult(
                dataset=dataset,
                format_detected=column_info.get('format', 'unknown'),
                warnings=[]
            )

            self._current_dataset = dataset
            self._finalize_load(result)

        except Exception as e:
            dpg.set_value("import_status", f"Error: {str(e)}")
            self._update_status(f"Error: {str(e)}")

    def _on_column_config_cancel(self):
        """Handle column configuration cancellation."""
        self._pending_path = None
        self._pending_df = None
        self._update_status("Cancelled")
        dpg.set_value("import_status", "Import cancelled")

    def _load_reference_file(self, ref_file: Path):
        """Load a reference file and show config dialog."""
        import pandas as pd

        self._update_status(f"Found reference: {ref_file.name}")

        if ref_file.suffix.lower() == '.csv':
            ref_df = pd.read_csv(ref_file)
        else:
            ref_df = pd.read_excel(ref_file)

        # Detect columns in reference file
        column_info = io_utils.detect_columns(ref_df)
        column_info['all_columns'] = list(ref_df.columns)
        column_info['format'] = 'reference'
        column_info['ref_file'] = str(ref_file)

        # Store for later
        self._pending_df = ref_df

        # Show column config for reference merge
        show_column_config(
            column_info=column_info,
            on_confirm=self._on_reference_config_confirm,
            on_cancel=self._on_reference_config_cancel
        )
        self._update_status("Configure reference columns...")

    def _show_reference_file_selection(self, ref_files: list):
        """Show dialog to select which reference file to use."""
        dialog_id = "ref_file_select_dialog"

        # Clean up existing dialog if any
        if dpg.does_item_exist(dialog_id):
            dpg.delete_item(dialog_id)

        self._ref_file_options = ref_files

        with dpg.window(
            label="Select Reference File",
            tag=dialog_id,
            modal=True,
            width=500,
            height=350,
            pos=[550, 300],
            no_resize=True,
        ):
            dpg.add_text("Multiple CSV/Excel files found in folder.")
            dpg.add_text("Select which file contains the reference data:")
            dpg.add_spacer(height=15)

            # List files as radio buttons
            file_names = [f.name for f in ref_files]
            dpg.add_radio_button(
                items=file_names,
                tag=f"{dialog_id}_selection",
                default_value=file_names[0]
            )

            dpg.add_spacer(height=15)

            # Option to skip reference
            dpg.add_checkbox(
                label="Skip reference file (load spectra only)",
                tag=f"{dialog_id}_skip",
                default_value=False
            )

            dpg.add_spacer(height=20)

            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: self._on_ref_file_select_cancel(dialog_id),
                    width=100
                )
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="Continue",
                    callback=lambda: self._on_ref_file_select_confirm(dialog_id),
                    width=120
                )

    def _on_ref_file_select_confirm(self, dialog_id: str):
        """Handle reference file selection confirmation."""
        skip = dpg.get_value(f"{dialog_id}_skip")

        if skip:
            # Load without reference
            dpg.delete_item(dialog_id)
            if hasattr(self, '_pending_load_result') and self._pending_load_result:
                self._finalize_load(self._pending_load_result)
            return

        # Get selected file
        selected_name = dpg.get_value(f"{dialog_id}_selection")
        selected_file = None
        for f in self._ref_file_options:
            if f.name == selected_name:
                selected_file = f
                break

        dpg.delete_item(dialog_id)

        if selected_file:
            self._load_reference_file(selected_file)

    def _on_ref_file_select_cancel(self, dialog_id: str):
        """Handle reference file selection cancellation."""
        dpg.delete_item(dialog_id)
        # Load without reference
        if hasattr(self, '_pending_load_result') and self._pending_load_result:
            self._finalize_load(self._pending_load_result)

    def _on_reference_config_confirm(self, config: dict):
        """Handle reference file configuration confirmation."""
        try:
            import numpy as np
            import pandas as pd
            from ..core.types import SpectralDataset, LoadResult

            # Use v3's align_xy function for matching
            from ..core import io as v3_io

            ref_df = self._pending_df
            column_info = config['column_info']
            id_column = config.get('id_column')
            target_column = config.get('target_column')

            if not id_column:
                raise ValueError("ID column is required to merge reference data")

            # Get the already-loaded spectral dataset
            dataset = self._current_dataset

            # Convert to DataFrame format for v3's align_xy
            X_df = pd.DataFrame(
                dataset.X,
                index=dataset.sample_ids,
                columns=dataset.wavelengths
            )

            # Index reference by ID column
            ref_df_indexed = ref_df.set_index(id_column)

            # Use v3's align_xy with fuzzy matching
            X_aligned, y_aligned, align_info = v3_io.align_xy(
                X_df, ref_df_indexed, id_column, target_column,
                return_alignment_info=True
            )

            n_matched = len(X_aligned)
            if n_matched == 0:
                raise ValueError("No samples matched between spectral data and reference file")

            # Handle target type
            target_type = None
            y = None
            if target_column and y_aligned is not None:
                if pd.api.types.is_numeric_dtype(y_aligned):
                    y = y_aligned.values.astype(float)
                    target_type = 'regression'
                else:
                    y = y_aligned.values
                    target_type = 'classification'

            # Extract ALL metadata columns from reference (not just target)
            # This ensures categorical data, multiple Y values, and all other columns are retained
            metadata_columns = {}
            all_ref_cols = list(ref_df_indexed.columns)
            print(f"DEBUG [reference_config]: Total columns in reference file: {len(all_ref_cols)}")
            print(f"DEBUG [reference_config]: Columns to extract: {[c for c in all_ref_cols if c != target_column]}")

            for col in all_ref_cols:
                # Include ALL columns - user can choose target later in Build
                try:
                    # Get values for matched samples
                    matched_ids = X_aligned.index
                    values = [ref_df_indexed.loc[idx, col] if idx in ref_df_indexed.index else None
                              for idx in matched_ids]
                    metadata_columns[col] = values
                    n_valid = sum(1 for v in values if v is not None)
                    print(f"DEBUG [reference_config]: Extracted column '{col}' with {n_valid}/{len(values)} values")
                except Exception as e:
                    print(f"DEBUG [reference_config]: Failed to extract column '{col}': {e}")

            print(f"DEBUG [reference_config]: Retained {len(metadata_columns)} metadata columns: {list(metadata_columns.keys())}")

            # Create merged dataset
            merged_dataset = SpectralDataset(
                X=X_aligned.values.astype(float),
                wavelengths=np.array(X_aligned.columns, dtype=float),
                sample_ids=list(X_aligned.index.astype(str)),
                y=y,
                target_name=target_column,
                metadata={
                    'file_format': dataset.metadata.get('file_format', 'unknown'),
                    'target_type': target_type,
                    'reference_file': column_info.get('ref_file', 'unknown')
                },
                metadata_columns=metadata_columns
            )

            result = LoadResult(
                dataset=merged_dataset,
                format_detected=dataset.metadata.get('file_format', 'unknown'),
                warnings=[]
            )

            self._current_dataset = merged_dataset
            self._finalize_load(result)

            # Show match info
            if n_matched < dataset.n_samples:
                unmatched = dataset.n_samples - n_matched
                dpg.set_value("import_status",
                    dpg.get_value("import_status") + f"\nMatched {n_matched}/{dataset.n_samples} samples"
                )

        except Exception as e:
            import traceback
            traceback.print_exc()

            # Provide helpful error message
            error_msg = str(e)
            if "No samples matched" in error_msg or "match" in error_msg.lower():
                error_msg = (
                    "No samples matched. Make sure the ID column contains values "
                    "that match the spectral filenames (without extension).\n"
                    f"Original error: {e}"
                )

            dpg.set_value("import_status", f"Error: {error_msg}")
            self._update_status("Error - check ID column selection")

            # Offer to load without reference
            if hasattr(self, '_pending_load_result') and self._pending_load_result:
                dpg.set_value("import_status",
                    dpg.get_value("import_status") + "\n\nClick 'Browse Folder' again and cancel the column config to load spectra only."
                )

    def _on_reference_config_cancel(self):
        """Handle reference configuration cancellation - load without reference."""
        # Just load the spectral data without reference
        if hasattr(self, '_pending_load_result') and self._pending_load_result:
            self._finalize_load(self._pending_load_result)
        self._pending_path = None
        self._pending_df = None
        self._pending_load_result = None

    def _finalize_load(self, result):
        """Finalize loading and update UI."""
        # Update UI
        target_info = ""
        if result.dataset.has_target:
            target_type = result.dataset.metadata.get('target_type', 'unknown')
            if target_type == 'classification':
                # Count unique classes
                n_classes = len(set(result.dataset.y))
                target_info = f"\nTarget: {result.dataset.target_name} ({n_classes} classes)"
            else:
                target_info = f"\nTarget: {result.dataset.target_name} (numeric)"

        status_text = (
            f"Loaded: {result.dataset.n_samples} samples x "
            f"{result.dataset.n_wavelengths} wavelengths\n"
            f"Format: {result.format_detected}\n"
            f"Range: {result.dataset.wavelength_range[0]:.0f} - "
            f"{result.dataset.wavelength_range[1]:.0f} nm"
            f"{target_info}"
        )

        dpg.set_value("import_status", status_text)
        dpg.set_value(
            "data_status",
            f"{result.dataset.n_samples} samples loaded"
        )

        # Update data grid
        if self._data_grid:
            self._data_grid.set_data(result.dataset)

        # Update spectra plot
        if self._spectra_plot:
            self._spectra_plot.set_data(result.dataset)

        # Update PCA plot
        if self._pca_plot:
            self._pca_plot.set_data(result.dataset)

        # Update Data Quality panel
        if hasattr(self, '_data_quality_panel') and self._data_quality_panel:
            self._data_quality_panel.set_data(result.dataset)

        # Switch to Data tab to show the loaded data
        self._switch_panel("data")

        # Enable Build panel if target is available
        if result.dataset.has_target:
            dpg.configure_item("build_run_btn", enabled=True)

            # Auto-detect task type from target
            target_type = result.dataset.metadata.get('target_type', 'regression')
            if target_type == 'classification':
                dpg.set_value("build_task_type", "Classification")
            else:
                dpg.set_value("build_task_type", "Regression")

            # Update tier info
            self._update_tier_info()

            # Update validation status if holdout is enabled
            if dpg.get_value("build_validation_enable"):
                self._update_validation_status()

        # Populate target variable dropdown with available columns
        self._populate_target_variable_combo()

        self._update_status("Ready")

    def _update_status(self, message: str):
        """Update status bar message."""
        dpg.set_value("status_text", message)

    def _update_explore_plots(self):
        """Update the Explore tab plots with current dataset."""
        if self._current_dataset is None:
            return

        if self._spectra_plot:
            self._spectra_plot.set_data(self._current_dataset)

        if self._pca_plot:
            self._pca_plot.set_data(self._current_dataset)

    def _show_about(self, sender=None, app_data=None):
        """Show about dialog."""
        with dpg.window(
            label="About",
            modal=True,
            width=400,
            height=200,
            pos=(600, 400),
            no_resize=True,
        ):
            dpg.add_text("Spectral Predict v3")
            dpg.add_text("High-performance spectroscopy analysis")
            dpg.add_spacer(height=10)
            dpg.add_text("Built with Dear PyGui", color=COLORS["text_muted"])
            dpg.add_spacer(height=20)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.delete_item(dpg.get_item_parent(dpg.last_item()))
            )

    def _on_exit(self, sender=None, app_data=None):
        """Handle exit action."""
        dpg.stop_dearpygui()

    def _toggle_sound_notification(self, sender=None, app_data=None):
        """Toggle sound notification setting."""
        self._sound_notification_enabled = dpg.get_value("menu_sound_notification")

    def _on_theme_menu_click(self, sender, app_data, user_data):
        """Handle theme menu item click."""
        theme_name = user_data
        if theme_name:
            self._on_theme_change(theme_name)

    def _on_theme_change(self, theme_name: str):
        """Handle theme selection change."""
        from .theme import create_theme

        # Delete old theme if it exists
        if hasattr(self, '_theme_id') and self._theme_id is not None:
            try:
                dpg.delete_item(self._theme_id)
            except Exception:
                pass

        # Create and apply new theme
        new_theme = create_theme(theme_name)
        self._theme_id = new_theme
        dpg.bind_theme(new_theme)

        # Store current theme name
        self._current_theme = theme_name

    def _play_completion_sound(self):
        """Play a sound notification when search completes."""
        if not self._sound_notification_enabled:
            return

        try:
            import sys
            if sys.platform == 'win32':
                import winsound
                # Play system asterisk sound (non-blocking)
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
            elif sys.platform == 'darwin':
                # macOS - use system beep
                import os
                os.system('afplay /System/Library/Sounds/Glass.aiff &')
            else:
                # Linux - try to use system beep
                print('\a', end='', flush=True)
        except Exception:
            pass  # Silently ignore if sound can't be played

    def _resize_callback(self):
        """Handle window resize."""
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()
        dpg.configure_item("main_window", width=width, height=height)

    # -------------------------------------------------------------------------
    # Selection sync methods
    # -------------------------------------------------------------------------

    def _on_pca_select(self, indices: list):
        """Handle selection from PCA plot."""
        import sys
        print(f"APP: _on_pca_select called with indices={indices}", file=sys.stderr)
        self._selection_manager.select(indices, source="pca_plot")

    def _on_spectra_select(self, indices: list):
        """Handle selection from spectra plot."""
        import sys
        print(f"APP: _on_spectra_select called with indices={indices}", file=sys.stderr)
        self._selection_manager.select(indices, source="spectra_plot")

    def _on_selection_change(self, indices: set, source: str):
        """Handle selection change from any source."""
        import sys
        print(f"APP: _on_selection_change called with indices={indices}, source={source}", file=sys.stderr)
        n_selected = len(indices)

        # Update selection count text
        text_exists = dpg.does_item_exist("selection_count_text")
        print(f"APP: selection_count_text exists: {text_exists}", file=sys.stderr)
        if text_exists:
            dpg.set_value("selection_count_text", f"{n_selected} selected")
            print(f"APP: Set text to '{n_selected} selected'", file=sys.stderr)

        # Enable/disable assign button
        btn_exists = dpg.does_item_exist("assign_group_btn")
        print(f"APP: assign_group_btn exists: {btn_exists}", file=sys.stderr)
        if btn_exists:
            dpg.configure_item("assign_group_btn", enabled=(n_selected > 0))
            print(f"APP: Enabled button: {n_selected > 0}", file=sys.stderr)

        # Sync selection to PCA plot (if not the source)
        if source != "pca_plot" and self._pca_plot:
            self._pca_plot.set_selection(indices)

        # Sync selection to spectra plot (if not the source)
        if source != "spectra_plot" and self._spectra_plot:
            self._spectra_plot.set_selection(indices)

    def _on_assign_to_group(self, sender=None, app_data=None):
        """Handle 'Assign to Group' button click."""
        n_selected = self._selection_manager.selection_count
        if n_selected == 0:
            return

        # Get existing metadata columns
        existing_columns = []
        if self._current_dataset and self._current_dataset.metadata_columns:
            existing_columns = list(self._current_dataset.metadata_columns.keys())

        # Show the group assignment dialog
        show_group_assign(
            existing_columns=existing_columns,
            n_selected=n_selected,
            on_assign=self._on_group_assign_confirm
        )

    def _on_group_assign_confirm(self, column: str, value: str):
        """Handle group assignment confirmation."""
        if not self._current_dataset:
            return

        selected = self._selection_manager.selected_indices

        # Add column if it doesn't exist
        if column not in self._current_dataset.metadata_columns:
            # Initialize with empty strings for all samples
            self._current_dataset.metadata_columns[column] = [""] * self._current_dataset.n_samples

        # Set the value for selected samples
        for idx in selected:
            if 0 <= idx < len(self._current_dataset.metadata_columns[column]):
                self._current_dataset.metadata_columns[column][idx] = value

        # Refresh the data grid
        if self._data_grid:
            self._data_grid.set_data(self._current_dataset)

        # Update status
        self._update_status(f"Assigned '{value}' to {len(selected)} samples in '{column}'")

        # Clear selection
        self._selection_manager.clear()

    # -------------------------------------------------------------------------
    # Data editing methods
    # -------------------------------------------------------------------------

    def _on_edit_mode_toggle(self, sender=None, app_data=None):
        """Handle edit mode toggle."""
        enabled = dpg.get_value("edit_mode_checkbox")

        # Enable/disable edit controls
        dpg.configure_item("delete_row_btn", enabled=enabled)
        dpg.configure_item("duplicate_row_btn", enabled=enabled)
        dpg.configure_item("insert_row_btn", enabled=enabled)
        dpg.configure_item("add_column_btn", enabled=enabled)
        dpg.configure_item("delete_column_btn", enabled=enabled)
        dpg.configure_item("undo_btn", enabled=enabled)
        dpg.configure_item("redo_btn", enabled=enabled)
        dpg.configure_item("fill_down_btn", enabled=enabled)

        # Show/hide row selection input
        dpg.configure_item("row_selection_group", show=enabled)

        # Set edit mode on data grid
        if self._data_grid:
            self._data_grid.set_edit_mode(enabled)

        self._update_status(f"Edit mode {'enabled' if enabled else 'disabled'}")

    def _parse_row_indices(self) -> List[int]:
        """Parse row indices from input field."""
        text = dpg.get_value("row_indices_input")
        if not text:
            return []

        try:
            indices = [int(x.strip()) for x in text.split(',') if x.strip()]
            return indices
        except ValueError:
            self._update_status("Error: Invalid row indices format")
            return []

    def _on_delete_row(self, sender=None, app_data=None):
        """Handle delete row button click."""
        if not self._data_grid:
            return

        indices = self._parse_row_indices()
        if not indices:
            # Use selected row if no indices specified
            selected_row = self._data_grid.get_selected_row()
            if selected_row is not None:
                indices = [selected_row]
            else:
                self._update_status("Click a row to select it, or enter row indices")
                return

        # Store indices for callback
        self._pending_delete_indices = indices

        # Confirm deletion
        with dpg.window(
            label="Confirm Delete",
            modal=True,
            show=True,
            tag="confirm_delete_window",
            autosize=True,
            pos=[400, 300]
        ):
            dpg.add_text(f"Delete {len(indices)} row(s)? This can be undone.")
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Delete",
                    callback=self._do_confirm_delete_rows,
                    width=145
                )
                dpg.add_button(
                    label="Cancel",
                    callback=self._cancel_delete_rows,
                    width=145
                )

    def _do_confirm_delete_rows(self, sender=None, app_data=None):
        """Execute row deletion from stored indices."""
        if dpg.does_item_exist("confirm_delete_window"):
            dpg.delete_item("confirm_delete_window")

        if self._data_grid and hasattr(self, '_pending_delete_indices'):
            indices = self._pending_delete_indices
            self._data_grid.delete_rows(indices)
            dpg.set_value("row_indices_input", "")
            self._update_status(f"Deleted {len(indices)} row(s)")
            del self._pending_delete_indices

    def _cancel_delete_rows(self, sender=None, app_data=None):
        """Cancel row deletion."""
        if dpg.does_item_exist("confirm_delete_window"):
            dpg.delete_item("confirm_delete_window")
        if hasattr(self, '_pending_delete_indices'):
            del self._pending_delete_indices

    def _on_duplicate_row(self, sender=None, app_data=None):
        """Handle duplicate row button click."""
        if not self._data_grid:
            return

        indices = self._parse_row_indices()
        if not indices:
            # Use selected row if no indices specified
            selected_row = self._data_grid.get_selected_row()
            if selected_row is not None:
                indices = [selected_row]
            else:
                self._update_status("Click a row to select it, or enter row indices")
                return

        self._data_grid.duplicate_rows(indices)
        dpg.set_value("row_indices_input", "")  # Clear input
        self._update_status(f"Duplicated {len(indices)} row(s)")

    def _on_insert_row(self, sender=None, app_data=None):
        """Handle insert row button click - insert after selected row or at end."""
        if not self._data_grid or not self._current_dataset:
            return

        # Get selected row position, default to end
        selected_row = self._data_grid.get_selected_row()
        if selected_row is not None:
            # Insert after selected row
            position = selected_row + 1
            self._data_grid.insert_row(position)
            self._update_status(f"Inserted row at position {position} (after selected row)")
        else:
            # Insert at end
            self._data_grid.insert_row(-1)
            self._update_status("Inserted row at end (click a row first to insert after it)")

    def _on_add_column(self, sender=None, app_data=None):
        """Handle add column button click."""
        if not self._data_grid:
            self._update_status("Error: Data grid not initialized")
            return

        if not self._current_dataset:
            self._update_status("Error: No dataset loaded - load data first")
            return

        # Delete existing window if it exists (clean up any stale state)
        if dpg.does_item_exist("add_column_window"):
            dpg.delete_item("add_column_window")

        # Get viewport center for dialog positioning
        vp_width = dpg.get_viewport_client_width()
        vp_height = dpg.get_viewport_client_height()
        pos_x = (vp_width - 350) // 2
        pos_y = (vp_height - 200) // 2

        # Create dialog for column name
        with dpg.window(
            label="Add Column",
            modal=True,
            show=True,
            tag="add_column_window",
            autosize=True,
            pos=[pos_x, pos_y],
            no_close=True,
            no_resize=True,
            no_move=False
        ):
            dpg.add_text("Enter name for new column:")
            dpg.add_spacer(height=8)
            dpg.add_input_text(
                tag="add_column_name_input",
                width=300,
                hint="Column name"
            )
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Add Column",
                    callback=self._confirm_add_column,
                    width=145
                )
                dpg.add_button(
                    label="Cancel",
                    callback=self._cancel_add_column,
                    width=145
                )

    def _cancel_add_column(self, sender=None, app_data=None):
        """Cancel add column dialog."""
        if dpg.does_item_exist("add_column_window"):
            dpg.delete_item("add_column_window")

    def _confirm_add_column(self, sender=None, app_data=None):
        """Confirm and execute column addition."""
        column_name = dpg.get_value("add_column_name_input")

        if dpg.does_item_exist("add_column_window"):
            dpg.delete_item("add_column_window")

        if not column_name or not column_name.strip():
            self._update_status("Error: Column name cannot be empty")
            return

        column_name = column_name.strip()

        # Check if column already exists
        if self._current_dataset and self._current_dataset.metadata_columns:
            if column_name in self._current_dataset.metadata_columns:
                self._update_status(f"Error: Column '{column_name}' already exists")
                return

        if self._data_grid and self._data_grid._dataset:
            success = self._data_grid.add_column(column_name, column_type='metadata', default_value="")
            if success:
                self._update_status(f"Added column '{column_name}'")
            else:
                self._update_status(f"Error: Could not add column '{column_name}'")
        else:
            self._update_status("Error: No data loaded in grid")

    def _on_delete_column(self, sender=None, app_data=None):
        """Handle delete column button click."""
        if not self._data_grid:
            self._update_status("Error: Data grid not initialized")
            return

        if not self._current_dataset:
            self._update_status("Error: No dataset loaded")
            return

        # Get list of metadata columns from the data grid's dataset (same as current_dataset)
        ds = self._data_grid._dataset if self._data_grid else self._current_dataset
        if not ds or not ds.metadata_columns:
            self._update_status("No metadata columns to delete")
            return

        columns = list(ds.metadata_columns.keys())
        if not columns:
            self._update_status("No metadata columns to delete")
            return

        # Delete existing window if it exists
        if dpg.does_item_exist("delete_column_window"):
            dpg.delete_item("delete_column_window")

        # Get viewport center for dialog positioning
        vp_width = dpg.get_viewport_client_width()
        vp_height = dpg.get_viewport_client_height()
        pos_x = (vp_width - 350) // 2
        pos_y = (vp_height - 200) // 2

        # Create dialog to select column
        with dpg.window(
            label="Delete Column",
            modal=True,
            show=True,
            tag="delete_column_window",
            autosize=True,
            pos=[pos_x, pos_y],
            no_close=True,
            no_resize=True,
            no_move=False
        ):
            dpg.add_text("Select column to delete:")
            dpg.add_spacer(height=8)
            dpg.add_combo(
                items=columns,
                tag="delete_column_combo",
                width=300,
                default_value=columns[0] if columns else ""
            )
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Delete",
                    callback=self._confirm_delete_column,
                    width=145
                )
                dpg.add_button(
                    label="Cancel",
                    callback=self._cancel_delete_column,
                    width=145
                )

    def _confirm_delete_column(self, sender=None, app_data=None):
        """Confirm and execute column deletion."""
        column_name = dpg.get_value("delete_column_combo")

        if dpg.does_item_exist("delete_column_window"):
            dpg.delete_item("delete_column_window")

        if not column_name:
            self._update_status("Error: No column selected")
            return

        # Verify column exists before attempting delete
        ds = self._data_grid._dataset if self._data_grid else self._current_dataset
        if not ds or not ds.metadata_columns or column_name not in ds.metadata_columns:
            self._update_status(f"Error: Column '{column_name}' not found")
            return

        if self._data_grid:
            success = self._data_grid.delete_column(column_name)
            if success:
                self._update_status(f"Deleted column '{column_name}'")
            else:
                self._update_status(f"Error: Could not delete column '{column_name}'")

    def _cancel_delete_column(self, sender=None, app_data=None):
        """Cancel delete column dialog."""
        if dpg.does_item_exist("delete_column_window"):
            dpg.delete_item("delete_column_window")

    def _on_undo(self, sender=None, app_data=None):
        """Handle undo button click."""
        if self._data_grid:
            self._data_grid.undo()
            self._update_status("Undo")

    def _on_redo(self, sender=None, app_data=None):
        """Handle redo button click."""
        if self._data_grid:
            self._data_grid.redo()
            self._update_status("Redo")

    def _on_fill_down(self, sender=None, app_data=None):
        """Handle fill down button click - fills selected column from selected row to end."""
        if not self._data_grid or not self._current_dataset:
            self._update_status("Error: No data loaded")
            return

        # Get selected cell
        selected_row = self._data_grid.get_selected_row()
        selected_col = self._data_grid.get_selected_col()

        if selected_row is None or selected_col is None:
            self._update_status("Click a cell first to select the source value for fill down")
            return

        # Get the column type
        col_type = self._data_grid._get_column_type(selected_col)
        if col_type == 'id':
            self._update_status("Cannot fill down the ID column")
            return

        # Perform fill down
        self._data_grid.fill_down(selected_row, selected_col)
        self._update_status(f"Filled down from row {selected_row} to end")

    # -------------------------------------------------------------------------
    # Build panel methods
    # -------------------------------------------------------------------------

    def _on_build_mode_change(self, sender=None, app_data=None):
        """Handle build mode toggle between Manual and Auto."""
        mode = dpg.get_value("build_mode_radio")

        if mode == "Manual Build":
            self._build_mode = 'manual'
            dpg.configure_item("auto_build_options", show=False)
            dpg.configure_item("manual_build_options", show=True)
            dpg.set_value("build_mode_desc", "Manual: Configure a single model with specific settings")
            dpg.configure_item("build_run_btn", label="Train Model")
        else:
            self._build_mode = 'auto'
            dpg.configure_item("auto_build_options", show=True)
            dpg.configure_item("manual_build_options", show=False)
            dpg.set_value("build_mode_desc", "Auto: Automated search with preprocessing combos, variable selection")
            dpg.configure_item("build_run_btn", label="Run Model Search")

    def _on_wl_range_toggle(self, sender=None, app_data=None):
        """Handle wavelength range restriction toggle."""
        enabled = dpg.get_value("build_enable_wl_range")
        if enabled and self._current_dataset is not None:
            # Set default range to current data range
            wl_min = float(self._current_dataset.wavelengths.min())
            wl_max = float(self._current_dataset.wavelengths.max())
            dpg.set_value("build_wl_range_min", wl_min)
            dpg.set_value("build_wl_range_max", wl_max)
            self._update_wl_range_info()
        elif not enabled:
            dpg.set_value("build_wl_range_info", "")

    def _on_ga_preprocess_toggle(self, sender=None, app_data=None):
        """Handle GA preprocessing optimization toggle."""
        enabled = dpg.get_value("build_ga_preprocess")
        if dpg.does_item_exist("ga_preprocess_params_group"):
            dpg.configure_item("ga_preprocess_params_group", show=enabled)

        # List of preprocessing checkboxes to enable/disable
        preproc_checkboxes = [
            "build_preproc_raw", "build_preproc_snv",
            "build_preproc_deriv1", "build_preproc_deriv2",
            "build_preproc_deriv3", "build_preproc_deriv4",
            "build_preproc_snv_deriv1", "build_preproc_snv_deriv2",
            "build_preproc_snv_deriv3", "build_preproc_snv_deriv4",
            "build_preproc_deriv1_snv", "build_preproc_deriv2_snv",
            "build_preproc_deriv3_snv", "build_preproc_deriv4_snv",
        ]

        # Default values for restoration
        preproc_defaults = {
            "build_preproc_raw": True,
            "build_preproc_snv": True,
            "build_preproc_deriv1": True,
            "build_preproc_deriv2": True,
            "build_preproc_deriv3": False,
            "build_preproc_deriv4": False,
            "build_preproc_snv_deriv1": True,
            "build_preproc_snv_deriv2": True,
            "build_preproc_snv_deriv3": False,
            "build_preproc_snv_deriv4": False,
            "build_preproc_deriv1_snv": False,
            "build_preproc_deriv2_snv": False,
            "build_preproc_deriv3_snv": False,
            "build_preproc_deriv4_snv": False,
        }

        if enabled:
            # === GA PREPROCESSING ENABLED ===
            # Uncheck AND disable preprocessing checkboxes - GA will explore all options
            for tag in preproc_checkboxes:
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, False)
                    dpg.configure_item(tag, enabled=False)

            # Disable window size inputs
            for tag in ["build_window_primary", "build_window_secondary"]:
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, enabled=False)

            # Disable baseline method and hide parameter groups
            if dpg.does_item_exist("build_baseline_method"):
                dpg.set_value("build_baseline_method", "None")
                dpg.configure_item("build_baseline_method", enabled=False)
            for group in ["baseline_poly_params", "baseline_asls_params", "baseline_airpls_params"]:
                if dpg.does_item_exist(group):
                    dpg.configure_item(group, show=False)

            # Disable smoothing checkbox and hide params
            if dpg.does_item_exist("build_enable_smoothing"):
                dpg.set_value("build_enable_smoothing", False)
                dpg.configure_item("build_enable_smoothing", enabled=False)
            if dpg.does_item_exist("smoothing_params"):
                dpg.configure_item("smoothing_params", show=False)

            # Show info message
            if dpg.does_item_exist("ga_preprocess_info"):
                dpg.set_value("ga_preprocess_info", "GA explores ALL preprocessing options automatically")
        else:
            # === GA PREPROCESSING DISABLED ===
            # Only re-enable if not in Bayesian/NSGA-II mode
            search_mode = dpg.get_value("build_search_mode") if dpg.does_item_exist("build_search_mode") else "Grid Search"
            is_auto_mode = "Bayesian" in search_mode or "NSGA-II" in search_mode

            if not is_auto_mode:
                # Re-enable preprocessing checkboxes and restore defaults
                for tag in preproc_checkboxes:
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, preproc_defaults.get(tag, False))
                        dpg.configure_item(tag, enabled=True)

                # Re-enable window size inputs
                for tag in ["build_window_primary", "build_window_secondary"]:
                    if dpg.does_item_exist(tag):
                        dpg.configure_item(tag, enabled=True)

                # Re-enable baseline method
                if dpg.does_item_exist("build_baseline_method"):
                    dpg.configure_item("build_baseline_method", enabled=True)

                # Re-enable smoothing checkbox
                if dpg.does_item_exist("build_enable_smoothing"):
                    dpg.configure_item("build_enable_smoothing", enabled=True)

            # Clear info message
            if dpg.does_item_exist("ga_preprocess_info"):
                dpg.set_value("ga_preprocess_info", "")

    def _set_wl_range_preset(self, preset: str):
        """Set wavelength range from preset."""
        presets = {
            "full": (0, 50000),  # Will be clamped to data range
            "uv": (10, 400),
            "vis": (400, 800),
            "nir": (800, 2500),
            "swir": (1000, 2500),
            "mir": (2500, 25000),
        }

        if preset not in presets:
            return

        wl_min, wl_max = presets[preset]

        # Enable the checkbox if not already enabled
        dpg.set_value("build_enable_wl_range", True)

        # Clamp to data range if data is loaded
        if self._current_dataset is not None:
            data_min = float(self._current_dataset.wavelengths.min())
            data_max = float(self._current_dataset.wavelengths.max())

            if preset == "full":
                wl_min, wl_max = data_min, data_max
            else:
                # Clamp preset to data range
                wl_min = max(wl_min, data_min)
                wl_max = min(wl_max, data_max)

        dpg.set_value("build_wl_range_min", wl_min)
        dpg.set_value("build_wl_range_max", wl_max)
        self._update_wl_range_info()

    def _update_wl_range_info(self):
        """Update wavelength range info text."""
        if self._current_dataset is None:
            dpg.set_value("build_wl_range_info", "Load data to see wavelength count")
            return

        wl_min = dpg.get_value("build_wl_range_min")
        wl_max = dpg.get_value("build_wl_range_max")

        if wl_min >= wl_max:
            dpg.set_value("build_wl_range_info", "Error: Min must be less than Max")
            return

        # Count wavelengths in range
        wavelengths = self._current_dataset.wavelengths
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        n_in_range = mask.sum()
        n_total = len(wavelengths)

        info = f"{n_in_range} of {n_total} wavelengths selected ({wl_min:.0f}-{wl_max:.0f} nm)"

        # Warn if range is very small
        if n_in_range < 50:
            info += "\nNote: Small range may limit variable selection methods"

        dpg.set_value("build_wl_range_info", info)

    def _on_validation_holdout_change(self, sender=None, app_data=None):
        """Handle validation holdout checkbox toggle."""
        enabled = dpg.get_value("build_validation_enable")
        dpg.configure_item("build_validation_options", show=enabled)

        # Update status text if data is loaded
        if enabled and self._current_dataset is not None:
            self._update_validation_status()
        else:
            dpg.set_value("build_validation_status", "")

    def _update_validation_status(self):
        """Update the validation holdout status text based on current dataset."""
        if self._current_dataset is None:
            return

        n_total = self._current_dataset.X.shape[0]

        # Parse percentage
        percent_str = dpg.get_value("build_validation_percent")
        percent = int(percent_str.replace("%", "")) / 100.0

        n_val = int(n_total * percent)
        n_cal = n_total - n_val

        # Get method name
        method_str = dpg.get_value("build_validation_method")
        method_name = method_str.split(" ")[0]  # e.g., "SPXY" from "SPXY (recommended)"

        status = f"Training: {n_cal} samples | Validation: {n_val} samples ({method_name})"
        dpg.set_value("build_validation_status", status)

    def _on_manual_model_change(self, sender=None, app_data=None):
        """Handle manual model selection change - show appropriate hyperparameters."""
        model = dpg.get_value("build_manual_model")

        # Hide all parameter groups first
        all_param_groups = [
            "manual_params_pls", "manual_params_ridge", "manual_params_lasso",
            "manual_params_elasticnet", "manual_params_rf", "manual_params_lgbm",
            "manual_params_xgb", "manual_params_catboost", "manual_params_svm",
            "manual_params_neuralboosted"
        ]
        for group in all_param_groups:
            if dpg.does_item_exist(group):
                dpg.configure_item(group, show=False)

        # Show appropriate parameter group
        if model in ["PLS", "PLS-DA"]:
            dpg.configure_item("manual_params_pls", show=True)
        elif model == "Ridge":
            dpg.configure_item("manual_params_ridge", show=True)
        elif model == "Lasso":
            dpg.configure_item("manual_params_lasso", show=True)
        elif model == "ElasticNet":
            dpg.configure_item("manual_params_elasticnet", show=True)
        elif model == "RandomForest":
            dpg.configure_item("manual_params_rf", show=True)
        elif model == "LightGBM":
            dpg.configure_item("manual_params_lgbm", show=True)
        elif model == "XGBoost":
            dpg.configure_item("manual_params_xgb", show=True)
        elif model == "CatBoost":
            dpg.configure_item("manual_params_catboost", show=True)
        elif model in ["SVR", "SVM"]:
            dpg.configure_item("manual_params_svm", show=True)
        elif model == "NeuralBoosted":
            dpg.configure_item("manual_params_neuralboosted", show=True)

        # Update model list for classification task
        task_type = dpg.get_value("build_task_type").lower()
        if task_type == "classification":
            # Update dropdown items for classification
            dpg.configure_item("build_manual_model",
                items=["PLS-DA", "RandomForest", "LightGBM", "XGBoost", "CatBoost", "SVM", "NeuralBoosted"])
        else:
            dpg.configure_item("build_manual_model",
                items=["PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest",
                       "LightGBM", "XGBoost", "CatBoost", "SVR", "NeuralBoosted"])

    def _update_tier_info(self):
        """Update the tier info text based on current selection."""
        from ..core.model_config import MODEL_TIERS, CLASSIFICATION_TIERS, HYPERPARAMETER_GRIDS

        tier = dpg.get_value("build_tier").lower()
        task_type = dpg.get_value("build_task_type").lower()

        if tier == "custom":
            # Custom tier - user selects models via checkboxes
            desc = "Custom model selection"
            models = self._get_custom_selected_models(task_type)
            recommended = "When you want specific models only"

            info_lines = [
                f"{desc}",
                f"Selected: {', '.join(models) if models else 'None'}",
                f"Recommended for: {recommended}",
                "(Select models using checkboxes above)"
            ]
        else:
            if task_type == "classification":
                tier_info = CLASSIFICATION_TIERS.get(tier, {})
            else:
                tier_info = MODEL_TIERS.get(tier, {})

            desc = tier_info.get('description', '')
            models = tier_info.get('models', [])
            recommended = tier_info.get('recommended_for', '')

            info_lines = [
                f"{desc}",
                f"Models: {', '.join(models)}",
                f"Recommended for: {recommended}",
                "(Tier only affects which models are tested)"
            ]

        dpg.set_value("build_tier_info", '\n'.join(info_lines))

        # Estimate configurations based on current UI selections
        self._update_config_estimate()

    def _get_custom_selected_models(self, task_type: str) -> list:
        """Get list of models selected in custom tier."""
        models = []

        if task_type == "classification":
            # Classification models
            if dpg.does_item_exist("custom_model_plsda") and dpg.get_value("custom_model_plsda"):
                models.append("PLS-DA")
            if dpg.does_item_exist("custom_model_rf_cls") and dpg.get_value("custom_model_rf_cls"):
                models.append("RandomForest")
            if dpg.does_item_exist("custom_model_lgbm_cls") and dpg.get_value("custom_model_lgbm_cls"):
                models.append("LightGBM")
            if dpg.does_item_exist("custom_model_xgb_cls") and dpg.get_value("custom_model_xgb_cls"):
                models.append("XGBoost")
            if dpg.does_item_exist("custom_model_catboost_cls") and dpg.get_value("custom_model_catboost_cls"):
                models.append("CatBoost")
            if dpg.does_item_exist("custom_model_svm") and dpg.get_value("custom_model_svm"):
                models.append("SVM")
            if dpg.does_item_exist("custom_model_neuralboosted_cls") and dpg.get_value("custom_model_neuralboosted_cls"):
                models.append("NeuralBoosted")
        else:
            # Regression models
            if dpg.does_item_exist("custom_model_pls") and dpg.get_value("custom_model_pls"):
                models.append("PLS")
            if dpg.does_item_exist("custom_model_ridge") and dpg.get_value("custom_model_ridge"):
                models.append("Ridge")
            if dpg.does_item_exist("custom_model_lasso") and dpg.get_value("custom_model_lasso"):
                models.append("Lasso")
            if dpg.does_item_exist("custom_model_elasticnet") and dpg.get_value("custom_model_elasticnet"):
                models.append("ElasticNet")
            if dpg.does_item_exist("custom_model_rf") and dpg.get_value("custom_model_rf"):
                models.append("RandomForest")
            if dpg.does_item_exist("custom_model_lgbm") and dpg.get_value("custom_model_lgbm"):
                models.append("LightGBM")
            if dpg.does_item_exist("custom_model_xgb") and dpg.get_value("custom_model_xgb"):
                models.append("XGBoost")
            if dpg.does_item_exist("custom_model_catboost") and dpg.get_value("custom_model_catboost"):
                models.append("CatBoost")
            if dpg.does_item_exist("custom_model_svr") and dpg.get_value("custom_model_svr"):
                models.append("SVR")
            if dpg.does_item_exist("custom_model_neuralboosted") and dpg.get_value("custom_model_neuralboosted"):
                models.append("NeuralBoosted")

        return models

    def _update_config_estimate(self):
        """Update the estimated configuration count based on current selections."""
        try:
            tier = dpg.get_value("build_tier").lower()
            task_type = dpg.get_value("build_task_type").lower()

            from ..core.model_config import MODEL_TIERS, CLASSIFICATION_TIERS, get_hyperparameter_grid

            # Get models for tier or custom selection
            if tier == "custom":
                models = self._get_custom_selected_models(task_type)
            elif task_type == "classification":
                models = CLASSIFICATION_TIERS.get(tier, {}).get('models', [])
            else:
                models = MODEL_TIERS.get(tier, {}).get('models', [])

            # Count selected preprocessing methods
            n_preproc = 0
            if dpg.get_value("build_preproc_raw"):
                n_preproc += 1
            if dpg.get_value("build_preproc_snv"):
                n_preproc += 1

            # Count window sizes from new input fields
            window_count = 0
            primary_window = dpg.get_value("build_window_primary")
            if primary_window >= 5:
                window_count += 1
            secondary_window = dpg.get_value("build_window_secondary")
            if secondary_window >= 5:
                window_count += 1
            window_count = max(1, window_count)

            if dpg.get_value("build_preproc_deriv1"):
                n_preproc += window_count
            if dpg.get_value("build_preproc_deriv2"):
                n_preproc += window_count
            if dpg.get_value("build_preproc_snv_deriv1"):
                n_preproc += window_count
            if dpg.get_value("build_preproc_snv_deriv2"):
                n_preproc += window_count
            if dpg.get_value("build_preproc_deriv1_snv"):
                n_preproc += window_count
            if dpg.get_value("build_preproc_deriv2_snv"):
                n_preproc += window_count

            n_preproc = max(1, n_preproc)

            # Count hyperparameter configs per model
            total_model_configs = 0
            for model in models:
                grid = get_hyperparameter_grid(model)
                total_model_configs += len(grid)

            # Base configs (preprocessing × model hyperparams)
            base_configs = n_preproc * total_model_configs

            # Variable selection multiplier
            var_counts_str = dpg.get_value("build_var_counts")
            try:
                var_counts = [int(x.strip()) for x in var_counts_str.split(",") if x.strip()]
            except:
                var_counts = [20, 50, 100, 250]

            varsel_methods = sum([
                dpg.get_value("build_varsel_importance"),
                dpg.get_value("build_varsel_vip"),
                dpg.get_value("build_varsel_spa"),
                dpg.get_value("build_varsel_uve"),
                dpg.get_value("build_varsel_uve_spa"),
                dpg.get_value("build_varsel_ga_pls"),
                # Note: iPLS is now handled as Phase 4, not counted here
            ])

            if varsel_methods > 0:
                # Variable selection adds: preproc × models × var_counts
                varsel_configs = n_preproc * len(models) * len(var_counts)
            else:
                varsel_configs = 0

            total = base_configs + varsel_configs

            dpg.set_value("build_config_estimate", f"Estimated configurations: ~{total:,}")
        except:
            dpg.set_value("build_config_estimate", "")

    def _on_target_variable_change(self, sender=None, app_data=None):
        """Handle target variable selection change."""
        if not self._current_dataset:
            return

        selected_target = dpg.get_value("build_target_variable")
        if not selected_target:
            return

        # If selecting the original target, use existing y values
        if selected_target == self._current_dataset.target_name:
            # Already set, just update info
            self._update_target_info()
            return

        # Switch to a different metadata column as target
        if selected_target in self._current_dataset.metadata_columns:
            import numpy as np

            # Save current y as metadata if it has a name and isn't already saved
            if self._current_dataset.target_name and self._current_dataset.y is not None:
                old_target_name = self._current_dataset.target_name
                if old_target_name not in self._current_dataset.metadata_columns:
                    self._current_dataset.metadata_columns[old_target_name] = list(self._current_dataset.y)

            # Get new target values
            new_y_values = self._current_dataset.metadata_columns[selected_target]

            # Convert to numpy array, handling numeric/categorical
            try:
                # Try numeric first
                new_y = np.array(new_y_values, dtype=float)
                # Use pd.isna for compatibility with all types
                mask = ~pd.isna(new_y)
                n_unique = len(np.unique(new_y[mask]))
                if n_unique <= 10:
                    target_type = "classification"
                else:
                    target_type = "regression"
            except (ValueError, TypeError):
                # Categorical - encode as integers
                unique_vals = list(set(v for v in new_y_values if v is not None and str(v).strip()))
                val_to_int = {v: i for i, v in enumerate(sorted(unique_vals))}
                new_y = np.array([val_to_int.get(v, -1) for v in new_y_values], dtype=float)
                new_y[new_y < 0] = np.nan
                target_type = "classification"

            # Update dataset
            self._current_dataset.y = new_y
            self._current_dataset.target_name = selected_target
            self._current_dataset.metadata['target_type'] = target_type

            # Remove new target from metadata_columns to prevent duplication in display
            if selected_target in self._current_dataset.metadata_columns:
                del self._current_dataset.metadata_columns[selected_target]

            # Update task type combo to match
            dpg.set_value("build_task_type", target_type.capitalize())
            self._on_tier_change()  # Update model lists

            # Update info text
            self._update_target_info()

            # Update data grid if visible
            if self._data_grid:
                self._data_grid.set_data(self._current_dataset)

    def _update_target_info(self):
        """Update the target variable info text."""
        if not self._current_dataset or not self._current_dataset.has_target:
            dpg.set_value("build_target_info", "")
            return

        import numpy as np
        import pandas as pd
        y = self._current_dataset.y
        # Use pd.isna for compatibility with string/object types
        valid_y = y[~pd.isna(y)] if y is not None else np.array([])

        if len(valid_y) == 0:
            dpg.set_value("build_target_info", "No valid values")
            return

        n_unique = len(np.unique(valid_y))
        target_type = self._current_dataset.metadata.get('target_type', 'unknown')

        if target_type == "classification":
            info = f"{n_unique} classes, {len(valid_y)} samples"
        else:
            info = f"Range: {valid_y.min():.2f} - {valid_y.max():.2f}"

        dpg.set_value("build_target_info", info)

    def _populate_target_variable_combo(self):
        """Populate the target variable combo box with available columns."""
        if not self._current_dataset:
            dpg.configure_item("build_target_variable", items=[], default_value="")
            dpg.set_value("build_target_info", "")
            return

        # Build list of available targets
        available_targets = []

        # Current target first (if exists)
        if self._current_dataset.target_name:
            available_targets.append(self._current_dataset.target_name)

        # Add metadata columns that could be targets
        if self._current_dataset.metadata_columns:
            for col_name in sorted(self._current_dataset.metadata_columns.keys()):
                if col_name not in available_targets:
                    available_targets.append(col_name)

        # Update combo
        current_target = self._current_dataset.target_name or ""
        dpg.configure_item("build_target_variable", items=available_targets)
        dpg.set_value("build_target_variable", current_target)

        # Update info
        self._update_target_info()

    def _on_tier_change(self, sender=None, app_data=None):
        """Handle tier or task type selection change."""
        tier = dpg.get_value("build_tier").lower()
        task_type = dpg.get_value("build_task_type").lower()

        # Show/hide custom model selection based on tier
        is_custom = (tier == "custom")
        if dpg.does_item_exist("custom_model_selection"):
            dpg.configure_item("custom_model_selection", show=is_custom)

        # Show appropriate custom model list (regression vs classification)
        if is_custom:
            if dpg.does_item_exist("custom_regression_models"):
                dpg.configure_item("custom_regression_models", show=(task_type != "classification"))
            if dpg.does_item_exist("custom_classification_models"):
                dpg.configure_item("custom_classification_models", show=(task_type == "classification"))

        # Update validation method options based on task type
        if dpg.does_item_exist("build_validation_method"):
            if task_type == "classification":
                dpg.configure_item(
                    "build_validation_method",
                    items=["Stratified (recommended)", "Kennard-Stone", "Random"],
                    default_value="Stratified (recommended)"
                )
                dpg.set_value("build_validation_method", "Stratified (recommended)")
            else:
                dpg.configure_item(
                    "build_validation_method",
                    items=["SPXY (recommended)", "DUPLEX", "Kennard-Stone", "Random"],
                    default_value="SPXY (recommended)"
                )
                dpg.set_value("build_validation_method", "SPXY (recommended)")

        self._update_tier_info()

        # Update manual model list based on task type
        if task_type == "classification":
            dpg.configure_item("build_manual_model",
                items=["PLS-DA", "RandomForest", "LightGBM", "XGBoost", "CatBoost", "SVM", "NeuralBoosted"])
            dpg.set_value("build_manual_model", "PLS-DA")
        else:
            dpg.configure_item("build_manual_model",
                items=["PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest",
                       "LightGBM", "XGBoost", "CatBoost", "SVR", "NeuralBoosted"])
            dpg.set_value("build_manual_model", "PLS")

    def _on_search_mode_change(self, sender=None, app_data=None):
        """Handle search mode selection change (Grid vs Bayesian vs NSGA-II).

        Bayesian mode is TRUE AUTOML:
        - User controls: which models to consider (with smart defaults)
        - Optuna discovers: preprocessing, window sizes, variable selection, hyperparameters

        NSGA-II mode is MULTI-OBJECTIVE:
        - Optimizes trade-offs between error, wavelength count, and complexity
        - Returns Pareto front of non-dominated solutions
        """
        search_mode = dpg.get_value("build_search_mode")
        is_bayesian = ("Bayesian" in search_mode)
        is_nsga2 = ("NSGA-II" in search_mode)

        # Show/hide Bayesian options (n_trials slider)
        if dpg.does_item_exist("bayesian_options"):
            dpg.configure_item("bayesian_options", show=is_bayesian)

        # Show/hide NSGA-II options
        if dpg.does_item_exist("nsga2_options"):
            dpg.configure_item("nsga2_options", show=is_nsga2)

        # List of preprocessing checkboxes to enable/disable
        preproc_checkboxes = [
            "build_preproc_raw", "build_preproc_snv",
            "build_preproc_deriv1", "build_preproc_deriv2",
            "build_preproc_deriv3", "build_preproc_deriv4",
            "build_preproc_snv_deriv1", "build_preproc_snv_deriv2",
            "build_preproc_snv_deriv3", "build_preproc_snv_deriv4",
            "build_preproc_deriv1_snv", "build_preproc_deriv2_snv",
            "build_preproc_deriv3_snv", "build_preproc_deriv4_snv",
        ]

        # Window size checkboxes
        window_checkboxes = [
            "build_window_7", "build_window_11", "build_window_17",
            "build_window_19", "build_window_21", "build_window_25",
        ]

        # Variable selection checkboxes
        varsel_checkboxes = [
            "build_varsel_importance", "build_varsel_vip", "build_varsel_spa",
            "build_varsel_uve", "build_varsel_uve_spa", "build_varsel_ga_pls",
        ]

        if is_bayesian or is_nsga2:
            # === BAYESIAN AUTOML or NSGA-II MODE ===
            # Uncheck AND disable preprocessing checkboxes (will explore all)
            for tag in preproc_checkboxes:
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, False)
                    dpg.configure_item(tag, enabled=False)

            # Disable window size inputs
            for tag in ["build_window_primary", "build_window_secondary"]:
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, enabled=False)

            # Uncheck AND disable variable selection checkboxes (handled internally)
            for tag in varsel_checkboxes:
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, False)
                    dpg.configure_item(tag, enabled=False)

            # Disable var counts input
            if dpg.does_item_exist("build_var_counts"):
                dpg.configure_item("build_var_counts", enabled=False)

            if is_bayesian:
                # Show custom model selection so user can choose which models
                if dpg.does_item_exist("custom_model_selection"):
                    dpg.configure_item("custom_model_selection", show=True)

                # Auto-select all models for current task type
                task_type = dpg.get_value("build_task_type").lower()
                if task_type == "regression":
                    # Select all regression models by default
                    for tag in ["custom_model_pls", "custom_model_ridge", "custom_model_lasso",
                               "custom_model_elasticnet", "custom_model_rf", "custom_model_lgbm",
                               "custom_model_xgb", "custom_model_catboost", "custom_model_svr",
                               "custom_model_neuralboosted"]:
                        if dpg.does_item_exist(tag):
                            dpg.set_value(tag, True)
                else:
                    # Select all classification models by default
                    for tag in ["custom_model_plsda", "custom_model_rf_cls", "custom_model_lgbm_cls",
                               "custom_model_xgb_cls", "custom_model_catboost_cls", "custom_model_svm",
                               "custom_model_neuralboosted_cls"]:
                        if dpg.does_item_exist(tag):
                            dpg.set_value(tag, True)

                info = "AutoML: Optuna explores ALL preprocessing & variable selection. You control which models."
            elif is_nsga2:
                # NSGA-II mode - show custom model selection so user can choose which models
                if dpg.does_item_exist("custom_model_selection"):
                    dpg.configure_item("custom_model_selection", show=True)

                # Auto-select all models for current task type
                task_type = dpg.get_value("build_task_type").lower()
                if task_type == "regression":
                    # Select all regression models by default
                    for tag in ["custom_model_pls", "custom_model_ridge", "custom_model_lasso",
                               "custom_model_elasticnet", "custom_model_rf", "custom_model_lgbm",
                               "custom_model_xgb", "custom_model_catboost", "custom_model_svr",
                               "custom_model_neuralboosted"]:
                        if dpg.does_item_exist(tag):
                            dpg.set_value(tag, True)
                else:
                    # Select all classification models by default
                    for tag in ["custom_model_plsda", "custom_model_rf_cls", "custom_model_lgbm_cls",
                               "custom_model_xgb_cls", "custom_model_catboost_cls", "custom_model_svm",
                               "custom_model_neuralboosted_cls"]:
                        if dpg.does_item_exist(tag):
                            dpg.set_value(tag, True)

                info = "NSGA-II: Pareto optimization of error vs wavelengths vs complexity. You control which models."
        else:
            # === GRID SEARCH MODE ===
            # Re-enable preprocessing checkboxes and restore defaults
            preproc_defaults = {
                "build_preproc_raw": True,
                "build_preproc_snv": True,
                "build_preproc_deriv1": True,
                "build_preproc_deriv2": True,
                "build_preproc_deriv3": False,
                "build_preproc_deriv4": False,
                "build_preproc_snv_deriv1": True,
                "build_preproc_snv_deriv2": True,
                "build_preproc_snv_deriv3": False,
                "build_preproc_snv_deriv4": False,
                "build_preproc_deriv1_snv": False,
                "build_preproc_deriv2_snv": False,
                "build_preproc_deriv3_snv": False,
                "build_preproc_deriv4_snv": False,
            }
            for tag in preproc_checkboxes:
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, preproc_defaults.get(tag, False))
                    dpg.configure_item(tag, enabled=True)

            # Re-enable window size inputs
            for tag in ["build_window_primary", "build_window_secondary"]:
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, enabled=True)

            # Re-enable variable selection checkboxes and restore defaults
            varsel_defaults = {
                "build_varsel_importance": True,
                "build_varsel_vip": False,
                "build_varsel_spa": False,
                "build_varsel_uve": False,
                "build_varsel_uve_spa": False,
                "build_varsel_ga_pls": False,
            }
            for tag in varsel_checkboxes:
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, varsel_defaults.get(tag, False))
                    dpg.configure_item(tag, enabled=True)

            # Re-enable var counts input
            if dpg.does_item_exist("build_var_counts"):
                dpg.configure_item("build_var_counts", enabled=True)

            # Restore tier-based model selection (hide custom if not Custom tier)
            tier = dpg.get_value("build_tier")
            if tier != "Custom" and dpg.does_item_exist("custom_model_selection"):
                dpg.configure_item("custom_model_selection", show=False)

            info = "Grid search tests your selected preprocessing combinations exhaustively"

        if dpg.does_item_exist("build_search_mode_info"):
            dpg.set_value("build_search_mode_info", info)

    def _on_baseline_method_change(self, sender=None, app_data=None):
        """Handle baseline method selection change."""
        method = dpg.get_value("build_baseline_method")

        # Show/hide parameter groups
        show_poly = (method == "Polynomial")
        show_als = (method in ["AsLS", "airPLS"])

        if dpg.does_item_exist("baseline_poly_params"):
            dpg.configure_item("baseline_poly_params", show=show_poly)
        if dpg.does_item_exist("baseline_als_params"):
            dpg.configure_item("baseline_als_params", show=show_als)

    def _on_smoothing_toggle(self, sender=None, app_data=None):
        """Handle smoothing checkbox toggle."""
        enabled = dpg.get_value("build_enable_smoothing")

        if dpg.does_item_exist("smoothing_params"):
            dpg.configure_item("smoothing_params", show=enabled)

    def _on_run_search(self, sender=None, app_data=None):
        """Run model search in background thread."""
        if self._search_running or not self._current_dataset:
            return

        if not self._current_dataset.has_target:
            dpg.set_value("build_progress_text", "Error: No target variable loaded")
            return

        # Get common settings from UI
        task_type = dpg.get_value("build_task_type").lower()
        cv_folds = dpg.get_value("build_cv_folds")
        self._cv_folds = cv_folds  # Store for diagnostics

        # Store CV object with fixed random_state for reproducible R² in diagnostics
        from sklearn.model_selection import KFold, StratifiedKFold
        if task_type == 'classification':
            self._cv_object = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            self._cv_object = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Get X and y as numpy arrays
        X_full = self._current_dataset.X.copy()
        y_full = self._current_dataset.y.copy()
        wavelengths = self._current_dataset.wavelengths

        # Filter out samples with NaN/missing target values
        import numpy as np
        if np.issubdtype(y_full.dtype, np.floating):
            valid_mask = ~np.isnan(y_full)
        else:
            valid_mask = np.ones(len(y_full), dtype=bool)

        if not np.all(valid_mask):
            n_excluded = np.sum(~valid_mask)
            n_valid = np.sum(valid_mask)
            print(f"DEBUG: Excluding {n_excluded} samples with missing target values, {n_valid} remaining")
            dpg.set_value("build_progress_text", f"Excluding {n_excluded} samples with missing target values...")
            X_full = X_full[valid_mask]
            y_full = y_full[valid_mask]
            # Store mapping for later reference
            self._valid_sample_mask = valid_mask
        else:
            self._valid_sample_mask = None

        # Import v3 standalone modules
        from ..core.search import run_search
        from ..core.sample_selection import split_validation

        # Handle validation holdout
        validation_enabled = dpg.get_value("build_validation_enable")
        if validation_enabled:
            # Parse percentage
            percent_str = dpg.get_value("build_validation_percent")
            val_ratio = int(percent_str.replace("%", "")) / 100.0

            # Parse method
            method_str = dpg.get_value("build_validation_method")
            method_map = {
                "SPXY": "spxy",
                "DUPLEX": "duplex",
                "Kennard-Stone": "kennard-stone",
                "Random": "random",
                "Stratified": "stratified"
            }
            method = method_map.get(method_str.split(" ")[0], "spxy")

            # Split data
            try:
                print(f"DEBUG: Splitting data with method={method}, val_ratio={val_ratio}, task_type={task_type}")
                print(f"DEBUG: X_full shape={X_full.shape}, y_full shape={y_full.shape}, y dtype={y_full.dtype}")
                self._calibration_indices, self._validation_indices = split_validation(
                    X_full, y_full, val_ratio=val_ratio, method=method, random_state=42,
                    task_type=task_type
                )
                print(f"DEBUG: Split complete - cal={len(self._calibration_indices)}, val={len(self._validation_indices)}")

                # Extract calibration and validation sets
                X = X_full[self._calibration_indices]
                y = y_full[self._calibration_indices]

                # Store validation set
                self._validation_set = {
                    'X': X_full[self._validation_indices],
                    'y': y_full[self._validation_indices],
                    'indices': self._validation_indices,
                    'sample_ids': [self._current_dataset.sample_ids[i] for i in self._validation_indices]
                        if hasattr(self._current_dataset, 'sample_ids') and self._current_dataset.sample_ids is not None
                        else None
                }
                print(f"DEBUG: Validation set stored: {self._validation_set is not None}")

                n_val = len(self._validation_indices)
                n_cal = len(self._calibration_indices)
                dpg.set_value("build_progress_text", f"Holdout: {n_val} validation samples | Training on {n_cal} samples...")
            except Exception as e:
                print(f"DEBUG: Error splitting data: {e}")
                import traceback
                traceback.print_exc()
                dpg.set_value("build_progress_text", f"Error splitting data: {str(e)}")
                return
        else:
            # No holdout - use full dataset
            X = X_full
            y = y_full
            self._validation_set = None
            self._calibration_indices = None
            self._validation_indices = None

        # Determine mode and build search parameters
        mode = self._build_mode  # 'manual' or 'auto'

        if mode == 'manual':
            # Manual mode: single model with user-specified settings
            model_name = dpg.get_value("build_manual_model")
            preprocessing = dpg.get_value("build_manual_preproc")
            window_size = dpg.get_value("build_manual_window")

            # Get model-specific hyperparameters
            model_params = self._get_manual_model_params(model_name)

            # Wavelength range selection for manual mode
            wl_range_min = None
            wl_range_max = None
            if dpg.get_value("build_enable_wl_range"):
                wl_range_min = dpg.get_value("build_wl_range_min")
                wl_range_max = dpg.get_value("build_wl_range_max")
                if wl_range_min >= wl_range_max:
                    dpg.set_value("build_progress_text", "Error: Invalid wavelength range (min >= max)")
                    return

            # Check for absorbance conversion (shared with auto mode)
            convert_absorbance = dpg.get_value("build_convert_absorbance")

            search_kwargs = {
                'mode': 'manual',
                'model_name': model_name,
                'preprocessing': preprocessing,
                'window_size': window_size,
                'model_params': model_params,
                'wavelengths': wavelengths,
                'wl_range_min': wl_range_min,
                'wl_range_max': wl_range_max,
                'convert_absorbance': convert_absorbance,
            }

            # Add baseline correction parameters (manual mode)
            baseline_method = dpg.get_value("build_baseline_method")
            if baseline_method and baseline_method != "None":
                search_kwargs['baseline_method'] = baseline_method.lower()
                if baseline_method == "Polynomial":
                    search_kwargs['baseline_params'] = {
                        'degree': dpg.get_value("build_baseline_poly_degree"),
                        'n_segments': dpg.get_value("build_baseline_poly_segments")
                    }
                elif baseline_method in ["AsLS", "airPLS"]:
                    lam_str = dpg.get_value("build_baseline_lambda").split()[0]
                    lam = float(lam_str)
                    search_kwargs['baseline_params'] = {
                        'lam': lam,
                        'p': dpg.get_value("build_baseline_p")
                    }
            else:
                search_kwargs['baseline_method'] = None
                search_kwargs['baseline_params'] = None

            # Update UI
            dpg.configure_item("build_run_btn", enabled=False, label="Training...")
            start_message = f"Training {model_name} with {preprocessing}..."
            if convert_absorbance:
                start_message += " (absorbance)"
            self._search_preprocess = preprocessing

        else:
            # Auto mode: full automated search with user-selected options
            tier = dpg.get_value("build_tier").lower()

            # Check for auto modes that have their own preprocessing exploration
            search_mode_check = dpg.get_value("build_search_mode") if dpg.does_item_exist("build_search_mode") else "Grid Search"
            is_bayesian = "Bayesian" in search_mode_check
            is_nsga2 = "NSGA-II" in search_mode_check
            is_ga_preproc = dpg.get_value("build_ga_preprocess") if dpg.does_item_exist("build_ga_preprocess") else False

            if is_bayesian or is_nsga2 or is_ga_preproc:
                # Auto modes explore all preprocessing internally - pass None to use defaults
                # Bayesian: search.py lines 377-389 creates comprehensive preproc configs
                # NSGA-II: nsga2_search.py lines 56-69 has internal PREPROC_TYPES
                # GA preprocessing: ga_preprocessing.py lines 37-69 has internal PREPROC_TYPES
                preproc_methods = None
            else:
                # Grid search mode: gather user-selected preprocessing methods from checkboxes
                preproc_methods = []
                if dpg.get_value("build_preproc_raw"):
                    preproc_methods.append('raw')
                if dpg.get_value("build_preproc_snv"):
                    preproc_methods.append('snv')
                if dpg.get_value("build_preproc_deriv1"):
                    preproc_methods.append('deriv1')
                if dpg.get_value("build_preproc_deriv2"):
                    preproc_methods.append('deriv2')
                if dpg.get_value("build_preproc_deriv3"):
                    preproc_methods.append('deriv3')
                if dpg.get_value("build_preproc_deriv4"):
                    preproc_methods.append('deriv4')
                if dpg.get_value("build_preproc_snv_deriv1"):
                    preproc_methods.append('snv_deriv1')
                if dpg.get_value("build_preproc_snv_deriv2"):
                    preproc_methods.append('snv_deriv2')
                if dpg.get_value("build_preproc_snv_deriv3"):
                    preproc_methods.append('snv_deriv3')
                if dpg.get_value("build_preproc_snv_deriv4"):
                    preproc_methods.append('snv_deriv4')
                if dpg.get_value("build_preproc_deriv1_snv"):
                    preproc_methods.append('deriv1_snv')
                if dpg.get_value("build_preproc_deriv2_snv"):
                    preproc_methods.append('deriv2_snv')
                if dpg.get_value("build_preproc_deriv3_snv"):
                    preproc_methods.append('deriv3_snv')
                if dpg.get_value("build_preproc_deriv4_snv"):
                    preproc_methods.append('deriv4_snv')

                if not preproc_methods:
                    preproc_methods = ['snv']  # Fallback default for grid search

            # Gather user-specified window sizes from input fields
            if is_bayesian or is_nsga2 or is_ga_preproc:
                # Auto modes use their own window size exploration
                window_sizes = None
            else:
                window_sizes = []
                primary_window = dpg.get_value("build_window_primary")
                if primary_window >= 5:
                    window_sizes.append(primary_window)
                secondary_window = dpg.get_value("build_window_secondary")
                if secondary_window >= 5:
                    window_sizes.append(secondary_window)

                if not window_sizes:
                    window_sizes = [17]  # Fallback default for grid search

            # Gather variable selection methods
            if is_bayesian or is_nsga2:
                # Bayesian and NSGA-II handle variable selection internally
                varsel_methods = None
            else:
                varsel_methods = []
                if dpg.get_value("build_varsel_importance"):
                    varsel_methods.append('importance')
                if dpg.get_value("build_varsel_vip"):
                    varsel_methods.append('vip')
                if dpg.get_value("build_varsel_spa"):
                    varsel_methods.append('spa')
                if dpg.get_value("build_varsel_uve"):
                    varsel_methods.append('uve')
                if dpg.get_value("build_varsel_uve_spa"):
                    varsel_methods.append('uve_spa')
                if dpg.get_value("build_varsel_ga_pls"):
                    varsel_methods.append('ga_pls')
            # Note: iPLS is now handled separately via enable_ipls (Phase 4)

            # Gather variable counts
            var_counts_str = dpg.get_value("build_var_counts")
            try:
                var_counts = [int(x.strip()) for x in var_counts_str.split(",") if x.strip()]
            except:
                var_counts = [20, 50, 100, 250]

            # Region analysis
            enable_regions = dpg.get_value("build_enable_regions")
            n_regions = dpg.get_value("build_n_regions")

            # iPLS interval analysis
            enable_ipls = dpg.get_value("build_enable_ipls")
            ipls_n_intervals = dpg.get_value("build_ipls_n_intervals")
            ipls_forward = dpg.get_value("build_ipls_forward")
            ipls_backward = dpg.get_value("build_ipls_backward")

            # Gather variable selection parameters
            varsel_params = {}
            if dpg.get_value("build_varsel_spa"):
                varsel_params['spa'] = {
                    'n_features': dpg.get_value("build_spa_n_features"),
                    'n_random_starts': dpg.get_value("build_spa_n_random_starts"),
                }
            if dpg.get_value("build_varsel_uve"):
                varsel_params['uve'] = {
                    'cutoff_multiplier': dpg.get_value("build_uve_cutoff"),
                }
            if dpg.get_value("build_varsel_uve_spa"):
                varsel_params['uve_spa'] = {
                    'cutoff_multiplier': dpg.get_value("build_uve_spa_cutoff"),
                    'n_features': dpg.get_value("build_uve_spa_n_features"),
                }
            if dpg.get_value("build_varsel_ga_pls"):
                varsel_params['ga_pls'] = {
                    'population_size': dpg.get_value("build_ga_pls_population"),
                    'n_generations': dpg.get_value("build_ga_pls_generations"),
                    'n_runs': dpg.get_value("build_ga_pls_n_runs"),
                    'selection_threshold': dpg.get_value("build_ga_pls_threshold"),
                }
            # Note: iPLS params are now read separately via enable_ipls controls

            # PLS max latent variables
            pls_max_lv = dpg.get_value("hyperparam_pls_max_lv")

            # Wavelength range selection
            wl_range_min = None
            wl_range_max = None
            if dpg.get_value("build_enable_wl_range"):
                wl_range_min = dpg.get_value("build_wl_range_min")
                wl_range_max = dpg.get_value("build_wl_range_max")
                if wl_range_min >= wl_range_max:
                    dpg.set_value("build_progress_text", "Error: Invalid wavelength range (min >= max)")
                    return

            # Check for absorbance conversion
            convert_absorbance = dpg.get_value("build_convert_absorbance")

            # Parse custom hyperparameter grids from UI
            custom_hyperparam_grids = self._get_auto_model_hyperparam_grids(task_type)

            # Get search mode (Grid Search vs Bayesian vs NSGA-II)
            search_mode_ui = dpg.get_value("build_search_mode")
            if "NSGA-II" in search_mode_ui:
                search_mode = 'nsga2'
            elif "Bayesian" in search_mode_ui:
                search_mode = 'bayesian'
            else:
                search_mode = 'grid'
            bayesian_n_trials = dpg.get_value("build_bayesian_trials") if search_mode == 'bayesian' else 50

            # NSGA-II parameters
            nsga2_population = dpg.get_value("build_nsga2_population") if search_mode == 'nsga2' else 50
            nsga2_generations = dpg.get_value("build_nsga2_generations") if search_mode == 'nsga2' else 100
            nsga2_min_wavelengths = dpg.get_value("build_nsga2_min_wavelengths") if search_mode == 'nsga2' else 10

            search_kwargs = {
                'mode': 'auto',
                'tier': tier,
                'wavelengths': wavelengths,
                'preproc_methods': preproc_methods,
                'window_sizes': window_sizes,
                'varsel_methods': varsel_methods,
                'varsel_params': varsel_params,
                'var_counts': var_counts,
                'enable_regions': enable_regions,
                'n_regions': n_regions,
                'enable_ipls': enable_ipls,
                'ipls_n_intervals': ipls_n_intervals,
                'ipls_forward': ipls_forward,
                'ipls_backward': ipls_backward,
                'pls_max_lv': pls_max_lv,
                # Wavelength range selection
                'wl_range_min': wl_range_min,
                'wl_range_max': wl_range_max,
                # Data conversion
                'convert_absorbance': convert_absorbance,
                # Custom hyperparameter grids from UI
                'custom_hyperparam_grids': custom_hyperparam_grids if custom_hyperparam_grids else None,
                # Search mode (Grid Search vs Bayesian vs NSGA-II)
                'search_mode': search_mode,
                'bayesian_n_trials': bayesian_n_trials,
                # NSGA-II parameters
                'nsga2_population': nsga2_population,
                'nsga2_generations': nsga2_generations,
                'nsga2_min_wavelengths': nsga2_min_wavelengths,
            }

            # Add baseline correction parameters
            baseline_method = dpg.get_value("build_baseline_method")
            if baseline_method and baseline_method != "None":
                search_kwargs['baseline_method'] = baseline_method.lower()
                if baseline_method == "Polynomial":
                    search_kwargs['baseline_params'] = {
                        'degree': dpg.get_value("build_baseline_poly_degree"),
                        'n_segments': dpg.get_value("build_baseline_poly_segments")
                    }
                elif baseline_method in ["AsLS", "airPLS"]:
                    # Parse lambda from combo string (e.g., "1e5 (moderate)" -> 1e5)
                    lam_str = dpg.get_value("build_baseline_lambda").split()[0]
                    lam = float(lam_str)
                    search_kwargs['baseline_params'] = {
                        'lam': lam,
                        'p': dpg.get_value("build_baseline_p")
                    }
            else:
                search_kwargs['baseline_method'] = None
                search_kwargs['baseline_params'] = None

            # Add smoothing parameters
            if dpg.get_value("build_enable_smoothing"):
                search_kwargs['enable_smoothing'] = True
                if dpg.does_item_exist("build_smoothing_window"):
                    search_kwargs['smoothing_window'] = dpg.get_value("build_smoothing_window")
                if dpg.does_item_exist("build_smoothing_order"):
                    search_kwargs['smoothing_order'] = dpg.get_value("build_smoothing_order")

            # Add GA preprocessing optimization parameters
            if dpg.does_item_exist("build_ga_preprocess") and dpg.get_value("build_ga_preprocess"):
                search_kwargs['ga_preprocess'] = True
                search_kwargs['ga_preprocess_population'] = dpg.get_value("build_ga_preprocess_population")
                search_kwargs['ga_preprocess_generations'] = dpg.get_value("build_ga_preprocess_generations")
                search_kwargs['ga_preprocess_cv_folds'] = dpg.get_value("build_ga_preprocess_cv_folds")

            # Handle custom tier, Bayesian mode, or NSGA-II mode - all use explicit model selection
            # These modes are AutoML: user controls models, the search explores preprocessing
            if tier == "custom" or search_mode == 'bayesian' or search_mode == 'nsga2':
                custom_models = self._get_custom_selected_models(task_type)
                if not custom_models:
                    dpg.set_value("build_progress_text", "Error: No models selected")
                    return
                search_kwargs['custom_models'] = custom_models

            # Update UI
            dpg.configure_item("build_run_btn", enabled=False, label="Searching...")
            if search_mode == 'nsga2':
                start_message = f"Starting NSGA-II optimization (pop={nsga2_population}, gen={nsga2_generations})..."
            elif search_mode == 'bayesian':
                start_message = f"Starting Bayesian AutoML with {len(custom_models)} models, {bayesian_n_trials} trials..."
            elif tier == "custom":
                start_message = f"Starting custom search with {len(custom_models)} models..."
            else:
                start_message = f"Starting {tier} tier search ({len(preproc_methods)} preproc methods)..."
            self._search_preprocess = "auto"

        # Update UI state
        self._search_running = True
        dpg.configure_item("build_progress_bar", show=True)
        dpg.set_value("build_progress_bar", 0.0)
        dpg.set_value("build_progress_text", start_message)

        # Track time for estimates
        import time
        self._search_start_time = time.time()

        def format_time(seconds):
            """Format seconds as human-readable time."""
            if seconds < 60:
                return f"{int(seconds)}s"
            elif seconds < 3600:
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins}m {secs}s"
            else:
                hours = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                return f"{hours}h {mins}m"

        def progress_callback(info):
            """Update progress from background thread."""
            message = info.get('message', '')
            current = info.get('current', 0)
            total = info.get('total', 1)
            best_score = info.get('best_score')

            progress = current / max(total, 1)
            dpg.set_value("build_progress_bar", progress)

            # Calculate time estimates
            elapsed = time.time() - self._search_start_time
            time_str = ""
            if current > 0 and total > 0:
                rate = elapsed / current  # seconds per item
                remaining = rate * (total - current)
                time_str = f" | Elapsed: {format_time(elapsed)} | ETA: {format_time(remaining)}"

            if best_score is not None:
                dpg.set_value("build_progress_text", f"[{current}/{total}] {message} (best: {best_score:.4f}){time_str}")
            else:
                dpg.set_value("build_progress_text", f"[{current}/{total}] {message}{time_str}")

        def search_thread():
            """Run search in background."""
            try:
                # Run the search with v3's unified interface
                results = run_search(
                    X=X,
                    y=y,
                    task_type=task_type,
                    folds=cv_folds,
                    random_state=42,
                    progress_callback=progress_callback,
                    **search_kwargs
                )

                # Store results and signal completion
                self._search_results = results
                self._on_search_complete()

            except Exception as e:
                import traceback
                traceback.print_exc()
                dpg.set_value("build_progress_text", f"Error: {str(e)}")
                self._search_running = False
                btn_label = "Train Model" if mode == 'manual' else "Run Model Search"
                dpg.configure_item("build_run_btn", enabled=True, label=btn_label)

        # Start search in background thread
        self._search_thread = threading.Thread(target=search_thread, daemon=True)
        self._search_thread.start()

    def _get_auto_model_hyperparam_grids(self, task_type: str) -> dict:
        """
        Parse AUTO mode hyperparameter UI inputs into grid format.

        Returns dict keyed by model name with list of param dicts, e.g.:
        {'Ridge': [{'alpha': 0.001}, {'alpha': 0.01}, ...], ...}
        """
        from itertools import product

        def parse_values(text: str, dtype='float'):
            """Parse comma-separated values into list of typed values."""
            if not text or not text.strip():
                return []
            values = []
            for v in text.split(','):
                v = v.strip()
                if not v:
                    continue
                try:
                    if dtype == 'float':
                        values.append(float(v))
                    elif dtype == 'int':
                        values.append(int(v))
                    elif dtype == 'bool':
                        values.append(v.lower() in ('true', '1', 'yes'))
                    elif dtype == 'int_or_none':
                        val = int(v)
                        values.append(None if val == 0 else val)
                    elif dtype == 'int_or_neg':
                        values.append(int(v))
                    elif dtype == 'str':
                        values.append(v)
                    elif dtype == 'gamma':
                        # gamma can be 'scale', 'auto', or float
                        if v.lower() in ('scale', 'auto'):
                            values.append(v.lower())
                        else:
                            values.append(float(v))
                except:
                    pass
            return values

        grids = {}

        try:
            # Ridge
            alphas = parse_values(dpg.get_value("hyperparam_ridge_alpha"), 'float')
            tols = parse_values(dpg.get_value("hyperparam_ridge_tol"), 'float')
            if alphas:
                if tols and len(tols) > 1:
                    grids['Ridge'] = [{'alpha': a, 'tol': t} for a, t in product(alphas, tols)]
                else:
                    grids['Ridge'] = [{'alpha': a} for a in alphas]

            # Lasso
            alphas = parse_values(dpg.get_value("hyperparam_lasso_alpha"), 'float')
            selections = parse_values(dpg.get_value("hyperparam_lasso_selection"), 'str')
            if alphas:
                if selections and len(selections) > 1:
                    grids['Lasso'] = [{'alpha': a, 'selection': s} for a, s in product(alphas, selections)]
                else:
                    grids['Lasso'] = [{'alpha': a} for a in alphas]

            # ElasticNet
            alphas = parse_values(dpg.get_value("hyperparam_elasticnet_alpha"), 'float')
            l1_ratios = parse_values(dpg.get_value("hyperparam_elasticnet_l1"), 'float')
            selections = parse_values(dpg.get_value("hyperparam_elasticnet_selection"), 'str')
            if alphas and l1_ratios:
                if selections and len(selections) > 1:
                    grids['ElasticNet'] = [
                        {'alpha': a, 'l1_ratio': r, 'selection': s}
                        for a, r, s in product(alphas, l1_ratios, selections)
                    ]
                else:
                    grids['ElasticNet'] = [
                        {'alpha': a, 'l1_ratio': r}
                        for a, r in product(alphas, l1_ratios)
                    ]

            # RandomForest
            estimators = parse_values(dpg.get_value("hyperparam_rf_estimators"), 'int')
            depths = parse_values(dpg.get_value("hyperparam_rf_depth"), 'int_or_none')
            max_leaf = parse_values(dpg.get_value("hyperparam_rf_max_leaf"), 'int_or_none')
            min_impurity = parse_values(dpg.get_value("hyperparam_rf_min_impurity"), 'float')
            if estimators and depths:
                # Include max_leaf_nodes and min_impurity_decrease if multiple values provided
                if (max_leaf and len(max_leaf) > 1) or (min_impurity and len(min_impurity) > 1):
                    ml = max_leaf if max_leaf else [None]
                    mi = min_impurity if min_impurity else [0.0]
                    grids['RandomForest'] = [
                        {'n_estimators': n, 'max_depth': d, 'max_leaf_nodes': m, 'min_impurity_decrease': i}
                        for n, d, m, i in product(estimators, depths, ml, mi)
                    ]
                else:
                    grids['RandomForest'] = [
                        {'n_estimators': n, 'max_depth': d}
                        for n, d in product(estimators, depths)
                    ]

            # LightGBM
            estimators = parse_values(dpg.get_value("hyperparam_lgbm_estimators"), 'int')
            lrs = parse_values(dpg.get_value("hyperparam_lgbm_lr"), 'float')
            leaves = parse_values(dpg.get_value("hyperparam_lgbm_leaves"), 'int')
            if estimators and lrs and leaves:
                grids['LightGBM'] = [
                    {'n_estimators': n, 'learning_rate': lr, 'num_leaves': nl}
                    for n, lr, nl in product(estimators, lrs, leaves)
                ]

            # XGBoost
            estimators = parse_values(dpg.get_value("hyperparam_xgb_estimators"), 'int')
            lrs = parse_values(dpg.get_value("hyperparam_xgb_lr"), 'float')
            depths = parse_values(dpg.get_value("hyperparam_xgb_depth"), 'int')
            if estimators and lrs and depths:
                grids['XGBoost'] = [
                    {'n_estimators': n, 'learning_rate': lr, 'max_depth': d}
                    for n, lr, d in product(estimators, lrs, depths)
                ]

            # CatBoost
            iters = parse_values(dpg.get_value("hyperparam_catboost_iters"), 'int')
            lrs = parse_values(dpg.get_value("hyperparam_catboost_lr"), 'float')
            depths = parse_values(dpg.get_value("hyperparam_catboost_depth"), 'int')
            if iters and lrs and depths:
                grids['CatBoost'] = [
                    {'iterations': n, 'learning_rate': lr, 'depth': d}
                    for n, lr, d in product(iters, lrs, depths)
                ]

            # SVR / SVM
            kernels = parse_values(dpg.get_value("hyperparam_svm_kernel"), 'str')
            cs = parse_values(dpg.get_value("hyperparam_svm_c"), 'float')
            shrinking = parse_values(dpg.get_value("hyperparam_svm_shrinking"), 'bool')
            if kernels and cs:
                if shrinking and len(shrinking) > 1:
                    grids['SVR'] = [
                        {'kernel': k, 'C': c, 'shrinking': s}
                        for k, c, s in product(kernels, cs, shrinking)
                    ]
                    grids['SVM'] = [
                        {'kernel': k, 'C': c, 'shrinking': s}
                        for k, c, s in product(kernels, cs, shrinking)
                    ]
                else:
                    grids['SVR'] = [
                        {'kernel': k, 'C': c}
                        for k, c in product(kernels, cs)
                    ]
                    grids['SVM'] = [
                        {'kernel': k, 'C': c}
                        for k, c in product(kernels, cs)
                    ]

            # NeuralBoosted
            nb_estimators = parse_values(dpg.get_value("hyperparam_nb_estimators"), 'int')
            nb_lrs = parse_values(dpg.get_value("hyperparam_nb_lr"), 'float')
            nb_hidden = parse_values(dpg.get_value("hyperparam_nb_hidden"), 'int')
            nb_activation = parse_values(dpg.get_value("hyperparam_nb_activation"), 'str')
            if nb_estimators and nb_lrs and nb_hidden:
                if nb_activation and len(nb_activation) > 1:
                    grids['NeuralBoosted'] = [
                        {'n_estimators': n, 'learning_rate': lr, 'hidden_layer_size': h, 'activation': a}
                        for n, lr, h, a in product(nb_estimators, nb_lrs, nb_hidden, nb_activation)
                    ]
                else:
                    grids['NeuralBoosted'] = [
                        {'n_estimators': n, 'learning_rate': lr, 'hidden_layer_size': h}
                        for n, lr, h in product(nb_estimators, nb_lrs, nb_hidden)
                    ]

            # MLP (Multi-Layer Perceptron)
            def parse_hidden_layers(text: str):
                """Parse hidden_layer_sizes from text like '(50,), (100,), (50, 50)'"""
                if not text or not text.strip():
                    return []
                layers = []
                # Split by ), and process each tuple
                import re
                pattern = r'\([\d,\s]+\)'
                matches = re.findall(pattern, text)
                for m in matches:
                    try:
                        # Convert "(50,)" or "(50, 50)" to tuple
                        inner = m.strip('()').strip()
                        if inner:
                            vals = tuple(int(x.strip()) for x in inner.split(',') if x.strip())
                            if vals:
                                layers.append(vals)
                    except:
                        pass
                return layers

            hidden_layers = parse_hidden_layers(dpg.get_value("hyperparam_mlp_hidden"))
            activations = parse_values(dpg.get_value("hyperparam_mlp_activation"), 'str')
            alphas = parse_values(dpg.get_value("hyperparam_mlp_alpha"), 'float')
            if hidden_layers and activations and alphas:
                grids['MLP'] = [
                    {'hidden_layer_sizes': hls, 'activation': act, 'alpha': a}
                    for hls, act, a in product(hidden_layers, activations, alphas)
                ]
        except Exception as e:
            print(f"Error parsing hyperparameter grids: {e}")

        return grids

    def _get_manual_model_params(self, model_name: str) -> dict:
        """Get hyperparameters for manual mode based on model type."""
        params = {}

        if model_name in ["PLS", "PLS-DA"]:
            params['n_components'] = dpg.get_value("build_pls_components")
            params['scale'] = dpg.get_value("build_pls_scale")

        elif model_name == "Ridge":
            params['alpha'] = dpg.get_value("build_ridge_alpha")
            params['fit_intercept'] = dpg.get_value("build_ridge_intercept")
            params['solver'] = dpg.get_value("build_ridge_solver")

        elif model_name == "Lasso":
            params['alpha'] = dpg.get_value("build_lasso_alpha")
            params['max_iter'] = dpg.get_value("build_lasso_maxiter")
            params['tol'] = dpg.get_value("build_lasso_tol")
            params['fit_intercept'] = dpg.get_value("build_lasso_intercept")

        elif model_name == "ElasticNet":
            params['alpha'] = dpg.get_value("build_elasticnet_alpha")
            params['l1_ratio'] = dpg.get_value("build_elasticnet_l1")
            params['max_iter'] = dpg.get_value("build_elasticnet_maxiter")
            params['tol'] = dpg.get_value("build_elasticnet_tol")
            params['fit_intercept'] = dpg.get_value("build_elasticnet_intercept")

        elif model_name == "RandomForest":
            params['n_estimators'] = dpg.get_value("build_rf_estimators")
            max_depth = dpg.get_value("build_rf_depth")
            params['max_depth'] = None if max_depth == 0 else max_depth
            params['min_samples_split'] = dpg.get_value("build_rf_min_split")
            params['min_samples_leaf'] = dpg.get_value("build_rf_min_leaf")
            max_features = dpg.get_value("build_rf_max_features")
            if max_features == "None":
                params['max_features'] = None
            elif max_features in ["sqrt", "log2"]:
                params['max_features'] = max_features
            else:
                params['max_features'] = float(max_features)
            params['bootstrap'] = dpg.get_value("build_rf_bootstrap")

        elif model_name == "LightGBM":
            params['n_estimators'] = dpg.get_value("build_lgbm_estimators")
            params['learning_rate'] = dpg.get_value("build_lgbm_lr")
            params['num_leaves'] = dpg.get_value("build_lgbm_leaves")
            max_depth = dpg.get_value("build_lgbm_depth")
            params['max_depth'] = max_depth if max_depth != -1 else -1
            params['min_child_samples'] = dpg.get_value("build_lgbm_min_child")
            params['subsample'] = dpg.get_value("build_lgbm_subsample")
            params['colsample_bytree'] = dpg.get_value("build_lgbm_colsample")
            params['reg_alpha'] = dpg.get_value("build_lgbm_reg_alpha")
            params['reg_lambda'] = dpg.get_value("build_lgbm_reg_lambda")

        elif model_name == "XGBoost":
            params['n_estimators'] = dpg.get_value("build_xgb_estimators")
            params['learning_rate'] = dpg.get_value("build_xgb_lr")
            params['max_depth'] = dpg.get_value("build_xgb_depth")
            params['min_child_weight'] = dpg.get_value("build_xgb_min_child")
            params['subsample'] = dpg.get_value("build_xgb_subsample")
            params['colsample_bytree'] = dpg.get_value("build_xgb_colsample")
            params['gamma'] = dpg.get_value("build_xgb_gamma")
            params['reg_alpha'] = dpg.get_value("build_xgb_reg_alpha")
            params['reg_lambda'] = dpg.get_value("build_xgb_reg_lambda")

        elif model_name == "CatBoost":
            params['iterations'] = dpg.get_value("build_catboost_iters")
            params['learning_rate'] = dpg.get_value("build_catboost_lr")
            params['depth'] = dpg.get_value("build_catboost_depth")
            params['l2_leaf_reg'] = dpg.get_value("build_catboost_l2")
            params['border_count'] = dpg.get_value("build_catboost_border")
            params['bagging_temperature'] = dpg.get_value("build_catboost_bagging")
            params['random_strength'] = dpg.get_value("build_catboost_randstrength")

        elif model_name in ["SVR", "SVM"]:
            params['kernel'] = dpg.get_value("build_svm_kernel")
            params['C'] = dpg.get_value("build_svm_c")
            params['gamma'] = dpg.get_value("build_svm_gamma")
            if model_name == "SVR":
                params['epsilon'] = dpg.get_value("build_svm_epsilon")
            params['degree'] = dpg.get_value("build_svm_degree")
            params['coef0'] = dpg.get_value("build_svm_coef0")
            params['max_iter'] = dpg.get_value("build_svm_maxiter")

        elif model_name == "NeuralBoosted":
            params['n_estimators'] = dpg.get_value("build_nb_estimators")
            params['learning_rate'] = dpg.get_value("build_nb_lr")
            params['hidden_layer_size'] = dpg.get_value("build_nb_hidden")
            params['max_iter'] = dpg.get_value("build_nb_maxiter")
            params['early_stopping'] = dpg.get_value("build_nb_earlystop")
            params['tol'] = dpg.get_value("build_nb_tol")

        return params

    def _on_search_complete(self):
        """Handle search completion - populate results table."""
        self._search_running = False
        btn_label = "Train Model" if self._build_mode == 'manual' else "Run Model Search"
        dpg.configure_item("build_run_btn", enabled=True, label=btn_label)
        dpg.configure_item("build_progress_bar", show=False)

        if self._search_results is None or len(self._search_results) == 0:
            dpg.set_value("build_progress_text", "No results found")
            return

        results_df = self._search_results
        n_results = len(results_df)

        # Calculate total elapsed time
        import time
        total_time = ""
        if hasattr(self, '_search_start_time') and self._search_start_time:
            elapsed = time.time() - self._search_start_time
            if elapsed < 60:
                total_time = f" in {int(elapsed)}s"
            elif elapsed < 3600:
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                total_time = f" in {mins}m {secs}s"
            else:
                hours = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                total_time = f" in {hours}h {mins}m"

        dpg.set_value("build_progress_text", f"Complete! {n_results} model configurations tested{total_time} (click column headers to sort)")

        # Play completion sound
        self._play_completion_sound()

        # Hide placeholder, show table
        dpg.configure_item("build_results_placeholder", show=False)
        dpg.configure_item("build_results_table", show=True)

        # Enable export menu items
        dpg.configure_item("menu_export_results_csv", enabled=True)
        dpg.configure_item("menu_export_results_excel", enabled=True)

        # Check for NSGA-II results and show Pareto plot
        is_nsga2 = hasattr(results_df, 'attrs') and 'nsga2_result' in results_df.attrs
        if is_nsga2:
            nsga2_result = results_df.attrs['nsga2_result']
            self._show_pareto_plot(nsga2_result)
            dpg.configure_item("build_pareto_container", show=True)
        else:
            # Hide Pareto plot for non-NSGA-II results
            dpg.configure_item("build_pareto_container", show=False)
            self._pareto_plot = None

        # Populate the table with all results
        self._populate_results_table(results_df)

    def _show_pareto_plot(self, nsga2_result):
        """Display Pareto front visualization for NSGA-II results."""
        # Clear existing pareto plot children
        if dpg.does_item_exist("build_pareto_plot_parent"):
            children = dpg.get_item_children("build_pareto_plot_parent", slot=1)
            if children:
                for child in children:
                    dpg.delete_item(child)

        # Get data from NSGA-II result
        pareto_front = nsga2_result.get('pareto_front')
        pareto_solutions = nsga2_result.get('pareto_solutions')
        knee_idx = nsga2_result.get('knee_idx', -1)

        if pareto_front is None or len(pareto_front) == 0:
            dpg.add_text("No Pareto solutions found", parent="build_pareto_plot_parent", color=COLORS["text_muted"])
            return

        # Create ParetoPlot component
        def on_solution_select(idx, data):
            """Handle solution selection in Pareto plot."""
            # Optionally highlight corresponding row in results table
            if self._search_results is not None and idx < len(self._search_results):
                # Could scroll to or highlight the row - for now just log
                print(f"Selected Pareto solution {idx}: {data}")

        self._pareto_plot = ParetoPlot(
            parent="build_pareto_plot_parent",
            width=500,
            height=350,
            on_solution_select=on_solution_select
        )

        # Set the data
        self._pareto_plot.set_data(
            pareto_front=pareto_front,
            pareto_solutions=pareto_solutions,
            knee_idx=knee_idx,
            objective_labels=['Error (RMSE)', 'Wavelength Fraction', 'Complexity']
        )

        # Auto-select knee point
        if knee_idx >= 0:
            self._pareto_plot.select_knee()

    def _on_results_sort(self, sender, sort_specs):
        """Handle sorting when user clicks on a column header."""
        if sort_specs is None or len(sort_specs) == 0:
            return

        if self._search_results is None or len(self._search_results) == 0:
            return

        # sort_specs is a list of (column_id, direction) tuples
        # direction: 1 = ascending, -1 = descending
        col_id, direction = sort_specs[0]
        ascending = (direction == 1)

        # Get column tag from col_id
        col_tag = None
        for item in ["col_model", "col_preproc", "col_metric1", "col_metric2", "col_vars", "col_varsel", "col_params", "col_topvars"]:
            if dpg.does_item_exist(item) and dpg.get_item_info(item)['id'] == col_id:
                col_tag = item
                break

        if col_tag is None:
            return

        # Determine which DataFrame column to sort by
        results_df = self._search_results
        sort_col = None

        if col_tag == "col_model":
            for c in ['Model', 'model']:
                if c in results_df.columns:
                    sort_col = c
                    break
        elif col_tag == "col_preproc":
            for c in ['Preprocessing', 'preprocessing']:
                if c in results_df.columns:
                    sort_col = c
                    break
        elif col_tag == "col_metric1":
            for c in ['RMSE', 'rmse', 'Accuracy', 'accuracy']:
                if c in results_df.columns:
                    sort_col = c
                    break
        elif col_tag == "col_metric2":
            for c in ['R2', 'r2', 'ROC_AUC', 'roc_auc', 'AUC']:
                if c in results_df.columns:
                    sort_col = c
                    break
        elif col_tag == "col_vars":
            for c in ['N_Variables', 'n_variables']:
                if c in results_df.columns:
                    sort_col = c
                    break
        elif col_tag == "col_varsel":
            for c in ['Variables', 'variables']:
                if c in results_df.columns:
                    sort_col = c
                    break
        elif col_tag == "col_params":
            for c in ['Params', 'params']:
                if c in results_df.columns:
                    sort_col = c
                    break
        elif col_tag == "col_topvars":
            for c in ['Top_Variables', 'top_variables']:
                if c in results_df.columns:
                    sort_col = c
                    break

        if sort_col is None:
            return

        # Sort the dataframe
        sorted_df = results_df.sort_values(sort_col, ascending=ascending)

        # Repopulate the table
        self._populate_results_table(sorted_df)

    def _populate_results_table(self, results_df):
        """Populate the results table with the given dataframe."""
        # Clear existing rows
        for child in dpg.get_item_children("build_results_table", 1) or []:
            dpg.delete_item(child)

        # Determine if regression or classification
        is_regression = 'RMSE' in results_df.columns or 'rmse' in results_df.columns

        # Update column labels based on task type
        if is_regression:
            dpg.configure_item("col_metric1", label="RMSE")
            dpg.configure_item("col_metric2", label="R²")
        else:
            dpg.configure_item("col_metric1", label="Accuracy")
            dpg.configure_item("col_metric2", label="AUC")

        # Show Save Model controls
        dpg.configure_item("build_save_spacer", show=True)
        dpg.configure_item("build_save_group", show=True)

        # Show Ensemble section
        dpg.configure_item("build_ensemble_group", show=True)

        # Track row items for highlighting
        self._result_row_items = {}

        # Track checkboxes for multi-select
        self._result_checkboxes = {}

        # Show ALL results
        for rank, (idx, row) in enumerate(results_df.iterrows(), 1):
            row_index = rank - 1  # 0-based index for selection
            row_items = []
            with dpg.table_row(parent="build_results_table", tag=f"result_row_{row_index}"):
                # Checkbox for selection
                checkbox = dpg.add_checkbox(
                    callback=lambda s, a, u: self._on_result_checkbox_changed(u),
                    user_data=row_index
                )
                self._result_checkboxes[row_index] = checkbox
                row_items.append(checkbox)

                # Rank number
                item = dpg.add_text(str(rank))
                row_items.append(item)

                # Model name
                item = dpg.add_text(str(row.get('Model', row.get('model', 'Unknown'))))
                row_items.append(item)

                # Preprocessing
                preprocess = row.get('Preprocessing', row.get('preprocessing', 'raw'))
                if self._search_preprocess == "auto":
                    display_preprocess = preprocess
                else:
                    display_preprocess = self._search_preprocess
                item = dpg.add_text(display_preprocess if display_preprocess else "raw")
                row_items.append(item)

                # Primary metric (RMSE or Accuracy)
                if is_regression:
                    rmse = row.get('RMSE', row.get('rmse', 0))
                    item = dpg.add_text(f"{rmse:.4f}")
                else:
                    acc = row.get('Accuracy', row.get('accuracy', 0))
                    item = dpg.add_text(f"{acc:.3f}")
                row_items.append(item)

                # Secondary metric (R² or AUC)
                if is_regression:
                    r2 = row.get('R2', row.get('r2', 0))
                    item = dpg.add_text(f"{r2:.3f}")
                else:
                    auc = row.get('ROC_AUC', row.get('roc_auc', row.get('AUC', 0)))
                    item = dpg.add_text(f"{auc:.3f}" if auc else "-")
                row_items.append(item)

                # Number of variables
                n_vars = row.get('N_Variables', row.get('n_variables', '-'))
                item = dpg.add_text(str(n_vars))
                row_items.append(item)

                # Variable selection method
                var_tag = row.get('Variables', row.get('variables', 'full'))
                item = dpg.add_text(str(var_tag))
                row_items.append(item)

                # Hyperparameters
                params = row.get('Params', row.get('params', ''))
                item = dpg.add_text(str(params) if params else "-")
                row_items.append(item)

                # Top Variables (wavelengths)
                top_vars = row.get('Top_Variables', row.get('top_variables', ''))
                item = dpg.add_text(str(top_vars) if top_vars else "-")
                row_items.append(item)

            # Store items for this row
            self._result_row_items[row_index] = row_items

    def _on_result_checkbox_changed(self, row_index: int):
        """Handle checkbox change for result row selection."""
        # Initialize selected indices set if needed
        if not hasattr(self, '_selected_result_indices'):
            self._selected_result_indices = set()

        # Get checkbox state
        checkbox = self._result_checkboxes.get(row_index)
        if checkbox and dpg.does_item_exist(checkbox):
            is_checked = dpg.get_value(checkbox)

            if is_checked:
                self._selected_result_indices.add(row_index)
            else:
                self._selected_result_indices.discard(row_index)

        # Update legacy single-selection tracking (for save model compatibility)
        if self._selected_result_indices:
            self._selected_result_index = min(self._selected_result_indices)
        else:
            self._selected_result_index = None

        # Highlight selected rows
        self._update_result_row_highlighting()

        # Update UI buttons
        has_selection = len(self._selected_result_indices) > 0
        dpg.configure_item("build_save_btn", enabled=has_selection)
        dpg.configure_item("build_diagnostics_btn", enabled=has_selection)

        # Show/enable validation test button if there's a validation set
        if self._validation_set is not None:
            dpg.configure_item("build_validation_test_btn", show=True, enabled=has_selection)

        # Update ensemble and compare controls (need 2+ models)
        n_selected = len(self._selected_result_indices)
        can_create_ensemble = n_selected >= 2
        dpg.configure_item("create_ensemble_btn", enabled=can_create_ensemble)
        if n_selected >= 2:
            dpg.set_value("ensemble_status_text", f"{n_selected} models selected - ready to create ensemble")
        elif n_selected == 1:
            dpg.set_value("ensemble_status_text", "Select 1 more model to create ensemble")
        else:
            dpg.set_value("ensemble_status_text", "Select 2+ models using checkboxes above")

        # Update status text
        self._update_selection_status()

    def _update_result_row_highlighting(self):
        """Update row highlighting based on checkbox selections."""
        if not hasattr(self, '_result_row_items'):
            return

        selected = getattr(self, '_selected_result_indices', set())

        for row_index, items in self._result_row_items.items():
            is_selected = row_index in selected
            for item in items:
                if dpg.does_item_exist(item):
                    item_type = dpg.get_item_type(item)
                    if "Text" in item_type:
                        color = COLORS["accent_primary"] if is_selected else COLORS["text_primary"]
                        dpg.configure_item(item, color=color)

    def _update_selection_status(self):
        """Update the status text based on current selection."""
        selected = getattr(self, '_selected_result_indices', set())
        n_selected = len(selected)

        if n_selected == 0:
            dpg.set_value("build_save_status", "Select models using checkboxes")
        elif n_selected == 1:
            row_index = list(selected)[0]
            if self._search_results is not None and row_index < len(self._search_results):
                row = self._search_results.iloc[row_index]
                model_name = row.get('Model', row.get('model', 'Unknown'))
                preproc = row.get('Preprocessing', row.get('preprocessing', 'raw'))
                n_vars = row.get('N_Variables', row.get('n_variables', '-'))
                var_tag = row.get('Variables', row.get('variables', 'full'))

                is_regression = 'RMSE' in self._search_results.columns or 'rmse' in self._search_results.columns
                if is_regression:
                    rmse = row.get('RMSE', row.get('rmse', 0))
                    r2 = row.get('R2', row.get('r2', 0))
                    metrics_str = f"RMSE={rmse:.4f}, R²={r2:.3f}"
                else:
                    acc = row.get('Accuracy', row.get('accuracy', 0))
                    metrics_str = f"Accuracy={acc:.3f}"

                status = f"Selected: {model_name} | {preproc} | {metrics_str} | {n_vars} vars ({var_tag})"
                dpg.set_value("build_save_status", status)
        else:
            dpg.set_value("build_save_status", f"{n_selected} models selected")

    def _on_result_row_selected(self, row_index: int):
        """Handle selection of a result row (legacy - now uses checkboxes)."""
        # Toggle the checkbox for this row
        if hasattr(self, '_result_checkboxes'):
            checkbox = self._result_checkboxes.get(row_index)
            if checkbox and dpg.does_item_exist(checkbox):
                current = dpg.get_value(checkbox)
                dpg.set_value(checkbox, not current)
                self._on_result_checkbox_changed(row_index)

    def _on_save_model(self, sender=None, app_data=None):
        """Handle Save Model button click - save the selected model configuration."""
        if self._selected_result_index is None:
            return

        if self._search_results is None or self._selected_result_index >= len(self._search_results):
            dpg.set_value("build_save_status", "Error: Invalid selection")
            return

        # Get the selected result row
        selected_row = self._search_results.iloc[self._selected_result_index]

        # Store row for callback
        self._pending_save_row = selected_row

        # Show file save dialog
        def save_callback(filepath):
            row = getattr(self, '_pending_save_row', None)
            if row is not None:
                self._save_model_to_file(row, filepath)
            self._pending_save_row = None

        self.show_save_dialog(save_callback, dialog_type="model", initial_dir=get_desktop_path())

    def _save_model_to_file(self, selected_row, filepath: str):
        """
        Retrain the selected model configuration on full data and save it.

        Parameters
        ----------
        selected_row : pd.Series
            Row from results DataFrame with model configuration
        filepath : str
            Path to save the model
        """
        from ..core import model_io
        from ..core.models import get_model
        from ..core.preprocess import SNV, SavgolDerivative
        import numpy as np

        try:
            # Update status
            dpg.set_value("build_save_status", "Retraining model on full dataset...")

            # Get model configuration
            model_name = selected_row.get('Model', selected_row.get('model'))
            preprocessing = selected_row.get('Preprocessing', selected_row.get('preprocessing', 'raw'))
            params_str = selected_row.get('Params', selected_row.get('params', ''))

            # Parse parameters from string
            params = {}
            if params_str and params_str != 'default' and params_str != '-':
                # Parse "n_components=10" or "alpha=1.0, max_iter=5000" style strings
                for param_pair in params_str.split(','):
                    param_pair = param_pair.strip()
                    if '=' in param_pair:
                        key, val = param_pair.split('=', 1)
                        key = key.strip()
                        val = val.strip()
                        # Handle None values
                        if val.lower() == 'none':
                            params[key] = None
                        else:
                            # Try to convert to appropriate type
                            try:
                                if '.' in val:
                                    params[key] = float(val)
                                else:
                                    params[key] = int(val)
                            except ValueError:
                                params[key] = val

            # Get full training data
            if self._current_dataset is None:
                dpg.set_value("build_save_status", "Error: No dataset loaded")
                return

            X = self._current_dataset.X
            y = self._current_dataset.y
            wavelengths = self._current_dataset.wavelengths
            target_name = self._current_dataset.target_name or 'Target'
            task_type = self._current_dataset.metadata.get('target_type', 'regression')

            # Apply preprocessing
            X_processed = X.copy()
            if preprocessing != 'raw':
                if preprocessing == 'snv':
                    X_processed = SNV().fit_transform(X_processed)
                elif preprocessing.startswith('deriv1_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_processed = SavgolDerivative(deriv=1, window=window).fit_transform(X_processed)
                elif preprocessing.startswith('deriv2_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_processed = SavgolDerivative(deriv=2, window=window).fit_transform(X_processed)
                elif preprocessing.startswith('snv_deriv1_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_snv = SNV().fit_transform(X_processed)
                    X_processed = SavgolDerivative(deriv=1, window=window).fit_transform(X_snv)
                elif preprocessing.startswith('snv_deriv2_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_snv = SNV().fit_transform(X_processed)
                    X_processed = SavgolDerivative(deriv=2, window=window).fit_transform(X_snv)

            # Handle variable selection if used
            variable_indices = None
            var_tag = selected_row.get('Variables', selected_row.get('variables', 'full'))
            if var_tag != 'full' and var_tag.startswith('top'):
                # Variable selection was used - we need to get the indices
                # For now, we'll save without variable selection (using full spectrum)
                # This is a simplification - full implementation would require re-running variable selection
                pass

            # Create and train model
            model = get_model(model_name, task_type=task_type, **params)
            if model is None:
                dpg.set_value("build_save_status", f"Error: Model {model_name} not available")
                return

            model.fit(X_processed, y)

            # Get training metrics
            is_regression = task_type == 'regression'
            if is_regression:
                rmse = selected_row.get('RMSE', selected_row.get('rmse', 0))
                r2 = selected_row.get('R2', selected_row.get('r2', 0))
                metrics = {'RMSE': rmse, 'R2': r2}
            else:
                acc = selected_row.get('Accuracy', selected_row.get('accuracy', 0))
                auc = selected_row.get('ROC_AUC', selected_row.get('roc_auc', 0))
                metrics = {'Accuracy': acc, 'ROC_AUC': auc if auc else None}

            # Create model bundle with applicability domain
            # Pass X_processed (preprocessed training data) for AD computation
            model_bundle = model_io.create_model_bundle(
                model=model,
                model_name=model_name,
                preprocessing=preprocessing,
                wavelengths=wavelengths if wavelengths is not None else np.arange(X.shape[1]),
                target_name=target_name,
                task_type=task_type,
                metrics=metrics,
                params=params,
                variable_indices=variable_indices,
                X_train=X_processed  # Include preprocessed training data for AD
            )

            # Save to file
            model_io.save_model(model_bundle, filepath)

            # Update status
            dpg.set_value("build_save_status", f"Model saved successfully to: {Path(filepath).name}")

        except Exception as e:
            dpg.set_value("build_save_status", f"Error saving model: {str(e)}")
            print(f"Error in _save_model_to_file: {e}")
            import traceback
            traceback.print_exc()

    def _on_show_diagnostics(self, sender=None, app_data=None):
        """Show model diagnostics for selected model(s). Supports multi-model toggle."""
        from sklearn.model_selection import cross_val_predict, KFold
        from ..core.models import get_model
        from ..core.preprocess import SNV, SavgolDerivative
        import numpy as np

        # Get all selected indices (use set if available, otherwise single index)
        selected_indices = []
        if hasattr(self, '_selected_result_indices') and self._selected_result_indices:
            selected_indices = sorted(self._selected_result_indices)
        elif self._selected_result_index is not None:
            selected_indices = [self._selected_result_index]

        if not selected_indices:
            dpg.set_value("build_save_status", "Select a model first")
            return

        if self._search_results is None:
            dpg.set_value("build_save_status", "Error: No results available")
            return

        if self._current_dataset is None:
            dpg.set_value("build_save_status", "No dataset available")
            return

        try:
            n_models = len(selected_indices)
            dpg.set_value("build_save_status", f"Computing diagnostics for {n_models} model(s)...")

            # Get training data - use calibration subset if validation holdout was enabled
            X = self._current_dataset.X
            y = self._current_dataset.y
            if self._calibration_indices is not None:
                X = X[self._calibration_indices]
                y = y[self._calibration_indices]

            task_type = dpg.get_value("build_task_type").lower()

            # Get sample IDs if available
            sample_ids = None
            if hasattr(self._current_dataset, 'sample_ids') and self._current_dataset.sample_ids:
                sample_ids = list(self._current_dataset.sample_ids)

            # Get CV object for reproducible results
            cv_obj = getattr(self, '_cv_object', None)
            if cv_obj is None:
                cv_obj = KFold(n_splits=getattr(self, '_cv_folds', 5), shuffle=True, random_state=42)

            # Build list of model diagnostics data
            models_data = []

            for idx in selected_indices:
                if idx >= len(self._search_results):
                    continue

                selected_row = self._search_results.iloc[idx]

                # Get model configuration
                model_name = selected_row.get('Model', selected_row.get('model'))
                preprocessing = selected_row.get('Preprocessing', selected_row.get('preprocessing', 'raw'))
                params_str = selected_row.get('Params', selected_row.get('params', ''))

                # Parse parameters
                params = {}
                if params_str and params_str != 'default' and params_str != '-':
                    for param_pair in params_str.split(','):
                        param_pair = param_pair.strip()
                        if '=' in param_pair:
                            key, val = param_pair.split('=', 1)
                            key = key.strip()
                            val = val.strip()
                            if val.lower() == 'none':
                                params[key] = None
                            else:
                                try:
                                    if '.' in val:
                                        params[key] = float(val)
                                    else:
                                        params[key] = int(val)
                                except ValueError:
                                    params[key] = val

                # Apply preprocessing
                X_proc = X.copy()
                if preprocessing == 'snv':
                    X_proc = SNV().fit_transform(X_proc)
                elif preprocessing.startswith('deriv1_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_proc = SavgolDerivative(deriv=1, window=window).fit_transform(X_proc)
                elif preprocessing.startswith('deriv2_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_proc = SavgolDerivative(deriv=2, window=window).fit_transform(X_proc)
                elif preprocessing.startswith('snv_deriv'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    deriv = 1 if 'deriv1' in preprocessing else 2
                    X_snv = SNV().fit_transform(X_proc)
                    X_proc = SavgolDerivative(deriv=deriv, window=window).fit_transform(X_snv)
                elif 'deriv' in preprocessing and '_snv' in preprocessing:
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    deriv = 1 if 'deriv1' in preprocessing else (2 if 'deriv2' in preprocessing else 1)
                    X_deriv = SavgolDerivative(deriv=deriv, window=window).fit_transform(X_proc)
                    X_proc = SNV().fit_transform(X_deriv)

                # Apply variable selection if indices were stored
                variable_indices = selected_row.get('variable_indices', None)
                if variable_indices is not None:
                    if not isinstance(variable_indices, np.ndarray):
                        variable_indices = np.array(variable_indices)
                    X_proc = X_proc[:, variable_indices]

                # Create model and get CV predictions
                model = get_model(model_name, task_type=task_type, random_state=42, **params)
                if model is None:
                    continue

                y_pred = cross_val_predict(model, X_proc, y, cv=cv_obj)

                # Add to models list
                models_data.append({
                    'name': f"{model_name} ({preprocessing})",
                    'y_true': y,
                    'y_pred': y_pred,
                    'task_type': task_type,
                    'sample_ids': sample_ids
                })

            if not models_data:
                dpg.set_value("build_save_status", "Error: Could not compute diagnostics")
                return

            # Update diagnostics panel with all models
            self._diagnostics_panel.set_multiple_models(models_data)

            if n_models == 1:
                dpg.set_value("build_save_status", f"Diagnostics: {models_data[0]['name']}")
            else:
                dpg.set_value("build_save_status", f"Diagnostics for {len(models_data)} models (use < > to toggle)")

        except Exception as e:
            dpg.set_value("build_save_status", f"Diagnostics error: {str(e)}")
            print(f"Error in _on_show_diagnostics: {e}")
            import traceback
            traceback.print_exc()

    def _on_test_validation(self, sender=None, app_data=None):
        """Test the selected model on the held-out validation set."""
        import numpy as np
        from ..core.models import get_model
        from ..core.preprocess import SNV, SavgolDerivative
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

        if self._selected_result_index is None:
            return

        if self._validation_set is None:
            dpg.set_value("build_save_status", "No validation set available")
            return

        if self._search_results is None or self._selected_result_index >= len(self._search_results):
            dpg.set_value("build_save_status", "Error: Invalid selection")
            return

        try:
            # Get selected result row
            selected_row = self._search_results.iloc[self._selected_result_index]

            # Get model configuration
            model_name = selected_row.get('Model', selected_row.get('model'))
            preprocessing = selected_row.get('Preprocessing', selected_row.get('preprocessing', 'raw'))
            params_str = selected_row.get('Params', selected_row.get('params', ''))
            task_type = dpg.get_value("build_task_type").lower()

            # Parse parameters
            params = {}
            if params_str and params_str != 'default' and params_str != '-':
                for param_pair in params_str.split(','):
                    param_pair = param_pair.strip()
                    if '=' in param_pair:
                        key, val = param_pair.split('=', 1)
                        key = key.strip()
                        val = val.strip()
                        # Handle None values
                        if val.lower() == 'none':
                            params[key] = None
                        else:
                            try:
                                if '.' in val:
                                    params[key] = float(val)
                                else:
                                    params[key] = int(val)
                            except ValueError:
                                params[key] = val

            # Get calibration and validation data
            X_cal = self._current_dataset.X[self._calibration_indices]
            y_cal = self._current_dataset.y[self._calibration_indices]
            X_val = self._validation_set['X']
            y_val = self._validation_set['y']

            # Apply preprocessing to both sets
            def apply_preprocessing(X, preproc_name):
                X_out = X.copy()
                if preproc_name == 'raw':
                    return X_out
                elif preproc_name == 'snv':
                    return SNV().fit_transform(X_out)
                elif preproc_name.startswith('deriv1_w'):
                    window = int(preproc_name.split('w')[1]) if 'w' in preproc_name else 7
                    return SavgolDerivative(deriv=1, window=window).fit_transform(X_out)
                elif preproc_name.startswith('deriv2_w'):
                    window = int(preproc_name.split('w')[1]) if 'w' in preproc_name else 7
                    return SavgolDerivative(deriv=2, window=window).fit_transform(X_out)
                elif preproc_name.startswith('snv_deriv1_w'):
                    window = int(preproc_name.split('w')[1]) if 'w' in preproc_name else 7
                    X_snv = SNV().fit_transform(X_out)
                    return SavgolDerivative(deriv=1, window=window).fit_transform(X_snv)
                elif preproc_name.startswith('snv_deriv2_w'):
                    window = int(preproc_name.split('w')[1]) if 'w' in preproc_name else 7
                    X_snv = SNV().fit_transform(X_out)
                    return SavgolDerivative(deriv=2, window=window).fit_transform(X_snv)
                elif preproc_name.startswith('deriv1_snv_w'):
                    window = int(preproc_name.split('w')[1]) if 'w' in preproc_name else 7
                    X_deriv = SavgolDerivative(deriv=1, window=window).fit_transform(X_out)
                    return SNV().fit_transform(X_deriv)
                elif preproc_name.startswith('deriv2_snv_w'):
                    window = int(preproc_name.split('w')[1]) if 'w' in preproc_name else 7
                    X_deriv = SavgolDerivative(deriv=2, window=window).fit_transform(X_out)
                    return SNV().fit_transform(X_deriv)
                return X_out

            X_cal_proc = apply_preprocessing(X_cal, preprocessing)
            X_val_proc = apply_preprocessing(X_val, preprocessing)

            # Handle classification with string labels
            label_encoder = None
            y_cal_fit = y_cal
            y_val_compare = y_val
            is_regression = task_type == 'regression'

            if not is_regression:
                # Check if labels are strings/objects
                if y_cal.dtype == object or not np.issubdtype(y_cal.dtype, np.number):
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    y_cal_fit = label_encoder.fit_transform(y_cal)
                    # Note: y_val stays as original for comparison since model predicts original labels

            # Train model on calibration set
            model = get_model(model_name, task_type=task_type, **params)
            if model is None:
                dpg.set_value("build_save_status", f"Error: Model {model_name} not available")
                return

            model.fit(X_cal_proc, y_cal_fit)

            # Predict on validation set
            y_pred = model.predict(X_val_proc)

            # For models that return encoded values, decode them
            if label_encoder is not None and hasattr(model, '_label_encoder') and model._label_encoder is None:
                # Model doesn't have its own encoder, so it returns encoded values
                y_pred = label_encoder.inverse_transform(y_pred.astype(int))

            # Calculate metrics
            if is_regression:
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_r2 = r2_score(y_val, y_pred)
                metrics_text = f"Validation RMSE: {val_rmse:.4f} | R2: {val_r2:.3f}"
            else:
                val_acc = accuracy_score(y_val, y_pred)
                metrics_text = f"Validation Accuracy: {val_acc:.3f}"

            # Show validation results section
            dpg.configure_item("build_validation_results_group", show=True)
            dpg.set_value("build_validation_results_text", metrics_text)

            # Clear existing table rows
            children = dpg.get_item_children("build_validation_table", 1)
            if children:
                for child in children:
                    dpg.delete_item(child)

            dpg.configure_item("build_validation_table", show=True)

            # Add validation result rows
            sample_ids = self._validation_set.get('sample_ids') or [f"S{i}" for i in range(len(y_val))]
            residuals = y_pred - y_val

            # Determine status thresholds based on residual distribution
            if is_regression:
                residual_std = np.std(residuals)
                for i in range(len(y_val)):
                    with dpg.table_row(parent="build_validation_table"):
                        dpg.add_text(str(sample_ids[i]) if sample_ids[i] else f"Sample {i+1}")
                        dpg.add_text(f"{y_val[i]:.3f}")
                        dpg.add_text(f"{y_pred[i]:.3f}")
                        dpg.add_text(f"{residuals[i]:+.3f}")

                        # Status based on residual
                        abs_res = abs(residuals[i])
                        if abs_res <= residual_std:
                            status = "good"
                            color = COLORS["accent_primary"]
                        elif abs_res <= 2 * residual_std:
                            status = "caution"
                            color = COLORS["accent_warning"]
                        else:
                            status = "outlier"
                            color = COLORS["accent_error"]
                        dpg.add_text(status, color=color)
            else:
                # Classification
                for i in range(len(y_val)):
                    with dpg.table_row(parent="build_validation_table"):
                        dpg.add_text(str(sample_ids[i]) if sample_ids[i] else f"Sample {i+1}")
                        dpg.add_text(str(y_val[i]))
                        dpg.add_text(str(y_pred[i]))
                        dpg.add_text("-")

                        if y_val[i] == y_pred[i]:
                            dpg.add_text("correct", color=COLORS["accent_primary"])
                        else:
                            dpg.add_text("incorrect", color=COLORS["accent_error"])

            # Update diagnostics panel with validation predictions
            self._diagnostics_panel.set_data(
                y_true=y_val,
                y_pred=y_pred,
                task_type=task_type,
                model_name=f"{model_name} ({preprocessing}) - Validation",
                sample_ids=list(sample_ids) if sample_ids is not None else None
            )

            dpg.set_value("build_save_status", f"Validation complete: {len(y_val)} samples tested")

        except Exception as e:
            dpg.set_value("build_save_status", f"Validation error: {str(e)}")
            print(f"Error in _on_test_validation: {e}")
            import traceback
            traceback.print_exc()

    def _on_create_ensemble(self, sender=None, app_data=None):
        """Create an ensemble from selected models."""
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import mean_squared_error, r2_score

        if not hasattr(self, '_selected_result_indices') or len(self._selected_result_indices) < 2:
            dpg.set_value("ensemble_status_text", "Select at least 2 models")
            return

        if self._search_results is None or self._current_dataset is None:
            dpg.set_value("ensemble_status_text", "No search results available")
            return

        try:
            dpg.set_value("ensemble_status_text", "Creating ensemble...")

            # Get ensemble settings
            method_display = dpg.get_value("ensemble_method_combo")
            n_regions = dpg.get_value("ensemble_n_regions")

            # Map display name to ensemble type
            method_map = {
                "Simple Average": "simple_average",
                "Region Weighted": "region_weighted",
                "Mixture of Experts": "mixture_experts",
                "Stacking": "stacking"
            }
            ensemble_type = method_map.get(method_display, "region_weighted")

            # Get training data
            X = self._current_dataset.X
            y = self._current_dataset.y
            task_type = self._current_dataset.metadata.get('target_type', 'regression')

            from ..core.models import get_model
            from ..core.preprocess import SNV, SavgolDerivative

            # Train each selected model and collect predictions
            models = []
            model_names = []
            preprocessors = []
            cv_predictions = []

            sorted_indices = sorted(self._selected_result_indices)

            for row_index in sorted_indices:
                row = self._search_results.iloc[row_index]
                model_name = row.get('Model', row.get('model'))
                preprocessing = row.get('Preprocessing', row.get('preprocessing', 'raw'))
                params_str = row.get('Params', row.get('params', ''))

                # Parse parameters
                params = {}
                if params_str and params_str != 'default' and params_str != '-':
                    for param_pair in params_str.split(','):
                        param_pair = param_pair.strip()
                        if '=' in param_pair:
                            key, val = param_pair.split('=', 1)
                            key = key.strip()
                            val = val.strip()
                            if val.lower() == 'none':
                                params[key] = None
                            else:
                                try:
                                    if '.' in val:
                                        params[key] = float(val)
                                    else:
                                        params[key] = int(val)
                                except ValueError:
                                    params[key] = val

                # Apply preprocessing
                X_proc = X.copy()
                if preprocessing == 'snv':
                    X_proc = SNV().fit_transform(X_proc)
                elif preprocessing.startswith('deriv1_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_proc = SavgolDerivative(deriv=1, window=window).fit_transform(X_proc)
                elif preprocessing.startswith('deriv2_w'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    X_proc = SavgolDerivative(deriv=2, window=window).fit_transform(X_proc)
                elif preprocessing.startswith('snv_deriv'):
                    window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                    deriv = 1 if 'deriv1' in preprocessing else 2
                    X_snv = SNV().fit_transform(X_proc)
                    X_proc = SavgolDerivative(deriv=deriv, window=window).fit_transform(X_snv)

                # Create and train model
                model = get_model(model_name, task_type=task_type, **params)
                if model is not None:
                    model.fit(X_proc, y)
                    models.append(model)
                    model_names.append(f"{model_name}_{preprocessing}")
                    preprocessors.append(None)  # Already preprocessed

                    # Get CV predictions for evaluation
                    try:
                        cv_pred = cross_val_predict(model, X_proc, y, cv=5)
                        cv_predictions.append(cv_pred)
                    except:
                        cv_predictions.append(model.predict(X_proc))

            if len(models) < 2:
                dpg.set_value("ensemble_status_text", "Failed to train enough models")
                return

            # Create ensemble
            ensemble = create_ensemble(
                models=models,
                model_names=model_names,
                X=X,
                y=y,
                ensemble_type=ensemble_type,
                n_regions=n_regions,
                preprocessors=preprocessors
            )

            # Get ensemble CV predictions
            ensemble_pred = ensemble.predict(X)

            # Calculate metrics
            if task_type == 'regression':
                ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
                ensemble_r2 = r2_score(y, ensemble_pred)

                # Calculate individual model metrics
                individual_metrics = []
                for i, cv_pred in enumerate(cv_predictions):
                    rmse = np.sqrt(mean_squared_error(y, cv_pred))
                    r2 = r2_score(y, cv_pred)
                    individual_metrics.append((model_names[i], rmse, r2))

                # Update UI with results
                dpg.configure_item("ensemble_results_group", show=True)
                dpg.set_value(
                    "ensemble_results_text",
                    f"Ensemble ({method_display}): RMSE={ensemble_rmse:.4f}, R²={ensemble_r2:.3f}"
                )

                # Populate comparison table
                dpg.configure_item("ensemble_comparison_table", show=True)
                children = dpg.get_item_children("ensemble_comparison_table", 1)
                if children:
                    for child in children:
                        dpg.delete_item(child)

                # Add ensemble row
                with dpg.table_row(parent="ensemble_comparison_table"):
                    dpg.add_text(f"ENSEMBLE ({method_display})", color=COLORS["accent_primary"])
                    dpg.add_text(f"{ensemble_rmse:.4f}", color=COLORS["accent_primary"])
                    dpg.add_text(f"{ensemble_r2:.3f}", color=COLORS["accent_primary"])
                    dpg.add_text(f"{len(models)} models combined")

                # Add individual model rows
                for name, rmse, r2 in individual_metrics:
                    improvement = ((rmse - ensemble_rmse) / rmse) * 100 if rmse > 0 else 0
                    with dpg.table_row(parent="ensemble_comparison_table"):
                        dpg.add_text(name[:25])
                        dpg.add_text(f"{rmse:.4f}")
                        dpg.add_text(f"{r2:.3f}")
                        if improvement > 0:
                            dpg.add_text(f"+{improvement:.1f}% vs ensemble", color=COLORS["text_muted"])
                        else:
                            dpg.add_text(f"{improvement:.1f}% vs ensemble", color=COLORS["text_muted"])

            # Store ensemble for saving
            self._current_ensemble = ensemble
            self._ensemble_models = models
            self._ensemble_model_names = model_names
            dpg.configure_item("save_ensemble_btn", enabled=True)

            # Show validation button if validation set exists
            if self._validation_set is not None:
                dpg.configure_item("test_ensemble_validation_btn", show=True, enabled=True)

            dpg.set_value("ensemble_status_text", f"Ensemble created with {len(models)} models")

        except Exception as e:
            dpg.set_value("ensemble_status_text", f"Error: {str(e)}")
            print(f"Error in _on_create_ensemble: {e}")
            import traceback
            traceback.print_exc()

    def _on_save_ensemble(self, sender=None, app_data=None):
        """Save the created ensemble model."""
        if not hasattr(self, '_current_ensemble') or self._current_ensemble is None:
            return

        from ..core import model_io

        def save_callback(filepath):
            try:
                # Create ensemble bundle
                ensemble_bundle = {
                    'model': self._current_ensemble,
                    'model_name': 'Ensemble',
                    'ensemble_type': dpg.get_value("ensemble_method_combo"),
                    'base_models': self._ensemble_model_names,
                    'wavelengths': self._current_dataset.wavelengths if self._current_dataset else None,
                    'target_name': self._current_dataset.target_name if self._current_dataset else 'Target',
                    'task_type': self._current_dataset.metadata.get('target_type', 'regression') if self._current_dataset else 'regression',
                    'preprocessing': 'various',  # Multiple preprocessing methods in ensemble
                }

                model_io.save_model(ensemble_bundle, filepath)
                dpg.set_value("ensemble_status_text", f"Ensemble saved to: {Path(filepath).name}")
            except Exception as e:
                dpg.set_value("ensemble_status_text", f"Error saving: {str(e)}")

        self.show_save_dialog(save_callback, dialog_type="model", initial_dir=get_desktop_path())

    def _on_test_ensemble_validation(self, sender=None, app_data=None):
        """Test the ensemble on validation set."""
        from sklearn.metrics import mean_squared_error, r2_score

        if not hasattr(self, '_current_ensemble') or self._current_ensemble is None:
            return

        if self._validation_set is None:
            dpg.set_value("ensemble_status_text", "No validation set available")
            return

        try:
            X_val = self._validation_set['X']
            y_val = self._validation_set['y']

            # Predict using ensemble
            y_pred = self._current_ensemble.predict(X_val)

            # Calculate metrics
            task_type = self._current_dataset.metadata.get('target_type', 'regression') if self._current_dataset else 'regression'

            if task_type == 'regression':
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_r2 = r2_score(y_val, y_pred)
                dpg.set_value(
                    "ensemble_status_text",
                    f"Validation: RMSE={val_rmse:.4f}, R²={val_r2:.3f} ({len(y_val)} samples)"
                )
            else:
                from sklearn.metrics import accuracy_score
                val_acc = accuracy_score(y_val, y_pred)
                dpg.set_value(
                    "ensemble_status_text",
                    f"Validation Accuracy: {val_acc:.3f} ({len(y_val)} samples)"
                )

        except Exception as e:
            dpg.set_value("ensemble_status_text", f"Validation error: {str(e)}")
            print(f"Error in _on_test_ensemble_validation: {e}")

    def _on_load_model(self, sender=None, app_data=None):
        """Handle Load Model button - open file dialog to select model."""
        self.show_open_model_dialog(self._load_model_file, get_desktop_path())

    def _load_model_file(self, filepath: str):
        """Load a model from file and add to loaded models list."""
        from ..core import model_io
        import numpy as np

        try:
            # Load model bundle
            model_bundle = model_io.load_model(filepath)

            # Add filename to bundle for display
            model_bundle['filename'] = Path(filepath).name

            # Add to loaded models list
            self._loaded_models.append(model_bundle)

            # Update UI to show loaded models
            self._update_loaded_models_ui()

            # Show "Add Additional Model" button after first model
            if len(self._loaded_models) == 1:
                dpg.configure_item("predict_add_model_btn", show=True)
                # Enable spectra loading
                dpg.configure_item("predict_load_file_btn", enabled=True)
                dpg.configure_item("predict_load_folder_btn", enabled=True)

            # Clear any previous predictions when models change
            self._predictions = None
            dpg.configure_item("predict_run_btn", enabled=False)
            dpg.configure_item("predict_export_btn", enabled=False)

            # Update status
            dpg.set_value("predict_status", f"Model loaded: {Path(filepath).name} ({len(self._loaded_models)} total)")

        except Exception as e:
            dpg.set_value("predict_status", f"Error loading model: {str(e)}")
            print(f"Error in _load_model_file: {e}")
            import traceback
            traceback.print_exc()

    def _update_loaded_models_ui(self):
        """Update the UI to display all loaded models."""
        import numpy as np

        # Clear existing model widgets
        if dpg.does_item_exist("predict_models_container"):
            for child in dpg.get_item_children("predict_models_container", 1) or []:
                dpg.delete_item(child)

        # Show models list if we have models
        if len(self._loaded_models) > 0:
            dpg.configure_item("predict_models_list", show=True)
            dpg.configure_item("predict_model_count", show=True)
            dpg.set_value("predict_model_count", f"{len(self._loaded_models)} model(s) loaded")

            # Add each model to the list
            for i, model_bundle in enumerate(self._loaded_models):
                with dpg.group(parent="predict_models_container", horizontal=True):
                    # Model info text
                    model_name = model_bundle.get('model_name', 'Unknown')
                    target_name = model_bundle.get('target_name', 'Unknown')
                    preproc = model_bundle.get('preprocessing', 'raw')
                    has_ad = model_bundle.get('has_applicability_domain', False)
                    ad_marker = " [AD]" if has_ad else ""

                    info_text = f"{i+1}. {model_name} ({target_name}) - {preproc}{ad_marker}"
                    dpg.add_text(info_text, color=COLORS["text_primary"])

                    # Remove button
                    dpg.add_button(
                        label="X",
                        callback=lambda s, a, u: self._remove_model(u),
                        user_data=i,
                        width=30
                    )
        else:
            dpg.configure_item("predict_models_list", show=False)
            dpg.configure_item("predict_model_count", show=False)

    def _remove_model(self, model_index: int):
        """Remove a model from the loaded models list."""
        if 0 <= model_index < len(self._loaded_models):
            removed = self._loaded_models.pop(model_index)
            self._update_loaded_models_ui()

            # Hide "Add Additional Model" if no models left
            if len(self._loaded_models) == 0:
                dpg.configure_item("predict_add_model_btn", show=False)
                dpg.configure_item("predict_load_file_btn", enabled=False)
                dpg.configure_item("predict_load_folder_btn", enabled=False)
                dpg.configure_item("predict_run_btn", enabled=False)

            # Clear predictions
            self._predictions = None
            dpg.configure_item("predict_export_btn", enabled=False)

            dpg.set_value("predict_status", f"Model removed: {removed.get('filename', 'Unknown')}")

    def _on_load_predict_spectra_file_btn(self, sender=None, app_data=None):
        """Handle Load File button - open file dialog to select single spectra file."""
        self.show_open_dialog(self._load_predict_spectra_from_path, get_desktop_path())

    def _on_load_predict_spectra_folder_btn(self, sender=None, app_data=None):
        """Handle Load Folder button - open folder dialog to select spectra folder."""
        self.show_folder_dialog(self._load_predict_spectra_from_path, get_desktop_path())

    def _load_predict_spectra_from_path(self, path: str):
        """Load spectra from file or folder and validate wavelength compatibility."""
        import numpy as np
        from pathlib import Path

        try:
            from ..core import io_utils

            path_obj = Path(path)
            is_folder = path_obj.is_dir()

            dpg.set_value("predict_status", f"Loading {'folder' if is_folder else 'file'}: {path_obj.name}...")

            # Use the universal loader that handles both files and folders
            X_new, wavelengths_new, sample_ids = io_utils.load_spectra(path)

            # Store data
            self._predict_spectra = X_new
            self._predict_wavelengths = wavelengths_new
            self._predict_sample_ids = sample_ids

            # Update UI
            dpg.configure_item("predict_spectra_info", show=True)

            format_type = "folder" if is_folder else "file"
            dpg.set_value("predict_spectra_count", f"{len(X_new)} samples from {format_type}, {X_new.shape[1]} wavelengths")
            dpg.set_value("predict_spectra_range", f"{wavelengths_new[0]:.1f} - {wavelengths_new[-1]:.1f} nm")

            # Check wavelength compatibility with loaded models
            if len(self._loaded_models) > 0:
                # Check against first model (primary model)
                model_wavelengths = self._loaded_models[0].get('wavelengths', np.array([]))

                # Check if wavelengths match
                if not np.array_equal(wavelengths_new, model_wavelengths):
                    # Check overlap
                    overlap_min = max(wavelengths_new.min(), model_wavelengths.min())
                    overlap_max = min(wavelengths_new.max(), model_wavelengths.max())

                    if overlap_min >= overlap_max:
                        warning = "WARNING: No wavelength overlap with models! Prediction will likely fail."
                        dpg.set_value("predict_wavelength_warning", warning)
                        dpg.configure_item("predict_wavelength_warning", show=True, color=COLORS["accent_error"])
                    else:
                        warning = f"Note: Wavelengths differ from models. Will interpolate to match training range."
                        dpg.set_value("predict_wavelength_warning", warning)
                        dpg.configure_item("predict_wavelength_warning", show=True, color=COLORS["accent_warning"])
                else:
                    dpg.configure_item("predict_wavelength_warning", show=False)

            # Enable prediction button
            dpg.configure_item("predict_run_btn", enabled=True)
            dpg.set_value("predict_status", f"Spectra loaded: {path_obj.name} ({len(X_new)} samples)")

        except Exception as e:
            dpg.set_value("predict_status", f"Error loading spectra: {str(e)}")
            print(f"Error in _load_predict_spectra_from_path: {e}")
            import traceback
            traceback.print_exc()

    def _on_run_prediction(self, sender=None, app_data=None):
        """Handle Predict button - run predictions with all loaded models."""
        if len(self._loaded_models) == 0 or self._predict_spectra is None:
            dpg.set_value("predict_status", "Error: Load model(s) and spectra first")
            return

        from ..core import model_io
        import numpy as np

        try:
            dpg.set_value("predict_status", "Running predictions with all models...")
            dpg.configure_item("predict_run_btn", enabled=False)

            # Dictionary to store predictions from each model
            all_predictions = {}
            all_ad_status = {}
            interpolation_used = False
            has_any_ad = False

            # Run prediction with each loaded model
            for i, model_bundle in enumerate(self._loaded_models):
                model_name = model_bundle.get('model_name', f'Model{i+1}')
                target_name = model_bundle.get('target_name', 'Prediction')

                # Run prediction
                predictions, info = model_io.apply_model(
                    model_bundle,
                    self._predict_spectra,
                    self._predict_wavelengths
                )

                # Store predictions with model identifier
                model_key = f"{model_name}_{target_name}"
                all_predictions[model_key] = predictions

                if info['interpolated']:
                    interpolation_used = True

                # Compute applicability domain status if available
                if model_bundle.get('has_applicability_domain', False):
                    # Need to apply preprocessing before AD check
                    from ..core.preprocess import SNV, SavgolDerivative
                    X_processed = self._predict_spectra.copy()

                    # Apply same preprocessing as model
                    preprocessing = model_bundle.get('preprocessing', 'raw')
                    if preprocessing != 'raw':
                        # Parse and apply preprocessing (same logic as in apply_model)
                        if preprocessing == 'snv':
                            X_processed = SNV().fit_transform(X_processed)
                        elif preprocessing.startswith('deriv1_w'):
                            window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                            X_processed = SavgolDerivative(deriv=1, window=window).fit_transform(X_processed)
                        elif preprocessing.startswith('deriv2_w'):
                            window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                            X_processed = SavgolDerivative(deriv=2, window=window).fit_transform(X_processed)
                        elif preprocessing.startswith('snv_deriv1_w'):
                            window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                            X_snv = SNV().fit_transform(X_processed)
                            X_processed = SavgolDerivative(deriv=1, window=window).fit_transform(X_snv)
                        elif preprocessing.startswith('snv_deriv2_w'):
                            window = int(preprocessing.split('w')[1]) if 'w' in preprocessing else 7
                            X_snv = SNV().fit_transform(X_processed)
                            X_processed = SavgolDerivative(deriv=2, window=window).fit_transform(X_snv)

                    # Compute AD status
                    ad_results = model_io.compute_applicability_status(model_bundle, X_processed)
                    all_ad_status[model_key] = ad_results['status']
                    has_any_ad = True
                else:
                    all_ad_status[model_key] = np.array(['N/A'] * len(self._predict_spectra))

            # Compute consensus if multiple regression models
            regression_models = []
            for i, model_bundle in enumerate(self._loaded_models):
                if model_bundle.get('task_type', 'regression') == 'regression':
                    model_name = model_bundle.get('model_name', f'Model{i+1}')
                    target_name = model_bundle.get('target_name', 'Prediction')
                    model_key = f"{model_name}_{target_name}"
                    regression_models.append(model_key)

            consensus_predictions = None
            if len(regression_models) >= 2:
                # Compute consensus as simple average
                pred_arrays = [all_predictions[key] for key in regression_models]
                consensus_predictions = np.mean(pred_arrays, axis=0)

            # Store all results
            self._predictions = {
                'individual': all_predictions,
                'consensus': consensus_predictions,
                'ad_status': all_ad_status if has_any_ad else None
            }

            # Build status message
            status_parts = [f"Predictions complete with {len(self._loaded_models)} model(s)"]
            if interpolation_used:
                status_parts.append("wavelengths interpolated")
            if consensus_predictions is not None:
                status_parts.append(f"consensus computed from {len(regression_models)} models")

            dpg.set_value("predict_status", " - ".join(status_parts))

            # Populate results table
            self._populate_prediction_results()

            # Store predictions for export
            individual_preds = self._predictions.get('individual', {})
            if individual_preds:
                first_key = sorted(individual_preds.keys())[0]
                self._last_predictions = {
                    'predictions': individual_preds[first_key],
                    'sample_ids': self._predict_sample_ids,
                    'all_predictions': individual_preds,
                    'consensus': self._predictions.get('consensus')
                }

            # Enable export
            dpg.configure_item("predict_export_btn", enabled=True)
            dpg.configure_item("menu_export_predictions", enabled=True)
            dpg.configure_item("predict_run_btn", enabled=True)

        except Exception as e:
            dpg.set_value("predict_status", f"Error during prediction: {str(e)}")
            dpg.configure_item("predict_run_btn", enabled=True)
            print(f"Error in _on_run_prediction: {e}")
            import traceback
            traceback.print_exc()

    def _populate_prediction_results(self):
        """Populate the prediction results table with multi-model support."""
        if self._predictions is None:
            return

        import numpy as np

        # Extract prediction data
        individual_preds = self._predictions.get('individual', {})
        consensus_preds = self._predictions.get('consensus', None)
        ad_status = self._predictions.get('ad_status', None)

        if len(individual_preds) == 0:
            return

        # Clear existing table and rebuild columns
        if dpg.does_item_exist("predict_results_table"):
            dpg.delete_item("predict_results_table")

        # Create new table with dynamic columns
        with dpg.table(
            tag="predict_results_table",
            parent="predict_results_parent",
            header_row=True,
            borders_innerH=True,
            borders_innerV=True,
            borders_outerH=True,
            borders_outerV=True,
            resizable=True,
            scrollY=True,
            height=-1,
            show=True
        ):
            # Sample ID column
            dpg.add_table_column(label="Sample ID", width_fixed=True, init_width_or_weight=150)

            # Columns for each model
            model_keys = sorted(individual_preds.keys())
            for model_key in model_keys:
                dpg.add_table_column(label=model_key, width_fixed=True, init_width_or_weight=120)

            # Consensus column if available
            if consensus_preds is not None:
                dpg.add_table_column(label="Consensus", width_fixed=True, init_width_or_weight=120)

            # AD Status column if available
            if ad_status is not None:
                dpg.add_table_column(label="AD Status", width_fixed=True, init_width_or_weight=120)

        # Populate rows
        sample_ids = self._predict_sample_ids if self._predict_sample_ids is not None else [f"Sample_{i+1}" for i in range(len(next(iter(individual_preds.values()))))]

        for i, sample_id in enumerate(sample_ids):
            with dpg.table_row(parent="predict_results_table"):
                # Sample ID
                dpg.add_text(str(sample_id))

                # Individual model predictions
                for model_key in model_keys:
                    pred_val = individual_preds[model_key][i]
                    if isinstance(pred_val, (int, np.integer)):
                        dpg.add_text(str(pred_val))
                    elif isinstance(pred_val, (float, np.floating)):
                        dpg.add_text(f"{pred_val:.4f}")
                    else:
                        dpg.add_text(str(pred_val))

                # Consensus prediction
                if consensus_preds is not None:
                    cons_val = consensus_preds[i]
                    dpg.add_text(f"{cons_val:.4f}")

                # AD Status with color coding
                if ad_status is not None:
                    # Get AD status from first model that has it
                    status_text = 'N/A'
                    status_color = COLORS["text_muted"]

                    for model_key in model_keys:
                        if model_key in ad_status:
                            status_text = ad_status[model_key][i]
                            if status_text == 'good':
                                status_color = (0, 200, 0)  # Green
                            elif status_text == 'caution':
                                status_color = (200, 200, 0)  # Yellow
                            elif status_text == 'extrapolation':
                                status_color = (200, 0, 0)  # Red
                            break

                    dpg.add_text(status_text, color=status_color)

        # Hide placeholder
        dpg.configure_item("predict_results_placeholder", show=False)

    def _on_export_predictions(self, sender=None, app_data=None):
        """Handle Export button - save predictions to CSV."""
        if self._predictions is None:
            dpg.set_value("predict_status", "Error: No predictions to export")
            return

        self.show_save_dialog(self._export_predictions_to_file, dialog_type="csv", initial_dir=get_desktop_path())

    def _export_predictions_to_file(self, filepath: str):
        """Export multi-model predictions to CSV file."""
        import pandas as pd

        try:
            # Extract prediction data
            individual_preds = self._predictions.get('individual', {})
            consensus_preds = self._predictions.get('consensus', None)
            ad_status = self._predictions.get('ad_status', None)

            # Get sample IDs
            sample_ids = self._predict_sample_ids if self._predict_sample_ids is not None else [f"Sample_{i+1}" for i in range(len(next(iter(individual_preds.values()))))]

            # Build DataFrame
            data = {'Sample_ID': sample_ids}

            # Add individual model predictions
            model_keys = sorted(individual_preds.keys())
            for model_key in model_keys:
                data[model_key] = individual_preds[model_key]

            # Add consensus if available
            if consensus_preds is not None:
                data['Consensus'] = consensus_preds

            # Add AD status if available
            if ad_status is not None:
                # Use AD status from first model that has it
                for model_key in model_keys:
                    if model_key in ad_status:
                        data['AD_Status'] = ad_status[model_key]
                        break

            df = pd.DataFrame(data)

            # Save to CSV
            df.to_csv(filepath, index=False)

            dpg.set_value("predict_status", f"Predictions exported to: {Path(filepath).name}")

        except Exception as e:
            dpg.set_value("predict_status", f"Error exporting predictions: {str(e)}")
            print(f"Error in _export_predictions_to_file: {e}")
            import traceback
            traceback.print_exc()

    # --- Export Functions (File Menu) ---

    def _on_export_results_csv(self, sender=None, app_data=None):
        """Export model search results to CSV."""
        if self._search_results is None:
            return

        def save_callback(filepath):
            try:
                success = export_utils.export_results_to_csv(
                    self._search_results,
                    filepath,
                    include_index=False
                )
                if success:
                    dpg.set_value("build_progress_text", f"Results exported to: {Path(filepath).name}")
            except Exception as e:
                dpg.set_value("build_progress_text", f"Export error: {str(e)}")

        self.show_save_dialog(save_callback, dialog_type="csv", initial_dir=get_desktop_path())

    def _on_export_results_excel(self, sender=None, app_data=None):
        """Export model search results to Excel."""
        if self._search_results is None:
            return

        def save_callback(filepath):
            try:
                success = export_utils.export_results_to_excel(
                    self._search_results,
                    filepath,
                    sheet_name='Model Results',
                    include_index=False
                )
                if success:
                    dpg.set_value("build_progress_text", f"Results exported to: {Path(filepath).name}")
            except Exception as e:
                dpg.set_value("build_progress_text", f"Export error: {str(e)}")

        self.show_save_dialog(save_callback, dialog_type="xlsx", initial_dir=get_desktop_path())

    def _on_export_predictions_v2(self, sender=None, app_data=None):
        """Export predictions from Predict panel to CSV (alternate version)."""
        if not hasattr(self, '_last_predictions') or self._last_predictions is None:
            dpg.set_value("predict_status", "No predictions to export")
            return

        def save_callback(filepath):
            try:
                y_pred = self._last_predictions.get('predictions', np.array([]))
                y_true = self._last_predictions.get('actual', None)
                sample_names = self._last_predictions.get('sample_ids', None)

                if y_true is not None:
                    success = export_utils.export_predictions_to_csv(
                        y_true, y_pred, filepath,
                        sample_names=sample_names,
                        include_residuals=True
                    )
                else:
                    # Just predictions, no actual values
                    df = pd.DataFrame({
                        'Sample': sample_names if sample_names else [f'S{i+1}' for i in range(len(y_pred))],
                        'Predicted': y_pred
                    })
                    df.to_csv(filepath, index=False)
                    success = True

                if success:
                    dpg.set_value("predict_status", f"Predictions exported to: {Path(filepath).name}")
            except Exception as e:
                dpg.set_value("predict_status", f"Export error: {str(e)}")

        self.show_save_dialog(save_callback, dialog_type="csv", initial_dir=get_desktop_path())

    def _on_export_preprocessed(self, sender=None, app_data=None):
        """Export current preprocessed dataset to CSV."""
        if self._current_dataset is None:
            return

        ds = self._current_dataset

        def save_callback(filepath):
            try:
                success = export_utils.export_preprocessed_data_to_csv(
                    ds.X,
                    ds.wavelengths if ds.wavelengths is not None else np.arange(ds.X.shape[1]),
                    filepath,
                    sample_names=ds.sample_ids if hasattr(ds, 'sample_ids') else None,
                    y=ds.y if ds.has_target else None,
                    y_name=ds.target_name or 'Target'
                )
                if success:
                    dpg.set_value("build_progress_text", f"Data exported to: {Path(filepath).name}")
            except Exception as e:
                dpg.set_value("build_progress_text", f"Export error: {str(e)}")

        self.show_save_dialog(save_callback, dialog_type="csv", initial_dir=get_desktop_path())

    def _setup_tooltips(self):
        """Set up tooltips for all UI elements."""
        # === BUILD MODE & BASIC SETTINGS ===
        add_tooltip("build_mode_radio",
            "Manual: Build single model with specific settings.\n"
            "Auto: Automated search with preprocessing combos, variable selection, and hyperparameter optimization.")
        add_tooltip("build_task_type", TOOLTIP_CONTENT['ui']['task_type_regression'] + "\n" + TOOLTIP_CONTENT['ui']['task_type_classification'])
        add_tooltip("build_cv_folds", TOOLTIP_CONTENT['ui']['cv_folds'])
        add_tooltip("build_target_variable", TOOLTIP_CONTENT['ui']['target_variable'])

        # === MODEL TIER ===
        add_tooltip("build_tier",
            "Quick: PLS only (fastest).\n"
            "Standard: PLS, Ridge, ElasticNet, RandomForest, LightGBM.\n"
            "Comprehensive: All available models.\n"
            "Custom: Select specific models.")

        # === SEARCH MODE ===
        add_tooltip("build_search_mode",
            TOOLTIP_CONTENT['search_modes']['grid_search'] + "\n\n" +
            TOOLTIP_CONTENT['search_modes']['bayesian'] + "\n\n" +
            TOOLTIP_CONTENT['search_modes']['nsga2'])

        # === BAYESIAN OPTIONS ===
        add_tooltip("build_bayesian_trials",
            "Number of Optuna optimization trials. More trials = better optimization but longer runtime. "
            "20-50 for quick search, 100-200 for thorough optimization.")

        # === NSGA-II OPTIONS ===
        add_tooltip("build_nsga2_population", TOOLTIP_CONTENT['nsga2']['population_size'])
        add_tooltip("build_nsga2_generations", TOOLTIP_CONTENT['nsga2']['generations'])
        add_tooltip("build_nsga2_min_wavelengths", TOOLTIP_CONTENT['nsga2']['min_wavelengths'])

        # === CUSTOM MODEL SELECTION (Regression) ===
        add_tooltip("custom_model_pls", TOOLTIP_CONTENT['models']['PLS'])
        add_tooltip("custom_model_ridge", TOOLTIP_CONTENT['models']['Ridge'])
        add_tooltip("custom_model_lasso", TOOLTIP_CONTENT['models']['Lasso'])
        add_tooltip("custom_model_elasticnet", TOOLTIP_CONTENT['models']['ElasticNet'])
        add_tooltip("custom_model_rf", TOOLTIP_CONTENT['models']['RandomForest'])
        add_tooltip("custom_model_lgbm", TOOLTIP_CONTENT['models']['LightGBM'])
        add_tooltip("custom_model_xgb", TOOLTIP_CONTENT['models']['XGBoost'])
        add_tooltip("custom_model_catboost", TOOLTIP_CONTENT['models']['CatBoost'])
        add_tooltip("custom_model_svr", TOOLTIP_CONTENT['models']['SVR'])
        add_tooltip("custom_model_neuralboosted", TOOLTIP_CONTENT['models']['NeuralBoosted'])

        # === CUSTOM MODEL SELECTION (Classification) ===
        add_tooltip("custom_model_plsda", TOOLTIP_CONTENT['models']['PLS-DA'])
        add_tooltip("custom_model_rf_cls", TOOLTIP_CONTENT['models']['RandomForest'])
        add_tooltip("custom_model_lgbm_cls", TOOLTIP_CONTENT['models']['LightGBM'])
        add_tooltip("custom_model_xgb_cls", TOOLTIP_CONTENT['models']['XGBoost'])
        add_tooltip("custom_model_catboost_cls", TOOLTIP_CONTENT['models']['CatBoost'])
        add_tooltip("custom_model_svm", TOOLTIP_CONTENT['models']['SVM'])
        add_tooltip("custom_model_neuralboosted_cls", TOOLTIP_CONTENT['models']['NeuralBoosted'])

        # === DATA CONVERSION ===
        add_tooltip("build_convert_absorbance",
            "Convert reflectance data to absorbance using log10(1/R). "
            "Enable if your spectrometer outputs reflectance and you want to work in absorbance space.")

        # === PREPROCESSING METHODS ===
        add_tooltip("build_preproc_raw", TOOLTIP_CONTENT['preprocessing']['Raw'])
        add_tooltip("build_preproc_snv", TOOLTIP_CONTENT['preprocessing']['SNV'])
        add_tooltip("build_preproc_deriv1", TOOLTIP_CONTENT['preprocessing']['SG1'])
        add_tooltip("build_preproc_deriv2", TOOLTIP_CONTENT['preprocessing']['SG2'])
        add_tooltip("build_preproc_deriv3", TOOLTIP_CONTENT['preprocessing']['SG3'])
        add_tooltip("build_preproc_deriv4", TOOLTIP_CONTENT['preprocessing']['SG4'])
        add_tooltip("build_preproc_snv_deriv1", TOOLTIP_CONTENT['preprocessing']['SNV'] + "\nThen applies 1st derivative.")
        add_tooltip("build_preproc_snv_deriv2", TOOLTIP_CONTENT['preprocessing']['SNV'] + "\nThen applies 2nd derivative.")
        add_tooltip("build_preproc_snv_deriv3", TOOLTIP_CONTENT['preprocessing']['SNV'] + "\nThen applies 3rd derivative.")
        add_tooltip("build_preproc_snv_deriv4", TOOLTIP_CONTENT['preprocessing']['SNV'] + "\nThen applies 4th derivative.")
        add_tooltip("build_preproc_deriv1_snv", TOOLTIP_CONTENT['preprocessing']['deriv_snv'])
        add_tooltip("build_preproc_deriv2_snv", "2nd derivative then SNV.\n" + TOOLTIP_CONTENT['preprocessing']['deriv_snv'])
        add_tooltip("build_preproc_deriv3_snv", "3rd derivative then SNV.\n" + TOOLTIP_CONTENT['preprocessing']['deriv_snv'])
        add_tooltip("build_preproc_deriv4_snv", "4th derivative then SNV.\n" + TOOLTIP_CONTENT['preprocessing']['deriv_snv'])

        # === ADVANCED PREPROCESSING ===
        add_tooltip("build_enable_msc", TOOLTIP_CONTENT['advanced_preprocessing']['msc'])
        add_tooltip("build_msc_reference", "Reference spectrum for MSC. Mean uses average spectrum, Median is more robust to outliers.")

        # === GA PREPROCESSING ===
        add_tooltip("build_ga_preprocess", TOOLTIP_CONTENT['ga_preprocessing']['enable'])
        add_tooltip("build_ga_preprocess_population", TOOLTIP_CONTENT['ga_preprocessing']['population'])
        add_tooltip("build_ga_preprocess_generations", TOOLTIP_CONTENT['ga_preprocessing']['generations'])
        add_tooltip("build_ga_preprocess_cv_folds", TOOLTIP_CONTENT['ga_preprocessing']['cv_folds'])

        # === WAVELENGTH RANGE ===
        add_tooltip("build_enable_wl_range", TOOLTIP_CONTENT['wavelength_range']['enable'])
        add_tooltip("build_wl_range_min", TOOLTIP_CONTENT['wavelength_range']['from_nm'])
        add_tooltip("build_wl_range_max", TOOLTIP_CONTENT['wavelength_range']['to_nm'])

        # === INTERFERENCE REMOVAL ===
        add_tooltip("build_wavelength_exclude_moisture",
            "Exclude common water absorption bands (1400-1500 nm and 1900-2000 nm) that are often noisy in NIR spectroscopy.")
        add_tooltip("build_wavelength_exclude_custom",
            "Additional wavelength ranges to exclude. Format: 'start1-end1, start2-end2' (e.g., '1350-1450, 1850-1950').")
        add_tooltip("build_enable_osc", TOOLTIP_CONTENT['advanced_preprocessing']['osc'])
        add_tooltip("build_osc_n_components",
            "Number of orthogonal components to remove. 1-2 usually sufficient. More components may remove useful signal.")
        add_tooltip("build_enable_epo", TOOLTIP_CONTENT['advanced_preprocessing']['epo'])
        add_tooltip("build_enable_glsw", TOOLTIP_CONTENT['advanced_preprocessing']['glsw'])
        add_tooltip("build_glsw_method",
            "Covariance: Estimates wavelength noise from spectral covariance.\n"
            "Residual: Uses model residuals to estimate wavelength reliability.")

        # === BASELINE CORRECTION ===
        add_tooltip("build_baseline_method",
            "None: No baseline correction.\n"
            "Polynomial: Fit polynomial baseline (good for Raman).\n"
            "AsLS: Asymmetric Least Squares (good for fluorescence).\n"
            "airPLS: Adaptive iteratively reweighted penalized LS.")
        add_tooltip("build_baseline_poly_degree", "Polynomial degree. 1=linear, 2=quadratic, 3=cubic. Higher degrees fit more complex baselines.")
        add_tooltip("build_baseline_poly_segments", "Number of segments for piecewise polynomial fitting. More segments = more flexible baseline.")
        add_tooltip("build_baseline_lambda", "Smoothness parameter. Higher values = smoother baseline.")
        add_tooltip("build_baseline_p", "Asymmetry parameter. Lower values make baseline fit tighter under peaks. 0.001-0.01 typical.")

        # === SMOOTHING ===
        add_tooltip("build_enable_smoothing", TOOLTIP_CONTENT['advanced_preprocessing']['smoothing_sg'])
        add_tooltip("build_smooth_window", "Savitzky-Golay window length (must be odd). Larger = more smoothing.")
        add_tooltip("build_smooth_polyorder", "Polynomial order for smoothing. Lower = more smoothing but may distort peaks.")

        # === SG WINDOW SIZES ===
        add_tooltip("build_window_primary", TOOLTIP_CONTENT['preprocessing']['window_size'])
        add_tooltip("build_window_secondary", "Secondary window size for two-step derivative (0 = disabled). Uncommon - used for very specific applications.")

        # === VARIABLE SELECTION METHODS ===
        add_tooltip("build_varsel_importance", TOOLTIP_CONTENT['variable_selection']['feature_importance'])
        add_tooltip("build_varsel_vip", TOOLTIP_CONTENT['variable_selection']['vip'])
        add_tooltip("build_varsel_spa", TOOLTIP_CONTENT['variable_selection']['spa'])
        add_tooltip("build_varsel_uve", TOOLTIP_CONTENT['variable_selection']['uve'])
        add_tooltip("build_varsel_uve_spa", TOOLTIP_CONTENT['variable_selection']['uve_spa'])
        add_tooltip("build_varsel_ga_pls", TOOLTIP_CONTENT['variable_selection']['ga_pls'])

        # === SPA PARAMETERS ===
        add_tooltip("build_spa_n_features", "Number of wavelengths to select with SPA. Typical range: 20-100.")
        add_tooltip("build_spa_n_random_starts", "Number of random starting points for SPA. More starts = better exploration but slower.")

        # === UVE PARAMETERS ===
        add_tooltip("build_uve_cutoff", "Cutoff multiplier for UVE reliability threshold. Higher = more aggressive elimination.")

        # === UVE-SPA PARAMETERS ===
        add_tooltip("build_uve_spa_cutoff", "Cutoff multiplier for UVE step in UVE-SPA.")
        add_tooltip("build_uve_spa_n_features", "Number of wavelengths to select in SPA step after UVE filtering.")

        # === GA-PLS PARAMETERS ===
        add_tooltip("build_ga_pls_population", "GA population size per run. 32-64 typical.")
        add_tooltip("build_ga_pls_generations", "GA generations per run. 50-100 typical.")
        add_tooltip("build_ga_pls_n_runs", "Number of independent GA runs. More runs = more robust consensus selection.")
        add_tooltip("build_ga_pls_threshold", "Selection threshold across runs. 0.6 = wavelength must be selected in 60% of runs.")

        # === VARIABLE COUNTS ===
        add_tooltip("build_var_counts", "Variable counts to test. Comma-separated list of integers. More counts = more models to compare.")

        # === REGION ANALYSIS ===
        add_tooltip("build_enable_regions", TOOLTIP_CONTENT['variable_selection']['region_analysis'])
        add_tooltip("build_n_regions", "Number of top spectral regions to identify from selected wavelengths.")

        # === iPLS ===
        add_tooltip("build_enable_ipls", TOOLTIP_CONTENT['variable_selection']['ipls'])
        add_tooltip("build_ipls_n_intervals", "Number of spectral intervals to divide the spectrum into. 20-40 typical.")
        add_tooltip("build_ipls_forward", "Forward iPLS: Start with best interval, add intervals that improve model.")
        add_tooltip("build_ipls_backward", "Backward iPLS: Start with all intervals, remove intervals that don't hurt model.")

        # === HYPERPARAMETER INPUTS ===
        # PLS
        add_tooltip("hyperparam_pls_max_lv", TOOLTIP_CONTENT['hyperparameters']['pls_max_n_components'])
        add_tooltip("hyperparam_pls_scale", TOOLTIP_CONTENT['hyperparameters']['pls_scale'])
        add_tooltip("hyperparam_pls_maxiter", TOOLTIP_CONTENT['hyperparameters']['pls_max_iter'])
        add_tooltip("hyperparam_pls_tol", TOOLTIP_CONTENT['hyperparameters']['pls_tol'])

        # Ridge
        add_tooltip("hyperparam_ridge_alpha", TOOLTIP_CONTENT['hyperparameters']['ridge_alpha'])

        # Lasso
        add_tooltip("hyperparam_lasso_alpha", TOOLTIP_CONTENT['hyperparameters']['lasso_alpha'])

        # ElasticNet
        add_tooltip("hyperparam_elasticnet_alpha", TOOLTIP_CONTENT['hyperparameters']['elasticnet_alpha'])
        add_tooltip("hyperparam_elasticnet_l1", TOOLTIP_CONTENT['hyperparameters']['elasticnet_l1_ratio'])

        # RandomForest
        add_tooltip("hyperparam_rf_estimators", TOOLTIP_CONTENT['hyperparameters']['rf_n_estimators'])
        add_tooltip("hyperparam_rf_depth", TOOLTIP_CONTENT['hyperparameters']['rf_max_depth'])
        add_tooltip("hyperparam_rf_min_split", TOOLTIP_CONTENT['hyperparameters']['rf_min_samples_split'])
        add_tooltip("hyperparam_rf_min_leaf", TOOLTIP_CONTENT['hyperparameters']['rf_min_samples_leaf'])
        add_tooltip("hyperparam_rf_max_features", TOOLTIP_CONTENT['hyperparameters']['rf_max_features'])
        add_tooltip("hyperparam_rf_bootstrap", TOOLTIP_CONTENT['hyperparameters']['rf_bootstrap'])
        add_tooltip("hyperparam_rf_max_leaf", TOOLTIP_CONTENT['hyperparameters']['rf_max_leaf_nodes'])
        add_tooltip("hyperparam_rf_min_impurity", TOOLTIP_CONTENT['hyperparameters']['rf_min_impurity_decrease'])

        # LightGBM
        add_tooltip("hyperparam_lgbm_estimators", TOOLTIP_CONTENT['hyperparameters']['lightgbm_n_estimators'])
        add_tooltip("hyperparam_lgbm_lr", TOOLTIP_CONTENT['hyperparameters']['lightgbm_learning_rate'])
        add_tooltip("hyperparam_lgbm_leaves", TOOLTIP_CONTENT['hyperparameters']['lightgbm_num_leaves'])
        add_tooltip("hyperparam_lgbm_depth", TOOLTIP_CONTENT['hyperparameters']['lightgbm_max_depth'])
        add_tooltip("hyperparam_lgbm_min_child", TOOLTIP_CONTENT['hyperparameters']['lightgbm_min_child_samples'])
        add_tooltip("hyperparam_lgbm_subsample", TOOLTIP_CONTENT['hyperparameters']['lightgbm_subsample'])
        add_tooltip("hyperparam_lgbm_colsample", TOOLTIP_CONTENT['hyperparameters']['lightgbm_colsample_bytree'])
        add_tooltip("hyperparam_lgbm_reg_alpha", TOOLTIP_CONTENT['hyperparameters']['lightgbm_reg_alpha'])
        add_tooltip("hyperparam_lgbm_reg_lambda", TOOLTIP_CONTENT['hyperparameters']['lightgbm_reg_lambda'])

        # Register theme change callback to reset tooltip theme
        on_theme_change(lambda _: reset_tooltip_theme())

    def run(self):
        """Run the application main loop."""
        dpg.setup_dearpygui()

        # Set up tooltips after UI is built
        self._setup_tooltips()

        dpg.show_viewport()

        # Set main window to fill viewport
        dpg.set_primary_window("main_window", True)

        # Set up file drop callback
        # dpg.set_viewport_file_callback(self._on_file_drop)

        # Load pending project if specified via command line
        if self._pending_project_load:
            # Use dpg.split_frame to defer loading until after first render
            def load_pending():
                self._load_project_file(self._pending_project_load)
                self._pending_project_load = None
            dpg.split_frame(callback=load_pending)

        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    """Entry point for the application."""
    app = SpectralPredictApp()
    app.run()


if __name__ == "__main__":
    main()
