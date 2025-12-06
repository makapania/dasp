"""
Main application window for Spectral Predict v3.

Built on Dear PyGui for GPU-accelerated performance.
"""

import dearpygui.dearpygui as dpg
import os
import threading
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any

from .theme import create_theme, COLORS
from .components import show_column_config, DataGrid, SpectraPlot, PCAPlot, show_group_assign
from ..core import Engine
from ..core import io_utils
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


def open_file_dialog_tk(callback: Callable, initial_dir: Path = None, title: str = "Select File"):
    """
    Open native file dialog using tkinter (works reliably on Windows).
    Runs in separate thread to not block DPG.
    """
    def _run_dialog():
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring to front

        filetypes = [
            ("All Spectral Files", "*.csv *.xlsx *.xls *.asd *.sig *.spc"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("ASD files", "*.asd *.sig"),
            ("SPC files", "*.spc"),
            ("All files", "*.*"),
        ]

        path = filedialog.askopenfilename(
            title=title,
            initialdir=str(initial_dir or get_desktop_path()),
            filetypes=filetypes
        )

        root.destroy()

        if path:
            callback(path)

    # Run in thread to not block
    thread = threading.Thread(target=_run_dialog, daemon=True)
    thread.start()


def open_folder_dialog_tk(callback: Callable, initial_dir: Path = None, title: str = "Select Folder"):
    """
    Open native folder dialog using tkinter (works reliably on Windows).
    Runs in separate thread to not block DPG.
    """
    def _run_dialog():
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring to front

        path = filedialog.askdirectory(
            title=title,
            initialdir=str(initial_dir or get_desktop_path())
        )

        root.destroy()

        if path:
            callback(path)

    # Run in thread to not block
    thread = threading.Thread(target=_run_dialog, daemon=True)
    thread.start()


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

        # Create the main window
        self._create_main_window()

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
                        label="Open File...",
                        callback=self._on_open_file
                    )
                    dpg.add_menu_item(
                        label="Open Folder...",
                        callback=self._on_open_folder
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
        """Create the Import panel content."""
        with dpg.child_window(height=-30, border=False):
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
                    width=100
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
                    width=100
                )
                dpg.add_button(
                    label="Delete Column",
                    callback=self._on_delete_column,
                    tag="delete_column_btn",
                    enabled=False,
                    width=100
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

    def _create_explore_panel(self):
        """Create the Explore panel content with spectra and PCA visualization."""
        with dpg.child_window(height=-30, border=False, tag="explore_panel_container"):
            # Header with layout control
            with dpg.group(horizontal=True):
                dpg.add_text("Data Exploration", color=COLORS["text_secondary"])
                dpg.add_spacer(width=30)
                with dpg.group(horizontal=True, tag="shared_bins_group"):
                    dpg.add_text("Color bins:", color=COLORS["text_muted"])
                    dpg.add_combo(
                        items=["1", "2", "3", "4", "5", "6", "7", "8"],
                        default_value="4",
                        callback=self._on_bins_change,
                        tag="shared_bins_combo",
                        width=50
                    )
                dpg.add_spacer(width=30)
                dpg.add_text("PCA width:", color=COLORS["text_muted"])
                dpg.add_slider_int(
                    default_value=500,
                    min_value=300,
                    max_value=800,
                    callback=self._on_pca_width_change,
                    tag="pca_width_slider",
                    width=150
                )
                dpg.add_spacer(width=30)
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

            # Two plots side by side
            with dpg.group(horizontal=True):
                # Spectra plot (left, takes remaining width)
                with dpg.child_window(width=-500, height=-1, border=True, tag="spectra_container"):
                    self._spectra_plot = SpectraPlot(
                        parent="spectra_container",
                        tag="main_spectra_plot",
                        on_select=self._on_spectra_select
                    )

                # PCA plot (right, adjustable width)
                with dpg.child_window(width=-1, height=-1, border=True, tag="pca_container"):
                    self._pca_plot = PCAPlot(
                        parent="pca_container",
                        tag="main_pca_plot",
                        on_select=self._on_pca_select
                    )

    def _on_pca_width_change(self, sender, app_data):
        """Handle PCA panel width change."""
        new_width = app_data
        # Update spectra container to leave room for PCA
        dpg.configure_item("spectra_container", width=-new_width)

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

    def _create_build_panel(self):
        """Create the Build panel content with Manual/Auto mode and model training configuration."""
        self._search_running = False
        self._search_thread = None
        self._search_results = None
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

                        # Preprocessing Options (user-selectable, not tied to tier)
                        dpg.add_text("Preprocessing Methods", color=COLORS["text_muted"])
                        dpg.add_spacer(height=5)

                        dpg.add_checkbox(label="Raw (no preprocessing)", tag="build_preproc_raw", default_value=False)
                        dpg.add_checkbox(label="SNV", tag="build_preproc_snv", default_value=True)
                        dpg.add_checkbox(label="1st Derivative", tag="build_preproc_deriv1", default_value=True)
                        dpg.add_checkbox(label="2nd Derivative", tag="build_preproc_deriv2", default_value=True)
                        dpg.add_checkbox(label="SNV + 1st Derivative", tag="build_preproc_snv_deriv1", default_value=True)
                        dpg.add_checkbox(label="SNV + 2nd Derivative", tag="build_preproc_snv_deriv2", default_value=True)
                        dpg.add_checkbox(label="1st Derivative + SNV", tag="build_preproc_deriv1_snv", default_value=True)
                        dpg.add_checkbox(label="2nd Derivative + SNV", tag="build_preproc_deriv2_snv", default_value=True)

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

                        # Interference Removal Options (Sprint 3)
                        with dpg.collapsing_header(label="Interference Removal (Advanced)", default_open=False):
                            dpg.add_spacer(height=5)

                            # Wavelength Exclusion
                            dpg.add_checkbox(
                                label="Enable Wavelength Exclusion",
                                tag="build_enable_wavelength_exclude",
                                default_value=False
                            )
                            dpg.add_text(
                                "Exclude noisy spectral regions (e.g., moisture bands)",
                                color=COLORS["text_muted"],
                                wrap=350
                            )
                            dpg.add_spacer(height=5)
                            dpg.add_checkbox(
                                label="Exclude standard moisture bands (1400-1500, 1900-2000 nm)",
                                tag="build_wavelength_exclude_moisture",
                                default_value=True
                            )
                            dpg.add_text("Custom exclusion ranges (nm, e.g., 1400-1500,2300-2400):", color=COLORS["text_muted"])
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

                        # Advanced Options (collapsible)
                        with dpg.collapsing_header(label="Advanced Options", default_open=False):
                            dpg.add_spacer(height=5)

                            # SG Window sizes
                            dpg.add_text("SG Window Sizes:", color=COLORS["text_muted"])
                            with dpg.group(horizontal=True):
                                dpg.add_input_int(
                                    default_value=17,
                                    min_value=5,
                                    max_value=51,
                                    tag="build_window_primary",
                                    width=80,
                                    step=2,
                                    callback=self._on_window_size_change
                                )
                                dpg.add_spacer(width=20)
                                dpg.add_input_int(
                                    default_value=0,
                                    min_value=0,
                                    max_value=51,
                                    tag="build_window_secondary",
                                    width=80,
                                    step=2,
                                    callback=self._on_window_size_change
                                )
                            dpg.add_text("(Second field optional, 0=unused; must be odd)", color=COLORS["text_muted"])

                            dpg.add_spacer(height=10)

                            # Variable Selection
                            dpg.add_text("Variable Selection Methods:", color=COLORS["text_muted"])
                            dpg.add_checkbox(label="Feature Importance (RF)", tag="build_varsel_importance", default_value=True)
                            dpg.add_checkbox(label="VIP (PLS-based)", tag="build_varsel_vip", default_value=False)
                            dpg.add_checkbox(label="SPA (Successive Projections)", tag="build_varsel_spa", default_value=False, callback=self._on_varsel_checkbox_change)
                            dpg.add_checkbox(label="UVE (Uninformative Variable Elim.)", tag="build_varsel_uve", default_value=False, callback=self._on_varsel_checkbox_change)
                            dpg.add_checkbox(label="UVE-SPA (Hybrid)", tag="build_varsel_uve_spa", default_value=False, callback=self._on_varsel_checkbox_change)
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

                    # Results table (hidden initially)
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
                        height=-300,
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

    def _switch_panel(self, panel_name: str):
        """Switch to a specific panel tab."""
        tab_map = {
            "import": "tab_import",
            "data": "tab_data",
            "explore": "tab_explore",
            "build": "tab_build",
            "predict": "tab_predict",
        }
        if panel_name in tab_map:
            dpg.set_value("main_tabs", tab_map[panel_name])

    def _on_open_file(self, sender=None, app_data=None):
        """Handle open file action using native Windows dialog."""
        open_file_dialog_tk(
            callback=self._load_file,
            initial_dir=get_desktop_path(),
            title="Select Spectral File"
        )

    def _on_open_folder(self, sender=None, app_data=None):
        """Handle open folder action using native Windows dialog."""
        open_folder_dialog_tk(
            callback=self._load_file,
            initial_dir=get_desktop_path(),
            title="Select Spectral Data Folder"
        )

    def _load_file(self, path: str):
        """Preview a file and show column configuration dialog."""
        import pandas as pd

        self._pending_path = path
        path_obj = Path(path)

        try:
            self._update_status(f"Analyzing {path_obj.name}...")

            # Detect format
            format_type = io_utils.detect_format(path)

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

            X = df[wl_cols].values.astype(float)
            wavelengths = np.array([float(c) for c in wl_cols])

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

            # Gather metadata columns
            metadata_columns = {}
            for col in column_info.get('metadata_columns', []):
                if col != id_column and col != target_column:
                    metadata_columns[col] = list(df[col])

            # Also include target candidates that aren't the selected target
            for col in column_info.get('target_candidates', []):
                if col != target_column and col not in metadata_columns:
                    metadata_columns[col] = list(df[col])

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

            # Use v1's proven align_xy function for matching
            import sys
            from pathlib import Path
            V1_PATH = Path(__file__).parent.parent.parent / "src"
            if str(V1_PATH) not in sys.path:
                sys.path.insert(0, str(V1_PATH))
            from spectral_predict import io as v1_io

            ref_df = self._pending_df
            column_info = config['column_info']
            id_column = config.get('id_column')
            target_column = config.get('target_column')

            if not id_column:
                raise ValueError("ID column is required to merge reference data")

            # Get the already-loaded spectral dataset
            dataset = self._current_dataset

            # Convert to DataFrame format for v1's align_xy
            X_df = pd.DataFrame(
                dataset.X,
                index=dataset.sample_ids,
                columns=dataset.wavelengths
            )

            # Index reference by ID column
            ref_df_indexed = ref_df.set_index(id_column)

            # Use v1's align_xy with fuzzy matching
            X_aligned, y_aligned, align_info = v1_io.align_xy(
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

            # Extract metadata columns from reference
            metadata_columns = {}
            for col in ref_df_indexed.columns:
                if col != target_column:
                    try:
                        # Get values for matched samples
                        matched_ids = X_aligned.index
                        values = [ref_df_indexed.loc[idx, col] if idx in ref_df_indexed.index else None
                                  for idx in matched_ids]
                        metadata_columns[col] = values
                    except:
                        pass

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
            self._update_status("No row indices specified")
            return

        # Confirm deletion
        with dpg.window(
            label="Confirm Delete",
            modal=True,
            show=True,
            tag="confirm_delete_window",
            no_resize=True,
            width=300,
            height=120,
            pos=[400, 300]
        ):
            dpg.add_text(f"Delete {len(indices)} row(s)?")
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Delete",
                    callback=lambda: self._confirm_delete_rows(indices),
                    width=140
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("confirm_delete_window"),
                    width=140
                )

    def _confirm_delete_rows(self, indices: List[int]):
        """Confirm and execute row deletion."""
        if dpg.does_item_exist("confirm_delete_window"):
            dpg.delete_item("confirm_delete_window")

        if self._data_grid:
            self._data_grid.delete_rows(indices)
            dpg.set_value("row_indices_input", "")  # Clear input
            self._update_status(f"Deleted {len(indices)} row(s)")

    def _on_duplicate_row(self, sender=None, app_data=None):
        """Handle duplicate row button click."""
        if not self._data_grid:
            return

        indices = self._parse_row_indices()
        if not indices:
            self._update_status("No row indices specified")
            return

        self._data_grid.duplicate_rows(indices)
        dpg.set_value("row_indices_input", "")  # Clear input
        self._update_status(f"Duplicated {len(indices)} row(s)")

    def _on_insert_row(self, sender=None, app_data=None):
        """Handle insert row button click."""
        if not self._data_grid:
            return

        self._data_grid.insert_row()
        self._update_status("Inserted new row at end")

    def _on_add_column(self, sender=None, app_data=None):
        """Handle add column button click."""
        if not self._data_grid:
            return

        # Create dialog for column name
        with dpg.window(
            label="Add Column",
            modal=True,
            show=True,
            tag="add_column_window",
            no_resize=True,
            width=350,
            height=140,
            pos=[400, 300]
        ):
            dpg.add_text("Column name:")
            dpg.add_spacer(height=5)
            dpg.add_input_text(
                tag="add_column_name_input",
                width=-1,
                hint="Enter column name"
            )
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Add",
                    callback=self._confirm_add_column,
                    width=160
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("add_column_window"),
                    width=160
                )

    def _confirm_add_column(self, sender=None, app_data=None):
        """Confirm and execute column addition."""
        column_name = dpg.get_value("add_column_name_input")

        if dpg.does_item_exist("add_column_window"):
            dpg.delete_item("add_column_window")

        if not column_name:
            self._update_status("Error: Column name cannot be empty")
            return

        if self._data_grid:
            self._data_grid.add_column(column_name, column_type='metadata', default_value="")
            self._update_status(f"Added column '{column_name}'")

    def _on_delete_column(self, sender=None, app_data=None):
        """Handle delete column button click."""
        if not self._data_grid or not self._current_dataset:
            return

        # Get list of metadata columns
        if not self._current_dataset.metadata_columns:
            self._update_status("No metadata columns to delete")
            return

        columns = list(self._current_dataset.metadata_columns.keys())

        # Create dialog to select column
        with dpg.window(
            label="Delete Column",
            modal=True,
            show=True,
            tag="delete_column_window",
            no_resize=True,
            width=350,
            height=140,
            pos=[400, 300]
        ):
            dpg.add_text("Select column to delete:")
            dpg.add_spacer(height=5)
            dpg.add_combo(
                items=columns,
                tag="delete_column_combo",
                width=-1,
                default_value=columns[0] if columns else ""
            )
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Delete",
                    callback=self._confirm_delete_column,
                    width=160
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("delete_column_window"),
                    width=160
                )

    def _confirm_delete_column(self, sender=None, app_data=None):
        """Confirm and execute column deletion."""
        column_name = dpg.get_value("delete_column_combo")

        if dpg.does_item_exist("delete_column_window"):
            dpg.delete_item("delete_column_window")

        if self._data_grid:
            self._data_grid.delete_column(column_name)
            self._update_status(f"Deleted column '{column_name}'")

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
                n_unique = len(np.unique(new_y[~np.isnan(new_y)]))
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
        y = self._current_dataset.y
        valid_y = y[~np.isnan(y)] if y is not None else np.array([])

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

        # Get X and y as numpy arrays
        X_full = self._current_dataset.X.copy()
        y_full = self._current_dataset.y.copy()
        wavelengths = self._current_dataset.wavelengths

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

            search_kwargs = {
                'mode': 'manual',
                'model_name': model_name,
                'preprocessing': preprocessing,
                'window_size': window_size,
                'model_params': model_params,
                'wavelengths': wavelengths,
            }

            # Update UI
            dpg.configure_item("build_run_btn", enabled=False, label="Training...")
            start_message = f"Training {model_name} with {preprocessing}..."
            self._search_preprocess = preprocessing

        else:
            # Auto mode: full automated search with user-selected options
            tier = dpg.get_value("build_tier").lower()

            # Gather user-selected preprocessing methods
            preproc_methods = []
            if dpg.get_value("build_preproc_raw"):
                preproc_methods.append('raw')
            if dpg.get_value("build_preproc_snv"):
                preproc_methods.append('snv')
            if dpg.get_value("build_preproc_deriv1"):
                preproc_methods.append('deriv1')
            if dpg.get_value("build_preproc_deriv2"):
                preproc_methods.append('deriv2')
            if dpg.get_value("build_preproc_snv_deriv1"):
                preproc_methods.append('snv_deriv1')
            if dpg.get_value("build_preproc_snv_deriv2"):
                preproc_methods.append('snv_deriv2')
            if dpg.get_value("build_preproc_deriv1_snv"):
                preproc_methods.append('deriv1_snv')
            if dpg.get_value("build_preproc_deriv2_snv"):
                preproc_methods.append('deriv2_snv')

            if not preproc_methods:
                preproc_methods = ['snv']  # Fallback default

            # Gather user-specified window sizes from input fields
            window_sizes = []
            primary_window = dpg.get_value("build_window_primary")
            if primary_window >= 5:
                window_sizes.append(primary_window)
            secondary_window = dpg.get_value("build_window_secondary")
            if secondary_window >= 5:
                window_sizes.append(secondary_window)

            if not window_sizes:
                window_sizes = [17]  # Fallback default

            # Gather variable selection methods
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
            # Note: iPLS params are now read separately via enable_ipls controls

            # PLS max latent variables
            pls_max_lv = dpg.get_value("hyperparam_pls_max_lv")

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
            }

            # Handle custom tier - pass explicit model list
            if tier == "custom":
                custom_models = self._get_custom_selected_models(task_type)
                if not custom_models:
                    dpg.set_value("build_progress_text", "Error: No models selected for custom tier")
                    return
                search_kwargs['custom_models'] = custom_models

            # Update UI
            dpg.configure_item("build_run_btn", enabled=False, label="Searching...")
            if tier == "custom":
                start_message = f"Starting custom search with {len(custom_models)} models..."
            else:
                start_message = f"Starting {tier} tier search ({len(preproc_methods)} preproc methods)..."
            self._search_preprocess = "auto"

        # Update UI state
        self._search_running = True
        dpg.configure_item("build_progress_bar", show=True)
        dpg.set_value("build_progress_bar", 0.0)
        dpg.set_value("build_progress_text", start_message)

        def progress_callback(info):
            """Update progress from background thread."""
            message = info.get('message', '')
            current = info.get('current', 0)
            total = info.get('total', 1)
            best_score = info.get('best_score')

            progress = current / max(total, 1)
            dpg.set_value("build_progress_bar", progress)

            if best_score is not None:
                dpg.set_value("build_progress_text", f"[{current}/{total}] {message} (best: {best_score:.4f})")
            else:
                dpg.set_value("build_progress_text", f"[{current}/{total}] {message}")

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
        dpg.set_value("build_progress_text", f"Complete! {n_results} model configurations tested (click column headers to sort)")

        # Hide placeholder, show table
        dpg.configure_item("build_results_placeholder", show=False)
        dpg.configure_item("build_results_table", show=True)

        # Populate the table with all results
        self._populate_results_table(results_df)

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

        # Show/enable validation test button if there's a validation set
        if self._validation_set is not None:
            dpg.configure_item("build_validation_test_btn", show=True, enabled=has_selection)

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

        # Show file save dialog
        def save_callback(filepath):
            self._save_model_to_file(selected_row, filepath)

        # Open save dialog in thread
        def _run_save_dialog():
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            # Suggest filename based on model config
            model_name = selected_row.get('Model', selected_row.get('model', 'model'))
            preproc = selected_row.get('Preprocessing', selected_row.get('preprocessing', 'raw'))
            default_name = f"{model_name}_{preproc}.pkl"

            filepath = filedialog.asksaveasfilename(
                title="Save Model",
                initialdir=str(get_desktop_path()),
                initialfile=default_name,
                defaultextension=".pkl",
                filetypes=[("Model files", "*.pkl"), ("All files", "*.*")]
            )

            root.destroy()

            if filepath:
                save_callback(filepath)

        import threading
        thread = threading.Thread(target=_run_save_dialog, daemon=True)
        thread.start()

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

            # Train model on calibration set
            model = get_model(model_name, task_type=task_type, **params)
            if model is None:
                dpg.set_value("build_save_status", f"Error: Model {model_name} not available")
                return

            model.fit(X_cal_proc, y_cal)

            # Predict on validation set
            y_pred = model.predict(X_val_proc)

            # Calculate metrics
            is_regression = task_type == 'regression'
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

            dpg.set_value("build_save_status", f"Validation complete: {len(y_val)} samples tested")

        except Exception as e:
            dpg.set_value("build_save_status", f"Validation error: {str(e)}")
            print(f"Error in _on_test_validation: {e}")
            import traceback
            traceback.print_exc()

    def _on_load_model(self, sender=None, app_data=None):
        """Handle Load Model button - open file dialog to select model."""
        def load_callback(filepath):
            self._load_model_file(filepath)

        # Open file dialog in thread
        def _run_load_dialog():
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            filepath = filedialog.askopenfilename(
                title="Load Model",
                initialdir=str(get_desktop_path()),
                filetypes=[("Model files", "*.pkl"), ("All files", "*.*")]
            )

            root.destroy()

            if filepath:
                load_callback(filepath)

        import threading
        thread = threading.Thread(target=_run_load_dialog, daemon=True)
        thread.start()

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
        def load_callback(filepath):
            self._load_predict_spectra_from_path(filepath)

        open_file_dialog_tk(
            callback=load_callback,
            initial_dir=get_desktop_path(),
            title="Select Spectra File for Prediction"
        )

    def _on_load_predict_spectra_folder_btn(self, sender=None, app_data=None):
        """Handle Load Folder button - open folder dialog to select spectra folder."""
        def load_callback(folderpath):
            self._load_predict_spectra_from_path(folderpath)

        open_folder_dialog_tk(
            callback=load_callback,
            initial_dir=get_desktop_path(),
            title="Select Folder with Spectral Files"
        )

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
                        dpg.configure_item("predict_wavelength_warning", show=True, color=COLORS["error"])
                    else:
                        warning = f"Note: Wavelengths differ from models. Will interpolate to match training range."
                        dpg.set_value("predict_wavelength_warning", warning)
                        dpg.configure_item("predict_wavelength_warning", show=True, color=COLORS["warning"])
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

            # Enable export
            dpg.configure_item("predict_export_btn", enabled=True)
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

        def export_callback(filepath):
            self._export_predictions_to_file(filepath)

        # Open save dialog in thread
        def _run_export_dialog():
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            filepath = filedialog.asksaveasfilename(
                title="Export Predictions",
                initialdir=str(get_desktop_path()),
                initialfile="predictions.csv",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            root.destroy()

            if filepath:
                export_callback(filepath)

        import threading
        thread = threading.Thread(target=_run_export_dialog, daemon=True)
        thread.start()

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

    def run(self):
        """Run the application main loop."""
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Set main window to fill viewport
        dpg.set_primary_window("main_window", True)

        # Set up file drop callback
        # dpg.set_viewport_file_callback(self._on_file_drop)

        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    """Entry point for the application."""
    app = SpectralPredictApp()
    app.run()


if __name__ == "__main__":
    main()
