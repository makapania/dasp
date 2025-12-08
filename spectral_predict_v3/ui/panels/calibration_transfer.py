"""
Calibration Transfer Panel for Spectral Predict v3.

Provides a wizard workflow for calibration transfer between instruments:
1. Load Data - Load master and slave instrument data
2. Select Transfer Samples - Choose matching samples
3. Configure Method - Select transfer method and parameters
4. Apply & Validate - Execute transfer and show metrics
"""

import dearpygui.dearpygui as dpg
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import threading

from ...core.calibration_transfer import (
    estimate_ds, apply_ds,
    estimate_pds, apply_pds,
    estimate_tsr, apply_tsr,
    estimate_ctai, apply_ctai,
    estimate_nspfce, apply_nspfce,
    estimate_jypls_inv, apply_jypls_inv,
    TransferModel,
    save_transfer_model,
    load_transfer_model,
)
from ...core.instrument_profiles import characterize_instrument
from ...core.sample_selection import kennard_stone
from ...core import io_utils
from ..theme import COLORS
from ..tooltips import add_tooltip, TOOLTIP_CONTENT


class CalibrationTransferPanel:
    """
    Wizard-style panel for calibration transfer between instruments.
    """

    def __init__(self, parent_tag: str,
                 show_open_dialog_callback=None,
                 show_folder_dialog_callback=None,
                 show_save_dialog_callback=None):
        """
        Initialize calibration transfer panel.

        Parameters
        ----------
        parent_tag : str
            DearPyGui tag of parent container
        show_open_dialog_callback : callable, optional
            Callback to show file open dialog
        show_folder_dialog_callback : callable, optional
            Callback to show folder selection dialog
        show_save_dialog_callback : callable, optional
            Callback to show save dialog
        """
        self.parent_tag = parent_tag
        self.show_open_dialog_callback = show_open_dialog_callback
        self.show_folder_dialog_callback = show_folder_dialog_callback
        self.show_save_dialog_callback = show_save_dialog_callback
        self.current_step = 0
        self.current_method = "tsr"

        # Data storage
        self.master_wavelengths = None
        self.master_spectra = None
        self.master_sample_ids = None
        self.master_instrument_id = "Master"

        self.slave_wavelengths = None
        self.slave_spectra = None
        self.slave_sample_ids = None
        self.slave_instrument_id = "Slave"

        self.transfer_indices = None
        self.y_reference = None  # For methods that need Y

        # Transfer results
        self.transfer_model = None
        self.transfer_params = None
        self.transferred_spectra = None
        self.quality_metrics = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the calibration transfer UI."""
        with dpg.child_window(parent=self.parent_tag, tag=f"{self.parent_tag}_ct_main"):
            # Header
            dpg.add_text("Calibration Transfer Wizard", tag=f"{self.parent_tag}_ct_title")
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Step indicator
            with dpg.group(horizontal=True, tag=f"{self.parent_tag}_ct_steps"):
                dpg.add_button(label="1. Load Data", tag=f"{self.parent_tag}_ct_step1_btn", callback=lambda: self._go_to_step(0))
                dpg.add_text(">", color=COLORS["text_muted"])
                dpg.add_button(label="2. Select Samples", tag=f"{self.parent_tag}_ct_step2_btn", callback=lambda: self._go_to_step(1))
                dpg.add_text(">", color=COLORS["text_muted"])
                dpg.add_button(label="3. Configure Method", tag=f"{self.parent_tag}_ct_step3_btn", callback=lambda: self._go_to_step(2))
                dpg.add_text(">", color=COLORS["text_muted"])
                dpg.add_button(label="4. Apply & Validate", tag=f"{self.parent_tag}_ct_step4_btn", callback=lambda: self._go_to_step(3))

            dpg.add_separator()
            dpg.add_spacer(height=20)

            # Content area for steps
            with dpg.child_window(tag=f"{self.parent_tag}_ct_content", border=True, height=-80):
                self._build_step_1()

            # Navigation buttons
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True, tag=f"{self.parent_tag}_ct_nav"):
                dpg.add_button(label="← Previous", tag=f"{self.parent_tag}_ct_prev", callback=self._previous_step, enabled=False)
                dpg.add_spacer(width=20)
                dpg.add_button(label="Next →", tag=f"{self.parent_tag}_ct_next", callback=self._next_step)
                dpg.add_spacer(width=50)
                dpg.add_button(label="Reset", callback=self._reset_wizard)

        # Set up tooltips for step buttons
        self._setup_step_tooltips()

    def _setup_step_tooltips(self):
        """Set up tooltips for calibration transfer wizard."""
        # Step navigation tooltips
        add_tooltip(f"{self.parent_tag}_ct_step1_btn",
            "Load spectral data from master and slave instruments. "
            "Master is the reference instrument, slave is the one to be calibrated.")
        add_tooltip(f"{self.parent_tag}_ct_step2_btn",
            "Select transfer samples - paired measurements on both instruments. "
            "These samples are used to build the transfer model.")
        add_tooltip(f"{self.parent_tag}_ct_step3_btn",
            "Choose transfer method and configure parameters. "
            "Different methods work better for different types of instrument differences.")
        add_tooltip(f"{self.parent_tag}_ct_step4_btn",
            "Apply the transfer model and evaluate quality metrics. "
            "View before/after comparison and save the transfer model.")

    def _setup_method_tooltips(self, parent: str):
        """Add tooltips to method-specific parameters (called after building params)."""
        params_tag = f"{parent}_method_params"

        # TSR parameters
        if dpg.does_item_exist(f"{params_tag}_slope_bias"):
            add_tooltip(f"{params_tag}_slope_bias",
                "Apply slope and bias correction to refine the transfer. "
                "Usually improves results, especially for instruments with different response characteristics.")
        if dpg.does_item_exist(f"{params_tag}_regularization"):
            add_tooltip(f"{params_tag}_regularization",
                "Regularization strength. Higher values create smoother, more stable transfers "
                "but may underfit. 0 = no regularization.")

        # PDS parameters
        if dpg.does_item_exist(f"{params_tag}_window"):
            add_tooltip(f"{params_tag}_window", TOOLTIP_CONTENT['calibration_transfer']['param_pds_window'])

        # DS parameters
        if dpg.does_item_exist(f"{params_tag}_lambda"):
            add_tooltip(f"{params_tag}_lambda", TOOLTIP_CONTENT['calibration_transfer']['param_ds_lambda'])

        # NS-PFCE parameters
        if dpg.does_item_exist(f"{params_tag}_use_vcpa"):
            add_tooltip(f"{params_tag}_use_vcpa", TOOLTIP_CONTENT['calibration_transfer']['param_nspfce_wavelength_selection'])
        if dpg.does_item_exist(f"{params_tag}_max_iter"):
            add_tooltip(f"{params_tag}_max_iter", TOOLTIP_CONTENT['calibration_transfer']['param_nspfce_max_iter'])

        # JYPLS parameters
        if dpg.does_item_exist(f"{params_tag}_n_components"):
            add_tooltip(f"{params_tag}_n_components", TOOLTIP_CONTENT['calibration_transfer']['param_jypls_components'])

    def _go_to_step(self, step: int):
        """Navigate to specific step."""
        self.current_step = step
        self._update_ui()

    def _next_step(self):
        """Go to next step."""
        if self.current_step < 3:
            self.current_step += 1
            self._update_ui()

    def _previous_step(self):
        """Go to previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_ui()

    def _update_ui(self):
        """Update UI based on current step."""
        # Clear content
        if dpg.does_item_exist(f"{self.parent_tag}_ct_content"):
            dpg.delete_item(f"{self.parent_tag}_ct_content", children_only=True)

        # Build step content
        if self.current_step == 0:
            self._build_step_1()
        elif self.current_step == 1:
            self._build_step_2()
        elif self.current_step == 2:
            self._build_step_3()
        elif self.current_step == 3:
            self._build_step_4()

        # Update navigation
        dpg.configure_item(f"{self.parent_tag}_ct_prev", enabled=(self.current_step > 0))
        dpg.configure_item(f"{self.parent_tag}_ct_next", enabled=(self.current_step < 3))

    def _build_step_1(self):
        """Step 1: Load Data"""
        parent = f"{self.parent_tag}_ct_content"

        with dpg.group(parent=parent):
            dpg.add_text("Step 1: Load Instrument Data")
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Master instrument
            dpg.add_text("Master Instrument (Reference)")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag=f"{parent}_master_path", width=350, hint="Master instrument file/folder path")
                dpg.add_button(label="File...", callback=lambda: self._browse_file("master"), width=60)
                dpg.add_button(label="Folder...", callback=lambda: self._browse_folder("master"), width=70)

            dpg.add_spacer(height=5)
            dpg.add_text("", tag=f"{parent}_master_status", color=(150, 150, 150))

            dpg.add_spacer(height=15)

            # Slave instrument
            dpg.add_text("Slave Instrument (To be transferred)")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag=f"{parent}_slave_path", width=350, hint="Slave instrument file/folder path")
                dpg.add_button(label="File...", callback=lambda: self._browse_file("slave"), width=60)
                dpg.add_button(label="Folder...", callback=lambda: self._browse_folder("slave"), width=70)

            dpg.add_spacer(height=5)
            dpg.add_text("", tag=f"{parent}_slave_status", color=(150, 150, 150))

            dpg.add_spacer(height=20)

            # Preview plot area
            with dpg.child_window(tag=f"{parent}_preview_area", height=250, border=True):
                dpg.add_text("Load both datasets to see preview comparison", color=(150, 150, 150))

            dpg.add_spacer(height=15)

            # Info
            with dpg.group():
                dpg.add_text("Supported formats:", color=(100, 150, 255))
                dpg.add_text("  • CSV files (rows=samples, columns=wavelengths)")
                dpg.add_text("  • Folders with ASD, SPC, or other spectral files")
                dpg.add_text("  • Sample IDs should match between instruments")

    def _build_step_2(self):
        """Step 2: Select Transfer Samples"""
        parent = f"{self.parent_tag}_ct_content"

        with dpg.group(parent=parent):
            dpg.add_text("Step 2: Select Transfer Samples")
            dpg.add_separator()
            dpg.add_spacer(height=10)

            if self.master_spectra is None or self.slave_spectra is None:
                dpg.add_text("Please load data in Step 1 first.", color=(255, 100, 100))
                return

            dpg.add_text(f"Master samples: {self.master_spectra.shape[0]}")
            dpg.add_text(f"Slave samples: {self.slave_spectra.shape[0]}")
            dpg.add_spacer(height=10)

            # Selection method
            dpg.add_text("Selection Method:")
            dpg.add_radio_button(
                items=["All Samples", "Kennard-Stone", "Random", "Manual"],
                tag=f"{parent}_selection_method",
                default_value="Kennard-Stone"
            )

            dpg.add_spacer(height=10)

            # Number of samples
            dpg.add_text("Number of transfer samples:")
            dpg.add_slider_int(
                tag=f"{parent}_n_transfer",
                default_value=min(20, self.master_spectra.shape[0]),
                min_value=5,
                max_value=self.master_spectra.shape[0],
                width=300
            )

            dpg.add_spacer(height=20)
            dpg.add_button(label="Select Samples", callback=self._select_transfer_samples, width=200)

            dpg.add_spacer(height=10)
            dpg.add_text("", tag=f"{parent}_selection_status")

    def _build_step_3(self):
        """Step 3: Configure Transfer Method"""
        parent = f"{self.parent_tag}_ct_content"

        with dpg.group(parent=parent):
            dpg.add_text("Step 3: Configure Transfer Method")
            dpg.add_separator()
            dpg.add_spacer(height=10)

            if self.transfer_indices is None:
                dpg.add_text("Please select transfer samples in Step 2 first.", color=(255, 100, 100))
                return

            dpg.add_text(f"Transfer samples selected: {len(self.transfer_indices)}")
            dpg.add_spacer(height=10)

            # Method selection
            dpg.add_text("Transfer Method:")
            dpg.add_combo(
                items=["DS (Direct Standardization)",
                       "PDS (Piecewise Direct Standardization)",
                       "TSR (Transfer Sample Regression)",
                       "CTAI (Affine Invariance)",
                       "NS-PFCE (Parameter-Free)",
                       "JYPLS-inv (Joint-Y PLS)"],
                tag=f"{parent}_method",
                default_value="TSR (Transfer Sample Regression)",
                callback=self._on_method_changed,
                width=350
            )

            # Add tooltip for method combo
            add_tooltip(f"{parent}_method",
                "DS: " + TOOLTIP_CONTENT['calibration_transfer']['method_DS'][:100] + "...\n\n"
                "PDS: " + TOOLTIP_CONTENT['calibration_transfer']['method_PDS'][:100] + "...\n\n"
                "TSR: " + TOOLTIP_CONTENT['calibration_transfer']['method_TSR'][:100] + "...\n\n"
                "CTAI: " + TOOLTIP_CONTENT['calibration_transfer']['method_CTAI'][:100] + "...\n\n"
                "NS-PFCE: " + TOOLTIP_CONTENT['calibration_transfer']['method_NSPFCE'][:100] + "...\n\n"
                "JYPLS-inv: " + TOOLTIP_CONTENT['calibration_transfer']['method_JYPLS'][:100] + "...")

            dpg.add_spacer(height=20)

            # Method-specific parameters
            with dpg.child_window(tag=f"{parent}_method_params", height=200, border=True):
                self._build_tsr_params(parent)

        # Add tooltips to method-specific parameters
        self._setup_method_tooltips(parent)

    def _build_tsr_params(self, parent: str):
        """Build TSR-specific parameters."""
        params_tag = f"{parent}_method_params"

        if dpg.does_item_exist(params_tag):
            dpg.delete_item(params_tag, children_only=True)

        with dpg.group(parent=params_tag):
            dpg.add_text("TSR Parameters:")
            dpg.add_spacer(height=10)

            dpg.add_checkbox(
                label="Slope & Bias Correction",
                tag=f"{params_tag}_slope_bias",
                default_value=True
            )

            dpg.add_text("Regularization:")
            dpg.add_slider_float(
                tag=f"{params_tag}_regularization",
                default_value=0.0,
                min_value=0.0,
                max_value=1.0,
                width=200
            )

    def _build_pds_params(self, parent: str):
        """Build PDS-specific parameters."""
        params_tag = f"{parent}_method_params"

        if dpg.does_item_exist(params_tag):
            dpg.delete_item(params_tag, children_only=True)

        with dpg.group(parent=params_tag):
            dpg.add_text("PDS Parameters:")
            dpg.add_spacer(height=10)

            dpg.add_text("Window Size:")
            dpg.add_slider_int(
                tag=f"{params_tag}_window",
                default_value=11,
                min_value=5,
                max_value=31,
                width=200
            )

    def _build_ds_params(self, parent: str):
        """Build DS-specific parameters."""
        params_tag = f"{parent}_method_params"

        if dpg.does_item_exist(params_tag):
            dpg.delete_item(params_tag, children_only=True)

        with dpg.group(parent=params_tag):
            dpg.add_text("DS (Direct Standardization) Parameters:")
            dpg.add_spacer(height=10)

            dpg.add_text("Ridge Regularization (Lambda):")
            dpg.add_slider_float(
                tag=f"{params_tag}_lambda",
                default_value=0.001,
                min_value=0.0001,
                max_value=1.0,
                format="%.4f",
                width=200
            )
            dpg.add_text("  Higher values = more regularization", color=(150, 150, 150))

    def _build_ctai_params(self, parent: str):
        """Build CTAI-specific parameters (none required)."""
        params_tag = f"{parent}_method_params"

        if dpg.does_item_exist(params_tag):
            dpg.delete_item(params_tag, children_only=True)

        with dpg.group(parent=params_tag):
            dpg.add_text("CTAI (Affine Invariance) Parameters:")
            dpg.add_spacer(height=10)
            dpg.add_text("This method has no configurable parameters.", color=(150, 150, 150))
            dpg.add_spacer(height=10)
            dpg.add_text("CTAI treats master spectra as Y and slave spectra as X,")
            dpg.add_text("then uses linear regression to find the transformation.")

    def _build_nspfce_params(self, parent: str):
        """Build NS-PFCE-specific parameters."""
        params_tag = f"{parent}_method_params"

        if dpg.does_item_exist(params_tag):
            dpg.delete_item(params_tag, children_only=True)

        with dpg.group(parent=params_tag):
            dpg.add_text("NS-PFCE (Parameter-Free) Parameters:")
            dpg.add_spacer(height=10)

            dpg.add_checkbox(
                label="Use VCPA-IRIV Wavelength Selection",
                tag=f"{params_tag}_use_vcpa",
                default_value=True
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Max Iterations:")
            dpg.add_slider_int(
                tag=f"{params_tag}_max_iter",
                default_value=100,
                min_value=10,
                max_value=500,
                width=200
            )

    def _build_jypls_params(self, parent: str):
        """Build JYPLS-inv specific parameters."""
        params_tag = f"{parent}_method_params"

        if dpg.does_item_exist(params_tag):
            dpg.delete_item(params_tag, children_only=True)

        with dpg.group(parent=params_tag):
            dpg.add_text("JYPLS-inv (Joint-Y PLS) Parameters:")
            dpg.add_spacer(height=10)

            dpg.add_text("Number of PLS Components:")
            dpg.add_slider_int(
                tag=f"{params_tag}_n_components",
                default_value=10,
                min_value=2,
                max_value=30,
                width=200
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Note: JYPLS-inv works best when Y reference values are available.", color=(255, 200, 100))
            dpg.add_text("Using sample indices as fallback if no Y values provided.", color=(150, 150, 150))

    def _build_step_4(self):
        """Step 4: Apply & Validate"""
        parent = f"{self.parent_tag}_ct_content"

        with dpg.group(parent=parent):
            dpg.add_text("Step 4: Apply Transfer & Validate")
            dpg.add_separator()
            dpg.add_spacer(height=10)

            dpg.add_button(label="Run Calibration Transfer", callback=self._run_transfer, width=250)

            dpg.add_spacer(height=20)

            # Results area
            with dpg.child_window(tag=f"{parent}_results", border=True, height=-50):
                dpg.add_text("Results will appear here after running transfer.")

            dpg.add_spacer(height=10)
            dpg.add_button(label="Save Transfer Model...", callback=self._save_model, width=200)

    def _browse_file(self, instrument_type: str):
        """Browse for instrument data file or folder."""
        parent = f"{self.parent_tag}_ct_content"

        def on_file_selected(filepath):
            if filepath:
                # Update path input field
                dpg.set_value(f"{parent}_{instrument_type}_path", filepath)
                # Load the file
                self._load_instrument_file(instrument_type, filepath)

        # Use open dialog for files (CSV, etc.)
        if self.show_open_dialog_callback:
            self.show_open_dialog_callback(on_file_selected)

    def _browse_folder(self, instrument_type: str):
        """Browse for instrument data folder (for ASD/SPC files)."""
        parent = f"{self.parent_tag}_ct_content"

        def on_folder_selected(folderpath):
            if folderpath:
                # Update path input field
                dpg.set_value(f"{parent}_{instrument_type}_path", folderpath)
                # Load the folder
                self._load_instrument_file(instrument_type, folderpath)

        if self.show_folder_dialog_callback:
            self.show_folder_dialog_callback(on_folder_selected)

    def _load_instrument_file(self, instrument_type: str, filepath: str):
        """Load spectral data from file or folder."""
        parent = f"{self.parent_tag}_ct_content"
        status_tag = f"{parent}_{instrument_type}_status"

        try:
            # Use io_utils to load spectra (handles files and folders)
            X, wavelengths, sample_ids = io_utils.load_spectra(filepath)

            # Store the data
            if instrument_type == "master":
                self.master_spectra = X
                self.master_wavelengths = wavelengths
                self.master_sample_ids = sample_ids
                self.master_instrument_id = Path(filepath).stem
            else:
                self.slave_spectra = X
                self.slave_wavelengths = wavelengths
                self.slave_sample_ids = sample_ids
                self.slave_instrument_id = Path(filepath).stem

            # Update status
            status_msg = f"Loaded: {X.shape[0]} samples, {X.shape[1]} wavelengths ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm)"
            dpg.set_value(status_tag, status_msg)
            dpg.configure_item(status_tag, color=(100, 255, 100))

            # Update preview plot if both datasets loaded
            self._update_preview_plot()

        except Exception as e:
            dpg.set_value(status_tag, f"Error: {str(e)}")
            dpg.configure_item(status_tag, color=(255, 100, 100))

    def _select_transfer_samples(self):
        """Select transfer samples based on chosen method."""
        parent = f"{self.parent_tag}_ct_content"

        if self.master_spectra is None:
            return

        method = dpg.get_value(f"{parent}_selection_method")
        n_transfer = dpg.get_value(f"{parent}_n_transfer")

        if method == "All Samples":
            self.transfer_indices = np.arange(self.master_spectra.shape[0])
        elif method == "Kennard-Stone":
            self.transfer_indices = kennard_stone(self.master_spectra, n_samples=n_transfer)
        elif method == "Random":
            self.transfer_indices = np.random.choice(
                self.master_spectra.shape[0],
                size=n_transfer,
                replace=False
            )
        else:  # Manual
            dpg.set_value(f"{parent}_selection_status",
                         "Manual selection not yet implemented. Using Kennard-Stone.")
            self.transfer_indices = kennard_stone(self.master_spectra, n_samples=n_transfer)

        dpg.set_value(f"{parent}_selection_status",
                     f"Selected {len(self.transfer_indices)} transfer samples using {method}")

    def _on_method_changed(self):
        """Handle transfer method change."""
        parent = f"{self.parent_tag}_ct_content"
        method = dpg.get_value(f"{parent}_method")

        if "DS" in method and "PDS" not in method:
            self._build_ds_params(parent)
        elif "PDS" in method:
            self._build_pds_params(parent)
        elif "TSR" in method:
            self._build_tsr_params(parent)
        elif "CTAI" in method:
            self._build_ctai_params(parent)
        elif "NS-PFCE" in method:
            self._build_nspfce_params(parent)
        elif "JYPLS" in method:
            self._build_jypls_params(parent)
        else:
            self._build_tsr_params(parent)

        # Add tooltips to newly built parameters
        self._setup_method_tooltips(parent)

    def _run_transfer(self):
        """Run calibration transfer."""
        parent = f"{self.parent_tag}_ct_content"

        if self.master_spectra is None or self.slave_spectra is None:
            return

        method_str = dpg.get_value(f"{parent}_method")
        params_tag = f"{parent}_method_params"

        # Extract method type
        if "DS" in method_str and "PDS" not in method_str:
            method = "ds"
        elif "PDS" in method_str:
            method = "pds"
        elif "TSR" in method_str:
            method = "tsr"
        elif "CTAI" in method_str:
            method = "ctai"
        elif "NS-PFCE" in method_str:
            method = "nspfce"
        elif "JYPLS" in method_str:
            method = "jypls-inv"
        else:
            method = "tsr"

        self.current_method = method

        # Get transfer sample spectra
        X_master_transfer = self.master_spectra[self.transfer_indices]
        X_slave_transfer = self.slave_spectra[self.transfer_indices]

        # Run in thread to avoid blocking
        def _run():
            try:
                params = None

                if method == "ds":
                    # Get DS parameters
                    ridge_lambda = 0.001
                    if dpg.does_item_exist(f"{params_tag}_lambda"):
                        ridge_lambda = dpg.get_value(f"{params_tag}_lambda")
                    params = estimate_ds(X_master_transfer, X_slave_transfer, ridge_lambda=ridge_lambda)
                    self.transferred_spectra = apply_ds(self.slave_spectra, params)

                elif method == "pds":
                    # Get PDS parameters
                    window_size = 11
                    if dpg.does_item_exist(f"{params_tag}_window"):
                        window_size = dpg.get_value(f"{params_tag}_window")
                    # Ensure odd window size
                    if window_size % 2 == 0:
                        window_size += 1
                    params = estimate_pds(X_master_transfer, X_slave_transfer, window_size=window_size)
                    self.transferred_spectra = apply_pds(self.slave_spectra, params)

                elif method == "tsr":
                    # Get TSR parameters
                    regularization = 0.0
                    if dpg.does_item_exist(f"{params_tag}_regularization"):
                        regularization = dpg.get_value(f"{params_tag}_regularization")
                    params = estimate_tsr(X_master_transfer, X_slave_transfer, self.transfer_indices)
                    self.transferred_spectra = apply_tsr(self.slave_spectra, params)

                elif method == "ctai":
                    params = estimate_ctai(X_master_transfer, X_slave_transfer)
                    self.transferred_spectra = apply_ctai(self.slave_spectra, params)

                elif method == "nspfce":
                    # Get NS-PFCE parameters
                    max_iter = 100
                    use_vcpa = True
                    if dpg.does_item_exist(f"{params_tag}_max_iter"):
                        max_iter = dpg.get_value(f"{params_tag}_max_iter")
                    if dpg.does_item_exist(f"{params_tag}_use_vcpa"):
                        use_vcpa = dpg.get_value(f"{params_tag}_use_vcpa")
                    params = estimate_nspfce(X_master_transfer, X_slave_transfer, max_iter=max_iter)
                    self.transferred_spectra = apply_nspfce(self.slave_spectra, params)

                elif method == "jypls-inv":
                    # Get JYPLS parameters
                    n_components = 10
                    if dpg.does_item_exist(f"{params_tag}_n_components"):
                        n_components = dpg.get_value(f"{params_tag}_n_components")
                    # JYPLS requires Y reference values - use sample indices if not provided
                    y_ref = self.y_reference if self.y_reference is not None else np.arange(len(self.transfer_indices))
                    params = estimate_jypls_inv(X_master_transfer, X_slave_transfer, y_ref, n_components=n_components)
                    self.transferred_spectra = apply_jypls_inv(self.slave_spectra, params)

                # Calculate quality metrics
                rmse_before = np.sqrt(np.mean((self.slave_spectra - self.master_spectra) ** 2))
                rmse_after = np.sqrt(np.mean((self.transferred_spectra - self.master_spectra) ** 2))

                self.quality_metrics = {
                    'rmse_before': rmse_before,
                    'rmse_after': rmse_after,
                    'improvement': rmse_before / rmse_after if rmse_after > 0 else float('inf'),
                    'method': method
                }

                self.transfer_params = params

                # Update UI on main thread
                self._display_results()

            except Exception as e:
                import traceback
                traceback.print_exc()
                # Show error in results area
                if dpg.does_item_exist(f"{parent}_results"):
                    dpg.delete_item(f"{parent}_results", children_only=True)
                    with dpg.group(parent=f"{parent}_results"):
                        dpg.add_text(f"Transfer Error: {str(e)}", color=(255, 100, 100))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _display_results(self):
        """Display transfer results."""
        parent = f"{self.parent_tag}_ct_content"
        results_tag = f"{parent}_results"

        if dpg.does_item_exist(results_tag):
            dpg.delete_item(results_tag, children_only=True)

        with dpg.group(parent=results_tag):
            dpg.add_text("Transfer Complete!", color=(100, 255, 100))
            dpg.add_separator()
            dpg.add_spacer(height=10)

            if self.quality_metrics:
                dpg.add_text(f"Method: {self.quality_metrics.get('method', 'unknown').upper()}")
                dpg.add_text(f"RMSE Before: {self.quality_metrics['rmse_before']:.6f}")
                dpg.add_text(f"RMSE After: {self.quality_metrics['rmse_after']:.6f}")
                improvement = self.quality_metrics['improvement']
                if improvement > 1:
                    dpg.add_text(f"Improvement: {improvement:.2f}x better", color=(100, 255, 100))
                else:
                    dpg.add_text(f"Improvement: {improvement:.2f}x (no improvement)", color=(255, 200, 100))

            dpg.add_spacer(height=10)

        # Add the results visualization plot
        self._update_results_plot()

    def _save_model(self):
        """Save transfer model."""
        if self.transfer_params is None:
            return

        def on_save_path_selected(filepath):
            if not filepath:
                return

            try:
                # Ensure .pkl extension
                if not filepath.endswith('.pkl'):
                    filepath = filepath + '.pkl'

                # Create TransferModel object
                model = TransferModel(
                    master_id=self.master_instrument_id,
                    slave_id=self.slave_instrument_id,
                    method=self.current_method,
                    params=self.transfer_params,
                    wavelengths=self.master_wavelengths,
                    meta={
                        'n_transfer_samples': len(self.transfer_indices) if self.transfer_indices is not None else 0,
                        'n_wavelengths': len(self.master_wavelengths) if self.master_wavelengths is not None else 0,
                        'quality_metrics': self.quality_metrics
                    }
                )

                # Save model
                save_transfer_model(model, filepath)

                # Update UI to show success
                parent = f"{self.parent_tag}_ct_content"
                results_tag = f"{parent}_results"
                if dpg.does_item_exist(results_tag):
                    dpg.add_text(f"Model saved to: {Path(filepath).name}", parent=results_tag, color=(100, 255, 100))

            except Exception as e:
                print(f"Error saving model: {e}")

        if self.show_save_dialog_callback:
            self.show_save_dialog_callback(on_save_path_selected, "model")

    def _reset_wizard(self):
        """Reset wizard to initial state."""
        self.current_step = 0
        self.master_spectra = None
        self.slave_spectra = None
        self.master_wavelengths = None
        self.slave_wavelengths = None
        self.master_sample_ids = None
        self.slave_sample_ids = None
        self.transfer_indices = None
        self.transfer_params = None
        self.transferred_spectra = None
        self.quality_metrics = None
        self._update_ui()

    def _update_preview_plot(self):
        """Update the preview plot showing master vs slave spectra."""
        parent = f"{self.parent_tag}_ct_content"
        preview_tag = f"{parent}_preview_area"

        if not dpg.does_item_exist(preview_tag):
            return

        # Clear existing content
        dpg.delete_item(preview_tag, children_only=True)

        # Check if both datasets are loaded
        if self.master_spectra is None or self.slave_spectra is None:
            with dpg.group(parent=preview_tag):
                dpg.add_text("Load both datasets to see preview comparison", color=(150, 150, 150))
            return

        # Create plot
        plot_tag = f"{preview_tag}_plot"
        with dpg.plot(label="Spectra Comparison", parent=preview_tag, tag=plot_tag,
                      height=220, width=-1):
            dpg.add_plot_legend()

            # X axis (wavelengths)
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Wavelength (nm)")

            # Y axis (absorbance)
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Absorbance")

            # Calculate mean and std for both datasets
            master_mean = np.mean(self.master_spectra, axis=0)
            master_std = np.std(self.master_spectra, axis=0)
            slave_mean = np.mean(self.slave_spectra, axis=0)
            slave_std = np.std(self.slave_spectra, axis=0)

            # Use common wavelengths (master's wavelengths)
            wavelengths = self.master_wavelengths.astype(float).tolist()

            # Plot master mean
            dpg.add_line_series(
                wavelengths,
                master_mean.tolist(),
                label=f"Master (n={self.master_spectra.shape[0]})",
                parent=y_axis
            )

            # Plot slave mean
            dpg.add_line_series(
                wavelengths,
                slave_mean.tolist(),
                label=f"Slave (n={self.slave_spectra.shape[0]})",
                parent=y_axis
            )

    def _update_results_plot(self):
        """Update the results plot showing before/after transfer."""
        parent = f"{self.parent_tag}_ct_content"
        results_tag = f"{parent}_results"

        if not dpg.does_item_exist(results_tag):
            return

        if self.transferred_spectra is None:
            return

        # Create plot showing before/after
        plot_tag = f"{results_tag}_plot"
        if dpg.does_item_exist(plot_tag):
            dpg.delete_item(plot_tag)

        with dpg.plot(label="Transfer Results", parent=results_tag, tag=plot_tag,
                      height=180, width=-1):
            dpg.add_plot_legend()

            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Wavelength (nm)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Absorbance")

            wavelengths = self.master_wavelengths.astype(float).tolist()

            # Plot master mean (target)
            master_mean = np.mean(self.master_spectra, axis=0)
            dpg.add_line_series(
                wavelengths,
                master_mean.tolist(),
                label="Master (target)",
                parent=y_axis
            )

            # Plot original slave mean
            slave_mean = np.mean(self.slave_spectra, axis=0)
            dpg.add_line_series(
                wavelengths,
                slave_mean.tolist(),
                label="Slave (before)",
                parent=y_axis
            )

            # Plot transferred mean
            transferred_mean = np.mean(self.transferred_spectra, axis=0)
            dpg.add_line_series(
                wavelengths,
                transferred_mean.tolist(),
                label="Transferred (after)",
                parent=y_axis
            )
