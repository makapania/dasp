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
)
from ...core.instrument_profiles import characterize_instrument
from ...core.sample_selection import kennard_stone


class CalibrationTransferPanel:
    """
    Wizard-style panel for calibration transfer between instruments.
    """

    def __init__(self, parent_tag: str):
        """
        Initialize calibration transfer panel.

        Parameters
        ----------
        parent_tag : str
            DearPyGui tag of parent container
        """
        self.parent_tag = parent_tag
        self.current_step = 0

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
                dpg.add_text("→")
                dpg.add_button(label="2. Select Samples", tag=f"{self.parent_tag}_ct_step2_btn", callback=lambda: self._go_to_step(1))
                dpg.add_text("→")
                dpg.add_button(label="3. Configure Method", tag=f"{self.parent_tag}_ct_step3_btn", callback=lambda: self._go_to_step(2))
                dpg.add_text("→")
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
                dpg.add_input_text(tag=f"{parent}_master_path", width=400, hint="Master instrument file path")
                dpg.add_button(label="Browse...", callback=lambda: self._browse_file("master"))

            dpg.add_spacer(height=5)
            dpg.add_text("", tag=f"{parent}_master_status", color=(150, 150, 150))

            dpg.add_spacer(height=20)

            # Slave instrument
            dpg.add_text("Slave Instrument (To be transferred)")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag=f"{parent}_slave_path", width=400, hint="Slave instrument file path")
                dpg.add_button(label="Browse...", callback=lambda: self._browse_file("slave"))

            dpg.add_spacer(height=5)
            dpg.add_text("", tag=f"{parent}_slave_status", color=(150, 150, 150))

            dpg.add_spacer(height=30)

            # Info
            with dpg.group():
                dpg.add_text("Requirements:", color=(100, 150, 255))
                dpg.add_text("  • Both datasets must contain the same samples")
                dpg.add_text("  • Wavelength ranges should overlap")
                dpg.add_text("  • Sample IDs must match between instruments")

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

            dpg.add_spacer(height=20)

            # Method-specific parameters
            with dpg.child_window(tag=f"{parent}_method_params", height=200, border=True):
                self._build_tsr_params(parent)

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

    def _build_nspfce_params(self, parent: str):
        """Build NS-PFCE-specific parameters."""
        params_tag = f"{parent}_method_params"

        if dpg.does_item_exist(params_tag):
            dpg.delete_item(params_tag, children_only=True)

        with dpg.group(parent=params_tag):
            dpg.add_text("NS-PFCE Parameters:")
            dpg.add_spacer(height=10)

            dpg.add_checkbox(
                label="Use VCPA-IRIV Wavelength Selection",
                tag=f"{params_tag}_use_vcpa",
                default_value=True
            )

            dpg.add_text("Max Iterations:")
            dpg.add_slider_int(
                tag=f"{params_tag}_max_iter",
                default_value=100,
                min_value=10,
                max_value=500,
                width=200
            )

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
        """Browse for instrument data file."""
        # Placeholder for file dialog
        dpg.set_value(f"{self.parent_tag}_ct_content_{instrument_type}_status",
                     f"File browsing not yet implemented. Use input box to enter path.")

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

        if "PDS" in method:
            self._build_pds_params(parent)
        elif "NS-PFCE" in method:
            self._build_nspfce_params(parent)
        else:
            self._build_tsr_params(parent)

    def _run_transfer(self):
        """Run calibration transfer."""
        parent = f"{self.parent_tag}_ct_content"

        if self.master_spectra is None or self.slave_spectra is None:
            return

        method_str = dpg.get_value(f"{parent}_method")

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

        # Run in thread to avoid blocking
        def _run():
            try:
                if method == "tsr":
                    params = estimate_tsr(
                        self.master_spectra,
                        self.slave_spectra,
                        self.transfer_indices
                    )
                    self.transferred_spectra = apply_tsr(self.slave_spectra, params)

                elif method == "ctai":
                    params = estimate_ctai(self.master_spectra, self.slave_spectra)
                    self.transferred_spectra = apply_ctai(self.slave_spectra, params)

                # Calculate quality metrics
                rmse_before = np.sqrt(np.mean((self.slave_spectra - self.master_spectra) ** 2))
                rmse_after = np.sqrt(np.mean((self.transferred_spectra - self.master_spectra) ** 2))

                self.quality_metrics = {
                    'rmse_before': rmse_before,
                    'rmse_after': rmse_after,
                    'improvement': rmse_before / rmse_after
                }

                self.transfer_params = params

                # Update UI
                self._display_results()

            except Exception as e:
                print(f"Transfer error: {e}")

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
                dpg.add_text(f"RMSE Before: {self.quality_metrics['rmse_before']:.6f}")
                dpg.add_text(f"RMSE After: {self.quality_metrics['rmse_after']:.6f}")
                dpg.add_text(f"Improvement: {self.quality_metrics['improvement']:.2f}x")

    def _save_model(self):
        """Save transfer model."""
        if self.transfer_params is None:
            return

        # Placeholder for save dialog
        print("Save model dialog not yet implemented")

    def _reset_wizard(self):
        """Reset wizard to initial state."""
        self.current_step = 0
        self.master_spectra = None
        self.slave_spectra = None
        self.transfer_indices = None
        self.transfer_params = None
        self._update_ui()
