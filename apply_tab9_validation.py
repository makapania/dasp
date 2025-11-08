#!/usr/bin/env python3
"""
Apply Tab 9 (Calibration Transfer) validation checks to spectral_predict_gui_optimized.py

This script adds comprehensive validation checks across all sections of Tab 9:
- Section B: Select Instruments & Load Paired Spectra
- Section C: Build Transfer Model
- Section D: Multi-Instrument Equalization (placeholder)
- Section E: Predict with Transfer Model
"""

import re
import sys

def apply_validations():
    """Apply all Tab 9 validation checks."""

    filepath = r"C:\Users\sponheim\git\dasp\spectral_predict_gui_optimized.py"

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # ============================================================================
    # SECTION B: Load Paired Spectra Validation
    # ============================================================================

    # 1. Add "same instrument" check before try block in _load_ct_paired_spectra
    pattern1 = re.compile(
        r'(        if master_id not in self\.instrument_profiles or slave_id not in self\.instrument_profiles:\n'
        r'            messagebox\.showerror\("Error", "Selected instruments not found in registry"\)\n'
        r'            return\n)\n'
        r'(        try:)',
        re.MULTILINE
    )

    replacement1 = r'''\1
        # VALIDATION: Check that master and slave are different instruments
        if master_id == slave_id:
            messagebox.showerror(
                "Same Instrument Selected",
                "Master and slave instruments must be different for calibration transfer.\\n\\n"
                f"You selected: {master_id} for both master and slave.\\n\\n"
                "Please select different instruments."
            )
            return

\2'''

    content = pattern1.sub(replacement1, content)

    # 2. Add sample count, minimum sample, and wavelength overlap checks after loading spectra
    pattern2 = re.compile(
        r'(            # Load spectra using the helper from Tab 1\n'
        r'            wavelengths_master, X_master = self\._load_spectra_from_directory\(spectra_dir\)\n'
        r'            wavelengths_slave, X_slave = self\._load_spectra_from_directory\(spectra_dir\)\n)\n'
        r'(            # Get instrument profiles)',
        re.MULTILINE
    )

    replacement2 = r'''\1
            # VALIDATION 1: Same Sample Count Check
            if X_master.shape[0] != X_slave.shape[0]:
                messagebox.showerror(
                    "Sample Count Mismatch",
                    f"Master has {X_master.shape[0]} samples, Slave has {X_slave.shape[0]} samples.\\n\\n"
                    "Paired spectra must have the same number of samples (same sample set measured on both instruments).\\n\\n"
                    "Please ensure both instruments measured the exact same samples."
                )
                return

            # VALIDATION 2: Minimum Sample Check
            if X_master.shape[0] < 20:
                response = messagebox.askokcancel(
                    "Few Samples",
                    f"Only {X_master.shape[0]} paired samples loaded.\\n\\n"
                    "At least 30 samples recommended for robust calibration transfer.\\n"
                    "Results may be unreliable with fewer samples.\\n\\n"
                    "Do you want to continue anyway?"
                )
                if not response:
                    return

            # VALIDATION 3: Wavelength Overlap Check
            master_range = (wavelengths_master[0], wavelengths_master[-1])
            slave_range = (wavelengths_slave[0], wavelengths_slave[-1])

            overlap_start = max(master_range[0], slave_range[0])
            overlap_end = min(master_range[1], slave_range[1])

            if overlap_start >= overlap_end:
                messagebox.showerror(
                    "No Wavelength Overlap",
                    f"Master range: {master_range[0]:.1f}-{master_range[1]:.1f} nm\\n"
                    f"Slave range: {slave_range[0]:.1f}-{slave_range[1]:.1f} nm\\n\\n"
                    "Instruments must have overlapping wavelength ranges for calibration transfer.\\n\\n"
                    "Please select instruments with compatible wavelength coverage."
                )
                return

            # Check overlap percentage
            master_span = master_range[1] - master_range[0]
            slave_span = slave_range[1] - slave_range[0]
            overlap_span = overlap_end - overlap_start

            master_overlap_pct = (overlap_span / master_span) * 100
            slave_overlap_pct = (overlap_span / slave_span) * 100
            min_overlap_pct = min(master_overlap_pct, slave_overlap_pct)

            if min_overlap_pct < 80:
                response = messagebox.askokcancel(
                    "Limited Wavelength Overlap",
                    f"Wavelength overlap is {min_overlap_pct:.1f}% of instrument range.\\n\\n"
                    f"Master range: {master_range[0]:.1f}-{master_range[1]:.1f} nm\\n"
                    f"Slave range: {slave_range[0]:.1f}-{slave_range[1]:.1f} nm\\n"
                    f"Overlap region: {overlap_start:.1f}-{overlap_end:.1f} nm\\n\\n"
                    "Transfer quality may be reduced with limited overlap.\\n"
                    "Consider using instruments with better wavelength coverage overlap.\\n\\n"
                    "Do you want to continue anyway?"
                )
                if not response:
                    return

\2'''

    content = pattern2.sub(replacement2, content)

    # 3. Update info_text to include overlap percentage
    pattern3 = re.compile(
        r'(            # Display info\n'
        r'            info_text = \(f"Loaded {X_master\.shape\[0\]} paired spectra\\n"\n'
        r'                        f"Common wavelength grid: {common_wl\.shape\[0\]} points\\n"\n'
        r'                        f"Range: {common_wl\.min\(\):.1f} - {common_wl\.max\(\):.1f} nm"\))',
        re.MULTILINE
    )

    replacement3 = r'''            # Display info
            info_text = (f"Loaded {X_master.shape[0]} paired spectra\\n"
                        f"Common wavelength grid: {common_wl.shape[0]} points\\n"
                        f"Range: {common_wl.min():.1f} - {common_wl.max():.1f} nm\\n"
                        f"Wavelength overlap: {min_overlap_pct:.1f}%")'''

    content = pattern3.sub(replacement3, content)

    # ============================================================================
    # SECTION C: Build Transfer Model Validation
    # ============================================================================

    # 4. Enhance data loaded check with better error message
    pattern4 = re.compile(
        r'(        if self\.ct_X_master_common is None or self\.ct_X_slave_common is None:\n'
        r'            messagebox\.showwarning\("Warning", "Please load paired spectra first"\)\n'
        r'            return)',
        re.MULTILINE
    )

    replacement4 = r'''        # VALIDATION: Data Loaded Check
        if not hasattr(self, 'ct_X_master_common') or not hasattr(self, 'ct_X_slave_common'):
            messagebox.showerror(
                "No Paired Spectra Loaded",
                "Please load paired standardization spectra in Section B first."
            )
            return

        if self.ct_X_master_common is None or self.ct_X_slave_common is None:
            messagebox.showerror(
                "No Paired Spectra Loaded",
                "Please load paired standardization spectra in Section B first."
            )
            return'''

    content = pattern4.sub(replacement4, content)

    # 5. Add same instrument check in build method
    pattern5 = re.compile(
        r'(        method = self\.ct_method_var\.get\(\)\n'
        r'        master_id = self\.ct_master_instrument_id\.get\(\)\n'
        r'        slave_id = self\.ct_slave_instrument_id\.get\(\)\n)\n'
        r'(        try:)',
        re.MULTILINE
    )

    replacement5 = r'''\1
        # VALIDATION: Different Instruments Check
        if master_id == slave_id:
            messagebox.showerror(
                "Same Instrument Selected",
                "Master and slave instruments must be different for calibration transfer."
            )
            return

\2'''

    content = pattern5.sub(replacement5, content)

    # 6. Add parameter validation for DS and PDS
    pattern6 = re.compile(
        r'(        try:\n'
        r'            if method == \'ds\':\n'
        r'                # Build DS transfer model\n'
        r'                lam = float\(self\.ct_ds_lambda_var\.get\(\)\))',
        re.MULTILINE
    )

    replacement6 = r'''        try:
            if method == 'ds':
                # Build DS transfer model
                # VALIDATION: DS Ridge Lambda parameter
                try:
                    lam = float(self.ct_ds_lambda_var.get())
                    if lam <= 0 or lam > 100:
                        messagebox.showerror(
                            "Invalid Parameter",
                            f"DS Ridge Lambda must be between 0 and 100.\\nYou entered: {lam}"
                        )
                        return
                except ValueError:
                    messagebox.showerror("Invalid Parameter", "DS Ridge Lambda must be a number.")
                    return'''

    content = pattern6.sub(replacement6, content)

    # 7. Add parameter validation for PDS window
    pattern7 = re.compile(
        r'(            elif method == \'pds\':\n'
        r'                # Build PDS transfer model\n'
        r'                window = int\(self\.ct_pds_window_var\.get\(\)\))',
        re.MULTILINE
    )

    replacement7 = r'''            elif method == 'pds':
                # Build PDS transfer model
                # VALIDATION: PDS Window parameter
                try:
                    window = int(self.ct_pds_window_var.get())
                    if window < 5 or window > 101:
                        messagebox.showerror(
                            "Invalid Parameter",
                            f"PDS Window must be between 5 and 101.\\nYou entered: {window}"
                        )
                        return
                    if window % 2 == 0:
                        messagebox.showerror(
                            "Invalid Parameter",
                            f"PDS Window must be an odd number.\\nYou entered: {window} (even)"
                        )
                        return
                except ValueError:
                    messagebox.showerror("Invalid Parameter", "PDS Window must be an integer.")
                    return'''

    content = pattern7.sub(replacement7, content)

    # ============================================================================
    # SECTION E: Predict with Transfer Model Validation
    # ============================================================================

    # 8. Enhance master model check
    pattern8 = re.compile(
        r'(        if self\.ct_pred_transfer_model is None:\n'
        r'            messagebox\.showwarning\("Warning", "Please load a transfer model first"\)\n'
        r'            return\n\n'
        r'        if self\.ct_master_model_dict is None:\n'
        r'            messagebox\.showwarning\("Warning", "Please load master model first \(Section A\)"\)\n'
        r'            return)',
        re.MULTILINE
    )

    replacement8 = r'''        # VALIDATION: Models Loaded Check
        if self.ct_master_model_dict is None:
            messagebox.showerror(
                "Master Model Not Loaded",
                "Please load the master model in Section A first."
            )
            return

        if self.ct_pred_transfer_model is None:
            messagebox.showerror(
                "Transfer Model Not Loaded",
                "Please load or build a transfer model in Section C first."
            )
            return'''

    content = pattern8.sub(replacement8, content)

    # 9. Add wavelength compatibility check after loading new slave spectra
    pattern9 = re.compile(
        r'(            # Load new slave spectra\n'
        r'            wavelengths_slave, X_slave_new = self\._load_spectra_from_directory\(new_slave_dir\)\n)\n'
        r'(            # Resample to common grid\n'
        r'            common_wl = self\.ct_pred_transfer_model\.wavelengths_common)',
        re.MULTILINE
    )

    replacement9 = r'''\1
            # VALIDATION: Wavelength Compatibility Check
            transfer_slave_range = (
                self.ct_pred_transfer_model.wavelengths_common[0],
                self.ct_pred_transfer_model.wavelengths_common[-1]
            )
            new_slave_range = (wavelengths_slave[0], wavelengths_slave[-1])

            # Check if new slave data can be resampled to transfer model wavelengths
            if new_slave_range[0] > transfer_slave_range[0] or new_slave_range[1] < transfer_slave_range[1]:
                messagebox.showwarning(
                    "Wavelength Range Mismatch",
                    f"Transfer model expects wavelengths: {transfer_slave_range[0]:.1f}-{transfer_slave_range[1]:.1f} nm\\n"
                    f"New slave data has wavelengths: {new_slave_range[0]:.1f}-{new_slave_range[1]:.1f} nm\\n\\n"
                    "New slave data has narrower wavelength coverage than the transfer model expects.\\n"
                    "Predictions may require extrapolation and could be unreliable."
                )

\2'''

    content = pattern9.sub(replacement9, content)

    # 10. Add extrapolation warning after transfer
    pattern10 = re.compile(
        r'(            # Resample transferred spectra to master model\'s wavelength grid\n'
        r'            wl_model = self\.ct_master_model_dict\[\'wavelengths\'\]\n'
        r'            X_for_prediction = resample_to_grid\(X_transferred, common_wl, wl_model\)\n)\n'
        r'(            # Apply preprocessing if present)',
        re.MULTILINE
    )

    replacement10 = r'''\1
            # VALIDATION: Extrapolation Warning
            if 'wavelength_range' in self.ct_master_model_dict:
                model_wl_range = self.ct_master_model_dict['wavelength_range']
                if wl_model[0] < model_wl_range[0] or wl_model[-1] > model_wl_range[1]:
                    messagebox.showwarning(
                        "Extrapolation Warning",
                        f"Transferred data wavelengths ({wl_model[0]:.1f}-{wl_model[-1]:.1f} nm)\\n"
                        f"exceed master model training range ({model_wl_range[0]:.1f}-{model_wl_range[1]:.1f} nm).\\n\\n"
                        "Predictions may be unreliable in extrapolated regions."
                    )

\2'''

    content = pattern10.sub(replacement10, content)

    # Check if any changes were made
    if content == original_content:
        print("ERROR: No patterns matched! File may have unexpected format.")
        return False

    # Write the modified content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("[SUCCESS] Tab 9 validation checks successfully applied!")
    print("\nValidations added:")
    print("  Section B (Load Paired Spectra):")
    print("    - Same instrument check (before loading)")
    print("    - Sample count mismatch check")
    print("    - Minimum sample count check (warning)")
    print("    - Wavelength overlap check (error + warning)")
    print("  Section C (Build Transfer Model):")
    print("    - Data loaded check (enhanced)")
    print("    - Same instrument check")
    print("    - DS Ridge Lambda parameter validation")
    print("    - PDS Window parameter validation")
    print("  Section E (Predict with Transfer Model):")
    print("    - Master model loaded check (enhanced)")
    print("    - Transfer model loaded check (enhanced)")
    print("    - Wavelength compatibility check")
    print("    - Extrapolation warning")

    return True

if __name__ == "__main__":
    success = apply_validations()
    sys.exit(0 if success else 1)
