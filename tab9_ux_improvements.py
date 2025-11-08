"""
Tab 9 UX Improvements for Calibration Transfer
================================================

This file contains all code modifications needed to implement UX improvements
for Tab 9 (Calibration Transfer) in spectral_predict_gui_optimized.py.

Agent 3 - UX Improvements Implementation
"""

# ============================================================================
# PART 1: Add to __init__ method (around line 148, after existing Tab 9 vars)
# ============================================================================

INIT_STATUS_VARS = '''
        # Tab 9 Section Status Tracking
        self.ct_section_a_complete = False  # Master model loaded
        self.ct_section_b_complete = False  # Instruments & paired spectra loaded
        self.ct_section_c_complete = False  # Transfer model built
        self.ct_section_d_complete = False  # Equalization done
        self.ct_section_e_complete = False  # Predictions made

        # Tab 9 UI References (for enable/disable control)
        self.ct_section_b_buttons = []  # Buttons to enable after section A
        self.ct_section_c_button = None  # Build button to enable after section B
        self.ct_section_d_buttons = []  # Equalization buttons
        self.ct_section_e_buttons = []  # Prediction buttons

        # Tab 9 Status Labels
        self.ct_status_labels = {}  # Dict of section -> status label widget
        self.ct_workflow_labels = {}  # Dict of step -> workflow label widget
'''

# ============================================================================
# PART 2: Helper Methods for Status Management
# ============================================================================

HELPER_METHODS = '''
    def _create_help_button(self, parent, help_text, title="Help"):
        """Create a help button that shows info on click."""
        help_label = ttk.Label(parent, text="ℹ️", style='TLabel', cursor="hand2")
        help_label.bind("<Button-1>", lambda e: messagebox.showinfo(title, help_text))
        return help_label

    def _update_ct_section_status(self, section, complete):
        """Update status indicator for a calibration transfer section.

        Args:
            section: 'a', 'b', 'c', 'd', or 'e'
            complete: True if section is complete, False otherwise
        """
        # Update internal state
        if section == 'a':
            self.ct_section_a_complete = complete
        elif section == 'b':
            self.ct_section_b_complete = complete
        elif section == 'c':
            self.ct_section_c_complete = complete
        elif section == 'd':
            self.ct_section_d_complete = complete
        elif section == 'e':
            self.ct_section_e_complete = complete

        # Update status label
        if section in self.ct_status_labels:
            label = self.ct_status_labels[section]
            if complete:
                label.config(text="✓ Complete", foreground="#27AE60", font=('Segoe UI', 10, 'bold'))
            else:
                label.config(text="○ Pending", foreground="#95A5A6", font=('Segoe UI', 10))

        # Update workflow guide
        self._update_ct_workflow_guide()

        # Update button states
        self._update_ct_button_states()

    def _update_ct_workflow_guide(self):
        """Update workflow guide colors based on section completion."""
        workflow_steps = {
            'a': self.ct_section_a_complete,
            'b': self.ct_section_b_complete,
            'c': self.ct_section_c_complete,
            'd': self.ct_section_d_complete,
            'e': self.ct_section_e_complete
        }

        for step, label in self.ct_workflow_labels.items():
            if workflow_steps.get(step, False):
                label.config(foreground="#27AE60", font=('Segoe UI', 9, 'bold'))
            elif step == 'a' or (step == 'b' and workflow_steps['a']) or \
                 (step == 'c' and workflow_steps['b']) or \
                 (step == 'e' and workflow_steps['a']):
                # Required step or next available step
                label.config(foreground="#E67E22", font=('Segoe UI', 9, 'bold'))
            else:
                label.config(foreground="#95A5A6", font=('Segoe UI', 9))

    def _update_ct_button_states(self):
        """Enable/disable buttons based on section completion states."""
        # Section B buttons: enable only when section A complete
        for button in self.ct_section_b_buttons:
            if self.ct_section_a_complete:
                button.config(state='normal')
            else:
                button.config(state='disabled')

        # Section C button: enable only when section B complete
        if self.ct_section_c_button:
            if self.ct_section_b_complete:
                self.ct_section_c_button.config(state='normal')
            else:
                self.ct_section_c_button.config(state='disabled')

        # Section D buttons: enable only when instruments registered
        for button in self.ct_section_d_buttons:
            if self.instrument_profiles:
                button.config(state='normal')
            else:
                button.config(state='disabled')

        # Section E buttons: enable only when master model + transfer model loaded
        for button in self.ct_section_e_buttons:
            if self.ct_master_model_dict and self.ct_pred_transfer_model:
                button.config(state='normal')
            else:
                button.config(state='disabled')

    def _validate_ct_ds_lambda(self, *args):
        """Validate DS Ridge Lambda parameter and show visual feedback."""
        try:
            value = float(self.ct_ds_lambda_var.get())
            if value < 0.0001 or value > 1.0:
                self.ct_ds_lambda_entry.config(foreground='#E74C3C')  # Red
                self.ct_ds_lambda_warning.config(
                    text="⚠ Recommended: 0.001-1.0",
                    foreground='#E67E22'
                )
            else:
                self.ct_ds_lambda_entry.config(foreground='#27AE60')  # Green
                self.ct_ds_lambda_warning.config(text="")
        except ValueError:
            self.ct_ds_lambda_entry.config(foreground='#E74C3C')
            self.ct_ds_lambda_warning.config(
                text="⚠ Invalid number",
                foreground='#E74C3C'
            )

    def _validate_ct_pds_window(self, *args):
        """Validate PDS Window parameter and show visual feedback."""
        try:
            value = int(self.ct_pds_window_var.get())
            if value < 5 or value > 101:
                self.ct_pds_window_entry.config(foreground='#E74C3C')
                self.ct_pds_window_warning.config(
                    text="⚠ Recommended: 5-101",
                    foreground='#E67E22'
                )
            elif value % 2 == 0:
                self.ct_pds_window_entry.config(foreground='#E67E22')
                self.ct_pds_window_warning.config(
                    text="⚠ Should be odd number",
                    foreground='#E67E22'
                )
            else:
                self.ct_pds_window_entry.config(foreground='#27AE60')
                self.ct_pds_window_warning.config(text="")
        except ValueError:
            self.ct_pds_window_entry.config(foreground='#E74C3C')
            self.ct_pds_window_warning.config(
                text="⚠ Invalid number",
                foreground='#E74C3C'
            )
'''

# ============================================================================
# PART 3: Updated _create_tab9_calibration_transfer method
# ============================================================================

UPDATED_CREATE_TAB9 = '''
    def _create_tab9_calibration_transfer(self):
        """Tab 9: Calibration Transfer & Equalized Prediction with UX Improvements."""
        self.tab9 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab9, text='  🔄 Calibration Transfer  ')

        # Create scrollable content
        canvas = tk.Canvas(self.tab9, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab9, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='TFrame')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = ttk.Frame(scrollable_frame, style='TFrame', padding="30")
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Calibration Transfer & Equalized Prediction",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 10))

        desc_label = ttk.Label(main_frame,
                              text="Build transfer models between instruments and make equalized predictions",
                              style='Caption.TLabel')
        desc_label.pack(pady=(0, 20))

        # ===================================================================
        # WORKFLOW GUIDE (NEW)
        # ===================================================================
        workflow_frame = ttk.LabelFrame(main_frame, text="📋 Workflow Guide",
                                       style='Card.TFrame', padding=15)
        workflow_frame.pack(fill='x', pady=(0, 15))

        workflow_text = ttk.Frame(workflow_frame, style='TFrame')
        workflow_text.pack(fill='x')

        # Create workflow labels
        self.ct_workflow_labels['a'] = ttk.Label(workflow_text,
            text="A. Load Master Model", style='TLabel')
        self.ct_workflow_labels['a'].pack(side='left', padx=5)

        ttk.Label(workflow_text, text="→", style='TLabel').pack(side='left', padx=3)

        self.ct_workflow_labels['b'] = ttk.Label(workflow_text,
            text="B. Select Instruments & Load Paired Spectra", style='TLabel')
        self.ct_workflow_labels['b'].pack(side='left', padx=5)

        ttk.Label(workflow_text, text="→", style='TLabel').pack(side='left', padx=3)

        self.ct_workflow_labels['c'] = ttk.Label(workflow_text,
            text="C. Build Transfer Model", style='TLabel')
        self.ct_workflow_labels['c'].pack(side='left', padx=5)

        ttk.Label(workflow_text, text="→", style='TLabel').pack(side='left', padx=3)

        self.ct_workflow_labels['d'] = ttk.Label(workflow_text,
            text="D. Export Equalized Spectra (Optional)", style='TLabel')
        self.ct_workflow_labels['d'].pack(side='left', padx=5)

        ttk.Label(workflow_text, text="→", style='TLabel').pack(side='left', padx=3)

        self.ct_workflow_labels['e'] = ttk.Label(workflow_text,
            text="E. Predict with Transfer Model", style='TLabel')
        self.ct_workflow_labels['e'].pack(side='left', padx=5)

        # ===================================================================
        # SECTION A: Load Master Model
        # ===================================================================
        section_a = ttk.LabelFrame(main_frame, text="A) Load Master Model",
                                  style='Card.TFrame', padding=15)
        section_a.pack(fill='x', pady=(0, 15))

        # Status indicator
        status_frame_a = ttk.Frame(section_a, style='TFrame')
        status_frame_a.pack(anchor='w', pady=(0, 10))
        self.ct_status_labels['a'] = ttk.Label(status_frame_a, text="○ Pending",
                                                foreground="#95A5A6",
                                                font=('Segoe UI', 10))
        self.ct_status_labels['a'].pack(side='left')

        ttk.Label(section_a, text="Load a trained PLS/PCR model (the 'master' instrument) for calibration transfer:",
                 style='TLabel').pack(anchor='w', pady=(0, 10))

        load_model_frame = ttk.Frame(section_a, style='TFrame')
        load_model_frame.pack(fill='x')

        self.ct_master_model_path_var = tk.StringVar()
        model_entry = ttk.Entry(load_model_frame, textvariable=self.ct_master_model_path_var,
                               width=60, state='readonly')
        model_entry.pack(side='left', padx=(0, 10))

        ttk.Button(load_model_frame, text="Browse Model...",
                  command=self._browse_ct_master_model).pack(side='left', padx=(0, 10))
        ttk.Button(load_model_frame, text="Load Model",
                  command=self._load_ct_master_model_ux,
                  style='Accent.TButton').pack(side='left')

        # Model info display
        self.ct_model_info_text = tk.Text(section_a, height=4, width=80, state='disabled',
                                          wrap='word', relief='flat', bg='#f0f0f0')
        self.ct_model_info_text.pack(fill='x', pady=(10, 0))

        # ===================================================================
        # SECTION B: Select Instruments & Load Paired Spectra
        # ===================================================================
        section_b = ttk.LabelFrame(main_frame, text="B) Select Instruments & Load Paired Spectra",
                                  style='Card.TFrame', padding=15)
        section_b.pack(fill='x', pady=(0, 15))

        # Status indicator
        status_frame_b = ttk.Frame(section_b, style='TFrame')
        status_frame_b.pack(anchor='w', pady=(0, 10))
        self.ct_status_labels['b'] = ttk.Label(status_frame_b, text="⚠ Required",
                                                foreground="#E67E22",
                                                font=('Segoe UI', 10))
        self.ct_status_labels['b'].pack(side='left')

        # Help text for paired spectra
        help_frame_b = ttk.Frame(section_b, style='TFrame')
        help_frame_b.pack(fill='x', pady=(0, 10))

        help_button = self._create_help_button(
            help_frame_b,
            "Paired Spectra Information\\n\\n"
            "Paired spectra are identical samples measured on BOTH the master and slave "
            "instruments. These samples are used to build the calibration transfer model.\\n\\n"
            "Requirements:\\n"
            "• Same physical samples measured on both instruments\\n"
            "• Ideally 20-50 samples covering the range of variation\\n"
            "• Files should be in the same directory with identical naming",
            "What are Paired Spectra?"
        )
        help_button.pack(side='left', padx=(0, 10))

        note_label = ttk.Label(help_frame_b,
            text="Note: Register instruments in Tab 8 (Instrument Lab) first",
            style='Caption.TLabel', foreground='#E67E22')
        note_label.pack(side='left')

        ttk.Label(section_b,
                 text="Select master and slave instruments, then load paired standardization spectra:",
                 style='TLabel').pack(anchor='w', pady=(0, 10))

        # Instrument selection grid
        inst_grid = ttk.Frame(section_b, style='TFrame')
        inst_grid.pack(fill='x', pady=(0, 10))

        ttk.Label(inst_grid, text="Master Instrument:", style='TLabel').grid(
            row=0, column=0, sticky='w', padx=(0, 10))
        self.ct_master_instrument_combo = ttk.Combobox(inst_grid,
            textvariable=self.ct_master_instrument_id, state='readonly', width=30)
        self.ct_master_instrument_combo.grid(row=0, column=1, sticky='w', padx=(0, 20))

        ttk.Label(inst_grid, text="Slave Instrument:", style='TLabel').grid(
            row=0, column=2, sticky='w', padx=(0, 10))
        self.ct_slave_instrument_combo = ttk.Combobox(inst_grid,
            textvariable=self.ct_slave_instrument_id, state='readonly', width=30)
        self.ct_slave_instrument_combo.grid(row=0, column=3, sticky='w')

        refresh_btn = ttk.Button(inst_grid, text="Refresh from Registry",
                  command=self._refresh_ct_instrument_combos)
        refresh_btn.grid(row=0, column=4, padx=(20, 0))
        self.ct_section_b_buttons.append(refresh_btn)

        # Load paired spectra
        load_spectra_frame = ttk.Frame(section_b, style='TFrame')
        load_spectra_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(load_spectra_frame, text="Standardization Spectra Directory:",
                 style='TLabel').pack(anchor='w', pady=(0, 5))

        spectra_entry_frame = ttk.Frame(load_spectra_frame, style='TFrame')
        spectra_entry_frame.pack(fill='x')

        self.ct_spectra_dir_var = tk.StringVar()
        spectra_entry = ttk.Entry(spectra_entry_frame, textvariable=self.ct_spectra_dir_var,
                                 width=60, state='readonly')
        spectra_entry.pack(side='left', padx=(0, 10))

        browse_spectra_btn = ttk.Button(spectra_entry_frame, text="Browse Directory...",
                  command=self._browse_ct_spectra_dir)
        browse_spectra_btn.pack(side='left', padx=(0, 10))
        self.ct_section_b_buttons.append(browse_spectra_btn)

        load_spectra_btn = ttk.Button(spectra_entry_frame, text="Load Paired Spectra",
                  command=self._load_ct_paired_spectra_ux, style='Accent.TButton')
        load_spectra_btn.pack(side='left')
        self.ct_section_b_buttons.append(load_spectra_btn)

        # Spectra info
        self.ct_spectra_info_text = tk.Text(section_b, height=3, width=80, state='disabled',
                                           wrap='word', relief='flat', bg='#f0f0f0')
        self.ct_spectra_info_text.pack(fill='x', pady=(10, 0))

        # ===================================================================
        # SECTION C: Build Transfer Mapping (DS/PDS)
        # ===================================================================
        section_c = ttk.LabelFrame(main_frame, text="C) Build Transfer Mapping",
                                  style='Card.TFrame', padding=15)
        section_c.pack(fill='x', pady=(0, 15))

        # Status indicator
        status_frame_c = ttk.Frame(section_c, style='TFrame')
        status_frame_c.pack(anchor='w', pady=(0, 10))
        self.ct_status_labels['c'] = ttk.Label(status_frame_c, text="○ Pending",
                                                foreground="#95A5A6",
                                                font=('Segoe UI', 10))
        self.ct_status_labels['c'].pack(side='left')

        ttk.Label(section_c, text="Configure and build calibration transfer model (DS or PDS):",
                 style='TLabel').pack(anchor='w', pady=(0, 10))

        # Method selection and parameters
        method_frame = ttk.Frame(section_c, style='TFrame')
        method_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(method_frame, text="Transfer Method:", style='TLabel').pack(
            side='left', padx=(0, 10))
        self.ct_method_var = tk.StringVar(value='ds')
        ttk.Radiobutton(method_frame, text="Direct Standardization (DS)",
                       variable=self.ct_method_var, value='ds').pack(side='left', padx=(0, 20))
        ttk.Radiobutton(method_frame, text="Piecewise DS (PDS)",
                       variable=self.ct_method_var, value='pds').pack(side='left', padx=(0, 10))

        # Method help button
        method_help = self._create_help_button(
            method_frame,
            "Transfer Method Selection\\n\\n"
            "DS (Direct Standardization):\\n"
            "• Global linear transformation\\n"
            "• Fast, simple, works well for similar instruments\\n"
            "• Use when instrument responses are linearly related\\n\\n"
            "PDS (Piecewise Direct Standardization):\\n"
            "• Local non-linear transformation\\n"
            "• More flexible, adapts to local variations\\n"
            "• Use when instruments have wavelength-dependent differences",
            "Transfer Method Help"
        )
        method_help.pack(side='left')

        # Parameters with validation
        params_frame = ttk.Frame(section_c, style='TFrame')
        params_frame.pack(fill='x', pady=(0, 10))

        # DS Ridge Lambda
        ttk.Label(params_frame, text="DS Ridge Lambda:", style='TLabel').grid(
            row=0, column=0, sticky='w', padx=(0, 10))
        self.ct_ds_lambda_var = tk.StringVar(value='0.001')
        self.ct_ds_lambda_entry = ttk.Entry(params_frame,
            textvariable=self.ct_ds_lambda_var, width=15)
        self.ct_ds_lambda_entry.grid(row=0, column=1, sticky='w', padx=(0, 10))

        # DS Lambda help
        ds_help = self._create_help_button(
            params_frame,
            "DS Ridge Lambda (Regularization Parameter)\\n\\n"
            "Controls smoothness vs. flexibility of the transfer:\\n\\n"
            "• Higher values (e.g., 0.1-1.0): Smoother, more stable transfer\\n"
            "• Lower values (e.g., 0.001-0.01): More flexible, fits data closely\\n\\n"
            "Recommended range: 0.001 to 1.0\\n"
            "Default: 0.001 works well for most cases",
            "DS Ridge Lambda Help"
        )
        ds_help.grid(row=0, column=2, padx=(0, 10))

        ttk.Label(params_frame, text="(Recommended: 0.001-1.0)",
                 style='Caption.TLabel').grid(row=0, column=3, sticky='w', padx=(0, 20))

        self.ct_ds_lambda_warning = ttk.Label(params_frame, text="", style='Caption.TLabel')
        self.ct_ds_lambda_warning.grid(row=0, column=4, sticky='w')

        # PDS Window
        ttk.Label(params_frame, text="PDS Window:", style='TLabel').grid(
            row=1, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
        self.ct_pds_window_var = tk.StringVar(value='11')
        self.ct_pds_window_entry = ttk.Entry(params_frame,
            textvariable=self.ct_pds_window_var, width=15)
        self.ct_pds_window_entry.grid(row=1, column=1, sticky='w', padx=(0, 10), pady=(10, 0))

        # PDS Window help
        pds_help = self._create_help_button(
            params_frame,
            "PDS Window Size\\n\\n"
            "Number of neighboring wavelengths used for local transfer:\\n\\n"
            "• Larger windows (e.g., 31-51): Smoother transfer, more averaging\\n"
            "• Smaller windows (e.g., 11-21): More local adaptation\\n\\n"
            "Requirements:\\n"
            "• Must be an ODD number\\n"
            "• Recommended range: 11 to 51\\n"
            "• Default: 11 is a good starting point",
            "PDS Window Help"
        )
        pds_help.grid(row=1, column=2, padx=(0, 10), pady=(10, 0))

        ttk.Label(params_frame, text="(Recommended: 11-51, must be odd)",
                 style='Caption.TLabel').grid(row=1, column=3, sticky='w',
                                             padx=(0, 20), pady=(10, 0))

        self.ct_pds_window_warning = ttk.Label(params_frame, text="", style='Caption.TLabel')
        self.ct_pds_window_warning.grid(row=1, column=4, sticky='w', pady=(10, 0))

        # Bind validation
        self.ct_ds_lambda_var.trace('w', self._validate_ct_ds_lambda)
        self.ct_pds_window_var.trace('w', self._validate_ct_pds_window)

        # Build button
        self.ct_section_c_button = ttk.Button(section_c, text="Build Transfer Model",
                  command=self._build_ct_transfer_model_ux, style='Accent.TButton')
        self.ct_section_c_button.pack(pady=(0, 10))

        # Transfer model info
        self.ct_transfer_info_text = tk.Text(section_c, height=4, width=80, state='disabled',
                                            wrap='word', relief='flat', bg='#f0f0f0')
        self.ct_transfer_info_text.pack(fill='x')

        # Save transfer model
        save_tm_frame = ttk.Frame(section_c, style='TFrame')
        save_tm_frame.pack(fill='x', pady=(10, 0))

        ttk.Button(save_tm_frame, text="Save Transfer Model...",
                  command=self._save_ct_transfer_model).pack(side='left')

        # ===================================================================
        # SECTION D: Export Equalized Spectra (Optional)
        # ===================================================================
        section_d = ttk.LabelFrame(main_frame, text="D) Export Equalized Spectra (Optional)",
                                  style='Card.TFrame', padding=15)
        section_d.pack(fill='x', pady=(0, 15))

        # Status indicator
        status_frame_d = ttk.Frame(section_d, style='TFrame')
        status_frame_d.pack(anchor='w', pady=(0, 10))
        self.ct_status_labels['d'] = ttk.Label(status_frame_d, text="○ Optional",
                                                foreground="#95A5A6",
                                                font=('Segoe UI', 10))
        self.ct_status_labels['d'].pack(side='left')

        ttk.Label(section_d,
                 text="Use equalization to combine multi-instrument datasets into a common domain:",
                 style='TLabel').pack(anchor='w', pady=(0, 10))

        eq_button_frame = ttk.Frame(section_d, style='TFrame')
        eq_button_frame.pack(fill='x')

        load_eq_btn = ttk.Button(eq_button_frame, text="Load Multi-Instrument Dataset...",
                  command=self._load_multiinstrument_dataset)
        load_eq_btn.pack(side='left', padx=(0, 10))
        self.ct_section_d_buttons.append(load_eq_btn)

        export_eq_btn = ttk.Button(eq_button_frame, text="Equalize & Export...",
                  command=self._equalize_and_export, style='Accent.TButton')
        export_eq_btn.pack(side='left')
        self.ct_section_d_buttons.append(export_eq_btn)

        # ===================================================================
        # SECTION E: Predict with Transfer Model
        # ===================================================================
        section_e = ttk.LabelFrame(main_frame, text="E) Predict with Transfer Model",
                                  style='Card.TFrame', padding=15)
        section_e.pack(fill='x', pady=(0, 15))

        # Status indicator
        status_frame_e = ttk.Frame(section_e, style='TFrame')
        status_frame_e.pack(anchor='w', pady=(0, 10))
        self.ct_status_labels['e'] = ttk.Label(status_frame_e, text="○ Pending",
                                                foreground="#95A5A6",
                                                font=('Segoe UI', 10))
        self.ct_status_labels['e'].pack(side='left')

        ttk.Label(section_e,
                 text="Load new slave spectra, apply calibration transfer, and predict using master model:",
                 style='TLabel').pack(anchor='w', pady=(0, 10))

        # Load transfer model for prediction
        load_tm_frame = ttk.Frame(section_e, style='TFrame')
        load_tm_frame.pack(fill='x', pady=(0, 10))

        self.ct_pred_tm_path_var = tk.StringVar()
        tm_entry = ttk.Entry(load_tm_frame, textvariable=self.ct_pred_tm_path_var,
                            width=60, state='readonly')
        tm_entry.pack(side='left', padx=(0, 10))

        browse_tm_btn = ttk.Button(load_tm_frame, text="Browse Transfer Model...",
                  command=self._browse_ct_pred_transfer_model)
        browse_tm_btn.pack(side='left', padx=(0, 10))
        self.ct_section_e_buttons.append(browse_tm_btn)

        load_tm_btn = ttk.Button(load_tm_frame, text="Load TM",
                  command=self._load_ct_pred_transfer_model_ux)
        load_tm_btn.pack(side='left')
        self.ct_section_e_buttons.append(load_tm_btn)

        # Load new slave spectra
        load_new_spectra_frame = ttk.Frame(section_e, style='TFrame')
        load_new_spectra_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(load_new_spectra_frame, text="New Slave Spectra Directory:",
                 style='TLabel').pack(anchor='w', pady=(0, 5))

        new_spectra_entry_frame = ttk.Frame(load_new_spectra_frame, style='TFrame')
        new_spectra_entry_frame.pack(fill='x')

        self.ct_new_slave_dir_var = tk.StringVar()
        new_slave_entry = ttk.Entry(new_spectra_entry_frame,
                                   textvariable=self.ct_new_slave_dir_var,
                                   width=60, state='readonly')
        new_slave_entry.pack(side='left', padx=(0, 10))

        browse_new_btn = ttk.Button(new_spectra_entry_frame, text="Browse Directory...",
                  command=self._browse_ct_new_slave_dir)
        browse_new_btn.pack(side='left', padx=(0, 10))
        self.ct_section_e_buttons.append(browse_new_btn)

        predict_btn = ttk.Button(new_spectra_entry_frame, text="Load & Predict",
                  command=self._load_and_predict_ct_ux, style='Accent.TButton')
        predict_btn.pack(side='left')
        self.ct_section_e_buttons.append(predict_btn)

        # Prediction results
        self.ct_prediction_text = tk.Text(section_e, height=6, width=80, state='disabled',
                                         wrap='word', relief='flat', bg='#f0f0f0')
        self.ct_prediction_text.pack(fill='x', pady=(10, 0))

        # Export predictions
        export_pred_btn = ttk.Button(section_e, text="Export Predictions...",
                  command=self._export_ct_predictions)
        export_pred_btn.pack(pady=(10, 0), anchor='w')
        self.ct_section_e_buttons.append(export_pred_btn)

        # Initialize button states and workflow guide
        self._update_ct_button_states()
        self._update_ct_workflow_guide()
'''

# ============================================================================
# PART 4: Updated wrapper methods with UX updates
# ============================================================================

UPDATED_METHODS = '''
    def _load_ct_master_model_ux(self):
        """Load master model with UX status updates."""
        if not HAS_CALIBRATION_TRANSFER:
            messagebox.showerror("Error", "Calibration transfer modules not available")
            return

        filepath = self.ct_master_model_path_var.get()
        if not filepath:
            messagebox.showwarning("Warning", "Please browse and select a master model file")
            return

        try:
            import pickle
            with open(filepath, 'rb') as f:
                self.ct_master_model_dict = pickle.load(f)

            # Display model info
            model_type = self.ct_master_model_dict.get('model_type', 'Unknown')
            n_components = self.ct_master_model_dict.get('n_components', 'N/A')
            wl_model = self.ct_master_model_dict.get('wavelengths', np.array([]))

            info_text = (f"Model Type: {model_type}\\n"
                        f"Components: {n_components}\\n"
                        f"Wavelength Range: {wl_model.min():.1f} - {wl_model.max():.1f} nm\\n"
                        f"Number of Wavelengths: {len(wl_model)}")

            self.ct_model_info_text.config(state='normal')
            self.ct_model_info_text.delete('1.0', tk.END)
            self.ct_model_info_text.insert('1.0', info_text)
            self.ct_model_info_text.config(state='disabled')

            # Update status
            self._update_ct_section_status('a', True)

            messagebox.showinfo("Success", "Master model loaded successfully")
        except Exception as e:
            self._update_ct_section_status('a', False)
            messagebox.showerror("Error", f"Failed to load model:\\n{str(e)}")

    def _load_ct_paired_spectra_ux(self):
        """Load paired spectra with UX status updates."""
        if not HAS_CALIBRATION_TRANSFER:
            messagebox.showerror("Error", "Calibration transfer modules not available")
            return

        master_id = self.ct_master_instrument_id.get()
        slave_id = self.ct_slave_instrument_id.get()
        spectra_dir = self.ct_spectra_dir_var.get()

        if not master_id or not slave_id:
            messagebox.showwarning("Warning", "Please select both master and slave instruments")
            return

        if not spectra_dir:
            messagebox.showwarning("Warning", "Please browse and select spectra directory")
            return

        if master_id not in self.instrument_profiles or slave_id not in self.instrument_profiles:
            messagebox.showerror("Error", "Selected instruments not found in registry")
            return

        try:
            # Load spectra using the helper from Tab 1
            wavelengths_master, X_master = self._load_spectra_from_directory(spectra_dir)
            wavelengths_slave, X_slave = self._load_spectra_from_directory(spectra_dir)

            # Get instrument profiles
            master_prof = self.instrument_profiles[master_id]
            slave_prof = self.instrument_profiles[slave_id]

            # Choose common grid
            common_wl = choose_common_grid(
                {master_id: master_prof, slave_id: slave_prof},
                [master_id, slave_id]
            )

            # Resample both to common grid
            self.ct_X_master_common = resample_to_grid(X_master, wavelengths_master, common_wl)
            self.ct_X_slave_common = resample_to_grid(X_slave, wavelengths_slave, common_wl)
            self.ct_wavelengths_common = common_wl

            # Display info
            info_text = (f"Loaded {X_master.shape[0]} paired spectra\\n"
                        f"Common wavelength grid: {common_wl.shape[0]} points\\n"
                        f"Range: {common_wl.min():.1f} - {common_wl.max():.1f} nm")

            self.ct_spectra_info_text.config(state='normal')
            self.ct_spectra_info_text.delete('1.0', tk.END)
            self.ct_spectra_info_text.insert('1.0', info_text)
            self.ct_spectra_info_text.config(state='disabled')

            # Update status
            self._update_ct_section_status('b', True)

            messagebox.showinfo("Success", "Paired spectra loaded and resampled to common grid")
        except Exception as e:
            self._update_ct_section_status('b', False)
            messagebox.showerror("Error", f"Failed to load spectra:\\n{str(e)}")

    def _build_ct_transfer_model_ux(self):
        """Build transfer model with UX status updates."""
        if not HAS_CALIBRATION_TRANSFER:
            messagebox.showerror("Error", "Calibration transfer modules not available")
            return

        if self.ct_X_master_common is None or self.ct_X_slave_common is None:
            messagebox.showwarning("Warning", "Please load paired spectra first")
            return

        method = self.ct_method_var.get()
        master_id = self.ct_master_instrument_id.get()
        slave_id = self.ct_slave_instrument_id.get()

        try:
            if method == 'ds':
                # Build DS transfer model
                lam = float(self.ct_ds_lambda_var.get())
                A = estimate_ds(self.ct_X_master_common, self.ct_X_slave_common, lam=lam)

                # Create TransferModel object
                from spectral_predict.calibration_transfer import TransferModel
                self.ct_transfer_model = TransferModel(
                    master_id=master_id,
                    slave_id=slave_id,
                    method='ds',
                    wavelengths_common=self.ct_wavelengths_common,
                    params={'A': A},
                    meta={'lambda': lam, 'note': 'DS transfer built in GUI'}
                )

                info_text = (f"Transfer Method: Direct Standardization (DS)\\n"
                            f"Master: {master_id} → Slave: {slave_id}\\n"
                            f"Ridge Lambda: {lam}\\n"
                            f"Matrix Shape: {A.shape}")

            elif method == 'pds':
                # Build PDS transfer model
                window = int(self.ct_pds_window_var.get())
                B = estimate_pds(self.ct_X_master_common, self.ct_X_slave_common, window=window)

                from spectral_predict.calibration_transfer import TransferModel
                self.ct_transfer_model = TransferModel(
                    master_id=master_id,
                    slave_id=slave_id,
                    method='pds',
                    wavelengths_common=self.ct_wavelengths_common,
                    params={'B': B, 'window': window},
                    meta={'note': 'PDS transfer built in GUI'}
                )

                info_text = (f"Transfer Method: Piecewise Direct Standardization (PDS)\\n"
                            f"Master: {master_id} → Slave: {slave_id}\\n"
                            f"Window Size: {window}\\n"
                            f"Coefficient Matrix Shape: {B.shape}")

            # Display transfer model info
            self.ct_transfer_info_text.config(state='normal')
            self.ct_transfer_info_text.delete('1.0', tk.END)
            self.ct_transfer_info_text.insert('1.0', info_text)
            self.ct_transfer_info_text.config(state='disabled')

            # Update status
            self._update_ct_section_status('c', True)

            messagebox.showinfo("Success", f"{method.upper()} transfer model built successfully")
        except Exception as e:
            self._update_ct_section_status('c', False)
            messagebox.showerror("Error", f"Failed to build transfer model:\\n{str(e)}")

    def _load_ct_pred_transfer_model_ux(self):
        """Load transfer model for prediction with UX status updates."""
        if not HAS_CALIBRATION_TRANSFER:
            messagebox.showerror("Error", "Calibration transfer modules not available")
            return

        path_prefix = self.ct_pred_tm_path_var.get()
        if not path_prefix:
            messagebox.showwarning("Warning", "Please browse and select a transfer model")
            return

        try:
            self.ct_pred_transfer_model = load_transfer_model(path_prefix)

            # Update button states (will check if both model and TM are loaded)
            self._update_ct_button_states()

            messagebox.showinfo("Success",
                f"Transfer model loaded:\\n"
                f"Method: {self.ct_pred_transfer_model.method.upper()}\\n"
                f"Master: {self.ct_pred_transfer_model.master_id}\\n"
                f"Slave: {self.ct_pred_transfer_model.slave_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load transfer model:\\n{str(e)}")

    def _load_and_predict_ct_ux(self):
        """Load new slave spectra, apply transfer, and predict with sample ID improvements."""
        if not HAS_CALIBRATION_TRANSFER:
            messagebox.showerror("Error", "Calibration transfer modules not available")
            return

        if self.ct_pred_transfer_model is None:
            messagebox.showwarning("Warning", "Please load a transfer model first")
            return

        if self.ct_master_model_dict is None:
            messagebox.showwarning("Warning", "Please load master model first (Section A)")
            return

        new_slave_dir = self.ct_new_slave_dir_var.get()
        if not new_slave_dir:
            messagebox.showwarning("Warning", "Please browse and select new slave spectra directory")
            return

        try:
            import glob
            from pathlib import Path

            # Get list of files for sample IDs
            asd_files = sorted(glob.glob(os.path.join(new_slave_dir, "*.asd")))
            csv_files = sorted(glob.glob(os.path.join(new_slave_dir, "*.csv")))
            spc_files = sorted(glob.glob(os.path.join(new_slave_dir, "*.spc")))

            # Determine which files are present
            if asd_files:
                sample_files = asd_files
            elif csv_files:
                sample_files = csv_files
            elif spc_files:
                sample_files = spc_files
            else:
                raise ValueError("No supported spectral files found")

            # Extract sample IDs from filenames
            sample_ids = [Path(f).stem for f in sample_files]

            # Load new slave spectra
            wavelengths_slave, X_slave_new = self._load_spectra_from_directory(new_slave_dir)

            # Resample to common grid
            common_wl = self.ct_pred_transfer_model.wavelengths_common
            X_slave_common = resample_to_grid(X_slave_new, wavelengths_slave, common_wl)

            # Apply transfer model
            if self.ct_pred_transfer_model.method == 'ds':
                A = self.ct_pred_transfer_model.params['A']
                X_transferred = apply_ds(X_slave_common, A)
            elif self.ct_pred_transfer_model.method == 'pds':
                B = self.ct_pred_transfer_model.params['B']
                window = self.ct_pred_transfer_model.params['window']
                X_transferred = apply_pds(X_slave_common, B, window)

            # Resample transferred spectra to master model's wavelength grid
            wl_model = self.ct_master_model_dict['wavelengths']
            X_for_prediction = resample_to_grid(X_transferred, common_wl, wl_model)

            # Apply preprocessing if present
            if 'preprocessing' in self.ct_master_model_dict:
                prep = self.ct_master_model_dict['preprocessing']
                X_for_prediction = prep.transform(X_for_prediction)

            # Predict
            model = self.ct_master_model_dict['model']
            y_pred = model.predict(X_for_prediction).ravel()

            # Store predictions with actual sample IDs
            self.ct_pred_y_pred = y_pred
            self.ct_pred_sample_ids = sample_ids

            # Display results with actual sample IDs
            pred_text = f"Transferred {len(y_pred)} spectra using {self.ct_pred_transfer_model.method.upper()}\\n"
            pred_text += f"Predictions (first 10):\\n"
            for i in range(min(10, len(y_pred))):
                pred_text += f"  {self.ct_pred_sample_ids[i]}: {y_pred[i]:.3f}\\n"

            if len(y_pred) > 10:
                pred_text += f"  ... and {len(y_pred) - 10} more\\n"

            pred_text += f"\\nMean: {y_pred.mean():.3f}, Std: {y_pred.std():.3f}"

            self.ct_prediction_text.config(state='normal')
            self.ct_prediction_text.delete('1.0', tk.END)
            self.ct_prediction_text.insert('1.0', pred_text)
            self.ct_prediction_text.config(state='disabled')

            # Update status
            self._update_ct_section_status('e', True)

            messagebox.showinfo("Success", f"Predictions generated for {len(y_pred)} samples")
        except Exception as e:
            self._update_ct_section_status('e', False)
            messagebox.showerror("Error", f"Failed to predict:\\n{str(e)}")
'''

# ============================================================================
# SUMMARY OF CHANGES
# ============================================================================

IMPLEMENTATION_SUMMARY = """
================================================================================
TAB 9 UX IMPROVEMENTS - IMPLEMENTATION SUMMARY
================================================================================

This file contains all code modifications needed for comprehensive UX
improvements to Tab 9 (Calibration Transfer).

CHANGES REQUIRED:

1. __init__ method (around line 148):
   - Add status tracking variables (ct_section_a_complete, etc.)
   - Add UI reference lists for button control
   - Add status label dictionaries

2. Helper Methods (new methods to add):
   - _create_help_button(): Creates clickable info icons
   - _update_ct_section_status(): Updates status indicators
   - _update_ct_workflow_guide(): Updates workflow colors
   - _update_ct_button_states(): Enable/disable buttons
   - _validate_ct_ds_lambda(): Validates DS parameter
   - _validate_ct_pds_window(): Validates PDS parameter

3. _create_tab9_calibration_transfer() method (replace existing):
   - Add workflow guide at top
   - Add status indicators to each section
   - Add help buttons and tooltips
   - Add parameter validation UI
   - Store button references for enable/disable

4. Updated wrapper methods (replace existing):
   - _load_ct_master_model_ux(): Adds status updates
   - _load_ct_paired_spectra_ux(): Adds status updates
   - _build_ct_transfer_model_ux(): Adds status updates
   - _load_ct_pred_transfer_model_ux(): Adds status updates
   - _load_and_predict_ct_ux(): Adds sample ID parsing

UX FEATURES IMPLEMENTED:

✓ Section status indicators (✓ Complete, ⚠ Required, ○ Pending)
✓ Workflow guide with color-coded steps
✓ Help tooltips for confusing parameters
✓ Inline notes about prerequisites
✓ Parameter validation with visual feedback
✓ Sample ID extraction from filenames
✓ Smart button enable/disable based on workflow state
✓ Color-coded warnings and recommendations

COLOR SCHEME:
- Green (#27AE60): Complete/Valid
- Orange (#E67E22): Required/Warning
- Gray (#95A5A6): Pending/Disabled
- Red (#E74C3C): Error/Invalid

WORKFLOW LOGIC:
- Section A: Always enabled (entry point)
- Section B: Enabled after A complete
- Section C: Enabled after B complete
- Section D: Enabled when instruments registered
- Section E: Enabled when master model + transfer model loaded
"""

print(IMPLEMENTATION_SUMMARY)
