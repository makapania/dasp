"""
PHASE 2: Complete Tab 7 Model Development UI Implementation

This file contains the full UI implementation for Tab 7 (Model Development).
Ready to paste into spectral_predict_gui_optimized.py replacing lines 1313-1387.

Author: Claude
Date: 2025-11-07
"""

# =============================================================================
# MAIN TAB 7 METHOD (replaces lines 1313-1387)
# =============================================================================

def _create_tab7_model_development(self):
    """Tab 7: Model Development - Fresh implementation with full control.

    This tab allows users to:
    - Develop models from scratch with full parameter control
    - Load models from Results tab for refinement
    - Configure all model parameters (type, preprocessing, wavelengths, hyperparameters)
    - Run cross-validation with diagnostic plots
    - Save developed models
    """
    self.tab7 = ttk.Frame(self.notebook, style='TFrame')
    self.notebook.add(self.tab7, text='  🔬 Model Development  ')

    # Create scrollable canvas
    canvas = tk.Canvas(self.tab7, bg=self.colors['bg'], highlightthickness=0)
    scrollbar = ttk.Scrollbar(self.tab7, orient="vertical", command=canvas.yview)
    content_frame = ttk.Frame(canvas, style='TFrame', padding="30")

    content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=content_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # =============================================================================
    # HEADER SECTION
    # =============================================================================

    header_frame = ttk.Frame(content_frame, style='TFrame')
    header_frame.pack(fill='x', pady=(0, 20))

    # Title
    ttk.Label(header_frame, text="Model Development",
              style='Title.TLabel').pack(side='left')

    # Mode indicator (updates when model loaded from Results)
    self.tab7_mode_label = ttk.Label(header_frame, text="Mode: Fresh Development",
                                      style='Heading.TLabel', foreground=self.colors['success'])
    self.tab7_mode_label.pack(side='left', padx=20)

    # Reset button
    ttk.Button(header_frame, text="🔄 Reset to Fresh",
               command=self._tab7_reset_to_fresh,
               style='Modern.TButton').pack(side='right')

    # Configuration info display (shows loaded model info)
    info_frame = ttk.LabelFrame(content_frame, text="Configuration Info", padding=10)
    info_frame.pack(fill='x', pady=(0, 20))

    self.tab7_config_text = tk.Text(info_frame, height=5, width=120,
                                    font=('Consolas', 9), wrap=tk.WORD,
                                    bg='#F0F8FF', relief='flat', state='disabled')
    self.tab7_config_text.pack(fill='x')

    # Initialize with fresh mode message
    self.tab7_config_text.config(state='normal')
    self.tab7_config_text.insert('1.0',
        "🆕 Fresh Development Mode\n"
        "Configure all parameters below to develop a new model from scratch.\n"
        "Or switch to Results tab and click 'Load to Model Development' to refine an existing model.")
    self.tab7_config_text.config(state='disabled')

    # =============================================================================
    # PARAMETER CONTROLS (2-COLUMN LAYOUT)
    # =============================================================================

    params_frame = ttk.LabelFrame(content_frame, text="Model Parameters", padding=15)
    params_frame.pack(fill='both', expand=True, pady=(0, 20))

    # Create left and right columns
    left_col = ttk.Frame(params_frame, style='TFrame')
    left_col.grid(row=0, column=0, sticky='nsew', padx=(0, 20))

    right_col = ttk.Frame(params_frame, style='TFrame')
    right_col.grid(row=0, column=1, sticky='nsew')

    params_frame.columnconfigure(0, weight=1)
    params_frame.columnconfigure(1, weight=1)

    # -----------------------------------------------------------------------------
    # LEFT COLUMN: Core Parameters
    # -----------------------------------------------------------------------------

    row = 0

    # Model Type
    ttk.Label(left_col, text="Model Type:", style='TLabel').grid(
        row=row, column=0, sticky='w', pady=5)
    self.tab7_model_type = tk.StringVar(value='PLS')
    model_combo = ttk.Combobox(left_col, textvariable=self.tab7_model_type,
                               values=['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted'],
                               state='readonly', width=25)
    model_combo.grid(row=row, column=1, sticky='w', pady=5)
    model_combo.bind('<<ComboboxSelected>>', self._tab7_update_hyperparameters)
    row += 1

    # Task Type
    ttk.Label(left_col, text="Task Type:", style='TLabel').grid(
        row=row, column=0, sticky='w', pady=5)
    self.tab7_task_type = tk.StringVar(value='regression')
    task_combo = ttk.Combobox(left_col, textvariable=self.tab7_task_type,
                              values=['regression', 'classification'],
                              state='readonly', width=25)
    task_combo.grid(row=row, column=1, sticky='w', pady=5)
    row += 1

    # Backend
    ttk.Label(left_col, text="Backend:", style='TLabel').grid(
        row=row, column=0, sticky='w', pady=5)
    self.tab7_backend = tk.StringVar(value='python')
    backend_combo = ttk.Combobox(left_col, textvariable=self.tab7_backend,
                                 values=['python'],
                                 state='readonly', width=25)
    backend_combo.grid(row=row, column=1, sticky='w', pady=5)
    row += 1

    # Preprocessing
    ttk.Label(left_col, text="Preprocessing:", style='TLabel').grid(
        row=row, column=0, sticky='w', pady=5)
    self.tab7_preprocessing = tk.StringVar(value='raw')
    preproc_combo = ttk.Combobox(left_col, textvariable=self.tab7_preprocessing,
                                 values=['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2',
                                        'deriv_snv', 'msc', 'msc_sg1', 'msc_sg2', 'deriv_msc'],
                                 state='readonly', width=25)
    preproc_combo.grid(row=row, column=1, sticky='w', pady=5)
    row += 1

    # Window Size (for Savitzky-Golay)
    ttk.Label(left_col, text="Window Size:", style='TLabel').grid(
        row=row, column=0, sticky='w', pady=5)

    window_frame = ttk.Frame(left_col, style='TFrame')
    window_frame.grid(row=row, column=1, sticky='w', pady=5)

    self.tab7_window_size = tk.IntVar(value=11)
    for i, val in enumerate([7, 11, 17, 19]):
        ttk.Radiobutton(window_frame, text=str(val), variable=self.tab7_window_size,
                       value=val).grid(row=0, column=i, padx=5)
    row += 1

    # CV Folds
    ttk.Label(left_col, text="CV Folds:", style='TLabel').grid(
        row=row, column=0, sticky='w', pady=5)
    self.tab7_cv_folds = tk.IntVar(value=5)
    cv_spin = ttk.Spinbox(left_col, from_=3, to=10, textvariable=self.tab7_cv_folds,
                          width=23)
    cv_spin.grid(row=row, column=1, sticky='w', pady=5)
    row += 1

    # -----------------------------------------------------------------------------
    # RIGHT COLUMN: Wavelength Specification
    # -----------------------------------------------------------------------------

    row = 0

    # Wavelength specification label with format info
    wl_label_text = ("Wavelength Specification:\n"
                    "(One wavelength per line, or ranges like '400-700')")
    ttk.Label(right_col, text=wl_label_text, style='TLabel',
              justify='left').grid(row=row, column=0, columnspan=2, sticky='w', pady=(0, 5))
    row += 1

    # Wavelength text widget
    wl_text_frame = ttk.Frame(right_col, style='TFrame')
    wl_text_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(0, 10))

    self.tab7_wl_spec = tk.Text(wl_text_frame, height=4, width=40,
                                font=('Consolas', 10), wrap=tk.WORD)
    self.tab7_wl_spec.pack(side='left', fill='both', expand=True)

    wl_scrollbar = ttk.Scrollbar(wl_text_frame, orient='vertical',
                                 command=self.tab7_wl_spec.yview)
    wl_scrollbar.pack(side='right', fill='y')
    self.tab7_wl_spec.config(yscrollcommand=wl_scrollbar.set)

    # Bind update event for real-time wavelength count
    self.tab7_wl_spec.bind('<KeyRelease>', self._tab7_update_wavelength_count)
    row += 1

    # Wavelength preset buttons
    preset_frame = ttk.Frame(right_col, style='TFrame')
    preset_frame.grid(row=row, column=0, columnspan=2, sticky='w', pady=(0, 10))

    ttk.Label(preset_frame, text="Presets:", style='TLabel').pack(side='left', padx=(0, 10))
    ttk.Button(preset_frame, text="All",
               command=lambda: self._tab7_apply_wl_preset('all'),
               style='Modern.TButton').pack(side='left', padx=2)
    ttk.Button(preset_frame, text="NIR (1000-2500)",
               command=lambda: self._tab7_apply_wl_preset('nir'),
               style='Modern.TButton').pack(side='left', padx=2)
    ttk.Button(preset_frame, text="Visible (400-700)",
               command=lambda: self._tab7_apply_wl_preset('visible'),
               style='Modern.TButton').pack(side='left', padx=2)
    ttk.Button(preset_frame, text="Custom Range...",
               command=self._tab7_custom_range_dialog,
               style='Modern.TButton').pack(side='left', padx=2)
    row += 1

    # Wavelength count display
    self.tab7_wl_count_label = ttk.Label(right_col, text="Wavelengths: 0",
                                         style='Subheading.TLabel')
    self.tab7_wl_count_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(0, 10))
    row += 1

    # -----------------------------------------------------------------------------
    # Dynamic Hyperparameter Frame
    # -----------------------------------------------------------------------------

    self.tab7_hyperparam_container = ttk.LabelFrame(right_col, text="Hyperparameters",
                                                    padding=10)
    self.tab7_hyperparam_container.grid(row=row, column=0, columnspan=2,
                                        sticky='ew', pady=(10, 0))

    # This frame will be populated dynamically based on model type
    self.tab7_hyperparam_frame = ttk.Frame(self.tab7_hyperparam_container, style='TFrame')
    self.tab7_hyperparam_frame.pack(fill='both', expand=True)

    # Dictionary to store hyperparameter widgets for easy access
    self.tab7_hyperparam_widgets = {}

    # Initialize hyperparameters for default model (PLS)
    self._tab7_create_hyperparam_widgets('PLS')

    # =============================================================================
    # ACTION BUTTONS
    # =============================================================================

    action_frame = ttk.Frame(content_frame, style='TFrame')
    action_frame.pack(fill='x', pady=20)

    # Run Model button
    self.tab7_run_btn = ttk.Button(action_frame, text="▶ Run Model Development",
                                   command=self._tab7_run_model,
                                   style='Accent.TButton')
    self.tab7_run_btn.pack(side='left', padx=(0, 10))

    # Save Model button (disabled initially)
    self.tab7_save_btn = ttk.Button(action_frame, text="💾 Save Model",
                                    command=self._tab7_save_model,
                                    style='Modern.TButton', state='disabled')
    self.tab7_save_btn.pack(side='left')

    # =============================================================================
    # RESULTS DISPLAY
    # =============================================================================

    results_frame = ttk.LabelFrame(content_frame, text="Results", padding=10)
    results_frame.pack(fill='both', expand=True, pady=(0, 20))

    self.tab7_results_text = tk.Text(results_frame, height=8, width=120,
                                     font=('Consolas', 10), wrap=tk.WORD,
                                     bg='#FFFFFF', relief='solid', borderwidth=1)
    self.tab7_results_text.pack(fill='both', expand=True)

    # Initialize with placeholder
    self.tab7_results_text.insert('1.0',
        "No results yet. Configure parameters above and click 'Run Model Development'.")
    self.tab7_results_text.config(state='disabled')

    # =============================================================================
    # DIAGNOSTIC PLOTS (3 side-by-side)
    # =============================================================================

    plots_frame = ttk.LabelFrame(content_frame, text="Diagnostic Plots", padding=10)
    plots_frame.pack(fill='both', expand=True, pady=(0, 20))

    plots_container = ttk.Frame(plots_frame, style='TFrame')
    plots_container.pack(fill='both', expand=True)

    # Configure columns for equal width
    plots_container.columnconfigure(0, weight=1)
    plots_container.columnconfigure(1, weight=1)
    plots_container.columnconfigure(2, weight=1)

    # Plot 1: Prediction (Observed vs Predicted)
    plot1_frame = ttk.LabelFrame(plots_container, text="Prediction Plot", padding=5)
    plot1_frame.grid(row=0, column=0, sticky='nsew', padx=5)

    self.tab7_plot1_frame = ttk.Frame(plot1_frame, style='TFrame',
                                      width=300, height=250)
    self.tab7_plot1_frame.pack(fill='both', expand=True)
    self.tab7_plot1_frame.pack_propagate(False)

    # Placeholder label
    ttk.Label(self.tab7_plot1_frame,
              text="Prediction plot will appear here\nafter running model",
              style='Caption.TLabel', justify='center').place(relx=0.5, rely=0.5, anchor='center')

    # Plot 2: Residuals
    plot2_frame = ttk.LabelFrame(plots_container, text="Residual Diagnostics", padding=5)
    plot2_frame.grid(row=0, column=1, sticky='nsew', padx=5)

    self.tab7_plot2_frame = ttk.Frame(plot2_frame, style='TFrame',
                                      width=300, height=250)
    self.tab7_plot2_frame.pack(fill='both', expand=True)
    self.tab7_plot2_frame.pack_propagate(False)

    ttk.Label(self.tab7_plot2_frame,
              text="Residual plot will appear here\nafter running model",
              style='Caption.TLabel', justify='center').place(relx=0.5, rely=0.5, anchor='center')

    # Plot 3: Model Comparison
    plot3_frame = ttk.LabelFrame(plots_container, text="Model Comparison", padding=5)
    plot3_frame.grid(row=0, column=2, sticky='nsew', padx=5)

    self.tab7_plot3_frame = ttk.Frame(plot3_frame, style='TFrame',
                                      width=300, height=250)
    self.tab7_plot3_frame.pack(fill='both', expand=True)
    self.tab7_plot3_frame.pack_propagate(False)

    ttk.Label(self.tab7_plot3_frame,
              text="Comparison plot will appear here\nafter multiple runs",
              style='Caption.TLabel', justify='center').place(relx=0.5, rely=0.5, anchor='center')

    # =============================================================================
    # STATUS LABEL
    # =============================================================================

    self.tab7_status = ttk.Label(content_frame, text="Ready. Configure parameters and run model.",
                                 style='Caption.TLabel')
    self.tab7_status.pack(pady=10)

    # Initialize run history for comparison plot
    self.tab7_run_history = []


# =============================================================================
# HELPER METHODS (add these after _create_tab7_model_development)
# =============================================================================

def _tab7_create_hyperparam_widgets(self, model_type):
    """Create hyperparameter widgets based on model type.

    Args:
        model_type: One of 'PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted'
    """
    # Clear existing widgets
    for widget in self.tab7_hyperparam_frame.winfo_children():
        widget.destroy()
    self.tab7_hyperparam_widgets.clear()

    row = 0

    if model_type == 'PLS':
        # n_components (1-50)
        ttk.Label(self.tab7_hyperparam_frame, text="n_components:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['n_components'] = tk.IntVar(value=10)
        ttk.Spinbox(self.tab7_hyperparam_frame, from_=1, to=50,
                   textvariable=self.tab7_hyperparam_widgets['n_components'],
                   width=20).grid(row=row, column=1, sticky='w', pady=5)

    elif model_type in ['Ridge', 'Lasso']:
        # alpha (float entry)
        ttk.Label(self.tab7_hyperparam_frame, text="alpha:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['alpha'] = tk.StringVar(value='1.0')
        ttk.Entry(self.tab7_hyperparam_frame,
                 textvariable=self.tab7_hyperparam_widgets['alpha'],
                 width=22).grid(row=row, column=1, sticky='w', pady=5)

    elif model_type == 'RandomForest':
        # n_estimators
        ttk.Label(self.tab7_hyperparam_frame, text="n_estimators:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['n_estimators'] = tk.IntVar(value=100)
        ttk.Spinbox(self.tab7_hyperparam_frame, from_=10, to=500,
                   textvariable=self.tab7_hyperparam_widgets['n_estimators'],
                   width=20).grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # max_depth
        ttk.Label(self.tab7_hyperparam_frame, text="max_depth:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['max_depth'] = tk.StringVar(value='None')
        ttk.Entry(self.tab7_hyperparam_frame,
                 textvariable=self.tab7_hyperparam_widgets['max_depth'],
                 width=22).grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # max_features
        ttk.Label(self.tab7_hyperparam_frame, text="max_features:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['max_features'] = tk.StringVar(value='sqrt')
        ttk.Combobox(self.tab7_hyperparam_frame,
                    textvariable=self.tab7_hyperparam_widgets['max_features'],
                    values=['auto', 'sqrt', 'log2'], state='readonly',
                    width=20).grid(row=row, column=1, sticky='w', pady=5)

    elif model_type == 'MLP':
        # hidden_layer_sizes
        ttk.Label(self.tab7_hyperparam_frame, text="hidden_layer_sizes:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['hidden_layer_sizes'] = tk.StringVar(value='(100,)')
        ttk.Entry(self.tab7_hyperparam_frame,
                 textvariable=self.tab7_hyperparam_widgets['hidden_layer_sizes'],
                 width=22).grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # learning_rate_init
        ttk.Label(self.tab7_hyperparam_frame, text="learning_rate_init:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['learning_rate_init'] = tk.StringVar(value='0.001')
        ttk.Entry(self.tab7_hyperparam_frame,
                 textvariable=self.tab7_hyperparam_widgets['learning_rate_init'],
                 width=22).grid(row=row, column=1, sticky='w', pady=5)

    elif model_type == 'NeuralBoosted':
        # n_estimators
        ttk.Label(self.tab7_hyperparam_frame, text="n_estimators:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['n_estimators'] = tk.IntVar(value=50)
        ttk.Spinbox(self.tab7_hyperparam_frame, from_=10, to=200,
                   textvariable=self.tab7_hyperparam_widgets['n_estimators'],
                   width=20).grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # learning_rate
        ttk.Label(self.tab7_hyperparam_frame, text="learning_rate:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['learning_rate'] = tk.StringVar(value='0.1')
        ttk.Entry(self.tab7_hyperparam_frame,
                 textvariable=self.tab7_hyperparam_widgets['learning_rate'],
                 width=22).grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # hidden_layer_size
        ttk.Label(self.tab7_hyperparam_frame, text="hidden_layer_size:",
                  style='TLabel').grid(row=row, column=0, sticky='w', pady=5)
        self.tab7_hyperparam_widgets['hidden_layer_size'] = tk.IntVar(value=10)
        ttk.Spinbox(self.tab7_hyperparam_frame, from_=3, to=20,
                   textvariable=self.tab7_hyperparam_widgets['hidden_layer_size'],
                   width=20).grid(row=row, column=1, sticky='w', pady=5)

    # Add help text
    row += 1
    help_texts = {
        'PLS': 'Number of PLS components to use (more = more complex)',
        'Ridge': 'Regularization strength (higher = more regularization)',
        'Lasso': 'Regularization strength (higher = more regularization)',
        'RandomForest': 'Configure ensemble parameters (None = unlimited)',
        'MLP': 'Format: (layer1, layer2, ...) e.g., (100,50)',
        'NeuralBoosted': 'Boosted neural network ensemble parameters'
    }

    if model_type in help_texts:
        ttk.Label(self.tab7_hyperparam_frame, text=help_texts[model_type],
                  style='Caption.TLabel', wraplength=250).grid(
                      row=row, column=0, columnspan=2, sticky='w', pady=(10, 0))


def _tab7_update_hyperparameters(self, event=None):
    """Callback when model type changes - recreate hyperparameter widgets."""
    model_type = self.tab7_model_type.get()
    self._tab7_create_hyperparam_widgets(model_type)
    self.tab7_status.config(text=f"Model type changed to {model_type}. Configure hyperparameters.")


def _tab7_update_wavelength_count(self, event=None):
    """Update wavelength count label in real-time."""
    try:
        wl_text = self.tab7_wl_spec.get('1.0', 'end').strip()

        if not wl_text:
            self.tab7_wl_count_label.config(text="Wavelengths: 0")
            return

        # Parse wavelengths (simplified - Phase 4 will have full parser)
        wavelengths = []
        lines = wl_text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for range (e.g., "400-700")
            if '-' in line and not line.startswith('-'):
                parts = line.split('-')
                if len(parts) == 2:
                    try:
                        start = float(parts[0].strip())
                        end = float(parts[1].strip())
                        # Estimate count (assume 1nm intervals)
                        count = int(end - start) + 1
                        wavelengths.extend(range(int(start), int(end) + 1))
                    except:
                        pass
            else:
                # Single wavelength
                try:
                    wl = float(line)
                    wavelengths.append(wl)
                except:
                    pass

        # Remove duplicates
        wavelengths = sorted(set(wavelengths))
        count = len(wavelengths)

        self.tab7_wl_count_label.config(text=f"Wavelengths: {count}")

    except Exception as e:
        self.tab7_wl_count_label.config(text="Wavelengths: Error")


def _tab7_apply_wl_preset(self, preset):
    """Apply wavelength presets.

    Args:
        preset: One of 'all', 'nir', 'visible'
    """
    self.tab7_wl_spec.delete('1.0', 'end')

    if preset == 'all':
        # Use all available wavelengths from loaded data
        if self.X_original is not None:
            wavelengths = [str(int(wl)) for wl in self.X_original.columns[:50]]  # First 50 for display
            self.tab7_wl_spec.insert('1.0', '\n'.join(wavelengths) + '\n...')
            self.tab7_status.config(text="All wavelengths selected (from loaded data)")
        else:
            self.tab7_wl_spec.insert('1.0', 'Load data first in Tab 1 to use this preset')
            self.tab7_status.config(text="No data loaded. Load data in Tab 1 first.")

    elif preset == 'nir':
        self.tab7_wl_spec.insert('1.0', '1000-2500')
        self.tab7_status.config(text="NIR range (1000-2500 nm) selected")

    elif preset == 'visible':
        self.tab7_wl_spec.insert('1.0', '400-700')
        self.tab7_status.config(text="Visible range (400-700 nm) selected")

    # Update count
    self._tab7_update_wavelength_count()


def _tab7_custom_range_dialog(self):
    """Show dialog for custom wavelength range."""
    dialog = tk.Toplevel(self.root)
    dialog.title("Custom Wavelength Range")
    dialog.geometry("400x200")
    dialog.configure(bg=self.colors['bg'])

    # Center dialog
    dialog.transient(self.root)
    dialog.grab_set()

    # Content
    frame = ttk.Frame(dialog, style='TFrame', padding=20)
    frame.pack(fill='both', expand=True)

    ttk.Label(frame, text="Enter Wavelength Range",
              style='Heading.TLabel').pack(pady=(0, 20))

    # Start wavelength
    start_frame = ttk.Frame(frame, style='TFrame')
    start_frame.pack(fill='x', pady=5)
    ttk.Label(start_frame, text="Start (nm):", style='TLabel').pack(side='left', padx=(0, 10))
    start_var = tk.StringVar(value='400')
    ttk.Entry(start_frame, textvariable=start_var, width=15).pack(side='left')

    # End wavelength
    end_frame = ttk.Frame(frame, style='TFrame')
    end_frame.pack(fill='x', pady=5)
    ttk.Label(end_frame, text="End (nm):", style='TLabel').pack(side='left', padx=(0, 10))
    end_var = tk.StringVar(value='700')
    ttk.Entry(end_frame, textvariable=end_var, width=15).pack(side='left')

    # Buttons
    btn_frame = ttk.Frame(frame, style='TFrame')
    btn_frame.pack(pady=20)

    def apply():
        try:
            start = float(start_var.get())
            end = float(end_var.get())

            if start >= end:
                messagebox.showerror("Invalid Range",
                                   "Start wavelength must be less than end wavelength.")
                return

            self.tab7_wl_spec.delete('1.0', 'end')
            self.tab7_wl_spec.insert('1.0', f'{int(start)}-{int(end)}')
            self._tab7_update_wavelength_count()
            self.tab7_status.config(text=f"Custom range ({int(start)}-{int(end)} nm) applied")
            dialog.destroy()

        except ValueError:
            messagebox.showerror("Invalid Input",
                               "Please enter valid numbers for wavelengths.")

    ttk.Button(btn_frame, text="Apply", command=apply,
               style='Accent.TButton').pack(side='left', padx=5)
    ttk.Button(btn_frame, text="Cancel", command=dialog.destroy,
               style='Modern.TButton').pack(side='left', padx=5)


def _tab7_reset_to_fresh(self):
    """Reset all parameters to defaults for fresh development."""
    # Confirm reset
    if not messagebox.askyesno("Reset to Fresh",
                               "This will reset all parameters to defaults. Continue?"):
        return

    # Reset all parameters
    self.tab7_model_type.set('PLS')
    self.tab7_task_type.set('regression')
    self.tab7_backend.set('python')
    self.tab7_preprocessing.set('raw')
    self.tab7_window_size.set(11)
    self.tab7_cv_folds.set(5)
    self.tab7_wl_spec.delete('1.0', 'end')

    # Reset mode label
    self.tab7_mode_label.config(text="Mode: Fresh Development",
                                foreground=self.colors['success'])

    # Reset config text
    self.tab7_config_text.config(state='normal')
    self.tab7_config_text.delete('1.0', 'end')
    self.tab7_config_text.insert('1.0',
        "🆕 Fresh Development Mode\n"
        "Configure all parameters below to develop a new model from scratch.\n"
        "Or switch to Results tab and click 'Load to Model Development' to refine an existing model.")
    self.tab7_config_text.config(state='disabled')

    # Reset results
    self.tab7_results_text.config(state='normal')
    self.tab7_results_text.delete('1.0', 'end')
    self.tab7_results_text.insert('1.0',
        "No results yet. Configure parameters above and click 'Run Model Development'.")
    self.tab7_results_text.config(state='disabled')

    # Disable save button
    self.tab7_save_btn.config(state='disabled')

    # Recreate hyperparameters for PLS
    self._tab7_create_hyperparam_widgets('PLS')

    # Update counts
    self._tab7_update_wavelength_count()

    # Clear run history
    self.tab7_run_history = []

    # Update status
    self.tab7_status.config(text="Reset to fresh development mode. Configure parameters and run model.")


def _tab7_run_model(self):
    """Run model development with current parameters.

    Phase 4 will implement full execution logic. For now, show placeholder.
    """
    # Validate data is loaded
    if self.X is None or self.y is None:
        messagebox.showerror("No Data",
                           "Please load data in Tab 1 first before running model development.")
        return

    # Validate wavelengths specified
    wl_text = self.tab7_wl_spec.get('1.0', 'end').strip()
    if not wl_text:
        messagebox.showerror("No Wavelengths",
                           "Please specify wavelengths before running model.")
        return

    # Get parameters
    model_type = self.tab7_model_type.get()
    task_type = self.tab7_task_type.get()
    preprocessing = self.tab7_preprocessing.get()
    cv_folds = self.tab7_cv_folds.get()

    # Show placeholder message
    self.tab7_status.config(text="Phase 4 will implement model execution...")

    messagebox.showinfo("Phase 4 Coming Soon",
        f"Model Development Execution (Phase 4)\n\n"
        f"Configuration captured:\n"
        f"  • Model: {model_type}\n"
        f"  • Task: {task_type}\n"
        f"  • Preprocessing: {preprocessing}\n"
        f"  • CV Folds: {cv_folds}\n\n"
        f"Phase 4 will implement:\n"
        f"  ✓ Full model training with cross-validation\n"
        f"  ✓ Diagnostic plot generation\n"
        f"  ✓ Results display\n"
        f"  ✓ Model comparison across runs\n\n"
        f"UI is complete and ready for Phase 4 integration!")


def _tab7_save_model(self):
    """Save developed model to .dasp file.

    Phase 4+ will implement model saving.
    """
    messagebox.showinfo("Not Implemented",
        "Model saving will be implemented in Phase 4 after execution is complete.\n\n"
        "This will save the developed model to a .dasp file that can be loaded\n"
        "in Tab 8 (Model Prediction) for inference on new data.")


# =============================================================================
# END OF TAB 7 IMPLEMENTATION
# =============================================================================

"""
INTEGRATION INSTRUCTIONS:

1. Open spectral_predict_gui_optimized.py

2. REPLACE lines 1313-1387 with the `_create_tab7_model_development` method above

3. ADD the 8 helper methods after the main method (around line 1390):
   - _tab7_create_hyperparam_widgets
   - _tab7_update_hyperparameters
   - _tab7_update_wavelength_count
   - _tab7_apply_wl_preset
   - _tab7_custom_range_dialog
   - _tab7_reset_to_fresh
   - _tab7_run_model
   - _tab7_save_model

4. Save and test!

TESTING CHECKLIST:
[ ] Tab 7 displays with all UI elements
[ ] Model type dropdown changes hyperparameters dynamically
[ ] Wavelength presets work (All, NIR, Visible, Custom)
[ ] Wavelength count updates in real-time as you type
[ ] Custom range dialog opens and applies ranges
[ ] Reset button clears all parameters
[ ] Run button shows Phase 4 placeholder message
[ ] All 3 plot frames display with placeholders
[ ] Status label updates with actions

NEXT PHASES:
- Phase 3: Implement loading models from Results tab
- Phase 4: Implement full execution engine with CV and plots
- Phase 5: Implement model saving
"""
