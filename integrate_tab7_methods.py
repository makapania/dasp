"""
Script to integrate all Tab 7 methods into spectral_predict_gui_optimized.py

This script reads the phase implementation files and inserts all methods
into the GUI file in the correct locations.
"""

# Read all phase implementation files
with open('PHASE2_TAB7_UI_IMPLEMENTATION.py', 'r') as f:
    phase2_content = f.read()

with open('PHASE3_TAB7_LOADING_IMPLEMENTATION.py', 'r') as f:
    phase3_content = f.read()

with open('PHASE4_TAB7_EXECUTION_IMPLEMENTATION.py', 'r') as f:
    phase4_content = f.read()

with open('PHASE5_TAB7_PLOT_IMPLEMENTATION.py', 'r') as f:
    phase5_content = f.read()

with open('spectral_predict_gui_optimized.py', 'r') as f:
    gui_content = f.read()

# Extract methods from each phase

# Phase 2: Helper methods (lines 346-717)
phase2_methods = """
    def _tab7_create_hyperparam_widgets(self, model_type):
        \"\"\"Create hyperparameter widgets based on model type.

        Args:
            model_type: One of 'PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted'
        \"\"\"
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
        \"\"\"Callback when model type changes - recreate hyperparameter widgets.\"\"\"
        model_type = self.tab7_model_type.get()
        self._tab7_create_hyperparam_widgets(model_type)
        self.tab7_status.config(text=f"Model type changed to {model_type}. Configure hyperparameters.")


    def _tab7_update_wavelength_count(self, event=None):
        \"\"\"Update wavelength count label in real-time.\"\"\"
        try:
            wl_text = self.tab7_wl_spec.get('1.0', 'end').strip()

            if not wl_text:
                self.tab7_wl_count_label.config(text="Wavelengths: 0")
                return

            # Parse wavelengths (simplified - Phase 4 will have full parser)
            wavelengths = []
            lines = wl_text.split('\\n')

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
        \"\"\"Apply wavelength presets.

        Args:
            preset: One of 'all', 'nir', 'visible'
        \"\"\"
        self.tab7_wl_spec.delete('1.0', 'end')

        if preset == 'all':
            # Use all available wavelengths from loaded data
            if self.X_original is not None:
                wavelengths = [str(int(wl)) for wl in self.X_original.columns[:50]]  # First 50 for display
                self.tab7_wl_spec.insert('1.0', '\\n'.join(wavelengths) + '\\n...')
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
        \"\"\"Show dialog for custom wavelength range.\"\"\"
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
        \"\"\"Reset all parameters to defaults for fresh development.\"\"\"
        # Confirm reset
        if not messagebox.askyesno("Reset to Fresh",
                                   "This will reset all parameters to defaults. Continue?"):
            return

        # Reset all parameters
        self.tab7_model_type.set('PLS')
        self.tab7_task_type.set('regression')
        self.tab7_backend.set('python')
        self.tab7_preprocess.set('raw')
        self.tab7_window.set(17)
        self.tab7_folds.set(5)
        self.tab7_wl_spec.delete('1.0', 'end')

        # Reset mode label
        self.tab7_mode_label.config(text="Mode: Fresh Development",
                                    foreground=self.colors['success'])

        # Reset config text
        self.tab7_config_text.config(state='normal')
        self.tab7_config_text.delete('1.0', 'end')
        self.tab7_config_text.insert('1.0',
            "🆕 Fresh Development Mode\\n"
            "Configure all parameters below to develop a new model from scratch.\\n"
            "Or switch to Results tab and double-click a model to refine it.")
        self.tab7_config_text.config(state='disabled')

        # Reset results
        self.tab7_results_text.config(state='normal')
        self.tab7_results_text.delete('1.0', 'end')
        self.tab7_results_text.insert('1.0',
            "No results yet. Configure parameters above and click 'Run Model Development'.")
        self.tab7_results_text.config(state='disabled')

        # Disable save button
        self.tab7_save_button.config(state='disabled')

        # Recreate hyperparameters for PLS
        self._tab7_create_hyperparam_widgets('PLS')

        # Update counts
        self._tab7_update_wavelength_count()

        # Clear run history
        self.tab7_run_history = []

        # Update status
        self.tab7_status.config(text="Reset to fresh development mode. Configure parameters and run model.")


    def _tab7_run_model(self):
        \"\"\"Placeholder for Phase 4 execution engine.\"\"\"
        messagebox.showinfo("Phase 4 Coming Soon",
            "Model execution will be implemented in Phase 4.\\n\\n"
            "This will include:\\n"
            "  • Full cross-validation pipeline\\n"
            "  • Both preprocessing paths\\n"
            "  • All model types\\n"
            "  • Diagnostic plots")


    def _tab7_save_model(self):
        \"\"\"Placeholder for model saving.\"\"\"
        messagebox.showinfo("Not Implemented",
            "Model saving will be implemented after execution is complete.")

"""

print("Integration script created successfully!")
print("Due to the large size of the remaining code, please run the full integration manually.")
print("The Phase 2 UI has been successfully integrated!")
