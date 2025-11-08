#!/usr/bin/env python3
"""
Integration script for Tab 7 Model Development

This script integrates all agent deliverables into spectral_predict_gui_optimized.py:
- Agent 2: Tab 7 UI
- Agent 3: Model loading engine
- Agent 4: Execution engine
- Agent 5: Diagnostic plots

It also:
- Renames current Tab 7 (Model Prediction) to Tab 8
- Updates Results tab double-click to use new Tab 7
- Fixes n_folds field in Python backend
"""

import re
import shutil
from pathlib import Path
from datetime import datetime

def backup_file(filepath):
    """Create backup of file before modification."""
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ Backup created: {backup_path}")
    return backup_path

def read_file(filepath):
    """Read file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Updated: {filepath}")

def integrate_gui_changes():
    """Integrate Tab 7 into GUI file."""
    gui_file = Path('spectral_predict_gui_optimized.py')

    if not gui_file.exists():
        print(f"❌ Error: {gui_file} not found!")
        return False

    print(f"\n{'='*80}")
    print(f"INTEGRATING TAB 7 MODEL DEVELOPMENT INTO GUI")
    print(f"{'='*80}\n")

    # Create backup
    backup_path = backup_file(gui_file)

    # Read current content
    content = read_file(gui_file)

    # Step 1: Rename Tab 7 to Tab 8 in _create_ui()
    print("\n[Step 1/6] Renaming Tab 7 (Model Prediction) to Tab 8...")

    # Update the method call in _create_ui()
    content = content.replace(
        'self._create_tab7_model_prediction()',
        'self._create_tab8_model_prediction()'
    )

    # Update the method definition
    content = content.replace(
        'def _create_tab7_model_prediction(self):',
        'def _create_tab8_model_prediction(self):'
    )

    # Update references to self.tab7 in the old prediction tab
    # We need to be careful - only replace in the Model Prediction section
    # Find the Model Prediction method and replace within it
    tab7_pred_start = content.find('def _create_tab8_model_prediction(self):')
    if tab7_pred_start != -1:
        # Find the next method definition
        tab7_pred_end = content.find('\n    def _', tab7_pred_start + 100)
        if tab7_pred_end == -1:
            tab7_pred_end = len(content)

        # Replace self.tab7 with self.tab8 in this section only
        section = content[tab7_pred_start:tab7_pred_end]
        section = section.replace('self.tab7', 'self.tab8')
        content = content[:tab7_pred_start] + section + content[tab7_pred_end:]

    print("  ✓ Tab 7 renamed to Tab 8")

    # Step 2: Add new Tab 7 creation call in _create_ui()
    print("\n[Step 2/6] Adding Tab 7 Model Development creation call...")

    # Find the _create_ui method and add our new tab call
    create_ui_pattern = r'(self\._create_tab6_refine_model\(\)\s*\n)'
    replacement = r'\1        self._create_tab7_model_development()  # NEW: Fresh Model Development\n'
    content = re.sub(create_ui_pattern, replacement, content)

    print("  ✓ Tab 7 creation call added")

    # Step 3: Insert Tab 7 UI code after Tab 6
    print("\n[Step 3/6] Inserting Tab 7 UI code (Agent 2)...")

    # Find insertion point (after Tab 6, before helper methods section)
    insertion_point_pattern = r'(self\.refine_status\.grid\(row=row, column=0, columnspan=2\)\s*\n\s*# === Helper Methods ===)'

    tab7_ui_code = '''

    # ====================================================================
    # TAB 7: Model Development (Fresh Implementation)
    # ====================================================================

    def _create_tab7_model_development(self):
        """Tab 7: Model Development - Complete control over model parameters with live refinement."""
        self.tab7 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab7, text='  🔬 Model Development  ')

        # Initialize storage variables for Tab 7
        self.tab7_loaded_config = None  # Config loaded from Results tab
        self.tab7_fitted_model = None  # Fitted model after execution
        self.tab7_fitted_preprocessor = None  # Fitted preprocessor
        self.tab7_performance = None  # Performance metrics dict
        self.tab7_wavelengths = None  # Wavelengths used in model
        self.tab7_config = None  # Full configuration dict
        self.tab7_y_true = None  # True values from CV
        self.tab7_y_pred = None  # Predicted values from CV
        self.tab7_X_cv = None  # X data for leverage calculation

        # Create scrollable content
        canvas = tk.Canvas(self.tab7, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab7, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas, style='TFrame', padding="30")

        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Title
        ttk.Label(content_frame, text="Model Development", style='Title.TLabel').grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1

        # Instructions
        ttk.Label(content_frame,
            text="Build and refine models with full control over all parameters. Load results from the Results tab or start fresh.",
            style='Caption.TLabel', wraplength=1200).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 25))
        row += 1

        # ===================================================================
        # Mode & Info Display
        # ===================================================================
        mode_frame = ttk.LabelFrame(content_frame, text="Development Mode", padding=15)
        mode_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        row += 1

        mode_control_frame = ttk.Frame(mode_frame)
        mode_control_frame.pack(fill='x')

        self.tab7_mode_label = ttk.Label(mode_control_frame,
            text="Status: Fresh Development",
            style='Subheading.TLabel', foreground='#2E7D32')
        self.tab7_mode_label.pack(side='left', padx=5)

        ttk.Button(mode_control_frame, text="Reset to Fresh",
                   command=self._tab7_reset_to_fresh).pack(side='right', padx=5)

        # Configuration info
        ttk.Label(content_frame, text="Configuration Information", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(15, 5))
        row += 1

        config_frame = ttk.LabelFrame(content_frame, text="Loaded Model Details", padding=15)
        config_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        row += 1

        self.tab7_config_text = tk.Text(config_frame, height=6, width=100, font=('Consolas', 9),
                                         bg='#F5F5F5', fg=self.colors['text'], wrap=tk.WORD,
                                         relief='flat', borderwidth=0)
        self.tab7_config_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.tab7_config_text.insert('1.0', "No model loaded. Configure parameters below to start fresh development.")
        self.tab7_config_text.config(state='disabled')

        # ===================================================================
        # Adjustable Parameters
        # ===================================================================
        ttk.Label(content_frame, text="Model Configuration", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        row += 1

        params_outer_frame = ttk.LabelFrame(content_frame, text="Adjustable Parameters", padding=20)
        params_outer_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        row += 1

        # Two-column layout
        params_left = ttk.Frame(params_outer_frame)
        params_left.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E), padx=(0, 20))

        params_right = ttk.Frame(params_outer_frame)
        params_right.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E))

        params_outer_frame.columnconfigure(0, weight=1)
        params_outer_frame.columnconfigure(1, weight=1)

        # LEFT COLUMN
        left_row = 0

        # Model Type
        ttk.Label(params_left, text="Model Type:", style='Subheading.TLabel').grid(
            row=left_row, column=0, sticky=tk.W, pady=(0, 5))
        left_row += 1

        self.tab7_model_type = tk.StringVar(value='PLS')
        model_combo = ttk.Combobox(params_left, textvariable=self.tab7_model_type,
                                   width=25, state='readonly')
        model_combo['values'] = ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']
        model_combo.grid(row=left_row, column=0, sticky=tk.W, pady=(0, 15))
        model_combo.bind('<<ComboboxSelected>>', self._tab7_update_hyperparameters)
        left_row += 1

        # Task Type
        ttk.Label(params_left, text="Task Type:", style='Subheading.TLabel').grid(
            row=left_row, column=0, sticky=tk.W, pady=(0, 5))
        left_row += 1

        self.tab7_task_type = tk.StringVar(value='regression')
        task_frame = ttk.Frame(params_left)
        task_frame.grid(row=left_row, column=0, sticky=tk.W, pady=(0, 15))
        ttk.Radiobutton(task_frame, text="Regression",
                        variable=self.tab7_task_type, value='regression').pack(side='left', padx=(0, 10))
        ttk.Radiobutton(task_frame, text="Classification",
                        variable=self.tab7_task_type, value='classification').pack(side='left')
        left_row += 1

        # Backend
        ttk.Label(params_left, text="Backend:", style='Subheading.TLabel').grid(
            row=left_row, column=0, sticky=tk.W, pady=(0, 5))
        left_row += 1

        self.tab7_backend = tk.StringVar(value='python')
        backend_frame = ttk.Frame(params_left)
        backend_frame.grid(row=left_row, column=0, sticky=tk.W, pady=(0, 15))
        ttk.Radiobutton(backend_frame, text="Python",
                        variable=self.tab7_backend, value='python').pack(side='left', padx=(0, 10))
        ttk.Radiobutton(backend_frame, text="Julia",
                        variable=self.tab7_backend, value='julia').pack(side='left')
        left_row += 1

        # Preprocessing
        ttk.Label(params_left, text="Preprocessing:", style='Subheading.TLabel').grid(
            row=left_row, column=0, sticky=tk.W, pady=(0, 5))
        left_row += 1

        self.tab7_preprocess = tk.StringVar(value='raw')
        preprocess_combo = ttk.Combobox(params_left, textvariable=self.tab7_preprocess,
                                        width=25, state='readonly')
        preprocess_combo['values'] = ['raw', 'snv', 'msc', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2',
                                       'deriv_snv', 'deriv_msc', 'msc_sg1', 'msc_sg2']
        preprocess_combo.grid(row=left_row, column=0, sticky=tk.W, pady=(0, 15))
        left_row += 1

        # Window Size
        ttk.Label(params_left, text="Window Size:", style='Subheading.TLabel').grid(
            row=left_row, column=0, sticky=tk.W, pady=(0, 5))
        left_row += 1

        self.tab7_window = tk.IntVar(value=17)
        window_frame = ttk.Frame(params_left)
        window_frame.grid(row=left_row, column=0, sticky=tk.W, pady=(0, 15))
        for w in [7, 11, 17, 19, 21, 23, 25]:
            ttk.Radiobutton(window_frame, text=f"{w}",
                            variable=self.tab7_window, value=w).pack(side='left', padx=3)
        left_row += 1

        # CV Folds
        ttk.Label(params_left, text="CV Folds:", style='Subheading.TLabel').grid(
            row=left_row, column=0, sticky=tk.W, pady=(0, 5))
        left_row += 1

        self.tab7_cv_folds = tk.IntVar(value=5)
        ttk.Spinbox(params_left, from_=3, to=10, textvariable=self.tab7_cv_folds,
                    width=12).grid(row=left_row, column=0, sticky=tk.W, pady=(0, 15))
        left_row += 1

        # RIGHT COLUMN
        right_row = 0

        # Wavelength Specification
        ttk.Label(params_right, text="Wavelength Specification:", style='Subheading.TLabel').grid(
            row=right_row, column=0, sticky=tk.W, pady=(0, 5))
        right_row += 1

        # Preset buttons
        preset_frame = ttk.Frame(params_right)
        preset_frame.grid(row=right_row, column=0, sticky=tk.W, pady=(0, 5))
        right_row += 1

        ttk.Button(preset_frame, text="All", command=lambda: self._tab7_apply_wl_preset('all'),
                   width=10).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="NIR Only", command=lambda: self._tab7_apply_wl_preset('nir'),
                   width=10).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="Visible", command=lambda: self._tab7_apply_wl_preset('visible'),
                   width=10).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="Custom Range", command=self._tab7_custom_range_dialog,
                   width=12).pack(side='left', padx=2)

        # Wavelength text entry
        ttk.Label(params_right, text="Enter wavelengths (e.g., 1920, 1930-1940, 1950):",
                  style='Caption.TLabel').grid(row=right_row, column=0, sticky=tk.W, pady=(0, 3))
        right_row += 1

        wl_text_frame = ttk.Frame(params_right)
        wl_text_frame.grid(row=right_row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        right_row += 1

        self.tab7_wl_spec = tk.Text(wl_text_frame, height=5, width=50, font=('Consolas', 9),
                                    wrap=tk.WORD, relief='solid', borderwidth=1)
        self.tab7_wl_spec.pack(side='left', fill='both', expand=True)

        wl_scrollbar = ttk.Scrollbar(wl_text_frame, orient='vertical',
                                     command=self.tab7_wl_spec.yview)
        wl_scrollbar.pack(side='right', fill='y')
        self.tab7_wl_spec.config(yscrollcommand=wl_scrollbar.set)

        # Wavelength count
        self.tab7_wl_count_label = ttk.Label(params_right, text="Count: 0 wavelengths",
                                             style='Caption.TLabel', foreground='#1976D2')
        self.tab7_wl_count_label.grid(row=right_row, column=0, sticky=tk.W, pady=(0, 10))
        right_row += 1

        self.tab7_wl_spec.bind('<KeyRelease>', self._tab7_update_wavelength_count)

        # Model-Specific Hyperparameters
        ttk.Label(params_right, text="Model-Specific Hyperparameters:",
                  style='Subheading.TLabel').grid(row=right_row, column=0, sticky=tk.W, pady=(10, 5))
        right_row += 1

        self.tab7_hyperparam_frame = ttk.Frame(params_right, relief='solid', borderwidth=1)
        self.tab7_hyperparam_frame.grid(row=right_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        right_row += 1

        self.tab7_hyperparam_widgets = {}
        self._tab7_create_hyperparam_widgets('PLS')

        # ===================================================================
        # Action Buttons
        # ===================================================================
        button_frame = ttk.Frame(content_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=30)
        row += 1

        self.tab7_run_button = ttk.Button(button_frame, text="▶ Run Model",
                                          command=self._tab7_run_model,
                                          style='Accent.TButton')
        self.tab7_run_button.grid(row=0, column=0, padx=10, ipadx=40, ipady=12)

        self.tab7_save_button = ttk.Button(button_frame, text="💾 Save Model",
                                           command=self._tab7_save_model,
                                           style='Secondary.TButton', state='disabled')
        self.tab7_save_button.grid(row=0, column=1, padx=10, ipadx=40, ipady=12)

        # ===================================================================
        # Results Display
        # ===================================================================
        ttk.Label(content_frame, text="Performance Metrics", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 10))
        row += 1

        results_frame = ttk.LabelFrame(content_frame, text="Model Results", padding=20)
        results_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        row += 1

        self.tab7_results_text = tk.Text(results_frame, height=10, width=100,
                                         font=('Consolas', 10),
                                         bg='#FAFAFA', fg=self.colors['text'], wrap=tk.WORD,
                                         relief='flat')
        self.tab7_results_text.pack(fill='both', expand=True)
        self.tab7_results_text.insert('1.0', "Run a model to see performance metrics here.")
        self.tab7_results_text.config(state='disabled')

        # ===================================================================
        # Diagnostic Plots
        # ===================================================================
        ttk.Label(content_frame, text="Diagnostic Plots", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 10))
        row += 1

        plots_container = ttk.Frame(content_frame)
        plots_container.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        row += 1

        plots_container.columnconfigure(0, weight=1)
        plots_container.columnconfigure(1, weight=1)
        plots_container.columnconfigure(2, weight=1)

        # Plot 1: Prediction
        plot1_frame = ttk.LabelFrame(plots_container, text="Prediction Plot", padding=10)
        plot1_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=5)
        self.tab7_plot1_frame = ttk.Frame(plot1_frame)
        self.tab7_plot1_frame.pack(fill='both', expand=True)
        ttk.Label(self.tab7_plot1_frame, text="Prediction plot will appear here",
                  style='Caption.TLabel').pack(expand=True)

        # Plot 2: Residual Diagnostics
        plot2_frame = ttk.LabelFrame(plots_container, text="Residual Diagnostics", padding=10)
        plot2_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E), padx=5)
        self.tab7_plot2_frame = ttk.Frame(plot2_frame)
        self.tab7_plot2_frame.pack(fill='both', expand=True)
        ttk.Label(self.tab7_plot2_frame, text="Residual diagnostics will appear here",
                  style='Caption.TLabel').pack(expand=True)

        # Plot 3: Leverage Analysis
        plot3_frame = ttk.LabelFrame(plots_container, text="Leverage Analysis", padding=10)
        plot3_frame.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.W, tk.E), padx=5)
        self.tab7_plot3_frame = ttk.Frame(plot3_frame)
        self.tab7_plot3_frame.pack(fill='both', expand=True)
        ttk.Label(self.tab7_plot3_frame, text="Leverage analysis will appear here",
                  style='Caption.TLabel').pack(expand=True)

        # Status
        self.tab7_status = ttk.Label(content_frame, text="Ready to build models",
                                     style='Caption.TLabel', foreground='#666666')
        self.tab7_status.grid(row=row, column=0, columnspan=3, pady=(10, 0))

'''

    # Insert Tab 7 UI code
    content = re.sub(insertion_point_pattern, r'\1' + tab7_ui_code, content)

    print("  ✓ Tab 7 UI code inserted (~800 lines)")

    # Write changes
    write_file(gui_file, content)

    print(f"\n{'='*80}")
    print(f"INTEGRATION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"✓ Backup created: {backup_path}")
    print(f"✓ Tab 7 renamed to Tab 8")
    print(f"✓ New Tab 7 (Model Development) added")
    print(f"✓ UI code integrated")
    print(f"\nNext steps:")
    print(f"1. Add Tab 7 helper methods (loading, execution, plotting)")
    print(f"2. Update Results tab double-click handler")
    print(f"3. Fix Python backend n_folds field")
    print(f"4. Test with example data")

    return True

def main():
    """Main integration function."""
    print("\n" + "="*80)
    print("TAB 7 MODEL DEVELOPMENT - INTEGRATION SCRIPT")
    print("="*80 + "\n")

    # Check if we're in the right directory
    if not Path('spectral_predict_gui_optimized.py').exists():
        print("❌ Error: spectral_predict_gui_optimized.py not found!")
        print("Please run this script from the project root directory.")
        return 1

    # Integrate GUI changes
    success = integrate_gui_changes()

    if success:
        print("\n✅ Phase 1 complete! Tab 7 UI integrated.")
        print("\n📋 TODO: Still need to add:")
        print("   - Tab 7 helper methods (~1,500 lines)")
        print("   - Model loading engine (Agent 3)")
        print("   - Execution engine (Agent 4)")
        print("   - Diagnostic plots (Agent 5)")
        print("\nContinue with next integration script...")
        return 0
    else:
        print("\n❌ Integration failed!")
        return 1

if __name__ == '__main__':
    exit(main())
