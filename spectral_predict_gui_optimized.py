"""
Spectral Predict - Redesigned 7-Tab GUI Application (OPTIMIZED)

Tab 1: Import & Preview - Data loading + spectral plots
Tab 2: Data Quality Check - Outlier detection and exclusion
Tab 3: Analysis Configuration - All analysis settings
Tab 4: Analysis Progress - Live progress monitor
Tab 5: Results - Analysis results table (clickable to refine)
Tab 6: Custom Model Development - Interactive model refinement
Tab 7: Model Prediction - Load saved models and predict on new data

OPTIMIZED VERSION:
- Neural Boosted max_iter reduced from 500 to 100 (Phase A optimization)
- Implements evidence-based optimizations for 2-3x speedup
- Phase 3: Integrated outlier detection system with unified exclusion
- Phase 3+: Model prediction tab for applying saved .dasp models
"""

import sys
import os
import ast
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime
import numpy as np
import pandas as pd

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import spectral_predict modules
try:
    from spectral_predict.preprocess import SavgolDerivative
    HAS_DERIVATIVES = True
except ImportError:
    HAS_DERIVATIVES = False
    SavgolDerivative = None

try:
    from spectral_predict.outlier_detection import generate_outlier_report
    HAS_OUTLIER_DETECTION = True
except ImportError:
    HAS_OUTLIER_DETECTION = False
    generate_outlier_report = None


class SpectralPredictApp:
    """Main application window with 6-tab design."""

    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Predict - Automated Spectral Analysis (OPTIMIZED)")

        # Set window size - use zoomed/maximized for better visibility
        try:
            self.root.state('zoomed')  # Windows/Linux
        except:
            # Fallback for systems that don't support 'zoomed'
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = int(screen_width * 0.85)
            window_height = int(screen_height * 0.85)
            self.root.geometry(f"{window_width}x{window_height}")

        # Configure modern theme
        self._configure_style()

        # Data variables
        self.X = None  # Spectral data (filtered by wavelength)
        self.X_original = None  # Original unfiltered spectral data
        self.y = None  # Target data
        self.ref = None  # Reference dataframe

        # Results storage
        self.results_df = None  # Analysis results dataframe
        self.selected_model_config = None  # Selected model configuration for refinement

        # Refined/saved model storage (for model persistence)
        self.refined_model = None  # Fitted model from Custom Model Development tab
        self.refined_preprocessor = None  # Fitted preprocessing pipeline
        self.refined_performance = None  # Performance metrics dict (R2, RMSE, etc.)
        self.refined_wavelengths = None  # List of wavelengths used in refined model (subset for derivative+subset)
        self.refined_full_wavelengths = None  # List of ALL wavelengths (for derivative+subset preprocessing)
        self.refined_config = None  # Configuration dict for refined model

        # Model Prediction Tab (Tab 7) variables
        self.loaded_models = []  # List of model dicts from load_model()
        self.prediction_data = None  # DataFrame with new spectral data
        self.predictions_df = None  # Results dataframe

        # GUI variables
        self.spectral_data_path = tk.StringVar()  # Unified path for spectral data
        self.detected_type = None  # Auto-detected type: "asd", "csv", or "spc"
        self.reference_file = tk.StringVar()
        self.spectral_file_column = tk.StringVar()
        self.id_column = tk.StringVar()
        self.target_column = tk.StringVar()
        self.wavelength_min = tk.StringVar(value="")
        self.wavelength_max = tk.StringVar(value="")

        # Analysis variables
        self.output_dir = tk.StringVar(value="outputs")
        self.folds = tk.IntVar(value=5)
        self.lambda_penalty = tk.DoubleVar(value=0.15)
        self.max_n_components = tk.IntVar(value=24)
        self.max_iter = tk.IntVar(value=100)  # OPTIMIZED: Reduced from 500 to 100 (Phase A)
        self.show_progress = tk.BooleanVar(value=True)

        # Reflectance/Absorbance toggle
        self.use_absorbance = tk.BooleanVar(value=False)

        # Spectrum exclusion tracking
        self.excluded_spectra = set()  # Set of indices of excluded spectra

        # Model selection
        self.use_pls = tk.BooleanVar(value=True)
        self.use_ridge = tk.BooleanVar(value=False)  # Default off (baseline model)
        self.use_lasso = tk.BooleanVar(value=False)  # Default off (baseline model)
        self.use_randomforest = tk.BooleanVar(value=True)
        self.use_mlp = tk.BooleanVar(value=True)
        self.use_neuralboosted = tk.BooleanVar(value=True)

        # Preprocessing method selection
        self.use_raw = tk.BooleanVar(value=True)
        self.use_snv = tk.BooleanVar(value=True)
        self.use_sg1 = tk.BooleanVar(value=True)  # 1st derivative
        self.use_sg2 = tk.BooleanVar(value=True)  # 2nd derivative
        self.use_deriv_snv = tk.BooleanVar(value=False)  # deriv_snv (less common combo)

        # Subset Analysis options
        self.enable_variable_subsets = tk.BooleanVar(value=True)  # Top-N variable analysis
        self.enable_region_subsets = tk.BooleanVar(value=True)  # Spectral region analysis
        self.n_top_regions = tk.IntVar(value=5)  # Number of top regions to analyze (5, 10, 15, 20)

        # Top-N variable counts (checkboxes for each)
        self.var_10 = tk.BooleanVar(value=True)
        self.var_20 = tk.BooleanVar(value=True)
        self.var_50 = tk.BooleanVar(value=True)
        self.var_100 = tk.BooleanVar(value=True)
        self.var_250 = tk.BooleanVar(value=True)
        self.var_500 = tk.BooleanVar(value=False)
        self.var_1000 = tk.BooleanVar(value=False)

        # Window size selections (default: only 17 checked)
        self.window_7 = tk.BooleanVar(value=False)
        self.window_11 = tk.BooleanVar(value=False)
        self.window_17 = tk.BooleanVar(value=True)
        self.window_19 = tk.BooleanVar(value=False)

        # Advanced model options (NeuralBoosted)
        self.n_estimators_50 = tk.BooleanVar(value=False)
        self.n_estimators_100 = tk.BooleanVar(value=True)  # Default
        self.lr_005 = tk.BooleanVar(value=False)
        self.lr_01 = tk.BooleanVar(value=True)  # Default
        self.lr_02 = tk.BooleanVar(value=True)  # Default

        # Variable selection method (NEW - for advanced methods)
        self.variable_selection_method = tk.StringVar(value='importance')  # 'importance', 'spa', 'uve', 'uve_spa', 'ipls'
        self.apply_uve_prefilter = tk.BooleanVar(value=False)  # Apply UVE before main selection
        self.uve_cutoff_multiplier = tk.DoubleVar(value=1.0)  # UVE threshold (0.7-1.5)
        self.uve_n_components = tk.StringVar(value="")  # PLS components for UVE (empty = auto)
        self.spa_n_random_starts = tk.IntVar(value=10)  # Random starts for SPA
        self.ipls_n_intervals = tk.IntVar(value=20)  # Number of intervals for iPLS

        # CSV export option
        self.export_preprocessed_csv = tk.BooleanVar(value=False)

        # Outlier detection variables (Phase 3)
        self.n_pca_components = tk.IntVar(value=5)
        self.y_min_bound = tk.StringVar(value="")
        self.y_max_bound = tk.StringVar(value="")
        self.outlier_report = None  # Store most recent outlier detection report

        # Progress tracking
        self.progress_monitor = None
        self.analysis_thread = None
        self.analysis_start_time = None

        # Plotting
        self.plot_frames = {}
        self.plot_canvases = {}

        self._create_ui()

    def _configure_style(self):
        """Configure modern art gallery aesthetic."""
        style = ttk.Style()

        # Art gallery color palette
        self.colors = {
            'bg': '#F5F5F5',
            'panel': '#FFFFFF',
            'text': '#2C3E50',
            'text_light': '#7F8C8D',
            'accent': '#3498DB',
            'accent_dark': '#2980B9',
            'success': '#27AE60',
            'border': '#E8E8E8',
            'shadow': '#D0D0D0'
        }

        self.root.configure(bg=self.colors['bg'])

        style.configure('Modern.TButton', font=('Segoe UI', 10), padding=(15, 8))
        style.configure('Accent.TButton', font=('Segoe UI', 11, 'bold'), padding=(20, 12))
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=self.colors['text'])
        style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'), foreground=self.colors['text'])
        style.configure('Subheading.TLabel', font=('Segoe UI', 11, 'bold'), foreground=self.colors['accent'])
        style.configure('Caption.TLabel', font=('Segoe UI', 9), foreground=self.colors['text_light'])
        style.configure('TNotebook', background=self.colors['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', font=('Segoe UI', 11), padding=(20, 10))

    def _create_ui(self):
        """Create 7-tab user interface."""
        # Create notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self._create_tab1_import_preview()
        self._create_tab2_data_quality_check()
        self._create_tab3_analysis_config()
        self._create_tab4_progress()
        self._create_tab5_results()
        self._create_tab6_refine_model()
        self._create_tab7_model_prediction()

        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _create_tab1_import_preview(self):
        """Tab 1: Import & Preview - Data loading + spectral plots."""
        # Create main tab frame
        self.tab1 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab1, text='  📁 Import & Preview  ')

        # Create canvas for scrolling
        canvas = tk.Canvas(self.tab1, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab1, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas, style='TFrame', padding="30")

        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Title
        ttk.Label(content_frame, text="Import & Preview", style='Title.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 20))
        row += 1

        # === SECTION 1: Input Data ===
        ttk.Label(content_frame, text="1. Input Data", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 1

        # Spectral data directory (unified input with auto-detection)
        ttk.Label(content_frame, text="Spectral File Directory:").grid(row=row, column=0, sticky=tk.W, pady=10)
        ttk.Entry(content_frame, textvariable=self.spectral_data_path, width=60).grid(row=row, column=1, padx=10)
        ttk.Button(content_frame, text="Browse...", command=self._browse_spectral_data, style='Modern.TButton').grid(row=row, column=2)
        row += 1

        # Detection status label
        self.detection_status = ttk.Label(content_frame, text="", style='Caption.TLabel')
        self.detection_status.grid(row=row, column=1, sticky=tk.W, padx=10, pady=(0, 10))
        row += 1

        # Reference file
        ttk.Label(content_frame, text="Reference CSV:").grid(row=row, column=0, sticky=tk.W, pady=10)
        ttk.Entry(content_frame, textvariable=self.reference_file, width=60).grid(row=row, column=1, padx=10)
        ttk.Button(content_frame, text="Browse...", command=self._browse_reference_file, style='Modern.TButton').grid(row=row, column=2)
        row += 1

        # === SECTION 2: Column Names ===
        ttk.Label(content_frame, text="2. Column Names", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 15))
        row += 1

        ttk.Label(content_frame, text="Spectral File Column:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.spectral_file_combo = ttk.Combobox(content_frame, textvariable=self.spectral_file_column, width=35)
        self.spectral_file_combo.grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1

        ttk.Label(content_frame, text="Specimen ID Column:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.id_combo = ttk.Combobox(content_frame, textvariable=self.id_column, width=35)
        self.id_combo.grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1

        ttk.Label(content_frame, text="Target Variable Column:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.target_combo = ttk.Combobox(content_frame, textvariable=self.target_column, width=35)
        self.target_combo.grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1

        # Auto-detect button
        ttk.Button(content_frame, text="🔍 Auto-Detect Columns", command=self._auto_detect_columns,
                  style='Modern.TButton').grid(row=row, column=1, pady=15)
        row += 1

        # === Wavelength Range ===
        ttk.Label(content_frame, text="Wavelength Range (nm):", style='Subheading.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        row += 1

        wl_frame = ttk.Frame(content_frame)
        wl_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        ttk.Entry(wl_frame, textvariable=self.wavelength_min, width=12).grid(row=0, column=0, padx=5)
        ttk.Label(wl_frame, text="to").grid(row=0, column=1, padx=5)
        ttk.Entry(wl_frame, textvariable=self.wavelength_max, width=12).grid(row=0, column=2, padx=5)
        ttk.Label(wl_frame, text="(auto-fills after load)", style='Caption.TLabel').grid(row=0, column=3, padx=10)
        self.update_wl_button = ttk.Button(wl_frame, text="Update Plots", command=self._update_wavelengths, style='Modern.TButton', state='disabled')
        self.update_wl_button.grid(row=0, column=4, padx=15)

        # === Reflectance/Absorbance Toggle ===
        ttk.Label(content_frame, text="Data Transformation:", style='Subheading.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        row += 1

        transform_frame = ttk.Frame(content_frame)
        transform_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1

        self.absorbance_checkbox = ttk.Checkbutton(transform_frame, text="Convert to Absorbance (log10(1/R))",
                                                    variable=self.use_absorbance,
                                                    command=self._toggle_absorbance,
                                                    state='disabled')
        self.absorbance_checkbox.grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(transform_frame, text="(Toggle to view data as absorbance instead of reflectance)",
                 style='Caption.TLabel').grid(row=0, column=1, sticky=tk.W, padx=10)

        # === Spectrum Exclusion Controls ===
        ttk.Label(content_frame, text="Spectrum Selection:", style='Subheading.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15, 5))
        row += 1

        exclusion_frame = ttk.Frame(content_frame)
        exclusion_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1

        self.reset_exclusions_button = ttk.Button(exclusion_frame, text="Reset Exclusions",
                                                  command=self._reset_exclusions,
                                                  style='Modern.TButton',
                                                  state='disabled')
        self.reset_exclusions_button.grid(row=0, column=0, padx=5)

        self.exclusion_status = ttk.Label(exclusion_frame, text="No spectra excluded", style='Caption.TLabel')
        self.exclusion_status.grid(row=0, column=1, sticky=tk.W, padx=10)

        ttk.Label(exclusion_frame, text="(Click individual spectra in plots to toggle visibility)",
                 style='Caption.TLabel').grid(row=0, column=2, sticky=tk.W, padx=10)

        # Load Data Button
        self.load_button = ttk.Button(content_frame, text="📊 Load Data & Generate Plots",
                                     command=self._load_and_plot_data, style='Accent.TButton')
        self.load_button.grid(row=row, column=0, columnspan=3, pady=30)
        row += 1

        # Status
        self.tab1_status = ttk.Label(content_frame, text="Ready to load data", style='Caption.TLabel')
        self.tab1_status.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        # === Plotting Area ===
        ttk.Label(content_frame, text="Spectral Plots", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        row += 1

        # Create sub-notebook for plots
        self.plot_notebook = ttk.Notebook(content_frame)
        self.plot_notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        content_frame.grid_rowconfigure(row, weight=1)

        # Placeholder for plots
        placeholder = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(placeholder, text="Load data to see plots")
        ttk.Label(placeholder, text="Load data to generate spectral plots", style='Caption.TLabel').pack(expand=True)

    def _create_tab2_data_quality_check(self):
        """Tab 2: Data Quality Check - Outlier detection and exclusion."""
        self.tab2 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab2, text='  🔍 Data Quality Check  ')

        # Create scrollable content
        canvas = tk.Canvas(self.tab2, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab2, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas, style='TFrame', padding="30")

        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Title
        ttk.Label(content_frame, text="Data Quality Check", style='Title.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 20))
        row += 1

        # === SECTION 1: Controls ===
        ttk.Label(content_frame, text="1. Outlier Detection Parameters", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 1

        controls_frame = ttk.LabelFrame(content_frame, text="Detection Settings", padding="20")
        controls_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # PCA Components
        ttk.Label(controls_frame, text="PCA Components:").grid(row=0, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        ttk.Spinbox(controls_frame, from_=2, to=20, textvariable=self.n_pca_components, width=12).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(controls_frame, text="Number of PCs for outlier detection", style='Caption.TLabel').grid(row=0, column=2, sticky=tk.W, padx=10)

        # Y Range (optional)
        ttk.Label(controls_frame, text="Y Range (optional):").grid(row=1, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        y_range_frame = ttk.Frame(controls_frame)
        y_range_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W)
        ttk.Label(y_range_frame, text="Min:").pack(side='left', padx=(0, 5))
        ttk.Entry(y_range_frame, textvariable=self.y_min_bound, width=10).pack(side='left', padx=(0, 10))
        ttk.Label(y_range_frame, text="Max:").pack(side='left', padx=(0, 5))
        ttk.Entry(y_range_frame, textvariable=self.y_max_bound, width=10).pack(side='left')

        # Buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=20)
        self.run_outlier_btn = ttk.Button(button_frame, text="Run Outlier Detection",
                                          command=self._run_outlier_detection, style='Accent.TButton')
        self.run_outlier_btn.pack(side='left', padx=5)
        self.export_report_btn = ttk.Button(button_frame, text="Export Report",
                                           command=self._export_outlier_report, style='Modern.TButton')
        self.export_report_btn.pack(side='left', padx=5)

        # === SECTION 2: Visualizations ===
        ttk.Label(content_frame, text="2. Outlier Detection Plots", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 15))
        row += 1

        # Create notebook for visualization tabs
        self.outlier_plot_notebook = ttk.Notebook(content_frame)
        self.outlier_plot_notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        content_frame.grid_rowconfigure(row, weight=1)
        row += 1

        # Create placeholder frames for each plot type
        self.pca_plot_frame = ttk.Frame(self.outlier_plot_notebook)
        self.outlier_plot_notebook.add(self.pca_plot_frame, text="PCA Scores")

        self.t2_plot_frame = ttk.Frame(self.outlier_plot_notebook)
        self.outlier_plot_notebook.add(self.t2_plot_frame, text="Hotelling T²")

        self.q_plot_frame = ttk.Frame(self.outlier_plot_notebook)
        self.outlier_plot_notebook.add(self.q_plot_frame, text="Q-Residuals")

        self.maha_plot_frame = ttk.Frame(self.outlier_plot_notebook)
        self.outlier_plot_notebook.add(self.maha_plot_frame, text="Mahalanobis")

        self.y_dist_plot_frame = ttk.Frame(self.outlier_plot_notebook)
        self.outlier_plot_notebook.add(self.y_dist_plot_frame, text="Y Distribution")

        # === SECTION 3: Summary Table ===
        ttk.Label(content_frame, text="3. Outlier Summary", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 15))
        row += 1

        # Create frame for treeview
        tree_frame = ttk.Frame(content_frame)
        tree_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        content_frame.grid_rowconfigure(row, weight=1)
        row += 1

        # Create Treeview with scrollbars
        columns = ('Sample', 'Y_Value', 'T2', 'Q', 'Maha', 'Y', 'Flags')
        self.outlier_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', selectmode='extended', height=10)

        # Define column headings
        self.outlier_tree.heading('Sample', text='Sample')
        self.outlier_tree.heading('Y_Value', text='Y Value')
        self.outlier_tree.heading('T2', text='T² Outlier')
        self.outlier_tree.heading('Q', text='Q Outlier')
        self.outlier_tree.heading('Maha', text='Maha Outlier')
        self.outlier_tree.heading('Y', text='Y Outlier')
        self.outlier_tree.heading('Flags', text='Total Flags')

        # Configure column widths
        self.outlier_tree.column('Sample', width=80, anchor='center')
        self.outlier_tree.column('Y_Value', width=100, anchor='center')
        self.outlier_tree.column('T2', width=80, anchor='center')
        self.outlier_tree.column('Q', width=80, anchor='center')
        self.outlier_tree.column('Maha', width=80, anchor='center')
        self.outlier_tree.column('Y', width=80, anchor='center')
        self.outlier_tree.column('Flags', width=100, anchor='center')

        # Add scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.outlier_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.outlier_tree.xview)
        self.outlier_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.outlier_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # === SECTION 4: Selection Controls ===
        ttk.Label(content_frame, text="4. Sample Selection & Exclusion", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 15))
        row += 1

        selection_frame = ttk.LabelFrame(content_frame, text="Auto-Select Outliers", padding="20")
        selection_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Selection checkboxes
        self.select_all_flagged = tk.BooleanVar(value=False)
        self.select_high_conf = tk.BooleanVar(value=True)
        self.select_moderate_conf = tk.BooleanVar(value=False)

        ttk.Checkbutton(selection_frame, text="Select all flagged samples",
                       variable=self.select_all_flagged,
                       command=self._auto_select_flagged).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(selection_frame, text="High confidence (3+ flags)",
                       variable=self.select_high_conf,
                       command=self._auto_select_high_confidence).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(selection_frame, text="Moderate confidence (2 flags)",
                       variable=self.select_moderate_conf,
                       command=self._auto_select_moderate_confidence).grid(row=2, column=0, sticky=tk.W, pady=5)

        # Status and action buttons
        self.outlier_selection_status = ttk.Label(selection_frame, text="No samples selected", style='Caption.TLabel')
        self.outlier_selection_status.grid(row=3, column=0, pady=10)

        ttk.Button(selection_frame, text="Mark Selected for Exclusion",
                  command=self._mark_selected_for_exclusion,
                  style='Accent.TButton').grid(row=4, column=0, pady=10)

        # Overall status
        self.tab2_status = ttk.Label(content_frame, text="Load data and run outlier detection to begin", style='Caption.TLabel')
        self.tab2_status.grid(row=row, column=0, columnspan=3)

    def _create_tab3_analysis_config(self):
        """Tab 3: Analysis Configuration - All analysis settings."""
        self.tab3 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab3, text='  ⚙️ Analysis Configuration  ')

        # Create scrollable content
        canvas = tk.Canvas(self.tab3, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab3, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas, style='TFrame', padding="30")

        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Title
        ttk.Label(content_frame, text="Analysis Configuration", style='Title.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        row += 1

        # === Analysis Options ===
        ttk.Label(content_frame, text="Analysis Options", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        row += 1

        options_frame = ttk.LabelFrame(content_frame, text="General Settings", padding="20")
        options_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # CV Folds
        ttk.Label(options_frame, text="CV Folds:").grid(row=0, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        ttk.Spinbox(options_frame, from_=3, to=10, textvariable=self.folds, width=12).grid(row=0, column=1, sticky=tk.W)

        # Lambda penalty
        ttk.Label(options_frame, text="Complexity Penalty:").grid(row=1, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        ttk.Entry(options_frame, textvariable=self.lambda_penalty, width=12).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(options_frame, text="(higher = prefer simpler models)", style='Caption.TLabel').grid(row=1, column=2, sticky=tk.W, padx=10)

        # Max PLS components
        ttk.Label(options_frame, text="Max Latent Variables:").grid(row=2, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        ttk.Spinbox(options_frame, from_=2, to=100, textvariable=self.max_n_components, width=12).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(options_frame, text="(PLS components)", style='Caption.TLabel').grid(row=2, column=2, sticky=tk.W, padx=10)

        # Max iterations
        ttk.Label(options_frame, text="Max Iterations:").grid(row=3, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        ttk.Spinbox(options_frame, from_=100, to=5000, increment=100, textvariable=self.max_iter, width=12).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(options_frame, text="(for MLP/Neural Boosted)", style='Caption.TLabel').grid(row=3, column=2, sticky=tk.W, padx=10)

        # Output directory
        ttk.Label(options_frame, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        ttk.Entry(options_frame, textvariable=self.output_dir, width=25).grid(row=4, column=1, sticky=tk.W)

        # Progress monitor
        ttk.Checkbutton(options_frame, text="Show live progress monitor", variable=self.show_progress).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=10)

        # === Preprocessing Methods ===
        ttk.Label(content_frame, text="Preprocessing Methods", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        preprocess_frame = ttk.LabelFrame(content_frame, text="Select Preprocessing", padding="20")
        preprocess_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        ttk.Checkbutton(preprocess_frame, text="✓ Raw (no preprocessing)", variable=self.use_raw).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Baseline, unprocessed spectra", style='Caption.TLabel').grid(row=0, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(preprocess_frame, text="✓ SNV (Standard Normal Variate)", variable=self.use_snv).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Scatter correction", style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(preprocess_frame, text="✓ SG1 (1st derivative)", variable=self.use_sg1).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Removes baseline drift", style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(preprocess_frame, text="✓ SG2 (2nd derivative)", variable=self.use_sg2).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Peak enhancement", style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, padx=15)

        # Advanced: deriv_snv option
        ttk.Checkbutton(preprocess_frame, text="deriv_snv (advanced)", variable=self.use_deriv_snv).grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Derivative then SNV (less common)", style='Caption.TLabel').grid(row=4, column=1, sticky=tk.W, padx=15)

        # Derivative window size settings
        ttk.Label(preprocess_frame, text="Derivative Window Sizes:", style='Subheading.TLabel').grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        ttk.Label(preprocess_frame, text="Select one or more (default: 17 only)", style='Caption.TLabel').grid(row=7, column=0, columnspan=2, sticky=tk.W)

        window_frame = ttk.Frame(preprocess_frame)
        window_frame.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Checkbutton(window_frame, text="Window=7", variable=self.window_7).grid(row=0, column=0, padx=5, pady=2)
        ttk.Checkbutton(window_frame, text="Window=11", variable=self.window_11).grid(row=0, column=1, padx=5, pady=2)
        ttk.Checkbutton(window_frame, text="Window=17 ⭐", variable=self.window_17).grid(row=0, column=2, padx=5, pady=2)
        ttk.Checkbutton(window_frame, text="Window=19", variable=self.window_19).grid(row=0, column=3, padx=5, pady=2)

        # === Subset Analysis ===
        ttk.Label(content_frame, text="Subset Analysis", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        subset_frame = ttk.LabelFrame(content_frame, text="Variable & Region Subsets", padding="20")
        subset_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Enable/disable toggles
        ttk.Checkbutton(subset_frame, text="✓ Enable Top-N Variable Analysis", variable=self.enable_variable_subsets).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=5)
        ttk.Label(subset_frame, text="Test models using only the N most important wavelengths", style='Caption.TLabel').grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=(20, 0))

        ttk.Checkbutton(subset_frame, text="✓ Enable Spectral Region Analysis", variable=self.enable_region_subsets).grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(10, 5))
        ttk.Label(subset_frame, text="Test models using auto-detected spectral regions (e.g., 2000-2050nm)", style='Caption.TLabel').grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=(20, 0))

        # Region analysis depth (radio buttons)
        ttk.Label(subset_frame, text="Region Analysis Depth:", style='Caption.TLabel').grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=(20, 0), pady=(5, 2))
        region_depth_frame = ttk.Frame(subset_frame)
        region_depth_frame.grid(row=5, column=0, columnspan=4, sticky=tk.W, padx=(20, 0), pady=(0, 5))
        ttk.Radiobutton(region_depth_frame, text="Shallow (5 regions)", variable=self.n_top_regions, value=5).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(region_depth_frame, text="Medium (10 regions)", variable=self.n_top_regions, value=10).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(region_depth_frame, text="Deep (15 regions)", variable=self.n_top_regions, value=15).grid(row=0, column=2, padx=5)
        ttk.Radiobutton(region_depth_frame, text="Thorough (20 regions)", variable=self.n_top_regions, value=20).grid(row=0, column=3, padx=5)

        # Top-N variable counts
        ttk.Label(subset_frame, text="Top-N Variable Counts:", style='Subheading.TLabel').grid(row=6, column=0, columnspan=4, sticky=tk.W, pady=(15, 5))
        ttk.Label(subset_frame, text="Select which N values to test (default: 10, 20, 50, 100, 250)", style='Caption.TLabel').grid(row=7, column=0, columnspan=4, sticky=tk.W)

        var_frame = ttk.Frame(subset_frame)
        var_frame.grid(row=8, column=0, columnspan=4, sticky=tk.W, pady=5)

        ttk.Checkbutton(var_frame, text="N=10 ⭐", variable=self.var_10).grid(row=0, column=0, padx=5, pady=2)
        ttk.Checkbutton(var_frame, text="N=20 ⭐", variable=self.var_20).grid(row=0, column=1, padx=5, pady=2)
        ttk.Checkbutton(var_frame, text="N=50 ⭐", variable=self.var_50).grid(row=0, column=2, padx=5, pady=2)
        ttk.Checkbutton(var_frame, text="N=100 ⭐", variable=self.var_100).grid(row=0, column=3, padx=5, pady=2)
        ttk.Checkbutton(var_frame, text="N=250 ⭐", variable=self.var_250).grid(row=1, column=0, padx=5, pady=2)
        ttk.Checkbutton(var_frame, text="N=500", variable=self.var_500).grid(row=1, column=1, padx=5, pady=2)
        ttk.Checkbutton(var_frame, text="N=1000", variable=self.var_1000).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(subset_frame, text="💡 More subsets = more comprehensive results but longer runtime",
                 style='Caption.TLabel', foreground=self.colors['accent']).grid(row=7, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))

        # === Variable Selection Methods ===
        ttk.Label(content_frame, text="Variable Selection Methods 🆕", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        varsel_frame = ttk.LabelFrame(content_frame, text="Advanced Variable Selection", padding="20")
        varsel_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Method selection (radio buttons)
        ttk.Label(varsel_frame, text="Selection Method:", style='Subheading.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        ttk.Radiobutton(varsel_frame, text="Feature Importance (default)",
                       variable=self.variable_selection_method, value='importance').grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Uses model-specific importance scores",
                 style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, padx=15)

        ttk.Radiobutton(varsel_frame, text="SPA (Successive Projections)",
                       variable=self.variable_selection_method, value='spa').grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Collinearity-aware selection",
                 style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)

        ttk.Radiobutton(varsel_frame, text="UVE (Uninformative Variable Elimination)",
                       variable=self.variable_selection_method, value='uve').grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Filters noisy variables",
                 style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, padx=15)

        ttk.Radiobutton(varsel_frame, text="UVE-SPA Hybrid",
                       variable=self.variable_selection_method, value='uve_spa').grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Combines noise filtering + collinearity reduction",
                 style='Caption.TLabel').grid(row=4, column=1, sticky=tk.W, padx=15)

        ttk.Radiobutton(varsel_frame, text="iPLS (Interval PLS)",
                       variable=self.variable_selection_method, value='ipls').grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Region-based analysis",
                 style='Caption.TLabel').grid(row=5, column=1, sticky=tk.W, padx=15)

        # UVE Prefilter option
        ttk.Checkbutton(varsel_frame, text="Apply UVE Pre-filter (removes noisy variables first)",
                       variable=self.apply_uve_prefilter).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))

        # Method parameters
        ttk.Label(varsel_frame, text="Method Parameters:", style='Subheading.TLabel').grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(15, 8))

        params_frame = ttk.Frame(varsel_frame)
        params_frame.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=5)

        # UVE parameters
        ttk.Label(params_frame, text="UVE Cutoff:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(params_frame, textvariable=self.uve_cutoff_multiplier, width=8).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(params_frame, text="(0.7-1.5, default: 1.0)", style='Caption.TLabel').grid(row=0, column=2, sticky=tk.W, padx=10)

        ttk.Label(params_frame, text="UVE Components:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=5)
        ttk.Entry(params_frame, textvariable=self.uve_n_components, width=8).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(params_frame, text="(leave empty for auto)", style='Caption.TLabel').grid(row=1, column=2, sticky=tk.W, padx=10)

        ttk.Label(params_frame, text="SPA Random Starts:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=5)
        ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.spa_n_random_starts, width=8).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(params_frame, text="(default: 10)", style='Caption.TLabel').grid(row=2, column=2, sticky=tk.W, padx=10)

        ttk.Label(params_frame, text="iPLS Intervals:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=5)
        ttk.Spinbox(params_frame, from_=5, to=50, textvariable=self.ipls_n_intervals, width=8).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(params_frame, text="(default: 20)", style='Caption.TLabel').grid(row=3, column=2, sticky=tk.W, padx=10)

        ttk.Label(varsel_frame, text="📚 See VARIABLE_SELECTION_IMPLEMENTATION.md for method details",
                 style='Caption.TLabel', foreground=self.colors['accent']).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # === Model Selection ===
        ttk.Label(content_frame, text="Models to Test", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        models_frame = ttk.LabelFrame(content_frame, text="Select Models", padding="20")
        models_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        ttk.Checkbutton(models_frame, text="✓ PLS (Partial Least Squares)", variable=self.use_pls).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Linear, fast, interpretable", style='Caption.TLabel').grid(row=0, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ Ridge Regression", variable=self.use_ridge).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="L2 regularized linear, fast baseline", style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ Lasso Regression", variable=self.use_lasso).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="L1 regularized linear, sparse solutions", style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ Random Forest", variable=self.use_randomforest).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Nonlinear, robust", style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ MLP (Multi-Layer Perceptron)", variable=self.use_mlp).grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Deep learning", style='Caption.TLabel').grid(row=4, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ Neural Boosted", variable=self.use_neuralboosted).grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Gradient boosting with NNs", style='Caption.TLabel').grid(row=5, column=1, sticky=tk.W, padx=15)

        # === Advanced Model Options ===
        ttk.Label(content_frame, text="Advanced Model Options", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        advanced_frame = ttk.LabelFrame(content_frame, text="Neural Boosted Hyperparameters", padding="20")
        advanced_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # n_estimators options
        ttk.Label(advanced_frame, text="n_estimators (boosting rounds):", style='Subheading.TLabel').grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5))
        nest_frame = ttk.Frame(advanced_frame)
        nest_frame.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5)
        ttk.Checkbutton(nest_frame, text="50", variable=self.n_estimators_50).grid(row=0, column=0, padx=5)
        ttk.Checkbutton(nest_frame, text="100 ⭐", variable=self.n_estimators_100).grid(row=0, column=1, padx=5)
        ttk.Label(nest_frame, text="(default: 100 only)", style='Caption.TLabel').grid(row=0, column=2, padx=10)

        # Learning rate options
        ttk.Label(advanced_frame, text="Learning rates:", style='Subheading.TLabel').grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(15, 5))
        lr_frame = ttk.Frame(advanced_frame)
        lr_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=5)
        ttk.Checkbutton(lr_frame, text="0.05", variable=self.lr_005).grid(row=0, column=0, padx=5)
        ttk.Checkbutton(lr_frame, text="0.1 ⭐", variable=self.lr_01).grid(row=0, column=1, padx=5)
        ttk.Checkbutton(lr_frame, text="0.2 ⭐", variable=self.lr_02).grid(row=0, column=2, padx=5)
        ttk.Label(lr_frame, text="(default: 0.1, 0.2)", style='Caption.TLabel').grid(row=0, column=3, padx=10)

        # Info label
        ttk.Label(advanced_frame, text="💡 Selecting more options = more comprehensive analysis but longer runtime",
                 style='Caption.TLabel', foreground=self.colors['accent']).grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))

        # CSV export checkbox
        ttk.Checkbutton(content_frame, text="Export preprocessed data CSV (2nd derivative)",
                       variable=self.export_preprocessed_csv).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
        row += 1

        # Run button
        ttk.Button(content_frame, text="▶ Run Analysis", command=self._run_analysis,
                  style='Accent.TButton').grid(row=row, column=0, columnspan=2, pady=20, ipadx=30, ipady=10)
        row += 1

        self.tab3_status = ttk.Label(content_frame, text="Configure analysis settings above", style='Caption.TLabel')
        self.tab3_status.grid(row=row, column=0, columnspan=2)

    def _create_tab4_progress(self):
        """Tab 4: Analysis Progress - Live progress monitor."""
        self.tab4 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab4, text='  📊 Analysis Progress  ')

        content_frame = ttk.Frame(self.tab4, style='TFrame', padding="30")
        content_frame.pack(fill='both', expand=True)

        # Header with title and best model info side-by-side
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill='x', pady=(0, 20))

        # Left side: Title
        ttk.Label(header_frame, text="Analysis Progress", style='Title.TLabel').pack(side='left', anchor=tk.W)

        # Right side: Best model info
        best_model_frame = ttk.Frame(header_frame)
        best_model_frame.pack(side='right', anchor=tk.E)

        ttk.Label(best_model_frame, text="Best Model So Far:", style='Heading.TLabel').pack(anchor=tk.E)
        self.best_model_info = ttk.Label(best_model_frame, text="(none yet)", style='Caption.TLabel', foreground=self.colors['success'])
        self.best_model_info.pack(anchor=tk.E)

        # Progress info with time estimate
        progress_info_frame = ttk.Frame(content_frame)
        progress_info_frame.pack(fill='x', pady=10)

        self.progress_info = ttk.Label(progress_info_frame, text="No analysis running", style='Heading.TLabel')
        self.progress_info.pack(side='left', anchor=tk.W)

        self.time_estimate_label = ttk.Label(progress_info_frame, text="", style='Caption.TLabel')
        self.time_estimate_label.pack(side='right', anchor=tk.E)

        # Progress text area
        self.progress_text = tk.Text(content_frame, height=30, width=120, font=('Consolas', 10), bg='#FAFAFA', fg=self.colors['text'])
        self.progress_text.pack(fill='both', expand=True, pady=10)

        scrollbar = ttk.Scrollbar(content_frame, command=self.progress_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.progress_text.config(yscrollcommand=scrollbar.set)

        # Status
        self.progress_status = ttk.Label(content_frame, text="Waiting for analysis to start...", style='Caption.TLabel')
        self.progress_status.pack(pady=10)

    def _create_tab5_results(self):
        """Tab 5: Results - Display analysis results in a table."""
        self.tab5 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab5, text='  📊 Results  ')

        content_frame = ttk.Frame(self.tab5, style='TFrame', padding="30")
        content_frame.pack(fill='both', expand=True)

        # Title
        ttk.Label(content_frame, text="Analysis Results", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 20))

        # Instructions
        instructions = ttk.Label(content_frame,
            text="Click on any result row to load it into the 'Custom Model Development' tab for further tuning.",
            style='Caption.TLabel')
        instructions.pack(anchor=tk.W, pady=(0, 10))

        # Create frame for treeview and scrollbars
        tree_frame = ttk.Frame(content_frame)
        tree_frame.pack(fill='both', expand=True)

        # Create Treeview with scrollbars
        self.results_tree = ttk.Treeview(tree_frame, show='headings', selectmode='browse')

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.results_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Bind double-click event
        self.results_tree.bind('<Double-Button-1>', self._on_result_double_click)

        # Status label
        self.results_status = ttk.Label(content_frame, text="No results yet. Run an analysis to see results here.",
                                       style='Caption.TLabel')
        self.results_status.pack(pady=10)

    def _create_tab6_refine_model(self):
        """Tab 6: Custom Model Development - Interactive model parameter refinement."""
        self.tab6 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab6, text='  🔧 Custom Model Development  ')

        # Create scrollable content
        canvas = tk.Canvas(self.tab6, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab6, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas, style='TFrame', padding="30")

        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Title
        ttk.Label(content_frame, text="Custom Model Development", style='Title.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        row += 1

        # Instructions
        ttk.Label(content_frame,
            text="Double-click a result from the Results tab to load it here for refinement, or click 'Reset to Defaults' for fresh development.",
            style='Caption.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        row += 1

        # === Mode Control Frame ===
        mode_frame = ttk.LabelFrame(content_frame, text="Development Mode", padding=10)
        mode_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Mode status label
        self.refine_mode_label = ttk.Label(mode_frame, text="Mode: Fresh Development", style='Caption.TLabel')
        self.refine_mode_label.pack(side='left', padx=5)

        # Reset button
        ttk.Button(mode_frame, text="Reset to Defaults",
                   command=self._load_default_parameters).pack(side='right', padx=5)

        # === Selected Model Info ===
        ttk.Label(content_frame, text="Selected Model Configuration", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        row += 1

        info_frame = ttk.LabelFrame(content_frame, text="Current Configuration", padding="20")
        info_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.refine_model_info = tk.Text(info_frame, height=8, width=80, font=('Consolas', 10),
                                         bg='#FAFAFA', fg=self.colors['text'], wrap=tk.WORD)
        self.refine_model_info.pack(fill='both', expand=True)
        self.refine_model_info.insert('1.0', "No model selected. Double-click a result in the Results tab to load it here.")
        self.refine_model_info.config(state='disabled')

        # === Refinement Parameters ===
        ttk.Label(content_frame, text="Refinement Parameters", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        params_frame = ttk.LabelFrame(content_frame, text="Adjust Parameters", padding="20")
        params_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Wavelength specification header
        ttk.Label(params_frame, text="Wavelength Specification:", style='Subheading.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)

        # Wavelength presets
        preset_frame = ttk.Frame(params_frame)
        preset_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(preset_frame, text="Quick presets:", style='Caption.TLabel').pack(side='left', padx=(0, 5))
        ttk.Button(preset_frame, text="All", command=lambda: self._apply_wl_preset('all'),
                   width=8).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="NIR Only", command=lambda: self._apply_wl_preset('nir'),
                   width=8).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="Visible", command=lambda: self._apply_wl_preset('visible'),
                   width=8).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="Custom Range...", command=self._custom_range_dialog,
                   width=12).pack(side='left', padx=2)

        # Instructions
        ttk.Label(params_frame, text="Enter wavelengths as individual values or ranges (e.g., 1920, 1930-1940, 1950)",
                  style='Caption.TLabel').grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Text box for wavelength specification
        wl_spec_frame = ttk.Frame(params_frame)
        wl_spec_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        self.refine_wl_spec = tk.Text(wl_spec_frame, height=6, width=60, font=('Consolas', 9), wrap=tk.WORD)
        self.refine_wl_spec.pack(side='left', fill='both', expand=True)

        wl_spec_scrollbar = ttk.Scrollbar(wl_spec_frame, orient='vertical', command=self.refine_wl_spec.yview)
        wl_spec_scrollbar.pack(side='right', fill='y')
        self.refine_wl_spec.config(yscrollcommand=wl_spec_scrollbar.set)

        # Button to preview wavelength selection
        ttk.Button(params_frame, text="Preview Selected Wavelengths",
                   command=self._preview_wavelength_selection).grid(row=4, column=0, sticky=tk.W, pady=5)

        # Wavelength count display (real-time)
        self.refine_wl_count_label = ttk.Label(params_frame, text="Wavelengths: 0 selected",
                                                style='Caption.TLabel')
        self.refine_wl_count_label.grid(row=4, column=1, sticky=tk.W, padx=(10, 0))

        # Bind update to text widget for real-time feedback
        self.refine_wl_spec.bind('<KeyRelease>', self._update_wavelength_count)

        # Window size (for derivatives)
        ttk.Label(params_frame, text="Window Size:", style='Subheading.TLabel').grid(row=5, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_window = tk.IntVar(value=17)
        window_frame = ttk.Frame(params_frame)
        window_frame.grid(row=6, column=0, sticky=tk.W, pady=5)
        for w in [7, 11, 17, 19]:
            ttk.Radiobutton(window_frame, text=f"{w}", variable=self.refine_window, value=w).pack(side='left', padx=5)

        # Model Type
        ttk.Label(params_frame, text="Model Type:", style='Subheading.TLabel').grid(row=7, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_model_type = tk.StringVar(value='PLS')
        model_combo = ttk.Combobox(params_frame, textvariable=self.refine_model_type, width=20, state='readonly')
        model_combo['values'] = ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']
        model_combo.grid(row=8, column=0, sticky=tk.W, pady=5)

        # Task Type
        ttk.Label(params_frame, text="Task Type:", style='Subheading.TLabel').grid(row=9, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_task_type = tk.StringVar(value='regression')
        task_frame = ttk.Frame(params_frame)
        task_frame.grid(row=10, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(task_frame, text="Regression", variable=self.refine_task_type, value='regression').pack(side='left', padx=5)
        ttk.Radiobutton(task_frame, text="Classification", variable=self.refine_task_type, value='classification').pack(side='left', padx=5)

        # Preprocessing Method
        ttk.Label(params_frame, text="Preprocessing:", style='Subheading.TLabel').grid(row=11, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_preprocess = tk.StringVar(value='raw')
        preprocess_combo = ttk.Combobox(params_frame, textvariable=self.refine_preprocess, width=25, state='readonly')
        preprocess_combo['values'] = ['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']
        preprocess_combo.grid(row=12, column=0, sticky=tk.W, pady=5)

        # CV Folds
        ttk.Label(params_frame, text="CV Folds:", style='Subheading.TLabel').grid(row=13, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_folds = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=3, to=10, textvariable=self.refine_folds, width=12).grid(row=14, column=0, sticky=tk.W)

        # Max iterations (for neural models)
        ttk.Label(params_frame, text="Max Iterations:", style='Subheading.TLabel').grid(row=15, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_max_iter = tk.IntVar(value=100)
        ttk.Spinbox(params_frame, from_=100, to=5000, increment=100, textvariable=self.refine_max_iter, width=12).grid(row=16, column=0, sticky=tk.W)

        # Button frame for Run and Save buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=30)
        row += 1

        self.refine_run_button = ttk.Button(button_frame, text="▶ Run Refined Model", command=self._run_refined_model,
                  style='Accent.TButton', state='disabled')
        self.refine_run_button.grid(row=0, column=0, padx=10, ipadx=30, ipady=10)

        self.refine_save_button = ttk.Button(button_frame, text="💾 Save Model", command=self._save_refined_model,
                  style='Secondary.TButton', state='disabled')
        self.refine_save_button.grid(row=0, column=1, padx=10, ipadx=30, ipady=10)

        # Results display
        ttk.Label(content_frame, text="Refined Model Results", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        results_frame = ttk.LabelFrame(content_frame, text="Performance Metrics", padding="20")
        results_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.refine_results_text = tk.Text(results_frame, height=10, width=80, font=('Consolas', 10),
                                           bg='#FAFAFA', fg=self.colors['text'], wrap=tk.WORD)
        self.refine_results_text.pack(fill='both', expand=True)
        self.refine_results_text.insert('1.0', "Run a refined model to see results here.")
        self.refine_results_text.config(state='disabled')

        # Status
        self.refine_status = ttk.Label(content_frame, text="No model loaded", style='Caption.TLabel')
        self.refine_status.grid(row=row, column=0, columnspan=2)

    # === Helper Methods ===

    def _browse_spectral_data(self):
        """Browse for spectral data and auto-detect type."""
        directory = filedialog.askdirectory(title="Select Spectral Data Directory")

        if not directory:
            return

        # Store path
        self.spectral_data_path.set(directory)
        path = Path(directory)

        # Auto-detect file type
        # Priority: ASD > CSV > SPC

        # Check for ASD files
        asd_files = list(path.glob("*.asd"))
        if asd_files:
            self.detected_type = "asd"
            self.detection_status.config(
                text=f"✓ Detected {len(asd_files)} ASD files",
                foreground=self.colors['success']
            )

            # Auto-detect reference CSV
            csv_files = list(path.glob("*.csv"))
            if len(csv_files) == 1:
                self.reference_file.set(str(csv_files[0]))
                self._auto_detect_columns()
                # No popup needed - status label shows detection
            elif len(csv_files) > 1:
                # Update status to guide user - no popup needed
                self.detection_status.config(
                    text=f"✓ Detected {len(asd_files)} ASD files - {len(csv_files)} CSVs found, select reference manually",
                    foreground=self.colors['accent']
                )
            return

        # Check for CSV files
        csv_files = list(path.glob("*.csv"))
        if csv_files:
            if len(csv_files) == 1:
                # Single CSV - use as spectral data
                self.spectral_data_path.set(str(csv_files[0]))
                self.detected_type = "csv"
                self.detection_status.config(
                    text="✓ Detected CSV spectra file - select reference CSV below",
                    foreground=self.colors['success']
                )
                # No popup needed - status label guides user
            else:
                # Multiple CSVs - need user to clarify
                self.detected_type = "csv"
                self.detection_status.config(
                    text=f"⚠ Found {len(csv_files)} CSV files - select files manually",
                    foreground=self.colors['accent']
                )
                # No popup needed - status label guides user
            return

        # Check for SPC files (GRAMS/Thermo Galactic)
        spc_files = list(path.glob("*.spc"))
        if spc_files:
            self.detected_type = "spc"
            self.detection_status.config(
                text=f"✓ Detected {len(spc_files)} SPC files",
                foreground=self.colors['success']
            )

            # Auto-detect reference CSV
            csv_files = list(path.glob("*.csv"))
            if len(csv_files) == 1:
                self.reference_file.set(str(csv_files[0]))
                self._auto_detect_columns()
                # No popup needed - status label shows detection
            elif len(csv_files) > 1:
                # Update status to guide user - no popup needed
                self.detection_status.config(
                    text=f"✓ Detected {len(spc_files)} SPC files - {len(csv_files)} CSVs found, select reference manually",
                    foreground=self.colors['accent']
                )
            return

        # No supported files found
        self.detected_type = None
        self.detection_status.config(
            text="✗ No supported spectral files found",
            foreground='red'
        )
        messagebox.showwarning("No Spectral Data",
            "No supported spectral files found in this directory.\n\nSupported formats:\n• .asd (ASD files)\n• .csv (CSV spectral data)\n• .spc (GRAMS/Thermo Galactic)")

    def _browse_reference_file(self):
        """Browse for reference CSV file."""
        filename = filedialog.askopenfilename(title="Select Reference CSV", filetypes=[("CSV files", "*.csv")])
        if filename:
            self.reference_file.set(filename)
            self._auto_detect_columns()

    def _auto_detect_columns(self):
        """Auto-detect column names from reference CSV."""
        if not self.reference_file.get():
            return

        try:
            df = pd.read_csv(self.reference_file.get(), nrows=5)
            columns = list(df.columns)

            # Update comboboxes
            self.spectral_file_combo['values'] = columns
            self.id_combo['values'] = columns
            self.target_combo['values'] = columns

            # Auto-select if possible
            if len(columns) >= 3:
                self.spectral_file_column.set(columns[0])
                self.id_column.set(columns[1])
                self.target_column.set(columns[2])

            self.tab1_status.config(text=f"✓ Detected {len(columns)} columns")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read reference file:\n{e}")

    def _load_and_plot_data(self):
        """Load data and generate spectral plots."""
        try:
            from spectral_predict.io import read_csv_spectra, read_reference_csv, align_xy, read_asd_dir, read_spc_dir

            self.tab1_status.config(text="Loading data...")
            self.root.update()

            # Check if spectral data has been selected and detected
            if not self.spectral_data_path.get():
                messagebox.showwarning("Missing Input", "Please select spectral data directory")
                return

            if not self.detected_type:
                messagebox.showwarning("No Data Detected",
                    "Could not detect spectral data type.\n\nPlease ensure the directory contains:\n• .asd files\n• .csv files\n• .spc files (GRAMS)")
                return

            # Load spectral data based on detected type
            if self.detected_type == "asd":
                X = read_asd_dir(self.spectral_data_path.get())
            elif self.detected_type == "csv":
                X = read_csv_spectra(self.spectral_data_path.get())
            elif self.detected_type == "spc":
                X = read_spc_dir(self.spectral_data_path.get())
            else:
                messagebox.showerror("Error", f"Unknown data type: {self.detected_type}")
                return

            # Load reference data
            if not self.reference_file.get():
                messagebox.showwarning("Missing Input", "Please select reference CSV file")
                return

            ref = read_reference_csv(self.reference_file.get(), self.spectral_file_column.get())

            # Align data
            X_aligned, y_aligned = align_xy(X, ref, self.spectral_file_column.get(), self.target_column.get())

            # Store original unfiltered data
            self.X_original = X_aligned
            self.y = y_aligned
            self.ref = ref

            # Auto-populate wavelength range ONLY if empty
            if not self.wavelength_min.get().strip() and not self.wavelength_max.get().strip():
                wavelengths = self.X_original.columns.astype(float)
                self.wavelength_min.set(str(int(wavelengths.min())))
                self.wavelength_max.set(str(int(wavelengths.max())))

            # Apply wavelength filtering
            self._apply_wavelength_filter()

            # Generate plots
            self._generate_plots()

            self.tab1_status.config(text=f"✓ Loaded {len(self.X)} samples × {self.X.shape[1]} wavelengths")
            # Enable interactive controls
            self.update_wl_button.config(state='normal')
            self.absorbance_checkbox.config(state='normal')
            self.reset_exclusions_button.config(state='normal')
            # No popup needed - status label shows success and plots are visible

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load data:\n{e}")
            self.tab1_status.config(text="✗ Error loading data")

    def _apply_wavelength_filter(self):
        """Apply wavelength filtering to X_original and store in self.X."""
        if self.X_original is None:
            return

        # Get wavelength range
        wl_min = self.wavelength_min.get().strip()
        wl_max = self.wavelength_max.get().strip()

        # Start with full data
        self.X = self.X_original.copy()

        # Apply filtering
        if wl_min or wl_max:
            wavelengths = self.X.columns.astype(float)
            if wl_min:
                self.X = self.X.loc[:, wavelengths >= float(wl_min)]
            if wl_max:
                wavelengths = self.X.columns.astype(float)
                self.X = self.X.loc[:, wavelengths <= float(wl_max)]

    def _update_wavelengths(self):
        """Update wavelength filter and regenerate plots."""
        if self.X_original is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        try:
            # Validate wavelength inputs
            wl_min = self.wavelength_min.get().strip()
            wl_max = self.wavelength_max.get().strip()

            if wl_min:
                float(wl_min)  # Validate it's a number
            if wl_max:
                float(wl_max)  # Validate it's a number

            # Apply new filter
            self._apply_wavelength_filter()

            # Regenerate plots
            self._generate_plots()

            # Update status
            self.tab1_status.config(text=f"✓ Updated to {len(self.X)} samples × {self.X.shape[1]} wavelengths")

        except ValueError as e:
            messagebox.showerror("Invalid Input", "Wavelength values must be numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update wavelengths:\n{e}")

    def _toggle_absorbance(self):
        """Toggle between reflectance and absorbance display."""
        if self.X is None:
            return

        # Regenerate plots with current transformation state
        self._generate_plots()

    def _reset_exclusions(self):
        """Reset all spectrum exclusions."""
        self.excluded_spectra.clear()
        self._update_exclusion_status()
        # Regenerate plots to restore all spectra
        self._generate_plots()

    def _update_exclusion_status(self):
        """Update the exclusion status label."""
        n_excluded = len(self.excluded_spectra)
        if n_excluded == 0:
            self.exclusion_status.config(text="No spectra excluded")
        elif n_excluded == 1:
            self.exclusion_status.config(text="1 spectrum excluded")
        else:
            self.exclusion_status.config(text=f"{n_excluded} spectra excluded")

    def _on_spectrum_click(self, event):
        """Handle clicking on a spectrum line to toggle its visibility."""
        line = event.artist
        sample_idx = int(line.get_gid())  # Get stored sample index

        if sample_idx in self.excluded_spectra:
            # Re-include the spectrum
            self.excluded_spectra.remove(sample_idx)
            line.set_alpha(0.3)  # Restore normal alpha
            line.set_linewidth(1.0)  # Restore normal linewidth
        else:
            # Exclude the spectrum
            self.excluded_spectra.add(sample_idx)
            line.set_alpha(0.05)  # Make nearly transparent
            line.set_linewidth(0.5)  # Make thinner

        event.canvas.draw()
        self._update_exclusion_status()

    def _apply_transformation(self, data):
        """Apply absorbance transformation if enabled."""
        if self.use_absorbance.get():
            # Convert reflectance to absorbance: A = log10(1/R)
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            data_safe = np.maximum(data, epsilon)
            return np.log10(1.0 / data_safe)
        else:
            return data

    def _generate_plots(self):
        """Generate spectral plots in the plot notebook."""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Matplotlib Required", "Matplotlib is required for plotting")
            return

        # Clear existing plots
        for widget in self.plot_notebook.winfo_children():
            widget.destroy()

        # Determine y-axis label based on transformation state
        ylabel = "Absorbance" if self.use_absorbance.get() else "Reflectance"

        # Apply transformation to raw data
        data_transformed = self._apply_transformation(self.X.values)

        # Create plot tabs
        self._create_plot_tab("Raw Spectra", data_transformed, ylabel, "blue", is_raw=True)

        # Generate derivative plots if available
        if HAS_DERIVATIVES:
            # 1st derivative
            deriv1 = SavgolDerivative(deriv=1, window=7)
            X_deriv1 = deriv1.transform(self.X.values)
            self._create_plot_tab("1st Derivative", X_deriv1, "First Derivative", "green", is_raw=False)

            # 2nd derivative
            deriv2 = SavgolDerivative(deriv=2, window=7)
            X_deriv2 = deriv2.transform(self.X.values)
            self._create_plot_tab("2nd Derivative", X_deriv2, "Second Derivative", "red", is_raw=False)
        else:
            messagebox.showwarning("Derivatives Unavailable",
                "Could not import SavgolDerivative. Only raw spectra will be plotted.")

    def _create_plot_tab(self, title, data, ylabel, color, is_raw=False):
        """Create an interactive plot tab with click-to-toggle and zoom/pan.

        Args:
            title: Title for the plot tab
            data: Spectral data to plot (samples x wavelengths)
            ylabel: Y-axis label
            color: Line color
            is_raw: Whether this is raw data (enables click interaction only for raw)
        """
        frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(frame, text=title)

        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        wavelengths = self.X.columns.values
        n_samples = len(data)

        # Determine plotting strategy
        if n_samples <= 50:
            alpha = 0.3
            indices = range(n_samples)
        else:
            alpha = 0.5
            indices = np.random.choice(n_samples, size=50, replace=False)

        # Plot with interactive features (only for raw spectra to keep it simple)
        for i in indices:
            # Determine if this spectrum is currently excluded
            if i in self.excluded_spectra:
                current_alpha = 0.05
                current_linewidth = 0.5
            else:
                current_alpha = alpha
                current_linewidth = 1.0

            line, = ax.plot(wavelengths, data[i, :], alpha=current_alpha,
                          color=color, linewidth=current_linewidth)

            # Make clickable only for raw spectra
            if is_raw:
                line.set_gid(str(i))  # Store sample index as gid
                line.set_picker(5)  # Enable picking with 5-point tolerance

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} (n={n_samples})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if "Derivative" in title:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Add click handler only for raw spectra
        if is_raw:
            fig.canvas.mpl_connect('pick_event', self._on_spectrum_click)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        # Add navigation toolbar for zoom/pan
        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Pack canvas below toolbar
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ========== OUTLIER DETECTION METHODS (Phase 3) ==========

    def _run_outlier_detection(self):
        """Run outlier detection analysis."""
        if not HAS_OUTLIER_DETECTION:
            messagebox.showerror("Module Missing",
                "Outlier detection module not found. Please check installation.")
            return

        if self.X is None or self.y is None:
            messagebox.showerror("Error", "Please load data first in the 'Import & Preview' tab")
            return

        try:
            # Get parameters
            n_components = self.n_pca_components.get()
            y_min = float(self.y_min_bound.get()) if self.y_min_bound.get().strip() else None
            y_max = float(self.y_max_bound.get()) if self.y_max_bound.get().strip() else None

            # Apply absorbance transformation if enabled
            X_data = self._apply_transformation(self.X.values)

            # Run detection
            self.tab2_status.config(text="Running outlier detection...")
            self.root.update()

            self.outlier_report = generate_outlier_report(
                X_data, self.y.values, n_components, y_min, y_max
            )

            # Update visualizations
            self._plot_pca_scores()
            self._plot_hotelling_t2()
            self._plot_q_residuals()
            self._plot_mahalanobis()
            self._plot_y_distribution()

            # Populate table
            self._populate_outlier_table()

            # Update status
            n_high = len(self.outlier_report['high_confidence_outliers'])
            n_moderate = len(self.outlier_report['moderate_confidence_outliers'])
            n_low = len(self.outlier_report['low_confidence_outliers'])

            self.tab2_status.config(
                text=f"Detection complete: {n_high} high confidence, {n_moderate} moderate, {n_low} low confidence outliers"
            )

            messagebox.showinfo("Success",
                f"Outlier detection complete!\n\n"
                f"High confidence (3+ flags): {n_high}\n"
                f"Moderate confidence (2 flags): {n_moderate}\n"
                f"Low confidence (1 flag): {n_low}")

        except Exception as e:
            messagebox.showerror("Error", f"Outlier detection failed:\n{str(e)}")
            self.tab2_status.config(text="Error during outlier detection")

    def _plot_pca_scores(self):
        """Plot PCA scores (PC1 vs PC2) colored by Y value."""
        if not HAS_MATPLOTLIB or self.outlier_report is None:
            return

        # Clear existing plot
        for widget in self.pca_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        scores = self.outlier_report['pca']['scores']
        y_values = self.y.values
        outliers = self.outlier_report['pca']['outlier_flags']

        # Scatter plot
        scatter = ax.scatter(scores[:, 0], scores[:, 1], c=y_values,
                           cmap='viridis', alpha=0.6, edgecolors='black', linewidths=0.5)

        # Highlight outliers
        if np.any(outliers):
            ax.scatter(scores[outliers, 0], scores[outliers, 1],
                      facecolors='none', edgecolors='red', linewidths=2, s=100,
                      label='T² Outliers')

        ax.set_xlabel(f'PC1 ({self.outlier_report["pca"]["variance_explained"][0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({self.outlier_report["pca"]["variance_explained"][1]*100:.1f}%)')
        ax.set_title('PCA Score Plot (PC1 vs PC2)')
        ax.grid(True, alpha=0.3)
        if np.any(outliers):
            ax.legend()

        fig.colorbar(scatter, ax=ax, label='Y Value')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.pca_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_hotelling_t2(self):
        """Plot Hotelling T² statistics with threshold."""
        if not HAS_MATPLOTLIB or self.outlier_report is None:
            return

        # Clear existing plot
        for widget in self.t2_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        t2_values = self.outlier_report['pca']['hotelling_t2']
        threshold = self.outlier_report['pca']['t2_threshold']
        outliers = self.outlier_report['pca']['outlier_flags']

        # Bar chart
        colors = ['red' if o else 'steelblue' for o in outliers]
        ax.bar(range(len(t2_values)), t2_values, color=colors, alpha=0.7)
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'95% Threshold ({threshold:.2f})')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Hotelling T²')
        ax.set_title('Hotelling T² Statistic')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.t2_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_q_residuals(self):
        """Plot Q-residuals with threshold."""
        if not HAS_MATPLOTLIB or self.outlier_report is None:
            return

        # Clear existing plot
        for widget in self.q_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        q_values = self.outlier_report['q_residuals']['q_residuals']
        threshold = self.outlier_report['q_residuals']['q_threshold']
        outliers = self.outlier_report['q_residuals']['outlier_flags']

        # Bar chart
        colors = ['red' if o else 'steelblue' for o in outliers]
        ax.bar(range(len(q_values)), q_values, color=colors, alpha=0.7)
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'95% Threshold ({threshold:.2f})')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Q-Residual (SPE)')
        ax.set_title('Q-Residuals (Squared Prediction Error)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.q_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_mahalanobis(self):
        """Plot Mahalanobis distances with threshold."""
        if not HAS_MATPLOTLIB or self.outlier_report is None:
            return

        # Clear existing plot
        for widget in self.maha_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        distances = self.outlier_report['mahalanobis']['distances']
        threshold = self.outlier_report['mahalanobis']['threshold']
        outliers = self.outlier_report['mahalanobis']['outlier_flags']

        # Bar chart
        colors = ['red' if o else 'steelblue' for o in outliers]
        ax.bar(range(len(distances)), distances, color=colors, alpha=0.7)
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold ({threshold:.2f})')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Mahalanobis Distance')
        ax.set_title('Mahalanobis Distance in PCA Space')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.maha_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_y_distribution(self):
        """Plot Y value distribution with outliers highlighted."""
        if not HAS_MATPLOTLIB or self.outlier_report is None:
            return

        # Clear existing plot
        for widget in self.y_dist_plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        y_values = self.y.values
        outliers = self.outlier_report['y_consistency']['all_outliers']

        # Histogram
        ax1.hist(y_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        if np.any(outliers):
            ax1.hist(y_values[outliers], bins=30, color='red', alpha=0.7,
                    edgecolor='black', label='Y Outliers')
            ax1.legend()
        ax1.set_xlabel('Y Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Y Value Distribution')
        ax1.grid(True, alpha=0.3, axis='y')

        # Box plot
        bp = ax2.boxplot([y_values], vert=False, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)

        if np.any(outliers):
            ax2.scatter(y_values[outliers], np.ones(np.sum(outliers)),
                       color='red', s=100, zorder=3, label='Y Outliers')
            ax2.legend()

        ax2.set_xlabel('Y Value')
        ax2.set_title('Y Value Box Plot')
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3, axis='x')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.y_dist_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _populate_outlier_table(self):
        """Populate the outlier summary table."""
        if self.outlier_report is None:
            return

        # Clear existing
        for item in self.outlier_tree.get_children():
            self.outlier_tree.delete(item)

        # Get summary
        summary = self.outlier_report['outlier_summary']

        # Add rows
        for idx, row in summary.iterrows():
            values = (
                row['Sample_Index'],
                f"{row['Y_Value']:.2f}",
                "✓" if row['T2_Outlier'] else "",
                "✓" if row['Q_Outlier'] else "",
                "✓" if row['Maha_Outlier'] else "",
                "✓" if row['Y_Outlier'] else "",
                row['Total_Flags']
            )

            # Color code by flags
            if row['Total_Flags'] >= 3:
                tag = 'high'
            elif row['Total_Flags'] == 2:
                tag = 'moderate'
            else:
                tag = 'normal'

            self.outlier_tree.insert('', 'end', values=values, tags=(tag,))

        # Configure tags
        self.outlier_tree.tag_configure('high', background='#ffcccc')
        self.outlier_tree.tag_configure('moderate', background='#ffffcc')
        self.outlier_tree.tag_configure('normal', background='white')

    def _auto_select_flagged(self):
        """Auto-select all flagged samples."""
        if self.outlier_report is None:
            return

        # Clear selection
        for item in self.outlier_tree.selection():
            self.outlier_tree.selection_remove(item)

        if self.select_all_flagged.get():
            # Select all with at least 1 flag
            summary = self.outlier_report['outlier_summary']
            flagged = summary[summary['Total_Flags'] > 0]

            for idx, row in flagged.iterrows():
                # Find the tree item corresponding to this row
                for item in self.outlier_tree.get_children():
                    if int(self.outlier_tree.item(item, 'values')[0]) == row['Sample_Index']:
                        self.outlier_tree.selection_add(item)
                        break

        self._update_outlier_selection_status()

    def _auto_select_high_confidence(self):
        """Auto-select high confidence outliers (3+ flags)."""
        if self.outlier_report is None:
            return

        # Clear selection
        for item in self.outlier_tree.selection():
            self.outlier_tree.selection_remove(item)

        if self.select_high_conf.get():
            # Select samples with 3+ flags
            summary = self.outlier_report['outlier_summary']
            high_conf = summary[summary['Total_Flags'] >= 3]

            for idx, row in high_conf.iterrows():
                for item in self.outlier_tree.get_children():
                    if int(self.outlier_tree.item(item, 'values')[0]) == row['Sample_Index']:
                        self.outlier_tree.selection_add(item)
                        break

        self._update_outlier_selection_status()

    def _auto_select_moderate_confidence(self):
        """Auto-select moderate confidence outliers (2 flags)."""
        if self.outlier_report is None:
            return

        # Clear selection
        for item in self.outlier_tree.selection():
            self.outlier_tree.selection_remove(item)

        if self.select_moderate_conf.get():
            # Select samples with 2 flags
            summary = self.outlier_report['outlier_summary']
            moderate_conf = summary[summary['Total_Flags'] == 2]

            for idx, row in moderate_conf.iterrows():
                for item in self.outlier_tree.get_children():
                    if int(self.outlier_tree.item(item, 'values')[0]) == row['Sample_Index']:
                        self.outlier_tree.selection_add(item)
                        break

        self._update_outlier_selection_status()

    def _update_outlier_selection_status(self):
        """Update the selection status label."""
        n_selected = len(self.outlier_tree.selection())
        self.outlier_selection_status.config(text=f"{n_selected} samples selected")

    def _mark_selected_for_exclusion(self):
        """Add selected samples to unified exclusion set."""
        selected = self.outlier_tree.selection()

        if not selected:
            messagebox.showwarning("No Selection", "Please select samples to exclude")
            return

        # Get selected indices
        added_count = 0
        for item in selected:
            values = self.outlier_tree.item(item, 'values')
            sample_idx = int(values[0])
            if sample_idx not in self.excluded_spectra:
                self.excluded_spectra.add(sample_idx)
                added_count += 1

        # Update plots in Tab 1 if data is loaded
        if self.X is not None:
            self._generate_plots()

        messagebox.showinfo("Success",
            f"Added {added_count} new samples to exclusion list.\n"
            f"Total excluded: {len(self.excluded_spectra)}\n\n"
            f"These samples will be excluded from analysis.")

    def _export_outlier_report(self):
        """Export outlier detection report to CSV."""
        if self.outlier_report is None:
            messagebox.showerror("Error", "Run outlier detection first")
            return

        try:
            # Ask for file location
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Outlier Report"
            )

            if filepath:
                summary = self.outlier_report['outlier_summary']
                summary.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Report exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report:\n{str(e)}")

    # ========== END OUTLIER DETECTION METHODS ==========

    def _export_preprocessed_csv(self, window_size=None):
        """
        Export preprocessed spectral data (2nd derivative) to CSV.

        Parameters
        ----------
        window_size : int, optional
            Window size for Savitzky-Golay filter. If None, uses first selected window size or defaults to 17.
        """
        try:
            from spectral_predict.preprocess import SavgolDerivative
            import pandas as pd

            # Determine window size
            if window_size is None:
                # Collect window sizes from checkboxes
                window_sizes = []
                if self.window_7.get():
                    window_sizes.append(7)
                if self.window_11.get():
                    window_sizes.append(11)
                if self.window_17.get():
                    window_sizes.append(17)
                if self.window_19.get():
                    window_sizes.append(19)

                # Use first selected or default to 17
                window_size = window_sizes[0] if window_sizes else 17

            # Apply second derivative preprocessing
            preprocessor = SavgolDerivative(deriv=2, window=window_size, polyorder=3)
            X_preprocessed = preprocessor.transform(self.X.values)

            # Create DataFrame with wavelength column names
            df_preprocessed = pd.DataFrame(
                X_preprocessed,
                columns=self.X.columns,
                index=self.X.index
            )

            # Add response variable as first column
            target_name = self.target_column.get()
            df_export = pd.DataFrame({target_name: self.y})
            df_export = pd.concat([df_export, df_preprocessed], axis=1)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = output_dir / f"preprocessed_data_{target_name}_w{window_size}_{timestamp}.csv"

            # Save to CSV
            df_export.to_csv(csv_path, index=False)

            # Log success
            self._log_progress(f"✓ Preprocessed CSV exported: {csv_path}")
            self._log_progress(f"  - Window size: {window_size}, Polyorder: 3 (2nd derivative)")
            self._log_progress(f"  - Shape: {df_export.shape[0]} samples × {df_export.shape[1]} columns")

            return str(csv_path)

        except Exception as e:
            error_msg = f"Failed to export preprocessed CSV: {str(e)}"
            self._log_progress(f"✗ {error_msg}")
            messagebox.showerror("Export Error", error_msg)
            return None

    def _run_analysis(self):
        """Run analysis in background thread."""
        # Validate data is loaded
        if self.X is None or self.y is None:
            messagebox.showwarning("No Data", "Please load data first in the 'Import & Preview' tab")
            return

        # Validate at least one model selected
        selected_models = []
        if self.use_pls.get():
            selected_models.append("PLS")
        if self.use_ridge.get():
            selected_models.append("Ridge")
        if self.use_lasso.get():
            selected_models.append("Lasso")
        if self.use_randomforest.get():
            selected_models.append("RandomForest")
        if self.use_mlp.get():
            selected_models.append("MLP")
        if self.use_neuralboosted.get():
            selected_models.append("NeuralBoosted")

        if not selected_models:
            messagebox.showwarning("No Models", "Please select at least one model to test")
            return

        # Switch to Analysis Progress tab (index 3)
        # Tab indices: 0=Import, 1=Quality Check, 2=Analysis Config, 3=Analysis Progress, 4=Results, 5=Custom Model Dev
        self.notebook.select(3)

        # Clear progress text
        self.progress_text.delete('1.0', tk.END)
        self.progress_info.config(text="Starting analysis...")
        self.progress_status.config(text="Analysis in progress...")
        self.best_model_info.config(text="(none yet)")
        self.time_estimate_label.config(text="")

        # Reset start time
        self.analysis_start_time = datetime.now()

        # Export preprocessed CSV if requested
        if self.export_preprocessed_csv.get():
            self._log_progress("\n" + "="*70)
            self._log_progress("EXPORTING PREPROCESSED DATA CSV")
            self._log_progress("="*70)
            csv_path = self._export_preprocessed_csv()
            if csv_path:
                self._log_progress(f"✓ CSV export complete\n")
            else:
                self._log_progress(f"✗ CSV export failed\n")

        # Run in thread
        self.analysis_thread = threading.Thread(target=self._run_analysis_thread, args=(selected_models,))
        self.analysis_thread.start()

    def _run_analysis_thread(self, selected_models):
        """Run analysis in background thread."""
        try:
            from spectral_predict.search import run_search
            from spectral_predict.report import write_markdown_report

            # Determine task type
            if self.y.nunique() == 2:
                task_type = "classification"
            elif self.y.dtype == 'object' or self.y.nunique() < 10:
                task_type = "classification"
            else:
                task_type = "regression"

            # Collect preprocessing method selections
            preprocessing_methods = {
                'raw': self.use_raw.get(),
                'snv': self.use_snv.get(),
                'sg1': self.use_sg1.get(),
                'sg2': self.use_sg2.get(),
                'deriv_snv': self.use_deriv_snv.get()
            }

            # Collect subset analysis settings
            enable_variable_subsets = self.enable_variable_subsets.get()
            enable_region_subsets = self.enable_region_subsets.get()

            # DEBUG: Print what we're getting from the GUI
            print("\n" + "="*70)
            print("GUI DEBUG: Subset Analysis Settings")
            print("="*70)
            print(f"enable_variable_subsets checkbox value: {self.enable_variable_subsets.get()}")
            print(f"enable_region_subsets checkbox value: {self.enable_region_subsets.get()}")

            # Collect top-N variable counts
            variable_counts = []
            print(f"var_10 checkbox: {self.var_10.get()}")
            if self.var_10.get():
                variable_counts.append(10)
            print(f"var_20 checkbox: {self.var_20.get()}")
            if self.var_20.get():
                variable_counts.append(20)
            print(f"var_50 checkbox: {self.var_50.get()}")
            if self.var_50.get():
                variable_counts.append(50)
            print(f"var_100 checkbox: {self.var_100.get()}")
            if self.var_100.get():
                variable_counts.append(100)
            print(f"var_250 checkbox: {self.var_250.get()}")
            if self.var_250.get():
                variable_counts.append(250)
            print(f"var_500 checkbox: {self.var_500.get()}")
            if self.var_500.get():
                variable_counts.append(500)
            print(f"var_1000 checkbox: {self.var_1000.get()}")
            if self.var_1000.get():
                variable_counts.append(1000)

            print(f"\nCollected variable_counts: {variable_counts}")
            print(f"Final enable_variable_subsets: {enable_variable_subsets}")
            print(f"Final enable_region_subsets: {enable_region_subsets}")
            print(f"Final n_top_regions: {self.n_top_regions.get()}")
            print("="*70 + "\n")

            # Collect window sizes from checkboxes
            window_sizes = []
            if self.window_7.get():
                window_sizes.append(7)
            if self.window_11.get():
                window_sizes.append(11)
            if self.window_17.get():
                window_sizes.append(17)
            if self.window_19.get():
                window_sizes.append(19)

            # Default to window size 17 if none specified
            if not window_sizes:
                window_sizes = [17]

            # Collect n_estimators options
            n_estimators_list = []
            if self.n_estimators_50.get():
                n_estimators_list.append(50)
            if self.n_estimators_100.get():
                n_estimators_list.append(100)

            # Default to 100 if none selected
            if not n_estimators_list:
                n_estimators_list = [100]

            # Collect learning rate options
            learning_rates = []
            if self.lr_005.get():
                learning_rates.append(0.05)
            if self.lr_01.get():
                learning_rates.append(0.1)
            if self.lr_02.get():
                learning_rates.append(0.2)

            # Default to [0.1, 0.2] if none selected
            if not learning_rates:
                learning_rates = [0.1, 0.2]

            self._log_progress(f"\n{'='*70}")
            self._log_progress(f"ANALYSIS CONFIGURATION")
            self._log_progress(f"{'='*70}")
            self._log_progress(f"Task type: {task_type}")
            self._log_progress(f"Models: {', '.join(selected_models)}")
            self._log_progress(f"Preprocessing: {', '.join([k for k, v in preprocessing_methods.items() if v])}")
            self._log_progress(f"Window sizes: {window_sizes}")
            self._log_progress(f"n_estimators: {n_estimators_list}")
            self._log_progress(f"Learning rates: {learning_rates}")
            self._log_progress(f"\n** SUBSET ANALYSIS SETTINGS **")
            self._log_progress(f"Variable subsets: {'ENABLED' if enable_variable_subsets else 'DISABLED'}")
            self._log_progress(f"  enable_variable_subsets value: {enable_variable_subsets}")
            if enable_variable_subsets:
                self._log_progress(f"  Variable counts selected: {variable_counts if variable_counts else 'NONE'}")
                if not variable_counts:
                    self._log_progress(f"  ⚠️ WARNING: Variable subsets enabled but no counts selected!")
            else:
                self._log_progress(f"  ⚠️ Variable subsets are DISABLED - no subset analysis will run")
            self._log_progress(f"Region subsets: {'ENABLED' if enable_region_subsets else 'DISABLED'}")
            if enable_region_subsets:
                self._log_progress(f"  Region analysis depth: {self.n_top_regions.get()} regions")
            self._log_progress(f"Data: {len(self.X)} samples × {self.X.shape[1]} wavelengths")
            self._log_progress(f"{'='*70}\n")

            # Run search
            # Filter out excluded spectra
            if self.excluded_spectra:
                mask = ~np.isin(np.arange(len(self.X)), list(self.excluded_spectra))
                X_filtered = self.X[mask]
                y_filtered = self.y[mask]

                # Update progress with exclusion info
                self.root.after(0, lambda: self.progress_text.insert(tk.END,
                    f"\nℹ️ Excluding {len(self.excluded_spectra)} user-selected spectra from analysis...\n"))
                self.root.after(0, lambda: self.progress_text.see(tk.END))
            else:
                X_filtered = self.X
                y_filtered = self.y

            # Parse UVE n_components (empty string = None)
            uve_n_comp = None
            if self.uve_n_components.get().strip():
                try:
                    uve_n_comp = int(self.uve_n_components.get())
                except ValueError:
                    self._log_progress("⚠️ Warning: Invalid UVE n_components, using auto-determination")

            results_df = run_search(
                X_filtered,
                y_filtered,
                task_type=task_type,
                folds=self.folds.get(),
                lambda_penalty=self.lambda_penalty.get(),
                max_n_components=self.max_n_components.get(),
                max_iter=self.max_iter.get(),
                models_to_test=selected_models,
                preprocessing_methods=preprocessing_methods,
                window_sizes=window_sizes,
                n_estimators_list=n_estimators_list,
                learning_rates=learning_rates,
                enable_variable_subsets=enable_variable_subsets,
                variable_counts=variable_counts if variable_counts else None,
                enable_region_subsets=enable_region_subsets,
                n_top_regions=self.n_top_regions.get(),
                progress_callback=self._progress_callback,
                # Variable selection parameters (NEW)
                variable_selection_method=self.variable_selection_method.get(),
                apply_uve_prefilter=self.apply_uve_prefilter.get(),
                uve_cutoff_multiplier=self.uve_cutoff_multiplier.get(),
                uve_n_components=uve_n_comp,
                spa_n_random_starts=self.spa_n_random_starts.get(),
                ipls_n_intervals=self.ipls_n_intervals.get()
            )

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            results_path = output_dir / f"results_{self.target_column.get()}_{timestamp}.csv"
            results_df.to_csv(results_path, index=False)

            # Generate report
            report_dir = Path("reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            write_markdown_report(self.target_column.get(), results_df, str(report_dir))

            # Store results for Results tab
            self.results_df = results_df

            # Populate Results tab
            self.root.after(0, lambda: self._populate_results_table(results_df))

            self._log_progress(f"\n✓ Analysis complete!")
            self._log_progress(f"Results saved to: {results_path}")

            self.root.after(0, lambda: self.progress_status.config(text="✓ Analysis complete!"))
            self.root.after(0, lambda: self.progress_info.config(text="Analysis Complete"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Analysis complete!\n\nResults: {results_path}\n\nView results in the 'Results' tab."))

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self._log_progress(f"\n✗ Error: {e}\n{error_msg}")
            self.root.after(0, lambda: self.progress_status.config(text="✗ Analysis failed"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{e}"))

    def _progress_callback(self, info):
        """Handle progress updates."""
        msg = info.get('message', '')
        self._log_progress(msg)

        current = info.get('current', 0)
        total = info.get('total', 1)
        best_model = info.get('best_model', None)

        # Calculate time estimate
        if self.analysis_start_time and current > 0:
            elapsed = (datetime.now() - self.analysis_start_time).total_seconds()
            time_per_config = elapsed / current
            remaining_configs = total - current
            estimated_remaining = time_per_config * remaining_configs

            # Format time remaining
            if estimated_remaining < 60:
                time_str = f"~{int(estimated_remaining)}s remaining"
            elif estimated_remaining < 3600:
                time_str = f"~{int(estimated_remaining / 60)}m {int(estimated_remaining % 60)}s remaining"
            else:
                hours = int(estimated_remaining / 3600)
                minutes = int((estimated_remaining % 3600) / 60)
                time_str = f"~{hours}h {minutes}m remaining"
        else:
            time_str = "Calculating..."

        # Update progress info
        self.root.after(0, lambda: self.progress_info.config(text=f"Progress: {current}/{total} configurations"))
        self.root.after(0, lambda: self.time_estimate_label.config(text=time_str))

        # Update best model display
        if best_model:
            # Determine task type from best_model dict
            if 'RMSE' in best_model:
                # Regression
                model_text = f"{best_model['Model']} | {best_model['Preprocess']}"
                if best_model.get('Deriv'):
                    model_text += f" (d{best_model['Deriv']})"
                model_text += f"\nRMSE: {best_model['RMSE']:.4f} | R²: {best_model['R2']:.4f}"

                # Add top wavelengths if available
                if 'top_vars' in best_model and best_model['top_vars'] != 'N/A':
                    top_vars = best_model['top_vars'].split(',')[:5]  # First 5
                    model_text += f"\nTop λ: {', '.join(top_vars)} nm"
            else:
                # Classification
                model_text = f"{best_model['Model']} | {best_model['Preprocess']}"
                if best_model.get('Deriv'):
                    model_text += f" (d{best_model['Deriv']})"
                model_text += f"\nAcc: {best_model['Accuracy']:.4f}"
                if 'ROC_AUC' in best_model and not np.isnan(best_model['ROC_AUC']):
                    model_text += f" | AUC: {best_model['ROC_AUC']:.4f}"

                # Add top wavelengths if available
                if 'top_vars' in best_model and best_model['top_vars'] != 'N/A':
                    top_vars = best_model['top_vars'].split(',')[:5]  # First 5
                    model_text += f"\nTop λ: {', '.join(top_vars)} nm"

            self.root.after(0, lambda text=model_text: self.best_model_info.config(text=text))

    def _log_progress(self, message):
        """Log message to progress text area."""
        self.root.after(0, lambda: self._append_progress(message))

    def _append_progress(self, message):
        """Append message to progress text (must be called from main thread)."""
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)

    def _on_tab_changed(self, event):
        """Handle tab change events."""
        pass  # Placeholder for future enhancements

    def _show_help(self):
        """Show help dialog."""
        help_text = """Spectral Predict - Quick Start

1. IMPORT & PREVIEW Tab:
   - Select ASD directory or CSV file
   - Select reference CSV
   - Auto-detect columns
   - Load data to see plots

2. ANALYSIS CONFIGURATION Tab:
   - Configure analysis options
   - Select models to test
   - Run analysis

3. ANALYSIS PROGRESS Tab:
   - Auto-switches during analysis
   - Shows live progress
   - Displays results when complete

4. RESULTS Tab:
   - View all model results
   - Double-click a row to refine

5. REFINE MODEL Tab:
   - Tweak model parameters
   - Run refined models
"""
        messagebox.showinfo("Help", help_text)

    def _populate_results_table(self, results_df):
        """Populate the results table with analysis results."""
        if results_df is None or len(results_df) == 0:
            self.results_status.config(text="No results to display")
            return

        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Set up columns
        columns = list(results_df.columns)
        self.results_tree['columns'] = columns

        # Configure column headings
        for col in columns:
            self.results_tree.heading(col, text=col)
            # Set column width based on content
            if col in ['Model', 'Preprocess', 'Subset']:
                width = 120
            elif col in ['top_vars']:
                width = 200
            else:
                width = 80
            self.results_tree.column(col, width=width, anchor='center')

        # Insert data rows
        for idx, row in results_df.iterrows():
            values = [row[col] for col in columns]
            self.results_tree.insert('', 'end', iid=str(idx), values=values)

        # Update status
        self.results_status.config(text=f"Displaying {len(results_df)} results. Double-click a row to refine the model.")

    def _on_result_double_click(self, event):
        """Handle double-click on a result row."""
        selection = self.results_tree.selection()
        if not selection:
            return

        # Get the selected row index
        item_id = selection[0]
        row_idx = int(item_id)

        if self.results_df is None:
            return

        # Get the selected model configuration
        # CRITICAL: Use .loc (label-based) not .iloc (position-based)
        # because treeview IID uses the dataframe's original index labels
        model_config = self.results_df.loc[row_idx].to_dict()
        self.selected_model_config = model_config

        # Validation logging
        rank = model_config.get('Rank', 'N/A')
        r2_or_acc = model_config.get('R2', model_config.get('Accuracy', 'N/A'))
        model_name = model_config.get('Model', 'N/A')
        print(f"✓ Loading Rank {rank}: {model_name} (R²/Acc={r2_or_acc}, n_vars={model_config.get('n_vars', 'N/A')})")

        # Populate the Custom Model Development tab
        self._load_model_for_refinement(model_config)

        # Switch to the Custom Model Development tab
        self.notebook.select(5)  # Tab 6 (index 5)

    def _validate_data_for_refinement(self):
        """Validate that required data is available for refinement."""
        if self.X_original is None:
            messagebox.showwarning(
                "Data Not Loaded",
                "Please load data in the Data Upload tab before using Custom Model Development."
            )
            return False

        if self.y is None:
            messagebox.showwarning(
                "Target Variable Missing",
                "Please ensure target variable (y) is loaded in the Data Upload tab."
            )
            return False

        return True

    def _load_default_parameters(self):
        """Load default parameters for fresh model development."""
        if not self._validate_data_for_refinement():
            return

        # Set all wavelengths
        wavelengths = list(self.X_original.columns.astype(float).values)
        wl_spec = self._format_wavelengths_as_spec(wavelengths)

        self.refine_wl_spec.config(state='normal')
        self.refine_wl_spec.delete('1.0', 'end')
        self.refine_wl_spec.insert('1.0', wl_spec)

        # Default parameters
        self.refine_model_type.set('PLS')
        self.refine_task_type.set('regression')
        self.refine_preprocess.set('raw')
        self.refine_window.set(17)
        self.refine_folds.set(5)
        self.refine_max_iter.set(100)

        # Update model info display
        self.refine_model_info.config(state='normal')
        self.refine_model_info.delete('1.0', 'end')
        self.refine_model_info.insert('1.0', "Default parameters loaded. Ready for fresh model development.")
        self.refine_model_info.config(state='disabled')

        # Update mode label and status
        self.refine_mode_label.config(text="Mode: Fresh Development (Defaults Loaded)")
        self.refine_status.config(text="Ready to develop custom model")

        # Enable the run button
        self.refine_run_button.config(state='normal')

        # Update the wavelength count display
        self._update_wavelength_count()

    def _validate_refinement_parameters(self):
        """Validate all refinement parameters before execution."""
        errors = []

        # Validate wavelength specification
        wl_spec_text = self.refine_wl_spec.get('1.0', 'end')
        if not wl_spec_text or wl_spec_text.strip() == '':
            errors.append("Wavelength specification is empty")

        # Validate model type
        if self.refine_model_type.get() not in ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']:
            errors.append("Invalid model type selected")

        # Validate CV folds
        if self.refine_folds.get() < 3:
            errors.append("CV folds must be at least 3")

        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False

        return True

    def _load_model_for_refinement(self, config):
        """Load a model configuration into the Custom Model Development tab."""
        # Validate data availability
        if not self._validate_data_for_refinement():
            return

        # Build configuration text
        info_text = f"""Model: {config.get('Model', 'N/A')}
Preprocessing: {config.get('Preprocess', 'N/A')}
Subset: {config.get('SubsetTag', config.get('Subset', 'N/A'))}
Window Size: {config.get('Window', 'N/A')}
"""

        # Add performance metrics
        if 'RMSE' in config and not pd.isna(config.get('RMSE')):
            # Regression
            info_text += f"""
Performance (Regression):
  RMSE: {config.get('RMSE', 'N/A')}
  R²: {config.get('R2', 'N/A')}
"""
        elif 'Accuracy' in config and not pd.isna(config.get('Accuracy')):
            # Classification
            info_text += f"""
Performance (Classification):
  Accuracy: {config.get('Accuracy', 'N/A')}
"""
            if 'ROC_AUC' in config and not pd.isna(config['ROC_AUC']):
                info_text += f"  ROC AUC: {config.get('ROC_AUC', 'N/A')}\n"

        # Add wavelength information
        n_vars = config.get('n_vars', 'N/A')
        full_vars = config.get('full_vars', 'N/A')
        subset_tag = config.get('SubsetTag', config.get('Subset', 'full'))

        info_text += f"\nWavelengths: {n_vars} of {full_vars} used"
        if subset_tag != 'full' and subset_tag != 'N/A':
            info_text += f" ({subset_tag})"
        info_text += "\n"

        if 'top_vars' in config and config['top_vars'] != 'N/A':
            top_vars_list = config['top_vars'].split(',')
            n_shown = len(top_vars_list)
            if n_shown < n_vars and n_vars != 'N/A':
                info_text += f"Most important {n_shown} wavelengths shown below (model used {n_vars} total):\n"
            else:
                info_text += f"Wavelengths used:\n"
            info_text += f"  {config['top_vars']}\n"

        # Add hyperparameters if available
        if 'n_estimators' in config and not pd.isna(config['n_estimators']):
            info_text += f"\nHyperparameters:\n"
            info_text += f"  n_estimators: {config.get('n_estimators', 'N/A')}\n"
            if 'learning_rate' in config:
                info_text += f"  learning_rate: {config.get('learning_rate', 'N/A')}\n"

        # Update the info text widget
        self.refine_model_info.config(state='normal')
        self.refine_model_info.delete('1.0', tk.END)
        self.refine_model_info.insert('1.0', info_text)
        self.refine_model_info.config(state='disabled')

        # Populate refinement parameters with current values
        # Populate wavelength specification - Show ONLY wavelengths used in the selected model
        print(f"DEBUG: X_original is None? {self.X_original is None}")

        if self.X_original is not None:
            try:
                print(f"DEBUG: X_original shape: {self.X_original.shape}")

                # Extract all available wavelengths
                all_wavelengths = self.X_original.columns.astype(float).values
                print(f"DEBUG: Total available wavelengths: {len(all_wavelengths)}")

                subset_tag = config.get('SubsetTag', config.get('Subset', 'full'))
                n_vars = config.get('n_vars', len(all_wavelengths))

                # Determine which wavelengths to show
                model_wavelengths = None

                # For subset models: Prefer all_vars (complete list) over top_vars (top 30)
                # This fixes the variable count mismatch for models with >30 variables
                if 'all_vars' in config and config['all_vars'] != 'N/A' and config['all_vars']:
                    print(f"DEBUG: Model has all_vars, parsing complete wavelength list")
                    try:
                        # Parse wavelengths from all_vars string (e.g., "1520.0, 1540.0, 1560.0, ...")
                        all_vars_str = str(config['all_vars']).strip()
                        wavelength_strings = [w.strip() for w in all_vars_str.split(',')]
                        model_wavelengths = [float(w) for w in wavelength_strings if w]
                        model_wavelengths = sorted(model_wavelengths)  # Sort for formatting
                        print(f"DEBUG: Parsed {len(model_wavelengths)} wavelengths from all_vars")
                    except Exception as e:
                        print(f"WARNING: Could not parse all_vars: {e}")
                        model_wavelengths = None

                # Fallback to top_vars for backward compatibility with old results
                if model_wavelengths is None and 'top_vars' in config and config['top_vars'] != 'N/A' and config['top_vars']:
                    print(f"DEBUG: Falling back to top_vars (may be incomplete for large subsets)")
                    try:
                        # Parse wavelengths from top_vars string (e.g., "1520.0, 1540.0, 1560.0")
                        top_vars_str = str(config['top_vars']).strip()
                        wavelength_strings = [w.strip() for w in top_vars_str.split(',')]
                        model_wavelengths = [float(w) for w in wavelength_strings if w]
                        model_wavelengths = sorted(model_wavelengths)  # Sort for formatting
                        print(f"DEBUG: Parsed {len(model_wavelengths)} wavelengths from top_vars")
                    except Exception as e:
                        print(f"WARNING: Could not parse top_vars: {e}")
                        model_wavelengths = None

                # For full models or if parsing failed: Use all wavelengths
                if model_wavelengths is None:
                    if subset_tag == 'full' or subset_tag == 'N/A':
                        print(f"DEBUG: Full model - using all {len(all_wavelengths)} wavelengths")
                        model_wavelengths = list(all_wavelengths)
                    else:
                        # Fallback: use all wavelengths
                        print(f"WARNING: Subset model but no top_vars, using all wavelengths")
                        model_wavelengths = list(all_wavelengths)

                # Format the wavelengths (ranges for consecutive, individual otherwise)
                wl_spec = self._format_wavelengths_as_spec(model_wavelengths)
                print(f"DEBUG: Formatted {len(model_wavelengths)} wavelengths into {len(wl_spec)} character spec")

                # FALLBACK: If formatter returns empty, use simple range
                if not wl_spec or len(wl_spec) == 0:
                    print("WARNING: Formatter returned empty, using fallback")
                    if len(model_wavelengths) > 0:
                        wl_spec = f"{model_wavelengths[0]:.1f}-{model_wavelengths[-1]:.1f}"
                    else:
                        wl_spec = "# ERROR: No wavelengths available"

                # Ensure widget is in normal state before editing
                self.refine_wl_spec.config(state='normal')
                self.refine_wl_spec.delete('1.0', 'end')
                self.refine_wl_spec.insert('1.0', wl_spec)

                # Verify insertion
                content = self.refine_wl_spec.get('1.0', 'end-1c')
                print(f"DEBUG: Text widget content length: {len(content)} characters")

                if len(content) == 0:
                    print("ERROR: Text widget is empty after insertion!")

            except Exception as e:
                print(f"ERROR loading wavelengths: {e}")
                import traceback
                traceback.print_exc()

                # Show error to user in the text widget
                self.refine_wl_spec.config(state='normal')
                self.refine_wl_spec.delete('1.0', 'end')
                self.refine_wl_spec.insert('1.0', f"# ERROR: Could not load wavelengths\n# {str(e)}\n# Please check console for details")

        else:
            print("WARNING: X_original is None - data not loaded")
            self.refine_wl_spec.config(state='normal')
            self.refine_wl_spec.delete('1.0', 'end')
            self.refine_wl_spec.insert('1.0', "# ERROR: Data not loaded\n# Please load data in Data Upload tab first")

        # Set window size
        window = config.get('Window', 17)
        if not pd.isna(window) and window in [7, 11, 17, 19]:
            self.refine_window.set(int(window))

        # Set model type
        model_name = config.get('Model', 'PLS')
        if model_name in ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']:
            self.refine_model_type.set(model_name)

        # Set task type (auto-detect from data)
        if self.y is not None:
            if self.y.nunique() == 2 or self.y.dtype == 'object' or self.y.nunique() < 10:
                self.refine_task_type.set('classification')
            else:
                self.refine_task_type.set('regression')

        # Set preprocessing method
        preprocess = config.get('Preprocess', 'raw')
        deriv = config.get('Deriv', None)

        # Convert from search.py naming to GUI naming
        if preprocess == 'deriv' and deriv == 1:
            gui_preprocess = 'sg1'
        elif preprocess == 'deriv' and deriv == 2:
            gui_preprocess = 'sg2'
        elif preprocess == 'snv_deriv':
            # SNV then derivative - NOW PROPERLY SUPPORTED!
            gui_preprocess = 'snv_sg1' if deriv == 1 else 'snv_sg2'
        elif preprocess == 'deriv_snv':
            gui_preprocess = 'deriv_snv'
        elif preprocess in ['raw', 'snv']:
            gui_preprocess = preprocess
        else:
            gui_preprocess = 'raw'  # Fallback

        if gui_preprocess in ['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']:
            self.refine_preprocess.set(gui_preprocess)

        # Enable the run button
        self.refine_run_button.config(state='normal')
        self.refine_status.config(text=f"Loaded: {config.get('Model', 'N/A')} | {config.get('Preprocess', 'N/A')}")

        # Update mode label to indicate loaded from results
        rank = config.get('Rank', 'N/A')
        self.refine_mode_label.config(text=f"Mode: Loaded from Results (Rank {rank})")

        # Update the wavelength count display
        self._update_wavelength_count()

    def _run_refined_model(self):
        """Run the refined model with user-specified parameters."""
        if self.X is None or self.y is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        # Validate refinement parameters
        if not self._validate_refinement_parameters():
            return

        # Disable button during execution
        self.refine_run_button.config(state='disabled')
        self.refine_status.config(text="Running refined model...")

        # Run in thread
        thread = threading.Thread(target=self._run_refined_model_thread)
        thread.start()

    def _run_refined_model_thread(self):
        """Execute the refined model in a background thread."""
        try:
            from spectral_predict.models import get_model
            from spectral_predict.preprocess import SavgolDerivative, SNV
            from sklearn.model_selection import KFold, StratifiedKFold
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
            from sklearn.base import clone

            # Parse wavelength specification
            available_wl = self.X_original.columns.astype(float).values
            wl_spec_text = self.refine_wl_spec.get('1.0', 'end')
            selected_wl = self._parse_wavelength_spec(wl_spec_text, available_wl)

            if not selected_wl:
                raise ValueError("No valid wavelengths selected. Please check your wavelength specification.")

            # Filter data source to selected wavelengths
            # Create mapping from float wavelengths to actual column names
            if self.X is not None:
                wavelength_columns = self.X.columns
            else:
                wavelength_columns = self.X_original.columns
            wl_to_col = {float(col): col for col in wavelength_columns}

            # Get the actual column names for selected wavelengths
            selected_cols = [wl_to_col[wl] for wl in selected_wl if wl in wl_to_col]

            if not selected_cols:
                raise ValueError(f"Could not find matching wavelengths. Selected: {len(selected_wl)}, Found: 0")

            # Determine how many folds we'll run so we can validate sample counts
            n_folds = self.refine_folds.get()

            # Determine data source (respect current wavelength filter)
            if self.X is not None:
                X_source = self.X
            else:
                X_source = self.X_original

            # Align sample selection with the main analysis (respect excluded spectra)
            total_samples = len(X_source)
            excluded_indices = sorted(idx for idx in self.excluded_spectra if 0 <= idx < total_samples)
            if excluded_indices:
                excluded_set = set(excluded_indices)
                include_indices = [i for i in range(total_samples) if i not in excluded_set]
                if len(include_indices) < n_folds:
                    raise ValueError(
                        f"Only {len(include_indices)} samples remain after exclusions; "
                        f"{n_folds}-fold CV requires at least {n_folds} samples."
                    )
                print(f"DEBUG: Applying {len(excluded_indices)} excluded spectra for refinement "
                      f"({len(include_indices)} samples remain).")
                X_base_df = X_source.iloc[include_indices]
                y_series = self.y.iloc[include_indices]
            else:
                X_base_df = X_source
                y_series = self.y

            wl_summary = f"{len(selected_wl)} wavelengths ({selected_wl[0]:.1f} to {selected_wl[-1]:.1f} nm)"

            # Get user-selected preprocessing method and map to build_preprocessing_pipeline format
            preprocess = self.refine_preprocess.get()
            window = self.refine_window.get()

            # Map GUI preprocessing names to search.py format
            preprocess_name_map = {
                'raw': 'raw',
                'snv': 'snv',
                'sg1': 'deriv',
                'sg2': 'deriv',
                'snv_sg1': 'snv_deriv',
                'snv_sg2': 'snv_deriv',
                'deriv_snv': 'deriv_snv'
            }

            deriv_map = {
                'raw': 0,
                'snv': 0,
                'sg1': 1,
                'sg2': 2,
                'snv_sg1': 1,
                'snv_sg2': 2,
                'deriv_snv': 1
            }

            polyorder_map = {
                'raw': 2,
                'snv': 2,
                'sg1': 2,
                'sg2': 3,
                'snv_sg1': 2,
                'snv_sg2': 3,
                'deriv_snv': 2
            }

            preprocess_name = preprocess_name_map.get(preprocess, 'raw')

            # Use actual derivative order from loaded config if available (fixes deriv_snv mismatch)
            # Otherwise fall back to hardcoded map for custom models
            if self.selected_model_config is not None:
                config_deriv = self.selected_model_config.get('Deriv', None)
                if config_deriv is not None and not pd.isna(config_deriv):
                    deriv = int(config_deriv)
                    # Determine polyorder based on actual derivative order
                    if deriv == 0:
                        polyorder = 2
                    elif deriv == 1:
                        polyorder = 2
                    elif deriv == 2:
                        polyorder = 3
                    else:
                        polyorder = 2  # Fallback
                    print(f"DEBUG: Using deriv={deriv}, polyorder={polyorder} from loaded config")
                else:
                    # No valid deriv in config, use map
                    deriv = deriv_map.get(preprocess, 0)
                    polyorder = polyorder_map.get(preprocess, 2)
            else:
                # No config loaded, use map (custom model creation)
                deriv = deriv_map.get(preprocess, 0)
                polyorder = polyorder_map.get(preprocess, 2)

            # CRITICAL FIX: Detect if we have derivative preprocessing + variable subset
            # This matches the behavior in search.py (lines 434-449)
            is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']
            base_full_vars = len(X_base_df.columns)
            if self.selected_model_config is not None:
                cfg_full_vars = self.selected_model_config.get('full_vars')
                if cfg_full_vars is not None and not pd.isna(cfg_full_vars):
                    try:
                        base_full_vars = int(cfg_full_vars)
                    except (TypeError, ValueError):
                        pass
            is_subset = len(selected_wl) < base_full_vars
            use_full_spectrum_preprocessing = is_derivative and is_subset

            if use_full_spectrum_preprocessing:
                print("DEBUG: Derivative + subset detected. Using full-spectrum preprocessing (matching search.py).")
                print(f"DEBUG: This fixes the R² discrepancy for non-contiguous wavelength selections.")

            # Get user-selected task type
            task_type = self.refine_task_type.get()

            # Get user-selected model
            model_name = self.refine_model_type.get()

            # Extract hyperparameters from loaded model config (if available)
            # This ensures we reproduce the exact same model that was selected from results
            n_components = 10  # Default fallback
            if self.selected_model_config is not None and 'LVs' in self.selected_model_config:
                lvs_value = self.selected_model_config.get('LVs')
                if not pd.isna(lvs_value):
                    n_components = int(lvs_value)
                    print(f"DEBUG: Using n_components={n_components} from loaded model config")

            model = get_model(
                model_name,
                task_type=task_type,
                n_components=n_components,  # Use exact n_components from original model
                max_n_components=self.max_n_components.get(),
                max_iter=self.refine_max_iter.get()
            )

            # Reapply tuned hyperparameters from the search results when available
            params_from_search = {}
            if self.selected_model_config is not None:
                raw_params = self.selected_model_config.get('Params')
                if isinstance(raw_params, str) and raw_params.strip():
                    try:
                        parsed = ast.literal_eval(raw_params)
                        if isinstance(parsed, dict):
                            params_from_search = parsed
                    except (ValueError, SyntaxError) as parse_err:
                        print(f"WARNING: Could not parse saved Params '{raw_params}': {parse_err}")

            if params_from_search:
                try:
                    model.set_params(**params_from_search)
                    print(f"DEBUG: Applied saved search parameters: {params_from_search}")
                except Exception as e:
                    print(f"WARNING: Failed to apply saved parameters {params_from_search}: {e}")

            # Build preprocessing pipeline and prepare data
            from spectral_predict.preprocess import build_preprocessing_pipeline
            from sklearn.pipeline import Pipeline

            # Prepare cross-validation
            y_array = y_series.values
            if task_type == "regression":
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            if use_full_spectrum_preprocessing:
                # === PATH A: Derivative + Subset (matches search.py lines 434-449) ===
                # 1. Build preprocessing pipeline WITHOUT model
                prep_steps = build_preprocessing_pipeline(
                    preprocess_name,
                    deriv,
                    window,
                    polyorder
                )
                prep_pipeline = Pipeline(prep_steps)

                # 2. Preprocess FULL spectrum (all wavelengths)
                X_full = X_base_df.values
                print(f"DEBUG: Preprocessing full spectrum ({X_full.shape[1]} wavelengths)...")
                X_full_preprocessed = prep_pipeline.fit_transform(X_full)

                # 3. Find indices of selected wavelengths in original data
                all_wavelengths = X_base_df.columns.astype(float).values
                wavelength_indices = []
                for wl in selected_wl:
                    idx = np.where(np.abs(all_wavelengths - wl) < 0.01)[0]
                    if len(idx) > 0:
                        wavelength_indices.append(idx[0])

                # 4. Subset the PREPROCESSED data (not raw!)
                X_work = X_full_preprocessed[:, wavelength_indices]
                print(f"DEBUG: Subsetted to {X_work.shape[1]} wavelengths after preprocessing.")
                print(f"DEBUG: This preserves derivative context from full spectrum.")

                # 5. Build pipeline with ONLY the model (skip preprocessing - already done!)
                pipe_steps = [('model', model)]
                pipe = Pipeline(pipe_steps)

                print(f"DEBUG: Pipeline steps: {[name for name, _ in pipe_steps]} (preprocessing already applied)")

            else:
                # === PATH B: Raw/SNV or Full-Spectrum (existing behavior) ===
                # Subset raw data first, then preprocess inside CV
                X_work = X_base_df[selected_cols].values

                # Build full pipeline with preprocessing + model
                pipe_steps = build_preprocessing_pipeline(
                    preprocess_name,
                    deriv,
                    window,
                    polyorder
                )
                pipe_steps.append(('model', model))
                pipe = Pipeline(pipe_steps)

                print(f"DEBUG: Pipeline steps: {[name for name, _ in pipe_steps]} (preprocessing inside CV)")

            # Collect metrics for each fold
            fold_metrics = []
            X_raw = X_work  # For derivative+subset, this is preprocessed; for others, it's raw

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_raw, y_array)):
                # Clone ENTIRE PIPELINE for this fold (not just model)
                pipe_fold = clone(pipe)

                # Split data
                X_train, X_test = X_raw[train_idx], X_raw[test_idx]
                y_train, y_test = y_array[train_idx], y_array[test_idx]

                # Fit pipeline (preprocessing + model) and predict
                pipe_fold.fit(X_train, y_train)
                y_pred = pipe_fold.predict(X_test)

                if task_type == "regression":
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    fold_metrics.append({"rmse": rmse, "r2": r2, "mae": mae})
                else:
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    fold_metrics.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

            # Compute mean and std across folds
            results = {}
            if task_type == "regression":
                results['rmse_mean'] = np.mean([m['rmse'] for m in fold_metrics])
                results['rmse_std'] = np.std([m['rmse'] for m in fold_metrics])
                results['r2_mean'] = np.mean([m['r2'] for m in fold_metrics])
                results['r2_std'] = np.std([m['r2'] for m in fold_metrics])
                results['mae_mean'] = np.mean([m['mae'] for m in fold_metrics])
                results['mae_std'] = np.std([m['mae'] for m in fold_metrics])
            else:
                results['accuracy_mean'] = np.mean([m['accuracy'] for m in fold_metrics])
                results['accuracy_std'] = np.std([m['accuracy'] for m in fold_metrics])
                results['precision_mean'] = np.mean([m['precision'] for m in fold_metrics])
                results['precision_std'] = np.std([m['precision'] for m in fold_metrics])
                results['recall_mean'] = np.mean([m['recall'] for m in fold_metrics])
                results['recall_std'] = np.std([m['recall'] for m in fold_metrics])
                results['f1_mean'] = np.mean([m['f1'] for m in fold_metrics])
                results['f1_std'] = np.std([m['f1'] for m in fold_metrics])

            # Format results with detailed diagnostics
            if task_type == "regression":
                # Add comparison to loaded model if available
                loaded_r2 = "N/A"
                r2_diff = "N/A"
                if self.selected_model_config is not None and 'R2' in self.selected_model_config:
                    loaded_r2_value = self.selected_model_config.get('R2')
                    if not pd.isna(loaded_r2_value):
                        loaded_r2 = f"{loaded_r2_value:.4f}"
                        r2_diff_value = results['r2_mean'] - loaded_r2_value
                        r2_diff = f"{r2_diff_value:+.4f}"

                results_text = f"""Refined Model Results:

Cross-Validation Performance ({self.refine_folds.get()} folds):
  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}
  R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}
  MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}

COMPARISON TO LOADED MODEL:
  Original R² (from Results tab): {loaded_r2}
  Refined R² (just computed):     {results['r2_mean']:.4f}
  Difference:                     {r2_diff}

Configuration:
  Model: {model_name}
  Task Type: {task_type}
  Preprocessing: {preprocess}
  Window Size: {window}
  Wavelengths: {wl_summary}
  Features: {len(selected_wl)}
  Samples: {X_raw.shape[0]}
  CV Folds: {self.refine_folds.get()}
  n_components: {n_components}

DEBUG INFO:
  Loaded LVs from config: {self.selected_model_config.get('LVs', 'N/A') if self.selected_model_config else 'N/A'}
  Loaded n_vars from config: {self.selected_model_config.get('n_vars', 'N/A') if self.selected_model_config else 'N/A'}
  Loaded Preprocessing: {self.selected_model_config.get('Preprocess', 'N/A') if self.selected_model_config else 'N/A'}
  Loaded Deriv: {self.selected_model_config.get('Deriv', 'N/A') if self.selected_model_config else 'N/A'}
  Loaded Window: {self.selected_model_config.get('Window', 'N/A') if self.selected_model_config else 'N/A'}
  Processing Path: {'Full-spectrum preprocessing (derivative+subset fix)' if use_full_spectrum_preprocessing else 'Standard (subset then preprocess)'}

NOTE: {'Derivative + subset detected! Using full-spectrum preprocessing to match search.py behavior and preserve derivative context.' if use_full_spectrum_preprocessing else ''}
"""
            else:
                results_text = f"""Refined Model Results:

Cross-Validation Performance ({self.refine_folds.get()} folds):
  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}
  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}
  Recall: {results['recall_mean']:.4f} ± {results['recall_std']:.4f}
  F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}

Configuration:
  Model: {model_name}
  Task Type: {task_type}
  Preprocessing: {preprocess}
  Window Size: {window}
  Wavelengths: {wl_summary}
  Features: {len(selected_wl)}
  Samples: {X_raw.shape[0]}
  CV Folds: {self.refine_folds.get()}
"""

            # Fit final pipeline on full dataset for model persistence
            # Clone the pipeline and fit on all data
            final_pipe = clone(pipe)
            final_pipe.fit(X_raw, y_array)

            # Extract model and preprocessor from pipeline for saving
            final_model = final_pipe.named_steps['model']

            # Build preprocessor from pipeline steps (excluding the model)
            if use_full_spectrum_preprocessing:
                # For derivative + subset: preprocessor was already fitted on full spectrum
                # We need to save that preprocessor, not create a new one
                final_preprocessor = prep_pipeline  # Already fitted
                print("DEBUG: Using full-spectrum preprocessor (already fitted)")
            elif len(pipe_steps) > 1:  # Has preprocessing steps
                final_preprocessor = Pipeline(pipe_steps[:-1])  # All steps except model
                final_preprocessor.fit(X_raw)  # Fit on raw data
                print("DEBUG: Fitting preprocessor on subset data")
            else:
                final_preprocessor = None
                print("DEBUG: No preprocessor (raw data)")

            # Store the fitted model and metadata for later saving
            self.refined_model = final_model
            self.refined_preprocessor = final_preprocessor
            self.refined_wavelengths = list(selected_wl)
            self.refined_performance = results
            self.refined_config = {
                'model_name': model_name,
                'task_type': task_type,
                'preprocessing': preprocess,
                'window': window,
                'n_vars': len(selected_wl),
                'n_samples': X_raw.shape[0],
                'cv_folds': n_folds,
                'use_full_spectrum_preprocessing': use_full_spectrum_preprocessing
            }

            # Store full wavelengths for derivative + subset case
            if use_full_spectrum_preprocessing:
                self.refined_full_wavelengths = list(all_wavelengths)
            else:
                self.refined_full_wavelengths = None

            # Update UI
            self.root.after(0, lambda: self._update_refined_results(results_text))

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            error_text = f"Error running refined model:\n\n{e}\n\n{error_msg}"
            self.root.after(0, lambda: self._update_refined_results(error_text, is_error=True))

    def _update_refined_results(self, results_text, is_error=False):
        """Update the refined results display."""
        self.refine_results_text.config(state='normal')
        self.refine_results_text.delete('1.0', tk.END)
        self.refine_results_text.insert('1.0', results_text)
        self.refine_results_text.config(state='disabled')

        # Re-enable buttons
        self.refine_run_button.config(state='normal')

        if is_error:
            self.refine_status.config(text="✗ Error running refined model")
            self.refine_save_button.config(state='disabled')
            messagebox.showerror("Error", "Failed to run refined model. See results area for details.")
        else:
            self.refine_status.config(text="✓ Refined model complete")
            # Enable Save Model button after successful run
            self.refine_save_button.config(state='normal')
            messagebox.showinfo("Success", "Refined model analysis complete!")

    def _save_refined_model(self):
        """Save the current refined model to a .dasp file."""
        # Check if model has been trained
        if self.refined_model is None:
            messagebox.showerror(
                "No Model Trained",
                "Please run a refined model first before saving.\n\n"
                "Click 'Run Refined Model' to train a model, then you can save it."
            )
            return

        try:
            from spectral_predict.model_io import save_model
            from datetime import datetime

            # Ask for save location
            default_name = f"model_{self.refined_config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dasp"

            filepath = filedialog.asksaveasfilename(
                defaultextension=".dasp",
                filetypes=[("DASP Model", "*.dasp"), ("All files", "*.*")],
                initialfile=default_name,
                title="Save Trained Model"
            )

            if not filepath:
                return  # User cancelled

            # Build comprehensive metadata
            metadata = {
                'model_name': self.refined_config['model_name'],
                'task_type': self.refined_config['task_type'],
                'preprocessing': self.refined_config['preprocessing'],
                'window': self.refined_config['window'],
                'wavelengths': self.refined_wavelengths,
                'n_vars': self.refined_config['n_vars'],
                'n_samples': self.refined_config['n_samples'],
                'cv_folds': self.refined_config['cv_folds'],
                'performance': {},
                'use_full_spectrum_preprocessing': self.refined_config.get('use_full_spectrum_preprocessing', False),
                'full_wavelengths': self.refined_full_wavelengths  # All wavelengths for derivative+subset
            }

            # Add performance metrics based on task type
            if self.refined_config['task_type'] == 'regression':
                metadata['performance'] = {
                    'RMSE': self.refined_performance.get('rmse_mean'),
                    'RMSE_std': self.refined_performance.get('rmse_std'),
                    'R2': self.refined_performance.get('r2_mean'),
                    'R2_std': self.refined_performance.get('r2_std'),
                    'MAE': self.refined_performance.get('mae_mean'),
                    'MAE_std': self.refined_performance.get('mae_std')
                }
            else:  # classification
                metadata['performance'] = {
                    'Accuracy': self.refined_performance.get('accuracy_mean'),
                    'Accuracy_std': self.refined_performance.get('accuracy_std'),
                    'Precision': self.refined_performance.get('precision_mean'),
                    'Precision_std': self.refined_performance.get('precision_std'),
                    'Recall': self.refined_performance.get('recall_mean'),
                    'Recall_std': self.refined_performance.get('recall_std'),
                    'F1': self.refined_performance.get('f1_mean'),
                    'F1_std': self.refined_performance.get('f1_std')
                }

            # Save the model
            save_model(
                model=self.refined_model,
                preprocessor=self.refined_preprocessor,
                metadata=metadata,
                filepath=filepath
            )

            # Show success message
            messagebox.showinfo(
                "Model Saved",
                f"Model successfully saved to:\n\n{filepath}\n\n"
                f"You can now load this model in the Model Prediction tab "
                f"to make predictions on new data."
            )

            # Update status
            self.refine_status.config(text=f"✓ Model saved to {Path(filepath).name}")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            messagebox.showerror(
                "Save Error",
                f"Failed to save model:\n\n{str(e)}\n\nSee console for details."
            )
            print(f"Error saving model:\n{error_msg}")

    def _format_wavelengths_as_spec(self, wavelengths):
        """
        Format a list of wavelengths into a compact specification string.
        Groups consecutive wavelengths into ranges.

        Example: [1500, 1501, 1502, 1505, 1506, 1510]
                 -> "1500.0-1502.0, 1505.0-1506.0, 1510.0"
        """
        # Enhanced validation
        if wavelengths is None:
            print("WARNING: _format_wavelengths_as_spec received None")
            return ""

        # Convert numpy array to list if needed
        if hasattr(wavelengths, 'tolist'):
            wavelengths = wavelengths.tolist()

        # Ensure it's a list
        if not isinstance(wavelengths, list):
            try:
                wavelengths = list(wavelengths)
            except Exception as e:
                print(f"ERROR: Cannot convert wavelengths to list: {e}")
                return ""

        if len(wavelengths) == 0:
            print("WARNING: _format_wavelengths_as_spec received empty list")
            return ""

        wavelengths = sorted(list(set(wavelengths)))  # Remove duplicates and sort

        # Group consecutive wavelengths (within 1.5 nm)
        ranges = []
        start = wavelengths[0]
        end = wavelengths[0]

        for i in range(1, len(wavelengths)):
            if wavelengths[i] - end <= 1.5:  # Consecutive
                end = wavelengths[i]
            else:
                # Save the range
                if abs(end - start) < 0.1:  # Single wavelength
                    ranges.append(f"{start:.1f}")
                else:  # Range
                    ranges.append(f"{start:.1f}-{end:.1f}")
                start = wavelengths[i]
                end = wavelengths[i]

        # Don't forget the last range
        if abs(end - start) < 0.1:
            ranges.append(f"{start:.1f}")
        else:
            ranges.append(f"{start:.1f}-{end:.1f}")

        return ", ".join(ranges)

    def _parse_wavelength_spec(self, spec_text, available_wavelengths):
        """
        Parse wavelength specification string into list of wavelengths.

        Format: "1920, 1930-1940, 1950, 1960-2000"
        - Individual wavelengths: 1920, 1950
        - Ranges: 1930-1940, 1960-2000
        - Comments: Lines starting with # are ignored

        Returns list of wavelengths that exist in available_wavelengths.
        """
        selected = []
        spec_text = spec_text.strip()

        if not spec_text:
            return list(available_wavelengths)  # Return all if empty

        # Remove comment lines (lines starting with #)
        lines = spec_text.split('\n')
        clean_lines = [line for line in lines if not line.strip().startswith('#')]
        spec_text = ' '.join(clean_lines)

        # Split by commas
        parts = [p.strip() for p in spec_text.split(',')]

        for part in parts:
            if '-' in part and not part.startswith('-'):
                # Range specification
                try:
                    start, end = part.split('-')
                    start_wl = float(start.strip())
                    end_wl = float(end.strip())

                    # Find wavelengths in this range
                    for wl in available_wavelengths:
                        if start_wl <= wl <= end_wl:
                            selected.append(wl)
                except ValueError:
                    # Invalid range, skip
                    continue
            else:
                # Individual wavelength
                try:
                    wl = float(part.strip())
                    # Find closest wavelength in available
                    if wl in available_wavelengths:
                        selected.append(wl)
                    else:
                        # Find closest match
                        closest = min(available_wavelengths, key=lambda x: abs(x - wl))
                        if abs(closest - wl) < 5:  # Within 5 nm tolerance
                            selected.append(closest)
                except ValueError:
                    # Invalid wavelength, skip
                    continue

        # Remove duplicates and sort
        selected = sorted(list(set(selected)))
        return selected

    def _preview_wavelength_selection(self):
        """Preview the wavelength selection with a plot."""
        if self.X_original is None:
            messagebox.showwarning("No Data", "Please load data first to preview wavelength selection.")
            return

        # Get available wavelengths
        available_wl = self.X_original.columns.astype(float).values

        # Parse wavelength specification
        spec_text = self.refine_wl_spec.get('1.0', 'end')
        try:
            selected_wl = self._parse_wavelength_spec(spec_text, available_wl)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Error parsing wavelength specification:\n{e}")
            return

        if not selected_wl:
            messagebox.showwarning("No Wavelengths", "No valid wavelengths found in specification.")
            return

        # Create preview window
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Wavelength Selection Preview")
        preview_window.geometry("800x500")

        # Info text
        info_text = f"Selected {len(selected_wl)} wavelengths out of {len(available_wl)} available"
        ttk.Label(preview_window, text=info_text, font=('Arial', 12, 'bold')).pack(pady=10)

        # Create plot
        if HAS_MATPLOTLIB:
            fig = Figure(figsize=(10, 4))
            ax = fig.add_subplot(111)

            # Create binary indicator (1 = selected, 0 = not selected)
            selected_set = set(selected_wl)
            indicators = [1 if wl in selected_set else 0 for wl in available_wl]

            # Plot
            ax.fill_between(available_wl, 0, indicators, alpha=0.3, color='blue', label='Selected')
            ax.plot(available_wl, indicators, 'b-', linewidth=0.5)
            ax.set_xlabel('Wavelength (nm)', fontsize=10)
            ax.set_ylabel('Selected', fontsize=10)
            ax.set_title(f'Wavelength Selection: {len(selected_wl)}/{len(available_wl)} wavelengths', fontsize=12)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Embed plot
            canvas = FigureCanvasTkAgg(fig, master=preview_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # Show wavelength list in text box
        list_frame = ttk.LabelFrame(preview_window, text="Selected Wavelengths", padding="10")
        list_frame.pack(fill='both', expand=True, padx=10, pady=10)

        wl_text = tk.Text(list_frame, height=5, font=('Consolas', 9), wrap=tk.WORD)
        wl_text.pack(fill='both', expand=True)

        # Format wavelengths nicely
        wl_str = ', '.join([f"{wl:.1f}" for wl in selected_wl[:50]])  # Show first 50
        if len(selected_wl) > 50:
            wl_str += f", ... ({len(selected_wl) - 50} more)"
        wl_text.insert('1.0', wl_str)
        wl_text.config(state='disabled')

        # Close button
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)


    def _update_wavelength_count(self, event=None):
        """Update wavelength count display in real-time."""
        try:
            if self.X_original is None:
                self.refine_wl_count_label.config(text="Wavelengths: Data not loaded")
                return

            wl_spec_text = self.refine_wl_spec.get('1.0', 'end')
            available_wl = self.X_original.columns.astype(float).values
            selected_wl = self._parse_wavelength_spec(wl_spec_text, available_wl)

            count = len(selected_wl)
            if count > 0:
                range_text = f"({selected_wl[0]:.1f} - {selected_wl[-1]:.1f} nm)"
                self.refine_wl_count_label.config(text=f"Wavelengths: {count} selected {range_text}")
            else:
                self.refine_wl_count_label.config(text="Wavelengths: 0 selected (check specification)")
        except Exception as e:
            self.refine_wl_count_label.config(text="Wavelengths: Invalid specification")
            print(f"DEBUG: Error updating wavelength count: {e}")

    def _apply_wl_preset(self, preset_type):
        """Apply wavelength preset."""
        if self.X_original is None:
            messagebox.showinfo("No Data", "Please load data first")
            return

        wavelengths = self.X_original.columns.astype(float).values

        if preset_type == 'all':
            selected = wavelengths
        elif preset_type == 'nir':
            selected = [wl for wl in wavelengths if wl >= 780]
        elif preset_type == 'visible':
            selected = [wl for wl in wavelengths if 400 <= wl <= 780]
        else:
            return

        if len(selected) == 0:
            messagebox.showinfo("No Wavelengths", f"No wavelengths found for {preset_type} preset in your data")
            return

        wl_spec = self._format_wavelengths_as_spec(list(selected))
        self.refine_wl_spec.delete('1.0', 'end')
        self.refine_wl_spec.insert('1.0', wl_spec)
        self._update_wavelength_count()

    def _custom_range_dialog(self):
        """Show dialog for custom wavelength range."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Custom Wavelength Range")
        dialog.geometry("350x180")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Start wavelength (nm):", padding=5).pack(pady=5)
        start_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=start_var, width=20).pack()

        ttk.Label(dialog, text="End wavelength (nm):", padding=5).pack(pady=5)
        end_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=end_var, width=20).pack()

        def apply_range():
            try:
                start = float(start_var.get())
                end = float(end_var.get())

                if self.X_original is None:
                    messagebox.showwarning("No Data", "Please load data first")
                    dialog.destroy()
                    return

                wavelengths = self.X_original.columns.astype(float).values
                selected = [wl for wl in wavelengths if start <= wl <= end]

                if len(selected) == 0:
                    messagebox.showwarning("No Wavelengths",
                                          f"No wavelengths found in range {start}-{end} nm")
                    return

                wl_spec = self._format_wavelengths_as_spec(selected)
                self.refine_wl_spec.delete('1.0', 'end')
                self.refine_wl_spec.insert('1.0', wl_spec)
                self._update_wavelength_count()
                dialog.destroy()

            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers")

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Apply", command=apply_range).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)

    # === Tab 7: Model Prediction Methods ===

    def _create_tab7_model_prediction(self):
        """Create Tab 7: Model Prediction - Load models and make predictions on new data."""
        self.tab7 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab7, text='  🔮 Model Prediction  ')

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
        ttk.Label(content_frame, text="Model Prediction", style='Title.TLabel').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        row += 1

        # Instructions
        ttk.Label(content_frame,
            text="Load saved .dasp model files and apply them to new spectral data for predictions.",
            style='Caption.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        row += 1

        # === Step 1: Load Models ===
        step1_frame = ttk.LabelFrame(content_frame, text="Step 1: Load Saved Models", padding="20")
        step1_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=20, pady=10)
        row += 1

        # Button frame for load and clear
        button_frame1 = ttk.Frame(step1_frame)
        button_frame1.grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Button(button_frame1, text="📂 Load Model File(s)",
                   command=self._load_model_for_prediction).pack(side='left', padx=5)
        ttk.Button(button_frame1, text="🗑️ Clear All Models",
                   command=self._clear_loaded_models).pack(side='left', padx=5)

        # Loaded models display
        ttk.Label(step1_frame, text="Loaded Models:", style='Subheading.TLabel').grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))

        models_text_frame = ttk.Frame(step1_frame)
        models_text_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.loaded_models_text = tk.Text(models_text_frame, height=8, width=90,
                                          font=('Consolas', 9),
                                          bg='#FAFAFA', fg=self.colors['text'],
                                          wrap=tk.WORD, state='disabled')
        self.loaded_models_text.pack(side='left', fill='both', expand=True)

        models_scrollbar = ttk.Scrollbar(models_text_frame, orient='vertical',
                                        command=self.loaded_models_text.yview)
        models_scrollbar.pack(side='right', fill='y')
        self.loaded_models_text.config(yscrollcommand=models_scrollbar.set)

        # === Step 2: Load Data ===
        step2_frame = ttk.LabelFrame(content_frame, text="Step 2: Load Data for Prediction", padding="20")
        step2_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=20, pady=10)
        row += 1

        # Data source selection
        ttk.Label(step2_frame, text="Data Source:", style='Subheading.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=5)

        self.pred_data_source = tk.StringVar(value='directory')
        source_frame = ttk.Frame(step2_frame)
        source_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Radiobutton(source_frame, text="Directory (ASD/SPC)",
                       variable=self.pred_data_source, value='directory').pack(side='left', padx=5)
        ttk.Radiobutton(source_frame, text="CSV File",
                       variable=self.pred_data_source, value='csv').pack(side='left', padx=5)

        # File path entry
        ttk.Label(step2_frame, text="Path:", style='Caption.TLabel').grid(
            row=2, column=0, sticky=tk.W, pady=(10, 5))

        path_frame = ttk.Frame(step2_frame)
        path_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.pred_data_path = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.pred_data_path, width=60).pack(side='left', fill='x', expand=True)
        ttk.Button(path_frame, text="Browse...",
                   command=self._browse_prediction_data).pack(side='left', padx=5)

        # Load button and status
        button_frame2 = ttk.Frame(step2_frame)
        button_frame2.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame2, text="📊 Load Data",
                   command=self._load_prediction_data).pack(side='left', padx=5)

        self.pred_data_status = ttk.Label(step2_frame, text="No data loaded", style='Caption.TLabel')
        self.pred_data_status.grid(row=5, column=0, columnspan=2, pady=5)

        # === Step 3: Run Predictions ===
        step3_frame = ttk.LabelFrame(content_frame, text="Step 3: Make Predictions", padding="20")
        step3_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=20, pady=10)
        row += 1

        button_frame3 = ttk.Frame(step3_frame)
        button_frame3.grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Button(button_frame3, text="🚀 Run All Models",
                   command=self._run_predictions, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame3, text="📥 Export to CSV",
                   command=self._export_predictions).pack(side='left', padx=5)

        # Progress bar
        self.pred_progress = ttk.Progressbar(step3_frame, mode='determinate', length=400)
        self.pred_progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Status label
        self.pred_status = ttk.Label(step3_frame, text="Ready", style='Caption.TLabel')
        self.pred_status.grid(row=2, column=0, columnspan=2)

        # === Step 4: View Results ===
        step4_frame = ttk.LabelFrame(content_frame, text="Step 4: View Predictions", padding="20")
        step4_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=10)
        content_frame.grid_rowconfigure(row, weight=1)
        row += 1

        # Predictions table
        ttk.Label(step4_frame, text="Prediction Results:", style='Subheading.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))

        tree_frame = ttk.Frame(step4_frame)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        step4_frame.grid_rowconfigure(1, weight=1)
        step4_frame.grid_columnconfigure(0, weight=1)

        self.predictions_tree = ttk.Treeview(tree_frame, height=12, show='headings')
        self.predictions_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbars for treeview
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.predictions_tree.yview)
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.predictions_tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.predictions_tree.xview)
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.predictions_tree.configure(xscrollcommand=hsb.set)

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Statistics display
        ttk.Label(step4_frame, text="Statistics:", style='Subheading.TLabel').grid(
            row=2, column=0, sticky=tk.W, pady=(15, 5))

        stats_text_frame = ttk.Frame(step4_frame)
        stats_text_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        self.pred_stats_text = tk.Text(stats_text_frame, height=10, width=90,
                                       font=('Consolas', 9),
                                       bg='#FAFAFA', fg=self.colors['text'],
                                       wrap=tk.WORD, state='disabled')
        self.pred_stats_text.pack(side='left', fill='both', expand=True)

        stats_scrollbar = ttk.Scrollbar(stats_text_frame, orient='vertical',
                                       command=self.pred_stats_text.yview)
        stats_scrollbar.pack(side='right', fill='y')
        self.pred_stats_text.config(yscrollcommand=stats_scrollbar.set)

    def _load_model_for_prediction(self):
        """Browse and load one or more .dasp model files."""
        filepaths = filedialog.askopenfilenames(
            title="Select DASP Model File(s)",
            filetypes=[("DASP Model", "*.dasp"), ("All files", "*.*")]
        )

        if not filepaths:
            return

        try:
            from spectral_predict.model_io import load_model

            # Load each model
            loaded_count = 0
            failed_models = []

            for filepath in filepaths:
                try:
                    # Load the model
                    model_dict = load_model(filepath)

                    # Add file information
                    model_dict['filepath'] = filepath
                    model_dict['filename'] = Path(filepath).name

                    # Add to loaded models list
                    self.loaded_models.append(model_dict)
                    loaded_count += 1

                except Exception as e:
                    failed_models.append((Path(filepath).name, str(e)))

            # Update display
            self._update_loaded_models_display()

            # Show appropriate message
            if loaded_count > 0 and not failed_models:
                messagebox.showinfo("Success",
                    f"Successfully loaded {loaded_count} model(s).")
            elif loaded_count > 0 and failed_models:
                error_msg = f"Successfully loaded {loaded_count} model(s), but {len(failed_models)} failed:\n\n"
                for name, error in failed_models[:3]:  # Show first 3 failures
                    error_msg += f"- {name}: {error}\n"
                if len(failed_models) > 3:
                    error_msg += f"... and {len(failed_models) - 3} more"
                messagebox.showwarning("Partial Success", error_msg)
            else:
                error_msg = f"Failed to load all {len(failed_models)} model(s):\n\n"
                for name, error in failed_models[:3]:
                    error_msg += f"- {name}: {error}\n"
                if len(failed_models) > 3:
                    error_msg += f"... and {len(failed_models) - 3} more"
                messagebox.showerror("Load Error", error_msg)

        except Exception as e:
            messagebox.showerror("Load Error",
                f"Unexpected error during model loading:\n{str(e)}")

    def _update_loaded_models_display(self):
        """Update the list of loaded models display."""
        self.loaded_models_text.config(state='normal')
        self.loaded_models_text.delete('1.0', 'end')

        if not self.loaded_models:
            self.loaded_models_text.insert('1.0', "No models loaded. Click 'Load Model File(s)' to add models.")
        else:
            for i, model_dict in enumerate(self.loaded_models, 1):
                metadata = model_dict['metadata']

                # Extract key information
                model_name = metadata.get('model_name', 'Unknown')
                preprocessing = metadata.get('preprocessing', 'Unknown')
                n_vars = metadata.get('n_vars', 'Unknown')

                # Performance metrics
                perf = metadata.get('performance', {})
                r2 = perf.get('R2', perf.get('R2_cv', 'N/A'))
                rmse = perf.get('RMSE', perf.get('RMSE_cv', 'N/A'))

                # Format R2 and RMSE
                if isinstance(r2, (int, float)):
                    r2_str = f"{r2:.4f}"
                else:
                    r2_str = str(r2)

                if isinstance(rmse, (int, float)):
                    rmse_str = f"{rmse:.4f}"
                else:
                    rmse_str = str(rmse)

                filename = model_dict.get('filename', 'Unknown')

                # Build display text
                text = f"[{i}] {filename}\n"
                text += f"    Model: {model_name}  |  Preprocessing: {preprocessing}\n"
                text += f"    R²: {r2_str}  |  RMSE: {rmse_str}  |  Variables: {n_vars}\n"
                text += f"    Path: {model_dict.get('filepath', 'Unknown')}\n"
                text += "\n"

                self.loaded_models_text.insert('end', text)

        self.loaded_models_text.config(state='disabled')

    def _clear_loaded_models(self):
        """Clear all loaded models."""
        if self.loaded_models:
            response = messagebox.askyesno("Confirm Clear",
                f"Clear all {len(self.loaded_models)} loaded model(s)?")
            if response:
                self.loaded_models = []
                self._update_loaded_models_display()
                messagebox.showinfo("Cleared", "All models have been cleared.")
        else:
            messagebox.showinfo("No Models", "No models are currently loaded.")

    def _browse_prediction_data(self):
        """Browse for spectral data directory or CSV file."""
        source = self.pred_data_source.get()

        if source == 'directory':
            path = filedialog.askdirectory(title="Select Spectral Data Directory")
        else:  # csv
            path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

        if path:
            self.pred_data_path.set(path)

    def _load_prediction_data(self):
        """Load spectral data for predictions."""
        path_str = self.pred_data_path.get()

        if not path_str:
            messagebox.showerror("No Path", "Please select a data source first.")
            return

        path = Path(path_str)

        if not path.exists():
            messagebox.showerror("Path Error", f"Path does not exist:\n{path_str}")
            return

        source = self.pred_data_source.get()

        try:
            from spectral_predict.io import read_asd_dir, read_spc_dir, read_csv_spectra

            if source == 'directory':
                # Try to detect file type
                asd_files = list(path.glob("*.asd"))
                spc_files = list(path.glob("*.spc"))

                if asd_files:
                    self.pred_status.config(text="Loading ASD files...")
                    self.root.update()
                    self.prediction_data = read_asd_dir(str(path))
                elif spc_files:
                    self.pred_status.config(text="Loading SPC files...")
                    self.root.update()
                    self.prediction_data = read_spc_dir(str(path))
                else:
                    messagebox.showerror("No Files",
                        "No ASD or SPC files found in the selected directory.")
                    return
            else:  # csv
                self.pred_status.config(text="Loading CSV file...")
                self.root.update()
                self.prediction_data = read_csv_spectra(str(path))

            # Update status
            n_samples = len(self.prediction_data)
            n_wavelengths = len(self.prediction_data.columns)

            self.pred_data_status.config(
                text=f"✓ Loaded {n_samples} spectra with {n_wavelengths} wavelengths"
            )

            messagebox.showinfo("Success",
                f"Data loaded successfully:\n{n_samples} spectra\n{n_wavelengths} wavelengths")

        except Exception as e:
            messagebox.showerror("Load Error",
                f"Failed to load data:\n{str(e)}")
            self.pred_data_status.config(text="Load failed")

    def _run_predictions(self):
        """Apply all loaded models to prediction data."""
        # Validate inputs
        if not self.loaded_models:
            messagebox.showerror("No Models",
                "Please load at least one model first.")
            return

        if self.prediction_data is None:
            messagebox.showerror("No Data",
                "Please load prediction data first.")
            return

        try:
            from spectral_predict.model_io import predict_with_model

            # Initialize results dataframe
            results = pd.DataFrame()
            results['Sample'] = self.prediction_data.index

            # Setup progress bar
            self.pred_progress['maximum'] = len(self.loaded_models)
            self.pred_progress['value'] = 0

            # Apply each model
            successful_models = 0
            for i, model_dict in enumerate(self.loaded_models):
                metadata = model_dict['metadata']
                model_name = metadata.get('model_name', 'Unknown')
                preprocessing = metadata.get('preprocessing', 'raw')
                filename = model_dict.get('filename', f'Model_{i+1}')

                # Update status
                self.pred_status.config(text=f"Running {filename}...")
                self.root.update()

                try:
                    # Make predictions
                    predictions = predict_with_model(
                        model_dict,
                        self.prediction_data,
                        validate_wavelengths=True
                    )

                    # Store predictions with descriptive column name
                    col_name = f"{model_name}_{preprocessing}"

                    # Handle duplicate column names
                    counter = 1
                    original_col_name = col_name
                    while col_name in results.columns:
                        col_name = f"{original_col_name}_{counter}"
                        counter += 1

                    results[col_name] = predictions
                    successful_models += 1

                except Exception as e:
                    error_msg = f"Model '{filename}' failed:\n{str(e)}"
                    print(error_msg)  # Log to console
                    # Continue with other models

                # Update progress
                self.pred_progress['value'] = i + 1
                self.root.update()

            # Store results
            self.predictions_df = results

            # Display results
            self._display_predictions()

            # Update status
            if successful_models == len(self.loaded_models):
                self.pred_status.config(text=f"✓ Complete! {successful_models} models applied successfully.")
            else:
                failed = len(self.loaded_models) - successful_models
                self.pred_status.config(
                    text=f"⚠ Complete with warnings: {successful_models} succeeded, {failed} failed."
                )

            messagebox.showinfo("Predictions Complete",
                f"Successfully applied {successful_models} of {len(self.loaded_models)} models.\n"
                f"Results are displayed below.")

        except Exception as e:
            messagebox.showerror("Prediction Error",
                f"An error occurred during predictions:\n{str(e)}")
            self.pred_status.config(text="Error occurred")

    def _display_predictions(self):
        """Display predictions in treeview table."""
        # Clear existing items
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)

        if self.predictions_df is None or self.predictions_df.empty:
            return

        # Set columns
        columns = list(self.predictions_df.columns)
        self.predictions_tree['columns'] = columns

        # Configure column headings and widths
        for col in columns:
            self.predictions_tree.heading(col, text=col)
            if col == 'Sample':
                self.predictions_tree.column(col, width=150, anchor='w')
            else:
                self.predictions_tree.column(col, width=120, anchor='e')

        # Populate rows
        for idx, row in self.predictions_df.iterrows():
            values = []
            for col in columns:
                val = row[col]
                # Format numeric values
                if isinstance(val, (int, float)) and col != 'Sample':
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))

            self.predictions_tree.insert('', 'end', values=values)

        # Calculate and display statistics
        self._update_prediction_statistics()

    def _update_prediction_statistics(self):
        """Calculate and display prediction statistics."""
        if self.predictions_df is None or self.predictions_df.empty:
            return

        self.pred_stats_text.config(state='normal')
        self.pred_stats_text.delete('1.0', 'end')

        stats_text = "Prediction Statistics:\n"
        stats_text += "=" * 60 + "\n\n"

        # Calculate stats for each prediction column (skip 'Sample' column)
        prediction_cols = [col for col in self.predictions_df.columns if col != 'Sample']

        if not prediction_cols:
            stats_text += "No prediction columns found.\n"
        else:
            for col in prediction_cols:
                values = self.predictions_df[col].dropna()

                if len(values) > 0:
                    stats_text += f"{col}:\n"
                    stats_text += f"  Count: {len(values)}\n"
                    stats_text += f"  Mean:  {values.mean():.4f}\n"
                    stats_text += f"  Std:   {values.std():.4f}\n"
                    stats_text += f"  Min:   {values.min():.4f}\n"
                    stats_text += f"  Max:   {values.max():.4f}\n"
                    stats_text += f"  Median:{values.median():.4f}\n"
                    stats_text += "\n"

        self.pred_stats_text.insert('1.0', stats_text)
        self.pred_stats_text.config(state='disabled')

    def _export_predictions(self):
        """Export predictions to CSV file."""
        if self.predictions_df is None or self.predictions_df.empty:
            messagebox.showerror("No Predictions",
                "No predictions to export. Run predictions first.")
            return

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"predictions_{timestamp}.csv"

        # Ask user for save location
        filepath = filedialog.asksaveasfilename(
            title="Export Predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if not filepath:
            return

        try:
            # Export to CSV
            self.predictions_df.to_csv(filepath, index=False)

            messagebox.showinfo("Export Successful",
                f"Predictions exported to:\n{filepath}\n\n"
                f"{len(self.predictions_df)} samples exported.")

        except Exception as e:
            messagebox.showerror("Export Error",
                f"Failed to export predictions:\n{str(e)}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SpectralPredictApp(root)

    # Add menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Quick Start", command=app._show_help)
    help_menu.add_separator()
    help_menu.add_command(label="Exit", command=root.quit)

    root.mainloop()


if __name__ == "__main__":
    main()
