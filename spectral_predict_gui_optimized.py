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
    from matplotlib.patches import Patch
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
        self.results_sort_column = None  # Current column being sorted
        self.results_sort_reverse = False  # Sort direction (False = ascending, True = descending)

        # Refined/saved model storage (for model persistence)
        self.refined_model = None  # Fitted model from Custom Model Development tab
        self.refined_preprocessor = None  # Fitted preprocessing pipeline
        self.refined_performance = None  # Performance metrics dict (R2, RMSE, etc.)
        self.refined_wavelengths = None  # List of wavelengths used in refined model (subset for derivative+subset)
        self.refined_full_wavelengths = None  # List of ALL wavelengths (for derivative+subset preprocessing)
        self.refined_config = None  # Configuration dict for refined model

        # Model Prediction Tab (Tab 8) variables
        self.loaded_models = []  # List of model dicts from load_model()
        self.prediction_data = None  # DataFrame with new spectral data
        self.predictions_df = None  # Results dataframe
        self.predictions_model_map = {}  # Map column names to model metadata

        # Tab 7 Model Development variables
        self.tab7_trained_model = None  # Trained model from Tab 7
        self.tab7_preprocessing_pipeline = None  # Fitted preprocessing pipeline
        self.tab7_wavelengths = None  # Wavelengths used (after preprocessing)
        self.tab7_full_wavelengths = None  # Full wavelengths (for derivative+subset)
        self.tab7_config = None  # Configuration dict
        self.tab7_performance = None  # Performance metrics dict
        self.tab7_y_true = None  # True values (for plotting)
        self.tab7_y_pred = None  # Predicted values (for plotting)
        self.tab7_loaded_config = None  # Config loaded from Results tab
        self.tab7_run_history = []  # History of model runs
        self.tab7_performance_history = []  # Performance tracking for comparison plot

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

        # Validation set tracking
        self.validation_enabled = tk.BooleanVar(value=False)
        self.validation_percentage = tk.DoubleVar(value=20.0)
        self.validation_algorithm = tk.StringVar(value="SPXY")
        self.validation_indices = set()  # Sample indices in validation set
        self.validation_X = None  # Stored validation spectral data
        self.validation_y = None  # Stored validation target data

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
        self.use_msc = tk.BooleanVar(value=False)  # MSC (Multiplicative Scatter Correction)
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
        self.custom_window_size = tk.StringVar(value="")  # Custom window size entry

        # Advanced model options (NeuralBoosted)
        self.n_estimators_50 = tk.BooleanVar(value=False)
        self.n_estimators_100 = tk.BooleanVar(value=True)  # Default
        self.custom_n_estimators = tk.StringVar(value="")  # Custom n_estimators entry
        self.lr_005 = tk.BooleanVar(value=False)
        self.lr_01 = tk.BooleanVar(value=True)  # Default
        self.lr_02 = tk.BooleanVar(value=True)  # Default
        self.custom_learning_rate = tk.StringVar(value="")  # Custom learning rate entry

        # Variable selection methods (multiple selection enabled)
        self.varsel_importance = tk.BooleanVar(value=True)  # Default enabled
        self.varsel_spa = tk.BooleanVar(value=False)
        self.varsel_uve = tk.BooleanVar(value=False)
        self.varsel_uve_spa = tk.BooleanVar(value=False)
        self.varsel_ipls = tk.BooleanVar(value=False)
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

        # Backend selection
        self.backend_choice = tk.StringVar(value="julia")  # Default to Julia
        self.julia_available = tk.BooleanVar(value=False)
        self.python_available = tk.BooleanVar(value=True)

        # Progress tracking
        self.progress_monitor = None
        self.analysis_thread = None
        self.analysis_start_time = None

        # Plotting
        self.plot_frames = {}
        self.plot_canvases = {}

        # Detect available backends
        self._detect_available_backends()

        self._create_ui()

    def _detect_available_backends(self):
        """Detect which backends are available."""
        # Check Python backend
        try:
            from spectral_predict.search import run_search
            self.python_available.set(True)
        except ImportError:
            self.python_available.set(False)

        # Check Julia backend
        try:
            from spectral_predict_julia_bridge import check_julia_installation
            status = check_julia_installation()
            self.julia_available.set(status.get('ready', False))
        except (ImportError, Exception):
            self.julia_available.set(False)

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
        self._create_tab7_model_development()  # NEW: Fresh Model Development
        self._create_tab8_model_prediction()  # Renamed from tab7

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

        # === Backend Selection ===
        ttk.Label(content_frame, text="Computation Backend", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
        row += 1

        backend_frame = ttk.Frame(content_frame)
        backend_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))

        ttk.Radiobutton(backend_frame, text="Julia (Fast - 2-5x speedup, Recommended)",
                        variable=self.backend_choice, value="julia").pack(anchor=tk.W)
        ttk.Radiobutton(backend_frame, text="Python (Compatible - all features work)",
                        variable=self.backend_choice, value="python").pack(anchor=tk.W)

        # Status indicator
        self.backend_status = ttk.Label(backend_frame, text="",
                                        style='Caption.TLabel')
        self.backend_status.pack(anchor=tk.W, pady=(5, 0))

        row += 1

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

        ttk.Checkbutton(preprocess_frame, text="MSC (Multiplicative Scatter Correction)", variable=self.use_msc).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Reference-based scatter correction", style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(preprocess_frame, text="✓ SG1 (1st derivative)", variable=self.use_sg1).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Removes baseline drift", style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(preprocess_frame, text="✓ SG2 (2nd derivative)", variable=self.use_sg2).grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Peak enhancement", style='Caption.TLabel').grid(row=4, column=1, sticky=tk.W, padx=15)

        # Advanced: deriv_snv option
        ttk.Checkbutton(preprocess_frame, text="deriv_snv (advanced)", variable=self.use_deriv_snv).grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Label(preprocess_frame, text="Derivative then SNV (less common)", style='Caption.TLabel').grid(row=5, column=1, sticky=tk.W, padx=15)

        # Derivative window size settings
        ttk.Label(preprocess_frame, text="Derivative Window Sizes:", style='Subheading.TLabel').grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        ttk.Label(preprocess_frame, text="Select one or more (default: 17 only)", style='Caption.TLabel').grid(row=7, column=0, columnspan=2, sticky=tk.W)

        window_frame = ttk.Frame(preprocess_frame)
        window_frame.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Checkbutton(window_frame, text="Window=7", variable=self.window_7).grid(row=0, column=0, padx=5, pady=2)
        ttk.Checkbutton(window_frame, text="Window=11", variable=self.window_11).grid(row=0, column=1, padx=5, pady=2)
        ttk.Checkbutton(window_frame, text="Window=17 ⭐", variable=self.window_17).grid(row=0, column=2, padx=5, pady=2)
        ttk.Checkbutton(window_frame, text="Window=19", variable=self.window_19).grid(row=0, column=3, padx=5, pady=2)

        # Custom window size entry
        custom_window_frame = ttk.Frame(preprocess_frame)
        custom_window_frame.grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(custom_window_frame, text="Custom:", style='Subheading.TLabel').grid(row=0, column=0, padx=(0, 5))
        ttk.Entry(custom_window_frame, textvariable=self.custom_window_size, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(custom_window_frame, text="(odd number 5-51)", style='Caption.TLabel').grid(row=0, column=2, padx=5)

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

        # Method selection (checkboxes - multiple selection enabled)
        ttk.Label(varsel_frame, text="Selection Methods (select one or more):", style='Subheading.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        ttk.Checkbutton(varsel_frame, text="Feature Importance (default)",
                       variable=self.varsel_importance).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Uses model-specific importance scores",
                 style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(varsel_frame, text="✓ SPA (Successive Projections)",
                       variable=self.varsel_spa).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Collinearity-aware selection",
                 style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(varsel_frame, text="✓ UVE (Uninformative Variable Elimination)",
                       variable=self.varsel_uve).grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Filters noisy variables",
                 style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(varsel_frame, text="✓ UVE-SPA Hybrid",
                       variable=self.varsel_uve_spa).grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Label(varsel_frame, text="Combines noise filtering + collinearity reduction",
                 style='Caption.TLabel').grid(row=4, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(varsel_frame, text="✓ iPLS (Interval PLS)",
                       variable=self.varsel_ipls).grid(row=5, column=0, sticky=tk.W, pady=2)
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

        # Custom n_estimators entry
        ttk.Label(nest_frame, text="Custom:", style='Subheading.TLabel').grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        ttk.Entry(nest_frame, textvariable=self.custom_n_estimators, width=8).grid(row=1, column=1, padx=5, pady=(5, 0))
        ttk.Label(nest_frame, text="(positive integer)", style='Caption.TLabel').grid(row=1, column=2, padx=5, pady=(5, 0))

        # Learning rate options
        ttk.Label(advanced_frame, text="Learning rates:", style='Subheading.TLabel').grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(15, 5))
        lr_frame = ttk.Frame(advanced_frame)
        lr_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=5)
        ttk.Checkbutton(lr_frame, text="0.05", variable=self.lr_005).grid(row=0, column=0, padx=5)
        ttk.Checkbutton(lr_frame, text="0.1 ⭐", variable=self.lr_01).grid(row=0, column=1, padx=5)
        ttk.Checkbutton(lr_frame, text="0.2 ⭐", variable=self.lr_02).grid(row=0, column=2, padx=5)
        ttk.Label(lr_frame, text="(default: 0.1, 0.2)", style='Caption.TLabel').grid(row=0, column=3, padx=10)

        # Custom learning rate entry
        ttk.Label(lr_frame, text="Custom:", style='Subheading.TLabel').grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.custom_learning_rate, width=8).grid(row=1, column=1, padx=5, pady=(5, 0))
        ttk.Label(lr_frame, text="(0.001-1.0)", style='Caption.TLabel').grid(row=1, column=2, padx=5, pady=(5, 0))

        # Info label
        ttk.Label(advanced_frame, text="💡 Selecting more options = more comprehensive analysis but longer runtime",
                 style='Caption.TLabel', foreground=self.colors['accent']).grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))

        # CSV export checkbox
        ttk.Checkbutton(content_frame, text="Export preprocessed data CSV (2nd derivative)",
                       variable=self.export_preprocessed_csv).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
        row += 1

        # === Validation Set Configuration ===
        ttk.Label(content_frame, text="Validation Set Configuration 🆕", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        validation_frame = ttk.LabelFrame(content_frame, text="Holdout Validation Set", padding="20")
        validation_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Enable checkbox
        ttk.Checkbutton(validation_frame, text="Enable Validation Set (holdout samples for independent testing)",
                       variable=self.validation_enabled).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        # Percentage slider
        ttk.Label(validation_frame, text="Validation Set Size:", style='Subheading.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)

        val_pct_frame = ttk.Frame(validation_frame)
        val_pct_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Scale(val_pct_frame, from_=5, to=40, variable=self.validation_percentage, orient=tk.HORIZONTAL, length=300).pack(side=tk.LEFT, padx=(0, 10))

        val_pct_display = ttk.Label(val_pct_frame, text="20%")
        val_pct_display.pack(side=tk.LEFT)

        # Update label when slider changes
        def update_pct_label(*args):
            val_pct_display.config(text=f"{int(self.validation_percentage.get())}%")
        self.validation_percentage.trace_add('write', update_pct_label)

        # Algorithm selection
        ttk.Label(validation_frame, text="Selection Algorithm:", style='Subheading.TLabel').grid(row=3, column=0, sticky=tk.W, pady=(15, 5))

        algo_frame = ttk.Frame(validation_frame)
        algo_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)

        ttk.Radiobutton(algo_frame, text="Kennard-Stone",
                       variable=self.validation_algorithm, value="Kennard-Stone").grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Label(algo_frame, text="Maximizes spectral diversity only (ignores y distribution)",
                 style='Caption.TLabel').grid(row=0, column=1, sticky=tk.W)

        ttk.Radiobutton(algo_frame, text="SPXY ⭐",
                       variable=self.validation_algorithm, value="SPXY").grid(row=1, column=0, sticky=tk.W, padx=(0, 15), pady=3)
        ttk.Label(algo_frame, text="Balances spectral and target diversity (recommended)",
                 style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, pady=3)

        ttk.Radiobutton(algo_frame, text="Random",
                       variable=self.validation_algorithm, value="Random").grid(row=2, column=0, sticky=tk.W, padx=(0, 15), pady=3)
        ttk.Label(algo_frame, text="Simple random selection",
                 style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, pady=3)

        ttk.Radiobutton(algo_frame, text="Stratified",
                       variable=self.validation_algorithm, value="Stratified").grid(row=3, column=0, sticky=tk.W, padx=(0, 15), pady=3)
        ttk.Label(algo_frame, text="Ensures balanced target variable distribution",
                 style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, pady=3)

        # Buttons
        button_frame = ttk.Frame(validation_frame)
        button_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(15, 5))

        ttk.Button(button_frame, text="Create Validation Set", command=self._create_validation_set).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Reset", command=self._reset_validation_set).pack(side=tk.LEFT)

        # Status label
        self.validation_status_label = ttk.Label(validation_frame, text="No validation set created", style='Caption.TLabel')
        self.validation_status_label.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        ttk.Label(validation_frame, text="💡 Validation set will be held out during model training and used for independent testing",
                 style='Caption.TLabel', foreground=self.colors['accent']).grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

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

        # Button frame for actions
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=10, fill='x')

        ttk.Button(button_frame, text="📥 Export Results to CSV",
                   command=self._export_results_table).pack(side='left', padx=5)

        # Status label
        self.results_status = ttk.Label(content_frame, text="No results yet. Run an analysis to see results here.",
                                       style='Caption.TLabel')
        self.results_status.pack(pady=5)

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
        preprocess_combo['values'] = ['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv', 'msc', 'msc_sg1', 'msc_sg2', 'deriv_msc']
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

        # Prediction plot
        ttk.Label(content_frame, text="Prediction Plot", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        plot_frame = ttk.LabelFrame(content_frame, text="Reference vs Predicted", padding="20")
        plot_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.refine_plot_frame = ttk.Frame(plot_frame)
        self.refine_plot_frame.pack(fill='both', expand=True)

        # Residual Diagnostics (regression only)
        ttk.Label(content_frame, text="Residual Diagnostics", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        residual_diagnostics_frame = ttk.LabelFrame(content_frame, text="Residual Analysis", padding="20")
        residual_diagnostics_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Add explanatory text for residual diagnostics
        residual_help_frame = ttk.Frame(residual_diagnostics_frame)
        residual_help_frame.pack(fill='x', padx=10, pady=(5, 15))

        residual_help_text = (
            "Residuals Analysis: Good models show randomly scattered residuals around zero with no patterns.\n\n"
            "• Residuals vs Fitted: Look for random scatter. Patterns (curves, funnels) indicate model issues.\n"
            "• Residuals vs Index: Check for systematic trends across samples.\n"
            "• Q-Q Plot: Points should follow the red diagonal line. Deviations suggest non-normal residuals.\n\n"
            "✓ Good: Random scatter, points on diagonal | ⚠ Warning: Patterns, curved Q-Q plot"
        )

        residual_help_label = ttk.Label(residual_help_frame, text=residual_help_text,
                                        style='Caption.TLabel', justify='left', wraplength=1200)
        residual_help_label.pack(anchor='w')

        self.residual_diagnostics_frame = ttk.Frame(residual_diagnostics_frame)
        self.residual_diagnostics_frame.pack(fill='both', expand=True)

        # Leverage Diagnostics (linear models only)
        ttk.Label(content_frame, text="Leverage Analysis", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        leverage_frame = ttk.LabelFrame(content_frame, text="Influential Samples (Hat Values)", padding="20")
        leverage_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Add explanatory text for leverage diagnostics
        leverage_help_frame = ttk.Frame(leverage_frame)
        leverage_help_frame.pack(fill='x', padx=10, pady=(5, 15))

        leverage_help_text = (
            "Leverage Analysis: Identifies influential samples that strongly affect the model.\n\n"
            "• Interpretation: High-leverage points (red, above threshold) have unusual feature values.\n"
            "• Orange line (2p/n): Moderate influence | Red line (3p/n): High influence\n"
            "  where p = number of model parameters, n = number of samples\n\n"
            "✓ Good: Most points below orange line | ⚠ Warning: Many red points may indicate data quality issues"
        )

        leverage_help_label = ttk.Label(leverage_help_frame, text=leverage_help_text,
                                        style='Caption.TLabel', justify='left', wraplength=1200)
        leverage_help_label.pack(anchor='w')

        self.leverage_plot_frame = ttk.Frame(leverage_frame)
        self.leverage_plot_frame.pack(fill='both', expand=True)

        # Status
        self.refine_status = ttk.Label(content_frame, text="No model loaded", style='Caption.TLabel')
        self.refine_status.grid(row=row, column=0, columnspan=2)

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
            "Or switch to Results tab and double-click a model to refine it.")
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
        self.tab7_preprocess = tk.StringVar(value='raw')
        preproc_combo = ttk.Combobox(left_col, textvariable=self.tab7_preprocess,
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

        self.tab7_window = tk.IntVar(value=17)
        for i, val in enumerate([7, 11, 17, 19]):
            ttk.Radiobutton(window_frame, text=str(val), variable=self.tab7_window,
                           value=val).grid(row=0, column=i, padx=5)
        row += 1

        # CV Folds
        ttk.Label(left_col, text="CV Folds:", style='TLabel').grid(
            row=row, column=0, sticky='w', pady=5)
        self.tab7_folds = tk.IntVar(value=5)
        cv_spin = ttk.Spinbox(left_col, from_=3, to=10, textvariable=self.tab7_folds,
                              width=23)
        cv_spin.grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # Max Iterations
        ttk.Label(left_col, text="Max Iterations:", style='TLabel').grid(
            row=row, column=0, sticky='w', pady=5)
        self.tab7_max_iter = tk.IntVar(value=100)
        iter_spin = ttk.Spinbox(left_col, from_=50, to=500, textvariable=self.tab7_max_iter,
                                width=23)
        iter_spin.grid(row=row, column=1, sticky='w', pady=5)
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
        self.tab7_run_button = ttk.Button(action_frame, text="▶ Run Model Development",
                                       command=self._tab7_run_model,
                                       style='Accent.TButton')
        self.tab7_run_button.pack(side='left', padx=(0, 10))

        # Save Model button (disabled initially)
        self.tab7_save_button = ttk.Button(action_frame, text="💾 Save Model",
                                        command=self._tab7_save_model,
                                        style='Modern.TButton', state='disabled')
        self.tab7_save_button.pack(side='left')

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

    # =============================================================================
    # TAB 7 HELPER METHODS
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
        self.tab7_preprocess.set('raw')
        self.tab7_window.set(11)
        self.tab7_folds.set(5)
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
        """
        Entry point when user clicks 'Run Model' button.

        This method:
        1. Validates that data is loaded
        2. Validates all parameters
        3. Disables the run button
        4. Launches background thread for execution
        """
        # Check if data is loaded
        if self.X is None or self.y is None:
            messagebox.showwarning("No Data", "Please load data first in Tab 1: Import & Preview")
            return

        # Validate parameters
        if not self._tab7_validate_parameters():
            return

        # Disable button during execution
        self.tab7_run_button.config(state='disabled')
        self.tab7_status.config(text="Running model...")

        # Clear previous results
        self.tab7_results_text.config(state='normal')
        self.tab7_results_text.delete('1.0', tk.END)
        self.tab7_results_text.insert('1.0', "Executing model... please wait.\n\nThis may take 30-60 seconds depending on model complexity.")
        self.tab7_results_text.config(state='disabled')

        # Run in background thread to avoid freezing GUI
        import threading
        thread = threading.Thread(target=self._tab7_run_model_thread)
        thread.start()

    def _tab7_validate_parameters(self):
        """
        Validate all parameters before execution.

        Checks:
        - Wavelength specification is not empty
        - Wavelength parsing succeeds
        - Model type is valid
        - CV folds is in valid range (3-10)

        Returns:
            bool: True if all validations pass, False otherwise
        """
        errors = []

        # Validate wavelength specification
        wl_spec_text = self.tab7_wl_spec.get('1.0', 'end').strip()
        if not wl_spec_text:
            errors.append("Wavelength specification is empty.\n\nPlease specify wavelengths (e.g., '1500-2500' or '1500-1800, 2000-2400')")
        else:
            # Try to parse wavelengths
            try:
                available_wl = self.X_original.columns.astype(float).values
                parsed_wl = self._parse_wavelength_spec(wl_spec_text, available_wl)
                if not parsed_wl:
                    errors.append("No valid wavelengths found in specification.\n\nPlease check your wavelength ranges.")
            except Exception as e:
                errors.append(f"Failed to parse wavelength specification:\n{str(e)}")

        # Validate model type
        model_type = self.tab7_model_type.get()
        valid_models = ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']
        if model_type not in valid_models:
            errors.append(f"Invalid model type: {model_type}\n\nValid options: {', '.join(valid_models)}")

        # Validate CV folds
        cv_folds = self.tab7_folds.get()
        if cv_folds < 3 or cv_folds > 10:
            errors.append(f"CV folds must be between 3 and 10 (got {cv_folds})")

        # Show errors if any
        if errors:
            messagebox.showerror("Validation Error", "\n\n".join(errors))
            return False

        return True

    def _tab7_run_model_thread(self):
        """Main execution engine running in background thread."""
        try:
            # ===================================================================
            # IMPORTS
            # ===================================================================
            from spectral_predict.models import get_model
            from spectral_predict.preprocess import build_preprocessing_pipeline
            from sklearn.model_selection import KFold, StratifiedKFold
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.pipeline import Pipeline
            from sklearn.base import clone

            print("\n" + "="*80)
            print("TAB 7: MODEL DEVELOPMENT - EXECUTION START")
            print("="*80)

            # ===================================================================
            # STEP 1: PARSE PARAMETERS
            # ===================================================================
            print("\n[STEP 1] Parsing parameters...")

            # Get basic parameters
            model_name = self.tab7_model_type.get()
            task_type = self.tab7_task_type.get()
            preprocess = self.tab7_preprocess.get()
            window = self.tab7_window.get()
            n_folds = self.tab7_folds.get()
            max_iter = self.tab7_max_iter.get()

            print(f"  Model Type: {model_name}")
            print(f"  Task Type: {task_type}")
            print(f"  Preprocessing: {preprocess}")
            print(f"  Window Size: {window}")
            print(f"  CV Folds: {n_folds}")
            print(f"  Max Iterations: {max_iter}")

            # Parse wavelength specification
            available_wl = self.X_original.columns.astype(float).values
            wl_spec_text = self.tab7_wl_spec.get('1.0', 'end')
            selected_wl = self._parse_wavelength_spec(wl_spec_text, available_wl)

            if not selected_wl:
                raise ValueError("No valid wavelengths selected. Please check your wavelength specification.")

            print(f"  Wavelengths: {len(selected_wl)} selected ({selected_wl[0]:.1f} to {selected_wl[-1]:.1f} nm)")

            # Map GUI preprocessing names to search.py format
            preprocess_name_map = {
                'raw': 'raw',
                'snv': 'snv',
                'sg1': 'deriv',
                'sg2': 'deriv',
                'snv_sg1': 'snv_deriv',
                'snv_sg2': 'snv_deriv',
                'deriv_snv': 'deriv_snv',
                'msc': 'msc',
                'msc_sg1': 'msc_deriv',
                'msc_sg2': 'msc_deriv',
                'deriv_msc': 'deriv_msc'
            }

            deriv_map = {
                'raw': 0, 'snv': 0,
                'sg1': 1, 'sg2': 2,
                'snv_sg1': 1, 'snv_sg2': 2,
                'deriv_snv': 1,
                'msc': 0, 'msc_sg1': 1, 'msc_sg2': 2,
                'deriv_msc': 1
            }

            polyorder_map = {
                'raw': 2, 'snv': 2,
                'sg1': 2, 'sg2': 3,
                'snv_sg1': 2, 'snv_sg2': 3,
                'deriv_snv': 2,
                'msc': 2, 'msc_sg1': 2, 'msc_sg2': 3,
                'deriv_msc': 2
            }

            preprocess_name = preprocess_name_map.get(preprocess, 'raw')
            deriv = deriv_map.get(preprocess, 0)
            polyorder = polyorder_map.get(preprocess, 2)

            print(f"  Preprocessing Config: {preprocess_name} (deriv={deriv}, polyorder={polyorder})")

            # Extract hyperparameters from hyperparam_widgets
            hyperparams = {}
            if hasattr(self, 'tab7_hyperparam_widgets') and self.tab7_hyperparam_widgets:
                for param_name, widget in self.tab7_hyperparam_widgets.items():
                    try:
                        value = widget.get()
                        if param_name in ['n_components', 'n_estimators', 'max_depth']:
                            hyperparams[param_name] = int(value)
                        else:
                            hyperparams[param_name] = float(value)
                    except Exception as e:
                        print(f"  Warning: Could not extract {param_name}: {e}")

            # Set model-specific defaults ONLY (prevent cross-contamination)
            if model_name == 'PLS':
                if 'n_components' not in hyperparams:
                    hyperparams['n_components'] = 10
            elif model_name in ['Ridge', 'Lasso']:
                # DIAGNOSTIC: Alpha extraction from widgets
                print(f"\n🔍 DIAGNOSTIC [Execution]: Alpha extraction for {model_name}")
                print(f"  Extracted from widgets: {hyperparams.get('alpha', 'NOT FOUND')}")
                print(f"  'alpha' in hyperparams? {'alpha' in hyperparams}")
                if 'alpha' not in hyperparams:
                    print(f"  ⚠️  Alpha NOT in hyperparams - using default 1.0")
                    hyperparams['alpha'] = 1.0
                else:
                    print(f"  ✅ Alpha extracted successfully: {hyperparams['alpha']}")
            elif model_name == 'RandomForest':
                if 'n_estimators' not in hyperparams:
                    hyperparams['n_estimators'] = 100
                if 'max_depth' not in hyperparams:
                    hyperparams['max_depth'] = None
            elif model_name == 'MLP':
                if 'learning_rate_init' not in hyperparams:
                    hyperparams['learning_rate_init'] = 0.001
            elif model_name == 'NeuralBoosted':
                if 'n_estimators' not in hyperparams:
                    hyperparams['n_estimators'] = 100
                if 'learning_rate' not in hyperparams:
                    hyperparams['learning_rate'] = 0.1
                if 'hidden_layer_size' not in hyperparams:
                    hyperparams['hidden_layer_size'] = 50

            print(f"  Model-specific hyperparameters for {model_name}: {hyperparams}")
            print(f"  ✓ No cross-contamination from other model types")

            # ===================================================================
            # STEP 2: FILTER DATA
            # ===================================================================
            print("\n[STEP 2] Filtering data...")

            # Start with original data
            X_source = self.X_original if self.X_original is not None else self.X
            y_series = self.y.copy()

            print(f"  Initial shape: X={X_source.shape}, y={y_series.shape}")

            # Apply excluded spectra (from outlier detection)
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

                print(f"  Applying {len(excluded_indices)} excluded spectra ({len(include_indices)} remain)")
                X_base_df = X_source.iloc[include_indices]
                y_series = y_series.iloc[include_indices]
            else:
                X_base_df = X_source.copy()
                y_series = y_series.copy()

            # CRITICAL FIX #1: Exclude validation set (if enabled)
            if self.validation_enabled.get() and self.validation_indices:
                initial_samples = len(X_base_df)
                X_base_df = X_base_df[~X_base_df.index.isin(self.validation_indices)]
                y_series = y_series[~y_series.index.isin(self.validation_indices)]

                n_val = len(self.validation_indices)
                n_removed = initial_samples - len(X_base_df)
                n_cal = len(X_base_df)

                if n_cal < n_folds:
                    raise ValueError(
                        f"Only {n_cal} calibration samples remain after validation set exclusion; "
                        f"{n_folds}-fold CV requires at least {n_folds} samples."
                    )

                print(f"  Excluding {n_removed} validation samples")
                print(f"  Calibration: {n_cal} samples | Validation: {n_val} samples")
                print(f"  This matches the data split used in the Results tab")

            # CRITICAL FIX #2: Reset DataFrame index
            X_base_df = X_base_df.reset_index(drop=True)
            y_series = y_series.reset_index(drop=True)

            print(f"  Reset index after exclusions")
            print(f"  Final shape: X={X_base_df.shape}, y={y_series.shape}")
            print(f"  This ensures CV folds match Results tab (sequential row indexing)")

            # ===================================================================
            # STEP 3: BUILD PREPROCESSING PIPELINE
            # ===================================================================
            print("\n[STEP 3] Building preprocessing pipeline...")

            # Create mapping from float wavelengths to actual column names
            wavelength_columns = X_base_df.columns
            wl_to_col = {float(col): col for col in wavelength_columns}

            # Get the actual column names for selected wavelengths
            selected_cols = [wl_to_col[wl] for wl in selected_wl if wl in wl_to_col]

            if not selected_cols:
                raise ValueError(f"Could not find matching wavelengths. Selected: {len(selected_wl)}, Found: 0")

            # CRITICAL DECISION: Determine preprocessing path
            is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv', 'msc_sg1', 'msc_sg2', 'deriv_msc']
            base_full_vars = len(X_base_df.columns)
            is_subset = len(selected_wl) < base_full_vars
            use_full_spectrum_preprocessing = is_derivative and is_subset

            if use_full_spectrum_preprocessing:
                # PATH A: Derivative + Subset
                print("  PATH A: Derivative + Subset detected")
                print("  Will preprocess FULL spectrum first, then subset")

                # Build preprocessing pipeline WITHOUT model
                prep_steps = build_preprocessing_pipeline(preprocess_name, deriv, window, polyorder)
                prep_pipeline = Pipeline(prep_steps)

                # Preprocess FULL spectrum
                X_full = X_base_df.values
                print(f"  Preprocessing full spectrum ({X_full.shape[1]} wavelengths)...")
                X_full_preprocessed = prep_pipeline.fit_transform(X_full)

                # Find indices of selected wavelengths
                all_wavelengths = X_base_df.columns.astype(float).values
                wavelength_indices = []
                for wl in selected_wl:
                    idx = np.where(np.abs(all_wavelengths - wl) < 0.01)[0]
                    if len(idx) > 0:
                        wavelength_indices.append(idx[0])

                # Subset the PREPROCESSED data
                X_work = X_full_preprocessed[:, wavelength_indices]
                print(f"  Subsetted to {X_work.shape[1]} wavelengths after preprocessing")

                # Extract n_components only for PLS (prevent passing to non-PLS models)
                if model_name == 'PLS':
                    n_components = hyperparams.get('n_components', 10)
                else:
                    n_components = 10  # Default for get_model() signature (not used by other models)

                # Build pipeline with ONLY the model
                model = get_model(
                    model_name,
                    task_type=task_type,
                    n_components=n_components,
                    max_n_components=24,
                    max_iter=max_iter
                )

                # Apply model-specific hyperparameters using set_params()
                params_to_set = {}
                if model_name in ['Ridge', 'Lasso'] and 'alpha' in hyperparams:
                    params_to_set['alpha'] = hyperparams['alpha']
                elif model_name == 'RandomForest':
                    if 'n_estimators' in hyperparams:
                        params_to_set['n_estimators'] = hyperparams['n_estimators']
                    if 'max_depth' in hyperparams:
                        params_to_set['max_depth'] = hyperparams['max_depth']
                elif model_name == 'MLP':
                    if 'learning_rate_init' in hyperparams:
                        params_to_set['learning_rate_init'] = hyperparams['learning_rate_init']
                    if 'hidden_layer_sizes' in hyperparams:
                        params_to_set['hidden_layer_sizes'] = hyperparams['hidden_layer_sizes']
                elif model_name == 'NeuralBoosted':
                    if 'n_estimators' in hyperparams:
                        params_to_set['n_estimators'] = hyperparams['n_estimators']
                    if 'learning_rate' in hyperparams:
                        params_to_set['learning_rate'] = hyperparams['learning_rate']
                    if 'hidden_layer_size' in hyperparams:
                        params_to_set['hidden_layer_size'] = hyperparams['hidden_layer_size']

                if params_to_set:
                    model.set_params(**params_to_set)
                    print(f"  Applied hyperparameters: {params_to_set}")

                pipe_steps = [('model', model)]
                pipe = Pipeline(pipe_steps)

                print(f"  Pipeline: {[name for name, _ in pipe_steps]} (preprocessing already applied)")

            else:
                # PATH B: Standard preprocessing
                print("  PATH B: Standard preprocessing (subset first, then preprocess)")

                # Subset raw data first
                X_work = X_base_df[selected_cols].values

                # Extract n_components only for PLS (prevent passing to non-PLS models)
                if model_name == 'PLS':
                    n_components = hyperparams.get('n_components', 10)
                else:
                    n_components = 10  # Default for get_model() signature (not used by other models)

                # Build full pipeline with preprocessing + model
                pipe_steps = build_preprocessing_pipeline(preprocess_name, deriv, window, polyorder)
                model = get_model(
                    model_name,
                    task_type=task_type,
                    n_components=n_components,
                    max_n_components=24,
                    max_iter=max_iter
                )

                # Apply model-specific hyperparameters using set_params()
                params_to_set = {}
                if model_name in ['Ridge', 'Lasso'] and 'alpha' in hyperparams:
                    params_to_set['alpha'] = hyperparams['alpha']
                elif model_name == 'RandomForest':
                    if 'n_estimators' in hyperparams:
                        params_to_set['n_estimators'] = hyperparams['n_estimators']
                    if 'max_depth' in hyperparams:
                        params_to_set['max_depth'] = hyperparams['max_depth']
                elif model_name == 'MLP':
                    if 'learning_rate_init' in hyperparams:
                        params_to_set['learning_rate_init'] = hyperparams['learning_rate_init']
                    if 'hidden_layer_sizes' in hyperparams:
                        params_to_set['hidden_layer_sizes'] = hyperparams['hidden_layer_sizes']
                elif model_name == 'NeuralBoosted':
                    if 'n_estimators' in hyperparams:
                        params_to_set['n_estimators'] = hyperparams['n_estimators']
                    if 'learning_rate' in hyperparams:
                        params_to_set['learning_rate'] = hyperparams['learning_rate']
                    if 'hidden_layer_size' in hyperparams:
                        params_to_set['hidden_layer_size'] = hyperparams['hidden_layer_size']

                if params_to_set:
                    model.set_params(**params_to_set)
                    print(f"  Applied hyperparameters: {params_to_set}")

                pipe_steps.append(('model', model))
                pipe = Pipeline(pipe_steps)

                print(f"  Pipeline: {[name for name, _ in pipe_steps]} (preprocessing inside CV)")

            # ===================================================================
            # STEP 4: CROSS-VALIDATION
            # ===================================================================
            print(f"\n[STEP 4] Running {n_folds}-fold cross-validation...")

            # CRITICAL: Use shuffle=True to match Results tab behavior
            # Fixed: shuffle=False was causing catastrophic R² differences (issue #DASP-001)
            y_array = y_series.values
            if task_type == "regression":
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                print("  Using KFold (shuffle=True, random_state=42) to match Results tab")
            else:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                print("  Using StratifiedKFold (shuffle=True, random_state=42) to match Results tab")

            # Collect metrics for each fold
            fold_metrics = []
            all_y_true = []
            all_y_pred = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_work, y_array)):
                pipe_fold = clone(pipe)
                X_train, X_test = X_work[train_idx], X_work[test_idx]
                y_train, y_test = y_array[train_idx], y_array[test_idx]

                pipe_fold.fit(X_train, y_train)
                y_pred = pipe_fold.predict(X_test)

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)

                if task_type == "regression":
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    bias = np.mean(y_pred - y_test)
                    fold_metrics.append({"rmse": rmse, "r2": r2, "mae": mae, "bias": bias})
                    print(f"  Fold {fold_idx+1}/{n_folds}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
                else:
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    fold_metrics.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
                    print(f"  Fold {fold_idx+1}/{n_folds}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

            # ===================================================================
            # STEP 5: TRAIN FINAL MODEL
            # ===================================================================
            print("\n[STEP 5] Training final model on full dataset...")

            final_pipe = clone(pipe)
            final_pipe.fit(X_work, y_array)

            final_model = final_pipe.named_steps['model']

            # Build preprocessor from pipeline steps
            if use_full_spectrum_preprocessing:
                final_preprocessor = prep_pipeline
                print("  Using full-spectrum preprocessor (already fitted)")
            elif len(pipe_steps) > 1:
                final_preprocessor = Pipeline(pipe_steps[:-1])
                final_preprocessor.fit(X_work)
                print("  Fitted preprocessor on subset data")
            else:
                final_preprocessor = None
                print("  No preprocessor (raw data)")

            # Store wavelengths AFTER preprocessing
            if final_preprocessor is not None:
                dummy_input = X_work[:1]
                transformed = final_preprocessor.transform(dummy_input)
                n_features_after_preprocessing = transformed.shape[1]

                if use_full_spectrum_preprocessing:
                    refined_wavelengths = list(selected_wl)
                else:
                    n_trimmed = len(selected_wl) - n_features_after_preprocessing
                    if n_trimmed > 0:
                        trim_per_side = n_trimmed // 2
                        refined_wavelengths = list(selected_wl[trim_per_side:len(selected_wl)-trim_per_side])
                        print(f"  Derivative preprocessing trimmed {n_trimmed} edge wavelengths")
                    else:
                        refined_wavelengths = list(selected_wl)
            else:
                refined_wavelengths = list(selected_wl)

            print(f"  Model trained on {X_work.shape[0]} samples × {len(refined_wavelengths)} features")

            # Store model artifacts
            self.tab7_trained_model = final_model
            self.tab7_preprocessing_pipeline = final_preprocessor
            self.tab7_wavelengths = refined_wavelengths
            self.tab7_full_wavelengths = list(all_wavelengths) if use_full_spectrum_preprocessing else None

            # ===================================================================
            # STEP 6: CALCULATE METRICS
            # ===================================================================
            print("\n[STEP 6] Computing performance metrics...")

            results = {}
            if task_type == "regression":
                results['rmse_mean'] = np.mean([m['rmse'] for m in fold_metrics])
                results['rmse_std'] = np.std([m['rmse'] for m in fold_metrics])
                results['r2_mean'] = np.mean([m['r2'] for m in fold_metrics])
                results['r2_std'] = np.std([m['r2'] for m in fold_metrics])
                results['mae_mean'] = np.mean([m['mae'] for m in fold_metrics])
                results['mae_std'] = np.std([m['mae'] for m in fold_metrics])
                results['bias_mean'] = np.mean([m['bias'] for m in fold_metrics])
                results['bias_std'] = np.std([m['bias'] for m in fold_metrics])

                print(f"  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
                print(f"  R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
                print(f"  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
                print(f"  Bias: {results['bias_mean']:.4f} ± {results['bias_std']:.4f}")

                # DIAGNOSTIC: Compare with expected R² from Results tab
                if hasattr(self, 'tab7_expected_r2') and self.tab7_expected_r2 is not None:
                    print(f"\n🔍 DIAGNOSTIC [Validation]: R² Comparison")
                    print(f"  Results tab R²: {self.tab7_expected_r2:.4f}")
                    print(f"  Tab 7 R²:       {results['r2_mean']:.4f}")
                    r2_diff = abs(results['r2_mean'] - self.tab7_expected_r2)
                    print(f"  Difference:     {r2_diff:.4f} ({r2_diff*100:.2f} percentage points)")
                    if r2_diff < 0.001:
                        print(f"  ✅ MATCH! (tolerance: 0.001)")
                    elif r2_diff < 0.01:
                        print(f"  ⚠️  CLOSE (tolerance: 0.01)")
                    else:
                        print(f"  ❌ MISMATCH! Expected difference < 0.01")
                        print(f"  This indicates a BUG in configuration loading/execution!")
            else:
                results['accuracy_mean'] = np.mean([m['accuracy'] for m in fold_metrics])
                results['accuracy_std'] = np.std([m['accuracy'] for m in fold_metrics])
                results['precision_mean'] = np.mean([m['precision'] for m in fold_metrics])
                results['precision_std'] = np.std([m['precision'] for m in fold_metrics])
                results['recall_mean'] = np.mean([m['recall'] for m in fold_metrics])
                results['recall_std'] = np.std([m['recall'] for m in fold_metrics])
                results['f1_mean'] = np.mean([m['f1'] for m in fold_metrics])
                results['f1_std'] = np.std([m['f1'] for m in fold_metrics])

                print(f"  Accuracy:  {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
                print(f"  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
                print(f"  Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
                print(f"  F1 Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

            # Store predictions and configuration
            self.tab7_y_true = np.array(all_y_true)
            self.tab7_y_pred = np.array(all_y_pred)
            self.tab7_config = {
                'model_name': model_name,
                'task_type': task_type,
                'preprocessing': preprocess,
                'window': window,
                'n_vars': len(refined_wavelengths),
                'n_samples': X_work.shape[0],
                'cv_folds': n_folds,
                'use_full_spectrum_preprocessing': use_full_spectrum_preprocessing
            }
            self.tab7_performance = results

            # ===================================================================
            # STEP 7: UPDATE GUI (THREAD-SAFE)
            # ===================================================================
            print("\n[STEP 7] Updating GUI...")

            wl_summary = f"{len(selected_wl)} wavelengths ({selected_wl[0]:.1f} to {selected_wl[-1]:.1f} nm)"

            if task_type == "regression":
                results_text = f"""Model Development Results:

Cross-Validation Performance ({n_folds} folds):
  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}
  R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}
  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}
  Bias: {results['bias_mean']:.4f} ± {results['bias_std']:.4f}

Configuration:
  Model:        {model_name}
  Task Type:    {task_type}
  Preprocessing: {preprocess}
  Window Size:  {window}
  Wavelengths:  {wl_summary}
  Features:     {len(refined_wavelengths)}
  Samples:      {X_work.shape[0]}
  CV Folds:     {n_folds}

Processing Details:
  Path: {'Full-spectrum preprocessing (derivative+subset)' if use_full_spectrum_preprocessing else 'Standard (subset then preprocess)'}
  CV Strategy: {'KFold' if task_type == 'regression' else 'StratifiedKFold'} (shuffle=False, deterministic)
  Validation Set: {'Excluded' if self.validation_enabled.get() else 'Not used'}

The model is ready to be saved. Click 'Save Model' to export as .dasp file.
"""
            else:
                results_text = f"""Model Development Results:

Cross-Validation Performance ({n_folds} folds):
  Accuracy:  {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}
  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}
  Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}
  F1 Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}

Configuration:
  Model:        {model_name}
  Task Type:    {task_type}
  Preprocessing: {preprocess}
  Window Size:  {window}
  Wavelengths:  {wl_summary}
  Features:     {len(refined_wavelengths)}
  Samples:      {X_work.shape[0]}
  CV Folds:     {n_folds}

Processing Details:
  Path: {'Full-spectrum preprocessing (derivative+subset)' if use_full_spectrum_preprocessing else 'Standard (subset then preprocess)'}
  CV Strategy: {'KFold' if task_type == 'regression' else 'StratifiedKFold'} (shuffle=False, deterministic)
  Validation Set: {'Excluded' if self.validation_enabled.get() else 'Not used'}

The model is ready to be saved. Click 'Save Model' to export as .dasp file.
"""

            # Thread-safe GUI update
            self.root.after(0, lambda: self._tab7_update_results(results_text))

            print("\n" + "="*80)
            print("TAB 7: MODEL DEVELOPMENT - EXECUTION COMPLETE")
            print("="*80 + "\n")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            error_text = f"""Error running model:

{str(e)}

Full traceback:
{error_msg}

Please check:
1. Data is loaded correctly (Tab 1)
2. Wavelength specification is valid
3. All parameters are within valid ranges
4. Sufficient samples remain after exclusions

If the problem persists, please report this error.
"""
            print(f"\nERROR: {error_text}")
            self.root.after(0, lambda: self._tab7_update_results(error_text, is_error=True))

    def _tab7_update_results(self, results_text, is_error=False):
        """
        Update the results display on the main thread (thread-safe).

        Args:
            results_text (str): Formatted results text to display
            is_error (bool): Whether this is an error message
        """
        # Update results text widget
        self.tab7_results_text.config(state='normal')
        self.tab7_results_text.delete('1.0', tk.END)
        self.tab7_results_text.insert('1.0', results_text)
        self.tab7_results_text.config(state='disabled')

        # Re-enable run button
        self.tab7_run_button.config(state='normal')

        if is_error:
            # Error state
            self.tab7_status.config(text="✗ Error running model", foreground='red')
            self.tab7_save_button.config(state='disabled')
            messagebox.showerror("Error", "Failed to run model. See results area for details.")
        else:
            # Success state
            self.tab7_status.config(text="✓ Model training complete", foreground='green')
            self.tab7_save_button.config(state='normal')

            # Generate diagnostic plots
            if hasattr(self, 'tab7_y_true') and hasattr(self, 'tab7_y_pred') and hasattr(self, 'tab7_config'):
                try:
                    plot_results = {
                        'y_true': self.tab7_y_true,
                        'y_pred': self.tab7_y_pred,
                        'model_name': self.tab7_config.get('model_name', 'Model'),
                        'r2': self.tab7_performance.get('r2_mean', 0.0),
                        'rmse': self.tab7_performance.get('rmse_mean', 0.0)
                    }
                    self._tab7_generate_plots(plot_results)
                except Exception as e:
                    print(f"Warning: Could not generate diagnostic plots: {e}")
                    import traceback
                    traceback.print_exc()

            # Show success message
            messagebox.showinfo("Success",
                f"Model training complete!\n\n"
                f"R² = {self.tab7_performance.get('r2_mean', 0):.4f}\n\n"
                f"Click 'Save Model' to export as .dasp file.")

    def _tab7_save_model(self):
        """
        Save the trained model to a .dasp file.

        Saves:
        - Trained model object
        - Preprocessing pipeline (fitted)
        - Wavelengths used (after preprocessing)
        - Full wavelengths (for derivative+subset)
        - Performance metrics
        - Configuration metadata
        - Validation set info
        """
        from datetime import datetime
        from pathlib import Path

        # Check if model has been trained
        if not hasattr(self, 'tab7_trained_model') or self.tab7_trained_model is None:
            messagebox.showerror(
                "No Model Trained",
                "Please run a model first before saving.\n\n"
                "Click 'Run Model' to train a model, then you can save it."
            )
            return

        try:
            from spectral_predict.model_io import save_model

            # Ask for save location
            default_name = f"model_{self.tab7_config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dasp"

            # Get initial directory
            initial_dir = None
            if hasattr(self, 'spectral_data_path') and self.spectral_data_path.get():
                data_path = Path(self.spectral_data_path.get())
                initial_dir = str(data_path.parent if data_path.is_file() else data_path)

            filepath = filedialog.asksaveasfilename(
                defaultextension=".dasp",
                filetypes=[("DASP Model", "*.dasp"), ("All files", "*.*")],
                initialfile=default_name,
                initialdir=initial_dir,
                title="Save Trained Model"
            )

            if not filepath:
                return  # User cancelled

            # Build comprehensive metadata
            metadata = {
                'model_name': self.tab7_config['model_name'],
                'task_type': self.tab7_config['task_type'],
                'preprocessing': self.tab7_config['preprocessing'],
                'window': self.tab7_config['window'],
                'wavelengths': self.tab7_wavelengths,
                'n_vars': self.tab7_config['n_vars'],
                'n_samples': self.tab7_config['n_samples'],
                'cv_folds': self.tab7_config['cv_folds'],
                'performance': {},
                'use_full_spectrum_preprocessing': self.tab7_config.get('use_full_spectrum_preprocessing', False),
                'full_wavelengths': self.tab7_full_wavelengths,
                'validation_set_enabled': self.validation_enabled.get() if hasattr(self, 'validation_enabled') else False,
                'validation_indices': list(self.validation_indices) if hasattr(self, 'validation_indices') and self.validation_indices else [],
                'validation_size': len(self.validation_indices) if hasattr(self, 'validation_indices') and self.validation_indices else 0,
                'validation_algorithm': self.validation_algorithm.get() if hasattr(self, 'validation_algorithm') and hasattr(self, 'validation_enabled') and self.validation_enabled.get() else None,
                'created_timestamp': datetime.now().isoformat(),
                'created_by': 'SpectralPredict GUI - Tab 7 Model Development'
            }

            # Add performance metrics
            if self.tab7_config['task_type'] == 'regression':
                metadata['performance'] = {
                    'RMSE': self.tab7_performance.get('rmse_mean'),
                    'RMSE_std': self.tab7_performance.get('rmse_std'),
                    'R2': self.tab7_performance.get('r2_mean'),
                    'R2_std': self.tab7_performance.get('r2_std'),
                    'MAE': self.tab7_performance.get('mae_mean'),
                    'MAE_std': self.tab7_performance.get('mae_std'),
                    'Bias': self.tab7_performance.get('bias_mean'),
                    'Bias_std': self.tab7_performance.get('bias_std')
                }
            else:  # classification
                metadata['performance'] = {
                    'Accuracy': self.tab7_performance.get('accuracy_mean'),
                    'Accuracy_std': self.tab7_performance.get('accuracy_std'),
                    'Precision': self.tab7_performance.get('precision_mean'),
                    'Precision_std': self.tab7_performance.get('precision_std'),
                    'Recall': self.tab7_performance.get('recall_mean'),
                    'Recall_std': self.tab7_performance.get('recall_std'),
                    'F1': self.tab7_performance.get('f1_mean'),
                    'F1_std': self.tab7_performance.get('f1_std')
                }

            # Save the model
            save_model(
                model=self.tab7_trained_model,
                preprocessor=self.tab7_preprocessing_pipeline,
                metadata=metadata,
                filepath=filepath
            )

            # Show success message
            messagebox.showinfo(
                "Model Saved",
                f"Model successfully saved to:\n\n{filepath}\n\n"
                f"Performance:\n"
                f"  R² = {metadata['performance'].get('R2', 0):.4f}\n"
                f"  RMSE = {metadata['performance'].get('RMSE', 0):.4f}\n\n"
                f"You can now load this model in Tab 8 (Model Prediction) "
                f"to make predictions on new data."
            )

            # Update status
            self.tab7_status.config(text=f"✓ Model saved to {Path(filepath).name}")
            print(f"\n✓ Model saved successfully to: {filepath}")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            messagebox.showerror(
                "Save Error",
                f"Failed to save model:\n\n{str(e)}\n\nSee console for details."
            )
            print(f"Error saving model:\n{error_msg}")

    # =============================================================================
    # TAB 7 DIAGNOSTIC PLOTS (PHASE 5)
    # =============================================================================

    def _tab7_clear_plots(self):
        """Clear all Tab 7 diagnostic plots."""
        if not hasattr(self, 'tab7_plot1_frame'):
            return
        for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
            if frame is not None:
                for widget in frame.winfo_children():
                    widget.destroy()

    def _tab7_show_plot_placeholder(self, frame, message):
        """Show placeholder message in plot frame."""
        if frame is None:
            return
        label = ttk.Label(frame, text=message, justify='center', anchor='center',
                         foreground='#999999', font=('Helvetica', 10))
        label.pack(expand=True)

    def _tab7_plot_predictions(self, y_true, y_pred, model_name="Model"):
        """Plot observed vs predicted scatter plot for Tab 7."""
        if not HAS_MATPLOTLIB:
            self._tab7_show_plot_placeholder(self.tab7_plot1_frame,
                "Matplotlib not available\nCannot generate plots")
            return

        # Clear existing widgets
        for widget in self.tab7_plot1_frame.winfo_children():
            widget.destroy()

        try:
            y_true = np.asarray(y_true).flatten()
            y_pred = np.asarray(y_pred).flatten()

            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)

            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black',
                      linewidths=0.5, color='steelblue', label='Predictions')

            # 1:1 reference line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            margin = (max_val - min_val) * 0.05
            ax.plot([min_val - margin, max_val + margin],
                   [min_val - margin, max_val + margin],
                   'r--', lw=2, label='1:1 Line', zorder=1)

            # Calculate statistics
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            bias = np.mean(y_pred - y_true)
            n = len(y_true)

            # Statistics box
            stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nBias = {bias:.4f}\nn = {n}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'),
                   fontsize=9, family='monospace')

            ax.set_xlabel('Observed Values', fontsize=10)
            ax.set_ylabel('Predicted Values', fontsize=10)
            ax.set_title(f'{model_name}\nModel Development Performance',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.legend(loc='lower right', fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.tab7_plot1_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            print(f"Error creating prediction plot: {e}")
            import traceback
            traceback.print_exc()
            self._tab7_show_plot_placeholder(self.tab7_plot1_frame,
                                            f"Error creating plot:\n{str(e)}")

    def _tab7_plot_residuals(self, y_true, y_pred):
        """Plot residual diagnostics (4-panel) for Tab 7."""
        if not HAS_MATPLOTLIB:
            self._tab7_show_plot_placeholder(self.tab7_plot2_frame,
                "Matplotlib not available\nCannot generate plots")
            return

        # Clear existing widgets
        for widget in self.tab7_plot2_frame.winfo_children():
            widget.destroy()

        try:
            from spectral_predict.diagnostics import compute_residuals, qq_plot_data
            from scipy.ndimage import uniform_filter1d

            y_true = np.asarray(y_true).flatten()
            y_pred = np.asarray(y_pred).flatten()

            residuals, std_residuals = compute_residuals(y_true, y_pred)

            fig = Figure(figsize=(5, 4), dpi=100)

            # SUBPLOT 1: Residuals vs Fitted
            ax1 = fig.add_subplot(221)
            ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='black',
                       linewidths=0.5, s=30, color='steelblue')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero')

            # Add trend line
            try:
                sorted_idx = np.argsort(y_pred)
                window_size = max(3, len(y_pred) // 10)
                smoothed = uniform_filter1d(residuals[sorted_idx], size=window_size, mode='nearest')
                ax1.plot(y_pred[sorted_idx], smoothed, 'orange', linewidth=2, alpha=0.7, label='Trend')
            except:
                pass

            ax1.set_xlabel('Fitted Values', fontsize=8)
            ax1.set_ylabel('Residuals', fontsize=8)
            ax1.set_title('Residuals vs Fitted', fontsize=9, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax1.tick_params(labelsize=7)
            ax1.legend(fontsize=6, loc='best')

            # SUBPLOT 2: Residuals vs Index
            ax2 = fig.add_subplot(222)
            indices = np.arange(len(residuals))
            ax2.scatter(indices, residuals, alpha=0.6, edgecolors='black',
                       linewidths=0.5, s=30, color='steelblue')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

            # Highlight outliers
            large_resid_mask = np.abs(std_residuals) > 2.5
            if np.any(large_resid_mask):
                ax2.scatter(indices[large_resid_mask], residuals[large_resid_mask],
                           color='red', s=50, marker='x', linewidths=2,
                           label=f'Outliers (>2.5σ)', zorder=5)
                ax2.legend(fontsize=6, loc='best')

            ax2.set_xlabel('Sample Index', fontsize=8)
            ax2.set_ylabel('Residuals', fontsize=8)
            ax2.set_title('Residuals vs Index', fontsize=9, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax2.tick_params(labelsize=7)

            # SUBPLOT 3: Q-Q Plot
            ax3 = fig.add_subplot(223)
            theoretical_q, sample_q = qq_plot_data(residuals)
            ax3.scatter(theoretical_q, sample_q, alpha=0.6, edgecolors='black',
                       linewidths=0.5, s=30, color='steelblue')
            min_q = min(theoretical_q.min(), sample_q.min())
            max_q = max(theoretical_q.max(), sample_q.max())
            ax3.plot([min_q, max_q], [min_q, max_q], 'r--',
                    linewidth=1.5, label='Normal line')
            ax3.set_xlabel('Theoretical Quantiles', fontsize=8)
            ax3.set_ylabel('Sample Quantiles', fontsize=8)
            ax3.set_title('Q-Q Plot (Normality)', fontsize=9, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax3.tick_params(labelsize=7)
            ax3.legend(fontsize=6, loc='best')

            # SUBPLOT 4: Histogram
            ax4 = fig.add_subplot(224)
            n_bins = min(20, max(10, len(residuals) // 5))
            counts, bins, patches = ax4.hist(residuals, bins=n_bins, alpha=0.7,
                                            edgecolor='black', color='steelblue')

            # Color bars by distance from center
            bin_centers = (bins[:-1] + bins[1:]) / 2
            std_resid_value = np.std(residuals)
            for patch, center in zip(patches, bin_centers):
                if abs(center) > 2 * std_resid_value:
                    patch.set_facecolor('coral')

            ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
            ax4.set_xlabel('Residuals', fontsize=8)
            ax4.set_ylabel('Frequency', fontsize=8)
            ax4.set_title('Distribution', fontsize=9, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
            ax4.tick_params(labelsize=7)
            ax4.legend(fontsize=6, loc='best')

            fig.tight_layout(pad=1.5)

            canvas = FigureCanvasTkAgg(fig, self.tab7_plot2_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            print(f"Error creating residual plots: {e}")
            import traceback
            traceback.print_exc()
            self._tab7_show_plot_placeholder(self.tab7_plot2_frame,
                                            f"Error creating plot:\n{str(e)}")

    def _tab7_plot_model_comparison(self, performance_history):
        """Plot model comparison bar chart for Tab 7."""
        if not HAS_MATPLOTLIB:
            self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
                "Matplotlib not available\nCannot generate plots")
            return

        # Clear existing widgets
        for widget in self.tab7_plot3_frame.winfo_children():
            widget.destroy()

        # Check if we have enough models
        if len(performance_history) == 0:
            self._tab7_show_plot_placeholder(self.tab7_plot3_frame, "No model runs to compare")
            return
        if len(performance_history) == 1:
            self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
                                            "Run multiple models\nfor comparison")
            return

        try:
            labels = [entry['label'] for entry in performance_history]
            r2_scores = [entry['r2'] for entry in performance_history]
            rmse_scores = [entry['rmse'] for entry in performance_history]

            # Sort by R²
            sorted_indices = np.argsort(r2_scores)[::-1]
            labels = [labels[i] for i in sorted_indices]
            r2_scores = [r2_scores[i] for i in sorted_indices]
            rmse_scores = [rmse_scores[i] for i in sorted_indices]

            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)

            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, r2_scores, alpha=0.8, edgecolor='black', linewidth=1)

            # Color bars by performance tier
            max_r2 = max(r2_scores) if r2_scores else 1.0
            colors = []
            for r2 in r2_scores:
                if r2 >= max_r2 * 0.95:
                    colors.append('#4CAF50')  # Green
                elif r2 >= max_r2 * 0.85:
                    colors.append('#FFC107')  # Yellow
                else:
                    colors.append('#F44336')  # Red

            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Add value labels
            for i, (r2, rmse) in enumerate(zip(r2_scores, rmse_scores)):
                ax.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
                if r2 > 0.15:
                    ax.text(i, r2 / 2, f'RMSE:\n{rmse:.3f}', ha='center', va='center',
                           fontsize=7, color='white', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('R² Score', fontsize=10)
            ax.set_title('Model Comparison\n(Training Performance)',
                        fontsize=11, fontweight='bold')

            # X-axis labels
            ax.set_xticks(x_pos)
            display_names = [name[:20] + '...' if len(name) > 20 else name for name in labels]
            ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)

            # Y-axis limits
            y_min = min(0, min(r2_scores) - 0.1)
            y_max = min(1.1, max(r2_scores) * 1.15) if r2_scores else 1.1
            ax.set_ylim([y_min, y_max])

            ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

            # Legend
            legend_elements = [
                Patch(facecolor='#4CAF50', edgecolor='black', label='Top tier (≥95% of best)'),
                Patch(facecolor='#FFC107', edgecolor='black', label='Good (≥85% of best)'),
                Patch(facecolor='#F44336', edgecolor='black', label='Needs improvement (<85%)')
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=7)

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.tab7_plot3_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            print(f"Error creating comparison plot: {e}")
            import traceback
            traceback.print_exc()
            self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
                                            f"Error creating plot:\n{str(e)}")

    def _tab7_generate_plots(self, results):
        """Generate all diagnostic plots for Tab 7 Model Development."""
        try:
            if not HAS_MATPLOTLIB:
                self._tab7_clear_plots()
                for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                    self._tab7_show_plot_placeholder(frame,
                        "Matplotlib not available\nInstall matplotlib for plots")
                return

            if 'y_true' not in results or 'y_pred' not in results:
                self._tab7_clear_plots()
                for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                    self._tab7_show_plot_placeholder(frame, "No data available\nfor diagnostic plots")
                return

            y_true = results['y_true']
            y_pred = results['y_pred']
            model_name = results.get('model_name', 'Model')

            if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
                self._tab7_clear_plots()
                for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                    self._tab7_show_plot_placeholder(frame, "Invalid data\nCannot generate plots")
                return

            # Generate plots
            self._tab7_plot_predictions(y_true, y_pred, model_name)
            self._tab7_plot_residuals(y_true, y_pred)

            # Update performance history
            if not hasattr(self, 'tab7_performance_history'):
                self.tab7_performance_history = []

            history_entry = {
                'label': model_name,
                'r2': results.get('r2', 0.0),
                'rmse': results.get('rmse', 0.0)
            }
            self.tab7_performance_history.append(history_entry)

            # Keep last 10 runs
            if len(self.tab7_performance_history) > 10:
                self.tab7_performance_history = self.tab7_performance_history[-10:]

            self._tab7_plot_model_comparison(self.tab7_performance_history)

            print("✓ Tab 7 diagnostic plots generated successfully")

        except Exception as e:
            print(f"Error generating Tab 7 plots: {e}")
            import traceback
            traceback.print_exc()
            self._tab7_clear_plots()
            error_msg = f"Error generating plots:\n{str(e)[:50]}"
            for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                self._tab7_show_plot_placeholder(frame, error_msg)

    # =============================================================================
    # TAB 7 LOADING METHODS (PHASE 3)
    # =============================================================================

    def _load_model_to_NEW_tab7(self, config):
        """Load a model configuration from Results tab into NEW Tab 7 (Model Development).

        FAIL LOUD wavelength validation - no silent fallbacks!
        """
        import ast

        print("\n" + "="*80)
        print("LOADING MODEL INTO NEW TAB 7 - MODEL DEVELOPMENT")
        print("="*80)

        # STEP 1: Validate Data
        print("\n[STEP 1/7] Validating data availability...")
        if self.X_original is None:
            raise RuntimeError("X_original is not available. Load data in Tab 1.")
        if self.y is None:
            raise RuntimeError("Target variable (y) is not available. Load data in Tab 1.")
        if len(self.X_original.columns) == 0:
            raise RuntimeError("No wavelengths found in X_original.")

        print(f"✓ Data validation passed: X={self.X_original.shape}, y={len(self.y)}, wavelengths={len(self.X_original.columns)}")

        # DIAGNOSTIC: Print full config dict
        print("\n" + "="*80)
        print("🔍 DIAGNOSTIC: FULL CONFIG DICT")
        print("="*80)
        for key, value in sorted(config.items()):
            value_type = type(value).__name__
            if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
                value_display = "NaN"
            elif value is None:
                value_display = "None"
            else:
                value_display = str(value)[:100]  # Truncate long values
            print(f"  {key:20s} = {value_display:50s} (type: {value_type})")
        print("="*80 + "\n")

        # STEP 2: Build Config Info
        print("\n[STEP 2/7] Building configuration information...")
        model_name = config.get('Model', 'N/A')
        rank = config.get('Rank', 'N/A')
        preprocess = config.get('Preprocess', 'N/A')
        subset_tag = config.get('SubsetTag', config.get('Subset', 'full'))
        window = config.get('Window', 'N/A')
        n_vars = config.get('n_vars', 'N/A')
        full_vars = config.get('full_vars', 'N/A')

        info_text = f"Model: {model_name} (Rank {rank})\nPreprocessing: {preprocess}\nSubset: {subset_tag}\nWindow Size: {window}\n"

        if 'RMSE' in config and not pd.isna(config.get('RMSE')):
            rmse, r2 = config.get('RMSE', 'N/A'), config.get('R2', 'N/A')
            info_text += f"\nPerformance: RMSE={rmse}, R²={r2}\n"
        elif 'Accuracy' in config and not pd.isna(config.get('Accuracy')):
            accuracy = config.get('Accuracy', 'N/A')
            info_text += f"\nPerformance: Accuracy={accuracy}\n"

        info_text += f"\nWavelengths: {n_vars} of {full_vars} used"
        if subset_tag not in ['full', 'N/A']:
            info_text += f" ({subset_tag})"
        info_text += "\n"

        print(f"✓ Configuration text built")

        # STEP 3: CRITICAL - Load Wavelengths with FAIL LOUD
        print("\n[STEP 3/7] Loading wavelengths with strict validation...")
        print("⚠️  CRITICAL SECTION: FAIL LOUD validation - no silent fallbacks!")

        all_wavelengths = self.X_original.columns.astype(float).values
        is_subset_model = (subset_tag not in ['full', 'N/A'])

        if is_subset_model:
            print(f"  Subset model detected: '{subset_tag}' with {n_vars} variables")
            if 'all_vars' not in config or not config['all_vars'] or config['all_vars'] == 'N/A':
                raise ValueError(
                    f"CRITICAL ERROR: Missing 'all_vars' field for subset model!\n"
                    f"  Model: {model_name} (Rank {rank})\n"
                    f"  Subset: {subset_tag}\n\n"
                    f"SOLUTION: Re-run the analysis to generate complete results.")

            all_vars_str = str(config['all_vars']).strip()
            wavelength_strings = [w.strip() for w in all_vars_str.split(',')]
            model_wavelengths = [float(w) for w in wavelength_strings if w]

            expected_count = int(n_vars) if n_vars != 'N/A' else len(model_wavelengths)
            if len(model_wavelengths) != expected_count:
                raise ValueError(
                    f"CRITICAL ERROR: Wavelength count mismatch!\n"
                    f"  Expected: {expected_count}, Parsed: {len(model_wavelengths)}\n"
                    f"SOLUTION: Re-run the analysis.")

            available_wl_set = set(all_wavelengths)
            invalid_wls = [w for w in model_wavelengths if w not in available_wl_set]
            if invalid_wls:
                raise ValueError(
                    f"CRITICAL ERROR: Invalid wavelengths in 'all_vars'!\n"
                    f"  Found {len(invalid_wls)} wavelengths not in current dataset\n"
                    f"SOLUTION: Load the original dataset.")

            model_wavelengths = sorted(model_wavelengths)
            print(f"  ✓ Validation passed: {len(model_wavelengths)} wavelengths valid")
        else:
            print(f"  Full model detected - using all {len(all_wavelengths)} wavelengths")
            model_wavelengths = list(all_wavelengths)

        print(f"✓ Wavelength loading complete: {len(model_wavelengths)} wavelengths")

        # STEP 4: Format Wavelengths
        print("\n[STEP 4/7] Formatting wavelengths...")
        wl_display_text = self._format_wavelengths_for_NEW_tab7(model_wavelengths)
        self.tab7_wl_spec.delete('1.0', 'end')
        self.tab7_wl_spec.insert('1.0', wl_display_text)
        print(f"✓ Wavelength widget updated")

        # STEP 5: Load Hyperparameters
        print("\n[STEP 5/7] Loading hyperparameters...")
        def _get_config_value(keys):
            for k in keys:
                if k in config:
                    v = config.get(k)
                    if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v) != 'N/A':
                        return v
            return None

        hyper_lines = []
        if model_name == 'PLS':
            lv_val = _get_config_value(['LVs', 'n_components', 'n_LVs'])
            if lv_val: hyper_lines.append(f"  n_components: {lv_val}")
        elif model_name in ['Ridge', 'Lasso']:
            alpha_val = _get_config_value(['Alpha', 'alpha'])
            # DIAGNOSTIC: Alpha extraction (display only)
            print(f"\n🔍 DIAGNOSTIC [Display]: Alpha extraction for {model_name}")
            print(f"  alpha_val = {alpha_val}")
            print(f"  type = {type(alpha_val)}")
            print(f"  is None? {alpha_val is None}")
            if alpha_val is not None:
                print(f"  is NaN? {pd.isna(alpha_val) if isinstance(alpha_val, (float, np.floating)) else False}")
            if alpha_val: hyper_lines.append(f"  alpha: {alpha_val}")
        elif model_name == 'RandomForest':
            n_est = _get_config_value(['n_estimators', 'n_trees'])
            if n_est: hyper_lines.append(f"  n_estimators: {n_est}")
            max_d = _get_config_value(['max_depth', 'MaxDepth'])
            if max_d: hyper_lines.append(f"  max_depth: {max_d}")
        elif model_name == 'MLP':
            hidden = _get_config_value(['Hidden', 'hidden_layer_sizes'])
            if hidden: hyper_lines.append(f"  hidden_layer_sizes: {hidden}")
            lr = _get_config_value(['LR_init', 'learning_rate_init'])
            if lr: hyper_lines.append(f"  learning_rate_init: {lr}")
        elif model_name == 'NeuralBoosted':
            n_est = _get_config_value(['n_estimators'])
            if n_est: hyper_lines.append(f"  n_estimators: {n_est}")
            lr = _get_config_value(['LearningRate', 'learning_rate'])
            if lr: hyper_lines.append(f"  learning_rate: {lr}")
            hidden_size = _get_config_value(['HiddenSize', 'hidden_layer_size'])
            if hidden_size: hyper_lines.append(f"  hidden_layer_size: {hidden_size}")

        if hyper_lines:
            info_text += "\nHyperparameters:\n" + "\n".join(hyper_lines) + "\n"
            print(f"✓ Loaded {len(hyper_lines)} hyperparameters")

        # STEP 6: Populate GUI Controls
        print("\n[STEP 6/7] Populating GUI controls...")
        self.tab7_config_text.config(state='normal')
        self.tab7_config_text.delete('1.0', tk.END)
        self.tab7_config_text.insert('1.0', info_text)
        self.tab7_config_text.config(state='disabled')

        if model_name in ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']:
            self.tab7_model_type.set(model_name)

            # CRITICAL: Create and populate hyperparameter widgets
            self._tab7_create_hyperparam_widgets(model_name)

            # Populate widgets with extracted values (use .set() for Var objects)
            if model_name == 'PLS':
                lv_val = _get_config_value(['LVs', 'n_components', 'n_LVs'])
                if lv_val and 'n_components' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['n_components'].set(int(lv_val))
            elif model_name in ['Ridge', 'Lasso']:
                alpha_val = _get_config_value(['Alpha', 'alpha'])
                # DIAGNOSTIC: Alpha extraction (widget population)
                print(f"\n🔍 DIAGNOSTIC [Widget]: Alpha extraction for {model_name}")
                print(f"  alpha_val = {alpha_val}")
                print(f"  type = {type(alpha_val)}")
                print(f"  is None? {alpha_val is None}")
                if alpha_val is not None:
                    print(f"  is NaN? {pd.isna(alpha_val) if isinstance(alpha_val, (float, np.floating)) else False}")
                print(f"  'alpha' in widgets? {'alpha' in self.tab7_hyperparam_widgets}")

                # CRITICAL FIX: Use 'is not None' instead of truthiness check
                # This handles alpha=0.0 case (rare but valid - would be falsy!)
                if alpha_val is not None and 'alpha' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['alpha'].set(str(alpha_val))
                    widget_value_after = self.tab7_hyperparam_widgets['alpha'].get()
                    print(f"  ✅ Widget.set() called with: {str(alpha_val)}")
                    print(f"  Widget.get() returns: {widget_value_after}")
                    print(f"  Match? {widget_value_after == str(alpha_val)}")
                else:
                    if 'alpha' in self.tab7_hyperparam_widgets:
                        widget_value = self.tab7_hyperparam_widgets['alpha'].get()
                        print(f"  ⚠️  Widget.set() NOT called (alpha_val={alpha_val})")
                        print(f"  Widget retains default value: {widget_value}")
                        # FAIL LOUD: Raise error if alpha is None for Ridge/Lasso
                        if alpha_val is None:
                            raise ValueError(
                                f"CRITICAL ERROR: Alpha parameter not found in config for {model_name} model!\n"
                                f"  Rank: {rank}\n"
                                f"  Available fields: {list(config.keys())}\n\n"
                                f"This indicates Results tab is not storing alpha correctly.\n"
                                f"SOLUTION: Re-run the analysis to generate complete results.")
                    else:
                        print(f"  ⚠️  Widget.set() NOT called ('alpha' not in widgets)")

            elif model_name == 'RandomForest':
                n_est = _get_config_value(['n_estimators', 'n_trees'])
                if n_est and 'n_estimators' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['n_estimators'].set(int(n_est))
                max_d = _get_config_value(['max_depth', 'MaxDepth'])
                if max_d and 'max_depth' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['max_depth'].set(str(max_d))
            elif model_name == 'MLP':
                hidden = _get_config_value(['Hidden', 'hidden_layer_sizes'])
                if hidden and 'hidden_layer_sizes' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['hidden_layer_sizes'].set(str(hidden))
                lr = _get_config_value(['LR_init', 'learning_rate_init'])
                if lr and 'learning_rate_init' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['learning_rate_init'].set(str(lr))
            elif model_name == 'NeuralBoosted':
                n_est = _get_config_value(['n_estimators'])
                if n_est and 'n_estimators' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['n_estimators'].set(int(n_est))
                lr = _get_config_value(['LearningRate', 'learning_rate'])
                if lr and 'learning_rate' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['learning_rate'].set(str(lr))
                hidden_size = _get_config_value(['HiddenSize', 'hidden_layer_size'])
                if hidden_size and 'hidden_layer_size' in self.tab7_hyperparam_widgets:
                    self.tab7_hyperparam_widgets['hidden_layer_size'].set(int(hidden_size))

            print(f"✓ Hyperparameter widgets populated")

        if self.y is not None:
            if self.y.nunique() < 10:
                self.tab7_task_type.set('classification')
            else:
                self.tab7_task_type.set('regression')

        # Convert preprocessing names
        deriv = config.get('Deriv', None)
        if preprocess == 'deriv' and deriv == 1:
            gui_preprocess = 'sg1'
        elif preprocess == 'deriv' and deriv == 2:
            gui_preprocess = 'sg2'
        elif preprocess == 'snv_deriv':
            gui_preprocess = 'snv_sg1' if deriv == 1 else 'snv_sg2'
        elif preprocess in ['raw', 'snv', 'msc', 'deriv_snv', 'deriv_msc', 'msc_sg1', 'msc_sg2']:
            gui_preprocess = preprocess
        else:
            gui_preprocess = 'raw'

        self.tab7_preprocess.set(gui_preprocess)

        if window != 'N/A' and not pd.isna(window):
            window_val = int(float(window))
            if window_val in [7, 11, 17, 19]:
                self.tab7_window.set(window_val)

        n_folds = config.get('n_folds', 5)
        if not pd.isna(n_folds) and 3 <= int(n_folds) <= 10:
            self.tab7_folds.set(int(n_folds))

        print("✓ GUI controls populated")

        # STEP 7: Finalize
        print("\n[STEP 7/7] Finalizing...")
        self.tab7_loaded_config = config.copy()

        # Store expected R² for comparison after execution
        expected_r2 = config.get('R2', None)
        expected_accuracy = config.get('Accuracy', None)
        if expected_r2 is not None and not pd.isna(expected_r2):
            self.tab7_expected_r2 = float(expected_r2)
            print(f"  📊 Expected R² from Results tab: {self.tab7_expected_r2:.4f}")
        elif expected_accuracy is not None and not pd.isna(expected_accuracy):
            self.tab7_expected_accuracy = float(expected_accuracy)
            print(f"  📊 Expected Accuracy from Results tab: {self.tab7_expected_accuracy:.4f}")
        else:
            self.tab7_expected_r2 = None
            self.tab7_expected_accuracy = None

        self.tab7_mode_label.config(text=f"Mode: Loaded from Results (Rank {rank})",
                                    foreground=self.colors['success'])
        self._tab7_update_wavelength_count()
        self.tab7_status.config(text=f"Loaded: {model_name} | {gui_preprocess} | Rank {rank}")

        print(f"\n{'='*80}")
        print(f"✅ MODEL LOADING COMPLETE: {model_name} (Rank {rank})")
        print(f"   - {len(model_wavelengths)} wavelengths loaded")
        print(f"   - Ready for development in Tab 7")
        print(f"{'='*80}\n")

    def _format_wavelengths_for_NEW_tab7(self, wavelengths_list):
        """Format wavelength list for display in Tab 7."""
        if not wavelengths_list:
            return "# No wavelengths specified"

        wls = sorted(wavelengths_list)
        n_wls = len(wls)

        if n_wls <= 50:
            wl_strings = [f"{w:.1f}" for w in wls]
            return ", ".join(wl_strings)
        else:
            first_10 = [f"{w:.1f}" for w in wls[:10]]
            last_10 = [f"{w:.1f}" for w in wls[-10:]]
            return (", ".join(first_10) + ", ..., " + ", ".join(last_10) +
                   f"\n\n# Total: {n_wls} wavelengths\n# Range: {wls[0]:.1f}-{wls[-1]:.1f} nm")

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

    # ==================== Validation Set Selection Methods ====================

    def _validation_kennard_stone(self, X, n_samples):
        """
        Select validation samples using Kennard-Stone algorithm.
        Maximizes Euclidean distance in X-space for spectral diversity.

        Args:
            X: pandas DataFrame of spectral data
            n_samples: number of samples to select

        Returns:
            list of indices to include in validation set
        """
        from scipy.spatial.distance import pdist, squareform

        X_array = X.values
        n_total = len(X_array)

        if n_samples >= n_total:
            raise ValueError(f"Validation set size ({n_samples}) must be less than total samples ({n_total})")

        # Compute pairwise Euclidean distances
        distances = squareform(pdist(X_array, metric='euclidean'))

        # Start with the two samples that are farthest apart
        max_dist_idx = np.unravel_index(distances.argmax(), distances.shape)
        selected = [max_dist_idx[0], max_dist_idx[1]]

        # Iteratively select samples that maximize minimum distance to already selected
        remaining = list(set(range(n_total)) - set(selected))

        while len(selected) < n_samples:
            # For each remaining sample, find minimum distance to selected samples
            min_distances = []
            for idx in remaining:
                min_dist = min(distances[idx, s] for s in selected)
                min_distances.append((min_dist, idx))

            # Select the sample with maximum minimum distance
            _, best_idx = max(min_distances)
            selected.append(best_idx)
            remaining.remove(best_idx)

        # Convert array indices to DataFrame indices
        return X.index[selected].tolist()

    def _validation_spxy(self, X, y, n_samples):
        """
        Select validation samples using SPXY algorithm.
        Maximizes distance in both X-space and Y-space.

        Args:
            X: pandas DataFrame of spectral data
            y: pandas Series of target values
            n_samples: number of samples to select

        Returns:
            list of indices to include in validation set
        """
        from scipy.spatial.distance import pdist, squareform

        X_array = X.values
        y_array = y.values.reshape(-1, 1)
        n_total = len(X_array)

        if n_samples >= n_total:
            raise ValueError(f"Validation set size ({n_samples}) must be less than total samples ({n_total})")

        # Normalize X and y to [0, 1]
        X_norm = (X_array - X_array.min(axis=0)) / (X_array.max(axis=0) - X_array.min(axis=0) + 1e-10)
        y_norm = (y_array - y_array.min()) / (y_array.max() - y_array.min() + 1e-10)

        # Compute distances in X-space
        dist_X = squareform(pdist(X_norm, metric='euclidean'))

        # Compute distances in y-space
        dist_y = squareform(pdist(y_norm, metric='euclidean'))

        # Combine distances: d_SPXY = d_X + d_y
        distances = dist_X + dist_y

        # Start with the two samples that are farthest apart
        max_dist_idx = np.unravel_index(distances.argmax(), distances.shape)
        selected = [max_dist_idx[0], max_dist_idx[1]]

        # Iteratively select samples that maximize minimum distance to already selected
        remaining = list(set(range(n_total)) - set(selected))

        while len(selected) < n_samples:
            # For each remaining sample, find minimum distance to selected samples
            min_distances = []
            for idx in remaining:
                min_dist = min(distances[idx, s] for s in selected)
                min_distances.append((min_dist, idx))

            # Select the sample with maximum minimum distance
            _, best_idx = max(min_distances)
            selected.append(best_idx)
            remaining.remove(best_idx)

        # Convert array indices to DataFrame indices
        return X.index[selected].tolist()

    def _validation_random(self, X, y, n_samples):
        """
        Select validation samples using random sampling.

        Args:
            X: pandas DataFrame of spectral data
            y: pandas Series of target values
            n_samples: number of samples to select

        Returns:
            list of indices to include in validation set
        """
        from sklearn.model_selection import train_test_split

        n_total = len(X)

        if n_samples >= n_total:
            raise ValueError(f"Validation set size ({n_samples}) must be less than total samples ({n_total})")

        # Calculate test size as fraction
        test_size = n_samples / n_total

        # Use train_test_split for random selection with fixed random_state
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_val.index.tolist()

    def _validation_stratified(self, X, y, n_samples):
        """
        Select validation samples using stratified sampling.
        For classification: stratifies by class.
        For regression: bins y into quartiles and stratifies.

        Args:
            X: pandas DataFrame of spectral data
            y: pandas Series of target values
            n_samples: number of samples to select

        Returns:
            list of indices to include in validation set
        """
        from sklearn.model_selection import train_test_split

        n_total = len(X)

        if n_samples >= n_total:
            raise ValueError(f"Validation set size ({n_samples}) must be less than total samples ({n_total})")

        # Calculate test size as fraction
        test_size = n_samples / n_total

        # Determine if y is continuous or categorical
        if y.dtype in ['object', 'category'] or len(y.unique()) < 10:
            # Classification: use y directly for stratification
            stratify = y
        else:
            # Regression: bin y into quartiles for pseudo-stratification
            stratify = pd.qcut(y, q=4, labels=False, duplicates='drop')

        try:
            _, X_val, _, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
            return X_val.index.tolist()
        except ValueError as e:
            # If stratification fails (e.g., too few samples per bin), fall back to random
            self._log(f"Stratified sampling failed, using random: {e}")
            return self._validation_random(X, y, n_samples)

    def _create_validation_set(self):
        """Create validation set using the selected algorithm."""
        if self.X is None or self.y is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        try:
            # Get current data (after exclusions)
            X_available = self.X[~self.X.index.isin(self.excluded_spectra)]
            y_available = self.y[~self.y.index.isin(self.excluded_spectra)]

            if len(X_available) < 10:
                messagebox.showwarning("Insufficient Data",
                                     "Need at least 10 samples to create validation set")
                return

            # Calculate number of validation samples
            val_pct = self.validation_percentage.get() / 100.0
            n_val = int(len(X_available) * val_pct)

            if n_val < 3:
                messagebox.showwarning("Validation Set Too Small",
                                     f"Validation set would have only {n_val} samples. Increase percentage or dataset size.")
                return

            if n_val > len(X_available) * 0.4:
                messagebox.showwarning("Validation Set Too Large",
                                     f"Validation set of {n_val} samples is more than 40% of data. Consider reducing percentage.")
                return

            # Select validation samples based on algorithm
            algorithm = self.validation_algorithm.get()

            if algorithm == "Kennard-Stone":
                selected_indices = self._validation_kennard_stone(X_available, n_val)
            elif algorithm == "SPXY":
                selected_indices = self._validation_spxy(X_available, y_available, n_val)
            elif algorithm == "Random":
                selected_indices = self._validation_random(X_available, y_available, n_val)
            elif algorithm == "Stratified":
                selected_indices = self._validation_stratified(X_available, y_available, n_val)
            else:
                messagebox.showerror("Invalid Algorithm", f"Unknown algorithm: {algorithm}")
                return

            # Store validation set
            self.validation_indices = set(selected_indices)
            self.validation_X = self.X.loc[selected_indices]
            self.validation_y = self.y.loc[selected_indices]
            self.validation_enabled.set(True)

            # Update status label
            n_cal = len(X_available) - n_val
            if hasattr(self, 'validation_status_label'):
                self.validation_status_label.config(
                    text=f"✓ {n_val} validation samples selected ({algorithm})\n"
                         f"Calibration: {n_cal} samples | Validation: {n_val} samples"
                )

            messagebox.showinfo("Success",
                              f"Created validation set with {n_val} samples using {algorithm} algorithm\n"
                              f"Calibration: {n_cal} samples\n"
                              f"Validation: {n_val} samples")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create validation set:\n{e}")
            import traceback
            traceback.print_exc()

    def _reset_validation_set(self):
        """Reset/clear the validation set."""
        self.validation_indices.clear()
        self.validation_X = None
        self.validation_y = None
        self.validation_enabled.set(False)

        if hasattr(self, 'validation_status_label'):
            self.validation_status_label.config(text="No validation set created")

        messagebox.showinfo("Validation Set Reset", "Validation set has been cleared")

    # ==================== End Validation Set Methods ====================

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
            # Dynamic backend selection
            backend = self.backend_choice.get()

            if backend == "julia":
                from spectral_predict_julia_bridge import run_search_julia as run_search
                self._log_progress("Using Julia backend (optimized, 2-5x faster)\n")
            else:  # python
                from spectral_predict.search import run_search
                self._log_progress("Using Python backend (compatible, all features work)\n")

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
                'msc': self.use_msc.get(),
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


            # Add custom window size if provided
            custom_win = self.custom_window_size.get().strip()
            if custom_win:
                try:
                    custom_val = int(custom_win)
                    if 5 <= custom_val <= 51 and custom_val % 2 == 1:  # Odd number in range
                        if custom_val not in window_sizes:
                            window_sizes.append(custom_val)
                    else:
                        self._log_progress(f"Warning: Custom window size {custom_val} must be odd and in range 5-51. Skipping.")
                except ValueError:
                    self._log_progress(f"Warning: Invalid custom window size ''{custom_win}''. Must be an integer. Skipping.")

            # Default to window size 17 if none specified
            if not window_sizes:
                window_sizes = [17]

            # Collect n_estimators options
            n_estimators_list = []
            if self.n_estimators_50.get():
                n_estimators_list.append(50)
            if self.n_estimators_100.get():
                n_estimators_list.append(100)


            # Add custom n_estimators if provided
            custom_nest = self.custom_n_estimators.get().strip()
            if custom_nest:
                try:
                    custom_val = int(custom_nest)
                    if custom_val > 0:
                        if custom_val not in n_estimators_list:
                            n_estimators_list.append(custom_val)
                    else:
                        self._log_progress(f"Warning: Custom n_estimators {custom_val} must be positive. Skipping.")
                except ValueError:
                    self._log_progress(f"Warning: Invalid custom n_estimators ''{custom_nest}''. Must be an integer. Skipping.")

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


            # Add custom learning rate if provided
            custom_lr = self.custom_learning_rate.get().strip()
            if custom_lr:
                try:
                    custom_val = float(custom_lr)
                    if 0.001 <= custom_val <= 1.0:
                        if custom_val not in learning_rates:
                            learning_rates.append(custom_val)
                    else:
                        self._log_progress(f"Warning: Custom learning rate {custom_val} must be in range 0.001-1.0. Skipping.")
                except ValueError:
                    self._log_progress(f"Warning: Invalid custom learning rate ''{custom_lr}''. Must be a number. Skipping.")

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

            # Filter out validation set (if enabled)
            if self.validation_enabled.get() and self.validation_indices:
                # Remove validation samples from training data
                X_filtered = X_filtered[~X_filtered.index.isin(self.validation_indices)]
                y_filtered = y_filtered[~y_filtered.index.isin(self.validation_indices)]

                n_val = len(self.validation_indices)
                n_cal = len(X_filtered)

                # Update progress with validation info
                self.root.after(0, lambda: self.progress_text.insert(tk.END,
                    f"\n🔬 Validation Set Enabled:\n"))
                self.root.after(0, lambda: self.progress_text.insert(tk.END,
                    f"   • Calibration samples: {n_cal}\n"))
                self.root.after(0, lambda: self.progress_text.insert(tk.END,
                    f"   • Validation samples (held out): {n_val}\n"))
                self.root.after(0, lambda: self.progress_text.insert(tk.END,
                    f"   • Algorithm: {self.validation_algorithm.get()}\n\n"))
                self.root.after(0, lambda: self.progress_text.see(tk.END))

                self._log_progress(f"\n🔬 VALIDATION SET:")
                self._log_progress(f"   Calibration samples: {n_cal}")
                self._log_progress(f"   Validation samples (held out): {n_val}")
                self._log_progress(f"   Selection algorithm: {self.validation_algorithm.get()}")
                self._log_progress(f"")

            # Parse UVE n_components (empty string = None)
            uve_n_comp = None
            if self.uve_n_components.get().strip():
                try:
                    uve_n_comp = int(self.uve_n_components.get())
                except ValueError:
                    self._log_progress("⚠️ Warning: Invalid UVE n_components, using auto-determination")

            # Collect selected variable selection methods
            # NOTE: Method names must match Julia bridge expectations:
            # 'importance', 'SPA', 'UVE', 'iPLS', 'UVE-SPA'
            selected_varsel_methods = []
            if self.varsel_importance.get():
                selected_varsel_methods.append('importance')
            if self.varsel_spa.get():
                selected_varsel_methods.append('SPA')
            if self.varsel_uve.get():
                selected_varsel_methods.append('UVE')
            if self.varsel_uve_spa.get():
                selected_varsel_methods.append('UVE-SPA')
            if self.varsel_ipls.get():
                selected_varsel_methods.append('iPLS')

            # Default to importance if none selected
            if not selected_varsel_methods:
                selected_varsel_methods = ['importance']
                self._log_progress("⚠️ No variable selection method selected, defaulting to 'importance'")

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
                # Variable selection parameters (NEW - supports multiple methods)
                variable_selection_methods=selected_varsel_methods,
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

            # FIX: Check if NeuralBoosted was selected but produced no results
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
            if (self.results_df is None or len(self.results_df) == 0):
                if 'NeuralBoosted' in selected_models:
                    warning_msg = (
                        "NeuralBoosted training failed for all configurations.\n\n"
                        "This model requires specific conditions to train successfully.\n"
                        "Check the console output for detailed error messages.\n\n"
                        "Note: Other models may have completed successfully."
                    )
                    self.root.after(0, lambda: messagebox.showwarning(
                        "NeuralBoosted Training Failed", warning_msg
                    ))
                    self._log_progress("\n[WARN] WARNING: NeuralBoosted produced no results (all training attempts failed)")

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
            error_str = str(e)  # Capture error message before lambda
            self._log_progress(f"\n✗ Error: {e}\n{error_msg}")
            self.root.after(0, lambda: self.progress_status.config(text="✗ Analysis failed"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{error_str}"))

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

    def _sort_results_by_column(self, col):
        """Sort results table by the specified column."""
        if self.results_df is None or len(self.results_df) == 0:
            return

        # Toggle sort direction if clicking the same column
        if self.results_sort_column == col:
            self.results_sort_reverse = not self.results_sort_reverse
        else:
            self.results_sort_column = col
            self.results_sort_reverse = False  # Start with ascending

        # Create a copy to sort
        sorted_df = self.results_df.copy()

        # Convert column to numeric if possible for proper sorting
        try:
            sorted_df[col] = pd.to_numeric(sorted_df[col])
        except (ValueError, TypeError):
            pass  # Keep as string if not numeric

        # Sort the dataframe
        sorted_df = sorted_df.sort_values(by=col, ascending=not self.results_sort_reverse)

        # Repopulate with sorted data
        self._populate_results_table(sorted_df, is_sorted=True)

    def _populate_results_table(self, results_df, is_sorted=False):
        """Populate the results table with analysis results."""
        if results_df is None or len(results_df) == 0:
            self.results_status.config(text="No results to display")
            return

        # Store original results if this is the first population (not a sort)
        if not is_sorted:
            self.results_df = results_df
            self.results_sort_column = None
            self.results_sort_reverse = False

        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Set up columns
        columns = list(results_df.columns)
        self.results_tree['columns'] = columns

        # Configure column headings with sort indicators
        for col in columns:
            # Add sort indicator if this column is currently sorted
            header_text = col
            if self.results_sort_column == col:
                if self.results_sort_reverse:
                    header_text = f"{col} ▼"  # Descending (high to low)
                else:
                    header_text = f"{col} ▲"  # Ascending (low to high)

            # Bind click event to sort by this column
            self.results_tree.heading(col, text=header_text,
                                     command=lambda c=col: self._sort_results_by_column(c))

            # Set column width based on content
            if col in ['Model', 'Preprocess', 'Subset']:
                width = 120
            elif col in ['top_vars']:
                width = 200
            elif col in ['Activation', 'Hidden', 'HiddenSize']:
                width = 90
            elif col in ['LearningRate', 'LR_init', 'Alpha']:
                width = 100
            elif col in ['n_estimators', 'max_depth']:
                width = 95
            elif col in ['Params']:
                # Make Params column wider since it contains full parameter dict
                width = 250
            else:
                width = 80
            self.results_tree.column(col, width=width, anchor='center')

        # Insert data rows with improved preprocessing display
        for idx, row in results_df.iterrows():
            values = []
            for col in columns:
                value = row[col]
                # Make preprocessing names more descriptive
                if col == 'Preprocess' and isinstance(value, str):
                    deriv = row.get('Deriv', None)
                    if value == 'deriv_snv':
                        if deriv == 1:
                            value = 'Deriv1→SNV'  # Derivative THEN SNV
                        elif deriv == 2:
                            value = 'Deriv2→SNV'
                        else:
                            value = 'Deriv→SNV'
                    elif value == 'snv_deriv':
                        if deriv == 1:
                            value = 'SNV→Deriv1'  # SNV THEN Derivative
                        elif deriv == 2:
                            value = 'SNV→Deriv2'
                        else:
                            value = 'SNV→Deriv'
                    elif value == 'msc_deriv':
                        if deriv == 1:
                            value = 'MSC→Deriv1'
                        elif deriv == 2:
                            value = 'MSC→Deriv2'
                        else:
                            value = 'MSC→Deriv'
                    elif value == 'deriv_msc':
                        if deriv == 1:
                            value = 'Deriv1→MSC'
                        elif deriv == 2:
                            value = 'Deriv2→MSC'
                        else:
                            value = 'Deriv→MSC'
                    elif value == 'deriv':
                        if deriv == 1:
                            value = 'Deriv1'
                        elif deriv == 2:
                            value = 'Deriv2'
                    # raw, snv, msc stay as-is
                values.append(value)
            self.results_tree.insert('', 'end', iid=str(idx), values=values)

        # AUTO-LOAD TOP MODEL: When analysis completes, automatically load rank #1 model into Tab 7
        if not is_sorted and len(results_df) > 0:
            try:
                # Get the top-ranking model (rank 1)
                top_model = results_df.iloc[0]
                model_name = top_model.get('Model', '?')
                rank = top_model.get('Rank', '?')
                r2_or_acc = top_model.get('R2', top_model.get('Accuracy', '?'))

                print(f"\n{'='*80}")
                print(f"⚡ AUTO-LOAD: Top-ranking model detected")
                print(f"{'='*80}")
                print(f"  Rank: {rank}")
                print(f"  Model: {model_name}")
                print(f"  Performance: {r2_or_acc}")
                print(f"  Loading into Tab 7 and auto-running...")

                # Schedule auto-load after 1 second (let user see Results tab first)
                def auto_load_top_model():
                    try:
                        model_config = top_model.to_dict()
                        self._load_model_to_NEW_tab7(model_config)
                        # Switch to Tab 7
                        self.notebook.select(6)
                        # Auto-run after 500ms
                        self.root.after(500, self._tab7_run_model)
                    except Exception as e:
                        print(f"⚠️  AUTO-LOAD FAILED: {e}")
                        import traceback
                        traceback.print_exc()

                self.root.after(1000, auto_load_top_model)
                print(f"✅ Auto-load scheduled")
                print(f"{'='*80}\n")
            except Exception as e:
                print(f"⚠️  Could not auto-load top model: {e}")

        # Update status
        self.results_status.config(text=f"Displaying {len(results_df)} results. Double-click a row to refine the model.")

    def _on_result_double_click(self, event):
        """Handle double-click on result row - load into Custom Model Development tab."""
        selection = self.results_tree.selection()
        if not selection:
            return

        if self.results_df is None:
            return

        try:
            # Get the selected row index
            item_id = selection[0]
            row_idx = int(item_id)

            # Get the selected model configuration
            # CRITICAL: Use .loc (label-based) not .iloc (position-based)
            # because treeview IID uses the dataframe's original index labels
            model_config = self.results_df.loc[row_idx].to_dict()
            self.selected_model_config = model_config

            # Log validation info
            rank = model_config.get('Rank', '?')
            model_name = model_config.get('Model', '?')
            r2_or_acc = model_config.get('R2', model_config.get('Accuracy', '?'))
            n_vars = model_config.get('n_vars', '?')

            print(f"\n{'='*80}")
            print(f"USER ACTION: Double-clicked Rank {rank} in Results tab")
            print(f"{'='*80}")
            print(f"  Model: {model_name}")
            print(f"  Performance: {r2_or_acc}")
            print(f"  Variables: {n_vars}")
            print(f"  Loading into NEW Tab 7 (Model Development)...")

            # Load into NEW Tab 7 (Model Development) using robust logic with FAIL LOUD
            self._load_model_to_NEW_tab7(model_config)

            # Switch to NEW Tab 7 (index 6: 0=Import, 1=Quality, 2=Config, 3=Progress, 4=Results, 5=Tab6, 6=Tab7, 7=Tab8)
            self.notebook.select(6)

            print(f"✅ Successfully loaded and switched to Custom Model Development tab")

            # AUTO-RUN: Run the model after 500ms delay to allow GUI to update
            print(f"⚡ AUTO-RUN: Scheduling model execution in 500ms...")
            self.root.after(500, self._tab7_run_model)

            print(f"{'='*80}\n")

        except ValueError as ve:
            # Our validation errors - show to user with full details
            error_msg = str(ve)
            messagebox.showerror(
                "Model Loading Failed - Data Validation Error",
                error_msg
            )
            print(f"\n❌ VALIDATION ERROR:\n{error_msg}\n")
            import traceback
            traceback.print_exc()

        except Exception as e:
            # Unexpected errors - show generic message
            messagebox.showerror(
                "Model Loading Failed",
                f"Failed to load model configuration:\n\n{str(e)}\n\n"
                f"Check the console for detailed error information."
            )
            print(f"\n❌ ERROR loading model: {e}\n")
            import traceback
            traceback.print_exc()

    def _export_results_table(self):
        """Export the current results table to a CSV file."""
        if self.results_df is None or len(self.results_df) == 0:
            messagebox.showwarning(
                "No Results",
                "No results to export. Run an analysis first.")
            return

        try:
            # Get default directory from spectral data path
            initial_dir = None
            if self.spectral_data_path.get():
                data_path = Path(self.spectral_data_path.get())
                initial_dir = str(data_path.parent if data_path.is_file() else data_path)

            # Ask user for save location
            default_name = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=default_name,
                initialdir=initial_dir,
                title="Export Results to CSV"
            )

            if not filepath:
                return  # User cancelled

            # Export the dataframe
            self.results_df.to_csv(filepath, index=False)

            messagebox.showinfo(
                "Export Successful",
                f"Results exported successfully to:\n\n{filepath}"
            )

            self.results_status.config(text=f"✓ Results exported to {Path(filepath).name}")

        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export results:\n\n{str(e)}"
            )

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
        """Legacy wrapper - NOW REDIRECTS TO NEW TAB 7 instead of old Tab 6."""
        print("⚠️  Legacy call detected - redirecting to NEW Tab 7 (Model Development)")
        self._load_model_to_NEW_tab7(config)
        self.notebook.select(6)  # Switch to NEW Tab 7

    def _load_model_to_tab7(self, config):
        """
        OLD Tab 6 loading method - NOW REDIRECTS TO NEW TAB 7.

        This method is DEPRECATED and only kept for backward compatibility.
        All model loading now goes to Tab 7 (Model Development) via _load_model_to_NEW_tab7().

        Args:
            config (dict): Model configuration dictionary from results DataFrame

        Raises:
            ValueError: If critical data is missing or invalid (wavelength mismatch, etc.)
            RuntimeError: If data validation fails
        """
        # REDIRECT TO NEW TAB 7 METHOD
        print("⚠️  OLD Tab 6 method called - redirecting to NEW Tab 7 (Model Development)")
        self._load_model_to_NEW_tab7(config)
        self.notebook.select(6)  # Switch to NEW Tab 7
        return

        # ====================================================================
        # OLD CODE BELOW - NO LONGER EXECUTED (kept for reference only)
        # ====================================================================
        print("\n" + "="*80)
        print("LOADING MODEL INTO CUSTOM MODEL DEVELOPMENT TAB")
        print("="*80)

        # ====================================================================
        # STEP 1: Validate Data Availability
        # ====================================================================
        print("\n[STEP 1/7] Validating data availability...")

        if not self._validate_data_for_refinement():
            raise RuntimeError(
                "Data validation failed!\n"
                "Required data (X, y, wavelengths) is not available.\n"
                "Please ensure data is loaded in the Data Upload tab."
            )

        print("✓ Data validation passed: X, y, and wavelengths are available")
        print(f"  - X shape: {self.X_original.shape}")
        print(f"  - y length: {len(self.y)}")
        print(f"  - Available wavelengths: {len(self.X_original.columns)}")

        # ====================================================================
        # STEP 2: Build Configuration Info Text
        # ====================================================================
        print("\n[STEP 2/7] Building configuration information display...")

        # Extract basic info
        model_name = config.get('Model', 'N/A')
        rank = config.get('Rank', 'N/A')
        preprocess = config.get('Preprocess', 'N/A')
        subset_tag = config.get('SubsetTag', config.get('Subset', 'full'))
        window = config.get('Window', 'N/A')

        info_text = f"""Model: {model_name} (Rank {rank})
Preprocessing: {preprocess}
Subset: {subset_tag}
Window Size: {window}
"""

        # Add performance metrics
        if 'RMSE' in config and not pd.isna(config.get('RMSE')):
            # Regression model
            rmse = config.get('RMSE', 'N/A')
            r2 = config.get('R2', 'N/A')
            info_text += f"""
Performance (Regression):
  RMSE: {rmse}
  R²: {r2}
"""
            print(f"✓ Regression model: RMSE={rmse}, R²={r2}")
        elif 'Accuracy' in config and not pd.isna(config.get('Accuracy')):
            # Classification model
            accuracy = config.get('Accuracy', 'N/A')
            info_text += f"""
Performance (Classification):
  Accuracy: {accuracy}
"""
            if 'ROC_AUC' in config and not pd.isna(config['ROC_AUC']):
                roc_auc = config.get('ROC_AUC', 'N/A')
                info_text += f"  ROC AUC: {roc_auc}\n"
            print(f"✓ Classification model: Accuracy={accuracy}")

        # Add wavelength count info
        n_vars = config.get('n_vars', 'N/A')
        full_vars = config.get('full_vars', 'N/A')
        info_text += f"\nWavelengths: {n_vars} of {full_vars} used"
        if subset_tag not in ['full', 'N/A']:
            info_text += f" ({subset_tag})"
        info_text += "\n"

        print(f"✓ Configuration text built: {model_name}, {n_vars} wavelengths")

        # ====================================================================
        # STEP 3: CRITICAL - Load Wavelengths with FAIL LOUD Validation
        # ====================================================================
        print("\n[STEP 3/7] Loading wavelengths with strict validation...")
        print("⚠️  CRITICAL SECTION: FAIL LOUD validation - no silent fallbacks!")

        model_wavelengths = None
        all_wavelengths = self.X_original.columns.astype(float).values

        # Check if this is a subset model or full model
        is_subset_model = (subset_tag not in ['full', 'N/A'])

        if is_subset_model:
            print(f"  Subset model detected: '{subset_tag}' with {n_vars} variables")

            # CRITICAL: For subset models, we MUST have all_vars field
            if 'all_vars' not in config or not config['all_vars'] or config['all_vars'] == 'N/A':
                raise ValueError(
                    f"CRITICAL ERROR: Missing 'all_vars' field for subset model!\n"
                    f"  Model: {model_name} (Rank {rank})\n"
                    f"  Subset: {subset_tag}\n"
                    f"  Expected variables: {n_vars}\n\n"
                    f"The 'all_vars' field is REQUIRED for subset models to identify\n"
                    f"the exact wavelengths used. Without it, loading this model\n"
                    f"would cause R² discrepancies.\n\n"
                    f"SOLUTION: Re-run the analysis to generate complete results."
                )

            # Parse all_vars field
            print(f"  Parsing 'all_vars' field...")
            try:
                all_vars_str = str(config['all_vars']).strip()
                wavelength_strings = [w.strip() for w in all_vars_str.split(',')]
                parsed_wavelengths = [float(w) for w in wavelength_strings if w]

                print(f"  ✓ Parsed {len(parsed_wavelengths)} wavelengths from 'all_vars'")

                # CRITICAL VALIDATION: Count must match n_vars
                expected_count = int(n_vars) if n_vars != 'N/A' else len(parsed_wavelengths)

                if len(parsed_wavelengths) != expected_count:
                    raise ValueError(
                        f"CRITICAL ERROR: Wavelength count mismatch!\n"
                        f"  Model: {model_name} (Rank {rank})\n"
                        f"  Expected: {expected_count} wavelengths (from n_vars field)\n"
                        f"  Parsed: {len(parsed_wavelengths)} wavelengths (from all_vars field)\n"
                        f"  Subset: {subset_tag}\n\n"
                        f"This indicates a data integrity issue in the results table.\n"
                        f"The 'all_vars' field does not match the 'n_vars' count.\n\n"
                        f"SOLUTION: Re-run the analysis to generate consistent results."
                    )

                # Validate that all parsed wavelengths exist in available data
                available_wl_set = set(all_wavelengths)
                invalid_wls = [w for w in parsed_wavelengths if w not in available_wl_set]

                if invalid_wls:
                    raise ValueError(
                        f"CRITICAL ERROR: Invalid wavelengths in 'all_vars'!\n"
                        f"  Model: {model_name} (Rank {rank})\n"
                        f"  Found {len(invalid_wls)} wavelengths not in current dataset:\n"
                        f"  {invalid_wls[:10]}{'...' if len(invalid_wls) > 10 else ''}\n\n"
                        f"This likely means the current dataset is different from\n"
                        f"the one used to generate these results.\n\n"
                        f"SOLUTION: Load the original dataset or re-run the analysis."
                    )

                model_wavelengths = sorted(parsed_wavelengths)
                print(f"  ✓ Validation passed: All {len(model_wavelengths)} wavelengths are valid")

            except ValueError as ve:
                # Re-raise ValueError (our validation errors)
                raise
            except Exception as e:
                raise ValueError(
                    f"CRITICAL ERROR: Failed to parse 'all_vars' field!\n"
                    f"  Model: {model_name} (Rank {rank})\n"
                    f"  Error: {str(e)}\n"
                    f"  all_vars content: {str(config.get('all_vars', 'MISSING'))[:200]}...\n\n"
                    f"The wavelength data in the results table is malformed.\n\n"
                    f"SOLUTION: Re-run the analysis to generate valid results."
                )

        else:
            # Full model - use all available wavelengths
            print(f"  Full model detected - using all {len(all_wavelengths)} wavelengths")
            model_wavelengths = list(all_wavelengths)

            # Validate count matches (optional check for full models)
            if n_vars != 'N/A' and int(n_vars) != len(all_wavelengths):
                print(f"  ⚠️  WARNING: n_vars ({n_vars}) doesn't match available wavelengths ({len(all_wavelengths)})")
                print(f"      This may indicate the dataset has changed since results were generated")

        # Final validation: ensure we have wavelengths
        if model_wavelengths is None or len(model_wavelengths) == 0:
            raise RuntimeError(
                f"CRITICAL ERROR: No wavelengths loaded!\n"
                f"  Model: {model_name} (Rank {rank})\n"
                f"  Subset: {subset_tag}\n"
                f"  This should never happen - indicates a logic error in loading code."
            )

        print(f"✓ Wavelength loading complete: {len(model_wavelengths)} wavelengths validated")

        # ====================================================================
        # STEP 4: Format Wavelengths for Display
        # ====================================================================
        print("\n[STEP 4/7] Formatting wavelengths for display...")

        try:
            # Use the new formatting helper
            wl_display_text = self._format_wavelengths_for_tab7(model_wavelengths)
            print(f"✓ Formatted {len(model_wavelengths)} wavelengths ({len(wl_display_text)} characters)")

        except Exception as e:
            print(f"⚠️  WARNING: Wavelength formatting failed: {e}")
            # Fallback to simple comma-separated list
            wl_display_text = ", ".join([f"{w:.1f}" for w in model_wavelengths[:100]])
            if len(model_wavelengths) > 100:
                wl_display_text += f", ... ({len(model_wavelengths) - 100} more)"

        # Update wavelength specification widget
        self.refine_wl_spec.config(state='normal')
        self.refine_wl_spec.delete('1.0', 'end')
        self.refine_wl_spec.insert('1.0', wl_display_text)

        # Verify insertion
        content = self.refine_wl_spec.get('1.0', 'end-1c')
        if len(content) == 0:
            raise RuntimeError("ERROR: Wavelength text widget is empty after insertion!")

        print(f"✓ Wavelength widget updated: {len(content)} characters inserted")

        # ====================================================================
        # STEP 5: Load Hyperparameters from Config
        # ====================================================================
        print("\n[STEP 5/7] Loading hyperparameters...")

        # Helper function to get config value with field name variations
        def _get_config_value(keys):
            """Get config value, trying multiple possible field names."""
            for k in keys:
                if k in config:
                    v = config.get(k)
                    if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v) != 'N/A':
                        return v

            # Fallback: try to parse from Params field (legacy support)
            params_str = config.get('Params')
            if params_str:
                try:
                    parsed = ast.literal_eval(params_str)
                    if isinstance(parsed, dict):
                        for k in keys:
                            if k in parsed:
                                return parsed[k]
                except Exception:
                    pass
            return None

        # Build hyperparameters display
        hyper_lines = []

        if model_name == 'PLS':
            lv_val = _get_config_value(['LVs', 'n_components', 'n_LVs'])
            if lv_val is not None:
                hyper_lines.append(f"  n_components (LVs): {lv_val}")

        elif model_name in ['Ridge', 'Lasso']:
            alpha_val = _get_config_value(['Alpha', 'alpha'])
            if alpha_val is not None:
                hyper_lines.append(f"  alpha: {alpha_val}")

        elif model_name == 'RandomForest':
            n_est = _get_config_value(['n_estimators', 'n_trees'])
            if n_est is not None:
                hyper_lines.append(f"  n_estimators: {n_est}")
            max_d = _get_config_value(['max_depth', 'MaxDepth'])
            if max_d is not None:
                hyper_lines.append(f"  max_depth: {max_d}")
            max_f = _get_config_value(['max_features'])
            if max_f is not None:
                hyper_lines.append(f"  max_features: {max_f}")

        elif model_name == 'MLP':
            hidden = _get_config_value(['Hidden', 'hidden_layer_sizes'])
            if hidden is not None:
                hyper_lines.append(f"  hidden_layer_sizes: {hidden}")
            lr_init = _get_config_value(['LR_init', 'learning_rate_init', 'learning_rate'])
            if lr_init is not None:
                hyper_lines.append(f"  learning_rate_init: {lr_init}")

        elif model_name == 'NeuralBoosted':
            n_est = _get_config_value(['n_estimators'])
            if n_est is not None:
                hyper_lines.append(f"  n_estimators: {n_est}")
            lr = _get_config_value(['LearningRate', 'learning_rate'])
            if lr is not None:
                hyper_lines.append(f"  learning_rate: {lr}")
            hidden_size = _get_config_value(['HiddenSize', 'hidden_layer_size'])
            if hidden_size is not None:
                hyper_lines.append(f"  hidden_layer_size: {hidden_size}")
            act = _get_config_value(['Activation', 'activation'])
            if act is not None:
                hyper_lines.append(f"  activation: {act}")

        if hyper_lines:
            info_text += "\nHyperparameters:\n" + "\n".join(hyper_lines) + "\n"
            print(f"✓ Loaded {len(hyper_lines)} hyperparameters")
        else:
            print("  (No hyperparameters found in config)")

        # ====================================================================
        # STEP 6: Populate GUI Controls
        # ====================================================================
        print("\n[STEP 6/7] Populating GUI controls...")

        # Update info text display
        self.refine_model_info.config(state='normal')
        self.refine_model_info.delete('1.0', tk.END)
        self.refine_model_info.insert('1.0', info_text)
        self.refine_model_info.config(state='disabled')
        print("  ✓ Model info display updated")

        # Set model type
        if model_name in ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']:
            self.refine_model_type.set(model_name)
            print(f"  ✓ Model type: {model_name}")
        else:
            print(f"  ⚠️  WARNING: Unknown model type '{model_name}', defaulting to PLS")
            self.refine_model_type.set('PLS')

        # Set task type (auto-detect from data)
        if self.y is not None:
            if self.y.nunique() == 2 or self.y.dtype == 'object' or self.y.nunique() < 10:
                self.refine_task_type.set('classification')
                print("  ✓ Task type: classification (auto-detected)")
            else:
                self.refine_task_type.set('regression')
                print("  ✓ Task type: regression (auto-detected)")

        # Set preprocessing method (handle various naming conventions)
        deriv = config.get('Deriv', None)

        # Convert from search.py naming to GUI naming
        if preprocess == 'deriv' and deriv == 1:
            gui_preprocess = 'sg1'
        elif preprocess == 'deriv' and deriv == 2:
            gui_preprocess = 'sg2'
        elif preprocess == 'snv_deriv':
            gui_preprocess = 'snv_sg1' if deriv == 1 else 'snv_sg2'
        elif preprocess == 'deriv_snv':
            gui_preprocess = 'deriv_snv'
        elif preprocess == 'msc_deriv':
            gui_preprocess = 'msc_sg1' if deriv == 1 else 'msc_sg2'
        elif preprocess == 'deriv_msc':
            gui_preprocess = 'deriv_msc'
        elif preprocess in ['raw', 'snv', 'msc']:
            gui_preprocess = preprocess
        else:
            print(f"  ⚠️  WARNING: Unknown preprocessing '{preprocess}', defaulting to 'raw'")
            gui_preprocess = 'raw'

        if gui_preprocess in ['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv',
                              'msc', 'msc_sg1', 'msc_sg2', 'deriv_msc']:
            self.refine_preprocess.set(gui_preprocess)
            print(f"  ✓ Preprocessing: {gui_preprocess}")

        # Set window size
        if window != 'N/A' and not pd.isna(window):
            window_val = int(float(window))
            if window_val in [7, 11, 17, 19]:
                self.refine_window.set(window_val)
                print(f"  ✓ Window size: {window_val}")
            else:
                print(f"  ⚠️  WARNING: Invalid window={window_val}, using default (17)")
                self.refine_window.set(17)

        # Set CV folds (CRITICAL: Use same folds as Results tab for reproducibility)
        n_folds = config.get('n_folds', 5)
        if not pd.isna(n_folds) and int(n_folds) in range(3, 11):
            self.refine_folds.set(int(n_folds))
            print(f"  ✓ CV folds: {n_folds} (ensures same CV strategy as Results)")
        else:
            print(f"  ⚠️  WARNING: n_folds not in config or invalid, using default (5)")
            self.refine_folds.set(5)

        # ====================================================================
        # STEP 7: Update Mode Label and Enable Controls
        # ====================================================================
        print("\n[STEP 7/7] Finalizing...")

        # Store config for later use
        self.tab7_loaded_config = config.copy()

        # Update mode label
        self.refine_mode_label.config(
            text=f"Mode: Loaded from Results (Rank {rank})",
            foreground='#2E7D32'  # Green color to indicate loaded state
        )
        print(f"✓ Mode label updated: Loaded from Results (Rank {rank})")

        # Enable Run button
        self.refine_run_button.config(state='normal')
        print("✓ Run button enabled")

        # Update wavelength count display
        self._update_wavelength_count()

        # Update status
        if hasattr(self, 'refine_status'):
            self.refine_status.config(
                text=f"Loaded: {model_name} | {gui_preprocess} | Rank {rank}"
            )

        print("\n" + "="*80)
        print(f"✅ MODEL LOADING COMPLETE: {model_name} (Rank {rank})")
        print(f"   - {len(model_wavelengths)} wavelengths loaded and validated")
        print(f"   - Preprocessing: {gui_preprocess}")
        print(f"   - Ready for refinement")
        print("="*80 + "\n")

    def _plot_refined_predictions(self):
        """Plot reference vs predicted values for refined model."""
        if not HAS_MATPLOTLIB:
            return

        if not hasattr(self, 'refined_y_true') or not hasattr(self, 'refined_y_pred'):
            return

        # Clear existing plot
        for widget in self.refine_plot_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        y_true = self.refined_y_true
        y_pred = self.refined_y_pred

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidths=0.5, s=50)

        # Add error bars if prediction intervals available (using ±1 SE)
        if hasattr(self, 'refined_prediction_intervals') and self.refined_prediction_intervals is not None:
            # Reconstruct standard errors from CV folds
            n_samples = len(y_true)
            std_errors_full = np.zeros(n_samples)

            for fold_data in self.refined_prediction_intervals:
                test_idx = fold_data['test_idx']
                std_errors_full[test_idx] = fold_data['std_err']

            # Add error bars showing ±1 SE (much more reasonable than 95% CI)
            ax.errorbar(y_true, y_pred,
                        yerr=std_errors_full,  # ±1 SE
                        fmt='none', ecolor='gray', alpha=0.3, linewidth=0.5, capsize=2,
                        label='±1 SE (Jackknife)')

        # 1:1 line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

        # Calculate statistics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Add statistics text box
        stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nn = {len(y_true)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, family='monospace')

        ax.set_xlabel('Reference Values', fontsize=11)
        ax.set_ylabel('Predicted Values', fontsize=11)
        ax.set_title('Cross-Validation: Reference vs Predicted', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.refine_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_residual_diagnostics(self):
        """Plot three residual diagnostic plots in Tab 6."""
        if not HAS_MATPLOTLIB:
            return

        # Only for regression
        if not hasattr(self, 'refined_config') or self.refined_config.get('task_type') != 'regression':
            return

        if not hasattr(self, 'refined_y_true') or not hasattr(self, 'refined_y_pred'):
            return

        from spectral_predict.diagnostics import compute_residuals, qq_plot_data

        # Clear existing plot
        for widget in self.residual_diagnostics_frame.winfo_children():
            widget.destroy()

        y_true = self.refined_y_true
        y_pred = self.refined_y_pred

        # Compute residuals
        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # Create 1x3 subplot figure
        fig = Figure(figsize=(18, 5))

        # Plot 1: Residuals vs Fitted
        ax1 = fig.add_subplot(131)
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=40)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Fitted Values', fontsize=10)
        ax1.set_ylabel('Residuals', fontsize=10)
        ax1.set_title('Residuals vs Fitted', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals vs Index
        ax2 = fig.add_subplot(132)
        indices = np.arange(len(residuals))
        ax2.scatter(indices, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=40)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Sample Index', fontsize=10)
        ax2.set_ylabel('Residuals', fontsize=10)
        ax2.set_title('Residuals vs Index', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Q-Q Plot
        ax3 = fig.add_subplot(133)
        theoretical_q, sample_q = qq_plot_data(residuals)
        ax3.scatter(theoretical_q, sample_q, alpha=0.6, edgecolors='black', linewidths=0.5, s=40)

        # Add reference line
        min_q = min(theoretical_q.min(), sample_q.min())
        max_q = max(theoretical_q.max(), sample_q.max())
        ax3.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=2)

        ax3.set_xlabel('Theoretical Quantiles', fontsize=10)
        ax3.set_ylabel('Sample Quantiles', fontsize=10)
        ax3.set_title('Q-Q Plot (Normality)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.residual_diagnostics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add dynamic model assessment below the plots
        self._add_residual_assessment(residuals, std_residuals)

    def _add_residual_assessment(self, residuals, std_residuals):
        """Add a dynamic assessment box that evaluates residual quality."""
        # Create assessment frame
        assessment_frame = ttk.Frame(self.residual_diagnostics_frame)
        assessment_frame.pack(fill='x', padx=10, pady=(10, 5))

        # Analyze residuals
        assessment_lines = ["Model Assessment:"]
        issues = []

        # Check for outliers (residuals > 3 standard deviations)
        outlier_threshold = 3.0
        outlier_count = np.sum(np.abs(std_residuals) > outlier_threshold)
        if outlier_count > 0:
            issues.append(f"⚠ {outlier_count} potential outlier(s) detected (|residual| > 3σ)")
        else:
            assessment_lines.append("✓ No significant outliers detected")

        # Check for normality using Q-Q plot deviation
        # Simple check: compare quantiles at extremes
        from spectral_predict.diagnostics import qq_plot_data
        theoretical_q, sample_q = qq_plot_data(residuals)

        # Calculate deviation from diagonal at extremes (first and last 10%)
        n_check = max(1, len(theoretical_q) // 10)
        lower_dev = np.mean(np.abs(sample_q[:n_check] - theoretical_q[:n_check]))
        upper_dev = np.mean(np.abs(sample_q[-n_check:] - theoretical_q[-n_check:]))
        residual_std = np.std(residuals)

        # If deviation is more than 50% of std, flag it
        if lower_dev > 0.5 * residual_std or upper_dev > 0.5 * residual_std:
            issues.append("⚠ Q-Q plot shows deviation from normality at extremes")
        else:
            assessment_lines.append("✓ Residuals appear normally distributed")

        # Check for heteroscedasticity (changing variance)
        # Split residuals into lower and upper half by fitted values
        if hasattr(self, 'refined_y_pred'):
            y_pred = self.refined_y_pred
            sorted_indices = np.argsort(y_pred)
            n_half = len(residuals) // 2
            lower_half_var = np.var(residuals[sorted_indices[:n_half]])
            upper_half_var = np.var(residuals[sorted_indices[n_half:]])

            # If variance ratio is > 2, flag it
            var_ratio = max(lower_half_var, upper_half_var) / (min(lower_half_var, upper_half_var) + 1e-10)
            if var_ratio > 2.0:
                issues.append("⚠ Possible heteroscedasticity (non-constant variance)")
            else:
                assessment_lines.append("✓ Residual variance appears constant")

        # Add issues to assessment
        if issues:
            assessment_lines.extend(issues)

        # Create assessment text
        assessment_text = "\n".join(assessment_lines)

        # Choose background color based on issues
        if len(issues) == 0:
            bg_color = '#d4edda'  # Light green
        elif len(issues) <= 2:
            bg_color = '#fff3cd'  # Light yellow
        else:
            bg_color = '#f8d7da'  # Light red

        # Create label with colored background
        assessment_label = tk.Label(assessment_frame, text=assessment_text,
                                    bg=bg_color, fg='#000000',
                                    font=('TkDefaultFont', 9, 'bold'),
                                    justify='left', anchor='w',
                                    padx=15, pady=10, relief='solid', borderwidth=1)
        assessment_label.pack(fill='x')

    def _add_leverage_assessment(self, leverage, threshold_2p, threshold_3p, n_samples):
        """Add a dynamic assessment box that evaluates leverage distribution."""
        # Create assessment frame
        assessment_frame = ttk.Frame(self.leverage_plot_frame)
        assessment_frame.pack(fill='x', padx=10, pady=(10, 5))

        # Analyze leverage
        assessment_lines = ["Leverage Assessment:"]
        issues = []

        # Count high and moderate leverage points
        n_high = np.sum(leverage > threshold_3p)
        n_moderate = np.sum((leverage > threshold_2p) & (leverage <= threshold_3p))
        n_normal = n_samples - n_high - n_moderate

        # Calculate percentages
        pct_high = (n_high / n_samples) * 100
        pct_moderate = (n_moderate / n_samples) * 100
        pct_normal = (n_normal / n_samples) * 100

        # Check for concerning patterns
        if pct_high > 10:
            issues.append(f"⚠ {n_high} high-leverage points ({pct_high:.1f}%) - Consider investigating data quality")
        elif pct_high > 5:
            issues.append(f"⚠ {n_high} high-leverage points ({pct_high:.1f}%) - Some influential samples detected")
        elif n_high > 0:
            assessment_lines.append(f"✓ {n_high} high-leverage point(s) ({pct_high:.1f}%) - Within acceptable range")
        else:
            assessment_lines.append("✓ No high-leverage points detected")

        # Check moderate leverage
        if pct_moderate > 20:
            issues.append(f"⚠ {n_moderate} moderate-leverage points ({pct_moderate:.1f}%) - Higher than expected")
        elif n_moderate > 0:
            assessment_lines.append(f"✓ {n_moderate} moderate-leverage point(s) ({pct_moderate:.1f}%) - Normal distribution")

        # Overall assessment
        if pct_normal >= 80:
            assessment_lines.append(f"✓ {n_normal} samples ({pct_normal:.1f}%) have normal leverage - Good data distribution")
        elif pct_normal >= 70:
            assessment_lines.append(f"✓ {n_normal} samples ({pct_normal:.1f}%) have normal leverage - Acceptable")

        # Add issues to assessment
        if issues:
            assessment_lines.extend(issues)

        # Create assessment text
        assessment_text = "\n".join(assessment_lines)

        # Choose background color based on issues
        if len(issues) == 0:
            bg_color = '#d4edda'  # Light green
        elif pct_high <= 10:
            bg_color = '#fff3cd'  # Light yellow
        else:
            bg_color = '#f8d7da'  # Light red

        # Create label with colored background
        assessment_label = tk.Label(assessment_frame, text=assessment_text,
                                    bg=bg_color, fg='#000000',
                                    font=('TkDefaultFont', 9, 'bold'),
                                    justify='left', anchor='w',
                                    padx=15, pady=10, relief='solid', borderwidth=1)
        assessment_label.pack(fill='x')

    def _plot_leverage_diagnostics(self):
        """Plot leverage (hat values) to identify influential samples."""
        if not HAS_MATPLOTLIB:
            return

        # Only for regression with linear/PLS models
        if not hasattr(self, 'refined_config'):
            return

        task_type = self.refined_config.get('task_type')
        model_name = self.refined_config.get('model_name')

        # Leverage only meaningful for linear models (PLS, Ridge, Lasso)
        if task_type != 'regression' or model_name not in ['PLS', 'Ridge', 'Lasso']:
            return

        if not hasattr(self, 'refined_X_cv') or self.refined_X_cv is None:
            return  # Need X data for leverage calculation

        from spectral_predict.diagnostics import compute_leverage

        # Clear existing plot
        for widget in self.leverage_plot_frame.winfo_children():
            widget.destroy()

        # Compute leverage on the CV data
        X_data = self.refined_X_cv
        leverage, threshold_2p = compute_leverage(X_data)

        # Calculate 3p/n threshold manually
        n_samples, n_features = X_data.shape
        n_params = n_features + 1  # Include intercept
        threshold_3p = 3.0 * n_params / n_samples

        # Create figure
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # Determine colors based on leverage thresholds
        colors = []
        for h in leverage:
            if h > threshold_3p:
                colors.append('red')  # High leverage
            elif h > threshold_2p:
                colors.append('orange')  # Moderate leverage
            else:
                colors.append('steelblue')  # Normal

        indices = np.arange(len(leverage))
        ax.scatter(indices, leverage, c=colors, alpha=0.7, edgecolors='black', linewidths=0.5, s=60)

        # Add threshold lines
        ax.axhline(y=threshold_2p, color='orange', linestyle='--', linewidth=2,
                   label=f'Moderate Leverage (2p/n = {threshold_2p:.3f})')
        ax.axhline(y=threshold_3p, color='red', linestyle='--', linewidth=2,
                   label=f'High Leverage (3p/n = {threshold_3p:.3f})')

        # Label high-leverage points
        high_leverage_indices = np.where(leverage > threshold_3p)[0]
        for idx in high_leverage_indices:
            ax.annotate(f'{idx}', (idx, leverage[idx]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Leverage (Hat Values)', fontsize=11)
        ax.set_title('Leverage Plot - Influential Samples', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Add info text
        n_high = np.sum(leverage > threshold_3p)
        n_moderate = np.sum((leverage > threshold_2p) & (leverage <= threshold_3p))
        info_text = f'High leverage: {n_high} samples\nModerate leverage: {n_moderate} samples'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.leverage_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add dynamic leverage assessment below the plot
        self._add_leverage_assessment(leverage, threshold_2p, threshold_3p, n_samples)

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

            # Filter out validation set (if enabled) - CRITICAL FIX
            # This ensures Model Development uses the same data split as the main search
            if self.validation_enabled.get() and self.validation_indices:
                # Remove validation samples from training data
                initial_samples = len(X_base_df)
                X_base_df = X_base_df[~X_base_df.index.isin(self.validation_indices)]
                y_series = y_series[~y_series.index.isin(self.validation_indices)]

                n_val = len(self.validation_indices)
                n_removed = initial_samples - len(X_base_df)
                n_cal = len(X_base_df)

                if n_cal < n_folds:
                    raise ValueError(
                        f"Only {n_cal} calibration samples remain after validation set exclusion; "
                        f"{n_folds}-fold CV requires at least {n_folds} samples."
                    )

                print(f"DEBUG: Excluding {n_removed} validation samples from Model Development")
                print(f"DEBUG: Calibration: {n_cal} samples | Validation: {n_val} samples")
                print(f"DEBUG: This matches the data split used in the main search (Results tab)")

            # CRITICAL FIX: Reset DataFrame index to ensure sequential 0-based indexing
            # After exclusions and validation splits, index may have gaps (e.g., [0,1,2,5,7,9,...])
            # Julia's Matrix conversion creates sequential indices, so we must match that behavior
            # Without this, CV folds assign different physical rows despite same indices
            X_base_df = X_base_df.reset_index(drop=True)
            y_series = y_series.reset_index(drop=True)
            print(f"DEBUG: Reset index after exclusions")
            print(f"DEBUG:   X_base_df shape: {X_base_df.shape}, first 5 indices: {list(X_base_df.index[:5])}")
            print(f"DEBUG:   y_series shape: {y_series.shape}, first 5 y values: {list(y_series.values[:5])}")
            print(f"DEBUG:   This ensures CV folds match Julia backend (sequential row indexing)")

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
                'deriv_snv': 'deriv_snv',
                'msc': 'msc',
                'msc_sg1': 'msc_deriv',
                'msc_sg2': 'msc_deriv',
                'deriv_msc': 'deriv_msc'
            }

            deriv_map = {
                'raw': 0,
                'snv': 0,
                'sg1': 1,
                'sg2': 2,
                'snv_sg1': 1,
                'snv_sg2': 2,
                'deriv_snv': 1,
                'msc': 0,
                'msc_sg1': 1,
                'msc_sg2': 2,
                'deriv_msc': 1
            }

            polyorder_map = {
                'raw': 2,
                'snv': 2,
                'sg1': 2,
                'sg2': 3,
                'snv_sg1': 2,
                'snv_sg2': 3,
                'deriv_snv': 2,
                'msc': 2,
                'msc_sg1': 2,
                'msc_sg2': 3,
                'deriv_msc': 2
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
            is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv', 'msc_sg1', 'msc_sg2', 'deriv_msc']
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

            # CRITICAL FIX: Reapply tuned hyperparameters from the search results
            # Robustly handle naming differences and parse 'Params' as fallback.
            params_from_search = {}
            if self.selected_model_config is not None:
                print(f"DEBUG: Loading hyperparameters for model_name='{model_name}'")
                print(f"DEBUG: Config keys available: {list(self.selected_model_config.keys())}")

                # Helper: get first present, non-missing value across candidate keys or from Params
                def _get_cfg(keys):
                    for k in keys:
                        if k in self.selected_model_config:
                            v = self.selected_model_config.get(k)
                            if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v) != 'N/A':
                                return v
                    # Fallback: try to parse Params column if available
                    params_str = self.selected_model_config.get('Params')
                    if params_str:
                        try:
                            parsed = ast.literal_eval(params_str)
                            if isinstance(parsed, dict):
                                for k in keys:
                                    if k in parsed:
                                        return parsed[k]
                        except Exception:
                            pass
                    return None

                # Extract hyperparameters based on model type
                if model_name in ['Ridge', 'Lasso']:
                    alpha_val = _get_cfg(['alpha', 'Alpha'])
                    if alpha_val is not None:
                        try:
                            params_from_search['alpha'] = float(alpha_val)
                            print(f"DEBUG: ✓ Loaded alpha={params_from_search['alpha']} for {model_name}")
                        except Exception as e:
                            print(f"WARNING: Could not parse alpha='{alpha_val}': {e}")
                    else:
                        print(f"DEBUG: ✗ Alpha not found for {model_name} (will use default)")

                elif model_name == 'RandomForest':
                    n_est = _get_cfg(['n_estimators', 'n_trees'])
                    if n_est is not None:
                        try:
                            params_from_search['n_estimators'] = int(n_est)
                            print(f"DEBUG: Loaded n_estimators={params_from_search['n_estimators']} for RandomForest")
                        except Exception as e:
                            print(f"WARNING: Could not parse n_estimators='{n_est}': {e}")
                    max_f = _get_cfg(['max_features'])
                    if max_f is not None and str(max_f) not in ['N/A', 'missing']:
                        params_from_search['max_features'] = str(max_f)
                        print(f"DEBUG: Loaded max_features={params_from_search['max_features']} for RandomForest")
                    max_d = _get_cfg(['max_depth', 'MaxDepth'])
                    if max_d is not None:
                        try:
                            if str(max_d).lower() in ['nothing', 'none', 'null', 'n/a']:
                                params_from_search['max_depth'] = None
                                print("DEBUG: Set RandomForest max_depth=None (unlimited)")
                            else:
                                params_from_search['max_depth'] = int(max_d)
                                print(f"DEBUG: Loaded max_depth={params_from_search['max_depth']} for RandomForest")
                        except Exception as e:
                            print(f"WARNING: Could not parse max_depth='{max_d}': {e}")
                    # Set random_state for reproducibility
                    params_from_search['random_state'] = 42
                    print("DEBUG: Set RandomForest random_state=42 for reproducibility")

                elif model_name == 'MLP':
                    lr_init = _get_cfg(['learning_rate_init', 'LR_init', 'learning_rate'])
                    if lr_init is not None:
                        try:
                            params_from_search['learning_rate_init'] = float(lr_init)
                            print(f"DEBUG: Loaded learning_rate_init={params_from_search['learning_rate_init']} for MLP")
                        except Exception as e:
                            print(f"WARNING: Could not parse learning_rate_init='{lr_init}': {e}")
                    hidden = _get_cfg(['hidden_layer_sizes', 'Hidden'])
                    if hidden is not None:
                        try:
                            # Handle Hidden string like "128-64" or "64"
                            if isinstance(hidden, str):
                                parts = [p for p in hidden.split('-') if p.strip()]
                                sizes = tuple(int(p) for p in parts) if len(parts) > 1 else (int(hidden),)
                                params_from_search['hidden_layer_sizes'] = sizes
                            else:
                                params_from_search['hidden_layer_sizes'] = tuple(hidden) if isinstance(hidden, (list, tuple)) else (int(hidden),)
                            print(f"DEBUG: Loaded hidden_layer_sizes={params_from_search['hidden_layer_sizes']} for MLP")
                        except Exception as e:
                            print(f"WARNING: Could not parse hidden_layer_sizes='{hidden}': {e}")

                elif model_name == 'NeuralBoosted':
                    n_est_nb = _get_cfg(['n_estimators'])
                    if n_est_nb is not None:
                        try:
                            params_from_search['n_estimators'] = int(n_est_nb)
                        except Exception as e:
                            print(f"WARNING: Could not parse n_estimators='{n_est_nb}': {e}")
                    lr_nb = _get_cfg(['learning_rate', 'LearningRate'])
                    if lr_nb is not None:
                        try:
                            params_from_search['learning_rate'] = float(lr_nb)
                        except Exception as e:
                            print(f"WARNING: Could not parse learning_rate='{lr_nb}': {e}")
                    hidden_nb = _get_cfg(['hidden_layer_size', 'HiddenSize'])
                    if hidden_nb is not None:
                        try:
                            params_from_search['hidden_layer_size'] = int(hidden_nb)
                        except Exception as e:
                            print(f"WARNING: Could not parse hidden_layer_size='{hidden_nb}': {e}")
                    act_nb = _get_cfg(['activation', 'Activation'])
                    if act_nb is not None:
                        params_from_search['activation'] = str(act_nb)
                    if params_from_search:
                        print(f"DEBUG: Loaded NeuralBoosted params: {params_from_search}")

            if params_from_search:
                try:
                    model.set_params(**params_from_search)
                    print(f"DEBUG: ✓ Applied tuned hyperparameters from Results: {params_from_search}")
                except Exception as e:
                    print(f"WARNING: Failed to apply saved parameters {params_from_search}: {e}")

            # Build preprocessing pipeline and prepare data
            from spectral_predict.preprocess import build_preprocessing_pipeline
            from sklearn.pipeline import Pipeline

            # Prepare cross-validation
            # CRITICAL FIX: Use shuffle=False to ensure identical fold splits as Julia backend
            # Julia and Python use different RNG algorithms, so even with same seed (42),
            # they create different splits when shuffle=True. Using shuffle=False ensures
            # deterministic, data-order-based folds that match between backends.
            y_array = y_series.values
            if task_type == "regression":
                cv = KFold(n_splits=n_folds, shuffle=False)  # No shuffle for consistency
            else:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=False)  # No shuffle for consistency

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
            all_y_true = []
            all_y_pred = []
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

                # Store predictions for plotting
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)

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

            # Compute prediction intervals for PLS models (jackknife method)
            # Do this BEFORE building results_text so we can include interval info
            prediction_intervals = None
            avg_std_error = None

            if model_name == 'PLS' and task_type == 'regression' and len(X_raw) < 300:
                # Only compute for PLS regression with reasonable sample size
                # Skip if n > 300 (too slow - jackknife is O(n^2 * fit_time))
                from spectral_predict.diagnostics import jackknife_prediction_intervals

                print("DEBUG: Computing jackknife prediction intervals (may take 1-2 min)...")
                all_intervals = []

                # Recompute CV to get intervals for each fold
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_raw, y_array)):
                    X_train, X_test = X_raw[train_idx], X_raw[test_idx]
                    y_train, y_test = y_array[train_idx], y_array[test_idx]

                    # Clone and fit ENTIRE pipeline (not just model!)
                    # CRITICAL: Pass entire pipeline to jackknife function to preserve preprocessing
                    pipe_fold = clone(pipe)
                    pipe_fold.fit(X_train, y_train)

                    # Compute intervals - pass PIPELINE, not extracted model
                    try:
                        _, lower, upper, std_err = jackknife_prediction_intervals(
                            pipe_fold,  # Pass entire pipeline (preprocessing + model)
                            X_train, y_train, X_test, confidence=0.95
                        )

                        all_intervals.append({
                            'test_idx': test_idx,
                            'lower': lower,
                            'upper': upper,
                            'std_err': std_err
                        })
                        print(f"DEBUG: Fold {fold_idx+1}/{n_folds} intervals computed (n_test={len(test_idx)})")
                    except Exception as e:
                        print(f"WARNING: Failed to compute intervals for fold {fold_idx}: {e}")
                        all_intervals = None
                        break

                if all_intervals is not None and len(all_intervals) > 0:
                    prediction_intervals = all_intervals
                    # Compute average standard error for display
                    all_std_errors = []
                    for fold_data in prediction_intervals:
                        all_std_errors.extend(fold_data['std_err'])
                    avg_std_error = np.mean(all_std_errors)
                    print(f"DEBUG: Prediction intervals computed successfully. Avg SE: ±{avg_std_error:.4f}")
                else:
                    print("DEBUG: Prediction interval computation failed - continuing without intervals.")
            else:
                if model_name == 'PLS' and task_type == 'regression' and len(X_raw) >= 300:
                    print(f"DEBUG: Skipping prediction intervals (n={len(X_raw)} >= 300, too slow)")

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

                # Build interval info text if available
                interval_text = ""
                if avg_std_error is not None:
                    interval_text = f"""
Prediction Uncertainty (Standard Error):
  Average SE: ±{avg_std_error:.4f}
  Method: Jackknife (leave-one-out)
  Note: Error bars (±1 SE) shown in prediction plot
"""

                results_text = f"""Refined Model Results:

Cross-Validation Performance ({self.refine_folds.get()} folds):
  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}
  R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}
  MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}
{interval_text}
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

            # CRITICAL FIX: Store wavelengths AFTER preprocessing, not before
            # Derivatives remove edge wavelengths, so model expects fewer features than original
            if final_preprocessor is not None:
                # Apply preprocessor to get actual feature count
                dummy_input = X_raw[:1]  # Single sample for testing
                transformed = final_preprocessor.transform(dummy_input)
                n_features_after_preprocessing = transformed.shape[1]

                if use_full_spectrum_preprocessing:
                    # Derivative + subset: wavelengths already determined by subset indices
                    self.refined_wavelengths = list(selected_wl)  # Already the subset wavelengths
                else:
                    # Regular preprocessing: derivatives trim edges
                    # Calculate which wavelengths remain after edge trimming
                    n_trimmed = len(selected_wl) - n_features_after_preprocessing
                    if n_trimmed > 0:
                        # Edges were trimmed symmetrically
                        trim_per_side = n_trimmed // 2
                        self.refined_wavelengths = list(selected_wl[trim_per_side:len(selected_wl)-trim_per_side])
                        print(f"DEBUG: Derivative preprocessing trimmed {n_trimmed} edge wavelengths")
                        print(f"DEBUG: Storing {len(self.refined_wavelengths)} wavelengths (after trimming)")
                    else:
                        # No trimming (raw/SNV/MSC)
                        self.refined_wavelengths = list(selected_wl)
            else:
                # No preprocessor - use original wavelengths
                self.refined_wavelengths = list(selected_wl)

            self.refined_performance = results
            self.refined_config = {
                'model_name': model_name,
                'task_type': task_type,
                'preprocessing': preprocess,
                'window': window,
                'n_vars': len(self.refined_wavelengths),
                'n_samples': X_raw.shape[0],
                'cv_folds': n_folds,
                'use_full_spectrum_preprocessing': use_full_spectrum_preprocessing
            }

            # Store predictions for plotting
            self.refined_y_true = np.array(all_y_true)
            self.refined_y_pred = np.array(all_y_pred)

            # Store prediction intervals for plotting
            self.refined_prediction_intervals = prediction_intervals

            # Store preprocessed X data for leverage calculation
            # CRITICAL: Must use PREPROCESSED X, not raw X
            if len(pipe_steps) > 1:
                # Pipeline has preprocessing steps - extract transformed X
                preprocessor = Pipeline(pipe_steps[:-1])  # All steps except model
                preprocessor.fit(X_raw)  # Fit on raw data
                X_transformed = preprocessor.transform(X_raw)
                self.refined_X_cv = X_transformed
                print(f"DEBUG: Stored preprocessed X for leverage calculation (shape: {X_transformed.shape})")
            else:
                # No preprocessing - use raw X
                self.refined_X_cv = X_raw.copy()
                print(f"DEBUG: Stored raw X for leverage calculation (shape: {X_raw.shape})")

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
            # Plot the predictions
            self._plot_refined_predictions()
            # Plot diagnostic plots
            self._plot_residual_diagnostics()
            self._plot_leverage_diagnostics()
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

            # Get initial directory from spectral data path
            initial_dir = None
            if self.spectral_data_path.get():
                data_path = Path(self.spectral_data_path.get())
                # If it's a file, get its parent directory; if it's a directory, use it
                initial_dir = str(data_path.parent if data_path.is_file() else data_path)

            filepath = filedialog.asksaveasfilename(
                defaultextension=".dasp",
                filetypes=[("DASP Model", "*.dasp"), ("All files", "*.*")],
                initialfile=default_name,
                initialdir=initial_dir,
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
                'full_wavelengths': self.refined_full_wavelengths,  # All wavelengths for derivative+subset
                # Validation set metadata
                'validation_set_enabled': self.validation_enabled.get(),
                'validation_indices': list(self.validation_indices) if self.validation_indices else [],
                'validation_size': len(self.validation_indices) if self.validation_indices else 0,
                'validation_algorithm': self.validation_algorithm.get() if self.validation_enabled.get() else None
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

    def _format_wavelengths_for_tab7(self, wavelengths_list):
        """
        Format wavelength list for display in Custom Model Development tab.

        For small lists (<= 50), shows all wavelengths as comma-separated values.
        For large lists (> 50), shows first 10, "...", last 10, with total count.

        Args:
            wavelengths_list (list): List of wavelength values (floats)

        Returns:
            str: Formatted wavelength string for display

        Examples:
            Small list (25 wavelengths):
            "1520.0, 1540.0, 1560.0, ... (full list)"

            Large list (200 wavelengths):
            "1520.0, 1540.0, ..., 2480.0, 2500.0 (200 wavelengths total)"
        """
        if not wavelengths_list:
            return "# No wavelengths specified"

        # Sort wavelengths for consistent display
        wls = sorted(wavelengths_list)
        n_wls = len(wls)

        if n_wls <= 50:
            # Show all wavelengths
            wl_strings = [f"{w:.1f}" for w in wls]
            formatted = ", ".join(wl_strings)
            return formatted
        else:
            # Show condensed format: first 10 + "..." + last 10
            first_10 = [f"{w:.1f}" for w in wls[:10]]
            last_10 = [f"{w:.1f}" for w in wls[-10:]]

            formatted = (
                ", ".join(first_10) +
                ", ..., " +
                ", ".join(last_10) +
                f"\n\n# Total: {n_wls} wavelengths selected" +
                f"\n# Range: {wls[0]:.1f} nm to {wls[-1]:.1f} nm"
            )
            return formatted

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

    # === Tab 8: Model Prediction Methods ===

    def _create_tab8_model_prediction(self):
        """Create Tab 8: Model Prediction - Load models and make predictions on new data."""
        self.tab8 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab8, text='  🔮 Model Prediction  ')

        # Create scrollable content
        canvas = tk.Canvas(self.tab8, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab8, orient="vertical", command=canvas.yview)
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
                       variable=self.pred_data_source, value='directory',
                       command=self._on_pred_source_change).pack(side='left', padx=5)
        ttk.Radiobutton(source_frame, text="CSV File",
                       variable=self.pred_data_source, value='csv',
                       command=self._on_pred_source_change).pack(side='left', padx=5)
        ttk.Radiobutton(source_frame, text="Use Pre-Selected Validation Set 🔬",
                       variable=self.pred_data_source, value='validation',
                       command=self._on_pred_source_change).pack(side='left', padx=5)

        # File path entry
        self.pred_path_label = ttk.Label(step2_frame, text="Path:", style='Caption.TLabel')
        self.pred_path_label.grid(row=2, column=0, sticky=tk.W, pady=(10, 5))

        path_frame = ttk.Frame(step2_frame)
        path_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.pred_data_path = tk.StringVar()
        self.pred_path_entry = ttk.Entry(path_frame, textvariable=self.pred_data_path, width=60)
        self.pred_path_entry.pack(side='left', fill='x', expand=True)
        self.pred_browse_button = ttk.Button(path_frame, text="Browse...",
                   command=self._browse_prediction_data)
        self.pred_browse_button.pack(side='left', padx=5)

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

    def _on_pred_source_change(self):
        """Handle data source radio button change - enable/disable path widgets."""
        source = self.pred_data_source.get()

        if source == 'validation':
            # Disable path entry and browse button when validation set is selected
            self.pred_path_entry.config(state='disabled')
            self.pred_browse_button.config(state='disabled')
            self.pred_data_path.set("(Using pre-selected validation set)")
        else:
            # Enable path entry and browse button for directory and CSV options
            self.pred_path_entry.config(state='normal')
            self.pred_browse_button.config(state='normal')
            if self.pred_data_path.get() == "(Using pre-selected validation set)":
                self.pred_data_path.set("")

    def _load_prediction_data(self):
        """Load spectral data for predictions."""
        source = self.pred_data_source.get()

        # Handle validation set separately
        if source == 'validation':
            if self.validation_X is None or self.validation_y is None:
                messagebox.showerror("No Validation Set",
                    "No validation set has been created yet.\n\n"
                    "Please go to the Analysis Configuration tab and create a validation set first.")
                return

            try:
                self.prediction_data = self.validation_X.copy()

                # Update status
                n_samples = len(self.prediction_data)
                n_wavelengths = len(self.prediction_data.columns)

                self.pred_data_status.config(
                    text=f"✓ Loaded validation set: {n_samples} spectra with {n_wavelengths} wavelengths"
                )

                messagebox.showinfo("Success",
                    f"Validation set loaded successfully:\n"
                    f"{n_samples} spectra\n"
                    f"{n_wavelengths} wavelengths\n"
                    f"Algorithm: {self.validation_algorithm.get()}")

                return

            except Exception as e:
                messagebox.showerror("Load Error",
                    f"Failed to load validation set:\n{str(e)}")
                self.pred_data_status.config(text="Load failed")
                return

        # For directory and CSV options, need a path
        path_str = self.pred_data_path.get()

        if not path_str:
            messagebox.showerror("No Path", "Please select a data source first.")
            return

        path = Path(path_str)

        if not path.exists():
            messagebox.showerror("Path Error", f"Path does not exist:\n{path_str}")
            return

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

            # Clear and initialize model map
            self.predictions_model_map = {}

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

                    # Store mapping to model metadata
                    self.predictions_model_map[col_name] = metadata

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

        # Check if we're using validation set (has actual y values)
        is_validation = (self.pred_data_source.get() == 'validation' and
                        self.validation_y is not None)

        if is_validation:
            stats_text = "🔬 VALIDATION SET RESULTS\n"
            stats_text += "=" * 60 + "\n"
            stats_text += f"Algorithm: {self.validation_algorithm.get()}\n"
            stats_text += f"Validation Samples: {len(self.validation_y)}\n"
            stats_text += "=" * 60 + "\n\n"
        else:
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

                    # Add variable information from model metadata
                    if col in self.predictions_model_map:
                        metadata = self.predictions_model_map[col]
                        n_vars = metadata.get('n_vars', 'Unknown')
                        wavelengths = metadata.get('wavelengths', None)

                        stats_text += f"  • Variables: {n_vars}\n"

                        # Format wavelengths if available and not too many
                        if wavelengths is not None:
                            wl_spec = self._format_wavelengths_as_spec(wavelengths)
                            if wl_spec and len(wl_spec) < 500:  # Only show if reasonable length
                                stats_text += f"  • Wavelengths: {wl_spec}\n"

                    # If validation set, calculate performance metrics
                    if is_validation:
                        try:
                            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                            # Get actual values aligned with predictions
                            y_true = self.validation_y.loc[self.predictions_df['Sample']].values
                            y_pred = values.values

                            # Calculate metrics
                            r2 = r2_score(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            mae = mean_absolute_error(y_true, y_pred)

                            stats_text += f"  ✓ R² Score:     {r2:.4f}\n"
                            stats_text += f"  ✓ RMSE:         {rmse:.4f}\n"
                            stats_text += f"  ✓ MAE:          {mae:.4f}\n"
                            stats_text += f"  • Samples:      {len(y_true)}\n"
                            stats_text += f"  • Pred Mean:    {y_pred.mean():.4f}\n"
                            stats_text += f"  • Actual Mean:  {y_true.mean():.4f}\n"
                            stats_text += f"  • Pred Range:   [{y_pred.min():.4f}, {y_pred.max():.4f}]\n"
                            stats_text += f"  • Actual Range: [{y_true.min():.4f}, {y_true.max():.4f}]\n"
                        except Exception as e:
                            stats_text += f"  ⚠ Could not calculate validation metrics: {e}\n"
                            stats_text += f"  • Count:  {len(values)}\n"
                            stats_text += f"  • Mean:   {values.mean():.4f}\n"
                            stats_text += f"  • Std:    {values.std():.4f}\n"
                    else:
                        # Regular prediction statistics
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

        # Get initial directory from spectral data path
        initial_dir = None
        if self.spectral_data_path.get():
            data_path = Path(self.spectral_data_path.get())
            initial_dir = str(data_path.parent if data_path.is_file() else data_path)

        # Ask user for save location
        filepath = filedialog.asksaveasfilename(
            title="Export Predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename,
            initialdir=initial_dir
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
