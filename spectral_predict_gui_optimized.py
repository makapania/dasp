"""
Spectral Predict - Redesigned 5-Tab GUI Application (OPTIMIZED)

Tab 1: Import & Preview - Data loading + spectral plots
Tab 2: Analysis Configuration - All analysis settings
Tab 3: Analysis Progress - Live progress monitor
Tab 4: Results - Analysis results table (clickable to refine)
Tab 5: Refine Model - Interactive model refinement

OPTIMIZED VERSION:
- Neural Boosted max_iter reduced from 500 to 100 (Phase A optimization)
- Implements evidence-based optimizations for 2-3x speedup
"""

import sys
import os
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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


class SpectralPredictApp:
    """Main application window with 5-tab design."""

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

        # Model selection
        self.use_pls = tk.BooleanVar(value=True)
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
        """Create 5-tab user interface."""
        # Create notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self._create_tab1_import_preview()
        self._create_tab2_analysis_config()
        self._create_tab3_progress()
        self._create_tab4_results()
        self._create_tab5_refine_model()

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

    def _create_tab2_analysis_config(self):
        """Tab 2: Analysis Configuration - All analysis settings."""
        self.tab2 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab2, text='  ⚙️ Analysis Configuration  ')

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

        # === Model Selection ===
        ttk.Label(content_frame, text="Models to Test", style='Heading.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        models_frame = ttk.LabelFrame(content_frame, text="Select Models", padding="20")
        models_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        ttk.Checkbutton(models_frame, text="✓ PLS (Partial Least Squares)", variable=self.use_pls).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Linear, fast, interpretable", style='Caption.TLabel').grid(row=0, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ Random Forest", variable=self.use_randomforest).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Nonlinear, robust", style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ MLP (Multi-Layer Perceptron)", variable=self.use_mlp).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Deep learning", style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)

        ttk.Checkbutton(models_frame, text="✓ Neural Boosted", variable=self.use_neuralboosted).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Label(models_frame, text="Gradient boosting with NNs", style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, padx=15)

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

        # Run button
        ttk.Button(content_frame, text="▶ Run Analysis", command=self._run_analysis,
                  style='Accent.TButton').grid(row=row, column=0, columnspan=2, pady=40, ipadx=30, ipady=10)
        row += 1

        self.tab2_status = ttk.Label(content_frame, text="Configure analysis settings above", style='Caption.TLabel')
        self.tab2_status.grid(row=row, column=0, columnspan=2)

    def _create_tab3_progress(self):
        """Tab 3: Analysis Progress - Live progress monitor."""
        self.tab3 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab3, text='  📊 Analysis Progress  ')

        content_frame = ttk.Frame(self.tab3, style='TFrame', padding="30")
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

    def _create_tab4_results(self):
        """Tab 4: Results - Display analysis results in a table."""
        self.tab4 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab4, text='  📊 Results  ')

        content_frame = ttk.Frame(self.tab4, style='TFrame', padding="30")
        content_frame.pack(fill='both', expand=True)

        # Title
        ttk.Label(content_frame, text="Analysis Results", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 20))

        # Instructions
        instructions = ttk.Label(content_frame,
            text="Click on any result row to load it into the 'Refine Model' tab for further tuning.",
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

    def _create_tab5_refine_model(self):
        """Tab 5: Refine Model - Interactive model parameter refinement."""
        self.tab5 = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.tab5, text='  🔧 Refine Model  ')

        # Create scrollable content
        canvas = tk.Canvas(self.tab5, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab5, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas, style='TFrame', padding="30")

        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # Title
        ttk.Label(content_frame, text="Refine Model", style='Title.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        row += 1

        # Instructions
        ttk.Label(content_frame,
            text="Double-click a result from the Results tab to load it here for refinement.",
            style='Caption.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        row += 1

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

        # Wavelength range
        ttk.Label(params_frame, text="Wavelength Range (nm):", style='Subheading.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        wl_frame = ttk.Frame(params_frame)
        wl_frame.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.refine_wl_min = tk.StringVar()
        self.refine_wl_max = tk.StringVar()
        ttk.Entry(wl_frame, textvariable=self.refine_wl_min, width=12).grid(row=0, column=0, padx=5)
        ttk.Label(wl_frame, text="to").grid(row=0, column=1, padx=5)
        ttk.Entry(wl_frame, textvariable=self.refine_wl_max, width=12).grid(row=0, column=2, padx=5)

        # Window size (for derivatives)
        ttk.Label(params_frame, text="Window Size:", style='Subheading.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_window = tk.IntVar(value=17)
        window_frame = ttk.Frame(params_frame)
        window_frame.grid(row=3, column=0, sticky=tk.W, pady=5)
        for w in [7, 11, 17, 19]:
            ttk.Radiobutton(window_frame, text=f"{w}", variable=self.refine_window, value=w).pack(side='left', padx=5)

        # CV Folds
        ttk.Label(params_frame, text="CV Folds:", style='Subheading.TLabel').grid(row=4, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_folds = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=3, to=10, textvariable=self.refine_folds, width=12).grid(row=5, column=0, sticky=tk.W)

        # Max iterations (for neural models)
        ttk.Label(params_frame, text="Max Iterations:", style='Subheading.TLabel').grid(row=6, column=0, sticky=tk.W, pady=(15, 5))
        self.refine_max_iter = tk.IntVar(value=100)
        ttk.Spinbox(params_frame, from_=100, to=5000, increment=100, textvariable=self.refine_max_iter, width=12).grid(row=7, column=0, sticky=tk.W)

        # Run refined model button
        self.refine_run_button = ttk.Button(content_frame, text="▶ Run Refined Model", command=self._run_refined_model,
                  style='Accent.TButton', state='disabled')
        self.refine_run_button.grid(row=row, column=0, columnspan=2, pady=30, ipadx=30, ipady=10)
        row += 1

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
            # Enable the wavelength update button
            self.update_wl_button.config(state='normal')
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

    def _generate_plots(self):
        """Generate spectral plots in the plot notebook."""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Matplotlib Required", "Matplotlib is required for plotting")
            return

        # Clear existing plots
        for widget in self.plot_notebook.winfo_children():
            widget.destroy()

        # Create plot tabs
        self._create_plot_tab("Raw Spectra", self.X.values, "Reflectance", "blue")

        # Generate derivative plots if available
        if HAS_DERIVATIVES:
            # 1st derivative
            deriv1 = SavgolDerivative(deriv=1, window=7)
            X_deriv1 = deriv1.transform(self.X.values)
            self._create_plot_tab("1st Derivative", X_deriv1, "First Derivative", "green")

            # 2nd derivative
            deriv2 = SavgolDerivative(deriv=2, window=7)
            X_deriv2 = deriv2.transform(self.X.values)
            self._create_plot_tab("2nd Derivative", X_deriv2, "Second Derivative", "red")
        else:
            messagebox.showwarning("Derivatives Unavailable",
                "Could not import SavgolDerivative. Only raw spectra will be plotted.")

    def _create_plot_tab(self, title, data, ylabel, color):
        """Create a plot tab."""
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

        # Plot
        for i in indices:
            ax.plot(wavelengths, data[i, :], alpha=alpha, color=color, linewidth=1)

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} (n={n_samples})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if "Derivative" in title:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        if self.use_randomforest.get():
            selected_models.append("RandomForest")
        if self.use_mlp.get():
            selected_models.append("MLP")
        if self.use_neuralboosted.get():
            selected_models.append("NeuralBoosted")

        if not selected_models:
            messagebox.showwarning("No Models", "Please select at least one model to test")
            return

        # Switch to progress tab
        self.notebook.select(2)

        # Clear progress text
        self.progress_text.delete('1.0', tk.END)
        self.progress_info.config(text="Starting analysis...")
        self.progress_status.config(text="Analysis in progress...")
        self.best_model_info.config(text="(none yet)")
        self.time_estimate_label.config(text="")

        # Reset start time
        self.analysis_start_time = datetime.now()

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
            results_df = run_search(
                self.X,
                self.y,
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
                progress_callback=self._progress_callback
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
        model_config = self.results_df.iloc[row_idx].to_dict()
        self.selected_model_config = model_config

        # Populate the Refine Model tab
        self._load_model_for_refinement(model_config)

        # Switch to the Refine Model tab
        self.notebook.select(4)  # Tab 5 (index 4)

    def _load_model_for_refinement(self, config):
        """Load a model configuration into the Refine Model tab."""
        # Build configuration text
        info_text = f"""Model: {config.get('Model', 'N/A')}
Preprocessing: {config.get('Preprocess', 'N/A')}
Subset: {config.get('Subset', 'N/A')}
Window Size: {config.get('Window', 'N/A')}
"""

        # Add performance metrics
        if 'RMSE' in config:
            # Regression
            info_text += f"""
Performance (Regression):
  RMSE: {config.get('RMSE', 'N/A')}
  R²: {config.get('R2', 'N/A')}
  MAE: {config.get('MAE', 'N/A')}
"""
        else:
            # Classification
            info_text += f"""
Performance (Classification):
  Accuracy: {config.get('Accuracy', 'N/A')}
  Precision: {config.get('Precision', 'N/A')}
  Recall: {config.get('Recall', 'N/A')}
  F1 Score: {config.get('F1', 'N/A')}
"""
            if 'ROC_AUC' in config and not pd.isna(config['ROC_AUC']):
                info_text += f"  ROC AUC: {config.get('ROC_AUC', 'N/A')}\n"

        # Add top wavelengths if available
        if 'top_vars' in config and config['top_vars'] != 'N/A':
            info_text += f"\nTop Wavelengths: {config['top_vars']}\n"

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
        # Get wavelength range from current data
        if self.X is not None:
            wavelengths = self.X.columns.astype(float)
            self.refine_wl_min.set(str(int(wavelengths.min())))
            self.refine_wl_max.set(str(int(wavelengths.max())))

        # Set window size
        window = config.get('Window', 17)
        if not pd.isna(window) and window in [7, 11, 17, 19]:
            self.refine_window.set(int(window))

        # Enable the run button
        self.refine_run_button.config(state='normal')
        self.refine_status.config(text=f"Loaded: {config.get('Model', 'N/A')} | {config.get('Preprocess', 'N/A')}")

    def _run_refined_model(self):
        """Run the refined model with user-specified parameters."""
        if self.selected_model_config is None:
            messagebox.showwarning("No Model", "Please select a model from the Results tab first")
            return

        if self.X is None or self.y is None:
            messagebox.showwarning("No Data", "Please load data first")
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
            from spectral_predict.scoring import cross_validate_model
            from sklearn.model_selection import KFold

            config = self.selected_model_config

            # Apply wavelength filtering to get X_subset
            X_work = self.X_original.copy()
            wl_min = self.refine_wl_min.get().strip()
            wl_max = self.refine_wl_max.get().strip()

            if wl_min or wl_max:
                wavelengths = X_work.columns.astype(float)
                if wl_min:
                    X_work = X_work.loc[:, wavelengths >= float(wl_min)]
                if wl_max:
                    wavelengths = X_work.columns.astype(float)
                    X_work = X_work.loc[:, wavelengths <= float(wl_max)]

            # Apply preprocessing
            preprocess = config.get('Preprocess', 'raw')
            window = self.refine_window.get()

            if preprocess == 'snv':
                X_processed = SNV().transform(X_work.values)
            elif preprocess == 'sg1':
                X_processed = SavgolDerivative(deriv=1, window=window).transform(X_work.values)
            elif preprocess == 'sg2':
                X_processed = SavgolDerivative(deriv=2, window=window).transform(X_work.values)
            elif preprocess == 'deriv_snv':
                X_temp = SavgolDerivative(deriv=1, window=window).transform(X_work.values)
                X_processed = SNV().transform(X_temp)
            else:  # raw
                X_processed = X_work.values

            # Determine task type
            if self.y.nunique() == 2 or self.y.dtype == 'object' or self.y.nunique() < 10:
                task_type = "classification"
            else:
                task_type = "regression"

            # Get model
            model_name = config.get('Model', 'PLS')
            model = get_model(
                model_name,
                task_type=task_type,
                max_n_components=self.max_n_components.get(),
                max_iter=self.refine_max_iter.get()
            )

            # Run cross-validation
            cv = KFold(n_splits=self.refine_folds.get(), shuffle=True, random_state=42)
            results = cross_validate_model(model, X_processed, self.y.values, cv, task_type)

            # Format results
            if task_type == "regression":
                results_text = f"""Refined Model Results:

Cross-Validation Performance ({self.refine_folds.get()} folds):
  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}
  R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}
  MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}

Configuration:
  Model: {model_name}
  Preprocessing: {preprocess}
  Window Size: {window}
  Wavelength Range: {wl_min or 'full'} - {wl_max or 'full'} nm
  Features: {X_processed.shape[1]}
  Samples: {X_processed.shape[0]}
  CV Folds: {self.refine_folds.get()}
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
  Preprocessing: {preprocess}
  Window Size: {window}
  Wavelength Range: {wl_min or 'full'} - {wl_max or 'full'} nm
  Features: {X_processed.shape[1]}
  Samples: {X_processed.shape[0]}
  CV Folds: {self.refine_folds.get()}
"""

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

        # Re-enable button
        self.refine_run_button.config(state='normal')

        if is_error:
            self.refine_status.config(text="✗ Error running refined model")
            messagebox.showerror("Error", "Failed to run refined model. See results area for details.")
        else:
            self.refine_status.config(text="✓ Refined model complete")
            messagebox.showinfo("Success", "Refined model analysis complete!")


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
