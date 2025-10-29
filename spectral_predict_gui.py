"""
Spectral Predict - Redesigned 3-Tab GUI Application

Tab 1: Import & Preview - Data loading + spectral plots
Tab 2: Analysis Configuration - All analysis settings
Tab 3: Analysis Progress - Live progress monitor
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
    """Main application window with 3-tab design."""

    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Predict - Automated Spectral Analysis")

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
        self.max_iter = tk.IntVar(value=500)
        self.show_progress = tk.BooleanVar(value=True)

        # Model selection
        self.use_pls = tk.BooleanVar(value=True)
        self.use_randomforest = tk.BooleanVar(value=True)
        self.use_mlp = tk.BooleanVar(value=True)
        self.use_neuralboosted = tk.BooleanVar(value=True)

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
        """Create 3-tab user interface."""
        # Create notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self._create_tab1_import_preview()
        self._create_tab2_analysis_config()
        self._create_tab3_progress()

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

            self._log_progress(f"Task type: {task_type}")
            self._log_progress(f"Models: {', '.join(selected_models)}")
            self._log_progress(f"Data: {len(self.X)} samples × {self.X.shape[1]} wavelengths\n")

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

            self._log_progress(f"\n✓ Analysis complete!")
            self._log_progress(f"Results saved to: {results_path}")

            self.root.after(0, lambda: self.progress_status.config(text="✓ Analysis complete!"))
            self.root.after(0, lambda: self.progress_info.config(text="Analysis Complete"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Analysis complete!\n\nResults: {results_path}"))

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
"""
        messagebox.showinfo("Help", help_text)


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
