"""
Spectral Predict - REDESIGNED GUI v2 with Sidebar Navigation
Purple/Magenta "Lush Nexus" Theme - Better text visibility and space usage

Features:
- Left sidebar navigation (like dashboard reference)
- Purple/magenta color scheme
- High contrast text (all visible)
- Grid layout using full width
- All content visible without scrolling
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


class SpectralNexusApp:
    """Main application with sidebar navigation."""

    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Nexus - Next Generation Analysis")

        # Set window size
        try:
            self.root.state('zoomed')
        except:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = int(screen_width * 0.95)
            window_height = int(screen_height * 0.95)
            self.root.geometry(f"{window_width}x{window_height}")

        # Configure color scheme
        self._configure_colors()

        # Data variables
        self.X = None
        self.y = None
        self.ref = None

        # GUI variables
        self.spectral_data_path = tk.StringVar()
        self.detected_type = None
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

        # Plotting
        self.plot_frames = {}
        self.plot_canvases = {}

        # Current view
        self.current_view = "import"

        self._create_ui()

    def _configure_colors(self):
        """Configure purple/magenta lush color scheme."""
        self.colors = {
            # Backgrounds - rich purples
            'sidebar': '#1A0033',          # Deep purple-black
            'sidebar_hover': '#2D1B4E',    # Purple on hover
            'main_bg': '#0F0A1F',          # Very dark purple
            'card_bg': '#1C1535',          # Dark purple card
            'input_bg': '#2A1F47',         # Input background

            # Accents - vibrant
            'accent_pink': '#FF00FF',      # Hot magenta
            'accent_purple': '#B794F6',    # Soft purple
            'accent_cyan': '#00F5FF',      # Electric cyan
            'accent_gold': '#FFD700',      # Gold

            # Text - high contrast
            'text_white': '#FFFFFF',       # Pure white
            'text_light': '#E0E0E0',       # Light gray
            'text_dim': '#A0A0A0',         # Dimmed gray
            'text_purple': '#D4BBFF',      # Light purple

            # Status
            'success': '#00FF88',          # Bright green
            'warning': '#FFB800',          # Amber
            'error': '#FF4466',            # Red
            'info': '#64FFDA',             # Cyan

            # Borders
            'border': '#4A3A6A',           # Purple border
            'glow': '#FF00FF',             # Magenta glow
        }

        self.root.configure(bg=self.colors['main_bg'])

    def _create_ui(self):
        """Create sidebar + main content area."""
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['main_bg'])
        main_container.pack(fill='both', expand=True)

        # === LEFT SIDEBAR ===
        self.sidebar = tk.Frame(main_container,
            bg=self.colors['sidebar'],
            width=240)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        # Logo/Title
        logo_frame = tk.Frame(self.sidebar, bg=self.colors['sidebar'])
        logo_frame.pack(fill='x', pady=(30, 40))

        logo_text = tk.Label(logo_frame,
            text="Spectral\nNexus",
            font=('Segoe UI', 22, 'bold'),
            bg=self.colors['sidebar'],
            fg=self.colors['accent_pink'],
            justify='center')
        logo_text.pack()

        # Navigation buttons
        self.nav_buttons = {}

        nav_items = [
            ("📁", "IMPORT", "import"),
            ("⚙️", "CONFIG", "config"),
            ("📊", "PROGRESS", "progress")
        ]

        for icon, label, view_id in nav_items:
            btn = self._create_nav_button(icon, label, view_id)
            self.nav_buttons[view_id] = btn

        # Highlight first button
        self._highlight_nav_button("import")

        # Bottom section
        bottom_frame = tk.Frame(self.sidebar, bg=self.colors['sidebar'])
        bottom_frame.pack(side='bottom', fill='x', pady=30)

        help_label = tk.Label(bottom_frame,
            text="SPECTRAL PREDICT\nv2.0",
            font=('Segoe UI', 8),
            bg=self.colors['sidebar'],
            fg=self.colors['text_dim'],
            justify='center')
        help_label.pack()

        # === RIGHT CONTENT AREA ===
        self.content_area = tk.Frame(main_container, bg=self.colors['main_bg'])
        self.content_area.pack(side='right', fill='both', expand=True)

        # Create different views (only show one at a time)
        self.views = {}
        self._create_import_view()
        self._create_config_view()
        self._create_progress_view()

        # Show initial view
        self._show_view("import")

    def _create_nav_button(self, icon, label, view_id):
        """Create a sidebar navigation button."""
        btn_frame = tk.Frame(self.sidebar, bg=self.colors['sidebar'])
        btn_frame.pack(fill='x', pady=2)

        btn = tk.Button(btn_frame,
            text=f"{icon}  {label}",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['sidebar'],
            fg=self.colors['text_light'],
            activebackground=self.colors['sidebar_hover'],
            activeforeground=self.colors['accent_pink'],
            relief='flat',
            anchor='w',
            padx=30,
            pady=15,
            cursor='hand2',
            command=lambda: self._switch_view(view_id))
        btn.pack(fill='x')

        # Hover effects
        btn.bind('<Enter>', lambda e: btn.config(
            bg=self.colors['sidebar_hover'],
            fg=self.colors['accent_pink']) if view_id != self.current_view else None)
        btn.bind('<Leave>', lambda e: btn.config(
            bg=self.colors['sidebar'],
            fg=self.colors['text_light']) if view_id != self.current_view else None)

        return btn

    def _highlight_nav_button(self, view_id):
        """Highlight the active navigation button."""
        for vid, btn in self.nav_buttons.items():
            if vid == view_id:
                btn.config(bg=self.colors['sidebar_hover'], fg=self.colors['accent_pink'])
            else:
                btn.config(bg=self.colors['sidebar'], fg=self.colors['text_light'])

    def _switch_view(self, view_id):
        """Switch between different views."""
        self.current_view = view_id
        self._highlight_nav_button(view_id)
        self._show_view(view_id)

    def _show_view(self, view_id):
        """Show specific view, hide others."""
        for vid, view_frame in self.views.items():
            if vid == view_id:
                view_frame.pack(fill='both', expand=True)
            else:
                view_frame.pack_forget()

    def _create_import_view(self):
        """Create the Import & Preview view."""
        view = tk.Frame(self.content_area, bg=self.colors['main_bg'])
        self.views["import"] = view

        # Scrollable canvas
        canvas = tk.Canvas(view, bg=self.colors['main_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(view, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=self.colors['main_bg'])

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === HEADER ===
        header = tk.Frame(scroll_frame, bg=self.colors['main_bg'])
        header.pack(fill='x', padx=40, pady=(30, 20))

        title = tk.Label(header,
            text="Import & Preview",
            font=('Segoe UI', 32, 'bold'),
            bg=self.colors['main_bg'],
            fg=self.colors['text_white'])
        title.pack(anchor='w')

        subtitle = tk.Label(header,
            text="Load spectral data and visualize transformations",
            font=('Segoe UI', 13),
            bg=self.colors['main_bg'],
            fg=self.colors['text_dim'])
        subtitle.pack(anchor='w', pady=(8, 0))

        # === GRID LAYOUT FOR CARDS ===
        grid_container = tk.Frame(scroll_frame, bg=self.colors['main_bg'])
        grid_container.pack(fill='both', expand=True, padx=40, pady=20)

        # Configure grid - 2 columns
        grid_container.columnconfigure(0, weight=1)
        grid_container.columnconfigure(1, weight=1)

        # ROW 1: Input Files (left) + Column Config (right)
        input_card = self._create_card(grid_container, "📁 Input Files")
        input_card.grid(row=0, column=0, sticky='nsew', padx=(0, 15), pady=(0, 20))

        config_card = self._create_card(grid_container, "🔧 Column Configuration")
        config_card.grid(row=0, column=1, sticky='nsew', padx=(15, 0), pady=(0, 20))

        # ROW 2: Wavelength Range (full width)
        wl_card = self._create_card(grid_container, "🌈 Wavelength Range")
        wl_card.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(0, 20))

        # === POPULATE INPUT CARD ===
        self._add_file_input(input_card, "Spectral File Directory", self.spectral_data_path,
            self._browse_spectral_data, 0)

        # Status
        status_frame = tk.Frame(input_card, bg=self.colors['card_bg'])
        status_frame.grid(row=1, column=0, columnspan=2, sticky='w', padx=20, pady=(5, 15))

        self.detection_icon = tk.Label(status_frame,
            text="⚡",
            font=('Segoe UI', 14),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dim'])
        self.detection_icon.pack(side='left', padx=(0, 10))

        self.detection_status = tk.Label(status_frame,
            text="No data loaded",
            font=('Segoe UI', 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dim'])
        self.detection_status.pack(side='left')

        self._add_file_input(input_card, "Reference CSV", self.reference_file,
            self._browse_reference_file, 2)

        # === POPULATE CONFIG CARD ===
        self._add_dropdown(config_card, "Spectral File Column", self.spectral_file_column, 0)
        self.spectral_file_combo = self.last_combo

        self._add_dropdown(config_card, "Specimen ID Column", self.id_column, 1)
        self.id_combo = self.last_combo

        self._add_dropdown(config_card, "Target Variable Column", self.target_column, 2)
        self.target_combo = self.last_combo

        # Auto-detect button
        auto_btn = tk.Button(config_card,
            text="🔍 Auto-Detect Columns",
            command=self._auto_detect_columns,
            bg=self.colors['accent_purple'],
            fg=self.colors['text_white'],
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            pady=12,
            cursor='hand2')
        auto_btn.grid(row=3, column=0, columnspan=2, pady=(15, 20))

        # === POPULATE WAVELENGTH CARD ===
        wl_inner = tk.Frame(wl_card, bg=self.colors['card_bg'])
        wl_inner.grid(row=0, column=0, padx=20, pady=20)

        tk.Label(wl_inner,
            text="Select Range (nm):",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_white']).pack(side='left', padx=(0, 20))

        min_entry = tk.Entry(wl_inner,
            textvariable=self.wavelength_min,
            font=('Segoe UI', 13),
            bg=self.colors['input_bg'],
            fg=self.colors['text_white'],
            insertbackground=self.colors['accent_cyan'],
            relief='flat',
            width=12,
            justify='center')
        min_entry.pack(side='left', ipady=8, padx=5)

        tk.Label(wl_inner,
            text="to",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dim']).pack(side='left', padx=15)

        max_entry = tk.Entry(wl_inner,
            textvariable=self.wavelength_max,
            font=('Segoe UI', 13),
            bg=self.colors['input_bg'],
            fg=self.colors['text_white'],
            insertbackground=self.colors['accent_cyan'],
            relief='flat',
            width=12,
            justify='center')
        max_entry.pack(side='left', ipady=8, padx=5)

        tk.Label(wl_inner,
            text="(auto-fills after load)",
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dim']).pack(side='left', padx=20)

        # === LOAD BUTTON ===
        btn_frame = tk.Frame(scroll_frame, bg=self.colors['main_bg'])
        btn_frame.pack(fill='x', padx=40, pady=(10, 20))

        load_btn = tk.Button(btn_frame,
            text="📊 Load Data & Generate Plots",
            command=self._load_and_plot_data,
            bg=self.colors['accent_pink'],
            fg=self.colors['text_white'],
            font=('Segoe UI', 15, 'bold'),
            relief='flat',
            pady=18,
            cursor='hand2')
        load_btn.pack(fill='x')

        load_btn.bind('<Enter>', lambda e: load_btn.config(bg=self.colors['glow']))
        load_btn.bind('<Leave>', lambda e: load_btn.config(bg=self.colors['accent_pink']))

        self.tab1_status = tk.Label(btn_frame,
            text="Ready to load spectral data",
            font=('Segoe UI', 11),
            bg=self.colors['main_bg'],
            fg=self.colors['text_dim'])
        self.tab1_status.pack(pady=(12, 0))

        # === PLOTS ===
        plot_header = tk.Frame(scroll_frame, bg=self.colors['main_bg'])
        plot_header.pack(fill='x', padx=40, pady=(30, 15))

        tk.Label(plot_header,
            text="Spectral Visualizations",
            font=('Segoe UI', 24, 'bold'),
            bg=self.colors['main_bg'],
            fg=self.colors['text_white']).pack(anchor='w')

        # Plot area
        self.plot_container = tk.Frame(scroll_frame, bg=self.colors['card_bg'])
        self.plot_container.pack(fill='both', expand=True, padx=40, pady=(0, 40))

        placeholder = tk.Label(self.plot_container,
            text="🌌\n\nLoad data to see spectral plots",
            font=('Segoe UI', 16),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dim'],
            justify='center')
        placeholder.pack(expand=True, pady=100)

    def _create_config_view(self):
        """Create configuration view."""
        view = tk.Frame(self.content_area, bg=self.colors['main_bg'])
        self.views["config"] = view

        label = tk.Label(view,
            text="Analysis Configuration\n\n(Coming soon)",
            font=('Segoe UI', 20, 'bold'),
            bg=self.colors['main_bg'],
            fg=self.colors['text_white'],
            justify='center')
        label.pack(expand=True)

    def _create_progress_view(self):
        """Create progress view."""
        view = tk.Frame(self.content_area, bg=self.colors['main_bg'])
        self.views["progress"] = view

        label = tk.Label(view,
            text="Analysis Progress\n\n(Coming soon)",
            font=('Segoe UI', 20, 'bold'),
            bg=self.colors['main_bg'],
            fg=self.colors['text_white'],
            justify='center')
        label.pack(expand=True)

    def _create_card(self, parent, title):
        """Create a card container."""
        # Outer frame for border
        outer = tk.Frame(parent,
            bg=self.colors['border'],
            highlightthickness=0)

        # Inner card
        card = tk.Frame(outer,
            bg=self.colors['card_bg'],
            highlightthickness=0)
        card.pack(fill='both', expand=True, padx=2, pady=2)

        # Title
        title_label = tk.Label(card,
            text=title,
            font=('Segoe UI', 15, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['accent_cyan'])
        title_label.grid(row=0, column=0, columnspan=2, sticky='w', padx=20, pady=(20, 15))

        # Configure grid
        card.columnconfigure(0, weight=1)

        return card

    def _add_file_input(self, card, label_text, var, command, row_offset):
        """Add file input field to card."""
        # Get next available row
        row = len(card.grid_slaves()) + row_offset

        # Label
        tk.Label(card,
            text=label_text,
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_white']).grid(
                row=row, column=0, columnspan=2, sticky='w', padx=20, pady=(10, 5))

        # Entry + Button frame
        frame = tk.Frame(card, bg=self.colors['card_bg'])
        frame.grid(row=row+1, column=0, columnspan=2, sticky='ew', padx=20, pady=(0, 15))
        frame.columnconfigure(0, weight=1)

        entry = tk.Entry(frame,
            textvariable=var,
            font=('Segoe UI', 11),
            bg=self.colors['input_bg'],
            fg=self.colors['text_white'],
            insertbackground=self.colors['accent_cyan'],
            relief='flat')
        entry.pack(side='left', fill='x', expand=True, ipady=10, padx=(0, 10))

        btn = tk.Button(frame,
            text="Browse...",
            command=command,
            bg=self.colors['border'],
            fg=self.colors['text_white'],
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2')
        btn.pack(side='right')

        btn.bind('<Enter>', lambda e: btn.config(bg=self.colors['accent_cyan']))
        btn.bind('<Leave>', lambda e: btn.config(bg=self.colors['border']))

    def _add_dropdown(self, card, label_text, var, index):
        """Add dropdown to card."""
        row = index * 2

        tk.Label(card,
            text=label_text,
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_white']).grid(
                row=row, column=0, columnspan=2, sticky='w', padx=20, pady=(10, 5))

        # Custom styled combobox
        combo_frame = tk.Frame(card, bg=self.colors['input_bg'])
        combo_frame.grid(row=row+1, column=0, columnspan=2, sticky='ew', padx=20, pady=(0, 15))

        combo = ttk.Combobox(combo_frame,
            textvariable=var,
            font=('Segoe UI', 11),
            state='readonly')
        combo.pack(fill='x', ipady=8, padx=2, pady=2)

        self.last_combo = combo

    # === ORIGINAL FUNCTIONALITY ===

    def _browse_spectral_data(self):
        """Browse for spectral data."""
        directory = filedialog.askdirectory(title="Select Spectral Data Directory")
        if not directory:
            return

        self.spectral_data_path.set(directory)
        path = Path(directory)

        # Check for ASD
        asd_files = list(path.glob("*.asd"))
        if asd_files:
            self.detected_type = "asd"
            self.detection_icon.config(text="✓", fg=self.colors['success'])
            self.detection_status.config(
                text=f"Detected {len(asd_files)} ASD files",
                fg=self.colors['success'])

            csv_files = list(path.glob("*.csv"))
            if len(csv_files) == 1:
                self.reference_file.set(str(csv_files[0]))
                self._auto_detect_columns()
            return

        # Check for CSV
        csv_files = list(path.glob("*.csv"))
        if csv_files:
            if len(csv_files) == 1:
                self.spectral_data_path.set(str(csv_files[0]))
                self.detected_type = "csv"
                self.detection_icon.config(text="✓", fg=self.colors['success'])
                self.detection_status.config(
                    text="Detected CSV spectra file",
                    fg=self.colors['success'])
            else:
                self.detected_type = "csv"
                self.detection_icon.config(text="⚠", fg=self.colors['warning'])
                self.detection_status.config(
                    text=f"Found {len(csv_files)} CSV files - select manually",
                    fg=self.colors['warning'])
            return

        # Not found
        self.detected_type = None
        self.detection_icon.config(text="✗", fg=self.colors['error'])
        self.detection_status.config(
            text="No supported files found",
            fg=self.colors['error'])

    def _browse_reference_file(self):
        """Browse for reference CSV."""
        filename = filedialog.askopenfilename(
            title="Select Reference CSV",
            filetypes=[("CSV files", "*.csv")])
        if filename:
            self.reference_file.set(filename)
            self._auto_detect_columns()

    def _auto_detect_columns(self):
        """Auto-detect columns."""
        if not self.reference_file.get():
            return

        try:
            df = pd.read_csv(self.reference_file.get(), nrows=5)
            columns = list(df.columns)

            self.spectral_file_combo['values'] = columns
            self.id_combo['values'] = columns
            self.target_combo['values'] = columns

            if len(columns) >= 3:
                self.spectral_file_column.set(columns[0])
                self.id_column.set(columns[1])
                self.target_column.set(columns[2])

            self.tab1_status.config(
                text=f"✓ Detected {len(columns)} columns",
                fg=self.colors['success'])
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")

    def _load_and_plot_data(self):
        """Load and plot data."""
        try:
            from spectral_predict.io import read_csv_spectra, read_reference_csv, align_xy, read_asd_dir

            self.tab1_status.config(text="Loading data...", fg=self.colors['info'])
            self.root.update()

            if not self.spectral_data_path.get() or not self.detected_type:
                messagebox.showwarning("Missing Input", "Please select spectral data")
                return

            # Load spectral
            if self.detected_type == "asd":
                X = read_asd_dir(self.spectral_data_path.get())
            elif self.detected_type == "csv":
                X = read_csv_spectra(self.spectral_data_path.get())
            else:
                messagebox.showerror("Error", "Unsupported format")
                return

            if not self.reference_file.get():
                messagebox.showwarning("Missing Input", "Please select reference CSV")
                return

            ref = read_reference_csv(self.reference_file.get(), self.spectral_file_column.get())
            X_aligned, y_aligned = align_xy(X, ref, self.spectral_file_column.get(), self.target_column.get())

            self.X = X_aligned
            self.y = y_aligned
            self.ref = ref

            # Wavelength range
            wavelengths = self.X.columns.astype(float)
            self.wavelength_min.set(str(int(wavelengths.min())))
            self.wavelength_max.set(str(int(wavelengths.max())))

            # Filter
            wl_min = self.wavelength_min.get().strip()
            wl_max = self.wavelength_max.get().strip()
            if wl_min or wl_max:
                wavelengths = self.X.columns.astype(float)
                if wl_min:
                    self.X = self.X.loc[:, wavelengths >= float(wl_min)]
                if wl_max:
                    wavelengths = self.X.columns.astype(float)
                    self.X = self.X.loc[:, wavelengths <= float(wl_max)]

            self._generate_plots()

            self.tab1_status.config(
                text=f"✓ Loaded {len(self.X)} samples × {self.X.shape[1]} wavelengths",
                fg=self.colors['success'])

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load:\n{e}")
            self.tab1_status.config(text="✗ Error", fg=self.colors['error'])

    def _generate_plots(self):
        """Generate plots."""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Matplotlib Required", "Install matplotlib for plots")
            return

        # Clear container
        for widget in self.plot_container.winfo_children():
            widget.destroy()

        # Create notebook for plots
        notebook = ttk.Notebook(self.plot_container)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        plt.style.use('dark_background')

        # Raw
        self._create_plot_tab(notebook, "Raw Spectra", self.X.values, "Reflectance", '#FF00FF')

        # Derivatives
        if HAS_DERIVATIVES:
            deriv1 = SavgolDerivative(deriv=1, window=7)
            X_deriv1 = deriv1.transform(self.X.values)
            self._create_plot_tab(notebook, "1st Derivative", X_deriv1, "First Derivative", '#B794F6')

            deriv2 = SavgolDerivative(deriv=2, window=7)
            X_deriv2 = deriv2.transform(self.X.values)
            self._create_plot_tab(notebook, "2nd Derivative", X_deriv2, "Second Derivative", '#00F5FF')

    def _create_plot_tab(self, notebook, title, data, ylabel, color):
        """Create plot tab."""
        frame = tk.Frame(notebook, bg=self.colors['card_bg'])
        notebook.add(frame, text=f"  {title}  ")

        fig = Figure(figsize=(16, 6), facecolor=self.colors['card_bg'])
        ax = fig.add_subplot(111, facecolor=self.colors['main_bg'])

        wavelengths = self.X.columns.values
        n_samples = len(data)

        if n_samples <= 50:
            alpha = 0.5
            indices = range(n_samples)
        else:
            alpha = 0.7
            indices = np.random.choice(n_samples, size=50, replace=False)

        for i in indices:
            ax.plot(wavelengths, data[i, :], alpha=alpha, color=color, linewidth=1.5)

        ax.set_xlabel('Wavelength (nm)', fontsize=14, color='white', fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, color='white', fontweight='bold')
        ax.set_title(f'{title} (n={n_samples})', fontsize=18, fontweight='bold',
                    color='white', pad=15)

        ax.grid(True, alpha=0.15, color='#666666', linestyle='--')
        ax.tick_params(colors='white', which='both', labelsize=11)

        for spine in ax.spines.values():
            spine.set_edgecolor(self.colors['border'])
            spine.set_linewidth(2)

        if "Derivative" in title:
            ax.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SpectralNexusApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
