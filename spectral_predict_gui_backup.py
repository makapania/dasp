"""
Spectral Predict - Standalone GUI Application

Double-click this file to launch the application.
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
from datetime import datetime

# Check for required dependencies before proceeding
def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        error_msg = (
            f"ERROR: Missing required packages: {', '.join(missing)}\n\n"
            f"Please use the launcher script to run the GUI:\n\n"
            f"  Unix/Mac/Linux:  ./run_gui.sh\n"
            f"  Windows:         run_gui.bat\n\n"
            f"Or install manually:\n"
            f"  pip install {' '.join(missing)}\n\n"
            f"Or install with virtual environment:\n"
            f"  python3 -m venv .venv\n"
            f"  source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows\n"
            f"  pip install -e ."
        )
        # Try to show GUI error if tkinter works
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing Dependencies", error_msg)
            root.destroy()
        except:
            pass
        # Always print to console
        print("\n" + "="*70)
        print(error_msg)
        print("="*70 + "\n")
        sys.exit(1)

# Run dependency check
check_dependencies()

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


class SpectralPredictApp:
    """Main application window for Spectral Predict."""

    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Predict - Automated Spectral Analysis")
        self.root.geometry("850x1050")  # Increased height for model selection section

        # Variables
        self.input_type = tk.StringVar(value="asd")
        self.asd_dir = tk.StringVar()
        self.spectra_file = tk.StringVar()
        self.reference_file = tk.StringVar()
        self.spectral_file_column = tk.StringVar()  # NEW: For linking to spectral files
        self.id_column = tk.StringVar()  # Specimen ID (for tracking only)
        self.target_column = tk.StringVar()
        self.output_dir = tk.StringVar(value="outputs")
        self.folds = tk.IntVar(value=5)
        self.lambda_penalty = tk.DoubleVar(value=0.15)
        self.max_n_components = tk.IntVar(value=24)  # Maximum PLS components
        self.max_iter = tk.IntVar(value=500)  # Maximum MLP iterations
        self.wavelength_min = tk.StringVar(value="")  # Min wavelength (auto-populate)
        self.wavelength_max = tk.StringVar(value="")  # Max wavelength (auto-populate)
        self.use_gui = tk.BooleanVar(value=True)
        self.show_progress = tk.BooleanVar(value=True)  # NEW: Show progress monitor

        # Model selection (all enabled by default)
        self.use_pls = tk.BooleanVar(value=True)
        self.use_randomforest = tk.BooleanVar(value=True)
        self.use_mlp = tk.BooleanVar(value=True)
        self.use_neuralboosted = tk.BooleanVar(value=True)

        # Progress monitor
        self.progress_monitor = None
        self.analysis_thread = None

        self._create_ui()

    def _create_ui(self):
        """Create the user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title
        title = ttk.Label(main_frame, text="Spectral Predict",
                         font=("Arial", 20, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        subtitle = ttk.Label(main_frame,
                            text="Automated spectral analysis with preprocessing and model ranking",
                            font=("Arial", 10))
        subtitle.grid(row=1, column=0, columnspan=3, pady=(0, 5))

        models_info = ttk.Label(main_frame,
                               text="Models tested: PLS, Random Forest, MLP, Neural Boosted",
                               font=("Arial", 9, "italic"),
                               foreground="darkblue")
        models_info.grid(row=2, column=0, columnspan=3, pady=(0, 15))

        current_row = 3

        # === INPUT DATA SECTION ===
        section_label = ttk.Label(main_frame, text="1. Input Data",
                                 font=("Arial", 12, "bold"))
        section_label.grid(row=current_row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        current_row += 1

        # Input type selection
        input_frame = ttk.LabelFrame(main_frame, text="Spectral Data Type", padding="10")
        input_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1

        ttk.Radiobutton(input_frame, text="ASD files (directory)",
                       variable=self.input_type, value="asd",
                       command=self._update_input_fields).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(input_frame, text="CSV file (wide format)",
                       variable=self.input_type, value="csv",
                       command=self._update_input_fields).grid(row=0, column=1, sticky=tk.W)

        # ASD directory selection
        self.asd_frame = ttk.Frame(main_frame)
        self.asd_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(self.asd_frame, text="ASD Directory:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(self.asd_frame, textvariable=self.asd_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.asd_frame, text="Browse...",
                  command=self._browse_asd_dir).grid(row=0, column=2)
        current_row += 1

        # CSV file selection
        self.csv_frame = ttk.Frame(main_frame)
        self.csv_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(self.csv_frame, text="Spectra CSV:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(self.csv_frame, textvariable=self.spectra_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.csv_frame, text="Browse...",
                  command=self._browse_spectra_file).grid(row=0, column=2)
        current_row += 1

        # CSV ID column (only for CSV input)
        self.csv_id_frame = ttk.Frame(main_frame)
        self.csv_id_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(self.csv_id_frame, text="  CSV ID Column:").grid(row=0, column=0, sticky=tk.W)
        self.csv_id_var = tk.StringVar()
        self.csv_id_combo = ttk.Combobox(self.csv_id_frame, textvariable=self.csv_id_var, width=30)
        self.csv_id_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(self.csv_id_frame, text="(First column in spectra CSV)",
                 font=("Arial", 8, "italic")).grid(row=0, column=2, sticky=tk.W)
        current_row += 1

        # Reference file
        ttk.Label(main_frame, text="Reference CSV:").grid(row=current_row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.reference_file, width=50).grid(row=current_row, column=1, padx=5)
        ttk.Button(main_frame, text="Browse...",
                  command=self._browse_reference_file).grid(row=current_row, column=2)
        current_row += 1

        # Auto-detect button
        self.detect_button = ttk.Button(main_frame, text="🔍 Auto-Detect Columns",
                                       command=self._auto_detect_columns)
        self.detect_button.grid(row=current_row, column=1, pady=5)
        current_row += 1

        # === COLUMN SPECIFICATION ===
        section_label = ttk.Label(main_frame, text="2. Column Names (from Reference CSV)",
                                 font=("Arial", 12, "bold"))
        section_label.grid(row=current_row, column=0, columnspan=3, sticky=tk.W, pady=(15, 5))
        current_row += 1

        # Explanatory text
        explanation = ttk.Label(main_frame,
                               text="Your reference CSV should contain these three types of columns:",
                               font=("Arial", 9))
        explanation.grid(row=current_row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        current_row += 1

        # Column 1: Spectral File Column
        ttk.Label(main_frame, text="1. Spectral File Column:").grid(row=current_row, column=0, sticky=tk.W, pady=5)
        self.spectral_file_combo = ttk.Combobox(main_frame, textvariable=self.spectral_file_column, width=30)
        self.spectral_file_combo.grid(row=current_row, column=1, sticky=tk.W, padx=5)
        ttk.Label(main_frame, text='(e.g., "spectrum1", "spectrum5")',
                 font=("Arial", 8, "italic")).grid(row=current_row, column=2, sticky=tk.W)
        current_row += 1
        ttk.Label(main_frame, text='Links to spectral data files - CRITICAL for matching data',
                 font=("Arial", 8, "italic"), foreground="blue").grid(row=current_row, column=1, columnspan=2, sticky=tk.W, padx=5)
        current_row += 1

        # Column 2: Specimen ID Column
        ttk.Label(main_frame, text="2. Specimen ID Column:").grid(row=current_row, column=0, sticky=tk.W, pady=5)
        self.id_combo = ttk.Combobox(main_frame, textvariable=self.id_column, width=30)
        self.id_combo.grid(row=current_row, column=1, sticky=tk.W, padx=5)
        ttk.Label(main_frame, text='(e.g., "grass", "bone3")',
                 font=("Arial", 8, "italic")).grid(row=current_row, column=2, sticky=tk.W)
        current_row += 1
        ttk.Label(main_frame, text='For tracking only - NOT used in analysis',
                 font=("Arial", 8, "italic"), foreground="gray").grid(row=current_row, column=1, columnspan=2, sticky=tk.W, padx=5)
        current_row += 1

        # Column 3: Target Variable Column
        ttk.Label(main_frame, text="3. Target Variable Column:").grid(row=current_row, column=0, sticky=tk.W, pady=5)
        self.target_combo = ttk.Combobox(main_frame, textvariable=self.target_column, width=30)
        self.target_combo.grid(row=current_row, column=1, sticky=tk.W, padx=5)
        ttk.Label(main_frame, text='(e.g., "%N", "%collagen")',
                 font=("Arial", 8, "italic")).grid(row=current_row, column=2, sticky=tk.W)
        current_row += 1
        ttk.Label(main_frame, text='CRITICAL: This is what the models will learn to predict',
                 font=("Arial", 8, "italic"), foreground="blue").grid(row=current_row, column=1, columnspan=2, sticky=tk.W, padx=5)
        current_row += 1

        # === OPTIONS ===
        section_label = ttk.Label(main_frame, text="3. Analysis Options",
                                 font=("Arial", 12, "bold"))
        section_label.grid(row=current_row, column=0, columnspan=3, sticky=tk.W, pady=(15, 5))
        current_row += 1

        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1

        # CV Folds
        ttk.Label(options_frame, text="CV Folds:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(options_frame, from_=3, to=10, textvariable=self.folds,
                   width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        # Lambda penalty
        ttk.Label(options_frame, text="Complexity Penalty:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(options_frame, textvariable=self.lambda_penalty,
                 width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(options_frame, text="(higher = prefer simpler models)",
                 font=("Arial", 8, "italic")).grid(row=1, column=2, sticky=tk.W)

        # Max PLS components
        ttk.Label(options_frame, text="Max Latent Variables:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(options_frame, from_=2, to=100, textvariable=self.max_n_components,
                   width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Label(options_frame, text="(PLS components to test)",
                 font=("Arial", 8, "italic")).grid(row=2, column=2, sticky=tk.W)

        # Max iterations
        ttk.Label(options_frame, text="Max Iterations:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(options_frame, from_=100, to=5000, increment=100, textvariable=self.max_iter,
                   width=10).grid(row=3, column=1, sticky=tk.W, padx=5)
        ttk.Label(options_frame, text="(for MLP and Neural Boosted)",
                 font=("Arial", 8, "italic")).grid(row=3, column=2, sticky=tk.W)

        # Wavelength range
        ttk.Label(options_frame, text="Wavelength Range (nm):").grid(row=4, column=0, sticky=tk.W, pady=5)
        wavelength_frame = ttk.Frame(options_frame)
        wavelength_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Entry(wavelength_frame, textvariable=self.wavelength_min, width=8).grid(row=0, column=0, padx=2)
        ttk.Label(wavelength_frame, text="to").grid(row=0, column=1, padx=2)
        ttk.Entry(wavelength_frame, textvariable=self.wavelength_max, width=8).grid(row=0, column=2, padx=2)
        ttk.Label(wavelength_frame, text="(auto-fills after data load)",
                 font=("Arial", 8, "italic")).grid(row=0, column=3, padx=5)

        # Output directory
        ttk.Label(options_frame, text="Output Directory:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(options_frame, textvariable=self.output_dir,
                 width=20).grid(row=5, column=1, sticky=tk.W, padx=5)

        # Interactive mode
        ttk.Checkbutton(options_frame, text="Show interactive data preview (GUI)",
                       variable=self.use_gui).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Progress monitor option
        ttk.Checkbutton(options_frame, text="Show live progress monitor",
                       variable=self.show_progress).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)

        # === MODEL SELECTION ===
        section_label = ttk.Label(main_frame, text="4. Models to Test",
                                 font=("Arial", 12, "bold"))
        section_label.grid(row=current_row, column=0, columnspan=3, sticky=tk.W, pady=(15, 5))
        current_row += 1

        models_frame = ttk.LabelFrame(main_frame, text="Select Models (uncheck to skip)", padding="10")
        models_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1

        ttk.Checkbutton(models_frame, text="✓ PLS (Partial Least Squares)",
                       variable=self.use_pls).grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Label(models_frame, text="Linear, fast, interpretable",
                 font=("Arial", 8, "italic"), foreground="gray").grid(row=0, column=1, sticky=tk.W, padx=10)

        ttk.Checkbutton(models_frame, text="✓ Random Forest",
                       variable=self.use_randomforest).grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Label(models_frame, text="Nonlinear, robust, good for complex data",
                 font=("Arial", 8, "italic"), foreground="gray").grid(row=1, column=1, sticky=tk.W, padx=10)

        ttk.Checkbutton(models_frame, text="✓ MLP (Multi-Layer Perceptron)",
                       variable=self.use_mlp).grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Label(models_frame, text="Deep learning, captures nonlinearity",
                 font=("Arial", 8, "italic"), foreground="gray").grid(row=2, column=1, sticky=tk.W, padx=10)

        ttk.Checkbutton(models_frame, text="✓ Neural Boosted",
                       variable=self.use_neuralboosted).grid(row=3, column=0, sticky=tk.W, pady=3)
        ttk.Label(models_frame, text="Gradient boosting with neural networks, interpretable",
                 font=("Arial", 8, "italic"), foreground="gray").grid(row=3, column=1, sticky=tk.W, padx=10)

        # === RUN BUTTON ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=current_row, column=0, columnspan=3, pady=20)
        current_row += 1

        self.run_button = ttk.Button(button_frame, text="▶ Run Analysis",
                                     command=self._run_analysis,
                                     style="Accent.TButton")
        self.run_button.grid(row=0, column=0, padx=5, ipadx=20, ipady=10)

        ttk.Button(button_frame, text="Help",
                  command=self._show_help).grid(row=0, column=1, padx=5)

        ttk.Button(button_frame, text="Exit",
                  command=self.root.quit).grid(row=0, column=2, padx=5)

        # === STATUS ===
        self.status_label = ttk.Label(main_frame, text="Ready to run analysis",
                                     font=("Arial", 9), foreground="green")
        self.status_label.grid(row=current_row, column=0, columnspan=3, pady=10)

        # Initialize UI state
        self._update_input_fields()

    def _update_input_fields(self):
        """Update which input fields are shown based on input type."""
        if self.input_type.get() == "asd":
            # Show ASD, hide CSV
            self.asd_frame.grid()
            self.csv_frame.grid_remove()
            self.csv_id_frame.grid_remove()
        else:
            # Show CSV, hide ASD
            self.asd_frame.grid_remove()
            self.csv_frame.grid()
            self.csv_id_frame.grid()

    def _browse_asd_dir(self):
        """Browse for ASD directory and auto-import reference CSV if found."""
        directory = filedialog.askdirectory(title="Select ASD Directory")
        if directory:
            self.asd_dir.set(directory)

            # Look for CSV files in the selected directory
            csv_files = list(Path(directory).glob("*.csv"))

            if len(csv_files) == 1:
                # Exactly one CSV file found - automatically import it
                csv_path = str(csv_files[0])
                self.reference_file.set(csv_path)
                # Automatically try to detect columns
                self._auto_detect_columns()
                # Show info message
                messagebox.showinfo(
                    "CSV File Found",
                    f"Automatically imported reference file:\n{csv_files[0].name}"
                )
            elif len(csv_files) > 1:
                # Multiple CSV files found - let user choose
                messagebox.showinfo(
                    "Multiple CSV Files",
                    f"Found {len(csv_files)} CSV files in directory.\n"
                    f"Please select the reference file manually."
                )
            # If no CSV files found, user will select manually (no message needed)

    def _browse_spectra_file(self):
        """Browse for spectra CSV file."""
        filename = filedialog.askopenfilename(
            title="Select Spectra CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.spectra_file.set(filename)
            # Auto-detect ID column in spectra CSV
            self._detect_spectra_columns(filename)

    def _browse_reference_file(self):
        """Browse for reference CSV file."""
        filename = filedialog.askopenfilename(
            title="Select Reference CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.reference_file.set(filename)
            # Automatically try to detect columns
            self._auto_detect_columns()

    def _detect_spectra_columns(self, filename):
        """Detect ID column from spectra CSV file."""
        try:
            import pandas as pd

            # Read first row
            df = pd.read_csv(filename, nrows=5)
            columns = list(df.columns)

            if len(columns) == 0:
                return

            # Populate dropdown
            self.csv_id_combo['values'] = columns

            # First column is usually the ID
            id_guess = columns[0]

            # Count wavelength-like columns
            wavelength_cols = 0
            for col in columns[1:]:
                try:
                    wl_value = float(col)
                    if 200 <= wl_value <= 3000:
                        wavelength_cols += 1
                except (ValueError, TypeError):
                    pass

            msg = f"Detected spectra CSV: {Path(filename).name}\n\n"
            msg += f"Total columns: {len(columns)}\n"
            msg += f"Wavelength columns: {wavelength_cols}\n"
            msg += f"Suggested ID column: {id_guess}\n\n"
            msg += f"First 5 columns: {', '.join(columns[:5])}\n\n"
            msg += "Use this ID column?"

            response = messagebox.askyesno("Detect Spectra Columns", msg)
            if response:
                self.csv_id_var.set(id_guess)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n\n{str(e)}")

    def _auto_detect_columns(self):
        """Auto-detect column names from reference CSV."""
        if not self.reference_file.get():
            messagebox.showwarning("No File", "Please select a reference CSV file first")
            return

        try:
            import pandas as pd

            # Read first few rows
            df = pd.read_csv(self.reference_file.get(), nrows=10)
            columns = list(df.columns)

            if len(columns) == 0:
                messagebox.showerror("Error", "CSV file appears to be empty")
                return

            # Populate all three dropdowns
            self.spectral_file_combo['values'] = columns
            self.id_combo['values'] = columns
            self.target_combo['values'] = columns

            # Make intelligent guesses
            # First column is usually the spectral file column
            spectral_file_guess = columns[0]

            # Categorize remaining columns
            numeric_cols = []
            wavelength_like_cols = []
            non_numeric_cols = []

            for col in columns[1:]:  # Skip first column (spectral file)
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                    # Check if column name looks like a wavelength (numeric value)
                    try:
                        wl_value = float(col)
                        if 200 <= wl_value <= 3000:  # Typical spectral range in nm
                            wavelength_like_cols.append(col)
                    except (ValueError, TypeError):
                        pass
                else:
                    non_numeric_cols.append(col)

            # Determine if this is a spectra CSV or reference CSV
            has_spectra = len(wavelength_like_cols) > 50  # If >50 wavelength columns, likely spectra file

            # Specimen ID: First non-numeric column after spectral file, or second column if none
            if non_numeric_cols:
                specimen_id_guess = non_numeric_cols[0]
            elif len(columns) > 1:
                specimen_id_guess = columns[1]
            else:
                specimen_id_guess = ""

            # Target variable: First numeric column that's not a wavelength
            target_candidates = [c for c in numeric_cols if c not in wavelength_like_cols]
            if target_candidates:
                target_guess = target_candidates[0]
                # Any other numeric columns are additional target variables
                additional_target_cols = target_candidates[1:]
            elif numeric_cols:
                target_guess = numeric_cols[0]
            elif len(columns) > 2:
                target_guess = columns[2]
            else:
                target_guess = ""

            # Check for wrong file type
            if has_spectra:
                messagebox.showwarning(
                    "Wrong File Type",
                    f"WARNING: This file appears to contain spectral data!\n\n"
                    f"Detected {len(wavelength_like_cols)} wavelength columns.\n\n"
                    f"This should be selected as spectral data input (ASD or CSV spectra),\n"
                    f"not as a reference file.\n\n"
                    f"The reference file should only contain:\n"
                    f"  • Spectral file names\n"
                    f"  • Specimen IDs (optional)\n"
                    f"  • Target variable(s)"
                )
                return

            # Automatically set all three values (no popup)
            self.spectral_file_column.set(spectral_file_guess)
            self.id_column.set(specimen_id_guess)
            self.target_column.set(target_guess)

            # Update status with what was detected
            status_msg = f"✓ Auto-detected: File='{spectral_file_guess}', ID='{specimen_id_guess}', Target='{target_guess}'"
            if additional_target_cols:
                status_msg += f" | Additional targets available: {', '.join(additional_target_cols)}"

            self.status_label.config(text=status_msg, foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file:\n\n{str(e)}")

    def _validate_inputs(self):
        """Validate that all required inputs are provided."""
        errors = []

        # Check input data
        if self.input_type.get() == "asd":
            if not self.asd_dir.get():
                errors.append("Please select an ASD directory")
            elif not Path(self.asd_dir.get()).exists():
                errors.append("ASD directory does not exist")
        else:
            if not self.spectra_file.get():
                errors.append("Please select a spectra CSV file")
            elif not Path(self.spectra_file.get()).exists():
                errors.append("Spectra CSV file does not exist")

        # Check reference
        if not self.reference_file.get():
            errors.append("Please select a reference CSV file")
        elif not Path(self.reference_file.get()).exists():
            errors.append("Reference CSV file does not exist")

        # Check column names
        if not self.spectral_file_column.get():
            errors.append("Please enter the Spectral File column name")
        if not self.target_column.get():
            errors.append("Please enter the target variable name")
        # Note: specimen ID column is optional (for tracking only)

        return errors

    def _run_analysis(self):
        """Run the spectral analysis."""
        # Validate inputs
        errors = self._validate_inputs()
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return

        # Check if we should show progress monitor
        if self.show_progress.get():
            # Run with progress monitor (in-process with threading)
            self._run_analysis_with_progress()
        else:
            # Run in subprocess (old method, no progress)
            self._run_analysis_subprocess()

    def _run_analysis_with_progress(self):
        """Run analysis with live progress monitor."""
        try:
            # Import required modules
            from spectral_predict.progress_monitor import ProgressMonitor
            from spectral_predict.io import read_csv_spectra, read_reference_csv, align_xy, read_asd_dir
            from spectral_predict.interactive_gui import run_interactive_loading_gui
            from spectral_predict.search import run_search
            from spectral_predict.report import write_markdown_report

            # Update status
            self.status_label.config(text="Loading data...", foreground="blue")
            self.run_button.config(state="disabled")
            self.root.update()

            # STEP 1: Load data
            # Read spectral data
            if self.input_type.get() == "asd":
                X = read_asd_dir(self.asd_dir.get())
            else:
                X = read_csv_spectra(self.spectra_file.get())

            # Read reference data
            ref = read_reference_csv(
                self.reference_file.get(),
                self.spectral_file_column.get()
            )

            # Align X and y (extract target column)
            X_aligned, y_aligned = align_xy(
                X,
                ref,
                self.spectral_file_column.get(),
                self.target_column.get()
            )

            # Auto-populate wavelength range if not already set
            if not self.wavelength_min.get() or not self.wavelength_max.get():
                try:
                    wavelengths = X_aligned.columns.astype(float)
                    self.wavelength_min.set(str(int(wavelengths.min())))
                    self.wavelength_max.set(str(int(wavelengths.max())))
                except:
                    pass  # If wavelengths aren't numeric, just skip

            # Apply wavelength range filtering BEFORE interactive GUI (so plot shows trimmed range)
            wavelength_min_str = self.wavelength_min.get().strip()
            wavelength_max_str = self.wavelength_max.get().strip()

            if wavelength_min_str or wavelength_max_str:
                try:
                    # Get wavelength column names (they should be numeric)
                    wavelengths = X_aligned.columns.astype(float)

                    # Apply min filter
                    if wavelength_min_str:
                        wl_min = float(wavelength_min_str)
                        X_aligned = X_aligned.loc[:, wavelengths >= wl_min]
                        wavelengths = X_aligned.columns.astype(float)

                    # Apply max filter
                    if wavelength_max_str:
                        wl_max = float(wavelength_max_str)
                        X_aligned = X_aligned.loc[:, wavelengths <= wl_max]

                    print(f"Wavelength range filtered to {len(X_aligned.columns)} wavelengths " +
                          f"({X_aligned.columns.astype(float).min():.1f} - {X_aligned.columns.astype(float).max():.1f} nm)")
                except Exception as e:
                    messagebox.showwarning("Wavelength Filter Warning",
                        f"Could not apply wavelength filter: {e}\nProceeding with full spectrum.")

            # STEP 2: Show interactive GUI (if enabled)
            if self.use_gui.get():
                self.status_label.config(text="Opening interactive preview...", foreground="blue")
                self.root.update()

                # Run interactive GUI and get results
                interactive_result = run_interactive_loading_gui(
                    X_aligned,
                    y_aligned,
                    id_column=self.id_column.get(),
                    target=self.target_column.get()
                )

                # Check if user clicked continue
                if not interactive_result.get('user_continue', False):
                    self.status_label.config(text="Analysis cancelled by user", foreground="orange")
                    self.run_button.config(state="normal")
                    return

                # Use the potentially modified data from interactive GUI
                X_aligned = interactive_result['X']
                y_aligned = interactive_result['y']

            # STEP 3: Show progress monitor and run analysis
            self.status_label.config(text="Starting model search...", foreground="blue")
            self.root.update()

            # Determine task type
            if y_aligned.nunique() == 2:
                task_type = "binary_classification"
            elif y_aligned.dtype == 'object' or y_aligned.nunique() < 10:
                task_type = "multiclass_classification"
            else:
                task_type = "regression"

            # Create progress monitor (estimate total models)
            estimated_total = 350
            self.progress_monitor = ProgressMonitor(parent=self.root, total_models=estimated_total)
            self.progress_monitor.show()

            # Force the progress monitor to display
            self.root.update()

            # Note: Wavelength filtering already applied before interactive GUI
            # X_aligned now contains the trimmed spectral range

            # Build list of selected models
            selected_models = []
            if self.use_pls.get():
                selected_models.append("PLS")
            if self.use_randomforest.get():
                selected_models.append("RandomForest")
            if self.use_mlp.get():
                selected_models.append("MLP")
            if self.use_neuralboosted.get():
                selected_models.append("NeuralBoosted")

            # Validate at least one model is selected
            if not selected_models:
                raise ValueError("Please select at least one model to test")

            # Store data for thread to access
            self._analysis_data = {
                'X_aligned': X_aligned,
                'y_aligned': y_aligned,
                'task_type': task_type,
                'folds': self.folds.get(),
                'lambda_penalty': self.lambda_penalty.get(),
                'max_n_components': self.max_n_components.get(),
                'max_iter': self.max_iter.get(),
                'output_dir': self.output_dir.get(),
                'target_name': self.target_column.get(),
                'models_to_test': selected_models
            }

            # Define analysis function to run in thread
            def run_analysis_thread():
                try:
                    # Get data from stored dictionary
                    data = self._analysis_data

                    # Run search with progress callback
                    results_df = run_search(
                        data['X_aligned'],
                        data['y_aligned'],
                        task_type=data['task_type'],
                        folds=data['folds'],
                        lambda_penalty=data['lambda_penalty'],
                        max_n_components=data['max_n_components'],
                        max_iter=data['max_iter'],
                        models_to_test=data['models_to_test'],
                        progress_callback=self._update_progress_safe
                    )

                    # Create timestamp for unique filenames
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Save results with timestamp
                    output_dir = Path(data['output_dir'])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    results_filename = f"results_{data['target_name']}_{timestamp}.csv"
                    results_path = output_dir / results_filename
                    results_df.to_csv(results_path, index=False)

                    # Generate report with timestamp
                    report_dir = Path("reports")
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_filename = f"{data['target_name']}_{timestamp}.md"
                    report_path = report_dir / report_filename
                    write_markdown_report(
                        data['target_name'],  # target (correct parameter name)
                        results_df,           # df_ranked
                        str(report_dir)       # out_dir
                    )

                    # Complete progress monitor (schedule on main thread)
                    def on_complete():
                        if self.progress_monitor:
                            self.progress_monitor.complete(
                                success=True,
                                message=f"Analysis complete! Results saved to {results_path}"
                            )
                        self.status_label.config(
                            text="✓ Analysis complete! Check outputs/ directory",
                            foreground="green"
                        )
                        messagebox.showinfo(
                            "Success",
                            f"Analysis completed successfully!\n\n"
                            f"Results saved to:\n"
                            f"  - {results_path}\n"
                            f"  - {report_path}"
                        )
                        self.run_button.config(state="normal")

                    self.root.after(0, on_complete)

                except Exception as e:
                    # Handle errors (schedule on main thread)
                    error_msg = str(e)
                    import traceback
                    traceback.print_exc()  # Print to console for debugging

                    def on_error():
                        if self.progress_monitor:
                            self.progress_monitor.complete(
                                success=False,
                                message=f"Error: {error_msg}"
                            )
                        self.status_label.config(
                            text="✗ Analysis failed",
                            foreground="red"
                        )
                        messagebox.showerror(
                            "Error",
                            f"Analysis failed:\n\n{error_msg}"
                        )
                        self.run_button.config(state="normal")

                    self.root.after(0, on_error)

            # Start analysis in background thread
            self.analysis_thread = threading.Thread(target=run_analysis_thread, daemon=True)
            self.analysis_thread.start()

        except Exception as e:
            self.status_label.config(text="✗ Error starting analysis", foreground="red")
            messagebox.showerror("Error", f"Failed to start analysis:\n\n{str(e)}")
            self.run_button.config(state="normal")

    def _update_progress_safe(self, progress_data):
        """Thread-safe progress update."""
        if self.progress_monitor is not None:
            # Schedule update on main thread
            self.root.after(0, lambda: self.progress_monitor.update(progress_data))

    def _run_analysis_subprocess(self):
        """Run analysis in subprocess (old method without progress)."""
        # Build command
        cmd = [sys.executable, "-m", "spectral_predict.cli"]

        # Add input
        if self.input_type.get() == "asd":
            cmd.extend(["--asd-dir", self.asd_dir.get()])
        else:
            cmd.extend(["--spectra", self.spectra_file.get()])

        # Add required arguments
        cmd.extend([
            "--reference", self.reference_file.get(),
            "--id-column", self.spectral_file_column.get(),
            "--target", self.target_column.get(),
        ])

        # Add optional arguments
        cmd.extend([
            "--folds", str(self.folds.get()),
            "--lambda-penalty", str(self.lambda_penalty.get()),
            "--max-n-components", str(self.max_n_components.get()),
            "--max-iter", str(self.max_iter.get()),
            "--outdir", self.output_dir.get(),
        ])

        # Add GUI flag
        if not self.use_gui.get():
            cmd.append("--no-interactive")

        # Update status
        self.status_label.config(text="Running analysis...", foreground="blue")
        self.run_button.config(state="disabled")
        self.root.update()

        # Run in subprocess
        try:
            script_dir = Path(__file__).parent

            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.status_label.config(
                    text="✓ Analysis complete! Check outputs/ directory",
                    foreground="green"
                )
                messagebox.showinfo(
                    "Success",
                    f"Analysis completed successfully!\n\n"
                    f"Results saved to:\n"
                    f"  - {self.output_dir.get()}/results.csv\n"
                    f"  - reports/{self.target_column.get()}.md"
                )
            else:
                self.status_label.config(text="✗ Analysis failed", foreground="red")
                messagebox.showerror(
                    "Error",
                    f"Analysis failed with error:\n\n{result.stderr[-500:]}"
                )
        except Exception as e:
            self.status_label.config(text="✗ Error running analysis", foreground="red")
            messagebox.showerror("Error", f"Failed to run analysis:\n\n{str(e)}")
        finally:
            self.run_button.config(state="normal")

    def _show_help(self):
        """Show help dialog."""
        help_text = """
Spectral Predict - Quick Start Guide

1. INPUT DATA
   • Select either ASD files (directory) or CSV file
   • Select your reference CSV with target variables

2. COLUMN NAMES (Three types from reference CSV)
   • Spectral File Column: Links to spectral data (e.g., "spectrum1")
   • Specimen ID Column: For tracking only (e.g., "grass", "bone3")
   • Target Variable: The value to predict (e.g., "%N", "%collagen")

3. OPTIONS
   • CV Folds: Number of cross-validation folds (default: 5)
   • Complexity Penalty: Weight for model simplicity (default: 0.15, higher = prefer simpler)
   • Max Latent Variables: Maximum PLS components to test (default: 24)
   • Max Iterations: Maximum iterations for neural networks (default: 500)
   • Wavelength Range: Trim spectrum to specific range (e.g., 780-2350 for NIR)
     Auto-populated after data load, edit to restrict analysis range
   • Interactive Preview: Opens GUI to explore data before analysis

4. RUN
   • Click "Run Analysis" to start
   • If interactive mode is enabled, a preview GUI will appear
   • After confirming, the model search will run
   • Results are saved to outputs/ and reports/

Example:
   • ASD Directory: example/quick_start/
   • Reference CSV: example/quick_start/reference.csv
   • ID Column: File Number
   • Target: %Collagen
        """
        messagebox.showinfo("Help", help_text)


def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = SpectralPredictApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
