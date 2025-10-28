"""GUI-based interactive loading phase using matplotlib and tkinter."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from .preprocess import SavgolDerivative


class InteractiveLoadingGUI:
    """GUI for interactive data loading and exploration."""

    def __init__(self, X, y, id_column=None, target=None):
        """
        Initialize GUI.

        Parameters
        ----------
        X : pd.DataFrame
            Spectral data
        y : pd.Series
            Target values
        id_column : str
            ID column name
        target : str
            Target name
        """
        self.X_original = X.copy()
        self.X = X.copy()
        self.y = y
        self.id_column = id_column
        self.target = target
        self.converted_to_absorbance = False
        self.screening_results = None
        self.user_continue = False

        # Create main window
        self.root = tk.Tk()
        self.root.title("Spectral Predict - Interactive Data Loading")
        self.root.geometry("1400x900")

        # Create main layout
        self._create_layout()

    def _create_layout(self):
        """Create the main GUI layout."""
        # Top frame for info and controls
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Title
        title_label = ttk.Label(
            top_frame,
            text="Interactive Data Loading Phase",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=5)

        # Data info
        info_text = f"Dataset: {len(self.X)} samples × {self.X.shape[1]} wavelengths"
        if self.target:
            info_text += f" | Target: '{self.target}' (range: {self.y.min():.2f} - {self.y.max():.2f})"
        info_label = ttk.Label(top_frame, text=info_text, font=("Arial", 10))
        info_label.pack(pady=5)

        # Data range info
        min_val = self.X.min().min()
        max_val = self.X.max().max()
        range_text = f"Spectral value range: {min_val:.4f} to {max_val:.4f}"
        self.range_label = ttk.Label(top_frame, text=range_text, font=("Arial", 9))
        self.range_label.pack(pady=2)

        # Control buttons frame
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(pady=10)

        # Absorbance conversion button
        self.abs_button = ttk.Button(
            control_frame,
            text="Convert to Absorbance (A = log10(1/R))",
            command=self._convert_to_absorbance
        )
        self.abs_button.pack(side=tk.LEFT, padx=5)

        # Continue button
        self.continue_button = ttk.Button(
            control_frame,
            text="Continue to Model Search →",
            command=self._continue_clicked,
            style="Accent.TButton"
        )
        self.continue_button.pack(side=tk.LEFT, padx=5)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self._create_data_preview_tab()
        self._create_raw_spectra_tab()
        self._create_derivative1_tab()
        self._create_derivative2_tab()
        self._create_screening_tab()

    def _create_data_preview_tab(self):
        """Create data preview tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Data Preview")

        # Create treeview for data
        columns = ['Sample_ID', 'Target'] + [f"{wl:.1f}" for wl in self.X.columns[:10]]
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)

        # Configure columns
        tree.column('Sample_ID', width=150)
        tree.column('Target', width=80)
        for col in columns[2:]:
            tree.column(col, width=80)

        # Add headings
        for col in columns:
            tree.heading(col, text=col)

        # Add data (first 10 samples)
        for idx in self.X.index[:10]:
            values = [idx, f"{self.y.loc[idx]:.2f}"]
            values += [f"{val:.6f}" for val in self.X.loc[idx, self.X.columns[:10]].values]
            tree.insert('', tk.END, values=values)

        # Add scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Pack
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Add info label
        info_label = ttk.Label(
            frame,
            text=f"Showing first 10 samples and first 10 wavelengths. Total: {len(self.X)} samples × {self.X.shape[1]} wavelengths",
            font=("Arial", 9, "italic")
        )
        info_label.grid(row=2, column=0, columnspan=2, pady=10)

    def _create_raw_spectra_tab(self):
        """Create raw spectra plot tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📈 Raw Spectra")

        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        wavelengths = self.X.columns.values
        n_samples = len(self.X)

        # Plot settings
        if n_samples <= 50:
            alpha = 0.3
            indices = range(n_samples)
        else:
            alpha = 0.5
            indices = np.random.choice(n_samples, size=50, replace=False)

        for i in indices:
            ax.plot(wavelengths, self.X.iloc[i, :], alpha=alpha, color='blue', linewidth=1)

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title(f'Raw Spectra (n={n_samples})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        canvas.flush_events()  # Force rendering

        self.raw_fig = fig
        self.raw_ax = ax
        self.raw_canvas = canvas

    def _create_derivative1_tab(self):
        """Create first derivative plot tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📉 1st Derivative")

        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # Compute derivative
        deriv1_transformer = SavgolDerivative(deriv=1, window=7)
        X_deriv1 = deriv1_transformer.transform(self.X.values)

        wavelengths = self.X.columns.values
        n_samples = len(self.X)

        if n_samples <= 50:
            alpha = 0.3
            indices = range(n_samples)
        else:
            alpha = 0.5
            indices = np.random.choice(n_samples, size=50, replace=False)

        for i in indices:
            ax.plot(wavelengths, X_deriv1[i, :], alpha=alpha, color='green', linewidth=1)

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('First Derivative', fontsize=12)
        ax.set_title(f'First Derivative Spectra (SG window=7, n={n_samples})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_derivative2_tab(self):
        """Create second derivative plot tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 2nd Derivative")

        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # Compute derivative
        deriv2_transformer = SavgolDerivative(deriv=2, window=7)
        X_deriv2 = deriv2_transformer.transform(self.X.values)

        wavelengths = self.X.columns.values
        n_samples = len(self.X)

        if n_samples <= 50:
            alpha = 0.3
            indices = range(n_samples)
        else:
            alpha = 0.5
            indices = np.random.choice(n_samples, size=50, replace=False)

        for i in indices:
            ax.plot(wavelengths, X_deriv2[i, :], alpha=alpha, color='red', linewidth=1)

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Second Derivative', fontsize=12)
        ax.set_title(f'Second Derivative Spectra (SG window=7, n={n_samples})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_screening_tab(self):
        """Create predictor screening tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔍 Predictor Screening")

        if self.y is None:
            label = ttk.Label(frame, text="No target variable available for screening",
                            font=("Arial", 12))
            label.pack(pady=50)
            return

        # Compute screening
        self.screening_results = self._compute_screening()

        # Create figure with two subplots
        fig = Figure(figsize=(12, 10))

        # Plot 1: Correlation
        ax1 = fig.add_subplot(2, 1, 1)
        correlations = self.screening_results['correlations']
        wavelengths = correlations.index.values
        top_wls = self.screening_results['top_wavelengths'][:10]

        ax1.plot(wavelengths, correlations.values, color='blue', linewidth=1.5)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        for wl in top_wls:
            ax1.axvline(x=wl, color='red', alpha=0.3, linestyle=':', linewidth=1)
        ax1.set_xlabel('Wavelength (nm)', fontsize=11)
        ax1.set_ylabel('Correlation with Target', fontsize=11)
        ax1.set_title('Predictor Screening: Correlation by Wavelength',
                      fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Absolute correlation
        ax2 = fig.add_subplot(2, 1, 2)
        abs_corr = self.screening_results['abs_correlations']
        ax2.plot(wavelengths, abs_corr.values, color='purple', linewidth=1.5)
        for wl in top_wls:
            ax2.axvline(x=wl, color='red', alpha=0.3, linestyle=':', linewidth=1)
        ax2.set_xlabel('Wavelength (nm)', fontsize=11)
        ax2.set_ylabel('|Correlation|', fontsize=11)
        ax2.set_title('Absolute Correlation with Target', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Add info panel
        info_frame = ttk.Frame(frame)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Top wavelengths
        top_text = f"Top 10 wavelengths correlated with '{self.target}':\n"
        for i, (wl, corr) in enumerate(zip(
            self.screening_results['top_wavelengths'][:10],
            self.screening_results['top_correlations'][:10]
        ), 1):
            top_text += f"  {i:2d}. {wl:8.2f} nm -> r = {corr:+.4f}\n"

        top_label = ttk.Label(info_frame, text=top_text, font=("Courier", 9),
                             justify=tk.LEFT)
        top_label.pack(side=tk.LEFT, padx=20)

        # Interpretation
        max_abs_corr = self.screening_results['abs_correlations'].max()
        if max_abs_corr > 0.7:
            interp = f"Strong correlations detected (max |r| = {max_abs_corr:.3f})\n"
            interp += "Your variables of interest are likely present in the spectra"
            color = "green"
        elif max_abs_corr > 0.4:
            interp = f"Moderate correlations detected (max |r| = {max_abs_corr:.3f})\n"
            interp += "Modeling may be possible but could be challenging"
            color = "orange"
        else:
            interp = f"Weak correlations detected (max |r| = {max_abs_corr:.3f})\n"
            interp += "Warning: Target may not be well-predicted from these spectra"
            color = "red"

        interp_label = ttk.Label(info_frame, text=interp, font=("Arial", 10, "bold"),
                                foreground=color)
        interp_label.pack(side=tk.RIGHT, padx=20)

    def _compute_screening(self):
        """Compute predictor screening."""
        correlations = pd.Series(index=self.X.columns, dtype=float)

        for wl in self.X.columns:
            corr = np.corrcoef(self.X[wl].values, self.y.values)[0, 1]
            correlations[wl] = corr

        abs_corr = correlations.abs()
        top_indices = abs_corr.nlargest(20).index

        return {
            'correlations': correlations,
            'top_wavelengths': list(top_indices),
            'top_correlations': correlations[top_indices].values,
            'abs_correlations': abs_corr
        }

    def _convert_to_absorbance(self):
        """Handle absorbance conversion."""
        if self.converted_to_absorbance:
            messagebox.showinfo("Already Converted",
                               "Data has already been converted to absorbance.")
            return

        response = messagebox.askyesno(
            "Convert to Absorbance",
            "Convert reflectance to absorbance using A = log10(1/R)?\n\n"
            "This is commonly used in programs like Unscrambler.\n\n"
            "Note: All subsequent analysis will use absorbance values."
        )

        if response:
            # Convert
            self.X = self.X.clip(lower=1e-6)
            self.X = np.log10(1.0 / self.X)
            self.converted_to_absorbance = True

            # Update range label
            min_val = self.X.min().min()
            max_val = self.X.max().max()
            range_text = f"Spectral value range: {min_val:.4f} to {max_val:.4f} (ABSORBANCE)"
            self.range_label.config(text=range_text)

            # Disable button
            self.abs_button.config(state='disabled', text="✓ Converted to Absorbance")

            # Show success
            messagebox.showinfo("Conversion Complete",
                               f"Successfully converted to absorbance.\n\n"
                               f"New range: {min_val:.4f} to {max_val:.4f}")

            # Update plots would require recreating them - could add that feature

    def _continue_clicked(self):
        """Handle continue button click."""
        self.user_continue = True
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Run the GUI and wait for user."""
        self.root.mainloop()

        return {
            'X': self.X,
            'y': self.y,
            'converted_to_absorbance': self.converted_to_absorbance,
            'screening_results': self.screening_results,
            'user_continue': self.user_continue
        }


def run_interactive_loading_gui(X, y=None, id_column=None, target=None):
    """
    Run GUI-based interactive loading phase.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data
    y : pd.Series, optional
        Target values
    id_column : str, optional
        ID column name
    target : str, optional
        Target name

    Returns
    -------
    dict
        Results including processed X, y, and user choices
    """
    gui = InteractiveLoadingGUI(X, y, id_column, target)
    return gui.run()
