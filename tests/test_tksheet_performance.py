"""
Performance Benchmark for tksheet - Test BEFORE Integration

This script tests tksheet performance with realistic dataset sizes to determine
if it's fast enough for the Data Viewer tab.

REQUIREMENTS:
- Install: pip install tksheet
- Or use: .venv/Scripts/python.exe -m pip install tksheet

DECISION CRITERIA:
- Initial load: < 1 second
- Scrolling: Instant feel (< 50ms)
- Memory: Reasonable (< 500MB)
- Overall: Feels like Excel

If tksheet passes -> Worth integrating
If tksheet fails -> Keep optimized pagination system
"""

import pytest

pytestmark = pytest.mark.skip(reason="GUI benchmark - not a unit test")

import tkinter as tk
from tkinter import ttk
import time
import sys

# Check if tksheet is installed
try:
    from tksheet import Sheet
except ImportError:
    Sheet = None

import numpy as np
import pandas as pd


class TksheetPerformanceTest:
    """Test tksheet performance with realistic spectral data sizes."""

    def __init__(self, root):
        self.root = root
        self.root.title("tksheet Performance Benchmark")
        self.root.geometry("1200x800")

        # Test configuration
        self.n_samples = 1000  # Typical dataset size
        self.n_wavelengths = 2000  # Full spectral range

        # Create UI
        self._create_ui()

        self.status_label.config(text="Generating test data...")
        self.root.update()

        self._generate_test_data()

        # Auto-run benchmark on startup
        self.root.after(100, self.run_benchmark)

    def _create_ui(self):
        """Create the test UI."""
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill='x')

        ttk.Label(control_frame, text="tksheet Performance Test",
                 font=('Arial', 14, 'bold')).pack(side='left', padx=(0, 20))

        ttk.Button(control_frame, text="Run Benchmark",
                  command=self.run_benchmark).pack(side='left', padx=5)

        ttk.Button(control_frame, text="Test Scrolling (Manual)",
                  command=self.test_scrolling).pack(side='left', padx=5)

        ttk.Button(control_frame, text="Clear",
                  command=self.clear_sheet).pack(side='left', padx=5)

        self.status_label = ttk.Label(control_frame, text="Ready",
                                     font=('Arial', 10))
        self.status_label.pack(side='left', padx=20)

        result_frame = ttk.Frame(self.root, padding=10)
        result_frame.pack(fill='x')

        self.result_text = tk.Text(result_frame, height=8, font=('Courier', 9))
        self.result_text.pack(fill='x')

        sheet_frame = ttk.Frame(self.root, padding=10)
        sheet_frame.pack(fill='both', expand=True)

        self.sheet = Sheet(
            sheet_frame,
            data=[],
            headers=[],
            height=600,
            width=1180,
            show_row_index=True,
            show_header=True,
            show_top_left=True,
        )

        self.sheet.enable_bindings(
            "single_select",
            "row_select",
            "column_width_resize",
            "double_click_column_resize",
            "arrowkeys",
            "right_click_popup_menu",
            "rc_select",
            "copy",
        )

        self.sheet.grid(row=0, column=0, sticky='nsew')
        sheet_frame.grid_rowconfigure(0, weight=1)
        sheet_frame.grid_columnconfigure(0, weight=1)

    def _generate_test_data(self):
        """Generate realistic spectral data for testing."""
        wavelengths = np.linspace(350, 2500, self.n_wavelengths)
        self.data = np.random.random((self.n_samples, self.n_wavelengths)) * 0.5 + 0.3
        for i in range(self.n_samples):
            for _ in range(3):
                center = np.random.randint(0, self.n_wavelengths)
                width = np.random.randint(50, 200)
                amplitude = np.random.random() * 0.3
                x = np.arange(self.n_wavelengths)
                self.data[i] += amplitude * np.exp(-((x - center) / width) ** 2)
        self.data = np.clip(self.data, 0.1, 1.0)
        self.df = pd.DataFrame(
            self.data,
            columns=[f"{wl:.1f}" for wl in wavelengths],
            index=[f"Sample_{i+1}" for i in range(self.n_samples)]
        )

    def run_benchmark(self):
        """Run comprehensive performance benchmark."""
        self.status_label.config(text="Testing initial load time...")
        self.root.update()

        start = time.time()
        data_list = self.df.values.tolist()
        headers_list = self.df.columns.tolist()
        formatted_data = [[f"{val:.5f}" for val in row] for row in data_list]
        load_time = time.time() - start

        self.status_label.config(text="Testing sheet population...")
        self.root.update()

        start = time.time()
        self.sheet.set_sheet_data(formatted_data)
        self.sheet.headers(headers_list)
        self.root.update()
        populate_time = time.time() - start

        total_load = load_time + populate_time
        self.status_label.config(text=f"Benchmark complete - total load: {total_load:.3f}s")

    def test_scrolling(self):
        """Manual scrolling test instructions."""
        self.status_label.config(text="Test scrolling now!")

    def clear_sheet(self):
        """Clear the sheet."""
        self.sheet.set_sheet_data([[]])
        self.sheet.headers([])
        self.status_label.config(text="Sheet cleared")


def test_tksheet_benchmark():
    """Run the tksheet performance benchmark (requires GUI)."""
    if Sheet is None:
        pytest.skip("tksheet not installed")

    root = tk.Tk()
    app = TksheetPerformanceTest(root)
    root.mainloop()


if __name__ == "__main__":
    if Sheet is None:
        print("ERROR: tksheet not installed")
        sys.exit(1)
    root = tk.Tk()
    app = TksheetPerformanceTest(root)
    root.mainloop()
