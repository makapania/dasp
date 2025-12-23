"""
Performance Benchmark for tksheet - Test BEFORE Integration

This script tests tksheet performance with realistic dataset sizes to determine
if it's fast enough for the Data Viewer tab.

REQUIREMENTS:
- Install: pip install tksheet
- Or use: .venv/Scripts/python.exe -m pip install tksheet

DECISION CRITERIA:
✓ Initial load: < 1 second
✓ Scrolling: Instant feel (< 50ms)
✓ Memory: Reasonable (< 500MB)
✓ Overall: Feels like Excel

If tksheet passes → Worth integrating
If tksheet fails → Keep optimized pagination system
"""

import tkinter as tk
from tkinter import ttk
import time
import sys

# Check if tksheet is installed
try:
    from tksheet import Sheet
except ImportError:
    print("ERROR: tksheet not installed")
    print("Install with: .venv/Scripts/python.exe -m pip install tksheet")
    sys.exit(1)

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

        # Generate test data
        print("\n" + "="*60)
        print("TKSHEET PERFORMANCE BENCHMARK")
        print("="*60)
        print(f"Dataset size: {self.n_samples} samples × {self.n_wavelengths} wavelengths")
        print(f"Total cells: {self.n_samples * self.n_wavelengths:,}")
        print("="*60 + "\n")

        self.status_label.config(text="Generating test data...")
        self.root.update()

        self._generate_test_data()

        # Auto-run benchmark on startup
        self.root.after(100, self.run_benchmark)

    def _create_ui(self):
        """Create the test UI."""
        # Top controls
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

        # Status
        self.status_label = ttk.Label(control_frame, text="Ready",
                                     font=('Arial', 10))
        self.status_label.pack(side='left', padx=20)

        # Results display
        result_frame = ttk.Frame(self.root, padding=10)
        result_frame.pack(fill='x')

        self.result_text = tk.Text(result_frame, height=8, font=('Courier', 9))
        self.result_text.pack(fill='x')

        # Sheet container
        sheet_frame = ttk.Frame(self.root, padding=10)
        sheet_frame.pack(fill='both', expand=True)

        # Create tksheet
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

        # Enable all bindings for realistic testing
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

        # Instructions
        instr_frame = ttk.Frame(self.root, padding=10)
        instr_frame.pack(fill='x')

        instructions = """
PERFORMANCE TEST INSTRUCTIONS:
1. Benchmark runs automatically on startup
2. After benchmark completes, manually scroll the sheet (use mouse wheel and scrollbars)
3. EVALUATE: Does scrolling feel instant and smooth like Excel?
4. Check memory usage in Task Manager during scrolling
5. Try selecting cells, navigating with arrow keys

DECISION: If scrolling feels instant and smooth → tksheet is good
         If there's any lag or jerkiness → stick with current pagination
"""
        ttk.Label(instr_frame, text=instructions, font=('Courier', 9),
                 justify='left').pack(anchor='w')

    def _generate_test_data(self):
        """Generate realistic spectral data for testing."""
        start = time.time()

        # Generate spectral-like data (wavelengths 350-2500nm)
        wavelengths = np.linspace(350, 2500, self.n_wavelengths)

        # Generate realistic spectral curves (absorption/reflectance patterns)
        self.data = np.random.random((self.n_samples, self.n_wavelengths)) * 0.5 + 0.3

        # Add some structure to make it realistic
        for i in range(self.n_samples):
            # Add Gaussian peaks (like absorption bands)
            for _ in range(3):
                center = np.random.randint(0, self.n_wavelengths)
                width = np.random.randint(50, 200)
                amplitude = np.random.random() * 0.3
                x = np.arange(self.n_wavelengths)
                self.data[i] += amplitude * np.exp(-((x - center) / width) ** 2)

        # Normalize to realistic range (0.1 - 1.0)
        self.data = np.clip(self.data, 0.1, 1.0)

        # Create DataFrame
        self.df = pd.DataFrame(
            self.data,
            columns=[f"{wl:.1f}" for wl in wavelengths],
            index=[f"Sample_{i+1}" for i in range(self.n_samples)]
        )

        elapsed = time.time() - start
        print(f"✓ Test data generated in {elapsed:.3f}s")
        print(f"  Shape: {self.df.shape}")
        print(f"  Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")

    def run_benchmark(self):
        """Run comprehensive performance benchmark."""
        results = []
        results.append("="*60)
        results.append("TKSHEET PERFORMANCE BENCHMARK RESULTS")
        results.append("="*60)

        # Test 1: Initial load time
        self.status_label.config(text="Testing initial load time...")
        self.root.update()

        start = time.time()

        # Convert to list format (tksheet requires lists, not numpy arrays)
        data_list = self.df.values.tolist()
        headers_list = self.df.columns.tolist()

        # Format cells (realistic - format to 5 decimal places)
        formatted_data = []
        for row in data_list:
            formatted_data.append([f"{val:.5f}" for val in row])

        load_time = time.time() - start

        results.append(f"\n1. DATA PREPARATION TIME: {load_time:.3f}s")
        if load_time < 0.5:
            results.append("   ✓ EXCELLENT - Very fast preparation")
        elif load_time < 1.0:
            results.append("   ✓ GOOD - Acceptable preparation time")
        else:
            results.append("   ✗ SLOW - Data preparation is slow")

        # Test 2: Sheet population time
        self.status_label.config(text="Testing sheet population...")
        self.root.update()

        start = time.time()
        self.sheet.set_sheet_data(formatted_data)
        self.sheet.headers(headers_list)
        self.root.update()  # Force UI update
        populate_time = time.time() - start

        results.append(f"\n2. SHEET POPULATION TIME: {populate_time:.3f}s")
        if populate_time < 0.5:
            results.append("   ✓ EXCELLENT - Instant load")
        elif populate_time < 1.0:
            results.append("   ✓ GOOD - Fast load")
        elif populate_time < 2.0:
            results.append("   ~ ACCEPTABLE - Noticeable but OK")
        else:
            results.append("   ✗ SLOW - Too slow for good UX")

        total_load = load_time + populate_time
        results.append(f"\n   TOTAL LOAD TIME: {total_load:.3f}s")

        # Test 3: Highlight performance (simulate excluded samples)
        self.status_label.config(text="Testing highlighting...")
        self.root.update()

        excluded_rows = list(range(0, 100, 5))  # Every 5th row in first 100

        start = time.time()
        for row_idx in excluded_rows:
            self.sheet[row_idx].bg = "#FFE0E0"  # Pink background
        self.root.update()
        highlight_time = time.time() - start

        results.append(f"\n3. HIGHLIGHTING TIME ({len(excluded_rows)} rows): {highlight_time:.3f}s")
        if highlight_time < 0.1:
            results.append("   ✓ EXCELLENT - Instant highlighting")
        elif highlight_time < 0.5:
            results.append("   ✓ GOOD - Fast highlighting")
        else:
            results.append("   ✗ SLOW - Highlighting is slow")

        # Test 4: Memory estimate
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024**2

        results.append(f"\n4. MEMORY USAGE: {memory_mb:.1f} MB")
        if memory_mb < 300:
            results.append("   ✓ EXCELLENT - Low memory usage")
        elif memory_mb < 500:
            results.append("   ✓ GOOD - Reasonable memory usage")
        elif memory_mb < 800:
            results.append("   ~ ACCEPTABLE - High but OK")
        else:
            results.append("   ✗ HIGH - Memory usage is concerning")

        # Overall assessment
        results.append("\n" + "="*60)
        results.append("OVERALL ASSESSMENT:")
        results.append("="*60)

        if total_load < 1.0 and highlight_time < 0.5:
            results.append("✓✓✓ EXCELLENT PERFORMANCE")
            results.append("    → tksheet is fast enough - RECOMMEND INTEGRATION")
        elif total_load < 2.0 and highlight_time < 1.0:
            results.append("✓✓ GOOD PERFORMANCE")
            results.append("   → tksheet should work well - Worth testing more")
        else:
            results.append("✗ SLOW PERFORMANCE")
            results.append("  → tksheet too slow - KEEP CURRENT PAGINATION")

        results.append("\nNEXT STEP: Manually test scrolling!")
        results.append("Use mouse wheel and scrollbars to scroll the sheet.")
        results.append("Does it feel smooth and instant like Excel?")
        results.append("="*60)

        # Display results
        result_text = "\n".join(results)
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('1.0', result_text)

        # Also print to console
        print(result_text)

        self.status_label.config(text="✓ Benchmark complete - Now test scrolling manually!")

    def test_scrolling(self):
        """Manual scrolling test instructions."""
        msg = """
MANUAL SCROLLING TEST:

1. Use your mouse wheel to scroll vertically
2. Use the horizontal scrollbar to scroll sideways
3. Try clicking and dragging the scrollbars
4. Use arrow keys to navigate between cells

EVALUATE:
- Does it feel instant and responsive?
- Is there any lag or jerkiness?
- Does it feel like Excel or commercial software?

If YES → tksheet is good enough
If NO → stick with current pagination system
"""
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('1.0', msg)
        self.status_label.config(text="Test scrolling now! (See instructions)")

    def clear_sheet(self):
        """Clear the sheet."""
        self.sheet.set_sheet_data([[]])
        self.sheet.headers([])
        self.result_text.delete('1.0', 'end')
        self.status_label.config(text="Sheet cleared")


def main():
    """Run the performance test."""
    print("\nStarting tksheet Performance Benchmark...")
    print("Window will open and test will run automatically.\n")

    root = tk.Tk()
    app = TksheetPerformanceTest(root)
    root.mainloop()


if __name__ == "__main__":
    main()
