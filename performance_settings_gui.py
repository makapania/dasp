#!/usr/bin/env python3
"""
Performance Settings GUI Panel for DASP

Provides user-friendly controls for GPU acceleration and grid parallelization settings.
Can be integrated into existing tkinter GUI or run standalone.

Usage:
    from performance_settings_gui import PerformanceSettingsPanel

    # In your GUI:
    settings_panel = PerformanceSettingsPanel(parent_frame)
    settings_panel.pack(fill='both', expand=True)

    # Get config when running analysis:
    perf_config = settings_panel.get_config()
    results = run_search(X, y, perf_config=perf_config, ...)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from performance_config import PerformanceConfig
from hardware_detection import detect_hardware


class PerformanceSettingsPanel(ttk.Frame):
    """
    GUI panel for performance configuration settings.

    Features:
    - Quick mode selection (Power/Balanced/Light/Auto)
    - Advanced settings (CPU slider, GPU checkbox, parallel grid)
    - Hardware detection and display
    - Runtime estimation
    - Save/load preferences
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Detect hardware
        self.hw_config = detect_hardware(verbose=False)

        # Load saved preferences (or defaults)
        try:
            self.saved_config = PerformanceConfig.from_user_preferences()
            initial_mode = self.saved_config.mode
        except:
            initial_mode = 'balanced'

        # Create UI
        self._create_widgets(initial_mode)

        # Update estimates
        self._update_all()

    def _create_widgets(self, initial_mode='balanced'):
        """Create all GUI widgets"""

        # Main container with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)

        # ============================================================
        # TITLE
        # ============================================================
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(title_frame, text="⚡ Performance Settings",
                 font=('TkDefaultFont', 14, 'bold')).pack(side='left')

        ttk.Button(title_frame, text="ℹ Help",
                  command=self._show_help).pack(side='right')

        # ============================================================
        # MODE SELECTION
        # ============================================================
        mode_frame = ttk.LabelFrame(main_frame, text="Select Mode", padding="10")
        mode_frame.pack(fill='x', pady=(0, 10))

        self.mode_var = tk.StringVar(value=initial_mode)

        # Mode descriptions
        modes = [
            ('auto', '🔄 Auto (Recommended)',
             'Automatically detect hardware and use optimal settings'),
            ('balanced', '⚖️ Balanced Mode',
             'Good speed, leaves CPU free for multitasking  (45-60 min)'),
            ('power', '🚀 Power Mode',
             'Maximum speed, uses all resources  (30-45 min)'),
            ('light', '🌙 Light Mode',
             'Minimal impact, work normally while analyzing  (90-120 min)'),
        ]

        for value, label, description in modes:
            rb_frame = ttk.Frame(mode_frame)
            rb_frame.pack(fill='x', pady=2)

            ttk.Radiobutton(rb_frame, text=label, variable=self.mode_var,
                           value=value, command=self._on_mode_change).pack(side='left')

            ttk.Label(rb_frame, text=description,
                     foreground='gray').pack(side='left', padx=(10, 0))

        # ============================================================
        # ADVANCED SETTINGS (Collapsible)
        # ============================================================
        self.show_advanced = tk.BooleanVar(value=False)

        advanced_toggle = ttk.Checkbutton(main_frame, text="▼ Advanced Settings",
                                         variable=self.show_advanced,
                                         command=self._toggle_advanced)
        advanced_toggle.pack(anchor='w', pady=(5, 0))

        self.advanced_frame = ttk.Frame(main_frame)
        # Initially hidden

        # Advanced content
        advanced_content = ttk.LabelFrame(self.advanced_frame, text="Fine-Tune Settings",
                                         padding="10")
        advanced_content.pack(fill='x', pady=(5, 0))

        # CPU Usage Slider
        cpu_frame = ttk.Frame(advanced_content)
        cpu_frame.pack(fill='x', pady=5)

        ttk.Label(cpu_frame, text="Maximum CPU Usage:").pack(anchor='w')

        cpu_slider_frame = ttk.Frame(cpu_frame)
        cpu_slider_frame.pack(fill='x')

        self.cpu_slider = ttk.Scale(cpu_slider_frame, from_=10, to=100,
                                    orient='horizontal', command=self._on_cpu_change)
        self.cpu_slider.set(60)
        self.cpu_slider.pack(side='left', fill='x', expand=True)

        self.cpu_value_label = ttk.Label(cpu_slider_frame, text="60%", width=6)
        self.cpu_value_label.pack(side='left', padx=(5, 0))

        self.cpu_cores_label = ttk.Label(cpu_frame, text="", foreground='gray')
        self.cpu_cores_label.pack(anchor='w')

        # GPU Checkbox
        gpu_frame = ttk.Frame(advanced_content)
        gpu_frame.pack(fill='x', pady=5)

        self.gpu_var = tk.BooleanVar(value=self.hw_config['gpu_available'])

        gpu_text = "Use GPU acceleration"
        if self.hw_config['gpu_available']:
            gpu_text += f" ({self.hw_config['gpu_type']} detected)"
        else:
            gpu_text += " (not available)"

        self.gpu_checkbox = ttk.Checkbutton(gpu_frame, text=gpu_text,
                                           variable=self.gpu_var,
                                           command=self._on_setting_change)
        self.gpu_checkbox.pack(anchor='w')

        if not self.hw_config['gpu_available']:
            self.gpu_checkbox.state(['disabled'])

        gpu_help = ttk.Label(gpu_frame,
                            text="Disable if you need GPU for gaming/video editing",
                            foreground='gray', font=('TkDefaultFont', 9))
        gpu_help.pack(anchor='w', padx=(20, 0))

        # Parallel Grid Checkbox
        parallel_frame = ttk.Frame(advanced_content)
        parallel_frame.pack(fill='x', pady=5)

        self.parallel_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(parallel_frame, text="Parallel Grid Search (test multiple models simultaneously)",
                       variable=self.parallel_var,
                       command=self._on_setting_change).pack(anchor='w')

        parallel_help = ttk.Label(parallel_frame,
                                 text="Disable for debugging or when you need subset analysis",
                                 foreground='gray', font=('TkDefaultFont', 9))
        parallel_help.pack(anchor='w', padx=(20, 0))

        # ============================================================
        # HARDWARE INFO
        # ============================================================
        hw_frame = ttk.LabelFrame(main_frame, text="💻 Hardware Detected", padding="10")
        hw_frame.pack(fill='x', pady=(10, 10))

        hw_grid = ttk.Frame(hw_frame)
        hw_grid.pack(fill='x')

        # Create grid layout
        row = 0

        # CPU
        ttk.Label(hw_grid, text="CPU:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=row, column=0, sticky='w', padx=(0, 5))
        ttk.Label(hw_grid, text=f"{self.hw_config['n_cores']} cores").grid(
            row=row, column=1, sticky='w')
        row += 1

        # RAM
        ttk.Label(hw_grid, text="RAM:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=row, column=0, sticky='w', padx=(0, 5))
        ttk.Label(hw_grid, text=f"{self.hw_config['memory_gb']:.1f} GB").grid(
            row=row, column=1, sticky='w')
        row += 1

        # GPU
        ttk.Label(hw_grid, text="GPU:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=row, column=0, sticky='w', padx=(0, 5))
        gpu_status = f"✓ {self.hw_config['gpu_type']}" if self.hw_config['gpu_available'] else "✗ Not available"
        gpu_color = 'green' if self.hw_config['gpu_available'] else 'red'
        ttk.Label(hw_grid, text=gpu_status, foreground=gpu_color).grid(
            row=row, column=1, sticky='w')
        row += 1

        # Performance Tier
        ttk.Label(hw_grid, text="Tier:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=row, column=0, sticky='w', padx=(0, 5))
        tier_names = {1: "Standard", 2: "Parallel", 3: "Power"}
        tier_name = tier_names.get(self.hw_config['tier'], "Unknown")
        ttk.Label(hw_grid, text=f"Tier {self.hw_config['tier']} - {tier_name}").grid(
            row=row, column=1, sticky='w')

        # ============================================================
        # RUNTIME ESTIMATE
        # ============================================================
        estimate_frame = ttk.LabelFrame(main_frame, text="⏱️ Estimated Runtime", padding="10")
        estimate_frame.pack(fill='x', pady=(0, 10))

        self.estimate_label = ttk.Label(estimate_frame, text="",
                                       font=('TkDefaultFont', 11))
        self.estimate_label.pack(anchor='w')

        self.speedup_label = ttk.Label(estimate_frame, text="",
                                      foreground='green', font=('TkDefaultFont', 10))
        self.speedup_label.pack(anchor='w')

        ttk.Label(estimate_frame, text="(For typical 3,000 model analysis run)",
                 foreground='gray', font=('TkDefaultFont', 9)).pack(anchor='w')

        # ============================================================
        # ACTION BUTTONS
        # ============================================================
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        ttk.Button(button_frame, text="💾 Save as Default",
                  command=self._save_preferences).pack(side='left', padx=(0, 5))

        ttk.Button(button_frame, text="🔄 Reset to Auto",
                  command=self._reset_to_auto).pack(side='left')

        # Status label
        self.status_label = ttk.Label(main_frame, text="", foreground='blue')
        self.status_label.pack(pady=(5, 0))

    def _toggle_advanced(self):
        """Show/hide advanced settings"""
        if self.show_advanced.get():
            self.advanced_frame.pack(fill='x', pady=(5, 0))
        else:
            self.advanced_frame.pack_forget()

    def _on_mode_change(self, *args):
        """Handle mode selection change"""
        mode = self.mode_var.get()

        # Update advanced settings based on mode
        mode_settings = {
            'auto': {'cpu': 60, 'gpu': self.hw_config['gpu_available'], 'parallel': True},
            'power': {'cpu': 100, 'gpu': self.hw_config['gpu_available'], 'parallel': True},
            'balanced': {'cpu': 60, 'gpu': self.hw_config['gpu_available'], 'parallel': True},
            'light': {'cpu': 30, 'gpu': False, 'parallel': False},
        }

        if mode in mode_settings:
            settings = mode_settings[mode]
            self.cpu_slider.set(settings['cpu'])
            if self.hw_config['gpu_available']:
                self.gpu_var.set(settings['gpu'])
            self.parallel_var.set(settings['parallel'])

        self._update_all()

    def _on_cpu_change(self, value):
        """Handle CPU slider change"""
        cpu_percent = int(float(value))
        self.cpu_value_label.config(text=f"{cpu_percent}%")

        # Calculate cores
        n_cores = self.hw_config['n_cores']
        n_workers = max(1, int(n_cores * cpu_percent / 100))
        n_workers = min(n_workers, n_cores - 1)  # Leave 1 for OS

        self.cpu_cores_label.config(text=f"Using {n_workers} of {n_cores} cores")

        self._update_all()

    def _on_setting_change(self, *args):
        """Handle any setting change"""
        self._update_all()

    def _update_all(self):
        """Update all dynamic elements"""
        self._update_cpu_display()
        self._update_estimate()

    def _update_cpu_display(self):
        """Update CPU cores display"""
        cpu_percent = int(self.cpu_slider.get())
        n_cores = self.hw_config['n_cores']
        n_workers = max(1, int(n_cores * cpu_percent / 100))
        n_workers = min(n_workers, n_cores - 1)

        self.cpu_value_label.config(text=f"{cpu_percent}%")
        self.cpu_cores_label.config(text=f"Using {n_workers} of {n_cores} cores")

    def _update_estimate(self):
        """Update runtime estimate"""
        mode = self.mode_var.get()
        use_gpu = self.gpu_var.get() and self.hw_config['gpu_available']
        use_parallel = self.parallel_var.get()
        cpu_percent = int(self.cpu_slider.get())
        n_cores = self.hw_config['n_cores']

        # Calculate estimated runtime
        if use_gpu and use_parallel and cpu_percent >= 80 and n_cores >= 8:
            estimate = "30-45 minutes"
            speedup = "8-10x faster"
        elif use_gpu and use_parallel and cpu_percent >= 50:
            estimate = "45-60 minutes"
            speedup = "5-7x faster"
        elif use_gpu and use_parallel:
            estimate = "60-90 minutes"
            speedup = "3-5x faster"
        elif use_gpu:
            estimate = "2-2.5 hours"
            speedup = "2-2.5x faster"
        elif use_parallel and cpu_percent >= 50:
            estimate = "2.5-3 hours"
            speedup = "1.7-2x faster"
        else:
            estimate = "3-5 hours"
            speedup = "Similar to baseline"

        self.estimate_label.config(text=f"Estimated: {estimate}")
        self.speedup_label.config(text=f"Speedup: {speedup}")

    def _show_help(self):
        """Show help dialog"""
        help_text = """Performance Settings Help

MODES:
• Auto: Automatically detects your hardware and uses optimal settings
• Balanced: Good speed while leaving resources for multitasking (recommended)
• Power: Maximum speed using all CPU + GPU (for dedicated analysis runs)
• Light: Minimal resource usage, can work normally while analyzing

ADVANCED SETTINGS:
• CPU Usage: Control how much of your CPU to use (10-100%)
• GPU Acceleration: Uses GPU for 15-20x faster boosting models
• Parallel Grid: Tests multiple models simultaneously for 4-8x additional speedup

TIPS:
• For overnight runs: Use Power Mode
• For daytime work: Use Balanced Mode
• Gaming while analyzing: Disable GPU to free it for games
• Small datasets: Disable Parallel Grid (overhead not worth it)

HARDWARE REQUIREMENTS:
• GPU acceleration requires NVIDIA GPU with CUDA support
• Parallel grid benefits most with 8+ CPU cores
• Works on any hardware - automatically adjusts settings
"""
        messagebox.showinfo("Performance Settings Help", help_text)

    def _save_preferences(self):
        """Save current settings as default"""
        config = self.get_config()
        config.save_preferences()
        self.status_label.config(text="✓ Settings saved as default", foreground='green')
        self.after(3000, lambda: self.status_label.config(text=""))

    def _reset_to_auto(self):
        """Reset to auto mode"""
        self.mode_var.set('auto')
        self._on_mode_change()
        self.status_label.config(text="✓ Reset to auto mode", foreground='blue')
        self.after(3000, lambda: self.status_label.config(text=""))

    def get_config(self):
        """
        Get PerformanceConfig from current UI settings.

        Returns
        -------
        config : PerformanceConfig
            Configuration object ready to pass to run_search()
        """
        mode = self.mode_var.get()

        if mode == 'auto':
            return PerformanceConfig(mode='auto')
        elif mode in ['power', 'balanced', 'light']:
            # Use predefined mode but allow advanced overrides if shown
            if self.show_advanced.get():
                # Custom mode with user settings
                return PerformanceConfig(
                    mode='custom',
                    max_cpu_percent=int(self.cpu_slider.get()),
                    use_gpu=self.gpu_var.get(),
                    parallel_grid=self.parallel_var.get()
                )
            else:
                # Use predefined mode
                return PerformanceConfig(mode=mode)
        else:
            # Custom mode
            return PerformanceConfig(
                mode='custom',
                max_cpu_percent=int(self.cpu_slider.get()),
                use_gpu=self.gpu_var.get(),
                parallel_grid=self.parallel_var.get()
            )


# ============================================================
# STANDALONE TEST/DEMO
# ============================================================

class PerformanceSettingsDemo(tk.Tk):
    """Standalone demo window for testing the settings panel"""

    def __init__(self):
        super().__init__()

        self.title("Performance Settings Panel - Demo")
        self.geometry("600x700")

        # Create settings panel
        self.settings_panel = PerformanceSettingsPanel(self)
        self.settings_panel.pack(fill='both', expand=True, padx=10, pady=10)

        # Add test button
        test_frame = ttk.Frame(self)
        test_frame.pack(fill='x', padx=10, pady=(0, 10))

        ttk.Button(test_frame, text="Test: Get Config",
                  command=self._test_get_config).pack(side='left', padx=5)

        ttk.Button(test_frame, text="Test: Print Summary",
                  command=self._test_print_summary).pack(side='left', padx=5)

        # Output text area
        output_frame = ttk.LabelFrame(self, text="Test Output", padding="10")
        output_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.output_text = tk.Text(output_frame, height=8, wrap='word')
        self.output_text.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.output_text.config(yscrollcommand=scrollbar.set)

    def _test_get_config(self):
        """Test getting config from panel"""
        config = self.settings_panel.get_config()

        output = f"""PerformanceConfig Retrieved:
═══════════════════════════
Mode: {config.mode}
CPU Usage: {config.max_cpu_percent}%
Workers: {config.n_workers}
GPU Enabled: {config.use_gpu}
Parallel Grid: {config.parallel_grid}

Ready to pass to run_search():
    results = run_search(X, y, perf_config=config, ...)
"""

        self.output_text.delete('1.0', 'end')
        self.output_text.insert('1.0', output)

    def _test_print_summary(self):
        """Test printing config summary"""
        import io
        import sys

        config = self.settings_panel.get_config()

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        config.print_summary()

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        self.output_text.delete('1.0', 'end')
        self.output_text.insert('1.0', output)


if __name__ == '__main__':
    # Run standalone demo
    app = PerformanceSettingsDemo()
    app.mainloop()
