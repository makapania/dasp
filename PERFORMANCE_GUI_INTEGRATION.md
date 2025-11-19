# Performance Settings GUI - Integration Guide

## Quick Start

### Run Standalone Demo
```bash
python performance_settings_gui.py
```

This opens a standalone window where you can:
- Test all settings
- See hardware detection
- Get runtime estimates
- Export PerformanceConfig

## Integration into Existing GUI

### Option 1: Simple Integration (Recommended)

Add the settings panel to your existing GUI:

```python
from performance_settings_gui import PerformanceSettingsPanel

class YourMainGUI:
    def __init__(self):
        # ... your existing setup ...

        # Add performance settings (in settings tab, dialog, or main window)
        self.perf_settings = PerformanceSettingsPanel(self.settings_tab)
        self.perf_settings.pack(fill='both', expand=True)

    def run_analysis_button_click(self):
        # Get config from GUI panel
        perf_config = self.perf_settings.get_config()

        # Pass to run_search
        results = run_search(
            X=self.X_train,
            y=self.y_train,
            task_type=self.task_type,
            perf_config=perf_config,  # ← From GUI
            # ... all other existing parameters
        )
```

### Option 2: Popup Dialog

Create settings dialog window:

```python
import tkinter as tk
from tkinter import ttk
from performance_settings_gui import PerformanceSettingsPanel

class PerformanceSettingsDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Performance Settings")
        self.geometry("600x700")

        # Settings panel
        self.panel = PerformanceSettingsPanel(self)
        self.panel.pack(fill='both', expand=True, padx=10, pady=10)

        # OK/Cancel buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=(0, 10))

        ttk.Button(button_frame, text="OK", command=self.ok).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side='right')

        self.result = None

    def ok(self):
        self.result = self.panel.get_config()
        self.destroy()

# Usage in main GUI:
def show_performance_settings():
    dialog = PerformanceSettingsDialog(self.root)
    dialog.wait_window()

    if dialog.result:
        self.perf_config = dialog.result
        print(f"Settings updated: {self.perf_config}")
```

### Option 3: Menu Integration

Add to menu bar:

```python
# In menu creation:
settings_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Settings", menu=settings_menu)
settings_menu.add_command(label="Performance Settings...", command=self.show_perf_settings)

def show_perf_settings(self):
    dialog = PerformanceSettingsDialog(self.root)
    self.root.wait_window(dialog)
    if dialog.result:
        self.perf_config = dialog.result
```

## Features

### Mode Selection
- **Auto**: Detects hardware, picks best mode
- **Balanced**: 60% CPU + GPU (recommended for multitasking)
- **Power**: 100% CPU + GPU (maximum speed)
- **Light**: 30% CPU, no GPU (background processing)

### Advanced Settings (Collapsible)
- **CPU Usage Slider**: 10-100% with core count display
- **GPU Checkbox**: Enable/disable GPU (auto-detects availability)
- **Parallel Grid**: Enable/disable parallel model testing

### Hardware Info Display
- CPU cores detected
- RAM detected
- GPU detected (type if available)
- Performance tier (1-3)

### Runtime Estimation
- Real-time estimate based on current settings
- Speedup vs baseline
- Updates as settings change

### Persistence
- Save preferences button
- Auto-loads saved preferences on startup
- Reset to auto button

## Example: Full Integration

```python
import tkinter as tk
from tkinter import ttk
from performance_settings_gui import PerformanceSettingsPanel
from spectral_predict.search import run_search

class SpectralAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spectral Analysis")

        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)

        # Data tab
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="Data")
        # ... your data loading UI ...

        # Analysis tab
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Analysis")
        # ... your analysis parameters ...

        # Performance tab
        perf_tab = ttk.Frame(notebook)
        notebook.add(perf_tab, text="⚡ Performance")

        # Add performance settings panel
        self.perf_settings = PerformanceSettingsPanel(perf_tab)
        self.perf_settings.pack(fill='both', expand=True, padx=10, pady=10)

        # Results tab
        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="Results")
        # ... your results display ...

        # Run button (in analysis tab)
        ttk.Button(analysis_tab, text="Run Analysis",
                  command=self.run_analysis).pack()

    def run_analysis(self):
        """Run analysis with performance config from GUI"""

        # Get performance config
        perf_config = self.perf_settings.get_config()

        # Show what we're using
        print(f"Running analysis with:")
        print(f"  Mode: {perf_config.mode_name}")
        print(f"  CPU: {perf_config.max_cpu_percent}% ({perf_config.n_workers} workers)")
        print(f"  GPU: {perf_config.use_gpu}")
        print(f"  Parallel Grid: {perf_config.parallel_grid}")

        # Run search with config
        results = run_search(
            X=self.X_train,
            y=self.y_train,
            task_type=self.task_type,
            perf_config=perf_config,  # ← From GUI
            folds=self.folds,
            models_to_test=self.selected_models,
            # ... other parameters
        )

        # Display results
        self.display_results(results)

if __name__ == '__main__':
    app = SpectralAnalysisGUI()
    app.mainloop()
```

## Minimal Integration (No GUI Changes)

If you don't want to modify your GUI, just use auto mode:

```python
from performance_config import PerformanceConfig

# At the top of your analysis function:
perf_config = PerformanceConfig(mode='auto')

# Pass to run_search:
results = run_search(X, y, perf_config=perf_config, ...)
```

This gives automatic GPU + parallel acceleration without any GUI changes.

## Testing

Run the standalone demo:
```bash
python performance_settings_gui.py
```

Test buttons:
- **Test: Get Config** - Shows PerformanceConfig object
- **Test: Print Summary** - Shows formatted summary

## Customization

### Change Colors/Fonts

```python
# Modify in _create_widgets():
title_label = ttk.Label(title_frame, text="⚡ Performance Settings",
                       font=('Arial', 16, 'bold'),  # Custom font
                       foreground='blue')  # Custom color
```

### Add Custom Modes

```python
# In _create_widgets(), add to modes list:
modes = [
    # ... existing modes ...
    ('custom_fast', '⚡ Custom Fast',
     'Your custom mode description  (estimated time)')
]

# In _on_mode_change(), add settings:
mode_settings = {
    # ... existing settings ...
    'custom_fast': {'cpu': 80, 'gpu': True, 'parallel': True}
}
```

### Modify Estimates

```python
# In _update_estimate(), adjust estimates:
if use_gpu and use_parallel and cpu_percent >= 80 and n_cores >= 8:
    estimate = "20-30 minutes"  # Your custom estimate
    speedup = "10-15x faster"
```

## Troubleshooting

### "Module not found: performance_config"
Make sure `performance_config.py` and `hardware_detection.py` are in the same directory or in Python path.

### GUI doesn't show GPU
- GPU detection failed or not available
- Install CUDA and GPU-enabled XGBoost
- Run standalone demo to see detection messages

### Settings not saving
- Check write permissions in home directory
- Preferences saved to `~/.dasp_performance.json`
- Check console for error messages

### Runtime estimates seem off
- Estimates are for typical 3,000 model runs
- Adjust estimates in `_update_estimate()` based on your data
- Actual runtime depends on dataset size and model complexity

## API Reference

### PerformanceSettingsPanel

**Methods:**
- `get_config()` → Returns PerformanceConfig object
- `_save_preferences()` → Saves current settings to disk
- `_reset_to_auto()` → Resets to auto mode

**Properties:**
- `mode_var` → Current mode (StringVar)
- `cpu_slider` → CPU usage slider (Scale)
- `gpu_var` → GPU enabled (BooleanVar)
- `parallel_var` → Parallel grid enabled (BooleanVar)
- `hw_config` → Detected hardware (dict)

**Events:**
- Mode changes automatically update advanced settings
- Any setting change updates runtime estimate
- Settings persist on save

## Best Practices

1. **Add to settings tab** - Keeps main UI clean
2. **Load preferences on startup** - Remembers user choices
3. **Show estimate** - Helps users understand impact
4. **Provide help button** - Explains options
5. **Default to balanced** - Good for most users
6. **Allow reset** - Easy to recover from mistakes

## Support

For issues or questions:
1. Run standalone demo to test panel in isolation
2. Check `PERFORMANCE_OPTIMIZATION_COMPLETE.md` for details
3. Review `GPU_ACCELERATION_COMPLETE.md` for GPU setup
4. Test with: `python -c "from performance_config import PerformanceConfig; PerformanceConfig(mode='auto').print_summary()"`
