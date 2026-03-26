# Plan: Fix Window Icon Shows Feather Instead of Program Logo

## Status: ✅ COMPLETED

## Problem
The bundled PyInstaller app shows the default Python/feather icon in:
1. The Windows taskbar
2. The top-left corner of the window title bar

The actual program logo (`asp_logo.ico`) should be displayed instead.

## Root Cause
1. The `asp_logo.ico` file is used for the .exe file icon (spec line 337: `icon='asp_logo.ico'`), but this only affects Windows Explorer - not the running window
2. No `iconbitmap()` call exists in the GUI code to set the window/taskbar icon
3. The .ico file is NOT bundled as a data file for runtime access

## Solution - Two Changes

### 1. ✅ Bundle the .ico file for runtime access
**File:** `spectral_predict.spec` - line 293

Added `asp_logo.ico` to the datas list:
```python
datas=[
    # Bundle the source modules
    ('src/spectral_predict', 'src/spectral_predict'),
    # Bundle logo files
    ('asp_logo_final.png', '.'),
    ('asp_logo.ico', '.'),  # Window/taskbar icon
    # Bundle example data (for testing)
    ('example/BoneCollagen.csv', 'example'),
] + all_datas,
```

### 2. ✅ Set window icon in main()
**File:** `spectral_predict_gui_optimized.py` - lines 41415-41422

Added icon setting after `root = tk.Tk()`:
```python
def main():
    """Main entry point."""
    root = tk.Tk()

    # Set window icon (taskbar and title bar)
    try:
        from src.spectral_predict.resource_paths import get_resource_path
        icon_path = get_resource_path('asp_logo.ico')
        if icon_path.exists():
            root.iconbitmap(str(icon_path))
    except Exception:
        pass  # Fail silently if icon can't be loaded

    app = SpectralPredictApp(root)
    # ... rest of function
```

## Verification Steps
1. Rebuild the bundle:
   ```bash
   .venv311/Scripts/python.exe -m PyInstaller spectral_predict.spec --clean -y
   ```

2. Run the bundled app:
   ```bash
   dist/SpectralPredict/SpectralPredict.exe
   ```

3. Verify:
   - Taskbar shows the ASP logo (not feather)
   - Window title bar (top-left) shows the ASP logo
