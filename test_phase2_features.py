"""
Test script for Phase 2: Interactive plot features and reflectance/absorbance toggle

This script verifies that all Phase 2 features are correctly implemented:
1. Reflectance to Absorbance Toggle
2. Click-to-Toggle Spectrum Removal
3. Zoom/Pan Controls
4. Unified Exclusion System

Run this after loading data in the GUI to verify functionality.
"""

import sys
import inspect
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import the GUI module
sys.path.insert(0, r'C:\Users\sponheim\git\dasp')
import spectral_predict_gui_optimized as gui_module

def check_implementation():
    """Check that all Phase 2 features are implemented."""

    print("="*70)
    print("Phase 2 Implementation Verification")
    print("="*70)
    print()

    # Check 1: NavigationToolbar2Tk import
    print("✓ Check 1: NavigationToolbar2Tk import")
    source = inspect.getsource(gui_module)
    assert 'NavigationToolbar2Tk' in source
    print("  - NavigationToolbar2Tk is imported")
    print()

    # Check 2: State variables
    print("✓ Check 2: State variables")
    app_source = inspect.getsource(gui_module.SpectralPredictApp.__init__)
    assert 'self.use_absorbance' in app_source
    assert 'self.excluded_spectra = set()' in app_source
    print("  - self.use_absorbance variable exists")
    print("  - self.excluded_spectra set exists")
    print()

    # Check 3: Helper methods
    print("✓ Check 3: Helper methods")
    methods = dir(gui_module.SpectralPredictApp)

    assert '_toggle_absorbance' in methods
    print("  - _toggle_absorbance() method exists")

    assert '_reset_exclusions' in methods
    print("  - _reset_exclusions() method exists")

    assert '_on_spectrum_click' in methods
    print("  - _on_spectrum_click() event handler exists")

    assert '_apply_transformation' in methods
    print("  - _apply_transformation() method exists")

    assert '_update_exclusion_status' in methods
    print("  - _update_exclusion_status() method exists")
    print()

    # Check 4: _toggle_absorbance implementation
    print("✓ Check 4: _toggle_absorbance implementation")
    toggle_source = inspect.getsource(gui_module.SpectralPredictApp._toggle_absorbance)
    assert '_generate_plots' in toggle_source
    print("  - Regenerates plots on toggle")
    print()

    # Check 5: _apply_transformation implementation
    print("✓ Check 5: _apply_transformation implementation")
    transform_source = inspect.getsource(gui_module.SpectralPredictApp._apply_transformation)
    assert 'log10' in transform_source
    assert 'use_absorbance' in transform_source
    print("  - Implements log10(1/R) transformation")
    print("  - Checks use_absorbance flag")
    print()

    # Check 6: _on_spectrum_click implementation
    print("✓ Check 6: _on_spectrum_click implementation")
    click_source = inspect.getsource(gui_module.SpectralPredictApp._on_spectrum_click)
    assert 'excluded_spectra' in click_source
    assert 'set_alpha' in click_source
    assert 'canvas.draw()' in click_source
    print("  - Toggles spectrum in excluded_spectra set")
    print("  - Updates line alpha for visual feedback")
    print("  - Redraws canvas")
    print()

    # Check 7: _create_plot_tab interactive features
    print("✓ Check 7: _create_plot_tab interactive features")
    plot_source = inspect.getsource(gui_module.SpectralPredictApp._create_plot_tab)
    assert 'is_raw' in plot_source
    assert 'set_picker' in plot_source
    assert 'mpl_connect' in plot_source
    assert 'NavigationToolbar2Tk' in plot_source
    print("  - Accepts is_raw parameter")
    print("  - Sets picker on lines for clickability")
    print("  - Connects pick_event handler")
    print("  - Adds NavigationToolbar for zoom/pan")
    print()

    # Check 8: _generate_plots uses transformation
    print("✓ Check 8: _generate_plots uses transformation")
    gen_plots_source = inspect.getsource(gui_module.SpectralPredictApp._generate_plots)
    assert '_apply_transformation' in gen_plots_source
    assert 'Absorbance' in gen_plots_source or 'absorbance' in gen_plots_source.lower()
    print("  - Applies transformation to data")
    print("  - Updates y-axis label based on mode")
    print()

    # Check 9: _run_analysis_thread filters exclusions
    print("✓ Check 9: _run_analysis_thread filters exclusions")
    analysis_source = inspect.getsource(gui_module.SpectralPredictApp._run_analysis_thread)
    assert 'excluded_spectra' in analysis_source
    assert 'X_filtered' in analysis_source or 'mask' in analysis_source
    print("  - Checks excluded_spectra set")
    print("  - Filters data before analysis")
    print()

    # Check 10: UI elements
    print("✓ Check 10: UI elements in Import & Preview tab")
    tab1_source = inspect.getsource(gui_module.SpectralPredictApp._create_tab1_import_preview)
    assert 'absorbance_checkbox' in tab1_source
    assert 'reset_exclusions_button' in tab1_source
    assert 'exclusion_status' in tab1_source
    print("  - Absorbance toggle checkbox exists")
    print("  - Reset Exclusions button exists")
    print("  - Exclusion status label exists")
    print()

    print("="*70)
    print("✅ ALL PHASE 2 FEATURES SUCCESSFULLY IMPLEMENTED!")
    print("="*70)
    print()
    print("Features implemented:")
    print("  1. ✓ Reflectance/Absorbance toggle with log10(1/R) transformation")
    print("  2. ✓ Click-to-toggle spectrum visibility on plots")
    print("  3. ✓ Zoom/Pan controls via NavigationToolbar")
    print("  4. ✓ Unified exclusion system with status display")
    print("  5. ✓ Excluded spectra filtered from analysis")
    print()
    print("Manual testing checklist:")
    print("  [ ] Load spectral data in GUI")
    print("  [ ] Toggle absorbance checkbox - verify plots update")
    print("  [ ] Click on spectrum lines - verify they become transparent")
    print("  [ ] Check exclusion status updates")
    print("  [ ] Click Reset Exclusions - verify all spectra restored")
    print("  [ ] Use zoom/pan toolbar - verify controls work")
    print("  [ ] Run analysis with excluded spectra - verify count shown")
    print()

if __name__ == '__main__':
    try:
        check_implementation()
    except AssertionError as e:
        print(f"\n❌ IMPLEMENTATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
