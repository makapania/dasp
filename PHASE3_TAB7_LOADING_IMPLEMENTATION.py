"""
PHASE 3: Tab 7 Model Loading Engine Implementation
====================================================

This module contains the complete implementation for loading model configurations
from the Results tab into the NEW Tab 7 (Model Development).

CRITICAL DIFFERENCES FROM TAB 6:
- Uses tab7_ namespace (not refine_)
- Fresh implementation for NEW Tab 7 placeholder
- FAIL LOUD wavelength validation (no silent fallbacks)
- Ready for integration into spectral_predict_gui_optimized.py

INTEGRATION INSTRUCTIONS:
1. Add these methods to the SpectralPredictApp class
2. Update _on_result_double_click to call _load_model_to_NEW_tab7
3. Create Tab 7 UI controls with tab7_ prefix (future phase)
4. Test with subset models to verify fail-loud wavelength validation

Author: Claude Code Agent 3
Date: 2025-11-07
"""

import ast
import pandas as pd
from tkinter import messagebox
import tkinter as tk


# ============================================================================
# METHOD 1: Main Loading Logic - _load_model_to_NEW_tab7
# ============================================================================

def _load_model_to_NEW_tab7(self, config):
    """
    Load a model configuration from Results tab into NEW Tab 7 (Model Development).

    This method implements robust data transfer with comprehensive error handling
    and FAIL LOUD validation for wavelength loading.

    CRITICAL: This is for the NEW Tab 7, not the existing Tab 6!
    Uses tab7_ namespace for all GUI controls.

    Args:
        config (dict): Model configuration dictionary from results DataFrame

    Critical Features:
        - Validates data availability before loading
        - FAIL LOUD wavelength validation (never silent fallback!)
        - Comprehensive error messages for debugging
        - Handles field name variations for hyperparameters
        - Supports both regression and classification models

    Raises:
        ValueError: If critical data is missing or invalid
        RuntimeError: If data validation fails
    """
    print("\n" + "="*80)
    print("LOADING MODEL INTO NEW TAB 7 - MODEL DEVELOPMENT")
    print("="*80)

    # ========================================================================
    # STEP 1: Validate Data Availability
    # ========================================================================
    print("\n[STEP 1/7] Validating data availability...")

    # Check if X_original exists
    if self.X_original is None:
        raise RuntimeError(
            "Data validation failed!\n"
            "X_original is not available.\n"
            "Please ensure spectral data is loaded in the Data Upload tab."
        )

    # Check if y exists
    if self.y is None:
        raise RuntimeError(
            "Data validation failed!\n"
            "Target variable (y) is not available.\n"
            "Please ensure data is loaded in the Data Upload tab."
        )

    # Check if wavelengths exist
    if len(self.X_original.columns) == 0:
        raise RuntimeError(
            "Data validation failed!\n"
            "No wavelengths found in X_original.\n"
            "Please ensure spectral data is properly loaded."
        )

    print("✓ Data validation passed: X, y, and wavelengths are available")
    print(f"  - X shape: {self.X_original.shape}")
    print(f"  - y length: {len(self.y)}")
    print(f"  - Available wavelengths: {len(self.X_original.columns)}")

    # ========================================================================
    # STEP 2: Build Configuration Info Text
    # ========================================================================
    print("\n[STEP 2/7] Building configuration information display...")

    # Extract basic info
    model_name = config.get('Model', 'N/A')
    rank = config.get('Rank', 'N/A')
    preprocess = config.get('Preprocess', 'N/A')
    subset_tag = config.get('SubsetTag', config.get('Subset', 'full'))
    window = config.get('Window', 'N/A')

    info_text = f"""Model: {model_name} (Rank {rank})
Preprocessing: {preprocess}
Subset: {subset_tag}
Window Size: {window}
"""

    # Add performance metrics
    if 'RMSE' in config and not pd.isna(config.get('RMSE')):
        # Regression model
        rmse = config.get('RMSE', 'N/A')
        r2 = config.get('R2', 'N/A')
        info_text += f"""
Performance (Regression):
  RMSE: {rmse}
  R²: {r2}
"""
        print(f"✓ Regression model: RMSE={rmse}, R²={r2}")
    elif 'Accuracy' in config and not pd.isna(config.get('Accuracy')):
        # Classification model
        accuracy = config.get('Accuracy', 'N/A')
        info_text += f"""
Performance (Classification):
  Accuracy: {accuracy}
"""
        if 'ROC_AUC' in config and not pd.isna(config['ROC_AUC']):
            roc_auc = config.get('ROC_AUC', 'N/A')
            info_text += f"  ROC AUC: {roc_auc}\n"
        print(f"✓ Classification model: Accuracy={accuracy}")

    # Add wavelength count info
    n_vars = config.get('n_vars', 'N/A')
    full_vars = config.get('full_vars', 'N/A')
    info_text += f"\nWavelengths: {n_vars} of {full_vars} used"
    if subset_tag not in ['full', 'N/A']:
        info_text += f" ({subset_tag})"
    info_text += "\n"

    print(f"✓ Configuration text built: {model_name}, {n_vars} wavelengths")

    # ========================================================================
    # STEP 3: CRITICAL - Load Wavelengths with FAIL LOUD Validation
    # ========================================================================
    print("\n[STEP 3/7] Loading wavelengths with strict validation...")
    print("⚠️  CRITICAL SECTION: FAIL LOUD validation - no silent fallbacks!")

    model_wavelengths = None
    all_wavelengths = self.X_original.columns.astype(float).values

    # Check if this is a subset model or full model
    is_subset_model = (subset_tag not in ['full', 'N/A'])

    if is_subset_model:
        print(f"  Subset model detected: '{subset_tag}' with {n_vars} variables")

        # CRITICAL: For subset models, we MUST have all_vars field
        if 'all_vars' not in config or not config['all_vars'] or config['all_vars'] == 'N/A':
            raise ValueError(
                f"CRITICAL ERROR: Missing 'all_vars' field for subset model!\n"
                f"  Model: {model_name} (Rank {rank})\n"
                f"  Subset: {subset_tag}\n"
                f"  Expected variables: {n_vars}\n\n"
                f"The 'all_vars' field is REQUIRED for subset models to identify\n"
                f"the exact wavelengths used. Without it, loading this model\n"
                f"would cause R² discrepancies.\n\n"
                f"SOLUTION: Re-run the analysis to generate complete results."
            )

        # Parse all_vars field
        print(f"  Parsing 'all_vars' field...")
        try:
            all_vars_str = str(config['all_vars']).strip()
            wavelength_strings = [w.strip() for w in all_vars_str.split(',')]
            parsed_wavelengths = [float(w) for w in wavelength_strings if w]

            print(f"  ✓ Parsed {len(parsed_wavelengths)} wavelengths from 'all_vars'")

            # CRITICAL VALIDATION: Count must match n_vars
            expected_count = int(n_vars) if n_vars != 'N/A' else len(parsed_wavelengths)

            if len(parsed_wavelengths) != expected_count:
                raise ValueError(
                    f"CRITICAL ERROR: Wavelength count mismatch!\n"
                    f"  Model: {model_name} (Rank {rank})\n"
                    f"  Expected: {expected_count} wavelengths (from n_vars field)\n"
                    f"  Parsed: {len(parsed_wavelengths)} wavelengths (from all_vars field)\n"
                    f"  Subset: {subset_tag}\n\n"
                    f"This indicates a data integrity issue in the results table.\n"
                    f"The 'all_vars' field does not match the 'n_vars' count.\n\n"
                    f"SOLUTION: Re-run the analysis to generate consistent results."
                )

            # Validate that all parsed wavelengths exist in available data
            available_wl_set = set(all_wavelengths)
            invalid_wls = [w for w in parsed_wavelengths if w not in available_wl_set]

            if invalid_wls:
                raise ValueError(
                    f"CRITICAL ERROR: Invalid wavelengths in 'all_vars'!\n"
                    f"  Model: {model_name} (Rank {rank})\n"
                    f"  Found {len(invalid_wls)} wavelengths not in current dataset:\n"
                    f"  {invalid_wls[:10]}{'...' if len(invalid_wls) > 10 else ''}\n\n"
                    f"This likely means the current dataset is different from\n"
                    f"the one used to generate these results.\n\n"
                    f"SOLUTION: Load the original dataset or re-run the analysis."
                )

            model_wavelengths = sorted(parsed_wavelengths)
            print(f"  ✓ Validation passed: All {len(model_wavelengths)} wavelengths are valid")

        except ValueError as ve:
            # Re-raise ValueError (our validation errors)
            raise
        except Exception as e:
            raise ValueError(
                f"CRITICAL ERROR: Failed to parse 'all_vars' field!\n"
                f"  Model: {model_name} (Rank {rank})\n"
                f"  Error: {str(e)}\n"
                f"  all_vars content: {str(config.get('all_vars', 'MISSING'))[:200]}...\n\n"
                f"The wavelength data in the results table is malformed.\n\n"
                f"SOLUTION: Re-run the analysis to generate valid results."
            )

    else:
        # Full model - use all available wavelengths
        print(f"  Full model detected - using all {len(all_wavelengths)} wavelengths")
        model_wavelengths = list(all_wavelengths)

        # Validate count matches (optional check for full models)
        if n_vars != 'N/A' and int(n_vars) != len(all_wavelengths):
            print(f"  ⚠️  WARNING: n_vars ({n_vars}) doesn't match available wavelengths ({len(all_wavelengths)})")
            print(f"      This may indicate the dataset has changed since results were generated")

    # Final validation: ensure we have wavelengths
    if model_wavelengths is None or len(model_wavelengths) == 0:
        raise RuntimeError(
            f"CRITICAL ERROR: No wavelengths loaded!\n"
            f"  Model: {model_name} (Rank {rank})\n"
            f"  Subset: {subset_tag}\n"
            f"  This should never happen - indicates a logic error in loading code."
        )

    print(f"✓ Wavelength loading complete: {len(model_wavelengths)} wavelengths validated")

    # ========================================================================
    # STEP 4: Format Wavelengths for Display
    # ========================================================================
    print("\n[STEP 4/7] Formatting wavelengths for display...")

    try:
        # Use the formatting helper
        wl_display_text = self._format_wavelengths_for_NEW_tab7(model_wavelengths)
        print(f"✓ Formatted {len(model_wavelengths)} wavelengths ({len(wl_display_text)} characters)")

    except Exception as e:
        print(f"⚠️  WARNING: Wavelength formatting failed: {e}")
        # Fallback to simple comma-separated list
        wl_display_text = ", ".join([f"{w:.1f}" for w in model_wavelengths[:100]])
        if len(model_wavelengths) > 100:
            wl_display_text += f", ... ({len(model_wavelengths) - 100} more)"

    # Update wavelength specification widget (Tab 7 namespace)
    self.tab7_wl_spec.config(state='normal')
    self.tab7_wl_spec.delete('1.0', 'end')
    self.tab7_wl_spec.insert('1.0', wl_display_text)

    # Verify insertion
    content = self.tab7_wl_spec.get('1.0', 'end-1c')
    if len(content) == 0:
        raise RuntimeError("ERROR: Tab 7 wavelength text widget is empty after insertion!")

    print(f"✓ Tab 7 wavelength widget updated: {len(content)} characters inserted")

    # ========================================================================
    # STEP 5: Load Hyperparameters from Config
    # ========================================================================
    print("\n[STEP 5/7] Loading hyperparameters...")

    # Helper function to get config value with field name variations
    def _get_config_value(keys):
        """Get config value, trying multiple possible field names."""
        for k in keys:
            if k in config:
                v = config.get(k)
                if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v) != 'N/A':
                    return v

        # Fallback: try to parse from Params field (legacy support)
        params_str = config.get('Params')
        if params_str:
            try:
                parsed = ast.literal_eval(params_str)
                if isinstance(parsed, dict):
                    for k in keys:
                        if k in parsed:
                            return parsed[k]
            except Exception:
                pass
        return None

    # Build hyperparameters display
    hyper_lines = []

    if model_name == 'PLS':
        lv_val = _get_config_value(['LVs', 'n_components', 'n_LVs'])
        if lv_val is not None:
            hyper_lines.append(f"  n_components (LVs): {lv_val}")

    elif model_name in ['Ridge', 'Lasso']:
        alpha_val = _get_config_value(['Alpha', 'alpha'])
        if alpha_val is not None:
            hyper_lines.append(f"  alpha: {alpha_val}")

    elif model_name == 'RandomForest':
        n_est = _get_config_value(['n_estimators', 'n_trees'])
        if n_est is not None:
            hyper_lines.append(f"  n_estimators: {n_est}")
        max_d = _get_config_value(['max_depth', 'MaxDepth'])
        if max_d is not None:
            hyper_lines.append(f"  max_depth: {max_d}")
        max_f = _get_config_value(['max_features'])
        if max_f is not None:
            hyper_lines.append(f"  max_features: {max_f}")

    elif model_name == 'MLP':
        hidden = _get_config_value(['Hidden', 'hidden_layer_sizes'])
        if hidden is not None:
            hyper_lines.append(f"  hidden_layer_sizes: {hidden}")
        lr_init = _get_config_value(['LR_init', 'learning_rate_init', 'learning_rate'])
        if lr_init is not None:
            hyper_lines.append(f"  learning_rate_init: {lr_init}")

    elif model_name == 'NeuralBoosted':
        n_est = _get_config_value(['n_estimators'])
        if n_est is not None:
            hyper_lines.append(f"  n_estimators: {n_est}")
        lr = _get_config_value(['LearningRate', 'learning_rate'])
        if lr is not None:
            hyper_lines.append(f"  learning_rate: {lr}")
        hidden_size = _get_config_value(['HiddenSize', 'hidden_layer_size'])
        if hidden_size is not None:
            hyper_lines.append(f"  hidden_layer_size: {hidden_size}")
        act = _get_config_value(['Activation', 'activation'])
        if act is not None:
            hyper_lines.append(f"  activation: {act}")

    if hyper_lines:
        info_text += "\nHyperparameters:\n" + "\n".join(hyper_lines) + "\n"
        print(f"✓ Loaded {len(hyper_lines)} hyperparameters")
    else:
        print("  (No hyperparameters found in config)")

    # ========================================================================
    # STEP 6: Populate GUI Controls (Tab 7 namespace)
    # ========================================================================
    print("\n[STEP 6/7] Populating Tab 7 GUI controls...")

    # Update info text display
    self.tab7_config_text.config(state='normal')
    self.tab7_config_text.delete('1.0', tk.END)
    self.tab7_config_text.insert('1.0', info_text)
    self.tab7_config_text.config(state='disabled')
    print("  ✓ Tab 7 config text display updated")

    # Set model type
    if model_name in ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']:
        self.tab7_model_type.set(model_name)
        print(f"  ✓ Model type: {model_name}")
    else:
        print(f"  ⚠️  WARNING: Unknown model type '{model_name}', defaulting to PLS")
        self.tab7_model_type.set('PLS')

    # Set task type (auto-detect from data)
    if self.y is not None:
        if self.y.nunique() == 2 or self.y.dtype == 'object' or self.y.nunique() < 10:
            self.tab7_task_type.set('classification')
            print("  ✓ Task type: classification (auto-detected)")
        else:
            self.tab7_task_type.set('regression')
            print("  ✓ Task type: regression (auto-detected)")

    # Set preprocessing method (handle various naming conventions)
    deriv = config.get('Deriv', None)

    # Convert from search.py naming to GUI naming
    if preprocess == 'deriv' and deriv == 1:
        gui_preprocess = 'sg1'
    elif preprocess == 'deriv' and deriv == 2:
        gui_preprocess = 'sg2'
    elif preprocess == 'snv_deriv':
        gui_preprocess = 'snv_sg1' if deriv == 1 else 'snv_sg2'
    elif preprocess == 'deriv_snv':
        gui_preprocess = 'deriv_snv'
    elif preprocess == 'msc_deriv':
        gui_preprocess = 'msc_sg1' if deriv == 1 else 'msc_sg2'
    elif preprocess == 'deriv_msc':
        gui_preprocess = 'deriv_msc'
    elif preprocess in ['raw', 'snv', 'msc']:
        gui_preprocess = preprocess
    else:
        print(f"  ⚠️  WARNING: Unknown preprocessing '{preprocess}', defaulting to 'raw'")
        gui_preprocess = 'raw'

    if gui_preprocess in ['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv',
                          'msc', 'msc_sg1', 'msc_sg2', 'deriv_msc']:
        self.tab7_preprocess.set(gui_preprocess)
        print(f"  ✓ Preprocessing: {gui_preprocess}")

    # Set window size
    if window != 'N/A' and not pd.isna(window):
        window_val = int(float(window))
        if window_val in [7, 11, 17, 19]:
            self.tab7_window.set(window_val)
            print(f"  ✓ Window size: {window_val}")
        else:
            print(f"  ⚠️  WARNING: Invalid window={window_val}, using default (17)")
            self.tab7_window.set(17)

    # Set CV folds (CRITICAL: Use same folds as Results tab for reproducibility)
    n_folds = config.get('n_folds', 5)
    if not pd.isna(n_folds) and int(n_folds) in range(3, 11):
        self.tab7_cv_folds.set(int(n_folds))
        print(f"  ✓ CV folds: {n_folds} (ensures same CV strategy as Results)")
    else:
        print(f"  ⚠️  WARNING: n_folds not in config or invalid, using default (5)")
        self.tab7_cv_folds.set(5)

    # ========================================================================
    # STEP 7: Update Mode Label and Enable Controls
    # ========================================================================
    print("\n[STEP 7/7] Finalizing...")

    # Store config for later use
    self.tab7_loaded_config = config.copy()

    # Update mode label
    self.tab7_mode_label.config(
        text=f"Mode: Loaded from Results (Rank {rank})",
        foreground='#2E7D32'  # Green color to indicate loaded state
    )
    print(f"✓ Mode label updated: Loaded from Results (Rank {rank})")

    # Enable Run button
    if hasattr(self, 'tab7_run_button'):
        self.tab7_run_button.config(state='normal')
        print("✓ Tab 7 Run button enabled")

    # Update wavelength count display
    if hasattr(self, 'tab7_wl_count_label'):
        self.tab7_wl_count_label.config(
            text=f"Wavelengths: {len(model_wavelengths)} selected"
        )
        print(f"✓ Wavelength count label updated: {len(model_wavelengths)}")

    # Update status
    if hasattr(self, 'tab7_status'):
        self.tab7_status.config(
            text=f"Loaded: {model_name} | {gui_preprocess} | Rank {rank}"
        )

    print("\n" + "="*80)
    print(f"✅ MODEL LOADING COMPLETE: {model_name} (Rank {rank})")
    print(f"   - {len(model_wavelengths)} wavelengths loaded and validated")
    print(f"   - Preprocessing: {gui_preprocess}")
    print(f"   - Ready for development in NEW Tab 7")
    print("="*80 + "\n")


# ============================================================================
# METHOD 2: Format Wavelengths Helper for NEW Tab 7
# ============================================================================

def _format_wavelengths_for_NEW_tab7(self, wavelengths_list):
    """
    Format wavelength list for display in NEW Tab 7 (Model Development).

    For small lists (<= 50), shows all wavelengths as comma-separated values.
    For large lists (> 50), shows first 10, "...", last 10, with total count.

    Args:
        wavelengths_list (list): List of wavelength values (floats)

    Returns:
        str: Formatted wavelength string for display

    Examples:
        Small list (25 wavelengths):
        "1520.0, 1540.0, 1560.0, ... (full list)"

        Large list (200 wavelengths):
        "1520.0, 1540.0, ..., 2480.0, 2500.0 (200 wavelengths total)"
    """
    if not wavelengths_list:
        return "# No wavelengths specified"

    # Sort wavelengths for consistent display
    wls = sorted(wavelengths_list)
    n_wls = len(wls)

    if n_wls <= 50:
        # Show all wavelengths
        wl_strings = [f"{w:.1f}" for w in wls]
        formatted = ", ".join(wl_strings)
        return formatted
    else:
        # Show condensed format: first 10 + "..." + last 10
        first_10 = [f"{w:.1f}" for w in wls[:10]]
        last_10 = [f"{w:.1f}" for w in wls[-10:]]

        formatted = (
            ", ".join(first_10) +
            ", ..., " +
            ", ".join(last_10) +
            f"\n\n# Total: {n_wls} wavelengths selected" +
            f"\n# Range: {wls[0]:.1f} nm to {wls[-1]:.1f} nm"
        )
        return formatted


# ============================================================================
# METHOD 3: Updated Double-Click Handler for NEW Tab 7
# ============================================================================

def _on_result_double_click_NEW_TAB7(self, event):
    """
    Handle double-click on result row - load into NEW Tab 7 (Model Development).

    This is an UPDATED version that routes to NEW Tab 7 instead of Tab 6.
    Replace the existing _on_result_double_click with this version when
    Tab 7 UI is ready.

    INTEGRATION NOTE:
    - Current implementation loads into Tab 6 (index 5)
    - This version loads into NEW Tab 7 (index 6)
    - Switch when Tab 7 UI controls are implemented
    """
    selection = self.results_tree.selection()
    if not selection:
        return

    if self.results_df is None:
        return

    try:
        # Get the selected row index
        item_id = selection[0]
        row_idx = int(item_id)

        # Get the selected model configuration
        # CRITICAL: Use .loc (label-based) not .iloc (position-based)
        # because treeview IID uses the dataframe's original index labels
        model_config = self.results_df.loc[row_idx].to_dict()
        self.selected_model_config = model_config

        # Log validation info
        rank = model_config.get('Rank', '?')
        model_name = model_config.get('Model', '?')
        r2_or_acc = model_config.get('R2', model_config.get('Accuracy', '?'))
        n_vars = model_config.get('n_vars', '?')

        print(f"\n{'='*80}")
        print(f"USER ACTION: Double-clicked Rank {rank} in Results tab")
        print(f"{'='*80}")
        print(f"  Model: {model_name}")
        print(f"  Performance: {r2_or_acc}")
        print(f"  Variables: {n_vars}")
        print(f"  Loading into NEW Tab 7 (Model Development)...")

        # Load into NEW Tab 7 using NEW robust logic
        self._load_model_to_NEW_tab7(model_config)

        # Switch to NEW Tab 7 (index 6, since tabs are 0-indexed)
        # Tab 0=Import, 1=Quality, 2=Config, 3=Progress, 4=Results, 5=Tab6, 6=Tab7, 7=Predict
        self.notebook.select(6)

        print(f"✅ Successfully loaded and switched to NEW Tab 7 (Model Development)")
        print(f"{'='*80}\n")

    except ValueError as ve:
        # Our validation errors - show to user with full details
        error_msg = str(ve)
        messagebox.showerror(
            "Model Loading Failed - Data Validation Error",
            error_msg
        )
        print(f"\n❌ VALIDATION ERROR:\n{error_msg}\n")
        import traceback
        traceback.print_exc()

    except Exception as e:
        # Unexpected errors - show generic message
        messagebox.showerror(
            "Model Loading Failed",
            f"Failed to load model configuration:\n\n{str(e)}\n\n"
            f"Check the console for detailed error information."
        )
        print(f"\n❌ ERROR loading model: {e}\n")
        import traceback
        traceback.print_exc()


# ============================================================================
# INTEGRATION CHECKLIST
# ============================================================================

"""
INTEGRATION CHECKLIST FOR SPECTRAL_PREDICT_GUI_OPTIMIZED.PY:

1. ADD METHODS TO CLASS:
   - Copy _load_model_to_NEW_tab7() into SpectralPredictApp class
   - Copy _format_wavelengths_for_NEW_tab7() into SpectralPredictApp class
   - Keep as separate methods from existing Tab 6 methods

2. UPDATE DOUBLE-CLICK HANDLER (when Tab 7 UI is ready):
   - Replace _on_result_double_click with _on_result_double_click_NEW_TAB7
   - Or add conditional: if tab7_ready: load_to_tab7() else: load_to_tab6()

3. CREATE TAB 7 UI CONTROLS (Future Phase):
   All controls must use tab7_ prefix:
   - self.tab7_config_text (Text widget, read-only) - shows model info
   - self.tab7_wl_spec (Text widget) - wavelength specification
   - self.tab7_model_type (StringVar) - model type dropdown
   - self.tab7_task_type (StringVar) - regression/classification radio
   - self.tab7_preprocess (StringVar) - preprocessing dropdown
   - self.tab7_window (IntVar) - window size radio buttons
   - self.tab7_cv_folds (IntVar) - CV folds spinbox
   - self.tab7_mode_label (Label) - mode indicator
   - self.tab7_run_button (Button) - run model button
   - self.tab7_wl_count_label (Label) - wavelength count display
   - self.tab7_status (Label) - status message
   - self.tab7_loaded_config (dict) - stored config for reference

4. TEST FAIL-LOUD WAVELENGTH VALIDATION:
   After integration, test with:
   a. Subset model with valid all_vars -> Should load successfully
   b. Subset model missing all_vars -> Should show CRITICAL ERROR dialog
   c. Subset model with count mismatch -> Should show CRITICAL ERROR dialog
   d. Full spectrum model -> Should load all wavelengths successfully
   e. Model with invalid wavelengths -> Should show CRITICAL ERROR dialog

5. VERIFY NAMESPACE SEPARATION:
   - Tab 6 uses: refine_* variables
   - Tab 7 uses: tab7_* variables
   - NO overlap or conflicts between the two
   - Both tabs can coexist independently

6. LOGGING AND DEBUGGING:
   - All methods print detailed progress to console
   - Error messages are actionable with clear solutions
   - Each step (1-7) is logged with checkmarks
   - User sees clear dialogs for validation errors

7. EXPECTED BEHAVIOR AFTER INTEGRATION:
   - Double-click result in Results tab
   - Loads model into NEW Tab 7 (not Tab 6)
   - Automatically switches to Tab 7
   - All controls populated with loaded values
   - Wavelength validation FAILS LOUD on errors
   - Ready to run model with same configuration
"""


# ============================================================================
# TESTING SCENARIOS
# ============================================================================

"""
CRITICAL TEST SCENARIOS (Post-Integration):

TEST 1: Full Spectrum Model Loading
------------------------------------
Setup: Double-click a PLS full-spectrum model from Results
Expected:
  - Loads into Tab 7 (not Tab 6)
  - All wavelengths loaded
  - Config text shows model info
  - Controls populated correctly
  - No errors

TEST 2: Subset Model with Valid all_vars
----------------------------------------
Setup: Double-click a top50 PLS subset model with all_vars field
Expected:
  - Parses all_vars successfully
  - Validates 50 wavelengths
  - Loads exactly 50 wavelengths (not full spectrum)
  - Wavelength display shows condensed format or full list
  - No fallback to full spectrum

TEST 3: Subset Model Missing all_vars (FAIL LOUD)
-------------------------------------------------
Setup: Manually edit results_df to remove all_vars, then double-click
Expected:
  - Error dialog appears
  - Message: "CRITICAL ERROR: Missing 'all_vars' field"
  - Shows model name, rank, subset tag
  - Suggests re-running analysis
  - Does NOT silently fall back to full spectrum
  - Does NOT load partial data

TEST 4: Subset Model with Count Mismatch (FAIL LOUD)
----------------------------------------------------
Setup: Manually edit all_vars to have wrong number of wavelengths
Expected:
  - Error dialog appears
  - Message: "CRITICAL ERROR: Wavelength count mismatch!"
  - Shows expected vs parsed counts
  - Does NOT load

TEST 5: Subset Model with Invalid Wavelengths (FAIL LOUD)
---------------------------------------------------------
Setup: Manually edit all_vars to include wavelengths not in dataset
Expected:
  - Error dialog appears
  - Message: "CRITICAL ERROR: Invalid wavelengths in 'all_vars'!"
  - Lists invalid wavelengths
  - Suggests loading correct dataset
  - Does NOT load

TEST 6: NeuralBoosted Model with Hyperparameters
------------------------------------------------
Setup: Double-click NeuralBoosted model from Results
Expected:
  - Loads all hyperparameters (n_estimators, learning_rate, etc.)
  - Config text shows hyperparameters section
  - Controls populated with correct values

TEST 7: Classification Model
----------------------------
Setup: Double-click classification model (if available)
Expected:
  - Task type auto-detected as classification
  - Shows Accuracy (not RMSE/R²)
  - Shows ROC AUC if available

TEST 8: Preprocessing Conversion
--------------------------------
Setup: Double-click models with various preprocessing
Expected:
  - search.py naming converts to GUI naming correctly
  - 'deriv' + Deriv=1 -> 'sg1'
  - 'deriv' + Deriv=2 -> 'sg2'
  - 'snv_deriv' -> 'snv_sg1' or 'snv_sg2'
  - All other preprocessings map correctly

TEST 9: Large Wavelength Set Display
------------------------------------
Setup: Load model with >50 wavelengths
Expected:
  - Wavelength display shows condensed format
  - First 10, "...", last 10
  - Summary line with total count and range

TEST 10: Data Not Loaded (FAIL LOUD)
------------------------------------
Setup: Fresh app start, double-click without loading data
Expected:
  - Error dialog appears
  - Message: "Data validation failed!"
  - Clear instruction to load data in Tab 1
  - Does NOT crash

PERFORMANCE BENCHMARKS:
- Loading subset model (50 wls): < 0.5 seconds
- Loading full model (2000 wls): < 1 second
- Wavelength formatting: < 0.1 seconds
- No GUI freeze during loading
"""


# ============================================================================
# FUTURE ENHANCEMENTS (Not in Current Scope)
# ============================================================================

"""
POTENTIAL FUTURE ENHANCEMENTS:

1. Wavelength Validation Preview:
   - Before loading, show preview of wavelengths to be loaded
   - User can confirm before populating Tab 7

2. Batch Loading:
   - Load multiple models into comparison view
   - Side-by-side parameter comparison

3. Parameter Locking:
   - Lock specific parameters (e.g., always use same preprocessing)
   - Only modify other parameters

4. Undo/Redo for Loading:
   - Save previous loaded state
   - Allow user to revert to previous model

5. Export Loaded Configuration:
   - Save loaded config as JSON for documentation
   - Include wavelengths, hyperparameters, metrics

6. Load from External Results:
   - Import results CSV from external analysis
   - Parse and load into Tab 7

7. Smart Default Hyperparameters:
   - Suggest hyperparameter ranges based on loaded model
   - "Tune around this value" feature

8. Cross-Dataset Wavelength Mapping:
   - If dataset changed, attempt to map wavelengths
   - Warn user about differences
   - Option to interpolate missing wavelengths
"""


# ============================================================================
# ERROR MESSAGE TEMPLATES (For Reference)
# ============================================================================

ERROR_TEMPLATES = {
    'missing_all_vars': """CRITICAL ERROR: Missing 'all_vars' field for subset model!
  Model: {model_name} (Rank {rank})
  Subset: {subset_tag}
  Expected variables: {n_vars}

The 'all_vars' field is REQUIRED for subset models to identify
the exact wavelengths used. Without it, loading this model
would cause R² discrepancies.

SOLUTION: Re-run the analysis to generate complete results.""",

    'count_mismatch': """CRITICAL ERROR: Wavelength count mismatch!
  Model: {model_name} (Rank {rank})
  Expected: {expected_count} wavelengths (from n_vars field)
  Parsed: {parsed_count} wavelengths (from all_vars field)
  Subset: {subset_tag}

This indicates a data integrity issue in the results table.
The 'all_vars' field does not match the 'n_vars' count.

SOLUTION: Re-run the analysis to generate consistent results.""",

    'invalid_wavelengths': """CRITICAL ERROR: Invalid wavelengths in 'all_vars'!
  Model: {model_name} (Rank {rank})
  Found {n_invalid} wavelengths not in current dataset:
  {invalid_wls}

This likely means the current dataset is different from
the one used to generate these results.

SOLUTION: Load the original dataset or re-run the analysis.""",

    'parse_failed': """CRITICAL ERROR: Failed to parse 'all_vars' field!
  Model: {model_name} (Rank {rank})
  Error: {error}
  all_vars content: {all_vars_preview}...

The wavelength data in the results table is malformed.

SOLUTION: Re-run the analysis to generate valid results.""",

    'no_data': """Data validation failed!
Required data (X, y, wavelengths) is not available.
Please ensure data is loaded in the Data Upload tab.""",

    'no_wavelengths': """CRITICAL ERROR: No wavelengths loaded!
  Model: {model_name} (Rank {rank})
  Subset: {subset_tag}

This should never happen - indicates a logic error in loading code."""
}


# ============================================================================
# END OF PHASE 3 IMPLEMENTATION
# ============================================================================

"""
FILE SUMMARY:

This file contains THREE methods ready for integration:

1. _load_model_to_NEW_tab7(self, config)
   - Main loading engine for NEW Tab 7
   - 7-step process with validation
   - FAIL LOUD wavelength validation
   - Comprehensive error handling
   - ~400 lines

2. _format_wavelengths_for_NEW_tab7(self, wavelengths_list)
   - Formats wavelengths for display
   - Handles small (<= 50) and large (> 50) lists
   - Condensed format for large lists
   - ~50 lines

3. _on_result_double_click_NEW_TAB7(self, event)
   - Updated double-click handler
   - Routes to NEW Tab 7 (index 6)
   - Comprehensive error handling
   - ~80 lines

TOTAL: ~530 lines of production-ready code

NEXT STEPS:
1. Integrate methods into spectral_predict_gui_optimized.py
2. Create Tab 7 UI with tab7_* controls (separate phase)
3. Test with subset and full spectrum models
4. Verify FAIL LOUD validation works correctly

CRITICAL SUCCESS FACTORS:
✓ Uses tab7_ namespace (not refine_)
✓ FAIL LOUD wavelength validation
✓ No silent fallbacks
✓ Actionable error messages
✓ Comprehensive logging
✓ Ready for incremental testing
"""
