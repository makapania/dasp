"""
AGENT 3: Model Loading Engine - Robust Results→Tab 6 Data Transfer
Implementation for spectral_predict_gui_optimized.py

This module contains the complete, production-ready implementation for loading
model configurations from the Results tab into the Custom Model Development tab (Tab 6).

CRITICAL FIX: Wavelength loading with FAIL LOUD validation - no silent fallbacks!
"""

# ============================================================================
# METHOD 1: Main Loading Logic - _load_model_to_tab7
# ============================================================================

def _load_model_to_tab7(self, config):
    """
    Load a model configuration from Results tab into Tab 6 (Custom Model Development).

    This method implements robust data transfer with comprehensive error handling
    and FAIL LOUD validation for wavelength loading.

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
    print("LOADING MODEL INTO CUSTOM MODEL DEVELOPMENT TAB")
    print("="*80)

    # ========================================================================
    # STEP 1: Validate Data Availability
    # ========================================================================
    print("\n[STEP 1/7] Validating data availability...")

    if not self._validate_data_for_refinement():
        raise RuntimeError(
            "Data validation failed!\n"
            "Required data (X, y, wavelengths) is not available.\n"
            "Please ensure data is loaded in the Data Upload tab."
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
                f"  all_vars content: {config.get('all_vars', 'MISSING')[:200]}...\n\n"
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
        # Use the new formatting helper
        wl_display_text = self._format_wavelengths_for_tab7(model_wavelengths)
        print(f"✓ Formatted {len(model_wavelengths)} wavelengths ({len(wl_display_text)} characters)")

    except Exception as e:
        print(f"⚠️  WARNING: Wavelength formatting failed: {e}")
        # Fallback to simple comma-separated list
        wl_display_text = ", ".join([f"{w:.1f}" for w in model_wavelengths[:100]])
        if len(model_wavelengths) > 100:
            wl_display_text += f", ... ({len(model_wavelengths) - 100} more)"

    # Update wavelength specification widget
    self.refine_wl_spec.config(state='normal')
    self.refine_wl_spec.delete('1.0', 'end')
    self.refine_wl_spec.insert('1.0', wl_display_text)

    # Verify insertion
    content = self.refine_wl_spec.get('1.0', 'end-1c')
    if len(content) == 0:
        raise RuntimeError("ERROR: Wavelength text widget is empty after insertion!")

    print(f"✓ Wavelength widget updated: {len(content)} characters inserted")

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
    # STEP 6: Populate GUI Controls
    # ========================================================================
    print("\n[STEP 6/7] Populating GUI controls...")

    # Update info text display
    self.refine_model_info.config(state='normal')
    self.refine_model_info.delete('1.0', tk.END)
    self.refine_model_info.insert('1.0', info_text)
    self.refine_model_info.config(state='disabled')
    print("  ✓ Model info display updated")

    # Set model type
    if model_name in ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']:
        self.refine_model_type.set(model_name)
        print(f"  ✓ Model type: {model_name}")
    else:
        print(f"  ⚠️  WARNING: Unknown model type '{model_name}', defaulting to PLS")
        self.refine_model_type.set('PLS')

    # Set task type (auto-detect from data)
    if self.y is not None:
        if self.y.nunique() == 2 or self.y.dtype == 'object' or self.y.nunique() < 10:
            self.refine_task_type.set('classification')
            print("  ✓ Task type: classification (auto-detected)")
        else:
            self.refine_task_type.set('regression')
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
        self.refine_preprocess.set(gui_preprocess)
        print(f"  ✓ Preprocessing: {gui_preprocess}")

    # Set window size
    if window != 'N/A' and not pd.isna(window):
        window_val = int(float(window))
        if window_val in [7, 11, 17, 19]:
            self.refine_window.set(window_val)
            print(f"  ✓ Window size: {window_val}")
        else:
            print(f"  ⚠️  WARNING: Invalid window={window_val}, using default (17)")
            self.refine_window.set(17)

    # Set CV folds (CRITICAL: Use same folds as Results tab for reproducibility)
    n_folds = config.get('n_folds', 5)
    if not pd.isna(n_folds) and int(n_folds) in range(3, 11):
        self.refine_folds.set(int(n_folds))
        print(f"  ✓ CV folds: {n_folds} (ensures same CV strategy as Results)")
    else:
        print(f"  ⚠️  WARNING: n_folds not in config or invalid, using default (5)")
        self.refine_folds.set(5)

    # ========================================================================
    # STEP 7: Update Mode Label and Enable Controls
    # ========================================================================
    print("\n[STEP 7/7] Finalizing...")

    # Store config for later use
    self.tab7_loaded_config = config.copy()

    # Update mode label
    self.refine_mode_label.config(
        text=f"Mode: Loaded from Results (Rank {rank})",
        foreground='#2E7D32'  # Green color to indicate loaded state
    )
    print(f"✓ Mode label updated: Loaded from Results (Rank {rank})")

    # Enable Run button
    self.refine_run_button.config(state='normal')
    print("✓ Run button enabled")

    # Update wavelength count display
    self._update_wavelength_count()

    # Update status
    if hasattr(self, 'refine_status'):
        self.refine_status.config(
            text=f"Loaded: {model_name} | {gui_preprocess} | Rank {rank}"
        )

    print("\n" + "="*80)
    print(f"✅ MODEL LOADING COMPLETE: {model_name} (Rank {rank})")
    print(f"   - {len(model_wavelengths)} wavelengths loaded and validated")
    print(f"   - Preprocessing: {gui_preprocess}")
    print(f"   - Ready for refinement")
    print("="*80 + "\n")


# ============================================================================
# METHOD 2: Format Wavelengths Helper
# ============================================================================

def _format_wavelengths_for_tab7(self, wavelengths_list):
    """
    Format wavelength list for display in Custom Model Development tab.

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
# METHOD 3: Updated Double-Click Handler
# ============================================================================

def _on_result_double_click(self, event):
    """
    Handle double-click on result row - load into Custom Model Development tab.

    This replaces the existing handler to use the new robust loading logic.
    """
    selection = self.results_tree.selection()
    if not selection:
        return

    try:
        # Get row index (treeview IID = DataFrame index label)
        row_idx = int(selection[0])

        # CRITICAL: Use .loc (label-based) not .iloc (position-based)
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
        print(f"  Loading into Custom Model Development tab...")

        # Load into Custom Model Development tab (Tab 6) using NEW robust logic
        self._load_model_to_tab7(model_config)

        # Switch to Custom Model Development tab (index 5, since tabs are 0-indexed)
        self.notebook.select(5)

        print(f"✅ Successfully loaded and switched to Custom Model Development tab")
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
# TESTING CHECKLIST (for documentation - don't implement)
# ============================================================================

"""
TESTING CHECKLIST FOR VALIDATION:

1. Load PLS model with top50 subset
   - Verify 50 wavelengths are loaded
   - Verify all_vars field is parsed correctly
   - Verify n_vars matches parsed count
   - Verify no fallback to full spectrum

2. Load Ridge model with full spectrum
   - Verify all wavelengths loaded
   - Verify alpha parameter is set correctly
   - Verify preprocessing settings transferred

3. Load RandomForest with region subset
   - Verify subset wavelengths loaded (not all)
   - Verify hyperparameters (n_estimators, max_depth) loaded
   - Verify no silent fallback

4. Load model with missing all_vars field
   - Verify FAIL LOUD error is raised
   - Verify error message is clear and actionable
   - Verify user sees error dialog

5. Load model with malformed all_vars field
   - Verify parsing error is caught
   - Verify FAIL LOUD error with details
   - Verify no silent fallback

6. Load model with wavelength count mismatch
   - Verify count validation catches discrepancy
   - Verify error message shows both counts
   - Verify no silent fallback

7. Load NeuralBoosted model
   - Verify all hyperparameters loaded (n_estimators, learning_rate, etc.)
   - Verify activation function setting
   - Verify hidden layer size

8. Load classification model
   - Verify task type auto-detects as classification
   - Verify accuracy metric displayed
   - Verify ROC AUC shown if available

9. Load model with various preprocessing
   - raw, snv, sg1, sg2
   - snv_sg1, snv_sg2
   - deriv_snv, deriv_msc
   - Verify all convert correctly to GUI naming

10. Edge case: Load model with missing data
    - Verify _validate_data_for_refinement catches it
    - Verify clear error message
    - Verify no crash

PERFORMANCE TESTING:
- Large subset (500+ wavelengths): Should load in < 1 second
- Full spectrum (2000+ wavelengths): Should load in < 2 seconds
- Wavelength formatting should not hang GUI

ERROR MESSAGE QUALITY:
- All error messages must be actionable
- Must identify the specific problem
- Must suggest a solution
- Must include relevant context (rank, model name, etc.)
"""
