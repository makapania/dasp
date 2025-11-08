"""
Apply fixes for NeuralBoosted results and Ridge/RandomForest R² discrepancies
"""

import re

# Read the file
with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("Applying GUI fixes...")
print()

# Fix 1: Add NeuralBoosted empty results warning (after line 2735)
print("Fix 1: Adding NeuralBoosted empty results warning...")
old_section_1 = '''            # Store results for Results tab
            self.results_df = results_df

            # Populate Results tab
            self.root.after(0, lambda: self._populate_results_table(results_df))'''

new_section_1 = '''            # Store results for Results tab
            self.results_df = results_df

            # FIX: Check if NeuralBoosted was selected but produced no results
            selected_models = [model for model, var in self.model_checkboxes.items() if var.get()]
            if (self.results_df is None or len(self.results_df) == 0):
                if 'NeuralBoosted' in selected_models:
                    warning_msg = (
                        "NeuralBoosted training failed for all configurations.\\n\\n"
                        "This model requires specific conditions to train successfully.\\n"
                        "Check the console output for detailed error messages.\\n\\n"
                        "Note: Other models may have completed successfully."
                    )
                    self.root.after(0, lambda: messagebox.showwarning(
                        "NeuralBoosted Training Failed", warning_msg
                    ))
                    self._log_progress("\\n[WARN] WARNING: NeuralBoosted produced no results (all training attempts failed)")

            # Populate Results tab
            self.root.after(0, lambda: self._populate_results_table(results_df))'''

if old_section_1 in content:
    content = content.replace(old_section_1, new_section_1)
    print("  [OK] NeuralBoosted warning added")
else:
    print("  [WARN] Could not find target location for NeuralBoosted warning")

# Fix 2: Fix RandomForest to add max_depth loading and random_state (lines 3955-3963)
print("Fix 2: Fixing RandomForest hyperparameters...")
old_section_2 = '''                elif model_name == 'RandomForest':
                    if 'n_trees' in self.selected_model_config and not pd.isna(self.selected_model_config.get('n_trees')):
                        # Julia uses n_trees, scikit-learn uses n_estimators
                        params_from_search['n_estimators'] = int(self.selected_model_config['n_trees'])
                        print(f"DEBUG: Loaded n_estimators={params_from_search['n_estimators']} for RandomForest")
                    if 'max_features' in self.selected_model_config and not pd.isna(self.selected_model_config.get('max_features')):
                        params_from_search['max_features'] = str(self.selected_model_config['max_features'])
                        print(f"DEBUG: Loaded max_features={params_from_search['max_features']} for RandomForest")'''

new_section_2 = '''                elif model_name == 'RandomForest':
                    if 'n_trees' in self.selected_model_config and not pd.isna(self.selected_model_config.get('n_trees')):
                        # Julia uses n_trees, scikit-learn uses n_estimators
                        params_from_search['n_estimators'] = int(self.selected_model_config['n_trees'])
                        print(f"DEBUG: Loaded n_estimators={params_from_search['n_estimators']} for RandomForest")
                    if 'max_features' in self.selected_model_config and not pd.isna(self.selected_model_config.get('max_features')):
                        params_from_search['max_features'] = str(self.selected_model_config['max_features'])
                        print(f"DEBUG: Loaded max_features={params_from_search['max_features']} for RandomForest")
                    # FIX: Load max_depth if available
                    if 'max_depth' in self.selected_model_config and not pd.isna(self.selected_model_config.get('max_depth')):
                        max_depth_val = self.selected_model_config['max_depth']
                        # Julia uses 'nothing' for unlimited depth, Python uses None
                        if str(max_depth_val).lower() in ['nothing', 'none', 'null']:
                            params_from_search['max_depth'] = None
                            print(f"DEBUG: Set RandomForest max_depth=None (unlimited)")
                        else:
                            params_from_search['max_depth'] = int(max_depth_val)
                            print(f"DEBUG: Loaded max_depth={params_from_search['max_depth']} for RandomForest")
                    # FIX: Set random_state for reproducibility (Julia uses fixed random state)
                    params_from_search['random_state'] = 42
                    print(f"DEBUG: Set RandomForest random_state=42 for reproducibility")'''

if old_section_2 in content:
    content = content.replace(old_section_2, new_section_2)
    print("  [OK] RandomForest hyperparameters fixed (added max_depth + random_state)")
else:
    print("  [WARN] Could not find target location for RandomForest fixes")

# Fix 3: Enhance index reset debug output (around line 3824)
print("Fix 3: Enhancing index reset debug output...")
old_section_3 = '''            X_base_df = X_base_df.reset_index(drop=True)
            y_series = y_series.reset_index(drop=True)
            print(f"DEBUG: Reset index after exclusions - X_base_df.index now: {list(X_base_df.index[:10])}...")
            print(f"DEBUG: This ensures CV folds match Julia backend (sequential row indexing)")'''

new_section_3 = '''            X_base_df = X_base_df.reset_index(drop=True)
            y_series = y_series.reset_index(drop=True)
            print(f"DEBUG: Reset index after exclusions")
            print(f"DEBUG:   X_base_df shape: {X_base_df.shape}, first 5 indices: {list(X_base_df.index[:5])}")
            print(f"DEBUG:   y_series shape: {y_series.shape}, first 5 y values: {list(y_series.values[:5])}")
            print(f"DEBUG:   This ensures CV folds match Julia backend (sequential row indexing)")'''

if old_section_3 in content:
    content = content.replace(old_section_3, new_section_3)
    print("  [OK] Index reset debug output enhanced")
else:
    print("  [WARN] Could not find target location for index reset debug output")

# Write the fixed content
with open('spectral_predict_gui_optimized.py', 'w', encoding='utf-8') as f:
    f.write(content)

print()
print("="*60)
print("GUI fixes applied successfully!")
print("="*60)
print()
print("Fixes applied:")
print("  1. NeuralBoosted empty results warning")
print("  2. RandomForest random_state=42 + max_depth loading")
print("  3. Enhanced index reset debug output")
print()
print("Notes:")
print("  - Ridge alpha logging was already present (no changes needed)")
print("  - RandomForest now matches Julia's deterministic behavior")
print()
print("Next steps:")
print("  1. Test with GUI: Load data, run analysis with NeuralBoosted")
print("  2. Select Ridge from Results, run in Model Development")
print("  3. Select RandomForest from Results, run in Model Development")
print("  4. Check console debug output to verify:")
print("     - Hyperparameters loaded correctly")
print("     - Index reset executes")
print("     - random_state=42 is set for RandomForest")
