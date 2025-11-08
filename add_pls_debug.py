#!/usr/bin/env python
"""Add PLS debug logging to the GUI file."""

# Read the file
with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line "if model_name == 'Ridge' or model_name == 'Lasso':"
# and insert PLS debug code before it
pls_debug = '''                if model_name == 'PLS':
                    # PLS n_components already set during model initialization above
                    # Just log for verification
                    print(f"DEBUG: PLS model created with n_components={n_components}")
                    print(f"DEBUG: Model actual n_components: {model.n_components}")
                    if 'LVs' in self.selected_model_config:
                        expected_lvs = self.selected_model_config['LVs']
                        print(f"DEBUG: Expected LVs from Results: {expected_lvs}")
                        if model.n_components != expected_lvs:
                            print(f"WARNING: PLS n_components mismatch! Expected {expected_lvs}, got {model.n_components}")

                el'''

new_lines = []
for i, line in enumerate(lines):
    if "if model_name == 'Ridge' or model_name == 'Lasso':" in line and i > 3960 and i < 3970:
        # Insert PLS debug before this line
        new_lines.append(pls_debug + line)
    else:
        new_lines.append(line)

# Write back
with open('spectral_predict_gui_optimized.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Added PLS debug logging")
