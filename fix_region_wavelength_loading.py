#!/usr/bin/env python
"""Fix region model wavelength loading to fail loudly instead of falling back to full spectrum"""

with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the fallback code and make it more aggressive
for i, line in enumerate(lines):
    # Find the line with "WARNING: Subset model but no top_vars, using all wavelengths"
    if 'WARNING: Subset model but no top_vars, using all wavelengths' in line:
        # Replace the fallback behavior for region models
        # Instead of silently using all wavelengths, show error
        if i > 0 and 'model_wavelengths = list(all_wavelengths)' in lines[i+1]:
            # Insert a check before the fallback
            indent = '                        '
            new_lines = [
                f'{indent}# CRITICAL: For region models, do NOT fall back to all wavelengths\n',
                f'{indent}if "region" in subset_tag.lower():\n',
                f'{indent}    print(f"ERROR: Region model {{subset_tag}} has no wavelength data in all_vars or top_vars!")\n',
                f'{indent}    print(f"DEBUG: subset_tag={{subset_tag}}")\n',
                f'{indent}    print(f"DEBUG: all_vars={{config.get(\'all_vars\', \'N/A\')}}")\n',
                f'{indent}    print(f"DEBUG: top_vars={{config.get(\'top_vars\', \'N/A\')}}")\n',
                f'{indent}    raise ValueError(f"Cannot load region model: wavelength data missing from Results. SubsetTag={{subset_tag}}")\n',
            ]
            lines[i+1:i+1] = new_lines
            break

# Also add debug output for all_vars parsing
for i, line in enumerate(lines):
    if 'all_vars_str = str(config[\'all_vars\']).strip()' in line:
        # Add debug before parsing
        indent = ' ' * (len(line) - len(line.lstrip()))
        new_line = f'{indent}print(f"DEBUG: all_vars raw value: {{config[\'all_vars\'][:200]}}...")  # First 200 chars\n'
        lines.insert(i, new_line)
        break

with open('spectral_predict_gui_optimized.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Added strict validation for region model wavelength loading")
