#!/usr/bin/env python
"""Add overall R² calculation from concatenated predictions"""

with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where results dict is created and add overall R² calculation
for i, line in enumerate(lines):
    # Look for the line after fold metrics are computed
    if "# Compute mean and std across folds" in line:
        # Add overall R² calculation before this line
        indent = '            '
        new_code = f'''{indent}# Compute overall R² from concatenated predictions (more standard metric)
{indent}if task_type == "regression":
{indent}    overall_r2 = r2_score(all_y_true, all_y_pred)
{indent}    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
{indent}    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
{indent}else:
{indent}    overall_acc = accuracy_score(all_y_true, all_y_pred)
{indent}    overall_prec = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
{indent}    overall_rec = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
{indent}    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)

'''
        lines.insert(i, new_code)
        break

# Find where results dict is populated and add overall metrics
for i, line in enumerate(lines):
    if "results['mae_std'] = np.std([m['mae'] for m in fold_metrics])" in line:
        indent = '                '
        new_code = f'''{indent}# Add overall metrics from concatenated predictions
{indent}results['r2_overall'] = overall_r2
{indent}results['rmse_overall'] = overall_rmse
{indent}results['mae_overall'] = overall_mae
'''
        lines.insert(i+1, new_code)
        break

# Update the results text to show both R² values
for i, line in enumerate(lines):
    if "Cross-Validation Performance" in line and "folds" in line and i > 4200:
        # Find the R² line (should be a few lines down)
        for j in range(i, min(i+10, len(lines))):
            if "R²:" in lines[j] and "r2_mean" in lines[j]:
                # Replace with both values
                indent = ' ' * (len(lines[j]) - len(lines[j].lstrip()))
                new_line = f'{indent}R² (mean of folds): {{results[\'r2_mean\']:.4f}} ± {{results[\'r2_std\']:.4f}}\n'
                new_line2 = f'{indent}R² (overall): {{results[\'r2_overall\']:.4f}}\n'
                lines[j] = new_line
                lines.insert(j+1, new_line2)
                break
        break

with open('spectral_predict_gui_optimized.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Added overall R² calculation and display")
