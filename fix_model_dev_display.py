#!/usr/bin/env python
"""Fix Model Development tab to show correct parameters and preprocessing"""

with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Replace hardcoded n_components with model-specific parameters
old_config_section = """Configuration:
  Model: {model_name}
  Task Type: {task_type}
  Preprocessing: {preprocess}
  Window Size: {window}
  Wavelengths: {wl_summary}
  Features: {len(selected_wl)}
  Samples: {X_raw.shape[0]}
  CV Folds: {self.refine_folds.get()}
  n_components: {n_components}"""

new_config_section = """Configuration:
  Model: {model_name}
  Task Type: {task_type}
  Preprocessing: {preprocess}
  Window Size: {window}
  Wavelengths: {wl_summary}
  Features: {len(selected_wl)}
  Samples: {X_raw.shape[0]}
  CV Folds: {self.refine_folds.get()}
{model_params_str}"""

content = content.replace(old_config_section, new_config_section)

# Find where to insert model_params_str definition (before results_text)
# Look for the line before "results_text = f"
import_marker = '                results_text = f"""Refined Model Results:'
insert_before_marker = '                results_text = f"""Refined Model Results:'

if insert_before_marker in content:
    # Build model-specific params string
    params_code = '''                # Build model-specific parameters display
                model_params_str = ""
                if model_name == 'PLS':
                    model_params_str = f"  n_components: {n_components}"
                elif model_name in ['Ridge', 'Lasso']:
                    alpha_val = "N/A"
                    if hasattr(model, 'alpha'):
                        alpha_val = f"{model.alpha}"
                    elif self.selected_model_config and 'alpha' in self.selected_model_config:
                        alpha_val = f"{self.selected_model_config['alpha']}"
                    model_params_str = f"  alpha: {alpha_val}"
                elif model_name == 'RandomForest':
                    n_est = getattr(model, 'n_estimators', 'N/A')
                    max_d = getattr(model, 'max_depth', 'N/A')
                    max_f = getattr(model, 'max_features', 'N/A')
                    model_params_str = f"  n_estimators: {n_est}\\n  max_depth: {max_d}\\n  max_features: {max_f}"
                elif model_name == 'MLP':
                    hidden = getattr(model, 'hidden_layer_sizes', 'N/A')
                    lr = getattr(model, 'learning_rate_init', 'N/A')
                    model_params_str = f"  hidden_layer_sizes: {hidden}\\n  learning_rate_init: {lr}"
                elif model_name == 'NeuralBoosted':
                    n_est = getattr(model, 'n_estimators', 'N/A')
                    lr = getattr(model, 'learning_rate', 'N/A')
                    hidden = getattr(model, 'hidden_layer_size', 'N/A')
                    model_params_str = f"  n_estimators: {n_est}\\n  learning_rate: {lr}\\n  hidden_layer_size: {hidden}"
                else:
                    model_params_str = f"  n_components: {n_components}"

'''

    content = content.replace(insert_before_marker, params_code + insert_before_marker)

# Write back
with open('spectral_predict_gui_optimized.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed model parameter display")
