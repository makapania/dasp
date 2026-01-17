"""
Code generator for reproducible analysis scripts.

Generates standalone Python scripts and Jupyter notebooks from
model configurations and analysis results for scientific publication.
"""

import json
import base64
import gzip
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .templates.header import HEADER_TEMPLATE, DATA_LOADING_TEMPLATE, DATA_LOADING_CLASSIFICATION_TEMPLATE
from .templates.preprocessing import get_preprocessing_template, SNV_TEMPLATE, SAVGOL_DERIVATIVE_TEMPLATE
from .templates.variable_selection import get_variable_selection_template
from .templates.models import get_model_template, get_model_imports, MODEL_IMPORTS
from .templates.validation import (
    get_cross_validation_template,
    get_metrics_template,
    FINAL_MODEL_TEMPLATE,
    PREDICTION_TEMPLATE,
)
from .templates.visualization import get_visualization_code, VISUALIZATION_IMPORTS


@dataclass
class ExportOptions:
    """Options for code export."""
    include_data_loading: bool = True
    include_preprocessing: bool = True
    include_variable_selection: bool = True
    include_cross_validation: bool = True
    include_visualization: bool = False
    include_comments: bool = True
    include_prediction_template: bool = True

    # Output format
    format: str = 'script'  # 'script' or 'notebook'

    # Data path placeholder
    data_path: str = 'your_data.csv'
    target_column: str = 'target'

    # Data embedding options
    include_data: bool = False  # Embed actual data in script
    data_X: Optional[np.ndarray] = None  # Spectral data array
    data_y: Optional[np.ndarray] = None  # Target values
    wavelengths: Optional[np.ndarray] = None  # Wavelength array
    colab_ready: bool = False  # Add Colab-specific features (pip install, badge)


class CodeGenerator:
    """
    Generate standalone Python scripts from model configurations.

    This class takes model information and generates a complete,
    reproducible Python script that can be shared with reviewers.

    Parameters
    ----------
    model_config : dict
        Model configuration containing:
        - model_name: str (e.g., 'PLS', 'Ridge', 'RandomForest')
        - preprocessing: str (e.g., 'snv', 'sg1', 'raw')
        - target_name: str
        - task_type: str ('regression' or 'classification')
        - params: dict of hyperparameters
        - metrics: dict of performance metrics
        - variable_indices: list or None
        - wavelengths: list or array
        - cv_folds: int
    options : ExportOptions, optional
        Export configuration options

    Examples
    --------
    >>> from spectral_predict.code_generator import CodeGenerator, ExportOptions
    >>> config = {
    ...     'model_name': 'PLS',
    ...     'preprocessing': 'snv',
    ...     'target_name': 'protein',
    ...     'task_type': 'regression',
    ...     'params': {'n_components': 8},
    ...     'metrics': {'RMSE': 0.45, 'R2': 0.92},
    ...     'cv_folds': 5
    ... }
    >>> options = ExportOptions(include_visualization=True)
    >>> generator = CodeGenerator(config, options)
    >>> script = generator.generate_script()
    """

    def __init__(self, model_config: Dict[str, Any], options: ExportOptions = None):
        self.config = model_config
        self.options = options or ExportOptions()

        # Extract commonly used values
        self.model_name = model_config.get('model_name', 'Unknown')
        self.preprocessing = model_config.get('preprocessing', 'raw')
        self.task_type = model_config.get('task_type', 'regression')
        self.target_name = model_config.get('target_name', 'target')
        self.params = model_config.get('params', {})
        self.metrics = model_config.get('metrics', {})
        self.variable_indices = model_config.get('variable_indices', None)
        self.wavelengths = model_config.get('wavelengths', None)
        self.cv_folds = model_config.get('cv_folds', 5)
        self.imbalance_method = model_config.get('imbalance_method', None)

        # Update options with target column from config
        if self.target_name:
            self.options.target_column = self.target_name

        # Validate data embedding options
        if self.options.include_data:
            self._validate_data_embedding()

    def _validate_data_embedding(self):
        """Validate data embedding requirements."""
        if self.options.data_X is None or self.options.data_y is None:
            raise ValueError("data_X and data_y must be provided when include_data=True")

        # Check data size (100 MB limit)
        data_size_mb = self._estimate_data_size_mb()
        if data_size_mb > 100:
            raise ValueError(
                f"Data size ({data_size_mb:.1f} MB) exceeds 100 MB limit. "
                f"Consider exporting without embedded data."
            )

    def _estimate_data_size_mb(self) -> float:
        """Estimate compressed data size in MB."""
        if self.options.data_X is None:
            return 0.0

        # Estimate: serialize to JSON, compress, base64 encode
        total_elements = self.options.data_X.size + self.options.data_y.size
        if self.options.wavelengths is not None:
            total_elements += self.options.wavelengths.size

        # Rough estimate: 8 bytes per float * compression ratio * base64 overhead
        estimated_bytes = total_elements * 8 * 0.3 * 1.33  # ~30% compression, 33% base64 overhead
        return estimated_bytes / (1024 * 1024)

    def generate_script(self) -> str:
        """
        Generate a complete standalone Python script.

        Returns
        -------
        str
            Complete Python script as a string
        """
        sections = []

        # 1. Header with imports
        sections.append(self._render_header())

        # 2. Preprocessing functions
        if self.options.include_preprocessing and self.preprocessing != 'raw':
            sections.append(self._render_preprocessing_functions())

        # 3. Variable selection functions (if used)
        if self.options.include_variable_selection and self.variable_indices is not None:
            varsel_code = self._render_variable_selection_functions()
            if varsel_code:
                sections.append(varsel_code)

        # 4. Data loading (embedded or file-based)
        if self.options.include_data:
            sections.append(self._generate_embedded_data_section())
        elif self.options.include_data_loading:
            sections.append(self._render_data_loading())

        # 5. Preprocessing application
        if self.options.include_preprocessing:
            sections.append(self._render_preprocessing_application())

        # 6. Variable selection application
        if self.options.include_variable_selection and self.variable_indices is not None:
            sections.append(self._render_variable_selection_application())

        # 7. Imbalance handling (if classification)
        if self.imbalance_method and self.task_type == 'classification':
            sections.append(self._render_imbalance_handling())

        # 8. Model instantiation
        sections.append(self._render_model())

        # 9. Cross-validation
        if self.options.include_cross_validation:
            sections.append(self._render_cross_validation())
            sections.append(self._render_metrics())

        # 9. Final model training
        sections.append(self._render_final_model())

        # 10. Visualization (optional)
        if self.options.include_visualization:
            sections.append(self._render_visualization())

        # 11. Prediction template (optional)
        if self.options.include_prediction_template:
            sections.append(PREDICTION_TEMPLATE)

        return '\n'.join(sections)

    def generate_notebook(self) -> dict:
        """
        Generate a Jupyter notebook structure.

        Returns
        -------
        dict
            Notebook structure compatible with nbformat
        """
        cells = []

        # Title cell with optional Colab badge
        title_text = f"# Spectral Analysis: {self.model_name}\n\n"
        if self.options.colab_ready:
            title_text += (
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                "(https://colab.research.google.com/)\n\n"
            )
        title_text += (
            f"Generated by Spectral Predict on {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"**Model**: {self.model_name}  \n"
            f"**Preprocessing**: {self.preprocessing}  \n"
            f"**Task**: {self.task_type}  \n"
        )
        cells.append(self._make_markdown_cell(title_text))

        # Colab pip install cell (if requested)
        if self.options.colab_ready:
            cells.append(self._make_markdown_cell(
                "## 0. Install Dependencies (Colab Only)\n\n"
                "Run this cell if using Google Colab."
            ))
            pip_packages = self._get_pip_packages()
            cells.append(self._make_code_cell(f"!pip install -q {' '.join(pip_packages)}"))

        # Imports cell
        cells.append(self._make_markdown_cell("## 1. Setup and Imports"))
        cells.append(self._make_code_cell(self._get_imports_code()))

        # Preprocessing functions
        if self.preprocessing != 'raw':
            cells.append(self._make_markdown_cell("## 2. Preprocessing Functions"))
            cells.append(self._make_code_cell(self._render_preprocessing_functions()))

        # Variable selection functions
        if self.variable_indices is not None:
            varsel_code = self._render_variable_selection_functions()
            if varsel_code:
                cells.append(self._make_markdown_cell("## 3. Variable Selection"))
                cells.append(self._make_code_cell(varsel_code))

        # Data loading (embedded or file-based)
        section_num = 4
        if self.options.include_data:
            cells.append(self._make_markdown_cell(
                f"## {section_num}. Load Embedded Data\n\n"
                "Data is embedded in this notebook for portability."
            ))
            cells.append(self._make_code_cell(self._generate_embedded_data_section()))
        else:
            cells.append(self._make_markdown_cell(
                f"## {section_num}. Data Loading\n\n"
                "Update the file path and column names for your data."
            ))
            cells.append(self._make_code_cell(self._render_data_loading()))

        # Apply preprocessing
        section_num += 1
        cells.append(self._make_markdown_cell(f"## {section_num}. Apply Preprocessing"))
        cells.append(self._make_code_cell(self._render_preprocessing_application()))

        # Apply variable selection
        if self.variable_indices is not None:
            section_num += 1
            cells.append(self._make_markdown_cell(f"## {section_num}. Apply Variable Selection"))
            cells.append(self._make_code_cell(self._render_variable_selection_application()))

        # Model and cross-validation
        section_num += 1
        cells.append(self._make_markdown_cell(f"## {section_num}. Model Training and Evaluation"))
        model_cv_code = (
            self._render_model() + '\n' +
            self._render_cross_validation() + '\n' +
            self._render_metrics()
        )
        cells.append(self._make_code_cell(model_cv_code))

        # Final model
        section_num += 1
        cells.append(self._make_markdown_cell(f"## {section_num}. Train Final Model"))
        cells.append(self._make_code_cell(self._render_final_model()))

        # Visualization
        if self.options.include_visualization:
            section_num += 1
            cells.append(self._make_markdown_cell(f"## {section_num}. Visualization"))
            cells.append(self._make_code_cell(self._render_visualization()))

        # Create notebook structure
        notebook = {
            'nbformat': 4,
            'nbformat_minor': 5,
            'metadata': {
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'
                },
                'language_info': {
                    'name': 'python',
                    'version': '3.9.0'
                }
            },
            'cells': cells
        }

        if self.options.colab_ready:
            # Add Colab metadata
            notebook['metadata']['colab'] = {
                'name': f'spectral_analysis_{self.model_name}.ipynb',
                'provenance': []
            }

        return notebook

    def _get_pip_packages(self) -> List[str]:
        """Get list of required pip packages for installation."""
        packages = ['numpy', 'pandas', 'scikit-learn', 'scipy']

        if self.model_name.lower() == 'lightgbm':
            packages.append('lightgbm')
        elif self.model_name.lower() == 'xgboost':
            packages.append('xgboost')
        elif self.model_name.lower() == 'catboost':
            packages.append('catboost')

        if self.options.include_visualization:
            packages.append('matplotlib')

        if self.imbalance_method and self.imbalance_method.lower() == 'smote':
            packages.append('imbalanced-learn')

        return packages

    def save_script(self, filepath: str) -> None:
        """Save generated script to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_script())

    def save_notebook(self, filepath: str) -> None:
        """Save generated notebook to file."""
        notebook = self.generate_notebook()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)

    # =========================================================================
    # Data embedding methods
    # =========================================================================

    @staticmethod
    def _encode_array(arr: np.ndarray) -> str:
        """Encode numpy array to base64+gzip string."""
        # Convert to JSON-serializable format
        data_bytes = json.dumps(arr.tolist()).encode('utf-8')
        # Compress with gzip
        compressed = gzip.compress(data_bytes, compresslevel=9)
        # Encode to base64
        encoded = base64.b64encode(compressed).decode('ascii')
        return encoded

    @staticmethod
    def _generate_decode_function() -> str:
        """Generate function to decode embedded data."""
        return '''
def _decode_embedded_data(encoded_str):
    """Decode base64+gzip encoded data."""
    import base64
    import gzip
    import json
    import numpy as np

    decoded = base64.b64decode(encoded_str.encode('ascii'))
    decompressed = gzip.decompress(decoded)
    data_list = json.loads(decompressed.decode('utf-8'))
    return np.array(data_list)
'''

    def _generate_embedded_data_section(self) -> str:
        """Generate embedded data section for script."""
        if not self.options.include_data:
            return ""

        sections = [
            "\n# =============================================================================",
            "# EMBEDDED DATA",
            "# =============================================================================\n",
            self._generate_decode_function(),
        ]

        # Encode data arrays
        X_encoded = self._encode_array(self.options.data_X)
        y_encoded = self._encode_array(self.options.data_y)

        sections.append("\n# Spectral data (base64+gzip encoded)")
        sections.append(f"_X_ENCODED = '''{X_encoded}'''")

        sections.append("\n# Target values (base64+gzip encoded)")
        sections.append(f"_Y_ENCODED = '''{y_encoded}'''")

        if self.options.wavelengths is not None:
            wl_encoded = self._encode_array(self.options.wavelengths)
            sections.append("\n# Wavelengths (base64+gzip encoded)")
            sections.append(f"_WAVELENGTHS_ENCODED = '''{wl_encoded}'''")

        # Decode and assign
        sections.append("\n# Decode embedded data")
        sections.append("X = _decode_embedded_data(_X_ENCODED)")
        sections.append("y = _decode_embedded_data(_Y_ENCODED)")

        if self.options.wavelengths is not None:
            sections.append("wavelengths = _decode_embedded_data(_WAVELENGTHS_ENCODED)")

        sections.append(f'\nprint(f"Loaded embedded data: X shape = {{X.shape}}, y shape = {{y.shape}}")')

        return '\n'.join(sections)

    # =========================================================================
    # Private methods for rendering sections
    # =========================================================================

    def _render_header(self) -> str:
        """Render the script header with imports."""
        # Collect all necessary imports
        imports = [
            'import numpy as np',
            'import pandas as pd',
        ]

        # Model-specific import (adjusted for task type)
        model_name_for_import = self.model_name
        if self.task_type == 'classification':
            # Adjust model name for classification variants
            if self.model_name == 'MLP' or self.model_name == 'MLPRegressor':
                model_name_for_import = 'MLPClassifier'
            elif self.model_name == 'RandomForest':
                model_name_for_import = 'RandomForestClassifier'
            elif self.model_name == 'LightGBM':
                model_name_for_import = 'LightGBMClassifier'
            elif self.model_name == 'XGBoost':
                model_name_for_import = 'XGBoostClassifier'
            elif self.model_name == 'CatBoost':
                model_name_for_import = 'CatBoostClassifier'
            elif self.model_name == 'SVR':
                model_name_for_import = 'SVC'
            elif self.model_name == 'PLS':
                model_name_for_import = 'PLSDA'

        model_import = get_model_imports(model_name_for_import)
        if model_import:
            imports.append(model_import)

        # Scipy for preprocessing
        if 'deriv' in self.preprocessing.lower() or 'sg' in self.preprocessing.lower():
            imports.append('from scipy.signal import savgol_filter')

        # sklearn imports
        sklearn_imports = [
            'from sklearn.model_selection import cross_val_predict, KFold',
            'from sklearn.metrics import mean_squared_error, r2_score',
        ]
        if self.task_type == 'classification':
            sklearn_imports = [
                'from sklearn.model_selection import cross_val_predict, StratifiedKFold',
                'from sklearn.metrics import accuracy_score, f1_score, confusion_matrix',
            ]
        imports.extend(sklearn_imports)

        # Imbalance handling imports
        if self.imbalance_method:
            if self.imbalance_method.lower() == 'smote':
                imports.append('from imblearn.over_sampling import SMOTE')

        # Build model details string
        model_details = self._format_model_details()

        # Variable selection info
        var_sel_info = ''
        if self.variable_indices is not None:
            var_sel_method = self.config.get('variable_selection_method', 'custom')
            n_vars = len(self.variable_indices) if hasattr(self.variable_indices, '__len__') else 'N/A'
            var_sel_info = f'Variable Selection: {var_sel_method} ({n_vars} variables)'

        # Extra packages
        extra_packages = ''
        if self.model_name.lower() in ['lightgbm', 'xgboost', 'catboost']:
            pkg = self.model_name.lower()
            extra_packages = f', {pkg}'
        if self.options.include_visualization:
            extra_packages += ', matplotlib'
        if self.imbalance_method and self.imbalance_method.lower() == 'smote':
            extra_packages += ', imbalanced-learn'

        return HEADER_TEMPLATE.format(
            date=datetime.now().strftime('%Y-%m-%d'),
            model_name=self.model_name,
            model_details=model_details,
            preprocessing=self.preprocessing,
            variable_selection_info=var_sel_info,
            cv_folds=self.cv_folds,
            extra_packages=extra_packages,
            imports='\n'.join(imports)
        )

    def _format_model_details(self) -> str:
        """Format model parameters as string."""
        details = []
        for key, value in self.params.items():
            details.append(f'{key}: {value}')
        return ', '.join(details) if details else 'default parameters'

    def _get_imports_code(self) -> str:
        """Get just the imports for notebook."""
        imports = [
            'import numpy as np',
            'import pandas as pd',
        ]

        model_import = get_model_imports(self.model_name)
        if model_import:
            imports.append(model_import)

        if 'deriv' in self.preprocessing.lower() or 'sg' in self.preprocessing.lower():
            imports.append('from scipy.signal import savgol_filter')

        if self.task_type == 'classification':
            imports.extend([
                'from sklearn.model_selection import cross_val_predict, StratifiedKFold',
                'from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report',
            ])
        else:
            imports.extend([
                'from sklearn.model_selection import cross_val_predict, KFold',
                'from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error',
            ])

        if self.options.include_visualization:
            imports.append('import matplotlib.pyplot as plt')

        return '\n'.join(imports)

    def _render_preprocessing_functions(self) -> str:
        """Render preprocessing function definitions."""
        # Map v1 preprocessing names to template names
        preproc_map = {
            'snv': 'snv',
            'sg1': 'deriv1_w7',
            'sg2': 'deriv2_w7',
            'deriv_snv': 'deriv1_w7',  # Will also add SNV
        }

        preproc_key = preproc_map.get(self.preprocessing.lower(), self.preprocessing)
        template_code, _ = get_preprocessing_template(preproc_key)

        # For deriv_snv, also add SNV
        if 'deriv_snv' in self.preprocessing.lower():
            from .templates.preprocessing import SNV_TEMPLATE
            template_code = SNV_TEMPLATE + '\n' + template_code

        if template_code:
            return (
                "\n# =============================================================================\n"
                "# PREPROCESSING FUNCTIONS\n"
                "# =============================================================================\n"
                + template_code
            )
        return ''

    def _render_variable_selection_functions(self) -> str:
        """Render variable selection function definitions."""
        method = self.config.get('variable_selection_method', '')
        if not method:
            return ''

        template = get_variable_selection_template(method)
        if template:
            return (
                "\n# =============================================================================\n"
                f"# VARIABLE SELECTION: {method.upper()}\n"
                "# =============================================================================\n"
                + template
            )
        return ''

    def _render_data_loading(self) -> str:
        """Render data loading section."""
        if self.task_type == 'classification':
            return DATA_LOADING_CLASSIFICATION_TEMPLATE.format(
                data_path=self.options.data_path,
                target_column=self.options.target_column
            )
        else:
            return DATA_LOADING_TEMPLATE.format(
                data_path=self.options.data_path,
                target_column=self.options.target_column
            )

    def _render_preprocessing_application(self) -> str:
        """Render preprocessing application code."""
        preproc = self.preprocessing.lower()

        if preproc == 'raw':
            return '\n# No preprocessing - using raw spectra\nX_processed = X.copy()\n'

        # Map v1 preprocessing to application code
        code_lines = [
            "\n# =============================================================================",
            "# APPLY PREPROCESSING",
            "# =============================================================================\n"
        ]

        if preproc == 'snv':
            code_lines.append("X_processed = apply_snv(X)")
        elif preproc == 'sg1':
            code_lines.append("X_processed = apply_savgol_derivative(X, derivative=1, window_length=7)")
        elif preproc == 'sg2':
            code_lines.append("X_processed = apply_savgol_derivative(X, derivative=2, window_length=7)")
        elif preproc == 'deriv_snv':
            code_lines.append("X_processed = apply_savgol_derivative(X, derivative=1, window_length=7)")
            code_lines.append("X_processed = apply_snv(X_processed)")
        else:
            # Try to use the template system
            _, application_code = get_preprocessing_template(preproc)
            code_lines.append(application_code)

        code_lines.append('\nprint(f"Preprocessed data shape: {X_processed.shape}")')

        return '\n'.join(code_lines)

    def _render_variable_selection_application(self) -> str:
        """Render variable selection application code."""
        if self.variable_indices is None:
            return '\n# No variable selection - using all variables\nX_final = X_processed\n'

        # Convert indices to list for cleaner output
        indices = self.variable_indices
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        # Format indices
        if len(indices) <= 20:
            indices_str = str(indices)
        else:
            indices_str = f"np.array({indices})"

        return (
            "\n# =============================================================================\n"
            "# APPLY VARIABLE SELECTION\n"
            "# =============================================================================\n\n"
            f"# Selected variable indices (from {self.config.get('variable_selection_method', 'selection')})\n"
            f"selected_indices = {indices_str}\n\n"
            "# Apply variable selection\n"
            "X_final = X_processed[:, selected_indices]\n"
            f'print(f"After variable selection: {{X_final.shape[1]}} variables selected")\n'
        )

    def _render_imbalance_handling(self) -> str:
        """Render imbalance handling code for classification."""
        if not self.imbalance_method:
            return ''

        method = self.imbalance_method.lower()

        if method == 'smote':
            return '''
# =============================================================================
# IMBALANCE HANDLING: SMOTE
# =============================================================================

# Apply SMOTE to handle class imbalance
smote = SMOTE(k_neighbors=5, random_state=42)

# Determine which variable to use
X_to_balance = X_final if 'X_final' in locals() else (X_processed if 'X_processed' in locals() else X)

# Apply SMOTE
X_balanced, y_balanced = smote.fit_resample(X_to_balance, y)
print(f"SMOTE applied: {X_to_balance.shape[0]} samples -> {X_balanced.shape[0]} samples")

# Update variable names to use balanced data
X_final = X_balanced
y = y_balanced
'''
        elif method == 'class_weight':
            # Note: class_weight='balanced' is set directly in model params
            return '''
# =============================================================================
# IMBALANCE HANDLING: Class Weighting
# =============================================================================

# Note: Model will use class_weight='balanced' parameter
# This automatically adjusts weights inversely proportional to class frequencies
'''
        else:
            return f'\n# Imbalance method: {self.imbalance_method} (not implemented in export)\n'

    def _render_model(self) -> str:
        """Render model instantiation code."""
        # If using class_weight for imbalance handling, add it to params
        params = self.params.copy()
        if self.imbalance_method and self.imbalance_method.lower() == 'class_weight':
            if self.task_type == 'classification':
                # Only add for models that support class_weight
                supports_class_weight = ['RandomForest', 'LogisticRegression', 'SVC']
                if any(m in self.model_name for m in supports_class_weight):
                    params['class_weight'] = 'balanced'

        model_code = get_model_template(
            self.model_name,
            params,
            self.task_type
        )
        return (
            "\n# =============================================================================\n"
            "# MODEL DEFINITION\n"
            "# =============================================================================\n"
            + model_code
        )

    def _render_cross_validation(self) -> str:
        """Render cross-validation code."""
        # Determine final variable name
        if self.variable_indices is not None:
            x_var = 'X_final'
        elif self.preprocessing != 'raw':
            x_var = 'X_processed'
        else:
            x_var = 'X'

        cv_code = get_cross_validation_template(self.task_type, self.cv_folds)

        # Replace X_final with appropriate variable
        cv_code = cv_code.replace('X_final', x_var)

        return cv_code

    def _render_metrics(self) -> str:
        """Render metrics calculation code."""
        return get_metrics_template(self.task_type, self.cv_folds)

    def _render_final_model(self) -> str:
        """Render final model training code."""
        # Determine which variable to use
        if self.variable_indices is not None:
            x_var = 'X_final'
        elif self.preprocessing != 'raw':
            x_var = 'X_processed'
        else:
            x_var = 'X'

        return f'''
# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

# Train the model on all data
model.fit({x_var}, y)
print(f"\\nFinal model trained on {{{x_var}.shape[0]}} samples with {{{x_var}.shape[1]}} features")
'''

    def _render_visualization(self) -> str:
        """Render visualization code."""
        has_var_sel = self.variable_indices is not None
        return get_visualization_code(
            self.task_type,
            include_spectra=True,
            include_variable_importance=has_var_sel
        )

    # =========================================================================
    # Notebook cell helpers
    # =========================================================================

    def _make_markdown_cell(self, source: str) -> dict:
        """Create a markdown notebook cell."""
        # Jupyter expects each line to end with \n (except possibly the last)
        lines = source.split('\n')
        # Add \n to all lines except the last
        source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        return {
            'cell_type': 'markdown',
            'metadata': {},
            'source': source_lines
        }

    def _make_code_cell(self, source: str) -> dict:
        """Create a code notebook cell."""
        # Jupyter expects each line to end with \n (except possibly the last)
        lines = source.split('\n')
        # Add \n to all lines except the last
        source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        return {
            'cell_type': 'code',
            'metadata': {},
            'source': source_lines,
            'outputs': [],
            'execution_count': None
        }


def generate_script_from_config(model_config: Dict[str, Any],
                                include_visualization: bool = False,
                                data_path: str = 'your_data.csv') -> str:
    """
    Convenience function to generate a script from model configuration.

    Parameters
    ----------
    model_config : dict
        Model configuration dictionary
    include_visualization : bool
        Include matplotlib visualization code
    data_path : str
        Placeholder data path in generated script

    Returns
    -------
    str
        Complete Python script
    """
    options = ExportOptions(
        include_visualization=include_visualization,
        data_path=data_path,
        target_column=model_config.get('target_name', 'target')
    )
    generator = CodeGenerator(model_config, options)
    return generator.generate_script()


def generate_notebook_from_config(model_config: Dict[str, Any],
                                  include_visualization: bool = True) -> dict:
    """
    Convenience function to generate a notebook from model configuration.

    Parameters
    ----------
    model_config : dict
        Model configuration dictionary
    include_visualization : bool
        Include visualization cells

    Returns
    -------
    dict
        Jupyter notebook structure
    """
    options = ExportOptions(
        include_visualization=include_visualization,
        format='notebook',
        target_column=model_config.get('target_name', 'target')
    )
    generator = CodeGenerator(model_config, options)
    return generator.generate_notebook()
