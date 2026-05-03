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

from .templates.header import HEADER_TEMPLATE, DATA_LOADING_TEMPLATE, DATA_LOADING_CLASSIFICATION_TEMPLATE, DATA_LOADING_ONE_CLASS_TEMPLATE
from .templates.preprocessing import get_preprocessing_template, SNV_TEMPLATE, SAVGOL_DERIVATIVE_TEMPLATE
from .templates.variable_selection import get_variable_selection_template
from .templates.models import get_model_template, get_model_imports, MODEL_IMPORTS, DEFAULT_PARAMS, ONE_CLASS_MODELS, ONE_CLASS_NEEDS_SCALING, PCASIMCA_CLASS_TEMPLATE
from .templates.validation import (
    get_cross_validation_template,
    get_metrics_template,
    get_final_model_template,
    get_prediction_template,
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
        # CV metadata may live at the top level (legacy) or under training_config
        # (canonical, set by search.py since the cv-strategy-overhaul work). Fall
        # back to training_config so exported scripts/notebooks/R wrappers always
        # reproduce the user's selected CV regime — otherwise an LOO model would
        # silently emit KFold(...) on export.
        _training_config = model_config.get('training_config') or {}
        self.cv_folds = model_config.get('cv_folds', _training_config.get('folds', 5))
        self.cv_strategy = model_config.get('cv_strategy', _training_config.get('cv_strategy', 'kfold'))
        self.cv_n_repeats = model_config.get('cv_n_repeats', _training_config.get('cv_n_repeats', 5))
        self.imbalance_method = model_config.get('imbalance_method', None)
        self.inlier_class_label = model_config.get('inlier_class_label', '')

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
        # Handle both numpy arrays and lists
        def get_size(obj):
            if hasattr(obj, 'size'):
                return obj.size
            elif hasattr(obj, '__len__'):
                return len(obj)
            return 0

        total_elements = get_size(self.options.data_X) + get_size(self.options.data_y)
        if self.options.wavelengths is not None:
            total_elements += get_size(self.options.wavelengths)

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

        # 2. Preprocessing functions (skip if data is embedded - already preprocessed)
        if self.options.include_preprocessing and self.preprocessing != 'raw' and not self.options.include_data:
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

        # 7. Imbalance handling (classification or regression)
        if self.imbalance_method:
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
            sections.append(get_prediction_template(self.task_type))

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

        # Dependency install cell (always include if non-standard packages are required)
        pip_packages = self._get_pip_packages()
        needs_install_cell = self.options.colab_ready or any(
            pkg in pip_packages for pkg in ['catboost', 'xgboost', 'lightgbm', 'imbalanced-learn']
        )
        if needs_install_cell:
            cells.append(self._make_markdown_cell(
                "## 0. Install Dependencies\n\n"
                "Run this cell first if you're missing any required packages.\n"
                "Skip if running locally with packages already installed."
            ))
            install_code = (
                "import sys\n"
                "import subprocess\n"
                "\n"
                f"pkgs = {pip_packages!r}\n"
                "print('Installing:', ' '.join(pkgs))\n"
                "try:\n"
                "    if 'google.colab' in sys.modules:\n"
                "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])\n"
                "    else:\n"
                "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])\n"
                "    print('Install complete.')\n"
                "except Exception as e:\n"
                "    print('Install failed in this environment:', e)\n"
                "    print('If this is a browser/pyodide notebook, install is not available.')\n"
                "    print('Run this notebook in Colab or a local Jupyter kernel with pip enabled.')\n"
            )
            cells.append(self._make_code_cell(install_code))

        # Imports cell
        cells.append(self._make_markdown_cell("## 1. Setup and Imports"))
        cells.append(self._make_code_cell(self._get_imports_code()))

        # Preprocessing functions (skip if data is embedded - already preprocessed)
        if self.preprocessing != 'raw' and not self.options.include_data:
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

        # Apply preprocessing (or skip note if embedded data)
        section_num += 1
        if self.options.include_data:
            cells.append(self._make_markdown_cell(
                f"## {section_num}. Preprocessing (Skipped)\n\n"
                "Embedded data is already preprocessed with the same preprocessing "
                f"used during model training ({self.preprocessing}). No additional "
                "preprocessing is applied to avoid double-processing."
            ))
        else:
            cells.append(self._make_markdown_cell(f"## {section_num}. Apply Preprocessing"))
        cells.append(self._make_code_cell(self._render_preprocessing_application()))

        # Apply variable selection
        if self.variable_indices is not None:
            section_num += 1
            cells.append(self._make_markdown_cell(f"## {section_num}. Apply Variable Selection"))
            cells.append(self._make_code_cell(self._render_variable_selection_application()))

        # Imbalance handling (classification or regression)
        if self.imbalance_method:
            section_num += 1
            cells.append(self._make_markdown_cell(f"## {section_num}. Imbalance Handling"))
            cells.append(self._make_code_cell(self._render_imbalance_handling()))

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

        # Add imbalanced-learn for classification resampling methods
        classification_resample_methods = [
            'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
            'tomek_links', 'smote_tomek', 'smote_enn'
        ]
        if self.imbalance_method and self.imbalance_method.lower() in classification_resample_methods:
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
    def _encode_array(arr) -> str:
        """Encode array (numpy or list) to base64+gzip string."""
        # Convert to JSON-serializable format
        # Handle both numpy arrays and lists
        if hasattr(arr, 'tolist'):
            data = arr.tolist()
        else:
            data = list(arr)
        data_bytes = json.dumps(data).encode('utf-8')
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

        if self.task_type == 'one_class':
            sections.append('\ny_oc = y')
            sections.append('inlier_indices = np.where(y_oc == 1)[0]')
            sections.append('outlier_indices = np.where(y_oc == -1)[0]')
            sections.append(f'print(f"Inliers: {{len(inlier_indices)}}, Outliers: {{len(outlier_indices)}}")')

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

        # Model-specific import (normalize aliases + task type)
        model_import = get_model_imports(self._resolve_model_class_name())
        if model_import:
            imports.append(model_import)

        # Pipeline/scaling imports (must match Model Development)
        if self._needs_standard_scaler() or self._needs_pls_da_pipeline():
            imports.append('from sklearn.pipeline import Pipeline')
            imports.append('from sklearn.preprocessing import StandardScaler')
        if self._needs_pls_da_pipeline():
            imports.append('from sklearn.linear_model import LogisticRegression')

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
        elif self.task_type == 'one_class':
            sklearn_imports = [
                'from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score',
            ]
            if self.model_name in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
                imports.append('from sklearn.preprocessing import StandardScaler')
            if self.model_name == 'EllipticEnvelope':
                imports.append('from sklearn.decomposition import PCA')
        imports.extend(sklearn_imports)

        # Imbalance handling imports
        if self.imbalance_method:
            method = self.imbalance_method.lower()
            class_methods = {
                'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                'tomek_links', 'smote_tomek', 'smote_enn'
            }
            if method in class_methods:
                imports.extend([
                    'from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE',
                    'from imblearn.under_sampling import RandomUnderSampler, TomekLinks',
                    'from imblearn.combine import SMOTETomek, SMOTEENN',
                ])

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
        classification_resample_methods = [
            'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
            'tomek_links', 'smote_tomek', 'smote_enn',
        ]
        if self.imbalance_method and self.imbalance_method.lower() in classification_resample_methods:
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

        model_import = get_model_imports(self._resolve_model_class_name())
        if model_import:
            imports.append(model_import)

        # Pipeline/scaling imports (must match Model Development)
        if self._needs_standard_scaler() or self._needs_pls_da_pipeline():
            imports.append('from sklearn.pipeline import Pipeline')
            imports.append('from sklearn.preprocessing import StandardScaler')
        if self._needs_pls_da_pipeline():
            imports.append('from sklearn.linear_model import LogisticRegression')

        if 'deriv' in self.preprocessing.lower() or 'sg' in self.preprocessing.lower():
            imports.append('from scipy.signal import savgol_filter')

        if self.task_type == 'classification':
            imports.extend([
                'from sklearn.model_selection import cross_val_predict, StratifiedKFold',
                'from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report',
            ])
        elif self.task_type == 'one_class':
            imports.extend([
                'from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score',
            ])
            if self.model_name in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
                imports.append('from sklearn.preprocessing import StandardScaler')
            if self.model_name == 'EllipticEnvelope':
                imports.append('from sklearn.decomposition import PCA')
        else:
            imports.extend([
                'from sklearn.model_selection import cross_val_predict, KFold',
                'from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error',
            ])

        if self.options.include_visualization:
            imports.append('import matplotlib.pyplot as plt')

        # Imbalance handling imports
        if self.imbalance_method:
            method = self.imbalance_method.lower()
            class_methods = {
                'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                'tomek_links', 'smote_tomek', 'smote_enn'
            }
            if method in class_methods:
                imports.extend([
                    'from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE',
                    'from imblearn.under_sampling import RandomUnderSampler, TomekLinks',
                    'from imblearn.combine import SMOTETomek, SMOTEENN',
                ])

        return '\n'.join(imports)

    def _render_preprocessing_functions(self) -> str:
        """Render preprocessing function definitions."""
        # If data is embedded, it's already preprocessed - no functions needed
        if self.options.include_data:
            return ''

        # Get window size from config (default 17, matching GUI default)
        window = self.config.get('window_size', 17)
        deriv_order = self.config.get('deriv_order', None)

        # Map v1 preprocessing names to template names, using actual window size
        preproc_map = {
            'snv': 'snv',
            'sg1': f'deriv1_w{window}',
            'sg2': f'deriv2_w{window}',
            'sg3': f'deriv3_w{window}',
            'snv_sg1': f'snv_deriv1_w{window}',
            'snv_sg2': f'snv_deriv2_w{window}',
            'deriv_snv': f'deriv{deriv_order}_w{window}' if deriv_order else f'deriv1_w{window}',  # Will also add SNV
            'snv_deriv': f'snv_deriv{deriv_order}_w{window}' if deriv_order else f'snv_deriv1_w{window}',
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
        if self.task_type == 'one_class':
            return DATA_LOADING_ONE_CLASS_TEMPLATE.format(
                data_path=self.options.data_path,
                target_column=self.options.target_column,
                inlier_class_label=self.inlier_class_label or 'Clean',
            )
        elif self.task_type == 'classification':
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
        # If data is embedded, it's already preprocessed - skip this step
        if self.options.include_data:
            return '''
# =============================================================================
# PREPROCESSING (SKIPPED - embedded data is already preprocessed)
# =============================================================================

# The embedded data has already been preprocessed with the same
# preprocessing used during model training. No additional preprocessing needed.
# Preprocessing applied: ''' + self.preprocessing + '''
X_processed = X.copy()
print(f"Using pre-processed embedded data: {X_processed.shape}")
'''

        preproc = self.preprocessing.lower()

        # Get window size from config (default 17, matching GUI default)
        window = self.config.get('window_size', 17)

        # Map v1 preprocessing to application code
        code_lines = [
            "\n# =============================================================================",
            "# APPLY PREPROCESSING",
            "# =============================================================================\n"
        ]

        # Track if we're using derivatives (edge trimming is optional)
        uses_derivative = preproc in ['sg1', 'sg2', 'sg3', 'snv_sg1', 'snv_sg2', 'snv_deriv', 'deriv_snv']

        if preproc == 'raw':
            # T-36 fix (post-merge review): cannot early-return here, otherwise
            # the autoscale block below is skipped for raw+autoscale exports.
            code_lines.append("# No preprocessing - using raw spectra")
            code_lines.append("X_processed = X.copy()")
        elif preproc == 'snv':
            code_lines.append("X_processed = apply_snv(X)")
        elif preproc == 'sg1':
            code_lines.append(f"X_processed = apply_savgol_derivative(X, derivative=1, window_length={window})")
        elif preproc == 'sg2':
            code_lines.append(f"X_processed = apply_savgol_derivative(X, derivative=2, window_length={window})")
        elif preproc == 'sg3':
            code_lines.append(f"X_processed = apply_savgol_derivative(X, derivative=3, window_length={window})")
        elif preproc == 'snv_sg1':
            code_lines.append("X_processed = apply_snv(X)")
            code_lines.append(f"X_processed = apply_savgol_derivative(X_processed, derivative=1, window_length={window})")
        elif preproc == 'snv_sg2':
            code_lines.append("X_processed = apply_snv(X)")
            code_lines.append(f"X_processed = apply_savgol_derivative(X_processed, derivative=2, window_length={window})")
        elif preproc == 'snv_deriv':
            deriv = self.config.get('deriv_order', 1) or 1
            code_lines.append("X_processed = apply_snv(X)")
            code_lines.append(f"X_processed = apply_savgol_derivative(X_processed, derivative={deriv}, window_length={window})")
        elif preproc == 'deriv_snv':
            deriv = self.config.get('deriv_order', 1) or 1
            code_lines.append(f"X_processed = apply_savgol_derivative(X, derivative={deriv}, window_length={window})")
            code_lines.append("X_processed = apply_snv(X_processed)")
        else:
            # Try to use the template system
            _, application_code = get_preprocessing_template(preproc)
            code_lines.append(application_code)

        # T-36: autoscale (UV scaling) — applied AFTER SNV/derivatives so the
        # exported pipeline reproduces the trained model exactly.
        # Use the robust parser so round-tripped CSV/JSON values like the string
        # "False" are not silently coerced to True by Python's bool() builtin.
        if self._autoscale_enabled():
            code_lines.append("\n# Autoscale (UV scaling): mean-center + unit variance per wavelength column")
            code_lines.append("from sklearn.preprocessing import StandardScaler as _AutoscaleStandardScaler")
            code_lines.append("X_processed = _AutoscaleStandardScaler().fit_transform(X_processed)")

        # Optional edge trimming for derivatives (disabled by default for GUI parity)
        trim_edges = bool(self.config.get('trim_derivative_edges', False))
        if uses_derivative and trim_edges:
            half_window = window // 2
            code_lines.append(f"\n# Trim {half_window} edge wavelengths on each side (SG boundary artifacts)")
            code_lines.append(f"X_processed = X_processed[:, {half_window}:-{half_window}]")
            code_lines.append(f"if 'wavelengths' in dir():")
            code_lines.append(f"    wavelengths = wavelengths[{half_window}:-{half_window}]")

        code_lines.append('\nprint(f"Preprocessed data shape: {X_processed.shape}")')

        return '\n'.join(code_lines)

    def _render_variable_selection_application(self) -> str:
        """Render variable selection application code."""
        if self.variable_indices is None:
            return '\n# No variable selection - using all variables\nX_final = X_processed\n'

        # If data is embedded, it's already subset to selected variables - skip this step
        if self.options.include_data:
            return (
                "\n# =============================================================================\n"
                "# VARIABLE SELECTION (SKIPPED - embedded data is already subset)\n"
                "# =============================================================================\n\n"
                "# The embedded data has already been subset to the selected variables\n"
                "# during model training. No additional variable selection needed.\n"
                "if 'wavelengths' in dir():\n"
                "    selected_indices = np.arange(len(wavelengths))\n"
                "else:\n"
                "    selected_indices = np.arange(X_processed.shape[1])\n"
                "X_final = X_processed.copy()\n"
                'print(f"Using pre-subset embedded data: {X_final.shape[1]} variables")\n'
            )

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

    def _render_model(self) -> str:
        """Render model instantiation code."""
        # NOTE: do NOT bake class_weight='balanced' into params here. Auto mode
        # may resolve at runtime to "no correction" on mild-imbalance data
        # (below the 3:1 threshold), and a baked kwarg would silently weight
        # the loss anyway, contradicting the Auto-mode contract. The catch-all
        # else branch + StandardScaler-wrapped path + PLS-DA path each emit a
        # runtime conditional gated on IMBALANCE_METHOD == 'class_weight'.
        params = self._normalize_model_params(self.params)

        if self.model_name.lower() == 'neuralboosted':
            warning = (
                "\n# WARNING: NeuralBoosted export cannot fully reproduce GUI behavior.\n"
                "# Consider exporting a .dasp model for exact predictions.\n"
            )
        else:
            warning = ""

        model_class = self._resolve_model_class_name()

        if self._needs_pls_da_pipeline():
            pls_params, lr_params = self._split_pls_da_params(self.params)
            class_weight_conditional = self._imbalance_method_is_class_weight_like()
            model_code = warning + self._render_pls_da_pipeline(
                pls_params, lr_params, class_weight_conditional=class_weight_conditional
            )
        elif self.task_type == 'one_class':
            model_code = warning + self._render_one_class_model(params)
        elif self._needs_standard_scaler() and not self._autoscale_enabled():
            default_key = self._resolve_default_param_key()
            params_full = DEFAULT_PARAMS.get(default_key, {}).copy()
            params_full.update(params)

            # Runtime-conditional balanced kwarg (gated on resolved IMBALANCE_METHOD).
            # MLP excluded — TypeError on class_weight, sklearn floor too low for
            # sample_weight at fit. Runtime falls back to unweighted per
            # search.py:4439-4444; mirror that here.
            scaled_balanced_kwarg = None
            if self._imbalance_method_is_class_weight_like() and not model_class.startswith('MLP'):
                scaled_balanced_kwarg = ('class_weight', 'balanced')

            if 'random_state' not in params_full:
                params_full['random_state'] = 42

            ctor_class = self._resolve_model_ctor_class()
            model_code = warning + self._render_scaled_pipeline(
                params_full, ctor_class, balanced_kwarg=scaled_balanced_kwarg
            )
        else:
            default_key = self._resolve_default_param_key()
            params_full = DEFAULT_PARAMS.get(default_key, {}).copy()
            params_full.update(params)

            # Per-library balanced-loss kwarg name (None = no constructor knob;
            # XGBoost is handled via sample_weight at fit() time). Threaded as a
            # *runtime conditional* on IMBALANCE_METHOD rather than baked into
            # the constructor literal — necessary for Auto mode, because Auto
            # may resolve to None on mild-imbalance data and we don't want a
            # baked balanced kwarg to silently weight the loss in that case.
            balanced_kwarg_name = None
            balanced_kwarg_value = None
            if self._imbalance_method_is_class_weight_like():
                if model_class.startswith('CatBoost'):
                    balanced_kwarg_name = 'auto_class_weights'
                    balanced_kwarg_value = 'Balanced'
                elif model_class.startswith('XGBoost') or model_class.startswith('XGB'):
                    pass  # sample_weight threaded at fit, not __init__
                else:
                    balanced_kwarg_name = 'class_weight'
                    balanced_kwarg_value = 'balanced'

            # Avoid injecting random_state into estimators that don't accept it (e.g., PLSRegression)
            ctor_class = self._resolve_model_ctor_class()
            if ctor_class != 'PLSRegression':
                if 'random_state' not in params_full:
                    params_full['random_state'] = 42
            if model_class.startswith('LightGBM') and 'verbosity' not in params_full:
                params_full['verbosity'] = -1
            if model_class.startswith('XGBoost'):
                params_full.setdefault('tree_method', 'hist')
                params_full.setdefault('verbosity', 0)
            if model_class.startswith('CatBoost'):
                # CatBoost does not allow both depth and max_depth
                if 'depth' in params_full and 'max_depth' in params_full:
                    params_full.pop('max_depth', None)
                # CatBoost does not allow multiple iteration synonyms
                if 'iterations' in params_full:
                    params_full.pop('n_estimators', None)
                    params_full.pop('num_boost_round', None)
                    params_full.pop('num_trees', None)
                if 'verbose' not in params_full:
                    params_full['verbose'] = 0
            if model_class.startswith('LightGBM'):
                # LightGBM does not allow both n_estimators and num_iterations
                if 'n_estimators' in params_full and 'num_iterations' in params_full:
                    params_full.pop('num_iterations', None)
                # Drop legacy aliases if present alongside n_estimators
                if 'n_estimators' in params_full and 'num_boost_round' in params_full:
                    params_full.pop('num_boost_round', None)
            if model_class.startswith('XGBoost'):
                # XGBoost does not allow both n_estimators and num_boost_round
                if 'n_estimators' in params_full and 'num_boost_round' in params_full:
                    params_full.pop('num_boost_round', None)
            if model_class.startswith('LGBM') or model_class.startswith('XGB'):
                params_full.setdefault('n_jobs', -1)

            params_full = self._serialize_param_value(params_full)
            params_literal = repr(params_full)

            if balanced_kwarg_name is not None:
                # Runtime-conditional injection. Required so that Auto mode's
                # post-data-load resolution can choose to NOT apply balanced
                # weighting on mild-imbalance data (below the 3:1 threshold).
                # Pre-fix-of-fixes the kwarg was baked into the constructor
                # literal, which contradicted the Auto mode "no correction"
                # contract on mild-imbalance data.
                balanced_value_literal = repr(balanced_kwarg_value)
                model_code = (
                    warning
                    + f"\n# {ctor_class} with full parameter pass-through\n"
                    + f"model_params = {params_literal}\n"
                    + f"if IMBALANCE_METHOD == 'class_weight':\n"
                    + f"    model_params[{balanced_kwarg_name!r}] = {balanced_value_literal}\n"
                    + f"model = {ctor_class}(**model_params)\n"
                )
            else:
                model_code = (
                    warning
                    + f"\n# {ctor_class} with full parameter pass-through\n"
                    + f"model_params = {params_literal}\n"
                    + f"model = {ctor_class}(**model_params)\n"
                )
        return (
            "\n# =============================================================================\n"
            "# MODEL DEFINITION\n"
            "# =============================================================================\n"
            + model_code
        )

    # Pipeline-specific params that should never be passed to model constructors
    _PIPELINE_PARAMS = {'memory', 'transform_input', 'verbose', 'steps', 'n_jobs'}

    def _normalize_model_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Strip pipeline prefixes from params for base estimators."""
        normalized = {}
        for key, value in (params or {}).items():
            # Skip Pipeline-specific params (defense-in-depth)
            if key in self._PIPELINE_PARAMS:
                continue
            if key.startswith('model__'):
                normalized[key[7:]] = value
            elif key.startswith('estimator__'):
                normalized[key[11:]] = value
            elif key.startswith('base_estimator__'):
                normalized[key[15:]] = value
            elif key.startswith('scaler__'):
                # Scaler params not used in exports yet
                continue
            elif key.startswith('pls__') or key.startswith('lr__'):
                # Handled separately for PLS-DA
                continue
            else:
                normalized[key] = value
        return normalized

    def _resolve_model_class_name(self) -> str:
        """Resolve model class name used in templates."""
        normalized = self.model_name.replace(' ', '').replace('-', '')

        aliases = {
            'RF': 'RandomForest',
            'LGBM': 'LightGBM',
            'XGB': 'XGBoost',
            'CB': 'CatBoost',
            'SVM': 'SVR',
            'NN': 'MLP',
        }

        if normalized.upper() in aliases:
            normalized = aliases[normalized.upper()]

        if self.task_type == 'classification':
            if normalized == 'RandomForest':
                normalized = 'RandomForestClassifier'
            elif normalized == 'LightGBM':
                normalized = 'LightGBMClassifier'
            elif normalized == 'XGBoost':
                normalized = 'XGBoostClassifier'
            elif normalized == 'CatBoost':
                normalized = 'CatBoostClassifier'
            elif normalized == 'SVR':
                normalized = 'SVC'
            elif normalized == 'PLS':
                normalized = 'PLSDA'
            elif normalized == 'MLP':
                normalized = 'MLPClassifier'
        elif self.task_type == 'one_class':
            oc_name_map = {
                'IsolationForest': 'IsolationForest',
                'OneClassSVM': 'OneClassSVM',
                'EllipticEnvelope': 'EllipticEnvelope',
                'LOF': 'LOF',
                'PCASIMCA': 'PCA-SIMCA',
            }
            if normalized in oc_name_map:
                normalized = oc_name_map[normalized]

        return normalized

    def _resolve_model_ctor_class(self) -> str:
        """Resolve actual constructor class name for instantiation."""
        normalized = self.model_name.replace(' ', '').replace('-', '')

        aliases = {
            'RF': 'RandomForest',
            'LGBM': 'LightGBM',
            'XGB': 'XGBoost',
            'CB': 'CatBoost',
            'SVM': 'SVR',
            'NN': 'MLP',
        }

        if normalized.upper() in aliases:
            normalized = aliases[normalized.upper()]

        if self.task_type == 'classification':
            if normalized == 'RandomForest':
                return 'RandomForestClassifier'
            if normalized == 'LightGBM':
                return 'LGBMClassifier'
            if normalized == 'XGBoost':
                return 'XGBClassifier'
            if normalized == 'CatBoost':
                return 'CatBoostClassifier'
            if normalized in ('SVR', 'SVM'):
                return 'SVC'
            if normalized == 'MLP':
                return 'MLPClassifier'
            if normalized == 'PLS':
                return 'PLSRegression'
        elif self.task_type == 'one_class':
            oc_ctor_map = {
                'IsolationForest': 'IsolationForest',
                'OneClassSVM': 'OneClassSVM',
                'EllipticEnvelope': 'EllipticEnvelope',
                'LOF': 'LocalOutlierFactor',
                'PCASIMCA': 'PCASIMCA',
            }
            if normalized in oc_ctor_map:
                return oc_ctor_map[normalized]
            return normalized
        else:
            if normalized == 'RandomForest':
                return 'RandomForestRegressor'
            if normalized == 'LightGBM':
                return 'LGBMRegressor'
            if normalized == 'XGBoost':
                return 'XGBRegressor'
            if normalized == 'CatBoost':
                return 'CatBoostRegressor'
            if normalized == 'MLP':
                return 'MLPRegressor'
            if normalized == 'PLS':
                return 'PLSRegression'

        return normalized

    def _resolve_default_param_key(self) -> str:
        """Resolve DEFAULT_PARAMS key for the current model."""
        normalized = self.model_name.replace(' ', '').replace('-', '')

        aliases = {
            'RF': 'RandomForest',
            'LGBM': 'LightGBM',
            'XGB': 'XGBoost',
            'CB': 'CatBoost',
            'SVM': 'SVR',
            'NN': 'MLP',
        }

        if normalized.upper() in aliases:
            normalized = aliases[normalized.upper()]

        if self.task_type == 'classification':
            if normalized == 'RandomForest':
                return 'RandomForestClassifier'
            if normalized == 'LightGBM':
                return 'LightGBMClassifier'
            if normalized == 'XGBoost':
                return 'XGBoostClassifier'
            if normalized == 'CatBoost':
                return 'CatBoostClassifier'
            if normalized in ('SVR', 'SVM'):
                return 'SVC'
            if normalized == 'MLP':
                return 'MLPClassifier'
            if normalized == 'PLS':
                return 'PLSDA'
        elif self.task_type == 'one_class':
            oc_default_map = {
                'IsolationForest': 'IsolationForest',
                'OneClassSVM': 'OneClassSVM',
                'EllipticEnvelope': 'EllipticEnvelope',
                'LOF': 'LOF',
                'PCASIMCA': 'PCA-SIMCA',
            }
            if normalized in oc_default_map:
                return oc_default_map[normalized]
        return normalized

    def _serialize_param_value(self, value: Any) -> Any:
        """Convert numpy types to plain Python types for code literals."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return tuple(self._serialize_param_value(v) for v in value)
        if isinstance(value, list):
            return [self._serialize_param_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_param_value(v) for k, v in value.items()}
        return value

    def _needs_pls_da_pipeline(self) -> bool:
        """Check if we need the PLS-DA pipeline (PLS scores + LR)."""
        normalized = self.model_name.replace(' ', '').replace('-', '').upper()
        return self.task_type == 'classification' and normalized in ('PLS', 'PLSDA')

    def _needs_standard_scaler(self) -> bool:
        """Check if model needs StandardScaler (matches GUI pipeline)."""
        normalized = self.model_name.replace(' ', '').replace('-', '')
        scale_models = {
            'SVR', 'SVC', 'SVM',
            'MLP', 'MLPRegressor', 'MLPClassifier',
            'NeuralBoosted',
            'Ridge', 'Lasso', 'ElasticNet'
        }
        return normalized in scale_models and not self._needs_pls_da_pipeline()

    def _imbalance_method_is_class_weight_like(self) -> bool:
        """Both 'class_weight' (explicit) and 'auto' (resolves to class_weight at
        runtime if data is imbalanced) trigger the same code-generation paths:
        per-library balanced kwargs in __init__ and the XGBoost sample_weight
        block at fit. For Auto + balanced data, the baked class_weight='balanced'
        is mathematically a no-op (uniform per-class weights ≈ 1) and the
        runtime resolution prevents the sample_weight block from firing."""
        if not self.imbalance_method:
            return False
        return self.imbalance_method.lower() in ('class_weight', 'auto')

    def _xgb_class_weight_needs_sample_weight(self) -> bool:
        """XGBoost classification under imbalance_method='class_weight' (or 'auto')
        threads sample_weight at fit() time because XGBoost has no class_weight
        constructor kwarg. Mirrors the runtime path in search.py."""
        if self.task_type != 'classification':
            return False
        if not self._imbalance_method_is_class_weight_like():
            return False
        model_class = self._resolve_model_class_name()
        return model_class.startswith('XGBoost') or model_class.startswith('XGB')

    def _autoscale_enabled(self) -> bool:
        """Robust parse of self.config['autoscale'] — handles bool, int, NaN, and the
        string forms ('True'/'False'/'1'/'0') that round-tripped CSVs may contain.
        Plain bool() is wrong on the string path: bool('False') is True."""
        raw = self.config.get('autoscale', False)
        if isinstance(raw, str):
            return raw.strip().lower() in ('true', '1', 'yes')
        try:
            import math
            if isinstance(raw, float) and math.isnan(raw):
                return False
        except Exception:
            pass
        return bool(raw)

    def _split_pls_da_params(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split params into PLS and LogisticRegression sets for PLS-DA."""
        pls_params: Dict[str, Any] = {}
        lr_params: Dict[str, Any] = {}
        for key, value in (params or {}).items():
            if key.startswith('pls__'):
                pls_params[key[5:]] = value
            elif key.startswith('lr__'):
                lr_params[key[4:]] = value
            elif key.startswith('scaler__'):
                continue
            else:
                # Allow unprefixed PLS params for older configs
                if key in ('n_components', 'max_iter', 'tol', 'scale'):
                    pls_params[key] = value
                else:
                    lr_params[key] = value
        return pls_params, lr_params

    def _render_pls_da_pipeline(
        self,
        pls_params: Dict[str, Any],
        lr_params: Dict[str, Any],
        class_weight_conditional: bool = False,
    ) -> str:
        """Render a PLS-DA pipeline (PLS scores + scaler + LR).

        ``class_weight_conditional``: when True, emit a runtime conditional
        that injects ``class_weight='balanced'`` into the LR params only when
        IMBALANCE_METHOD resolves to 'class_weight'. Required for Auto mode
        correctness — see the catch-all branch's runtime-conditional emission.
        """
        n_components = pls_params.pop('n_components', 10)
        pls_max_iter = pls_params.pop('max_iter', 500)
        pls_tol = pls_params.pop('tol', 1e-6)
        pls_scale = pls_params.pop('scale', False)

        lr_defaults = {
            'max_iter': 1000,
            'random_state': 42,
        }
        lr_defaults.update(lr_params or {})
        # Strip any literal class_weight; runtime conditional handles it.
        lr_defaults.pop('class_weight', None)
        lr_params_literal = repr(lr_defaults)
        class_weight_block = (
            "if IMBALANCE_METHOD == 'class_weight':\n"
            "    _lr_params['class_weight'] = 'balanced'\n"
            if class_weight_conditional else ""
        )

        return f'''
# PLS-DA pipeline (PLS scores -> StandardScaler -> LogisticRegression)
class PLSTransformer:
    """Minimal PLS transformer for classification pipelines."""
    def __init__(self, n_components=2, max_iter=500, tol=1e-6, scale=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.pls_ = None

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        self.pls_ = PLSRegression(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            scale=self.scale
        )
        y_1d = np.ravel(y) if hasattr(y, 'ndim') and y.ndim > 1 else y
        self.pls_.fit(X, y_1d)
        return self

    def transform(self, X):
        X_scores = self.pls_.transform(X)
        if X_scores.ndim == 1:
            X_scores = X_scores.reshape(-1, 1)
        return X_scores

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {{
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'scale': self.scale
        }}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

pls = PLSTransformer(
    n_components={n_components},
    max_iter={pls_max_iter},
    tol={pls_tol},
    scale={pls_scale}
)
import inspect as _inspect
_lr_params = {lr_params_literal}
{class_weight_block}_lr_sig = _inspect.signature(LogisticRegression)
_lr_filtered = {{k: v for k, v in _lr_params.items() if k in _lr_sig.parameters}}
lr = LogisticRegression(**_lr_filtered)
model = Pipeline([('pls', pls), ('scaler', StandardScaler()), ('lr', lr)])
'''

    def _render_scaled_pipeline(
        self,
        params_full: Dict[str, Any],
        ctor_class: str,
        balanced_kwarg: Optional[Tuple[str, Any]] = None,
    ) -> str:
        """Render a StandardScaler + model pipeline with full params.

        ``balanced_kwarg``: optional (name, value) tuple to inject as a
        runtime conditional gated on IMBALANCE_METHOD == 'class_weight'.
        Required for Auto mode correctness — see the catch-all branch's
        runtime-conditional emission for rationale.
        """
        params_full = self._serialize_param_value(params_full)
        params_literal = repr(params_full)
        if balanced_kwarg is not None:
            kwarg_name, kwarg_value = balanced_kwarg
            conditional = (
                f"if IMBALANCE_METHOD == 'class_weight':\n"
                f"    model_params[{kwarg_name!r}] = {kwarg_value!r}\n"
            )
        else:
            conditional = ""
        return (
            f"\n# {ctor_class} with full parameter pass-through\n"
            f"model_params = {params_literal}\n"
            + conditional
            + f"model = {ctor_class}(**model_params)\n"
            "\n# Add StandardScaler for scale-sensitive models\n"
            "model = Pipeline([('scaler', StandardScaler()), ('model', model)])\n"
        )

    def _render_one_class_model(self, params: Dict[str, Any]) -> str:
        """Render one-class model instantiation (no Pipeline, no y in fit)."""
        ctor_class = self._resolve_model_ctor_class()
        default_key = self._resolve_default_param_key()
        params_full = DEFAULT_PARAMS.get(default_key, {}).copy()
        params_full.update(params)

        if 'random_state' not in params_full and ctor_class not in ('OneClassSVM', 'PCASIMCA', 'LocalOutlierFactor'):
            params_full['random_state'] = 42
        if ctor_class == 'LocalOutlierFactor':
            params_full['novelty'] = True
        if ctor_class == 'IsolationForest' and 'n_jobs' not in params_full:
            params_full['n_jobs'] = 1
        if ctor_class == 'LocalOutlierFactor' and 'n_jobs' not in params_full:
            params_full['n_jobs'] = 1

        params_full = self._serialize_param_value(params_full)
        params_literal = repr(params_full)

        code = ''
        if ctor_class == 'PCASIMCA':
            code = PCASIMCA_CLASS_TEMPLATE + '\n'

        code += (
            f"\n# {ctor_class} (one-class)\n"
            f"model_params = {params_literal}\n"
            f"model = {ctor_class}(**model_params)\n"
        )
        return code

    def _render_cross_validation(self) -> str:
        """Render cross-validation code."""
        if self.imbalance_method:
            return self._render_cross_validation_with_imbalance()

        # Determine final variable name
        # Priority: imbalance resampling > variable selection > preprocessing > raw
        resampling_methods = ['smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                              'tomek_links', 'smote_tomek', 'smote_enn', 'smogn', 'smotetomek',
                              'oversample', 'undersample']

        if self.imbalance_method and self.imbalance_method.lower() in resampling_methods:
            # Resampling methods create X_final and modify y
            x_var = 'X_final'
        elif self.variable_indices is not None:
            x_var = 'X_final'
        elif self.preprocessing != 'raw':
            x_var = 'X_processed'
        else:
            x_var = 'X'

        cv_code = get_cross_validation_template(
            self.task_type, self.cv_folds,
            cv_strategy=self.cv_strategy, cv_n_repeats=self.cv_n_repeats,
            model_name=self.model_name, x_var=x_var,
        )

        # Replace X_final with appropriate variable (one-class template uses {x_var} directly)
        if self.task_type != 'one_class':
            cv_code = cv_code.replace('X_final', x_var)

        return cv_code

    def _render_metrics(self) -> str:
        """Render metrics calculation code."""
        return get_metrics_template(self.task_type, self.cv_folds)

    def _render_final_model(self) -> str:
        """Render final model training code."""
        if self.imbalance_method:
            return self._render_final_model_with_imbalance()

        resampling_methods = ['smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                              'tomek_links', 'smote_tomek', 'smote_enn', 'smogn', 'smotetomek',
                              'oversample', 'undersample']

        if self.imbalance_method and self.imbalance_method.lower() in resampling_methods:
            x_var = 'X_final'
        elif self.variable_indices is not None:
            x_var = 'X_final'
        elif self.preprocessing != 'raw':
            x_var = 'X_processed'
        else:
            x_var = 'X'

        return get_final_model_template(self.task_type, x_var=x_var, model_name=self.model_name)

    def _render_imbalance_handling(self) -> str:
        """Render imbalance handling support code."""
        if not self.imbalance_method:
            return ''

        method = self.imbalance_method.lower()
        params = self.config.get('imbalance_params', {}) or {}

        # T-19 Auto mode emission: when the user picked 'auto', the exported
        # script needs to resolve at runtime (against whichever y the user is
        # using) rather than baking a class_weight or no-correction decision
        # at code-gen time. Mirrors `imbalance.resolve_auto_imbalance` and the
        # runtime resolution in search.py / nsga2_search.py / unified_bayesian.py.
        # The mutation of IMBALANCE_METHOD to 'class_weight' or None lets the
        # downstream resampler and sample_weight gates fire correctly without
        # needing to know about 'auto'.
        # MLP exclusion warning (DeepSeek Q4): MLP can't accept class_weight as
        # a constructor kwarg, and sample_weight at fit requires sklearn≥1.7
        # which is above our pyproject floor. When Auto resolves to class_weight
        # but the model is MLP, the user should know that the model trains
        # unweighted despite the run printing "applying class_weight".
        is_mlp_model = self._resolve_model_class_name().startswith('MLP')
        mlp_warning_block = (
            '\n    if IMBALANCE_METHOD == "class_weight":\n'
            '        print("[Auto imbalance] note: MLP does not support '
            'class_weight; model will train unweighted. Use SMOTE or another '
            'resampling method for imbalanced MLP runs.")\n'
            if is_mlp_model else ''
        )

        auto_resolution_block = '''
# Auto-mode resolution: classification only. If imbalance ratio ≥ 3:1, behave
# like class_weight; otherwise no correction. Mutates IMBALANCE_METHOD so the
# downstream gates inside the CV/final-fit blocks see the resolved value.
if IMBALANCE_METHOD == "auto":
    from collections import Counter as _AutoCounter
    _auto_counts = _AutoCounter(y)
    if len(_auto_counts) >= 2:
        _auto_maj = max(_auto_counts.values())
        _auto_min = max(min(_auto_counts.values()), 1)
        _auto_ratio = _auto_maj / _auto_min
        if _auto_ratio >= 3.0:
            IMBALANCE_METHOD = "class_weight"
            print(f"[Auto imbalance] ratio {_auto_ratio:.1f}:1; applying class_weight")
        else:
            IMBALANCE_METHOD = None
            print(f"[Auto imbalance] ratio {_auto_ratio:.1f}:1 (below 3:1); no correction")
    else:
        IMBALANCE_METHOD = None
        print("[Auto imbalance] single class detected; no correction")''' + mlp_warning_block + '\n'

        return f'''
# =============================================================================
# IMBALANCE HANDLING CONFIG (used inside CV/final fit)
# =============================================================================

IMBALANCE_METHOD = "{method}"
IMBALANCE_PARAMS = {params}
{auto_resolution_block}

def _supports_sample_weight(model_obj):
    import inspect
    try:
        return 'sample_weight' in inspect.signature(model_obj.fit).parameters
    except Exception:
        return False

def _get_classification_resampler(method_name, params):
    method_map = {{
        'smote': SMOTE,
        'adasyn': ADASYN,
        'borderline_smote': BorderlineSMOTE,
        'random_undersampler': RandomUnderSampler,
        'tomek_links': TomekLinks,
        'smote_tomek': SMOTETomek,
        'smote_enn': SMOTEENN,
    }}
    if method_name not in method_map:
        return None
    cls = method_map[method_name]
    resampler_params = dict(params or {{}})
    resampler_params.setdefault('random_state', 42)
    # Combined methods don't accept k_neighbors directly
    if method_name in ('smote_tomek', 'smote_enn') and 'k_neighbors' in resampler_params:
        k = resampler_params.pop('k_neighbors')
        resampler_params['smote'] = SMOTE(k_neighbors=k)
    return cls(**resampler_params)

def _compute_regression_weights(y_vals, method_name, params):
    y_vals = np.asarray(y_vals).ravel()
    if method_name == 'binning':
        n_bins = int((params or {{}}).get('n_bins', 5))
        bin_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)
        bin_indices = np.digitize(y_vals, bins=bin_edges, right=False)
        bin_indices = np.clip(bin_indices, 1, n_bins)
        counts = np.bincount(bin_indices, minlength=n_bins + 1)
        weights = np.array([len(y_vals) / (n_bins * counts[idx]) for idx in bin_indices])
    elif method_name == 'rare_boost':
        boost_factor = float((params or {{}}).get('boost_factor', 2.0))
        median = np.median(y_vals)
        std = np.std(y_vals)
        if std == 0:
            weights = np.ones(len(y_vals))
        else:
            distances = np.abs(y_vals - median) / std
            weights = 1.0 + (boost_factor - 1.0) * (distances / distances.max())
    elif method_name == 'balanced':
        from sklearn.utils import compute_sample_weight
        n_bins = int((params or {{}}).get('n_bins', 10))
        bin_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)
        y_binned = np.digitize(y_vals, bins=bin_edges, right=False)
        y_binned = np.clip(y_binned, 1, n_bins)
        weights = compute_sample_weight('balanced', y_binned)
    else:
        return None

    weights = weights / weights.mean()
    return weights

def _regression_resample(X_vals, y_vals, method_name, params):
    # Minimal resampling to mirror GUI imbalance methods
    X_vals = np.asarray(X_vals)
    y_vals = np.asarray(y_vals).ravel()
    method_name = (method_name or '').lower()
    if method_name == 'undersample':
        n_bins = int((params or {{}}).get('n_bins', 10))
        sampling_strategy = (params or {{}}).get('sampling_strategy', 'auto')
        bin_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)
        bin_indices = np.digitize(y_vals, bins=bin_edges[:-1], right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
        if sampling_strategy == 'auto':
            target_count = int(np.median(bin_counts))
        elif sampling_strategy == 'mean':
            target_count = int(np.mean(bin_counts))
        elif isinstance(sampling_strategy, float):
            target_count = int(max(bin_counts) * sampling_strategy)
        else:
            target_count = int(np.median(bin_counts))

        rng = np.random.RandomState(int((params or {{}}).get('random_state', 42)))
        keep_idx = []
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            bin_sample_indices = np.where(bin_mask)[0]
            if len(bin_sample_indices) > target_count:
                selected = rng.choice(bin_sample_indices, size=target_count, replace=False)
                keep_idx.extend(selected)
            else:
                keep_idx.extend(bin_sample_indices)
        keep_idx = np.array(sorted(keep_idx))
        return X_vals[keep_idx], y_vals[keep_idx]

    if method_name in ('oversample', 'smogn', 'smotetomek'):
        # Simple SMOGN-style oversampling
        from sklearn.neighbors import NearestNeighbors
        n_bins = int((params or {{}}).get('n_bins', 5))
        k_neighbors = int((params or {{}}).get('k_neighbors', 5))
        rng = np.random.RandomState(int((params or {{}}).get('random_state', 42)))
        bin_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)
        bin_indices = np.digitize(y_vals, bins=bin_edges[:-1], right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        target = int(np.median(bin_counts))
        X_res, y_res = [X_vals.copy()], [y_vals.copy()]
        for bin_idx in range(n_bins):
            if bin_counts[bin_idx] >= target or bin_counts[bin_idx] < max(2, k_neighbors):
                continue
            mask = bin_indices == bin_idx
            X_bin = X_vals[mask]
            y_bin = y_vals[mask]
            nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X_bin)))
            nn.fit(X_bin)
            n_synth = target - bin_counts[bin_idx]
            for _ in range(n_synth):
                idx = rng.randint(len(X_bin))
                _, neighbors = nn.kneighbors([X_bin[idx]])
                neighbor_idx = rng.choice(neighbors[0][1:]) if len(neighbors[0]) > 1 else neighbors[0][0]
                alpha = rng.random()
                X_new = X_bin[idx] + alpha * (X_bin[neighbor_idx] - X_bin[idx])
                y_new = y_bin[idx] + alpha * (y_bin[neighbor_idx] - y_bin[idx])
                X_res.append(X_new.reshape(1, -1))
                y_res.append(np.array([y_new]))
        return np.vstack(X_res), np.concatenate(y_res)

    return X_vals, y_vals
'''

    def _render_cross_validation_with_imbalance(self) -> str:
        """Render cross-validation code with imbalance handling inside folds."""
        from spectral_predict.templates.validation import _cv_splitter_code

        # Determine X variable name
        if self.variable_indices is not None:
            x_var = 'X_final'
        elif self.preprocessing != 'raw':
            x_var = 'X_processed'
        else:
            x_var = 'X'

        cv_import, cv_constructor = _cv_splitter_code(
            self.task_type, self.cv_strategy, self.cv_folds, self.cv_n_repeats
        )

        xgb_sample_weight = self._xgb_class_weight_needs_sample_weight()
        # Conditional template emission: only thread sample_weight when XGBoost
        # is the model under imbalance_method='class_weight'. Other classifiers
        # bake balanced loss into __init__ kwargs and need a plain fit() call.
        # Indentation discipline: sample_weight_block_cv prefixes 4 spaces (lands
        # inside the for-loop body); sample_weight_block_final prefixes 0 spaces
        # (module-level). Don't normalize prefixes when refactoring.
        sample_weight_import = (
            'from sklearn.utils.class_weight import compute_sample_weight\n'
            if xgb_sample_weight else ''
        )
        sample_weight_block_cv = (
            "    fit_kwargs = {}\n"
            "    if IMBALANCE_METHOD == 'class_weight':\n"
            "        fit_kwargs['sample_weight'] = compute_sample_weight('balanced', y_train_fold)\n"
            if xgb_sample_weight else ""
        )
        cv_fit_kwargs_spread = ", **fit_kwargs" if xgb_sample_weight else ""

        if self.task_type == 'classification':
            return f'''
# =============================================================================
# CROSS-VALIDATION (with imbalance handling)
# =============================================================================

{cv_import}
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.base import clone
from collections import Counter
{sample_weight_import}
cv = {cv_constructor}

unique_classes = np.unique(y)
average_method = 'binary' if len(unique_classes) == 2 else 'macro'

# Per-sample prediction lists — under Repeated K-Fold each sample appears in
# multiple test folds; majority-vote reduction before scoring matches the
# backend (cv_utils.cross_val_predict_pooled).
fold_acc = []
fold_f1 = []
preds_per_sample = {{}}
truth_per_sample = {{}}

for train_idx, test_idx in cv.split({x_var}, y):
    X_train, X_test = {x_var}[train_idx], {x_var}[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Apply imbalance handling inside the fold
    X_train_fold, y_train_fold = X_train, y_train
    if IMBALANCE_METHOD in ['smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                            'tomek_links', 'smote_tomek', 'smote_enn']:
        resampler = _get_classification_resampler(IMBALANCE_METHOD, IMBALANCE_PARAMS)
        if resampler is not None:
            X_train_fold, y_train_fold = resampler.fit_resample(X_train, y_train)

    fold_model = clone(model)
{sample_weight_block_cv}    fold_model.fit(X_train_fold, y_train_fold{cv_fit_kwargs_spread})
    y_pred_fold = fold_model.predict(X_test)

    for local_i, sample_idx in enumerate(test_idx):
        preds_per_sample.setdefault(int(sample_idx), []).append(y_pred_fold[local_i])
        truth_per_sample[int(sample_idx)] = y_test[local_i]

    # Per-fold metrics kept for reference only — not used in headline numbers.
    fold_acc.append(accuracy_score(y_test, y_pred_fold))
    fold_f1.append(f1_score(y_test, y_pred_fold, average=average_method, zero_division=0))

# Majority-vote reduction (no-op for plain K-Fold / LOO since each sample appears once)
sorted_idx = sorted(preds_per_sample.keys())
all_y_true_arr = np.array([truth_per_sample[i] for i in sorted_idx])
all_y_pred_arr = np.array(
    [Counter(preds_per_sample[i]).most_common(1)[0][0] for i in sorted_idx]
)

# Keep y_pred_cv for compatibility with metrics template
y_pred_cv = all_y_pred_arr

accuracy = float(accuracy_score(all_y_true_arr, all_y_pred_arr))
f1 = float(f1_score(all_y_true_arr, all_y_pred_arr, average=average_method, zero_division=0))

print(f"\\nCross-validation Results ({self.cv_folds}-fold):")
print(f"  Accuracy: {{accuracy:.4f}} (pooled across folds — matches Model Development)")
print(f"  F1 Score (weighted): {{f1:.4f}} (pooled across folds)")
print("\\nConfusion Matrix:")
print(confusion_matrix(all_y_true_arr, all_y_pred_arr))
print("\\nClassification Report:")
print(classification_report(all_y_true_arr, all_y_pred_arr))
'''

        return f'''
# =============================================================================
# CROSS-VALIDATION (with imbalance handling; matches Model Development)
# =============================================================================

{cv_import}
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone

cv = {cv_constructor}

# Per-sample prediction lists — under Repeated K-Fold each sample appears in
# multiple test folds; average repeated predictions before scoring so pooled
# RMSE/R²/MAE match the backend (cv_utils.cross_val_predict_pooled).
fold_rmse = []
fold_r2 = []
fold_mae = []
preds_per_sample = {{}}
truth_per_sample = {{}}

for train_idx, test_idx in cv.split({x_var}):
    X_train, X_test = {x_var}[train_idx], {x_var}[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Apply imbalance handling inside the fold
    X_train_fold, y_train_fold = X_train, y_train
    sample_weight = None

    if IMBALANCE_METHOD in ['undersample', 'oversample', 'smogn', 'smotetomek']:
        X_train_fold, y_train_fold = _regression_resample(X_train, y_train, IMBALANCE_METHOD, IMBALANCE_PARAMS)
    elif IMBALANCE_METHOD in ['binning', 'rare_boost', 'balanced']:
        sample_weight = _compute_regression_weights(y_train, IMBALANCE_METHOD, IMBALANCE_PARAMS)

    fold_model = clone(model)
    fit_kwargs = {{}}
    if sample_weight is not None:
        if hasattr(fold_model, 'named_steps'):
            if 'model' in fold_model.named_steps:
                fit_kwargs['model__sample_weight'] = sample_weight
            elif 'lr' in fold_model.named_steps:
                fit_kwargs['lr__sample_weight'] = sample_weight
            else:
                fit_kwargs['sample_weight'] = sample_weight
        else:
            fit_kwargs['sample_weight'] = sample_weight
    fold_model.fit(X_train_fold, y_train_fold, **fit_kwargs)
    y_pred_fold = fold_model.predict(X_test).ravel()

    for local_i, sample_idx in enumerate(test_idx):
        preds_per_sample.setdefault(int(sample_idx), []).append(float(y_pred_fold[local_i]))
        truth_per_sample[int(sample_idx)] = y_test[local_i]

    # Per-fold metrics kept for reference only — not used in headline numbers.
    fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_fold)))
    fold_r2.append(r2_score(y_test, y_pred_fold))
    fold_mae.append(mean_absolute_error(y_test, y_pred_fold))

# Mean-per-sample reduction (no-op for plain K-Fold / LOO since each sample appears once)
sorted_idx = sorted(preds_per_sample.keys())
all_y_true_arr = np.array([truth_per_sample[i] for i in sorted_idx])
all_y_pred_arr = np.array([np.mean(preds_per_sample[i]) for i in sorted_idx])

rmse = float(np.sqrt(mean_squared_error(all_y_true_arr, all_y_pred_arr)))
r2 = float(r2_score(all_y_true_arr, all_y_pred_arr))
mae = float(mean_absolute_error(all_y_true_arr, all_y_pred_arr))
rpd = np.std(y) / rmse

# Keep y_pred_cv for compatibility with visualization
y_pred_cv = all_y_pred_arr
'''

    def _render_final_model_with_imbalance(self) -> str:
        """Render final model training code with imbalance handling."""
        # Determine X variable name
        if self.variable_indices is not None:
            x_var = 'X_final'
        elif self.preprocessing != 'raw':
            x_var = 'X_processed'
        else:
            x_var = 'X'

        if self.task_type == 'classification':
            xgb_sample_weight = self._xgb_class_weight_needs_sample_weight()
            sample_weight_import = (
                'from sklearn.utils.class_weight import compute_sample_weight\n'
                if xgb_sample_weight else ''
            )
            sample_weight_block_final = (
                "fit_kwargs = {}\n"
                "if IMBALANCE_METHOD == 'class_weight':\n"
                "    fit_kwargs['sample_weight'] = compute_sample_weight('balanced', y_train_full)\n"
                if xgb_sample_weight else ""
            )
            final_fit_kwargs_spread = ", **fit_kwargs" if xgb_sample_weight else ""
            return f'''
# =============================================================================
# TRAIN FINAL MODEL (with imbalance handling)
# =============================================================================
{sample_weight_import}
X_train_full, y_train_full = {x_var}, y
if IMBALANCE_METHOD in ['smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                        'tomek_links', 'smote_tomek', 'smote_enn']:
    resampler = _get_classification_resampler(IMBALANCE_METHOD, IMBALANCE_PARAMS)
    if resampler is not None:
        X_train_full, y_train_full = resampler.fit_resample(X_train_full, y_train_full)

{sample_weight_block_final}model.fit(X_train_full, y_train_full{final_fit_kwargs_spread})
print(f"\\nFinal model trained on {{X_train_full.shape[0]}} samples with {{X_train_full.shape[1]}} features")
'''

        return f'''
# =============================================================================
# TRAIN FINAL MODEL (with imbalance handling)
# =============================================================================

X_train_full, y_train_full = {x_var}, y
sample_weight = None

if IMBALANCE_METHOD in ['undersample', 'oversample', 'smogn', 'smotetomek']:
    X_train_full, y_train_full = _regression_resample(X_train_full, y_train_full, IMBALANCE_METHOD, IMBALANCE_PARAMS)
elif IMBALANCE_METHOD in ['binning', 'rare_boost', 'balanced']:
    sample_weight = _compute_regression_weights(y_train_full, IMBALANCE_METHOD, IMBALANCE_PARAMS)

fit_kwargs = {{}}
if sample_weight is not None:
    if hasattr(model, 'named_steps'):
        if 'model' in model.named_steps:
            fit_kwargs['model__sample_weight'] = sample_weight
        elif 'lr' in model.named_steps:
            fit_kwargs['lr__sample_weight'] = sample_weight
        else:
            fit_kwargs['sample_weight'] = sample_weight
    else:
        fit_kwargs['sample_weight'] = sample_weight

model.fit(X_train_full, y_train_full, **fit_kwargs)
print(f"\\nFinal model trained on {{X_train_full.shape[0]}} samples with {{X_train_full.shape[1]}} features")
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
