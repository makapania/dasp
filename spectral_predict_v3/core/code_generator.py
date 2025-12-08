"""
Code generator for reproducible analysis scripts.

Generates standalone Python scripts and Jupyter notebooks from
model bundles and analysis configurations.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
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


class CodeGenerator:
    """
    Generate standalone Python scripts from model bundles.

    This class takes a trained model bundle and generates a complete,
    reproducible Python script that can be shared with reviewers.

    Parameters
    ----------
    model_bundle : dict
        Model bundle containing:
        - model: trained model
        - model_name: str
        - preprocessing: str
        - wavelengths: np.ndarray
        - target_name: str
        - task_type: str ('regression' or 'classification')
        - metrics: dict
        - params: dict
        - variable_indices: np.ndarray or None
    options : ExportOptions, optional
        Export configuration options

    Examples
    --------
    >>> from spectral_predict_v3.core.code_generator import CodeGenerator, ExportOptions
    >>> options = ExportOptions(include_visualization=True)
    >>> generator = CodeGenerator(model_bundle, options)
    >>> script = generator.generate_script()
    >>> with open('analysis.py', 'w') as f:
    ...     f.write(script)
    """

    def __init__(self, model_bundle: Dict[str, Any], options: ExportOptions = None):
        self.bundle = model_bundle
        self.options = options or ExportOptions()

        # Extract commonly used values
        self.model_name = model_bundle.get('model_name', 'Unknown')
        self.preprocessing = model_bundle.get('preprocessing', 'raw')
        self.task_type = model_bundle.get('task_type', 'regression')
        self.target_name = model_bundle.get('target_name', 'target')
        self.params = model_bundle.get('params', {})
        self.metrics = model_bundle.get('metrics', {})
        self.variable_indices = model_bundle.get('variable_indices', None)
        self.wavelengths = model_bundle.get('wavelengths', None)

        # Update options with target column from bundle
        if self.target_name:
            self.options.target_column = self.target_name

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

        # 4. Data loading
        if self.options.include_data_loading:
            sections.append(self._render_data_loading())

        # 5. Preprocessing application
        if self.options.include_preprocessing:
            sections.append(self._render_preprocessing_application())

        # 6. Variable selection application
        if self.options.include_variable_selection and self.variable_indices is not None:
            sections.append(self._render_variable_selection_application())

        # 7. Model instantiation
        sections.append(self._render_model())

        # 8. Cross-validation
        if self.options.include_cross_validation:
            sections.append(self._render_cross_validation())
            sections.append(self._render_metrics())

        # 9. Final model training
        sections.append(FINAL_MODEL_TEMPLATE)

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

        # Title cell
        cells.append(self._make_markdown_cell(
            f"# Spectral Analysis: {self.model_name}\n\n"
            f"Generated by Spectral Predict v3 on {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"**Model**: {self.model_name}  \n"
            f"**Preprocessing**: {self.preprocessing}  \n"
            f"**Task**: {self.task_type}  \n"
        ))

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

        # Data loading
        cells.append(self._make_markdown_cell(
            "## 4. Data Loading\n\n"
            "Update the file path and column names for your data."
        ))
        cells.append(self._make_code_cell(self._render_data_loading()))

        # Apply preprocessing
        cells.append(self._make_markdown_cell("## 5. Apply Preprocessing"))
        cells.append(self._make_code_cell(self._render_preprocessing_application()))

        # Apply variable selection
        if self.variable_indices is not None:
            cells.append(self._make_markdown_cell("## 6. Apply Variable Selection"))
            cells.append(self._make_code_cell(self._render_variable_selection_application()))

        # Model and cross-validation
        cells.append(self._make_markdown_cell("## 7. Model Training and Evaluation"))
        model_cv_code = (
            self._render_model() + '\n' +
            self._render_cross_validation() + '\n' +
            self._render_metrics()
        )
        cells.append(self._make_code_cell(model_cv_code))

        # Final model
        cells.append(self._make_markdown_cell("## 8. Train Final Model"))
        cells.append(self._make_code_cell(FINAL_MODEL_TEMPLATE))

        # Visualization
        if self.options.include_visualization:
            cells.append(self._make_markdown_cell("## 9. Visualization"))
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

        return notebook

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
    # Private methods for rendering sections
    # =========================================================================

    def _render_header(self) -> str:
        """Render the script header with imports."""
        # Collect all necessary imports
        imports = [
            'import numpy as np',
            'import pandas as pd',
        ]

        # Model-specific import
        model_import = get_model_imports(self.model_name)
        if model_import:
            imports.append(model_import)

        # Scipy for preprocessing
        if 'deriv' in self.preprocessing:
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

        # Build model details string
        model_details = self._format_model_details()

        # Variable selection info
        var_sel_info = ''
        if self.variable_indices is not None:
            var_sel_method = self.bundle.get('variable_selection_method', 'custom')
            var_sel_info = f'Variable Selection: {var_sel_method} ({len(self.variable_indices)} variables)'

        # Extra packages
        extra_packages = ''
        if self.model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
            pkg = self.model_name.lower().replace('boost', '')
            extra_packages = f', {pkg}'
        if self.options.include_visualization:
            extra_packages += ', matplotlib'

        return HEADER_TEMPLATE.format(
            date=datetime.now().strftime('%Y-%m-%d'),
            model_name=self.model_name,
            model_details=model_details,
            preprocessing=self.preprocessing,
            variable_selection_info=var_sel_info,
            cv_folds=self.bundle.get('cv_folds', 5),
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

        if 'deriv' in self.preprocessing:
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
        template_code, _ = get_preprocessing_template(self.preprocessing)
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
        method = self.bundle.get('variable_selection_method', '')
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
        if self.preprocessing == 'raw':
            return '\n# No preprocessing - using raw spectra\nX_processed = X.copy()\n'

        _, application_code = get_preprocessing_template(self.preprocessing)
        return (
            "\n# =============================================================================\n"
            "# APPLY PREPROCESSING\n"
            "# =============================================================================\n\n"
            + application_code + '\n'
            + f'\nprint(f"Preprocessed data shape: {{X_processed.shape}}")\n'
        )

    def _render_variable_selection_application(self) -> str:
        """Render variable selection application code."""
        if self.variable_indices is None:
            return '\n# No variable selection - using all variables\nX_final = X_processed\n'

        # Convert indices to list for cleaner output
        indices = self.variable_indices
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        # Format indices - truncate if too many
        if len(indices) <= 20:
            indices_str = str(indices)
        else:
            indices_str = f"np.array({indices[:5]} + ... + {indices[-5:]})  # {len(indices)} variables total"
            # For actual use, embed all indices
            indices_str = f"np.array({indices})"

        return (
            "\n# =============================================================================\n"
            "# APPLY VARIABLE SELECTION\n"
            "# =============================================================================\n\n"
            f"# Selected variable indices (from {self.bundle.get('variable_selection_method', 'selection')})\n"
            f"selected_indices = {indices_str}\n\n"
            "# Apply variable selection\n"
            "X_final = X_processed[:, selected_indices]\n"
            f'print(f"After variable selection: {{X_final.shape[1]}} variables selected")\n'
        )

    def _render_model(self) -> str:
        """Render model instantiation code."""
        model_code = get_model_template(
            self.model_name,
            self.params,
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
        cv_folds = self.bundle.get('cv_folds', 5)

        # Determine final variable name
        if self.variable_indices is not None:
            x_var = 'X_final'
        elif self.preprocessing != 'raw':
            x_var = 'X_processed'
        else:
            x_var = 'X'

        cv_code = get_cross_validation_template(self.task_type, cv_folds)

        # Replace X_final with appropriate variable
        cv_code = cv_code.replace('X_final', x_var)

        return cv_code

    def _render_metrics(self) -> str:
        """Render metrics calculation code."""
        cv_folds = self.bundle.get('cv_folds', 5)
        return get_metrics_template(self.task_type, cv_folds)

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
        return {
            'cell_type': 'markdown',
            'metadata': {},
            'source': source.split('\n')
        }

    def _make_code_cell(self, source: str) -> dict:
        """Create a code notebook cell."""
        return {
            'cell_type': 'code',
            'metadata': {},
            'source': source.split('\n'),
            'outputs': [],
            'execution_count': None
        }


def generate_script_from_bundle(model_bundle: Dict[str, Any],
                                 include_visualization: bool = False,
                                 data_path: str = 'your_data.csv') -> str:
    """
    Convenience function to generate a script from a model bundle.

    Parameters
    ----------
    model_bundle : dict
        Model bundle from model_io.save_model()
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
        target_column=model_bundle.get('target_name', 'target')
    )
    generator = CodeGenerator(model_bundle, options)
    return generator.generate_script()


def generate_notebook_from_bundle(model_bundle: Dict[str, Any],
                                   include_visualization: bool = True) -> dict:
    """
    Convenience function to generate a notebook from a model bundle.

    Parameters
    ----------
    model_bundle : dict
        Model bundle from model_io.save_model()
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
        target_column=model_bundle.get('target_name', 'target')
    )
    generator = CodeGenerator(model_bundle, options)
    return generator.generate_notebook()
