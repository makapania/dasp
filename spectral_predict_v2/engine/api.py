"""
Engine API - Clean interface to the analysis engine.

This module provides a high-level API that the UI layer uses to interact
with the analysis functionality. It wraps the existing src/spectral_predict/
modules.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Add parent directory to path to import existing modules
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class LoadedData:
    """Container for loaded spectral data."""
    X: np.ndarray  # Spectral data (n_samples, n_wavelengths)
    y: Optional[np.ndarray]  # Target values (None for prediction-only data)
    wavelengths: np.ndarray  # Wavelength axis
    sample_ids: list[str]  # Sample identifiers
    target_column: Optional[str]  # Name of target column
    metadata: dict  # Additional metadata
    available_targets: Optional[list[str]] = None  # List of available target columns for selection


@dataclass
class AnalysisConfig:
    """Configuration for an analysis run."""
    # Required fields (no defaults) - must come first
    preprocessing_methods: list[str]  # e.g., ["raw", "snv", "sg1"]
    model_types: list[str]  # e.g., ["pls", "ridge", "randomforest"]

    # Optional fields (with defaults)
    # Task type
    task_type: str = "regression"  # "regression" or "classification"

    # Preprocessing
    wavelength_min: Optional[float] = None
    wavelength_max: Optional[float] = None

    # Models
    use_bayesian: bool = True
    n_bayesian_trials: int = 50

    # Variable selection
    variable_selection_enabled: bool = False
    variable_selection_methods: Optional[list[str]] = None  # e.g., ["uve", "spa"]

    # Validation
    n_folds: int = 5
    random_state: int = 42

    # Scoring
    complexity_penalty: float = 0.1
    variable_penalty: float = 0.05


@dataclass
class TrainedModel:
    """Container for a trained model."""
    model: Any  # The sklearn-compatible model object
    name: str  # Model name (e.g., "PLS-5")
    preprocessing: str  # Applied preprocessing
    variable_mask: Optional[np.ndarray]  # Boolean mask for selected variables
    config: dict  # Model configuration
    metrics: dict  # Performance metrics


class EngineAPI:
    """
    High-level API for the spectral analysis engine.

    This class wraps the existing analysis modules and provides a clean
    interface for the UI to use.
    """

    def __init__(self):
        self._io_module = None
        self._search_module = None
        self._preprocess_module = None
        self._models_module = None
        self._scoring_module = None

    def _lazy_import_io(self):
        """Lazy import of I/O module."""
        if self._io_module is None:
            try:
                from src.spectral_predict import io as io_module
                self._io_module = io_module
            except ImportError:
                raise ImportError(
                    "Could not import spectral_predict.io. "
                    "Make sure src/spectral_predict is in the Python path."
                )
        return self._io_module

    def _lazy_import_search(self):
        """Lazy import of search module."""
        if self._search_module is None:
            try:
                from src.spectral_predict import search as search_module
                self._search_module = search_module
            except ImportError:
                raise ImportError("Could not import spectral_predict.search.")
        return self._search_module

    def _lazy_import_preprocess(self):
        """Lazy import of preprocessing module."""
        if self._preprocess_module is None:
            try:
                from src.spectral_predict import preprocess as preprocess_module
                self._preprocess_module = preprocess_module
            except ImportError:
                raise ImportError("Could not import spectral_predict.preprocess.")
        return self._preprocess_module

    # --- Data Loading ---

    def load_data(
        self,
        file_path: str,
        target_column: Optional[str] = None,
        for_prediction: bool = False,
    ) -> LoadedData:
        """
        Load spectral data from a file.

        Supports CSV, Excel, ASD, SPC, JCAMP-DX formats.
        Auto-detects format and handles both spectra-only and combined (spectra+target) files.

        Args:
            file_path: Path to the data file
            target_column: Name of the target column (auto-detected if None)
            for_prediction: If True, don't require target column

        Returns:
            LoadedData object containing X, y, wavelengths, etc.
        """
        io = self._lazy_import_io()
        path = Path(file_path)

        # Detect format
        file_format = io.detect_format(file_path)

        # Try to load as combined file (spectra + targets) first for CSV/Excel
        if file_format in ['csv', 'excel']:
            try:
                if file_format == 'csv':
                    X_df, y_series, metadata_df, meta = io.read_combined_csv(
                        file_path,
                        y_col=target_column,
                        drop_na_y=not for_prediction
                    )
                else:
                    X_df, y_series, metadata_df, meta = io.read_combined_excel(
                        file_path,
                        y_col=target_column,
                        drop_na_y=not for_prediction
                    )

                # Extract data
                X = X_df.values.astype(np.float64)
                wavelengths = np.array([float(c) for c in X_df.columns])
                sample_ids = list(X_df.index.astype(str))

                # Get target if available
                y = y_series.values if y_series is not None else None
                detected_target = meta.get('y_col')

                # Find available target columns (non-wavelength, non-ID columns)
                available_targets = []
                if metadata_df is not None:
                    available_targets = list(metadata_df.columns)
                if detected_target and detected_target not in available_targets:
                    available_targets.insert(0, detected_target)

                return LoadedData(
                    X=X,
                    y=y,
                    wavelengths=wavelengths,
                    sample_ids=sample_ids,
                    target_column=detected_target,
                    metadata={
                        "file_path": str(file_path),
                        "file_format": file_format,
                        "n_samples": X.shape[0],
                        "n_wavelengths": X.shape[1],
                        "wavelength_range": (float(wavelengths.min()), float(wavelengths.max())),
                        **meta
                    },
                    available_targets=available_targets
                )

            except Exception as e:
                # Fall back to spectra-only loading
                print(f"Combined format loading failed: {e}, trying spectra-only")

        # Load as spectra-only file
        df, meta = io.read_spectra(file_path, format=file_format)

        X = df.values.astype(np.float64)
        wavelengths = np.array([float(c) for c in df.columns])
        sample_ids = list(df.index.astype(str))

        return LoadedData(
            X=X,
            y=None,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            target_column=None,
            metadata={
                "file_path": str(file_path),
                "file_format": file_format,
                "n_samples": X.shape[0],
                "n_wavelengths": X.shape[1],
                "wavelength_range": (float(wavelengths.min()), float(wavelengths.max())),
                **meta
            },
            available_targets=[]
        )

    def load_data_with_config(
        self,
        file_path: str,
        id_column: Optional[str] = None,
        target_column: Optional[str] = None,
        metadata_columns: Optional[list[str]] = None,
    ) -> LoadedData:
        """
        Load spectral data with explicit column configuration.

        This method gives full control over which columns are used for what purpose.

        Args:
            file_path: Path to the data file
            id_column: Column to use as sample ID (None = use row index)
            target_column: Column to use as target (y) variable
            metadata_columns: Columns to exclude from spectra (metadata)

        Returns:
            LoadedData object containing X, y, wavelengths, etc.
        """
        path = Path(file_path)
        metadata_columns = metadata_columns or []

        # Read the file
        if path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        # Set index if id_column specified
        if id_column and id_column in df.columns:
            df = df.set_index(id_column)
            sample_ids = list(df.index.astype(str))
        else:
            sample_ids = [f"sample_{i}" for i in range(len(df))]

        # Extract target column if specified
        y = None
        available_targets = []
        is_classification = False
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            # Check if it's categorical/string (classification) or numeric (regression)
            if pd.api.types.is_numeric_dtype(target_series):
                y = target_series.values.astype(np.float64)
            else:
                # Categorical - encode as integers for classification
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(target_series.values.astype(str))
                is_classification = True
            df = df.drop(columns=[target_column])
            available_targets.append(target_column)

        # Remove metadata columns
        cols_to_remove = [c for c in metadata_columns if c in df.columns]
        if cols_to_remove:
            # Store other numeric columns as potential targets
            for col in cols_to_remove:
                if pd.api.types.is_numeric_dtype(df[col]):
                    available_targets.append(col)
            df = df.drop(columns=cols_to_remove)

        # Remaining columns should be wavelengths - identify them
        wavelength_cols = []
        non_wavelength_cols = []

        for col in df.columns:
            try:
                wl = float(str(col))
                wavelength_cols.append((col, wl))
            except ValueError:
                # Non-numeric column name - check if it's numeric data
                if pd.api.types.is_numeric_dtype(df[col]):
                    available_targets.append(col)
                non_wavelength_cols.append(col)

        # Remove non-wavelength columns
        if non_wavelength_cols:
            df = df.drop(columns=non_wavelength_cols)

        if not wavelength_cols:
            raise ValueError("No wavelength columns found. Wavelength columns should have numeric names.")

        # Sort by wavelength and extract data
        wavelength_cols.sort(key=lambda x: x[1])
        sorted_cols = [c[0] for c in wavelength_cols]
        wavelengths = np.array([c[1] for c in wavelength_cols])

        df = df[sorted_cols]
        X = df.values.astype(np.float64)

        return LoadedData(
            X=X,
            y=y,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            target_column=target_column,
            metadata={
                "file_path": str(file_path),
                "n_samples": X.shape[0],
                "n_wavelengths": X.shape[1],
                "wavelength_range": (float(wavelengths.min()), float(wavelengths.max())),
                "is_classification": is_classification,
            },
            available_targets=available_targets
        )

    def load_reference_data(
        self,
        file_path: str,
        id_column: str,
    ) -> pd.DataFrame:
        """
        Load a separate reference file with target values.

        Args:
            file_path: Path to reference file (CSV or Excel)
            id_column: Column name containing sample IDs for matching

        Returns:
            DataFrame indexed by sample ID
        """
        io = self._lazy_import_io()
        return io.read_reference_csv(file_path, id_column)

    def detect_file_format(self, file_path: str) -> str:
        """Detect the format of a spectral data file."""
        io = self._lazy_import_io()
        return io.detect_format(file_path)

    # --- Preprocessing ---

    def apply_preprocessing(
        self,
        X: np.ndarray,
        method: str,
        wavelengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply preprocessing to spectral data.

        Args:
            X: Spectral data (n_samples, n_wavelengths)
            method: Preprocessing method ("snv", "sg1", "sg2", "msc", "raw")
            wavelengths: Optional wavelength axis (needed for some methods)

        Returns:
            Preprocessed spectral data
        """
        preprocess = self._lazy_import_preprocess()

        method = method.lower()

        if method == "raw":
            return X.copy()
        elif method == "snv":
            return preprocess.SNV().fit_transform(X)
        elif method == "sg1":
            return preprocess.SavitzkyGolayDerivative(
                window_length=15, polyorder=2, deriv=1
            ).fit_transform(X)
        elif method == "sg2":
            return preprocess.SavitzkyGolayDerivative(
                window_length=15, polyorder=2, deriv=2
            ).fit_transform(X)
        elif method == "msc":
            # MSC may be in interference module
            try:
                from src.spectral_predict.interference import MSC
                return MSC().fit_transform(X)
            except ImportError:
                return X  # Fallback
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def get_available_preprocessing(self) -> list[tuple[str, str]]:
        """Get list of available preprocessing methods."""
        return [
            ("raw", "Raw (No preprocessing)"),
            ("snv", "SNV (Standard Normal Variate)"),
            ("sg1", "SG1 (1st Derivative, Savitzky-Golay)"),
            ("sg2", "SG2 (2nd Derivative, Savitzky-Golay)"),
            ("msc", "MSC (Multiplicative Scatter Correction)"),
        ]

    # --- Analysis ---

    def run_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: AnalysisConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> pd.DataFrame:
        """
        Run automated analysis to find the best model.

        Args:
            X: Spectral data (n_samples, n_wavelengths)
            y: Target values
            config: Analysis configuration
            progress_callback: Optional callback for progress updates (progress, stage)

        Returns:
            DataFrame with results sorted by composite score
        """
        search = self._lazy_import_search()

        # Convert numpy arrays to pandas (expected by search module)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        # Determine task type
        task_type = "regression"
        if hasattr(config, 'task_type'):
            task_type = config.task_type
        elif len(np.unique(y)) <= 20:  # Heuristic for classification
            # Check if values look like classes
            if np.all(y == y.astype(int)) and np.min(y) >= 0:
                task_type = "classification"

        # Build preprocessing configs for search module
        # The search module expects: [{'name': 'snv', 'deriv': 0}, ...]
        preprocessing_configs = []
        for method in config.preprocessing_methods:
            method = method.lower()
            if method == "raw":
                preprocessing_configs.append({"name": "none", "deriv": 0})
            elif method == "snv":
                preprocessing_configs.append({"name": "snv", "deriv": 0})
            elif method == "sg1":
                preprocessing_configs.append({"name": "snv", "deriv": 1})
            elif method == "sg2":
                preprocessing_configs.append({"name": "snv", "deriv": 2})
            elif method == "msc":
                preprocessing_configs.append({"name": "msc", "deriv": 0})
            else:
                preprocessing_configs.append({"name": method, "deriv": 0})

        if not preprocessing_configs:
            preprocessing_configs = [{"name": "snv", "deriv": 0}]

        # Map model names to search module format
        model_name_map = {
            "pls": "PLS",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "elasticnet": "ElasticNet",
            "randomforest": "RandomForest",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "catboost": "CatBoost",
            "svr": "SVR",
            "svm": "SVM",
            "mlp": "MLP",
            "plsda": "PLS-DA",
        }
        models_to_test = [model_name_map.get(m.lower(), m) for m in config.model_types]

        # Run either Bayesian or grid search
        try:
            if config.use_bayesian:
                results_df, label_encoder = search.run_bayesian_search(
                    X_df, y_series,
                    task_type=task_type,
                    models_to_test=models_to_test,
                    preprocessing_methods=preprocessing_configs,
                    n_trials=config.n_bayesian_trials,
                    folds=config.n_folds,
                    random_state=config.random_state,
                    progress_callback=progress_callback,
                )
            else:
                results_df, label_encoder = search.run_search(
                    X_df, y_series,
                    task_type=task_type,
                    folds=config.n_folds,
                    progress_callback=progress_callback,
                )

            return results_df

        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Analysis failed: {str(e)}") from e

    def get_available_models(self, task_type: str = "regression") -> list[tuple[str, str]]:
        """Get list of available models for a task type."""
        if task_type == "regression":
            return [
                ("pls", "PLS (Partial Least Squares)"),
                ("ridge", "Ridge Regression"),
                ("lasso", "Lasso Regression"),
                ("elasticnet", "ElasticNet"),
                ("randomforest", "Random Forest"),
                ("xgboost", "XGBoost"),
                ("lightgbm", "LightGBM"),
                ("catboost", "CatBoost"),
                ("svr", "SVR (Support Vector Regression)"),
                ("mlp", "MLP (Neural Network)"),
            ]
        else:
            return [
                ("plsda", "PLS-DA"),
                ("randomforest", "Random Forest"),
                ("xgboost", "XGBoost"),
                ("lightgbm", "LightGBM"),
                ("svm", "SVM"),
                ("mlp", "MLP (Neural Network)"),
            ]

    # --- Model Training ---

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        preprocessing: str = "raw",
        n_components: int = 5,
        variable_mask: Optional[np.ndarray] = None,
        **hyperparams
    ) -> TrainedModel:
        """
        Train a specific model configuration.

        Args:
            X: Spectral data
            y: Target values
            model_type: Type of model (e.g., "pls", "ridge")
            preprocessing: Preprocessing to apply
            n_components: Number of components (for PLS, etc.)
            variable_mask: Optional boolean mask for variable selection
            **hyperparams: Additional hyperparameters

        Returns:
            TrainedModel object
        """
        from src.spectral_predict.models import get_model

        # Apply preprocessing
        X_proc = self.apply_preprocessing(X, preprocessing)

        # Apply variable selection if provided
        if variable_mask is not None:
            X_proc = X_proc[:, variable_mask]

        # Get and configure model
        model = get_model(model_type, n_components=n_components, **hyperparams)

        # Fit model
        model.fit(X_proc, y)

        # Calculate metrics
        y_pred = model.predict(X_proc)
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        bias = np.mean(y_pred - y)

        return TrainedModel(
            model=model,
            name=f"{model_type.upper()}-{n_components}",
            preprocessing=preprocessing,
            variable_mask=variable_mask,
            config={"model_type": model_type, "n_components": n_components, **hyperparams},
            metrics={"rmse": rmse, "r2": r2, "bias": bias}
        )

    # --- Prediction ---

    def predict(
        self,
        X_new: np.ndarray,
        model: TrainedModel,
        return_uncertainty: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Make predictions on new data.

        Args:
            X_new: New spectral data
            model: Trained model
            return_uncertainty: Whether to estimate prediction uncertainty

        Returns:
            Tuple of (predictions, uncertainty, applicability_domain_flags)
        """
        # Apply same preprocessing
        X_proc = self.apply_preprocessing(X_new, model.preprocessing)

        # Apply variable mask
        if model.variable_mask is not None:
            X_proc = X_proc[:, model.variable_mask]

        # Predict
        predictions = model.model.predict(X_proc)

        # Estimate uncertainty (placeholder - would use proper methods)
        uncertainty = None
        ad_flags = None

        if return_uncertainty:
            # Simple uncertainty estimate based on distance from training data
            # In reality, this would use proper methods like leverage, etc.
            uncertainty = np.ones(len(predictions)) * model.metrics.get("rmse", 0.5)

            # Applicability domain check (placeholder)
            ad_flags = np.ones(len(predictions), dtype=bool)

        return predictions, uncertainty, ad_flags

    # --- Model Persistence ---

    def save_model(
        self,
        model: TrainedModel,
        file_path: str,
        wavelengths: Optional[np.ndarray] = None,
        training_stats: Optional[dict] = None
    ):
        """
        Save a trained model to disk in .dasp format.

        The .dasp format includes:
        - The trained model object
        - Preprocessing configuration
        - Variable selection mask
        - Wavelength axis for applicability domain checking
        - Training data statistics for uncertainty estimation
        - Full configuration and metrics

        Args:
            model: TrainedModel object to save
            file_path: Destination file path
            wavelengths: Optional wavelength axis from training data
            training_stats: Optional dict with training data statistics (mean, std, etc.)
        """
        import joblib
        from datetime import datetime

        # Extract wavelengths from model config if not provided
        if wavelengths is None:
            wavelengths = model.config.get('wavelengths')

        # Build training stats if not provided
        if training_stats is None and 'y_true' in model.config:
            y = model.config['y_true']
            training_stats = {
                'y_mean': float(np.mean(y)),
                'y_std': float(np.std(y)),
                'y_min': float(np.min(y)),
                'y_max': float(np.max(y)),
                'n_samples': len(y),
            }

        # Clean config for serialization (remove large arrays)
        clean_config = {k: v for k, v in model.config.items()
                       if k not in ('y_pred_cv', 'y_true', 'wavelengths', 'loadings', 'coefficients')}

        save_data = {
            # Core model
            "model": model.model,
            "name": model.name,
            "preprocessing": model.preprocessing,
            "variable_mask": model.variable_mask,

            # Configuration
            "config": clean_config,
            "metrics": model.metrics,

            # Metadata for applicability domain
            "wavelengths": wavelengths,
            "training_stats": training_stats,

            # File metadata
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "format": "dasp",
        }

        joblib.dump(save_data, file_path)

    def load_model(self, file_path: str) -> TrainedModel:
        """
        Load a trained model from disk.

        Args:
            file_path: Path to .dasp file

        Returns:
            TrainedModel object with all configuration and metadata
        """
        import joblib

        data = joblib.load(file_path)

        # Reconstruct config with wavelengths and training stats
        config = data.get("config", {})
        if data.get("wavelengths") is not None:
            config["wavelengths"] = data["wavelengths"]
        if data.get("training_stats") is not None:
            config["training_stats"] = data["training_stats"]
        config["created"] = data.get("created")
        config["version"] = data.get("version", "1.0")

        return TrainedModel(
            model=data["model"],
            name=data["name"],
            preprocessing=data["preprocessing"],
            variable_mask=data.get("variable_mask"),
            config=config,
            metrics=data["metrics"],
        )

    def get_model_info(self, file_path: str) -> dict:
        """
        Get model information without fully loading the model.

        Useful for displaying model metadata in the UI before making predictions.

        Args:
            file_path: Path to .dasp file

        Returns:
            Dict with model metadata (name, metrics, config, etc.)
        """
        import joblib

        data = joblib.load(file_path)

        # Return metadata without the model object
        return {
            "name": data.get("name", "Unknown"),
            "preprocessing": data.get("preprocessing", "Unknown"),
            "metrics": data.get("metrics", {}),
            "config": data.get("config", {}),
            "wavelengths": data.get("wavelengths"),
            "training_stats": data.get("training_stats"),
            "created": data.get("created"),
            "version": data.get("version", "1.0"),
        }
