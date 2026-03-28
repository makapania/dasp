"""
GUI Test Harness for Spectral Predict V1.

Provides methods to interact with and validate the GUI without modifying
any existing application code.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:
    from spectral_predict_gui_optimized import SpectralPredictApp


class GUITestHarness:
    """
    Test harness wrapping SpectralPredictApp for automated testing.

    Provides methods to:
    - Wait for GUI to process events
    - Interact with widgets programmatically
    - Validate application state
    - Wait for async operations (analysis)
    """

    def __init__(self, app: SpectralPredictApp, visible: bool = False):
        """
        Initialize test harness.

        Args:
            app: The SpectralPredictApp instance to test
            visible: Whether the GUI window is visible
        """
        self.app = app
        self.root = app.root
        self.visible = visible

    # ==================== Wait Utilities ====================

    def wait_for_idle(self, timeout: float = 2.0) -> None:
        """
        Wait for GUI to finish processing events.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start = time.time()
        while time.time() - start < timeout:
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.05)

    def wait_for_condition(
        self,
        condition: Callable[[], bool],
        timeout: float = 30.0,
        poll_interval: float = 0.1,
        description: str = "condition"
    ) -> bool:
        """
        Wait for a condition to become True.

        Args:
            condition: Callable returning True when condition is met
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check condition
            description: Description for error message

        Returns:
            True if condition was met

        Raises:
            TimeoutError: If condition not met within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            self.root.update_idletasks()
            self.root.update()

            try:
                if condition():
                    return True
            except Exception:
                pass

            time.sleep(poll_interval)

        raise TimeoutError(f"Condition '{description}' not met within {timeout}s")

    def wait_for_analysis_complete(self, timeout: float = 300.0) -> bool:
        """
        Wait for analysis thread to complete and results to populate.

        Args:
            timeout: Maximum time to wait (default 5 minutes)

        Returns:
            True if analysis completed successfully
        """
        def check_complete():
            # Check if results exist
            if self.app.results_df is None:
                return False
            if len(self.app.results_df) == 0:
                return False

            # Check if analysis thread is done
            thread = getattr(self.app, 'analysis_thread', None)
            if thread is not None and thread.is_alive():
                return False

            return True

        return self.wait_for_condition(
            check_complete,
            timeout=timeout,
            poll_interval=0.5,
            description="analysis complete"
        )

    # ==================== Navigation ====================

    def select_tab(self, index: int) -> None:
        """
        Select a tab in the main notebook by index.

        Args:
            index: Tab index (0-based)
        """
        if hasattr(self.app, 'notebook'):
            tabs = self.app.notebook.tabs()
            if 0 <= index < len(tabs):
                self.app.notebook.select(index)
                self.wait_for_idle(0.3)

    def get_current_tab_index(self) -> int:
        """Get the currently selected tab index."""
        if hasattr(self.app, 'notebook'):
            return self.app.notebook.index(self.app.notebook.select())
        return -1

    # ==================== Widget Interaction ====================

    def set_var(self, var_name: str, value: Any) -> None:
        """
        Set a tkinter variable on the app.

        Args:
            var_name: Name of the variable attribute on app
            value: Value to set
        """
        var = getattr(self.app, var_name, None)
        if var is not None and hasattr(var, 'set'):
            var.set(value)
            self.root.update()

    def get_var(self, var_name: str) -> Any:
        """
        Get the value of a tkinter variable.

        Args:
            var_name: Name of the variable attribute on app

        Returns:
            The variable's value, or None if not found
        """
        var = getattr(self.app, var_name, None)
        if var is not None and hasattr(var, 'get'):
            return var.get()
        return None

    def invoke_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Invoke a method on the app.

        Args:
            method_name: Name of the method to invoke
            *args, **kwargs: Arguments to pass to method

        Returns:
            Return value of the method
        """
        method = getattr(self.app, method_name, None)
        if method is not None and callable(method):
            result = method(*args, **kwargs)
            self.wait_for_idle(0.2)
            return result
        raise AttributeError(f"Method '{method_name}' not found on app")

    def click_button(self, button: tk.Button | ttk.Button) -> None:
        """
        Simulate clicking a button.

        Args:
            button: The button widget to click
        """
        if hasattr(button, 'invoke'):
            button.invoke()
            self.wait_for_idle(0.2)

    def find_button_by_text(self, text: str, parent: tk.Widget | None = None) -> tk.Button | ttk.Button | None:
        """
        Find a button by its text content.

        Args:
            text: Text to search for (substring match)
            parent: Parent widget to search in (default: root)

        Returns:
            The button widget, or None if not found
        """
        parent = parent or self.root
        return self._find_widget_by_text(parent, (tk.Button, ttk.Button), text)

    def _find_widget_by_text(
        self,
        parent: tk.Widget,
        widget_types: tuple,
        text: str
    ) -> tk.Widget | None:
        """Recursively find widget by text."""
        for child in parent.winfo_children():
            if isinstance(child, widget_types):
                try:
                    widget_text = child.cget('text')
                    if text in str(widget_text):
                        return child
                except tk.TclError:
                    pass

            # Recurse into children
            found = self._find_widget_by_text(child, widget_types, text)
            if found:
                return found

        return None

    def find_widgets_by_type(
        self,
        widget_type: type,
        parent: tk.Widget | None = None
    ) -> list[tk.Widget]:
        """
        Find all widgets of a specific type.

        Args:
            widget_type: Type of widget to find
            parent: Parent widget to search in

        Returns:
            List of matching widgets
        """
        parent = parent or self.root
        results = []
        self._collect_widgets_by_type(parent, widget_type, results)
        return results

    def _collect_widgets_by_type(
        self,
        parent: tk.Widget,
        widget_type: type,
        results: list
    ) -> None:
        """Recursively collect widgets by type."""
        for child in parent.winfo_children():
            if isinstance(child, widget_type):
                results.append(child)
            self._collect_widgets_by_type(child, widget_type, results)

    # ==================== Treeview Interaction ====================

    def get_treeview_items(self, tree: ttk.Treeview) -> list[dict]:
        """
        Get all items from a Treeview as list of dicts.

        Args:
            tree: The Treeview widget

        Returns:
            List of dicts with 'id', 'text', 'values' for each row
        """
        items = []
        for item_id in tree.get_children():
            items.append({
                'id': item_id,
                'text': tree.item(item_id, 'text'),
                'values': tree.item(item_id, 'values')
            })
        return items

    def select_treeview_row(self, tree: ttk.Treeview, row_index: int) -> None:
        """
        Select a row in a Treeview by index.

        Args:
            tree: The Treeview widget
            row_index: Row index (0-based)
        """
        children = tree.get_children()
        if 0 <= row_index < len(children):
            item_id = children[row_index]
            tree.selection_set(item_id)
            tree.event_generate('<<TreeviewSelect>>')
            self.wait_for_idle(0.2)

    def double_click_treeview_row(self, tree: ttk.Treeview, row_index: int) -> None:
        """
        Simulate double-click on a Treeview row.

        Args:
            tree: The Treeview widget
            row_index: Row index (0-based)
        """
        children = tree.get_children()
        if 0 <= row_index < len(children):
            item_id = children[row_index]
            tree.selection_set(item_id)
            tree.focus(item_id)
            tree.event_generate('<Double-1>')
            self.wait_for_idle(0.3)

    # ==================== Analysis Configuration ====================

    def configure_quick_analysis(
        self,
        models: list[str] | None = None,
        preprocessing: list[str] | None = None,
        cv_folds: int = 3
    ) -> None:
        """
        Configure a quick analysis run with minimal settings.

        Args:
            models: List of model names to enable (default: ['PLS'])
            preprocessing: List of preprocessing methods (default: ['Raw', 'SNV'])
            cv_folds: Number of CV folds (default: 3)
        """
        models = models or ['PLS']
        preprocessing = preprocessing or ['Raw', 'SNV']

        # Set CV folds
        self.set_var('folds', cv_folds)

        # Disable all models first
        model_vars = [
            'use_pls', 'use_plsda', 'use_ridge', 'use_lasso',
            'use_elasticnet', 'use_randomforest', 'use_mlp',
            'use_svr', 'use_xgboost', 'use_lightgbm', 'use_catboost'
        ]
        for var in model_vars:
            if hasattr(self.app, var):
                getattr(self.app, var).set(False)

        # Enable selected models
        model_map = {
            'PLS': 'use_pls',
            'PLS-DA': 'use_plsda',
            'Ridge': 'use_ridge',
            'Lasso': 'use_lasso',
            'ElasticNet': 'use_elasticnet',
            'RandomForest': 'use_randomforest',
            'MLP': 'use_mlp',
            'SVR': 'use_svr',
            'XGBoost': 'use_xgboost',
            'LightGBM': 'use_lightgbm',
            'CatBoost': 'use_catboost',
        }
        for model in models:
            var_name = model_map.get(model)
            if var_name and hasattr(self.app, var_name):
                getattr(self.app, var_name).set(True)

        # Configure preprocessing
        preprocess_vars = [
            'use_raw', 'use_snv', 'use_sg1', 'use_sg2'
        ]
        for var in preprocess_vars:
            if hasattr(self.app, var):
                getattr(self.app, var).set(False)

        preprocess_map = {
            'Raw': 'use_raw',
            'SNV': 'use_snv',
            'SG1': 'use_sg1',
            'SG2': 'use_sg2',
        }
        for pp in preprocessing:
            var_name = preprocess_map.get(pp)
            if var_name and hasattr(self.app, var_name):
                getattr(self.app, var_name).set(True)

        self.wait_for_idle()

    def run_analysis(self, timeout: float = 300.0) -> bool:
        """
        Start analysis and wait for completion.

        Note: This requires the Tk mainloop to be running for threading.
        For headless testing, use run_analysis_direct() instead.

        Args:
            timeout: Maximum time to wait for analysis

        Returns:
            True if analysis completed successfully
        """
        # Trigger analysis
        if hasattr(self.app, '_run_analysis'):
            self.app._run_analysis()

        # Wait for completion
        return self.wait_for_analysis_complete(timeout)

    def apply_spxy_holdout(self, n_holdout: int = 8) -> tuple:
        """
        Apply SPXY sample selection to create holdout validation set.

        Args:
            n_holdout: Number of samples to hold out

        Returns:
            Tuple of (X_cal, y_cal, X_val, y_val)
        """
        import numpy as np
        from spectral_predict.sample_selection import spxy

        if self.app.X is None or self.app.y is None:
            raise ValueError("No data loaded in app")

        X = self.app.X.values if hasattr(self.app.X, 'values') else self.app.X
        y = self.app.y.values if hasattr(self.app.y, 'values') else self.app.y

        # Get validation indices using SPXY
        val_indices = spxy(X, y, n_samples=n_holdout)

        # Create mask for calibration samples
        all_indices = np.arange(len(X))
        cal_mask = ~np.isin(all_indices, val_indices)

        # Split data
        X_cal = self.app.X.iloc[cal_mask]
        y_cal = self.app.y.iloc[cal_mask]
        X_val = self.app.X.iloc[val_indices]
        y_val = self.app.y.iloc[val_indices]

        return X_cal, y_cal, X_val, y_val

    def run_analysis_direct(
        self,
        models: list[str] | None = None,
        preprocessing: list[str] | None = None,
        cv_folds: int = 3,
        holdout_samples: int = 0,
        holdout_method: str = 'spxy',
        tier: str | None = None
    ) -> bool:
        """
        Run analysis directly using backend, bypassing GUI threading.

        This is more reliable for automated testing because it doesn't
        require the Tk mainloop.

        Args:
            models: Models to test (default: ['PLS'])
            preprocessing: Preprocessing methods (default: ['Raw', 'SNV'])
            cv_folds: CV folds (default: 3)
            holdout_samples: Number of holdout validation samples (default: 0)
            holdout_method: Holdout method - 'spxy' or 'random' (default: 'spxy')
            tier: Model tier - 'quick', 'standard', 'comprehensive', 'experimental'
                  If None, automatically selects based on requested models

        Returns:
            True if analysis succeeded
        """
        from spectral_predict.search import run_search

        models = models or ['PLS']
        preprocessing = preprocessing or ['Raw', 'SNV']

        # Auto-select tier based on requested models if not specified
        if tier is None:
            # Models that require higher tiers
            experimental_only = {'SVR', 'MLP'}
            comprehensive_models = {'XGBoost', 'CatBoost', 'NeuralBoosted'}

            requested = set(models)
            if requested & experimental_only:
                tier = 'experimental'
            elif requested & comprehensive_models:
                tier = 'comprehensive'
            else:
                tier = 'standard'

        if self.app.X is None or self.app.y is None:
            return False

        # Apply holdout if requested
        X_train = self.app.X
        y_train = self.app.y
        X_val = None
        y_val = None

        if holdout_samples > 0:
            if holdout_method == 'spxy':
                X_train, y_train, X_val, y_val = self.apply_spxy_holdout(holdout_samples)
            else:
                # Random holdout
                import numpy as np
                n_total = len(self.app.X)
                indices = np.random.permutation(n_total)
                val_idx = indices[:holdout_samples]
                cal_idx = indices[holdout_samples:]
                X_train = self.app.X.iloc[cal_idx]
                y_train = self.app.y.iloc[cal_idx]
                X_val = self.app.X.iloc[val_idx]
                y_val = self.app.y.iloc[val_idx]

        # Determine task type
        task_type = self.app.task_type.get()
        if task_type == 'auto':
            # Auto-detect based on y dtype
            if pd.api.types.is_string_dtype(y_train.dtype) or not hasattr(y_train.dtype, 'kind'):
                task_type = 'classification'
            elif y_train.dtype.kind in 'iufb':  # int, uint, float, bool
                task_type = 'regression'
            else:
                task_type = 'regression'

        # Map preprocessing list to format expected by run_search
        # run_search expects preprocessing_methods dict like {'raw': True, 'snv': True, ...}
        preprocessing_dict = {
            'raw': 'Raw' in preprocessing,
            'snv': 'SNV' in preprocessing,
            'sg1': 'SG1' in preprocessing,
            'sg2': 'SG2' in preprocessing,
        }

        try:
            result = run_search(
                X=X_train,
                y=y_train,
                task_type=task_type,
                folds=cv_folds,
                models_to_test=models,
                preprocessing_methods=preprocessing_dict,
                validation_count=holdout_samples,
                tier=tier
            )

            # run_search returns (df_ranked, label_encoder)
            if isinstance(result, tuple):
                results_df = result[0]
            else:
                results_df = result

            # Store results in app
            self.app.results_df = results_df

            # Store holdout data if applicable
            if X_val is not None:
                self._holdout_X = X_val
                self._holdout_y = y_val

            return True

        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ==================== Results Validation ====================

    def get_results_df(self):
        """Get the analysis results DataFrame."""
        return self.app.results_df

    def validate_results(self) -> dict:
        """
        Validate that results are reasonable.

        Returns:
            Dict with validation results
        """
        results = {
            'has_results': False,
            'row_count': 0,
            'has_required_columns': False,
            'r2_valid': False,
            'accuracy_valid': False,
            'errors': []
        }

        df = self.app.results_df
        if df is None:
            results['errors'].append("No results DataFrame")
            return results

        results['has_results'] = True
        results['row_count'] = len(df)

        # Check required columns
        required_cols = {'Model', 'Rank'}
        if required_cols.issubset(df.columns):
            results['has_required_columns'] = True
        else:
            missing = required_cols - set(df.columns)
            results['errors'].append(f"Missing columns: {missing}")

        # Check R2 values (for regression)
        # Column could be 'R2', 'R2_CV', 'r2_cv', etc.
        r2_col = None
        for col in df.columns:
            if col.upper() in ('R2', 'R2_CV'):
                r2_col = col
                break

        if r2_col:
            r2_values = df[r2_col].dropna()
            if len(r2_values) > 0:
                # R2 can be arbitrarily negative for bad models
                # but should never exceed 1
                if (r2_values <= 1).all():
                    results['r2_valid'] = True
                else:
                    results['errors'].append(
                        f"R2 values out of range (>1): max={r2_values.max()}"
                    )

        # Check Accuracy values (for classification)
        if 'Accuracy' in df.columns or 'accuracy' in df.columns:
            acc_col = 'Accuracy' if 'Accuracy' in df.columns else 'accuracy'
            acc_values = df[acc_col].dropna()
            if len(acc_values) > 0:
                if (acc_values >= 0).all() and (acc_values <= 1).all():
                    results['accuracy_valid'] = True
                else:
                    results['errors'].append(
                        f"Accuracy values out of range: min={acc_values.min()}, max={acc_values.max()}"
                    )

        return results

    # ==================== Visibility Control ====================

    def show(self) -> None:
        """Make the window visible (for debugging)."""
        self.root.deiconify()
        self.root.update()
        self.visible = True

    def hide(self) -> None:
        """Hide the window."""
        self.root.withdraw()
        self.visible = False

    # ==================== Debug Helpers ====================

    def print_app_state(self) -> None:
        """Print current application state for debugging."""
        print("\n=== App State ===")
        print(f"X: {type(self.app.X).__name__}, shape: {self.app.X.shape if self.app.X is not None else 'None'}")
        print(f"y: {type(self.app.y).__name__}, len: {len(self.app.y) if self.app.y is not None else 'None'}")
        print(f"task_type: {self.get_var('task_type')}")
        print(f"results_df: {len(self.app.results_df) if self.app.results_df is not None else 'None'} rows")
        print(f"Current tab: {self.get_current_tab_index()}")
        print("=================\n")

    def screenshot(self, filename: str = "gui_screenshot.png") -> None:
        """
        Take a screenshot of the GUI (requires visible window).

        Args:
            filename: Output filename
        """
        if not self.visible:
            print("Cannot screenshot hidden window")
            return

        try:
            import pyautogui
            self.root.update()
            time.sleep(0.2)
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            print(f"Screenshot saved: {filename}")
        except ImportError:
            print("pyautogui not installed - cannot take screenshot")
