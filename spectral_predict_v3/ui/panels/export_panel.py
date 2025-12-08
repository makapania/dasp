"""
Export for Publication Panel for Spectral Predict v3.

Generates standalone Python scripts and Jupyter notebooks from
trained models for sharing with reviewers and ensuring reproducibility.
"""

import dearpygui.dearpygui as dpg
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ...core.code_generator import CodeGenerator, ExportOptions
from ..tooltips import add_tooltip, TOOLTIP_CONTENT


class ExportPanel:
    """
    Panel for exporting analysis code for publication.

    Allows users to generate standalone Python scripts or Jupyter notebooks
    that reproduce their analysis workflow.
    """

    def __init__(self, parent_tag: str, get_model_bundle_callback=None, show_save_dialog_callback=None):
        """
        Initialize export panel.

        Parameters
        ----------
        parent_tag : str
            DearPyGui tag of parent container
        get_model_bundle_callback : callable, optional
            Callback to get the current model bundle from the app
        show_save_dialog_callback : callable, optional
            Callback to show save dialog: show_save_dialog(callback, dialog_type, initial_dir)
        """
        self.parent_tag = parent_tag
        self.get_model_bundle = get_model_bundle_callback
        self.show_save_dialog = show_save_dialog_callback

        # Current model bundle
        self.model_bundle = None

        # Generated code
        self.generated_code = ""

        # Pending export data for async save callback
        self._pending_export_generator = None
        self._pending_export_format = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the export panel UI."""
        with dpg.child_window(parent=self.parent_tag, tag=f"{self.parent_tag}_export_main"):
            # Header
            dpg.add_text("Export for Publication", tag=f"{self.parent_tag}_export_title")
            dpg.add_text(
                "Generate standalone Python scripts for reviewers and reproducibility.",
                color=(150, 150, 150)
            )
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Two-column layout
            with dpg.group(horizontal=True):
                # Left column: Options
                with dpg.child_window(width=350, height=-50, border=True):
                    self._build_options_panel()

                dpg.add_spacer(width=10)

                # Right column: Preview
                with dpg.child_window(width=-1, height=-50, border=True):
                    self._build_preview_panel()

            # Bottom buttons
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Generate Preview",
                    callback=self._on_generate_preview,
                    tag=f"{self.parent_tag}_export_generate_btn"
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="Export to File",
                    callback=self._on_export_file,
                    tag=f"{self.parent_tag}_export_file_btn"
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Copy to Clipboard",
                    callback=self._on_copy_clipboard,
                    tag=f"{self.parent_tag}_export_copy_btn"
                )
                dpg.add_spacer(width=30)
                dpg.add_text("", tag=f"{self.parent_tag}_export_status", color=(100, 200, 100))

        # Set up tooltips
        self._setup_tooltips()

    def _setup_tooltips(self):
        """Set up tooltips for export panel elements."""
        # Format selection
        add_tooltip(f"{self.parent_tag}_export_format",
            TOOLTIP_CONTENT['export']['format_python'] + "\n" + TOOLTIP_CONTENT['export']['format_notebook'])

        # Include options
        add_tooltip(f"{self.parent_tag}_export_inc_data", TOOLTIP_CONTENT['export']['include_data_loading'])
        add_tooltip(f"{self.parent_tag}_export_inc_preproc", TOOLTIP_CONTENT['export']['include_preprocessing'])
        add_tooltip(f"{self.parent_tag}_export_inc_varsel", TOOLTIP_CONTENT['export']['include_variable_selection'])
        add_tooltip(f"{self.parent_tag}_export_inc_cv", TOOLTIP_CONTENT['export']['include_cv'])
        add_tooltip(f"{self.parent_tag}_export_inc_viz", TOOLTIP_CONTENT['export']['include_visualization'])
        add_tooltip(f"{self.parent_tag}_export_inc_pred", TOOLTIP_CONTENT['export']['include_prediction'])

        # Buttons
        add_tooltip(f"{self.parent_tag}_export_generate_btn",
            "Generate a preview of the export code without saving to file.")
        add_tooltip(f"{self.parent_tag}_export_file_btn",
            "Save the generated code to a file. Choose .py for Python script or .ipynb for Jupyter notebook.")
        add_tooltip(f"{self.parent_tag}_export_copy_btn",
            "Copy the generated code to clipboard for pasting into another application.")

        # Data path
        add_tooltip(f"{self.parent_tag}_export_data_path",
            "Placeholder path that will appear in the generated code. "
            "Replace with the actual path when using the exported code.")

    def _build_options_panel(self):
        """Build the options panel on the left."""
        dpg.add_text("Export Options", color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Format selection
        dpg.add_text("Output Format:")
        with dpg.group(horizontal=True):
            dpg.add_radio_button(
                items=["Python Script (.py)", "Jupyter Notebook (.ipynb)"],
                default_value="Python Script (.py)",
                tag=f"{self.parent_tag}_export_format",
                horizontal=False
            )

        dpg.add_spacer(height=15)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Include options
        dpg.add_text("Include in Export:")
        dpg.add_spacer(height=5)

        dpg.add_checkbox(
            label="Data loading example",
            default_value=True,
            tag=f"{self.parent_tag}_export_inc_data"
        )
        dpg.add_checkbox(
            label="Preprocessing functions",
            default_value=True,
            tag=f"{self.parent_tag}_export_inc_preproc"
        )
        dpg.add_checkbox(
            label="Variable selection code",
            default_value=True,
            tag=f"{self.parent_tag}_export_inc_varsel"
        )
        dpg.add_checkbox(
            label="Cross-validation",
            default_value=True,
            tag=f"{self.parent_tag}_export_inc_cv"
        )
        dpg.add_checkbox(
            label="Visualization (matplotlib)",
            default_value=False,
            tag=f"{self.parent_tag}_export_inc_viz"
        )
        dpg.add_checkbox(
            label="Prediction template",
            default_value=True,
            tag=f"{self.parent_tag}_export_inc_pred"
        )

        dpg.add_spacer(height=15)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Data path placeholder
        dpg.add_text("Data File Path (placeholder):")
        dpg.add_input_text(
            default_value="your_data.csv",
            tag=f"{self.parent_tag}_export_data_path",
            width=300
        )

        dpg.add_spacer(height=15)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Model info (read-only)
        dpg.add_text("Current Model:", color=(200, 200, 255))
        dpg.add_text("(No model selected)", tag=f"{self.parent_tag}_export_model_info", wrap=300)

    def _build_preview_panel(self):
        """Build the preview panel on the right."""
        dpg.add_text("Code Preview", color=(200, 200, 255))
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Preview text area
        dpg.add_input_text(
            multiline=True,
            readonly=True,
            default_value="# Click 'Generate Preview' to see the generated code\n\n"
                          "# Make sure you have:\n"
                          "# 1. Trained a model in the Build tab\n"
                          "# 2. Selected the model to export\n",
            tag=f"{self.parent_tag}_export_preview",
            width=-1,
            height=-1,
            tab_input=True
        )

    def set_model_bundle(self, model_bundle: Dict[str, Any]):
        """
        Set the model bundle to export.

        Parameters
        ----------
        model_bundle : dict
            Model bundle containing model, preprocessing, etc.
        """
        self.model_bundle = model_bundle

        # Update model info display
        if model_bundle:
            info = (
                f"Model: {model_bundle.get('model_name', 'Unknown')}\n"
                f"Preprocessing: {model_bundle.get('preprocessing', 'raw')}\n"
                f"Target: {model_bundle.get('target_name', 'Unknown')}\n"
                f"Task: {model_bundle.get('task_type', 'Unknown')}"
            )
            if model_bundle.get('variable_indices') is not None:
                n_vars = len(model_bundle['variable_indices'])
                info += f"\nVariables: {n_vars} selected"
        else:
            info = "(No model selected)"

        if dpg.does_item_exist(f"{self.parent_tag}_export_model_info"):
            dpg.set_value(f"{self.parent_tag}_export_model_info", info)

    def _get_export_options(self) -> ExportOptions:
        """Get export options from UI."""
        format_value = dpg.get_value(f"{self.parent_tag}_export_format")
        is_notebook = "Notebook" in format_value

        return ExportOptions(
            include_data_loading=dpg.get_value(f"{self.parent_tag}_export_inc_data"),
            include_preprocessing=dpg.get_value(f"{self.parent_tag}_export_inc_preproc"),
            include_variable_selection=dpg.get_value(f"{self.parent_tag}_export_inc_varsel"),
            include_cross_validation=dpg.get_value(f"{self.parent_tag}_export_inc_cv"),
            include_visualization=dpg.get_value(f"{self.parent_tag}_export_inc_viz"),
            include_prediction_template=dpg.get_value(f"{self.parent_tag}_export_inc_pred"),
            format='notebook' if is_notebook else 'script',
            data_path=dpg.get_value(f"{self.parent_tag}_export_data_path"),
            target_column=self.model_bundle.get('target_name', 'target') if self.model_bundle else 'target'
        )

    def _on_generate_preview(self, sender=None, app_data=None):
        """Generate code preview."""
        # Try to get model bundle from callback if not set
        if self.model_bundle is None and self.get_model_bundle:
            self.model_bundle = self.get_model_bundle()

        if self.model_bundle is None:
            dpg.set_value(
                f"{self.parent_tag}_export_preview",
                "# Error: No model selected\n\n"
                "# Please:\n"
                "# 1. Go to the Build tab and train a model\n"
                "# 2. Select a model from the results\n"
                "# 3. Return here to export"
            )
            self._set_status("No model to export", error=True)
            return

        try:
            options = self._get_export_options()
            generator = CodeGenerator(self.model_bundle, options)

            if options.format == 'notebook':
                # For notebook, show a summary
                notebook = generator.generate_notebook()
                n_cells = len(notebook['cells'])
                preview = (
                    f"# Jupyter Notebook Generated\n"
                    f"# Cells: {n_cells}\n\n"
                    f"# Cell types:\n"
                )
                for i, cell in enumerate(notebook['cells'][:10]):
                    cell_type = cell['cell_type']
                    source = cell.get('source', [''])[0][:50]
                    preview += f"# [{i+1}] {cell_type}: {source}...\n"
                if n_cells > 10:
                    preview += f"# ... and {n_cells - 10} more cells\n"

                preview += "\n# Click 'Export to File' to save the notebook"
                self.generated_code = preview
            else:
                self.generated_code = generator.generate_script()

            dpg.set_value(f"{self.parent_tag}_export_preview", self.generated_code)
            self._set_status(f"Preview generated ({len(self.generated_code)} chars)")

        except Exception as e:
            dpg.set_value(
                f"{self.parent_tag}_export_preview",
                f"# Error generating code:\n# {str(e)}"
            )
            self._set_status(f"Error: {str(e)}", error=True)

    def _on_export_file(self, sender=None, app_data=None):
        """Export code to file."""
        if not self.generated_code and self.model_bundle:
            self._on_generate_preview()

        if not self.model_bundle:
            self._set_status("No model to export", error=True)
            return

        try:
            options = self._get_export_options()
            generator = CodeGenerator(self.model_bundle, options)

            # Store pending export state for callback
            self._pending_export_generator = generator
            self._pending_export_format = options.format

            def save_callback(filepath):
                try:
                    gen = self._pending_export_generator
                    fmt = self._pending_export_format
                    if gen is None:
                        return

                    if fmt == 'notebook':
                        gen.save_notebook(filepath)
                    else:
                        gen.save_script(filepath)

                    dpg.set_value(
                        f"{self.parent_tag}_export_status",
                        f"Saved: {Path(filepath).name}"
                    )
                except Exception as e:
                    dpg.set_value(
                        f"{self.parent_tag}_export_status",
                        f"Error: {str(e)}"
                    )
                finally:
                    self._pending_export_generator = None
                    self._pending_export_format = None

            # Use DearPyGui dialog via callback if available
            if self.show_save_dialog:
                dialog_type = "py"  # Python files (can also handle .ipynb)
                self.show_save_dialog(save_callback, dialog_type=dialog_type)

        except Exception as e:
            self._set_status(f"Error: {str(e)}", error=True)

    def _on_copy_clipboard(self, sender=None, app_data=None):
        """Copy code to clipboard."""
        if not self.generated_code:
            self._on_generate_preview()

        if not self.generated_code:
            self._set_status("Nothing to copy", error=True)
            return

        try:
            import pyperclip
            pyperclip.copy(self.generated_code)
            self._set_status("Copied to clipboard!")
        except ImportError:
            # Fallback: try tkinter
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(self.generated_code)
                root.update()
                root.destroy()
                self._set_status("Copied to clipboard!")
            except Exception as e:
                self._set_status(f"Copy failed: {str(e)}", error=True)
        except Exception as e:
            self._set_status(f"Copy failed: {str(e)}", error=True)

    def _set_status(self, message: str, error: bool = False):
        """Set status message."""
        if dpg.does_item_exist(f"{self.parent_tag}_export_status"):
            color = (255, 100, 100) if error else (100, 200, 100)
            dpg.set_value(f"{self.parent_tag}_export_status", message)
            dpg.configure_item(f"{self.parent_tag}_export_status", color=color)
