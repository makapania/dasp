"""
Pareto Plot Component for NSGA-II visualization.

Interactive 2D scatter plot for displaying Pareto front solutions
with knee point highlighting and solution selection.
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Dict, List, Optional, Callable, Any


class ParetoPlot:
    """
    Interactive Pareto front visualization component.

    Displays multi-objective optimization results as a scatter plot
    with interactive selection and knee point highlighting.
    """

    def __init__(
        self,
        parent: int,
        width: int = 500,
        height: int = 400,
        on_solution_select: Optional[Callable[[int, Dict], None]] = None,
    ):
        """
        Initialize Pareto plot.

        Parameters
        ----------
        parent : int
            DearPyGui parent container ID
        width : int
            Plot width in pixels
        height : int
            Plot height in pixels
        on_solution_select : callable, optional
            Callback when a solution is selected: fn(solution_idx, solution_data)
        """
        self.parent = parent
        self.width = width
        self.height = height
        self.on_solution_select = on_solution_select

        self._pareto_front = None
        self._pareto_solutions = None
        self._knee_idx = -1
        self._selected_idx = -1
        self._objective_labels = ['Objective 1', 'Objective 2', 'Objective 3']

        self._plot_tag = None
        self._scatter_tag = None
        self._knee_tag = None
        self._selected_tag = None
        self._x_axis = None
        self._y_axis = None

        self._create_plot()

    def _create_plot(self):
        """Create the DearPyGui plot structure."""
        with dpg.group(parent=self.parent):
            # Objective selector
            with dpg.group(horizontal=True):
                dpg.add_text("X-Axis:")
                dpg.add_combo(
                    items=['Error', 'Wavelengths', 'Complexity'],
                    default_value='Error',
                    width=120,
                    tag=f"pareto_x_obj",
                    callback=self._on_axis_change
                )
                dpg.add_spacer(width=20)
                dpg.add_text("Y-Axis:")
                dpg.add_combo(
                    items=['Error', 'Wavelengths', 'Complexity'],
                    default_value='Wavelengths',
                    width=120,
                    tag=f"pareto_y_obj",
                    callback=self._on_axis_change
                )

            # The plot
            with dpg.plot(
                label="Pareto Front",
                width=self.width,
                height=self.height,
                tag="pareto_plot_main"
            ) as self._plot_tag:
                # Legend
                dpg.add_plot_legend()

                # X axis
                self._x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Error (RMSE)")

                # Y axis
                with dpg.plot_axis(dpg.mvYAxis, label="Wavelength Fraction") as self._y_axis:
                    # Pareto front scatter
                    self._scatter_tag = dpg.add_scatter_series(
                        [], [],
                        label="Solutions",
                        tag="pareto_scatter"
                    )

                    # Knee point marker
                    self._knee_tag = dpg.add_scatter_series(
                        [], [],
                        label="Knee Point",
                        tag="pareto_knee"
                    )

                    # Selected point marker
                    self._selected_tag = dpg.add_scatter_series(
                        [], [],
                        label="Selected",
                        tag="pareto_selected"
                    )

            # Solution info panel
            with dpg.group(tag="pareto_solution_info"):
                dpg.add_text("Select a solution to view details", tag="pareto_info_text")

    def set_data(
        self,
        pareto_front: np.ndarray,
        pareto_solutions: np.ndarray = None,
        knee_idx: int = -1,
        objective_labels: List[str] = None,
    ):
        """
        Set the Pareto front data.

        Parameters
        ----------
        pareto_front : ndarray, shape (n_solutions, n_objectives)
            Objective values for each Pareto solution
        pareto_solutions : ndarray, optional
            Decision variable values (for callbacks)
        knee_idx : int
            Index of the knee point solution
        objective_labels : list of str, optional
            Labels for each objective
        """
        self._pareto_front = pareto_front
        self._pareto_solutions = pareto_solutions
        self._knee_idx = knee_idx

        if objective_labels:
            self._objective_labels = objective_labels

        self._update_plot()

    def _update_plot(self):
        """Update the plot with current data."""
        if self._pareto_front is None or len(self._pareto_front) == 0:
            return

        # Get current axis selection
        x_obj_name = dpg.get_value("pareto_x_obj")
        y_obj_name = dpg.get_value("pareto_y_obj")

        obj_map = {'Error': 0, 'Wavelengths': 1, 'Complexity': 2}
        x_idx = obj_map.get(x_obj_name, 0)
        y_idx = obj_map.get(y_obj_name, 1)

        x_data = self._pareto_front[:, x_idx].tolist()
        y_data = self._pareto_front[:, y_idx].tolist()

        # Update scatter data
        dpg.set_value("pareto_scatter", [x_data, y_data])

        # Update axis labels
        dpg.set_item_label(self._x_axis, x_obj_name)
        dpg.set_item_label(self._y_axis, y_obj_name)

        # Update knee point
        if self._knee_idx >= 0 and self._knee_idx < len(self._pareto_front):
            knee_x = [self._pareto_front[self._knee_idx, x_idx]]
            knee_y = [self._pareto_front[self._knee_idx, y_idx]]
            dpg.set_value("pareto_knee", [knee_x, knee_y])
        else:
            dpg.set_value("pareto_knee", [[], []])

        # Update selected point
        if self._selected_idx >= 0 and self._selected_idx < len(self._pareto_front):
            sel_x = [self._pareto_front[self._selected_idx, x_idx]]
            sel_y = [self._pareto_front[self._selected_idx, y_idx]]
            dpg.set_value("pareto_selected", [sel_x, sel_y])
        else:
            dpg.set_value("pareto_selected", [[], []])

        # Auto-fit axes
        dpg.fit_axis_data(self._x_axis)
        dpg.fit_axis_data(self._y_axis)

    def _on_axis_change(self, sender=None, app_data=None):
        """Handle axis objective change."""
        self._update_plot()

    def select_solution(self, idx: int):
        """
        Select a solution by index.

        Parameters
        ----------
        idx : int
            Solution index to select
        """
        if self._pareto_front is None:
            return

        if idx < 0 or idx >= len(self._pareto_front):
            self._selected_idx = -1
            dpg.set_value("pareto_selected", [[], []])
            dpg.set_value("pareto_info_text", "No solution selected")
            return

        self._selected_idx = idx
        self._update_plot()

        # Update info text
        obj = self._pareto_front[idx]
        info = f"Solution {idx}: Error={obj[0]:.4f}, Wavelengths={obj[1]:.3f}, Complexity={obj[2]:.3f}"
        if idx == self._knee_idx:
            info += " [KNEE POINT]"
        dpg.set_value("pareto_info_text", info)

        # Callback
        if self.on_solution_select and self._pareto_solutions is not None:
            self.on_solution_select(idx, {
                'objectives': obj,
                'solution': self._pareto_solutions[idx] if self._pareto_solutions is not None else None,
                'is_knee': idx == self._knee_idx,
            })

    def select_knee(self):
        """Select the knee point solution."""
        if self._knee_idx >= 0:
            self.select_solution(self._knee_idx)

    def clear(self):
        """Clear the plot."""
        self._pareto_front = None
        self._pareto_solutions = None
        self._knee_idx = -1
        self._selected_idx = -1

        dpg.set_value("pareto_scatter", [[], []])
        dpg.set_value("pareto_knee", [[], []])
        dpg.set_value("pareto_selected", [[], []])
        dpg.set_value("pareto_info_text", "No data loaded")


def create_pareto_results_table(
    parent: int,
    pareto_df,
    on_row_select: Optional[Callable[[int], None]] = None,
) -> int:
    """
    Create a table displaying Pareto front results.

    Parameters
    ----------
    parent : int
        DearPyGui parent container
    pareto_df : pd.DataFrame
        Pareto results DataFrame from pareto_to_dataframe()
    on_row_select : callable, optional
        Callback when a row is selected

    Returns
    -------
    table_tag : int
        The table widget ID
    """
    if pareto_df is None or len(pareto_df) == 0:
        dpg.add_text("No Pareto solutions available", parent=parent)
        return None

    # Columns to display
    display_cols = ['Model', 'Preprocessing', 'N_Variables', 'RMSE', 'Complexity', 'Is_Knee']
    if 'Accuracy' in pareto_df.columns:
        display_cols = ['Model', 'Preprocessing', 'N_Variables', 'Accuracy', 'Complexity', 'Is_Knee']

    available_cols = [c for c in display_cols if c in pareto_df.columns]

    with dpg.table(
        parent=parent,
        header_row=True,
        borders_innerH=True,
        borders_innerV=True,
        borders_outerH=True,
        borders_outerV=True,
        resizable=True,
        scrollY=True,
        height=200,
        tag="pareto_results_table"
    ) as table_tag:
        # Add columns
        dpg.add_table_column(label="#")
        for col in available_cols:
            dpg.add_table_column(label=col)

        # Add rows
        for idx, row in pareto_df.iterrows():
            with dpg.table_row():
                dpg.add_text(str(idx))
                for col in available_cols:
                    val = row[col]
                    if isinstance(val, float):
                        dpg.add_text(f"{val:.4f}")
                    elif isinstance(val, bool):
                        dpg.add_text("*" if val else "")
                    else:
                        dpg.add_text(str(val))

    return table_tag
