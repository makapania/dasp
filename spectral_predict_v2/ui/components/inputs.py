"""
Input Components - Spectral Predict v2

Styled form input widgets with consistent appearance.
"""

from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY, INPUT


class StyledLineEdit(QLineEdit):
    """
    Styled single-line text input.

    Features:
    - Consistent styling with theme
    - Optional placeholder
    - Optional prefix/suffix text
    """

    def __init__(
        self,
        placeholder: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if placeholder:
            self.setPlaceholderText(placeholder)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            StyledLineEdit {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {INPUT["padding_x"]}px;
                min-height: {INPUT["height_md"] - 14}px;
                selection-background-color: {COLORS["accent_secondary"]};
                selection-color: #ffffff;
            }}
            StyledLineEdit:hover {{
                border-color: {COLORS["border_emphasis"]};
            }}
            StyledLineEdit:focus {{
                border-color: {COLORS["accent_secondary"]};
                background-color: {COLORS["bg_surface"]};
            }}
            StyledLineEdit:disabled {{
                background-color: {COLORS["bg_surface"]};
                color: {COLORS["text_tertiary"]};
                border-color: {COLORS["border_subtle"]};
            }}
            StyledLineEdit[readOnly="true"] {{
                background-color: {COLORS["bg_surface"]};
            }}
        """)


class StyledComboBox(QComboBox):
    """
    Styled dropdown/combobox.

    Features:
    - Consistent styling with theme
    - Custom dropdown arrow
    - Styled popup list
    """

    def __init__(
        self,
        items: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if items:
            self.addItems(items)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            StyledComboBox {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {INPUT["padding_x"]}px;
                padding-right: 30px;
                min-height: {INPUT["height_md"] - 14}px;
            }}
            StyledComboBox:hover {{
                border-color: {COLORS["border_emphasis"]};
            }}
            StyledComboBox:focus {{
                border-color: {COLORS["accent_secondary"]};
            }}
            StyledComboBox:disabled {{
                background-color: {COLORS["bg_surface"]};
                color: {COLORS["text_tertiary"]};
            }}
            StyledComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border: none;
                background: transparent;
            }}
            StyledComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {COLORS["text_secondary"]};
                margin-right: 8px;
            }}
            StyledComboBox::down-arrow:hover {{
                border-top-color: {COLORS["text_primary"]};
            }}
            StyledComboBox QAbstractItemView {{
                background-color: {COLORS["bg_overlay"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 4px;
                selection-background-color: {COLORS["accent_secondary"]};
                selection-color: #ffffff;
                outline: none;
            }}
            StyledComboBox QAbstractItemView::item {{
                padding: 6px 12px;
                min-height: 24px;
                border-radius: {RADIUS["sm"]}px;
            }}
            StyledComboBox QAbstractItemView::item:hover {{
                background-color: {COLORS["bg_elevated"]};
            }}
            StyledComboBox QAbstractItemView::item:selected {{
                background-color: {COLORS["accent_secondary"]};
            }}
        """)


class StyledSpinBox(QSpinBox):
    """
    Styled integer spinbox.

    Features:
    - Consistent styling with theme
    - Custom up/down buttons
    """

    def __init__(
        self,
        minimum: int = 0,
        maximum: int = 100,
        value: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            StyledSpinBox {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {INPUT["padding_x"]}px;
                padding-right: 20px;
                min-height: {INPUT["height_md"] - 14}px;
            }}
            StyledSpinBox:hover {{
                border-color: {COLORS["border_emphasis"]};
            }}
            StyledSpinBox:focus {{
                border-color: {COLORS["accent_secondary"]};
            }}
            StyledSpinBox::up-button, StyledSpinBox::down-button {{
                background-color: transparent;
                border: none;
                width: 16px;
            }}
            StyledSpinBox::up-button:hover, StyledSpinBox::down-button:hover {{
                background-color: {COLORS["bg_overlay"]};
            }}
            StyledSpinBox::up-arrow {{
                width: 8px;
                height: 8px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid {COLORS["text_secondary"]};
            }}
            StyledSpinBox::down-arrow {{
                width: 8px;
                height: 8px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid {COLORS["text_secondary"]};
            }}
        """)


class StyledDoubleSpinBox(QDoubleSpinBox):
    """
    Styled floating-point spinbox.

    Features:
    - Consistent styling with theme
    - Configurable precision
    """

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 100.0,
        value: float = 0.0,
        decimals: int = 2,
        step: float = 0.1,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            StyledDoubleSpinBox {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {INPUT["padding_x"]}px;
                padding-right: 20px;
                min-height: {INPUT["height_md"] - 14}px;
            }}
            StyledDoubleSpinBox:hover {{
                border-color: {COLORS["border_emphasis"]};
            }}
            StyledDoubleSpinBox:focus {{
                border-color: {COLORS["accent_secondary"]};
            }}
            StyledDoubleSpinBox::up-button, StyledDoubleSpinBox::down-button {{
                background-color: transparent;
                border: none;
                width: 16px;
            }}
            StyledDoubleSpinBox::up-button:hover, StyledDoubleSpinBox::down-button:hover {{
                background-color: {COLORS["bg_overlay"]};
            }}
            StyledDoubleSpinBox::up-arrow {{
                width: 8px;
                height: 8px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid {COLORS["text_secondary"]};
            }}
            StyledDoubleSpinBox::down-arrow {{
                width: 8px;
                height: 8px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid {COLORS["text_secondary"]};
            }}
        """)


class StyledSlider(QSlider):
    """
    Styled horizontal slider.

    Features:
    - Consistent styling with theme
    - Optional value display
    """

    def __init__(
        self,
        orientation: Qt.Orientation = Qt.Orientation.Horizontal,
        minimum: int = 0,
        maximum: int = 100,
        value: int = 50,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(orientation, parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            StyledSlider::groove:horizontal {{
                background-color: {COLORS["bg_overlay"]};
                height: 6px;
                border-radius: 3px;
            }}
            StyledSlider::handle:horizontal {{
                background-color: {COLORS["accent_secondary"]};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            StyledSlider::handle:horizontal:hover {{
                background-color: {COLORS["accent_secondary_hover"]};
            }}
            StyledSlider::sub-page:horizontal {{
                background-color: {COLORS["accent_secondary"]};
                border-radius: 3px;
            }}
            StyledSlider::groove:vertical {{
                background-color: {COLORS["bg_overlay"]};
                width: 6px;
                border-radius: 3px;
            }}
            StyledSlider::handle:vertical {{
                background-color: {COLORS["accent_secondary"]};
                width: 16px;
                height: 16px;
                margin: 0 -5px;
                border-radius: 8px;
            }}
            StyledSlider::add-page:vertical {{
                background-color: {COLORS["accent_secondary"]};
                border-radius: 3px;
            }}
        """)


class StyledCheckBox(QCheckBox):
    """
    Styled checkbox.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            StyledCheckBox {{
                spacing: 8px;
                color: {COLORS["text_primary"]};
            }}
            StyledCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["sm"]}px;
                background-color: {COLORS["bg_elevated"]};
            }}
            StyledCheckBox::indicator:hover {{
                border-color: {COLORS["accent_secondary"]};
            }}
            StyledCheckBox::indicator:checked {{
                background-color: {COLORS["accent_secondary"]};
                border-color: {COLORS["accent_secondary"]};
            }}
            StyledCheckBox:disabled {{
                color: {COLORS["text_tertiary"]};
            }}
            StyledCheckBox::indicator:disabled {{
                background-color: {COLORS["bg_surface"]};
                border-color: {COLORS["border_subtle"]};
            }}
        """)


class LabeledInput(QWidget):
    """
    A form field with label, input, and optional helper text.

    Layout:
        Label
        [Input Widget]
        Helper text (optional)

    Use this for consistent form field layouts.
    """

    def __init__(
        self,
        label: str,
        input_widget: QWidget,
        helper_text: str = "",
        required: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._input_widget = input_widget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["xs"])

        # Label row
        label_layout = QHBoxLayout()
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(SPACING["xs"])

        self._label = QLabel(label)
        self._label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-weight: {TYPOGRAPHY["weight_medium"]};
        """)
        label_layout.addWidget(self._label)

        if required:
            required_marker = QLabel("*")
            required_marker.setStyleSheet(f"color: {COLORS['accent_danger']};")
            label_layout.addWidget(required_marker)

        label_layout.addStretch()
        layout.addLayout(label_layout)

        # Input widget
        layout.addWidget(input_widget)

        # Helper text
        if helper_text:
            self._helper = QLabel(helper_text)
            self._helper.setStyleSheet(f"""
                color: {COLORS["text_secondary"]};
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            """)
            self._helper.setWordWrap(True)
            layout.addWidget(self._helper)
        else:
            self._helper = None

    def input_widget(self) -> QWidget:
        """Get the input widget."""
        return self._input_widget

    def set_helper_text(self, text: str):
        """Update the helper text."""
        if self._helper:
            self._helper.setText(text)

    def set_error(self, error_text: str):
        """Show error state with message."""
        if self._helper:
            self._helper.setText(error_text)
            self._helper.setStyleSheet(f"""
                color: {COLORS["accent_danger"]};
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            """)

        # Add error border to input
        if hasattr(self._input_widget, 'setStyleSheet'):
            current_style = self._input_widget.styleSheet()
            # This is a simple approach - in production, use properties
            self._input_widget.setProperty("error", True)
            self._input_widget.style().unpolish(self._input_widget)
            self._input_widget.style().polish(self._input_widget)

    def clear_error(self):
        """Clear error state."""
        if self._helper:
            self._helper.setStyleSheet(f"""
                color: {COLORS["text_secondary"]};
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            """)
        if hasattr(self._input_widget, 'setProperty'):
            self._input_widget.setProperty("error", False)
            self._input_widget.style().unpolish(self._input_widget)
            self._input_widget.style().polish(self._input_widget)


class RadioGroup(QWidget):
    """
    A group of radio buttons.

    Features:
    - Consistent styling
    - Single selection
    - Value access by index or label
    """

    selection_changed = Signal(int)  # Emits selected index

    def __init__(
        self,
        options: List[str],
        default_index: int = 0,
        orientation: Qt.Orientation = Qt.Orientation.Vertical,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._options = options

        if orientation == Qt.Orientation.Vertical:
            layout = QVBoxLayout(self)
        else:
            layout = QHBoxLayout(self)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])

        self._button_group = QButtonGroup(self)
        self._buttons: List[QRadioButton] = []

        for i, option in enumerate(options):
            radio = QRadioButton(option)
            radio.setStyleSheet(f"""
                QRadioButton {{
                    spacing: 8px;
                    color: {COLORS["text_primary"]};
                }}
                QRadioButton::indicator {{
                    width: 16px;
                    height: 16px;
                    border: 1px solid {COLORS["border_default"]};
                    border-radius: 8px;
                    background-color: {COLORS["bg_elevated"]};
                }}
                QRadioButton::indicator:hover {{
                    border-color: {COLORS["accent_secondary"]};
                }}
                QRadioButton::indicator:checked {{
                    background-color: {COLORS["accent_secondary"]};
                    border-color: {COLORS["accent_secondary"]};
                }}
            """)

            if i == default_index:
                radio.setChecked(True)

            self._button_group.addButton(radio, i)
            self._buttons.append(radio)
            layout.addWidget(radio)

        self._button_group.idClicked.connect(self.selection_changed.emit)

    def selected_index(self) -> int:
        """Get the index of the selected option."""
        return self._button_group.checkedId()

    def selected_value(self) -> str:
        """Get the text of the selected option."""
        idx = self.selected_index()
        if 0 <= idx < len(self._options):
            return self._options[idx]
        return ""

    def set_selected(self, index: int):
        """Set the selected option by index."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)


class SearchInput(QWidget):
    """
    Search input field with icon and clear button.
    """

    text_changed = Signal(str)
    search_submitted = Signal(str)

    def __init__(
        self,
        placeholder: str = "Search...",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Container with search styling
        self._container = QWidget()
        container_layout = QHBoxLayout(self._container)
        container_layout.setContentsMargins(SPACING["sm"], 0, SPACING["sm"], 0)
        container_layout.setSpacing(SPACING["sm"])

        # Search icon (as label with text)
        # In production, use actual icon
        self._icon_label = QLabel("\U0001F50D")  # Magnifying glass emoji
        self._icon_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        container_layout.addWidget(self._icon_label)

        # Input field
        self._input = QLineEdit()
        self._input.setPlaceholderText(placeholder)
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background-color: transparent;
                color: {COLORS["text_primary"]};
                border: none;
                padding: 6px 0;
            }}
        """)
        self._input.textChanged.connect(self.text_changed.emit)
        self._input.returnPressed.connect(
            lambda: self.search_submitted.emit(self._input.text())
        )
        container_layout.addWidget(self._input, 1)

        # Container styling
        self._container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS["bg_elevated"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
            }}
            QWidget:focus-within {{
                border-color: {COLORS["accent_secondary"]};
            }}
        """)

        layout.addWidget(self._container)

    def text(self) -> str:
        """Get the current search text."""
        return self._input.text()

    def set_text(self, text: str):
        """Set the search text."""
        self._input.setText(text)

    def clear(self):
        """Clear the search field."""
        self._input.clear()
