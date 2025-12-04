"""
Spectral Predict v2 - UI Components

Reusable styled components for the application.
"""

from .cards import Card, CardHeader, CollapsibleCard, StatusCard
from .buttons import (
    PrimaryButton,
    SecondaryButton,
    DangerButton,
    GhostButton,
    IconButton,
    OutlineButton,
)
from .inputs import (
    StyledLineEdit,
    StyledComboBox,
    StyledSpinBox,
    StyledDoubleSpinBox,
    StyledSlider,
    StyledCheckBox,
    LabeledInput,
)

__all__ = [
    # Cards
    "Card",
    "CardHeader",
    "CollapsibleCard",
    "StatusCard",
    # Buttons
    "PrimaryButton",
    "SecondaryButton",
    "DangerButton",
    "GhostButton",
    "IconButton",
    "OutlineButton",
    # Inputs
    "StyledLineEdit",
    "StyledComboBox",
    "StyledSpinBox",
    "StyledDoubleSpinBox",
    "StyledSlider",
    "StyledCheckBox",
    "LabeledInput",
]
