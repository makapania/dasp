"""
Spectral Predict v2 - Modern Automated Spectral Analysis Platform

Entry point for the application.
"""

import sys
from PySide6.QtWidgets import QApplication
from ui.app import SpectralPredictApp


def main():
    """Launch the Spectral Predict application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Spectral Predict")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("DASP")

    window = SpectralPredictApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
