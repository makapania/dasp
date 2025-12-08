"""
Spectral Predict v3 - Entry Point

GPU-accelerated spectroscopy analysis application.

This module handles:
- Application startup
- Command-line argument processing for file association
"""

import sys
from pathlib import Path


def main():
    """Entry point for the application."""
    from spectral_predict_v3.ui.app import SpectralPredictApp

    app = SpectralPredictApp()

    # Check for file argument (file association)
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
        if project_path.suffix.lower() == '.sproject' and project_path.exists():
            # Schedule project load after UI initialization
            app.schedule_project_load(str(project_path))

    app.run()


if __name__ == "__main__":
    main()
