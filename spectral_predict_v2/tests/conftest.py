"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_spectral_data():
    """Generate sample spectral data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 200
    
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    X = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0
    
    # Add some signal correlated with target
    signal = np.sin(wavelengths / 500) * 0.2
    y = np.random.randn(n_samples) * 5 + 20
    for i in range(n_samples):
        X[i] += signal * (y[i] - 20) / 5
    
    return {
        "X": X,
        "y": y,
        "wavelengths": wavelengths,
        "sample_ids": [f"sample_{i:03d}" for i in range(n_samples)]
    }


@pytest.fixture
def sample_csv_file(sample_spectral_data, tmp_path):
    """Create a temporary CSV file with spectral data."""
    data = sample_spectral_data
    
    df = pd.DataFrame(
        data["X"],
        columns=[str(w) for w in data["wavelengths"]]
    )
    df.insert(0, "sample_id", data["sample_ids"])
    df["target"] = data["y"]
    
    csv_path = tmp_path / "test_spectra.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def engine_api():
    """Create an EngineAPI instance."""
    from spectral_predict_v2.engine.api import EngineAPI
    return EngineAPI()


@pytest.fixture
def state_store():
    """Create a StateStore instance."""
    from spectral_predict_v2.orchestration.state_store import StateStore
    return StateStore()


@pytest.fixture
def config_manager(tmp_path):
    """Create a ConfigManager with temp directories."""
    from spectral_predict_v2.orchestration.config_manager import ConfigManager
    return ConfigManager(
        presets_dir=tmp_path / "builtin",
        user_dir=tmp_path / "user"
    )
