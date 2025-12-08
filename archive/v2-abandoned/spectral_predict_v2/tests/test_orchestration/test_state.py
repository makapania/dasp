"""
Tests for the Orchestration layer.

Tests cover:
- StateStore state management
- ConfigManager preset handling
- JobRunner background execution
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestStateStore:
    """Test StateStore functionality."""

    @pytest.fixture
    def state(self):
        """Create a StateStore instance."""
        from spectral_predict_v2.orchestration.state_store import StateStore
        return StateStore()

    def test_initial_state(self, state):
        """Test initial state values."""
        from spectral_predict_v2.orchestration.state_store import AppMode
        
        assert state.mode == AppMode.EXPLORE
        assert not state.has_data
        assert state.analysis.is_running == False

    def test_set_mode(self, state):
        """Test mode switching."""
        from spectral_predict_v2.orchestration.state_store import AppMode
        
        state.set_mode(AppMode.BUILD)
        assert state.mode == AppMode.BUILD
        
        state.set_mode(AppMode.PREDICT)
        assert state.mode == AppMode.PREDICT

    def test_load_data(self, state):
        """Test loading data into state."""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = np.random.randn(50)
        wavelengths = np.linspace(400, 2500, 100)
        
        state.load_data(
            X=X,
            y=y,
            wavelengths=wavelengths,
            file_path="/path/to/data.csv",
            target_column="protein",
            sample_ids=[f"sample_{i}" for i in range(50)]
        )
        
        assert state.has_data
        assert state.data.n_samples == 50
        assert state.data.n_wavelengths == 100
        assert state.data.target_column == "protein"

    def test_analysis_progress(self, state):
        """Test analysis progress tracking."""
        state.start_analysis()
        assert state.analysis.is_running
        
        state.update_progress(0.5, "Training models")
        assert state.analysis.progress == 0.5
        assert state.analysis.current_stage == "Training models"

    def test_pinned_models(self, state):
        """Test model pinning functionality."""
        # Pin first model
        result = state.toggle_pinned(0)
        assert result == True
        assert 0 in state.pinned_indices
        
        # Pin more
        state.toggle_pinned(1)
        state.toggle_pinned(2)
        state.toggle_pinned(3)
        
        assert len(state.pinned_indices) == 4
        
        # Try to pin fifth (should fail - max 4)
        result = state.toggle_pinned(4)
        assert result == False
        
        # Unpin
        result = state.toggle_pinned(0)
        assert result == False
        assert 0 not in state.pinned_indices

    def test_reset(self, state):
        """Test state reset."""
        # Load some data first
        X = np.random.randn(10, 50)
        y = np.random.randn(10)
        wavelengths = np.linspace(400, 2500, 50)
        
        state.load_data(X, y, wavelengths, "/path/data.csv", "target")
        assert state.has_data
        
        state.reset()
        assert not state.has_data


class TestConfigManager:
    """Test ConfigManager functionality."""

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create a ConfigManager with temp directories."""
        from spectral_predict_v2.orchestration.config_manager import ConfigManager
        return ConfigManager(
            presets_dir=tmp_path / "builtin",
            user_dir=tmp_path / "user"
        )

    def test_list_builtin_presets(self, config_manager):
        """Test listing built-in presets."""
        presets = config_manager.list_presets()
        
        assert len(presets) > 0
        preset_keys = [p[0] for p in presets]
        assert "nir_protein" in preset_keys
        assert "comprehensive" in preset_keys

    def test_get_preset(self, config_manager):
        """Test getting a specific preset."""
        preset = config_manager.get_preset("nir_protein")
        
        assert preset is not None
        assert preset.name == "NIR Protein (Grain)"
        assert "snv" in preset.preprocessing.methods

    def test_save_custom_preset(self, config_manager):
        """Test saving a custom preset."""
        from spectral_predict_v2.orchestration.config_manager import (
            AnalysisPreset, PreprocessingConfig, ModelConfig
        )
        
        custom = AnalysisPreset(
            name="My Custom Preset",
            description="A test preset",
            preprocessing=PreprocessingConfig(methods=["raw", "snv"]),
            models=ModelConfig(model_types=["pls", "ridge"])
        )
        
        config_manager.save_preset("my_custom", custom)
        
        # Verify it was saved
        loaded = config_manager.get_preset("my_custom")
        assert loaded is not None
        assert loaded.name == "My Custom Preset"

    def test_delete_custom_preset(self, config_manager):
        """Test deleting a custom preset."""
        from spectral_predict_v2.orchestration.config_manager import AnalysisPreset
        
        custom = AnalysisPreset(name="To Delete")
        config_manager.save_preset("to_delete", custom)
        
        result = config_manager.delete_preset("to_delete")
        assert result == True
        
        assert config_manager.get_preset("to_delete") is None


class TestJobRunner:
    """Test JobRunner functionality."""

    @pytest.fixture
    def job_runner(self):
        """Create a JobRunner instance."""
        from spectral_predict_v2.orchestration.job_runner import JobRunner
        return JobRunner()

    def test_no_active_jobs_initially(self, job_runner):
        """Test that there are no active jobs initially."""
        assert not job_runner.has_active_jobs()

    def test_job_execution(self, job_runner):
        """Test basic job execution."""
        import time
        
        result_holder = []
        
        def simple_job():
            time.sleep(0.1)
            return 42
        
        def on_complete(job_id, result):
            result_holder.append(result)
        
        job_runner.job_completed.connect(on_complete)
        
        job_id = job_runner.submit(simple_job, "test_job")
        assert job_id is not None
        
        # Wait for job to complete
        time.sleep(0.3)
        
        assert len(result_holder) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
