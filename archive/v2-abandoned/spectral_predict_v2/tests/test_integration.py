"""
Integration tests for Spectral Predict v2.

Tests end-to-end workflows:
- Load data -> Analyze -> Build model -> Predict
- Preset application
- Tools integration
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEndToEndWorkflow:
    """Test complete analysis workflows."""

    def test_load_analyze_predict_workflow(self, engine_api, sample_spectral_data, tmp_path):
        """Test the complete Explore -> Build -> Predict workflow."""
        data = sample_spectral_data
        
        # Step 1: Train a model (simulating what Build mode does)
        trained = engine_api.train_model(
            X=data["X"],
            y=data["y"],
            model_type="pls",
            preprocessing="snv",
            n_components=5
        )
        
        assert trained is not None
        assert trained.metrics["r2"] > 0  # Should have some predictive power
        
        # Step 2: Save the model
        model_path = tmp_path / "test_model.dasp"
        trained.config["y_true"] = data["y"]  # For training stats
        engine_api.save_model(trained, str(model_path))
        
        assert model_path.exists()
        
        # Step 3: Load the model
        loaded_model = engine_api.load_model(str(model_path))
        
        assert loaded_model is not None
        assert loaded_model.name == trained.name
        
        # Step 4: Make predictions on new data
        X_new = np.random.randn(10, data["X"].shape[1])
        predictions, uncertainty, ad_flags = engine_api.predict(
            X_new, loaded_model, return_uncertainty=True
        )
        
        assert len(predictions) == 10
        assert uncertainty is not None
        assert ad_flags is not None

    def test_preprocessing_pipeline(self, engine_api, sample_spectral_data):
        """Test that preprocessing produces consistent results."""
        X = sample_spectral_data["X"]
        
        # Apply SNV
        X_snv = engine_api.apply_preprocessing(X, "snv")
        
        # Verify SNV properties (zero mean, unit std per sample)
        for i in range(X_snv.shape[0]):
            assert abs(np.mean(X_snv[i])) < 1e-10
            assert abs(np.std(X_snv[i]) - 1.0) < 1e-10

    def test_multiple_model_comparison(self, engine_api, sample_spectral_data):
        """Test training multiple models for comparison."""
        data = sample_spectral_data
        
        models = {}
        for model_type in ["pls", "ridge"]:
            trained = engine_api.train_model(
                X=data["X"],
                y=data["y"],
                model_type=model_type,
                preprocessing="snv",
                n_components=5
            )
            models[model_type] = trained
        
        # Both should have reasonable R2
        for name, model in models.items():
            assert model.metrics["r2"] > 0, f"{name} should have positive R2"


class TestStateIntegration:
    """Test state management integration."""

    def test_data_loading_updates_state(self, state_store, sample_spectral_data):
        """Test that loading data updates state correctly."""
        data = sample_spectral_data
        
        state_store.load_data(
            X=data["X"],
            y=data["y"],
            wavelengths=data["wavelengths"],
            file_path="/test/path.csv",
            target_column="target",
            sample_ids=data["sample_ids"]
        )
        
        assert state_store.has_data
        assert state_store.data.n_samples == len(data["y"])
        
        # Get data summary
        summary = state_store.get_data_summary()
        assert "n_samples" in summary
        assert summary["n_samples"] == len(data["y"])

    def test_analysis_state_tracking(self, state_store, sample_spectral_data):
        """Test analysis state tracking."""
        data = sample_spectral_data
        
        state_store.load_data(
            X=data["X"],
            y=data["y"],
            wavelengths=data["wavelengths"],
            file_path="/test/path.csv",
            target_column="target"
        )
        
        # Start analysis
        state_store.start_analysis()
        assert state_store.analysis.is_running
        
        # Update progress
        state_store.update_progress(0.5, "Testing models")
        assert state_store.analysis.progress == 0.5
        
        # Complete with results
        results_df = pd.DataFrame({
            "model": ["PLS", "Ridge"],
            "preprocessing": ["SNV", "SNV"],
            "r2": [0.9, 0.85],
            "composite_score": [90, 85]
        })
        
        state_store.complete_analysis(results_df)
        
        assert not state_store.analysis.is_running
        assert state_store.analysis.n_models_evaluated == 2


class TestPresetIntegration:
    """Test preset system integration."""

    def test_preset_application(self, config_manager):
        """Test applying a preset returns correct configuration."""
        preset = config_manager.get_preset("nir_protein")
        
        assert preset is not None
        assert "snv" in preset.preprocessing.methods
        assert "pls" in preset.models.model_types

    def test_custom_preset_workflow(self, config_manager):
        """Test creating, saving, and loading custom preset."""
        from spectral_predict_v2.orchestration.config_manager import (
            AnalysisPreset, PreprocessingConfig, ModelConfig
        )
        
        # Create custom preset
        custom = AnalysisPreset(
            name="My NIR Bone Preset",
            description="Custom preset for bone analysis",
            preprocessing=PreprocessingConfig(
                methods=["raw", "snv", "sg1", "sg2"]
            ),
            models=ModelConfig(
                model_types=["pls", "ridge"],
                tier="comprehensive",
                use_bayesian=True
            )
        )
        
        # Save it
        config_manager.save_preset("my_nir_bone", custom)
        
        # Load and verify
        loaded = config_manager.get_preset("my_nir_bone")
        assert loaded is not None
        assert loaded.name == "My NIR Bone Preset"
        assert "sg2" in loaded.preprocessing.methods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
