"""
Config Manager - Preset and configuration management.

Handles loading/saving presets and user configurations.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    methods: list[str] = field(default_factory=lambda: ["raw", "snv"])
    sg_window: int = 15
    sg_polyorder: int = 2
    wavelength_min: Optional[float] = None
    wavelength_max: Optional[float] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    model_types: list[str] = field(default_factory=lambda: ["pls", "ridge"])
    tier: str = "standard"  # quick, standard, comprehensive, experimental
    use_bayesian: bool = True
    n_trials: int = 50  # For Bayesian optimization


@dataclass
class VariableSelectionConfig:
    """Variable selection configuration."""
    enabled: bool = False
    methods: list[str] = field(default_factory=list)  # uve, spa, ipls
    n_variables: list[int] = field(default_factory=lambda: [10, 20, 50])


@dataclass
class ValidationConfig:
    """Validation configuration."""
    n_folds: int = 5
    stratified: bool = True  # For classification
    random_state: Optional[int] = 42


@dataclass
class AnalysisPreset:
    """Complete analysis preset."""
    name: str
    description: str = ""
    task_type: str = "regression"  # regression, classification
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    variable_selection: VariableSelectionConfig = field(default_factory=VariableSelectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Scoring weights
    complexity_penalty: float = 0.1
    variable_penalty: float = 0.05


class ConfigManager:
    """
    Manages presets and application configuration.

    Presets are loaded from:
    1. Built-in presets (bundled with app)
    2. User presets (in user's config directory)
    """

    def __init__(self, presets_dir: Optional[str] = None, user_dir: Optional[str] = None):
        # Default to bundled presets directory
        if presets_dir is None:
            presets_dir = Path(__file__).parent.parent / "presets"
        self.presets_dir = Path(presets_dir)

        # User presets directory
        if user_dir is None:
            user_dir = Path.home() / ".spectral_predict" / "presets"
        self.user_dir = Path(user_dir)

        # Cache of loaded presets
        self._presets: dict[str, AnalysisPreset] = {}
        self._load_builtin_presets()

    def _load_builtin_presets(self):
        """Load built-in presets."""
        # Define built-in presets programmatically
        self._presets = {
            "nir_protein": AnalysisPreset(
                name="NIR Protein (Grain)",
                description="Protein prediction in wheat, corn, soy using NIR",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv", "sg1"],
                    sg_window=15,
                ),
                models=ModelConfig(
                    model_types=["pls", "ridge"],
                    tier="standard",
                    use_bayesian=True,
                ),
                variable_selection=VariableSelectionConfig(
                    enabled=True,
                    methods=["uve"],
                ),
            ),
            "nir_moisture": AnalysisPreset(
                name="NIR Moisture",
                description="Moisture prediction in grains and feeds",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv", "msc"],
                ),
                models=ModelConfig(
                    model_types=["pls"],
                    tier="quick",
                ),
            ),
            "nir_fat": AnalysisPreset(
                name="NIR Fat/Oil",
                description="Oil content prediction in seeds",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv", "sg2"],
                ),
                models=ModelConfig(
                    model_types=["pls", "elasticnet"],
                    tier="standard",
                ),
                variable_selection=VariableSelectionConfig(
                    enabled=True,
                    methods=["ipls"],
                ),
            ),
            "nir_bone": AnalysisPreset(
                name="NIR Bone",
                description="Bone mineral, collagen, carbonate prediction using NIR",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv", "sg1", "sg2"],
                ),
                models=ModelConfig(
                    model_types=["pls", "ridge"],
                    tier="standard",
                ),
                variable_selection=VariableSelectionConfig(
                    enabled=True,
                    methods=["uve"],
                ),
            ),
            "midir_bone": AnalysisPreset(
                name="Mid-IR Bone",
                description="FTIR bone composition analysis",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv", "baseline"],
                ),
                models=ModelConfig(
                    model_types=["pls"],
                    tier="standard",
                ),
            ),
            "midir_enamel": AnalysisPreset(
                name="Mid-IR Enamel",
                description="FTIR enamel composition analysis",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "msc", "baseline"],
                ),
                models=ModelConfig(
                    model_types=["pls", "ridge"],
                    tier="standard",
                ),
            ),
            "ftir_general": AnalysisPreset(
                name="FTIR General",
                description="General FTIR spectral analysis",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv", "sg1"],
                ),
                models=ModelConfig(
                    model_types=["pls", "randomforest"],
                    tier="standard",
                ),
                variable_selection=VariableSelectionConfig(
                    enabled=True,
                    methods=["uve"],
                ),
            ),
            "classification": AnalysisPreset(
                name="Classification",
                description="Spectral discrimination and classification",
                task_type="classification",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv"],
                ),
                models=ModelConfig(
                    model_types=["plsda", "randomforest", "svm"],
                    tier="standard",
                ),
                variable_selection=VariableSelectionConfig(
                    enabled=True,
                    methods=["spa"],
                ),
            ),
            "comprehensive": AnalysisPreset(
                name="Comprehensive",
                description="Full exploration with all methods",
                preprocessing=PreprocessingConfig(
                    methods=["raw", "snv", "msc", "sg1", "sg2"],
                ),
                models=ModelConfig(
                    model_types=["pls", "ridge", "lasso", "elasticnet",
                                 "randomforest", "xgboost", "lightgbm"],
                    tier="comprehensive",
                    use_bayesian=True,
                    n_trials=100,
                ),
                variable_selection=VariableSelectionConfig(
                    enabled=True,
                    methods=["uve", "spa"],
                    n_variables=[10, 20, 50, 100],
                ),
            ),
        }

        # Load YAML presets from disk if they exist
        if self.presets_dir.exists():
            self._load_presets_from_dir(self.presets_dir)

        if self.user_dir.exists():
            self._load_presets_from_dir(self.user_dir)

    def _load_presets_from_dir(self, directory: Path):
        """Load presets from YAML files in a directory."""
        for yaml_file in directory.glob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    preset = self._dict_to_preset(data)
                    # Use filename (without extension) as key
                    key = yaml_file.stem
                    self._presets[key] = preset
            except Exception as e:
                print(f"Warning: Could not load preset {yaml_file}: {e}")

    def _dict_to_preset(self, data: dict) -> AnalysisPreset:
        """Convert a dictionary to an AnalysisPreset."""
        return AnalysisPreset(
            name=data.get("name", "Unknown"),
            description=data.get("description", ""),
            task_type=data.get("task_type", "regression"),
            preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
            models=ModelConfig(**data.get("models", {})),
            variable_selection=VariableSelectionConfig(**data.get("variable_selection", {})),
            validation=ValidationConfig(**data.get("validation", {})),
            complexity_penalty=data.get("complexity_penalty", 0.1),
            variable_penalty=data.get("variable_penalty", 0.05),
        )

    def get_preset(self, name: str) -> Optional[AnalysisPreset]:
        """Get a preset by name."""
        return self._presets.get(name)

    def list_presets(self) -> list[tuple[str, str]]:
        """List all available presets as (key, display_name) tuples."""
        return [(key, preset.name) for key, preset in self._presets.items()]

    def save_preset(self, key: str, preset: AnalysisPreset):
        """Save a preset to the user directory."""
        self.user_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.user_dir / f"{key}.yaml"

        data = {
            "name": preset.name,
            "description": preset.description,
            "task_type": preset.task_type,
            "preprocessing": asdict(preset.preprocessing),
            "models": asdict(preset.models),
            "variable_selection": asdict(preset.variable_selection),
            "validation": asdict(preset.validation),
            "complexity_penalty": preset.complexity_penalty,
            "variable_penalty": preset.variable_penalty,
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        self._presets[key] = preset

    def delete_preset(self, key: str) -> bool:
        """Delete a user preset."""
        filepath = self.user_dir / f"{key}.yaml"
        if filepath.exists():
            filepath.unlink()
            if key in self._presets:
                del self._presets[key]
            return True
        return False
