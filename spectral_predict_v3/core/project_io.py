"""
Project I/O functionality for Spectral Predict v3.

Save and load complete project state to/from .sproject files.
The .sproject format is a ZIP archive containing:
- manifest.json: Version, metadata, file inventory
- data/: Spectral data (npz), samples (json), target (npz), metadata (json)
- config/: Column config, UI state, source provenance
- models/: Trained model bundles (pkl) with registry
- results/: Search results (csv)
"""

import zipfile
import json
import pickle
import io
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class ProjectJSONEncoder(json.JSONEncoder):
    """Handle numpy types and datetime in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class ColumnConfig:
    """Column configuration for the project."""
    id_column: Optional[str] = None
    target_column: Optional[str] = None
    target_type: Optional[str] = None  # 'numeric', 'categorical', None
    task_type: str = 'regression'  # 'regression', 'classification'
    column_types: Dict[str, str] = field(default_factory=dict)
    class_labels: Optional[List[str]] = None


@dataclass
class UIState:
    """UI state to restore."""
    build_settings: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    variable_selection: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_grids: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    selected_result_index: Optional[int] = None


@dataclass
class ProjectData:
    """Complete project data container."""
    # Core data
    X: np.ndarray = field(default_factory=lambda: np.array([]))
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_ids: List[str] = field(default_factory=list)
    y: Optional[np.ndarray] = None
    target_name: Optional[str] = None
    datasource: Optional[List[str]] = None
    metadata_columns: Dict[str, List[Any]] = field(default_factory=dict)

    # Configuration
    column_config: Optional[ColumnConfig] = None
    ui_state: Optional[UIState] = None

    # Provenance
    source_files: List[Dict[str, Any]] = field(default_factory=list)
    merge_strategy: str = ""
    merge_report: Dict[str, Any] = field(default_factory=dict)

    # Models
    model_bundles: List[Dict[str, Any]] = field(default_factory=list)
    model_registry: List[Dict[str, Any]] = field(default_factory=list)
    active_model_id: Optional[int] = None

    # Results
    search_results: Optional[pd.DataFrame] = None
    pareto_front: Optional[Dict[str, Any]] = None

    # Metadata
    project_name: str = "Untitled Project"
    description: str = ""
    created: Optional[str] = None
    modified: Optional[str] = None


def save_project(
    project: ProjectData,
    filepath: str,
    compress: bool = True
) -> None:
    """
    Save complete project state to .sproject file.

    Parameters
    ----------
    project : ProjectData
        Complete project data to save
    filepath : str
        Output path (will add .sproject extension if missing)
    compress : bool
        Whether to compress the ZIP archive (default True)
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() != '.sproject':
        filepath = filepath.with_suffix('.sproject')

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

    with zipfile.ZipFile(filepath, 'w', compression=compression) as zf:
        now = datetime.now().isoformat()

        # 1. Save spectral data
        with io.BytesIO() as buf:
            np.savez_compressed(buf, X=project.X, wavelengths=project.wavelengths)
            buf.seek(0)
            zf.writestr('data/spectral_data.npz', buf.read())

        # 2. Save samples
        samples_data = {
            'sample_ids': project.sample_ids,
            'datasource': project.datasource,
            'n_samples': len(project.sample_ids)
        }
        zf.writestr('data/samples.json', json.dumps(samples_data, cls=ProjectJSONEncoder, indent=2))

        # 3. Save target (if present)
        has_target = project.y is not None and len(project.y) > 0
        if has_target:
            with io.BytesIO() as buf:
                dtype_str = 'numeric' if np.issubdtype(project.y.dtype, np.number) else 'categorical'
                np.savez_compressed(buf, y=project.y, dtype=np.array([dtype_str]))
                buf.seek(0)
                zf.writestr('data/target.npz', buf.read())

        # 4. Save metadata columns
        zf.writestr('data/metadata_columns.json',
                    json.dumps(project.metadata_columns, cls=ProjectJSONEncoder, indent=2))

        # 5. Save column config
        if project.column_config:
            config_dict = {
                'id_column': project.column_config.id_column,
                'target_column': project.column_config.target_column,
                'target_type': project.column_config.target_type,
                'task_type': project.column_config.task_type,
                'column_types': project.column_config.column_types,
                'class_labels': project.column_config.class_labels
            }
            zf.writestr('config/column_config.json', json.dumps(config_dict, indent=2))

        # 6. Save UI state
        if project.ui_state:
            ui_dict = {
                'build_settings': project.ui_state.build_settings,
                'preprocessing': project.ui_state.preprocessing,
                'variable_selection': project.ui_state.variable_selection,
                'hyperparameter_grids': project.ui_state.hyperparameter_grids,
                'selected_result_index': project.ui_state.selected_result_index
            }
            zf.writestr('config/ui_state.json', json.dumps(ui_dict, cls=ProjectJSONEncoder, indent=2))

        # 7. Save source provenance
        sources_data = {
            'original_files': project.source_files,
            'merge_strategy': project.merge_strategy,
            'merge_report': project.merge_report
        }
        zf.writestr('config/sources.json', json.dumps(sources_data, cls=ProjectJSONEncoder, indent=2))

        # 8. Save models
        model_registry = []
        for idx, bundle in enumerate(project.model_bundles):
            # Write pickle
            with io.BytesIO() as buf:
                pickle.dump(bundle, buf)
                buf.seek(0)
                zf.writestr(f'models/model_{idx}.pkl', buf.read())

            # Add to registry
            model_registry.append({
                'id': idx,
                'file': f'model_{idx}.pkl',
                'name': f"{bundle.get('model_name', 'Unknown')}_{bundle.get('preprocessing', '')}",
                'model_type': bundle.get('model_name'),
                'preprocessing': bundle.get('preprocessing'),
                'task_type': bundle.get('task_type'),
                'target_name': bundle.get('target_name'),
                'metrics': bundle.get('metrics', {}),
                'created': bundle.get('created')
            })

        registry_data = {
            'models': model_registry,
            'active_model_id': project.active_model_id
        }
        zf.writestr('models/model_registry.json', json.dumps(registry_data, cls=ProjectJSONEncoder, indent=2))

        # 9. Save search results
        if project.search_results is not None:
            with io.StringIO() as buf:
                project.search_results.to_csv(buf, index=False)
                zf.writestr('results/search_results.csv', buf.getvalue())

        # 10. Save manifest
        manifest = {
            'format_version': '1.0',
            'software_version': '3.0',
            'created': project.created or now,
            'modified': now,
            'project_name': project.project_name,
            'description': project.description,
            'files': {
                'spectral_data': 'data/spectral_data.npz',
                'samples': 'data/samples.json',
                'target': 'data/target.npz' if has_target else None,
                'metadata_columns': 'data/metadata_columns.json',
                'column_config': 'config/column_config.json' if project.column_config else None,
                'ui_state': 'config/ui_state.json' if project.ui_state else None,
                'sources': 'config/sources.json',
                'model_registry': 'models/model_registry.json',
                'search_results': 'results/search_results.csv' if project.search_results is not None else None
            },
            'statistics': {
                'n_samples': project.X.shape[0] if len(project.X.shape) > 0 else 0,
                'n_wavelengths': project.X.shape[1] if len(project.X.shape) > 1 else 0,
                'n_models': len(project.model_bundles),
                'wavelength_range': [
                    float(project.wavelengths.min()) if len(project.wavelengths) > 0 else 0,
                    float(project.wavelengths.max()) if len(project.wavelengths) > 0 else 0
                ],
                'task_type': project.column_config.task_type if project.column_config else 'unknown'
            }
        }
        zf.writestr('manifest.json', json.dumps(manifest, indent=2))


def load_project(
    filepath: str,
    load_models: bool = True
) -> ProjectData:
    """
    Load project from .sproject file.

    Parameters
    ----------
    filepath : str
        Path to .sproject file
    load_models : bool
        Whether to load model bundles (set False for quick preview)

    Returns
    -------
    ProjectData
        Complete project data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Project file not found: {filepath}")

    with zipfile.ZipFile(filepath, 'r') as zf:
        # 1. Read manifest
        manifest = json.loads(zf.read('manifest.json'))

        # Version check
        format_version = manifest.get('format_version', '1.0')
        if format_version.split('.')[0] != '1':
            raise ValueError(f"Unsupported project format version: {format_version}")

        # 2. Load spectral data
        with io.BytesIO(zf.read('data/spectral_data.npz')) as buf:
            data = np.load(buf)
            X = data['X']
            wavelengths = data['wavelengths']

        # 3. Load samples
        samples_data = json.loads(zf.read('data/samples.json'))
        sample_ids = samples_data['sample_ids']
        datasource = samples_data.get('datasource')

        # 4. Load target (if present)
        y = None
        target_name = None
        if manifest['files'].get('target'):
            try:
                with io.BytesIO(zf.read('data/target.npz')) as buf:
                    target_data = np.load(buf, allow_pickle=True)
                    y = target_data['y']
            except KeyError:
                pass  # Target file not present

        # 5. Load metadata columns
        metadata_columns = json.loads(zf.read('data/metadata_columns.json'))

        # 6. Load column config
        column_config = None
        if manifest['files'].get('column_config'):
            try:
                config_dict = json.loads(zf.read('config/column_config.json'))
                column_config = ColumnConfig(**config_dict)
                target_name = column_config.target_column
            except KeyError:
                pass

        # 7. Load UI state
        ui_state = None
        if manifest['files'].get('ui_state'):
            try:
                ui_dict = json.loads(zf.read('config/ui_state.json'))
                ui_state = UIState(**ui_dict)
            except KeyError:
                pass

        # 8. Load source provenance
        source_files = []
        merge_strategy = ""
        merge_report = {}
        try:
            sources_data = json.loads(zf.read('config/sources.json'))
            source_files = sources_data.get('original_files', [])
            merge_strategy = sources_data.get('merge_strategy', '')
            merge_report = sources_data.get('merge_report', {})
        except KeyError:
            pass

        # 9. Load models
        model_bundles = []
        model_registry = []
        active_model_id = None

        if load_models:
            try:
                registry_data = json.loads(zf.read('models/model_registry.json'))
                model_registry = registry_data.get('models', [])
                active_model_id = registry_data.get('active_model_id')

                for model_info in model_registry:
                    model_file = f"models/{model_info['file']}"
                    try:
                        with io.BytesIO(zf.read(model_file)) as buf:
                            bundle = pickle.load(buf)
                            model_bundles.append(bundle)
                    except KeyError:
                        pass  # Model file not found
            except KeyError:
                pass

        # 10. Load search results
        search_results = None
        if manifest['files'].get('search_results'):
            try:
                csv_path = manifest['files']['search_results']
                with io.StringIO(zf.read(csv_path).decode('utf-8')) as buf:
                    search_results = pd.read_csv(buf)
            except KeyError:
                pass

        return ProjectData(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            y=y,
            target_name=target_name,
            datasource=datasource,
            metadata_columns=metadata_columns,
            column_config=column_config,
            ui_state=ui_state,
            source_files=source_files,
            merge_strategy=merge_strategy,
            merge_report=merge_report,
            model_bundles=model_bundles,
            model_registry=model_registry,
            active_model_id=active_model_id,
            search_results=search_results,
            project_name=manifest.get('project_name', 'Untitled'),
            description=manifest.get('description', ''),
            created=manifest.get('created'),
            modified=manifest.get('modified')
        )


def get_project_info(filepath: str) -> Dict[str, Any]:
    """
    Get project metadata without fully loading.

    Useful for file browsers and recent projects list.

    Parameters
    ----------
    filepath : str
        Path to .sproject file

    Returns
    -------
    dict
        Project metadata including name, description, statistics
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Project file not found: {filepath}")

    with zipfile.ZipFile(filepath, 'r') as zf:
        manifest = json.loads(zf.read('manifest.json'))

        return {
            'project_name': manifest.get('project_name', 'Untitled'),
            'description': manifest.get('description', ''),
            'created': manifest.get('created'),
            'modified': manifest.get('modified'),
            'format_version': manifest.get('format_version'),
            'software_version': manifest.get('software_version'),
            'statistics': manifest.get('statistics', {}),
            'filepath': str(filepath)
        }


def export_data_only(
    project: ProjectData,
    filepath: str,
    format: str = 'csv'
) -> None:
    """
    Export just the spectral data and metadata (no models).

    Parameters
    ----------
    project : ProjectData
        Project to export from
    filepath : str
        Output path
    format : str
        'csv' or 'xlsx'
    """
    filepath = Path(filepath)

    # Build DataFrame with all data
    data = {'Sample_ID': project.sample_ids}

    # Add target if present
    if project.y is not None and len(project.y) > 0:
        target_name = project.target_name or 'Target'
        data[target_name] = project.y

    # Add metadata columns
    for col_name, col_values in project.metadata_columns.items():
        data[col_name] = col_values

    # Add wavelength columns
    for i, wl in enumerate(project.wavelengths):
        data[f'{wl:.2f}'] = project.X[:, i]

    df = pd.DataFrame(data)

    if format.lower() == 'xlsx':
        if not filepath.suffix.lower() == '.xlsx':
            filepath = filepath.with_suffix('.xlsx')
        df.to_excel(filepath, index=False)
    else:
        if not filepath.suffix.lower() == '.csv':
            filepath = filepath.with_suffix('.csv')
        df.to_csv(filepath, index=False)


def import_model_to_project(
    project: ProjectData,
    model_bundle_path: str
) -> ProjectData:
    """
    Import an existing .pkl model bundle into a project.

    Parameters
    ----------
    project : ProjectData
        Existing project to add model to
    model_bundle_path : str
        Path to existing .pkl model bundle

    Returns
    -------
    ProjectData
        Updated project with new model
    """
    from .model_io import load_model

    bundle = load_model(model_bundle_path)

    # Verify compatibility (wavelength count)
    if 'wavelengths' in bundle:
        if len(bundle['wavelengths']) != len(project.wavelengths):
            raise ValueError(
                f"Wavelength mismatch: model has {len(bundle['wavelengths'])} wavelengths, "
                f"project has {len(project.wavelengths)}"
            )

    # Add to project
    project.model_bundles.append(bundle)

    # Update registry
    new_id = len(project.model_registry)
    project.model_registry.append({
        'id': new_id,
        'file': f'model_{new_id}.pkl',
        'name': f"{bundle.get('model_name', 'Unknown')}_{bundle.get('preprocessing', '')}",
        'model_type': bundle.get('model_name'),
        'preprocessing': bundle.get('preprocessing'),
        'task_type': bundle.get('task_type'),
        'target_name': bundle.get('target_name'),
        'metrics': bundle.get('metrics', {}),
        'created': bundle.get('created')
    })

    return project
