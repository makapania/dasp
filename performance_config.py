#!/usr/bin/env python3
"""
Performance Configuration with User Preferences

Allows users to control performance settings even if they have powerful hardware.
Users can opt for lighter resource usage to multitask while analyses run.

Usage:
    from performance_config import PerformanceConfig

    # Load user preferences
    config = PerformanceConfig.from_user_preferences()

    # Or create with specific settings
    config = PerformanceConfig(
        mode='balanced',  # 'power', 'balanced', or 'light'
        max_cpu_percent=50,  # Use max 50% of CPU
        use_gpu=False  # Disable GPU even if available
    )
"""

import json
from pathlib import Path
from hardware_detection import detect_hardware


class PerformanceConfig:
    """
    Performance configuration with user preferences.

    Balances auto-detection with user choice.
    """

    MODES = {
        'power': {
            'name': 'Power Mode',
            'description': 'Maximum performance (uses all resources)',
            'cpu_percent': 100,
            'use_all_cores': True,
            'use_gpu': True,
            'parallel_grid': True,
        },
        'balanced': {
            'name': 'Balanced Mode',
            'description': 'Good performance, leaves resources for multitasking',
            'cpu_percent': 60,
            'use_all_cores': False,  # Leave 40% free
            'use_gpu': True,
            'parallel_grid': True,
        },
        'light': {
            'name': 'Light Mode',
            'description': 'Minimal resource usage, slower but unobtrusive',
            'cpu_percent': 30,
            'use_all_cores': False,
            'use_gpu': False,  # Disable GPU (leaves it for graphics)
            'parallel_grid': False,  # Sequential (less memory)
        }
    }

    def __init__(self, mode='auto', max_cpu_percent=None, use_gpu=None,
                 parallel_grid=None, n_workers=None):
        """
        Create performance configuration.

        Parameters
        ----------
        mode : str
            'auto', 'power', 'balanced', or 'light'
        max_cpu_percent : int, optional
            Max CPU usage (0-100). Overrides mode setting.
        use_gpu : bool, optional
            Use GPU if available. Overrides mode setting.
        parallel_grid : bool, optional
            Parallelize grid search. Overrides mode setting.
        n_workers : int, optional
            Number of workers. If None, calculated from max_cpu_percent.
        """
        self.mode = mode

        # Auto-detect hardware
        self.hw_detected = detect_hardware(verbose=False)

        # Determine effective settings
        if mode == 'auto':
            # Use hardware detection tier
            self._apply_auto_mode()
        elif mode in self.MODES:
            # Use predefined mode
            self._apply_mode(mode)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from {list(self.MODES.keys())} or 'auto'")

        # Apply user overrides
        if max_cpu_percent is not None:
            self.max_cpu_percent = max_cpu_percent
        if use_gpu is not None:
            self.use_gpu = use_gpu
        if parallel_grid is not None:
            self.parallel_grid = parallel_grid

        # Calculate workers from CPU percent if not specified
        if n_workers is None:
            self.n_workers = self._calculate_workers()
        else:
            self.n_workers = n_workers

    def _apply_auto_mode(self):
        """Apply settings based on hardware detection."""
        tier = self.hw_detected['tier']

        if tier == 3:
            # Powerful hardware - default to balanced (not power!)
            # User likely wants to multitask
            self._apply_mode('balanced')
        elif tier == 2:
            self._apply_mode('balanced')
        else:
            self._apply_mode('light')

    def _apply_mode(self, mode):
        """Apply predefined mode settings."""
        settings = self.MODES[mode]
        self.mode_name = settings['name']
        self.description = settings['description']
        self.max_cpu_percent = settings['cpu_percent']
        self.use_gpu = settings['use_gpu'] and self.hw_detected['gpu_available']
        self.parallel_grid = settings['parallel_grid']

    def _calculate_workers(self):
        """Calculate number of workers from CPU percentage."""
        n_cores = self.hw_detected['n_cores']

        if self.max_cpu_percent >= 90:
            # Use all cores (leave 1 for OS)
            return max(1, n_cores - 1)
        else:
            # Use proportion of cores based on CPU percent
            workers = max(1, int(n_cores * self.max_cpu_percent / 100))
            # Always leave at least 1 core free for OS/other tasks
            return min(workers, n_cores - 1)

    @classmethod
    def from_user_preferences(cls, config_file='.dasp_performance.json'):
        """
        Load from saved user preferences.

        Parameters
        ----------
        config_file : str
            Path to config file (in user's home directory)

        Returns
        -------
        config : PerformanceConfig
        """
        config_path = Path.home() / config_file

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    prefs = json.load(f)

                return cls(
                    mode=prefs.get('mode', 'auto'),
                    max_cpu_percent=prefs.get('max_cpu_percent'),
                    use_gpu=prefs.get('use_gpu'),
                    parallel_grid=prefs.get('parallel_grid'),
                    n_workers=prefs.get('n_workers')
                )
            except Exception as e:
                print(f"Warning: Could not load preferences from {config_path}: {e}")
                print("Using auto-detection")
                return cls(mode='auto')
        else:
            # No saved preferences - use auto
            return cls(mode='auto')

    def save_preferences(self, config_file='.dasp_performance.json'):
        """
        Save current settings as user preferences.

        Parameters
        ----------
        config_file : str
            Path to config file (in user's home directory)
        """
        config_path = Path.home() / config_file

        prefs = {
            'mode': self.mode,
            'max_cpu_percent': self.max_cpu_percent,
            'use_gpu': self.use_gpu,
            'parallel_grid': self.parallel_grid,
            'n_workers': self.n_workers
        }

        try:
            with open(config_path, 'w') as f:
                json.dump(prefs, f, indent=2)
            print(f"✓ Preferences saved to {config_path}")
        except Exception as e:
            print(f"Warning: Could not save preferences: {e}")

    def get_model_params(self, model_name, base_params=None):
        """
        Get model parameters respecting user preferences.

        Parameters
        ----------
        model_name : str
            Model type: 'XGBoost', 'LightGBM', etc.
        base_params : dict, optional
            Base parameters to merge

        Returns
        -------
        params : dict
            Optimized parameters
        """
        if base_params is None:
            base_params = {}

        params = base_params.copy()

        # XGBoost
        if model_name == 'XGBoost':
            if self.use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
            else:
                params['tree_method'] = 'hist'

            # Respect n_workers for parallelism
            params['n_jobs'] = self.n_workers

        # LightGBM
        elif model_name == 'LightGBM':
            if self.use_gpu:
                params['device'] = 'gpu'
            else:
                params['device'] = 'cpu'

            params['n_jobs'] = self.n_workers

        # CatBoost
        elif model_name == 'CatBoost':
            if self.use_gpu:
                params['task_type'] = 'GPU'
            else:
                params['task_type'] = 'CPU'

            params['thread_count'] = self.n_workers

        # Sklearn models
        else:
            if 'n_jobs' not in params:
                params['n_jobs'] = self.n_workers

        return params

    def print_summary(self):
        """Print current configuration summary."""
        print("\n" + "=" * 70)
        print("PERFORMANCE CONFIGURATION")
        print("=" * 70)
        print(f"Mode: {self.mode_name}")
        print(f"Description: {self.description}")
        print()
        print("Settings:")
        print(f"  CPU Usage: {self.max_cpu_percent}% ({self.n_workers}/{self.hw_detected['n_cores']} cores)")
        print(f"  GPU: {'✓ Enabled' if self.use_gpu else '✗ Disabled'}")
        if self.hw_detected['gpu_available'] and not self.use_gpu:
            print(f"       (GPU available but not used - leaves GPU free for other tasks)")
        print(f"  Parallel Grid Search: {'✓ Enabled' if self.parallel_grid else '✗ Disabled (sequential)'}")
        print()
        print("Hardware Detected:")
        print(f"  CPU: {self.hw_detected['n_cores']} cores")
        print(f"  RAM: {self.hw_detected['memory_gb']:.1f} GB")
        print(f"  GPU: {'✓ Available' if self.hw_detected['gpu_available'] else '✗ Not available'}")
        if self.hw_detected['gpu_type']:
            print(f"       Type: {self.hw_detected['gpu_type']}")
        print("=" * 70 + "\n")

    def __repr__(self):
        return (f"PerformanceConfig(mode={self.mode}, cpu={self.max_cpu_percent}%, "
                f"workers={self.n_workers}, gpu={self.use_gpu})")


# Example GUI integration
class PerformanceSettingsDialog:
    """
    Example GUI for performance settings.

    This shows how to integrate into the Tkinter GUI.
    """

    @staticmethod
    def get_settings_ui_elements():
        """
        Returns UI element configuration for GUI.

        Returns
        -------
        elements : dict
            GUI element specifications
        """
        return {
            'mode': {
                'type': 'radio',
                'label': 'Performance Mode',
                'options': [
                    ('auto', 'Auto (Recommended)', 'Automatically detect and use optimal settings'),
                    ('power', 'Power Mode', 'Maximum performance - uses all CPU and GPU'),
                    ('balanced', 'Balanced Mode', 'Good performance, leaves ~40% CPU free for multitasking'),
                    ('light', 'Light Mode', 'Minimal impact - slower but lets you work normally')
                ],
                'default': 'balanced'  # Default to balanced, not power!
            },
            'advanced': {
                'type': 'expander',
                'label': 'Advanced Settings',
                'elements': {
                    'max_cpu_percent': {
                        'type': 'slider',
                        'label': 'Maximum CPU Usage',
                        'range': (10, 100),
                        'default': 60,
                        'suffix': '%'
                    },
                    'use_gpu': {
                        'type': 'checkbox',
                        'label': 'Use GPU (if available)',
                        'default': True,
                        'tooltip': 'Disable to leave GPU free for graphics/other tasks'
                    },
                    'parallel_grid': {
                        'type': 'checkbox',
                        'label': 'Parallel grid search',
                        'default': True,
                        'tooltip': 'Disable to reduce memory usage'
                    }
                }
            },
            'info': {
                'type': 'label',
                'text': 'Tip: Use Balanced or Light mode to multitask while analyses run'
            }
        }


# Example usage
if __name__ == '__main__':
    print("Example 1: Auto-detection (default to balanced)")
    config = PerformanceConfig(mode='auto')
    config.print_summary()

    print("\n" + "=" * 70)
    print("Example 2: User wants multitasking (even with powerful PC)")
    config = PerformanceConfig(
        mode='balanced',
        max_cpu_percent=50,  # Use only half CPU
        use_gpu=False  # Leave GPU free
    )
    config.print_summary()

    print("\n" + "=" * 70)
    print("Example 3: Custom settings")
    config = PerformanceConfig(
        mode='balanced',
        n_workers=4,  # Explicitly use 4 workers
        use_gpu=True
    )
    config.print_summary()

    # Save preferences
    # config.save_preferences()

    # Load saved preferences
    # config = PerformanceConfig.from_user_preferences()
