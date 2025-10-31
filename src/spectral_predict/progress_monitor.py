"""
Progress Monitor Window for Spectral Predict

Real-time progress tracking during model search with:
- Progress bar and percentage
- Current model being tested
- Best model found so far
- Estimated time remaining (ETA)
- Cancel button
"""

import tkinter as tk
from tkinter import ttk
import time
from datetime import timedelta


class ProgressMonitor:
    """Live progress window for spectral analysis."""

    def __init__(self, parent=None, total_models=0):
        """
        Initialize progress monitor window.

        Parameters
        ----------
        parent : tk.Tk, optional
            Parent window. If None, creates standalone window.
        total_models : int
            Total number of models to test
        """
        # Create window
        if parent is None:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel(parent)

        self.root.title("Spectral Predict - Analysis Progress")
        self.root.geometry("700x450")
        self.root.resizable(False, False)

        # State variables
        self.total_models = total_models
        self.current_model = 0
        self.start_time = time.time()
        self.cancel_requested = False
        self.best_model = None
        self.stage = "initializing"

        # ETA tracking
        self.last_update_time = time.time()
        self.updates_history = []  # Store (timestamp, model_number) tuples
        self.max_history = 20  # Keep last 20 updates for ETA calculation

        self._create_ui()

    def _create_ui(self):
        """Create the progress monitor UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title
        title = ttk.Label(main_frame, text="Analysis in Progress",
                         font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # === PROGRESS SECTION ===
        progress_frame = ttk.LabelFrame(main_frame, text="Overall Progress", padding="15")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=600
        )
        self.progress_bar.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))

        # Progress percentage
        self.progress_label = ttk.Label(
            progress_frame,
            text="0%",
            font=("Arial", 12, "bold")
        )
        self.progress_label.grid(row=1, column=0, sticky=tk.W)

        # Model counter
        self.counter_label = ttk.Label(
            progress_frame,
            text=f"Model 0 of {self.total_models}",
            font=("Arial", 10)
        )
        self.counter_label.grid(row=1, column=1, sticky=tk.E)

        # Time info frame
        time_frame = ttk.Frame(progress_frame)
        time_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))

        # Elapsed time
        ttk.Label(time_frame, text="Elapsed:").grid(row=0, column=0, sticky=tk.W)
        self.elapsed_label = ttk.Label(time_frame, text="00:00:00", font=("Courier", 10))
        self.elapsed_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))

        # ETA
        ttk.Label(time_frame, text="Estimated Remaining:").grid(row=0, column=2, sticky=tk.W)
        self.eta_label = ttk.Label(time_frame, text="Calculating...", font=("Courier", 10))
        self.eta_label.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))

        # === CURRENT STATUS SECTION ===
        status_frame = ttk.LabelFrame(main_frame, text="Current Task", padding="15")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Stage label
        self.stage_label = ttk.Label(
            status_frame,
            text="Initializing analysis...",
            font=("Arial", 10, "italic"),
            foreground="blue"
        )
        self.stage_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Current model
        self.current_label = ttk.Label(
            status_frame,
            text="Preparing models...",
            font=("Arial", 11),
            wraplength=600
        )
        self.current_label.grid(row=1, column=0, sticky=tk.W)

        # === BEST MODEL SECTION ===
        self.best_frame = ttk.LabelFrame(main_frame, text="Best Model So Far", padding="15")
        self.best_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Best model display
        self.best_label = ttk.Label(
            self.best_frame,
            text="No models tested yet",
            font=("Arial", 10),
            foreground="gray",
            wraplength=600,
            justify=tk.LEFT
        )
        self.best_label.grid(row=0, column=0, sticky=tk.W)

        # === CONTROL BUTTONS ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)

        # Cancel button
        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel Analysis",
            command=self._on_cancel
        )
        self.cancel_button.grid(row=0, column=0, padx=5)

        # Minimize button
        self.minimize_button = ttk.Button(
            button_frame,
            text="Minimize",
            command=self._on_minimize
        )
        self.minimize_button.grid(row=0, column=1, padx=5)

        # Status bar at bottom
        self.status_bar = ttk.Label(
            main_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Start updating elapsed time
        self._update_elapsed_time()

    def update(self, progress_data):
        """
        Update progress monitor with new data.

        Parameters
        ----------
        progress_data : dict
            Dictionary with keys:
            - stage: 'region_analysis' or 'model_testing'
            - message: Status message
            - current: Current model number
            - total: Total models to test
            - best_model: Best model dict (optional)
        """
        # Update stage
        stage = progress_data.get('stage', 'model_testing')
        self.stage = stage

        # Update total if provided
        total = progress_data.get('total', self.total_models)
        if total > 0 and total != self.total_models:
            self.total_models = total

        # Update current model number
        current = progress_data.get('current', 0)
        self.current_model = current

        # Track update time for ETA calculation
        current_time = time.time()
        self.updates_history.append((current_time, current))
        if len(self.updates_history) > self.max_history:
            self.updates_history.pop(0)

        # Update progress bar
        if self.total_models > 0:
            progress_percent = (current / self.total_models) * 100
            self.progress_bar['value'] = progress_percent
            self.progress_label.config(text=f"{progress_percent:.1f}%")
            self.counter_label.config(text=f"Model {current} of {self.total_models}")

        # Update stage label
        if stage == 'region_analysis':
            self.stage_label.config(text="Stage: Analyzing spectral regions", foreground="purple")
        elif stage == 'model_testing':
            self.stage_label.config(text="Stage: Testing model configurations", foreground="blue")

        # Update current task message
        message = progress_data.get('message', '')
        if message:
            self.current_label.config(text=message)

        # Update best model
        best_model = progress_data.get('best_model')
        if best_model is not None:
            self.best_model = best_model
            self._update_best_model_display()

        # Update ETA
        self._update_eta()

        # Update status bar
        self.status_bar.config(text=f"Running... {current}/{self.total_models} models tested")

        # Force UI update
        self.root.update_idletasks()

    def _update_best_model_display(self):
        """Update the best model display section."""
        if self.best_model is None:
            self.best_label.config(
                text="No models tested yet",
                foreground="gray"
            )
            # Reset header text when nothing has been tested yet
            if hasattr(self, 'best_frame') and self.best_frame.winfo_exists():
                self.best_frame.config(text="Best Model So Far")
            return

        # Extract model info (keys are capitalized in result dict)
        model_name = self.best_model.get('Model', self.best_model.get('model', 'Unknown'))
        preprocessing = self.best_model.get('Preprocess', self.best_model.get('preprocessing', 'Unknown'))
        n_vars = self.best_model.get('n_vars', 0)
        subset = self.best_model.get('SubsetTag', self.best_model.get('subset', 'full'))

        # Format performance metrics
        task_type = self.best_model.get('Task', self.best_model.get('task_type', 'regression'))

        if task_type == 'regression':
            rmse = self.best_model.get('RMSE', 0)
            r2 = self.best_model.get('R2', 0)
            metrics_text = f"RMSE: {rmse:.4f} | R²: {r2:.4f}"
        else:
            roc_auc = self.best_model.get('ROC_AUC', 0)
            accuracy = self.best_model.get('Accuracy', 0)
            metrics_text = f"ROC AUC: {roc_auc:.4f} | Accuracy: {accuracy:.4f}"

        # Update header to include variable count for quick glance
        if hasattr(self, 'best_frame') and self.best_frame.winfo_exists():
            try:
                self.best_frame.config(text=f"Best Model So Far — {n_vars} variables")
            except Exception:
                # Fallback in case of any UI issues
                self.best_frame.config(text="Best Model So Far")

        # Build display text
        display_text = (
            f"Model: {model_name}\n"
            f"Preprocessing: {preprocessing}\n"
            f"Variables: {n_vars} ({subset})\n"
            f"Performance: {metrics_text}"
        )

        self.best_label.config(
            text=display_text,
            foreground="darkgreen",
            font=("Arial", 10, "bold")
        )

    def _update_elapsed_time(self):
        """Update elapsed time display (runs every second)."""
        if hasattr(self, 'root') and self.root.winfo_exists():
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            self.elapsed_label.config(text=elapsed_str)

            # Schedule next update in 1 second
            self.root.after(1000, self._update_elapsed_time)

    def _update_eta(self):
        """Calculate and update ETA based on recent progress."""
        if len(self.updates_history) < 2 or self.current_model == 0:
            self.eta_label.config(text="Calculating...")
            return

        # Calculate models per second from recent history
        time_span = self.updates_history[-1][0] - self.updates_history[0][0]
        models_span = self.updates_history[-1][1] - self.updates_history[0][1]

        if time_span > 0 and models_span > 0:
            models_per_second = models_span / time_span
            remaining_models = self.total_models - self.current_model
            eta_seconds = remaining_models / models_per_second

            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{int(eta_seconds)}s"
            elif eta_seconds < 3600:
                minutes = int(eta_seconds / 60)
                seconds = int(eta_seconds % 60)
                eta_str = f"{minutes}m {seconds}s"
            else:
                hours = int(eta_seconds / 3600)
                minutes = int((eta_seconds % 3600) / 60)
                eta_str = f"{hours}h {minutes}m"

            self.eta_label.config(text=eta_str)
        else:
            self.eta_label.config(text="Calculating...")

    def _on_cancel(self):
        """Handle cancel button click."""
        self.cancel_requested = True
        self.cancel_button.config(state='disabled', text="Cancelling...")
        self.status_bar.config(text="Cancel requested - finishing current model...")

    def _on_minimize(self):
        """Minimize the window."""
        self.root.iconify()

    def is_cancelled(self):
        """Check if user requested cancellation."""
        return self.cancel_requested

    def complete(self, success=True, message=""):
        """
        Mark analysis as complete.

        Parameters
        ----------
        success : bool
            Whether analysis completed successfully
        message : str
            Completion message
        """
        if success:
            self.progress_bar['value'] = 100
            self.progress_label.config(text="100%")
            self.stage_label.config(text="Analysis Complete!", foreground="green")
            self.current_label.config(text=message or "All models tested successfully")
            self.status_bar.config(text="Complete - Results saved")
            self.cancel_button.config(text="Close", state='normal', command=self.close)
        else:
            self.stage_label.config(text="Analysis Failed", foreground="red")
            self.current_label.config(text=message or "An error occurred")
            self.status_bar.config(text="Failed")
            self.cancel_button.config(text="Close", state='normal', command=self.close)

    def close(self):
        """Close the progress monitor window."""
        self.root.destroy()

    def show(self):
        """Show the progress monitor window."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def run(self):
        """Run the progress monitor window (for standalone use)."""
        self.root.mainloop()


# Standalone testing
if __name__ == "__main__":
    import random

    # Demo progress monitor
    monitor = ProgressMonitor(total_models=100)

    def simulate_progress():
        """Simulate analysis progress."""
        for i in range(100):
            if monitor.is_cancelled():
                monitor.complete(success=False, message="Analysis cancelled by user")
                return

            # Simulate progress update
            models = ["PLS", "RandomForest", "PLS-DA"]
            preprocs = ["raw", "SNV", "MSC", "d1", "d2"]

            progress_data = {
                'stage': 'model_testing' if i > 5 else 'region_analysis',
                'message': f"Testing {random.choice(models)} with {random.choice(preprocs)} preprocessing",
                'current': i + 1,
                'total': 100,
                'best_model': {
                    'model': random.choice(models),
                    'preprocessing': random.choice(preprocs),
                    'n_vars': random.randint(10, 500),
                    'subset': random.choice(['full', 'top100', 'region1']),
                    'task_type': 'regression',
                    'RMSE': random.uniform(0.1, 0.5),
                    'R2': random.uniform(0.7, 0.95)
                }
            }

            monitor.update(progress_data)
            monitor.root.update()
            time.sleep(0.1)  # Simulate processing time

        monitor.complete(success=True, message="All 100 models tested successfully!")

    # Start simulation after a short delay
    monitor.root.after(500, simulate_progress)
    monitor.run()
