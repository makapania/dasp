"""
Job Runner - Background task execution with cancellation support.

Runs analysis jobs in separate threads to keep UI responsive.
"""

from typing import Any, Callable, Optional
from dataclasses import dataclass
from PySide6.QtCore import QObject, QThread, Signal
import traceback


@dataclass
class JobResult:
    """Result of a completed job."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None


class WorkerThread(QThread):
    """Thread that executes a callable."""

    finished_signal = Signal(JobResult)
    progress_signal = Signal(float, str)  # progress (0-1), stage description

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False

    def run(self):
        try:
            # Pass progress callback if the function accepts it
            if "progress_callback" in self.kwargs or self._accepts_progress_callback():
                self.kwargs["progress_callback"] = self._emit_progress

            result = self.func(*self.args, **self.kwargs)
            self.finished_signal.emit(JobResult(success=True, result=result))
        except Exception as e:
            self.finished_signal.emit(JobResult(
                success=False,
                error=str(e),
                traceback=traceback.format_exc()
            ))

    def _accepts_progress_callback(self) -> bool:
        """Check if the function accepts a progress_callback parameter."""
        import inspect
        try:
            sig = inspect.signature(self.func)
            return "progress_callback" in sig.parameters
        except (ValueError, TypeError):
            return False

    def _emit_progress(self, progress: float, stage: str = ""):
        """Emit progress signal (called from worker function)."""
        if not self._cancelled:
            self.progress_signal.emit(progress, stage)

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class JobRunner(QObject):
    """
    Manages background job execution.

    Usage:
        runner = JobRunner()
        runner.job_completed.connect(handle_result)
        runner.job_progress.connect(update_progress_bar)
        runner.run(my_analysis_function, X, y, n_components=5)
    """

    job_started = Signal(str)  # job_id
    job_progress = Signal(str, float, str)  # job_id, progress, stage
    job_completed = Signal(str, JobResult)  # job_id, result
    job_cancelled = Signal(str)  # job_id

    def __init__(self):
        super().__init__()
        self._active_jobs: dict[str, WorkerThread] = {}
        self._job_counter = 0

    def run(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Run a function in the background.

        Args:
            func: The function to run
            *args: Positional arguments for func
            job_id: Optional custom job ID
            **kwargs: Keyword arguments for func

        Returns:
            job_id: The ID of the started job
        """
        if job_id is None:
            self._job_counter += 1
            job_id = f"job_{self._job_counter}"

        thread = WorkerThread(func, *args, **kwargs)

        # Connect signals
        thread.progress_signal.connect(
            lambda p, s, jid=job_id: self.job_progress.emit(jid, p, s)
        )
        thread.finished_signal.connect(
            lambda r, jid=job_id: self._on_job_finished(jid, r)
        )

        self._active_jobs[job_id] = thread
        thread.start()

        self.job_started.emit(job_id)
        return job_id

    def cancel(self, job_id: str) -> bool:
        """
        Request cancellation of a job.

        Returns True if the job was found and cancellation requested.
        """
        if job_id in self._active_jobs:
            thread = self._active_jobs[job_id]
            thread.cancel()
            self.job_cancelled.emit(job_id)
            return True
        return False

    def cancel_all(self):
        """Cancel all active jobs."""
        for job_id in list(self._active_jobs.keys()):
            self.cancel(job_id)

    def is_running(self, job_id: str) -> bool:
        """Check if a job is currently running."""
        return job_id in self._active_jobs and self._active_jobs[job_id].isRunning()

    def has_active_jobs(self) -> bool:
        """Check if any jobs are currently running."""
        return any(t.isRunning() for t in self._active_jobs.values())

    def _on_job_finished(self, job_id: str, result: JobResult):
        """Handle job completion."""
        if job_id in self._active_jobs:
            del self._active_jobs[job_id]
        self.job_completed.emit(job_id, result)

    def wait_for_all(self, timeout_ms: int = 30000):
        """Wait for all jobs to complete (useful for testing)."""
        for thread in self._active_jobs.values():
            thread.wait(timeout_ms)
