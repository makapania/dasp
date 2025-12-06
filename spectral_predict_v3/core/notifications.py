"""
Sound notifications for Spectral Predict v3.

Provides cross-platform sound notifications with user settings:
- Windows: winsound beep
- macOS/Linux: print bell character or fallback
- User setting to enable/disable
"""

import sys
import platform
from typing import Optional


class NotificationManager:
    """
    Manages sound notifications with user preferences.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize notification manager.

        Parameters
        ----------
        enabled : bool
            Whether notifications are enabled by default
        """
        self.enabled = enabled
        self.platform = platform.system()

    def enable(self):
        """Enable sound notifications."""
        self.enabled = True

    def disable(self):
        """Disable sound notifications."""
        self.enabled = False

    def toggle(self) -> bool:
        """
        Toggle notification state.

        Returns
        -------
        bool
            New enabled state
        """
        self.enabled = not self.enabled
        return self.enabled

    def beep(self, frequency: int = 1000, duration: int = 200):
        """
        Play a beep sound.

        Parameters
        ----------
        frequency : int
            Frequency in Hz (Windows only)
        duration : int
            Duration in milliseconds (Windows only)
        """
        if not self.enabled:
            return

        try:
            if self.platform == 'Windows':
                self._beep_windows(frequency, duration)
            elif self.platform == 'Darwin':  # macOS
                self._beep_macos()
            else:  # Linux and others
                self._beep_linux()
        except Exception as e:
            # Silently fail if sound doesn't work
            print(f"Sound notification failed: {e}")

    def _beep_windows(self, frequency: int, duration: int):
        """Windows beep using winsound."""
        try:
            import winsound
            winsound.Beep(frequency, duration)
        except ImportError:
            # Fallback to bell character
            print('\a', end='', flush=True)

    def _beep_macos(self):
        """macOS beep using afplay or bell character."""
        try:
            import subprocess
            # Try to play system sound
            subprocess.run(
                ['afplay', '/System/Library/Sounds/Glass.aiff'],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            # Fallback to bell character
            print('\a', end='', flush=True)

    def _beep_linux(self):
        """Linux beep using bell character or paplay."""
        # Try bell character first
        print('\a', end='', flush=True)

        # Optionally try paplay for pulseaudio
        try:
            import subprocess
            subprocess.run(
                ['paplay', '/usr/share/sounds/freedesktop/stereo/complete.oga'],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1
            )
        except Exception:
            pass

    def notify_completion(self, task_name: str = "Analysis"):
        """
        Notify task completion with sound and message.

        Parameters
        ----------
        task_name : str
            Name of completed task
        """
        if self.enabled:
            print(f"\n{task_name} complete!")
            self.beep()

    def notify_error(self, error_msg: str = "Error occurred"):
        """
        Notify error with distinctive sound.

        Parameters
        ----------
        error_msg : str
            Error message
        """
        if self.enabled:
            print(f"\n{error_msg}")
            # Double beep for errors
            self.beep(frequency=800, duration=100)
            import time
            time.sleep(0.1)
            self.beep(frequency=600, duration=100)

    def notify_warning(self, warning_msg: str = "Warning"):
        """
        Notify warning with sound.

        Parameters
        ----------
        warning_msg : str
            Warning message
        """
        if self.enabled:
            print(f"\n{warning_msg}")
            self.beep(frequency=900, duration=150)


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """
    Get the global notification manager instance.

    Returns
    -------
    NotificationManager
        Global notification manager
    """
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager(enabled=True)
    return _notification_manager


def enable_notifications():
    """Enable sound notifications globally."""
    get_notification_manager().enable()


def disable_notifications():
    """Disable sound notifications globally."""
    get_notification_manager().disable()


def toggle_notifications() -> bool:
    """
    Toggle sound notifications globally.

    Returns
    -------
    bool
        New enabled state
    """
    return get_notification_manager().toggle()


def notify_completion(task_name: str = "Analysis"):
    """
    Notify task completion.

    Parameters
    ----------
    task_name : str
        Name of completed task
    """
    get_notification_manager().notify_completion(task_name)


def notify_error(error_msg: str = "Error occurred"):
    """
    Notify error.

    Parameters
    ----------
    error_msg : str
        Error message
    """
    get_notification_manager().notify_error(error_msg)


def notify_warning(warning_msg: str = "Warning"):
    """
    Notify warning.

    Parameters
    ----------
    warning_msg : str
        Warning message
    """
    get_notification_manager().notify_warning(warning_msg)


def beep(frequency: int = 1000, duration: int = 200):
    """
    Play a beep sound.

    Parameters
    ----------
    frequency : int
        Frequency in Hz (Windows only)
    duration : int
        Duration in milliseconds (Windows only)
    """
    get_notification_manager().beep(frequency, duration)


# Example usage
if __name__ == "__main__":
    print("Testing notification system...")

    # Test basic beep
    print("\n1. Basic beep:")
    beep()

    import time
    time.sleep(1)

    # Test completion notification
    print("\n2. Completion notification:")
    notify_completion("Test Task")

    time.sleep(1)

    # Test warning
    print("\n3. Warning notification:")
    notify_warning("This is a warning")

    time.sleep(1)

    # Test error
    print("\n4. Error notification:")
    notify_error("This is an error")

    time.sleep(1)

    # Test disable/enable
    print("\n5. Testing disable/enable:")
    disable_notifications()
    print("  Notifications disabled (should be silent):")
    beep()

    time.sleep(0.5)

    enable_notifications()
    print("  Notifications enabled (should beep):")
    beep()

    print("\nNotification test complete!")
