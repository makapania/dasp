"""Test the completion chime sound."""

import pytest

pytestmark = pytest.mark.skip(reason="Audio test - plays sound")

try:
    import winsound
    import time

    def test_chime_plays():
        """Test that wind chime sound plays without error."""
        chime_notes = [
            (523, 150),   # C5
            (659, 150),   # E5
            (784, 150),   # G5
            (1047, 200),  # C6 (slightly longer for nice ending)
        ]

        for freq, duration in chime_notes:
            winsound.Beep(freq, duration)
            time.sleep(0.08)

except ImportError:
    def test_chime_plays():
        pytest.skip("winsound not available (Windows only)")
