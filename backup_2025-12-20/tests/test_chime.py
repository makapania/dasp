"""Test the completion chime sound."""

try:
    import winsound
    import time

    print("Playing wind chime sound...")
    print("(This is what will play when your analysis completes)")

    # Wind chime-like sequence using major pentatonic scale
    chime_notes = [
        (523, 150),   # C5
        (659, 150),   # E5
        (784, 150),   # G5
        (1047, 200),  # C6 (slightly longer for nice ending)
    ]

    for freq, duration in chime_notes:
        winsound.Beep(freq, duration)
        time.sleep(0.08)  # Small gap between notes

    print("[SUCCESS] Chime played successfully!")
    print("\nThis pleasant wind chime will play automatically when:")
    print("  - Analysis completes")
    print("  - Results are populated")

except ImportError:
    print("winsound not available (Windows only)")
except Exception as e:
    print(f"Error playing chime: {e}")
