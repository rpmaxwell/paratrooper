"""Native-resolution screen capture.

grab() returns a (200, 320) uint8 palette-index frame (0 black, 1 cyan,
2 magenta, 3 white) straight from the X server. The palette has exactly
four colors, so the index is two bit tests: red >127 -> bit 1 (magenta,
white), green >127 -> bit 0 (cyan, white). Replaces the old 13 ms/frame
three-pass color match.
"""
import time

import mss
import numpy as np

from ..geometry import NATIVE_H, NATIVE_W, SCREEN_H, SCREEN_W, SCREEN_X0, SCREEN_Y0


class Screen:
    def __init__(self):
        self._sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        self._region = {"left": SCREEN_X0, "top": SCREEN_Y0, "width": SCREEN_W, "height": SCREEN_H}
        self._last = None

    def grab(self):
        """-> (t, frame, changed). `changed` is False when the frame is
        pixel-identical to the previous grab (no sprite moved since)."""
        img = self._sct.grab(self._region)
        t = time.time()
        bgra = np.frombuffer(img.raw, np.uint8).reshape(SCREEN_H, SCREEN_W, 4)[::2, ::2]
        frame = ((bgra[..., 2] > 127).astype(np.uint8) << 1) | (bgra[..., 1] > 127)
        changed = self._last is None or not np.array_equal(frame, self._last)
        self._last = frame
        return t, frame, changed


def to_rgb(frame):
    """Palette frame -> RGB (for saving/viewing)."""
    pal = np.array([[0, 0, 0], [85, 255, 255], [255, 85, 255], [255, 255, 255]], np.uint8)
    return pal[frame]
