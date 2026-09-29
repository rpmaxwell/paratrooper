#!/usr/bin/env python3
"""Stage 3: read the on-screen SCORE via fixed-width template matching
(no OCR), and detect the game-over/"done" state.

Geometry and templates were determined by inspecting real captured
frames (see scripts/templates/*.png, one clean glyph per digit taken
directly from in-game screenshots). All coordinates are in the
1024x768 Xvfb screenshot's pixel space.
"""
import pathlib

import numpy as np
from PIL import Image

MAGENTA = np.array([255, 85, 255])
WHITE = np.array([255, 255, 255])

# Score row: y band containing the SCORE:/HI-SCORE: line.
SCORE_Y0, SCORE_Y1 = 583, 600

# Digit cells are a fixed-width (16px), right-aligned grid. Cell 0 is
# the units place; the field's right edge (rightmost cell's left x)
# was found empirically at x=368 in the 1024x768 frame.
CELL_W = 16
RIGHTMOST_CELL_X0 = 368
MAX_DIGITS = 6  # generous headroom; PRD only needs the actual score

# Title/instructions/game-over screen shows fixed white text
# ("PRESS 'I' FOR INSTRUCTIONS" etc.) in this band; it's absent during
# active gameplay. >1000 white pixels here reliably means "not playing".
DONE_Y0, DONE_Y1 = 390, 470
DONE_WHITE_THRESHOLD = 1000

# A cell with fewer than this many "on" pixels is considered blank
# (no digit drawn there -- true for unused leading places, and for the
# whole field when score == 0).
BLANK_PIXEL_THRESHOLD = 10

_TEMPLATE_DIR = pathlib.Path(__file__).parent / "templates"
_TEMPLATES = {}
for _digit in "0123456789":
    _im = np.array(Image.open(_TEMPLATE_DIR / f"digit_{_digit}.png").convert("L")) > 128
    _TEMPLATES[_digit] = _im


def _cell_mask(frame: np.ndarray, x0: int) -> np.ndarray:
    band = frame[SCORE_Y0:SCORE_Y1, x0 : x0 + CELL_W]
    return np.all(band == MAGENTA, axis=-1)


def _match_digit(mask: np.ndarray) -> str:
    best_digit, best_score = None, -1
    for digit, template in _TEMPLATES.items():
        agree = (mask == template).sum()
        if agree > best_score:
            best_score, best_digit = agree, digit
    return best_digit


def read_score(frame: np.ndarray) -> int:
    """frame: HxWx3 RGB array (e.g. from mss). Returns the integer score."""
    digits = []
    for k in range(MAX_DIGITS - 1, -1, -1):
        x0 = RIGHTMOST_CELL_X0 - CELL_W * k
        mask = _cell_mask(frame, x0)
        if mask.sum() < BLANK_PIXEL_THRESHOLD:
            if digits:
                # blank cell after digits have started is unexpected; keep
                # scanning in case of a rendering glitch, don't break early.
                continue
            continue
        digits.append(_match_digit(mask))
    return int("".join(digits)) if digits else 0


def is_done(frame: np.ndarray) -> bool:
    band = frame[DONE_Y0:DONE_Y1, :]
    white_count = np.all(band == WHITE, axis=-1).sum()
    return white_count > DONE_WHITE_THRESHOLD


if __name__ == "__main__":
    import sys

    import mss

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        img = sct.grab(monitor)
        frame = np.array(img)[:, :, :3][:, :, ::-1]  # BGRA -> RGB

    score = read_score(frame)
    done = is_done(frame)
    print(f"score={score} done={done}")
