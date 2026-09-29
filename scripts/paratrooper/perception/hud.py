"""Score and game-over, native resolution (ported from read_score.py: the
digit templates and the game-over text band, halved; the score row starts
on an aligned row, so the templates downsample exactly)."""
import pathlib

import numpy as np
from PIL import Image

from ..geometry import MAGENTA, WHITE

_TPL_DIR = pathlib.Path(__file__).resolve().parents[2] / "templates"
_TEMPLATES = {d: (np.array(Image.open(_TPL_DIR / f"digit_{d}.png").convert("L")) > 128)[0::2, 0::2]
              for d in "0123456789"}
SCORE_Y0, SCORE_Y1 = 191, 200  # native rows (screen 583..600)
RIGHT_CELL_X0, CELL_W, MAX_DIGITS = 88, 8, 6
DONE_Y0, DONE_Y1, DONE_WHITE = 95, 135, 250


def read_score(frame):
    digits = []
    for k in range(MAX_DIGITS - 1, -1, -1):
        x0 = RIGHT_CELL_X0 - CELL_W * k
        m = frame[SCORE_Y0:SCORE_Y1, x0:x0 + CELL_W] == MAGENTA
        if m.sum() < 3:
            continue
        digits.append(max(_TEMPLATES, key=lambda d: (m == _TEMPLATES[d]).sum()))
    return int("".join(digits)) if digits else 0


def is_done(frame):
    return int((frame[DONE_Y0:DONE_Y1] == WHITE).sum()) > DONE_WHITE
