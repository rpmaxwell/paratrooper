"""Sprite detection on a native (200x320) palette frame.

One 8-connected labeling pass per color (scipy.ndimage), then exact
shape lookups. Every size below is the measured game-coordinate sprite
halved (see bomb_model / trooper_model / heli_model docstrings for the
measurements). Outputs are in GAME coordinates (geometry.to_game) so the
validated physics constants apply unchanged.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..geometry import CYAN, WHITE

_EIGHT = np.ones((3, 3), bool)
# a bomb: 4x4 native disc, corners missing (12 px)
_DISC = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], bool)


def _stacked_bombs(blob):
    """Two bombs in one column merge into a 4-wide blob 5..8 rows tall:
    round-4 bombers drop pairs 2-4 ticks apart, and bombs keep the plane's
    x speed, so a pair falls exactly in line. The bomb drawn last blanks its
    whole 4x4 cell, so an overlapped one shows only its uncovered rows.
    -> row offset of the lower disc if `blob` (bool, h x 4) is two stacked
    discs (either drawn on top, or touching), else None."""
    h = blob.shape[0]
    if blob.shape[1] != 4 or not 5 <= h <= 8:
        return None
    dy = h - 4
    upper_on_top = np.zeros((h, 4), bool)
    upper_on_top[dy:] = _DISC
    upper_on_top[:4] = _DISC
    lower_on_top = np.zeros((h, 4), bool)
    lower_on_top[:4] = _DISC
    lower_on_top[dy:] = _DISC
    union = upper_on_top | lower_on_top
    return dy if any(np.array_equal(blob, m) for m in (upper_on_top, lower_on_top, union)) else None
SKY_ROWS = 186          # native rows above the ground line
TURRET_X0, TURRET_X1 = 140, 180  # native columns of the turret/barrel


@dataclass
class Sprites:
    planes: list = field(default_factory=list)       # (x0, x1) any visible part
    planes_full: list = field(default_factory=list)  # x0 of fully visible planes
    heli_any: bool = False                           # any helicopter (incl. partial)
    helis: list = field(default_factory=list)        # (skid_x0, skid_y0), fully visible
    bombs: list = field(default_factory=list)        # (x0, y0)
    bodies: list = field(default_factory=list)       # airborne trooper body top-left (x, y)
    grounded: list = field(default_factory=list)     # bodies standing on the ground / a stack
    canopies: list = field(default_factory=list)     # implied body (x, y) under a canopy
    landed: list = field(default_factory=list)       # (x, head_y) of landed troopers' heads
    dots: list = field(default_factory=list)         # 1-native-px white dots (bullets), game coords


def _components(mask):
    lab, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return []
    sl = ndimage.find_objects(lab)
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    return [(s[1].start, s[0].start, s[1].stop - s[1].start, s[0].stop - s[0].start, int(counts[i + 1]))
            for i, s in enumerate(sl)]  # (x0, y0, w, h, n) native


def detect(frame):
    out = Sprites()
    cyan = _components(frame[:SKY_ROWS] == CYAN)
    plane_parts = [(x0, x0 + w - 1) for x0, y0, w, h, n in cyan if y0 <= 4]
    for x0, y0, w, h, n in cyan:
        gx, gy = 2 * x0, 2 * y0 + 1
        if h >= 6 and w >= 12 and y0 <= 2:
            out.planes.append((gx, gx + 2 * w - 1))
            if w == 24:
                out.planes_full.append(gx)
        elif h <= 2 and w >= 12 and 6 <= y0 <= 49 and not any(
                a <= x0 + w - 1 + 4 and x0 - 4 <= b for a, b in plane_parts):
            out.heli_any = True
            if w == 16 and h == 2:
                out.helis.append((gx, gy))
        elif (w, h, n) == (4, 6, 14):
            if not (TURRET_X0 <= x0 <= TURRET_X1 and y0 > 125):
                # a body whose bottom is at/below the old scan limit (game
                # y 371) is standing on the ground or on another trooper
                (out.grounded if gy > 359 else out.bodies).append((gx, gy))
        elif (w, h, n) == (12, 6, 58):
            out.canopies.append((gx + 8, gy + 32))
    for x0, y0, w, h, n in _components(frame[:SKY_ROWS] == WHITE):
        gx, gy = 2 * x0, 2 * y0 + 1
        if n == 1:
            out.dots.append((gx, gy))
        elif (w, h, n) == (4, 4, 12) and y0 >= 8:
            out.bombs.append((gx, gy))
        elif w == 4 and 5 <= h <= 8 and y0 >= 8:
            dy = _stacked_bombs(frame[y0:y0 + h, x0:x0 + 4] == WHITE)
            if dy is not None:
                out.bombs += [(gx, gy), (gx, gy + 2 * dy)]
        # heads low on screen: landed troopers, incl. ones standing on a stack
        # (world.py keeps only heads that stay put, so falling ones don't count)
        elif (w, h, n) == (2, 2, 4) and y0 >= 150 and not TURRET_X0 <= x0 <= TURRET_X1:
            out.landed.append((gx - 2, gy))
    return out
