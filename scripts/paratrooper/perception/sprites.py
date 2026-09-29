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
        # heads low on screen: landed troopers, incl. ones standing on a stack
        # (world.py keeps only heads that stay put, so falling ones don't count)
        elif (w, h, n) == (2, 2, 4) and y0 >= 150 and not TURRET_X0 <= x0 <= TURRET_X1:
            out.landed.append((gx - 2, gy))
    return out
