"""Discrete barrel position (0 = 25.7 deg full right .. 18 = 155.1 deg full
left) from a native frame: the farthest cyan pixel within radius 23 of the
pivot identifies it. Tip table derived from 25,627 recorded frames against
the validated game-coordinate detector: all 19 tips map one-to-one with
100% agreement. (175,165)/(144,165) are the destroyed-turret sprite.
"""
import numpy as np

PIVOT = (160, 159.5)
TIP_TO_POS = {
    (173, 153): 0, (172, 151): 1, (171, 150): 2, (169, 148): 3, (168, 147): 4,
    (166, 146): 5, (165, 146): 6, (163, 145): 7, (162, 145): 8, (159, 145): 9,
    (157, 145): 10, (156, 145): 11, (154, 146): 12, (153, 146): 13, (151, 147): 14,
    (150, 148): 15, (148, 150): 16, (147, 151): 17, (146, 153): 18,
}
_Y0, _Y1, _X0, _X1 = 130, 166, 130, 190
_yy, _xx = np.mgrid[_Y0:_Y1, _X0:_X1]
_DIST = np.hypot(_xx - PIVOT[0], _yy - PIVOT[1])
_INSIDE = _DIST <= 23


def barrel_pos(frame):
    """0..18, or None (mid-rotation sprite, occluded, or destroyed)."""
    cyan = (frame[_Y0:_Y1, _X0:_X1] == 1) & _INSIDE
    if not cyan.any():
        return None
    i = np.argmax(np.where(cyan, _DIST, -1))
    return TIP_TO_POS.get((int(_xx.flat[i]), int(_yy.flat[i])))
