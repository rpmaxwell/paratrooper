#!/usr/bin/env python3
"""Stage 5.1: measure the game's own simulated speed (paratrooper fall
rate) in real wall-clock time, without taking any actions ourselves.
Used to compare across different dosbox-x cycles/turbo settings and
determine whether the game is cycle-bound or wall-clock/timer-bound.
"""
import sys
import time

import mss
import numpy as np

CYAN = np.array([85, 255, 255])
Y0, Y1 = 215, 580


def grab(sct, monitor):
    img = np.array(sct.grab(monitor))
    return img[:, :, :3][:, :, ::-1]


TURRET_X0, TURRET_X1 = 470, 555  # exclude the turret's own cyan barrel highlight


def cyan_centroid_y(frame):
    band = frame[Y0:Y1, :].copy()
    band[:, TURRET_X0:TURRET_X1] = 0
    mask = np.all(band == CYAN, axis=-1)
    if not mask.any():
        return None
    ys, _ = np.where(mask)
    return float(ys.mean() + Y0)


def main(duration_s=8.0, interval=0.15):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        t0 = time.time()
        while time.time() - t0 < duration_s:
            t = time.time() - t0
            y = cyan_centroid_y(grab(sct, monitor))
            print(f"{t:.3f}\t{y}")
            time.sleep(interval)


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    main(duration)
