#!/usr/bin/env python3
"""Do bullets fired from the rotation-limit positions (0 = 25.7 deg,
18 = 155.1 deg) depend on HOW the turret was stopped there?

Live misses (planes2 evidence clips, 2026-09-27): 7 of 9 shots from the
limit positions flew (+-20, -6)/tick instead of the table's (+-20, -4) and
passed over canopies. Hypothesis: stopping the instant the limit sprite
appears leaves the internal angle short of the clamp; rotating on into
the stop for a few ticks clamps it.

Each trial: move away from the limit, goto(limit) either stopping on sight
("sight") or holding the rotation into the stop for HOLD_S first ("clamp"),
then fire one measured shot and read its step vector.

Usage: python3 calibrate_limit_lanes.py [trials_per_mode]
"""
import sys
import time
from collections import Counter

import mss
import numpy as np

from bomb_model import barrel_pos, to_index
from calibrate_bullets import follow, white_dots
from play_one_game import grab
from read_score import is_done
from turret import goto, key, read_pos

HOLD_S = 0.2


def to_limit(sct, mon, limit, mode):
    goto(sct, mon, 9)  # start from the middle
    if mode == "sight":
        return goto(sct, mon, limit)
    key("Left" if limit == 18 else "Right")
    t0 = time.time()
    while time.time() - t0 < 2.0 and read_pos(sct, mon, tries=1) != limit:
        pass
    time.sleep(HOLD_S)  # keep pushing into the rotation stop
    key("Up")
    time.sleep(0.1)
    return read_pos(sct, mon)


def main(n=20):
    res = {(lim, mode): Counter() for lim in (0, 18) for mode in ("sight", "clamp")}
    with mss.mss() as sct:
        mon = sct.monitors[1]
        for i in range(n * 4):
            if is_done(grab(sct, mon)):
                key("space")
                time.sleep(1.0)
                continue
            lim = (0, 18)[i % 2]
            mode = ("sight", "clamp")[(i // 2) % 2]
            p = to_limit(sct, mon, lim, mode)
            time.sleep(0.35)  # let the stop-bullet clear the spawn area
            t_fire = time.time()
            key("Up")
            samples, last = [], None
            while time.time() - t_fire < 0.6:
                idx = to_index(grab(sct, mon))
                if last is not None and np.array_equal(idx, last):
                    continue
                last = idx
                samples.append((time.time(), white_dots(idx)))
            tr = follow(samples, t_fire)
            v = tuple(tr["v"]) if tr else None
            res[(lim, mode)][v] += 1
            print(f"limit {lim} {mode}: at pos {p}, bullet v={v}", flush=True)
    for k, c in res.items():
        print(k, dict(c))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
