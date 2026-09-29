#!/usr/bin/env python3
"""Passive (read-only, no key presses) observation of whatever game is
currently being played -- safe to run alongside live training since it
never touches xdotool. Logs left/right walker counts every ~0.2s and
prints the counts at the exact moment is_done() flips True, across
however many game-overs occur during the observation window. Used to
determine the actual game-over trigger condition empirically (does it
correlate with total walker count per side? proximity to the base?
something else?) rather than assume.
"""
import sys
import time

import mss
import numpy as np

from read_score import is_done, read_score
from threats import BASE_X0, BASE_X1, WHITE, _mask_coords, count_walkers_per_side


def nearest_walker_dist_to_base(frame, pivot_x=512.0):
    """Distance from the base edge to the closest walker on each side,
    in case proximity (not raw count) is what actually matters."""
    xs, _ = _mask_coords(frame, WHITE, y0=558, y1=573)
    xs = xs[(xs < BASE_X0) | (xs > BASE_X1)]
    left = xs[xs < pivot_x]
    right = xs[xs >= pivot_x]
    left_dist = float(BASE_X0 - left.max()) if left.size else None
    right_dist = float(right.min() - BASE_X1) if right.size else None
    return left_dist, right_dist


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def main(duration_s=600.0, poll=0.2):
    prev_done = False
    gameovers = []
    history = []  # rolling recent (left, right, ld, rd) for context at the moment of done

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        t0 = time.time()
        while time.time() - t0 < duration_s:
            frame = grab(sct, monitor)
            done = is_done(frame)
            left, right = count_walkers_per_side(frame)
            ld, rd = nearest_walker_dist_to_base(frame)
            score = read_score(frame)
            history.append((left, right, ld, rd, score))
            if len(history) > 10:
                history.pop(0)

            if done and not prev_done:
                print(f"\n=== GAME OVER at t={time.time()-t0:.1f}s ===")
                print("last 10 samples before/at done (left, right, left_dist, right_dist, score):")
                for h in history:
                    print("  ", h)
                gameovers.append(history[-1])

            prev_done = done
            time.sleep(poll)

    print(f"\n\n{len(gameovers)} game-overs observed in {duration_s:.0f}s")
    for g in gameovers:
        print(" final state:", g)


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
    main(duration)
