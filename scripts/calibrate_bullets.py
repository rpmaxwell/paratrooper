#!/usr/bin/env python3
"""Measure, for every discrete barrel resting position, where a bullet
spawns and its exact per-tick velocity -- plus the latency from pressing
fire to the bullet first appearing on screen.

Bullets are 2x2 WHITE dots that move a fixed integer (dx, dy) every game
tick (~18.2 Hz), with the step vector depending only on the barrel
position. Bombs follow a fixed discrete parabola (see bomb_model.py), so
with this table a bomb intercept can be simulated exactly, tick by tick.

Each trial: rotate a random amount (Left/Right, random hold), press Up
(stop + fire), then capture every frame for 0.6s and follow the new
bullet outward from the barrel.

Usage: python3 calibrate_bullets.py [n_trials] [out_json]
"""
import hashlib
import json
import math
import random
import subprocess
import sys
import time

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from record_episode import to_index, GX0, GY0
from play_one_game import grab

# pivot in game-area (cropped) coordinates
PX, PY = 512 - GX0, 520 - GY0


def barrel_tip(idx):
    sub = idx[PY - 45:PY + 5, PX - 45:PX + 46] == 1
    ys, xs = np.nonzero(sub)
    if xs.size == 0:
        return None
    xs = xs + PX - 45
    ys = ys + PY - 45
    d = np.hypot(xs - PX, ys - PY)
    k = d <= 45
    if not k.any():
        return None
    i = int(np.argmax(np.where(k, d, -1)))
    return int(xs[i]), int(ys[i])


def barrel_signature(idx):
    """Exact cyan pixel pattern of the barrel -- identifies the discrete
    resting position without angle-estimation noise."""
    sub = (idx[PY - 45:PY + 5, PX - 45:PX + 46] == 1)
    return hashlib.md5(np.packbits(sub).tobytes()).hexdigest()[:10]


def white_dots(idx, rmax=140):
    """2x2 white dots (bullets) within rmax of the pivot, above it."""
    y0 = max(0, PY - rmax)
    w = idx[y0:PY, max(0, PX - rmax):PX + rmax] == 3
    # a bullet pixel's top-left: white, with the 2x2 block white and a
    # black ring around it
    ys, xs = np.nonzero(w[:-1, :-1] & w[1:, :-1] & w[:-1, 1:] & w[1:, 1:])
    out = []
    for y, x in zip(ys, xs):
        blk = w[max(0, y - 1):y + 3, max(0, x - 1):x + 3]
        if blk.sum() == 4:
            out.append((int(x + max(0, PX - rmax)), int(y + y0)))
    return out


def follow(samples, t_fire):
    """Find the first new bullet track after t_fire. samples: list of
    (t, dots) for DISTINCT frames only."""
    for i, (t, dots) in enumerate(samples):
        if t < t_fire:
            continue
        for d in dots:
            if math.hypot(d[0] - PX, d[1] - PY) > 70:
                continue
            # confirm with the next two distinct frames at constant velocity
            for j in range(i + 1, min(i + 3, len(samples))):
                for d2 in samples[j][1]:
                    v = (d2[0] - d[0], d2[1] - d[1])
                    if not (8 <= math.hypot(*v) <= 26 and v[1] < 0):
                        continue
                    pts = [d, d2]
                    for k in range(j + 1, len(samples)):
                        nxt = (pts[-1][0] + v[0], pts[-1][1] + v[1])
                        if nxt in samples[k][1]:
                            pts.append(nxt)
                        if len(pts) >= 4:
                            break
                    if len(pts) >= 4 and j == i + 1:
                        return dict(t_first=t, start=d, v=v, n=len(pts))
    return None


def main(n_trials=80, out_path="/captures/bullet_calibration.json"):
    trials = []
    with mss.mss() as sct:
        mon = sct.monitors[1]
        for n in range(n_trials):
            f = grab(sct, mon)
            if is_done(f):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(1.0)
                continue
            do_action(random.choice([1, 2]))
            time.sleep(random.uniform(0.0, 0.6))
            do_action(3)  # stop (no bullet expected to matter yet)
            time.sleep(0.25)
            idx = to_index(grab(sct, mon))
            tip, sig = barrel_tip(idx), barrel_signature(idx)
            t_fire = time.time()
            do_action(3)  # the measured shot, turret already at rest
            t_after = time.time()
            samples, last = [], None
            while time.time() - t_fire < 0.7:
                idx = to_index(grab(sct, mon))
                if last is not None and np.array_equal(idx, last):
                    continue
                last = idx
                samples.append((time.time(), white_dots(idx)))
            tr = follow(samples, t_fire)
            rec = dict(tip=tip, sig=sig, t_press=t_fire, t_after=t_after,
                       bullet=None)
            if tr:
                rec["bullet"] = dict(start=tr["start"], v=tr["v"], n=tr["n"],
                                     latency_s=tr["t_first"] - t_fire)
            ang = math.degrees(math.atan2(PY - tip[1], tip[0] - PX)) if tip else None
            print(f"trial {n}: tip={tip} ang={ang and round(ang,1)} bullet={rec['bullet']}", flush=True)
            trials.append(rec)
    with open(out_path, "w") as fh:
        json.dump(trials, fh, indent=1)
    print(f"saved {len(trials)} trials -> {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    out = sys.argv[2] if len(sys.argv) > 2 else "/captures/bullet_calibration.json"
    main(n, out)
