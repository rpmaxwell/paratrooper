#!/usr/bin/env python3
"""Helicopter model + intercept planner, same tick-simulation approach as
bomb_model.py (game-area coordinates, palette-indexed frames).

Measured from recordings (2026-09-26, 4 episodes, ~5000 tracked steps):
  * A helicopter is identified by its cyan skids: exactly 32x4.
  * Moves exactly 8 px/tick horizontally, never vertically.
  * Two lanes per wave, one flying each way -- but WHICH way changes by
    wave (wave 1: 17 left / 41 right; a later wave: 17 right / 41 left;
    others at 29/65, 41/89). Never infer direction from lane height:
    bomb_defense measures it from frame-to-frame motion.
  * Full sprite box relative to the skids' top-left (x0, y0):
      flying left : x0 .. x0+43,  y0-16 .. y0+3
      flying right: x0-12 .. x0+31, y0-16 .. y0+3
"""
import numpy as np

from bomb_model import BARREL, FIRE_LATENCY_TICKS, components, lane

HELI_VX = 8
SKID_W, SKID_H = 32, 4


def find_helis(idx):
    """Fully-visible helicopters -> list of (skid_x0, skid_y0)."""
    out = []
    for xs, ys in components(idx[:110] == 1):
        if xs.max() - xs.min() + 1 == SKID_W and ys.max() - ys.min() + 1 == SKID_H and ys.min() >= 12:
            out.append((int(xs.min()), int(ys.min())))
    return out


def heli_box(x0, y0, d):
    if d < 0:
        return x0, y0 - 16, x0 + 43, y0 + 3
    return x0 - 12, y0 - 16, x0 + 31, y0 + 3


def plane_box(x0, y0, d):
    """Plane: 48x20 hollow outline, always at the top (y 1..20), 8 px/tick
    (measured 2026-09-26, 1800+ sightings). x0 = left edge of its cyan."""
    return x0, 1, x0 + 47, 20


def bullet_hits_heli(x0, y0, d, pos, spawn_tick, box=heli_box, lane_=None):
    """Meeting tick (relative to the observation, tick 0) of a bullet from
    barrel `pos` spawning at `spawn_tick`, or None. Checks overlap at each
    tick plus the swept path between ticks."""
    (sx, sy), (vx, vy) = lane_ or (BARREL[pos][2], BARREL[pos][3])

    def overlap(hx, ux, uy):
        bx0, by0, bx1, by1 = box(hx, y0, d)
        return ux + 1 >= bx0 and ux <= bx1 and uy + 1 >= by0 and uy <= by1

    for j in range(0, 40):
        k = spawn_tick + j
        hx = x0 + HELI_VX * d * k
        if not -48 <= hx <= 640:
            return None  # helicopter has left the screen
        ux, uy = sx + vx * j, sy + vy * j
        if uy < -2 or not -2 <= ux <= 640:
            return None
        for s in ((1.0,) if j == 0 else (0.25, 0.5, 0.75, 1.0)):
            if overlap(hx - HELI_VX * d * (1 - s), ux - vx * (1 - s), uy - vy * (1 - s)):
                return k
    return None


def plan_heli(x0, y0, d, cur_pos, ticks_per_pos=1.1, settle_ticks=2, min_window=2,
              box=heli_box, allowed=None):
    """Best barrel position to shoot this helicopter from, preferring the
    earliest kill (helicopters are the lowest priority, so engagements
    should be short). Needs >= min_window consecutive hitting spawn ticks
    for timing robustness. Returns (pos, spawn_ticks, meet_tick) or None."""
    best = None
    for pos in (range(len(BARREL)) if allowed is None else allowed):
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        earliest = (int(np.ceil(moves * ticks_per_pos)) + (settle_ticks if moves else 0)
                    + FIRE_LATENCY_TICKS)
        run = []
        for f in range(earliest, earliest + 30):
            m = bullet_hits_heli(x0, y0, d, pos, f, box, lane(pos, cur_pos))
            if m is not None:
                run.append((f, m))
            elif run:
                break
        if len(run) < min_window:
            continue
        # aim for the middle of the window (+-1 tick of slop either side)
        mid = run[len(run) // 2]
        key = (mid[1], moves)
        if best is None or key < best[0]:
            best = (key, pos, [f for f, _ in run], mid[1])
    return None if best is None else best[1:]
