#!/usr/bin/env python3
"""Phase-2 (planes + bombs) vision and physics model, measured directly
from frame-by-frame recordings (record_episode.py, 2026-09-25).

All coordinates here are GAME-AREA coordinates: the 640x400 region of
the 1024x768 Xvfb frame starting at (GX0, GY0) = (192, 200). Frames are
palette-indexed: 0 black, 1 cyan, 2 magenta, 3 white (see to_index).

Findings this model encodes:
  * Everything moves on the game's ~18.2 Hz tick; positions are integer
    and change only on tick boundaries.
  * Phase 1 (helicopters) runs a fixed ~38s, then planes enter at ~38.2s.
    Helicopter = cyan skids 32x4 (y0 17/41 in wave 1, lower -- 29/65 --
    in later waves). Plane = cyan body 48x20 at y0 1.
  * Bomb = WHITE filled disc, bbox exactly 8x8, 48 px. It moves exactly
    8 px/tick horizontally (the plane's speed) and falls along the fixed
    discrete-gravity sequence BOMB_YS (vy += 2 px/tick every 4 ticks),
    landing on the turret ~33 ticks (~1.8s) after release. Release x
    varies with the plane position, so paths are x-translates of one
    curve: x(k) = x_release + 8*dir*k, y(k) = BOMB_YS[k].
  * Bullet = 2x2 WHITE dot, moving a fixed integer (dx, dy) per tick that
    depends only on the barrel's discrete resting position -- ~17-21
    px/tick, i.e. NOT hitscan.
"""
import numpy as np

GX0, GX1, GY0, GY1 = 192, 832, 200, 600
PALETTE = np.array([[0, 0, 0], [85, 255, 255], [255, 85, 255], [255, 255, 255]], dtype=np.uint8)
PIVOT = (320, 320)  # turret pivot, game-area coords (512, 520 in the Xvfb frame)

# bomb's top-left y at each tick after release (k=0 is the first frame seen)
BOMB_YS = [17, 19, 21, 23, 25, 29, 33, 37, 41, 47, 53, 59, 65, 73, 81, 89, 97,
           107, 117, 127, 137, 149, 161, 173, 185, 199, 213, 227, 241, 257,
           273, 289, 305, 323]
BOMB_TICK_OF_Y = {y: k for k, y in enumerate(BOMB_YS)}
BOMB_VX = 8
TICK_S = 1 / 18.2065

SKY_Y1 = 330  # game-area y above which sprites are "in the sky" (turret top ~ 300)


def to_index(frame):
    """Full RGB Xvfb frame -> palette-indexed game area (uint8 0..3)."""
    crop = frame[GY0:GY1, GX0:GX1]
    idx = np.zeros(crop.shape[:2], dtype=np.uint8)
    for i, c in enumerate(PALETTE[1:], start=1):
        idx[np.all(crop == c, axis=-1)] = i
    return idx


def components(mask):
    """8-connected components of a (sparse) boolean mask ->
    list of (xs, ys) int arrays."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    pts = set(zip(xs.tolist(), ys.tolist()))
    out = []
    while pts:
        seed = pts.pop()
        stack, comp = [seed], [seed]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in pts:
                        pts.remove(q)
                        stack.append(q)
                        comp.append(q)
        a = np.array(comp)
        out.append((a[:, 0], a[:, 1]))
    return out


def sky_sprites(idx):
    """Classify sky-band sprites. Returns dict with lists of
    'planes' [(x0, x1)], 'helis' [(x0, y0)], 'bombs' [(x0, y0)]."""
    planes, helis, bombs = [], [], []
    # helicopter lanes drop in later waves (skids at y0 17/41 in wave 1,
    # 29/65 in wave 2), so search a deep band rather than fixed lanes
    comps = components(idx[:110] == 1)
    # anything cyan that starts near the very top is (part of) a plane --
    # when planes overlap/redraw, a plane can split into pieces, and its
    # lower strip must not be mistaken for helicopter skids
    plane_parts = [(xs.min(), xs.max()) for xs, ys in comps if ys.min() <= 8]
    for xs, ys in comps:
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if h >= 12 and w >= 24 and ys.min() <= 4:
            planes.append((int(xs.min()), int(xs.max())))
        elif h <= 4 and w >= 24 and 12 <= ys.min() <= 100 and not any(
                a <= xs.max() + 8 and xs.min() - 8 <= b for a, b in plane_parts):
            helis.append((int(xs.min()), int(ys.min())))
    white = idx[:SKY_Y1] == 3
    for xs, ys in components(white):
        if xs.size == 48 and xs.max() - xs.min() == 7 and ys.max() - ys.min() == 7:
            y0 = int(ys.min())
            if y0 >= 16:  # a plane's white cockpit (y 9-14) is never a bomb
                bombs.append((int(xs.min()), y0))
    return dict(planes=planes, helis=helis, bombs=bombs)


def bomb_at(x_release, direction, k):
    """Top-left of a bomb k ticks after release (None once it has landed)."""
    if not 0 <= k < len(BOMB_YS):
        return None
    return x_release + BOMB_VX * direction * k, BOMB_YS[k]


class PhaseTracker:
    """phase 1 = helicopters, 'gap' = helicopters gone but no plane yet,
    phase 2 = at least one plane seen since the last helicopter."""

    def __init__(self):
        self.phase = 1
        self.last_heli_t = None
        self.first_plane_t = None

    def update(self, sprites, t):
        if sprites["helis"]:
            self.last_heli_t = t
            if self.phase != 1:
                self.phase = 1
                self.first_plane_t = None
        if sprites["planes"] and self.phase != 2:
            self.phase = 2
            self.first_plane_t = t
        elif self.phase == 1 and not sprites["helis"] and self.last_heli_t is not None \
                and t - self.last_heli_t > 0.5:
            self.phase = "gap"
        return self.phase


# ---------------------------------------------------------------------------
# Barrel positions -> bullets (calibrate_bullets.py, 90 live trials,
# 2026-09-25). The turret rests only at these 19 discrete positions.
# tip = farthest barrel (cyan) pixel from the pivot, the runtime key for
# "which position are we at". spawn = bullet's top-left on its first tick;
# v = its exact per-tick step. Right half measured; left half is the exact
# mirror (verified on every left position that was also measured: tip
# x -> 639-x, spawn x -> 640-x, vx -> -vx). 105.4 and 117.4 were not
# directly measured, only mirrored.
_RIGHT = [  # (angle, tip, spawn, v)
    (25.7, (347, 307), (360, 307), (20, -4)),
    (34.2, (345, 303), (356, 299), (18, -8)),
    (39.6, (343, 301), (352, 295), (16, -10)),
    (50.4, (339, 297), (352, 291), (16, -12)),
    (55.8, (337, 295), (344, 291), (12, -12)),
    (64.3, (333, 293), (340, 287), (10, -14)),
    (67.8, (331, 293), (336, 287), (8, -14)),
    (76.4, (327, 291), (332, 283), (6, -16)),
    (80.2, (325, 291), (324, 283), (2, -16)),
]
_CENTER = (93.9, (318, 291), (320, 283), (0, -16))
_LEFT_ANGLES = [155.1, 146.8, 141.6, 131.0, 125.8, 117.4, 114.0, 105.4, 101.7]
BARREL = sorted(
    _RIGHT + [_CENTER] + [
        (la, (639 - tip[0], tip[1]), (640 - sp[0], sp[1]), (-v[0], v[1]))
        for la, (_, tip, sp, v) in zip(_LEFT_ANGLES, _RIGHT)
    ])  # index 0 = 25.7 (full right) ... 18 = 155.1 (full left)
TIP_TO_POS = {tip: i for i, (_, tip, _, _) in enumerate(BARREL)}

# The two rotation-limit positions are really two lanes each, identical on
# screen (calibrate_limit_lanes.py, 2026-09-27): stopping the moment the
# limit sprite appears fires (+-20, -6)/tick in 20/21 shots; holding the
# rotation into the stop first fires the table's (+-20, -4) in 23/23. 7 of
# 9 limit-position misses in live play were this (bullet 10-16 px over the
# canopy). turret.goto records which state the turret is in.
LIMIT_SIGHT_LANE = {0: ((360, 303), (20, -6)), 18: ((280, 303), (-20, -6))}
LIMIT_STATE = {"clamped": False}  # set by turret.goto


def lane(pos, cur_pos):
    """(spawn, v) a bullet from `pos` will use, given where the turret is
    now: arriving at a limit (or sitting there unclamped) -> steeper lane."""
    if pos in LIMIT_SIGHT_LANE and (cur_pos != pos or not LIMIT_STATE["clamped"]):
        return LIMIT_SIGHT_LANE[pos]
    return BARREL[pos][2], BARREL[pos][3]

# Up-press -> bullet at its spawn point, in game ticks, measured from the
# moment the bomb frame we reacted to was grabbed. calibrate_bullets.py:
# first-visible latency 0.149-0.279s, median 0.183s (~3.3 ticks).
FIRE_LATENCY_TICKS = 3


def barrel_pos(idx):
    """Discrete barrel position index (0..18) or None (mid-rotation /
    occluded)."""
    sub = idx[PIVOT[1] - 45:PIVOT[1] + 5, PIVOT[0] - 45:PIVOT[0] + 46] == 1
    ys, xs = np.nonzero(sub)
    if xs.size == 0:
        return None
    xs = xs + PIVOT[0] - 45
    ys = ys + PIVOT[1] - 45
    d = np.hypot(xs - PIVOT[0], ys - PIVOT[1])
    d = np.where(d <= 45, d, -1)
    i = int(np.argmax(d))
    return TIP_TO_POS.get((int(xs[i]), int(ys[i])))


def _overlap(bx, by, ux, uy):
    # bomb 8x8 at (bx, by), bullet 2x2 at (ux, uy)
    return ux + 1 >= bx and ux <= bx + 7 and uy + 1 >= by and uy <= by + 7


def bullet_hits(x_release, direction, pos, spawn_tick, lane_=None):
    """Does a bullet from barrel position `pos`, at its spawn point on bomb
    tick `spawn_tick`, hit the bomb? Checks each tick's overlap plus the
    swept path between ticks (the two can pass through each other at a
    combined ~30 px/tick closing speed). Returns the meeting tick or None."""
    (sx, sy), (vx, vy) = lane_ or (BARREL[pos][2], BARREL[pos][3])
    for j in range(0, 40):
        k = spawn_tick + j
        b = bomb_at(x_release, direction, k)
        if b is None:
            return None
        ux, uy = sx + vx * j, sy + vy * j
        if uy < -2 or not -2 <= ux <= 640:
            return None
        if j == 0:
            if _overlap(b[0], b[1], ux, uy):
                return k
            continue
        pb = bomb_at(x_release, direction, k - 1)
        for s in (0.25, 0.5, 0.75, 1.0):
            if _overlap(pb[0] + (b[0] - pb[0]) * s, pb[1] + (b[1] - pb[1]) * s,
                        ux - vx + vx * s, uy - vy + vy * s):
                return k
    return None


def hit_window(x_release, direction, pos, lane_=None):
    """All spawn ticks for which a bullet from `pos` hits this bomb."""
    return [f for f in range(len(BOMB_YS)) if bullet_hits(x_release, direction, pos, f, lane_) is not None]


def plan_intercept(x_release, direction, k_now, cur_pos, ticks_per_pos=1.1, settle_ticks=2):
    """Choose the barrel position to shoot this bomb from.

    A position is usable from the first spawn tick we can still reach:
    now + rotation time (the turret steps ~1 position per tick) + settle
    margin + fire latency. Score = number of usable consecutive hitting
    spawn ticks (robustness to +-1 tick of timing error), ties broken by
    the shorter rotation. Returns (pos, usable_spawn_ticks) or None."""
    best = None
    for pos in range(len(BARREL)):
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        earliest = (k_now + int(np.ceil(moves * ticks_per_pos)) + (settle_ticks if moves else 0)
                    + FIRE_LATENCY_TICKS)
        usable = [f for f in hit_window(x_release, direction, pos, lane(pos, cur_pos)) if f >= earliest]
        if not usable:
            continue
        # longest run of consecutive ticks
        runs, run = [], [usable[0]]
        for f in usable[1:]:
            if f == run[-1] + 1:
                run.append(f)
            else:
                runs.append(run)
                run = [f]
        runs.append(run)
        run = max(runs, key=len)
        key = (min(len(run), 5), -moves)
        if best is None or key > best[0]:
            best = (key, pos, run)
    return None if best is None else (best[1], best[2])
