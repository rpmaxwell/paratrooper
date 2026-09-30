"""Intercept planning by table lookup. Same decision rules as the old,
validated planners (bomb_model.plan_intercept, trooper_model.plan_trooper,
heli_model.plan_heli); tools/check_tables.py checks them against each other.

All tick numbers are relative to the observation the plan is made from
(tick 0), and "spawn tick" means the tick a bullet appears at its lane's
spawn point. A plan is (pos, spawn_ticks, meet_tick[, clamp]).
"""
import math

import numpy as np

from . import model as M
from .tables import AX, AY, BR, TX, TY, tables

_T = tables()
_TROOPER, _HELI, _PLANE, _BOMB = _T["trooper"], _T["heli"], _T["plane"], _T["bomb"]
_TROOPER_SWEPT = _T["trooper_swept"]


def _earliest(moves, clamp=False):
    return (math.ceil(moves * M.TICKS_PER_POS) + (M.SETTLE_TICKS if moves else 0)
            + M.FIRE_LATENCY_TICKS + (M.CLAMP_TICKS if clamp else 0))


# ---- bombs --------------------------------------------------------------------
def bomb_meet(x_release, d, lane, f):
    """Meeting tick of a bullet on lane id `lane` spawning at bomb tick f."""
    ri = (x_release - BR[0]) // 8
    if not 0 <= ri < len(BR) or not 0 <= f < len(M.BOMB_YS) or (x_release - BR[0]) % 8:
        return None
    j = _BOMB[lane, 0 if d < 0 else 1, ri, f]
    return None if j < 0 else f + int(j)


def plan_bomb(x_release, d, k_now, cur_pos, clamped=False):
    """Barrel position with the widest reachable run of hitting spawn ticks
    (capped at 5), ties to the shorter move. -> (pos, run) or None."""
    best = None
    for pos in range(M.N_POS):
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        earliest = k_now + _earliest(moves)
        lane = M.lane_id(pos, cur_pos, clamped)
        usable = [f for f in range(max(0, earliest), len(M.BOMB_YS))
                  if bomb_meet(x_release, d, lane, f) is not None]
        if not usable:
            continue
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


# ---- troopers -------------------------------------------------------------------
def trooper_meet(x, y_spawn, vy, part, lane, f):
    """Meeting tick for a trooper whose body top is at y_spawn when the
    bullet spawns (tick f)."""
    xi, yi = (x - TX[0]) // 2, (y_spawn - TY[0]) // 2
    if not (0 <= xi < len(TX) and 0 <= yi < len(TY)):
        return None
    j = _TROOPER[lane, 0 if part == "canopy" else 1, 0 if vy == M.CANOPY_VY else 1, xi, yi]
    return None if j < 0 else f + int(j)


def _first_run(ok, meets, fs, min_len=1):
    """First run of consecutive True in ok -> (fs, meets) of that run."""
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None
    start = idx[0]
    stop = start
    while stop + 1 < ok.size and ok[stop + 1]:
        stop += 1
    if stop - start + 1 < min_len:
        return None
    return fs[start:stop + 1], meets[start:stop + 1]


_F25 = np.arange(25)


def plan_trooper(x, y, vy, part, cur_pos, clamped=False, free=False, chute_only=False, ground_y=None,
                 swept=False):
    """Trooper at body top (x, y) now, falling vy px/tick. Earliest meeting
    tick wins, ties to the wider window, then the shorter move. At the two
    rotation limits both lanes are options (stop on sight / clamp).
    -> (pos, spawn_ticks, meet_tick, clamp) or None. Vectorized over the
    spawn ticks; same decisions as the scalar version (check_tables)."""
    # ground_y: body top when it lands in this column -- higher on a stack
    max_k = ((M.GROUND_BODY_Y if ground_y is None else ground_y) - y) / vy - 1
    if part == "body" and free:
        max_k = min(max_k, M.FREE_MAX_MEET_TICKS)
    xi = (x - TX[0]) // 2
    if not 0 <= xi < len(TX):
        return None
    pi, vi = (0 if part == "canopy" else 1), (0 if vy == M.CANOPY_VY else 1)
    options = [(p, False) for p in range(M.N_POS)] + [(p, True) for p in M.SIGHT_LANE_ID]
    best = None
    for pos, clamp in options:
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        earliest = _earliest(moves, clamp)
        lane = pos if clamp else M.lane_id(pos, cur_pos, clamped)
        fs = earliest + _F25
        yi = (y + vy * fs - TY[0]) // 2
        valid = (yi >= 0) & (yi < len(TY))
        tab = _TROOPER_SWEPT if swept else _TROOPER  # swept: also count contact between ticks
        j = np.full(fs.shape, -1, np.int16)
        j[valid] = tab[lane, pi, vi, xi, yi[valid]]
        meets = fs + j
        ok = (j >= 0) & (meets <= max_k)
        if chute_only:
            # bullets travel upward: the shot must enter the chute box strictly
            # before it would touch the body (a body hit kills the trooper
            # outright -- no fall, no crush of the trooper below)
            jb = np.full(fs.shape, -1, np.int16)
            jb[valid] = tab[lane, 1, vi, xi, yi[valid]]
            ok &= (jb < 0) | (jb > j)
        run = _first_run(ok, meets, fs)
        if run is None:
            continue
        rf, rm = run
        mid = len(rf) // 2
        key = (int(rm[mid]), -len(rf), moves)
        if best is None or key < best[0]:
            best = (key, pos, [int(f) for f in rf], int(rm[mid]), clamp)
    return None if best is None else best[1:]


# ---- helicopters / planes ----------------------------------------------------------
def air_meet(kind, x0, y0, d, lane, f):
    """Meeting tick for an aircraft at x0 (skid/left x) now, lane y0."""
    x_spawn = x0 + M.HELI_VX * d * f
    xi, yi = (x_spawn - AX[0]) // 2, (y0 - AY[0]) // 2
    if not (0 <= xi < len(AX) and 0 <= yi < len(AY)):
        return None
    tab = _HELI if kind == "heli" else _PLANE
    j = tab[lane, 0 if d < 0 else 1, yi, xi]
    return None if j < 0 else f + int(j)


_F30 = np.arange(30)


def plan_air(kind, x0, y0, d, cur_pos, clamped=False, allowed=None, min_window=2):
    """Earliest meeting tick (lowest-priority targets: keep engagements
    short), >= min_window consecutive hitting spawn ticks; aims at the
    middle of the window. -> (pos, spawn_ticks, meet_tick) or None.
    Vectorized over spawn ticks."""
    yi = (y0 if kind == "heli" else AY[3]) - AY[0]
    yi //= 2
    if not 0 <= yi < len(AY):
        return None
    tab = _HELI if kind == "heli" else _PLANE
    di = 0 if d < 0 else 1
    best = None
    for pos in (range(M.N_POS) if allowed is None else allowed):
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        fs = _earliest(moves) + _F30
        xi = (x0 + M.HELI_VX * d * fs - AX[0]) // 2
        valid = (xi >= 0) & (xi < len(AX))
        j = np.full(fs.shape, -1, np.int16)
        j[valid] = tab[M.lane_id(pos, cur_pos, clamped), di, yi, xi[valid]]
        run = _first_run(j >= 0, fs + j, fs, min_window)
        if run is None:
            continue
        rf, rm = run
        mid = len(rf) // 2
        key = (int(rm[mid]), moves)
        if best is None or key < best[0]:
            best = (key, pos, [int(f) for f in rf], int(rm[mid]))
    return None if best is None else best[1:]
