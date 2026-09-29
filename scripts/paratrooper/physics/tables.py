"""Precomputed intercept tables: planning becomes lookups.

Every target type moves along a line that depends on one "phase" value,
so "does a bullet on lane L, spawning when the target is at phase p, hit
it -- and how many ticks later?" is a pure function of (L, p):

  troopers : phase = (x, y at spawn tick), per part (canopy/body) and fall
             speed (4 or 8 px/tick)            -> TROOPER[L, part, v, xi, yi]
  aircraft : phase = (lane y, x at spawn tick), per direction and kind
             (helicopter/plane)                -> AIR[kind][L, d, yi, xi]
  bombs    : phase = (release x, direction, spawn tick) -> BOMB[L, d, ri, f]

Values are j = ticks from spawn to meeting (-1 = miss); planners add the
spawn tick and apply reachability / landing cutoffs. Built with
physics.model.simulate (same semantics as the validated old simulators)
and cached in data/tables.npz; `python3 -m paratrooper.physics.tables`
rebuilds.
"""
import math
import pathlib

import numpy as np

from . import model as M

CACHE = pathlib.Path(__file__).resolve().parents[1] / "data" / "tables.npz"

# trooper grid: x even 0..638, y (body top at spawn tick) odd -99..399
TX = np.arange(0, 640, 2)
TY = np.arange(-99, 401, 2)
PARTS = ("canopy", "body")
VYS = (M.CANOPY_VY, M.FREE_VY)
# aircraft grid: skid y odd 11..121 (lanes seen: 17, 29, 41, 65, 89), x even -48..640
AY = np.arange(11, 123, 2)
AX = np.arange(-48, 642, 2)
KINDS = ("heli", "plane")
# bombs: release x -24..664 in steps of 8, spawn tick 0..33
BR = np.arange(-24, 665, 8)
MAX_J = 40


def _trooper_table():
    """Vectorized over the (x, y) grid, per lane / part / speed -- same
    tick + swept-substep semantics as model.simulate."""
    T = np.full((M.N_LANES, 2, 2, len(TX), len(TY)), -1, np.int8)
    X, Y = np.meshgrid(TX, TY, indexing="ij")
    for li, ((sx, sy), (vx, vy)) in enumerate(M.LANES):
        for pi, part in enumerate(PARTS):
            for vi, tv in enumerate(VYS):
                hit = np.full(X.shape, -1, np.int8)
                for j in range(MAX_J):
                    ux, uy = sx + vx * j, sy + vy * j
                    if uy < -2 or not -2 <= ux <= 640:
                        break
                    for s in ((1.0,) if j == 0 else (0.25, 0.5, 0.75, 1.0)):
                        px, py = ux - vx * (1 - s), uy - vy * (1 - s)
                        yk = Y + tv * (j - 1 + s)  # body top at that instant
                        if part == "canopy":
                            x0, y0, x1, y1 = X - 8, yk - 32, X + 15, yk - 5
                        else:
                            x0, y0, x1, y1 = X, yk - 4, X + 7, yk + 11
                        m = (px + 1 >= x0) & (px <= x1) & (py + 1 >= y0) & (py <= y1) & (hit < 0)
                        hit[m] = j
                T[li, pi, vi] = hit
    return T


def _trooper_bits_table():
    """Like _trooper_table, but every bullet age j (0..39) at which the
    bullet touches the box, as a bitmask: the hit-probability planner needs
    hits restricted to a phase (before / after the chute opens), not just
    the first one."""
    T = np.zeros((M.N_LANES, 2, 2, len(TX), len(TY)), np.uint64)
    X, Y = np.meshgrid(TX, TY, indexing="ij")
    for li, ((sx, sy), (vx, vy)) in enumerate(M.LANES):
        for pi, part in enumerate(PARTS):
            for vi, tv in enumerate(VYS):
                bits = np.zeros(X.shape, np.uint64)
                for j in range(MAX_J):
                    ux, uy = sx + vx * j, sy + vy * j
                    if uy < -2 or not -2 <= ux <= 640:
                        break
                    hit_j = np.zeros(X.shape, bool)
                    for s in ((1.0,) if j == 0 else (0.25, 0.5, 0.75, 1.0)):
                        px, py = ux - vx * (1 - s), uy - vy * (1 - s)
                        yk = Y + tv * (j - 1 + s)
                        if part == "canopy":
                            x0, y0, x1, y1 = X - 8, yk - 32, X + 15, yk - 5
                        else:
                            x0, y0, x1, y1 = X, yk - 4, X + 7, yk + 11
                        hit_j |= (px + 1 >= x0) & (px <= x1) & (py + 1 >= y0) & (py <= y1)
                    bits[hit_j] |= np.uint64(1 << j)
                T[li, pi, vi] = bits
    return T


def _air_table(kind):
    A = np.full((M.N_LANES, 2, len(AY), len(AX)), -1, np.int8)
    Y, X = np.meshgrid(AY, AX, indexing="ij")
    for li, ((sx, sy), (vx, vy)) in enumerate(M.LANES):
        for di, d in enumerate((-1, 1)):
            hit = np.full(X.shape, -1, np.int8)
            alive = np.ones(X.shape, bool)  # aircraft still on screen
            for j in range(MAX_J):
                ux, uy = sx + vx * j, sy + vy * j
                hx_int = X + M.HELI_VX * d * j
                alive &= (hx_int >= -48) & (hx_int <= 640)
                if uy < -2 or not -2 <= ux <= 640:
                    break
                for s in ((1.0,) if j == 0 else (0.25, 0.5, 0.75, 1.0)):
                    px, py = ux - vx * (1 - s), uy - vy * (1 - s)
                    hx = X + M.HELI_VX * d * (j - 1 + s)
                    if kind == "plane":
                        x0, y0, x1, y1 = hx, 1, hx + 47, 20
                    elif d < 0:
                        x0, y0, x1, y1 = hx, Y - 16, hx + 47, Y + 3
                    else:
                        x0, y0, x1, y1 = hx - 16, Y - 16, hx + 31, Y + 3
                    m = alive & (px + 1 >= x0) & (px <= x1) & (py + 1 >= y0) & (py <= y1) & (hit < 0)
                    hit[m] = j
            A[li, di] = hit
    return A


def _bomb_table():
    B = np.full((M.N_LANES, 2, len(BR), len(M.BOMB_YS)), -1, np.int8)
    for li, lane in enumerate(M.LANES):
        for di, d in enumerate((-1, 1)):
            for ri, xr in enumerate(BR):
                def box_at(kf, xr=xr, d=d):
                    k0 = math.floor(kf)
                    a = M.bomb_at(xr, d, k0)
                    b = M.bomb_at(xr, d, math.ceil(kf))
                    if a is None or b is None:
                        return None
                    fr = kf - k0
                    x, y = a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr
                    return x, y, x + 7, y + 7
                for f in range(len(M.BOMB_YS)):
                    m = M.simulate(lane, box_at, f)
                    if m is not None:
                        B[li, di, ri, f] = m - f
    return B


_T = None


def tables():
    global _T
    if _T is None:
        if CACHE.exists():
            z = np.load(CACHE)
            _T = {k: z[k] for k in z.files}
        else:
            _T = build()
    return _T


def build():
    t = {"trooper": _trooper_table(), "trooper_bits": _trooper_bits_table(),
         "heli": _air_table("heli"), "plane": _air_table("plane"), "bomb": _bomb_table()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **t)
    return t


if __name__ == "__main__":
    import time
    a = time.time()
    t = build()
    print({k: v.shape for k, v in t.items()}, f"built in {time.time() - a:.1f}s -> {CACHE}")
