"""Hit-probability planning for FREE-FALLING troopers.

A free-faller follows one of a known set of trajectories: free fall at
8 px/tick until its chute opens k ticks from now, then 4 px/tick with the
tall chute box on top (physics/chute.py gives P(k)). For a bullet on a lane
spawning f ticks from now,

    P(hit) = sum_k P(k) * [bullet meets the body before tick k
                           or the chute/body after tick k, before landing]

Each case is a bitmask lookup in the trooper_bits table (every bullet age
at which the bullet touches a box moving at constant speed), restricted to
the right phase -- vectorized over k and f, so a full plan over every
barrel position is a few ms.

Chuted troopers are deterministic (P = 1 or 0) and keep using plan.py.
"""
import math

import numpy as np

from . import chute
from . import model as M
from .tables import TX, TY, tables

_BITS = tables()["trooper_bits"]
_ONE = np.uint64(1)
_ALL = np.uint64((1 << 40) - 1)
N_F = 25          # spawn ticks considered from the earliest reachable one
HORIZON = 40      # chute-opening scenarios considered (the floor comes sooner)
PAIR_MIN_GAIN = 0.15   # a second bullet must add at least this much hit probability
SHOT_PENALTY = 0.15    # ... and is charged this much when comparing plans


def _earliest(moves, clamp=False):
    return (math.ceil(moves * M.TICKS_PER_POS) + (M.SETTLE_TICKS if moves else 0)
            + M.FIRE_LATENCY_TICKS + (M.CLAMP_TICKS if clamp else 0))


def _lookup(lane, part_i, vy_i, xi, y0):
    yi = (y0 - TY[0]) // 2
    ok = (yi >= 0) & (yi < len(TY))
    out = np.zeros(y0.shape, np.uint64)
    out[ok] = _BITS[lane, part_i, vy_i, xi, yi[ok]]
    return out


def _low_mask(n):
    """bits 0..n-1 set (n may be an array; <=0 -> 0, >=40 -> all)."""
    n = np.clip(n, 0, 40).astype(np.uint64)
    return np.where(n >= 40, _ALL, (_ONE << n) - _ONE)


def _lowest(bits):
    """index of the lowest set bit (bits != 0), else 99."""
    nz = bits != 0
    lsb = bits & (~bits + _ONE)
    return np.where(nz, np.log2(np.where(nz, lsb, _ONE).astype(np.float64)), 99).astype(np.int64)


def hit_matrix(x, y_now, s_now, lane, f0, chute_only=False, ground_y=None):
    """-> (P(k) [K], hit [K, N_F] bool, meet [K, N_F] ticks) for spawn ticks
    f0..f0+N_F-1 on `lane`, over chute-opening scenarios k = 1..HORIZON."""
    p = chute.open_distribution(s_now, y_now, HORIZON)
    ks = np.arange(1, HORIZON + 1)[:, None]          # [K, 1]
    fs = f0 + np.arange(N_F)[None, :]                # [1, F]
    xi = (x - TX[0]) // 2
    # before opening: body only, free fall from y_now
    free = np.broadcast_to(_lookup(lane, 1, 1, xi, y_now + M.FREE_VY * fs), (HORIZON, N_F))
    pre = free & _low_mask(ks - fs)                  # bullet ages j < k - f
    # after opening: chute + body at 4 px/tick; body top at spawn-age 0 would be y0c
    y0c = y_now + M.FREE_VY * ks + M.CANOPY_VY * (fs - ks)
    after_mask = ~_low_mask(ks - fs)                 # ages j >= k - f
    y_open = y_now + M.FREE_VY * ks
    t_land = ks + ((M.GROUND_BODY_Y if ground_y is None else ground_y) - y_open) / M.CANOPY_VY
    land_mask = _low_mask(np.floor(t_land - 1 - fs).astype(np.int64) + 1)  # meet before landing
    chute_bits = _lookup(lane, 0, 0, xi, y0c) & after_mask & land_mask
    body_after = _lookup(lane, 1, 0, xi, y0c) & after_mask & land_mask
    if chute_only:
        # count only scenarios where the bullet enters the chute box before
        # it would touch the body in either phase
        jc, jb = _lowest(chute_bits), _lowest(pre | body_after)
        hit = (chute_bits != 0) & (jc < jb)
        return p, hit, fs + np.where(hit, jc, 0)
    bits = pre | chute_bits | body_after
    hit = bits != 0
    lsb = bits & (~bits + _ONE)
    j = np.where(hit, np.log2(np.where(hit, lsb, _ONE).astype(np.float64)), 0).astype(np.int64)
    return p, hit, fs + j


def plan_free(x, y_now, s_now, cur_pos, clamped=False, min_p=0.0, allow_pair=True, chute_only=False,
              ground_y=None):
    """Best shot at a free-faller. Considers every barrel position (and the
    clamped limit lanes), every reachable spawn tick, and pairs of spawn
    ticks on the same lane (one bullet for each likely phase).
    -> dict(pos, clamp, spawns, p_hit, meet_last, p_single) or None."""
    best = None
    options = [(pos, False) for pos in range(M.N_POS)] + [(pos, True) for pos in M.SIGHT_LANE_ID]
    for pos, clamp in options:
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        f0 = _earliest(moves, clamp)
        lane = pos if clamp else M.lane_id(pos, cur_pos, clamped)
        p, hit, meet = hit_matrix(x, y_now, s_now, lane, f0, chute_only, ground_y)
        single = p @ hit                                           # [F]
        fi = int(np.argmax(single))
        cand = [(float(single[fi]), [fi])]
        if allow_pair and single[fi] < 0.9:
            # union of two spawn ticks >= 1 apart
            u = (p[:, None, None] * (hit[:, :, None] | hit[:, None, :])).sum(0)  # [F, F]
            np.fill_diagonal(u, 0)
            a, b = np.unravel_index(int(np.argmax(u)), u.shape)
            cand.append((float(u[a, b]), sorted([int(a), int(b)])))
        for p_hit, idx in cand:
            if p_hit < min_p or (len(idx) == 2 and p_hit - cand[0][0] < PAIR_MIN_GAIN):
                continue
            fire = [f0 + i for i in idx]
            hitting = hit[:, idx].any(1) & (p > 0.02)
            meet_last = int(meet[:, idx].max(where=hit[:, idx], initial=fire[-1]).max()) if hitting.any() else fire[-1]
            key = (round(p_hit - SHOT_PENALTY * (len(idx) - 1), 3), -meet_last, -moves)
            if best is None or key > best[0]:
                best = (key, dict(pos=pos, clamp=clamp, spawns=fire, p_hit=p_hit, meet_last=meet_last,
                                  p_single=float(single[fi]), moves=moves))
    return None if best is None else best[1]
