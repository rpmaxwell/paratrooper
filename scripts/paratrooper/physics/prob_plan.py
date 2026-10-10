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

PARATROOPER_PHIT selects how P(hit) is scored (per process, for A/B runs):
  "model"      -- as above: a drawn-position contact kills with P_DRAWN,
                  a between-ticks-only contact with P_SWEPT_ONLY.
  "calibrated" -- fitted on 5.5k replanned engagements (sim/phit.py,
                  2026-10-07); needs PARATROOPER_TROOPER_BOX=fitted_parts:
                  * the stop-press bullet: every move ends with an Up press
                    that fires down the destination lane, spawning
                    moves + STOP_BASE ticks after the observation (+-STOP_SLIP)
                    -- ~40% of all trooper kills, ignored by "model";
                  * timing slip: the plan's tick 0 is a tick early 17% of the
                    time, every bullet of the plan slipping together (SLIP);
                  * per contact: kills Q_KILL (drawn), Q_SWEPT (only between
                    drawn positions, frame data); a canopy-first kill is a
                    canopy kill (crush) CANOPY_SHARE of the time.
                  Crush (chute_only) bets drop from ~0.70 planned to ~0.30,
                  which is what they achieve.
"""
import math
import os

import numpy as np

from . import chute
from . import model as M
from .tables import P_DRAWN, P_SWEPT_ONLY, TX, TY, tables

_BITS = tables()["trooper_bits"]              # drawn-position contact
_BITS_SWEPT = tables()["trooper_bits_swept"]   # contact at drawn positions or between ticks
_ONE = np.uint64(1)
_ALL = np.uint64((1 << 40) - 1)
N_F = 25          # spawn ticks considered from the earliest reachable one
HORIZON = 40      # chute-opening scenarios considered (the floor comes sooner)
PAIR_MIN_GAIN = 0.15   # a second bullet must add at least this much hit probability
SHOT_PENALTY = 0.15    # ... and is charged this much when comparing plans

PHIT = os.environ.get("PARATROOPER_PHIT", "model")
if PHIT not in ("model", "calibrated"):
    raise ValueError(f"PARATROOPER_PHIT={PHIT!r}: expected 'model' or 'calibrated'")
if PHIT == "calibrated" and M.TROOPER_BOX != "fitted_parts":
    raise ValueError("PARATROOPER_PHIT=calibrated was fitted with PARATROOPER_TROOPER_BOX=fitted_parts")
Q_KILL, Q_SWEPT, CANOPY_SHARE = 0.90, 0.12, 0.75
SLIP = {-1: 0.01, 0: 0.82, 1: 0.17}                  # actual - planned spawn, all bullets of a plan
STOP_BASE = 4
STOP_SLIP = {-1: 0.05, 0: 0.61, 1: 0.29, 2: 0.05}   # stop bullet spawn - (moves + STOP_BASE)


def _earliest(moves, clamp=False):
    return (math.ceil(moves * M.TICKS_PER_POS) + (M.SETTLE_TICKS if moves else 0)
            + M.FIRE_LATENCY_TICKS + (M.CLAMP_TICKS if clamp else 0))


def _lookup(lane, part_i, vy_i, xi, y0, table=None):
    table = _BITS if table is None else table
    yi = (y0 - TY[0]) // 2
    ok = (yi >= 0) & (yi < len(TY))
    out = np.zeros(y0.shape, np.uint64)
    out[ok] = table[lane, part_i, vy_i, xi, yi[ok]]
    return out


def _phase_bits(table, lane, xi, y_now, ks, fs, after_mask, land_mask):
    """(pre, chute, body_after) contact bitmasks under one collision table."""
    free = np.broadcast_to(_lookup(lane, 1, 1, xi, y_now + M.FREE_VY * fs, table), (HORIZON, fs.shape[1]))
    pre = free & _low_mask(ks - fs)
    y0c = y_now + M.FREE_VY * ks + M.CANOPY_VY * (fs - ks)
    chute = _lookup(lane, 0, 0, xi, y0c, table) & after_mask & land_mask
    body_after = _lookup(lane, 1, 0, xi, y0c, table) & after_mask & land_mask
    return pre, chute, body_after


def _low_mask(n):
    """bits 0..n-1 set (n may be an array; <=0 -> 0, >=40 -> all)."""
    n = np.clip(n, 0, 40).astype(np.uint64)
    return np.where(n >= 40, _ALL, (_ONE << n) - _ONE)


def _lowest(bits):
    """index of the lowest set bit (bits != 0), else 99."""
    nz = bits != 0
    lsb = bits & (~bits + _ONE)
    return np.where(nz, np.log2(np.where(nz, lsb, _ONE).astype(np.float64)), 99).astype(np.int64)


def hit_matrix(x, y_now, s_now, lane, f0, chute_only=False, ground_y=None, n_f=N_F):
    """-> (P(k) [K], q [K, n_f] P(this bullet kills), meet [K, n_f] ticks) for
    spawn ticks f0..f0+n_f-1 on `lane`, over chute-opening scenarios k = 1..HORIZON."""
    p = chute.open_distribution(s_now, y_now, HORIZON)
    ks = np.arange(1, HORIZON + 1)[:, None]          # [K, 1]
    fs = f0 + np.arange(n_f)[None, :]                # [1, F]
    xi = (x - TX[0]) // 2
    after_mask = ~_low_mask(ks - fs)                 # bullet ages j >= k - f (chute open)
    y_open = y_now + M.FREE_VY * ks
    t_land = ks + ((M.GROUND_BODY_Y if ground_y is None else ground_y) - y_open) / M.CANOPY_VY
    land_mask = _low_mask(np.floor(t_land - 1 - fs).astype(np.int64) + 1)  # meet before landing
    pre, chute_bits, body_after = _phase_bits(_BITS, lane, xi, y_now, ks, fs, after_mask, land_mask)
    if chute_only:
        # scenarios where the bullet enters the chute box (at a drawn position)
        # before it would touch the body in either phase
        jc, jb = _lowest(chute_bits), _lowest(pre | body_after)
        hit = (chute_bits != 0) & (jc < jb)
        q_c = Q_KILL * CANOPY_SHARE if PHIT == "calibrated" else P_DRAWN
        return p, np.where(hit, q_c, 0.0), fs + np.where(hit, jc, 0)
    drawn = pre | chute_bits | body_after
    spre, schute, sbody = _phase_bits(_BITS_SWEPT, lane, xi, y_now, ks, fs, after_mask, land_mask)
    swept = spre | schute | sbody
    # per scenario: probability this bullet kills -- drawn-position contact is
    # near-certain, between-tick-only contact is a coin flip weighted by data
    q_d, q_s = (Q_KILL, Q_SWEPT) if PHIT == "calibrated" else (P_DRAWN, P_SWEPT_ONLY)
    q = np.where(drawn != 0, q_d, np.where(swept != 0, q_s, 0.0))
    bits = np.where(drawn != 0, drawn, swept)
    j = _lowest(bits)
    return p, q, fs + np.where(q > 0, j, 0)


def _calibrated_option(x, y_now, s_now, pos, cur_pos, clamped, clamp, moves, f0, lane, chute_only, ground_y):
    """PHIT="calibrated": P over the chute scenarios for every single spawn
    tick and (lazily) every pair of the planned bullets, timing slip
    averaged, with the stop bullet of the move to `pos` (which comes first).
    -> (p [K], single [N_F], pair() -> [N_F, N_F], q [K, N_F] unslipped (for
    hit masks), meet [K, N_F]).

    Per scenario, with S = the stop bullet's kill (crush) probability and
    A = its any-contact probability, a plan of bullets with per-bullet
    probabilities q_a, q_b scores  S + (1 - A) * (1 - (1 - q_a)(1 - q_b)),
    summed over scenarios as  p.S + w.q_a + w.q_b - (w * q_a).q_b  with
    w = p * (1 - A) -- matrix products instead of a [K, F, F] array."""
    stop = moves + STOP_BASE if (moves and cur_pos is not None) else None
    # the stop press happens at pos before any clamp: its own lane, usually the planned one
    s_lane = M.lane_id(pos, cur_pos, clamped) if stop is not None else None
    lo = f0 + min(SLIP)
    hi = f0 + N_F - 1 + max(SLIP)
    shared = stop is not None and s_lane == lane
    if shared:  # one table lookup covers the stop bullet too (its crush chance, under chute_only)
        lo = min(lo, stop + min(STOP_SLIP))
        hi = max(hi, stop + max(STOP_SLIP))
    p, q, meet = hit_matrix(x, y_now, s_now, lane, lo, chute_only, ground_y, n_f=hi - lo + 1)
    off = f0 - lo
    win = {d: q[:, off + d:off + d + N_F] for d in SLIP}          # spawn f0+i slipped by d
    base, w = 0.0, p
    if stop is not None:
        s_lo = stop + min(STOP_SLIP)
        n = max(STOP_SLIP) - min(STOP_SLIP) + 1

        def stop_p(cut):
            if shared:
                return sum(wt * q[:, stop + d - lo] for d, wt in STOP_SLIP.items())          # [K]
            _, qs, _ = hit_matrix(x, y_now, s_now, s_lane, s_lo, cut, ground_y, n_f=n)
            return sum(wt * qs[:, d - min(STOP_SLIP)] for d, wt in STOP_SLIP.items())
        s_kill = stop_p(chute_only)
        if chute_only:
            # crush: any contact of the stop bullet settles it, a canopy-first one as a crush
            _, qs, _ = hit_matrix(x, y_now, s_now, s_lane, s_lo, False, ground_y, n_f=n)
            s_any = sum(wt * qs[:, d - min(STOP_SLIP)] for d, wt in STOP_SLIP.items())
        else:
            s_any = s_kill
        base, w = float(p @ s_kill), p * (1 - s_any)
    single = base + sum(wt * (w @ win[d]) for d, wt in SLIP.items())

    def pair():
        u = np.full((N_F, N_F), base)
        for d, wt in SLIP.items():
            Q = win[d]
            a = w @ Q
            u += wt * (a[:, None] + a[None, :] - (Q * w[:, None]).T @ Q)
        return u
    return p, single, pair, q[:, off:off + N_F], meet[:, off:off + N_F]


def plan_free(x, y_now, s_now, cur_pos, clamped=False, min_p=0.0, allow_pair=True, chute_only=False,
              ground_y=None):
    """Best shot at a free-faller. Considers every barrel position (and the
    clamped limit lanes), every reachable spawn tick, and pairs of spawn
    ticks on the same lane (one bullet for each likely phase).
    -> dict(pos, clamp, spawns, p_hit, meet_last, p_single) or None."""
    if PHIT == "calibrated":
        return _plan_free_calibrated(x, y_now, s_now, cur_pos, clamped, min_p, allow_pair, chute_only, ground_y)
    best = None
    options = [(pos, False) for pos in range(M.N_POS)] + [(pos, True) for pos in M.SIGHT_LANE_ID]
    for pos, clamp in options:
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        f0 = _earliest(moves, clamp)
        lane = pos if clamp else M.lane_id(pos, cur_pos, clamped)
        p, q, meet = hit_matrix(x, y_now, s_now, lane, f0, chute_only, ground_y)
        hit = q > 0
        single = p @ q                                             # [F]
        fi = int(np.argmax(single))
        cand = [(float(single[fi]), [fi])]
        if allow_pair and single[fi] < 0.9:
            # two bullets: the scenario is covered unless both fail
            u = (p[:, None, None] * (1 - (1 - q[:, :, None]) * (1 - q[:, None, :]))).sum(0)  # [F, F]
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


def _plan_free_calibrated(x, y_now, s_now, cur_pos, clamped, min_p, allow_pair, chute_only, ground_y):
    """plan_free under PHIT="calibrated": same search and tie-breaks, the
    calibrated per-scenario probabilities (stop bullet, slip, fitted q)."""
    best = None
    options = [(pos, False) for pos in range(M.N_POS)] + [(pos, True) for pos in M.SIGHT_LANE_ID]
    for pos, clamp in options:
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        f0 = _earliest(moves, clamp)
        lane = pos if clamp else M.lane_id(pos, cur_pos, clamped)
        p, single, pair, q, meet = _calibrated_option(x, y_now, s_now, pos, cur_pos, clamped, clamp, moves,
                                                      f0, lane, chute_only, ground_y)
        hit = q > 0
        fi = int(np.argmax(single))
        cand = [(float(single[fi]), [fi])]
        if allow_pair and single[fi] < 0.9:
            u = pair()                                             # [F, F]
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
