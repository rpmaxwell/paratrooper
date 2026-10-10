"""How often does a planned trooper shot actually hit? Calibration data.

Every trooper engagement the policy starts (a Job: move -> fire its planned
bullets) is logged with the planner's p_hit in the bullet ctx. This module
joins those plans with what happened, so the planner's probabilities can
be checked and refitted:

    engagements(games) -> DataFrame, one row per engagement

An engagement is every bullet aimed at one target from one plan (ctx.t_obs),
plus the stop-press bullet the turret fired on the way there (ctx.kind
"stop"/"clamp" with the same target): stopping the rotation always fires,
down the planned lane a few ticks early, and that bullet makes a large share
of the kills. Columns:

  game, target, t_obs, state ("free"/"canopy" at plan time), part
  ("free", "canopy", "chute_only", "body"), p_hit (planner), n_planned,
  n_stop (stop/clamp bullets of this engagement), over_landed,
  y_obs, s_obs (ticks since the drop at plan time), tick0 (game tick of the
  plan's tick 0), plan_spawns / spawns (planned / actual bullet spawn ticks,
  game ticks), lane, x, first_y,
  open_tick (true chute-opening tick, if seen), fate, last_tick, last_y,
  killer (shot id the world credited), outcome:
     "planned"  one of the planned bullets killed it
     "stop"     the stop/clamp bullet killed it (planned ones never mattered)
     "other"    killed by a bullet of some other engagement
     "landed"   it landed
     "miss"     none of ours killed it here, and it died / was lost later
  hit (outcome in planned/stop), hit_planned, canopy_kill (chute_killed by
  this engagement), later (a later engagement on the same target exists)
"""
import collections
import json

import numpy as np
import pandas as pd

from ..physics import model as M


def _records(path):
    out = []
    for line in open(path):
        if '"bullet"' in line or '"trooper"' in line or '"game_start"' in line or '"press"' in line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def engagements(games):
    rows = []
    for gi, g in enumerate(games):
        recs = _records(g.path)
        variant = next((r.get("heli_box") for r in recs if r.get("type") == "game_start"), None)
        trs = {r["id"]: r for r in recs if r.get("type") == "trooper" and r.get("y_by_tick")}
        presses = [r for r in recs if r.get("type") == "press"]
        press_i = {r["shot"]: i for i, r in enumerate(presses)}
        by_target = collections.defaultdict(list)
        for r in recs:
            c = r.get("ctx") or {}
            if r.get("type") == "bullet" and c.get("target") in trs and c.get("kind") in ("trooper", "stop", "clamp") \
                    and (c.get("kind") == "trooper" or c.get("why") == "trooper"):
                by_target[c["target"]].append(r)
        for tid, bullets in by_target.items():
            tr = trs[tid]
            yb = tr["y_by_tick"]
            open_tick = _open_tick(yb)
            # split into engagements: a stop/clamp bullet belongs to the next plan
            jobs, pending = [], []
            for b in sorted(bullets, key=lambda b: (b.get("first_tick") or 10 ** 9, b["shot"])):
                c = b["ctx"]
                if c["kind"] != "trooper":
                    pending.append(b)
                    continue
                if jobs and jobs[-1]["t_obs"] == c.get("t_obs"):
                    jobs[-1]["planned"].append(b)
                else:
                    jobs.append(dict(t_obs=c.get("t_obs"), planned=[b], stop=pending))
                    pending = []
            killer = tr.get("fate_evidence") if isinstance(tr.get("fate_evidence"), int) else None
            fate = tr.get("fate")
            if killer is None and fate != "landed":
                killer = _infer_killer(tr, bullets)
            for i, j in enumerate(jobs):
                c = j["planned"][0]["ctx"]
                planned_ids = {b["shot"] for b in j["planned"]}
                stop_ids = {b["shot"] for b in j["stop"]}
                if killer in planned_ids:
                    outcome = "planned"
                elif killer in stop_ids:
                    outcome = "stop"
                elif fate == "landed" and i == len(jobs) - 1:
                    outcome = "landed"
                elif killer is not None and i == len(jobs) - 1:
                    outcome = "other"
                else:
                    outcome = "miss"
                tick0 = c["plan_spawn_tick"] - c["spawn"] if c.get("plan_spawn_tick") is not None else None
                drop_y = tr["first_y"] if tr["first_y"] <= 60 else 33
                # turret position the plan was made from: where the previous press left it
                first_shot = (j["stop"] or j["planned"])[0]["shot"]
                pi = press_i.get(first_shot)
                prev = presses[pi - 1] if pi else None
                rows.append(dict(
                    game=gi, variant=variant, target=tid, t_obs=j["t_obs"], state=c.get("state"), part=c.get("part"),
                    p_hit=c.get("p_hit"), n_planned=len(j["planned"]), n_stop=len(j["stop"]),
                    over_landed=bool(c.get("over_landed")), y_obs=c.get("y_obs"),
                    s_obs=max(0, ((c.get("y_obs") or drop_y) - drop_y) // M.FREE_VY), tick0=tick0,
                    plan_spawns=[(tick0 + b["ctx"]["spawn"]) if tick0 is not None else None for b in j["planned"]],
                    spawns=[b.get("first_tick") for b in j["planned"]],
                    found=[bool(b.get("found")) for b in j["planned"]],
                    lane=j["planned"][0].get("lane"), pos=c.get("pos"), clamp=c.get("clamp"),
                    cur_pos=prev.get("barrel") if prev else None, cur_clamped=bool(prev.get("clamped")) if prev else False,
                    stop_spawns=[b.get("first_tick") for b in j["stop"]],
                    stop_lanes=[b.get("lane") for b in j["stop"]],
                    x=tr["x"], first_y=tr["first_y"], first_tick=tr["first_tick"], open_tick=open_tick,
                    fate=fate, last_tick=yb[-1][0], last_y=yb[-1][1], killer=killer, outcome=outcome,
                    later=i < len(jobs) - 1, job_index=i, y_by_tick=yb))
    D = pd.DataFrame(rows)
    D["hit"] = D.outcome.isin(["planned", "stop"])
    D["hit_planned"] = D.outcome == "planned"
    D["canopy_kill"] = D.hit & (D.fate == "chute_killed")
    return D


def _infer_killer(tr, bullets, max_dist=40):
    """The world credits no bullet for many kills (fate "lost", or evidence
    "unexplained"). Credit one of OUR bullets aimed at it if it ended (not off
    screen) within a tick or two of the trooper's last sighting, close to it."""
    last_tick, last_y = tr["y_by_tick"][-1]
    best = None
    for b in bullets:
        e = b.get("end") or {}
        if e.get("kind") in (None, "left_screen") or e.get("tick") is None or not e.get("point"):
            continue
        if not -1 <= e["tick"] - last_tick <= 3:
            continue
        d = abs(e["point"][0] - tr["x"]) + abs(e["point"][1] - last_y)
        if d <= max_dist and (best is None or d < best[0]):
            best = (d, b["shot"])
    return None if best is None else best[1]


def _open_tick(yb):
    from ..physics import chute
    return chute.true_open_tick(yb)


def reliability(D, by=("state", "part"), bins=(0, .5, .6, .7, .8, .9, .95, 1.001), outcome="hit"):
    """Planned p_hit vs realized rate, per group and p_hit bin."""
    D = D[D.p_hit.notna()].copy()
    D["p_bin"] = pd.cut(D.p_hit.astype(float), list(bins), right=False)
    return (D.groupby(list(by) + ["p_bin"], observed=True)
             .agg(n=("p_hit", "size"), planned=("p_hit", "mean"), realized=(outcome, "mean"),
                  planned_bullets=("hit_planned", "mean"), stop_bullet=("outcome", lambda s: (s == "stop").mean()),
                  landed=("outcome", lambda s: (s == "landed").mean()))
             .round(3))


# ---- engagement hit probability ---------------------------------------------------
# P(an engagement kills its target), from first principles: every bullet of
# the engagement (the stop-press bullet included), every chute-opening
# scenario (physics/chute.py), a shared timing slip (the plan's tick 0 is off
# by a tick 17% of the time -- both bullets of a pair slip together), and the
# game's hitbox (physics/tables.py; "fitted_parts", the only variant left).
# The first bullet that touches the trooper decides; it kills with
# probability q (drawn-position contact is not quite certain), else the next
# one gets its chance.
_BITS_CACHE = {}
SLIP = {-1: 0.01, 0: 0.82, 1: 0.17}   # actual spawn - planned spawn, measured on 11.5k planned bullets
HORIZON = 48


def _bits(variant, swept=False):
    key = (variant, swept)
    if key not in _BITS_CACHE:
        if variant != "fitted_parts":
            raise ValueError(f"hitbox variant {variant!r}: only the game's box (fitted_parts) is tabled now")
        from ..physics.tables import tables
        _BITS_CACHE[key] = tables()["trooper_bits_swept" if swept else "trooper_bits"]
    return _BITS_CACHE[key]


def _lowest(bits):
    """lowest set bit index per element (99 if none)."""
    bits = np.asarray(bits, np.uint64)
    out = np.full(bits.shape, 99, np.int64)
    for j in range(40):
        m = (out == 99) & ((bits >> np.uint64(j)) & np.uint64(1)).astype(bool)
        out[m] = j
    return out


def _contacts(B, lane, x, y_spawn, part_i, vy_i):
    """bitmask of bullet ages touching a box (part, speed) whose body top is
    y_spawn (array) when the bullet spawns."""
    from ..physics.tables import TX, TY
    xi = (x - TX[0]) // 2
    yi = (np.asarray(y_spawn) - TY[0]) // 2
    out = np.zeros(np.shape(y_spawn), np.uint64)
    ok = (yi >= 0) & (yi < len(TY)) & (0 <= xi < len(TX))
    out[ok] = B[lane, part_i, vy_i, xi, yi[ok]]
    return out


def _low_mask(n):
    n = np.clip(np.asarray(n), 0, 40).astype(np.uint64)
    return np.where(n >= 40, np.uint64((1 << 40) - 1), (np.uint64(1) << n) - np.uint64(1))


def bullet_contacts(x, y_now, state, s_now, bullets, variant="fitted_parts", ground_y=None, horizon=HORIZON,
                    swept=False):
    """For each chute scenario k (free-fallers; one scenario for canopy
    troopers) and bullet (lane, spawn tick relative to the observation):
    -> (p[K], tick[K, n], canopy_first[K, n]) where tick is the tick of
    first contact (large = none) and canopy_first whether that contact is
    the canopy (else the body)."""
    from ..physics import chute
    B = _bits(variant, swept)
    gy = M.GROUND_BODY_Y if ground_y is None else ground_y
    lanes = np.array([b[0] for b in bullets])
    fs = np.array([b[1] for b in bullets], np.int64)[None, :]          # [1, n]
    NONE = 10 ** 6
    if state == "canopy":
        p = np.ones(1)
        ks = np.array([[-10 ** 3]])                                    # opened long ago
    else:
        p = chute.open_distribution(s_now, y_now, horizon)
        ks = np.arange(1, horizon + 1)[:, None]                        # [K, 1]
    K = len(p)
    tick = np.full((K, len(bullets)), NONE, np.int64)
    canopy_first = np.zeros((K, len(bullets)), bool)
    for bi, lane in enumerate(lanes):
        f = fs[:, bi:bi + 1]                                          # [1, 1]
        if state == "canopy":
            y_open, k_open = y_now, 0
            y0c = np.full((K, 1), y_now) + M.CANOPY_VY * f
            pre = np.zeros((K, 1), np.uint64)
            after = np.full((K, 1), np.uint64((1 << 40) - 1))
            t_land = (gy - y_now) / M.CANOPY_VY
        else:
            y0f = y_now + M.FREE_VY * f
            pre = _contacts(B, lane, x, np.broadcast_to(y0f, (K, 1)), 1, 1) & _low_mask(ks - f)
            y0c = y_now + M.FREE_VY * ks + M.CANOPY_VY * (f - ks)
            after = ~_low_mask(ks - f)
            t_land = ks + (gy - (y_now + M.FREE_VY * ks)) / M.CANOPY_VY
        land = _low_mask(np.floor(t_land - 1 - f).astype(np.int64) + 1)
        can = _contacts(B, lane, x, y0c, 0, 0) & after & land
        body = (pre | (_contacts(B, lane, x, y0c, 1, 0) & after & land))
        jc, jb = _lowest(can), _lowest(body)
        j = np.minimum(jc, jb)
        hit = j < 99
        tick[:, bi] = np.where(hit, f + j, NONE)[:, 0]
        canopy_first[:, bi] = (jc < jb)[:, 0]
    return p, tick, canopy_first


def engagement_p(x, y_now, state, s_now, bullets, variant="fitted_parts", q=0.94, slip=SLIP,
                 ground_y=None, slip_stop=False, q_swept=0.0, canopy_share=1.0):
    """-> (P(kill), P(the kill is a canopy kill)) for the bullets
    [(lane, spawn tick rel. to the observation, is_stop), ...]. A timing slip
    shifts every planned bullet (and the stop bullet if slip_stop). Each
    contact kills with q (q_swept if the bullet only crosses the box between
    drawn positions); of the kills whose first contact is the canopy, a
    share canopy_share are canopy kills (crushes), the rest body kills."""
    pk = pc = 0.0
    for d, w in slip.items():
        bl = [(ln, f + (d if (slip_stop or not st) else 0)) for ln, f, st in bullets]
        p, tick, cf = bullet_contacts(x, y_now, state, s_now, bl, variant, ground_y)
        qq = np.full(tick.shape, q)
        if q_swept:
            # contact only between drawn positions: a weaker chance (planner: prob_plan.Q_SWEPT)
            _, ts, cs = bullet_contacts(x, y_now, state, s_now, bl, variant, ground_y, swept=True)
            only = (tick >= 10 ** 6) & (ts < 10 ** 6)
            tick, cf = np.where(only, ts, tick), np.where(only, cs, cf)
            qq = np.where(only, q_swept, qq)
        order = np.argsort(tick, axis=1)
        t_sorted = np.take_along_axis(tick, order, 1)
        c_sorted = np.take_along_axis(cf, order, 1)
        q_sorted = np.take_along_axis(qq, order, 1)
        alive = np.ones(len(p))
        kill = np.zeros(len(p))
        can = np.zeros(len(p))
        for i in range(tick.shape[1]):
            touch = t_sorted[:, i] < 10 ** 6
            hit = alive * q_sorted[:, i] * touch
            kill += hit
            can += hit * c_sorted[:, i] * canopy_share
            alive = alive - hit
        pk += w * float(p @ kill)
        pc += w * float(p @ can)
    return pk, pc


# The stop press: every move ends with an Up press that fires down the
# destination lane. Its bullet spawns `moves + 4` ticks after the
# observation the plan was made from (5.5k engagements: median exactly
# that for 1..18 moves; spread below), so it is part of every plan that moves.
STOP_SLIP = {-1: 0.05, 0: 0.61, 1: 0.29, 2: 0.05}
STOP_BASE = 4


def stop_bullet(pos, cur_pos, clamped=False):
    """(lane, spawn tick rel. to the observation) of the stop-press bullet a
    move from cur_pos to pos fires, or None (no move, no stop bullet)."""
    if cur_pos is None or pos == cur_pos:
        return None
    return M.lane_id(pos, cur_pos, clamped), abs(pos - cur_pos) + STOP_BASE


# Fitted on 5.5k replanned engagements (A/B + recent runs, 2026-10): kills per
# drawn contact by the target's state at plan time, kills only between drawn
# positions (frame data), and the share of canopy-first kills that take the
# canopy (crushes): 0.72 fits free-fall bets, ~0.8 canopy shots (n=43).
Q_KILL = {"free": 0.90, "canopy": 0.80}
Q_SWEPT = 0.12
CANOPY_SHARE = 0.75


def plan_p(x, y_now, state, s_now, pos, spawns, cur_pos, clamped=False, clamp=False, ground_y=None,
           variant="fitted_parts", q=None, q_swept=None, canopy_share=None, with_stop=True):
    """P(kill), P(first kill is the canopy) for a plan (pos, planned spawn
    ticks) made from barrel cur_pos -- the stop bullet included, timing
    slips averaged over. Drop-in for the planner's p_hit when comparing
    options."""
    q = Q_KILL[state] if q is None else q
    q_swept = Q_SWEPT if q_swept is None else q_swept
    canopy_share = CANOPY_SHARE if canopy_share is None else canopy_share
    lane = pos if clamp else M.lane_id(pos, cur_pos, clamped)
    planned = [(lane, f, False) for f in spawns]
    sb = stop_bullet(pos, cur_pos, clamped) if with_stop else None
    if sb is None:
        return engagement_p(x, y_now, state, s_now, planned, variant, q, SLIP, ground_y, q_swept=q_swept,
                            canopy_share=canopy_share)
    pk = pc = 0.0
    for d, w in STOP_SLIP.items():
        bl = planned + [(sb[0], sb[1] + d, True)]
        a, b = engagement_p(x, y_now, state, s_now, bl, variant, q, SLIP, ground_y, q_swept=q_swept,
                            canopy_share=canopy_share)
        pk += w * a
        pc += w * b
    return pk, pc


def main(argv):
    """python3 -m paratrooper.sim.phit [run_dir ...]: planner p_hit vs outcome."""
    from . import telemetry
    games = telemetry.load(argv or None, root="..")
    D = engagements(games)
    pd.set_option("display.width", 200)
    print(f"{len(D)} engagements in {D.game.nunique()} games\n")
    print(D.groupby(["state", "part"]).outcome.value_counts(normalize=True).unstack(fill_value=0).round(3))
    print()
    print(reliability(D[D.outcome != "other"]))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
