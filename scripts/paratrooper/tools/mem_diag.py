"""Why do troopers get away? Exact answers from the game's memory.

    python3 -m paratrooper.tools.mem_diag <run_dir> [<run_dir> ...]

Needs games played with `paratrooper.run ... --mem` (game_<t0>.jsonl +
game_<t0>_mem.npz). From memory (perception/memstate.py) every trooper's
exact path (top row per tick, chute opening, end) and every bullet's exact
spawn tick, direction and path. Telemetry ticks are aligned to the game's
tick counter through bullet positions. Then, per trooper the bot engaged:

  hit         one of our bullets killed it
  geometry    with the game's own collision rule, the bullets fired could
              not have hit the trooper's real path; split by what would have
              fixed it: the planned spawn tick (timing slip), the planned
              barrel position (barrel error), or neither (chute opened /
              moved differently from the plan's assumption)
  consumed    a bullet's path met the trooper but the bullet died first
              (another target took it)
and per trooper that landed without a bullet ever aimed at it, how long it
was in the air and what the telemetry says about why (busy, never feasible).
"""
import collections
import glob
import json
import os
import sys

import numpy as np

from ..perception.memstate import MemState
from ..physics import model as M


# ---- memory timelines ------------------------------------------------------------
def timelines(npz):
    z = np.load(npz)
    mem, lo = z["mem"], int(z["lo"])
    ticks, troop, bullets, barrel = [], [], [], []
    last = None
    for snap in mem:
        ms = MemState(snap, lo)
        tk = ms.tick
        if tk == last:
            continue
        last = tk
        ticks.append(tk)
        troop.append({x: (y, canopy, ff) for x, y, canopy, ff in ms.troopers()})
        bullets.append(ms.bullets())
        barrel.append(ms.barrel())
    return np.array(ticks), troop, bullets, barrel


def trooper_paths(ticks, troop):
    """x -> list of lives: [(tick, body y, canopy, free_fall), ...]"""
    lives = collections.defaultdict(list)
    open_ = {}
    for tk, tr in zip(ticks, troop):
        for x, st in tr.items():
            if x not in open_:
                open_[x] = []
            open_[x].append((int(tk),) + st)
        for x in [x for x in open_ if x not in tr]:
            lives[x].append(open_.pop(x))
    for x, life in open_.items():
        lives[x].append(life)
    return lives


def bullet_paths(ticks, bullets):
    """list of (spawn tick, direction, [(tick, x, y)]) -- slots don't keep a
    bullet's identity, so a bullet is followed by its direction step."""
    out, live = [], []
    for tk, bl in zip(ticks, bullets):
        tk = int(tk)
        used = set()
        nxt = []
        for b in live:
            _, d, path = b
            t1, x1, y1 = path[-1]
            m = next((i for i, (x, y, dd) in enumerate(bl) if i not in used and dd == d and tk - t1 <= 2
                      and abs(x - x1) <= 48 * (tk - t1) and 0 <= y1 - y <= 40 * (tk - t1)), None)
            if m is None:
                out.append(b)
            else:
                used.add(m)
                path.append((tk, bl[m][0], bl[m][1]))
                nxt.append(b)
        for i, (x, y, d) in enumerate(bl):
            if i not in used:
                nxt.append((tk, d, [(tk, x, y)]))
        live = nxt
    return out + live


# ---- the game's trooper hit test (collision routine, CS:108C) -------------------------
def trooper_hit(ux, uy, bx, by, canopy):
    """Bullet drawn at (ux, uy) vs trooper body top-left (bx, by), game coords.
    -> None / 'canopy' / 'body'."""
    c, t = ux >> 3, (bx >> 3) + 1
    if not t - 2 <= c <= t:
        return None
    r = ((uy - 1) >> 1) - ((by - 1) >> 1) + 16        # rows below the record's top row
    if not 0 <= r <= 22:
        return None
    if canopy:
        return "canopy" if r <= 13 else "body"
    return "body" if r >= 10 else None


def diagnose(run_dirs):
    rows = []
    for d in run_dirs:
        for jf in sorted(glob.glob(os.path.join(d, "game_*.jsonl"))):
            mf = jf[:-6] + "_mem.npz"
            if not os.path.exists(mf):
                continue
            rows += _diagnose_game(jf, mf)
    return rows


def _diagnose_game(jf, mf):
    recs = [json.loads(l) for l in open(jf)]
    ticks, troop, bullets, barrel = timelines(mf)
    lives = trooper_paths(ticks, troop)
    bpaths = bullet_paths(ticks, bullets)
    # align telemetry ticks to memory ticks through trooper drops: the same
    # column, a life starting near the telemetry trooper's first tick
    first = int(ticks[0])
    off = collections.Counter()
    for r in recs:
        if r.get("type") == "trooper":
            for life in lives.get(r["x"], []):
                d = life[0][0] - r["first_tick"]
                if abs(d - first) < 400:
                    off[d] += 1
    if not off:
        return []
    offset = off.most_common(1)[0][0]
    shots = [r for r in recs if r.get("type") == "bullet"]
    out = []
    for tr in (r for r in recs if r.get("type") == "trooper"):
        x = tr["x"]
        t0 = tr["first_tick"] + offset
        life = next((l for l in lives.get(x, []) if l[0][0] - 3 <= t0 <= l[-1][0] + 3), None)
        if life is None:
            continue
        mine = [s for s in shots if (s.get("ctx") or {}).get("target") == tr["id"]]
        mine_ids = {s["shot"] for s in mine}
        killer = tr.get("fate_evidence") if isinstance(tr.get("fate_evidence"), int) else None
        row = dict(game=os.path.basename(jf), id=tr["id"], x=x, fate=tr.get("fate"), n_shots=len(mine),
                   ours=killer in mine_ids,
                   drop=life[0][0], end=life[-1][0], open=next((tk for tk, y, c, ff in life if c), None),
                   end_y=life[-1][1], feasible_ever=tr.get("feasible_ever"), busy_skips=tr.get("busy_skips"))
        if not mine:
            row["verdict"] = "never engaged"
            out.append(row)
            continue
        if killer in mine_ids:
            row["verdict"] = "hit"
            out.append(row)
            continue
        # every bullet path that came out of this trooper's engagements
        verdict = "geometry"
        for s in mine:
            if not s.get("found") or s.get("first_tick") is None:
                continue
            sp = s["first_tick"] + offset
            cand = [b for b in bpaths if abs(b[0] - sp) <= 1]
            for b in cand:
                hit = _meets(b[2], life, x)
                if hit:
                    end_tick = b[2][-1][0]
                    verdict = "hit" if row["end"] <= hit[0] + 1 else ("consumed" if end_tick < hit[0] else "rule mismatch")
                    break
            if verdict != "geometry":
                break
        if verdict == "geometry":
            verdict = _why_geometry(mine, offset, bpaths, life, x, barrel, ticks)
        row["verdict"] = verdict
        out.append(row)
    return out


def _meets(path, life, x):
    at = {tk: (y, c) for tk, y, c, _ in life}
    for tk, ux, uy in path:
        for tt in (tk - 1, tk):
            st = at.get(tt)
            if st and trooper_hit(ux, uy, x, st[0], st[1]):
                return tk, trooper_hit(ux, uy, x, st[0], st[1])
    return None


def _lane_path(lane, spawn):
    (sx, sy), (vx, vy) = M.LANES[lane]
    return [(spawn + j, sx + vx * j, sy + vy * j) for j in range(40)
            if -2 <= sx + vx * j <= 640 and sy + vy * j >= -2]


def _why_geometry(mine, offset, bpaths, life, x, barrel, ticks):
    """The bullets fired couldn't meet the real path: would the PLAN have?"""
    for s in mine:
        c = s.get("ctx") or {}
        if c.get("kind") != "trooper" or c.get("plan_spawn_tick") is None or c.get("pos") is None:
            continue
        lane = M.lane_id(c["pos"], c["pos"], c.get("clamp", False)) if not c.get("clamp") else c["pos"]
        planned = _lane_path(lane, c["plan_spawn_tick"] + offset)
        actual_lane = s.get("lane")
        if _meets(planned, life, x):
            if actual_lane is not None and actual_lane != lane:
                return "geometry: barrel error"
            return "geometry: timing slip"
    return "geometry: trooper path differed from the plan (chute timing)"


def main(argv):
    rows = diagnose(argv)
    print(f"{len(rows)} troopers in {len({r['game'] for r in rows})} games")
    def outcome(r):
        return "landed" if r["fate"] == "landed" else ("killed by our shot" if r.get("ours") else "killed otherwise")
    c = collections.Counter((outcome(r), r["verdict"]) for r in rows)
    for (o, v), n in sorted(c.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {o:20s} {v:60s} {n}")
    nev = [r for r in rows if r["verdict"] == "never engaged" and r["fate"] == "landed"]
    if nev:
        air = [r["end"] - r["drop"] for r in nev]
        print(f"\nlanded, never engaged: {len(nev)}; ticks in the air: median {np.median(air):.0f}; "
              f"feasible_ever {sum(1 for r in nev if r['feasible_ever'])}/{len(nev)}; "
              f"busy_skips>0 {sum(1 for r in nev if r['busy_skips'])}/{len(nev)}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
