"""Frame-by-frame review of crush shots (bullets aimed at a canopy trooper
that hangs over a landed one, ctx.over_landed) in recorded games.

    python3 -m paratrooper.tools.review_crush <run_dir> [<run_dir> ...]

Needs game_<t0>.jsonl + game_<t0>.npz (paratrooper.run --record). For each
crush bullet:
  * the bullet is found in the frames on its own lane lattice (that pins
    the telemetry tick of every frame: frame shows path index j <-> tick
    spawn + j), and followed until it is gone;
  * the target trooper is followed in its column;
  * outcome from the frames after: canopy burst and the trooper keeps
    falling (canopy kill), whole trooper burst (body kill), or nothing
    happened to it (survived) -- and whether it was already dead / its
    canopy already gone before our bullet got there (something else hit it);
  * where the bullet arrived relative to the trooper's body top (native px:
    canopy kills need rows <= -3, see sim/collision.py), and where the
    PLANNER thought the trooper would be at that tick (ctx.y_obs + 4 px/tick
    from the observation tick) -- the state-estimation error;
  * spawn tick vs plan (fire timing).
"""
import collections
import glob
import json
import os
import sys

import numpy as np

from ..geometry import CYAN
from ..perception import sprites as S
from ..physics import model as M

NL = [((sx // 2, (sy - 1) // 2), (vx // 2, vy // 2)) for (sx, sy), (vx, vy) in M.LANES]


def _native(gx, gy):
    return gx // 2, (gy - 1) // 2


def review_game(jsonl, npz):
    recs = [json.loads(l) for l in open(jsonl)]
    trs = {r["id"]: r for r in recs if r.get("type") == "trooper"}
    shots = [r for r in recs if r.get("type") == "bullet" and r.get("found")
             and (r.get("ctx") or {}).get("kind") == "trooper" and r["ctx"].get("over_landed")]
    if not shots:
        return []
    z = np.load(npz)
    F, T_ = z["frames"], z["t"]
    frame_tick = T_ / M.TICK_S                     # frame times are seconds since t0 (run.py)
    det = {}

    def D(k):
        if k not in det:
            det[k] = S.detect(F[k])
        return det[k]
    dots = [None] * len(F)

    def dots_at(k):
        if dots[k] is None:
            dots[k] = {_native(gx, gy) for gx, gy in D(k).dots}
        return dots[k]

    out = []
    for r in shots:
        c = r["ctx"]
        tr = trs.get(c.get("target"))
        lane = r["lane"]
        (sx, sy), (vx, vy) = NL[lane]
        p0 = _native(r["path"][0][1], r["path"][0][2])
        j0 = (p0[1] - sy) // vy
        spawn = r["path"][0][0] - j0                   # telemetry tick of path index 0
        # 1. find the bullet's first sighting in the frames
        # near the frames whose time matches the telemetry tick (the lane
        # lattice is shared by every bullet on that lane)
        near = np.flatnonzero(np.abs(frame_tick - r["path"][0][0]) <= 3)
        f0 = next((int(k) for k in near if p0 in dots_at(int(k))), None)
        if f0 is None or tr is None:
            out.append(dict(shot=r["shot"], verdict="not found in frames"))
            continue
        # 2. follow it: frame -> lattice index j (tick = spawn + j)
        seen = {f0: j0}
        j_last, f_last = j0, f0
        for k in range(f0 + 1, min(f0 + 40, len(F))):
            js = [j for j in range(j_last, j_last + 3)
                  if (sx + vx * j, sy + vy * j) in dots_at(k)]
            if js:
                j_last, f_last = js[0], k
                seen[k] = js[0]
            elif k - f_last > 4:
                break
        nxt = (sx + vx * (j_last + 1), sy + vy * (j_last + 1))
        exited = not (0 <= nxt[0] < 320 and 0 <= nxt[1] < 200)
        k_abs = spawn + j_last + 1                     # tick the bullet died (if it did)
        # 3. the target trooper in its column, per frame around the kill
        tx = tr["x"] // 2

        def body_in(k):
            b = [(x, y) for x, y in (_native(gx, gy) for gx, gy in D(k).bodies) if abs(x - tx) <= 1]
            cans = [(x, y) for x, y in (_native(gx, gy) for gx, gy in D(k).canopies) if abs(x - tx) <= 1]
            return b, cans
        b_last, can_last = body_in(f_last)
        canopy_at_last = bool(can_last)
        # trooper body top when the bullet died: from the last frame it was seen, + 2 px/tick
        y_body = None
        if b_last:
            y_body = min(b_last, key=lambda p: abs(p[1] - nxt[1]))[1] + 2   # one tick later
        # 4. outcome: frames 2..6 after the bullet's last sighting
        after = list(range(f_last + 2, min(f_last + 8, len(F))))
        canopy_after = any(body_in(k)[1] for k in after)
        falling = False
        if y_body is not None:
            for k in after:
                if (F[k][y_body: y_body + 60, max(0, tx - 1): tx + 5] == CYAN).sum() >= 8:
                    falling = True
                    break
        if exited:
            outcome = "bullet flew off screen"
        elif not canopy_at_last:
            outcome = "canopy already gone before our bullet"
        elif canopy_after:
            outcome = "trooper survived (bullet died elsewhere)"
        else:
            outcome = "canopy kill" if falling else "body kill"
        # 5. where did the bullet arrive, vs where the planner expected the trooper
        arr_r = nxt[1] - y_body if y_body is not None else None
        tick_obs = c["plan_spawn_tick"] - c["spawn"] if c.get("plan_spawn_tick") is not None else None
        y_pred = None
        if tick_obs is not None and c.get("y_obs") is not None and c.get("state") == "canopy":
            y_pred = _native(0, c["y_obs"] + M.CANOPY_VY * (k_abs - tick_obs))[1]
        out.append(dict(shot=r["shot"], target=tr["id"], fate=tr.get("fate"), lane=lane, outcome=outcome,
                        arrive_row=arr_r, arrive_col=(nxt[0] - tx) if y_body is not None else None,
                        y_err=(y_body - y_pred) if (y_body is not None and y_pred is not None) else None,
                        spawn_err=(r["first_tick"] - c["plan_spawn_tick"]) if c.get("plan_spawn_tick") is not None else None,
                        part=c.get("part"), p_hit=c.get("p_hit"), clamp=c.get("clamp"), state=c.get("state")))
    return out


def main(dirs):
    rows = []
    for d in dirs:
        for j in sorted(glob.glob(f"{d}/game_*.jsonl")):
            npz = j[:-6] + ".npz"
            if os.path.exists(npz):
                rows += [dict(r, game=os.path.basename(j)) for r in review_game(j, npz)]
    n = len(rows)
    print(f"{n} crush bullets in {len({r['game'] for r in rows})} recorded games\n")
    for st in ("free", "canopy"):
        sub = [r for r in rows if r.get("state") == st]
        print(f"OUTCOME (from the frames), shots planned while the trooper was {st}: {len(sub)}")
        for k, v in collections.Counter(r.get("outcome", r.get("verdict")) for r in sub).most_common():
            print(f"  {k:42s} {v:4d} ({v / max(len(sub), 1):.0%})")
    for oc in ("canopy kill", "body kill"):
        sub = [r for r in rows if r.get("outcome") == oc and r["arrive_row"] is not None and r.get("state") == "canopy"]
        if not sub:
            continue
        print(f"\n{oc.upper()} (canopy-state shots): bullet arrival row relative to the body top (canopy region <= -3)")
        print("  " + str(sorted(collections.Counter(r["arrive_row"] for r in sub).items())))
        print(f"  trooper y error (actual - planner's, native px): {sorted(collections.Counter(r['y_err'] for r in sub if r['y_err'] is not None).items())}")
        print(f"  spawn tick - plan: {sorted(collections.Counter(r['spawn_err'] for r in sub if r['spawn_err'] is not None).items())}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
