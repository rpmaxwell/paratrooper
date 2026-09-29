#!/usr/bin/env python3
"""Offline ground-truth audit of trooper shots.

Needs a run made with AUDIT_TROOPER_SHOTS=<n> (bomb_defense.py saves a
short frame clip around every planned trooper shot, plus shots_<t0>.json
with each Up press and what it was aimed at).

For every planned shot: find the bullet it fired, follow it tick by tick,
and classify each DRAWN bullet position against the target's actual
sprites -- dome (canopy pixels), gap (between dome and head: the strings),
body (body+head pixels), or clear -- then record what happened: canopy
gone, body gone, or both intact. Answers:
  * are the strings a hitbox, and are dome/body separate hitboxes?
  * does the game test collisions only where the bullet is drawn each
    tick, or along its path (our planner assumes along the path)?

Usage: python3 audit_trooper_shots.py <run_dir>
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

from bomb_model import components
from trooper_model import MASKS


def distinct(t, F):
    out, prev = [], None
    for i in range(len(t)):
        if prev is not None and np.array_equal(F[i], F[prev]):
            continue
        prev = i
        out.append((t[i], F[i]))
    return out


def sprites(f, x):
    """Canopy top-left and body top-left of the trooper in column x."""
    can = body = None
    for xs, ys in components(f[40:372] == 1):
        if xs.size == 232 and abs(int(xs.min()) + 8 - x) <= 1:
            can = (int(xs.min()), int(ys.min()) + 40)
        elif xs.size == 56 and int(xs.min()) == x:
            body = (int(xs.min()), int(ys.min()) + 40)
    return can, body


def dots(f):
    return [(int(xs.min()), int(ys.min())) for xs, ys in components(f[:372] == 3)
            if xs.size == 4 and xs.max() - xs.min() == 1]


def region(ux, uy, can, body):
    def touches(origin, mask):
        ox, oy = origin
        for by in (uy, uy + 1):
            for bx in (ux, ux + 1):
                r, c = by - oy, bx - ox
                if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]:
                    return True
        return False
    if can and touches(can, MASKS["canopy"]):
        return "dome"
    if body and touches((body[0], body[1] - 4), MASKS["body"]):
        return "body"
    if can and can[0] <= ux + 1 and ux <= can[0] + 23 and can[1] + 12 <= uy + 1 and uy <= can[1] + 27:
        return "gap"
    return "clear"


def lanes_for(pos):
    from bomb_model import BARREL, LIMIT_SIGHT_LANE
    out = [(tuple(BARREL[pos][2]), tuple(BARREL[pos][3]))]
    if pos in LIMIT_SIGHT_LANE:
        out.append(LIMIT_SIGHT_LANE[pos])
    return out


def audit(run_dir):
    rows = []
    for sf in sorted(glob.glob(f"{run_dir}/shots_*.json")):
        t0 = os.path.basename(sf)[6:-5]
        shots = [(t, c) for t, c in json.load(open(sf)) if c.get("kind") == "trooper"]
        clips = [np.load(p) for p in glob.glob(f"{run_dir}/evidence_trooper_shot_{t0}_*.npz")]
        for t_press, c in shots:
            clip = next((d for d in clips if d["t"][0] <= t_press <= d["t"][-1]), None)
            if clip is None:
                continue
            seq = [(t, f) for t, f in distinct(clip["t"], clip["frames"]) if t >= t_press - 0.01]
            D = [set(dots(f)) for _, f in seq]
            x = c["x"]
            # the bullet on this shot's lane: spawn + j*v, confirmed one tick later
            found = None
            for (sx, sy), v in lanes_for(c["pos"]):
                for i in range(min(8, len(seq) - 1)):
                    for jj in range(0, 4):
                        p = (sx + v[0] * jj, sy + v[1] * jj)
                        nxt = (p[0] + v[0], p[1] + v[1])
                        if p in D[i] and any(nxt in D[k] for k in range(i + 1, min(i + 4, len(seq)))):
                            found = (i, p, v)
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                rows.append(dict(c, verdict="bullet not found"))
                continue
            i, p, v = found
            last_i, last = i, p
            k = i + 1
            vanished = None
            while k < len(seq):
                nxt = (last[0] + v[0], last[1] + v[1])
                if nxt in D[k]:
                    last_i, last = k, nxt
                elif last in D[k]:
                    pass  # redraw within the same tick
                else:
                    vanished = k
                    break
                k += 1
            if vanished is None:
                rows.append(dict(c, v=v, verdict="left clip"))
                continue
            # collision point = where it would have been drawn this tick,
            # against the trooper as it was one tick on
            can, body = sprites(seq[last_i][1], x)
            vy = 4 if can else 8
            can1 = (can[0], can[1] + vy) if can else None
            body1 = (body[0], body[1] + vy) if body else None
            cp = (last[0] + v[0], last[1] + v[1])
            at = region(cp[0], cp[1], can1, body1)
            # swept: regions crossed between the last drawn point and cp
            crossed = []
            for sfrac in (0.25, 0.5, 0.75):
                q = (int(round(last[0] + v[0] * sfrac)), int(round(last[1] + v[1] * sfrac)))
                crossed.append(region(q[0], q[1], can, body))
            t_out = seq[vanished][0] + 0.2
            f_out = min(seq, key=lambda tf: abs(tf[0] - t_out))[1]
            can_o, body_o = sprites(f_out, x)
            outcome = ("canopy gone" if can and can_o is None and body_o is not None else
                       "body gone" if body and body_o is None else
                       "target intact" if (can_o or body_o) else "unclear")
            rows.append(dict(c, v=v, at=at, crossed=crossed, outcome=outcome,
                             had_canopy=can is not None, cp=cp, can1=can1))
    return rows


if __name__ == "__main__":
    rows = audit(sys.argv[1])
    ok = [r for r in rows if "outcome" in r]
    print(f"{len(rows)} trooper shots; bullet followed to its end: {len(ok)}; "
          f"{Counter(r.get('verdict') for r in rows if 'outcome' not in r)}")
    print("collision point (where it would be drawn next tick) -> outcome:")
    for (a, b), n in sorted(Counter((r["at"], r["outcome"]) for r in ok).items()):
        print(f"  {a:6s} -> {b:14s} {n}")
    print("aimed part -> outcome:", dict(Counter((r["part"], r["outcome"]) for r in ok)))
    for r in ok:
        if r["at"] in ("gap", "clear") or r["outcome"] == "target intact":
            print("   aimed", r["part"], "pos", r["pos"], "v", r["v"], "at", r["at"],
                  "crossed", r["crossed"], "->", r["outcome"])
