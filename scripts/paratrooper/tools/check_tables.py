"""Gate for the intercept tables: every table entry checked is recomputed by
brute force (physics.model.simulate against the game's hitboxes; bombs by
the game's test, bullet by bullet) -- identical meeting ticks required.
Also times the planners.
    python3 -m paratrooper.tools.check_tables [n]
"""
import math
import random
import sys
import time

from ..physics import model as M
from ..physics import plan as P
from ..physics import prob_plan as PP
from ..physics.tables import AX, AY, BR, KINDS, PARTS, TX, TY, VYS, tables


def _j(m):
    return -1 if m is None else m


def brute_trooper(lane, part, vy, x, y, swept):
    def box_at(t):
        return M.trooper_box(x, y + vy * t, part, free=vy == M.FREE_VY)
    return _j(M.simulate(M.LANES[lane], box_at, 0, swept=swept))


def brute_air(kind, lane, d, y0, x0):
    box = M.heli_box if kind == "heli" else M.plane_box

    def box_at(t):
        if not -48 <= x0 + M.HELI_VX * d * math.ceil(t) <= 640:
            return None  # off screen at that tick
        return box(x0 + M.HELI_VX * d * t, y0, d)
    return _j(M.simulate(M.LANES[lane], box_at, 0, swept=kind == "plane"))


def brute_bomb(lane, d, xr, f):
    """Bullet spawning at bomb tick f: each tick it is drawn, it is tested
    against the bomb where it was before that tick's move."""
    (sx, sy), (vx, vy) = M.LANES[lane]
    for j in range(40):
        ux, uy = sx + vx * j, sy + vy * j
        if uy < -2 or not -2 <= ux <= 640:
            return -1
        b = M.bomb_at(xr, d, f + j - 1)
        if b is None:
            if f + j - 1 < 0:
                continue
            return -1
        if M.bomb_hit_code(ux, uy, *b):
            return j
    return -1


def main(n=3000):
    rnd = random.Random(1)
    T = tables()
    res = {}
    for name, swept in (("trooper", False), ("trooper_swept", True)):
        bad = 0
        for _ in range(n):
            lane, pi, vi = rnd.randrange(M.N_LANES), rnd.randrange(2), rnd.randrange(2)
            xi, yi = rnd.randrange(len(TX)), rnd.randrange(len(TY))
            want = brute_trooper(lane, PARTS[pi], VYS[vi], int(TX[xi]), int(TY[yi]), swept)
            got = int(T[name][lane, pi, vi, xi, yi])
            if got != want:
                bad += 1
                if bad <= 3:
                    print(f"  {name} mismatch", (lane, PARTS[pi], VYS[vi], int(TX[xi]), int(TY[yi])), got, want)
        res[name] = bad
    for kind in KINDS:
        bad = 0
        for _ in range(n):
            lane, di = rnd.randrange(M.N_LANES), rnd.randrange(2)
            yi = rnd.randrange(len(AY)) if kind == "heli" else 3
            xi = rnd.randrange(len(AX))
            want = brute_air(kind, lane, (-1, 1)[di], int(AY[yi]), int(AX[xi]))
            got = int(T[kind][lane, di, yi, xi])
            if got != want:
                bad += 1
                if bad <= 3:
                    print(f"  {kind} mismatch", (lane, (-1, 1)[di], int(AY[yi]), int(AX[xi])), got, want)
        res[kind] = bad
    bad = 0
    for _ in range(n):
        lane, di, ri, f = rnd.randrange(M.N_LANES), rnd.randrange(2), rnd.randrange(len(BR)), rnd.randrange(len(M.BOMB_YS))
        want = brute_bomb(lane, (-1, 1)[di], int(BR[ri]), f)
        got = int(T["bomb"][lane, di, ri, f])
        if got != want:
            bad += 1
            if bad <= 3:
                print("  bomb mismatch", (lane, (-1, 1)[di], int(BR[ri]), f), got, want)
    res["bomb"] = bad
    for k, b in res.items():
        print(f"{k:14s}: {n - b}/{n} table entries identical to brute force")

    # planner timings
    def timed(f, cases):
        a = time.perf_counter()
        for c in cases:
            f(*c)
        return (time.perf_counter() - a) / len(cases) * 1000
    cur = [rnd.choice([None] + list(range(19))) for _ in range(300)]
    ms = {
        "plan_bomb": timed(P.plan_bomb, [((rnd.randrange(0, 11) * 8), 1, rnd.randrange(1, 25), c) for c in cur]),
        "plan_trooper": timed(P.plan_trooper, [(rnd.randrange(0, 80) * 8, rnd.randrange(20, 180) * 2 + 1,
                                                M.CANOPY_VY, "canopy", c) for c in cur]),
        "plan_air": timed(P.plan_air, [("heli", rnd.randrange(0, 76) * 8, 17, 1, c) for c in cur]),
        "plan_free": timed(PP.plan_free, [(rnd.randrange(0, 80) * 8, rnd.randrange(20, 115) * 2 + 1, 5, c)
                                          for c in cur[:50]]),
    }
    print("ms per plan:", {k: round(v, 3) for k, v in ms.items()})
    return sum(res.values()) == 0


if __name__ == "__main__":
    sys.exit(0 if main(*(int(a) for a in sys.argv[1:])) else 1)
