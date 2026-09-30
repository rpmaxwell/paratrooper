"""Phase 3 gate: table planners vs the old, validated simulators on random
inputs -- identical decisions required. Also times both.
    python3 -m paratrooper.tools.check_tables [n]
"""
import random
import sys
import time

import bomb_model as OB
import heli_model as OH
import trooper_model as OT

from ..physics import plan as P


def main(n=3000):
    rnd = random.Random(1)
    OB.LIMIT_STATE["clamped"] = False
    res = {}
    # bombs
    bad = 0; t_old = t_new = 0
    for _ in range(n):
        d = rnd.choice((-1, 1)); xr = rnd.randrange(0, 11) * 8 + (0 if d > 0 else 536)
        k = rnd.randrange(1, 25); cur = rnd.choice([None] + list(range(19)))
        a = time.perf_counter(); o = OB.plan_intercept(xr, d, k, cur); b = time.perf_counter()
        nw = P.plan_bomb(xr, d, k, cur); c = time.perf_counter()
        t_old += b - a; t_new += c - b
        o = None if o is None else (o[0], list(o[1]))
        bad += o != nw
    res["bomb"] = (bad, t_old, t_new)
    # troopers
    bad = 0; t_old = t_new = 0
    for _ in range(n):
        x = rnd.randrange(0, 80) * 8; y = rnd.randrange(20, 180) * 2 + 1
        state = rnd.choice(("canopy", "free")); part = "canopy" if state == "canopy" else "body"
        cur = rnd.choice([None] + list(range(19)))
        tr = OT.Trooper(x, y, 100.0); tr.state = state
        a = time.perf_counter(); o = OT.plan_trooper(tr, part, cur, 100.0); b = time.perf_counter()
        # the old planner counted contact between ticks: compare on the swept table
        nw = P.plan_trooper(x, y, tr.vy, part, cur, free=(state == "free"), swept=True); c = time.perf_counter()
        t_old += b - a; t_new += c - b
        o = None if o is None else (o[0], list(o[1]), o[2], o[3])
        if o != nw:
            bad += 1
            if bad <= 3:
                print("  trooper mismatch", (x, y, state, cur), "old", o, "new", nw)
    res["trooper"] = (bad, t_old, t_new)
    # helicopters + planes
    for kind in ("heli", "plane"):
        bad = 0; t_old = t_new = 0
        for _ in range(n):
            d = rnd.choice((-1, 1)); x0 = rnd.randrange(0, 76) * 8
            y0 = rnd.choice((17, 29, 41, 65, 89)) if kind == "heli" else 1
            cur = rnd.choice([None] + list(range(19)))
            allowed = None if kind == "heli" else range(max(0, cur - 5 if cur else 0), 19)
            box = OH.heli_box if kind == "heli" else OH.plane_box
            a = time.perf_counter(); o = OH.plan_heli(x0, y0, d, cur, box=box, allowed=allowed); b = time.perf_counter()
            nw = P.plan_air(kind, x0, y0, d, cur, allowed=allowed); c = time.perf_counter()
            t_old += b - a; t_new += c - b
            o = None if o is None else (o[0], list(o[1]), o[2])
            if o != nw:
                bad += 1
                if bad <= 3:
                    print(f"  {kind} mismatch", (x0, y0, d, cur), "old", o, "new", nw)
        res[kind] = (bad, t_old, t_new)
    for k, (bad, a, b) in res.items():
        print(f"{k:8s}: {n - bad}/{n} identical plans | old {a / n * 1000:.2f} ms, new {b / n * 1000:.3f} ms per plan")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
