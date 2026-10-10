"""Check prob_plan.hit_matrix against a brute-force two-phase simulation
(physics.model.simulate with the exact piecewise trajectory) on random
free-fallers, spawn ticks and chute-opening ticks.
    python3 -m paratrooper.tools.check_prob_plan [n]
"""
import random
import sys

import numpy as np

from ..physics import model as M
from ..physics import prob_plan as PP


def brute(x, y_now, lane, f, k, swept=False):
    t_land = k + (M.GROUND_BODY_Y - (y_now + M.FREE_VY * k)) / M.CANOPY_VY

    def y_at(t):
        return y_now + M.FREE_VY * t if t < k else y_now + M.FREE_VY * k + M.CANOPY_VY * (t - k)

    hits = []
    for part in ("body", "canopy"):
        def box_at(t, part=part):
            if part == "canopy" and t < k:
                return (-1000, -1000, -999, -999)  # no chute yet
            return M.trooper_box(x, y_at(t), part, free=t < k)
        m = M.simulate(M.LANES[lane], box_at, f, max_k=t_land - 1, swept=swept)
        if m is not None:
            hits.append(m)
    return min(hits) if hits else None


def main(n=3000):
    rnd = random.Random(7)
    agree = 0
    mism = []
    for _ in range(n):
        x = rnd.randrange(0, 80) * 8
        y = rnd.randrange(20, 115) * 2 + 1
        lane = rnd.randrange(M.N_LANES)
        f0 = rnd.randrange(3, 12)
        p, q, meet = PP.hit_matrix(x, y, 5, lane, f0)
        k = rnd.randrange(1, 30)
        fi = rnd.randrange(PP.N_F)
        b = brute(x, y, lane, f0 + fi, k)                 # drawn-position contact
        bs = brute(x, y, lane, f0 + fi, k, swept=True)    # any contact
        want = PP.Q_KILL if b is not None else (PP.Q_SWEPT if bs is not None else 0.0)
        ok = abs(q[k - 1, fi] - want) < 1e-9 and (want == 0 or (b if b is not None else bs) == meet[k - 1, fi])
        agree += ok
        if not ok and len(mism) < 5:
            mism.append((x, y, lane, f0 + fi, k, b, bs, float(q[k - 1, fi]), int(meet[k - 1, fi])))
    print(f"hit_matrix vs brute force: {agree}/{n} identical (contact type and meeting tick)")
    for m in mism:
        print("  mismatch (x, y, lane, spawn, open_k, brute_meet, table_hit, table_meet):", m)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
