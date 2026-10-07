"""PARATROOPER_HELI_BOX A/B gate: the "fitted" helicopter box (+8 px, one
tick of helicopter motion) should move the planned spawn window by about
one tick -- later against left-movers, earlier against right-movers --
for the same helicopter and barrel position. Runs each variant in its own
process (the variant is read at import).
    python3 -m paratrooper.tools.check_heli_box [n]
"""
import json
import os
import random
import subprocess
import sys


def _plans(variant, cases):
    code = ("import json, sys\n"
            "from paratrooper.physics import plan as P, model as M\n"
            "assert M.HELI_BOX == sys.argv[1], M.HELI_BOX\n"
            "out = [P.plan_air('heli', x0, y0, d, pos, allowed=[pos], min_window=1)\n"
            "       for x0, y0, d, pos in json.load(sys.stdin)]\n"
            "print(json.dumps(out))\n")
    r = subprocess.run([sys.executable, "-c", code, variant], input=json.dumps(cases),
                       capture_output=True, text=True, check=True,
                       env={**os.environ, "PARATROOPER_HELI_BOX": variant})
    return json.loads(r.stdout)


def main(n=2000):
    rnd = random.Random(1)
    cases = [(rnd.randrange(0, 76) * 8, rnd.choice((17, 29, 41, 65, 89)), rnd.choice((-1, 1)),
              rnd.randrange(19)) for _ in range(n)]
    old, new = _plans("model", cases), _plans("fitted", cases)
    ok = True
    for d, name, want in ((-1, "left ", 1), (1, "right", -1)):
        shifts = [new_p[1][0] - old_p[1][0] for c, old_p, new_p in zip(cases, old, new)
                  if c[2] == d and old_p and new_p]
        mean = sum(shifts) / len(shifts)
        share = sum(s == want for s in shifts) / len(shifts)
        print(f"{name}: first spawn tick fitted - model, mean {mean:+.2f}, "
              f"{share:.0%} exactly {want:+d} ({len(shifts)} helicopters both can hit)")
        ok &= abs(mean - want) < 0.5 and share > 0.5
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*map(int, sys.argv[1:])) else 1)
