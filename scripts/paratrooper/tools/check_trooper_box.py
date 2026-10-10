"""PARATROOPER_TROOPER_BOX A/B gate: how planning against troopers changes
under the "fitted" box (bigger body, canopy down to the body) for the same
troopers -- share with a plan, meeting tick, window width. Each variant
runs in its own process (the variant is read at import).
    python3 -m paratrooper.tools.check_trooper_box [n]
"""
import json
import os
import random
import subprocess
import sys


def _plans(variant, cases):
    code = ("import json, sys\n"
            "from paratrooper.physics import plan as P, prob_plan as PP, model as M\n"
            "assert M.TROOPER_BOX == sys.argv[1], M.TROOPER_BOX\n"
            "out = []\n"
            "for x, y, canopy, pos in json.load(sys.stdin):\n"
            "    if canopy:\n"
            "        out.append(P.plan_trooper(x, y, M.CANOPY_VY, 'canopy', pos))\n"
            "    else:\n"
            "        r = PP.plan_free(x, y, max(0, (y - 57) // M.FREE_VY), pos)\n"
            "        out.append(None if r is None else [r['pos'], r['spawns'], r['meet_last'], r['p_hit']])\n"
            "print(json.dumps(out))\n")
    r = subprocess.run([sys.executable, "-c", code, variant], input=json.dumps(cases),
                       capture_output=True, text=True, check=True,
                       env={**os.environ, "PARATROOPER_TROOPER_BOX": variant})
    return json.loads(r.stdout)


def main(n=2000, variant="fitted"):
    rnd = random.Random(1)
    cols = [32 + 24 * i for i in range(11)] + [368 + 24 * i for i in range(11)]
    cases = []
    for _ in range(n):
        canopy = rnd.random() < 0.5
        y = (rnd.randrange(60, 330) if canopy else 57 + 8 * rnd.randrange(0, 22))
        cases.append((rnd.choice(cols), y | 1, canopy, rnd.randrange(19)))
    old, new = _plans("model", cases), _plans(variant, cases)
    print(f"model vs {variant}")
    for canopy, name in ((False, "free-fallers (prob_plan.plan_free)"), (True, "canopy troopers (plan_trooper)")):
        idx = [i for i, c in enumerate(cases) if c[2] == canopy]
        has_o = sum(old[i] is not None for i in idx)
        has_n = sum(new[i] is not None for i in idx)
        both = [i for i in idx if old[i] and new[i]]
        dmeet = sum(new[i][2] - old[i][2] for i in both) / max(1, len(both))
        wid = (sum(len(old[i][1]) for i in both) / max(1, len(both)), sum(len(new[i][1]) for i in both) / max(1, len(both)))
        print(f"{name}: plan found model {has_o}/{len(idx)}, fitted {has_n}/{len(idx)}; "
              f"both: meeting tick fitted - model {dmeet:+.2f}, bullets/window {wid[0]:.2f} -> {wid[1]:.2f}")
        if not canopy:
            ph = [(old[i][3], new[i][3]) for i in both]
            print(f"    P(hit) of the chosen shot: model {sum(a for a, _ in ph) / len(ph):.3f} -> fitted {sum(b for _, b in ph) / len(ph):.3f}; "
                  f">= 0.5 (the bot engages): model {sum(a >= .5 for a, _ in ph)}, fitted {sum(b >= .5 for _, b in ph)} of {len(ph)}")
    ok = all((new[i] is not None) >= (old[i] is not None) for i in range(n))
    print("OK (fitted never loses a plan the model had)" if ok else "NOTE: some plans lost under fitted")
    return True


if __name__ == "__main__":
    main(*(int(a) if a.isdigit() else a for a in sys.argv[1:]))
