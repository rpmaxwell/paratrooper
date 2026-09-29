"""When does a free-falling trooper open its chute? A fitted model.

Measured from telemetry (tools: this module's fit(), 2026-09-28): chutes
open at a random time after the drop -- spread roughly evenly over ticks
(~2.7% of drops per tick, the same for every drop height and screen
column) -- but never lower than a floor at y ~= 249: a trooper still in
free fall there opens. So

    open tick (since drop) = min(random tick ~ d(s),  floor tick)
    floor tick             = first tick whose y >= FLOOR_Y

The opening tick is taken from each trooper's trajectory (the tick its
fall speed drops from 8 to 4 px), not from the tracker's canopy flag,
which confirms a tick or two late.

open_distribution() gives, for a trooper still falling now, the
probability of opening on each future tick -- what the hit-probability
planner (physics/prob_plan.py) sums over.
"""
import glob
import json
import pathlib

import numpy as np

from . import model as M

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "chute_model.json"
MAX_S = 60


def true_open_tick(y_by_tick):
    """Tick index (into y_by_tick) where the fall slows from 8 to 4 px/tick,
    or None. Uses consecutive-tick steps only."""
    for (t0, y0), (t1, y1), (t2, y2) in zip(y_by_tick, y_by_tick[1:], y_by_tick[2:]):
        if t1 == t0 + 1 and t2 == t1 + 1 and y1 - y0 == M.FREE_VY and y2 - y1 == M.CANOPY_VY:
            return t1
    return None


def fit(run_dirs, drop_ys=(33, 57)):
    """Kaplan-Meier style hazard of a *random* opening per tick since the
    drop, from trooper records seen from their drop. Troopers shot / lost /
    landed before opening are censored at their last free-fall tick; an
    opening at the floor is forced, so it is censored for the random part."""
    obs = []  # (s_end, event) per trooper: event = random opening at s_end
    ys_open = []
    for d in run_dirs:
        for f in glob.glob(f"{d}/game_*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                if r["type"] != "trooper" or r.get("first_y") not in drop_ys or not r.get("y_by_tick"):
                    continue
                t_open = true_open_tick(r["y_by_tick"])
                if t_open is not None:
                    ys_open.append(dict(r["y_by_tick"])[t_open])
                obs.append((r, t_open))
    floor_y = int(np.percentile(ys_open, 99))
    events, at_risk = np.zeros(MAX_S), np.zeros(MAX_S)
    for r, t_open in obs:
        yt = dict(r["y_by_tick"])
        if t_open is not None:
            s_end = t_open - r["first_tick"]
            forced = yt[t_open] >= floor_y - M.FREE_VY
        else:  # never seen opening: free fall observed up to its last free-fall tick
            s_end = r["y_by_tick"][-1][0] - r["first_tick"]
            forced = True  # censored
        s_end = min(s_end, MAX_S - 1)
        if forced and t_open is not None:
            at_risk[:s_end] += 1   # reached the floor: not at risk of a *random* opening there
        else:
            at_risk[:s_end + 1] += 1
            if not forced:
                events[s_end] += 1
    h_emp = np.where(at_risk >= 20, events / np.maximum(at_risk, 1), np.nan)
    # The empirical hazard rises like a uniform's (1/(b - s)): the random
    # opening tick is ~Uniform[1, b]. Fit b by matching the observed
    # survival over the well-sampled ticks, then use the smooth form (also
    # covers ticks past the drop-33 floor, where nobody is left to observe).
    ok = ~np.isnan(h_emp) & (events > 0)   # ticks where random openings were observable
    ok[0] = False
    s_obs = np.flatnonzero(ok)
    w = at_risk[s_obs]
    # weighted least squares on the hazard curve itself
    best = min(range(int(s_obs.max()) + 2, 200),
               key=lambda b: float((w * (h_emp[s_obs] - 1.0 / (b - s_obs)) ** 2).sum()))
    h = np.array([0.0] + [1.0 / (best - s) if s < best else 1.0 for s in range(1, MAX_S)])
    model = dict(floor_y=floor_y, uniform_max_tick=best, hazard=[float(x) for x in h],
                 hazard_empirical=[None if np.isnan(x) else float(x) for x in h_emp],
                 n_troopers=len(obs), n_random_openings=int(events.sum()),
                 at_risk=[int(x) for x in at_risk])
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(model, indent=1))
    return model


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = json.loads(DATA.read_text())
        _MODEL["hazard"] = np.array(_MODEL["hazard"])
    return _MODEL


def open_distribution(s_now, y_now, horizon=40):
    """For a trooper still free-falling s_now ticks after its drop, body top
    at y_now: P(chute opens k ticks from now), k = 1..horizon (index k-1).
    Random opening by the fitted hazard; still falling at the floor -> opens
    there. Sums to 1 (the floor always comes within the horizon)."""
    mdl = model()
    h = mdl["hazard"]
    p = np.zeros(horizon)
    alive = 1.0
    for k in range(1, horizon + 1):
        if y_now + M.FREE_VY * k >= mdl["floor_y"]:
            p[k - 1] = alive  # floor: opens now
            break
        hk = h[min(s_now + k, len(h) - 1)]
        p[k - 1] = alive * hk
        alive *= 1 - hk
    return p


if __name__ == "__main__":
    import sys
    m = fit(sys.argv[1:])
    h = np.array(m["hazard"])
    print(f"fitted on {m['n_troopers']} troopers seen from their drop "
          f"({m['n_random_openings']} random openings); floor y = {m['floor_y']}; "
          f"random opening ~ Uniform[1, {m['uniform_max_tick']}] ticks after the drop")
    print("hazard per tick since drop:", {s: round(float(v), 3) for s, v in enumerate(h[:32])})
