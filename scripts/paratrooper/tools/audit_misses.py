"""Audit every paratrooper that LANDED (a miss), from telemetry alone.

    python3 -m paratrooper.tools.audit_misses <run_dir> [<run_dir> ...] [--list]

For each landed trooper, replays its logged trajectory (x, per-tick y,
canopy-open tick) through the planner to decide WHY it got through:

  unreachable        no shot could hit it even with the turret already
                     waiting in the ideal position (pure geometry: e.g. a
                     canopy that opened low at the far side)
  out_of_position    hittable, but only from a turret already in place --
                     from where the turret actually was, rotating over took
                     too long. Shows what the turret was doing instead.
  busy               hittable from where the turret was, but never engaged
                     (turret busy with something else)
  engaged_missed     we fired at it and missed -- shows what the bullets did

Then the leading patterns: side / column, canopy-open height, game time,
how many other troopers were in the air, and what the turret was doing
(from the press log) during the trooper's reachable window.

Needs only the game_*.jsonl files (no frames). --list prints one line per
missed trooper for spot checks (frames: game_<t0>.npz if recorded).
"""
import glob
import json
import sys
from collections import Counter, defaultdict

from ..physics import model as M
from ..physics import plan as P
from ..physics import prob_plan as PP

TICK_S = M.TICK_S


def load(dirs):
    games = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/game_*.jsonl")):
            R = [json.loads(l) for l in open(f)]
            if any(r["type"] == "game" for r in R):
                games.append((f, R))
    return games


def reach(tr, p_min=0.5):
    """Per-tick reachability of a trooper from its logged trajectory:
    -> list of (tick, ideal_ok) where ideal_ok = a plan exists with the
    turret already at the right position. Free-fallers are judged the way
    the bot judges them: a shot with P(hit) >= p_min over every
    chute-opening scenario (physics/prob_plan.py)."""
    can = tr["canopy_open"][0] if tr.get("canopy_open") else None
    drop_y = tr["first_y"] if tr["first_y"] <= 60 else 33
    out = []
    for tk, y in tr["y_by_tick"]:
        if can is not None and tk >= can:
            ok = P.plan_trooper(tr["x"], y, M.CANOPY_VY, "canopy", None) is not None
        else:
            pp = PP.plan_free(tr["x"], y, max(0, (y - drop_y) // M.FREE_VY), None)
            ok = pp is not None and pp["p_hit"] >= p_min
        out.append((tk, ok))
    return out


def activity(presses, t0, t1):
    """What the turret was doing between ticks t0..t1, from the press log
    (every job ends in presses: aimed shots and turret stops carry the job
    kind / 'why')."""
    c = Counter()
    for p in presses:
        if t0 <= p["tick"] <= t1:
            ctx = p["ctx"] or {}
            c[ctx.get("kind") if ctx.get("kind") != "stop" else "move:" + str(ctx.get("why"))] += 1
    return c


def classify(games):
    rows = []
    for f, R in games:
        presses = [r for r in R if r["type"] == "press"]
        troopers = [r for r in R if r["type"] == "trooper" and r.get("y_by_tick")]
        bullets = {r["shot"]: r for r in R if r["type"] == "bullet"}
        # troopers alive per tick (for "how crowded was it")
        alive = defaultdict(int)
        for tr in troopers:
            for tk, _ in tr["y_by_tick"]:
                alive[tk] += 1
        for tr in troopers:
            if tr["fate"] != "landed":
                continue
            rch = reach(tr)
            ok_ticks = [tk for tk, ok in rch if ok]
            first, last = tr["y_by_tick"][0][0], tr["y_by_tick"][-1][0]
            win = (ok_ticks[0], ok_ticks[-1]) if ok_ticks else (first, last)
            if tr.get("engagements"):
                cause = "engaged_missed"
            elif not ok_ticks:
                cause = "unreachable"
            elif tr.get("feasible_ever"):
                cause = "busy"
            else:
                cause = "out_of_position"
            ends = Counter(bullets[s]["end"]["kind"] for s in tr.get("engagements", [])
                           if s in bullets and bullets[s].get("found"))
            crowd = max(alive[tk] for tk, _ in tr["y_by_tick"]) - 1
            rows.append(dict(
                game=f.split("/")[-1], id=tr["id"], cause=cause, x=tr["x"], side=tr.get("side"),
                first_y=tr["first_y"], canopy_y=tr["canopy_open"][1] if tr.get("canopy_open") else None,
                first_tick=first, landed_tick=last, reach_ticks=len(ok_ticks), window=win,
                activity=activity(presses, *win), others_in_air=crowd, stack=tr.get("stack_height"),
                bullet_ends=ends))
    return rows


def pct(n, d):
    return f"{n} ({n / max(1, d) * 100:.0f}%)"


def report(rows, n_games, listing=False):
    n = len(rows)
    print(f"{n} troopers landed in {n_games} games ({n / max(1, n_games):.1f} per game)\n")
    print("CAUSE")
    for c, k in Counter(r["cause"] for r in rows).most_common():
        print(f"  {c:16s} {pct(k, n)}")

    print("\nWHERE (column of landing, 64 px bins; turret is at 288-352)")
    bins = Counter((r["x"] // 64) * 64 for r in rows)
    for b in sorted(bins):
        by = Counter(r["cause"] for r in rows if (r["x"] // 64) * 64 == b)
        print(f"  x {b:3d}-{b + 63:3d}: {bins[b]:4d}  " + ", ".join(f"{c} {v}" for c, v in by.most_common()))

    print("\nCANOPY OPENED AT (game y; ground ~369)")
    cy = Counter((r["canopy_y"] // 40) * 40 if r["canopy_y"] is not None else "never" for r in rows)
    for b in sorted(cy, key=lambda v: (isinstance(v, str), v)):
        by = Counter(r["cause"] for r in rows if ((r["canopy_y"] // 40) * 40 if r["canopy_y"] is not None else "never") == b)
        print(f"  y {b!s:>5}: {cy[b]:4d}  " + ", ".join(f"{c} {v}" for c, v in by.most_common()))

    print("\nCROWDING (other troopers in the air at the same time)")
    cr = Counter(min(r["others_in_air"], 6) for r in rows)
    for b in sorted(cr):
        print(f"  {b}{'+' if b == 6 else ''} others: {cr[b]}")

    print("\nWHAT THE TURRET WAS DOING while an out-of-position / busy trooper was reachable")
    act = Counter()
    for r in rows:
        if r["cause"] in ("out_of_position", "busy"):
            act.update(r["activity"])
    tot = sum(act.values())
    for k, v in act.most_common(10):
        print(f"  {k:22s} {pct(v, tot)} of presses")

    em = [r for r in rows if r["cause"] == "engaged_missed"]
    if em:
        print("\nENGAGED AND MISSED: how the bullets ended")
        e = Counter()
        for r in em:
            e.update(r["bullet_ends"])
        print("  " + ", ".join(f"{k} {v}" for k, v in e.most_common()))

    if listing:
        print("\nLIST (game, trooper id, cause, x, canopy y, first/landed tick, reachable ticks, turret activity)")
        for r in sorted(rows, key=lambda r: (r["game"], r["landed_tick"])):
            print(f"  {r['game']} #{r['id']:<5} {r['cause']:15s} x={r['x']:3d} canopy_y={r['canopy_y']} "
                  f"ticks {r['first_tick']}-{r['landed_tick']} reachable={r['reach_ticks']:3d} "
                  f"turret: {dict(r['activity'].most_common(3))}")


def main(argv):
    dirs = [a for a in argv if not a.startswith("--")]
    games = load(dirs)
    report(classify(games), len(games), listing="--list" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
