"""Phase 2 gate: is the telemetry true?

    python3 -m paratrooper.tools.check_telemetry <run_dir>

1. closure    -- every press -> exactly one bullet record; every trooper ->
                 exactly one fate.
2. score      -- point values per event type fitted by least squares over
                 every on-screen score change; reports the share of score
                 changes the logged events explain exactly.
3. audit      -- for trooper-aimed shots, the independent frame-based audit
                 (scripts/audit_trooper_shots.py logic, on the recorded
                 frames) vs the telemetry's hit/miss verdict.
"""
import glob
import json
import sys
from collections import Counter

import numpy as np

EVENT_TYPES = ("shot", "heli", "plane", "bomb", "trooper_body", "trooper_chute", "trooper_fell",
               "trooper_fell_crush")


def load(run_dir):
    games = []
    for f in sorted(glob.glob(f"{run_dir}/game_*.jsonl")):
        recs = [json.loads(l) for l in open(f)]
        if any(r["type"] == "game" for r in recs):
            games.append((f, recs))
    return games


def closure(games):
    presses = bullets = dup = troopers = fates = shared = 0
    for _, R in games:
        p = {r["shot"] for r in R if r["type"] == "press"}
        b = Counter(r["shot"] for r in R if r["type"] == "bullet")
        presses += len(p)
        bullets += sum(1 for s in p if b.get(s) == 1)
        dup += sum(1 for s, n in b.items() if n > 1)
        # one physical bullet must never be two records: no shared (tick, x, y)
        seen_pts = Counter(tuple(pt) for r in R if r["type"] == "bullet" and r.get("found")
                           for pt in r["path"])
        shared += sum(1 for n in seen_pts.values() if n > 1)
        tr = [r for r in R if r["type"] == "trooper"]
        troopers += len(tr)
        fates += sum(1 for r in tr if r.get("fate") and r["fate"] != "open_at_game_over")
    print(f"closure: {bullets}/{presses} presses have exactly one bullet record ({dup} duplicates); "
          f"{fates}/{troopers} troopers closed with a fate (rest open at game over); "
          f"bullet positions claimed by two records: {shared}")


def _events_by_tick(R):
    """(tick, type) of every credited event, timed by the evidence bullet's end."""
    bullet_end = {r["shot"]: r["end"]["tick"] for r in R if r["type"] == "bullet" and r.get("found")}
    ev = [(r["tick"], "shot") for r in R if r["type"] == "press"]
    for r in R:
        if r["type"] == "aircraft" and r["fate"] == "shot_down" and r["evidence"] in bullet_end:
            ev.append((bullet_end[r["evidence"]], r["kind"]))
        elif r["type"] == "bomb" and r["fate"] == "shot" and r["evidence"] in bullet_end:
            ev.append((bullet_end[r["evidence"]], "bomb"))
        elif r["type"] == "trooper" and r.get("fate_evidence") in bullet_end:
            kind = "trooper_body" if r["fate"] == "body_killed" else "trooper_chute"
            ev.append((bullet_end[r["fate_evidence"]], kind))
        if r["type"] == "trooper" and r["fate"] == "chute_killed" and r.get("fell_dead") and r.get("y_by_tick"):
            # a chute kill scores when the trooper hits the ground
            ev.append((r["y_by_tick"][-1][0], "trooper_fell_crush" if r.get("crushed_landed_at") is not None
                       else "trooper_fell"))
    return ev


def score(games):
    """Match every on-screen score change to logged events, both ways.
    Scoring (measured below): each shot -1 one tick after the press unless
    the score is 0; kills +value, shown ~2 ticks before our credited tick
    (credit is timed by the bullet's end)."""
    # 1. point values from unambiguous changes: +v with exactly one kill nearby
    vals = {}
    for _, R in games:
        ev = _events_by_tick(R)
        for c in (r for r in R if r["type"] == "score" and r["delta"] > 0):
            shots = sum(1 for tk, t in ev if t == "shot" and c["tick"] - 3 <= tk <= c["tick"] - 1)
            kills = [t for tk, t in ev if t != "shot" and c["tick"] - 3 <= tk <= c["tick"] + 6]
            if len(kills) == 1:
                vals.setdefault(kills[0], Counter())[c["delta"] + shots] += 1
    value = {t: c.most_common(1)[0][0] for t, c in vals.items()}
    print("score: point values from clean single-kill changes:",
          {t: dict(c.most_common(3)) for t, c in vals.items()})
    # 2. sequential matching
    changes = matched = unexplained_pts = false_credit = 0
    for _, R in games:
        ev = sorted(_events_by_tick(R))
        used = [False] * len(ev)
        for c in (r for r in R if r["type"] == "score"):
            changes += 1
            need = c["delta"]
            # presses in the few ticks before explain -1 each (score > 0)
            for i, (tk, t) in enumerate(ev):
                if need < 0 and not used[i] and t == "shot" and c["tick"] - 4 <= tk <= c["tick"]:
                    used[i] = True
                    need += 1
            for i, (tk, t) in enumerate(ev):
                if need > 0 and not used[i] and t != "shot" and c["tick"] - 3 <= tk <= c["tick"] + 6 \
                        and value.get(t, 0) <= need + 1:
                    used[i] = True
                    need -= value.get(t, 0)
                    # a shot fired the same frame can offset a kill (+10 - 1)
            if need < 0:
                for i, (tk, t) in enumerate(ev):
                    if need < 0 and not used[i] and t == "shot" and c["tick"] - 6 <= tk <= c["tick"]:
                        used[i] = True
                        need += 1
            if need == 0:
                matched += 1
            elif need > 0:
                unexplained_pts += need
        false_credit += sum(1 for (tk, t), u in zip(ev, used) if t != "shot" and not u)
    print(f"score: {matched}/{changes} score changes fully explained ({matched / max(1, changes) * 100:.1f}%); "
          f"unexplained points {unexplained_pts} (kills not credited); "
          f"credited kills with no score change: {false_credit}")
    return value


def audit(games, run_dir):
    """Independent check of trooper-aimed shots from the recorded frames."""
    import sys as _s
    import types
    _s.modules.setdefault("mss", types.ModuleType("mss"))
    import audit_trooper_shots as A  # scripts/audit_trooper_shots.py (crop-coordinate audit)
    agree = disagree = skipped = 0
    examples = []
    for f, R in games:
        npz = f.replace(".jsonl", ".npz")
        try:
            d = np.load(npz)
        except FileNotFoundError:
            continue
        t, F = d["t"], d["frames"]
        # native -> the audit's 640x400 crop grid (game coords: y = 2*ny + 1)
        crop = np.zeros((len(F), 400, 640), np.uint8)
        crop[:, 1:, :] = np.repeat(np.repeat(F, 2, axis=1), 2, axis=2)[:, :399, :]
        seq_all = A.distinct(t, crop)
        bullets = {r["shot"]: r for r in R if r["type"] == "bullet"}
        trooper = {r["id"]: r for r in R if r["type"] == "trooper"}
        # group trooper-aimed shots per target (one engagement may fire 2)
        groups = {}
        for p in R:
            if p["type"] != "press":
                continue
            c = p["ctx"] or {}
            # aimed shots, plus the job's turret-stop / clamp bullet (same lane)
            if c.get("kind") == "trooper" or (c.get("why") == "trooper" and "target" in c):
                groups.setdefault(c["target"], []).append(p)
        times = [s[0] for s in seq_all]
        for tid, presses in groups.items():
            tr = trooper.get(tid)
            ends = [bullets[p["shot"]]["end"]["tick"] for p in presses
                    if bullets.get(p["shot"], {}).get("found")]
            if not tr or not ends:
                skipped += 1
                continue
            # telemetry: the trooper died (body or chute) during the window --
            # by whichever bullet; attribution is checked by score reconciliation
            death = (tr.get("canopy_lost") or [None])[0] if tr["fate"] == "chute_killed" else \
                (tr["y_by_tick"][-1][0] if tr["fate"] in ("body_killed", "lost") and tr.get("y_by_tick") else None)
            t_lo, t_hi = min(p["tick"] for p in presses), max(ends) + 5
            tele_hit = death is not None and t_lo - 1 <= death <= t_hi
            # audit: canopy/body state just before the first press vs just
            # after the last bullet ended
            t0 = min(p["tick"] for p in presses) / 18.2065
            t1 = max(ends) / 18.2065
            i_b = max(0, int(np.searchsorted(times, t0 - 0.05)) - 1)
            i_a = min(len(seq_all) - 1, int(np.searchsorted(times, t1 + 0.25)))
            c0, b0 = A.sprites(seq_all[i_b][1], tr["x"])
            c1, b1 = A.sprites(seq_all[i_a][1], tr["x"])
            chute_hit = c0 is not None and c1 is None and b1 is not None
            body_hit = b0 is not None and b0[1] < 330 and b1 is None
            audit_hit = chute_hit or body_hit
            if audit_hit == tele_hit:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 6:
                    examples.append((f[-20:], tid, tr["fate"], tr.get("fate_evidence"),
                                     [p["shot"] for p in presses], "tele", tele_hit, "audit", audit_hit))
    n = agree + disagree
    print(f"audit: trooper engagements {n} compared ({skipped} skipped): agreement {agree}/{n} "
          f"({agree / max(1, n) * 100:.1f}%)")
    for e in examples:
        print("   disagreement:", e)


def main(run_dir):
    games = load(run_dir)
    print(f"{len(games)} games")
    closure(games)
    score(games)
    audit(games, run_dir)


if __name__ == "__main__":
    main(sys.argv[1])
