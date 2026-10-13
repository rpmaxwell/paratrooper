"""Bullet audit: every bullet the tracker logged, checked against the game's
memory. Runs after a game (record.memory = true) or on a replay; the result
goes into summary.json as "bullet_audit", so every run -- and every A/B arm
-- reports how well telemetry matched the game.

Per tracked bullet: find its memory slot at its last sighting (same tick
numbering, same position), follow the slot until the game frees it, then
compare with how the tracker ended it:

  ended_early        the game's bullet flew on >= LOST_STEPS lane steps past the
                     tracker's last sighting (beyond the 2 ticks it extrapolates to find a hit)
  false_hit          ... and credited it as a hit
  collisions         the game freed it mid-screen (it hit something)
  credited           ... and the tracker credited a hit
  uncredited         ... and the tracker logged it as vanished / left_screen
  edge_exits         the game freed it at the screen edge
  edge_called_hit    ... but the tracker credited a hit
  credit_rate        credited / collisions
"""
from collections import Counter, defaultdict

from ..observe.memstate import MemState

HIT_KINDS = ("hit_candidate", "hit_by_score")
LOST_STEPS = 3      # world._what_it_hit extrapolates a lost bullet up to 2 ticks


def _slots(snaps, lo):
    return [{s: (x, y) for s, x, y, _ in MemState(sn, lo).bullet_slots()} for sn in snaps]


def _steps(p, origin, v):
    dx, dy = p[0] - origin[0], p[1] - origin[1]
    m = round(dy / v[1]) if v[1] else round(dx / v[0])
    return m if m >= 0 and (dx, dy) == (v[0] * m, v[1] * m) else None


def bullet_audit(events, ticks, snaps, lo):
    """events: one game's events; ticks[i]: the world tick of memory snapshot
    i (record/game.py memory.npz "tick"); snaps[i]: its state bytes."""
    slots = _slots(snaps, lo)
    by_tick = defaultdict(list)
    for i, tk in enumerate(ticks):
        by_tick[int(tk)].append(i)
    c = Counter()
    for e in events:
        if e.get("type") != "bullet" or not e.get("found") or not e.get("path") or not e.get("end"):
            continue
        c["bullets"] += 1
        tk, x, y = e["path"][-1]
        v = tuple(e["lane_v"])
        found = next(((i, s) for k in (tk, tk - 1, tk + 1) for i in by_tick.get(k, ())
                      for s, p in slots[i].items() if p == (x, y)), None)
        if found is None:
            c["unidentified"] += 1
            continue
        i, slot = found
        j, last = i, (x, y)
        while j + 1 < len(slots):
            p = slots[j + 1].get(slot)
            if p is None or _steps(p, (x, y), v) is None:
                break
            j, last = j + 1, p
        kind = e["end"]["kind"]
        if j + 1 >= len(slots):
            c["game_over"] += 1
            continue
        if _steps(last, (x, y), v) >= LOST_STEPS:
            c["ended_early"] += 1
            c["false_hit"] += kind in HIT_KINDS
            continue
        nx, ny = last[0] + v[0], last[1] + v[1]
        if 0 < nx < 640 and ny >= 0:
            c["collisions"] += 1
            c["credited" if kind in HIT_KINDS else "uncredited"] += 1
        else:
            c["edge_exits"] += 1
            c["edge_called_hit"] += kind in HIT_KINDS
    out = {k: c[k] for k in ("bullets", "unidentified", "ended_early", "false_hit", "collisions", "credited",
                             "uncredited", "edge_exits", "edge_called_hit", "game_over")}
    out["credit_rate"] = round(c["credited"] / c["collisions"], 3) if c["collisions"] else None
    return out
