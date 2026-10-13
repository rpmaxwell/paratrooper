"""Play a config, or A/B two configs, in an emulator container (host side).

    python -m paratrooper play CONFIG [--games N] [--instance N] [--set k=v ...]
    python -m paratrooper ab CONFIG_A CONFIG_B [--games N] [--instance N] [--set k=v ...]

play: one run directory (record/store.py), games played in the container.
ab: both arms in one run, alternating game by game (A,B then B,A, ...) so
drift over time hits both equally; at the end a comparison and both arms'
resolved configs go to configs/archive/ with status "pending" -- fill in
result and status (promoted | rejected) there.
"""
import argparse
import statistics

from .. import config as C
from ..emulator import docker
from ..record import store
from . import archive


def _check_emulator(instance, cfgs):
    emu = docker.running(instance)
    if emu is None:
        raise SystemExit(f"{docker.name(instance)} is not running: python -m paratrooper up"
                         + (f" --instance {instance}" if instance is not None else ""))
    for cfg in cfgs:
        want = dict(speed=cfg.emulator.speed, cycles=cfg.emulator.cycles)
        if want != emu:
            raise SystemExit(f"{cfg.name} wants emulator {want}, {docker.name(instance)} runs {emu}: "
                             f"python -m paratrooper up --speed {want['speed']} --cycles {want['cycles']}")
    return emu


def _get(row, key):
    for k in key.split("."):
        row = row.get(k) if isinstance(row, dict) else None
    return row


def _mean_se(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.1f} ± {sd / len(vals) ** 0.5:.1f}"


def _per_min(rows, key):
    """Per minute of play: games differ in length, so raw counts mislead."""
    return _mean_se([_get(r, key) / r["active_s"] * 60 if _get(r, key) is not None and r.get("active_s") else None
                     for r in rows])


def _pooled(rows, num, den):
    """Sum over games of num / sum of den, ± s.e. clustered by game (bullets
    within a game aren't independent, so a binomial s.e. is far too small)."""
    pairs = [(_get(r, num) or 0, _get(r, den) or 0) for r in rows]
    d = sum(y for _, y in pairs)
    if not d:
        return "-"
    p = sum(x for x, _ in pairs) / d
    k = len(pairs)
    se = (sum((x - p * y) ** 2 for x, y in pairs) * k / max(k - 1, 1)) ** 0.5 / d
    return f"{p:.3f} ± {se:.3f}"


# (label, how, key[, denominator])
COMPARE = (
    ("score", "mean", "score"),
    ("active minutes", "mean", "active_min"),
    ("per minute of play:", None, None),
    ("  troopers landed", "per_min", "world.trooper_landed"),
    ("  troopers body-killed", "per_min", "world.trooper_body_killed"),
    ("  troopers chute-killed", "per_min", "world.trooper_chute_killed"),
    ("  aircraft down", "per_min", "world.aircraft_down"),
    ("  bombs landed", "per_min", "world.bombs_landed"),
    ("  shots", "per_min", "world.shots"),
    ("  unexplained fates", "per_min", "world.unexplained_fates"),
    ("bullet audit (pooled):", None, None),
    ("  hits credited / collisions", "pooled", "bullet_audit.credited", "bullet_audit.collisions"),
    ("  false hits / collisions", "pooled", "bullet_audit.false_hit", "bullet_audit.collisions"),
    ("  lost / bullets", "pooled", "bullet_audit.ended_early", "bullet_audit.bullets"),
    ("  edge exits called hits", "pooled", "bullet_audit.edge_called_hit", "bullet_audit.edge_exits"),
    ("loop p99 ms", "mean", "loop_ms.p99"),
)


def compare(run):
    rows = store.summaries(run)
    for r in rows:
        r["active_min"] = r["active_s"] / 60 if r.get("active_s") else None
    arms = sorted({r["arm"] for r in rows})
    by = {a: [r for r in rows if r["arm"] == a] for a in arms}
    w = 30
    lines = [f"{'(mean ± s.e.)':{w}s}" + "".join(f"{a:>22s}" for a in arms),
             f"{'games':{w}s}" + "".join(f"{len(by[a]):>22d}" for a in arms)]
    for label, how, key, *den in COMPARE:
        if how is None:
            lines.append(label)
            continue
        cells = []
        for a in arms:
            rs = by[a]
            cells.append(_mean_se([_get(r, key) for r in rs]) if how == "mean" else
                         _per_min(rs, key) if how == "per_min" else _pooled(rs, key, den[0]))
        lines.append(f"{label:{w}s}" + "".join(f"{c:>22s}" for c in cells))
    return "\n".join(lines), rows


def play(argv):
    ap = argparse.ArgumentParser(prog="python -m paratrooper play")
    ap.add_argument("config")
    ap.add_argument("--games", type=int)
    ap.add_argument("--instance", type=int)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = C.load(a.config, a.set)
    emu = _check_emulator(a.instance, [cfg])
    run = store.new_run(cfg.name, {cfg.name: cfg}, emulator=emu)
    print(f"run {run}")
    docker.run_games(a.instance, run, cfg.name, a.games or cfg.games)
    print(compare(run)[0])


def ab(argv):
    ap = argparse.ArgumentParser(prog="python -m paratrooper ab")
    ap.add_argument("config_a")
    ap.add_argument("config_b")
    ap.add_argument("--games", type=int, help="per arm (default: arm A's games)")
    ap.add_argument("--instance", type=int)
    ap.add_argument("--set", action="append", default=[], help="applied to both arms")
    a = ap.parse_args(argv)
    ca, cb = C.load(a.config_a, a.set), C.load(a.config_b, a.set)
    na, nb = ca.name, cb.name
    if na == nb:
        na, nb = f"{na}_a", f"{nb}_b"
    emu = _check_emulator(a.instance, [ca, cb])
    run = store.new_run(f"ab_{na}_vs_{nb}", {na: ca, nb: cb}, emulator=emu)
    print(f"run {run}")
    games = a.games or ca.games
    for i in range(games):
        for arm in ((na, nb) if i % 2 == 0 else (nb, na)):
            docker.run_games(a.instance, run, arm, 1, start=i)
    table, rows = compare(run)
    print(table)
    for path in archive.add(run, summary_rows=rows):
        print(f"archived {path}")
