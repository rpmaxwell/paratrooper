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


def _stat(rows, key):
    vals = [_get(r, key) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.1f} ± {sd / len(vals) ** 0.5:.1f}"


def _get(row, key):
    for k in key.split("."):
        row = row.get(k) if isinstance(row, dict) else None
    return row


COMPARE = ("score", "world.trooper_landed", "world.trooper_body_killed", "world.trooper_chute_killed",
           "world.aircraft_down", "world.bombs_landed", "world.shots", "active_s")


def compare(run):
    rows = store.summaries(run)
    arms = sorted({r["arm"] for r in rows})
    lines = [f"{'(mean ± s.e.)':28s}" + "".join(f"{a:>22s}" for a in arms),
             f"{'games':28s}" + "".join(f"{sum(r['arm'] == a for r in rows):>22d}" for a in arms)]
    for key in COMPARE:
        lines.append(f"{key:28s}" + "".join(f"{_stat([r for r in rows if r['arm'] == a], key):>22s}" for a in arms))
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
