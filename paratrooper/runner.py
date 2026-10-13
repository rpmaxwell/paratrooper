"""Play games inside the emulator container (started by `python -m
paratrooper play|ab` on the host, which creates the run directory).

    python3 -m paratrooper.runner <run_dir> <arm> [--games N] [--start K]

Per distinct frame: grab -> (memory read) -> barrel + turret step FIRST (so
a goto stop is never delayed by other work) -> world update -> policy step
-> turret step. Each game is written to <run_dir>/<arm>/g<K>/ (record/store.py).
"""
import argparse
import hashlib
import json
import pathlib
import re
import time

import numpy as np

from . import config as C
from .control.turret import Turret
from .emulator.process_memory import STATE_HI, STATE_LO, GameMemory
from .observe.hud import is_done, read_score
from .observe.screen import Screen
from .record.game import GameRecorder
from .record.store import load_arm
from .record.writer import Writer
from .world.world import World

# identifies the code a game ran even when the tree was dirty: a hash of the package source
CODE_HASH = hashlib.sha1(b"".join(f.read_bytes() for f in sorted(pathlib.Path(__file__).parent.rglob("*.py")))).hexdigest()[:10]
STALL_S = 30.0  # no screen change this long = frozen / paused (quiet sky between waves lasts a few s)
EMULATOR_CONF = pathlib.Path("/config/dosbox-x.conf")


def _log(msg):
    print(msg, flush=True)  # live log even when stdout is redirected to a file


def running_emulator():
    """The settings the container's DOSBox-X was started with."""
    try:
        conf = EMULATOR_CONF.read_text()
    except OSError:
        return None
    cycles = re.search(r"^\s*cycles\s*=\s*(?:fixed\s+)?(\d+)", conf, re.M)
    turbo = re.search(r"^\s*turbo\s*=\s*true", conf, re.M)
    return dict(speed="turbo" if turbo else "realtime", cycles=int(cycles.group(1)) if cycles else None)


def wait_for_game(screen, keys, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, f, _ = screen.grab()
        if not is_done(f):
            return True
        keys.tap("space")
        time.sleep(0.8)
    return False


class Parts:
    """The swappable parts a config names, built once per process."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.observer = C.load_kind("observer", cfg.observer.kind)()
        self.keys = C.load_kind("actuator", cfg.actuator.kind)()
        self.policy_cls = C.load_kind("policy", cfg.policy.kind)
        self.params = C.policy_params(cfg)
        self.screen = Screen()
        self.mem = GameMemory() if (self.observer.needs_memory or cfg.record.memory) else None
        self.writer = Writer()


def play_game(parts, arm_dir, index, ids, log=_log):
    cfg, screen, keys, observer, mem = parts.cfg, parts.screen, parts.keys, parts.observer, parts.mem
    if not wait_for_game(screen, keys):
        raise RuntimeError("could not start a game")
    t0 = time.time()
    rec = GameRecorder(arm_dir, f"g{index:03d}", cfg.record, parts.writer)
    emit = rec.emit
    world = World(t0, emit)
    turret = Turret(keys, world)
    policy = parts.policy_cls(world, turret, log=lambda m: log(f"[{time.time() - t0:7.2f}] {m}"), params=parts.params)
    loop_ms = []
    emit(dict(type="game_start", tick=0, t0=t0, game=index, observer=cfg.observer.kind, policy=cfg.policy.kind,
              code=CODE_HASH, **ids))
    last_change, stall_start, stalled_s = time.time(), None, 0.0
    while True:
        t, frame, changed = screen.grab()
        keys.pump()
        if not changed:
            # watchdog: nothing moves for STALL_S -> the game is frozen/paused
            if stall_start is None and t - last_change > STALL_S:
                stall_start = last_change
                emit(dict(type="stall_start", tick=world.tick(t), at_s=round(last_change - t0, 1)))
                log(f"[{t - t0:7.2f}] STALL: no screen change for {STALL_S:.0f}s (game frozen or paused)")
            time.sleep(0.001)
            continue
        if stall_start is not None:
            stalled_s += t - stall_start
            emit(dict(type="stall_end", tick=world.tick(t), stalled_s=round(t - stall_start, 1)))
            log(f"[{t - t0:7.2f}] stall ended after {t - stall_start:.0f}s")
            stall_start = None
        last_change = t
        snap = None
        if mem is not None:
            snap = np.frombuffer(mem.read(STATE_LO, STATE_HI), np.uint8)
            if observer.needs_memory:
                t = time.time()  # the game's state as of now: stamp the world update with the read time
            rec.memory(t - t0, world.tick(t), snap)
        a = time.perf_counter()
        world.barrel = observer.barrel(frame, snap)
        turret.step(t)
        obs = observer.observe(frame, snap)
        world.update(t, frame, obs.sprites, obs.barrel, obs.mem_tick)
        if world.done:
            break
        policy.step(t)
        turret.step(t)
        loop_ms.append((time.perf_counter() - a) * 1000)
        rec.frame(t - t0, world.tick(t), frame)
    keys.release_all()
    dur = time.time() - t0
    score = read_score(frame)
    world.close_all(time.time())
    lp = np.array(loop_ms) if loop_ms else np.zeros(1)
    summary = dict(**ids, game=index, started=t0, score=score, duration_s=round(dur, 1),
                   stalled_s=round(stalled_s, 1), active_s=round(dur - stalled_s, 1), bombs_seen=world.bombs_seen,
                   world=world.stats, policy=policy.stats, code=CODE_HASH,
                   loop_ms=dict(p50=float(np.median(lp)), p99=float(np.percentile(lp, 99)), max=float(lp.max()),
                                frames=len(loop_ms)))
    emit(dict(type="game", tick=world.tick(time.time()), **{k: v for k, v in summary.items() if k not in ids}))
    rec.close(summary, mem_lo=STATE_LO, mem_base=mem.base if mem else None)
    log(f"GAME {index}: score {score} in {dur - stalled_s:.0f}s active ({stalled_s:.0f}s stalled) | loop p50 "
        f"{np.median(lp):.2f} ms p99 {np.percentile(lp, 99):.2f} ms max {lp.max():.1f} | {world.stats}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("arm")
    ap.add_argument("--games", type=int, default=None, help="default: the config's games")
    ap.add_argument("--start", type=int, default=0, help="index of the first game (names g<K>)")
    a = ap.parse_args(argv)
    cfg, manifest = load_arm(a.run_dir, a.arm)
    emu = running_emulator()
    if emu and emu["speed"] != cfg.emulator.speed:
        raise SystemExit(f"config wants emulator.speed={cfg.emulator.speed}, the container runs {emu['speed']}")
    ids = dict(run=manifest["run"], arm=a.arm, config_hash=manifest["arms"][a.arm]["config_hash"],
               git_sha=manifest["git"]["sha"], emulator=emu)
    parts = Parts(cfg)
    arm_dir = pathlib.Path(a.run_dir) / a.arm
    n = a.games or cfg.games
    for k in range(a.start, a.start + n):
        play_game(parts, arm_dir, k, ids)
    _log(json.dumps(dict(done=n, arm=a.arm)))


if __name__ == "__main__":
    main()
