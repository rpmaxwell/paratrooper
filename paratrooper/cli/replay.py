"""Replay a recorded game through the stack, offline: observer -> world ->
policy -> turret, with a fake keyboard and the clock set to each frame's
recorded time. The policy acts open loop (its keys don't reach the game), so
this checks perception, tracking, telemetry and decisions -- not play.

    python -m paratrooper replay <game> [<game> ...] [-c CONFIG] [--set k=v] [--no-policy]

<game>: a game directory (frames.npz, + memory.npz for the memory observer)
or an old capture file game_<t0>.npz (+ game_<t0>_mem.npz next to it).
"""
import argparse
import contextlib
import pathlib
import time
from collections import Counter

import numpy as np

from .. import config as C
from ..act.fake import FakeKeys
from ..control.turret import Turret
from ..emulator.process_memory import STATE_LO
from ..record.audit import bullet_audit
from ..world.world import World

BASE = pathlib.Path(__file__).resolve().parents[2] / "configs" / "base.toml"


def load_game(path):
    """-> (t, frames, snaps or None). Frames: N x 200 x 320 palette indices."""
    p = pathlib.Path(path)
    if p.is_dir():
        f, m = np.load(p / "frames.npz"), (np.load(p / "memory.npz") if (p / "memory.npz").exists() else None)
    else:
        f = np.load(p)
        mp = p.with_name(p.stem + "_mem.npz")
        m = np.load(mp) if mp.exists() else None
    t, frames = f["t"], f["frames"]
    if frames.shape[1:] == (400, 640):        # old full-resolution crops
        frames = frames[:, 1::2, ::2]
    snaps = None
    if m is not None:
        snaps = m["mem"]
        n = min(len(t), len(snaps))
        if not np.allclose(m["t"][:n], t[:n]):
            raise ValueError(f"{path}: frame and memory times differ")
        t, frames, snaps = t[:n], frames[:n], snaps[:n]
        if int(m["lo"]) != STATE_LO:
            raise ValueError(f"{path}: memory recorded from {int(m['lo']):#x}, expected {STATE_LO:#x}")
    return t, frames, snaps


@contextlib.contextmanager
def fake_clock():
    now = [0.0]
    real = time.time
    time.time = lambda: now[0]
    try:
        yield now
    finally:
        time.time = real


def replay(cfg, t, frames, snaps=None, policy=True, emit=None):
    """-> (events, world, policy or None). Mirrors runner.play_game's loop.
    world.frame_ticks: the world tick of every frame (skipped duplicates
    take the previous one), the numbering events use."""
    events = []
    emit = emit or events.append
    observer = C.load_kind("observer", cfg.observer.kind)()
    if observer.needs_memory and snaps is None:
        raise ValueError("the memory observer needs a recording with memory")
    with fake_clock() as now:
        now[0] = float(t[0])
        world = World(float(t[0]), emit, credit_by_score=cfg.world.credit_by_score)
        keys = FakeKeys()
        turret = Turret(keys, world)
        pol = None
        if policy:
            cls = C.load_kind("policy", cfg.policy.kind)
            pol = cls(world, turret, log=lambda m: None, params=C.policy_params(cfg))
        prev = None
        world.frame_ticks = ticks = []
        for i in range(len(t)):
            frame = frames[i]
            if prev is not None and np.array_equal(frame, prev):
                ticks.append(ticks[-1])
                continue
            prev = frame
            now[0] = ti = float(t[i])
            keys.pump()
            snap = snaps[i] if snaps is not None else None
            if pol:
                world.barrel = observer.barrel(frame, snap)
                turret.step(ti)
            obs = observer.observe(frame, snap)
            world.update(ti, frame, obs.sprites, obs.barrel, obs.mem_tick)
            ticks.append(world.tick(ti))
            if world.done:
                break
            if pol:
                pol.step(ti)
                turret.step(ti)
        world.close_all(now[0])
    return events, world, pol


def report(events):
    print("events:", dict(Counter(e["type"] for e in events)))
    tr = [e for e in events if e["type"] == "trooper"]
    print("trooper fates:", dict(Counter(e["fate"] for e in tr)))
    print("aircraft fates:", dict(Counter((e["kind"], e["fate"]) for e in events if e["type"] == "aircraft")))
    print("bomb fates:", dict(Counter(e["fate"] for e in events if e["type"] == "bomb")))
    print("decisions:", dict(Counter(e["chosen"] for e in events if e["type"] == "decision")))


def main(argv):
    ap = argparse.ArgumentParser(prog="python -m paratrooper replay")
    ap.add_argument("games", nargs="+")
    ap.add_argument("-c", "--config", default=str(BASE))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-policy", action="store_true")
    a = ap.parse_args(argv)
    cfg = C.load(a.config, a.set)
    allev = []
    for g in a.games:
        t, frames, snaps = load_game(g)
        t0 = time.perf_counter()
        events, world, _ = replay(cfg, t, frames, snaps, policy=not a.no_policy)
        print(f"{g}: {len(t)} frames in {time.perf_counter() - t0:.1f}s, {world.stats}")
        if snaps is not None:
            n = len(world.frame_ticks)
            print("  bullet audit:", bullet_audit(events, world.frame_ticks, snaps[:n], STATE_LO))
        allev += events
    report(allev)
