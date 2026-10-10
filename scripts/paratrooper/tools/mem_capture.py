"""Play one game (the normal stack) and snapshot the game's 64 KB memory
segment on every distinct frame, for mapping game variables.

    python3 -m paratrooper.tools.mem_capture [out_dir] [max_tick]

Stops at max_tick (default 1030: end of round 1, before the second
helicopter wave) or game over. Writes out_dir/mem_<t0>.npz with
t (s since t0, per frame), tick (world tick), frames (native), mem
(N x 65536 uint8, read right after the frame grab), and the usual
telemetry out_dir/game_<t0>.jsonl.
"""
import os
import sys
import time

import numpy as np

from ..control.turret import Turret
from ..io.keys import Keys
from ..io.memory import GameMemory
from ..io.screen import Screen
from ..perception.barrel import barrel_pos
from ..physics import model as M
from ..policy.heuristic import HeuristicPolicy
from ..run import wait_for_game
from ..telemetry.writer import Writer
from ..world.world import World


def main(argv):
    out_dir = argv[0] if argv else "/captures/mem1"
    max_tick = int(argv[1]) if len(argv) > 1 else 1030
    os.makedirs(out_dir, exist_ok=True)
    screen, keys, writer = Screen(), Keys(), Writer()
    mem = GameMemory()       # find the segment first: the scan takes ~1 s
    if not wait_for_game(screen, keys):
        raise RuntimeError("could not start a game")
    t0 = time.time()
    print(f"game segment at {mem.base:#x} (pid {mem.pid})", flush=True)
    writer.open(f"{out_dir}/game_{int(t0)}.jsonl")
    world = World(t0, writer.emit)
    turret = Turret(keys, world)
    policy = HeuristicPolicy(world, turret, log=lambda m: print(f"[{time.time() - t0:7.2f}] {m}", flush=True))
    writer.emit(dict(type="game_start", t0=t0, game=0, heli_box=M.HELI_BOX, mem_base=mem.base))
    ts, ticks, frames, snaps = [], [], [], []
    while True:
        t, frame, changed = screen.grab()
        keys.pump()
        if not changed:
            time.sleep(0.001)
            continue
        snap = mem.read()                    # right after the grab: the same game state
        world.barrel = barrel_pos(frame)
        turret.step(t)
        world.update(t, frame)
        ts.append(t - t0)
        ticks.append(world.tick(t))
        frames.append(frame)
        snaps.append(np.frombuffer(snap, np.uint8))
        if world.done or ticks[-1] >= max_tick:
            break
        policy.step(t)
        turret.step(t)
    keys.release_all()
    world.close_all(time.time())
    writer.emit(dict(type="mem_capture_end", tick=ticks[-1], frames=len(frames), done=world.done))
    writer.close()
    writer.flush()
    path = f"{out_dir}/mem_{int(t0)}.npz"
    np.savez_compressed(path, t=np.array(ts), tick=np.array(ticks), frames=np.stack(frames),
                        mem=np.stack(snaps), base=mem.base)
    print(f"{len(frames)} frames to tick {ticks[-1]} -> {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
