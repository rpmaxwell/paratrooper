"""Play games with the refactored stack and log verified telemetry.

    python3 -m paratrooper.run [n_games] [out_dir] [--record] [--mem]

Per distinct frame: grab (native) -> barrel read + turret step FIRST (so a
goto stop is never delayed by other work) -> world update -> policy step.
Per game, in out_dir: game_<t0>.jsonl (telemetry), a line in scores.csv,
and with --record the game's distinct native frames (game_<t0>.npz) for
offline audits; with --mem the game's state bytes from its memory
(io/memory.py, STATE_LO..STATE_HI) on every frame (game_<t0>_mem.npz).
"""
import os
import sys
import time

import numpy as np

from .control.turret import Turret
from .io.keys import Keys
from .io.memory import STATE_HI, STATE_LO, GameMemory
from .perception.memstate import MemState
from .io.screen import Screen
from .perception.barrel import barrel_pos
from .perception.hud import is_done, read_score
from .physics import model as M
from .policy.heuristic import HeuristicPolicy
from .telemetry.writer import Writer
from .world.world import World

# PARATROOPER_MEM_FEED=1: the world takes helicopters, troopers, bombs,
# bullets, the barrel and the tick clock from the game's memory (implies --mem)
MEM_FEED = os.environ.get("PARATROOPER_MEM_FEED") == "1"
STALL_S = 30.0  # no screen change this long = frozen / paused (quiet sky between waves lasts a few s)

CSV_HEADER = ("finished_at,game,score,duration_s,bombs_seen,bombs_shot,bombs_landed,troopers_body,"
              "troopers_chute,troopers_landed,crushed,aircraft_down,shots,bullets_found,"
              "bullets_not_found,unexplained_fates,loop_p50_ms,loop_p99_ms\n")


def _log(msg):
    print(msg, flush=True)  # live log even when stdout is redirected to a file


def wait_for_game(screen, keys, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, f, _ = screen.grab()
        if not is_done(f):
            return True
        keys.tap("space")
        time.sleep(0.8)
    return False


def play_game(screen, keys, writer, out_dir, n, record=False, log=_log, mem=None):
    if not wait_for_game(screen, keys):
        raise RuntimeError("could not start a game")
    t0 = time.time()
    writer.open(f"{out_dir}/game_{int(t0)}.jsonl")
    emit = writer.emit
    world = World(t0, emit)
    turret = Turret(keys, world)
    policy = HeuristicPolicy(world, turret, log=lambda m: log(f"[{time.time() - t0:7.2f}] {m}"))
    loop_ms, frames = [], []
    mem_t, mem_snaps = [], []     # --mem: the game's state bytes, read right after each grab
    emit(dict(type="game_start", t0=t0, game=n, heli_box=M.HELI_BOX, trooper_box=M.TROOPER_BOX,
              bomb_box=M.BOMB_BOX, phit=os.environ.get("PARATROOPER_PHIT", "model"),
              bomb_pairs=os.environ.get("PARATROOPER_BOMB_PAIRS") == "1", mem_feed=MEM_FEED))
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
        ms = None
        if mem is not None:
            snap = np.frombuffer(mem.read(STATE_LO, STATE_HI), np.uint8)
            if MEM_FEED:
                # the game's state as of now: stamp the world update with the read time
                t = time.time()
                ms = MemState(snap, STATE_LO)
            mem_t.append(t - t0)
            mem_snaps.append(snap)
        a = time.perf_counter()
        world.barrel = ms.barrel()[0] if ms is not None else barrel_pos(frame)
        turret.step(t)
        world.update(t, frame, mem=ms)
        if world.done:
            break
        policy.step(t)
        turret.step(t)
        loop_ms.append((time.perf_counter() - a) * 1000)
        if record:
            frames.append((t - t0, frame))
    keys.release_all()
    dur = time.time() - t0
    score = read_score(frame)
    world.close_all(time.time())
    lp = np.array(loop_ms) if loop_ms else np.zeros(1)
    st = world.stats
    summary = dict(type="game", game=n, score=score, duration_s=round(dur, 1),
                   stalled_s=round(stalled_s, 1), active_s=round(dur - stalled_s, 1), world=st,
                   policy=policy.stats, loop_ms=dict(p50=float(np.median(lp)), p99=float(np.percentile(lp, 99)),
                                                     max=float(lp.max()), frames=len(loop_ms)))
    emit(summary)
    writer.close()
    writer.flush()
    bombs_seen = world.bombs_seen  # every tracked bomb, incl. kills the telemetry couldn't credit
    csv = f"{out_dir}/scores.csv"
    new = not os.path.exists(csv)
    with open(csv, "a") as fh:
        if new:
            fh.write(CSV_HEADER)
        fh.write(",".join(str(v) for v in (
            time.strftime("%Y-%m-%d %H:%M:%S"), n, score, round(dur - stalled_s, 1), bombs_seen, st["bombs_shot"],
            st["bombs_landed"], st["trooper_body_killed"], st["trooper_chute_killed"], st["trooper_landed"],
            st["crushed"], st["aircraft_down"], st["shots"], st["bullets_found"], st["bullets_not_found"],
            st["unexplained_fates"], round(float(np.median(lp)), 2), round(float(np.percentile(lp, 99)), 2))) + "\n")
    if mem_snaps:
        np.savez_compressed(f"{out_dir}/game_{int(t0)}_mem.npz", t=np.array(mem_t), mem=np.stack(mem_snaps),
                            lo=STATE_LO, base=mem.base)
    if record and frames:
        np.savez_compressed(f"{out_dir}/game_{int(t0)}.npz", t=np.array([f[0] for f in frames]),
                            frames=np.stack([f[1] for f in frames]))
    log(f"GAME {n}: score {score} in {dur - stalled_s:.0f}s active ({stalled_s:.0f}s stalled) | loop p50 {np.median(lp):.2f} ms p99 "
        f"{np.percentile(lp, 99):.2f} ms max {lp.max():.1f} | {st}")
    return summary


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    n_games = int(args[0]) if args else 1
    out_dir = args[1] if len(args) > 1 else "/captures/refactor"
    record = "--record" in argv
    os.makedirs(out_dir, exist_ok=True)
    screen, keys, writer = Screen(), Keys(), Writer()
    mem = GameMemory() if ("--mem" in argv or MEM_FEED) else None
    for n in range(n_games):
        play_game(screen, keys, writer, out_dir, n, record=record, mem=mem)


if __name__ == "__main__":
    main(sys.argv[1:])
