"""Measure how fast the game runs under the current DOSBox-X config.

    python3 /repo/lab/turbo/measure_turbo.py [seconds]     (in the container; up with --speed turbo)

Samples, against the wall clock, the BIOS tick (0040:006C, advanced by the
emulated timer = emulated time) and the game's own tick counter (LAYOUT
"tick"). Reports:
  speed      BIOS ticks per wall second / 18.2065 (1.0 = real time)
  game/bios  game ticks per BIOS tick while a game is in play; 1.0 means
             the game finishes every tick's work in time, < 1.0 means the
             emulated CPU (cycles) is too slow and the game drops ticks
Restarts the game (space) whenever its tick stops (title / game over).
"""
import struct
import sys
import time

from paratrooper.act.xtest import Keys
from paratrooper.emulator.process_memory import LAYOUT, GameMemory

TICK_HZ = 18.2065


def guest_base(mem):
    """Host address of guest physical 0: the MCB just below the PSP names
    the PSP's own segment as owner."""
    mcb = mem._read_at(mem.base - 16, 16)
    assert mcb[0] in b"MZ", f"no MCB below the PSP ({mcb[:3]!r})"
    psp_seg = struct.unpack_from("<H", mcb, 1)[0]
    return mem.base - psp_seg * 16


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    keys = Keys()
    for _ in range(6):          # get a game going so the live image changes
        keys.tap("space")
        time.sleep(0.3)
    mem = GameMemory()
    g0 = guest_base(mem)
    tick_off = LAYOUT["tick"][0]

    def sample():
        bios = struct.unpack("<I", mem._read_at(g0 + 0x46C, 4))[0]
        game = struct.unpack("<H", mem.read(tick_off, tick_off + 2))[0]
        return time.time(), bios, game

    t_start, b_start, _ = prev = sample()
    play_bios = play_game = 0
    stalled_since, restarts, resets = None, 0, 0
    ivals = []                    # (bios, game) per sample interval
    while time.time() - t_start < dur:
        time.sleep(0.02)
        cur = sample()
        db, dg = (cur[1] - prev[1]) & 0xFFFFFFFF, (cur[2] - prev[2]) & 0xFFFF
        ivals.append((db, dg if dg <= db + 2 else -1))
        if dg > db + 2:          # counter reset (new game): u16 difference wrapped
            resets += 1
            stalled_since = None
        elif dg:
            play_bios += db
            play_game += dg
            stalled_since = None
        elif stalled_since is None:
            stalled_since = cur[0]
        elif cur[0] - stalled_since > 0.3:   # not in play: start a new game
            keys.tap("space")
            restarts += 1
            stalled_since = None
        prev = cur
    wall = prev[0] - t_start
    bios = (prev[1] - b_start) & 0xFFFFFFFF
    speed = bios / wall / TICK_HZ
    ratio = play_game / play_bios if play_bios else float("nan")
    # mid-game only: the interval and both neighbours advanced the game tick,
    # so a shortfall there is a dropped tick, not a game starting or ending
    mid = [ivals[i] for i in range(1, len(ivals) - 1) if min(ivals[i - 1][1], ivals[i][1], ivals[i + 1][1]) > 0]
    mb, mg = sum(b for b, _ in mid), sum(g for _, g in mid)
    short = sum(g < b for b, g in mid)
    print(f"mid-game: bios {mb} game {mg} game/bios {mg / mb if mb else float('nan'):.4f}  "
          f"intervals {len(mid)} short {short}  ticks/interval {mb / max(len(mid), 1):.1f}")
    print(f"wall {wall:.1f}s  bios_ticks {bios}  speed {speed:.2f}x  "
          f"in-play bios {play_bios} game {play_game}  game/bios {ratio:.3f}  restarts {restarts} resets {resets}")


if __name__ == "__main__":
    main()
