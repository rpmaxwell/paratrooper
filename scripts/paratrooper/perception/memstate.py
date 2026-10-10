"""Game state from the game's own memory (io/memory.py), in the same shapes
the vision detector (perception/sprites.py) produces, so the world model
can take it in place of detection: helicopters, troopers, bombs and bullet
dots are exact, the barrel is read from the game, and the tick counter
gives the game clock.

Decoded from the disassembly (tools/disasm.py; data offsets are the code's
DS offsets, DS = program segment + 0x110):
  tick        DS 1AB0  u16, +1 per game tick
  lanes       DS 1D91+bp / 1D99+bp / 1D9D+bp (bp = 2, 4): helicopter map base,
              band top row (native), direction (+-1). A helicopter is ONE
              nonzero byte in its lane's map, at column m (= game x // 8);
              it occupies columns m..m+5. Skid x = 8(m - 1) moving left,
              8(m + 1) moving right; skid row = top + 8.
  troopers    per screen column c (= game x // 8 + 1): screen address word at
              DS 1BA9 + 2c (sign: + canopy, - no canopy; 0 = none), top row
              word at DS 1C49 + 2c (= native body top - 16), free-fall flag
              byte at DS 1CE9 + 2c (1 until the chute opens)
  bombs       slots at DS 1EF2 (word, 0 = free), column at +0x3C, native row
              at +0x50 (byte), stride 2
  bullets     30 slots: native x at DS 1F59 + 2i (0 = free), native y at +0x3C,
              direction at +0x78
  barrel      DS 1F57: pointer, barrel position = 19 - ptr / 2 (0 and 0x28:
              the two clamped sight lanes)
"""
import numpy as np

from .sprites import Sprites

DS = 0x110
N_BULLETS = 30
N_BOMBS = 10


class MemState:
    """One snapshot of the state bytes (io/memory STATE_LO..STATE_HI)."""

    def __init__(self, snap, lo):
        self.b = np.frombuffer(snap, np.uint8) if not isinstance(snap, np.ndarray) else snap
        self.lo = lo

    def u8(self, ds):
        return int(self.b[ds + DS - self.lo])

    def u16(self, ds):
        o = ds + DS - self.lo
        return int(self.b[o]) | int(self.b[o + 1]) << 8

    def s16(self, ds):
        v = self.u16(ds)
        return v - 0x10000 if v >= 0x8000 else v

    # ------------------------------------------------------------------ fields
    @property
    def tick(self):
        return self.u16(0x1AB0)

    @property
    def phase(self):
        return self.u8(0x1AB2)

    @property
    def seed(self):
        return self.u16(0x1ADB)

    def barrel(self):
        """-> (barrel position 0..18, clamped)"""
        ptr = self.u16(0x1F57)
        if ptr == 0:
            return 18, True
        if ptr == 0x28:
            return 0, True
        return 19 - ptr // 2, False

    def helis(self):
        """-> [(skid x, skid y, dir)] game coords, all helicopters in the maps."""
        out = []
        for bp in (2, 4):
            base, top, d = self.u16(0x1D91 + bp), self.s16(0x1D99 + bp), self.s16(0x1D9D + bp)
            if not base or d not in (-1, 1):
                continue
            for m in range(-6, 86):
                o = base + m + DS - self.lo
                if 0 <= o < len(self.b) and self.b[o]:
                    x = 8 * (m - 1) if d < 0 else 8 * (m + 1)
                    out.append((x, 2 * (top + 8) + 1, d))
        return out

    def troopers(self):
        """-> [(body x, body y, canopy, free_fall)] game coords."""
        out = []
        for c in range(80):
            s = self.s16(0x1BA9 + 2 * c)
            if s == 0:
                continue
            top = self.s16(0x1C49 + 2 * c)
            out.append((8 * (c - 1), 2 * (top + 16) + 1, s > 0, bool(self.u8(0x1CE9 + 2 * c))))
        return out

    def bombs(self):
        out = []
        for i in range(N_BOMBS):
            if self.u16(0x1EF2 + 2 * i):
                out.append((8 * self.u8(0x1EF2 + 2 * i + 0x3C), 2 * self.u8(0x1EF2 + 2 * i + 0x50) + 1))
        return out

    def bullets(self):
        """-> [(x, y, direction)] game coords of live bullets."""
        out = []
        for i in range(N_BULLETS):
            x = self.u16(0x1F59 + 2 * i)
            if x:
                out.append((2 * x, 2 * self.u16(0x1F59 + 2 * i + 0x3C) + 1, self.u16(0x1F59 + 2 * i + 0x78)))
        return out


def sprites_from_memory(ms, seen):
    """Replace the moving-object fields of a vision Sprites (`seen`) with the
    game's own state; keep what memory doesn't hold yet (planes, landed
    heads, grounded bodies)."""
    s = Sprites()
    s.planes, s.planes_full, s.landed, s.grounded = seen.planes, seen.planes_full, seen.landed, seen.grounded
    hs = ms.helis()
    s.heli_any = bool(hs) or seen.heli_any
    s.helis = [(x, y) for x, y, _ in hs if 0 <= x <= 640 - 32]      # fully on screen, like the detector
    for x, y, canopy, _ in ms.troopers():
        if y > 359:                                                  # standing: vision's grounded/landed
            continue
        s.bodies.append((x, y))           # vision sees the body under a canopy too
        if canopy:
            s.canopies.append((x, y))
    s.bombs = ms.bombs()
    s.dots = [(x, y) for x, y, _ in ms.bullets()]
    return s
