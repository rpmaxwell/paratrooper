"""Non-blocking turret control, advanced once per frame.

The turret rotates while a direction key is "latched" and stops (firing a
bullet) on Up. goto stops on sight of the target position -- calibrated
exact (lead 0: 16/17 stops on target). At a rotation limit that leaves the
turret unclamped (steeper lane); clamp() holds into the stop first.
Every Up press is registered with the world as a bullet (world.note_press).
"""
import time

from ..physics import model as M

GOTO_TIMEOUT_S = 2.0
CLAMP_HOLD_S = 0.2
# A turn key sent while Up is still held (or within a tick of it) is
# swallowed by the game: 94 of ~13,300 turns never moved, every one right
# after another press (2026-10-01). Wait this long after an Up press.
KEY_GAP_S = 0.07
STUCK_S = 0.22      # no barrel movement this long after a turn key -> press it again


class Turret:
    def __init__(self, keys, world):
        self.keys, self.world = keys, world
        self.state = "idle"     # idle | moving | clamping
        self.target = None
        self._dir = 0
        self._t0 = 0.0
        self._ctx = None

    def busy(self):
        return self.state != "idle"

    def fire(self, ctx, pos=None):
        """Press Up; returns the shot id the world's bullet ledger uses."""
        t = self.keys.press("Up", ctx)
        self._last_up = t
        return self.world.note_press(t, ctx, pos)

    def _keys_settled(self):
        return "Up" not in self.keys._down and time.time() - getattr(self, "_last_up", 0) >= KEY_GAP_S

    def goto(self, pos, ctx=None):
        """Start moving to pos. -> True started, False already there (or
        barrel unreadable), None = not yet: a key was just pressed, retry
        next frame (a turn key right after Up is swallowed by the game)."""
        cur = self.world.barrel
        if cur is None or cur == pos:
            return False
        if not self._keys_settled():
            return None
        self._dir = 1 if pos > cur else -1
        self._turn_key = "Left" if self._dir > 0 else "Right"  # Left = counterclockwise
        self.keys.press(self._turn_key)
        self.state, self.target, self._t0 = "moving", pos, time.time()
        self._from, self._retried = cur, 0
        self._ctx = ctx or {"kind": "stop"}
        return True

    def clamp(self, pos):
        """At a rotation limit: hold into the stop so it fires the flatter
        (table) lane. Up after CLAMP_HOLD_S fires along it."""
        self.keys.press("Left" if pos == M.N_POS - 1 else "Right")
        self.state, self.target, self._t0 = "clamping", pos, time.time()
        self._ctx = {"kind": "clamp"}

    def stop(self, why="preempted"):
        if self.state != "idle":
            self.world.clamped = False
            self.fire({"kind": "stop", "why": why})
            self.state = "idle"

    def step(self, t):
        if self.state == "moving":
            p = self.world.barrel
            if p == self._from and t - self._t0 > STUCK_S * (self._retried + 1) and self._retried < 3 \
                    and self._keys_settled():
                self.keys.press(self._turn_key)  # the turn key didn't take: send it again
                self._retried += 1
            if (p is not None and (self.target - p) * self._dir <= 0) or t - self._t0 > GOTO_TIMEOUT_S:
                self.world.clamped = False  # stopped on sight: unclamped at a limit
                self.fire(self._ctx, p)
                self.state = "idle"
        elif self.state == "clamping" and t - self._t0 >= CLAMP_HOLD_S:
            self.world.clamped = True
            self.fire(self._ctx, self.target)
            self.state = "idle"
