"""In-process keyboard input through the X server's XTEST extension.

xdotool (the old path) starts a new process for every key: 23-32 ms per
press, the likely source of most +-1 tick fire-timing errors. XTEST on a
persistent connection sends the event in well under a millisecond.

press() is non-blocking: it sends key-down now and schedules the key-up
HOLD_S later; pump() (called every loop iteration) sends due key-ups.
Every press is recorded in `log` as (time, key, context) -- the raw
material for the telemetry bullet ledger.
"""
import time

from Xlib import X, XK, display
from Xlib.ext import xtest

HOLD_S = 0.02  # same hold the xdotool path used


class Keys:
    def __init__(self):
        self._d = display.Display()
        self._codes = {}
        self._down = {}  # key -> release time
        self.log = []

    def _code(self, name):
        if name not in self._codes:
            self._codes[name] = self._d.keysym_to_keycode(XK.string_to_keysym(name))
        return self._codes[name]

    def press(self, name, ctx=None):
        now = time.time()
        if name in self._down:  # re-press: release first so it registers
            self._release(name)
        xtest.fake_input(self._d, X.KeyPress, self._code(name))
        self._d.flush()
        self._down[name] = now + HOLD_S
        self.log.append((now, name, ctx))
        return now

    def _release(self, name):
        xtest.fake_input(self._d, X.KeyRelease, self._code(name))
        self._d.flush()
        self._down.pop(name, None)

    def pump(self):
        now = time.time()
        for name, due in list(self._down.items()):
            if now >= due:
                self._release(name)

    def tap(self, name, ctx=None):
        """Blocking press+release (for scripts, not the control loop)."""
        self.press(name, ctx)
        time.sleep(HOLD_S)
        self.pump()

    def release_all(self):
        for name in list(self._down):
            self._release(name)
