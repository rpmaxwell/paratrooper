"""Actuator for offline replay: records presses, sends nothing. Same
interface as xtest.Keys."""
import time

HOLD_S = 0.02


class FakeKeys:
    def __init__(self):
        self._down = {}
        self.log = []

    def press(self, name, ctx=None):
        now = time.time()
        self._down[name] = now + HOLD_S
        self.log.append((now, name, ctx))
        return now

    def pump(self):
        now = time.time()
        for name, due in list(self._down.items()):
            if now >= due:
                self._down.pop(name)

    def tap(self, name, ctx=None):
        self.press(name, ctx)

    def release_all(self):
        self._down.clear()
