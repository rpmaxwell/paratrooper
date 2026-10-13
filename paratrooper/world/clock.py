"""Game-tick clock.

The game advances on the PC timer, 18.2065 Hz. We never see the tick
itself, only its effect: sprites move, and each move happened at a tick
boundary shortly BEFORE the frame that shows it was captured. The boundary
phase is tracked as the smallest capture lag over a sliding window, with
wrap-around handled (a lag just under a full tick means the boundary is
slightly before the anchor, not a tick later). tick(t) numbers ticks
consistently for the whole game: every bullet step advances exactly one.
"""
from collections import deque

TICK_S = 1 / 18.2065
_WRAP = 0.75 * TICK_S


class GameClock:
    def __init__(self, t0):
        self.t0 = t0
        self._lags = deque(maxlen=40)

    def _lag(self, t):
        lag = (t - self.t0) % TICK_S
        return lag - TICK_S if lag > _WRAP else lag

    def observe_move(self, t):
        """A sprite moved in the frame captured at t."""
        self._lags.append(self._lag(t))
        if len(self._lags) >= 8:
            shift = min(self._lags)
            if abs(shift) > 0.002:  # boundary is `shift` after the anchor
                self.t0 += shift
                self._lags = deque(l - shift for l in self._lags)

    def tick(self, t):
        return int((t - self.t0) // TICK_S)
