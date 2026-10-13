"""Vision observer: everything from the screen -- sprite shapes
(sprites.py), the barrel's tip (barrel.py). Ticks are inferred from sprite
motion by the world's clock, which assumes the game runs in real time."""
from .barrel import barrel_pos
from .base import Observation, Observer
from .sprites import detect


class VisionObserver(Observer):
    def barrel(self, frame, snap=None):
        return barrel_pos(frame)

    def observe(self, frame, snap=None):
        return Observation(detect(frame), barrel_pos(frame))
