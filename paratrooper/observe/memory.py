"""Memory observer: helicopters, troopers, bombs, bullets, the barrel and
the tick counter from the game's own memory (memstate.py); planes, landed
heads and grounded bodies still from the screen (not decoded yet)."""
from ..emulator.process_memory import STATE_LO
from .base import Observation, Observer
from .memstate import MemState, sprites_from_memory
from .sprites import detect


class MemoryObserver(Observer):
    needs_memory = True

    def barrel(self, frame, snap=None):
        return MemState(snap, STATE_LO).barrel()[0]

    def observe(self, frame, snap=None):
        ms = MemState(snap, STATE_LO)
        return Observation(sprites_from_memory(ms, detect(frame)), ms.barrel()[0], ms.tick)
