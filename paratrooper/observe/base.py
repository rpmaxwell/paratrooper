"""What every observer returns. The world model takes an Observation and
never knows which observer made it -- swapping observers is a config change
([observer] kind = "vision" | "memory")."""
from dataclasses import dataclass

from .sprites import Sprites


@dataclass
class Observation:
    sprites: Sprites          # moving objects + landed heads, game coordinates
    barrel: int | None        # barrel position 0..18 (None: unreadable)
    mem_tick: int | None = None   # the game's own tick counter, when read from memory


class Observer:
    """observe(frame, snap) -> Observation. snap: the game's state bytes
    (emulator.process_memory STATE_LO..STATE_HI), read right after the frame;
    None unless needs_memory or the run records memory."""
    needs_memory = False

    def barrel(self, frame, snap=None):
        """Just the barrel -- read first each frame so turret control never
        waits for the full observation."""
        raise NotImplementedError

    def observe(self, frame, snap=None):
        raise NotImplementedError
