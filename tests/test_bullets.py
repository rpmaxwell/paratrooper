"""Bullet tracking (world/world.py) against what the game's memory shows:
bullets move one lane step every game tick and keep their memory slot
until the game frees them."""
from paratrooper.game import model as M
from paratrooper.observe.sprites import Sprites
from paratrooper.world import world as W


def test_assign_keeps_a_lane_chain_in_order():
    # two bullets one step apart; the clock says 1 tick passed but 2 did.
    # Bullet by bullet, the front one takes the back one's dot (2 steps
    # from the back = 1 from the front) and the back one is lost.
    p0, v = (300, 200), (2, -16)
    dot = lambda k: (p0[0] + v[0] * k, p0[1] + v[1] * k)
    front = [(0, 1, dot(2)), (1, 2, dot(3))]       # (cost, steps, dot) from front at dot(1)
    back = [(1, 2, dot(2))]                        # back at dot(0)
    assert W._assign([front, back]) == [(2, dot(3)), (2, dot(2))]


def test_assign_without_conflicts_is_first_choice():
    a, b = [(0, 1, (1, 1)), (1, 2, (2, 2))], [(0, 1, (5, 5))]
    assert W._assign([a, b]) == [(1, (1, 1)), (1, (5, 5))]


def _world():
    return W.World(0.0, lambda r: None)


def test_bullet_reaching_x0_left_the_screen():
    w = _world()
    lane = 20                                       # left sight lane, (-20, -6) per tick
    b = W.Bullet(1, 0.0, lane, {"kind": "stop"}, pos=(20, 225), t=0.0, first_tick=0, path=[(0, 20, 225)])
    w.bullets.append(b)
    w._end_bullet(b, 2)
    assert b.end["kind"] == "left_screen"


def test_memory_slot_is_the_bullets_identity():
    w = _world()
    lane = 9
    (sx, sy), (vx, vy) = M.LANES[lane]
    b = W.Bullet(1, 0.0, lane, {"kind": "stop"}, pos=(sx, sy), t=0.0, first_tick=0, path=[(0, sx, sy)], slot=4)
    w.bullets.append(b)
    s = Sprites(dots=[(sx + vx, sy + vy)], dot_slots=[4])
    w._bullets(s, M.TICK_S * 1.2, 1)
    assert b.pos == (sx + vx, sy + vy) and b in w.bullets
    w._bullets(Sprites(dots=[], dot_slots=[]), M.TICK_S * 2.2, 2)   # slot freed -> ended at once
    assert b not in w.bullets and b.end is not None
