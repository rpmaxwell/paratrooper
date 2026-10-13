"""Regression: a short recorded game (1,400 frames, with memory) replayed
through observer -> world -> policy gives exactly the recorded outcome.
expected.json was produced by the pre-rebuild code (byte-identical event
streams on three full games, both observers). A change that moves these
numbers changes behaviour -- if intended, regenerate expected.json and say
so in the commit."""
import json
import pathlib

import pytest

from paratrooper import config as C
from paratrooper.cli.replay import load_game, replay

FIX = pathlib.Path(__file__).parent / "fixtures" / "g_short"
BASE = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.toml"
EXPECTED = json.loads((FIX / "expected.json").read_text())["expected"]


@pytest.mark.parametrize("observer", ["vision", "memory"])
def test_replay_matches_recorded_outcome(observer):
    t, frames, snaps = load_game(FIX)
    events, world, _ = replay(C.load(BASE, [f"observer.kind={observer}"]), t, frames, snaps)
    want = EXPECTED[observer]
    assert world.stats == want["stats"]
    assert len(events) == want["n_events"]
    assert sum(e["type"] == "decision" for e in events) == want["decisions"]


def test_observers_agree_on_outcomes():
    """Vision and memory see the same troopers land and die on this game."""
    keys = ("trooper_landed", "trooper_chute_killed", "bombs_shot")
    assert {k: EXPECTED["vision"]["stats"][k] for k in keys} == {k: EXPECTED["memory"]["stats"][k] for k in keys}
