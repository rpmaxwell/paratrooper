import pathlib

import pytest

from paratrooper import config as C

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_every_shipped_config_loads():
    for p in (ROOT / "configs").rglob("*.toml"):
        C.load(p, playable=False)


def test_base_is_playable_and_defaults_resolve():
    cfg = C.load(ROOT / "configs" / "base.toml")
    r = cfg.resolved()
    assert r["policy"]["params"]["p_min"] == 0.5          # Params default filled in
    assert C.load(ROOT / "configs" / "observer" / "memory.toml").observer.kind == "memory"


def test_extends_merges_and_set_overrides(tmp_path):
    write(tmp_path, "a.toml", 'name = "a"\n[policy.params]\np_min = 0.6\nmax_free_plans = 3\n')
    b = write(tmp_path, "b.toml", 'extends = "a.toml"\nname = "b"\n[policy.params]\np_min = 0.7\n')
    cfg = C.load(b, ["policy.params.p_slack=0.1", "games=3"])
    assert cfg.name == "b" and cfg.games == 3
    assert cfg.policy.params == {"p_min": 0.7, "max_free_plans": 3, "p_slack": 0.1}


@pytest.mark.parametrize("text, msg", [
    ('[observer]\nkind = "telepathy"\n', "not in this code"),
    ('[observr]\nkind = "vision"\n', "unknown key"),
    ('[policy.params]\np_mni = 0.5\n', "unknown key"),
    ('[policy.params]\np_min = "high"\n', "expected float"),
    ('[record]\nframes = "some"\n', "record.frames"),
    ('[emulator]\nspeed = "turbo"\n', "turbo"),
])
def test_bad_configs_fail_at_load(tmp_path, text, msg):
    with pytest.raises(C.ConfigError, match=msg):
        C.load(write(tmp_path, "c.toml", text))


def test_turbo_loads_for_the_emulator_only():
    cfg = C.load(ROOT / "configs" / "emulator" / "turbo.toml", playable=False)
    assert (cfg.emulator.speed, cfg.emulator.cycles) == ("turbo", 300)


def test_archived_kind_that_is_gone_names_the_commit(tmp_path):
    p = write(tmp_path, "old.toml", '[policy]\nkind = "ev"\n[meta]\ngit_sha = "abc1234def"\n')
    with pytest.raises(C.ConfigError, match="git worktree add ../pt-abc1234 abc1234def"):
        C.load(p)


def test_resolved_round_trips_through_toml(tmp_path):
    cfg = C.load(ROOT / "configs" / "base.toml", ["policy.params.p_min=0.55"])
    p = write(tmp_path, "r.toml", C.dumps(cfg.resolved()))
    again = C.load(p)
    assert again.resolved() == cfg.resolved()
