import json
import tomllib

import numpy as np

from paratrooper import config as C
from paratrooper.cli import archive
from paratrooper.record import store
from paratrooper.record.game import GameRecorder, interesting_ticks, window_mask


def test_window_mask_keeps_frames_near_events():
    ticks = np.arange(0, 200)
    m = window_mask(ticks, [50, 120], 5)
    assert set(ticks[m]) == set(range(45, 56)) | set(range(115, 126))
    assert not window_mask(ticks, [], 5).any()


def test_interesting_ticks_are_misses_and_unexplained():
    ev = [dict(type="trooper", fate="landed", y_by_tick=[[10, 300], [14, 330]]),
          dict(type="trooper", fate="body_killed", fate_evidence=7, y_by_tick=[[3, 100]]),
          dict(type="bomb", fate="vanished_unexplained", last_k=5, first_tick=20),
          dict(type="bullet", end=dict(kind="vanished", tick=33))]
    assert interesting_ticks(ev) == [14, 20, 33]


def test_game_recorder_layout(tmp_path):
    cfg = C.Record(frames="windows", window_ticks=2, memory=True)
    rec = GameRecorder(tmp_path, "g000", cfg)
    for k in range(20):
        rec.frame(k * 0.055, k, np.full((200, 320), k % 4, np.uint8))
        rec.memory(k * 0.055, k, np.full(16, k, np.uint8))
    rec.emit(dict(type="trooper", fate="landed", y_by_tick=[[10, 330]]))
    rec.close(dict(score=5), mem_lo=0x1B00)
    d = tmp_path / "g000"
    assert [json.loads(l)["type"] for l in (d / "events.jsonl").read_text().splitlines()] == ["trooper"]
    f = np.load(d / "frames.npz")
    assert list(f["tick"]) == [8, 9, 10, 11, 12]
    assert len(np.load(d / "memory.npz")["tick"]) == 20
    assert json.loads((d / "summary.json").read_text())["frames_saved"] == 5


def test_run_and_archive(tmp_path):
    a = C.load(store.REPO / "configs" / "base.toml")
    b = C.load(store.REPO / "configs" / "base.toml", ["name=b", "policy.params.p_min=0.6"])
    run = store.new_run("ab_test", {"base": a, "b": b}, root=tmp_path / "runs")
    m = json.loads((run / "manifest.json").read_text())
    assert set(m["arms"]) == {"base", "b"} and len(m["git"]["sha"]) == 40
    assert C.load(run / "b" / "config.toml").policy.params["p_min"] == 0.6
    for arm, score in (("base", 100), ("b", 140), ("b", 160)):
        g = run / arm / f"g{score}"
        g.mkdir()
        (g / "summary.json").write_text(json.dumps(dict(arm=arm, score=score, started=score)))
    paths = archive.add(run, root=tmp_path / "archive")
    meta = {p.stem.split("__")[1]: tomllib.loads(p.read_text())["meta"] for p in paths}
    assert meta["b"]["mean_score"] == 150.0 and meta["b"]["status"] == "pending"
    again = C.load(paths[0])          # an archived config is directly runnable
    assert again.meta["git_sha"] == m["git"]["sha"]


def test_bullet_audit_on_fixture_with_memory():
    """With the memory observer every bullet is tracked by its slot, so the
    audit finds no lost bullets and no false hits."""
    import pathlib
    from paratrooper.cli.replay import load_game, replay
    from paratrooper.emulator.process_memory import STATE_LO
    from paratrooper.record.audit import bullet_audit
    fix = pathlib.Path(__file__).parent / "fixtures" / "g_short"
    t, frames, snaps = load_game(fix)
    audits = {}
    for credit in (False, True):
        cfg = C.load(store.REPO / "configs" / "base.toml", ["observer.kind=memory", f"world.credit_by_score={str(credit).lower()}"])
        ev, w, _ = replay(cfg, t, frames, snaps)
        a = audits[credit] = bullet_audit(ev, w.frame_ticks, snaps[:len(w.frame_ticks)], STATE_LO)
        assert a["bullets"] > 50 and a["ended_early"] == 0 and a["false_hit"] == 0
    # score crediting only adds credits, and none of them to bullets that flew on
    assert audits[True]["credited"] >= audits[False]["credited"] > 0.85 * audits[False]["collisions"]
