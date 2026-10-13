"""Where runs go. One layout for every run, live or replayed:

    runs/<YYYY-MM-DD>_<name>/
      manifest.json              git sha + branch, created, arms, emulator settings
      git.diff                   uncommitted changes the run included (if any)
      <arm>/config.toml          the arm's fully resolved config (re-runnable as is)
      <arm>/<game_id>/
        events.jsonl             one event per line, each with "type" and "tick"
        summary.json             score, counts, timing, run/arm/config ids (one row of the games table)
        frames.npz               t, tick, frames (N x 200 x 320 palette indices)   [record.frames]
        memory.npz               t, tick, mem (N x STATE bytes), lo, base           [record.memory]

`tick` is the same numbering in events, frames and memory, so an event can
be looked up in the frames around it (record/query.py). Every summary.json
carries run, arm and config hash, so all games of all runs form one table:
    duckdb: SELECT * FROM read_json_auto('runs/*/*/*/summary.json')
"""
import datetime
import hashlib
import json
import pathlib
import subprocess

from .. import config as C

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"


def config_hash(resolved):
    return hashlib.sha1(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:10]


def git_info(repo=REPO):
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    diff = git("diff", "HEAD")
    return dict(sha=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(diff)), diff


def new_run(name, arms, root=RUNS, emulator=None):
    """Create a run directory for `arms` ({arm name: Config}); host side
    (needs git). -> run dir path."""
    day = datetime.date.today().isoformat()
    base = root / f"{day}_{name}"
    run, n = base, 1
    while run.exists():
        n += 1
        run = base.with_name(f"{base.name}_{n}")
    run.mkdir(parents=True)
    git, diff = git_info()
    if diff:
        (run / "git.diff").write_text(diff)
    arm_info = {}
    for arm, cfg in arms.items():
        resolved = cfg.resolved()
        (run / arm).mkdir()
        (run / arm / "config.toml").write_text(C.dumps(resolved))
        arm_info[arm] = dict(config_hash=config_hash(resolved))
    manifest = dict(run=run.name, created=datetime.datetime.now().isoformat(timespec="seconds"),
                    git=git, arms=arm_info, emulator=emulator)
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return run


def load_arm(run, arm):
    """-> (Config, manifest) for one arm of a run."""
    run = pathlib.Path(run)
    cfg = C.load(run / arm / "config.toml")
    manifest = json.loads((run / "manifest.json").read_text())
    return cfg, manifest


def game_dirs(run):
    """Every game directory of a run, in play order."""
    return sorted((p.parent for p in pathlib.Path(run).glob("*/*/summary.json")),
                  key=lambda p: json.loads((p / "summary.json").read_text()).get("started", 0))


def summaries(run):
    return [json.loads((g / "summary.json").read_text()) for g in game_dirs(run)]
