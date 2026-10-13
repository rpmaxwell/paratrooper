"""Run configuration: TOML files -> dataclasses.

A config names one implementation per swappable part and its parameters:

    [observer]  kind = "memory"
    [policy]    kind = "heuristic"
    [policy.params]  p_min = 0.6

Files compose with `extends = "base.toml"` (path relative to the file;
tables merge, later values win) and command-line `--set policy.params.p_min=0.6`
overrides. Unknown keys, unknown kinds, wrong types and invalid combinations
fail at load time, before a game starts. resolved() is the fully merged
config, saved with every run and archived with every experiment.
"""
import dataclasses
import importlib
import pathlib
import tomllib
from dataclasses import dataclass, field

# kind name -> "module:attribute", imported only when used (the live
# observers and actuators need X11 libraries that offline tools don't)
KINDS = {
    "observer": {
        "vision": "paratrooper.observe.vision:VisionObserver",
        "memory": "paratrooper.observe.memory:MemoryObserver",
    },
    "actuator": {
        "xtest": "paratrooper.act.xtest:Keys",
    },
    "policy": {
        "heuristic": "paratrooper.policy.heuristic:HeuristicPolicy",
    },
}
SPEEDS = ("realtime", "turbo")
FRAMES = ("none", "windows", "all")


class ConfigError(ValueError):
    pass


@dataclass
class Emulator:
    speed: str = "realtime"     # realtime | turbo (emulated time runs as fast as the host allows)
    cycles: int = 3000          # emulated CPU speed; under turbo, lower = faster (lab/turbo)


@dataclass
class Observer:
    kind: str = "vision"


@dataclass
class Actuator:
    kind: str = "xtest"


@dataclass
class Policy:
    kind: str = "heuristic"
    params: dict = field(default_factory=dict)   # checked against the policy's Params dataclass


@dataclass
class World:
    # a bullet that ended mid-screen with no hit in view is credited from the
    # score: a gain matching a target's value (trooper 5, aircraft 10, bomb 30)
    # within a few ticks, given to the nearest such target on its path.
    # Promoted 2026-10-13 (configs/archive/2026-10-13_ab_base_vs_credit_by_score__*)
    credit_by_score: bool = True


@dataclass
class Record:
    frames: str = "none"        # none | windows (around misses and unexplained fates) | all
    window_ticks: int = 40      # frames kept on each side of an event of interest
    memory: bool = False        # the game's state bytes every frame (~0.5 MB/game)


@dataclass
class Config:
    name: str = "base"
    games: int = 1
    emulator: Emulator = field(default_factory=Emulator)
    observer: Observer = field(default_factory=Observer)
    actuator: Actuator = field(default_factory=Actuator)
    policy: Policy = field(default_factory=Policy)
    world: World = field(default_factory=World)
    record: Record = field(default_factory=Record)
    meta: dict = field(default_factory=dict)     # archive only: experiment, git_sha, runs, result, status

    def resolved(self):
        """Everything spelled out, the policy's default params included, so a
        saved config means the same thing after defaults change."""
        d = dataclasses.asdict(self)
        d["policy"]["params"] = dataclasses.asdict(policy_params(self))
        return d


# ---------------------------------------------------------------- loading
def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path, seen=()):
    path = pathlib.Path(path).resolve()
    if path in seen:
        raise ConfigError(f"extends cycle at {path}")
    raw = tomllib.loads(path.read_text())
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _merge(_read(path.parent / parent, seen + (path,)), raw)


def _literal(text):
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text  # bare string


def apply_sets(raw, sets):
    """sets: ["policy.params.p_min=0.6", ...]"""
    for s in sets or ():
        key, eq, val = s.partition("=")
        if not eq:
            raise ConfigError(f"--set expects key=value, got {s!r}")
        *path, last = key.strip().split(".")
        d = raw
        for p in path:
            d = d.setdefault(p, {})
        d[last] = _literal(val.strip())
    return raw


def _build(cls, raw, where):
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a table, got {raw!r}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; known: {sorted(fields)}")
    kw = {}
    for k, v in raw.items():
        f = fields[k]
        if dataclasses.is_dataclass(f.type):
            kw[k] = _build(f.type, v, f"{where}.{k}" if where else k)
        else:
            kw[k] = _check_type(v, f.type, f"{where}.{k}" if where else k)
    return cls(**kw)


def _check_type(v, typ, where):
    ok = {int: lambda x: isinstance(x, int) and not isinstance(x, bool),
          float: lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
          str: lambda x: isinstance(x, str), bool: lambda x: isinstance(x, bool),
          dict: lambda x: isinstance(x, dict)}.get(typ)
    if ok is not None and not ok(v):
        raise ConfigError(f"{where}: expected {typ.__name__}, got {v!r}")
    return float(v) if typ is float else v


def load_kind(slot, kind):
    """The class registered for a kind (imports its module)."""
    try:
        target = KINDS[slot][kind]
    except KeyError:
        raise ConfigError(f"{slot} kind {kind!r} is not in this code; known: {sorted(KINDS.get(slot, {}))}") from None
    mod, _, attr = target.partition(":")
    return getattr(importlib.import_module(mod), attr)


def policy_params(cfg):
    """The policy's Params dataclass filled from [policy.params]."""
    cls = load_kind("policy", cfg.policy.kind)
    return _build(cls.Params, cfg.policy.params, "policy.params")


def validate(cfg, check_policy=True, playable=True):
    for slot in ("observer", "actuator", "policy"):
        kind = getattr(cfg, slot).kind
        if kind not in KINDS[slot]:
            hint = ""
            sha = cfg.meta.get("git_sha")
            if sha:
                hint = f"; it ran at {sha}: git worktree add ../pt-{sha[:7]} {sha}"
            raise ConfigError(f"{slot} kind {kind!r} is not in this code{hint}")
    if cfg.emulator.speed not in SPEEDS:
        raise ConfigError(f"emulator.speed must be one of {SPEEDS}")
    if cfg.record.frames not in FRAMES:
        raise ConfigError(f"record.frames must be one of {FRAMES}")
    if cfg.games < 1:
        raise ConfigError("games must be >= 1")
    if playable and cfg.emulator.speed == "turbo":
        # the control loop, the world's tracking timeouts and the vision tick
        # clock count wall-clock seconds; under turbo a second is 20-90 game
        # ticks. Lifted once they count game ticks (CLAUDE.md, "Ticks inside").
        raise ConfigError("emulator.speed = 'turbo' cannot play yet: the bot's timing is in wall-clock seconds")
    if check_policy:
        policy_params(cfg)
    return cfg


def load(path, sets=(), check_policy=True, playable=True):
    raw = apply_sets(_read(path), sets)
    cfg = _build(Config, raw, "")
    return validate(cfg, check_policy, playable)


def from_dict(raw, check_policy=True, playable=True):
    return validate(_build(Config, raw, ""), check_policy, playable)


# ---------------------------------------------------------------- writing
def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {v!r} as TOML")


def dumps(d, _prefix=()):
    """Nested dict -> TOML text (scalars first, then sub-tables; None omitted)."""
    lines = [f"{k} = {_toml_value(v)}" for k, v in d.items() if v is not None and not isinstance(v, dict)]
    out = "\n".join(lines)
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out += f"\n\n[{'.'.join(_prefix + (k,))}]\n" + dumps(v, _prefix + (k,))
    return out.strip() + "\n"
