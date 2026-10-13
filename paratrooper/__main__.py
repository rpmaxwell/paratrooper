"""python -m paratrooper <command> -- host-side commands.

  up [--instance N] [--speed realtime|turbo] [--cycles C] [-c CONFIG]
                               start an emulator container
  down [--instance N]          stop it
  play CONFIG [...]            play games with a config (cli/play.py)
  ab CONFIG_A CONFIG_B [...]   A/B two configs, then archive both (cli/play.py)
  archive list|add             the experiment archive (cli/archive.py)
  replay GAME [...]            run a recorded game through the stack offline (cli/replay.py)
  runs [RUN]                   list runs, or one run's games
"""
import argparse
import sys


def _up(argv):
    from . import config as C
    from .emulator import docker
    ap = argparse.ArgumentParser(prog="python -m paratrooper up")
    ap.add_argument("--instance", type=int)
    ap.add_argument("-c", "--config", help="take speed and cycles from this config's [emulator]")
    ap.add_argument("--speed", choices=C.SPEEDS)
    ap.add_argument("--cycles", type=int)
    a = ap.parse_args(argv)
    emu = C.load(a.config, check_policy=False, playable=False).emulator if a.config else C.Emulator()
    docker.up(a.instance, a.speed or emu.speed, a.cycles or emu.cycles)


def _down(argv):
    from .emulator import docker
    ap = argparse.ArgumentParser(prog="python -m paratrooper down")
    ap.add_argument("--instance", type=int)
    docker.down(ap.parse_args(argv).instance)


def _runs(argv):
    import json
    from .cli.play import compare
    from .record.store import RUNS
    if argv:
        print(compare(RUNS / argv[0] if not argv[0].startswith(("/", ".")) else argv[0])[0])
        return
    for run in sorted(p for p in RUNS.glob("*") if (p / "manifest.json").exists()):
        m = json.loads((run / "manifest.json").read_text())
        n = len(list(run.glob("*/*/summary.json")))
        print(f"{run.name:50s} {n:4d} games  arms: {', '.join(m['arms'])}  git {m['git']['sha'][:8]}"
              + (" (dirty)" if m["git"]["dirty"] else ""))


def main():
    cmds = {"up": _up, "down": _down, "runs": _runs}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd in ("play", "ab"):
        from .cli import play
        getattr(play, cmd)(argv)
    elif cmd == "archive":
        from .cli import archive
        archive.main(argv)
    elif cmd == "replay":
        from .cli import replay
        replay.main(argv)
    elif cmd in cmds:
        cmds[cmd](argv)
    else:
        raise SystemExit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
