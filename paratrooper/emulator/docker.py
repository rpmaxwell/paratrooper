"""Emulator containers (host side): DOSBox-X + Xvfb + VNC in Docker, the
bot running inside next to it. Replaces run.sh / stop.sh.

    python -m paratrooper up [--instance N] [--speed realtime|turbo] [--cycles C]
    python -m paratrooper down [--instance N]

Instance N: container "paratrooper-N", VNC on 5900+N (no N: "paratrooper",
5900). The DOSBox-X config is generated from docker/dosbox-x.conf and the
emulator settings, and the settings are stored as container labels, so a
run can check it is playing on the emulator its config asks for. Mounts:
the repo read-only at /repo (PYTHONPATH), runs/ at /runs, the generated
config at /config/dosbox-x.conf, the game at /assets.
"""
import json
import os
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
IMAGE = "paratrooper-dosbox"
COM = pathlib.Path(os.environ.get("PARATROOPER_COM", "~/Documents/paratrooper/ParaTrooper.1982.com")).expanduser()


def name(instance=None):
    return "paratrooper" if instance is None else f"paratrooper-{instance}"


def conf_text(speed="realtime", cycles=3000):
    """docker/dosbox-x.conf with the emulator settings applied. Turbo lets
    emulated time run as fast as the host can: the game paces itself on the
    BIOS tick (INT 1Ah), which follows emulated time, so cycles becomes the
    throttle (lower = faster; lab/turbo/README.md)."""
    text = (REPO / "docker" / "dosbox-x.conf").read_text()
    text = re.sub(r"^cycles\s*=.*$", f"cycles    = fixed {int(cycles)}", text, flags=re.M)
    if speed == "turbo":
        text = re.sub(r"^(cycledown\s*=.*)$", r"\1\nturbo                   = true\n"
                      r"# default stops turbo at the first keypress -- the first move of a game\n"
                      r"stop turbo on key       = false\nstop turbo after second = 0", text, flags=re.M)
    elif speed != "realtime":
        raise ValueError(f"unknown speed {speed!r}")
    return text


def _vnc_password():
    pw = os.environ.get("VNC_PASSWORD")
    env = REPO / ".env"
    if not pw and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("VNC_PASSWORD="):
                pw = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not pw:
        raise SystemExit("Set VNC_PASSWORD (export it, or put VNC_PASSWORD=... in .env)")
    return pw


def up(instance=None, speed="realtime", cycles=3000, build=True):
    n = name(instance)
    if build:
        subprocess.run(["docker", "build", "-q", "-t", IMAGE, "-f", str(REPO / "docker" / "Dockerfile"),
                        str(REPO / "docker")], check=True, stdout=subprocess.DEVNULL)
    RUNS.mkdir(exist_ok=True)
    conf = RUNS / ".emulator" / f"{n}.conf"
    conf.parent.mkdir(exist_ok=True)
    conf.write_text(conf_text(speed, cycles))
    subprocess.run(["docker", "rm", "-f", n], capture_output=True)
    port = 5900 + (instance or 0)
    subprocess.run([
        "docker", "run", "-d", "--name", n,
        "--label", f"paratrooper.speed={speed}", "--label", f"paratrooper.cycles={cycles}",
        "-e", "VNC_PASSWORD", "-e", "PYTHONPATH=/repo", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-p", f"{port}:5900",
        "-v", f"{COM}:/assets/ParaTrooper.1982.com:ro",
        "-v", f"{conf}:/config/dosbox-x.conf:ro",
        "-v", f"{REPO}:/repo:ro",
        "-v", f"{RUNS}:/runs",
        IMAGE], check=True, stdout=subprocess.DEVNULL, env={**os.environ, "VNC_PASSWORD": _vnc_password()})
    print(f"{n}: {speed} (cycles {cycles}), VNC vnc://localhost:{port}")


def down(instance=None):
    subprocess.run(["docker", "rm", "-f", name(instance)], capture_output=True)
    print(f"stopped {name(instance)}")


def running(instance=None):
    """-> emulator settings of the running container, or None."""
    r = subprocess.run(["docker", "inspect", "-f", "{{json .State.Running}} {{json .Config.Labels}}", name(instance)],
                       capture_output=True, text=True)
    if r.returncode:
        return None
    up_, labels = r.stdout.split(" ", 1)
    if up_ != "true":
        return None
    labels = json.loads(labels) or {}
    return dict(speed=labels.get("paratrooper.speed", "realtime"), cycles=int(labels.get("paratrooper.cycles", 3000)))


def container_path(host_path):
    """runs/... on the host -> /runs/... in the container."""
    return "/runs/" + str(pathlib.Path(host_path).resolve().relative_to(RUNS.resolve()))


def run_games(instance, run_dir, arm, games, start=0):
    """Play games of one arm in the container; output streams to the terminal."""
    cmd = ["docker", "exec", name(instance), "python3", "-m", "paratrooper.runner",
           container_path(run_dir), arm, "--games", str(games), "--start", str(start)]
    subprocess.run(cmd, check=True)
