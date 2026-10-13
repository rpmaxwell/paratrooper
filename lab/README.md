# lab: exploration

Code here may import `paratrooper`; `paratrooper` never imports `lab`. Lab
code is not maintained with the package: when a script breaks after a
package change, fix it if the question is still open, else delete it (git
keeps it). Code moves into the package when a game run needs it, and moves
with a test.

Run scripts from the repo root with the package importable
(`uv pip install -e .`, or `PYTHONPATH=.`; in the container the repo is at
/repo and on PYTHONPATH).

| topic | what | status |
|---|---|---|
| [calibration/](calibration/) | fits that regenerate `paratrooper/game/data/*` and `sim/params`: `fit.py` (helicopter / drop model from telemetry), `phit.py` (p(hit) vs outcomes), `collision.py` (hitboxes from frames, since confirmed by the game's code) | rerun after new live runs |
| [disasm/](disasm/) | `disasm.py` (needs `pip install capstone`): full listing of the game's code with labels and cross-references | reference |
| [diagnostics/](diagnostics/) | `audit_misses.py` (why troopers landed), `review_crush.py`, `show_trooper.py`, `mem_diag.py` (memory vs vision), `check_telemetry.py` (self-checks of a run; its `audit()` needs `scripts/audit_trooper_shots.py` from tag `pre-rebuild`), `check_io.py` (live screen/keys latency) | tools; read captures in the old layout (`game_<t0>.jsonl` / `.npz`) |
| [turbo/](turbo/) | faster-than-real-time emulation | settled: [README](turbo/README.md) |
| [notebooks/](notebooks/) | `sim_spawn_drop.ipynb`: spawn / drop simulator vs telemetry | open |

The history of how the bot got here: [docs/dev_history.md](../docs/dev_history.md).
