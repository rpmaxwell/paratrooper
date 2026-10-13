"""Background JSONL writer: the control loop only appends to a queue.

One file per game: <out_dir>/game_<start>.jsonl, one JSON object per line,
each with a "type" field (trooper, bullet, bomb, aircraft, score, press,
phase, game). See telemetry/README section in refactor_plan.md.
"""
import json
import queue
import threading


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
    except ImportError:
        pass
    return str(o)


class Writer:
    def __init__(self):
        self._q = queue.Queue()
        self._fh = None
        threading.Thread(target=self._run, daemon=True).start()

    def open(self, path):
        self._q.put(("open", path))

    def emit(self, record):
        self._q.put(("rec", record))

    def close(self):
        self._q.put(("close", None))

    def flush(self, timeout=5.0):
        done = threading.Event()
        self._q.put(("sync", done))
        done.wait(timeout)

    def _run(self):
        while True:
            op, arg = self._q.get()
            if op == "open":
                if self._fh:
                    self._fh.close()
                self._fh = open(arg, "w")
            elif op == "rec" and self._fh:
                self._fh.write(json.dumps(arg, default=_default) + "\n")
            elif op == "close" and self._fh:
                self._fh.close()
                self._fh = None
            elif op == "sync":
                if self._fh:
                    self._fh.flush()
                arg.set()
