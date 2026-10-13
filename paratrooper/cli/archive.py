"""The experiment archive: configs/archive/<run>__<arm>.toml, one per arm of
every finished experiment -- winners and losers. Each is the arm's fully
resolved config plus a [meta] table (experiment, date, git_sha, run,
summary, result, status), so a parameter variant runs again as is, and a
variant whose code was later deleted names the commit to check out.

    python -m paratrooper archive list
    python -m paratrooper archive add RUN_DIR      (ab does this itself)
"""
import argparse
import json
import pathlib
import tomllib

from .. import config as C
from ..record import store

ARCHIVE = pathlib.Path(__file__).resolve().parents[2] / "configs" / "archive"


def add(run, summary_rows=None, root=ARCHIVE):
    run = pathlib.Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    rows = summary_rows if summary_rows is not None else store.summaries(run)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for arm in manifest["arms"]:
        raw = tomllib.loads((run / arm / "config.toml").read_text())
        scores = [r["score"] for r in rows if r["arm"] == arm]
        raw["meta"] = dict(experiment=manifest["run"], arm=arm, date=manifest["created"][:10],
                           git_sha=manifest["git"]["sha"], git_dirty=manifest["git"]["dirty"],
                           run=f"runs/{run.name}", games=len(scores),
                           mean_score=round(sum(scores) / len(scores), 1) if scores else None,
                           result="", status="pending")
        path = root / f"{run.name}__{arm}.toml"
        path.write_text(C.dumps(raw))
        out.append(path)
    return out


def list_(root=ARCHIVE):
    rows = []
    for p in sorted(root.glob("*.toml")):
        m = tomllib.loads(p.read_text()).get("meta", {})
        rows.append((m.get("date", ""), m.get("status", ""), str(m.get("mean_score", "")), str(m.get("games", "")),
                     p.stem, m.get("result", "")))
    w = [max([len(r[i]) for r in rows] + [len(h)]) for i, h in enumerate(("date", "status", "score", "n", "config"))]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(("date", "status", "score", "n", "config"))) + "  result")
    for r in rows:
        print("  ".join(r[i].ljust(w[i]) for i in range(5)) + "  " + r[5])


def main(argv):
    ap = argparse.ArgumentParser(prog="python -m paratrooper archive")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("add")
    p.add_argument("run")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        list_()
    else:
        for path in add(a.run):
            print(f"archived {path}")
