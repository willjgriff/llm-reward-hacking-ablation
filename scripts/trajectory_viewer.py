#!/usr/bin/env python
"""Local web UI for browsing saved trajectories.

    uv run scripts/trajectory_viewer.py [--data-root data] [--port 8765] [--extra-root ortho=../ortho/data]

Then open http://127.0.0.1:8765. Lists every run directory under the data root that holds a
trajectories.jsonl[.gz] (stage1 runs, sweep chunks, ...), shows one row per rollout with score,
reward and every hack signal as its own column, and opens the full transcript (chain of thought,
tool calls, observations, scorer output, stage-3 judge verdicts) on click. An --extra-root in the
ortho layout (Ethan's repo: <root>/inspect/<env>/<log>.jsonl) adds each of those files as a run
under "<name>/inspect/<env>", read through rhablation.ortho_records.

Read-only: nothing is written inside a run directory. The per-run row index is cached under
--cache-dir (default data/.viewer_cache), keyed by file size and mtime. Binds 127.0.0.1 only.
"""

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhablation import viewer  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "src" / "rhablation" / "viewer_static" / "index.html"


def parse_extra_root(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not name or "/" in name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH with a slash-free name, got {spec!r}")
    p = Path(path).resolve()
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"extra root not found: {p}")
    return name, p


def make_handler(data_root: Path, store: viewer.IndexStore, judge: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter than the default
            if "/api/index" not in (args[0] if args else ""):
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/" or url.path == "/index.html":
                    self._send(HTTPStatus.OK, STATIC.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self._json({"data_root": str(data_root), "extra_roots": {k: str(v) for k, v in store.extra_roots.items()}, "runs": viewer.discover_runs(data_root, store.extra_roots)})
                elif url.path == "/api/index":
                    self._json(store.get(q["run"]))
                elif url.path == "/api/trajectory":
                    path = viewer.run_file(data_root, q["run"], store.extra_roots)
                    rec = viewer.load_record(path, int(q["i"]), int(q.get("offset", -1)))
                    self._json(viewer.detail(rec, judge))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, FileNotFoundError, IndexError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="directory scanned for run dirs (default: data)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="row-index cache (default: <data-root>/.viewer_cache)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--extra-root", type=parse_extra_root, action="append", default=[], metavar="NAME=PATH",
                        help="a directory in the ortho layout (<PATH>/inspect/<env>/*.jsonl); repeatable")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        sys.exit(f"data root not found: {data_root}")
    extra_roots = dict(args.extra_root)
    cache_dir = (args.cache_dir or data_root / ".viewer_cache").resolve()
    store = viewer.IndexStore(data_root, cache_dir, extra_roots)
    judge = viewer.load_judge_labels(data_root)
    runs = viewer.discover_runs(data_root, extra_roots)
    extra = "".join(f"; {sum(r['id'].startswith(k + '/') for r in runs)} ortho files under {k}={v}" for k, v in extra_roots.items())
    print(f"{len(runs)} runs under {data_root}{extra}; judge verdicts for {len(judge)} trajectories; cache in {cache_dir}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(data_root, store, judge))
    print(f"serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
