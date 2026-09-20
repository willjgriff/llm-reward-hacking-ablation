"""Kill model-written processes that outlive their tool call under Inspect's `local` sandbox.

Why: when a tool call times out, the shell it started is killed but that shell's children are not.
A command like `python test.py | head -50` with an infinite loop leaves `python` running, reparented
to PID 1 and still holding the output pipe, so the tool call never returns, the sample never ends
and the run hangs (run `9b-promptA-full`: three samples blocked for 1.5-3 hours until the processes
were killed by hand). Docker sandboxes do not have this problem.

How: a daemon thread in the benchmark process records every process started under one of the
"roots": the benchmark process itself and Inspect's sandbox-tools servers (detached daemons that
execute the tool calls; recognised by their executable living in Inspect's tools directory). A
recorded process that is later found outside every root's process tree, i.e. orphaned, and stays
that way for `grace_s`, is killed. Roots are never killed, and nothing this run did not start is
touched. Each kill is appended to `<run_dir>/reaped_processes.jsonl`.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

POLL_S = 2.0


def sandbox_tools_dir() -> str | None:
    try:
        from inspect_ai.util._sandbox._cli import SANDBOX_TOOLS_DIR

        return SANDBOX_TOOLS_DIR
    except Exception:
        return None


class OrphanReaper:
    def __init__(self, log_path: Path, grace_s: float, poll_s: float = POLL_S) -> None:
        self.log_path = log_path
        self.grace_s = grace_s
        self.poll_s = poll_s
        self.me = psutil.Process(os.getpid())
        self.uid = os.getuid()
        self.tools_dir = sandbox_tools_dir()
        # (pid, create_time) identifies a process even if the pid is reused later.
        self.seen: dict[tuple[int, float], psutil.Process] = {}
        self.orphaned_since: dict[tuple[int, float], float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="orphan-reaper", daemon=True)

    def start(self) -> "OrphanReaper":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.poll_s + 5)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.sweep()
            except Exception as e:  # never let the reaper take the run down
                print(f"orphan reaper: sweep failed: {e!r}", flush=True)

    def _is_tools_process(self, p: psutil.Process) -> bool:
        """Inspect's own sandbox-tools binary (server or CLI), never model-written code."""
        if not self.tools_dir:
            return False
        try:
            return p.exe().startswith(self.tools_dir + "/")
        except psutil.Error:
            return False

    def _roots(self) -> list[psutil.Process]:
        roots = [self.me]
        if self.tools_dir:
            for p in psutil.process_iter(["uids"]):
                uids = p.info.get("uids")
                if uids and uids.real == self.uid and self._is_tools_process(p):
                    roots.append(p)
        return roots

    def sweep(self) -> None:
        now = time.time()
        alive: set[tuple[int, float]] = set()
        for root in self._roots():
            try:
                descendants = root.children(recursive=True)
            except psutil.Error:
                continue
            for p in descendants:
                try:
                    key = (p.pid, p.create_time())
                except psutil.Error:
                    continue
                alive.add(key)
                if not self._is_tools_process(p):
                    self.seen.setdefault(key, p)
        for key in alive:
            self.orphaned_since.pop(key, None)  # back under a root, e.g. seen mid-reparenting

        for key, p in list(self.seen.items()):
            if key in alive:
                continue
            if not p.is_running():  # also false when the pid was reused by another process
                self.seen.pop(key, None)
                self.orphaned_since.pop(key, None)
                continue
            since = self.orphaned_since.setdefault(key, now)
            if now - since >= self.grace_s:
                self._kill(p, now - since)
                self.seen.pop(key, None)
                self.orphaned_since.pop(key, None)

    def _kill(self, p: psutil.Process, orphaned_for_s: float) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "pid": p.pid, "orphaned_for_s": round(orphaned_for_s, 1)}
        try:
            with p.oneshot():
                record["cmdline"] = p.cmdline()
                record["age_s"] = round(time.time() - p.create_time(), 1)
                record["cpu_time_s"] = round(sum(p.cpu_times()[:2]), 1)
            p.kill()
            record["result"] = "killed"
        except psutil.NoSuchProcess:
            record["result"] = "already gone"
        except psutil.Error as e:
            record["result"] = f"failed: {e!r}"
        print(f"orphan reaper: {record}", flush=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
