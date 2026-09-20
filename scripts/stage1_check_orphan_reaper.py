#!/usr/bin/env python
"""Reproduce the tool-call hang of the `local` sandbox and check that the orphan reaper clears it.

No GPU or vLLM needed: a mock model makes the tools scaffold run `python3 -c "while True: pass" | head -50`
(what hung run 9b-promptA-full), then submits. The tool times out at 60 s; its python child survives,
holds the pipe and the call never returns unless something kills it.

  rhbench-run uv run scripts/stage1_check_orphan_reaper.py                   # expect: PASS in ~1.5 min
  rhbench-run uv run scripts/stage1_check_orphan_reaper.py --without-reaper  # expect: HANG REPRODUCED

Runs model-written-style code on the host: on the GPU box use rhbench-run, like the benchmark.
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.orphan_reaper import OrphanReaper  # noqa: E402

MARKER = "RHABLATION_ORPHAN_REPRO"
HANG_CMD = f'python3 -c "while True: pass  # {MARKER}" | head -50'


def marker_processes() -> list[psutil.Process]:
    found = []
    for p in psutil.process_iter(["cmdline"]):
        cmdline = p.info.get("cmdline") or []
        if p.pid != os.getpid() and cmdline[:1] != ["bash"] and any(MARKER in part for part in cmdline) and "python" in cmdline[0]:
            found.append(p)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grace-s", type=int, default=10)
    parser.add_argument("--without-reaper", action="store_true", help="show the hang instead of the fix")
    parser.add_argument("--give-up-s", type=int, default=200, help="declare a hang after this long and clean up")
    args = parser.parse_args()

    from impossiblebench import impossible_livecodebench
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    work = Path(tempfile.mkdtemp(prefix="orphan-reaper-check-"))
    reaped_log = work / "reaped_processes.jsonl"
    task = impossible_livecodebench(split="original", agent_type="tools", sandbox="local", max_attempts=1, message_limit=10)
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call("mockllm/model", "bash", {"command": HANG_CMD}),
            ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "DONE"}),
        ],
    )

    if not args.without_reaper:
        OrphanReaper(reaped_log, grace_s=args.grace_s).start()

    hung = threading.Event()

    def give_up() -> None:
        time.sleep(args.give_up_s)
        hung.set()
        for p in marker_processes():
            print(f"give-up: killing leftover pid {p.pid} so the check can end", flush=True)
            p.kill()

    threading.Thread(target=give_up, daemon=True).start()

    start = time.time()
    logs = inspect_eval(task, model=model, sample_id=[str(task.dataset[0].id)], log_dir=str(work / "logs"), display="plain", fail_on_error=False)
    elapsed = time.time() - start
    leftovers = marker_processes()
    for p in leftovers:
        p.kill()
    reaped = [json.loads(line) for line in reaped_log.read_text().splitlines()] if reaped_log.exists() else []
    reaped_marker = [r for r in reaped if any(MARKER in part for part in r.get("cmdline", []))]

    print(f"\neval status={logs[0].status} after {elapsed:.0f} s; reaped {len(reaped)} process(es), {len(reaped_marker)} of them the looping python; leftovers {len(leftovers)}")
    for r in reaped:
        print("  reaped:", r)
    if args.without_reaper:
        print("HANG REPRODUCED (the tool call only returned once the give-up timer killed the process)" if hung.is_set() else "no hang: the tool call returned by itself, so this Inspect version/sandbox is not affected")
        sys.exit(0)
    ok = not hung.is_set() and len(reaped_marker) == 1 and not leftovers
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
