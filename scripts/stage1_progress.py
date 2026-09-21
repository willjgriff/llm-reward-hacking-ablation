#!/usr/bin/env python
"""Stage 1: show progress of a running (or finished) benchmark run.

Reads <run-dir>/run_config.json (what was planned) and <run-dir>/progress.jsonl (written live by
rhablation.progress), plus live vLLM stats when the server is reachable.
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

STALL_MINUTES = 30
IMPOSSIBLE_SPLITS = {"oneoff", "conflicting"}


def read_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line may be half-written while the run is live
    return records


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def vllm_stats(base_url: str) -> dict | None:
    root = base_url.rstrip("/").removesuffix("/v1")

    def scrape() -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for line in httpx.get(f"{root}/metrics", timeout=5).text.splitlines():
            if line.startswith("vllm:"):
                name = line.split("{")[0].split(" ")[0]
                try:
                    totals[name] += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return totals

    try:
        first = scrape()
        time.sleep(5)
        second = scrape()
    except Exception:
        return None
    return {
        "running": int(second.get("vllm:num_requests_running", 0)),
        "waiting": int(second.get("vllm:num_requests_waiting", 0)),
        "gen_tok_s": (second.get("vllm:generation_tokens_total", 0) - first.get("vllm:generation_tokens_total", 0)) / 5,
    }


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def report(run_dir: Path, base_url: str | None) -> bool:
    """Print one progress report. Returns True when every planned sample has finished."""
    run_config = json.loads((run_dir / "run_config.json").read_text())
    cfg = run_config["config"]
    epochs = cfg.get("samples_per_task", 1)
    planned = {
        (split, agent): len(ids) * epochs
        for agent, by_split in run_config["selected_task_ids"].items()
        for split, ids in by_split.items()
    }
    records = read_progress(run_dir / "progress.jsonl")
    started, ended = set(), {}
    for r in records:
        key = (r.get("split"), r.get("agent_type"), r.get("task_id"), r.get("epoch"))
        if r["event"] == "sample_start":
            started.add(key)
        elif r["event"] == "sample_end":
            ended[key] = r

    now = datetime.now(timezone.utc)
    run_start = parse_ts(records[0]["ts"]) if records else parse_ts(run_config["created"])
    total_planned, total_done = sum(planned.values()), len(ended)
    finished = total_planned > 0 and total_done >= total_planned
    last_ts = max((parse_ts(r["ts"]) for r in ended.values()), default=None)
    clock_end = last_ts if finished and last_ts else now
    elapsed = (clock_end - run_start).total_seconds()

    print(f"run {run_config['run_id']} | model {cfg['model']} | {now.strftime('%H:%M:%S')} UTC | elapsed {fmt_duration(elapsed)}")
    def hit(done: list[dict], field: str) -> str:
        """'samples (calls)' for a per-call cap; '-' when the run predates the field or has no such cap."""
        values = [r.get(field) for r in done if r.get(field) is not None]
        return f"{sum(1 for v in values if v)} ({sum(values)})" if values else "-"

    print(f"  {'split':12s} {'agent':8s} {'done':>9s} {'running':>8s} {'pass':>5s} {'errors':>7s} {'msg-lim':>8s} {'oth-lim':>8s} {'maxtok':>9s} {'think-cut':>10s}   note")
    for (split, agent), n in sorted(planned.items()):
        done = [r for k, r in ended.items() if k[0] == split and k[1] == agent]
        running = sum(1 for k in started if k[0] == split and k[1] == agent and k not in ended)
        passes = sum(1 for r in done if r.get("score") == "C")
        errors = sum(1 for r in done if r.get("error"))
        msg_limit = sum(1 for r in done if r.get("limit") == "message")
        other_limit = sum(1 for r in done if r.get("limit") and r.get("limit") != "message")
        # Runners for other environments list their impossible variants in run_config.json.
        impossible = IMPOSSIBLE_SPLITS | set(run_config.get("impossible_variants") or [])
        note = "pass = verified hack" if split in impossible else "pass = solved (hack unknown)"
        print(
            f"  {split:12s} {agent:8s} {len(done):4d}/{n:<4d} {running:8d} {passes:5d} {errors:7d} {msg_limit:8d} {other_limit:8d}"
            f" {hit(done, 'calls_at_max_tokens'):>9s} {hit(done, 'calls_thinking_cut'):>10s}   {note}"
        )
    print("  msg-lim/oth-lim: samples stopped by the message limit / another sample-level limit. maxtok, think-cut: samples (calls)")
    print("  with a model call that ended at max_tokens / had its thinking cut at thinking_token_budget; both caps are per call.")

    if ended:
        mean_tokens = sum(r.get("output_tokens") or 0 for r in ended.values()) / len(ended)
        mean_seconds = sum(r.get("seconds") or 0 for r in ended.values()) / len(ended)
        rate = total_done / elapsed * 3600 if elapsed > 0 else 0
        print(f"  total {total_done}/{total_planned} | {rate:.0f} samples/h | mean {mean_tokens:,.0f} output tokens, {fmt_duration(mean_seconds)} per sample")
        if not finished and total_done >= 3 and rate > 0:
            print(f"  rough ETA {fmt_duration((total_planned - total_done) / rate * 3600)} (long-running stragglers make this optimistic)")
    else:
        print(f"  total 0/{total_planned} | no sample has finished yet")

    stats = vllm_stats(base_url) if base_url and not finished else None
    if stats:
        print(f"  vLLM: {stats['running']} running, {stats['waiting']} queued, {stats['gen_tok_s']:,.0f} generated tokens/s")

    if finished:
        print("  FINISHED: all planned samples are done.")
    else:
        idle_min = (now - (last_ts or run_start)).total_seconds() / 60
        print(f"  last sample finished {idle_min:.0f} min ago" if last_ts else f"  waiting for the first sample to finish ({idle_min:.0f} min so far)")
        if stats and stats["gen_tok_s"] == 0 and stats["running"] == 0:
            # A lone sample running its tests between attempts leaves vLLM idle for a few seconds.
            time.sleep(20)
            recheck = vllm_stats(base_url)
            if recheck and recheck["gen_tok_s"] == 0 and recheck["running"] == 0:
                print("  WARNING: vLLM has been idle for 30 s while samples are unfinished. The run may have stopped or crashed.")
        elif idle_min > STALL_MINUTES and not (stats and stats["gen_tok_s"] > 0):
            print(f"  WARNING: no sample finished in {STALL_MINUTES}+ min and vLLM activity could not be confirmed.")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="data/stage1/<run_id>")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="repeat every N seconds until finished")
    parser.add_argument("--vllm-base-url", default=None, help="default: the run's configured URL; 'none' to skip")
    args = parser.parse_args()

    base_url = args.vllm_base_url
    if base_url is None:
        base_url = json.loads((args.run_dir / "run_config.json").read_text())["config"].get("vllm_base_url")
    if base_url == "none":
        base_url = None

    while True:
        finished = report(args.run_dir, base_url)
        if finished or args.watch <= 0:
            break
        print()
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
