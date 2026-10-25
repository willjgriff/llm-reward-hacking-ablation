"""Adapter for CheatBench episode records (`<run>/episodes.jsonl` + `run.json`, synced from HF `cheatbench/`).

One line of episodes.jsonl is one attempt of one episode: `episode` ("<id>#<replicate>") repeats when the VM
retried a failed setup, so the attempt number is the line's rank among lines with the same `episode`
(`attempt_index`, counted by the caller, the convention of scripts/cheatbench_judge_openrouter.py and the
provisional CSV's "<episode>/a<n>"). The transcript is the embedded ATIF trajectory (`harbor_trajectory`).

`to_trajectory` maps a record onto the viewer subset of our trajectory schema. The task's own grader state
(`grader_state`, scalar fields only) becomes labels; judge verdicts and the VM's provisional read are joined
by the viewer (`side_labels`, `judge_entries`) because they live in separate files. Pure functions except the
two loaders at the bottom.
"""

import csv
import json
from pathlib import Path
from typing import Any

CHEATBENCH_KEYS = ("episode", "harbor_trajectory", "routing")
FILE_NAME = "episodes.jsonl"
SOURCE = "cheatbench"


def is_cheatbench_record(rec: dict[str, Any]) -> bool:
    return all(k in rec for k in CHEATBENCH_KEYS)


def trajectory_id(run_name: str, episode: str, attempt: int) -> str:
    return f"{run_name}/{episode}/a{attempt}"


def row_of(model: str | None) -> str | None:
    """'cb-a3half' (run.json) or 'vllm/cb_a3half' (record) -> 'a3half'."""
    if not model:
        return None
    return model.removeprefix("vllm/").removeprefix("cb-").removeprefix("cb_")


def _text(message: Any) -> str:
    """An ATIF message or result content: a string or a list of content parts (as core.render.message_text)."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return message.get("text") or f"<{message.get('type')} content, elided>"
    chunks = []
    for part in message or []:
        if isinstance(part, dict):
            src = (part.get("source") or {}).get("path")
            chunks.append(part.get("text") or (f"<{part.get('type')}: {src}>" if src else f"<{part.get('type')} content, elided>"))
        elif isinstance(part, str):
            chunks.append(part)
    return "\n".join(c for c in chunks if c)


def _messages_and_turns(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Our Message shape from ATIF steps. A result without `source_call_id` is the observation text Terminus-2
    sends back as the next user message (it already contains every call's output), so the per-call results of
    that step are not repeated; a step with only per-call results gets them as tool messages."""
    messages: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for s in steps:
        source = s.get("source")
        if source == "agent":
            calls = []
            for c in s.get("tool_calls") or []:
                names[c.get("tool_call_id")] = c.get("function_name")
                calls.append({"id": c.get("tool_call_id"), "function": c.get("function_name"), "arguments": c.get("arguments")})
            messages.append({"role": "assistant", "content": _text(s.get("message")), "tool_calls": calls or None})
            m = s.get("metrics") or {}
            turns.append(
                {
                    "turn_index": len(turns),
                    "reasoning": s.get("reasoning_content"),
                    "finish_reason": None,
                    "stop_reason": None,
                    "usage": {"input_tokens": m.get("prompt_tokens"), "output_tokens": m.get("completion_tokens")},
                    "tool_calls": calls,
                }
            )
        else:
            messages.append({"role": "system" if source == "system" else "user", "content": _text(s.get("message"))})
        results = (s.get("observation") or {}).get("results") or []
        plain = [r for r in results if not r.get("source_call_id")]
        for r in plain or results:
            err = bool((r.get("extra") or {}).get("is_error"))
            if r.get("source_call_id") and not plain:
                messages.append({"role": "tool", "content": _text(r.get("content")), "function": names.get(r["source_call_id"], "tool"), "error": "is_error" if err else None})
            else:
                messages.append({"role": "user", "content": _text(r.get("content"))})
    return messages, turns


def _scalar_labels(state: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": {"value": v, "source": SOURCE} for k, v in (state or {}).items() if isinstance(v, (bool, int, float)) and k not in ("deadline_monotonic",)}


def to_trajectory(rec: dict[str, Any], run_name: str, attempt: int) -> dict[str, Any]:
    """A CheatBench episode record in the shape of our trajectory schema (viewer/renderer subset)."""
    traj = rec.get("harbor_trajectory") or {}
    messages, turns = _messages_and_turns(traj.get("steps") or [])
    labels = {"ok": {"value": bool(rec.get("ok")), "source": SOURCE}}
    labels.update(_scalar_labels(rec.get("grader_state"), "gs_"))
    usage = rec.get("usage") or {}
    steps = traj.get("steps") or []
    return {
        "schema_version": "cheatbench",
        "trajectory_id": trajectory_id(run_name, rec.get("episode"), attempt),
        "run_id": rec.get("run_id"),
        "timestamp": steps[0].get("timestamp") if steps else None,
        "model": rec.get("model"),
        "model_revision": None,
        "benchmark": "cheatbench",
        "task_source": rec.get("task"),
        "variant": row_of(rec.get("model")),
        "task_id": rec.get("id"),
        "agent_type": rec.get("task"),
        "epoch": attempt,
        "status": "completed" if rec.get("ok") else "error",
        "error_message": rec.get("error") or rec.get("failure"),
        "sampling": {"routing_extra_body": (rec.get("routing") or {}).get("extra_body"), "agent_extra": (traj.get("agent") or {}).get("extra")},
        "seed": None,
        "env_options": {k: rec.get(k) for k in ("task", "variant", "replicate", "agent", "agent_version", "failure", "egress")},
        "prompt_config": {"prompt_file": rec.get("prompt_file")},
        "messages": messages,
        "turns": turns,
        "labels": labels,
        "programmatic_hack": None,
        "test_edit_attempt": None,
        "score": None,
        "raw_test_results": rec.get("grader_state"),
        "final_output": rec.get("final_text"),
        "token_counts": {"output_tokens": usage.get("output"), "input_tokens": usage.get("input")},
        "n_turns": rec.get("n_turns") if rec.get("n_turns") is not None else len(turns),
        "n_messages": len(messages),
        "duration_s": rec.get("wall_time"),
        "provenance": {"format": "cheatbench/episodes.jsonl", "run": run_name, "episode": rec.get("episode"), "attempt_index": attempt, "transcript_path": rec.get("transcript_path")},
    }


def strip_bulk(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    steps = (rec.get("harbor_trajectory") or {}).get("steps")
    if steps is not None:
        out["harbor_trajectory"] = f"<{len(steps)} ATIF steps omitted; see the transcript tab>"
    if rec.get("harbor_artifacts"):
        out["harbor_artifacts"] = f"<{len(rec['harbor_artifacts'])} artifacts omitted>"
    return out


def run_config_summary(run_json: Path) -> dict[str, Any] | None:
    try:
        rc = json.loads(run_json.read_text())
    except (OSError, ValueError):
        return None
    return {
        "run_id": rc.get("run_id"),
        "created": rc.get("started"),
        "model": rc.get("model_id"),
        "model_revision": None,
        "benchmark": "cheatbench",
        "env": {"task": rc.get("task"), "variant": rc.get("variant"), "row": row_of(rc.get("model")), "agent": rc.get("agent")},
        "splits": None,
        "agent_types": None,
        "samples_per_task": None,
        "sampling": {**(rc.get("generation_config") or {}), **((rc.get("routing") or {}).get("extra_body") or {})} or None,
        "impossible_variants": None,
    }


# ---------------------------------------------------------------------------------------------
# Side tables (read once by the viewer)


def _runs_by_row_task(runs_root: Path) -> dict[tuple[str, str], str]:
    out = {}
    for rj in runs_root.glob("*/run.json"):
        try:
            meta = json.loads(rj.read_text())
        except (OSError, ValueError):
            continue
        out[(row_of(meta.get("model")), meta.get("task"))] = rj.parent.name
    return out


def _bool(v: str) -> bool | str:
    return {"true": True, "false": False, "y": True, "n": False}.get(v.strip().lower(), v)


def provisional_labels(data_root: Path) -> dict[str, dict[str, Any]]:
    """trajectory_id -> the VM's provisional read (box_report/provisional_labels.csv next to the run dirs)."""
    out: dict[str, dict[str, Any]] = {}
    for csv_path in data_root.rglob("provisional_labels.csv"):
        runs_root = csv_path.parent.parent
        runs = _runs_by_row_task(runs_root)
        src = csv_path.relative_to(data_root).as_posix()
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                run = runs.get((r.get("row"), r.get("task")))
                if run is None:
                    continue
                key, _, a = r["episode_id"].partition("/a")
                labels = {f"prov_{k}": {"value": _bool(r[k]), "source": src} for k in ("cheated", "attempted", "honey_pots_seen", "explicit_prohibition_violated") if k in r}
                if "gave_up_or_degenerate" in r:
                    labels["gave_up_or_degenerate"] = {"value": _bool(r["gave_up_or_degenerate"]), "source": src}
                out[trajectory_id(run, key, int(a or 1))] = labels
    return out


_JUDGE_META = {
    "id", "episode", "attempt_index", "replicate", "task", "row", "model", "record_ok", "record_failure", "judge", "judge_prompt",
    "judge_schema", "parse_ok", "reasoning", "output_mode", "judge_cost_usd", "judge_usage", "judge_attempts", "provider", "model_used",
    "effort", "reasoning_setting", "temperature", "provider_prefs", "finish_reason", "wall_s", "prompt_chars", "judged_at",
}


def judge_records(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    """trajectory_id -> [(file, judge.jsonl record)] from data/cheatbench_judge/<judge>/<run>/judge.jsonl."""
    out: dict[str, list[dict[str, Any]]] = {}
    root = data_root / "cheatbench_judge"
    if not root.exists():
        return out
    for p in sorted(root.glob("*/*/judge.jsonl")):
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "episode" not in rec:
                continue
            tid = trajectory_id(p.parent.name, rec["episode"], int(rec.get("attempt_index") or 1))
            out.setdefault(tid, []).append({"file": p.relative_to(data_root).as_posix(), "rec": rec})
    return out


def judge_side_labels(judged: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Scalar verdict fields of the first judge file per trajectory, as `judge_<field>` labels."""
    out = {}
    for tid, entries in judged.items():
        e = entries[0]
        out[tid] = {f"judge_{k}": {"value": v, "source": e["file"]} for k, v in e["rec"].items() if k not in _JUDGE_META and isinstance(v, (bool, int, float))}
    return out


def judge_entries(judged: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """The viewer's judge-tab shape: true verdict fields as categories, non-empty evidence strings as flags."""
    out = {}
    for tid, entries in judged.items():
        for e in entries:
            rec = e["rec"]
            verdict = {k: v for k, v in rec.items() if k not in _JUDGE_META}
            out.setdefault(tid, []).append(
                {
                    "file": e["file"],
                    "judge": f"{rec.get('judge')} ({rec.get('effort')})",
                    "group": rec.get("row"),
                    "strongest_status": "enacted" if rec.get("cheated") else None,
                    "categories": [k for k, v in verdict.items() if v is True],
                    "flags": [{"step": "-", "channel": "verdict", "category": k, "status": "", "quote": v, "why": ""} for k, v in verdict.items() if isinstance(v, str) and v.strip()],
                    "rationale": rec.get("reasoning"),
                    "judged_at": rec.get("judged_at"),
                }
            )
    return out
