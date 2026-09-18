"""Inspect hook that appends run progress to a JSONL file (one line per task/sample event).

Importing this module registers the hook. It is active only while the environment variable
RHABLATION_PROGRESS_FILE points at the file to append to. Read it with scripts/stage1_progress.py.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from inspect_ai.hooks import Hooks, SampleEnd, SampleStart, TaskStart, hooks

ENV_VAR = "RHABLATION_PROGRESS_FILE"


def call_limit_counts(events: list[Any]) -> dict[str, int | None]:
    """Per-call token caps hit within one sample. These are not Inspect sample limits, so
    `sample.limit` never shows them: max_tokens and thinking_token_budget apply to each model call."""
    at_max_tokens = 0
    thinking_cut: int | None = None
    for ev in events:
        if getattr(ev, "event", None) != "model" or getattr(ev, "output", None) is None or ev.error:
            continue
        if ev.output.stop_reason in ("max_tokens", "model_length"):
            at_max_tokens += 1
        budget = ((ev.config.extra_body if ev.config else None) or {}).get("thinking_token_budget")
        if budget is not None:
            thinking_cut = thinking_cut or 0
            reasoning = getattr(ev.output.usage, "reasoning_tokens", None) if ev.output.usage else None
            # vLLM reports budget - 1 for a cut call (the forced </think> takes the last slot).
            if reasoning is not None and reasoning >= budget - 1:
                thinking_cut += 1
    return {"calls_at_max_tokens": at_max_tokens, "calls_thinking_cut": thinking_cut}


@hooks(name="rhablation_progress", description="Appends task and sample progress to a JSONL file.")
class ProgressHooks(Hooks):
    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}

    def enabled(self) -> bool:
        return bool(os.environ.get(ENV_VAR))

    def _write(self, record: dict[str, Any]) -> None:
        # Progress logging must never be able to break a run.
        try:
            record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
            with open(os.environ[ENV_VAR], "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    async def on_task_start(self, data: TaskStart) -> None:
        args = data.spec.task_args or {}
        info = {"task": data.spec.task, "split": args.get("split"), "agent_type": args.get("agent_type")}
        self._tasks[data.eval_id] = info
        self._write({"event": "task_start", "eval_id": data.eval_id, **info})

    async def on_sample_start(self, data: SampleStart) -> None:
        self._write(
            {
                "event": "sample_start",
                "eval_id": data.eval_id,
                "task_id": str(data.summary.id),
                "epoch": data.summary.epoch,
                **self._tasks.get(data.eval_id, {}),
            }
        )

    async def on_sample_end(self, data: SampleEnd) -> None:
        sample = data.sample
        record: dict[str, Any] = {
            "event": "sample_end",
            "eval_id": data.eval_id,
            "task_id": str(sample.id),
            "epoch": sample.epoch,
            **self._tasks.get(data.eval_id, {}),
        }
        try:
            score = next(iter((sample.scores or {}).values()), None)
            record.update(
                {
                    "score": score.value if score is not None else None,
                    "error": sample.error.message[:300] if sample.error is not None else None,
                    # Inspect's sample-level limit type (message, token, time, ...); per-call caps are below.
                    "limit": sample.limit.type if sample.limit is not None else None,
                    **call_limit_counts(sample.events or []),
                    "n_turns": sum(1 for m in sample.messages if m.role == "assistant"),
                    "output_tokens": sum(u.output_tokens for u in (sample.model_usage or {}).values()),
                    "seconds": sample.total_time,
                }
            )
        except Exception as e:  # still record that the sample finished
            record["progress_error"] = repr(e)[:200]
        self._write(record)
