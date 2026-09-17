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
                    "limit": sample.limit.type if sample.limit is not None else None,
                    "n_turns": sum(1 for m in sample.messages if m.role == "assistant"),
                    "output_tokens": sum(u.output_tokens for u in (sample.model_usage or {}).values()),
                    "seconds": sample.total_time,
                }
            )
        except Exception as e:  # still record that the sample finished
            record["progress_error"] = repr(e)[:200]
        self._write(record)
