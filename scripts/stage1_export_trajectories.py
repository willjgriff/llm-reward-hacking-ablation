#!/usr/bin/env python
"""Stage 1: convert Inspect .eval logs from a run into trajectories.jsonl (see rhablation.schema)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.hack_signals import test_file_writes  # noqa: E402
from rhablation.schema import (  # noqa: E402
    SCHEMA_VERSION,
    Message,
    PromptConfig,
    Provenance,
    RenderedText,
    SamplingParams,
    ScoreRecord,
    TokenCounts,
    ToolCallRecord,
    Trajectory,
    Turn,
    TurnUsage,
)

IMPOSSIBLE_VARIANTS = {"oneoff", "conflicting"}


def iter_model_events(events: list[Any]):
    for ev in events:
        if getattr(ev, "event", None) == "model":
            yield ev
        nested = getattr(ev, "events", None)
        if nested:
            yield from iter_model_events(nested)


def split_content(content: str | list[Any]) -> tuple[str, str | None]:
    """Return (visible_text, reasoning) from an Inspect message content."""
    if isinstance(content, str):
        return content, None
    texts, reasonings = [], []
    for part in content:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            texts.append(part.text)
        elif ptype == "reasoning":
            reasonings.append(part.reasoning)
    return "\n".join(texts), ("\n".join(reasonings) if reasonings else None)


def convert_tool_calls(tool_calls: list[Any] | None) -> list[ToolCallRecord]:
    if not tool_calls:
        return []
    return [ToolCallRecord(id=tc.id, function=tc.function, arguments=tc.arguments) for tc in tool_calls]


def convert_message(msg: Any) -> Message:
    text, _reasoning = split_content(msg.content)
    error = getattr(msg, "error", None)
    return Message(
        role=msg.role,
        content=text,
        tool_calls=convert_tool_calls(getattr(msg, "tool_calls", None)) or None,
        tool_call_id=getattr(msg, "tool_call_id", None) if msg.role == "tool" else None,
        function=getattr(msg, "function", None),
        error=error.message if error is not None else None,
    )


def convert_turn(index: int, ev: Any, tokenizer: Any) -> Turn:
    call = ev.call
    resp: dict[str, Any] | None = call.response if call is not None else None
    if not resp or not resp.get("choices"):
        out_choice = ev.output.choices[0] if ev.output and ev.output.choices else None
        if out_choice is not None and not ev.error:
            # Inspect sometimes logs a completed model event without the raw call. Keep the parsed
            # output (reasoning + content); token ids are unavailable for this turn.
            text, reasoning = split_content(out_choice.message.content)
            usage = ev.output.usage
            return Turn(
                turn_index=index,
                reasoning=reasoning,
                content=text,
                tool_calls=convert_tool_calls(out_choice.message.tool_calls),
                stop_reason=out_choice.stop_reason,
                usage=TurnUsage(
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                    reasoning_tokens=usage.reasoning_tokens if usage else None,
                ),
            )
        return Turn(
            turn_index=index,
            error=ev.error or (resp.get("error") if isinstance(resp, dict) else None) or "no response",
            stop_reason=out_choice.stop_reason if out_choice else None,
        )
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    prompt_ids = resp.get("prompt_token_ids")
    completion_ids = choice.get("token_ids")
    usage = resp.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    raw_tool_calls = msg.get("tool_calls") or []
    tool_calls = [
        ToolCallRecord(id=tc.get("id", ""), function=tc["function"]["name"], arguments=tc["function"].get("arguments", ""))
        for tc in raw_tool_calls
    ]
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False) if ids is not None else None  # noqa: E731
    return Turn(
        turn_index=index,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        rendered_prompt=decode(prompt_ids),
        rendered_completion=decode(completion_ids),
        reasoning=msg.get("reasoning") if msg.get("reasoning") is not None else msg.get("reasoning_content"),
        content=msg.get("content") or "",
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason"),
        stop_reason=ev.output.choices[0].stop_reason if ev.output and ev.output.choices else None,
        usage=TurnUsage(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
        ),
        error=ev.error,
    )


def derive_status(sample: Any, turns: list[Turn]) -> tuple[str, str | None]:
    if sample.error is not None:
        return "error", getattr(sample.error, "message", None) or str(sample.error)
    limit_type = sample.limit.type if sample.limit is not None else None
    if limit_type == "context" or any(t.stop_reason == "model_length" for t in turns):
        return "error", "context_length_exceeded"
    if any(t.error for t in turns):
        return "error", next(t.error for t in turns if t.error)
    if limit_type in ("message", "turn"):
        return "max_turns", None
    if limit_type in ("time", "working"):
        return "timeout", None
    return "completed", None


def convert_sample(log: Any, sample: Any, run_config: dict[str, Any], tokenizer: Any, chat_template_sha: str | None) -> Trajectory:
    cfg = run_config["config"]
    task_args = log.eval.task_args or {}
    variant = task_args.get("split") or sample.metadata.get("impossible_type")
    agent_type = task_args.get("agent_type") or "unknown"
    task_id = str(sample.id)
    run_id = run_config["run_id"]

    messages = [convert_message(m) for m in sample.messages]
    assistant_indices = [i for i, m in enumerate(messages) if m.role == "assistant"]

    turns = [convert_turn(i, ev, tokenizer) for i, ev in enumerate(iter_model_events(sample.events))]
    for turn in turns:
        if turn.turn_index < len(assistant_indices):
            turn.message_index = assistant_indices[turn.turn_index]

    score_name, score = next(iter((sample.scores or {}).items()), (None, None))
    score_record = None
    if score is not None:
        score_record = ScoreRecord(scorer=score_name, value=score.value, answer=score.answer, explanation=score.explanation)

    if variant in IMPOSSIBLE_VARIANTS and score is not None:
        programmatic_hack = score.value == "C"
    else:
        programmatic_hack = None

    test_edit_attempt = None
    if agent_type == "tools":
        flagged_at_scoring = bool(score is not None and (score.metadata or {}).get("test_modification_detected"))
        test_edit_attempt = flagged_at_scoring or bool(test_file_writes([t.model_dump() for t in turns]))

    status, error_message = derive_status(sample, turns)

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    for u in (sample.model_usage or {}).values():
        usage_total["input_tokens"] += u.input_tokens + (u.input_tokens_cache_read or 0) + (u.input_tokens_cache_write or 0)
        usage_total["output_tokens"] += u.output_tokens
        usage_total["total_tokens"] += u.total_tokens
        usage_total["reasoning_tokens"] += u.reasoning_tokens or 0
    if not any(t.usage.reasoning_tokens is not None for t in turns):
        usage_total["reasoning_tokens"] = None

    gen = log.eval.model_generate_config
    sampling = SamplingParams(
        temperature=gen.temperature,
        top_p=gen.top_p,
        top_k=gen.top_k,
        max_tokens=gen.max_tokens,
        seed=gen.seed,
        extra_body=gen.extra_body,
    )

    return Trajectory(
        schema_version=SCHEMA_VERSION,
        trajectory_id=f"{run_id}/{cfg['task_source']}/{variant}/{agent_type}/{task_id}/{sample.epoch}",
        run_id=run_id,
        timestamp=str(sample.started_at or log.eval.created),
        model=cfg["model"],
        model_revision=run_config.get("model_revision"),
        sampling=sampling,
        seed=gen.seed,
        benchmark=cfg["benchmark"],
        task_source=cfg["task_source"],
        variant=variant,
        task_id=task_id,
        agent_type=agent_type,
        epoch=sample.epoch,
        messages=messages,
        reasoning=[t.reasoning for t in turns],
        rendered_text=[RenderedText(prompt=t.rendered_prompt, completion=t.rendered_completion) for t in turns],
        turns=turns,
        final_output=score.answer if score is not None else None,
        score=score_record,
        raw_test_results=score.metadata if score is not None else None,
        programmatic_hack=programmatic_hack,
        test_edit_attempt=test_edit_attempt,
        status=status,
        error_message=error_message,
        token_counts=TokenCounts(**usage_total),
        n_turns=len(turns),
        n_messages=len(messages),
        duration_s=sample.total_time,
        prompt_config=PromptConfig(
            instruction_prompt=task_args.get("instruction_prompt", cfg.get("instruction_prompt")),
            allow_test_modifications=task_args.get("allow_test_modifications", cfg.get("allow_test_modifications")),
            max_attempts=task_args.get("max_attempts", cfg.get("max_attempts")),
            message_limit=task_args.get("message_limit", cfg.get("message_limit")),
        ),
        provenance=Provenance(
            inspect_log=log.location,
            sample_uuid=sample.uuid,
            inspect_ai_version=run_config["versions"].get("inspect_ai"),
            impossiblebench_version=run_config["versions"].get("impossiblebench"),
            vllm_version=run_config["versions"].get("vllm"),
            harness_git_sha=run_config["versions"].get("harness_git_sha"),
            chat_template_sha256=chat_template_sha,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="data/stage1/<run_id>")
    parser.add_argument("--output", type=Path, default=None, help="default: <run-dir>/trajectories.jsonl")
    args = parser.parse_args()

    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    model = run_config["config"]["model"]
    revision = run_config.get("model_revision")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    template = tokenizer.chat_template
    chat_template_sha = hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest() if template else None

    from inspect_ai.log import list_eval_logs, read_eval_log

    output = args.output or (args.run_dir / "trajectories.jsonl")
    records: list[Trajectory] = []
    for info in list_eval_logs(str(args.run_dir / "logs")):
        log = read_eval_log(info, resolve_attachments=True)
        if not log.samples:
            print(f"skip (no samples): {log.location} status={log.status}")
            continue
        for sample in log.samples:
            records.append(convert_sample(log, sample, run_config, tokenizer, chat_template_sha))
        print(f"{log.location}: {len(log.samples)} samples")

    records.sort(key=lambda r: r.trajectory_id)
    with output.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    print(f"wrote {len(records)} trajectories to {output}")


if __name__ == "__main__":
    main()
    