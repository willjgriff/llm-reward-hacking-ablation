"""Adapter for rollout records written by Ethan's `ortho` repo (`envs/convert.py` there).

One ortho record is one Inspect sample: `rollout_id` = "<log stem>/<sample id>/<epoch>", `env`
(grader / secret_number / impossible_bench), `config_id`, `condition`, `vectors`, `model`, `cfg`
(the run's EvalCfg), `items` (chat-completions messages, assistant ones with `reasoning`),
`turns[]` (one per assistant message with the env outputs that followed it and per-env cheat
flags), `labels`, `cheat_kinds`, `cheated`, `first_cheat_turn`, `n_turns` and `ids` (the last
request's prompt token ids followed by its completion ids, or null).

`to_trajectory` maps such a record onto the shape of our trajectory schema (`rhablation.schema`),
enough for the viewer and the transcript renderer: messages, per-call reasoning, labels as
`{value, source}`. Hack signals are never merged: `cheated` and `cheat_kinds` become labels and
`programmatic_hack` / `test_edit_attempt` stay null. Pure functions; nothing here reads files.
"""

import json
import re
from typing import Any

ORTHO_KEYS = ("rollout_id", "items")
_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def is_ortho_record(rec: dict[str, Any]) -> bool:
    return all(k in rec for k in ORTHO_KEYS)


def normalize(rec: dict[str, Any]) -> dict[str, Any]:
    """Our schema's shape for either kind of record."""
    return to_trajectory(rec) if is_ortho_record(rec) else rec


def _split_rollout_id(rollout_id: str) -> tuple[str, str, int | None]:
    """(log stem, sample id, epoch). The sample id may itself contain no '/'; the epoch is the last part."""
    parts = rollout_id.split("/")
    if len(parts) < 3:
        return rollout_id, rollout_id, None
    stem, epoch = parts[0], parts[-1]
    sample = "/".join(parts[1:-1])
    try:
        return stem, sample, int(epoch)
    except ValueError:
        return stem, sample, None


def _timestamp(stem: str) -> str | None:
    m = _DATE_PREFIX.match(stem)
    return f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}" if m else None


def _arguments(raw: Any) -> dict[str, Any] | str:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return raw
        return parsed if isinstance(parsed, dict) else raw
    return raw if isinstance(raw, dict) else {}


def _messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Our Message shape from chat-completions items; tool results get the name of the call they answer."""
    out: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    for it in items:
        role = it.get("role")
        if role == "assistant":
            calls = []
            for c in it.get("tool_calls") or []:
                fn = c.get("function") or {}
                name = fn.get("name") or c.get("name")
                cid = c.get("id") or f"call_{len(call_names)}"
                call_names[cid] = name
                calls.append({"id": cid, "function": name, "arguments": _arguments(fn.get("arguments", c.get("args")))})
            out.append({"role": "assistant", "content": it.get("content") or "", "tool_calls": calls or None})
        elif role == "tool":
            cid = it.get("tool_call_id")
            out.append({"role": "tool", "content": it.get("content") or "", "tool_call_id": cid, "function": call_names.get(cid)})
        else:
            out.append({"role": role, "content": it.get("content") or ""})
    return out


def _variant(rec: dict[str, Any], labels: dict[str, Any]) -> str | None:
    env = rec.get("env")
    if env == "grader" and all(k in labels for k in ("family", "side", "prompts", "sys")):
        return f"{labels['family']}.{labels['side']}.{labels['prompts']}.{labels['sys']}"
    if env == "impossible_bench" and "split" in labels:
        return labels["split"]
    return rec.get("config_id")


_SECRET_NUMBER_EXTRAS = ("won", "completed", "secret_number", "submitted_answer", "retries", "noise_prob", "max_guesses", "max_number", "answer_explanation")


def to_trajectory(rec: dict[str, Any]) -> dict[str, Any]:
    """An ortho record in the shape of our trajectory schema (viewer/renderer subset)."""
    stem, sample_id, epoch = _split_rollout_id(rec.get("rollout_id") or "")
    env = rec.get("env")
    source = f"ortho/{env}"
    raw_labels = rec.get("labels") or {}
    labels = {k: {"value": v, "source": source} for k, v in raw_labels.items()}
    labels["cheated"] = {"value": bool(rec.get("cheated")), "source": source}
    labels["cheat_kinds"] = {"value": ",".join(rec.get("cheat_kinds") or []), "source": source}

    items = rec.get("items") or []
    messages = _messages(items)
    ortho_turns = rec.get("turns") or []
    assistant_items = [it for it in items if it.get("role") == "assistant"]
    turns = []
    for i, it in enumerate(assistant_items):
        ot = ortho_turns[i] if i < len(ortho_turns) else {}
        turns.append(
            {
                "turn_index": i,
                "reasoning": it.get("reasoning") if it.get("reasoning") is not None else ot.get("reasoning"),
                "finish_reason": None,
                "stop_reason": None,
                "usage": None,
                "tool_calls": messages_tool_calls(messages, i),
                "cheated": ot.get("cheated"),
            }
        )

    env_options: dict[str, Any] = {k: rec.get(k) for k in ("env", "config_id", "condition", "vectors") if k in rec}
    env_options.update({k: rec[k] for k in _SECRET_NUMBER_EXTRAS if k in rec})
    if rec.get("first_cheat_turn") is not None:
        env_options["first_cheat_turn"] = rec["first_cheat_turn"]
    ids = rec.get("ids")

    return {
        "schema_version": "ortho",
        "trajectory_id": rec.get("rollout_id"),
        "run_id": stem,
        "timestamp": _timestamp(stem),
        "model": rec.get("model"),
        "model_revision": None,
        "benchmark": "ortho",
        "task_source": rec.get("config_id"),
        "variant": _variant(rec, raw_labels),
        "task_id": sample_id,
        "agent_type": env,
        "epoch": epoch,
        "status": "completed",
        "error_message": None,
        "sampling": rec.get("cfg"),
        "seed": None,
        "env_options": env_options,
        "prompt_config": None,
        "messages": messages,
        "turns": turns,
        "labels": labels,
        "programmatic_hack": None,
        "test_edit_attempt": None,
        "score": None,
        "raw_test_results": None,
        "final_output": None,
        "token_counts": {"ids": len(ids)} if ids is not None else None,
        "n_turns": len(turns),
        "n_messages": len(messages),
        "duration_s": None,
        "provenance": {"format": "ortho/convert.py", "rollout_id": rec.get("rollout_id")},
    }


_SYSTEM_TURN = re.compile(r"^<\|im_start\|>system\n(.*?)<\|im_end\|>\n", re.S)


def rendered_system_turn(rec: dict[str, Any], decode) -> str | None:
    """The system turn the chat template rendered at the start of the prompt, decoded from the record's
    stored `ids`, or None when the record has no ids or the prompt starts without one. `decode` maps a list
    of token ids to text (the tokenizer of `rec["model"]`). Qwen3.8's template injects a reasoning-effort
    system turn even when no system message was sent, so this is the only place it is visible."""
    ids = rec.get("ids")
    if not ids:
        return None
    m = _SYSTEM_TURN.match(decode(ids[:256]))
    return m.group(1) if m else None


def with_rendered_system_turn(traj: dict[str, Any], rec: dict[str, Any], decode) -> dict[str, Any]:
    """`traj` (from to_trajectory) with the system turn as the chat template rendered it: inserted as the
    first message when none was sent, or replacing the sent one when the template changed it (Qwen3.8
    prepends its reasoning-effort instruction to a supplied system prompt). Marked `rendered: True` so the
    UI can say where it came from; unchanged when the ids are missing or the rendering matches."""
    text = rendered_system_turn(rec, decode)
    if text is None:
        return traj
    messages = list(traj.get("messages") or [])
    if messages and messages[0].get("role") == "system":
        if (messages[0].get("content") or "").strip() == text.strip():
            return traj
        messages[0] = {**messages[0], "content": text, "rendered": True}
    else:
        messages.insert(0, {"role": "system", "content": text, "rendered": True})
    out = dict(traj)
    out["messages"] = messages
    out["n_messages"] = len(messages)
    return out


def messages_tool_calls(messages: list[dict[str, Any]], assistant_index: int) -> list[dict[str, Any]]:
    """The tool calls of the assistant_index-th assistant message (for hack_signals, which reads turns)."""
    n = -1
    for m in messages:
        if m.get("role") == "assistant":
            n += 1
            if n == assistant_index:
                return m.get("tool_calls") or []
    return []
