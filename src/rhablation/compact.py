"""Compact storage of trajectories (schema 1.2) and transparent reading of trajectories.jsonl[.gz].

Every model call's prompt repeats the previous call's prompt almost entirely, and the rendered strings
are `tokenizer.decode` of the stored token ids, so a full record is ~90% repetition. The compact form
stores each prompt as (length of the prefix shared with the previous call's prompt, remaining ids) and
drops `rendered_prompt` / `rendered_text`. `expand_trajectory` restores the full record exactly; call it
before using `prompt_token_ids`. It is a no-op on full records, so readers can call it unconditionally.
"""

import gzip
from pathlib import Path
from typing import Any, TextIO

from rhablation.schema import RenderedText, Trajectory


def compact_trajectory(record: Trajectory) -> Trajectory:
    """Copy of `record` with prefix-delta prompt ids and without the derived prompt strings."""
    out = record.model_copy(deep=True)
    prev: list[int] | None = None
    for turn in out.turns:
        if turn.prompt_token_ids is None:
            continue
        if turn.prompt_prefix_len is not None:
            raise ValueError(f"{record.trajectory_id} turn {turn.turn_index}: already compact")
        full = turn.prompt_token_ids
        k = 0
        if prev is not None:
            limit = min(len(prev), len(full))
            while k < limit and prev[k] == full[k]:
                k += 1
        turn.prompt_prefix_len = k
        turn.prompt_token_ids = full[k:]
        turn.rendered_prompt = None
        prev = full
    out.rendered_text = []
    return out


def expand_trajectory(record: Trajectory, tokenizer: Any = None) -> Trajectory:
    """Copy of `record` with full `prompt_token_ids` in every turn (`prompt_prefix_len` back to None).

    With a tokenizer, `rendered_prompt` and `rendered_text` are rebuilt where missing."""
    out = record.model_copy(deep=True)
    prev: list[int] | None = None
    for turn in out.turns:
        if turn.prompt_token_ids is None:
            continue
        k = turn.prompt_prefix_len
        if k is not None:
            if k > len(prev or []):
                raise ValueError(
                    f"{record.trajectory_id} turn {turn.turn_index}: prompt_prefix_len={k} exceeds the previous prompt ({len(prev or [])} tokens)"
                )
            turn.prompt_token_ids = (prev or [])[:k] + turn.prompt_token_ids
            turn.prompt_prefix_len = None
        prev = turn.prompt_token_ids
        if tokenizer is not None and turn.rendered_prompt is None:
            turn.rendered_prompt = tokenizer.decode(turn.prompt_token_ids, skip_special_tokens=False)
    if tokenizer is not None and not out.rendered_text:
        out.rendered_text = [RenderedText(prompt=t.rendered_prompt, completion=t.rendered_completion) for t in out.turns]
    return out


def is_compact(record: Trajectory) -> bool:
    return any(t.prompt_prefix_len is not None for t in record.turns)


def trajectories_path(run_dir: Path) -> Path:
    """<run_dir>/trajectories.jsonl, or the .gz next to it when only that exists."""
    plain = run_dir / "trajectories.jsonl"
    if not plain.exists() and plain.with_name(plain.name + ".gz").exists():
        return plain.with_name(plain.name + ".gz")
    return plain


def open_trajectories(path: Path, mode: str = "r") -> TextIO:
    """Open a trajectories file for text reading or writing; gzip when the name ends in .gz."""
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")
