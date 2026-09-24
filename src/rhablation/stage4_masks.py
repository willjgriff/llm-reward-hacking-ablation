"""Token positions of interest inside one trajectory's final sequence (pure Python, no torch).

In the TerminalVerifierEnv transcripts every model call's prompt is the previous call's prompt plus the
previous completion (chain of thought included) plus the tool output, so the final call's
`prompt + completion` contains the whole rollout exactly as the model saw it (verified on all 11,257
sweep trajectories, 2026-09-23). One causal forward over that sequence therefore gives every token's
residual for every turn. `stage4_manifest.final_sequence` asserts the property before anything relies on it.

Layout of every call: prompt ends `<|im_start|>assistant\\n<think>\\n`; the completion runs inside the open
think block until `</think>`, then the visible answer, then `<|im_end|>`.

Aggregations (weights over sequence positions, each summing to 1 over the trajectory):
- cot_mean: token-weighted mean over all chain-of-thought tokens of all turns (between the prompt end and
  `</think>`, exclusive)
- answer_mean: token-weighted mean over all visible answer tokens of all turns (after `</think>`, excluding
  the final `<|im_end|>`)
- prompt_last: mean over turns of the residual at the last prompt token of each call (heretic's position:
  the state just before the model starts generating)
- completion_last: mean over turns of the residual at the last generated token of each call before
  `<|im_end|>`
- completion_mean (derived at analysis time from cot_mean and answer_mean with the token counts)
"""

from dataclasses import dataclass, field

AGGREGATIONS = ("cot_mean", "answer_mean", "prompt_last", "completion_last")


class LayoutError(ValueError):
    pass


@dataclass
class TurnSpan:
    """Position ranges of one model call inside the final sequence."""

    prompt_len: int  # prompt occupies [0, prompt_len)
    completion_len: int  # completion occupies [prompt_len, prompt_len + completion_len)
    # Absolute position of the first </think>; None if the completion has none (cut while thinking, or the
    # model closed the block without the tag). A second </think> later in the completion is kept in the answer.
    think_close_pos: int | None = None
    n_think_close: int = 0
    ends_with_im_end: bool = False

    @property
    def start(self) -> int:
        return self.prompt_len

    @property
    def end(self) -> int:
        return self.prompt_len + self.completion_len


@dataclass
class Spans:
    seq_len: int
    turns: list[TurnSpan] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.turns)

    def cot_positions(self) -> list[int]:
        out: list[int] = []
        for t in self.turns:
            stop = t.think_close_pos if t.think_close_pos is not None else t.end
            out.extend(range(t.start, stop))
        return out

    def answer_positions(self) -> list[int]:
        out: list[int] = []
        for t in self.turns:
            if t.think_close_pos is None:
                continue
            stop = t.end - 1 if t.ends_with_im_end else t.end
            out.extend(range(t.think_close_pos + 1, stop))
        return out

    def prompt_last_positions(self) -> list[int]:
        return [t.prompt_len - 1 for t in self.turns]

    def completion_last_positions(self) -> list[int]:
        out = []
        for t in self.turns:
            last = t.end - 2 if t.ends_with_im_end else t.end - 1
            if last >= t.start:
                out.append(last)
        return out

    def completion_positions(self) -> list[int]:
        out: list[int] = []
        for t in self.turns:
            stop = t.end - 1 if t.ends_with_im_end else t.end
            out.extend(range(t.start, stop))
        return out

    def n_cot_tokens(self) -> int:
        return len(self.cot_positions())

    def n_answer_tokens(self) -> int:
        return len(self.answer_positions())

    def n_calls_without_think_close(self) -> int:
        return sum(1 for t in self.turns if t.think_close_pos is None)

    def n_calls_multi_think_close(self) -> int:
        return sum(1 for t in self.turns if t.n_think_close > 1)


def token_spans(
    ids: list[int],
    turn_lengths: list[tuple[int, int]],
    think_open_id: int,
    think_close_id: int,
    im_end_id: int,
    newline_id: int,
    strict: bool = True,
) -> Spans:
    """Build spans from the final sequence `ids` and per-call (prompt_len, completion_len).

    Raises LayoutError when the layout assumptions fail (strict) so that a change in the renderer or
    tokenizer cannot silently shift every mask."""
    spans = Spans(seq_len=len(ids))
    for i, (plen, clen) in enumerate(turn_lengths):
        if plen <= 0 or clen < 0 or plen + clen > len(ids):
            raise LayoutError(f"call {i}: prompt_len={plen} completion_len={clen} do not fit in {len(ids)} tokens")
        if strict and ids[plen - 2 : plen] != [think_open_id, newline_id]:
            raise LayoutError(f"call {i}: prompt does not end with <think>\\n (last ids {ids[plen - 2 : plen]})")
        completion = ids[plen : plen + clen]
        closes = [plen + j for j, tok in enumerate(completion) if tok == think_close_id]
        # Rare anomalies in the sweep data (15 of 10,993 trajectories): two </think> in one completion, or a
        # completion that ends with <|im_end|> without any </think>. They are kept and counted, not excluded.
        spans.turns.append(
            TurnSpan(
                prompt_len=plen,
                completion_len=clen,
                think_close_pos=closes[0] if closes else None,
                n_think_close=len(closes),
                ends_with_im_end=bool(completion) and completion[-1] == im_end_id,
            )
        )
    return spans


def aggregation_weights(spans: Spans, aggregations: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Sparse weights per aggregation: list of (position, weight), weights summing to 1 (empty if no tokens)."""
    out: dict[str, list[tuple[int, float]]] = {}
    for agg in aggregations:
        if agg == "cot_mean":
            pos = spans.cot_positions()
        elif agg == "answer_mean":
            pos = spans.answer_positions()
        elif agg == "prompt_last":
            pos = spans.prompt_last_positions()
        elif agg == "completion_last":
            pos = spans.completion_last_positions()
        else:
            raise ValueError(f"unknown aggregation {agg!r}")
        w = 1.0 / len(pos) if pos else 0.0
        out[agg] = [(p, w) for p in pos]
    return out
