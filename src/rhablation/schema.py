"""Trajectory record schema. One JSON object per line in trajectories.jsonl.

This is the contract between stage 1 and every later stage. Change SCHEMA_VERSION
whenever a field is added, removed, or changes meaning.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.1"  # 1.1: added test_edit_attempt

Variant = Literal["original", "oneoff", "conflicting"]
Status = Literal["completed", "error", "timeout", "max_turns"]
Role = Literal["system", "user", "assistant", "tool"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCallRecord(StrictModel):
    id: str
    function: str
    arguments: dict[str, Any] | str


class Message(StrictModel):
    """One message in the conversation the benchmark scaffold built, in order.

    Assistant `content` is the visible text only; chain of thought lives in
    Trajectory.reasoning / Trajectory.turns.
    """

    role: Role
    content: str
    tool_calls: list[ToolCallRecord] | None = None
    tool_call_id: str | None = None
    function: str | None = None
    error: str | None = None


class TurnUsage(StrictModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


class Turn(StrictModel):
    """One model call. `rendered_prompt` is exactly what the model saw at that
    call (chat template applied by vLLM, decoded from prompt_token_ids without
    skipping special tokens); `rendered_completion` is exactly what it produced,
    including <think> tags."""

    turn_index: int
    message_index: int | None = Field(
        default=None, description="Index into Trajectory.messages of the assistant message this call produced"
    )
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    rendered_prompt: str | None = None
    rendered_completion: str | None = None
    reasoning: str | None = None
    content: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    finish_reason: str | None = None
    stop_reason: str | None = Field(default=None, description="Inspect's normalized stop reason (e.g. model_length)")
    usage: TurnUsage = Field(default_factory=TurnUsage)
    error: str | None = None


class RenderedText(StrictModel):
    prompt: str | None = None
    completion: str | None = None


class SamplingParams(StrictModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    extra_body: dict[str, Any] | None = None


class ScoreRecord(StrictModel):
    scorer: str
    value: str | int | float | bool | None
    answer: str | None = None
    explanation: str | None = None


class TokenCounts(StrictModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None


class PromptConfig(StrictModel):
    instruction_prompt: str | None = None
    allow_test_modifications: bool | None = None
    max_attempts: int | None = None
    message_limit: int | None = None


class Provenance(StrictModel):
    inspect_log: str
    sample_uuid: str | None = None
    inspect_ai_version: str | None = None
    impossiblebench_version: str | None = None
    vllm_version: str | None = None
    harness_git_sha: str | None = None
    chat_template_sha256: str | None = None


class Trajectory(StrictModel):
    schema_version: str = SCHEMA_VERSION
    trajectory_id: str
    run_id: str
    timestamp: str

    model: str
    model_revision: str | None
    sampling: SamplingParams
    seed: int | None

    benchmark: str
    task_source: str
    variant: Variant
    task_id: str
    agent_type: str
    epoch: int

    messages: list[Message]
    reasoning: list[str | None] = Field(description="Chain of thought per assistant turn, same order as turns")
    rendered_text: list[RenderedText] = Field(description="Per-turn exact prompt/completion strings")
    turns: list[Turn]

    final_output: str | None
    score: ScoreRecord | None
    raw_test_results: dict[str, Any] | None
    programmatic_hack: bool | None = Field(
        description="True/False on impossible variants (pass == hack); null on original"
    )
    test_edit_attempt: bool | None = Field(
        default=None,
        description=(
            "tools scaffold only: True if any tool call wrote to test.py or the scorer flagged a "
            "modified test.py at final scoring (rhablation.hack_signals); null on the minimal scaffold, "
            "which has no file access. Independent of programmatic_hack: the scorer restores test.py "
            "before running tests, so an edited test never passes."
        ),
    )

    status: Status
    error_message: str | None

    token_counts: TokenCounts
    n_turns: int
    n_messages: int
    duration_s: float | None

    prompt_config: PromptConfig
    provenance: Provenance
