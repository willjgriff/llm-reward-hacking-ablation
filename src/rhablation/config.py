import argparse
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = 20
    max_tokens: int = 8192  # output tokens per model call: thinking + answer
    # vLLM forces </think> after this many thinking tokens, so the call still returns an answer.
    # null = unlimited (then a long chain of thought can use up max_tokens and return nothing).
    thinking_token_budget: int | None = 4096
    seed: int | None = 1234


class Stage1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    output_dir: Path = Path("data/stage1")

    model: str
    model_revision: str | None = None
    vllm_base_url: str = "http://localhost:8000/v1"

    benchmark: Literal["impossiblebench"] = "impossiblebench"
    task_source: Literal["livecodebench"] = "livecodebench"
    splits: list[Literal["original", "oneoff", "conflicting"]]
    agent_types: list[Literal["minimal", "tools"]]
    limit: int | None = None
    offset: int = 0
    task_ids: list[str] | None = None
    shuffle: bool = True
    samples_per_task: int = 1

    sandbox: Literal["docker", "local"] = "docker"

    instruction_prompt: str
    allow_test_modifications: bool = True
    max_attempts: int = 5
    message_limit: int = 50

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    max_connections: int = 8
    # Per-request HTTP timeout. The OpenAI SDK default (600 s, 2 retries) abandons any turn that
    # generates for more than 10 minutes and resamples it, which discards long chains of thought.
    client_timeout_s: int = 7200


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_lcb.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--vllm-base-url")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--agent-types", nargs="+")
    parser.add_argument("--limit", type=int, help="number of tasks; 0 = all remaining after --offset")
    parser.add_argument("--offset", type=int, help="skip this many tasks of the shuffled order first")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--samples-per-task", type=int)
    parser.add_argument("--max-attempts", type=int, help="submission attempts per task (package default 5; its example script uses 10)")
    parser.add_argument("--sandbox")
    parser.add_argument("--max-connections", type=int)
    parser.add_argument("--client-timeout-s", type=int, help="HTTP timeout per model call, in seconds")


def load_config(args: argparse.Namespace) -> Stage1Config:
    raw: dict[str, Any] = yaml.safe_load(args.config.read_text())
    for key in (
        "run_id", "output_dir", "model", "model_revision", "vllm_base_url", "splits",
        "agent_types", "limit", "offset", "task_ids", "samples_per_task", "max_attempts", "sandbox",
        "max_connections", "client_timeout_s",
    ):
        value = getattr(args, key, None)
        if value is not None:
            raw[key] = value
    if raw.get("limit") == 0:
        raw["limit"] = None
    return Stage1Config.model_validate(raw)
