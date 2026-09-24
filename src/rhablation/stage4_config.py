"""Stage 4 configuration: activation cache and reward-hacking directions.

One YAML file (configs/stage4_*.yaml) drives the three stage-4 scripts: manifest → activation cache →
directions. Everything the box needs to reproduce a run is in this file plus the manifest it writes.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Aggregations the cache builder can compute from one forward pass over a trajectory (see stage4_masks).
CACHED_AGGREGATIONS = ("cot_mean", "answer_mean", "prompt_last", "completion_last")
# Derived at analysis time from cached aggregations and the manifest's token counts.
DERIVED_AGGREGATIONS = ("completion_mean",)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassRule(StrictModel):
    """First matching rule (in file order) gives the class. A field left null is not tested."""

    name: str
    variant: str
    max_turns: int | None = None
    programmatic_hack: bool | None = None
    passed: bool | None = None
    tampered: bool | None = None


class DataConfig(StrictModel):
    sweep_dirs: list[Path] = Field(description="Directories whose subdirectories are run dirs with trajectories.jsonl[.gz]")
    exclude_run_ids: list[str] = Field(default_factory=list)


class SplitConfig(StrictModel):
    eval_fraction: float = 0.2
    seed: int = 0


class OrderConfig(StrictModel):
    priority_classes: list[str] = Field(description="Round-robin over these first; remaining classes follow")
    subset_per_class: int = 400
    seed: int = 0


class CacheConfig(StrictModel):
    storage_dtype: Literal["float16", "float32"] = "float16"
    aggregations: list[str] = Field(default_factory=lambda: list(CACHED_AGGREGATIONS))
    chunk_trajectories: int = 256
    max_tokens_per_batch: int = 32768
    max_batch_size: int = 16
    max_seq_len: int = 16384
    # Fraction of trajectories on which the teacher-forced completion NLL is computed (1.0 under --limit).
    nll_sample_rate: float = 0.05
    attn_implementation: str | None = None
    # Abort when any stored |activation| exceeds this (float16 max is 65504).
    max_abs_activation: float = 6.0e4

    @model_validator(mode="after")
    def _known_aggregations(self) -> "CacheConfig":
        unknown = set(self.aggregations) - set(CACHED_AGGREGATIONS)
        if unknown:
            raise ValueError(f"unknown cache aggregations {sorted(unknown)}; cached ones are {CACHED_AGGREGATIONS}")
        return self


class ContrastSpec(StrictModel):
    positive: list[str] = Field(description="Class names of the positive (hack) side")
    negative: list[str]
    positive_max_turns: int | None = None
    negative_max_turns: int | None = None
    description: str = ""


class AnalysisConfig(StrictModel):
    contrasts: dict[str, ContrastSpec]
    aggregations: list[str] = Field(default_factory=lambda: list(CACHED_AGGREGATIONS) + list(DERIVED_AGGREGATIONS))
    probe_C: float = 1.0
    # Logistic-regression probes are fitted on these layer indices (0 = embeddings) plus the best
    # projection layer; the projection AUROC is computed on every layer.
    probe_layer_stride: int = 8
    probe_max_iter: int = 500
    sample_sizes: list[int] = Field(default_factory=lambda: [50, 100, 200, 400])
    sample_reps: int = 10
    bootstrap_reps: int = 20
    winsorization_quantile: float | None = None
    covariates: list[str] = Field(default_factory=lambda: ["n_calls", "seq_len", "n_cot_tokens", "max_turns"])
    # Named pairs of contrasts whose directions are compared per layer (same aggregation).
    cosine_pairs: list[list[str]] = Field(default_factory=list)
    # Pairs (direction from A, scored on B's eval rows).
    transfer_pairs: list[list[str]] = Field(default_factory=list)
    seed: int = 0

    @model_validator(mode="after")
    def _known(self) -> "AnalysisConfig":
        allowed = set(CACHED_AGGREGATIONS) | set(DERIVED_AGGREGATIONS)
        unknown = set(self.aggregations) - allowed
        if unknown:
            raise ValueError(f"unknown analysis aggregations {sorted(unknown)}")
        for pair in self.cosine_pairs + self.transfer_pairs:
            if len(pair) != 2 or any(p not in self.contrasts for p in pair):
                raise ValueError(f"pair {pair} must name two contrasts from {sorted(self.contrasts)}")
        return self


class Stage4Config(StrictModel):
    manifest_id: str = "tv27b"
    model: str
    model_revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # Token ids of the chat template pieces the masks rely on (Qwen3.5/3.8 tokenizer).
    think_open_id: int = 248068
    think_close_id: int = 248069
    im_end_id: int = 248046
    newline_id: int = 198

    data: DataConfig
    classes: list[ClassRule]
    split: SplitConfig = Field(default_factory=SplitConfig)
    order: OrderConfig
    cache: CacheConfig = Field(default_factory=CacheConfig)
    analysis: AnalysisConfig

    @model_validator(mode="after")
    def _consistent(self) -> "Stage4Config":
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate class names")
        for c in self.order.priority_classes:
            if c not in names:
                raise ValueError(f"priority class {c!r} is not a defined class")
        for cname, spec in self.analysis.contrasts.items():
            for c in spec.positive + spec.negative:
                if c not in names:
                    raise ValueError(f"contrast {cname}: class {c!r} is not defined")
        return self


def load_stage4_config(path: Path, overrides: dict[str, Any] | None = None) -> Stage4Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        if value is not None:
            raw[key] = value
    return Stage4Config.model_validate(raw)
