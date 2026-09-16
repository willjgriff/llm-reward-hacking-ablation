#!/usr/bin/env python
"""Stage 1: run ImpossibleBench (LiveCodeBench) against a vLLM-served model.

Writes data/stage1/<run_id>/run_config.json and Inspect .eval logs under
data/stage1/<run_id>/logs/. Export to JSONL with stage1_export_trajectories.py.
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.config import Stage1Config, add_config_args, load_config  # noqa: E402

CONTEXT_MARGIN_TOKENS = 8192


def resolve_model_revision(model: str, revision: str | None) -> str:
    from huggingface_hub import HfApi

    try:
        return HfApi().model_info(model, revision=revision).sha
    except Exception as e:  # offline: fall back to the local snapshot ref
        from huggingface_hub.constants import HF_HUB_CACHE

        ref = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}" / "refs" / (revision or "main")
        if ref.exists():
            return ref.read_text().strip()
        raise RuntimeError(f"Could not resolve revision for {model}: {e}") from e


def preflight_vllm(cfg: Stage1Config) -> str | None:
    headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'local')}"}
    models = httpx.get(f"{cfg.vllm_base_url.rstrip('/')}/models", headers=headers, timeout=30).json()
    entry = next((m for m in models.get("data", []) if m.get("id") == cfg.model), None)
    if entry is None:
        raise RuntimeError(f"{cfg.model} not served at {cfg.vllm_base_url}; served: {[m.get('id') for m in models.get('data', [])]}")
    max_model_len = entry.get("max_model_len")
    required = cfg.max_attempts * cfg.sampling.max_tokens + CONTEXT_MARGIN_TOKENS
    if max_model_len is None or max_model_len < required:
        raise RuntimeError(
            f"vLLM max_model_len={max_model_len} < required {required} "
            f"(max_attempts={cfg.max_attempts} * max_tokens={cfg.sampling.max_tokens} + {CONTEXT_MARGIN_TOKENS})"
        )
    root = cfg.vllm_base_url.rstrip("/").removesuffix("/v1")
    try:
        return httpx.get(f"{root}/version", headers=headers, timeout=10).json().get("version")
    except Exception:
        return None


def pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def build_task(cfg: Stage1Config, split: str, agent_type: str):
    from impossiblebench import impossible_livecodebench

    return impossible_livecodebench(
        split=split,
        agent_type=agent_type,
        sandbox=cfg.sandbox,
        limit=None if cfg.task_ids else cfg.limit,
        shuffle=cfg.shuffle,
        instruction_prompt=cfg.instruction_prompt,
        allow_test_modifications=cfg.allow_test_modifications,
        max_attempts=cfg.max_attempts,
        message_limit=cfg.message_limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--display", default=None, help="Inspect display type (e.g. plain, rich)")
    args = parser.parse_args()
    cfg = load_config(args)

    os.environ.setdefault("VLLM_API_KEY", "local")

    run_id = cfg.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    run_dir = cfg.output_dir / run_id
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    model_revision = resolve_model_revision(cfg.model, cfg.model_revision)
    vllm_version = preflight_vllm(cfg)

    tasks = {}
    selected_ids: dict[str, dict[str, list[str]]] = {}
    for agent_type in cfg.agent_types:
        for split in cfg.splits:
            task = build_task(cfg, split, agent_type)
            ids = [str(s.id) for s in task.dataset]
            if cfg.task_ids:
                missing = set(cfg.task_ids) - set(ids)
                if missing:
                    raise RuntimeError(f"task_ids not found in {split}: {sorted(missing)}")
                ids = [i for i in ids if i in cfg.task_ids]
            tasks[(agent_type, split)] = task
            selected_ids.setdefault(agent_type, {})[split] = ids
        per_split = list(selected_ids[agent_type].values())
        if any(s != per_split[0] for s in per_split):
            raise RuntimeError(f"Selected task ids differ across splits for {agent_type}: {selected_ids[agent_type]}")
        print(f"[{agent_type}] {len(per_split[0])} tasks per split: {per_split[0]}")

    run_config = {
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "config": json.loads(cfg.model_dump_json()),
        "model_revision": model_revision,
        "versions": {
            "vllm": vllm_version,
            "inspect_ai": pkg_version("inspect_ai"),
            "impossiblebench": pkg_version("impossiblebench"),
            "harness_git_sha": git_sha(),
        },
        "selected_task_ids": selected_ids,
        "eval_logs": [],
    }
    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2))
    print(f"run_id={run_id} model_revision={model_revision} vllm={vllm_version}")

    from inspect_ai import eval as inspect_eval

    for (agent_type, split), task in tasks.items():
        print(f"\n=== {split} / {agent_type} ===")
        logs = inspect_eval(
            task,
            model=f"vllm/{cfg.model}",
            model_base_url=cfg.vllm_base_url,
            sample_id=cfg.task_ids,
            epochs=cfg.samples_per_task,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            top_k=cfg.sampling.top_k,
            seed=cfg.sampling.seed,
            max_tokens=cfg.sampling.max_tokens,
            # Inspect records top_k in the log but does not send it to OpenAI-compatible servers.
            extra_body={"return_token_ids": True, "top_k": cfg.sampling.top_k},
            max_connections=cfg.max_connections,
            max_samples=cfg.max_connections,
            max_sandboxes=cfg.max_connections,
            max_subprocesses=cfg.max_connections,
            log_dir=str(log_dir),
            log_samples=True,
            log_images=False,
            fail_on_error=False,
            display=args.display,
            tags=[f"run:{run_id}", f"split:{split}", f"agent:{agent_type}"],
            metadata={"run_id": run_id, "split": split, "agent_type": agent_type, "model_revision": model_revision},
        )
        for log in logs:
            metrics = {}
            if log.results:
                for score in log.results.scores:
                    metrics[score.name] = {k: v.value for k, v in score.metrics.items()}
            print(f"status={log.status} log={log.location}\nmetrics={json.dumps(metrics)}")
            run_config["eval_logs"].append(
                {"split": split, "agent_type": agent_type, "status": log.status, "location": log.location, "metrics": metrics}
            )
            run_config_path.write_text(json.dumps(run_config, indent=2))

    print(f"\nDone. Export with: uv run scripts/stage1_export_trajectories.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
