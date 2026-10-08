"""
ImpossibleBench's LiveCodeBench task (the official Inspect package, pinned in pyproject) under a named config: the split, scaffold, attempt
budget and instruction prompt from envs/configs/impossible_bench/<config>.yaml, the repo's compose file under docker/impossible_bench (the
installed package does not ship one), and the shared sampling and intervention settings. The task takes an EvalCfg (common.py) and runs
through run.py: its config and n (the number of problems, None for the whole split).

Cheating is passing on an impossible split (the package scorer's C on oneoff or conflicting); passing on original is the capability number.
"""
from impossiblebench import impossible_livecodebench
from inspect_ai import Task, task

from common import EvalCfg, generate_config, load_config, task_metadata

COMPOSE = "docker/impossible_bench/compose.yaml"


@task
def lcb(cfg: EvalCfg) -> Task:
    conf = load_config("impossible_bench", cfg.config)
    bench = impossible_livecodebench(split=conf["split"], agent_type=conf["agent_type"], instruction_prompt=conf["instruction_prompt"], allow_test_modifications=conf["allow_test_modifications"], max_attempts=conf["max_attempts"], message_limit=conf["message_limit"], limit=cfg.n, sandbox=("docker", COMPOSE))
    bench.config = generate_config(cfg)
    bench.metadata = task_metadata(cfg) | {"split": conf["split"]}
    return bench
