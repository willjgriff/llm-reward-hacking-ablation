"""
Shared pieces for the Inspect environments in this directory: the run config (EvalCfg), the config files under envs/configs, the request
settings an EvalCfg's intervention maps to, the server hooks, and the vector files. Inspect puts a task file's directory on sys.path, so the
tasks import this as `common`, and run.py at the repo root puts envs/ on sys.path for the same reason.

An EvalCfg is one run: an env under a named yaml config, the run's size, and the intervention, which is any of a lora adapter (the request's
model field), an add (vllm-lens steering vectors in every request: each listed layer gets alpha x its own row of the vector added to its output
at every position, under one of three scalings) and an ablation (a persistent hook on the server projecting each listed layer's row out of its
output, registered before the run and cleared after it by run.py). The named instances live in run.py and a run is `./run.py <name> ...`.

Vectors are utils.save_vector's data/vectors/<model>/<name>.safetensors, key "v" [n_layers, d_model], row i at layer layers[i] of the json sidecar.

    uv run python envs/common.py http://localhost:8000/v1    # clear the server's hooks by hand

Importing this module makes Inspect's vllm provider send tool schemas as the agent-interp-envs harness did, without the strict flag and the
additionalProperties it adds for OpenAI, so the server renders the same system prompt as that harness. It is a patch on the provider class
because the CLI resolves --model before it loads the task file.
"""
import base64
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cloudpickle
import httpx
import torch as t
from inspect_ai.model import GenerateConfig
from inspect_ai.model._openai import openai_chat_tools
from inspect_ai.model._providers.vllm import VLLMAPI
from inspect_ai.tool import ToolInfo
from omegaconf import OmegaConf
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / "configs"
VECTORS = ROOT / "data" / "vectors"
ENVS = ("grader", "secret_number", "impossible_bench")
SCALINGS = ("row", "unit", "resid_norm")
INTERVENTION = ("lora", "add_vector", "add_layers", "add_alpha", "add_scaling", "ablate_vector", "ablate_layers", "ablate_row", "vectors")

OmegaConf.register_new_resolver("pct", lambda x: int(float(x) * 100))
OmegaConf.register_new_resolver("pct_complement", lambda x: int((1 - float(x)) * 100))


@dataclass
class EvalCfg:
    env: str  # grader, secret_number or impossible_bench
    config: str  # envs/configs/<env>/<config>.yaml: the prompts, the scaffold settings and the model block sent with every request
    n: int | None = None  # grader: samples per side per family. secret_number: games. impossible_bench: problems, None for the whole split
    family: str | None = None  # grader: a family under envs/configs/grader/families, or all
    system: str | None = None  # grader: none, generic or very_hacker
    prompts: str | None = None  # grader: hack (the request with the other side's grader), clean (the side's own) or none (the request alone)
    seed: int | None = None  # secret_number: draws the secrets
    lora: str | None = None  # an adapter the server has loaded under that name
    add_vector: str | None = None  # a vector under data/vectors/<vectors>: each layer of add_layers gets its own row added to its output at every position
    add_layers: list[int] | None = None
    add_alpha: float | list[float] | None = None  # one alpha, or one per layer of add_layers. Positive: an add induces hacking, the suppression is the ablation
    add_scaling: str = "row"  # row: alpha x the row. unit: alpha x the unit row. resid_norm: alpha x the token's own residual norm x the unit row (vllm-lens norm_match)
    ablate_vector: str | None = None  # a vector whose rows are projected out of the outputs of ablate_layers, a hook on the server for the whole run
    ablate_layers: list[int] | None = None
    ablate_row: int | None = None  # that one layer's row projected out at every layer of ablate_layers; None for each layer's own row
    vectors: str = "Qwen3.6-27B"  # data/vectors/<vectors>
    model: str = "vllm/Qwen3.6-27B"  # the model as the server names it
    base_url: str = "http://localhost:8000/v1"  # the server, through the tunnel (ssh -N -f -L 8000:localhost:8000 vast)
    max_connections: int | None = 32  # concurrent requests, None for Inspect's default
    display: str = "rich"  # Inspect's display: rich is a progress panel, full the full-screen TUI
    name: str | None = None  # set by run.py from the instance's variable name


def plain_tools(self: VLLMAPI, tools: list[ToolInfo]) -> list[dict]:
    return openai_chat_tools(tools, exclude={"additionalProperties"})


VLLMAPI.tools_to_openai = plain_tools


def load_config(env: str, name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(CONFIGS / env / f"{name}.yaml"), resolve=True)


def load_vector(vectors: Path, name: str) -> tuple[t.Tensor, list[int]]:
    assert (vectors / f"{name}.safetensors").exists(), f"no vector {name} in {vectors}, which has {sorted(path.stem for path in vectors.glob('*.safetensors'))}"
    with safe_open(vectors / f"{name}.safetensors", "pt") as h:
        v = h.get_tensor("v")
    return v, json.loads((vectors / f"{name}.json").read_text())["layers"]


def condition(cfg: EvalCfg) -> str:
    """The run's label in the records and the rate tables: none when nothing is intervened on, else the instance's name."""
    return "none" if cfg.lora is None and cfg.add_vector is None and cfg.ablate_vector is None else cfg.name


def alphas(cfg: EvalCfg) -> list[float]:
    return cfg.add_alpha if isinstance(cfg.add_alpha, list) else [cfg.add_alpha] * len(cfg.add_layers)


def check(cfg: EvalCfg) -> None:
    """Assert that the fields fit together and that the files they name exist. Each env's task function checks its own task fields."""
    assert cfg.env in ENVS, f"{cfg.name}: env {cfg.env} is one of {ENVS}"
    assert (CONFIGS / cfg.env / f"{cfg.config}.yaml").exists(), f"{cfg.name}: no config {cfg.config} in {CONFIGS / cfg.env}, which has {sorted(p.stem for p in (CONFIGS / cfg.env).glob('*.yaml'))}"
    assert (cfg.env == "grader") == (cfg.family is not None) == (cfg.system is not None) == (cfg.prompts is not None), f"{cfg.name}: family, system and prompts are set exactly for grader"
    assert (cfg.env == "secret_number") == (cfg.seed is not None), f"{cfg.name}: seed is set exactly for secret_number"
    assert cfg.env == "impossible_bench" or cfg.n is not None, f"{cfg.name}: n is set for {cfg.env}"
    assert cfg.add_scaling in SCALINGS, f"{cfg.name}: add_scaling {cfg.add_scaling} is one of {SCALINGS}"
    assert (cfg.add_vector is None) == (cfg.add_layers is None) == (cfg.add_alpha is None), f"{cfg.name}: add_vector, add_layers and add_alpha are set together"
    if cfg.add_vector is not None:
        _, layers = load_vector(VECTORS / cfg.vectors, cfg.add_vector)
        assert set(cfg.add_layers) <= set(layers), f"{cfg.name}: {cfg.add_vector} has rows for layers {layers[0]} to {layers[-1]}, not {sorted(set(cfg.add_layers) - set(layers))}"
        assert len(alphas(cfg)) == len(cfg.add_layers), f"{cfg.name}: add_alpha lists {len(cfg.add_alpha)} alphas for {len(cfg.add_layers)} layers"
        assert all(a > 0 for a in alphas(cfg)), f"{cfg.name}: every alpha is positive, not {cfg.add_alpha}"
    assert (cfg.ablate_vector is None) == (cfg.ablate_layers is None), f"{cfg.name}: ablate_vector and ablate_layers are set together"
    assert cfg.ablate_vector is not None or cfg.ablate_row is None, f"{cfg.name}: ablate_row without ablate_vector"
    if cfg.ablate_vector is not None:
        _, layers = load_vector(VECTORS / cfg.vectors, cfg.ablate_vector)
        assert set(cfg.ablate_layers) <= set(layers), f"{cfg.name}: {cfg.ablate_vector} has rows for layers {layers[0]} to {layers[-1]}, not {sorted(set(cfg.ablate_layers) - set(layers))}"
        assert cfg.ablate_row is None or cfg.ablate_row in layers, f"{cfg.name}: {cfg.ablate_vector} has no row for layer {cfg.ablate_row}"
    assert cfg.model.startswith("vllm/"), f"{cfg.name}: interventions and token ids need the vLLM server, so model is vllm/<served name>, not {cfg.model}"
    assert cfg.base_url.endswith("/v1"), f"{cfg.name}: base_url ends in /v1, not {cfg.base_url}"


def tensor_json(v: t.Tensor) -> dict:
    """vllm-lens's wire format for a tensor, the uncompressed float32 variant."""
    arr = v.detach().float().cpu().numpy()
    return {"data": base64.b64encode(arr.tobytes()).decode(), "dtype": "float32", "original_dtype": "torch.float32", "shape": list(arr.shape)}


def steering_vectors(cfg: EvalCfg) -> list[dict]:
    """vllm-lens steering vectors for the add fields, one per layer: the layer's row (the unit row under unit and resid_norm) at scale alpha, norm matched under resid_norm."""
    V, layers = load_vector(VECTORS / cfg.vectors, cfg.add_vector)
    vectors = []
    for layer, alpha in zip(cfg.add_layers, alphas(cfg)):
        row = V[layers.index(layer)]
        row = row if cfg.add_scaling == "row" else row / row.norm()
        vectors.append({"activations": tensor_json(row[None]), "layer_indices": [layer], "scale": alpha, "norm_match": cfg.add_scaling == "resid_norm"})
    return vectors


def request_body(cfg: EvalCfg) -> dict:
    """The request fields the intervention adds. The ablation adds none: its hook lives on the server (set_hooks)."""
    body = {}
    if cfg.lora is not None:
        body["model"] = cfg.lora
    if cfg.add_vector is not None:
        body["vllm_xargs"] = {"apply_steering_vectors": json.dumps(steering_vectors(cfg))}
    return body


class ServerFunction:
    """Pickles as eval(source, namespace): the server compiles the function with its own Python, where cloudpickle would ship this interpreter's
    bytecode (unreadable by another minor version) and this torch's tensors. The namespace holds plain bytes and the torch module by name."""

    def __init__(self, source: str, namespace: dict):
        self.source, self.namespace = source, namespace

    def __reduce__(self):
        return (eval, (self.source, self.namespace))


# h <- h - (h . u) u with the layer's unit row built on the device once, per worker, from its float32 bytes
PROJECT_SRC = "lambda ctx, h: (lambda u: (h.float() - (h.float() @ u)[:, None] * u).to(h.dtype))(cache[ctx.layer_idx] if ctx.layer_idx in cache else cache.setdefault(ctx.layer_idx, torch.frombuffer(bytearray(rows[ctx.layer_idx]), dtype=torch.float32).to(h.device)))"


def projection_hook(cfg: EvalCfg) -> dict:
    """A vllm-lens Hook for the ablate fields: the projection at each listed layer's output, in the register endpoint's format."""
    V, layers = load_vector(VECTORS / cfg.vectors, cfg.ablate_vector)
    U = V / V.norm(dim=-1, keepdim=True)
    rows = {layer: U[layers.index(layer if cfg.ablate_row is None else cfg.ablate_row)].float().numpy().tobytes() for layer in cfg.ablate_layers}
    project = ServerFunction(PROJECT_SRC, {"rows": rows, "cache": {}, "torch": t})
    return {"fn": {"cloudpickle": base64.b64encode(cloudpickle.dumps(project)).decode()}, "layer_indices": cfg.ablate_layers, "pre": False}


def clear_hooks(base_url: str) -> dict:
    r = httpx.post(f"{base_url}/hooks/clear", timeout=120)
    r.raise_for_status()
    return r.json()


def set_hooks(base_url: str, cfg: EvalCfg) -> dict:
    """Make the server's persistent hooks exactly what the run needs: cleared, then the projection registered when the run ablates."""
    cleared = clear_hooks(base_url)
    if cfg.ablate_vector is None:
        return cleared
    r = httpx.post(f"{base_url}/hooks/register", json={"hooks": [projection_hook(cfg)]}, timeout=600)
    r.raise_for_status()
    return r.json()


def generate_config(cfg: EvalCfg) -> GenerateConfig:
    """The task's GenerateConfig: the yaml config's model block with the intervention's request fields merged into extra_body. The intervention
    fields are also the request's cache_salt, so the server's prefix cache never serves blocks computed under another intervention."""
    check(cfg)
    model = dict(load_config(cfg.env, cfg.config)["model"])
    salt = json.dumps([getattr(cfg, field) for field in INTERVENTION])
    extra_body = model.pop("extra_body") | request_body(cfg) | {"cache_salt": salt}
    return GenerateConfig(**model, extra_body=extra_body)


def task_metadata(cfg: EvalCfg) -> dict:
    """What every log and record carries: the env, the yaml config's name, the condition label, the vectors dir, and the whole EvalCfg."""
    return {"env": cfg.env, "config_id": cfg.config, "condition": condition(cfg), "vectors": cfg.vectors, "cfg": asdict(cfg)}


if __name__ == "__main__":
    print(clear_hooks(sys.argv[1]))
