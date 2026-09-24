"""Stage 4 forward passes: residual-stream capture reduced on the fly (torch; imported only by the GPU script).

Hooks on the token embedding, on every decoder layer but the last, and on the final norm reproduce the
`output_hidden_states=True` tuple of transformers (index 0 = embeddings, i = output of layer i-1, last =
post-norm), which is what heretic reads. Instead of keeping (layers × tokens × hidden) tensors, each hook
multiplies the layer's output by per-aggregation position weights and stores one vector per aggregation:
buffer[layer, batch, aggregation, hidden]. Statistics are accumulated in float32.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelParts:
    model: nn.Module
    text_model: nn.Module
    embed_tokens: nn.Module
    layers: nn.ModuleList
    norm: nn.Module
    lm_head: nn.Module
    n_layers: int
    hidden: int
    device: torch.device
    dtype: torch.dtype


def find_parts(model: nn.Module) -> ModelParts:
    """Locate the text decoder inside a (possibly multimodal) causal LM."""
    text = None
    for path in ("model.language_model", "language_model.model", "model", "language_model"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "layers") and hasattr(obj, "embed_tokens") and hasattr(obj, "norm"):
            text = obj
            break
    if text is None:
        raise ValueError(f"cannot find the text decoder (embed_tokens/layers/norm) in {type(model).__name__}")
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("model has no lm_head")
    p = next(text.parameters())
    return ModelParts(
        model=model, text_model=text, embed_tokens=text.embed_tokens, layers=text.layers, norm=text.norm, lm_head=lm_head,
        n_layers=len(text.layers), hidden=text.embed_tokens.embedding_dim, device=p.device, dtype=p.dtype,
    )


def load_model(model_id: str, revision: str | None, dtype: str, device_map: str = "cuda", attn_implementation: str | None = None) -> ModelParts:
    import importlib.util

    from transformers import AutoModelForCausalLM

    kwargs = {"dtype": getattr(torch, dtype)}
    # device_map needs accelerate (loads shards straight to the device); without it load on CPU and move.
    use_device_map = importlib.util.find_spec("accelerate") is not None
    if use_device_map:
        kwargs["device_map"] = device_map
    if revision:
        kwargs["revision"] = revision
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError) as e:
        if "accelerate" in str(e):
            raise
        # Multimodal checkpoints (e.g. Qwen3_5ForConditionalGeneration) are not always mapped by AutoModelForCausalLM.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if not use_device_map:
        model = model.to(device_map)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return find_parts(model)


def _tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


class ResidualReducer:
    """Registers hooks that reduce each residual layer output with per-batch position weights."""

    def __init__(self, parts: ModelParts, n_aggregations: int):
        self.parts = parts
        self.n_agg = n_aggregations
        self.n_layers_plus1 = parts.n_layers + 1
        self.weights: torch.Tensor | None = None  # (B, n_agg, T) float32
        self.mask: torch.Tensor | None = None  # (B, T) bool
        self.buffer: torch.Tensor | None = None  # (L1, B, n_agg, H) float32
        self.max_abs: torch.Tensor | None = None  # (L1,) float32 over non-padded positions
        self.handles = []
        self.handles.append(parts.embed_tokens.register_forward_hook(self._hook(0)))
        for i in range(parts.n_layers - 1):
            self.handles.append(parts.layers[i].register_forward_hook(self._hook(i + 1)))
        self.handles.append(parts.norm.register_forward_hook(self._hook(parts.n_layers)))

    def _hook(self, index: int):
        def fn(module, inputs, output):
            h = _tensor_output(output)
            if self.weights is None:
                return
            hf = h.float()
            self.buffer[index] = torch.einsum("bat,bth->bah", self.weights, hf)
            if self.mask is not None:
                self.max_abs[index] = hf.abs().amax(dim=-1).masked_fill(~self.mask, 0).max()
            else:
                self.max_abs[index] = hf.abs().max()

        return fn

    def set_batch(self, weights: torch.Tensor, mask: torch.Tensor) -> None:
        B, A, T = weights.shape
        self.weights = weights.to(self.parts.device, torch.float32)
        self.mask = mask.to(self.parts.device, torch.bool)
        self.buffer = torch.zeros((self.n_layers_plus1, B, A, self.parts.hidden), dtype=torch.float32, device=self.parts.device)
        self.max_abs = torch.zeros(self.n_layers_plus1, dtype=torch.float32, device=self.parts.device)

    def clear(self) -> None:
        self.weights = self.mask = self.buffer = self.max_abs = None

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def make_batches(items: list[tuple[int, int]], max_tokens_per_batch: int, max_batch_size: int) -> list[list[int]]:
    """Group (key, seq_len) items into batches of keys: longest first, padded size <= max_tokens_per_batch."""
    order = sorted(items, key=lambda kv: -kv[1])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for key, n in order:
        new_max = max(cur_max, n)
        if cur and ((len(cur) + 1) * new_max > max_tokens_per_batch or len(cur) >= max_batch_size):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = n
        cur.append(key)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def pad_batch(seqs: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return ids, mask


def weights_tensor(sparse: list[dict[str, list[tuple[int, float]]]], aggregations: list[str], T: int) -> torch.Tensor:
    """(B, n_agg, T) float32 from per-sequence sparse weights."""
    W = torch.zeros((len(sparse), len(aggregations), T), dtype=torch.float32)
    for b, per_agg in enumerate(sparse):
        for a, agg in enumerate(aggregations):
            for pos, w in per_agg[agg]:
                W[b, a, pos] = w
    return W


@torch.no_grad()
def forward_reduce(parts: ModelParts, reducer: ResidualReducer, ids: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor):
    """Run the text decoder once; returns (buffer (L1,B,A,H) float32 on CPU, max_abs (L1,), last_hidden (B,T,H) on device)."""
    reducer.set_batch(weights, mask)
    out = parts.text_model(input_ids=ids.to(parts.device), attention_mask=mask.to(parts.device, torch.long), use_cache=False)
    last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else _tensor_output(out)
    buffer, max_abs = reducer.buffer.cpu(), reducer.max_abs.cpu()
    reducer.clear()
    return buffer, max_abs, last_hidden


@torch.no_grad()
def teacher_forced_nll(parts: ModelParts, last_hidden: torch.Tensor, ids: torch.Tensor, positions: list[list[int]], chunk: int = 1024) -> list[dict]:
    """Mean NLL and top-1 accuracy of the tokens at `positions[b]` given everything before them."""
    results = []
    ids_dev = ids.to(parts.device)
    for b, pos in enumerate(positions):
        pos = [p for p in pos if p >= 1]
        if not pos:
            results.append({"nll": float("nan"), "top1": float("nan"), "n": 0})
            continue
        nll_sum, correct = 0.0, 0
        for start in range(0, len(pos), chunk):
            p = torch.tensor(pos[start : start + chunk], device=parts.device)
            logits = parts.lm_head(last_hidden[b, p - 1]).float()
            target = ids_dev[b, p]
            nll_sum += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
            correct += (logits.argmax(-1) == target).sum().item()
        results.append({"nll": nll_sum / len(pos), "top1": correct / len(pos), "n": len(pos)})
    return results


@torch.no_grad()
def hidden_states_reference(parts: ModelParts, ids: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """transformers' own output_hidden_states for one batch (reference for the hook check)."""
    out = parts.text_model(input_ids=ids.to(parts.device), attention_mask=mask.to(parts.device, torch.long), use_cache=False, output_hidden_states=True)
    return [h.float() for h in out.hidden_states]
