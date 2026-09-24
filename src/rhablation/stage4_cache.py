"""Stage 4 activation cache: per-trajectory, per-layer aggregated residuals as memory-mapped arrays (numpy only).

Layout of `<cache_dir>/`:
  cache_config.json   model, revision, shapes, dtype, aggregations, layer index convention, manifest sha256
  manifest.jsonl      copy of the manifest the cache was built from (row = manifest `order`)
  traj/<agg>.<f16|f32>   np.memmap of shape (n_layers_plus1, n_rows, hidden): layer-major so that one layer
                      of every trajectory is a contiguous read (`CacheReader.layer`)
  index.jsonl         one line per finished trajectory (order, trajectory_id, token counts, max |h|, nll)
  progress.json       rewritten by the builder after every chunk

Rows not listed in index.jsonl hold zeros and must be ignored; the builder resumes from index.jsonl.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

DTYPE_SUFFIX = {"float16": "f16", "float32": "f32"}
LAYER_INDEX_CONVENTION = (
    "transformers output_hidden_states: index 0 = token embeddings, index i (1..n_layers-1) = output of decoder "
    "layer i-1, index n_layers = output of the final norm (= what heretic sees as the last layer)"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_path(cache_dir: Path, agg: str, storage_dtype: str) -> Path:
    return Path(cache_dir) / "traj" / f"{agg}.{DTYPE_SUFFIX[storage_dtype]}"


class CacheWriter:
    def __init__(self, cache_dir: Path, config: dict, n_rows: int, n_layers_plus1: int, hidden: int):
        self.dir = Path(cache_dir)
        self.config = config
        self.n_rows, self.n_layers_plus1, self.hidden = n_rows, n_layers_plus1, hidden
        self.storage_dtype = config["storage_dtype"]
        self.aggregations: list[str] = list(config["aggregations"])
        (self.dir / "traj").mkdir(parents=True, exist_ok=True)
        cfg_path = self.dir / "cache_config.json"
        if cfg_path.exists():
            existing = json.loads(cfg_path.read_text())
            for key in ("model", "model_revision", "storage_dtype", "aggregations", "n_rows", "n_layers_plus1", "hidden", "manifest_sha256"):
                if existing.get(key) != config.get(key):
                    raise ValueError(f"{cfg_path}: {key} differs from the current run ({existing.get(key)!r} vs {config.get(key)!r}); use a new cache dir")
        else:
            cfg_path.write_text(json.dumps(config, indent=1) + "\n")
        shape = (n_layers_plus1, n_rows, hidden)
        self.arrays: dict[str, np.memmap] = {}
        for agg in self.aggregations:
            p = array_path(self.dir, agg, self.storage_dtype)
            mode = "r+" if p.exists() else "w+"
            self.arrays[agg] = np.memmap(p, dtype=np.dtype(self.storage_dtype), mode=mode, shape=shape)
        self.index_path = self.dir / "index.jsonl"

    def done_orders(self) -> set[int]:
        if not self.index_path.exists():
            return set()
        with self.index_path.open() as f:
            return {json.loads(line)["order"] for line in f if line.strip()}

    def write_rows(self, orders: list[int], values: dict[str, np.ndarray]) -> None:
        """values[agg]: float32 array (len(orders), n_layers_plus1, hidden)."""
        idx = np.asarray(orders)
        for agg in self.aggregations:
            v = values[agg]
            if v.shape != (len(orders), self.n_layers_plus1, self.hidden):
                raise ValueError(f"{agg}: got {v.shape}")
            self.arrays[agg][:, idx, :] = np.transpose(v, (1, 0, 2)).astype(self.storage_dtype)

    def append_index(self, entries: Iterable[dict]) -> None:
        for agg in self.aggregations:
            self.arrays[agg].flush()
        with self.index_path.open("a") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_progress(self, progress: dict) -> None:
        tmp = self.dir / "progress.json.tmp"
        tmp.write_text(json.dumps(progress, indent=1) + "\n")
        tmp.replace(self.dir / "progress.json")


class CacheReader:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.config = json.loads((self.dir / "cache_config.json").read_text())
        self.storage_dtype = self.config["storage_dtype"]
        self.n_rows = self.config["n_rows"]
        self.n_layers_plus1 = self.config["n_layers_plus1"]
        self.hidden = self.config["hidden"]
        self.aggregations: list[str] = list(self.config["aggregations"])
        self.index: dict[int, dict] = {}
        index_path = self.dir / "index.jsonl"
        if index_path.exists():
            with index_path.open() as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        self.index[e["order"]] = e
        self._arrays: dict[str, np.memmap] = {}

    def done_orders(self) -> set[int]:
        return set(self.index)

    def _array(self, agg: str) -> np.memmap:
        if agg not in self._arrays:
            if agg not in self.aggregations:
                raise KeyError(f"aggregation {agg!r} not in cache {self.aggregations}")
            self._arrays[agg] = np.memmap(
                array_path(self.dir, agg, self.storage_dtype),
                dtype=np.dtype(self.storage_dtype),
                mode="r",
                shape=(self.n_layers_plus1, self.n_rows, self.hidden),
            )
        return self._arrays[agg]

    def layer(self, agg: str, layer: int, orders: np.ndarray) -> np.ndarray:
        """(len(orders), hidden) float32 for one aggregation and layer index. Orders must be finished rows."""
        arr = self._array(agg)
        orders = np.asarray(orders)
        missing = [int(o) for o in orders if int(o) not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} requested rows are not in the cache index (e.g. {missing[:5]})")
        # Sorted fancy indexing on the contiguous layer block; restore the caller's order afterwards.
        srt = np.argsort(orders, kind="stable")
        block = np.asarray(arr[layer][orders[srt]], dtype=np.float32)
        out = np.empty_like(block)
        out[srt] = block
        return out
