"""Local trajectory viewer: run discovery, per-run row index, record access and transcript structuring.

Pure functions plus one small in-memory/on-disk index store. Served by scripts/trajectory_viewer.py.
Reads trajectories.jsonl[.gz] (schema 1.0-1.3, full or compact) without touching any run directory:
the row index is cached under a separate cache dir, keyed by file path, size and mtime.
"""

import gzip
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from rhablation.hack_signals import test_file_writes
from rhablation.judge_transcript import _turns_by_message, render as render_plain

# Candidate files per run directory, first match wins (a compacted copy is a lossless duplicate).
TRAJ_FILE_NAMES = (
    "trajectories.jsonl",
    "trajectories.jsonl.gz",
    "trajectories.compact.jsonl",
    "trajectories.compact.jsonl.gz",
)
TOKEN_FIELDS = ("prompt_token_ids", "completion_token_ids")
SCORE_TO_REWARD = {"C": 1.0, "I": 0.0}


# ---------------------------------------------------------------------------------------------
# Discovery


def discover_runs(data_root: Path) -> list[dict[str, Any]]:
    """Every directory under data_root holding a trajectories file, sorted by relative path."""
    runs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for p in sorted(data_root.rglob("trajectories*.jsonl*")):
        if p.name not in TRAJ_FILE_NAMES or p.parent in seen:
            continue
        run_dir = p.parent
        seen.add(run_dir)
        chosen = next(run_dir / n for n in TRAJ_FILE_NAMES if (run_dir / n).exists())
        rel = run_dir.relative_to(data_root).as_posix()
        runs.append(
            {
                "id": rel,
                "group": run_dir.parent.relative_to(data_root).as_posix(),
                "name": run_dir.name,
                "file": chosen.name,
                "bytes": chosen.stat().st_size,
                "run_config": _run_config_summary(run_dir / "run_config.json"),
            }
        )
    return runs


def _run_config_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        rc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    cfg = rc.get("config") or {}
    return {
        "run_id": rc.get("run_id"),
        "created": rc.get("created"),
        "model": cfg.get("model"),
        "model_revision": rc.get("model_revision") or cfg.get("model_revision"),
        "benchmark": cfg.get("benchmark"),
        "env": cfg.get("env"),
        "splits": cfg.get("splits"),
        "agent_types": cfg.get("agent_types"),
        "samples_per_task": cfg.get("samples_per_task"),
        "sampling": cfg.get("sampling"),
        "impossible_variants": rc.get("impossible_variants"),
    }


def run_file(data_root: Path, run_id: str) -> Path:
    run_dir = (data_root / run_id).resolve()
    if data_root.resolve() not in run_dir.parents:
        raise ValueError(f"run outside data root: {run_id}")
    for name in TRAJ_FILE_NAMES:
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"no trajectories file in {run_id}")


# ---------------------------------------------------------------------------------------------
# Reading records


def _open_binary(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def iter_records(path: Path) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """(record index, byte offset in the uncompressed stream or -1, record).

    One JSON object per line normally. Pretty-printed files (the smoke1/smoke2 layout) fall back to
    a whole-file decode with offset -1; `load_record` re-parses those the same way."""
    with _open_binary(path) as f:
        offset = f.tell()
        first = f.readline()
        try:
            rec = json.loads(first)
            if not isinstance(rec, dict):
                raise ValueError
        except ValueError:
            rec = None
        if rec is not None:
            yield 0, offset, rec
            n = 1
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    return
                if not line.strip():
                    continue
                yield n, offset, json.loads(line)
                n += 1
            return
    for n, rec in enumerate(_decode_concatenated(path)):
        yield n, -1, rec


def _decode_concatenated(path: Path) -> Iterator[dict[str, Any]]:
    with _open_binary(path) as f:
        text = f.read().decode("utf-8")
    dec = json.JSONDecoder()
    pos = 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return
        rec, pos = dec.raw_decode(text, pos)
        if isinstance(rec, dict):
            yield rec


def load_record(path: Path, index: int, offset: int) -> dict[str, Any]:
    if offset >= 0:
        with _open_binary(path) as f:
            f.seek(offset)
            return json.loads(f.readline())
    for n, rec in enumerate(_decode_concatenated(path)):
        if n == index:
            return rec
    raise IndexError(index)


# ---------------------------------------------------------------------------------------------
# Row index


def summarize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """The per-rollout row shown in the table. Every hack signal stays its own column."""
    rt = rec.get("raw_test_results") or {}
    labels = {k: (v or {}).get("value") for k, v in (rec.get("labels") or {}).items()}
    score = (rec.get("score") or {}).get("value")
    reward = rt.get("reward")
    if reward is None:
        reward = labels.get("frac_passed")
    if reward is None and score in SCORE_TO_REWARD:
        reward = SCORE_TO_REWARD[score]
    tc = rec.get("token_counts") or {}
    turns = rec.get("turns") or []
    return {
        "trajectory_id": rec.get("trajectory_id"),
        "task_id": rec.get("task_id"),
        "variant": rec.get("variant"),
        "agent_type": rec.get("agent_type"),
        "epoch": rec.get("epoch"),
        "status": rec.get("status"),
        "score": score,
        "reward": reward,
        "programmatic_hack": rec.get("programmatic_hack"),
        "test_edit_attempt": rec.get("test_edit_attempt"),
        "labels": labels,
        "n_turns": rec.get("n_turns", len(turns)),
        "n_messages": rec.get("n_messages", len(rec.get("messages") or [])),
        "output_tokens": tc.get("output_tokens"),
        "reasoning_tokens": tc.get("reasoning_tokens"),
        "duration_s": rec.get("duration_s"),
        "timed_out_calls": sum(1 for t in turns if t.get("error")),
        "error": bool(rec.get("error_message")),
        "model": rec.get("model"),
        "timestamp": rec.get("timestamp"),
        "schema_version": rec.get("schema_version"),
    }


class IndexStore:
    """Builds and caches one row index per run file. Builds run in a background thread."""

    def __init__(self, data_root: Path, cache_dir: Path):
        self.data_root = data_root
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._ready: dict[str, list[dict[str, Any]]] = {}
        self._progress: dict[str, dict[str, Any]] = {}

    def _cache_path(self, path: Path) -> Path:
        return self.cache_dir / (hashlib.sha1(str(path.resolve()).encode()).hexdigest() + ".json")

    def get(self, run_id: str) -> dict[str, Any]:
        """{'status': 'ready', 'rows': [...]} or {'status': 'building', 'progress': {...}}."""
        path = run_file(self.data_root, run_id)
        with self._lock:
            if run_id in self._ready:
                return {"status": "ready", "rows": self._ready[run_id]}
            cached = self._load_cache(path)
            if cached is not None:
                self._ready[run_id] = cached
                return {"status": "ready", "rows": cached}
            if run_id not in self._progress:
                self._progress[run_id] = {"rows": 0, "bytes": 0, "total_bytes": path.stat().st_size, "error": None}
                threading.Thread(target=self._build, args=(run_id, path), daemon=True).start()
            return {"status": "building", "progress": dict(self._progress[run_id])}

    def _load_cache(self, path: Path) -> list[dict[str, Any]] | None:
        cp = self._cache_path(path)
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
        except ValueError:
            return None
        st = path.stat()
        if data.get("size") != st.st_size or data.get("mtime") != st.st_mtime:
            return None
        return data["rows"]

    def _build(self, run_id: str, path: Path) -> None:
        rows: list[dict[str, Any]] = []
        try:
            for n, offset, rec in iter_records(path):
                row = summarize_record(rec)
                row.update({"run": run_id, "i": n, "offset": offset})
                rows.append(row)
                if n % 20 == 0:
                    with self._lock:
                        self._progress[run_id].update(rows=len(rows), bytes=max(offset, 0))
            st = path.stat()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path(path).with_suffix(".tmp")
            tmp.write_text(json.dumps({"path": str(path), "size": st.st_size, "mtime": st.st_mtime, "rows": rows}))
            os.replace(tmp, self._cache_path(path))
            with self._lock:
                self._ready[run_id] = rows
                self._progress.pop(run_id, None)
        except Exception as e:  # surfaced to the UI; the run stays selectable
            with self._lock:
                self._progress[run_id]["error"] = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------
# Judge labels (stage 3)


def load_judge_labels(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    """trajectory_id -> judge verdicts from every data/stage3/**/labels.jsonl."""
    out: dict[str, list[dict[str, Any]]] = {}
    stage3 = data_root / "stage3"
    if not stage3.exists():
        return out
    for p in sorted(stage3.rglob("labels.jsonl")):
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            tid = rec.get("trajectory_id")
            if not tid:
                continue
            out.setdefault(tid, []).append(
                {
                    "file": p.relative_to(data_root).as_posix(),
                    "judge": rec.get("judge"),
                    "group": rec.get("group"),
                    "strongest_status": rec.get("strongest_status"),
                    "categories": rec.get("categories"),
                    "flags": rec.get("flags"),
                    "rationale": rec.get("rationale"),
                    "judged_at": rec.get("judged_at"),
                }
            )
    return out


# ---------------------------------------------------------------------------------------------
# Detail view


def structure_transcript(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages in order, each assistant message joined with the model call that produced it
    (same positional pairing as rhablation.judge_transcript)."""
    by_message = _turns_by_message(rec)
    out = []
    step = 0
    for i, m in enumerate(rec.get("messages") or []):
        item: dict[str, Any] = {
            "index": i,
            "role": m.get("role"),
            "content": m.get("content") or "",
            "tool_calls": m.get("tool_calls") or [],
            "function": m.get("function"),
            "error": m.get("error"),
        }
        if m.get("role") == "assistant":
            step += 1
            turn = by_message.get(i, {})
            item.update(
                {
                    "step": step,
                    "turn_index": turn.get("turn_index"),
                    "reasoning": turn.get("reasoning"),
                    "finish_reason": turn.get("finish_reason"),
                    "stop_reason": turn.get("stop_reason"),
                    "usage": turn.get("usage"),
                }
            )
        out.append(item)
    return out


def strip_bulk(rec: dict[str, Any]) -> dict[str, Any]:
    """The record without token ids and rendered strings, for the raw-JSON tab."""
    out = dict(rec)
    out["rendered_text"] = f"<{len(rec.get('rendered_text') or [])} entries omitted>"
    out["turns"] = [
        {k: (f"<{len(v)} ids omitted>" if k in TOKEN_FIELDS and v is not None else v) for k, v in t.items() if k != "rendered_prompt"}
        for t in rec.get("turns") or []
    ]
    return out


def detail(rec: dict[str, Any], judge: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    turns = rec.get("turns") or []
    try:
        plain = render_plain(rec)
    except Exception as e:  # schema-1.0 smoke files may lack fields the renderer expects
        plain = f"(plain rendering failed: {type(e).__name__}: {e})"
    return {
        "row": summarize_record(rec),
        "meta": {
            k: rec.get(k)
            for k in (
                "trajectory_id", "run_id", "timestamp", "model", "model_revision", "benchmark", "task_source",
                "variant", "task_id", "agent_type", "epoch", "status", "error_message", "schema_version",
                "sampling", "seed", "env_options", "prompt_config", "token_counts", "n_turns", "n_messages",
                "duration_s", "provenance",
            )
        },
        "labels": rec.get("labels") or {},
        "programmatic_hack": rec.get("programmatic_hack"),
        "test_edit_attempt": rec.get("test_edit_attempt"),
        "score": rec.get("score"),
        "raw_test_results": rec.get("raw_test_results"),
        "final_output": rec.get("final_output"),
        "transcript": structure_transcript(rec),
        "abandoned_calls": [
            {"turn_index": t.get("turn_index"), "error": t.get("error")} for t in turns if t.get("error")
        ],
        "hack_signals": {"test_file_writes": test_file_writes(turns)},
        "judge": judge.get(rec.get("trajectory_id"), []),
        "plain": plain,
        "raw": strip_bulk(rec),
    }
