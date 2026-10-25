"""Local trajectory viewer: run discovery, per-run row index, record access and transcript structuring.

Pure functions plus one small in-memory/on-disk index store. Served by scripts/trajectory_viewer.py.
Reads trajectories.jsonl[.gz] (schema 1.0-1.3, full or compact) without touching any run directory:
the row index is cached under a separate cache dir, keyed by file path, size and mtime. Extra roots
in the ortho layout (`<root>/inspect/<env>/<log>.jsonl`, records of Ethan's `envs/convert.py`) are
read through `rhablation.ortho_records`, one file per run. CheatBench runs (`<run>/episodes.jsonl` + `run.json`)
are read through `rhablation.cheatbench_records`; their judge verdicts and provisional labels are joined from
side files (`side_labels`).
"""

import gzip
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from rhablation import cheatbench_records as cb
from rhablation.hack_signals import test_file_writes
from rhablation.judge_transcript import _turns_by_message, render as render_plain
from rhablation.ortho_records import is_ortho_record, normalize, with_rendered_system_turn

# Candidate files per run directory, first match wins (a compacted copy is a lossless duplicate).
# `records.jsonl[.gz]` is a run written in the ortho record format (sweep/benchmarks/ortho_grader).
TRAJ_FILE_NAMES = (
    "trajectories.jsonl",
    "trajectories.jsonl.gz",
    "trajectories.compact.jsonl",
    "trajectories.compact.jsonl.gz",
    "records.jsonl",
    "records.jsonl.gz",
    cb.FILE_NAME,
)
RUN_FILE_GLOBS = ("trajectories*.jsonl*", "records.jsonl*", cb.FILE_NAME)
TOKEN_FIELDS = ("prompt_token_ids", "completion_token_ids")
SCORE_TO_REWARD = {"C": 1.0, "I": 0.0}
# Second layout (Ethan's ortho repo): <extra root>/inspect/<env>/<inspect log name>.jsonl, one file = one run.
ORTHO_SUBDIR = "inspect"

ExtraRoots = dict[str, Path]  # name -> directory; run ids under it are "<name>/<relative path>"


# ---------------------------------------------------------------------------------------------
# Discovery


def discover_runs(data_root: Path, extra_roots: ExtraRoots | None = None) -> list[dict[str, Any]]:
    """Every directory under data_root holding a trajectories file, sorted by relative path, followed by
    every ortho-layout file under each extra root."""
    runs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for p in sorted(q for g in RUN_FILE_GLOBS for q in data_root.rglob(g)):
        if p.name not in TRAJ_FILE_NAMES or p.parent in seen:
            continue
        run_dir = p.parent
        seen.add(run_dir)
        chosen = next(run_dir / n for n in TRAJ_FILE_NAMES if (run_dir / n).exists())
        rel = run_dir.relative_to(data_root).as_posix()
        runs.append(
            {
                "id": rel,
                "group": (group := group_of(run_dir.relative_to(data_root))),
                "collection": collection_of(group),
                "name": run_dir.name,
                "file": chosen.name,
                "bytes": chosen.stat().st_size,
                "run_config": cb.run_config_summary(run_dir / "run.json") if chosen.name == cb.FILE_NAME else _run_config_summary(run_dir / "run_config.json"),
            }
        )
    for name, root in (extra_roots or {}).items():
        runs.extend(_discover_ortho(name, root))
    return runs


def _discover_ortho(name: str, root: Path) -> list[dict[str, Any]]:
    base = root / ORTHO_SUBDIR
    if not base.is_dir():
        return []
    runs = []
    for p in sorted(base.glob("*/*.jsonl")):
        rel = p.relative_to(root).as_posix()
        runs.append(
            {
                "id": f"{name}/{rel}",
                "group": (group := f"{name}/{p.parent.relative_to(root).as_posix()}"),
                "collection": collection_of(group),
                "name": p.stem,
                "file": p.name,
                "bytes": p.stat().st_size,
                "run_config": _ortho_config_summary(p),
            }
        )
    return runs


def group_of(rel: Path) -> str:
    """Sidebar group of a run dir (relative to the data root): its parent path, except under stage6/, whose layout
    is stage6/<checkpoint>/<run> (stages 5b-5e, run names s5b_..s5e_) or stage6/<checkpoint>/rhenvs/<run> (stage 7).
    There the group is stage6/<mission>/<checkpoint>, so one mission's checkpoints share a collection. A CheatBench
    run dir (<task>_cb-<row>_<run id>) is grouped by its row: <parent>/<row>."""
    parts = rel.parts
    m = re.match(r".+_cb-([^_]+)_[^_]+$", parts[-1])
    if m and "cheatbench" in rel.as_posix():
        return f"{rel.parent.as_posix()}/{m[1]}"
    if len(parts) >= 3 and parts[0] == "stage6":
        if parts[2] == "rhenvs":
            return f"stage6/stage7/{parts[1]}"
        m = re.match(r"s(5[a-z])_", parts[-1])
        if m:
            return f"stage6/stage{m[1]}/{parts[1]}"
    return rel.parent.as_posix()


def collection_of(group: str) -> str:
    """Sidebar collection of a group: its parent path (stage1 -> stage1, sweep/<model>/<env> -> sweep/<model>,
    <name>/inspect/<env> -> <name>/inspect)."""
    return group.rpartition("/")[0] or group


def _ortho_created(path: Path) -> str | None:
    """Inspect log names start with the run's start time, e.g. 2026-10-01T01-36-04-00-00_<task>_<id>.jsonl."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})", path.name)
    return f"{m[1]}T{m[2]}:{m[3]}:{m[4]}" if m else None


def _ortho_config_summary(path: Path) -> dict[str, Any] | None:
    """From the first record: the run's model, env, config and condition (ortho has no run_config.json)."""
    try:
        with path.open("rb") as f:
            rec = json.loads(f.readline())
    except (OSError, ValueError):
        return None
    cfg = rec.get("cfg") or {}
    return {
        "run_id": rec.get("rollout_id", "").split("/")[0] or None,
        "created": _ortho_created(path),
        "model": rec.get("model"),
        "model_revision": None,
        "benchmark": "ortho",
        "env": {k: rec.get(k) for k in ("env", "config_id", "condition") if k in rec},
        "splits": None,
        "agent_types": None,
        "samples_per_task": cfg.get("n"),
        "sampling": {k: v for k, v in cfg.items() if v is not None and k in ("family", "system", "prompts", "seed", "lora", "add_vector", "add_layers", "add_alpha", "add_scaling", "ablate_vector", "ablate_layers", "ablate_row")} or None,
        "impossible_variants": None,
    }


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


def run_file(data_root: Path, run_id: str, extra_roots: ExtraRoots | None = None) -> Path:
    """The trajectories file of a run id: a run dir under data_root, or "<name>/inspect/<env>/<file>.jsonl"
    under the extra root called <name> (an ortho-layout file is its own run)."""
    head, _, rest = run_id.partition("/")
    if extra_roots and head in extra_roots and rest.startswith(ORTHO_SUBDIR + "/"):
        root = extra_roots[head].resolve()
        p = (root / rest).resolve()
        if root not in p.parents or p.suffix != ".jsonl" or not p.is_file():
            raise FileNotFoundError(f"no ortho rollout file for {run_id}")
        return p
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
            rec = json.loads(f.readline())
        if path.name == cb.FILE_NAME and cb.is_cheatbench_record(rec):
            _annotate_cheatbench(rec, path, _cheatbench_attempt(path, offset, rec["episode"]))
        return rec
    for n, rec in enumerate(_decode_concatenated(path)):
        if n == index:
            return rec
    raise IndexError(index)


def _cheatbench_attempt(path: Path, offset: int, episode: str) -> int:
    """The attempt number of the record at `offset`: 1 + earlier lines of the same episode."""
    with _open_binary(path) as f:
        head = f.read(offset)
    return 1 + sum(json.loads(line).get("episode") == episode for line in head.splitlines() if line.strip())


def _annotate_cheatbench(rec: dict[str, Any], path: Path, attempt: int) -> None:
    rec["_viewer_run"] = path.parent.name
    rec["_viewer_attempt"] = attempt


def _normalize(rec: dict[str, Any]) -> dict[str, Any]:
    if "_viewer_run" in rec and cb.is_cheatbench_record(rec):
        return cb.to_trajectory(rec, rec["_viewer_run"], rec["_viewer_attempt"])
    return normalize(rec)


# ---------------------------------------------------------------------------------------------
# Row index


def summarize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """The per-rollout row shown in the table. Every hack signal stays its own column."""
    rec = _normalize(rec)
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


def parse_find_query(text: str, run_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split pasted text into (runs to search, ids to find). Tokens are separated by whitespace or commas. A token
    naming a run (its id, optionally with a leading `data/` and a trailing trajectories/records file name) scopes
    the search to that run; every other token is an id. No run token = search every run."""
    known = set(run_ids)
    scope: list[str] = []
    queries: list[str] = []
    for tok in re.split(r"[\s,]+", text.strip()):
        if not tok:
            continue
        cand = tok.strip("/")
        head, _, last = cand.rpartition("/")
        if last in TRAJ_FILE_NAMES:
            cand = head
        cand = cand.removeprefix("data/")
        if cand in known:
            if cand not in scope:
                scope.append(cand)
        elif tok not in queries:
            queries.append(tok)
    return scope, queries


def row_matches(row: dict[str, Any], q: str) -> bool:
    """A row matches its full trajectory id, any '/'-aligned tail of it (`eval_7919_253/2` for an ortho rollout id
    `<log>/eval_7919_253/2`), or its task id."""
    tid = str(row.get("trajectory_id") or "")
    return tid == q or tid.endswith("/" + q) or str(row.get("task_id")) == q


class IndexStore:
    """Builds and caches one row index per run file. Builds run in a background thread."""

    def __init__(self, data_root: Path, cache_dir: Path, extra_roots: ExtraRoots | None = None, side_labels: dict[str, dict[str, Any]] | None = None):
        self.data_root = data_root
        self.side_labels = side_labels or {}
        self.cache_dir = cache_dir
        self.extra_roots = extra_roots or {}
        self._lock = threading.Lock()
        self._ready: dict[str, list[dict[str, Any]]] = {}
        self._progress: dict[str, dict[str, Any]] = {}
        self._queue: list[tuple[str, Path]] = []  # runs waiting for the find worker
        self._worker: threading.Thread | None = None

    def _with_side(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rows with the side-table labels (judge verdicts, provisional reads) merged in at serve time, so the
        cached index does not go stale when those files change."""
        if not self.side_labels:
            return rows
        out = []
        for r in rows:
            extra = self.side_labels.get(r.get("trajectory_id"))
            out.append({**r, "labels": {**(r.get("labels") or {}), **{k: v["value"] for k, v in extra.items()}}} if extra else r)
        return out

    def _cache_path(self, path: Path) -> Path:
        return self.cache_dir / (hashlib.sha1(str(path.resolve()).encode()).hexdigest() + ".json")

    def get(self, run_id: str) -> dict[str, Any]:
        """{'status': 'ready', 'rows': [...]} or {'status': 'building', 'progress': {...}}."""
        path = run_file(self.data_root, run_id, self.extra_roots)
        with self._lock:
            if run_id in self._ready:
                return {"status": "ready", "rows": self._with_side(self._ready[run_id])}
            cached = self._load_cache(path)
            if cached is not None:
                self._ready[run_id] = cached
                return {"status": "ready", "rows": self._with_side(cached)}
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

    def find(self, text: str) -> dict[str, Any]:
        """Rows matching each id in `text` (see `parse_find_query`), over the runs named in it or else every run.
        Runs without an index are indexed one at a time in a background thread; `pending` counts them, so the
        caller polls until it is 0."""
        all_ids = [r["id"] for r in discover_runs(self.data_root, self.extra_roots)]
        scope, queries = parse_find_query(text, all_ids)
        matches: list[dict[str, Any]] = []
        missing, failed = [], []
        for run_id in scope or all_ids:
            path = run_file(self.data_root, run_id, self.extra_roots)
            with self._lock:
                rows = self._ready.get(run_id)
                if rows is None and (rows := self._load_cache(path)) is not None:
                    self._ready[run_id] = rows
                error = (self._progress.get(run_id) or {}).get("error")
            if error:
                failed.append(run_id)
                continue
            if rows is None:
                missing.append((run_id, path))
                continue
            for q in queries:
                matches.extend({"query": q, **r} for r in self._with_side(rows) if row_matches(r, q))
        if missing:
            self._queue_builds(missing)
        return {"queries": queries, "scope": scope, "matches": matches, "searched": len(scope or all_ids) - len(missing) - len(failed), "pending": len(missing), "failed": failed}

    def _queue_builds(self, items: list[tuple[str, Path]]) -> None:
        """Index runs sequentially in one worker thread (find over all runs must not start 200 builds at once)."""
        with self._lock:
            known = {r for r, _ in self._queue}
            self._queue.extend(it for it in items if it[0] not in known and it[0] not in self._progress)
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._drain, daemon=True)
            self._worker.start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                run_id, path = self._queue.pop(0)
                if run_id in self._ready or run_id in self._progress:
                    continue
                self._progress[run_id] = {"rows": 0, "bytes": 0, "total_bytes": path.stat().st_size, "error": None}
            self._build(run_id, path)

    def _build(self, run_id: str, path: Path) -> None:
        rows: list[dict[str, Any]] = []
        attempts: dict[str, int] = {}
        try:
            for n, offset, rec in iter_records(path):
                if path.name == cb.FILE_NAME and cb.is_cheatbench_record(rec):
                    attempts[rec["episode"]] = attempts.get(rec["episode"], 0) + 1
                    _annotate_cheatbench(rec, path, attempts[rec["episode"]])
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


def load_side_labels(data_root: Path) -> dict[str, dict[str, Any]]:
    """trajectory_id -> extra labels joined from side files: CheatBench judge verdicts and provisional reads."""
    out: dict[str, dict[str, Any]] = {}
    for table in (cb.provisional_labels(data_root), cb.judge_side_labels(cb.judge_records(data_root))):
        for tid, labels in table.items():
            out.setdefault(tid, {}).update(labels)
    return out


def load_judge_labels(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    """trajectory_id -> judge verdicts from every data/stage3/**/labels.jsonl and data/cheatbench_judge/."""
    out: dict[str, list[dict[str, Any]]] = cb.judge_entries(cb.judge_records(data_root))
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
            "rendered": bool(m.get("rendered")),
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
    """The record without token ids and rendered strings, for the raw-JSON tab. An ortho record is shown as
    stored (its own keys), minus `ids` and `items`."""
    if cb.is_cheatbench_record(rec):
        return cb.strip_bulk({k: v for k, v in rec.items() if not k.startswith("_viewer_")})
    out = dict(rec)
    if "ids" in out and "items" in out:
        out["ids"] = f"<{len(rec.get('ids') or [])} ids omitted>" if rec.get("ids") is not None else None
        out["items"] = f"<{len(rec.get('items') or [])} messages omitted; see the transcript tab>"
        return out
    out["rendered_text"] = f"<{len(rec.get('rendered_text') or [])} entries omitted>"
    out["turns"] = [
        {k: (f"<{len(v)} ids omitted>" if k in TOKEN_FIELDS and v is not None else v) for k, v in t.items() if k != "rendered_prompt"}
        for t in rec.get("turns") or []
    ]
    return out


_TOKENIZERS: dict[str, Any] = {}


def _decoder(model: str | None):
    """A token-id decoder for the record's model (the vLLM served name, `vllm/` prefix stripped), loaded
    once from the local HF cache; None when transformers or the tokenizer files are unavailable. Only used
    to recover the chat template's own system turn from ortho records, which store the sent messages."""
    if not model:
        return None
    name = model.removeprefix("vllm/")
    if name not in _TOKENIZERS:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(name)
            _TOKENIZERS[name] = lambda ids: tok.decode(ids, skip_special_tokens=False)
        except Exception:  # offline, no cache, or no transformers: the transcript just lacks the system turn
            _TOKENIZERS[name] = None
    return _TOKENIZERS[name]


def detail(rec: dict[str, Any], judge: dict[str, list[dict[str, Any]]], side_labels: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    raw = rec
    rec = _normalize(rec)
    extra = (side_labels or {}).get(rec.get("trajectory_id"))
    if extra:
        rec = {**rec, "labels": {**(rec.get("labels") or {}), **extra}}
    if is_ortho_record(raw):
        decode = _decoder(raw.get("model"))
        if decode is not None:
            rec = with_rendered_system_turn(rec, raw, decode)
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
        "raw": strip_bulk(raw),
    }
