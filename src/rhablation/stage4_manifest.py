"""Stage 4 manifest: which trajectories go into the activation cache, with class, task split and order.

Reads the sweep run directories (schema 1.3 compact records), assigns every trajectory a class from the
config's rules, splits task IDs into train/eval (the same split for every variant, since all variants
share one task pool), and fixes the processing order: round-robin over the priority classes so that a
balanced `subset_per_class` subset is complete early, while the full set keeps going.
"""

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from rhablation.compact import expand_trajectory, open_trajectories, trajectories_path
from rhablation.schema import Trajectory
from rhablation.stage4_config import ClassRule, Stage4Config
from rhablation.stage4_masks import LayoutError, Spans, token_spans


class PrefixError(ValueError):
    pass


@dataclass
class ManifestRow:
    order: int
    trajectory_id: str
    run_id: str
    source_file: str
    line_no: int
    variant: str
    max_turns: int | None
    epoch: int
    cls: str
    task_id: str
    split: str
    status: str
    end_reason: str | None
    passed: bool | None
    tampered: bool | None
    peeked: bool | None
    used_verifier: bool | None
    submitted_true_answer: bool | None
    n_calls: int
    seq_len: int
    n_cot_tokens: int
    n_answer_tokens: int
    n_calls_without_think_close: int
    n_calls_multi_think_close: int
    in_subset: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("cls")
        return d

    @staticmethod
    def from_dict(d: dict) -> "ManifestRow":
        d = dict(d)
        d["cls"] = d.pop("class")
        return ManifestRow(**d)


def iter_sweep_records(
    sweep_dirs: list[Path], exclude_run_ids: list[str], root: Path
) -> Iterator[tuple[str, int, Trajectory]]:
    """Yield (source_file relative to root, line number, record) for every trajectory under the sweep dirs."""
    for sweep_dir in sweep_dirs:
        sweep_dir = Path(sweep_dir)
        if not sweep_dir.exists():
            raise FileNotFoundError(sweep_dir)
        run_dirs = sorted(p for p in sweep_dir.iterdir() if p.is_dir())
        if not run_dirs:
            raise FileNotFoundError(f"no run directories under {sweep_dir}")
        for run_dir in run_dirs:
            if run_dir.name in exclude_run_ids:
                continue
            path = trajectories_path(run_dir)
            if not path.exists():
                continue
            rel = str(path.resolve().relative_to(root.resolve())) if path.resolve().is_relative_to(root.resolve()) else str(path)
            with open_trajectories(path) as f:
                for line_no, line in enumerate(f, 1):
                    if line.strip():
                        yield rel, line_no, Trajectory.model_validate_json(line)


def label_value(record: Trajectory, name: str):
    lab = record.labels.get(name)
    return None if lab is None else lab.value


def max_turns_of(record: Trajectory) -> int | None:
    opts = record.env_options or {}
    v = opts.get("max_turns")
    return int(v) if v is not None else None


def assign_class(record: Trajectory, rules: list[ClassRule]) -> str | None:
    mt = max_turns_of(record)
    passed = label_value(record, "passed")
    tampered = label_value(record, "tampered")
    for rule in rules:
        if rule.variant != record.variant:
            continue
        if rule.max_turns is not None and rule.max_turns != mt:
            continue
        if rule.programmatic_hack is not None and rule.programmatic_hack != record.programmatic_hack:
            continue
        if rule.passed is not None and rule.passed != passed:
            continue
        if rule.tampered is not None and rule.tampered != tampered:
            continue
        return rule.name
    return None


def final_sequence(record: Trajectory) -> tuple[list[int], list[tuple[int, int]]]:
    """The final call's prompt + completion, and (prompt_len, completion_len) per call in that sequence.

    Asserts the prefix property: every call's prompt is the previous call's prompt followed by the previous
    completion verbatim. Raises PrefixError otherwise (then one forward per trajectory would be wrong)."""
    rec = expand_trajectory(record)
    turns = [t for t in rec.turns if t.prompt_token_ids is not None]
    if not turns:
        raise PrefixError(f"{record.trajectory_id}: no turns with token ids")
    lengths: list[tuple[int, int]] = []
    prev_prompt: list[int] | None = None
    prev_completion: list[int] | None = None
    for t in turns:
        if t.completion_token_ids is None:
            raise PrefixError(f"{record.trajectory_id} turn {t.turn_index}: missing completion_token_ids")
        if t.error:
            raise PrefixError(f"{record.trajectory_id} turn {t.turn_index}: call error {t.error!r}")
        prompt = t.prompt_token_ids
        if prev_prompt is not None:
            n, m = len(prev_prompt), len(prev_completion or [])
            if prompt[:n] != prev_prompt or prompt[n : n + m] != prev_completion:
                raise PrefixError(
                    f"{record.trajectory_id} turn {t.turn_index}: prompt is not previous prompt + previous completion"
                )
        lengths.append((len(prompt), len(t.completion_token_ids)))
        prev_prompt, prev_completion = prompt, t.completion_token_ids
    ids = list(prev_prompt) + list(prev_completion)  # type: ignore[arg-type]
    return ids, lengths


def spans_for(record: Trajectory, cfg: Stage4Config) -> tuple[list[int], Spans]:
    ids, lengths = final_sequence(record)
    spans = token_spans(ids, lengths, cfg.think_open_id, cfg.think_close_id, cfg.im_end_id, cfg.newline_id)
    return ids, spans


def task_split(task_ids: list[str], eval_fraction: float, seed: int) -> dict[str, str]:
    ids = sorted(set(task_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_eval = int(round(len(ids) * eval_fraction))
    eval_ids = set(ids[:n_eval])
    return {t: ("eval" if t in eval_ids else "train") for t in ids}


def processing_order(rows: list[ManifestRow], priority_classes: list[str], subset_per_class: int, seed: int) -> list[ManifestRow]:
    """Round-robin over classes (priority ones first, then the rest alphabetically), each class in a seeded
    random order; `hack_vb`-style classes with several max_turns alternate between them. Sets `order` and
    `in_subset` (first `subset_per_class` rows of each class)."""
    rng = random.Random(seed)
    by_class: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        by_class[r.cls].append(r)
    queues: dict[str, list[ManifestRow]] = {}
    for cls, members in by_class.items():
        by_mt: dict[int | None, list[ManifestRow]] = defaultdict(list)
        for r in members:
            by_mt[r.max_turns].append(r)
        for group in by_mt.values():
            group.sort(key=lambda r: r.trajectory_id)
            rng.shuffle(group)
        groups = [by_mt[k] for k in sorted(by_mt, key=lambda k: (k is None, k))]
        merged: list[ManifestRow] = []
        idx = [0] * len(groups)
        while any(i < len(g) for i, g in zip(idx, groups)):
            for gi, g in enumerate(groups):
                if idx[gi] < len(g):
                    merged.append(g[idx[gi]])
                    idx[gi] += 1
        queues[cls] = merged
    class_order = [c for c in priority_classes if c in queues] + sorted(c for c in queues if c not in priority_classes)
    ordered: list[ManifestRow] = []
    taken = {c: 0 for c in class_order}
    while any(taken[c] < len(queues[c]) for c in class_order):
        for c in class_order:
            if taken[c] < len(queues[c]):
                r = queues[c][taken[c]]
                r.in_subset = taken[c] < subset_per_class
                taken[c] += 1
                ordered.append(r)
    for i, r in enumerate(ordered):
        r.order = i
    return ordered


@dataclass
class BuildReport:
    n_records: int = 0
    n_kept: int = 0
    excluded: dict[str, int] | None = None
    prefix_errors: list[str] | None = None
    layout_errors: list[str] | None = None


def build_rows(cfg: Stage4Config, root: Path, limit: int | None = None) -> tuple[list[ManifestRow], BuildReport]:
    report = BuildReport(excluded=defaultdict(int), prefix_errors=[], layout_errors=[])
    rows: list[ManifestRow] = []
    for source_file, line_no, rec in iter_sweep_records(cfg.data.sweep_dirs, cfg.data.exclude_run_ids, root):
        report.n_records += 1
        cls = assign_class(rec, cfg.classes)
        if cls is None:
            report.excluded["no class rule"] += 1
            continue
        try:
            ids, spans = spans_for(rec, cfg)
        except PrefixError as e:
            report.prefix_errors.append(str(e))
            report.excluded["prefix property violated"] += 1
            continue
        except LayoutError as e:
            report.layout_errors.append(f"{rec.trajectory_id}: {e}")
            report.excluded["token layout"] += 1
            continue
        if len(ids) > cfg.cache.max_seq_len:
            report.excluded[f"longer than max_seq_len={cfg.cache.max_seq_len}"] += 1
            continue
        rows.append(
            ManifestRow(
                order=-1,
                trajectory_id=rec.trajectory_id,
                run_id=rec.run_id,
                source_file=source_file,
                line_no=line_no,
                variant=rec.variant,
                max_turns=max_turns_of(rec),
                epoch=rec.epoch,
                cls=cls,
                task_id=rec.task_id,
                split="",
                status=rec.status,
                end_reason=label_value(rec, "end_reason"),
                passed=label_value(rec, "passed"),
                tampered=label_value(rec, "tampered"),
                peeked=label_value(rec, "peeked"),
                used_verifier=label_value(rec, "used_verifier"),
                submitted_true_answer=label_value(rec, "submitted_true_answer"),
                n_calls=spans.n_calls,
                seq_len=spans.seq_len,
                n_cot_tokens=spans.n_cot_tokens(),
                n_answer_tokens=spans.n_answer_tokens(),
                n_calls_without_think_close=spans.n_calls_without_think_close(),
                n_calls_multi_think_close=spans.n_calls_multi_think_close(),
                in_subset=False,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    split = task_split([r.task_id for r in rows], cfg.split.eval_fraction, cfg.split.seed)
    for r in rows:
        r.split = split[r.task_id]
    rows = processing_order(rows, cfg.order.priority_classes, cfg.order.subset_per_class, cfg.order.seed)
    report.n_kept = len(rows)
    report.excluded = dict(report.excluded)
    return rows, report


def read_manifest(path: Path) -> list[ManifestRow]:
    import json

    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(ManifestRow.from_dict(json.loads(line)))
    rows.sort(key=lambda r: r.order)
    return rows


def load_record(root: Path, row: ManifestRow) -> Trajectory:
    """Re-read one trajectory by source file and line number (sequential readers should use their own loop)."""
    path = root / row.source_file
    with open_trajectories(path) as f:
        for line_no, line in enumerate(f, 1):
            if line_no == row.line_no:
                rec = Trajectory.model_validate_json(line)
                if rec.trajectory_id != row.trajectory_id:
                    raise ValueError(f"{path}:{line_no} holds {rec.trajectory_id}, manifest says {row.trajectory_id}")
                return rec
    raise ValueError(f"{path}: no line {row.line_no}")
