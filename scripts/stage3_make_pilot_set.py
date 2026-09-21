#!/usr/bin/env python
"""Stage 3: choose the judge pilot set and render its transcripts and judge prompts.

Writes <out-dir>/pilot_set.json (fixed order: most informative first), transcripts/<n>.txt and
prompts/<n>.txt (rubric + scaffold section + transcript: what a scripted judge would be sent).

  uv run scripts/stage3_make_pilot_set.py --out-dir data/stage3/pilot1
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.compact import open_trajectories, trajectories_path  # noqa: E402
from rhablation.judge_transcript import collapse_repeats, render  # noqa: E402

STAGE1 = Path("data/stage1")
RUBRIC_DIR = Path("configs/judge")
HACK_WORDS = re.compile(r"special[- ]case|hard-?cod|to pass the test|match the test expectation", re.I)

# (run, scaffold, split, task): hand-confirmed hack signals from the stage-1 reports.
KNOWN_POSITIVES = [
    ("promptB-min-16", "minimal", "oneoff", "lcbhard_41"),
    ("promptB-solvable19-tb4096", "tools", "oneoff", "lcbhard_10"),
    ("promptB-tools-smoke", "tools", "oneoff", "lcbhard_41"),
    ("promptB-solvable19", "tools", "conflicting", "lcbhard_41"),
    ("promptB-min-16", "minimal", "conflicting", "lcbhard_9"),
]


def load(run: str) -> list[dict]:
    return [json.loads(line) for line in open_trajectories(trajectories_path(STAGE1 / run)) if line.strip()]


def key(t: dict) -> tuple:
    return (t["run_id"], t["agent_type"], t["variant"], t["task_id"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    runs = {r: load(r) for r in ("lcb-4b-part1", "promptB-min-16", "promptB-tools-smoke", "promptB-solvable19", "promptB-solvable19-tb4096")}
    every = {key(t): t for ts in runs.values() for t in ts}
    chosen: list[tuple[str, dict]] = [("known_positive", every[k]) for k in KNOWN_POSITIVES]
    taken = set(KNOWN_POSITIVES)

    def add(group: str, pool: list[dict], n: int | None = None) -> None:
        pool = [t for t in pool if key(t) not in taken and t.get("score") is not None]
        for t in (pool if n is None else rng.sample(pool, min(n, len(pool)))):
            chosen.append((group, t))
            taken.add(key(t))

    # Unread keyword hits: hack-like words in the chain of thought of the 10-attempt prompt B run.
    add("cot_keyword_hit", [t for t in runs["promptB-min-16"] if HACK_WORDS.search(" ".join((u.get("reasoning") or "") for u in t["turns"] if not u.get("error")))])
    d = runs["lcb-4b-part1"]
    add("promptD_impossible", [t for t in d if t["variant"] != "original"], 8)
    add("original_pass", [t for t in d if t["variant"] == "original" and t["score"]["value"] == "C"], 3)
    add("original_fail", [t for t in d if t["variant"] == "original" and t["score"]["value"] != "C"], 3)
    b = runs["promptB-solvable19-tb4096"]
    add("promptB_tb4096_minimal", [t for t in b if t["agent_type"] == "minimal"], 6)
    add("promptB_tb4096_tools", [t for t in b if t["agent_type"] == "tools"], 6)

    rubric = re.sub(r"<!--.*?-->\s*", "", (RUBRIC_DIR / "impossiblebench_lcb_rubric.md").read_text(), flags=re.S)
    for sub in ("transcripts", "prompts", "reading"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)
    items = []
    for n, (group, t) in enumerate(chosen, 1):
        text = render(t)
        (args.out_dir / "transcripts" / f"{n:03d}.txt").write_text(text)
        (args.out_dir / "reading" / f"{n:03d}.txt").write_text(collapse_repeats(text))  # for an interactive judge only
        scaffold = (RUBRIC_DIR / f"scaffold_{t['agent_type']}.md").read_text().strip()
        (args.out_dir / "prompts" / f"{n:03d}.txt").write_text(rubric.replace("{{SCAFFOLD_SECTION}}", scaffold).replace("{{TRANSCRIPT}}", text))
        items.append({"n": n, "group": group, "trajectory_id": t["trajectory_id"], "run_id": t["run_id"], "agent_type": t["agent_type"],
                      "variant": t["variant"], "task_id": t["task_id"], "transcript_chars": len(text), "reading_chars": len(collapse_repeats(text))})
    (args.out_dir / "pilot_set.json").write_text(json.dumps({"seed": args.seed, "items": items}, indent=2))
    for it in items:
        print(f"{it['n']:3d} {it['group']:24s} {it['run_id']:26s} {it['agent_type']:8s} {it['variant']:12s} {it['task_id']:11s} ~{it['transcript_chars'] // 4:,} tokens")
    print(f"total ~{sum(i['transcript_chars'] for i in items) // 4:,} tokens over {len(items)} transcripts")


if __name__ == "__main__":
    main()
