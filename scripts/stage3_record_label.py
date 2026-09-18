#!/usr/bin/env python
"""Stage 3: append one judged trajectory to <pilot-dir>/labels.jsonl, after checking every flag's
quote against the passage it names. Appends and flushes one record at a time, so an interrupted
labelling run loses nothing already recorded; a trajectory already in labels.jsonl is refused.

  uv run scripts/stage3_record_label.py --pilot-dir data/stage3/pilot1 --n 3 --judge <id> --label-file <json>
  uv run scripts/stage3_record_label.py --pilot-dir data/stage3/pilot1 --status      # what is done / next

The label file holds {"flags": [{step, channel, category, status, quote, why}, ...], "rationale": "..."}.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.judge_transcript import CATEGORIES, CHANNELS, STATUSES, check_flags  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot-dir", type=Path, required=True)
    p.add_argument("--n", type=int)
    p.add_argument("--judge")
    p.add_argument("--label-file", type=Path)
    p.add_argument("--status", action="store_true")
    args = p.parse_args()

    items = json.loads((args.pilot_dir / "pilot_set.json").read_text())["items"]
    labels_path = args.pilot_dir / "labels.jsonl"
    done = {json.loads(line)["trajectory_id"] for line in labels_path.open()} if labels_path.exists() else set()
    if args.status:
        todo = [i["n"] for i in items if i["trajectory_id"] not in done]
        print(f"{len(done)} of {len(items)} labelled; next: {todo[:5]}")
        return

    item = next(i for i in items if i["n"] == args.n)
    if item["trajectory_id"] in done:
        sys.exit(f"already labelled: {item['trajectory_id']}")
    label = json.loads(args.label_file.read_text())
    flags = label["flags"]
    for f in flags:
        assert f["category"] in CATEGORIES, f"unknown category {f['category']}"
        assert f["status"] in STATUSES, f"unknown status {f['status']}"
        assert f["channel"] in CHANNELS, f"unknown channel {f['channel']}"
    traj = next(
        t for line in (Path("data/stage1") / item["run_id"] / "trajectories.jsonl").open()
        if (t := json.loads(line))["trajectory_id"] == item["trajectory_id"]
    )
    problems = check_flags(traj, flags)
    if problems:
        sys.exit("NOT RECORDED, fix the quotes:\n  " + "\n  ".join(problems))
    strongest = max((f["status"] for f in flags), key=STATUSES.index, default=None)
    record = {
        "trajectory_id": item["trajectory_id"], "n": item["n"], "group": item["group"], "judge": args.judge,
        "judged_at": datetime.now(timezone.utc).isoformat(), "rubric": "configs/judge/impossiblebench_lcb_rubric.md",
        "strongest_status": strongest, "categories": sorted({f["category"] for f in flags}),
        "flags": flags, "rationale": label.get("rationale", ""),
    }
    with labels_path.open("a") as out:
        out.write(json.dumps(record) + "\n")
        out.flush()
    print(f"recorded n={item['n']} {item['trajectory_id']}: strongest={strongest} categories={record['categories']} ({len(done) + 1} of {len(items)})")


if __name__ == "__main__":
    main()
