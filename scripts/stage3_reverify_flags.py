#!/usr/bin/env python
"""Stage 3 audit: re-run the flag verification over an existing judge output without new API calls.

Every record's flags (kept and previously discarded) are re-checked with the current matcher
(rhablation.judge_audit.split_flags: exact / loose / fuzzy quote match, marked on each kept flag), and
strongest_status / categories are recomputed. The previous file is kept as labels.jsonl.bak-<utc>.

  uv run scripts/stage3_reverify_flags.py --audit-dir data/stage3/audit/mbpp_impossible [--judge <dir name>]
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.judge_audit import SPECS, load_manifest_classes, split_flags  # noqa: E402
from rhablation.judge_transcript import STATUSES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--judge", help="judge directory name (default: every judge under the audit dir)")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--manifest", type=Path, default=Path("data/stage4/manifest/tv27b/manifest.jsonl"))
    args = p.parse_args()

    meta = json.loads((args.audit_dir / "pilot_set.json").read_text())
    spec = SPECS[meta["env"]]
    if meta["env"] == "terminal_verifier":
        spec.context["manifest_classes"] = load_manifest_classes(args.manifest)
    items = {it["trajectory_id"]: it for it in meta["items"]}
    judge_dirs = [args.audit_dir / "judges" / args.judge] if args.judge else sorted(d for d in (args.audit_dir / "judges").glob("*") if (d / "labels.jsonl").exists())

    # Load every trajectory the judge files mention, once.
    needed: dict[str, set[int]] = {}
    for d in judge_dirs:
        for line in (d / "labels.jsonl").open():
            if line.strip():
                it = items.get(json.loads(line)["trajectory_id"])
                if it:
                    needed.setdefault(it["source_file"], set()).add(it["line_no"])
    trajs: dict[str, dict] = {}
    for sf, ln, traj in spec.iter_records(args.data_root):
        if ln in needed.get(str(sf), ()):
            trajs[traj["trajectory_id"]] = traj

    for d in judge_dirs:
        path = d / "labels.jsonl"
        records = [json.loads(line) for line in path.open() if line.strip()]
        before = Counter(r.get("strongest_status") or "none" for r in records)
        how = Counter()
        out = []
        for r in records:
            traj = trajs.get(r["trajectory_id"])
            if traj is None:
                out.append(r)
                continue
            raw = list(r.get("flags") or []) + [f["flag"] for f in (r.get("discarded_flags") or []) if isinstance(f.get("flag"), dict)]
            raw = [{k: v for k, v in f.items() if k != "quote_match"} for f in raw]
            kept, discarded = split_flags(spec, traj, raw)
            how.update(f["quote_match"] for f in kept)
            how["discarded"] += len(discarded)
            r.update({"flags": kept, "discarded_flags": discarded,
                      "strongest_status": max((f["status"] for f in kept), key=STATUSES.index, default=None),
                      "categories": sorted({f["category"] for f in kept}), "reverified_at": datetime.now(timezone.utc).isoformat()})
            out.append(r)
        after = Counter(r.get("strongest_status") or "none" for r in out)
        backup = path.with_name(f"labels.jsonl.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        path.rename(backup)
        with path.open("w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"{d.name}: {len(out)} records; quote matches {dict(how)}; strongest status before {dict(before)} -> after {dict(after)}; backup {backup.name}")


if __name__ == "__main__":
    main()
