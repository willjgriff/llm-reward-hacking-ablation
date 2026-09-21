#!/usr/bin/env python
"""Stage 1: rewrite a run's trajectories.jsonl in the compact form (schema 1.2, see rhablation.compact).

Writes trajectories.compact.jsonl[.gz] next to the original and checks that every record expands back to
the same prompt ids. The original file is never changed or deleted; replace it yourself once satisfied.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.compact import compact_trajectory, expand_trajectory, is_compact, open_trajectories, trajectories_path  # noqa: E402
from rhablation.schema import SCHEMA_VERSION, Trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="data/stage1/<run_id>")
    parser.add_argument("--gzip", action="store_true", help="write trajectories.compact.jsonl.gz")
    args = parser.parse_args()

    source = trajectories_path(args.run_dir)
    output = args.run_dir / ("trajectories.compact.jsonl.gz" if args.gzip else "trajectories.compact.jsonl")
    n = 0
    with open_trajectories(source) as fin, open_trajectories(output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = Trajectory.model_validate_json(line)
            if not is_compact(record):
                compact = compact_trajectory(record)
                if [t.prompt_token_ids for t in expand_trajectory(compact).turns] != [t.prompt_token_ids for t in record.turns]:
                    raise RuntimeError(f"{record.trajectory_id}: compact form does not restore the prompt ids")
                record = compact
            record.schema_version = SCHEMA_VERSION
            fout.write(record.model_dump_json() + "\n")
            n += 1
    print(f"{source} ({source.stat().st_size / 1e6:.1f} MB) -> {output} ({output.stat().st_size / 1e6:.1f} MB), {n} trajectories")


if __name__ == "__main__":
    main()
