# external/ortho: Ethan Hadley's `ortho` repo, environments only

Source: https://github.com/ekhadley/ortho, `main` at commit `1a93e987` (pushed 2026-10-04), copied from the
local checkout `../ortho` on 2026-10-07 with
`rsync -a --exclude __pycache__ --exclude docker ../ortho/envs/ external/ortho/envs/` plus `README.md`,
`pyproject.toml` and `uv.lock`. `envs/docker/` (secret_number and impossible_bench sandboxes) is left out:
the box cannot run docker and this tree only uses the `grader` task.

Rules: **never edit anything under `external/ortho/`**; adapt from outside (`sys.path` insert of
`external/ortho/envs`, our own runner). Refresh by re-running the copy and updating this file. The
rollout records these tasks produce are read by `src/rhablation/ortho_records.py`.

Not copied: `run.py`, `grader_lens.py`, `utils.py` (need his `mechtools` git dependency), `data/`,
`logs/`, `notes/`, `scripts/`.
