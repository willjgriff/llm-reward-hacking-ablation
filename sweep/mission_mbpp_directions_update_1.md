# Mission update 1 for `sweep/mission_mbpp_directions.md` (user, 2026-10-02, before the start)

Everything in the mission stays as it is. This adds one requirement: **record how long things take, in the
data, not only in the log.**

## What to record
1. **`box_report/mbpp_directions/timings.json`**, machine-readable, updated after every step and uploaded with
   the report: one entry per step and sub-step with `name`, `started` / `ended` (ISO UTC), `wall_s`,
   `gpu` (true/false), and the natural size measure (`tokens`, `rows`, `rollouts`, `configs`), so that rates
   can be derived (tok/s for the caches, rows/s for the likelihood grid, rollouts/h for the evals). Steps:
   restore (per item: code, data, each env, weights), manifest, MBPP cache, TV subset cache, directions
   (and the curves separately), each likelihood configuration, each heretic edit and export, each gate, the
   R1 merge and verification, each rollout run, MMLU, uploads, and the total.
2. **Inside each stage's own outputs**: `results.json` of the directions run gets a `timing` block (CPU wall
   time, per contrast if easy); every likelihood `*.json` records its wall time and rows scored per second;
   every rollout `run_config.json` / `chunk_summary.json` keeps the per-rollout seconds and rollouts/h the
   runners already write; `data/stage5d/adapters/<tag>/manifest.json` records the edit's wall time and the
   export's; the MMLU output directory gets a `timing.json`.
3. **`SUMMARY.md`**: a timing table (step, wall time, GPU or CPU, size, rate) next to the results, plus the
   sum of GPU hours and the total wall time from session start to the stop command.

Use `time.monotonic()` around the calls in the scripts you write, and `date -u` in the shell for runs you
launch by command; do not estimate afterwards from log timestamps when a measured number is available.
