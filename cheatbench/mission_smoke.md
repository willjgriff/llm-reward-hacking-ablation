# Mission: CheatBench smoke pass on Qwen3.8-27B (CheatBench VM)

Goal: find out which CheatBench tasks run end to end against our vLLM-served `Qwen/Qwen3.8-27B` at reasoning
effort low, and what one episode of each costs, so the user can choose tasks and sizes for a collection run.
No collection, no judge.

## Steps
1. **Preflight.** `systemctl status rhtunnel`; `curl -s localhost:8000/v1/models` must list `Qwen/Qwen3.8-27B`
   (anything else, e.g. an ablated checkpoint: record and stop). `df -h /`. Read `CONFIGURATION.md` and
   `README.md` of `/home/cheatbench/cheatbench` and each task's `README`/config before running it.
2. **Images.** Build every sandbox image the tasks below need, up front (`core.sandbox --build`, task builds),
   so no run waits on a build.
3. **One episode per task**, all tasks under `tasks/` except `software_engineering` (Docker) and any whose
   config needs a paid API or a credential at episode time (record and skip):
   `run.py <task> --model qwen3.8-27b-low --models-config configs/models.local.yaml --limit 1 --no-judge
   --max-concurrent 1 --run-id smoke-<task>`. Use the task's default variant; if it has a hack-cueing and a
   neutral variant, run one of each. On failure, read the error; fix only our side (model entry, `.env`,
   limits) and retry once; otherwise record and move on.
4. **Per run, record in `SUMMARY.md`:** status, turns, wall time, prompt/completion tokens (vLLM `/metrics` or
   the transcript), whether the episode hit `--max-turns` or the context limit, CheatBench's own result
   fields, and in one line what the model did (read `trajectory.json`; quote any sign of gaming the grader,
   as a note only, not a label).
5. **Upload** each run and the report as the rules say, verify, then finish (SUMMARY "What I would do next":
   which tasks look worth a collection run, at what concurrency, and the projected episodes/hour).

Time cap: 4 hours. If the cap is reached, finish the current episode, upload, and finish.
