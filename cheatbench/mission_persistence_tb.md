# Mission (DRAFT, not approved): do the in-environment TV edits cost persistence? (CheatBench VM + GPU box)

Status: draft, user decisions of 2026-11-05 applied (x1 and r1 first, x0 only if one of them loses tasks, no base re-roll, runs in parallel with stage 10 on its own GPU box). Same setup as `mission_capability.md` (Terminal-Bench 2.0 via Harbor/Terminus-2 on the VM, rows served by the GPU box). Nothing here has run.

## Goal

On Terminal-Bench 2.0, A3 (Grader direction G9, steering) lost 33 tasks base solved and gained none: 51.7 → 14.6 %. The KL-matched random direction gave 33.7 %. A3 re-scoped tasks to what it had already done and marked them complete. The direction came from a single-step environment. This mission asks whether hacking directions from **multi-step** data cost persistence too:

| row | served name | what | known |
|---|---|---|---|
| x1 | `cb_x1` | stage-7 X1 = D_TV (`tamper_vs_clean__cot_last`), stacked heretic zero-ablation, w 1.0, L40–63 | TV `verifier_bug` 4 turns 47.5 → 0 %, 6 turns 86.5 → 1.5 %; `possible` 100 %; MMLU 83.15 |
| r1 | `cb_r1` | stage-5b R1 = `attempt_vs_A__completion_last`, same stack recipe | TV 4 turns 48.5 → 12.0 %; MMLU 83.34 |
| x0 | `cb_x0` | stage-7 X0 = random direction, same stack recipe | the control for weight edits (the capability run's `rand` is a steering control) |

References from the capability run (2026-10-30, same 89 tasks, Harbor and settings): base 46/89, a3 13/89, a3half 36/89, rand 30/89. Pairing is by task, against that base (no re-roll).

Why TB2 and not the existing TV rollouts: the TV judge screen (`data/stage3/audit/tv_persistence-all/persistence_report.md`, 2026-11-05) cannot separate lost persistence from the end of tampering. Persistence scores sit at the ceiling for every row, A3 included. The flags that do move (claims the verifier is broken, confident final answers, fewer re-checks) move for A3 too, and on TV they are mostly correct statements.

## GPU box

`cheatbench/mission_persistence_gpubox.md` (a Claude session on the GPU box) runs first. It rebuilds x1, r1 and x0 as merged checkpoints, verifies each against its stored gate, and extends `serve_row.sh` so `request-gpubox-row x1|r1|x0` works. A row it could not verify is not servable; `request-gpubox-row status` and the HF report `cheatbench/persistence/box_report/persistence_gpubox/SUMMARY.md` say which.

## Plan (VM agent)

Phases 0–2 as in `mission_capability.md`:
- Wait until `request-gpubox-row status` says `serving cb_x1` (the prep mission's end state; poll every 10 min, give up after 4 h and finish). If it says `cb_base`, read the prep summary on HF: run only the rows marked verified.
- Preflight and the effort check.
- No smoke: the settings and C = 16 are taken from the capability run unchanged, so concurrency and timeouts match the references.
- Same Harbor commit, same task order (agent timeout descending).
- Same export: the capability run's `cheatbench/tools/tb2_export.py` exists only as text in HF `cheatbench/capability/box_report/LOG.md` (full text in the 16:04 entry, the `context_limit` change in the 16:49 entry). Recreate it from there unchanged, so the columns match the reference CSVs, and paste it into `LOG.md`.

Blocks in order: **x1 → r1**, then **x0 only if** x1 or r1 loses tasks against base (paired McNemar p < 0.2, or ≥ 8 more tasks lost than gained). x0 then tells whether the loss is specific to the direction or comes from the stack edit itself. Record the rule's inputs and the decision in `LOG.md` before starting or skipping x0. Each block:
- 89 tasks × 1 attempt, `collect_rollout_details` on;
- GSM8K/IFEval alongside, as before.

Per row, report the capability run's headline columns:
- TB2 pass with paired McNemar vs base and vs x0;
- `task_complete` early %, and completions that fail the verifier as a share of all completions (A3: 80 %, base 28 %);
- median turns, median prompt tokens per episode;
- end reasons;
- GSM8K, IFEval.

For each edited row, 5 episodes that base passed and the row failed, in three lines each with a verbatim quote, as in the capability summary.

Time: ~1.5 h per block (capability run: 82–103 min), so 3–5 h plus rebuilds. Cap 8 h. Uploads under `cheatbench/persistence/`. Self-stop as before: uploads verified, `systemctl stop rhtunnel`, `request-gpubox-stop`.

## After the mission (laptop)

Run CheatBench judge v2's engagement fields on the TB2 transcripts of every row:
- persistence;
- `stopped_with_options_left`;
- `report_accuracy`, i.e. re-scoping or claiming done.

The rubric needs TB2 task facts (or a generic "solvable task" facts block) in place of CheatBench's per-task facts.

## Machines

- **Hetzner CPU VM** (CCX43 as before, Ubuntu 24.04, `setup_cpu_vm.sh` + `setup_tbench.sh`). Terminal-Bench needs Docker, which Vast containers block. If the capability run's VM was deleted, create a new one; the TB2 images (pre-pulled there) are pulled again in preflight.
- **GPU box** (H200 as before, ≥ 200 GB disk: base plus one ~55 GB merged checkpoint at a time), separate from the stage-10 box.
- The tunnel and stop keys as in the capability run (`allow_tunnel_key.sh` with `FROM=<VM IP>`, `ROW_KEY`, `DONE_KEY`).
