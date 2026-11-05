# Mission (DRAFT, not approved): do the in-environment TV edits cost persistence? (CheatBench VM + GPU box)

Status: draft for the user, 2026-11-05. Same setup as `mission_capability.md` (Terminal-Bench 2.0 via Harbor/Terminus-2 on the VM, rows served by the GPU box). Nothing here has run.

## Goal

On Terminal-Bench 2.0, A3 (Grader direction G9, steering) lost 33 tasks base solved and gained none: 51.7 → 14.6 %. The KL-matched random direction gave 33.7 %. A3 re-scoped tasks to what it had already done and marked them complete. The direction came from a single-step environment. This mission asks whether hacking directions from **multi-step** data cost persistence too:

| row | served name | what | known |
|---|---|---|---|
| x1 | `cb_x1` | stage-7 X1 = D_TV (`tamper_vs_clean__cot_last`), stacked heretic zero-ablation, w 1.0, L40–63 | TV `verifier_bug` 4 turns 47.5 → 0 %, 6 turns 86.5 → 1.5 %; `possible` 100 %; MMLU 83.15 |
| r1 | `cb_r1` | stage-5b R1 = `attempt_vs_A__completion_last`, same stack recipe | TV 4 turns 48.5 → 12.0 %; MMLU 83.34 |
| x0 | `cb_x0` | stage-7 X0 = random direction, same stack recipe | the control for weight edits (the capability run's `rand` is a steering control) |
| s10 | `cb_s10_before` | stage-10 decision-sentence direction, A3 recipe (steering), if `mission_stage10_positions.md` has produced it | – |
| base2 | `cb_base` | base re-roll, optional | measures base-vs-base discordance on TB2 |

References from the capability run (2026-10-30, same 89 tasks, Harbor and settings): base 46/89, a3 13/89, a3half 36/89, rand 30/89. Pairing is by task. Rows are compared with base and with x0. The steering row s10 is compared with the capability run's `rand`.

The same TV judge screen on existing rollouts (`data/stage3/audit/tv_persistence-all/persistence_report.md`) is read next to this. TV can barely show giving up (episodes run to the turn limit), so TB2 is the measure that counts.

## GPU box changes needed before launch (laptop work, after approval)

- **`gpubox_setup.sh`:** build merged checkpoints for x1/r1/x0.
  - Re-run the recorded stacks: stage-7 matrix build code from the code mirror, and the direction files from HF `stage4/stage7/s7-tv`, `stage4/tv27b-full` and stage 7's random seed.
  - Add the MTP weights (`add_mtp_weights.py`), as stage 7 did.
  - **Verify** each against its stored gate: X1/X0 vs `stage7/rows/<id>/gate/gate.json`, R1 vs the stage-5c scores to ≤ 0.001 nats (the stage-5d check).
  - The stored adapters are not used: for stacks they hold only the last round.
- **Disk:** each merged checkpoint is ~55 GB, so build one row at a time. `serve_row.sh` builds the requested row if it is missing and deletes the previous merged row first. A build takes ~2 min plus verification.
- **`serve_row.sh`:**
  - accept `x1|r1|x0|s10` (merged rows served without the bias plugin; `s10` is bias-only like `cb_a3`);
  - accept the same names on the forced-command row key.

## Plan (VM agent)

Phases 0–2 as in `mission_capability.md`:
- Preflight and the effort check.
- No smoke: the settings and C = 16 are taken from the capability run unchanged, so concurrency and timeouts match the references.
- Same Harbor commit, same task order (agent timeout descending).

Blocks in order: **x1 → r1 → x0 → s10 (if ready) → base2 (if time)**. Each block:
- 89 tasks × 1 attempt, `collect_rollout_details` on;
- GSM8K/IFEval alongside, as before.

Per row, report the capability run's headline columns:
- TB2 pass with paired McNemar vs base and vs x0;
- `task_complete` early %, and completions that fail the verifier as a share of all completions (A3: 80 %, base 28 %);
- median turns, median prompt tokens per episode;
- end reasons;
- GSM8K, IFEval.

For each edited row, 5 episodes that base passed and the row failed, in three lines each with a verbatim quote, as in the capability summary.

Time: ~1.5 h per block (capability run: 82–103 min), so x1 + r1 + x0 ≈ 5 h plus rebuilds. Cap 10 h. Self-stop as before: uploads verified, `systemctl stop rhtunnel`, `request-gpubox-stop`.

## After the mission (laptop)

Run CheatBench judge v2's engagement fields on the TB2 transcripts of every row:
- persistence;
- `stopped_with_options_left`;
- `report_accuracy`, i.e. re-scoping or claiming done.

The rubric needs TB2 task facts (or a generic "solvable task" facts block) in place of CheatBench's per-task facts.

## Open questions for the user

1. Include x0? It is the right zero for weight edits but costs a 1.5 h block. Without it, x1/r1 are read only against base and the steering `rand`.
2. Include base2? It shows how much two base runs disagree at n = 89, one attempt each.
3. Two attempts per task for x1 and base instead of x0/base2? That gives more power on the main question for the same time.
