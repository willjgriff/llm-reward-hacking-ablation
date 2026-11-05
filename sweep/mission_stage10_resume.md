# Mission stage 10, resume: phase 4 rows, the rest of phase 3, report

Status: approved for launch 2026-11-06. The spec is still `sweep/mission_stage10_positions.md`: read it first. This file only says what is already done, what is left, and what changes because the work moves to a new box. Rules: `sweep/CLAUDE.unattended.md`.

## What happened

The first stage-10 box (H100, Vast 59286508) stopped answering SSH after 21:55 UTC on 2026-11-05. It was about halfway through phase 4. Its disk is not available. Everything below that is marked "on HF" is the complete record of that box; anything it did after 21:55 is lost.

## Done (on HF, `willjgriff/reward-hacking-ablation` @ `f7f5897`; do not redo)

- **Phase 1, forks:** `stage10/forks/` (pilot + `c001`–`c006`, 1,127 prefixes × 8 forks; 432 eligible mixed prefixes), `stage10/forks/exclude_task_ids.json`.
- **Decision-sentence rule v4** and its hand reading: `stage10/positions/v4/`, `stage10/reading/`, `sweep/box_report/stage10/rule_reading.md`.
- **Phase 2, activations:** `stage10/acts/` (82.5 GB; `fixed/{before_decision,decision_last,cot_last}.f16` at rows 0–64, sentence boundaries at rows 20–64). Download only what a step needs: no remaining step needs the boundary shards.
- **Phase 3, CPU part:** `stage10/directions/` (three directions, `results.md`, `splits.json`), `stage10/boundary_profile/`, `stage10/misc/alpha_calibration.json` (α_1: `before_decision` 1.4155, `cot_last` 2.3933, random seed 1 7.0038, G9 3.5328).
- **Phase 4, `S10_rand_steer`:** rollouts `stage6/S10_rand_steer/rhenvs/`, row files `stage10/rows/S10_rand_steer/`, metrics `stage10/metrics/S10_rand_steer.json`, MMLU 82.57. Its serving check: `stage10/bias_check/`.
- **Report so far:** `sweep/box_report/stage10/{LOG.md,SUMMARY.md}`. Continue both files; do not start new ones.
- **Code:** the stage-10 scripts exist only in the HF code mirror (`sweep/code/` @ `f7f5897`): `configs/stage10_rows.yaml`, `scripts/stage10_*.py`, `sweep/stage10_*.sh`, and the stage-9 scripts they call (`scripts/stage9_rows.py` etc.). Port them from there unchanged. Record in `LOG.md` any edit you have to make (paths, box differences) and why.

## Left to do, in this order

1. **Setup (≤ 1.5 h).** Reproduce the first box's environment, not a newer one:
   - vLLM **0.33.0** (the version `S10_rand_steer` ran under; stage 9 used 0.32.0), the same serve flags as `configs/stage10_rows.yaml` (MTP, 32,768 context, `max_num_seqs` 96);
   - rh-envs at the fork commit `2a524c1` in `external/rh-envs/`, as the first box did (its LOG has the steps);
   - the torch env, lm-eval, `Qwen/Qwen3.8-27B` @ `1d4bf0f`;
   - from HF: the directions, `alpha_calibration.json`, the `S10_rand_steer` row files, the forks, `stage10/positions/v4/`, the stage-9 id files and `S10ref` base copies (`stage6/B2_base`, `base_s9`) that the row config points to.
   **Harness check before any row:** rebuild `S10_rand_steer` from its stored recipe (same α, seed, window), and repeat its bias check against `stage10/bias_check/reference.json`. It must pass as it did on the first box (probability-weighted |Δ log p| ≈ 0.002 to the HF steering reference). Also confirm that the rebuilt bias shard is identical to `stage10/rows/S10_rand_steer/model-resid-bias.safetensors`. If either check fails, stop and report: the random row would then not be a valid control for rows from this box.
   Use the stored α values. Do not recalibrate.
2. **`S10_before_steer`.** This row was running when the first box went down, and nothing from it was uploaded, so run it from scratch.
3. **`S10_base`.** Base model on the same parts and ids, with MMLU. Its job is to tell (a) a random shift at L30–63 that is itself not neutral from (b) a harness that gives a lower base (`SUMMARY.md`, "Things the reader should know first"). It now runs second, not last, because every Δ-vs-base in the report depends on it.
4. **`S10_before_centred`.**
5. **`S10_rand_centred`.**
6. **Torch window 2** (`sweep/stage10_torch2.sh`): the J-lens readout, the other environments' `cot_last` caches, and the projection tables.
7. **`S10_cotlast_steer`,** only if at least 3 h remain before the cap after step 6. Otherwise drop it, as the mission says.

Each row as in the original mission and `sweep/stage10_phase4.sh`: build, rollouts on all parts, metrics, uploads, MMLU. After **every** row, update `SUMMARY.md` (including the rows table, re-rendered against `S10_base` once it exists) and upload `box_report/stage10/`. Then a lost box costs at most the row in flight.

## Reporting changes

- Once `S10_base` exists, report every row against **both** `S10_base` (this box) and `S10ref` (stage 9), plus against its arm's random control. Read the comparison with `S10_base` and the random control; `S10ref` is for continuity.
- Say in `SUMMARY.md` that `S10_rand_steer` ran on the first box and every other row on this one, and give the step-1 harness check that makes them comparable.
- Everything else (statistics, hazard tables, damage rule, handover of passing `S10_before_*` rows to Terminal-Bench) is unchanged from the original mission.

## Cap and stop

Cap 14 h from the session start (about 1.5 h setup, 4 × 2 h rows, about 1 h torch window, report). Within the cap, the order above decides what gets dropped: first `S10_cotlast_steer`, then the torch window. Never drop `S10_base` or a random control while keeping the row it controls. Self-stop as in `mission_mbpp_impossible_update_2.md`: uploads verified, credentials removed, box stopped.
