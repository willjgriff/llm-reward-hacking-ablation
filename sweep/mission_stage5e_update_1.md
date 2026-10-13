# Mission update 1 for `sweep/mission_stage5e.md` (user, 2026-10-08 23:30 UTC)

Everything else stays as it is. This reorders the remaining queue and moves one measurement forward, based
on the rows finished so far: A3 (`attempt_vs_neither_mt6` × `completion_last`) is the best row (tampered
0.0% [0, 1.9], possible 100%), while the whole `tamper_vs_A` family failed the gate (A1, A2 and the paired
A4, A5). At ~30 min per row the remaining queue ends at the 18 h cap, so the self-proposed slots would never
run. Treat the rows below as the 6 self-proposed rows, already justified.

## 1. New rows (A3's contrast on the other ingredients of the matrix)
Add a contrast `attempt_vs_neither_mt6` to the pos2 config (`{positive: [attempt_vb], positive_max_turns: 6,
negative: [neither_vb_mt6]}`; the pos2 cache holds all `neither_vb_mt6` rows and the mt6 half of the
`attempt_vb` subset) and its call-unit twin `attempt_vs_neither_mt6_call` (`unit: call`, same classes;
`neither_vb_mt6` has no decision call, so its negatives are all calls with k ≥ 1). Compute their directions
with the same criteria table as Block B, then:

| id | direction | method |
|---|---|---|
| A3b | `attempt_vs_neither_mt6` × `answer_first` (pos2) | default stack 40–63 (+ adaptive) |
| A3c | `attempt_vs_neither_mt6` × `cot_last` (pos2) | default (+ adaptive) |
| A3d | `attempt_vs_neither_mt6_call` × `decision_prompt_last` (pos2) | default (+ adaptive) |
| A3e | `attempt_vs_neither_mt6` × `action_mean` (pos2) | default (+ adaptive) |
| A3w | `attempt_vs_neither_mt6` × `completion_last` (blockA, as A3) | stack 40–63, **w = 0.5** (dose) |
| A3x | `attempt_vs_neither_mt6` × `answer_mean` (blockA) | default |

Same gate, same proxy, same env-eval policy. Report, next to each, the cosine of its direction to A3's at
the best layer, and whether A3w reproduces A3's 0% at half strength (a dose-response point for the report).

## 2. New queue order (after the row currently running)
B6, B13, B2, B12, B1, B4, **A3w, A3x, A3b, A3c, A3d, A3e**, C4, C5, C6, C7, then the deprioritised rows
B9, B10, B11 (the `tamper_vs_A` family failed at trajectory level, so these have the lowest prior now), B7,
B8, B5, then C1–C3, then the adaptive variants. The self-proposed budget is spent by section 1; do not add
further rows.

## 3. MMLU for the base and A3 now, not in the final pass
At the next point where vLLM is free (after the current row's env phase), run MMLU for the base model and
rebuild A3 for its MMLU (73 s build, ~20 min each), then delete the checkpoint and continue the queue.
A3's verdict under the decision rule needs it, and the final pass may be squeezed by the cap. If the cap
approaches, the stop order is: finish the current row, make sure the base and A3 MMLU numbers exist,
write the report, upload, stop.

## 4. Nothing else changes
The judge-based re-scoring of the eval rollouts (in-memory bypass routes the oracle cannot see) and the
extended write regex are done on the laptop afterwards from the uploaded run dirs and `proxy.json`; keep
storing the sampled command text per sample as you do now.
