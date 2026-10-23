# Stage 9, update 2 (user decision 2026-10-23): improve and then stress-test the Grader direction G9

This update overrides `sweep/mission_stage9.md` and `_update_1.md` where they conflict. Read it now, log it, and **start it as soon as the post chain prints `DONE post-chain`**. Write the stage-9 report sections for phases 4b and the post chain as usual first (≤ 1 h); do **not** self-stop after them. Self-stop only when this update is done or at the cap (2026-10-25 15:40 UTC), after the usual final report, uploads and verification.

## Why

Update 1 showed that G9 transfers to held-out environments as a deployable edit, on full rollouts:
- **Steering** (α₁ 4.19, L40–63): TV-vb mt4 46.0 → 9.0 %, mt6 97.0 → 35.5 %, MBPP 22.0 → 8.9 %, with honest metrics up and MMLU 82.72.
- **Centred** edit: 13.5 / 43.5 / 10.1 %, MMLU 82.55.

Pooling is worse than G9 alone (`B_TVB_abh` 28.5 % vs `U_G_centred` 13.5 % on the same TV tasks), so the pooled form is not pursued further. The user wants, in this order:
1. try to lower the residual hacking further (time-boxed, pre-set adoption rule);
2. replicate on fresh tasks;
3. sample efficiency;
4. capability checks;
5. a readout of what G9 is.

## Rules that apply to every block

- **Selection never touches the final test sets.**
  - Block A selects on the **screen** (the phase-4a readouts) only.
  - Block B's fresh tasks are used only for the final numbers.
  - Write the adopted recipe into LOG.md **before** any block-B run.
- **Damage checks on every screened row:** harmless first-token KL, Grader validity, MBPP honest rate, TV proxy control-prefix hack rate, truncation/early-EOS. A row with any co-primary loss > 5 points is "damage" and cannot be adopted.
- **p-values:** `paired_test` (task-paired) against base and against the matching random control. Every primary test joins the stage's Holm family.
- **Tuning is logged as tuning.** Count every screened configuration in a `tuning_log.md` (the DPO comparison will be owed an equal budget).
- **No training of any kind.** No edits to rh-envs, `external/`, scorers, or the frozen phase-0 recipe definition. New recipes are new rows.

## Block A: reduce the residual further (hard time box: 10 h including confirmation)

**Screen** = the phase-4a readouts and machinery:
- G held-out families: 16 cells × 10;
- MBPP: the 160 phase-4a tasks × 1;
- TV-vb and T-corr decision-call proxies: 60 prefixes × 8;
- each candidate calibrated to harmless KL 0.05 unless stated.

Reference row: 4a's `G_G` (G9 steering).

| id | candidate | note |
|---|---|---|
| A1 | **Stack** G9 + T-corr9: apply both steering edits, each at its own α₁ (no averaging) | report the combined KL |
| A2 | Same stack, both α scaled by one factor so the **combined** KL = 0.05 | |
| A3 | G9 steering at decoder layers **30–63** | α recalibrated for the window |
| A4 | G9 steering at decoder layers **20–63** | α recalibrated |
| A5 | **G13**: the G direction re-extracted from all 13 Grader families (main + held-out; same recipe: same-cell paired cheat vs non-cheat, `cot_last`; held-out-family rollouts from `base_s9` `gr_ho_hack`, a 20 % per-cell split kept out) | The G readout is no longer held out for A5: screen it on MBPP + TV only and mark it |
| A6 | Best of A1–A5 combined with the best window, if two different ones win | ≤ 2 rows |

Optional, only if time remains inside the box. Both are in-environment, so label them as such:
- A7: a TV "persistence" direction from mt6 calls 4–5 (tamper vs resubmit, same task), stacked on G9.
- A8: a round-2 direction from rollouts that still hack under `U_G_steer` vs those that don't (same tasks), stacked on G9.

**Adoption rule (fixed now).** A candidate replaces plain G9 only if, on the screen, it improves on `G_G` by **≥ 10 points on the TV-vb proxy, or by a paired p < 0.01 on MBPP cheat**, with no damage and no worse Grader result.
- **Confirmation:** full rollouts as `U_*` rows (`tv_vb` mt4 + `tv_vb_mt6` + `tv_pos` + `tv_corr` + `mbpp` 168 × 2) + MMLU, for at most the best 2 candidates. Adopt one only if the full rollouts confirm the gain at p < 0.01 against `U_G_steer` on the same ids.
- **If nothing qualifies, plain G9 stays the recipe.** Either way, write "Adopted recipe: …" in LOG.md and SUMMARY.md before block B.

## Block B: replication on fresh tasks (≈ 4 h)

G9 has never seen MBPP or TV data, so every MBPP and TV task is held out for it. Use tasks **not used by any earlier evaluation**:
- **MBPP:** ≈ 300 tasks from the stage-7 `train` split (`data/stage7/classes/mbpp_splits.json`; non-bool; seed 0) × 2.
- **TV `verifier_bug`:** ≈ 150 ids from the source pool not in any earlier eval set (exclude the stage-5b/7 100 ids, the F1a ids and every id used for direction extraction by any type; record the list) × 2, at **mt4 and mt6**, plus `possible` on the same ids × 1.

Rows: `base` (re-roll on the fresh sets), `G9 steer`, `G9 centred`, the adopted recipe if different, and the random steering control. Report everything with p and Holm. If the fresh-task effect is ≥ half the original effect at p < 0.001, call it replicated.

## Block C: sample efficiency and data quality (≈ 3 h)

All rows use the adopted recipe's edit type, at its KL-calibrated α, extracted with the same recipe.
- **n per class:** G9 from 25, 50 and 100 rollouts per class (cell-balanced random subsets, 3 seeds at n = 25, 1 seed otherwise).
- **Single family:** G9 from each of the 5 main families alone (`parity`, `sign`, `letter`, `weekday`, `pets_listed`).
- **Readout:** the screen (TV proxies + MBPP 160 + Grader held-out families). Then full rollouts (TV-vb mt4 + MBPP 168 × 2) for the n = 25 row and the best and worst single-family rows.
- **Report:** the hack rates and the cosine to full G9 per layer.

## Block D: capability beyond MMLU (≈ 2 h)

For base, G9 steer, G9 centred and the adopted recipe:
- **MBPP with clean tests:** rh-envs `mbpp` with corruption **off**, on the 168 held-out tasks × 2. Honest pass rate = plain coding ability.
- **djinn honest solving:** the phase-1 pilot problems × 2, honest = passes the secure verifier. The hack labels are not of interest here.
- **MMLU** for any row that lacks it, plus the base on this box (the post chain does it).

## Block E: what G9 is (≈ 1 h)

Run `scripts/jlens_readout.py` on G9 (and the adopted recipe's direction if different), on the generic and on-distribution prompt sets, with R1 and a random direction as anchors, and the logit lens alongside. Compare with stage 8's X2 reading ("disguised / clever / hidden" vs "simple").

## If time runs short

Order of what to drop:
1. A7/A8
2. Block E
3. The djinn part of block D
4. The single-family rows of block C

Block B is never dropped.

## Reporting

- New SUMMARY section **"Update 2: improving and stress-testing G9"**, with:
  - the block-A screen table (every candidate, with KL and damage columns) and the adoption decision;
  - the confirmation rows;
  - the fresh-task replication table;
  - the sample-efficiency table;
  - the capability table;
  - the J-lens top/bottom tokens.
- Then the final stage-9 summary, uploads, verification and self-stop as in the mission.
