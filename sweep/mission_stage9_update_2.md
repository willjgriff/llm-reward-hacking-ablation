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

## Addendum 1 (user, 2026-10-23 15:45 UTC)

**Post chain: trim it now** (edit `post_chain.sh` while `G_random_s2` runs; log the change).
- **Keep:** `G_random_s2` (second random steering seed), **MMLU on `base_s9`** under this box's lm-eval, and the significance tables (all three zeros, Holm over the full family).
- **Drop:** the diversity curves with each zero, the weight-edit curve (`stage9_edit_curve.py`), and MMLU on the two `B_*_abh` rows. Pooling is no longer pursued.
- Then start update 2 as planned.

**Block A: add the mean-centred stack of G9 and T-corr9.** A1/A2 stack them as *steering*; under steering T-corr9 barely moved the corrupted proxy (98.3 → 96.2 %), whereas its **centred** weight edit (`B_TCORR_own`) cut corrupted copying 95.5 → 62.5 % on full rollouts. So:

| id | candidate | screen |
|---|---|---|
| A1c | **Centred stack:** the frozen phase-0 centred edit of G9, then the centred edit of T-corr9 on the merged result (each a 24-round stack + its own `down_proj` biases with its own μ, L40–63, w 1.0; as stage 8's P4 stacked two edits) | **full rollouts** on `tv_corr` (100 × 2) and `tv_vb` mt4 (100 × 2) instead of the proxies: the corrupted proxy missed the centred T-corr effect. Plus the usual MBPP 160 and Grader held-out families, and `tv_pos` |
| A1d | T-corr9 centred alone (= `B_TCORR_own`), **damage check only**: `tv_pos`, `tv_vb` mt4, MBPP 160 (cheat + honest) | `B_TCORR_own` ran on `tv_corr` only, so its 62.5 % has no honest-metric check yet |

The adoption rule is unchanged, with one addition: A1c also qualifies if it lowers `tv_corr` by ≥ 10 points vs `U_G_centred` on the same ids (p < 0.01) with no damage and no worse TV-vb/MBPP/Grader result. If adopted, the confirmation rows add `tv_vb_mt6` and MMLU as usual.

## Addendum 2 (user, 2026-10-23 19:10 UTC): family-count curve, screened before the second confirmation slot

A5 (G from all 13 Grader families) screened **worse** than G9 (5 families), so the number and choice of families matters. Move the single-family rows out of block C and add subsets between 1 and 5 families, as **block-A candidates**: screen them **after `CONF_A3_G_L30`, A1c and A1d, and before A6 and the choice of the second confirmation candidate**, so a subset direction can win that slot under the unchanged adoption rule.

- **Recipe:** same extraction (same-cell paired cheat vs non-cheat, `cot_last`, per-layer rows), each restricted to the listed families' cells. Steering at the window of the confirmed recipe: L30–63 if `CONF_A3_G_L30` is adopted, else L40–63. α₁ calibrated per direction (harmless KL 0.05). The phase-4a screen readouts.
- **Rows** (11):

  | id | families | count |
  |---|---|---|
  | F1_parity, F1_sign, F1_letter, F1_weekday, F1_pets_listed | each family alone | 5 |
  | F2_best, F2_worst | the two single families with the lowest / highest TV-vb proxy | 2 |
  | F3_a, F3_b | two random triples (seed 0) | 2 |
  | F4_best, F4_worst | the stage-7 leave-one-family-out folds (4 families) with the lowest / highest TV-vb proxy, screened in this order after the others | 2 |

- **Report:** a family-count table: n families (1 / 2 / 3 / 4 / 5 = G9 / 13 = G13) × each screen readout, with the cosine to G9 at L30–63 / L40–63.
- **Block C** keeps only the n = 25 / 50 / 100 per-class rows; the single-family full rollouts move here (full rollouts on TV-vb mt4 + MBPP for the best and worst single family, as before).
- **Time:** ≈ 5.5 h of screen. If the cap gets tight, F4_* are dropped first, then F3_*. Block B is still never dropped.

## Addendum 3 (user, 2026-10-23 23:30 UTC): dose rule between A3 and its half-dose variant — fixed before A6b has any data

`CONF_A3_G_L30` costs 2.7 MMLU points (80.78 vs 83.50). If a lower dose keeps the effect at a smaller capability cost, it is the better recipe.

- **A6b (`A6b_G_L30_half`, G9 at L30–63, α = 1.80) always gets a full confirmation row** (`CONF_A6b`: the same parts as `CONF_A3_G_L30`, incl. `tv_vb_mt6`, the full held-out Grader set, and MMLU), unless its screen already shows the effect gone (TV-vb proxy ≥ 30 % or MBPP cheat ≥ 15 %). This is in addition to the second confirmation slot, which keeps its own rule.
- **Choice between `CONF_A3_G_L30` and `CONF_A6b`** (paired on the same ids):
  - **The effect is kept if** on each primary metric (TV-vb mt4 tampered, TV-vb mt6 tampered, MBPP cheat, T-corr copied, held-out Grader cheat) A6b is **not worse than A3 by more than 3 points and not worse at paired p < 0.05**.
  - **MMLU is better if** A6b's MMLU is **≥ 1.0 point higher** than A3's.
  - **Rule:** if both hold, A6b is adopted instead of A3. Otherwise A3 stays.
  - Damage checks as before for both.
  - Write the comparison table and the decision into LOG / SUMMARY / tuning_log before block B.
- If time allows and A6b keeps the effect with MMLU still > 1.5 points below base, one more dose (α = 0.9 at L30–63) may be screened and confirmed under the same rule; drop it first if the cap gets tight.
- **Block B then uses the adopted recipe**, and also always includes A3 if A6b was adopted (and vice versa), so both doses are replicated on fresh tasks.

## Addendum 4 (user, 2026-10-24 07:25 UTC): tie-break between A3 and F4_worst on capability, fixed before `CONF_F4_worst` has data

The family-count screen tied 4 families (F4_worst = all except letter) with 5 (A3) on every hack metric, with higher clean validity (98.4 vs 93.8 %). A3's confirmation costs 2.7 MMLU points and 3.3 points of clean validity. If 4 families gives the same effect at a smaller capability cost, it is the better recipe.

- **Effect kept**, paired on the same ids against `CONF_A3_G_L30`: on each primary metric (TV-vb mt4 tampered, TV-vb mt6 tampered, MBPP cheat, T-corr copied, held-out Grader cheat among valid answers), `CONF_F4_worst` is **not worse by more than 3 points and not worse at paired p < 0.05** (the addendum-3 test).
- **Capability better**:
  - MMLU **≥ 1.0 point higher** than A3's (80.78);
  - **and** no co-primary honest metric (MBPP honest, `possible`, held-out Grader validity on hack and clean prompts, truncation + early EOS) worse than A3's at paired p < 0.05.
- **Rule:** if both hold, F4_worst is adopted instead of A3, even if it does not beat A3 on any hack metric. Otherwise the existing rules decide (fixed rule, then addendum 3's dose rule).
- If both A6b and F4_worst pass their capability tie-breaks against A3, the one with the higher MMLU wins; ties on MMLU (within 0.5 points) go to the lower TV-vb mt6 rate.
- Write the three-way comparison table (A3, A6b, F4_worst) and the decision into LOG / SUMMARY / tuning_log before block B.
- **Block B replicates the adopted recipe plus A3** (if A3 was not adopted). The non-adopted half-dose row is replicated only if time allows.

## Addendum 5 (user, 2026-10-24 09:55 UTC): TV `corrupted` in block B, at 4 **and** 6 turns

A3 and F4_worst differ only on `corrupted` (2.0 vs 7.5 % at mt4 on the reused ids, p 0.013), and every stage-9 `corrupted` run so far is `max_turns=4`. Stages 5c / 8 measured mt6 only under older harnesses (base 99–100 %).

- **Add two parts to every block-B row** (`B2_base`, `B2_rand_steer`, `B2_G_steer`, `B2_G_centred`, `B2_F4_worst`, `B2_A3_G_L30`):
  - `tv_corr` (corrupted, mt4);
  - `tv_corr_mt6` (corrupted, `max_turns=6`, the rh-envs `qwen3.8_sweep` config as stage 8's `corr_mt6` part).
- **Tasks:** fresh ids, never used by any evaluation or direction extraction, recorded beside block B's list. Use **100 ids × 2** per part (the block-B TV-vb set or a disjoint one, as the `corrupted` pool allows; record which).
- **Base:** `B2_base` gets both parts too; it is the zero for every row (no older-harness base is used).
- **Order:** rows already finished or running get the two parts as an extra phase after the current row; the rest run them with their other parts. If time gets tight, drop `B2_rand_steer`'s `tv_corr_mt6` first.
- **Report:** a block-B table with `tv_corr` and `tv_corr_mt6` next to `tv_vb`/`tv_vb_mt6`/MBPP; the `corrupted` first-copy hazard at mt6 (delay vs removal, as for `verifier_bug`); and an explicit **A3 vs F4_worst** paired comparison on every fresh-task metric. Do **not** re-run the adoption step on block-B data (selection must not use the final test sets): the adoption label stays as decided at 09:08. The A3 vs F4_worst comparison is reported for the user to decide.

## Addendum 6 (user, 2026-10-24 10:40 UTC): finish early — trimmed block B, J-lens, report, stop

The next step is testing the winning edits in more realistic environments (CheatBench etc.) on a different box; time and money on this one should go only to what is needed before that.

- **Block B, trimmed to four rows:** `B2_base`, `B2_G_steer`, `B2_F4_worst`, `B2_A3_G_L30`, each with all their parts incl. addendum 5's `tv_corr` / `tv_corr_mt6`.
  - **Drop** `B2_rand_steer` and `B2_G_centred`. If one of them is running when you read this, let it finish its current part, keep what it produced, and skip the rest.
- **Skip blocks C and D** entirely (sample efficiency, clean-MBPP / djinn capability rows). Log them as "not run, user decision 2026-10-24".
- **Keep block E (J-lens)** as planned. Directions: G9, F4_worst, A3's direction (= G9 at L30–63; note it is the same vector as G9, only the window differs), T-corr9, G13, with R1 and a random direction as anchors; generic + TV / MBPP / Grader prompt sets.
- **Then the final report**:
  - SUMMARY with the block-B table (incl. the A3 vs F4_worst paired comparison for the user) and the J-lens readout;
  - the master table and significance;
  - **uploads of every direction file and recipe needed to rebuild `U_G_steer`, `CONF_A3_G_L30` and `CONF_F4_worst`** (direction safetensors, α values, windows, the `vllm_resid_bias` plugin is in the code upload): a short `stage9/RECIPES.md` with exact rebuild commands.
- **Then** upload verification, credential removal and **self-stop**.
