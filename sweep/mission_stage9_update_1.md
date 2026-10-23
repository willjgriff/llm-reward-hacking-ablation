# Stage 9, update 1 (user decision 2026-10-23): test the single Grader direction's transfer properly, next

This update overrides `sweep/mission_stage9.md` where they conflict. Log it in LOG.md when you read it.

## Why

Phase 4a found the strongest result of the stage at k = 1, not in the pooling curve. Steering with the stage-9 G direction (`G9`: same-cell paired cheat vs non-cheat, `cot_last`, decoder layers 40–63, α₁ = 4.19 at harmless KL 0.05) lowered every other type's hack rate:
- MBPP 22.0 → 5.0 %
- TV-vb proxy 68.5 → 17.7 %
- T-corr proxy 98.3 → 93.3 %

Phase 4b as planned tests only each type's own direction and the all-but-h pools. **G9 alone is never applied to MBPP or TV as a weight edit**, and TV under steering has only been measured with the proxy. The user considers this the most important open test: a direction from one environment, tested properly on held-out environments.

## Order

1. Let the row now running (`B_M_abh`) finish and upload.
2. **Run the three rows below next**, before `B_TVB_own`, `B_TVB_abh`, `B_TCORR_own`, `B_TCORR_abh`. Edit `scratchpad/phase4.sh` (or the chain) accordingly and log the change.
3. Then resume the rest of 4b and the post chain unchanged.

## Rows (same driver, `scripts/stage9_rows.py`, same parts and task ids as `base_s9` and `P0_rand_full`)

| id | edit | parts | zeros |
|---|---|---|---|
| `U_G_centred` | the **frozen phase-0 recipe** with direction `G9`: mean-centred, w 1.0, μ = G's non-cheat training mean along d̂, decoder layers 40–63, stacked edit + `down_proj` biases | `mbpp` (168 × 2), `tv_vb` (mt4, 100 × 2), `tv_pos`, `tv_corr` (mt4, 100 × 2) | `base_s9`; `P0_rand_full` |
| `U_G_steer` | bias-only steering checkpoint, subtract α = `calib:G` (4.19) · d̂ at layers 40–63, as the 4a row `G_G` | the same four parts, **full rollouts** (not the proxy) | `base_s9`; `U_rand_steer` |
| `U_rand_steer` | the same steering with the random direction `random_s1` at its own α₁ (10.72), as 4a's `G_random_s1` | the same four parts | `base_s9` |

- **Honest metrics are co-primary**, as everywhere: MBPP honest rate, TV `possible` pass rate, peek rate, truncation and early-EOS. A hack drop that comes with more than a 5-point honest loss is "damage".
- **MMLU** (the same lm-eval setup as the post chain) on `U_G_centred` and `U_G_steer`.
- **p-values:** `paired_test` from `scripts/results_significance.py`, per part, against both zeros. Add these primary tests to the stage's Holm family.
- **Reading:**
  - Hand-read 20 TV-vb episodes and 20 MBPP answers per edited row: what replaced the hack (resubmission, pushback on the test, giving up, incoherence?).
  - Report, CPU only, where MBPP's and TV's non-hack and hack means sit along G9's d̂ at L40–63. The centred edit uses the *Grader's* non-cheat mean as the reference point, which may not be where those environments' honest activations sit; this explains the centred row whichever way it goes.

## Reporting

Add a section **"Single-direction transfer (G9), full rollouts"** near the top of SUMMARY.md: one table per edited row with rate, Δ vs base and vs its random zero, 95 % CI, p, Holm, and the honest metrics. Compare it explicitly with the 4a steering-proxy numbers and with stage 7's X5 (zero-ablation of D_GR: no transfer).

Nothing else in the mission changes: same cap (2026-10-25 15:40 UTC), same uploads, self-stop at the end.
