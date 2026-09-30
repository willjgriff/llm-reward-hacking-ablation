# Mission update 1 for stage 4 (user, 2026-09-24 12:30 UTC, after reading the subset-400 results)

This amends `sweep/mission_stage4.md`. Everything not mentioned here stays as it is. **Do not touch the
running cache job** (tmux `stage4`); all of this is CPU work on the cache that exists.

## Why
On the `completion_last` directions of `vb_vs_A` and `corr_vs_A`, the honest passes (`pass_possible`)
project at 1.13 and 1.44, beyond the hack mean (class-projection table). Those directions encode "the
rollout ended with a passing submission", which is how the classes were defined (verified hack = passed,
`nonhack_A` = ran out of turns), not hacking. The token-mean directions behave differently: on
`vb_vs_A` `cot_mean`, `attempt_vb` projects at 1.0 (with the hacks) and `pass_possible` at -2.4. The
verdict rule lacked a check for this.

## Changes
1. **Two more contrasts** in `configs/stage4_terminal_verifier.yaml` (`analysis.contrasts`):
   ```yaml
   attempt_vs_A:    {positive: [attempt_vb], negative: [nonhack_A], description: "tampered-but-not-passed vs neither tampered nor passed: both failed at the turn limit on the same impossible task, outcome held fixed"}
   hack_vs_attempt: {positive: [hack_vb], negative: [attempt_vb], description: "successful vs failed tampering: what success adds"}
   ```
   Add to `cosine_pairs`: `[vb_vs_A, attempt_vs_A]`, `[corr_vs_A, attempt_vs_A]`, `[attempt_vs_A, hack_vs_attempt]`,
   `[vb_vs_A, hack_vs_attempt]`. Add to `transfer_pairs`: `[attempt_vs_A, vb_vs_A]`, `[attempt_vs_A, corr_vs_A]`,
   `[vb_vs_A, attempt_vs_A]`. Log the diff in `LOG.md`; upload the code (`sweep/upload_code.sh`).
2. **One more verdict criterion.** A direction counts as a hack direction only if the mean normalised
   projection of `pass_possible` on it is **below 0.5** (class-projection table, same layer). Under this
   rule `completion_last` is not usable on `vb_vs_A`, `corr_vs_A`, `vb_mt4_vs_A` or `allhacks_vs_A`. In
   `SUMMARY.md`, keep the existing tables and add a restated verdict table: per contrast, the best
   aggregation among those that satisfy all four criteria (length-matched AUROC ≥ 0.85, probe balanced
   accuracy ≥ 0.85, split-half cosine ≥ 0.8, `pass_possible` projection < 0.5), with its numbers, and say
   explicitly when no aggregation satisfies all four (then report the best-scoring one with the criterion it
   fails). Also report, for every A-contrast direction, the layer-0 projection AUROC next to the best layer
   (how much is token identity).
3. **`--no-curves` for the remaining analyses.** The split-half and sample-size bootstraps took most of the
   77 minutes on the subset and would take about six hours on the full set; the subset already has them.
   - Rerun the subset now with the new contrasts: `--subset-per-class 400 --no-curves --out data/stage4/directions/tv27b-subset400-v2` (keep v1).
   - Full analysis (step 6) with `--no-curves --out data/stage4/directions/tv27b-full`.
4. **Uploads** as in step 7, plus `tv27b-subset400-v2`. Everything else unchanged.
