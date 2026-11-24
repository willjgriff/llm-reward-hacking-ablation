# Second model, box G, update 1: KL-matched random controls (user, 2026-11-26)

**Why:** on Gemma 4 the unscaled random control `prand` destroys the model: first-token KL 19.7, MMLU −20.9. Your diagnosis: Gemma's late residual stream is hypersensitive to small isotropic shifts across many layers. A destroyed control makes "p < 0.01 vs its control" meaningless, and it cannot separate a direction-specific effect from generic damage. That matters now that `refdom` cuts Grader-on cheating 41.6 → 5.5% at an MMLU cost of −7.

**The recipe of the real rows does not change.** `refdom` and `gpdom` stay exactly as fitted and served. Only the control changes.

## What to do
1. **Stop the `prand` tier-1 evals** if they have started. Keep any `prand` results that already exist, and keep its capability numbers: they are reported as "unscaled random control: destroys the model".
2. **KL-matched random controls.** Use `prand`'s random rows, window and pins, with the hook scaled: `h ← h − s·((h·u) − c)·u`. You already measured s = 0.1 → KL 0.025.
   - Find s by bisection on heretic's harmless test prompts, in HF, so that the first-token KL matches each real row within ±15%:
     - `prandk_ref`: target `refdom`'s KL (0.72);
     - `prandk_gp`: target `gpdom`'s KL (2.41).
   - Log s, the KL reached and MMLU for each. If no s reaches the target before the model degenerates (valid Grader answers < 80% on a 64-answer check), use the largest s that keeps validity and log it.
3. **Serve and evaluate each `prandk_*` as a tier-1 row,** with the same evals as `refdom` / `gpdom`:
   - Grader on main + held-out (and off as context);
   - refusal off and on, harmful and harmless, judged;
   - MMLU + KL.

   Run them right after `refdom` and `gpdom` finish.
4. **The rule** now compares each real row with its KL-matched control (`refdom` vs `prandk_ref`, `gpdom` vs `prandk_gp`). In SUMMARY, state that this is a change from the mission and from Qwen's controls. On Qwen the random controls had near-zero KL and were never matched.
5. Everything else as before: no Arditi rows (no candidate passed KL < 0.1; report that), uploads, SUMMARY, self-stop.
