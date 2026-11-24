# Second model, box G, update 2: trim the KL-matched controls and finish (user, 2026-11-26)

**Why:** `prandk_ref` answers loop after thinking (Grader on main: valid 55%, 405 of 1,175 answers at the 8,192-token cap), so its thinking-on Grader runs cost about an hour each and give a control dominated by invalid answers. The thinking-off Grader already separates `refdom` from its control (held-out 1.2% vs 17.8%, base 18.0%). The remaining value is in the cheap evals.

## What to do
1. **Stop `prandk_ref_sm_g_on_heldout`** now. Keep what is finished and report it as not run (cost).
2. **Do not run `prandk_gp`'s Grader-on runs** (main or held-out). Report them as not run, for the same reason.
3. **Run the cheap evals:**
   - `prandk_ref`: refusal off and on, harmful and harmless.
   - `prandk_gp`: Grader off main and held-out, then refusal off and on, harmful and harmless.
4. **Judge the refusal answers** of both controls, harmful and harmless, with the same settings and `--all`, as for `refdom` and `gpdom`.
5. **Then finish:** tables, SUMMARY (rule read against the KL-matched controls where they exist; for the Grader-on cells, state that the control is degenerate at generation length: `prandk_ref` valid 55%, cheat 18.4% among valid), uploads, verified, then self-stop as in the mission.

Everything else as before.
