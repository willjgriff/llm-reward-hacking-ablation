# Mission update 1 for `sweep/mission_stage8_x2.md` (user, 2026-10-16 ~18:50 UTC)

Everything else stays as it is (rules, uploads, the 18 h cap, the final sequence).

## Why
Part 2 showed X2 does not transfer: at `verifier_bug` mt6 it tampers like the base (89.0 vs 86.5%), and nothing
changes on `corrupted`. X2's 4-turn effect reads as a **delay** of tampering, not a removed propensity. Part 1
already shows where the effect comes from (the shared component), so more X2 rows are not worth the GPU time.
The open question X2 raises is about our best in-environment direction: **does D_TV's 0% at 4 turns survive more
turns, or is it also a delay?** Every TerminalVerifier eval so far has been at 4 turns.

## Changes
1. **Let C1 finish** (suite E, metrics, uploads), then **stop the matrix chain before H1 builds**.
2. **Dropped:** H1, N100, N25 (part 5 rows; keep the directions and J-lens already uploaded), D1, D2 (part 4),
   C1b, and the extra MMLU pass. Record them in SUMMARY.md as "dropped by update 1", not as failures.
3. **New row X1T:** rebuild X1 from `stage7/stage7/rows/X1/recipe.json` (D_TV = `s7-tv`
   `tamper_vs_clean__cot_last`, stacked edit L40–63, w 1.0; fetch the direction from HF
   `stage4/stage7/s7-tv/directions/` if it is not on the box). Damage gate as usual. Serve it and run exactly the
   three part-2 parts with the same flags and ids as X2T: T1 `verifier_bug` mt6 × 2 (200), T2 `corrupted` mt6 × 2
   (200), T3 `corrupted` mt4 (100). The references are the part-2 base runs already on the box
   (`base_s8_*`: 86.5% tampered, 99% passed, 94% passed) and X2T.
4. **Report** X1T like X2T: rate with Wilson 95% interval, peek, tampered, passed, tampered-but-not-passed,
   interrupted/time-limit counts, the tampering route (`sed` / heredoc / `python -c` / other, from the command
   that first touched `verifier.py`), and for T2/T3 the copied-value counts. Add one headline line at the top of
   SUMMARY.md: "D_TV on other hacks: verifier_bug mt6 base → X1, corrupted mt6 …, corrupted mt4 …", and one
   sentence on whether D_TV's 4-turn effect is a removal or a delay. For comparison, R1 on `verifier_bug` mt6 gave
   30.3% tampered (stage 5c).
5. Then write the final report, upload, verify, remove the credentials file and self-stop, as the mission says.
   Uploads for X1T: rollouts under `stage6/X1T/rhenvs`, the row data under `stage8`.

Budget: ≈ 45 min for X1T after C1, then ≈ 45 min for the report and stop.
