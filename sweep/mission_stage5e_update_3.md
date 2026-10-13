# Mission update 3 for `sweep/mission_stage5e.md` (user, 2026-10-09 12:10 UTC)

Everything else stays as it is. **The hard cap is extended again, to 20:12 UTC (epoch 1791576720).** No new
row may start after **19:30 UTC (epoch 1791574200)**; final report, uploads and self-stop by 20:12. Update
every hardcoded deadline (`--deadline-epoch`, the update-2 watcher, the final-passes script) and log it.

The user wants **every row of the matrix run**, not a prioritised subset. Order from here:
1. Finish the final-pass list as queued (MMLU C7, **MMLU A3c**, extra env evals A3b/B2/A1, C5 re-run, MMLU
   A7/A3w/C6).
2. **B11** (`tamper_vs_A` × cot_last; cot_last is the position that turned `attempt_vs_A` into B1's 0%, and
   has never been tried with `tamper_vs_A`), **B5** (`attempt_vs_A` × full_mean, the mentor's "all tokens"
   position), **B7**, **B8**, then the adaptive-window variants (B6a, B7a, B9a and any other Block-B row whose
   best layer is below 36), then **B10**.
3. Regenerate the report, final uploads, self-stop as before. If anything is still unrun at 19:30, list it
   in SUMMARY.md.
