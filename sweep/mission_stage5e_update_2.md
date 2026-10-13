# Mission update 2 for `sweep/mission_stage5e.md` (user, 2026-10-09 07:45 UTC)

Everything else stays as it is. **The hard cap is extended by 5 h: 15:42 UTC (epoch 1791560520).** No new
row may start after **15:00 UTC (epoch 1791558000)**; the final report, uploads and self-stop happen by
15:42. Update every hardcoded deadline in your scripts (`--deadline-epoch`, `/root/s5e_stop_at_0755.sh`,
`/root/s5e_final_passes.sh`) and log the change.

## What to do with the extra time, in this order
1. If the 07:55 row-boundary stop has not fired yet, let the driver keep going instead; if it has, run your
   final-pass list **completely** (MMLU B13/B1/C7, extra env eval A3c, extra env evals A3b/B2/A1, the C5
   re-run, MMLU A7/A3w/C6) with the guard moved to 15:00.
2. Then the rows the original cap would have dropped: **C1, C2, C3** (hook rows: gate + resampling proxy
   only, as the mission says; these are the only rows on the steering-vs-orthogonalization axis, so they
   have priority), then **B7, B8**, then the adaptive-window variants (`<id>a`) of every Block-B row whose
   best layer is below 36, then B5, B10, B11 if time remains.
3. Regenerate the report, the final uploads and the self-stop sequence as before. Write the ordered list of
   rows that still did not run into SUMMARY.md.
