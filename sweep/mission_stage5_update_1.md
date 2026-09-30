# Mission update 1 for stage 5 (user, 2026-09-25 13:45 UTC, after the steering grid)

Two cheap follow-up checks on the one specific effect the grid found (`attempt_vs_A` ablate: hack and
attempt answer tokens +0.05 nats, control flat). Same model, same 400 rows, same code paths; about 30-40
minutes of GPU. Everything else in `sweep/mission_stage5.md` stays as it is.

## 1. Per-token attribution of the `attempt_vs_A` ablate effect
Question: is the +0.05 concentrated on the tokens of the tampering commands themselves (lexical: the
direction encodes "commands that touch `verifier.py`") or spread over the whole answer (propensity)?
- Re-run baseline and `attempt_vs_A__ablate__none` on the 100 `hack_vb`, 100 `attempt_vb` and 100
  `nonhack_A` rows, keeping the **per-token** NLL delta on answer tokens (store as compact arrays per row).
- Group answer tokens per row into: (a) tokens of calls whose command text contains `verifier.py`
  (from `raw_test_results["turns"][i]["command"]` or the call's visible answer text; state which you used),
  (b) tokens of all other calls. Report per class: mean Δ per token in (a) and (b), the share of the summed
  Δ that falls in (a), and the share of answer tokens that are in (a). Also report Δ per call index
  (turn 1, 2, ...) so a concentration in the tampering turn is visible.
- Also the top-30 tokens by summed Δ across the 200 hack/attempt rows (token string, count, summed Δ):
  if they are `verifier`, `.py`, `cat`, `>`, `sed`, ..., the effect is lexical.

## 2. Random-direction control
Question: how much of the small-α NLL rise, and of the ablate effect, does any direction produce?
- 3 random unit directions (seeds 0, 1, 2), one per hooked layer for `ablate` (same layers 40-63 as
  `attempt_vs_A`), and for `add`/`subtract` at α = 0.1 and 0.2 at layer 40 (same scale s as row 41).
- Same table columns as `results.md`; report each seed and the mean over seeds, next to the
  `attempt_vs_A` rows they mirror. The ablate control is the important one: if random-direction ablation
  over 24 layers also gives contrast ≈ 0 with a small uniform rise, the `attempt_vs_A` contrast of
  +0.047 is specific; if random ablation gives a similar contrast, it is not.

## Reporting
Append a section "After controls" to `box_report/stage5/SUMMARY.md` with both results and one paragraph
stating whether the `attempt_vs_A` effect is lexical or not, and whether it is specific. Upload
`data/stage5/steering/tv27b-full` again (new files) with `--dest-prefix stage5`, the report with
`--dest-prefix sweep`, and the code. Then stop.
