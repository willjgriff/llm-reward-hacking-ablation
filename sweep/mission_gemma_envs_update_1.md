# Gemma envs, update 1: a refusal column on every grid row (user, 2026-11-27)

Status: approved by the user, 2026-11-27.

**Why:** the grid will be drawn as one heat map per model, with refusal as its own row and column. The `refdom` row already exists. The refusal column is missing: no row is evaluated on refusal, so we cannot see whether a hacking direction also removes refusal (a sign that it is a generic edit, not a hacking direction), and `refdom`'s own cell is not measured at this box's serving path.

## What to add
1. **Refusal eval for every grid row,** at that row's grid dose: `base`, each `<env>dom`, `gpdom`, `refdom`, `langdom@K_lo`, `langdom@K_hi`, and `covrand` if it runs.
   - heretic's harmful test (100) and harmless test (100);
   - thinking **off** and **on** (the settings and configs of the previous Gemma mission, `<tag>_off` / `<tag>_on` for refusal);
   - label: heretic's keyword label on the box. The judge runs on the laptop afterwards.
2. **Run it as root** from `/root/ortho`, like the Grader and capability evals (it scores text only).
3. **Uploads:** these records are completions to harmful requests. Upload them **only to the private repo** (`upload_run.sh --private`, under `second_model/gemma4_31b/envs/refusal/`), never to the public one. The public `SUMMARY.md` may give the rates, but no answer text.
4. **Grid table:** add the columns "refusal off" and "refusal on" (harmful refused %, harmless refused %), keyword label, marked "keyword; judge pending".
5. **Order:** run each row's refusal eval right after that row's other single-turn evals. It is cheap (400 short generations per row).
6. **Drop order under time pressure:** the refusal eval is dropped only after GSM8K-on for the control rows (it goes after item 5 of the mission's list). Never drop it for `base`, `refdom` or the env rows.

Everything else as in the mission.
