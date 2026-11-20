# KTO refusal control, update 2: the LoRA baseline on a new box (laptop, 2026-11-20)

Read `sweep/mission_kto_refusal.md` and `sweep/mission_kto_refusal_update_1.md` first; this update changes only what is said here.

## Why
The second box finished `check` and `suite` and skipped `lora` because the table's rule, on heretic's keyword label, said a trained row cut refusal by 30 points or more. A laptop LLM judge (DeepSeek, 50 paired harmful-test answers per row) shows that label is wrong for the KTO rows: they change the refusal wording ("I'm not going to ...", which heretic's markers miss), not the behaviour. Judged as complying (partial + full): base 0 %, refdom 92 %, `rkto` low 0 %, `rktopt` low 10 %, `rktoall` off 28 % (+10 % degenerate), `rktoall` low 2 %. The rule read with the judge says the LoRA is needed, and the user decided to run it. The decision is the user's; the table's `LoRA baseline needed: False` does not stop this stage.

## What to run
1. Fetch the report into `box_report/kto_refusal/` (`kto_refusal/kto_refusal/{LOG,SUMMARY,notes}.md` in the private repo) and append to that LOG from here on.
2. Clone the fork as before (`claude` branch, **commit `f458285` or later**, into `/root/ortho`, branch `box/kto-refusal`), then stage `setup`.
3. Restore exactly as update 1 step 3 (`kto_refusal/{acts,vectors,records}` from the private repo, token only in the download command's environment), plus `records/grader/*.jsonl` → `/root/ortho/data/inspect/grader/` so the table has the earlier rows' grader runs. Check as update 1 step 4 and log it. `train_lora.py` reads the refusal records (Ethan's from setup + the restored harmless train neutrals) and `data/vectors/Qwen3.8-27B/refusal_rkto.json` (its neutral KL is the KL match).
4. Run stage `lora`. Since `f458285` it serves two adapters of the same training on one `cb_base` server: the KL-matched one (`refusallora_*`, the planned baseline) and the **final** one (`refusallorafinal_*`: whether the LoRA can remove refusal at all when not held to the direction's small KL budget). Then `./kto_refusal.py table` lists both.
5. No other runs, no changes to the recipe (`train_lora.py` hyperparameters, data, labels).

## Report
- SUMMARY: replace the tables with the new `table` output (both LoRA rows added), state the KL-matched step and its neutral KL against `rkto`'s 1.56 mnat/token, the final step and its neutral KL, the held-out screen of every saved adapter (`held_out.json`), and 3 compliant + 3 refusing harmful-test answers (reasoning off, first 300 characters) for each LoRA row. Add a line that heretic's label undercounts refusals on the KTO rows (laptop judge), so the laptop will judge the LoRA rows too.
- Uploads as in the mission, with `loras/` (`/root/ortho/data/loras/Qwen3.8-27B/refusal_kto_lora/`, all steps + `train.json` + `held_out.json`) and the new records, `.eval` logs and box logs.

## Budget and finish
4 h from your start on this box (setup ~40 min, LoRA training and scoring ~1.5–2 h, 12 eval runs ~40 min). If training would push past it, serve whatever adapters exist. Finish exactly as the mission's "Finishing" section: verified uploads, then `stop_box_when_idle.sh --now` as the last action.
