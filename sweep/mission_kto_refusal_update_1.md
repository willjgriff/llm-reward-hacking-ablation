# KTO refusal control, update 1: resume on a new box (laptop, 2026-11-19 ~23:00 UTC)

Read `sweep/mission_kto_refusal.md` first; this update changes only what is said here.

## What happened
The first box (instance 61481990) ran setup, sample and train, uploaded their outputs at 19:00 UTC, and was then stopped by the idle watchdog at about 19:45 UTC, during stage `check`. The watchdog's busy list covered this project's scripts but not ortho's (`kto_refusal.py`, `run.py`, `train_lora.py`). `check` runs on the GPU with no vLLM server up, and the session was waiting on it without writing, so the box looked idle. The instance could not be restarted, so you are on a new box. The fault was in the watchdog, not in the work. `scripts/stop_box_when_idle.sh` now also counts ortho's scripts, and any GPU utilisation of 10 % or more, as busy.

## Resume
1. Fetch the first box's report into `box_report/kto_refusal/` (`kto_refusal/kto_refusal/LOG.md` and `SUMMARY.md` in the private repo), and append to that LOG from here on.
2. Clone the fork as in the mission (`claude` branch into `/root/ortho`, branch `box/kto-refusal`), then run stage `setup`.
3. **Do not rerun `sample` or `train`.** Restore their outputs from the private repo as root. Read the token from `upload.env` into the environment of the download command only; never write it to a file. Download `kto_refusal/{acts,vectors,records}` and place them as follows:
   - `acts/*` → `/root/ortho/acts/kto_refusal/`: `base.pt`, `results.jsonl`, `vecs/*.pt`
   - `vectors/refusal_r*` → `/root/ortho/data/vectors/Qwen3.8-27B/`
   - `vectors/r*.safetensors` (rkto, rktopt, rktoall, rkrand, rkrandall) → `/root/hooks/`
   - `records/refusal/*.jsonl` → `/root/ortho/data/inspect/refusal/`: the harmless train neutrals (`refusal38_train_harmless`) and the smoke run
4. Check the restore and log it:
   - `./kto_refusal.py data` prints the first box's counts (796 / 798 / 400; 100 / 100 / 100);
   - `results.jsonl` holds rksmoke plus the five configs;
   - `~/hooks` holds the five hook files.
5. Run `check`, then `suite`, then `lora` if the table's rule says so, then finish, all exactly as in the mission. Uploads go to the same `kto_refusal/` paths (they add to what is there).
6. **New budget: 8 h from your start on this box**, which replaces the first box's 02:52 UTC end.
