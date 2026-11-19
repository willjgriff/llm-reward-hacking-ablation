# KTO refusal control, update 1 (laptop, 2026-11-19 ~22:45 UTC)

**What happened:** the box was stopped by the idle watchdog at about 19:45 UTC, during stage `check`. The watchdog's busy list covered this project's scripts but not ortho's (`kto_refusal.py`, `run.py`, `train_lora.py`). `check` runs on the GPU with no vLLM server up, and your session was waiting on it without writing, so the box looked idle. The fault is in the watchdog, not in your work.

**Fix (already on this box):** `scripts/stop_box_when_idle.sh` now counts ortho's scripts and any GPU utilisation of 10 % or more as busy. The laptop restarted the watchdog with the new version (30 idle minutes).

**Resume:**
1. Check what survived on disk (the stop keeps it): `acts/kto_refusal/results.jsonl` and `vecs/` (all five configs), `~/hooks/r*.safetensors`, and whether `acts/kto_refusal/check.json` exists. Log what you find.
2. If `check.json` is missing, rerun stage `check`. vLLM is not running after the restart, and tmux sessions are gone.
3. Continue with `suite`, then `lora` if the table's rule says so, exactly as in the mission.
4. **New budget: 8 h from your resume**, which replaces the 02:52 UTC end. The time the box was stopped does not count.

Everything else in `sweep/mission_kto_refusal.md` is unchanged, including the uploads and the final immediate stop.
