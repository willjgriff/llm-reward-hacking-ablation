# Second model, box G: Gemma 4 31B

Status: approved by the user 2026-11-25. Read `sweep/mission_second_model.md` first; it is the mission. This file only fixes the model, the failure path and the budget.

- **Model:** `google/gemma-4-31B-it` (`gemma4_text`, dense, 60 decoder layers, hidden size 5376, sliding-window attention with every 6th layer full; multimodal, serve text-only). Tag `gemma4_31b`.
- **If the pilot fails** (any gate, or the port): write `PILOT.md` and `SUMMARY.md` with what was measured, upload, and finish (stop the box). There is no fallback model on this box.
- **Budget:** hard end 18 h after the session start. Stop as soon as the work and uploads are done.
- **Disk:** about 62 GB of weights; keep at least 60 GB free for activations and logs.
