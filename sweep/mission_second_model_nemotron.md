# Second model, box N: Nemotron 3.5 Lightning, fallback GLM-4.7-Flash

Status: approved by the user 2026-11-25. Read `sweep/mission_second_model.md` first; it is the mission. This file only fixes the models, the order and the budget.

- **First model:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (`nemotron_h`: hybrid Mamba/attention MoE, 52 layers, hidden size 2688, 128 routed experts, ~3B active). Tag `nemotron35_30b`. The README names vLLM's reasoning parser (`nemotron_v3` / `nemotron_3`); use the one this vLLM version has.
- **Fallback:** `zai-org/GLM-4.7-Flash` (`glm4_moe_lite`: MoE with MLA attention, 47 layers, hidden size 2048, 64 experts, 4 active). Tag `glm47_flash`. It needed vLLM's main branch at release; use a release that supports it, in a separate env if the main vLLM env cannot serve both.
- **Order (fixed in advance):**
  1. Pilot Nemotron (Phase 1, including the 3 h port limit).
  2. **If Nemotron passes:** run the whole mission on Nemotron. Do not pilot GLM.
  3. **If Nemotron fails:** stop its server, upload its `PILOT.md` under `second_model/nemotron35_30b/`, download GLM-4.7-Flash and pilot it the same way. If GLM passes, run the whole mission on GLM.
  4. **If both fail:** write `SUMMARY.md` with both pilots, upload, and finish (stop the box). No other model.
- The two models never share the GPU. Delete Nemotron's weights before downloading GLM if disk is short.
- **Budget:** hard end 20 h after the session start (2 h more than box G, for the possible second pilot). Stop as soon as the work and uploads are done.
