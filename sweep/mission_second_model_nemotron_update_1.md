# Second model, box N, update 1: pilot GLM-4.7-Flash as well (user, 2026-11-26)

This overrides `sweep/mission_second_model_nemotron.md` step 2 ("If Nemotron passes … Do not pilot GLM"). Everything else stands.

## Why
Nemotron passes on refusal and Grader-on only; TV fails (3% tampered on, 0% off). Without a second reward-hacking environment there is no transfer test on this model, and `tvpdom` cannot be checked in its own environment. GLM-4.7-Flash is piloted to see whether it hacks on TV (and the Grader), which would give a second model with two environments.

## What to do
1. **Port first, without the GPU.** While Nemotron's current work runs, write the GLM-4.7-Flash support: model spec, rh-envs configs `glm47_flash_on` / `_off`, chat-template notes, layer windows, and the `vllm_proj.py` / HF-reader port for `glm4_moe_lite`. Find a vLLM that serves it, in a separate env if needed. Download the weights if disk allows (keep ≥ 40 GB free); otherwise download them at the swap.
2. **Swap at the next clean break.** Let every running eval or collection chunk on the Nemotron server finish. Do not kill a run in progress. Start no new Nemotron GPU work after that. Then stop the Nemotron server and serve GLM.
3. **Pilot GLM: the whole Phase 1,** with the same cells, sizes, gates, 3 h port limit and `PILOT.md` (under `box_report/second_model_glm47_flash/`, uploaded under `second_model/glm47_flash/`). Then run the judge on the base refusal cells.
4. **Resume Nemotron:** restart its server with the same flags and continue Phase 2/3 where it stopped. Log the gap.
5. **If GLM passes, and its passing cells include TV (on or off):** run GLM's Phase 2/3 after Nemotron's Phase 3 is done. The hard end moves to **2026-11-27 18:52 UTC** (32 h from the start), for that case only. Apply the mission's drop order to GLM if time runs short. Otherwise keep the 20 h hard end.

   If GLM passes **without** TV (refusal and Grader only, like Nemotron), report its pilot and do **not** run its Phase 2/3.
6. Add GLM's pilot table to `SUMMARY.md`, next to Nemotron's.
