# Capability of Ethan's saved rows (GSM8K, IFEval, Terminal-Bench 2)

Status: approved by the user 2026-11-17 ("go ahead and start those"); laptop-set-up, then unattended on the VM (`cheatbench/run_capability_kto.sh`).

## Why
grader_kto, b1 and the two KTO LoRAs carry the current transfer results (MBPP 27.3 → 11.3 %, held-out grader 30.9 → 7.0 %, direction vs LoRA) but have no capability number on any benchmark. mbpp_kto has Terminal-Bench only (42/89 vs base 46). A3 (G9) showed that a large cut in cheating can be the model giving up (TB2 46 → 13), so the direction-vs-LoRA comparison needs capability on both sides. Ethan never ran a random-direction row through the same hook.

## Rows
All Qwen/Qwen3.8-27B @ 1d4bf0f, vLLM 0.32.0 with ortho's `scripts/vllm_proj.py` (ekhadley/ortho @ 96d7d27, sha256 59abf11d…), served as Ethan's `scripts/kto_serve_row.sh` serves them: 131072 context, MTP 3 speculative tokens, 64 sequences, `--language-model-only`. `cheatbench/serve_row_kto.sh`, set up by `cheatbench/gpubox_setup_kto.sh`.

| row | served name | what | file (HF eekay/ortho) |
|---|---|---|---|
| base | `cb_base` | no hook | - |
| kto | `cb_kto` | grader_kto projection, one direction, layers 30-63, generated tokens only | `acts/kto_sweep/hooks/kto.safetensors` (sha256 ba8df20d…) |
| b1 | `cb_b1` | the same recipe trained on grader + MBPP | `acts/kto_sweep/hooks/b1.safetensors` (511efd7b…) |
| rand | `cb_rand` | a random unit direction (seed 1) in kto's format, same layers, `c[L] = mean_resid[L+1]·u` from grader_kto (kto's own c is built this way) | `cheatbench/make_rand_hook.py` |
| graderlora | `cb_graderlora` | rank-16 KTO LoRA, grader | `data/loras/Qwen3.8-27B/grader_kto_lora/step_0522` |
| mbpplora | `cb_mbpplora` | rank-16 KTO LoRA, MBPP | `data/loras/Qwen3.8-27B/mbpp_kto_lora/step_0483` |

## Evals (sampling fixed: temperature 1, top_p 1, top_k −1, 8192 tokens per call, `reasoning_effort` low; the rh-envs/CheatBench settings of `mission_capability.md`)
- **Phase A, GSM8K + IFEval** on every row including base (this serving stack differs from the October base run): `lmeval_chat.py` (lm-eval 0.4.9, `gsm8k_cot_llama,ifeval`, chat completions through the tunnel, as `cheatbench`), concurrency 32. ≈ 15-20 min a row.
- **Phase B, Terminal-Bench 2.0** (Harbor @ 4641fd5, Terminus-2, `max_turns` 150, 930 s client timeout, `collect_rollout_details`), all 89 tasks, longest agent timeout first, concurrency 16 for every row. Order: kto, b1, base, rand, graderlora, mbpplora. base is re-run on this GPU (an H100 PCIe, not the H200 of the October base run), because hardware changes agent timeouts. ≈ 1.5-1.7 h a row.
- Full MMLU: out of scope.

## Infrastructure
GPU box `gpubox-llm` (Vast H100 PCIe 80 GB, vLLM on 127.0.0.1:8000) and Hetzner VM `hetzner` (CCX43, Docker, Harbor), set up as in `cheatbench/README.md` with two changes: `gpubox_setup_kto.sh` in place of `gpubox_setup.sh`, and `ROW_SCRIPT=serve_row_kto.sh` for `allow_tunnel_key.sh` (the row key's forced command then accepts base kto b1 rand graderlora mbpplora status). Model-written commands run only inside Terminal-Bench task containers on the VM. GPU-box watchdog: 150 idle minutes.

## Outputs
HF `capability_kto/`: `lmeval_<row>/` (lm-eval results and samples), `tb2_kto-<row>/` (Harbor job dir, without .py/.sh/.ipynb files), `report/` (LOG.md, logs). Table to write on the laptop: row × GSM8K strict/flexible, IFEval prompt-strict/inst-loose, TB2 pass/89 with agent timeouts; TB2 paired by task vs this box's base (sign test / exact McNemar), as `docs/analysis/2026-10-19_significance_and_cosines.md`.

## Stops
`touch /root/capkto-skip-tb` on the VM: no new Terminal-Bench block. `touch /root/capkto-stop-at-end`: stop the tunnel and request the GPU-box stop when the run ends. The Hetzner VM bills until deleted in the console; the Vast box until destroyed.
