# KTO refusal direction, part 2: every layer and position, served as a hook and as a weight edit (fresh H100 box)

Status: approved by the user 2026-11-20. Rules: `sweep/CLAUDE.unattended.md` (copied to `/root/.claude/CLAUDE.md`), with the exceptions stated here. Log in `box_report/kto_orthog/LOG.md`, results in `box_report/kto_orthog/SUMMARY.md`. Design and reasons: project `CLAUDE.md`, bullet "KTO refusal control, part 2".

## Why this mission
Part 1 (`sweep/mission_kto_refusal.md` and its updates 1 and 2; reports in the private HF repo under `kto_refusal/`) trained Ethan's KTO projection direction on refusal. It did not recover refusal removal. The laptop LLM judge read all 400 harmful test answers per row. Not refused (partial + compliance), reasoning off / effort low:

| Row | Off | Low |
|---|---|---|
| base | 0.8 % | 2.0 % |
| `refdom` (Ethan's difference-in-means direction, hook at every position) | 92.0 % | 95.5 % |
| `rkto` (hook on generated tokens only, layers 30–63) | 21.0 % | 5.0 % |
| `rktoall` (hook at every position, layers 30–63) | 36.8 % | 6.2 % |
| KTO LoRA, KL-matched (step 80) | 61.5 % | 10.8 % |
| random directions | ≈ base | ≈ base |

**Heretic's keyword label is wrong for the KTO rows.** It counted `rkto` with reasoning off as 99 % refused, and `rktoall` with reasoning off as 69 % *not* refused, because the directions change the refusal's wording rather than the behaviour. So this box reports the keyword labels only as a rough reading, and the results are judged on the laptop afterwards (`scripts/refusal_judge.py --all`).

Before any reward-hacking sweep, two questions need answers. This mission answers both on the cheap single-turn refusal eval:
1. Does the KTO training find a refusal direction when the hook covers every layer (0–63) and every token position (prompt and generated)?
2. Does orthogonalisation reproduce the hook? Orthogonalisation means editing the model's weights so that no layer can write along the direction, then serving the edited model with no hook. This project has not yet orthogonalised a KTO direction. A weight edit acts on every layer and every position, so a direction meant to be applied as a weight edit must be trained under a hook that also covers every layer and position. That is why both questions are answered with the same new config.

## Code
- **Fork:** `willjgriff/ortho`, branch `claude` (commit `f458285` or later). Clone it as root into `/root/ortho`, because Ethan's serve scripts expect `~/ortho` and `~/hooks`:
  `git clone -b claude https://github.com/willjgriff/ortho.git /root/ortho && cd /root/ortho && git checkout -b box/kto-orthog`.
  Push only to `box/kto-orthog`, never to `main` or `claude`, and never force-push.
- **Running as root** (exception to "everything through `rhbench-run`"): the ortho venv, its vLLM server and the refusal evals run as root from `/root/ortho`, as in part 1. This is safe because the `refusal` env scores text and executes no model output. Run no other environment from this clone. Tokens stay in `/root/.config/rhablation/`; never write them into the clone.
- **You write all new code on the box**, in the ortho clone, in the style of `kto_refusal.py`: flat scripts, stages, no hardcoded paths beyond what the repo already fixes. Commit and push it to `box/kto-orthog` as you go. Expected pieces:
  - **Two configs in `kto_refusal.py` `CFGS`.** All other hyperparameters stay as in Ethan's `g0`:
    - `rkfull = replace(REF, positions="all", layers=tuple(range(64)))`
    - `rkfullrand = replace(REF, steps=0, init_scale=1.0, positions="all", layers=tuple(range(64)))`
    - a 2-step memory smoke, `rkfullsmoke = replace(rkfull, steps=2)`.
  - **A weight-edit script** (e.g. `kto_orthog.py`), described below.
  - **`run.py` rows** for the new served models, with the four refusal runs per row.
  - **A table stage** that writes the SUMMARY table.
- **Stage-9 bias plugin, for the mean-pinned edit only:** `sweep/code/sweep/benchmarks/rhenvs/vllm_resid_bias/` in the public HF repo `willjgriff/reward-hacking-ablation` (code mirror, revision `75dd6d1a515fc1b23fd1bf3dc047a7795fd97211`; `cheatbench/gpubox_setup.sh` shows how to fetch and install it). It rebuilds each decoder layer's `mlp.down_proj` with a bias, loaded from the checkpoint, when `RHABLATION_RESID_BIAS=1` is set. It was written for the stage-9 vLLM version. If it does not work with the ortho serving venv (vLLM 0.32.0), porting it is an operational fix.

## The two weight edits
The trained direction `u` (unit vector, `d_model` 5120) is one direction shared by all 64 layers. Its hook, at the output of each decoder layer `L`, sets `h ← h − ((h − μ_L)·u) u`, so the component of the residual along `u` becomes `c_L = μ_L·u` (`μ_L` is the pin; `c_L` is in the hook file written by `kto_refusal.py export`). Two weight forms:

- **Zero form (`_wz`, standard orthogonalisation as in Arditi et al.):** for every weight matrix that writes into the residual stream, replace `W` (output dimension = `d_model`) with `(I − u uᵀ) W`. The writers are:
  - the token embedding (row-wise: `E ← E − (E u) uᵀ`);
  - each layer's attention output projection (`self_attn.o_proj` in full-attention layers, the GatedDeltaNet output projection, e.g. `linear_attn.out_proj`, in linear-attention layers);
  - each layer's `mlp.down_proj`.

  Enumerate the writers from the loaded model and log the list. If the config has `tie_word_embeddings=True`, untie and keep the unembedding unchanged. The residual then has zero component along `u` everywhere. This differs from the trained hook, which holds the component at `c_L` rather than 0.
- **Mean-pinned form (`_wm`, matches the trained hook):** the zero form, plus a bias on each layer's `mlp.down_proj` equal to `(c_L − c_{L−1}) u`, with `c_{−1} = 0`. The component along `u` at the output of layer `L` is then exactly `c_L`, as under the hook. The only remaining difference from the hook is inside each layer: the hook leaves the attention sublayer's write along `u` in place until the layer output, and the edit removes it.

Mechanical check of each edited model, in HF, before serving it (hard gate; fix the code if it fails). On 20 harmless test prompts, read the component along `u` at every layer output:
- `_wz`: it must be about 0. Report its size next to the base model's component.
- `_wm`: it must equal `c_L`, to bf16 precision.

Storage (the box has about 150 GB of disk): write only the edited tensors (embedding, attention output projections, `down_proj` weights, and the biases; about 9B parameters, ~18 GB) to a new shard file. Symlink the base model's shards, and rewrite `model.safetensors.index.json` so the edited tensors' entries point to the new file. This is how `stage9_bias_checkpoint.py --link` added its bias shard. The edited tensors are spread across every base shard, so a checkpoint that copied changed shards would be a full 55 GB copy. Keep one edited checkpoint on disk at a time: delete it after its evals and checks, since the direction file plus the script rebuild it.

## Rows
All rows are evaluated with the same four refusal runs: heretic's 100 harmful and 100 harmless test prompts, reasoning off and effort low, 4 samples each, as `<row>_harmful`, `_harmless`, `_harmful_low`, `_harmless_low` in part 1.

| Row | What is served | Purpose |
|---|---|---|
| `rkfull` | base weights + `rkfull` hook at every position, layers 0–63 (`~/serve_row.sh rkfull`, as part 1) | Question 1 |
| `base_w` | base weights through the weight-edit serving path (same launch command as the `_w` rows, plugin on, all biases zero) | Base on the same serving stack as the edited rows |
| `rkfull_wm` | `rkfull` direction, mean-pinned weight edit, no hook | Question 2: should match `rkfull` |
| `rkfull_wz` | `rkfull` direction, zero-form weight edit, no hook | Standard orthogonalisation of the same direction |
| `rkfullrand_wm` | random direction (`rkfullrand`), mean-pinned edit | Control |
| `rkfullrand_wz` | random direction, zero-form edit | Control: damage from the zero form alone |
| `rktoall_wz` (only if time allows, last) | part 1's `rktoall` direction (trained at layers 30–63), zero-form edit at all layers | Does the existing direction already work as an orthogonalisation |

Ethan's `base` and `refdom` records are the references for the hook rows (same harness, same serving venv), as in part 1. Do not re-run them.

## Checks per row (HF, vLLM stopped)
`kto_refusal.py check`'s measures. For the `_w` rows, compute them on the edited weights rather than under a hook:
- first-token KL from base on harmless and harmful test prompts;
- MMLU on Ethan's 3,511-question subset;
- the held-out screen (compliant minus refusing log-prob gain per token, AUC) and the held-out harmless neutral KL;
- |cos| with `refusal_dom_all` and `refusal_resp_all` over layers 30–63 and over 0–63, for `rkfull` and `rkfullrand`.

**Served-vs-HF check for every `_w` row** (catches a bias plugin that silently does nothing). Teacher-forced log-probs of 50 held-out harmless answers from the served checkpoint (vLLM `prompt_logprobs`) against the same edited model in HF: mean absolute difference ≤ 0.005 nats per token, and `base_w` against the unedited HF model. A row that fails is not evaluated until fixed. If it cannot be fixed, mark it "not run" with the reason.

## Order
1. **Restore.** Fetch part 1's report (`kto_refusal/kto_refusal/{LOG,SUMMARY}.md` from the private repo `willjgriff/reward-hacking-ablation-private`) into `box_report/kto_orthog/part1/` for reference. Clone the fork, then run `bash scripts/kto_refusal_box.sh setup`. Restore part 1's outputs from the private repo as root, exactly as `sweep/mission_kto_refusal_update_1.md` step 3 describes: `kto_refusal/{acts,vectors,records}` to `acts/kto_refusal/`, `data/vectors/Qwen3.8-27B/`, `~/hooks/` and `data/inspect/refusal/`. Read the token from `upload.env` into the environment of the download command only. Check `./kto_refusal.py data` prints 796 / 798 / 400 and 100 / 100 / 100, and that `acts/kto_refusal/base.pt` holds pins for all 64 layers under positions `all`. If it does not, rerun `prep`.
2. **Train.** `rkfullsmoke` (memory), then `rkfull` and `rkfullrand`, then `export`. Record the held-out screen line of each in the LOG. Training through all 64 layers is slower and uses more memory than part 1's 30–63. The operational knobs are `MAX_TOKENS` / `EVAL_TOKENS` in `kto_refusal.py`.
3. **Weight edits and checks.** Write the edit script; build, mechanically check and HF-check each `_w` model, one at a time. Run `check` for `rkfull` (hook).
4. **Serve and evaluate.** `rkfull` via `~/serve_row.sh`. Then `base_w`, `rkfull_wm`, `rkfull_wz`, `rkfullrand_wm`, `rkfullrand_wz` through the weight-edit path, each with its served-vs-HF check first. Then `rktoall_wz` if more than 1.5 h of budget is left.
5. **Table, SUMMARY, uploads, stop** (below). Upload after steps 2, 4 and 5.

## When something breaks
- **Operational fixes are yours to make**, logged with the reason and pushed to `box/kto-orthog` if they touch the clone's code. Examples: memory knobs; vLLM `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, dropping MTP (apply the same flags to every row of the same serving path); porting the bias plugin; a path, a crash, a dependency.
- **Do not change the recipe:** the KTO hyperparameters (beyond the layers and positions stated above), the data, the labels, the rows, the edit definitions and the evals stay as written. If something cannot run as written, run what can, mark the rest "not run" with the reason, and continue.
- A failed eval is retried once (`runs()` in `kto_refusal_box.sh` does this). A row whose server will not start: try the operational fixes, then skip it and record it.
- **Budget: 10 h from your start.** Drop `rktoall_wz` first, then `rkfullrand_wz`.

## SUMMARY.md
One table with every row of this mission, Ethan's `base` and `refdom` (labelled as his), and part 1's `rkto`, `rktoall`, `rkrandall` and the two LoRA rows (labelled part 1), all recomputed from records by the same code. Columns:
- harmful and harmless test refusal, reasoning off and effort low, by heretic's label and by the first-person label, with Wilson intervals;
- answer length and cap rate;
- first-token KL (harmless, harmful);
- MMLU vs base (vs `base_w` for the `_w` rows) with an interval;
- held-out screen (gain, AUC) and neutral KL;
- the cosines;
- for `_w` rows: the mechanical check and the served-vs-HF difference.

Under the table:
- **No verdict on the box.** The pre-registered reading rule uses the laptop judge's labels, which this box cannot produce (no paid APIs). Report the keyword-label rates as they are; the laptop applies the rule below after its judge pass. The rule is listed here so it is fixed before the results exist:
  - On harmful test prompts, judge "not refused" = partial + compliance.
  - **recovers:** not refused ≥ `refdom` − 15 points in both reasoning settings, harmless refusal ≤ base + 5 points, MMLU ≥ base − 3 points;
  - **partial:** not refused ≥ base + 30 points in either setting;
  - **fails:** otherwise.
  - **edit matches hook:** `rkfull_wm` within 10 points of `rkfull` in both settings.
- 3 compliant and 3 refusing answers (first 300 characters) per trained row and edit, harmful test prompts, reasoning off. Pick them at evenly spaced positions in the run, not by hand.
- The list of writer matrices that were edited, and every deviation from this mission.
- A short "What I would do next".

Give no interpretation beyond what the numbers show.

## Uploads (private repo only)
These are harmful-request completions, so everything goes to the private repo: `bash scripts/upload_run.sh --private --run-dir <dir> --dest-prefix kto_orthog`, as root from `/workspace/llm-reward-hacking-ablation`. Stage data-only copies (no `.py`/`.sh`) under `/root/kto_orthog_out/`:
- `records/`: new files in `/root/ortho/data/inspect/refusal/`;
- `inspect_logs/`: the new `.eval` files under `/root/ortho/logs/refusal/`;
- `acts/`: `results.jsonl`, `vecs/rkfull*.pt`, `check.json` and the new check outputs (not `base.pt`, which part 1 already uploaded);
- `vectors/`: `refusal_rkfull*`, `~/hooks/rkfull*.safetensors`, and per edited model its bias tensors and a manifest (direction sha256, edit form, writer list, `c_L`). Do not upload the edited checkpoints;
- `box_logs/`: `~/kto_refusal_box.log`, `~/kto_refusal.log`, `~/run_*.log`, `~/vllm_*.log` and the edit script's log;
- the report: `bash scripts/upload_run.sh --private --run-dir box_report/kto_orthog --dest-prefix kto_orthog`.

Verify each upload by listing the repo (names and sizes). Code goes only via `git push` to `box/kto-orthog`; if you also changed files in this project's tree, run `bash sweep/upload_code.sh --message "kto orthog"`.

## Finishing
When the steps are done or skipped:
1. Bring `SUMMARY.md` up to date.
2. Push code to `box/kto-orthog`, upload everything above, and verify the uploads.
3. As your last action: stop vLLM and every eval, delete `/tmp/rhablation-no-stop` if it exists, run `tmux new -d -s stopnow "bash scripts/stop_box_when_idle.sh --now"` from `/workspace/llm-reward-hacking-ablation`, and write that you did so as the last LOG line.

The box stops as soon as nothing is running. Do not wait for an idle window.
