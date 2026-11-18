# KTO refusal control: Ethan's KTO projection recipe trained on refusal (fresh H100 box)

Status: approved by the user 2026-11-18. Rules: `sweep/CLAUDE.unattended.md` (copied to `/root/.claude/CLAUDE.md`), with the exceptions this mission states. Log in `box_report/kto_refusal/LOG.md`, results in `box_report/kto_refusal/SUMMARY.md`. Design and the reasons for every choice: project `CLAUDE.md`, bullet "KTO refusal control".

## The question
Does Ethan's KTO projection-direction training (`kto_sweep.py`'s recipe `g0`) find a single direction that removes refusal? This is a recovery test of the method on the behaviour abliteration was built for. The undesirable completions are the base model's refusals to heretic's 400 harmful train prompts. The desirable ones are row `refdom`'s compliant answers to the same prompts (`refdom` = Ethan's difference-of-means refusal direction projected out, so a single-direction solution exists by construction). The neutrals are the base model's answers to heretic's 400 harmless train prompts. The references are Ethan's rows (harmful-test refusal, reasoning off / effort low): base 99.5 / 93.5 %, `refdom` 19.5 / 3.3 %, `refdomall` 7.8 / 2.5 %, `refresp` 95.3 / 11.1 %, random rows about base.

## Code
All code is in the user's fork `willjgriff/ortho`, branch `claude` (commit `5394e56` or later): `kto_refusal.py` (docstring = the design), `scripts/kto_refusal_box.sh` (the stages), the `run.py` rows, and the `LORA_ENV=refusal` branch of `train_lora.py`. None of it has run on a GPU yet; its data stage was checked on the laptop against Ethan's records (796 base refusals, 798 compliant answers of which 608 hit the 1,024-token cap and are kept, 100/100/100 held out).

**Git (exception to the rules' `external/<name>/` location):** clone as root into `/root/ortho`, because Ethan's serve scripts expect `~/ortho` and `~/hooks`:
`git clone -b claude https://github.com/willjgriff/ortho.git /root/ortho && cd /root/ortho && git checkout -b box/kto-refusal`.
Push fixes to `box/kto-refusal` only (`git push -u origin box/kto-refusal`), never to `main` or `claude`, never force. Your token reaches `willjgriff/ortho` and `willjgriff/rh-envs` only; this mission needs no change to rh-envs.

**Running as root (exception to "everything through `rhbench-run`"):** the ortho venv, its vLLM server and its evals run as root from `/root/ortho`, as on Ethan's boxes. That is safe here because the only environments this mission runs, `refusal` and the `grader` side-effect runs, execute no model output: they score text. Run no other environment from this clone. The fork's code is the user's and Ethan's (trusted). The HF and GitHub tokens stay in `/root/.config/rhablation/`; never write them into `/root/ortho/.env` or any file in the clone (the downloads it needs are public).

## Stages
Run each as `bash scripts/kto_refusal_box.sh <stage> 2>&1 | tee -a ~/kto_refusal_box.log` from `/root/ortho`, in a tmux session or as a background task. Read the stage's end marker before going on. Rough times on one H100: setup 30–45 min, sample 15 min, train 1.5–2.5 h, check 30 min, suite 2.5–3.5 h, lora 2–3 h.

1. **setup**: venv, serving venv, weights, Ethan's refusal records and vectors from HF `eekay/ortho`. Then `./kto_refusal.py data` must print the counts above, except that the harmless train neutrals are missing until stage 2: until then it stops at an assertion about them, which is expected. Log the counts.
2. **sample**: a smoke eval on the base server, then `refusal38_train_harmless` (400 harmless train prompts, reasoning off). Then `./kto_refusal.py data` must pass and report ~400 neutrals.
3. **train**: prep, then the 2-step smoke config `rksmoke` (memory check), then `rkto`, `rktopt`, `rktoall`, `rkrand`, `rkrandall`, then the export. Record each config's held-out screen line (refusing / compliant gain, AUC, neutral KL) in the LOG.
4. **check**: first-token KL and MMLU under the every-position rows, cosines for all.
5. **suite**: each row served (`~/serve_row.sh <row>`), its six runs (`<row>_harmful`, `_harmless`, `_harmful_low`, `_harmless_low`, `_none`, `_clean`), then `./kto_refusal.py table`. Ethan's base and refdom rows are not re-run: his records (same harness, vLLM 0.32.0 with MTP) are the reference.
6. **lora, conditional.** The rule is fixed and applied by `./kto_refusal.py table`: the LoRA baseline is trained only if no trained KTO row (`rkto`, `rktopt`, `rktoall`) cuts harmful-test refusal (heretic's label) by 30 points or more from base in either reasoning setting. If the table says `LoRA baseline needed: True`, run the `lora` stage. Otherwise skip it and write why in the SUMMARY.

After each of stages 3, 5 and 6, upload (below) before going on.

## When something breaks
- **Operational fixes are yours to make**, recorded in the LOG with the reason, and pushed to `box/kto-refusal` if they touch the clone's code. Examples: memory (`MAX_TOKENS`, `EVAL_TOKENS` in `kto_refusal.py`; vLLM `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs` or dropping MTP in `~/serve_row.sh` on an H100), a crash or bug, a path, a dependency.
- **Do not change the recipe** to get a better result: the configs' hyperparameters (layers, beta, KL weight, lr, steps, k, positions, per-token), the data and its labels, the rows, the evals and the LoRA rule stay as written. If one cannot run as written, run what can, mark the rest "not run" with the reason, and go on.
- A failed eval is retried once by the script (`RUN_FAILED` after that). A row whose server will not start: try the operational fixes, then skip the row and record it.
- **Budget: 10 h from your start, all stages.** If the LoRA would push past it, skip the LoRA and say so.

## Uploads (private repo only)
Everything from this mission goes to the **private** HF repo: `bash scripts/upload_run.sh --private --run-dir <dir> --dest-prefix kto_refusal`, run as root from `/workspace/llm-reward-hacking-ablation`. These are harmful-request completions; never upload them anywhere public, and never into `eekay/ortho`. Stage data-only copies (no `.py`/`.sh`) under `/root/kto_refusal_out/` and upload each directory:
- `records/`: the new files in `/root/ortho/data/inspect/refusal/` and `/root/ortho/data/inspect/grader/`, i.e. those not downloaded from `eekay/ortho` (compare with the download list in the LOG)
- `inspect_logs/`: the `.eval` files this box wrote under `/root/ortho/logs/refusal/` and `/root/ortho/logs/grader/`
- `acts/`: `/root/ortho/acts/kto_refusal/` (`base.pt` included)
- `vectors/`: `/root/ortho/data/vectors/Qwen3.8-27B/refusal_r*` and `/root/hooks/r*.safetensors`
- `loras/`: `/root/ortho/data/loras/Qwen3.8-27B/refusal_kto_lora/` if trained
- `box_logs/`: `~/kto_refusal_box.log`, `~/kto_refusal.log`, `~/run_*.log`, `~/train_lora.log`, `~/vllm_*.log`
- the report: `bash scripts/upload_run.sh --private --run-dir box_report/kto_refusal --dest-prefix kto_refusal`

Verify each upload by listing the repo (names and sizes). Do not delete local data.

## SUMMARY.md
One table, with base and Ethan's rows (from his `rates.json` / records, labelled as his) next to `rkto`, `rktopt`, `rktoall`, `rkrand`, `rkrandall` and, if run, `refusallora`. Columns:
- harmful and harmless test refusal, reasoning off and effort low, by heretic's label and by the first-person label, with Wilson intervals;
- answer length and cap rate;
- harmless first-token KL;
- MMLU vs base (gen-only rows: unchanged by construction, say so);
- the held-out screen (compliant minus refusing gain per token, AUC) and the harmless neutral KL;
- mean and max |cos| with `refusal_dom_all` and `refusal_resp_all` over layers 30–63;
- the grader side-effect rates (`_none`, `_clean`) against base none38 / clean38.

Under the table:
- the LoRA decision and the numbers behind it;
- 3 compliant and 3 refusing answers (first 300 characters) per trained row on harmful test prompts with reasoning off, to show whether "compliant" means helpful or incoherent;
- every deviation from this mission.

No interpretation beyond what the numbers show; the laptop writes the reading.

## Finishing
When the stages are done or skipped:
1. Bring `SUMMARY.md` up to date and add a short "What I would do next".
2. Upload everything above, push any code fixes to `box/kto-refusal`, and verify the uploads.
3. As your last action, follow the rules' "Finishing" section: stop vLLM and every eval, delete `/tmp/rhablation-no-stop` if it exists, run `tmux new -d -s stopnow "bash scripts/stop_box_when_idle.sh --now"` from `/workspace/llm-reward-hacking-ablation`, and write that you did so as the last LOG line.

The box stops as soon as nothing is running; do not wait for an idle window.
