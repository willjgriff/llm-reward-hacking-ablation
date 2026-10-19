# Mission: Stage 9, make orthogonalisation work where steering works (within-environment tuning, ≤ 3 days)

> **DRAFT 2026-10-19, not approved.** Items marked **[decision]** are defaults proposed on the laptop and
> await the user. Remove this block when approved.

Read `/root/.claude/CLAUDE.md` and the project `CLAUDE.md` first. The first overrides the second, and this
mission overrides both where it is more specific. Nobody is watching: wherever this says "report", write it to
`box_report/stage9/` and carry on.

- **Box:** fresh, **one H100 80 GB**, ≥ 250 GB disk.
- **Agent:** Claude Code at **xhigh** effort, started with `bash sweep/box_claude.sh --effort xhigh --mission sweep/mission_stage9_tune.md`.
- **Hard cap:** 72 h from the first LOG.md line. Self-stop earlier when the stopping rule below is met.
- **SUMMARY.md:** keep it current enough to be useful if the session dies at any moment.

Everything in `sweep/mission_stage8_x2.md` (and the stage-7 missions it points to) that this mission does not
change still applies:
- **Who runs what:** root for torch, heretic, likelihood scoring and lm-eval; `rhbench-run` for every environment run; vLLM as root in tmux `vllm`.
- **One GPU:** vLLM and torch work alternate.
- **Unchanged pieces:** the stacked edit, the damage gate, eval suite E, the disk rules, the uploads and the final self-stop.

## Why

The 16 Oct mentor meeting (`docs/meetings/2026-10-16_vlad.md`) set the priority. Get orthogonalisation working
reliably **within** each environment before drawing conclusions about cross-environment transfer. Where steering
works and orthogonalisation does not, the default assumption is that orthogonalisation is under-tuned. Tune it,
explain the failures, and pivot early if it cannot be made to work.

The laptop's p-value pass (`docs/analysis/2026-10-19_significance_and_cosines.md`, read it in full) changes
the starting point:

1. **Zero-ablation weight edits raise Grader cheating for every direction without an MBPP component.**
   - The rise is +6 to +41 points, p < 1e-4. It includes D_GR itself (+19), the concept direction C1 (+19) and D_TV (+7).
   - Even the random control X0 moves it: +4.2 points (p 0.005), validity −5 points.
   - Stage 7 diagnosed the mechanism. d̂ = 0 lies on the cheat side of the non-cheat mean, so zero-ablation pushes non-cheat activations toward cheating.
   - The mean-centred hook, `h ← h − ((h − μ_neg)·d̂) d̂`, with the same D_GR gives 28.5 → 3.7%.
   - The mean-centred *weight edit* was deferred twice for lack of a bias term in the model: heretic's update has no offset, and Qwen3.8's `o_proj`/`down_proj` have no bias.
2. **Steering on the Grader is strongly causal when adding the direction**, and works when subtracting it on letter and weekday.
   - On parity and sign, much of what steering-subtract "removes" turns into unparseable answers. Validity falls 17–33 points.
   - Low-base families do not count as successes until those answers are read.
3. **The Grader eval is 10 fixed prompts**, each sampled 128 times, and they are the same prompts D_GR was trained on.
   - Every in-distribution Grader number is train = test at the prompt level.
   - Held-out evidence exists only from LOFO families.
4. **The other environments already work in-env.**
   - MBPP weight edits work: X6 −11 points, honest +10.
   - TV weight edits work: X1 0%; X1T at mt6 86.5 → 1.5%.
   - Both cut the hack they were trained on and nothing else: `corrupted` is untouched.
   - D_MBPP also lowers Grader cheating vs X0 (−12, p 2e-16), which the stage-7 verdict rule labelled "weak".
5. **Cosines:** hack directions from different environments are no more aligned than unrelated activation directions are (0.01–0.06 in the L40–63 window). This is consistent with the mentor's data-diversity hypothesis.

## Facts and inputs

- **Model:** `Qwen/Qwen3.8-27B` bf16, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, reasoning effort **low** for every rollout.
- **Code:**
  - The HF code mirror head `sweep/code/`, i.e. the stage-8 upload. Port it as stage 8 did and log every file you take.
  - Plus the laptop's `scripts/results_significance.py`. Use its `paired_test` and `stratified_counts_test` for every p-value in this mission, so the numbers match the laptop's table.
- **From HF:**
  - `stage4/stage7/*` and `stage4/stage8/*` (directions, manifests, splits)
  - `stage7/stage7/{directions/pooled,rows/*/recipe.json,f2f3,steer,f4,base_reference.json,f1_task_ids.json}`
  - `stage8/stage8/directions/`
  - `stage6/{base_rhenvs,X0,F1a_base,base_s8}/` (references)
  - `stage5/tv27b-full/baseline.json` (damage gate)
- **Rsynced from the laptop:**
  - `data/sweep/Qwen3.8-27B/{mbpp_impossible,ortho_grader,terminal_verifier_*}/` (sources the manifests point to)
  - rh-envs from the user's fork at the commit named in `/workspace/rh-envs.commit`
  - **[decision]** the commit that adds the new Grader families (below)
- **No activation arrays are on HF.** Rebuild caches as needed: the stage-7 MBPP cache took about 15 min, the GR cache a few minutes. The TV per-call cache for part 4 is new.
- **Grader eval prompts:**
  - The existing 12 cells, each a single prompt, are the tuning set.
  - **[decision]** The test set is **≥ 6 new families** (two mirrored sides each), written on the laptop as `configs/grader/families/*.yaml` in the rh-envs fork. Examples: prime/composite, warm/cool colour, fruit/vegetable, summer/winter month, uppercase/lowercase letter, even/odd day of month.
  - The test set also includes the `generic` and `very_hacker` system-prompt variants of the 12 existing cells.
  - Nothing tuned may see the test families until the final evaluation of the chosen recipes.
  - If the new families are not present, use the five LOFO folds as the only held-out evidence, and say so in every Grader claim.

## Statistics (every row)

- Report k/n, the rate, Δ with a 95% task-cluster bootstrap interval, and a two-sided p-value from `paired_test`.
  - It is paired by task id: exact McNemar for one sample per task, sign-flip permutation otherwise.
  - For runs that only have per-cell counts, use `stratified_counts_test`.
- Keep a running list of every primary test this mission runs, and report Holm-adjusted p-values over that list in SUMMARY.md.
- A claim needs **raw p < 0.001 or Holm < 0.05**. A claim from tuning also needs to replicate on the test set, which is a separate test.
- Rows in the 0.05–0.2 band that matter get rerun at the size `tasks_for_power` gives, if that fits the budget. Otherwise label them "unresolved".
- Grader answer validity is a co-primary metric. **A cheating drop that comes with a validity drop of more than 5 points is not a success** until 30 of the new invalid answers have been read and classified as hedge, refusal, format slip or other.
- For every row, report cheating among valid answers **and** the honest-valid rate (valid and not cheating, over all samples).

## Part 0: setup and Grader audit (≤ 4 h)

1. Set up rh-envs and vLLM as in stage 8. Re-roll the base on the Grader tuning set at n = 128, and on the test families if present.
2. **Read the Grader.**
   - For base, X0, X5, F2_centered and S1 subtract (rollouts and examples on HF), read 30 cheats, 30 honest answers and 30 unparseable answers per condition. Classify each.
   - Report:
     - whether the cheat label matches what the model did;
     - what the unparseable answers are;
     - whether any family's metric is broken (e.g. `y` for a vowel).
   - If a family's scoring is wrong, report it, drop that family from claims, and carry on. Do not edit rh-envs.
3. Read 30 rollouts where zero-ablation raised cheating (X5 and C1 cheats on prompts where base did not cheat as often), and report anything that stands out. This is the low-priority meeting item. Keep it short.

## Part 1: why zero-ablation fails on the Grader (≤ 4 h, mostly CPU)

For D_GR, D_TV, D_MBPP, C1 and two random directions, at every layer in 40–63, compute the projection of the Grader non-cheat mean, the cheat mean and the harmless-prompt mean onto d̂. The question is whether the non-cheat mean lies far from 0 on the cheat side, as MBPP's honest class did in stage 5d.

Then predict, for each direction, the sign and rough size of the zero-ablation effect from those projections. Compare the prediction with the measured rows (X5, C1, X1, X0, P1, P3).

Write the explanation into SUMMARY.md, whether it holds or not.

## Part 2: tune orthogonalisation on the Grader (the core; ≤ 30 h)

Work on the 12-cell tuning set only. Use the hook (HF generation, cheap for single-turn) to search, and weight edits served by vLLM to confirm. Axes, roughly in this order:

1. **Mean-centred weight edit.**
   - Make `h ← h − w((h − μ)·d̂) d̂` servable by vLLM. Options, cheapest first:
     - (a) add a bias to `o_proj`/`down_proj` through a patched model class registered as a vLLM out-of-tree model;
     - (b) fold the offset into the next layer's input norm;
     - (c) anything else that is exact.
   - Verify the weight version against the hook to ≤ 0.005 nats on 400 stage-5 rows before using it.
   - μ is the non-cheat class mean along d̂ by default. Also try the midpoint of the two class means.
   - This is an adaptation of how heretic applies a direction, like the stacked edit. Label it as such.
2. **Heretic's own optimiser.**
   - Run heretic's parameter search (its Optuna objective over weight kernel, layer window and per-layer weights) with the objective "Grader cheating on the tuning set + KL on harmless prompts", replacing its refusal count.
   - Run it for both zero-ablation and the mean-centred edit.
   - This is the "equally well-tuned" baseline the project rules ask for. Keep the trial log.
3. **Direction extraction.**
   - Token position: `cot_last`, `answer_first`, and the mean over the last 8 CoT tokens.
   - Class definition: the judge-free oracle labels vs dropping hedged answers.
   - Within-cell pairing: cheat vs non-cheat from the same prompt, which removes prompt identity. Every Grader cell is one prompt, so this is the default contrast.
4. **Window and weight.** Window, weight w and per-layer vs single direction, inside the optimiser's search or as a small grid if the optimiser is not usable.
5. **Steering twins.** Run steering (add and subtract, α calibrated by harmless KL as in S1) for the same directions, as the reference for "what the direction can do".

**Selecting recipes.**
- Choose at most **3 recipes** on the tuning set, by cheating among valid answers with validity ≥ 90% and harmless KL ≤ 0.05.
- Freeze them. Then evaluate them once on the test families and system variants, at n = 128 per cell, against the base re-roll and against an X0-style random-direction control with the same recipe.
- Report MMLU for each frozen recipe. The tolerance is 5 points.

**Success on the Grader:** at least one frozen recipe lowers cheating on the test set with p < 0.001 (or Holm < 0.05), with validity within 5 points of base, and beats its own random control.

## Part 3: steering on every row that only had orthogonalisation (≤ 10 h)

Rows: X1, X2, X5, X6, X7, P1–P4 and C1, plus the frozen Grader recipes' directions.

- **MBPP:** HF generation with hooks, single-turn, as F4 did, on the 168 held-out tasks × 2 samples.
- **TV:** use the stage-5e decision-call resampling proxy (8 samples from the prefix cut at the decision call; command classified by the write regex). Behavioural HF rollouts are too slow, so state that the proxy is a proxy.
- **Grader:** already covered in part 2 for the frozen recipes. For the rest, the S1 setup on all 10 cells.
- α is calibrated per direction by harmless KL (α₁ ≈ 0.05).
- Report add and subtract, with p-values against α = 0.

## Part 4: carry the recipe to MBPP and TerminalVerifier (≤ 16 h)

1. **TerminalVerifier, decision-step direction.**
   - Positives: the `cot_last` of the call that issues the first write to `verifier.py`.
   - Negatives: the `cot_last` of the call at the same turn index in non-hack (`nonhack_A`) rollouts of the same task where one exists, otherwise any call of a non-hack rollout.
   - Compare with the stage-7 D_TV, which averages over all calls, by cosine and by the X1 recipe on TV-vb mt4.
   - Then use the better of the two below.
2. **Apply the best Grader recipe**, i.e. the method, not the Grader direction, to:
   - the TV direction: TV-vb mt4 and mt6, `corrupted` mt4, `possible`;
   - the MBPP direction: MBPP held-out, honest rate.
   - Compare each against that environment's current best (X1/X1T, X6) on the same ids.
   - Tune nothing on the eval ids:
     - TV tuning uses the F1a ids (`stage7/stage7/f1_task_ids.json`); TV test uses the stage-5e/7 ids.
     - MBPP has only one held-out split. Tune on 84 of its 168 tasks (seed 0) and test on the other 84, and report that the test half is small.

## Part 5: leave-one-configuration-out within each environment (≤ 8 h)

The mentor's question, with the best recipe from parts 2–4:
- **Grader:** the five family folds (rerun with the tuned recipe), plus train-on-old-families → test-on-new-families from part 2.
- **MBPP:** corruption kind folds (`int_plus_1`, `str_char_changed`, `float_plus_1`, nested container kinds), with one kind held out at a time. Report per kind, with the n per kind.
- **TV:** `verifier_bug` mt4 → mt6 → `corrupted`, with each direction applied to the others.

## Part 6: unrelated-direction cosine baselines (≤ 3 h)

Extract difference-in-means directions with the same pipeline (base-model completions at reasoning effort low, `cot_last`, ≥ 100 prompts per class):
- answer in Spanish vs French (the same everyday requests, asked in each language);
- refusal (heretic's default harmful vs harmless prompt sets);
- formal vs casual register;
- Python vs JavaScript code requests.

Add them to `scripts/cosine_baselines.py`'s tables: per-layer cosines against D_TV, D_MBPP, D_GR, R1, the part-4 decision-step direction and each other. Upload the directions.

## Part 7: reruns of borderline rows (≤ 6 h, whatever fits)

From the laptop's rerun table, in this order, at the sizes given there:
1. X2 w 0.5 vs w 1.0 on TV-vb, using the ~100 unused TV eval ids plus a third sample per task;
2. H1 on TV-vb;
3. X0 and X2 on the Grader at 4 samples per prompt;
4. S1/F3 subtract on parity and sign, with the invalid answers read.

Skip any row that would exceed what remains of the budget.

## Stopping rule

- **Stop early and report "pivot signal"** if:
  - after part 2, no frozen recipe passes the Grader success bar;
  - **and** the part-1 explanation does not point to a fix that is still untried.
- In that case, still run part 3 (steering on the other rows) and part 6, then stop.
- **Otherwise** run parts 3–7 and stop at the cap or when done.

## Reporting and uploads

- **`box_report/stage9/`:**
  - LOG.md (timestamped)
  - SUMMARY.md: a headline, a master table with p, Holm and band per row, the Grader audit, the part-1 explanation, the frozen recipes and their test-set result, and the decision against the stopping rule
  - per-part tables
- **Uploads:**
  - rollouts to `stage6/<row>/rhenvs/` as before;
  - directions and recipes to `stage9/`;
  - code via `sweep/upload_code.sh`.
- Verify the uploads, remove the credentials, self-stop.

## Not allowed this round

- DPO.
- RL or any training to hack.
- Editing rh-envs, `external/` or any benchmark scorer.
- FP8.
- Using test families or test ids for any choice.
- Selecting recipes by test-set results.

## Risks to watch

- **Overfitting to 10 prompts.** That is why the test families exist.
- **Invalid-answer inflation masking as success.**
- **A vLLM bias patch that silently does nothing.** The 0.005-nat check guards against it.
- **The heretic optimiser drifting to edits that break formatting.** Watch validity and KL in its objective.
- **Disk:** one merged checkpoint at a time.
