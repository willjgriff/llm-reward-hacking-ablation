# Mission stage 10 (DRAFT, not approved): decision-sentence directions from forked TerminalVerifier rollouts

Status: draft for the user, 2026-11-05. Nothing here has run. Open questions are at the end.

## Why (team meeting 2026-11-04)

The Grader direction G9 (A3 recipe) removes hacking across environments but costs persistence: Terminal-Bench 2.0 51.7 → 14.6 % (random direction at the same KL: 33.7 %). G9 was extracted from a single-step environment at the response's `cot_last`. This mission builds hacking directions from **multi-step** TerminalVerifier data, at a position chosen by where the decision is expressed, with a contrast that holds the prefix fixed:

- **Forks from the decision point.** Take base rollouts that tampered, cut them just before the decision call (the first call whose command writes `verifier.py`, stage-5e regex), and resample that call. Prefixes whose forks split between tampering and not tampering give pairs with an identical history, so task, length and run-up cancel.
- **Position = the token just before the sentence that expresses the decision** (user decision 2026-11-05). In a tampering fork, the sentence that commits to editing (e.g. "Let me fix the verifier"). In a non-tampering fork, the sentence that declines the edit if the model considered it, otherwise the sentence that commits to the action it takes (e.g. resubmitting). That boundary token has a fixed identity (space, newline, period), like `cot_last`.

The question is whether a direction from this position (a) removes TV tampering, (b) transfers to the other environments, and (c) costs less persistence than G9 does.

## Fixed for the whole mission

- Model `Qwen/Qwen3.8-27B` @ `1d4bf0f`, bf16, reasoning effort low, rh-envs sampling (`qwen3.8_*` configs: temperature 1, top_p 1, no top_k). One H100 or H200.
- Train/eval split: the stage-4 task split (`data/stage4/manifest/tv27b/splits.json`). Prefixes come from **train** task ids only. Every evaluation uses the stage-9 block-B fresh ids or held-out ids, never a prefix's task.
- Edit recipe, fixed and not tuned: the stage-9 A3 recipe (steering, subtract, decoder layers 30–63, α calibrated to first-token harmless KL 0.05, `down_proj` biases, `stage9_bias_checkpoint.py`). The KL-matched random-direction row is the control (seed 1, same window).
- Statistics: every row vs base and vs the random row, paired by task (`scripts/results_significance.py:paired_test`), Holm over the mission's primary tests.

## Phase 1: forks (≤ 2 h)

1. Candidate prefixes: base rollouts of `verifier_bug` mt4 and mt6 from the sweep (`data/sweep/Qwen3.8-27B/terminal_verifier_*`), train split, oracle `tampered`, cut before the decision call. The stage-9 TV proxy (`stage9_tv_proxy.py`, code mirror `75dd6d1a`) already does the cut and the regex classification. Reuse it.
2. **Pilot:** 60 prefixes × 8 forks (each fork = the decision call only, `max_tokens` as in rh-envs). Report the per-fork tamper rate and the share of **mixed prefixes** (≥ 2 tampering and ≥ 2 non-tampering forks).
3. **Full:** draw prefixes until there are 400 mixed prefixes, or for 1.5 h. If fewer than 150 mixed prefixes exist, stop after phase 2 and report. Store every fork as an rh-envs-shaped record with the prefix id, fork index, regex class, and the token ids of prompt and completion.
4. **Decision sentence by rule** (the in-mission position). Split the fork's CoT into sentences, then:
   - **tampering fork:** the first sentence that proposes or commits to changing the verifier (regex over `fix|modify|edit|change|patch|sed|rewrite` near `verifier`), and
   - **non-tampering fork:** the first sentence that declines that, or else the first sentence that commits to the issued command (`submit|run|resubmit|python verifier.py`).
   Record the token index of the token before that sentence. Hand-read 20 forks of each class and report how often the rule picked the right sentence.

## Phase 2: activations (≤ 1.5 h)

One forward pass per fork (HF, vLLM stopped; `stage4_cache_activations.py` with a new aggregation). Save, for layers 0–64:
- `before_decision`: the token before the rule's decision sentence;
- `decision_last`: the last token of that sentence;
- `cot_last`: the `\n` before `</think>` of the decision call (the stage-9 position, as the comparison);
- **every sentence boundary** of the decision call's CoT (the token before each sentence), with its sentence index.

The last item lets the LLM-judge version of the position be computed later on CPU from the judge's sentence index, without another GPU pass. Its size is about forks × sentences × 65 × 5120 × 2 bytes. If that exceeds 100 GB, keep layers 20–64 only and say so.

## Phase 3: directions and the CPU checks

- Per position: the paired difference of means (`stage4_analysis.paired_diff_of_means`, prefix = pair unit), on 80 % of the prefixes. The other 20 % are held out.
- Report per position:
  - held-out projection AUROC, per layer;
  - split-half cosine;
  - cosine to G9, to the stage-7 `D_TV` (`tamper_vs_clean__cot_last`), to R1 and to the random/Spanish-French/refusal baselines (`scripts/cosine_baselines.py`);
  - where TV `possible`, MBPP honest/hack and Grader cheat/non-cheat sit along d̂ (stage-9 `g9_projection` format).
- J-lens readout (`scripts/jlens_readout.py`) of `before_decision` and `cot_last`.

## Phase 4: behaviour (≤ 6 h)

Rows, each a bias-only checkpoint served by vLLM with the plugin:

| row | direction | note |
|---|---|---|
| `S10_before` | `before_decision` | primary |
| `S10_cotlast` | `cot_last` from the same forks | same data, stage-9 position |
| `S10_rand` | random seed 1 | KL-matched control (also the reference for MMLU) |

Base rollouts on these ids exist from stage 9 block B (`stage6/B2_base`); re-roll the base only if the random row falls outside them.

Parts per row (stage-9 block-B ids):
- TV `verifier_bug` mt4 and mt6, 150 × 2;
- `corrupted` mt4, 100 × 2;
- `possible`, 150;
- MBPP, 300 × 2;
- Grader held-out families (16 cells × 128 + 32 clean);
- MMLU.

Report per call the first-edit hazard at mt6 (delay vs removal), and the stage-9 damage rule (honest/validity loss > 5 points).

**Handover to the persistence mission:** write the bias checkpoints of `S10_before` and `S10_cotlast` (recipe JSON plus bias shard) to HF, so `cheatbench/mission_persistence_tb.md` can serve them as Terminal-Bench rows.

## Report, uploads, stop

- `box_report/stage10/LOG.md` and `SUMMARY.md` (headline table, the mixed-prefix yield, rule accuracy from the hand-reading).
- Forks and per-sentence activations to HF under `stage10/`. The activations go to `stage10/acts/`, or stay on the stopped disk if they exceed 100 GB; say which.
- Rows to `stage6/S10_*`.
- Then self-stop as in `mission_mbpp_impossible_update_2.md`. Cap 14 h.

## After the mission (laptop, not the box)

- Run the LLM judge over the forks' decision calls to mark the decision sentence in both classes. The rubric is still to write; it is the stage-3 format with a `decision_sentence` field.
- Compute the judge-position direction from the stored sentence-boundary activations.
- Compare it with the rule position (cosine, held-out AUROC). It goes to behaviour in a later mission only if it differs from `before_decision` (cosine < 0.9).

## Open questions for the user

1. **Rule now, judge later** (above). The alternative is a two-mission split: forks → laptop judge → activations. That delays the behavioural result by one round trip.
2. **One box for both missions, or two in parallel?** The Terminal-Bench rows for X1/R1/X0 do not depend on this mission. Running them in parallel on a second GPU box saves about 6 h of wall time for the same GPU hours.
3. **Edit method.** The draft uses A3-recipe steering, because that is what transferred. Should the mean-centred weight edit be a second arm, or is that Ethan's workstream?
