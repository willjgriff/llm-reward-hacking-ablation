# Orthogonalization Against Reward Hacking

[SPAR project page](https://sparai.org/projects/f26/rec6s8CRgbmKsDlZ1/) · [Project doc](https://docs.google.com/document/d/1jdNhWacPyFmWbkDv_EPhm6EUnsgtRVCdDqkH3u3729c/edit?usp=sharing) · [Project notes](https://docs.google.com/document/d/1I8rx2zz3LTnX0frJaJ5A-KGQvwkEE9rNF0wo-7VbLnc/edit?usp=sharing)

## Overview

This project tests whether **orthogonalization** (activation-space ablation, as used to remove refusal from LLMs) can be used to remove **reward hacking** from production language models, and whether it outperforms **DPO** in two ways:

- Tolerance to low-quality training data
- Sample efficiency

Mentored as part of SPAR.

## Background

Orthogonalization is known to work well for removing refusal, where it is more sample-efficient and more tolerant of noisy data than alternatives like DPO. This project investigates whether the same advantages hold for reward hacking.

## Core Method

1. Take a production model (e.g. `Qwen/Qwen3.5-9B`) **after** all training/RL is complete.
2. Generate paired trajectories: hacked vs. not-hacked, for the same task.
3. Run forward passes, capture activations, compute a difference-of-means direction.
4. Ablate that direction from the model's weights (as `heretic` does for refusal).
5. Compare against a DPO baseline trained on the same data.

## Key Open Questions

| Question | Notes |
| --- | --- |
| Is reward hacking mediated by a single direction? | Unlike refusal (a fast, early-token decision), reward hacking is often long-context / multi-turn, and CoTs rarely state the decision explicitly. Prior: ~50% clean single direction, ~25% messy single direction, ~25% not single-direction (project-ending in the last case). |
| On-policy vs. off-policy data | Orthogonalization shouldn't need on-policy rollouts, but this is untested for agentic reward hacking specifically. |
| Intentional vs. accidental hacking | Need to verify via CoT that hacking reflects a genuine decision, not confusion — accidental hacking has no clean "decision" to ablate. |

## Scope (MVP)

- 2–4 training datasets of hack/no-hack trajectory pairs (varying size, realism, complexity)
- 2–4 reward-hacking evals (varying realism, complexity, egregiousness)
- A tuned orthogonalization pipeline (adapted from `heretic` or similar)
- A tuned DPO pipeline (adapted from e.g. NeMo RL)
- Comparison of eval results: orthogonalization vs. DPO, per dataset

**Stretch goals** (only after MVP): more variation across models/data/methods; testing whether orthogonalization also removes adjacent behaviors (sloppiness, overstating results); testing side effects on other misaligned behaviors (eval awareness, ambitiousness).

## Models

- Start small (4B–8B, e.g. Qwen3 family) to maximize iteration speed.
- Scale up as time/budget allow (likely feasible: <32B; unlikely feasible: frontier-scale).

## Evals

- **Reward hacking:** e.g. ImpossibleBench, EvilGenie — agentic SWE environments with known hacks, measured via LLM judge or programmatic check.
- **Capability degradation:** e.g. MMLU, SWE-Bench — must be unhackable, so degradation isn't confounded with successful reward-hack removal.

## First Step

Run a reward-hacking benchmark (e.g. ImpossibleBench) against a small local model (4–8B) to:

1. Establish baseline hacking propensity.
2. Generate real on-policy hack/no-hack transcripts (doubles as training data).
3. Read CoTs to sanity-check that hacking is intentional, not accidental.

## Output

- LessWrong blogpost in all cases (including negative results).
- Possible workshop/conference paper if results are strong enough.
- Mentees as first authors (equal contribution), mentor as last author.

## Risks

- Reward hacking may not be mediated by a single clean direction (see [Key Open Questions](#key-open-questions)).
- If orthogonalization fails outright or doesn't beat DPO, the project still yields a useful negative result and internal-mechanism insight.

## References

- Arditi et al., [Refusal in LLMs is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) ([blogpost](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction))
- [heretic](https://github.com/p-e-w/heretic) — SOTA orthogonalization library
- Steinhardt, [Research as a Stochastic Decision Process](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html)
