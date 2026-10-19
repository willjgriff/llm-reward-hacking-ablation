# P-values and cosine baselines for stages 5b–8 (2026-10-19)

Action items 1 and 3 from the 16 Oct meeting with Vlad: a p-value on every row, and baselines for the cosine similarities. Everything is recomputed from per-episode outcomes (HF `stage6/`, `stage7/stage7/{f2f3,steer,f4}`), not from the reports' rates.

- Full table (256 comparisons): [significance_2026-10-19/results.md](significance_2026-10-19/results.md), from `scripts/results_significance.py`
- Cosine tables: [cosine_baselines_2026-10-19/results.md](cosine_baselines_2026-10-19/results.md), from `scripts/cosine_baselines.py`

## How the p-values are computed

- **Question tested:** does the edited model's rate differ from the reference model's rate on the same eval set? All tests are two-sided, because several edits *raise* hacking.
- **Primary test: paired by task id.** Both models ran the same tasks, so under "no effect" a task's two outcomes are interchangeable.
  - With one sample per task (Grader, stage-7 MBPP steering), this is the exact McNemar test.
  - With several samples per task (TV has 2, MBPP has 2 or 4), it is a sign-flip permutation test over per-task differences.
  - These tests make no distributional assumptions. They are valid when samples within a task are correlated, and they only need tasks to be independent draws.
- **The rows' own intervals are too narrow.** Every TV "n = 200" is 100 tasks × 2 samples. The reports' Wilson intervals treat the 200 as independent, which overstates precision.
- **Unpaired Fisher exact:** reported next to each test for reference only, because it ignores the pairing.
- **Hook and steering runs (F2, F3, S1):** only per-cell counts were stored, so these use an unpaired permutation test within cells. Being unpaired, it is less powerful.
- **Δ intervals:** 95% bootstrap over tasks.
- **Multiple comparisons:**
  - Holm (controls the chance of *any* false positive) and Benjamini–Hochberg (controls the false-discovery rate), both applied over all 119 primary hack-rate tests.
  - Secondary metrics are unadjusted: honest pass rate, answer validity, and the comparisons against X0.
  - With 119 tests, about 6 would reach raw p < 0.05 by chance alone. **Treat "raw p < 0.05 but Holm ≥ 0.05" as unconfirmed.**
- **Grader rate:** cheating among parseable answers, excluding `pets`, as in the box reports. Validity is tested separately. A pair is dropped when either model's answer is unparseable.
- **MMLU:** only aggregate accuracy ± stderr was stored, so these are unpaired z-tests. They are conservative.
- **The Grader eval is 10 prompts, not a task sample.** Each Grader cell (family × side) is a single fixed prompt, sampled 128 times; D_GR was extracted from rollouts of the same 12 prompts (n = 1024 runs). A Grader p-value says the edited model behaves differently *on these prompts*; it says nothing about new prompts, and in-distribution Grader rows are train = test at the prompt level. Generalisation evidence on the Grader comes only from the LOFO folds (5 families, one or two prompts each). Tuning a method on the Grader therefore needs new prompts for its test set (see the stage-9 mission).

## What holds up (raw p < 0.001 and Holm < 0.05)

1. **Removing tampering within TerminalVerifier works.**
   - Every direction rolled out in 5b, 5e, 7 and 8 lowers `verifier_bug` mt4 tampering (A7 to 26%, the rest to 0–12%), p < 1e-5 each. The random controls do not (C8 p = 0.9, X0 p = 0.31).
   - The 6-turn removal is real: X1T took mt6 from 86.5% to 1.5%.
   - No direction changes `corrupted` (all p ≥ 0.25) except A3c (100 → 18%). A3c also stops the model peeking at the verifier (peek rate 9%), so it is a broader behaviour change, not a cleaner removal.
2. **X2 → TV transfer is real at mt4.**
   - 47.5 → 26.0%: p = 2e-5, Holm 0.001.
   - Fresh ids: 45.0 → 28.5%, p = 2e-4, Holm 0.01.
   - It does not reach mt6 (89.0 vs 86.5%, p = 0.54) or `corrupted`.
3. **MBPP directions reduce MBPP hacking.**
   - Every direction that contains D_MBPP lowers MBPP cheating by 8.3–11 points, p ≤ 6e-5: X6 (D_MBPP alone), X2, X4, X7, F1b, P2.
   - X6, X2, X4, X7 and F1b also raise the honest rate by 8–10 points (secondary, p ≤ 0.001).
4. **A second cross-environment transfer that the box's verdict rule hid: MBPP → Grader.**
   - X6 (D_MBPP alone) lowers Grader cheating 25.0 → 17.1% against base (p = 4e-8, Holm 4e-6), and 29.7 → 17.7% against X0 (p = 2e-16).
   - The box called it "weak" because its rule required a drop of at least 10 points.
   - Against X0, every MBPP-containing direction (X2, X4, X7, F1b, P2) lowers Grader cheating by 4.7–12 points, p ≤ 0.0012.
   - With steering instead of weight ablation, D_GR → MBPP also transfers: F4 subtract D_GR lowers MBPP cheating 17.3 → 7.7% (Holm 0.002). Weight ablation of D_GR on MBPP (X5) does nothing (p = 0.57).
5. **Zero-ablation raises Grader cheating for every direction without an MBPP component.**
   - Against base:
     - X5 (D_GR itself): +19
     - C1 (concept, no hacking data): +19
     - P1: +20
     - P3: +21
     - P4: +13
     - X8: +41
     - X1 (D_TV): +7
     - X3: +6
     - LOFO L3: +19
     - LOFO L5: +35
     - F2 zero-ablation hook: +17
     - All p ≤ 1e-4.
   - The random-direction control X0 also moves the Grader: +4.2 (p = 0.005, Holm 0.28), and validity −5 points (p = 3e-8).
   - **For Grader rows, X0 is the fairer reference than the base model.** The rise is a property of zero-ablation at L40–63, not of D_GR.
6. **P3 (u_MBPP − u_GR) raises TV tampering**, +19.5 points (p = 7e-5, Holm 0.004). The stage-8 SUMMARY labels it "null".
7. **The mean-centred hook works on the Grader.**
   - In-distribution D_GR: 28.5 → 3.7% (p < 1e-5), validity 95 → 92.5% (p = 0.14).
   - Held-out families: letter 51.6 → 1.6% and weekday 53.1 → 26.6%, both p < 1e-4. Sign 17.9 → 1.8% (p = 0.008, not Holm).
   - Parity and pets_listed start at 5–11%, too low to tell.
8. **Steering the Grader is causal in the "add" direction everywhere.**
   - Adding the direction raises cheating by 22–68 points; all p < 0.003 and all but parity survive Holm.
   - Subtracting it: letter 51.6 → 0% and weekday 53.1 → 9.4%, both p < 1e-5 with validity unchanged.

## What does not hold up, or is confounded

- **"Steering works on the Grader" rests on two families, letter and weekday.**
  - S1 (parity + sign): 7/116 base cheats → 0, p = 0.04 (Holm 1). Validity falls from 91% to 71–73% (p < 1e-4).
  - F3 sign: p = 0.0015, Holm 0.09, validity −17 points.
  - F3 parity: no effect on cheating, validity −33 points.
  - On parity and sign, much of the "cheating" removed by steering became unparseable answers. Stage 3 found such answers are often deliberate hedges. Low-base families should not count as steering successes until those answers are read.
- **Dose response is not established.**
  - X2 at w 0.5 vs w 1.0 on TV: 34.0 vs 26.0%, p = 0.08.
  - F1b vs base: p = 0.003, but Holm 0.17.
- **H1 (X2 from half the data):** 34.5% vs base p = 0.007 (Holm 0.37), and vs X2 p = 0.053. Robust to half the data is plausible, not shown.
- **X2, X4, X7, F1b on the Grader against base:** −3 to −4.5 points, p 0.002–0.04, none survives Holm. Against X0 they are −7 to −8 points with p < 1e-6. Which reference applies matters (see point 5 above).
- **R1 raises MBPP hacking:** +4.0 points, p = 0.0017 (the box's sign test gave 0.006). Holm 0.10, so unconfirmed.
- **Zero-ablation LOFO folds:** L1 (parity) +4.3 with p = 0.18 and L4 (weekday) −1.2 with p = 0.78 are chance; L2 (sign) +7.5 with p = 0.03 is unconfirmed. L3 and L5 are real increases.
- **P1 and P2 on TV:** +7.0 (p = 0.17) and −8.0 (p = 0.063), both in the 0.05–0.2 band.
- **MMLU:** B2 −1.24 (p = 0.004), X6 −1.10 (p = 0.01) and X4 −0.85 (p = 0.046) are likely real but small losses, inside the 5-point tolerance. Everything else is within noise.

## Rerun candidates and sizes

These are the rows in the 0.05–0.2 band, plus rows with p < 0.05 that fail Holm. The full list with sizes is in the "Rerun candidates" section of the full table. Sizes are samples per arm for 80% power at the observed effect. They use the unpaired formula with a design effect for the samples per task. Pairing only reduces the size, so these are upper bounds.

| row | claim it carries | Δ now | p | n per arm now | n needed at α 0.05 | n needed at α 0.001 |
|---|---|---|---|---|---|---|
| F1b vs X2 (TV) | dose response | +8.0 | 0.08 | 200 | 584 | 1,272 |
| H1 (TV) | robust to half the data | −13.0 | 0.007 | 200 | 244 | 532 |
| P2 (TV) | MBPP + random ≈ no transfer | −8.0 | 0.063 | 200 | 656 | 1,428 |
| R1 (MBPP) | R1 raises MBPP hacking | +4.0 | 0.0017 | 672 | ≤ 4,000 | ≤ 8,700 |
| X0 (GR) | random control moves the Grader | +4.2 | 0.005 | 1,130 | 1,742 | 3,789 |
| X2 (GR) | X2 lowers Grader cheating vs base | −4.5 | 0.0016 | 1,212 | 1,335 | 2,904 |
| F2 mean-centred, sign fold | mean-centred LOFO transfer | −16.1 | 0.008 | 56 | 53 | 116 |
| S1 / F3 subtract on parity, sign | steering on low-base families | −6 to −18 | 0.002–0.04 | 45–93 | 39–125 | 86–273 |

The F1b vs X2 row compares two edited models rather than an edit and the base; it is a secondary row of the full table.

**Pool limits:**
- TV `verifier_bug` has about 300 eval-split task ids (20% of 1,532). 200 are used (100 in stages 5e/7 and the mt6 runs, 100 fresh for F1a), so about 100 are left. Reaching the larger sizes needs more samples per task as well.
- MBPP has only 168 held-out tasks, and hacks cluster by task. More samples per task shrinks the error more slowly than more tasks would.
- Grader prompts are cheap. The 1,229-sample suite is single-sample per prompt, so going to 4 samples per prompt is easy.

## Cosine similarities: what "low" means here

From the cosine baselines tables:

| comparison | cosine, mean over L40–63 | in random-pair SDs |
|---|---|---|
| two random isotropic vectors, d = 5120 | SD 0.014, mean \|cos\| 0.011, 99th percentile 0.036 | — |
| max \|cos\| over 64 layers, random (what "max (L)" columns report) | median 0.036, 95th percentile 0.047 | — |
| hack directions vs stored random directions | −0.002 to +0.004 | ≤ 0.3 |
| **D_TV vs D_MBPP / D_GR, D_MBPP vs D_GR** | **+0.012 / +0.038 / +0.058** | **0.9 / 2.7 / 4.1** |
| hack directions vs the concept direction (covert vs simple, no hacking data) | −0.012 to −0.048 | 0.8–3.5 |
| unrelated activation directions vs each other (concept vs environment-identity) | −0.064 to +0.070 | up to 5 |
| D_TV vs R1 (same environment, different contrast and position) | +0.36 | 26 |
| same direction from less data (N25 / N100 / half) | 0.91–0.996 | — |

- **Reading:**
  - Cross-environment hack directions are above the isotropic-random level, but no more aligned than an unrelated concept direction is with them, or than unrelated activation directions are with each other. The residual stream is anisotropic, so random isotropic vectors set too low a bar.
  - Within TerminalVerifier, two quite different contrasts share a cosine of 0.36.
  - This fits the data-diversity hypothesis: each environment's hack has its own direction, and no shared "reward hacking" component is visible at these layers.
- **Pooled-direction cosines are fixed by construction.** A pooled direction is the normalised sum of its members' unit vectors, so cos(pool, member) = sqrt((1 + c) / 2), where c is the cosine between the members. With near-orthogonal members that is ≈ 0.71. The 0.6–0.76 member cosines in the stage-7 table match this formula to three decimals, so they carry no information about shared structure.
- **The meeting's 0.014 is the RMS of the random cosine (1/√d), not its expected absolute value (0.011).**
- **Not yet done:** Spanish-vs-French and refusal directions (Vlad's suggestion). They need a GPU forward pass and are in the stage-9 mission.
- **The environment-identity directions are only an upper reference.** These are differences of two environments' non-hack class means. They share a class mean with the hack directions, so for example D_MBPP vs (TV − MBPP) is +0.50. That makes them partly related by construction. The concept direction is the cleaner unrelated baseline.
