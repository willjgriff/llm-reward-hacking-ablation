# Second model, both boxes: TV judge on the pilot rollouts (user, 2026-11-26)

Do this next, before any new Phase 3 eval, then continue where you were.

**Why:** under `verifier_bug` the oracle's `tampered` sees only file changes. In-memory bypasses (`exec(open('verifier.py').read().replace(...))`, importing `verifier` and calling it with matching types) leave the file unchanged; on Qwen the judge found 29 such episodes. Every pilot TV cell failed its gate on the oracle alone, so we need to know whether the judge finds hidden hacking there.

**What:** run the TV judge (the mission's "Judge on the box" settings: stage-3 pipeline, judge env `tv_prompt_dom`, deepseek-v4-flash-0731, effort medium, rubric unchanged) on **every `verifier_bug` rollout this box has collected**, base model only:
- box G: Gemma pilot on mt4 (200) and off mt4 (100);
- box N: Nemotron pilot on mt4 (200) and off mt4 (100), Nemotron's on-mt6 pair collection (all episodes so far), GLM pilot on mt4 (200) and off mt4 (100).

About $0.0005 per rollout. The $15 stop still applies.

**Report** in LOG.md and in a `TV_JUDGE.md` next to `PILOT.md` (uploaded with the report). Give one row per run with:
- n, oracle tampered and passed;
- judge flags by status (considered / attempted / enacted);
- **judge-found bypasses the oracle missed** (enacted or attempted, `tampered` false), with the command quoted.

Add 5 quoted examples of the most serious judge-flagged episodes per model. Gate verdicts stay as they are; the laptop decides what follows.
