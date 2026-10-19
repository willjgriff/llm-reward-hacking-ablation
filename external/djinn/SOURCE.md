# external/djinn: EleutherAI's djinn framework, vendored unmodified

Source: https://github.com/EleutherAI/djinn, default branch at commit `b324baad3ec6498a84cd25cc68d0f1acadc4f467`,
shallow-cloned on 2026-10-19 and copied with `rsync -a --exclude .git` (`*.pyc` removed). License: MIT, per
the `pyproject.toml` classifier; the repository has no LICENSE file. No credentials are in the tree: API keys
are read from the environment only (`OPENROUTER_API_KEY`, never set on the box).

Why it is here: the dataset `EleutherAI/djinn-problems-v0.8` (698 train / 43 eval / 370 + 371 alternate
problems, 23 `exploit_type`s, each with a ground-truth solution and a reference exploit) does not contain the
verifiers that decide whether an exploit worked; they are `djinn/verifiers/{secure,insecure}/<exploit_type>.py`
and `djinn/sandbox/`. The box cannot reach GitHub, so the code travels with the repo rsync.

Rules: **never edit anything under `external/djinn/`**; adapt from outside (our own runner, `sys.path` insert of
`external/djinn`). The insecure verifiers execute model-written code: run them only through `rhbench-run`
(unprivileged, no credentials), as every other benchmark. The package's heavy dependencies (`vllm`, `trl`,
`verifiers`, `dspy-ai`, `e2b-code-interpreter`) are for its own training and generation tools and are not
needed to import the verifiers; install only what the verifiers import.
