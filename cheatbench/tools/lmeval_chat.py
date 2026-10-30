"""lm-eval 0.4.9 over vLLM chat completions with the eval rows' sampling settings (cheatbench/mission_capability.md).

lm-eval's command line cannot pass nested request fields, so this wrapper patches LocalChatCompletion's request body
before running lm-eval's own CLI:
  - temperature 1, top_p 1, top_k -1, max_tokens 8192, chat_template_kwargs {reasoning_effort: low}: the rh-envs /
    CheatBench settings, overriding each task's generation_kwargs (greedy, short max_gen_toks).
  - no stop strings: the tasks' `until` strings (GSM8K stops on "Q:") would also cut the model's reasoning, which
    vLLM generates before the answer; the chat template's end-of-turn token ends a reply.
  - a reply cut during reasoning has content null; it is scored as an empty answer instead of crashing the run.

Runs as the unprivileged `cheatbench` user on the CheatBench VM, against the GPU box through the tunnel:
  python -I lmeval_chat.py --row base --tasks gsm8k_cot_llama,ifeval --num-concurrent 16 --output-path out/base [--limit 20]
  python -I lmeval_chat.py --self-test     # prints the request body built for one message; no network
"""
import argparse
import json
import sys

SAMPLING = {
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "chat_template_kwargs": {"reasoning_effort": "low"},
}


def patch():
    from lm_eval.models.openai_completions import LocalChatCompletion

    original = LocalChatCompletion._create_payload
    first = []

    def create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        body = original(self, messages, generate=generate, gen_kwargs=dict(gen_kwargs or {}), seed=seed, eos=eos, **kwargs)
        for key in ("stop", "max_gen_toks", "do_sample", "until"):
            body.pop(key, None)
        body.update(json.loads(json.dumps(SAMPLING)))
        if not first:  # log the first request body once (messages shortened), for the mission's effort check
            first.append(1)
            shown = dict(body, messages=[{**m, "content": str(m.get("content"))[:200]} for m in body["messages"]])
            print("lmeval_chat first request body:", json.dumps(shown), file=sys.stderr, flush=True)
        return body

    def parse_generations(outputs, **kwargs):
        res = []
        for out in outputs if isinstance(outputs, list) else [outputs]:
            tmp = [None] * len(out["choices"])
            for choice in out["choices"]:
                tmp[choice["index"]] = choice["message"].get("content") or ""
            res += tmp
        return res

    LocalChatCompletion._create_payload = create_payload
    LocalChatCompletion.parse_generations = staticmethod(parse_generations)
    return LocalChatCompletion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--row", help="served row: base, a3, a3half, a3quarter, g40, rand (model cb_<row>)")
    ap.add_argument("--tasks", default="gsm8k_cot_llama,ifeval")
    ap.add_argument("--num-concurrent", type=int, default=16)
    ap.add_argument("--output-path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--fewshot-as-multiturn", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cls = patch()
    if args.self_test:
        body = cls._create_payload(
            type("M", (), {"model": "cb_test", "_max_gen_toks": 256})(),
            [{"role": "user", "content": "What is 2+2?"}],
            generate=True, gen_kwargs={"until": ["Q:"], "max_gen_toks": 256, "temperature": 0, "do_sample": False},
        )
        assert body["chat_template_kwargs"] == {"reasoning_effort": "low"} and "stop" not in body, body
        assert body["max_tokens"] == 8192 and body["temperature"] == 1.0 and body["top_p"] == 1.0, body
        print("self-test ok:", json.dumps(body))
        return
    if not (args.row and args.output_path):
        ap.error("--row and --output-path are required")

    from lm_eval.__main__ import cli_evaluate

    argv = [
        "lm_eval", "--model", "local-chat-completions",
        "--model_args", ",".join([
            f"model=cb_{args.row}", f"base_url={args.base_url}", f"num_concurrent={args.num_concurrent}",
            "tokenized_requests=False", "tokenizer_backend=None", "max_retries=5", "timeout=1800",
        ]),
        "--tasks", args.tasks, "--apply_chat_template", "--log_samples", "--seed", "0",
        "--output_path", args.output_path,
    ]
    if args.fewshot_as_multiturn:
        argv.append("--fewshot_as_multiturn")
    if args.limit:
        argv += ["--limit", str(args.limit)]
    print("lmeval_chat:", " ".join(argv), "| sampling", json.dumps(SAMPLING), file=sys.stderr, flush=True)
    sys.argv = argv
    cli_evaluate()


if __name__ == "__main__":
    main()
