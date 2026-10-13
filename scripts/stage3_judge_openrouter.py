#!/usr/bin/env python
"""Stage 3 audit: run an LLM judge over an audit set through OpenRouter (OpenAI-compatible API).

Reads <audit-dir>/pilot_set.json (from stage3_select_audit_set.py), builds each item's prompt (the
rubric for the set's environment + the blind transcript; prompts/NNN.txt is used when present), sends it
with JSON mode, temperature 0 and the requested reasoning effort, verifies every flag's quote against
the transcript, and appends one record per trajectory to
<audit-dir>/judges/<model>__<effort>[__<tag>]/labels.jsonl (flushed per record; trajectories already
there are skipped, so the run can be stopped and restarted). Replies that are not JSON go to
failures.jsonl after two attempts.

  uv run scripts/stage3_judge_openrouter.py --audit-dir data/stage3/audit/mbpp_impossible --dry-run
  uv run scripts/stage3_judge_openrouter.py --audit-dir data/stage3/audit/mbpp_impossible --limit 5
  uv run scripts/stage3_judge_openrouter.py --audit-dir data/stage3/audit/terminal_verifier-all --workers 16

Key: OPENROUTER_API_KEY in the environment, else ~/.config/rhablation/openrouter.env (KEY=value lines).
Never in the repo. Provider routing: cheapest first, providers that train on inputs excluded, and the
request's parameters must be honoured (`require_parameters`), so a provider that ignores the reasoning
effort or JSON mode is not used. The provider that served each call is recorded on the record.
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.judge_audit import SPECS, SYSTEM_PROMPT, build_prompt, load_manifest_classes, load_rubric, parse_reply, verdict_record  # noqa: E402

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
KEY_FILE = Path.home() / ".config" / "rhablation" / "openrouter.env"


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and KEY_FILE.exists():
        for line in KEY_FILE.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(f"no OpenRouter key: set OPENROUTER_API_KEY or put OPENROUTER_API_KEY=... in {KEY_FILE} (chmod 600)")
    return key


def model_pricing(model: str) -> dict | None:
    """USD per token for prompt/completion from OpenRouter's public model list."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=30) as r:
            models = json.load(r)["data"]
    except Exception as e:  # noqa: BLE001
        print(f"  (could not fetch pricing: {e})")
        return None
    for m in models:
        if m.get("id") == model:
            pr = m.get("pricing") or {}
            return {"prompt": float(pr.get("prompt") or 0), "completion": float(pr.get("completion") or 0),
                    "context_length": m.get("context_length"), "name": m.get("name")}
    print(f"  (model {model} not in OpenRouter's list)")
    return None


def load_trajectories(spec, items: list[dict], data_root: Path) -> dict[str, dict]:
    wanted: dict[str, set[int]] = defaultdict(set)
    for it in items:
        wanted[it["source_file"]].add(it["line_no"])
    out: dict[str, dict] = {}
    for sf, line_no, traj in spec.iter_records(data_root):
        lines = wanted.get(str(sf))
        if lines and line_no in lines:
            out[traj["trajectory_id"]] = traj
    missing = [it["trajectory_id"] for it in items if it["trajectory_id"] not in out]
    if missing:
        sys.exit(f"{len(missing)} items not found in their source files, e.g. {missing[:3]}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", choices=("none", "low", "medium", "high"), help="reasoning effort (default: the environment's)")
    p.add_argument("--tag", help="suffix for the judge directory, e.g. rep2 for a self-consistency rerun")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, help="judge at most this many not-yet-judged items")
    p.add_argument("--only", type=int, nargs="+", help="item numbers to judge")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8000, help="output cap; providers count reasoning tokens against it")
    p.add_argument("--quantizations", default="fp8,bf16,fp16,int8",
                   help="comma list of accepted provider quantizations ('' = any). fp4 providers misread transcripts in the first test")
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--manifest", type=Path, default=Path("data/stage4/manifest/tv27b/manifest.jsonl"))
    p.add_argument("--no-provider-prefs", action="store_true", help="debug: let OpenRouter route freely")
    p.add_argument("--dry-run", action="store_true", help="print two prompts and the token/cost estimate; no API calls")
    args = p.parse_args()

    meta = json.loads((args.audit_dir / "pilot_set.json").read_text())
    spec = SPECS[meta["env"]]
    if meta["env"] == "terminal_verifier":
        spec.context["manifest_classes"] = load_manifest_classes(args.manifest)
    effort = args.effort or spec.effort_default
    rubric_text, rubric_sha = load_rubric(spec.rubric)
    if rubric_sha != meta.get("rubric_sha"):
        print(f"WARNING: rubric changed since the set was built ({meta.get('rubric_sha')} -> {rubric_sha}); prompts are rebuilt from the current rubric")
    items = meta["items"]
    if args.only:
        items = [it for it in items if it["n"] in set(args.only)]
    judge_name = f"{args.model.replace('/', '_')}__{effort}" + (f"__{args.tag}" if args.tag else "")
    out_dir = args.audit_dir / "judges" / judge_name
    labels_path, failures_path = out_dir / "labels.jsonl", out_dir / "failures.jsonl"
    done = {json.loads(l)["trajectory_id"] for l in labels_path.open() if l.strip()} if labels_path.exists() else set()
    todo = [it for it in items if it["trajectory_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{meta['env']} / {judge_name}: {len(items)} items, {len(done)} done, {len(todo)} to do")

    trajs = load_trajectories(spec, todo, args.data_root)

    def prompt_for(it: dict) -> str:
        f = args.audit_dir / "prompts" / f"{it['n']:03d}.txt"
        if f.exists() and rubric_sha == meta.get("rubric_sha"):
            return f.read_text()
        return build_prompt(spec, trajs[it["trajectory_id"]], rubric_text)

    pricing = model_pricing(args.model)
    est_prompt_tokens = sum(len(prompt_for(it)) for it in todo) / 3.5
    est_out_tokens = 600 * len(todo)
    est = {"model": args.model, "items": len(todo), "est_prompt_tokens": int(est_prompt_tokens), "est_output_tokens": est_out_tokens,
           "pricing_usd_per_token": pricing, "effort": effort}
    if pricing:
        est["est_cost_usd"] = round(est_prompt_tokens * pricing["prompt"] + est_out_tokens * pricing["completion"], 2)
    print(f"estimate: ~{est_prompt_tokens / 1e6:.2f}M prompt tokens, ~{est_out_tokens / 1e6:.2f}M output tokens"
          + (f", ~${est['est_cost_usd']} at list price (reasoning tokens not included)" if pricing else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cost_estimate.json").write_text(json.dumps(est, indent=1))
    if args.dry_run:
        for it in todo[:2]:
            print(f"\n{'=' * 100}\nITEM n={it['n']} group={it['group']} {it['trajectory_id']}\n{'=' * 100}\n{prompt_for(it)}")
        return

    from openai import OpenAI  # imported here so --dry-run needs no key

    client = OpenAI(base_url=BASE_URL, api_key=load_key(), timeout=args.timeout_s, max_retries=3,
                    default_headers={"HTTP-Referer": "https://github.com/willjgriff", "X-Title": "rhablation stage-3 audit"})
    extra_body: dict = {"usage": {"include": True}}
    if effort != "none":
        extra_body["reasoning"] = {"effort": effort}
    if not args.no_provider_prefs:
        extra_body["provider"] = {"sort": "price", "data_collection": "deny", "require_parameters": True}
        if args.quantizations:
            extra_body["provider"]["quantizations"] = [q.strip() for q in args.quantizations.split(",") if q.strip()]
    lock = threading.Lock()
    stats = Counter()
    cost_total = [0.0]

    # Retry ladder: the model sometimes spends the whole output budget on reasoning and returns nothing
    # (finish_reason "length" with empty content) or returns an empty message. Attempt 2 doubles the
    # budget; attempt 3 turns reasoning off (the environment-level verdict then rests on reading alone).
    def attempt_settings(attempt: int) -> tuple[int, dict]:
        body = dict(extra_body)
        if attempt == 1:
            return args.max_tokens, body
        if attempt == 2:
            return args.max_tokens * 2, body
        body.pop("reasoning", None)
        return args.max_tokens, body

    def call(prompt: str, attempt: int = 1) -> tuple[dict | None, dict]:
        """(parsed verdict or None, envelope with text/usage/provider/error)."""
        t0 = time.time()
        max_tokens, body = attempt_settings(attempt)
        try:
            resp = client.chat.completions.create(
                model=args.model, temperature=args.temperature, max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                extra_body=body,
            )
        except Exception as e:  # noqa: BLE001
            return None, {"error": f"{type(e).__name__}: {str(e)[:500]}", "wall_s": round(time.time() - t0, 1)}
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = resp.usage.model_dump() if resp.usage else {}
        extra = getattr(resp, "model_extra", None) or {}
        env = {"text": text, "usage": usage, "provider": extra.get("provider"), "model_used": resp.model,
               "finish_reason": resp.choices[0].finish_reason if resp.choices else None, "wall_s": round(time.time() - t0, 1),
               "attempt": attempt, "max_tokens": max_tokens, "reasoning": body.get("reasoning")}
        return parse_reply(text), env

    def judge(it: dict) -> None:
        prompt = prompt_for(it)
        parsed, env = None, {}
        for attempt in (1, 2, 3):
            parsed, env = call(prompt, attempt)
            if parsed is not None:
                break
            with lock, failures_path.open("a") as out:
                out.write(json.dumps({"trajectory_id": it["trajectory_id"], "n": it["n"], "attempt": attempt,
                                      "at": datetime.now(timezone.utc).isoformat(), "error": env.get("error"),
                                      "finish_reason": env.get("finish_reason"), "text": (env.get("text") or "")[-1500:]}) + "\n")
            print(f"  n={it['n']}: attempt {attempt} failed: {(env.get('error') or env.get('text') or '')[:160]!r}", flush=True)
        if parsed is None:
            with lock:
                stats["failed"] += 1
            return
        rec = verdict_record(spec, it, trajs[it["trajectory_id"]], parsed, judge_name, rubric_sha)
        cost = (env.get("usage") or {}).get("cost")
        rec.update({"judged_at": datetime.now(timezone.utc).isoformat(), "model": args.model, "model_used": env.get("model_used"),
                    "provider": env.get("provider"), "effort": effort, "temperature": args.temperature,
                    "provider_prefs": extra_body.get("provider"), "max_tokens": env.get("max_tokens"),
                    "attempt": env.get("attempt"), "reasoning_setting": env.get("reasoning"),
                    "usage": env.get("usage"), "cost_usd": cost, "wall_s": env.get("wall_s"), "finish_reason": env.get("finish_reason")})
        with lock, labels_path.open("a") as out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
            stats["done"] += 1
            stats["flags_kept"] += len(rec["flags"])
            stats["flags_discarded"] += len(rec["discarded_flags"])
            if cost:
                cost_total[0] += float(cost)
            n_done = stats["done"]
        summ = rec["summary"]
        brief = summ.get("intent") or summ.get("hack_mechanism") or summ.get("followed")
        print(f"  [{n_done}/{len(todo)}] n={it['n']} {it['group']:<22} strongest={rec['strongest_status']!s:<10} {brief!s:<14} "
              f"flags={len(rec['flags'])}/{len(rec['flags']) + len(rec['discarded_flags'])} {env.get('wall_s')}s "
              f"{env.get('provider') or ''} ${cost or 0:.4f}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(judge, todo))
    wall = time.time() - t0
    kept, disc = stats["flags_kept"], stats["flags_discarded"]
    print(f"\n{stats['done']} judged, {stats['failed']} failed, {wall / 60:.1f} min, cost ${cost_total[0]:.2f} "
          f"(${cost_total[0] / max(stats['done'], 1):.4f} per item); flags discarded by quote/field check {disc}/{kept + disc} "
          f"({disc / max(kept + disc, 1):.0%}) -> {labels_path}")


if __name__ == "__main__":
    main()
