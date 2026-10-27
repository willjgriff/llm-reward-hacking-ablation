#!/usr/bin/env python
"""CheatBench judge v2, deliverable quality: render deliverables and write blind judging units.

Prompt: configs/judge/cheatbench_v2_quality_rubric.md (header comment dropped, criteria from its
"Criteria per task" part). One unit = one prompt for one judge call, in its own directory named by a hash
(no row, model or verdict in any path or file a judge reads); `manifest.jsonl` next to `units/` maps units
back to episodes and is never shown to a judge. Each unit's `verdict.json` is written by the judge (here a
local Claude Sonnet subagent that reads the unit with its file tools, because the OpenRouter judge model has
no image input; user decision 2026-10-27) and read by cheatbench_quality_report.py.

- svgs (`pair` mode): each steered row's drawing vs base's for the same brief, judged twice with the order
  swapped (the two orders go to different judges). `drawing.svg` is rendered by cairosvg at 800x600 as
  CheatBench's leaderboard does (PyMuPDF drops SVG gradients), and also given as source.
- knowledge_work (`single` mode, judged twice): 13 of 15 base deliverables copy the client's protected
  pipeline output (v1 `copied_final` / `copied_or_transformed_competitor` or `submitted_reference > 0`),
  so there is almost no base counterpart to pair with. Only `to_send/` is scored; a .docx/.pptx with a PDF
  of the same stem is scored as the PDF; Office files are converted by a local LibreOffice
  (`--soffice`, run headless with a throwaway profile), pages rendered at 110 DPI (at most --max-pages),
  text extracted. Brief = the GDPval prompt (inputs/ paths as the agent saw them), input files as text (.docx/.pptx from their XML, images as PNG).
  Copied deliverables are `excluded`, episodes with nothing in `to_send/` are `missing`; both are counted
  in the manifest, not judged.

  LD_LIBRARY_PATH=~/opt/lo/usr/lib/aarch64-linux-gnu uv run --group render scripts/cheatbench_quality_prepare.py --runs-root data/cheatbench_hf/cheatbench
"""

import argparse
import ast
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUBRIC = Path(__file__).resolve().parents[1] / "configs" / "judge" / "cheatbench_v2_quality_rubric.md"
TASK_DIRS = {"svgs": "svgs_workshop", "knowledge_work": "knowledge_work"}
STEERED = ("a3", "a3half", "a3quarter", "g40", "rand")


def load_rubric(path: Path) -> tuple[str, dict[str, tuple[str, str]]]:
    """(template, {task: (deliverable kind, criteria markdown)})."""
    text = re.sub(r"\A\s*<!--.*?-->\s*", "", path.read_text(), count=1, flags=re.S)
    template, _, crit = text.partition("\n# Criteria per task\n")
    out = {}
    for block in re.split(r"(?m)^(?=## )", crit):
        if block.startswith("## "):
            lines = block.strip().splitlines()
            kind = lines[1].removeprefix("DELIVERABLE_KIND:").strip()
            out[lines[0][3:].strip()] = (kind, "\n".join(lines[2:]).strip())
    return template.rstrip() + "\n", out


def first_attempts(metrics: Path, task: str) -> dict[tuple[str, str], dict]:
    """(row, key) -> metrics row of the lowest attempt index (the provisional/v1 episode set)."""
    out = {}
    for line in metrics.open():
        r = json.loads(line)
        if r["task"] == task:
            k = (r["row"], r["episode"])
            if k not in out or r["attempt_index"] < out[k]["attempt_index"]:
                out[k] = r
    return out


def v1_verdicts(judge_dir: Path, task_dir: str) -> dict[tuple[str, str, int], dict]:
    out = {}
    for f in judge_dir.glob(f"{task_dir}_cb-*_a3eval/judge.jsonl"):
        for line in f.open():
            r = json.loads(line)
            if r.get("parse_ok"):
                out[(r["row"], r["episode"], r["attempt_index"])] = r
    return out


def render_pdf(pdf: Path, out_prefix: Path, dpi: int, max_pages: int) -> tuple[list[Path], int, str]:
    import pymupdf  # noqa: PLC0415
    doc = pymupdf.open(pdf)
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        p = out_prefix.with_name(f"{out_prefix.name}_p{i + 1:02d}.png")
        page.get_pixmap(dpi=dpi).save(p)
        pages.append(p)
    return pages, len(doc), "\n\n".join(page.get_text() for page in doc)


def to_pdf(src: Path, soffice: str, workdir: Path) -> Path:
    out = workdir / "pdf"
    out.mkdir(exist_ok=True)
    profile = workdir / "lo_profile"
    subprocess.run([soffice, f"-env:UserInstallation=file://{profile}", "--headless", "--norestore", "--convert-to", "pdf",
                    "--outdir", str(out), str(src)], check=True, capture_output=True, timeout=300)
    pdf = out / (src.stem + ".pdf")
    if not pdf.exists():
        raise RuntimeError(f"LibreOffice produced no PDF for {src.name}")
    return pdf


def office_xml_text(path: Path) -> str:
    """Text of a .docx/.pptx from its XML parts (paragraph breaks kept); no Office process involved."""
    import zipfile  # noqa: PLC0415
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if re.fullmatch(r"word/document\.xml|ppt/slides/slide\d+\.xml", n))
        names.sort(key=lambda n: int(re.sub(r"\D", "", n) or 0))
        parts = []
        for n in names:
            xml = z.read(n).decode("utf-8", "replace")
            xml = re.sub(r"</w:p>|</a:p>", "\n", xml)
            parts.append(re.sub(r"<[^>]+>", "", xml))
    import html  # noqa: PLC0415
    return html.unescape("\n\n".join(parts))


def input_text(path: Path, cap: int) -> str | bytes:
    """An input file as text for the judge, or PNG bytes for an image input."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls", ".csv"):
            import pandas as pd  # noqa: PLC0415
            sheets = {"csv": pd.read_csv(path)} if suffix == ".csv" else pd.read_excel(path, sheet_name=None)
            text = "\n\n".join(f"## sheet {name}\n{df.to_csv(index=False)}" for name, df in sheets.items())
        elif suffix == ".pdf":
            import pymupdf  # noqa: PLC0415
            text = "\n\n".join(p.get_text() for p in pymupdf.open(path))
        elif suffix in (".docx", ".pptx"):
            text = office_xml_text(path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            import io  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
            buf = io.BytesIO()
            Image.open(path).convert("RGB").save(buf, "PNG")
            return buf.getvalue()
        else:
            raw = path.read_bytes()
            if b"\0" in raw[:4096]:
                return f"(binary {suffix} file, {len(raw)} bytes; not shown)"
            text = raw.decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        text = f"(could not extract: {type(e).__name__}: {e})"
    return text if len(text) <= cap else text[:cap] + f"\n[... truncated at {cap} of {len(text)} characters ...]"


def pick_files(to_send: Path) -> list[Path]:
    files = sorted(p for p in to_send.rglob("*") if p.is_file())
    pdf_stems = {p.stem for p in files if p.suffix.lower() == ".pdf"}
    return [p for p in files if not (p.suffix.lower() in (".docx", ".pptx", ".doc", ".ppt") and p.stem in pdf_stems)]


def uid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--metrics", type=Path, default=Path("data/cheatbench_judge_v2/metrics.jsonl"), help="cheatbench_judge_v2.py --metrics-only")
    p.add_argument("--v1-judge-dir", type=Path, default=Path("data/cheatbench_judge/deepseek_deepseek-v4-flash-0731__medium"))
    p.add_argument("--gdpval", type=Path, default=Path("data/gdpval"), help="openai/gdpval download (parquet + reference_files/)")
    p.add_argument("--out", type=Path, default=Path("data/cheatbench_quality"))
    p.add_argument("--rubric", type=Path, default=RUBRIC)
    p.add_argument("--soffice", default=str(Path.home() / "opt" / "lo" / "soffice-local"))
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--max-pages", type=int, default=25)
    p.add_argument("--text-cap", type=int, default=20000)
    p.add_argument("--reps", type=int, default=2, help="judgements per knowledge_work deliverable")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out = args.out.resolve()

    template, criteria = load_rubric(args.rubric)
    rng = random.Random(args.seed)
    if (args.out / "units").exists():
        sys.exit(f"{args.out / 'units'} exists; move it away first (verdicts live there)")
    (args.out / "units").mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="cbq-"))
    manifest = []

    def write_unit(task: str, deliverables: list[dict], brief: str, meta: dict, inputs: dict[str, str | bytes] | None = None) -> None:
        u = uid(task, meta["mode"], *(d["episode_ref"] for d in deliverables), str(meta.get("order", "")), str(meta.get("rep", "")))
        d_unit = args.out / "units" / u
        d_unit.mkdir()
        kind, crit = criteria[task]
        listing = []
        for label, d in zip("AB", deliverables):
            files = []
            for src, name in d["files"]:
                shutil.copy2(src, d_unit / f"{label}_{name}")
                files.append(f"{label}_{name}")
            listing.append(f"Deliverable {label}: {d['note']}\n" + "\n".join(f"  - {d_unit / f}" for f in files))
        if inputs:
            (d_unit / "inputs").mkdir()
            shown = []
            for name, content in inputs.items():
                f = d_unit / "inputs" / (f"{name}.png" if isinstance(content, bytes) else f"{name}.txt")
                f.write_bytes(content) if isinstance(content, bytes) else f.write_text(content)
                shown.append(f)
            brief += "\n\nThe input files the brief refers to (as text, images as PNG):\n" + "\n".join(f"  - {f}" for f in shown)
        prompt = (template.replace("{{DELIVERABLE_KIND}}", kind).replace("{{N}}", str(len(deliverables)))
                  .replace("{{CRITERIA}}", crit).replace("{{BRIEF}}", brief).replace("{{DELIVERABLES}}", "\n\n".join(listing)))
        (d_unit / "prompt.md").write_text(prompt)
        manifest.append({"unit": u, "task": task, **meta,
                         "deliverables": {lab: {k: v for k, v in d.items() if k not in ("files", "note")} for lab, d in zip("AB", deliverables)}})

    # svgs: pairs vs base, both orders
    svg = first_attempts(args.metrics, "svgs")
    try:
        import cairosvg  # noqa: PLC0415
    except OSError as e:
        sys.exit(f"cairosvg needs libcairo ({e}); set LD_LIBRARY_PATH to a directory holding libcairo.so.2")

    def svg_deliverable(row: str, key: str) -> dict | None:
        path = args.runs_root / f"{TASK_DIRS['svgs']}_cb-{row}_a3eval" / "trajectories" / key / "deliverables" / "drawing.svg"
        ref = {"row": row, "episode": key, "episode_ref": f"svgs/{row}/{key}"}
        if not path.exists():
            return None
        png = work / f"{uid(row, key)}.png"
        try:  # as CheatBench's leaderboard renders it (svgs/leaderboard_server.py)
            png.write_bytes(cairosvg.svg2png(bytestring=path.read_bytes(), output_width=800, output_height=600, unsafe=False))
        except Exception as e:  # noqa: BLE001
            print(f"  svgs {row} {key}: unrenderable ({e})")
            return None
        return {**ref, "files": [(png, "render.png"), (path, "drawing.svg.txt")], "note": "rendered image and SVG source"}

    counts = {}
    for row in STEERED:
        for (r, key), _ in sorted(svg.items()):
            if r != row:
                continue
            x, b = svg_deliverable(row, key), svg_deliverable("base", key)
            if x is None or b is None:
                side = "steered" if x is None else "base"
                manifest.append({"unit": None, "task": "svgs", "mode": "pair", "row": row, "key": key,
                                 "status": f"unrenderable_{side}"})
                counts[("svgs", row, f"unrenderable_{side}")] = counts.get(("svgs", row, f"unrenderable_{side}"), 0) + 1
                continue
            brief = (args.runs_root / f"{TASK_DIRS['svgs']}_cb-base_a3eval" / "trajectories" / key / "prompt.txt").read_text()
            brief = re.search(r"Please solve the following task\.\s*(.+?)\n\s*\n", brief, re.S).group(1).strip()
            first = rng.choice(("x", "base"))
            for order in (0, 1):
                xfirst = (first == "x") != bool(order)
                pair = [x, b] if xfirst else [b, x]
                write_unit("svgs", pair, brief, {"mode": "pair", "row": row, "key": key, "order": order,
                                                  "base_label": "B" if xfirst else "A", "status": "judged"})
            counts[("svgs", row, "pairs")] = counts.get(("svgs", row, "pairs"), 0) + 1

    # knowledge_work: singles
    import pandas as pd  # noqa: PLC0415
    gdp = pd.read_parquet(next(args.gdpval.glob("data/*.parquet")))
    gdp["key8"] = gdp.task_id.str[:8]
    kw = first_attempts(args.metrics, "knowledge_work")
    v1 = v1_verdicts(args.v1_judge_dir, TASK_DIRS["knowledge_work"])
    input_cache: dict[str, dict[str, str | bytes]] = {}
    for (row, key), m in sorted(kw.items()):
        meta = m["metrics"]
        v = v1.get((row, key, m["attempt_index"]), {})
        ep = {"row": row, "episode": key, "attempt_index": m["attempt_index"], "episode_ref": f"knowledge_work/{row}/{key}"}
        if v.get("copied_final") or v.get("copied_or_transformed_competitor") or (meta.get("submitted_reference") or 0) > 0:
            manifest.append({"unit": None, "task": "knowledge_work", "mode": "single", "row": row, "key": key, "status": "excluded",
                             "deliverables": {"A": ep}})
            counts[("knowledge_work", row, "excluded")] = counts.get(("knowledge_work", row, "excluded"), 0) + 1
            continue
        to_send = args.runs_root / f"knowledge_work_cb-{row}_a3eval" / "trajectories" / key / "deliverables" / "to_send"
        files = pick_files(to_send) if to_send.is_dir() else []
        if not files:
            manifest.append({"unit": None, "task": "knowledge_work", "mode": "single", "row": row, "key": key, "status": "missing",
                             "deliverables": {"A": ep}})
            counts[("knowledge_work", row, "missing")] = counts.get(("knowledge_work", row, "missing"), 0) + 1
            continue
        g = gdp[gdp.key8 == key.split("#")[0]].iloc[0]
        brief = g.prompt.replace("reference_files/", "inputs/")
        if key not in input_cache:
            refs = g.reference_files if not isinstance(g.reference_files, str) else ast.literal_eval(g.reference_files)
            input_cache[key] = {Path(r).name: input_text(args.gdpval / r, args.text_cap) for r in refs}
        out_files, notes = [], []
        for f in files:
            stem = uid(row, key, f.name)
            try:
                pdf = f if f.suffix.lower() == ".pdf" else to_pdf(f, args.soffice, work)
                pages, n_pages, text = render_pdf(pdf, work / stem, args.dpi, args.max_pages)
            except Exception as e:  # noqa: BLE001
                notes.append(f"{f.name}: could not be rendered ({type(e).__name__})")
                continue
            txt = work / f"{stem}.txt"
            txt.write_text(text if len(text) <= args.text_cap else text[: args.text_cap] + "\n[... truncated ...]")
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f.name)
            out_files += [(pg, f"{safe}_page{i + 1:02d}.png") for i, pg in enumerate(pages)] + [(txt, f"{safe}_text.txt")]
            notes.append(f"file `{f.name}` ({n_pages} page(s)" + (f", first {len(pages)} shown" if n_pages > len(pages) else "") + "), page images and extracted text")
        if not out_files:
            manifest.append({"unit": None, "task": "knowledge_work", "mode": "single", "row": row, "key": key, "status": "unrenderable",
                             "deliverables": {"A": ep}})
            counts[("knowledge_work", row, "unrenderable")] = counts.get(("knowledge_work", row, "unrenderable"), 0) + 1
            continue
        for rep in range(args.reps):
            write_unit("knowledge_work", [{**ep, "files": out_files, "note": "; ".join(notes)}], brief,
                       {"mode": "single", "row": row, "key": key, "rep": rep, "status": "judged"}, input_cache[key])
        counts[("knowledge_work", row, "judged")] = counts.get(("knowledge_work", row, "judged"), 0) + 1

    with (args.out / "manifest.jsonl").open("w") as out:
        for m in manifest:
            out.write(json.dumps(m) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    for k in sorted(counts):
        print(k, counts[k])
    print(f"{sum(1 for m in manifest if m['unit'])} units -> {args.out / 'units'}")


if __name__ == "__main__":
    main()
