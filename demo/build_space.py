"""Assemble the Hugging Face Space folder (``build/space/``) from the repo.

    python demo/build_space.py   # galleries from reports/generations/kitsune.jsonl.gz and generations_en/kitsune-en.jsonl.gz

Each gallery is 20 outputs of a released model on its held-out test prompts, chosen by a seeded random draw
stratified by genre × format (not cherry-picked). The same draws feed REPORT.md's sample sections. Japanese
samples carry an English translation written by the maintainer (``reports/translations_jp.json``, D-023).
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from kitsune import versions  # noqa: E402

OUT = ROOT / "build" / "space"
PKG_FILES = [
    "kitsune/__init__.py",
    "kitsune/versions.py",
    "kitsune/en.py",
    "kitsune/taxonomy.py",
    "kitsune/prompts.py",
    "kitsune/data/__init__.py",
    "kitsune/data/filters.py",
    "kitsune/data/policy.py",
]

SPACE_README = f"""---
title: Kitsune Tales
emoji: 🦊
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
license: apache-2.0
models:
  - {versions.HF_MODEL_REPO}
  - {versions.HF_GGUF_REPO}
  - {versions.HF_MODEL_REPO_EN}
  - {versions.HF_GGUF_REPO_EN}
short_description: Fantasy light-novel fiction in Japanese and English
---

<p align="center"><img src="logo.png" width="96" alt="Kitsune Tales logo"></p>

# Kitsune Tales playground

Two LoRA fine-tunes of Gemma 4 E4B that write original, general-audience fantasy light-novel fiction:
`{versions.MODEL_SLUG}` (Japanese) and `{versions.MODEL_SLUG_EN}` (English with Japanese anime themes).

- **Live**: a 4-bit GGUF (`Q4_K_M`) through llama.cpp on this Space's free CPU, about 1 to 3 minutes for a short story.
  Every sampling setting is adjustable (temperature, top-p, top-k, min-p, repetition and presence penalty, max tokens,
  seed); the defaults are the ones used in the evaluation. The panel reports output length against the requested range,
  speed and finish reason, shows the exact prompt, and offers the text as a download.
- **Gallery**: outputs of the released models on held-out test prompts, one seeded random pick per genre and format
  (not selected for quality). Japanese samples include an English translation.
- Requests are screened before generation and outputs after it (general-audience fantasy only).

Results, figures and code: [Site](https://kitsune-tales-qwen.vercel.app) · [GitHub](https://github.com/whoashish115/kitsune-tales-qwen) ·
[W&B](https://wandb.ai/whoashish115-base/kitsune-tales)
"""


def gallery(gen_path: Path, n_per_cell: int = 1, seed: int = 20261001) -> list[dict]:
    """One seeded-random seed-0 output per (primary genre, format) cell, trimmed to 20 in a fixed order."""
    rows = []
    with gzip.open(gen_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["suite"] == "test" and r["seed"] == 0:
                rows.append(r)
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["genres"][0], r["format"]), []).append(r)
    picked = []
    for key in sorted(cells):
        picked += rng.sample(cells[key], min(n_per_cell, len(cells[key])))
    rng.shuffle(picked)
    return [
        {
            "prompt_id": r["prompt_id"],
            "genres": r["genres"],
            "title": r["title"],
            "format": r["format"],
            "passage": r.get("passage"),
            "text": r["text"],
        }
        for r in picked[:20]
    ]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "kitsune" / "data").mkdir(parents=True)
    for rel in PKG_FILES:
        shutil.copy(ROOT / "src" / rel.removeprefix("kitsune/"), OUT / rel)
    shutil.copy(ROOT / "demo" / "app.py", OUT / "app.py")
    shutil.copy(ROOT / "demo" / "requirements.txt", OUT / "requirements.txt")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    tr_path = ROOT / "reports" / "translations_jp.json"
    translations = json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.exists() else {}
    import yaml

    release = yaml.safe_load((ROOT / "configs" / "release.yaml").read_text(encoding="utf-8"))
    for lang, gen in (
        ("jp", ROOT / "reports" / "generations" / f"{release['ja']['system']}.jsonl.gz"),
        ("en", ROOT / "reports" / "generations_en" / f"{release['en']['system']}.jsonl.gz"),
    ):
        if not gen.exists():
            print(f"gallery_{lang}: no generations yet ({gen.relative_to(ROOT)})")
            continue
        g = gallery(gen)
        if lang == "jp":
            for x in g:
                x["translation_en"] = translations.get(x["prompt_id"], "")
        (OUT / f"gallery_{lang}.json").write_text(
            json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (ROOT / "reports" / f"samples_gallery_{lang}.json").write_text(
            json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"gallery_{lang}: {len(g)} samples")
    print(f"space assembled in {OUT}")


if __name__ == "__main__":
    main()
