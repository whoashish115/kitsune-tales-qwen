"""Build the Hugging Face Space (Gradio on ZeroGPU) for both released models.

    python demo/build_hf_space.py [--out ../kitsune-release/huggingface/spaces/kitsune-tales]

The Space runs the bf16 merged models with transformers; every request borrows a ZeroGPU slice for its generation.
It also links the free Colab notebook (notebooks/playground.ipynb), which runs the 4-bit GGUF playground.
Explore data (all 270 held-out prompts per language, released model next to the base model) and the headline numbers
are read from reports/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from kitsune import en, versions
from kitsune.prompts import SYSTEM_PROMPT
from kitsune.taxonomy import FORMATS, GENRES, count_chars

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
GITHUB = "https://github.com/whoashish115/kitsune-tales-qwen"
SITE = "https://kitsune-tales-qwen.vercel.app"
COLAB = "https://colab.research.google.com/github/whoashish115/kitsune-tales-qwen/blob/main/notebooks/playground.ipynb"
HF = "https://huggingface.co"
PKG_FILES = [
    "__init__.py",
    "versions.py",
    "en.py",
    "taxonomy.py",
    "prompts.py",
    "data/__init__.py",
    "data/filters.py",
    "data/policy.py",
]

README = f"""---
title: Kitsune Tales Playground
emoji: 🦊
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
suggested_hardware: zero-a10g
license: apache-2.0
short_description: Japanese and English fantasy light-novel LLM playground
tags:
  - creative-writing
  - light-novel
  - fantasy
  - japanese
  - gemma
  - lora
models:
  - {versions.HF_MODEL_REPO}
  - {versions.HF_MODEL_REPO_EN}
datasets:
  - {versions.HF_DATASET_REPO}
  - {versions.HF_DATASET_REPO_EN}
---

# Kitsune Tales Playground

Write with [`{versions.MODEL_SLUG}`]({HF}/{versions.HF_MODEL_REPO}) (Japanese) and
[`{versions.MODEL_SLUG_EN}`]({HF}/{versions.HF_MODEL_REPO_EN}) (English with Japanese anime themes): genres, title,
format, passage and every sampling setting, streamed on ZeroGPU. Explore all 270 held-out outputs per language next to
the base model's. The same playground also runs in the free [Colab notebook]({COLAB}).

[Site]({SITE}) · [Report]({GITHUB}/blob/main/REPORT.md) · [GitHub]({GITHUB})
"""

REQUIREMENTS = f"""spaces
torch
transformers=={versions.GPU_PACKAGES["transformers"]}
accelerate=={versions.GPU_PACKAGES["accelerate"]}
"""


def _gens(lang: str, system: str) -> dict[str, dict]:
    sub = "generations" if lang == "jp" else "generations_en"
    with gzip.open(R / sub / f"{system}.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    return {r["prompt_id"]: r for r in rows if r["suite"] == "test" and r["seed"] == 0}


def _length(lang: str, text: str) -> int:
    return count_chars(text) if lang == "jp" else en.count_words(text)


def _band(lang: str, fmt: str) -> tuple[int, int]:
    spec = FORMATS[fmt] if lang == "jp" else en.FORMATS_EN[fmt]
    return spec.target_min, spec.target_max


def outputs(lang: str) -> list[dict]:
    ours = _gens(lang, "kitsune-sft" if lang == "jp" else "kitsune-en")
    base = _gens(lang, "base" if lang == "jp" else "base-en")
    tr = json.loads((R / "translations_jp.json").read_text(encoding="utf-8")) if lang == "jp" else {}
    items = []
    for pid in sorted(ours):
        o, b = ours[pid], base.get(pid)
        lo, hi = _band(lang, o["format"])
        lk, lb = _length(lang, o["text"]), _length(lang, b["text"]) if b else 0
        items.append(
            {
                "id": pid,
                "genres": o["genres"],
                "title": o["title"],
                "format": o["format"],
                "passage": o.get("passage") or "",
                "k": o["text"],
                "b": b["text"] if b else "",
                "lk": lk,
                "lb": lb,
                "band": [lo, hi],
                "tr": tr.get(pid, ""),
            }
        )
    return items


def meta() -> dict:
    site = json.loads((R / "site" / "kitsune.json").read_text(encoding="utf-8"))

    def m(lang: str, sid: str, part: str, key: str) -> float:
        s = next(x for x in site["eval"][lang]["systems"] if x["id"] == sid)
        return round(s[part][key]["mean"] * 100, 1)

    glance = [
        ["Japanese stories within the requested length", m("ja", "base", "test", "length_ok"), m("ja", "kitsune-sft", "test", "length_ok"), "270 × 3"],
        ["English stories within the requested length", m("en", "base-en", "test", "length_ok"), m("en", "kitsune-en", "test", "length_ok"), "270 × 3"],
        ["Disallowed requests carried out anyway, JP", m("ja", "base", "policy", "violation_rate_disallowed"), m("ja", "kitsune-sft", "policy", "violation_rate_disallowed"), "45 × 3"],
        ["Disallowed requests carried out anyway, EN", m("en", "base-en", "policy", "violation_rate_disallowed"), m("en", "kitsune-en", "policy", "violation_rate_disallowed"), "45 × 3"],
    ]  # fmt: skip
    return {
        "system": {"jp": SYSTEM_PROMPT, "en": en.SYSTEM_PROMPT_EN},
        "genres": [{"ja": g, "en": en.GENRE_NAME_EN[g]} for g in GENRES],
        "formats": [
            {"key": k, "en": en.FORMAT_NAME_EN[k], "jp": [f.target_min, f.target_max], "enw": [en.FORMATS_EN[k].target_min, en.FORMATS_EN[k].target_max]}
            for k, f in FORMATS.items()
        ],
        "models": {
            "jp": {"slug": versions.MODEL_SLUG, "repo": versions.HF_MODEL_REPO, "gguf": versions.HF_GGUF_REPO, "recipe": "SFT"},
            "en": {"slug": versions.MODEL_SLUG_EN, "repo": versions.HF_MODEL_REPO_EN, "gguf": versions.HF_GGUF_REPO_EN, "recipe": "SFT + DPO"},
        },
        "links": {"site": SITE, "github": GITHUB, "colab": COLAB, "report": f"{GITHUB}/blob/main/REPORT.md", "wandb": site["project"]["links"]["wandb"], "collection": site["project"]["links"]["collection"]},
        "glance": glance,
        "billed": site["budget"]["total_billed"],
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT.parent / "kitsune-release" / "huggingface" / "spaces" / "kitsune-tales",
    )
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "data").mkdir(parents=True)
    shutil.copy(ROOT / "demo" / "space_app.py", a.out / "app.py")
    for rel in PKG_FILES:
        dst = a.out / "kitsune" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "src" / rel, dst)
    (a.out / "data" / "meta.json").write_text(json.dumps(meta(), ensure_ascii=False), encoding="utf-8")
    for lang in ("jp", "en"):
        (a.out / "data" / f"outputs_{lang}.json").write_text(
            json.dumps(outputs(lang), ensure_ascii=False), encoding="utf-8"
        )
    (a.out / "README.md").write_text(README, encoding="utf-8", newline="\n")
    (a.out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8", newline="\n")
    shutil.copy(ROOT / "assets" / "logo.png", a.out / "logo.png")
    print(f"Space written to {a.out}")


if __name__ == "__main__":
    main()
