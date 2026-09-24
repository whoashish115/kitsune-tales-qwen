"""Hugging Face release: repository cards and uploads.

    python -m kitsune.hub stage          # write the local mirror of every Hub repo (cards, figures, logo, local files)
    python -m kitsune.hub sync           # create the repos (private) and upload each local mirror
    python -m kitsune.hub collection     # create or update the private collection

Local mirror (``$KITSUNE_RELEASE``, default ``../kitsune-release``):

    huggingface/models/<repo>/      one folder per model repo (merged, LoRA, GGUF)
    huggingface/datasets/<repo>/    one folder per dataset repo
    huggingface/spaces/<repo>/      the playground
    kitsune-tales-qwen-site/        the website repository

Every repo is created private; making it public is a separate, manual step on the Hub. Weights that live on
the Modal volume (merged models, the English LoRA, GGUF files) are uploaded by ``modal_app.hf_upload``; this module
uploads the cards next to them, so text and weights can be updated independently.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kitsune import versions

R = Path("reports")
RELEASE = Path(os.environ.get("KITSUNE_RELEASE", "../kitsune-release"))
GITHUB = "https://github.com/whoashish115/kitsune-tales-qwen"
SITE = "https://kitsune-tales-qwen.vercel.app"
HF = "https://huggingface.co"

REPOS = {
    "jp": versions.HF_MODEL_REPO,
    "jp-lora": versions.HF_ADAPTER_REPO,
    "jp-gguf": versions.HF_GGUF_REPO,
    "en": versions.HF_MODEL_REPO_EN,
    "en-lora": versions.HF_ADAPTER_REPO_EN,
    "en-gguf": versions.HF_GGUF_REPO_EN,
    "data-jp": versions.HF_DATASET_REPO,
    "data-en": versions.HF_DATASET_REPO_EN,
    "space": versions.HF_SPACE_REPO,
}
KIND = {"data-jp": "dataset", "data-en": "dataset", "space": "space"}
FIGS_MODEL = ["eval_metrics", "eval_lengths", "judge_preference", "safety", "sft_loss", "dpo_training"]
FIGS_DATA = ["data_funnel", "data_rejections"]


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _links(lang: str) -> str:
    k = "jp" if lang == "ja" else "en"
    items = [
        ("Site", SITE),
        ("GitHub", GITHUB),
        ("Report", f"{GITHUB}/blob/main/REPORT.md"),
        ("W&B", f"https://wandb.ai/whoashish115-base/{versions.WANDB_PROJECT}"),
        ("Demo", f"{HF}/spaces/{REPOS['space']}"),
        ("Weights", f"{HF}/{REPOS[k]}"),
        ("LoRA", f"{HF}/{REPOS[k + '-lora']}"),
        ("GGUF", f"{HF}/{REPOS[k + '-gguf']}"),
        ("Dataset", f"{HF}/datasets/{REPOS['data-' + k]}"),
    ]
    return " | ".join(f"[{a}]({b})" for a, b in items)


def _header(title: str, subtitle: str, lang: str) -> str:
    logo = '<img src="logo.png" width="44" alt="Kitsune Tales logo" align="absmiddle">'
    return f"# {logo} {title}\n\n{subtitle}\n\n{_links(lang)}\n"


def _split_front(md: str) -> tuple[str, str]:
    """(YAML front matter including the fences, body)."""
    if md.startswith("---"):
        end = md.index("\n---", 3) + 4
        return md[:end], md[end:].lstrip("\n")
    return "", md


def _figures(names: list[str]) -> str:
    captions = {
        "eval_metrics": "Automatic metrics with 95 % bootstrap CIs for every evaluated system.",
        "eval_lengths": "Output length against the requested range (shaded), all test generations.",
        "judge_preference": "Pairwise judge: net preference on full outputs (grey) and equal-length openings (blue).",
        "safety": "Held-out disallowed requests: refusals, safe redirects and violations.",
        "sft_loss": "SFT training and validation loss.",
        "dpo_training": "DPO loss, held-out preference accuracy and reward margin.",
        "data_funnel": "Synthetic data funnel.",
        "data_rejections": "The ten most frequent rejection reasons.",
    }
    return "\n".join(f"![{captions[n]}](figures/{n}.png)\n*{captions[n]}*\n" for n in names)


# ---------------------------------------------------------------------------------------------------------- models


def merged_card(lang: str) -> str:
    src = Path("docs/MODEL_CARD.md" if lang == "ja" else "docs/MODEL_CARD_EN.md").read_text(encoding="utf-8")
    front, body = _split_front(src)
    lines = body.splitlines()
    body = "\n".join(line for line in lines if not line.startswith("[GitHub]("))
    head = _header(
        "kitsune-tales-e4b-jp" if lang == "ja" else "kitsune-tales-e4b-en",
        "Original fantasy light-novel fiction in Japanese. LoRA SFT of Gemma 4 E4B, merged to bf16."
        if lang == "ja"
        else "Original fantasy fiction in English with Japanese anime and light-novel themes. LoRA SFT + DPO of Gemma 4 E4B, merged to bf16.",
        lang,
    )
    body = body.split("\n", 1)[1] if body.startswith("# ") else body
    figs = "\n## Figures\n\n" + _figures(FIGS_MODEL)
    marker = "## Intended use"
    body = body.replace(marker, figs + "\n" + marker, 1) if marker in body else body + figs
    return f"{front}\n\n{head}\n{body}"


def lora_card(lang: str) -> str:
    k = "jp" if lang == "ja" else "en"
    run = "sft-main" if lang == "ja" else "dpo-en-main"
    n_train = _load(R / "data" / "stats.json")["n_train"]
    front = f"""---
license: apache-2.0
base_model: {versions.BASE_MODEL}
library_name: peft
pipeline_tag: text-generation
language:
- {"ja" if lang == "ja" else "en"}
datasets:
- {REPOS["data-" + k]}
tags:
- lora
- peft
- gemma4
- creative-writing
- light-novel
- fantasy
---"""
    recipe = (
        f"Supervised fine-tuning on {n_train:,} examples, one epoch."
        if lang == "ja"
        else "Supervised fine-tuning on 6,647 examples, then DPO on 1,596 preference pairs (1,168 judge-labeled, 293 rule-based, 135 refusal pairs), both one epoch. This repo holds the DPO adapter, which already includes the SFT update."
    )
    return f"""{front}

{_header(f"kitsune-tales-e4b-{k}-lora", f"LoRA adapter (run `{run}`) behind [{REPOS[k]}]({HF}/{REPOS[k]}). The merged model card has the full evaluation.", lang)}
## Adapter

| | |
|---|---|
| Base model | `{versions.BASE_MODEL}` @ `{versions.BASE_REVISION[:12]}` |
| Rank / alpha / dropout | 32 / 64 / 0.05 |
| Target modules | every linear layer of the language model (attention and MLP) |
| Trainable parameters | 77.8M (0.97 % of the checkpoint) |
| Precision | bf16 weights and adapter, max length 2,048 tokens |
| Recipe | {recipe} |
| Optimizer | AdamW, cosine schedule, lr 2e-4 (SFT) / 2e-5 (DPO, beta 0.1), effective batch 16, seed 42 |

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{versions.BASE_MODEL}", revision="{versions.BASE_REVISION}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{REPOS[k + "-lora"]}")
tok = AutoTokenizer.from_pretrained("{REPOS[k + "-lora"]}")
```

Use the system prompt and request format from the [merged model card]({HF}/{REPOS[k]}); the evaluation used temperature 0.8,
top-p 0.95, top-k 50 and repetition penalty 1.05.

## Figures

{_figures(["sft_loss", "dpo_training"] if lang == "en" else ["sft_loss"])}
"""


def gguf_card(lang: str) -> str:
    k = "jp" if lang == "ja" else "en"
    g = (
        _load(R / f"gguf_{'en' if lang == 'en' else 'jp'}.json")
        if (R / f"gguf_{'en' if lang == 'en' else 'jp'}.json").exists()
        else {}
    )
    files = g.get("files", {})
    rows = "\n".join(
        f"| `{f}` | {v.get('gb', '')} GB | `{str(v.get('sha256', ''))[:16]}` |" for f, v in files.items()
    )
    front = f"""---
license: apache-2.0
base_model: {REPOS[k]}
pipeline_tag: text-generation
language:
- {"ja" if lang == "ja" else "en"}
tags:
- gguf
- llama.cpp
- gemma4
- creative-writing
- light-novel
- fantasy
---"""
    return f"""{front}

{_header(f"kitsune-tales-e4b-{k}-gguf", f"llama.cpp GGUF quantizations of [{REPOS[k]}]({HF}/{REPOS[k]}), converted with llama.cpp commit `7fe450e` and smoke-tested on CPU.", lang)}
## Files

| File | Size | SHA-256 (prefix) |
|---|---|---|
{rows}

Q4_K_M is the one the demo Space runs. Q8_0 is close to the bf16 model. The evaluation numbers in the merged model card
were measured on the bf16 model, not on these files.

## Run

```bash
llama-cli -m kitsune-tales-e4b-{k}-Q4_K_M.gguf -st --temp 0.8 --top-p 0.95 --top-k 50 --repeat-penalty 1.05 -n 900 -p "<prompt>"
```

The prompt is the Gemma 4 chat format with the system prompt from the merged model card:
`<|turn>system\\n{{system}}<turn|>\\n<|turn>user\\n{{request}}<turn|>\\n<|turn>model\\n`.
"""


# ---------------------------------------------------------------------------------------------------------- datasets


def dataset_card(lang: str) -> str:
    k = "jp" if lang == "ja" else "en"
    st = _load(R / ("data" if lang == "ja" else "data_en") / "stats.json")
    gens, kept = st["n_generations"], st["n_kept_synthetic"]
    unit = "characters" if lang == "ja" else "words"
    lengths = (
        "synopsis 200 to 500, short story 800 to 1,500, continuation 400 to 800"
        if lang == "ja"
        else "synopsis 150 to 350, short story 600 to 1,100, continuation 300 to 600"
    )
    front = f"""---
license: apache-2.0
language:
- {"ja" if lang == "ja" else "en"}
task_categories:
- text-generation
tags:
- synthetic
- creative-writing
- light-novel
- fantasy
- sft
size_categories:
- 1K<n<10K
pretty_name: Kitsune Tales {"Japanese" if lang == "ja" else "English"} fantasy SFT
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
---"""
    by_gen = st["by_generator_kept"]
    extra = (
        ""
        if lang == "ja"
        else """
**Name rebalancing.** The first generator reused a few invented names ("Elara" in 67 % of its stories). Before
filtering, overused names are replaced deterministically per sample from gender-matched pools, consistently across
passage and continuation and never inside the title. Afterwards the most frequent invented name appears in 7.7 % of
training stories.
"""
    )
    return f"""{front}

{_header(REPOS["data-" + k].split("/")[1], f"Synthetic instruction data for original, general-audience fantasy light-novel fiction in {'Japanese' if lang == 'ja' else 'English with Japanese anime themes'}. Used to train [{REPOS[k]}]({HF}/{REPOS[k]}).", lang)}
## Summary

| | |
|---|---|
| Train / validation | {st["n_train"]:,} / {st["n_val"]:,} examples |
| Synthetic stories kept | {kept:,} of {gens:,} generations ({kept / gens:.0%}) |
| Generators | Qwen/Qwen3.6-35B-A3B-FP8 ({by_gen.get("gen1", 0):,} kept), google/gemma-4-26B-A4B-it ({by_gen.get("gen2", 0):,} kept), both Apache-2.0 |
| Labels | every kept story was labeled by the other generator (cross-labeling) |
| Policy examples | {st["n_refusal_templates"]} templated refusals and off-genre redirects, in train |
| Formats and target lengths ({unit}) | {lengths} |
| SHA-256 | train `{st["sha256"]["train.jsonl"][:16]}`, validation `{st["sha256"]["val.jsonl"][:16]}` |

## Schema

One JSON object per line: `prompt` (the user request: genres, title, format, and the passage for continuations),
`response` (the story), `genres`, `title`, `format`, `language`, `generator` (model and revision), `filters_passed`,
`source` (`synthetic` or `policy`), `hash`, `id` and `meta` (generation settings).

## Construction

1. Requests are sampled from a fixed taxonomy of nine genres and three formats; titles come from templates or are
   brainstormed by the generators. Titles close to any held-out test title are removed.
2. Each generator writes stories; the other generator labels each story (fantasy, general audience, real people or
   existing IP, genre and title fit, quality).
3. Rule filters check script purity, length, repetition, markdown and meta text, safety and PII, real people and IP
   blocklists, and genre-tag consistency; MinHash removes near-duplicates. A story is kept only if it passes both the
   rules and the labels.
{extra}
## Filtering

![Data funnel](figures/data_funnel.png)

![Rejection reasons](figures/data_rejections.png)

## Intended use

For research on small-model specialisation and for fine-tuning story models. The stories are machine-written and carry
the generators' clichés; the lexicon-based safety filters miss paraphrases. Held-out test prompts are not included:
they live in the [GitHub repository]({GITHUB}) (`data/test_prompts*.jsonl`) with their SHA-256 hashes.

## License

Apache-2.0. The data was generated with Apache-2.0 models whose licenses allow training other models on their outputs.
"""


def space_readme() -> str:
    from importlib import util

    spec = util.spec_from_file_location("build_space", Path("demo/build_space.py"))
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.SPACE_README


# ---------------------------------------------------------------------------------------------------------- staging


def local_dir(key: str) -> Path:
    """Local mirror of one Hub repo."""
    sub = {"dataset": "datasets", "space": "spaces"}.get(KIND.get(key, "model"), "models")
    return RELEASE / "huggingface" / sub / REPOS[key].split("/")[1]


# Files that exist on this machine and belong in a repo; the merged weights, the English LoRA and the GGUF files
# live on the Modal volume and are uploaded from there (modal_hub.py).
LOCAL_FILES = {
    "jp-lora": (
        "data/adapters/sft-main/adapter",
        [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    ),
    "data-jp": ("data/processed", ["train.jsonl", "val.jsonl"]),
    "data-en": ("data/processed_en", ["train.jsonl", "val.jsonl"]),
}


def stage() -> None:
    cards = {
        "jp": merged_card("ja"),
        "en": merged_card("en"),
        "jp-lora": lora_card("ja"),
        "en-lora": lora_card("en"),
        "jp-gguf": gguf_card("ja"),
        "en-gguf": gguf_card("en"),
        "data-jp": dataset_card("ja"),
        "data-en": dataset_card("en"),
    }
    for key, card in cards.items():
        d = local_dir(key)
        (d / "figures").mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(card, encoding="utf-8", newline="\n")
        shutil.copy("assets/logo.png", d / "logo.png")
        for f in FIGS_DATA if key.startswith("data") else FIGS_MODEL:
            shutil.copy(R / "figures" / f"{f}.png", d / "figures" / f"{f}.png")
        src, names = LOCAL_FILES.get(key, ("", []))
        for n in names:
            shutil.copy(Path(src) / n, d / n)
        print("staged", d)
    space = local_dir("space")
    subprocess.run([sys.executable, "demo/build_hf_space.py", "--out", str(space)], check=True)
    items = [{"repo": REPOS[k], "type": KIND.get(k, "model"), "local": str(local_dir(k))} for k in REPOS]
    (RELEASE / "huggingface" / "repos.json").write_text(json.dumps(items, indent=1), encoding="utf-8")


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def sync() -> None:
    """Create every model and dataset repo as private and upload its local mirror (unchanged files are skipped)."""
    api = _api()
    for key in ("jp", "en", "jp-lora", "en-lora", "jp-gguf", "en-gguf", "data-jp", "data-en"):
        kind = KIND.get(key, "model")
        api.create_repo(REPOS[key], repo_type=kind, private=True, exist_ok=True)
        api.upload_folder(
            repo_id=REPOS[key],
            repo_type=kind,
            folder_path=str(local_dir(key)),
            commit_message="Sync card, figures and files",
        )
        print("synced", REPOS[key])


def upload_space() -> None:
    """The playground Space: Gradio on ZeroGPU (Hugging Face PRO). HF_TOKEN must be a Space secret while the models are private."""
    api = _api()
    api.create_repo(REPOS["space"], repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=REPOS["space"],
        repo_type="space",
        folder_path=str(local_dir("space")),
        delete_patterns=["*"],  # mirror the local folder exactly
        commit_message="Playground app (ZeroGPU)",
    )
    api.request_space_hardware(REPOS["space"], "zero-a10g")
    print("uploaded Space", REPOS["space"])


def collection() -> None:
    api = _api()
    c = api.create_collection(
        title="Kitsune Tales",
        namespace=versions.HF_NAMESPACE,
        description="Gemma 4 E4B fine-tuned for original fantasy light-novel fiction in Japanese and English: weights, LoRA, GGUF, datasets.",
        private=True,
        exists_ok=True,
    )
    notes = {
        "jp": "Japanese model (SFT), merged bf16",
        "en": "English model with Japanese anime themes (SFT + DPO), merged bf16",
        "jp-gguf": "Japanese, GGUF Q4_K_M / Q8_0",
        "en-gguf": "English, GGUF Q4_K_M / Q8_0",
        "jp-lora": "Japanese LoRA adapter",
        "en-lora": "English LoRA adapter",
        "data-jp": "Japanese SFT data (10,090 / 302)",
        "data-en": "English SFT data (6,647 / 193)",
        "space": "Playground for both models",
    }
    present = {i.item_id for i in c.items}
    for key, note in notes.items():
        if REPOS[key] not in present:
            api.add_collection_item(
                c.slug, item_id=REPOS[key], item_type=KIND.get(key, "model"), note=note, exists_ok=True
            )
    print("collection", f"{HF}/collections/{c.slug}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "sync", "upload-space", "collection"])
    a = ap.parse_args()
    {"stage": stage, "sync": sync, "upload-space": upload_space, "collection": collection}[a.cmd]()


if __name__ == "__main__":
    main()
