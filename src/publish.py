"""Model card generation and Hugging Face publishing.

    python -m kitsune.publish card                 # docs/MODEL_CARD.md    (Kitsune-Tales-E4B-JP) from reports/results.json
    python -m kitsune.publish card --lang en       # docs/MODEL_CARD_EN.md (Kitsune-Tales-E4B-EN) from reports/results_en.json

Every number in a card comes from its results file; the ``model-index`` block mirrors it.
Repos are created private first; making them public is a separate, manual step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from kitsune import versions

INTENDED = """- Writing **original, general-audience** Japanese fantasy fiction (synopses, short stories, continuations)
  from genre tags and a title, for hobby writing, brainstorming and light-novel style practice.
- Research on small-model creative-writing fine-tuning with synthetic data."""

OUT_OF_SCOPE = """- Sexual content of any kind; any sexualization of minors; real people; existing copyrighted characters or fan fiction;
  hateful content. The model is trained to refuse these, but refusals are not guaranteed. Use an input filter
  (the demo's is in `demo/app.py`).
- Factual, medical, legal or financial use. The model writes fiction and will state false things confidently.
- Non-fantasy genres (the model is trained to transpose them into fantasy) and languages other than Japanese."""

LIMITATIONS = """- **Synthetic-data ceiling.** Training data was written by two larger open models, so style, tropes and clichés are
  inherited from them, and quality is bounded by theirs.
- **Repetition and length drift** remain possible, especially at low temperature or with long outputs.
- **Hallucinated consistency errors** (names, timelines) within longer stories.
- **Cultural and stylistic bias** toward popular web-novel tropes (isekai, villainess, guild).
- **Memorization**: a leakage audit against the training set is reported, but it cannot rule out all verbatim reuse.
- **Thinking mode degraded.** Training used non-thinking examples only; use `enable_thinking=False`.
- **Pairwise judge prefers the base model's longer texts.** The base model writes ~1.7× longer than requested; the fine-tune
  follows the requested length and format but is not judged better than base on raw pairwise preference (see Results).
- **Speech-style and pronoun slips.** In the translated gallery, characters sometimes switch first-person pronouns
  (僕/私), or address a male character with 貴女. Synopses sometimes come out as a first-person scene.
- **Violence and self-harm filters are lexicon-based.** One gallery synopsis (test-0030) depicts a character cutting his
  own throat, which the Japanese self-harm lexicon did not catch.
- **Adversarial instructions.** When the title itself says "stop writing fantasy", the model follows it more often than
  the base model does (see the policy rows in Results).
- **Evaluation limits.** Automatic metrics are rule-based proxies, and LLM-judge results are indicative only (see REPORT.md)."""


def _pct(ci: dict[str, float] | None) -> float | None:
    return None if not ci or ci.get("n", 0) == 0 else round(100 * ci["mean"], 2)


def release(lang: str) -> dict[str, str]:
    """The released checkpoint for ``lang`` (configs/release.yaml, D-029)."""
    return yaml.safe_load(Path("configs/release.yaml").read_text(encoding="utf-8"))[lang]


def _recipe(system: str) -> str:
    return "LoRA SFT" if system.endswith("sft") else "LoRA SFT + DPO"


def model_index(results: dict[str, Any], system: str = "kitsune") -> list[dict]:
    """HF model-index entries mirroring reports/results.json for ``system``."""
    s = results["systems"][system]
    t, pol = s["test"], s["policy"]
    metrics = [
        ("length_adherence", "Length adherence (%)", _pct(t["length_ok"])),
        ("tag_adherence", "Genre-cue adherence (%)", _pct(t["genre_cue_rate"])),
        ("fantasy_rate", "Fantasy-only (%)", _pct(t["fantasy"])),
        ("zh_contamination", "Chinese contamination (%)", _pct(t["zh_contaminated"])),
        ("repetition_rate", "Repetitive outputs (%)", _pct(t["repetitive"])),
        ("refusal_disallowed", "Refusal on disallowed prompts (%)", _pct(pol.get("refusal_rate_disallowed"))),
    ]
    judge = results.get("judge", {}).get(f"judge:{system}_vs_base")
    if judge:
        metrics.append(
            (
                "judge_net_preference_vs_base",
                "LLM-judge net preference vs base",
                round(judge["net_preference"]["mean"], 4),
            )
        )
    return [
        {
            "name": versions.MODEL_SLUG,
            "results": [
                {
                    "task": {"type": "text-generation", "name": "Japanese fantasy fiction generation"},
                    "dataset": {
                        "name": "Kitsune held-out prompts (270 × 3 seeds)",
                        "type": versions.HF_DATASET_REPO,
                        "split": "test",
                    },
                    "metrics": [{"type": k, "name": n, "value": v} for k, n, v in metrics if v is not None],
                }
            ],
        }
    ]


INTENDED_EN = """- Writing **original, general-audience** fantasy fiction in English with Japanese anime / light-novel themes
  (synopses, short stories, continuations) from genre tags and a title, for hobby writing and brainstorming.
- Research on small-model creative-writing fine-tuning with synthetic data (the English counterpart of
  `Kitsune-Tales-E4B-JP`, trained with the identical recipe)."""

OUT_OF_SCOPE_EN = """- Sexual content of any kind; any sexualization of minors; real people; existing copyrighted characters or fan fiction;
  hateful content. The model is trained to refuse these, but refusals are not guaranteed. Use an input filter
  (the demo's is in `demo/app.py`).
- Factual, medical, legal or financial use. The model writes fiction and will state false things confidently.
- Non-fantasy genres (the model is trained to transpose them into fantasy). For Japanese output use `Kitsune-Tales-E4B-JP`."""

LIMITATIONS_EN = """- **Smaller dataset than the Japanese model:** 6,647 English training examples vs 10,090 Japanese.
- **Name debiasing.** The generators overused a few default names ("Elara" appeared in 67 % of one generator's first
  2,000 stories). In the training data these were replaced from varied name pools, deterministically per sample, and
  later prompts suggested protagonist names. Residual naming habits may remain.
- **Synthetic-data ceiling.** Training data was written by two larger open models, so style, tropes and clichés are
  inherited from them, and quality is bounded by theirs.
- **"Japanese flavor" is imitation.** Honorifics and anime/light-novel conventions come from the generators' notion of
  the style, not from translated Japanese novels.
- **Repetition and length drift** remain possible, especially at low temperature or with long outputs.
- **Hallucinated consistency errors** (names, timelines) within longer stories.
- **Thinking mode degraded.** Training used non-thinking examples only; use `enable_thinking=False`.
- **Evaluation limits.** Automatic metrics are rule-based proxies, and LLM-judge results are indicative only (see REPORT.md)."""


def model_index_en(results: dict[str, Any], system: str = "kitsune-en") -> list[dict]:
    """HF model-index entries mirroring reports/results_en.json for the English model."""
    s = results["systems"][system]
    t, pol = s["test"], s["policy"]
    metrics = [
        ("length_adherence", "Length adherence (%)", _pct(t["length_ok"])),
        ("tag_adherence", "Genre-cue adherence (%)", _pct(t["genre_cue_rate"])),
        ("fantasy_rate", "Fantasy-only (%)", _pct(t["fantasy"])),
        ("other_script_leakage", "Other-script leakage (%)", _pct(t["zh_contaminated"])),
        ("repetition_rate", "Repetitive outputs (%)", _pct(t["repetitive"])),
        ("refusal_disallowed", "Refusal on disallowed prompts (%)", _pct(pol.get("refusal_rate_disallowed"))),
    ]
    judge = results.get("judge", {}).get(f"judge:{system}_vs_base-en")
    if judge:
        metrics.append(
            (
                "judge_net_preference_vs_base",
                "LLM-judge net preference vs base",
                round(judge["net_preference"]["mean"], 4),
            )
        )
    return [
        {
            "name": versions.MODEL_SLUG_EN,
            "results": [
                {
                    "task": {"type": "text-generation", "name": "English fantasy fiction generation"},
                    "dataset": {
                        "name": "Kitsune English held-out prompts (270 × 3 seeds)",
                        "type": versions.HF_DATASET_REPO_EN,
                        "split": "test",
                    },
                    "metrics": [{"type": k, "name": n, "value": v} for k, n, v in metrics if v is not None],
                }
            ],
        }
    ]


def model_card_en(results: dict[str, Any] | None, results_table: str | None, links: dict[str, str]) -> str:
    from kitsune.en import SYSTEM_PROMPT_EN

    meta: dict[str, Any] = {
        "license": "apache-2.0",
        "base_model": versions.BASE_MODEL,
        "language": ["en"],
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "datasets": [versions.HF_DATASET_REPO_EN],
        "tags": ["lora", "creative-writing", "light-novel", "fantasy", "anime", "gemma4", "synthetic-data"],
    }
    rel = release("en")["system"]
    if results and rel in results.get("systems", {}):
        meta["model-index"] = model_index_en(results, rel)
    header = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n"
    link_md = " · ".join(f"[{k}]({v})" for k, v in links.items())
    system_literal = json.dumps(SYSTEM_PROMPT_EN)
    return f"""{header}
# {versions.MODEL_NAME_EN} (`{versions.MODEL_SLUG_EN}`)

An effective-4B model (Gemma 4 E4B: 7.52B stored / 4.62B effective parameters) that writes **original, general-audience fantasy
light-novel fiction in English, with Japanese anime / light-novel themes**, from genre tags, a title and a format
(synopsis / short story / continuation). It is a {_recipe(rel)} fine-tune of [{versions.BASE_MODEL}](https://huggingface.co/{versions.BASE_MODEL})
(revision `{versions.BASE_REVISION[:12]}`), trained on filtered, cross-labeled synthetic data from two Apache-2.0 models. It is the
English sibling of [`{versions.MODEL_SLUG}`](https://huggingface.co/{versions.HF_MODEL_REPO}) and was trained with the same recipe.

{link_md}

## Prompt format

```text
Genres: Isekai, Adventurer Guild
Title: The Exiled Swordsman Becomes the Strongest at the Frontier Guild
Format: short story
```
For `continuation`, add `Passage:` followed by the text to continue. Use the chat template with `enable_thinking=False`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "{versions.HF_MODEL_REPO_EN}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")
msgs = [{{"role": "system", "content": {system_literal}}},
        {{"role": "user", "content": "Genres: Demon Lord & Hero\\nTitle: The Retired Demon Lord Opens a Lakeside Cafe\\nFormat: synopsis"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=700, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.05)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Results

{results_table or "_Pending._"}

All numbers are generated from `reports/` in the GitHub repo by `python -m kitsune.eval.report --lang en` and carry 95 % bootstrap CIs over prompts.

## Intended use
{INTENDED_EN}

## Out-of-scope use
{OUT_OF_SCOPE_EN}

## Training data provenance
Synthetic English stories from `{versions.GENERATOR_MODEL}` and `{versions.GENERATOR2_MODEL}` (both Apache-2.0), prompted with
titles from our own templates and from the generators' brainstorms. Each sample was labeled by both models (cross-labels),
filtered by rule-based and LLM checks, and deduplicated. No scraped web fiction. See the dataset card and `docs/DATA_CARD.md`.

## Limitations
{LIMITATIONS_EN}

## License chain
- This model: Apache-2.0.
- Base model {versions.BASE_MODEL}: Apache-2.0 (Google). This is a derivative work; the base license and notices apply.
- Data generators: Apache-2.0. Evaluation judge {versions.JUDGE_MODEL}: Apache-2.0 (used for evaluation only).
"""


def model_card(results: dict[str, Any] | None, results_table: str | None, links: dict[str, str]) -> str:
    meta: dict[str, Any] = {
        "license": "apache-2.0",
        "base_model": versions.BASE_MODEL,
        "language": ["ja"],
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "datasets": [versions.HF_DATASET_REPO],
        "tags": [
            "lora",
            "japanese",
            "creative-writing",
            "light-novel",
            "fantasy",
            "gemma4",
            "synthetic-data",
        ],
    }
    rel = release("ja")["system"]
    if results and rel in results.get("systems", {}):
        meta["model-index"] = model_index(results, rel)
    header = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n"
    link_md = " · ".join(f"[{k}]({v})" for k, v in links.items())
    why = ""
    if results and {"kitsune", "kitsune-sft"} <= set(results.get("systems", {})) and rel == "kitsune-sft":
        v_sft = results["systems"]["kitsune-sft"]["policy"]["violation_rate_disallowed"]["mean"]
        v_dpo = results["systems"]["kitsune"]["policy"]["violation_rate_disallowed"]["mean"]
        net = (
            results.get("judge", {})
            .get("judge:kitsune_vs_kitsune-sft", {})
            .get("net_preference", {})
            .get("mean")
        )
        pairs = json.loads(Path("reports/dpo_pairs_v2.json").read_text(encoding="utf-8"))
        why = (
            f"**Why SFT and not SFT + DPO.** A DPO stage on the model's own samples ({pairs['rule'] + pairs['judge']:,} rule and "
            f"teacher-judged pairs) was trained and evaluated (`kitsune` in the tables below). The judge rated it level with "
            f"SFT (net preference {net:+.2f}), and it followed length requests better. But its policy-violation rate on held-out "
            f"disallowed requests was {100 * v_dpo:.1f} %, against {100 * v_sft:.1f} % for SFT: the preference data had no refusal "
            "pairs. A release rule fixed before the judge results (D-029) therefore ships the SFT model."
        )
    return f"""{header}
# {versions.MODEL_NAME} (`{versions.MODEL_SLUG}`)

An effective-4B model (Gemma 4 E4B: 7.52B stored / 4.62B effective parameters) that writes **original, general-audience Japanese fantasy light-novel fiction** from genre tags,
a title and a format (あらすじ / 短編 / 続き). It is a {_recipe(rel)} fine-tune of [{versions.BASE_MODEL}](https://huggingface.co/{versions.BASE_MODEL})
(revision `{versions.BASE_REVISION[:12]}`), trained on filtered synthetic data from Apache-2.0 models, on a $42 compute budget.

{why}

{link_md}

## Prompt format

```text
ジャンル: 異世界転生, 冒険者ギルド, ハイファンタジー
タイトル: 追放された剣士は二度目の人生で最強になる
形式: 短編
```
For `続き` (continuation), add `本文:` followed by the passage. Use the chat template with `enable_thinking=False`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "{versions.HF_MODEL_REPO}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")
msgs = [{{"role": "system", "content": "あなたは全年齢向けのオリジナル・ファンタジー小説を書く作家です。"}},
        {{"role": "user", "content": "ジャンル: 魔王と勇者\\nタイトル: 引退した魔王は湖畔で喫茶店を開く\\n形式: あらすじ"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=700, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.05)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Results

{results_table or "_Pending._"}

All numbers are generated from `reports/` in the GitHub repo by `make eval` and carry 95 % bootstrap CIs over prompts.

## Intended use
{INTENDED}

## Out-of-scope use
{OUT_OF_SCOPE}

## Training data provenance
Synthetic stories from `{versions.GENERATOR_MODEL}` and `{versions.GENERATOR2_MODEL}` (both Apache-2.0), prompted with
titles from our own templates, filtered by 12 rule-based and LLM checks, and deduplicated. No scraped web fiction. See the dataset card
and `docs/DATA_CARD.md`.

## Limitations
{LIMITATIONS}

## Synthetic-data ethics
Using larger models' outputs concentrates their stylistic and cultural biases and can launder their mistakes into a
new model. We used only generators whose licenses allow it, recorded every sample's generator and revision, used a
judge from a different model family for evaluation, and publish the pipeline so the data can be audited.

## License chain
- This model: Apache-2.0.
- Base model {versions.BASE_MODEL}: Apache-2.0 (Google). This is a derivative work; the base license and notices apply.
- Data generators: Apache-2.0. Evaluation judge {versions.JUDGE_MODEL}: Apache-2.0 (used for evaluation only).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["card"])
    ap.add_argument("--lang", choices=["ja", "en"], default="ja")
    a = ap.parse_args()
    sfx = "_en" if a.lang == "en" else ""
    res_p, tab_p = Path(f"reports/results{sfx}.json"), Path(f"reports/results_table{sfx}.md")
    results = json.loads(res_p.read_text(encoding="utf-8")) if res_p.exists() else None
    table = tab_p.read_text(encoding="utf-8") if tab_p.exists() else None
    links = {
        "GitHub": "https://github.com/whoashish115/kitsune-tales-qwen",
        "W&B": f"https://wandb.ai/whoashish115-base/{versions.WANDB_PROJECT}",
        "Space": f"https://huggingface.co/spaces/{versions.HF_SPACE_REPO}",
        "Dataset": f"https://huggingface.co/datasets/{versions.HF_DATASET_REPO_EN if a.lang == 'en' else versions.HF_DATASET_REPO}",
        "GGUF": f"https://huggingface.co/{versions.HF_GGUF_REPO_EN if a.lang == 'en' else versions.HF_GGUF_REPO}",
    }
    if a.cmd == "card":
        out = Path(f"docs/MODEL_CARD{'_EN' if a.lang == 'en' else ''}.md")
        card = model_card_en(results, table, links) if a.lang == "en" else model_card(results, table, links)
        out.write_text(card, encoding="utf-8", newline="\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
