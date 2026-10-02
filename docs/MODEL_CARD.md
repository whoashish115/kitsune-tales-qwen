---
license: apache-2.0
base_model: google/gemma-4-E4B-it
language:
- ja
library_name: transformers
pipeline_tag: text-generation
datasets:
- whoashish115/Kitsune-Tales-JP-Fantasy-SFT
tags:
- lora
- japanese
- creative-writing
- light-novel
- fantasy
- gemma4
- synthetic-data
model-index:
- name: Kitsune-Tales-E4B-JP
  results:
  - task:
      type: text-generation
      name: Japanese fantasy fiction generation
    dataset:
      name: Kitsune held-out prompts (270 × 3 seeds)
      type: whoashish115/Kitsune-Tales-JP-Fantasy-SFT
      split: test
    metrics:
    - type: length_adherence
      name: Length adherence (%)
      value: 50.25
    - type: tag_adherence
      name: Genre-cue adherence (%)
      value: 93.11
    - type: fantasy_rate
      name: Fantasy-only (%)
      value: 98.52
    - type: zh_contamination
      name: Chinese contamination (%)
      value: 0.12
    - type: repetition_rate
      name: Repetitive outputs (%)
      value: 0.0
    - type: refusal_disallowed
      name: Refusal on disallowed prompts (%)
      value: 74.81
    - type: judge_net_preference_vs_base
      name: LLM-judge net preference vs base
      value: -0.4414
---

# Kitsune-Tales-E4B-JP (`Kitsune-Tales-E4B-JP`)

An effective-4B model (Gemma 4 E4B: 7.52B stored / 4.62B effective parameters) that writes **original, general-audience Japanese fantasy light-novel fiction** from genre tags,
a title and a format (あらすじ / 短編 / 続き). It is a LoRA SFT fine-tune of [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
(revision `ee0ef6023621`), trained on filtered synthetic data from Apache-2.0 models, on a $42 compute budget.

**Why SFT and not SFT + DPO.** A DPO stage on the model's own samples (1,709 rule and teacher-judged pairs) was trained and evaluated (`kitsune` in the tables below). The judge rated it level with SFT (net preference +0.00), and it followed length requests better. But its policy-violation rate on held-out disallowed requests was 20.7 %, against 5.2 % for SFT: the preference data had no refusal pairs. A release rule fixed before the judge results (D-029) therefore ships the SFT model.

[GitHub](https://github.com/whoashish115/kitsune-tales-qwen) · [W&B](https://wandb.ai/whoashish115-base/kitsune-tales) · [Space](https://huggingface.co/spaces/whoashish115/Kitsune-Tales) · [Dataset](https://huggingface.co/datasets/whoashish115/Kitsune-Tales-JP-Fantasy-SFT) · [GGUF](https://huggingface.co/whoashish115/Kitsune-Tales-E4B-JP-GGUF)

## Prompt format

```text
ジャンル: 異世界転生, 冒険者ギルド, ハイファンタジー
タイトル: 追放された剣士は二度目の人生で最強になる
形式: 短編
```
For `続き` (continuation), add `本文:` followed by the passage. Use the chat template with `enable_thinking=False`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "whoashish115/Kitsune-Tales-E4B-JP"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")
msgs = [{"role": "system", "content": "あなたは全年齢向けのオリジナル・ファンタジー小説を書く作家です。"},
        {"role": "user", "content": "ジャンル: 魔王と勇者\nタイトル: 引退した魔王は湖畔で喫茶店を開く\n形式: あらすじ"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=700, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.05)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Results

Systems (all decoded identically: temperature 0.8, top-p 0.95, 3 seeds):

- `base`: Gemma 4 E4B instruct, zero-shot (the base model)
- `kitsune-sft`: LoRA SFT, **released as `Kitsune-Tales-E4B-JP`**
- `kitsune`: LoRA SFT + DPO v2 (quality pairs only)
- `qwen3.5-4b`: Qwen3.5-4B instruct, zero-shot
- `qwen3.5-9b`: Qwen3.5-9B instruct, zero-shot
- `teacher`: Qwen3.6-35B-A3B-FP8, the data generator, zero-shot

## Automatic metrics

| Metric (95 % CI) | `base` | `kitsune-sft` | `kitsune` | `qwen3.5-4b` | `qwen3.5-9b` | `teacher` |
|---|---|---|---|---|---|---|
| Length adherence (%) ↑ | 0.4 [0.0, 0.9] | 50.2 [46.2, 54.3] | 66.9 [62.6, 71.2] | 5.7 [4.0, 7.4] | 4.1 [2.7, 5.6] | 4.1 [2.6, 5.6] |
| Tag (genre-cue) adherence (%) ↑ | 90.6 [88.7, 92.5] | 93.1 [91.6, 94.6] | 92.6 [90.9, 94.2] | 86.7 [84.3, 89.0] | 91.8 [90.0, 93.5] | 91.7 [89.8, 93.5] |
| Title reflected (%) ↑ | 100.0 [100.0, 100.0] | 98.3 [96.6, 99.6] | 99.2 [98.3, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Fantasy-only (%) ↑ | 98.8 [97.7, 99.6] | 98.5 [97.5, 99.4] | 99.6 [99.1, 100.0] | 96.8 [95.1, 98.3] | 97.3 [95.4, 98.8] | 99.0 [98.1, 99.8] |
| Japanese script ratio (%) ↑ | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.9 [99.8, 99.9] | 99.9 [99.9, 99.9] | 99.8 [99.8, 99.9] |
| Chinese contamination (%) ↓ | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 0.2 [0.0, 0.6] | 18.0 [15.4, 20.7] | 14.0 [11.5, 16.4] | 22.3 [19.4, 25.3] |
| Repetitive outputs (%) ↓ | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 4.9 [3.3, 6.7] | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.6] |
| Degenerate outputs (%) ↓ | 16.7 [13.6, 19.9] | 0.5 [0.1, 1.0] | 0.4 [0.0, 0.9] | 14.4 [12.0, 16.9] | 10.0 [7.8, 12.3] | 3.3 [2.0, 4.8] |
| Unsafe (filter hit) (%) ↓ | 1.0 [0.4, 1.7] | 0.4 [0.0, 0.9] | 0.4 [0.0, 0.9] | 0.5 [0.1, 1.0] | 0.4 [0.0, 0.9] | 1.0 [0.4, 1.9] |
| False refusals (%) ↓ | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Markdown artifacts (%) ↓ | 75.9 [71.2, 80.4] | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 68.8 [63.5, 74.0] | 68.6 [63.2, 74.0] | 70.0 [64.9, 74.9] |
| English/markup leakage (%) ↓ | 1.0 [0.4, 1.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 12.1 [9.9, 14.4] | 10.4 [8.1, 12.7] | 21.0 [18.1, 23.8] |
| Self-BLEU across seeds ↓ | 0.352 [0.348, 0.356] | 0.270 [0.265, 0.274] | 0.266 [0.261, 0.270] | 0.280 [0.274, 0.285] | 0.302 [0.295, 0.308] | 0.285 [0.282, 0.289] |
| Distinct-2 across seeds ↑ | 0.527 [0.521, 0.533] | 0.607 [0.601, 0.613] | 0.621 [0.615, 0.628] | 0.490 [0.481, 0.500] | 0.527 [0.520, 0.535] | 0.554 [0.547, 0.560] |
| Refusal on disallowed prompts (%) ↑ | 0.0 [0.0, 0.0] | 74.8 [67.4, 82.2] | 48.9 [40.7, 57.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.5 [0.0, 3.7] |
| Policy violations on disallowed prompts (%) ↓ | 85.2 [78.5, 91.1] | 5.2 [1.5, 9.6] | 20.7 [14.1, 28.1] | 89.6 [84.4, 94.1] | 88.9 [83.7, 94.1] | 89.6 [84.4, 94.1] |
| Stays fantasy on adversarial prompts (%) ↑ | 100.0 [100.0, 100.0] | 52.8 [36.1, 69.4] | 69.4 [52.8, 83.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Stays fantasy on off-genre prompts (%) ↑ | 15.6 [6.7, 26.7] | 95.6 [88.9, 100.0] | 95.6 [88.9, 100.0] | 2.2 [0.0, 6.7] | 6.7 [0.0, 15.6] | 11.1 [2.2, 20.0] |
| Test generations (prompts × seeds) | 810 | 810 | 810 | 810 | 810 | 810 |

## LLM-as-judge (indicative)

| Comparison | Win | Tie | Loss | Net preference (95 % CI) | Position-consistent | Pairs |
|---|---|---|---|---|---|---|
| judge:excerpt-kitsune-sft_vs_excerpt-base | 35.1 % | 42.1 % | 22.8 % | 0.123 [-0.070, 0.316] | 57.9 % | 57 |
| judge:kitsune-sft_vs_base | 8.3 % | 39.3 % | 52.4 % | -0.441 [-0.545, -0.338] | 60.7 % | 145 |
| judge:kitsune-sft_vs_teacher | 9.7 % | 32.1 % | 58.2 % | -0.485 [-0.597, -0.373] | 67.9 % | 134 |
| judge:kitsune_vs_base | 11.0 % | 37.0 % | 52.1 % | -0.411 [-0.514, -0.301] | 63.7 % | 146 |
| judge:kitsune_vs_kitsune-sft | 32.4 % | 35.1 % | 32.4 % | 0.000 [-0.128, 0.135] | 64.9 % | 148 |
| judge:kitsune_vs_teacher | 11.5 % | 25.9 % | 62.6 % | -0.511 [-0.626, -0.388] | 74.1 % | 139 |

| Judge known-answer test | Accuracy (95 % CI) | By corruption | n |
|---|---|---|---|
| judge | 86.7 [76.7, 95.0] | chinese: 75 %, loop: 100 %, shuffle: 67 %, truncate: 92 %, wrong_story: 100 % | 60 |


All numbers are generated from `reports/` in the GitHub repo by `make eval` and carry 95 % bootstrap CIs over prompts.

## Intended use
- Writing **original, general-audience** Japanese fantasy fiction (synopses, short stories, continuations)
  from genre tags and a title, for hobby writing, brainstorming and light-novel style practice.
- Research on small-model creative-writing fine-tuning with synthetic data.

## Out-of-scope use
- Sexual content of any kind; any sexualization of minors; real people; existing copyrighted characters or fan fiction;
  hateful content. The model is trained to refuse these, but refusals are not guaranteed. Use an input filter
  (the demo's is in `demo/app.py`).
- Factual, medical, legal or financial use. The model writes fiction and will state false things confidently.
- Non-fantasy genres (the model is trained to transpose them into fantasy) and languages other than Japanese.

## Training data provenance
Synthetic stories from `Qwen/Qwen3.6-35B-A3B-FP8` and `google/gemma-4-26B-A4B-it` (both Apache-2.0), prompted with
titles from our own templates, filtered by 12 rule-based and LLM checks, and deduplicated. No scraped web fiction. See the dataset card
and `docs/DATA_CARD.md`.

## Limitations
- **Synthetic-data ceiling.** Training data was written by two larger open models, so style, tropes and clichés are
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
- **Evaluation limits.** Automatic metrics are rule-based proxies, and LLM-judge results are indicative only (see REPORT.md).

## Synthetic-data ethics
Using larger models' outputs concentrates their stylistic and cultural biases and can launder their mistakes into a
new model. We used only generators whose licenses allow it, recorded every sample's generator and revision, used a
judge from a different model family for evaluation, and publish the pipeline so the data can be audited.

## License chain
- This model: Apache-2.0.
- Base model google/gemma-4-E4B-it: Apache-2.0 (Google). This is a derivative work; the base license and notices apply.
- Data generators: Apache-2.0. Evaluation judge llm-jp/llm-jp-4-32b-a3b-thinking: Apache-2.0 (used for evaluation only).
