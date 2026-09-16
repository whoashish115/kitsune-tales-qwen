---
license: apache-2.0
base_model: google/gemma-4-E4B-it
language:
- en
library_name: transformers
pipeline_tag: text-generation
datasets:
- whoashish115/kitsune-tales-en-fantasy-sft
tags:
- lora
- creative-writing
- light-novel
- fantasy
- anime
- gemma4
- synthetic-data
model-index:
- name: kitsune-tales-e4b-en
  results:
  - task:
      type: text-generation
      name: English fantasy fiction generation
    dataset:
      name: Kitsune English held-out prompts (270 × 3 seeds)
      type: whoashish115/kitsune-tales-en-fantasy-sft
      split: test
    metrics:
    - type: length_adherence
      name: Length adherence (%)
      value: 80.86
    - type: tag_adherence
      name: Genre-cue adherence (%)
      value: 98.17
    - type: fantasy_rate
      name: Fantasy-only (%)
      value: 99.75
    - type: other_script_leakage
      name: Other-script leakage (%)
      value: 0.0
    - type: repetition_rate
      name: Repetitive outputs (%)
      value: 0.0
    - type: refusal_disallowed
      name: Refusal on disallowed prompts (%)
      value: 94.81
    - type: judge_net_preference_vs_base
      name: LLM-judge net preference vs base
      value: -0.5667
---

# Kitsune-Tales-E4B-EN (`kitsune-tales-e4b-en`)

An effective-4B model (Gemma 4 E4B: 7.52B stored / 4.62B effective parameters) that writes **original, general-audience fantasy
light-novel fiction in English, with Japanese anime / light-novel themes**, from genre tags, a title and a format
(synopsis / short story / continuation). It is a LoRA SFT + DPO fine-tune of [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
(revision `ee0ef6023621`), trained on filtered, cross-labeled synthetic data from two Apache-2.0 models. It is the
English sibling of [`kitsune-tales-e4b-jp`](https://huggingface.co/whoashish115/kitsune-tales-e4b-jp) and was trained with the same recipe.

[GitHub](https://github.com/whoashish115/kitsune-tales-qwen) · [W&B](https://wandb.ai/whoashish115-base/kitsune-tales) · [Space](https://huggingface.co/spaces/whoashish115/kitsune-tales) · [Dataset](https://huggingface.co/datasets/whoashish115/kitsune-tales-en-fantasy-sft) · [GGUF](https://huggingface.co/whoashish115/kitsune-tales-e4b-en-gguf)
## Prompt format

```text
Genres: Isekai, Adventurer Guild
Title: The Exiled Swordsman Becomes the Strongest at the Frontier Guild
Format: short story
```
For `continuation`, add `Passage:` followed by the text to continue. Use the chat template with `enable_thinking=False`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "whoashish115/kitsune-tales-e4b-en"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")
msgs = [{"role": "system", "content": "You write original, general-audience fantasy light novels in English, in the style of Japanese anime and web novels. Follow the requested genres, title and format. Never write sexual content, real people, or characters from existing works. Requests outside fantasy are rewritten as fantasy."},
        {"role": "user", "content": "Genres: Demon Lord & Hero\nTitle: The Retired Demon Lord Opens a Lakeside Cafe\nFormat: synopsis"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=700, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.05)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```
## Results

Systems (all decoded identically: temperature 0.8, top-p 0.95, 3 seeds):

- `base-en`: Gemma 4 E4B instruct, zero-shot (the base model)
- `kitsune-en-sft`: LoRA SFT
- `kitsune-en`: LoRA SFT + DPO (quality + safety pairs), **released as `kitsune-tales-e4b-en`**
## Automatic metrics

| Metric (95 % CI) | `base-en` | `kitsune-en-sft` | `kitsune-en` |
|---|---|---|---|
| Length adherence (%) ↑ | 22.1 [18.3, 26.2] | 70.7 [67.2, 74.2] | 80.9 [77.8, 83.8] |
| Tag (genre-cue) adherence (%) ↑ | 97.2 [96.1, 98.3] | 97.9 [97.0, 98.7] | 98.2 [97.5, 98.8] |
| Title reflected (%) ↑ | 100.0 [100.0, 100.0] | 98.1 [96.7, 99.3] | 98.9 [98.0, 99.6] |
| Fantasy-only (%) ↑ | 99.8 [99.3, 100.0] | 99.8 [99.4, 100.0] | 99.8 [99.4, 100.0] |
| Latin-script ratio (%) ↑ | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Other-script leakage (CJK/kana/…) (%) ↓ | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Repetitive outputs (%) ↓ | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Degenerate outputs (%) ↓ | 18.9 [15.8, 22.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Unsafe (filter hit) (%) ↓ | 0.7 [0.2, 1.4] | 0.4 [0.0, 0.9] | 0.6 [0.1, 1.2] |
| False refusals (%) ↓ | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Markdown artifacts (%) ↓ | 95.4 [93.2, 97.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Meta-text leakage (%) ↓ | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 0.2 [0.0, 0.6] |
| Self-BLEU across seeds ↓ | 0.111 [0.108, 0.116] | 0.070 [0.068, 0.073] | 0.069 [0.067, 0.072] |
| Distinct-2 across seeds ↑ | 0.861 [0.858, 0.863] | 0.872 [0.869, 0.876] | 0.875 [0.871, 0.879] |
| Refusal on disallowed prompts (%) ↑ | 0.0 [0.0, 0.0] | 93.3 [88.9, 97.0] | 94.8 [91.1, 98.5] |
| Policy violations on disallowed prompts (%) ↓ | 79.3 [72.6, 85.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Stays fantasy on adversarial prompts (%) ↑ | 100.0 [100.0, 100.0] | 72.2 [58.3, 86.1] | 66.7 [50.0, 80.6] |
| Stays fantasy on off-genre prompts (%) ↑ | 88.9 [80.0, 97.8] | 97.8 [93.3, 100.0] | 100.0 [100.0, 100.0] |
| Test generations (prompts × seeds) | 810 | 810 | 810 |
## LLM-as-judge (indicative)

| Comparison | Win | Tie | Loss | Net preference (95 % CI) | Position-consistent | Pairs |
|---|---|---|---|---|---|---|
| judge:excerpt-kitsune-en_vs_excerpt-base-en | 23.3 % | 50.0 % | 26.7 % | -0.033 [-0.217, 0.150] | 50.0 % | 60 |
| judge:kitsune-en_vs_base-en | 5.3 % | 32.7 % | 62.0 % | -0.567 [-0.660, -0.467] | 67.3 % | 150 |
| judge:kitsune-en_vs_kitsune-en-sft | 28.7 % | 49.3 % | 22.0 % | 0.067 [-0.047, 0.180] | 50.7 % | 150 |

| Judge known-answer test | Accuracy (95 % CI) | By corruption | n |
|---|---|---|---|
| judge | 93.3 [86.7, 98.3] | loop: 100 %, script_leak: 100 %, shuffle: 100 %, truncate: 100 %, wrong_story: 67 % | 60 |


All numbers are generated from `reports/` in the GitHub repo by `python -m kitsune.eval.report --lang en` and carry 95 % bootstrap CIs over prompts.
## Intended use
- Writing **original, general-audience** fantasy fiction in English with Japanese anime / light-novel themes
  (synopses, short stories, continuations) from genre tags and a title, for hobby writing and brainstorming.
- Research on small-model creative-writing fine-tuning with synthetic data (the English counterpart of
  `kitsune-tales-e4b-jp`, trained with the identical recipe).
## Training data provenance
Synthetic English stories from `Qwen/Qwen3.6-35B-A3B-FP8` and `google/gemma-4-26B-A4B-it` (both Apache-2.0), prompted with
titles from our own templates and from the generators' brainstorms. Each sample was labeled by both models (cross-labels),
filtered by rule-based and LLM checks, and deduplicated. No scraped web fiction. See the dataset card and `docs/DATA_CARD.md`.
## Limitations
- **Smaller dataset than the Japanese model:** 6,647 English training examples vs 10,090 Japanese.
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
- **Evaluation limits.** Automatic metrics are rule-based proxies, and LLM-judge results are indicative only (see REPORT.md).
