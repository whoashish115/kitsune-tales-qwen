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
