---
theme: default
title: Kitsune Tales
info: Fantasy light-novel fine-tunes of Gemma 4 E4B in Japanese and English. Every number comes from reports/ in the main repository.
author: Ashish Kumar
colorSchema: light
routerMode: hash
favicon: logo.png
download: false
titleTemplate: '%s'
drawings:
  enabled: false
transition: fade
fonts:
  sans: Zen Kaku Gothic New
  serif: Shippori Mincho B1
  mono: JetBrains Mono
layout: center
class: text-center
---

<img src="../assets/logo.png" class="mx-auto w-20 mb-6" alt="Kitsune Tales logo" />

# Kitsune Tales

Fantasy light-novel fine-tunes of Gemma 4 E4B in Japanese and English

<div class="mt-10 text-sm">Ashish Kumar</div>
<div class="kicker mt-1">Technical report v0.1 · September 2026</div>

---

<p class="section">Overview</p>

# Abstract

We fine-tune Gemma 4 E4B (4.6B effective parameters) with LoRA to write original, general-audience fantasy fiction from a request that names genres, a title and a format, in Japanese and in English. All training data is synthetic: written by two Apache-2.0 models and filtered by rules and cross-model labels.

On 270 held-out prompts per language, length adherence rises from **0.4 % to 50 %** (Japanese) and from **22 % to 81 %** (English), and policy violations fall from **85 % to 5 %** and from **79 % to 0 %**. A validated LLM judge prefers the base model on full outputs, but the preference disappears when both outputs are cut to the same length.

The full study, including data generation, cost **$41.34** of cloud GPU time.

---

<p class="section">Overview</p>

# Contributions

1. **Two open models.** `kitsune-tales-e4b-jp` (SFT) and `kitsune-tales-e4b-en` (SFT + DPO), released as merged weights, LoRA adapters and GGUF files
2. **A synthetic-data pipeline.** Two generators, cross-model labelling, rule filters and deduplication; both datasets are released
3. **An evaluation protocol.** Test prompts frozen before data generation, bootstrap confidence intervals, an LLM judge validated on known answers, and a length-matched comparison
4. **A finding about the judge.** Its preference for the base model is explained mostly by output length, not by prose quality

---

<p class="section">Introduction</p>

# Motivation

- Open creative-writing models are either large and costly to run, or general-purpose and weak on a genre's conventions
- Light-novel fantasy is a large genre with strong conventions: isekai and villainess plots, long descriptive titles, dialogue-heavy prose
- A small, specialised model that runs on a laptop is therefore a well-posed target

<div class="box mt-10 p-5">

**Research question.** How close can a 4B-effective model get to its 35B teacher on this task, on an individual's budget, under honest measurement?

</div>

---
layout: two-cols
---

<p class="section">Introduction</p>

# Task definition

A request names **one to three genres**, a **title** and a **format**. The response is prose only: no headings, markdown or commentary.

| Format | Japanese | English |
|---|---:|---:|
| Synopsis | 200–500 characters | 150–350 words |
| Short story | 800–1,500 characters | 600–1,100 words |
| Continuation | 400–800 characters | 300–600 words |

<p class="cap">Table 1. Target length by format.</p>

::right::

<div class="pl-8 pt-20 flex flex-col gap-5">

<div class="request">ジャンル: 魔王と勇者, スローライフ
タイトル: 引退した魔王は湖畔で喫茶店を開く
形式: あらすじ</div>

<div class="request">Genres: Slow Life, High Fantasy
Title: A Kicked-Out Summoner Wants a Quiet Life in the Frontier
Format: continuation</div>

<p class="text-sm">Nine genres in total. Requests for sexual content, real people, existing IP or hate are refused; non-fantasy requests are rewritten as fantasy.</p>

</div>

---

<p class="section">Method</p>

# Design principles

All four were fixed before any training data existed.

<div class="grid grid-cols-2 gap-5 mt-6">
<div class="box p-4">

### Frozen test set
270 prompts per language (9 genres × 3 formats × 10), hashed before data generation

</div>
<div class="box p-4">

### Pre-registered decisions
The base-model switch rule and the release rule were written before the results they decide

</div>
<div class="box p-4">

### Measured uncertainty
95 % bootstrap confidence intervals; the LLM judge must pass a known-answer test before use

</div>
<div class="box p-4">

### Open and budgeted
Apache-2.0 models, code and data; a budget guard checks every GPU job before launch

</div>
</div>

---

<p class="section">Method</p>

# Base model selection

Pre-registered rule: switch to Gemma 4 E4B if Qwen3.5-4B is clearly worse on script purity or coherence.

| Metric | Qwen3.5-4B | Gemma 4 E4B |
|---|---:|---:|
| Chinese words inside Japanese prose | 4 / 12 | **0 / 12** |
| Repetitive outputs | 1 / 12 | **0 / 12** |
| Fantasy-only outputs | 11 / 12 | **12 / 12** |
| Genre-cue adherence | 0.86 | **0.93** |

<p class="cap">Table 2. Zero-shot bake-off on 12 prompts disjoint from the test set.</p>

Gemma 4 E4B: 7.52B stored parameters, **4.62B effective** (2.90B are per-layer embedding tables).

---

<p class="section">Data</p>

# Data generation

No web fiction is used. Two Apache-2.0 models write the stories, and each labels the other's.

<div class="grid grid-cols-5 gap-3 my-4">
<div class="box p-3"><div class="kicker">1</div><div class="font-bold text-sm mb-1">Seeds</div><div class="text-xs leading-snug">Titles, genre mixes and formats; titles close to a test title are dropped</div></div>
<div class="box p-3"><div class="kicker">2</div><div class="font-bold text-sm mb-1">Generation</div><div class="text-xs leading-snug">Qwen3.6-35B-A3B and Gemma 4 26B-A4B write the stories</div></div>
<div class="box p-3"><div class="kicker">3</div><div class="font-bold text-sm mb-1">Cross-labelling</div><div class="text-xs leading-snug">Each model rates the other's stories for fantasy, audience, real people or IP, and fit</div></div>
<div class="box p-3"><div class="kicker">4</div><div class="font-bold text-sm mb-1">Filtering</div><div class="text-xs leading-snug">Script, length, repetition, safety and PII rules, then MinHash deduplication</div></div>
<div class="box p-3"><div class="kicker">5</div><div class="font-bold text-sm mb-1">Splitting</div><div class="text-xs leading-snug">Train and validation sets, plus 135 refusal and redirect templates</div></div>
</div>

| Language | Generations | Kept | Train / validation |
|---|---:|---:|---:|
| Japanese | 15,320 | 10,257 (67 %) | 10,090 / 302 |
| English | 7,640 | 6,705 (88 %) | 6,647 / 193 |

<p class="cap">Table 3. Dataset sizes after filtering and deduplication.</p>

---

<p class="section">Data</p>

# Data filtering

<img src="../reports/figures/data_funnel.png" class="fig" alt="Data funnel for the Japanese and English datasets" />

<p class="cap"><b>Figure 1.</b> Synthetic data funnel. Rule filters remove most rejected stories; the cross-model labels remove a further 1–2 % of generations.</p>

---

<p class="section">Training</p>

# Training setup

<div class="grid grid-cols-2 gap-10 text-sm">
<div>

### Supervised fine-tuning
- LoRA r = 32, α = 64 on all language-model linear layers: 77.8M trainable parameters (0.97 %)
- Loss on the assistant turn only, no packing
- Learning rate 2e-4, cosine schedule, batch 16, one epoch

</div>
<div>

### Preference tuning (DPO)
- Two SFT samples on each of 2,400 prompts, ranked by rules or by the teacher in both orders
- English adds 135 refusal-preference pairs
- Learning rate 2e-5, β = 0.1; frozen SFT adapter as reference

</div>
</div>

| Run | Data | Validation | GPU minutes (H100) |
|---|---:|---:|---:|
| Japanese SFT | 10,090 examples | loss 1.200 (ppl 3.32) | 40 |
| English SFT | 6,647 examples | loss 1.184 (ppl 3.27) | 34 |
| English DPO | 1,596 pairs | DPO loss 0.637 | 14 |

<p class="cap">Table 4. Released training runs.</p>

---

<p class="section">Training</p>

# Training dynamics

<img src="../reports/figures/sft_loss.png" class="fig" alt="SFT loss curves" />

<p class="cap"><b>Figure 2.</b> SFT loss. Validation loss follows the training loss to the end of the epoch, with no sign of overfitting.</p>

---

<p class="section">Training</p>

# Ablations

<img src="../reports/figures/ablations.png" class="fig" alt="Ablations on data share and LoRA rank" />

<p class="cap"><b>Figure 3.</b> Japanese ablations. Validation loss falls with data (10 %, 30 %, 100 %: 1.364, 1.288, 1.200). At a fixed 25 % subset, rank 64 improves on rank 16 by only 0.036.</p>

---

<p class="section">Evaluation</p>

# Evaluation protocol

<div class="grid grid-cols-2 gap-10">
<div>

### Setup
- 270 frozen prompts × 3 seeds per language
- Identical decoding for every system: temperature 0.8, top-p 0.95
- A separate policy suite with names and titles disjoint from training
- Japanese baselines: base model, Qwen3.5-4B, Qwen3.5-9B, the 35B teacher

</div>
<div>

### Measures
- Rule-based metrics: length, format, script, repetition, safety, refusals
- Pairwise LLM judge (llm-jp-4-32b-a3b-thinking); a verdict counts only if both orders agree
- Judge accuracy on known-answer pairs: 86.7 % (JP), 93.3 % (EN)
- JGLUE via lm-eval, perplexity, and a train/test overlap audit

</div>
</div>

---

<p class="section">Results</p>

# Main results

| Metric | Base model | Kitsune |
|---|---:|---:|
| Outputs within the requested length (JP / EN) ↑ | 0.4 % / 22 % | **50 % / 81 %** |
| Outputs with markdown or meta text (JP / EN) ↓ | 76 % / 95 % | **0 % / 0 %** |
| Degenerate outputs (JP / EN) ↓ | 17 % / 19 % | **0.5 % / 0 %** |
| Disallowed requests carried out (JP / EN) ↓ | 85 % / 79 % | **5 % / 0 %** |
| Validation perplexity (JP / EN) ↓ | 6.67 / 7.59 | **3.31 / 3.27** |

<p class="cap">Table 5. Base model against the released model for each language (Japanese SFT, English SFT + DPO). Test set: 270 prompts × 3 seeds; policy suite: 45 disallowed requests × 3 seeds.</p>

---
layout: two-cols
---

<p class="section">Results</p>

# Judge preference and output length

- On full outputs, the judge prefers the base model: net −0.44 (JP) and −0.57 (EN)
- The base model writes 1.7× longer (median 1,397 vs 828 characters), usually past the requested length
- With both outputs cut to the same opening length, the preference disappears: +0.12 [−0.07, +0.32] (JP), −0.03 [−0.22, +0.15] (EN)
- The base model's advantage is therefore mostly length that the request did not ask for

::right::

<div class="pl-6 pt-16">
<img src="../reports/figures/judge_preference.png" class="fig" alt="Judge net preference with confidence intervals" />
<p class="cap"><b>Figure 4.</b> Net preference with 95 % CIs. Grey: full outputs; diamonds: equal-length openings.</p>
</div>

---

<p class="section">Results</p>

# Safety and preference tuning

<img src="../reports/figures/safety.png" class="fig" style="max-height: 200px" alt="Outcomes on disallowed requests" />

<p class="cap"><b>Figure 5.</b> Outcomes on held-out disallowed requests (45 prompts × 3 seeds per language).</p>

- Quality-only DPO (Japanese) raised length adherence from 50 % to 67 % but lowered refusals from 75 % to 49 %
- English DPO with 135 refusal pairs added 10 points of length adherence and kept violations at 0 %
- Release rule: DPO ships only if violations stay within 5 points of SFT and the judge does not prefer SFT

---

<p class="section">Results</p>

# General ability and memorisation

<div class="grid grid-cols-2 gap-10">
<div>

| JGLUE task | Base | Kitsune JP |
|---|---:|---:|
| JCommonsenseQA | 59.4 | 65.8 |
| JNLI | 58.4 | 55.4 |
| MARC-ja | 93.0 | 92.6 |
| XWinograd | 68.8 | 65.6 |

<p class="cap">Table 6. Accuracy on 500 items per task. One gain of about 2 SE; the other changes are within about 1 SE.</p>

</div>
<div>

### Memorisation audit
- Japanese: no test output shares a 32-character span with the training stories
- English: 1.6 % of 32-character windows appear in training (base model: 0.2 %)
- One English output in 270 contains a span of more than 100 characters
- The audit returns 100 % on a known positive

</div>
</div>

---

<p class="section">Discussion</p>

# Limitations

- **No human evaluation.** Quality rests on rule-based metrics and one LLM judge, which is confounded by length
- **Synthetic-data ceiling.** The models inherit their generators' habits; the Japanese models lose clearly to the 35B teacher
- **Lexicon-based safety.** Filters miss paraphrases and flag idioms, so unsafe rates are upper bounds
- **Adversarial titles.** A title that asks the model to drop fantasy is followed more often than by the base model
- **Japanese DPO.** Preference tuning with refusal pairs was not run for Japanese; the Japanese release is SFT only
- **Small ablations.** The rank comparison has two points at a single data size

---

<p class="section">Discussion</p>

# Future work

- A small human preference study to calibrate the LLM judge
- Japanese DPO with refusal-preference pairs, the recipe that worked in English
- Adversarial-title examples in the training data
- A full LoRA rank sweep at the main data size

---

<p class="section">Release</p>

# Release and reproducibility

<div class="grid grid-cols-2 gap-10">
<div>

### Artefacts
- Merged bf16 weights for both models
- GGUF Q4_K_M and Q8_0: 5.4 (JP) and 6.0 (EN) tokens/s on 8 CPU cores
- LoRA adapters and both training datasets
- A ZeroGPU demo and a Colab notebook

</div>
<div>

### Reproducibility
- Every number in this deck is generated from `reports/`
- `make eval` rebuilds every table and figure on a CPU
- Each GPU step is a `make` target behind a budget guard
- Total compute: $41.34

</div>
</div>

---
layout: center
---

<p class="section text-center">Conclusion</p>

# Conclusion

<div class="text-left max-w-3xl mx-auto">

- A 4B-effective model fine-tuned on filtered synthetic data follows genre, title, format and length far better than its base, and refuses disallowed requests
- Prose quality is on par with the base model once length is controlled
- Preference tuning needs explicit refusal pairs to avoid eroding safety

</div>

<div class="mt-10 text-sm text-center">

[Site](https://kitsune-tales-qwen.vercel.app) · [Code](https://github.com/whoashish115/kitsune-tales-qwen) · [Models](https://huggingface.co/collections/whoashish115/kitsune-tales-6abd4e61de4896bb86692bc1) · [Demo](https://huggingface.co/spaces/whoashish115/kitsune-tales) · [Report](https://github.com/whoashish115/kitsune-tales-qwen/blob/main/REPORT.md)

</div>
