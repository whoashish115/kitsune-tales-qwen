---
theme: default
title: Kitsune Tales
info: Two small open models that write fantasy light-novel stories in Japanese and English. Every number comes from reports/ in the main repository.
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

Fantasy light-novel stories from a small open model, in Japanese and English

<div class="mt-10 text-sm">Ashish Kumar</div>
<div class="kicker mt-1">v0.1 · September 2026</div>

---

<p class="section">Overview</p>

# Project overview

Two small open models that write original fantasy light-novel stories. You give a genre, a title and a format; the model writes the story.

<div class="grid grid-cols-2 gap-4 mt-4 text-sm">
<div class="box p-4">

### Models
`kitsune-tales-e4b-jp` (Japanese) and `kitsune-tales-e4b-en` (English with Japanese anime themes)

</div>
<div class="box p-4">

### Base model
Gemma 4 E4B, 4.6B effective parameters; LoRA trains 77.8M of them

</div>
<div class="box p-4">

### Data
16,737 training examples, almost all synthetic stories from two larger open models

</div>
<div class="box p-4">

### Cost
$41.34 of cloud GPU time, data generation included

</div>
</div>

---

<p class="section">Overview</p>

# Why this project

- Open creative-writing models are either large and costly to run, or general-purpose and weak on a genre's conventions
- Light-novel fantasy has strong conventions: isekai and villainess plots, long descriptive titles, dialogue-heavy prose
- A small model that knows one genre well can run on a laptop

<div class="box mt-10 p-5">

**Goal.** Get a 4B-effective model as close as possible to its 35B teacher on this genre, on a small budget, and measure the result properly.

</div>

---
layout: two-cols
---

<p class="section">Overview</p>

# What the model does

A request names **one to three genres**, a **title** and a **format**. The answer is the story only: no headings, markdown or commentary.

| Format | Japanese | English |
|---|---:|---:|
| Synopsis | 200–500 characters | 150–350 words |
| Short story | 800–1,500 characters | 600–1,100 words |
| Continuation | 400–800 characters | 300–600 words |

<p class="cap">Target length for each format.</p>

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

<p class="section">Overview</p>

# How it was built

<div class="grid grid-cols-5 gap-3 mt-6">
<div class="box p-3"><div class="kicker">1</div><div class="font-bold text-sm mb-1">Data</div><div class="text-xs leading-snug">Two open models write stories and check each other's; rules and deduplication clean them</div></div>
<div class="box p-3"><div class="kicker">2</div><div class="font-bold text-sm mb-1">SFT</div><div class="text-xs leading-snug">LoRA fine-tuning teaches the format, the length and the refusals</div></div>
<div class="box p-3"><div class="kicker">3</div><div class="font-bold text-sm mb-1">DPO</div><div class="text-xs leading-snug">Preference tuning on pairs of the model's own outputs</div></div>
<div class="box p-3"><div class="kicker">4</div><div class="font-bold text-sm mb-1">Testing</div><div class="text-xs leading-snug">Held-out prompts, a safety suite and an LLM judge</div></div>
<div class="box p-3"><div class="kicker">5</div><div class="font-bold text-sm mb-1">Release</div><div class="text-xs leading-snug">Merged weights, GGUF files for laptops, a demo and the datasets</div></div>
</div>

<div class="mt-8">

Three rules held throughout:

- The 270 test prompts per language were frozen before any training data existed
- Choices such as the base model and which model to release followed rules written down in advance
- A budget check ran before every GPU job

</div>

---

<p class="section">Build</p>

# Choosing the base model

A short zero-shot test decided between the two candidates: switch to Gemma 4 E4B if Qwen3.5-4B is clearly worse on script purity or coherence.

| 12 test prompts | Qwen3.5-4B | Gemma 4 E4B |
|---|---:|---:|
| Chinese words inside Japanese prose | 4 / 12 | **0 / 12** |
| Repetitive outputs | 1 / 12 | **0 / 12** |
| Fantasy-only outputs | 11 / 12 | **12 / 12** |
| Genre-cue adherence | 0.86 | **0.93** |

Gemma 4 E4B stores 7.52B parameters but computes like a **4.62B** model, because 2.90B of them are per-layer embedding tables.

---

<p class="section">Build</p>

# Data

No web fiction: Qwen3.6-35B-A3B and Gemma 4 26B-A4B write the stories and label each other's; rules and MinHash deduplication clean the rest.

| Language | Generated | Kept | Train / validation |
|---|---:|---:|---:|
| Japanese | 15,320 | 10,257 (67 %) | 10,090 / 302 |
| English | 7,640 | 6,705 (88 %) | 6,647 / 193 |

<img src="../reports/figures/data_funnel.png" class="fig mt-3" style="max-height: 140px" alt="Data funnel for the Japanese and English datasets" />

---

<p class="section">Build</p>

# Training

<div class="grid grid-cols-2 gap-10 text-sm">
<div>

### Fine-tuning (SFT)
- LoRA r = 32, α = 64 on all language-model linear layers: 77.8M trainable parameters (0.97 %)
- Loss on the answer only; learning rate 2e-4, batch 16, one epoch on one H100
- More data helped more than a higher LoRA rank in the ablations

</div>
<div>

### Preference tuning (DPO)
- Two samples per prompt on 2,400 prompts, ranked by rules or by the teacher in both orders
- English adds 135 pairs that prefer a refusal
- Learning rate 2e-5, β = 0.1

</div>
</div>

| Run | Data | Validation | GPU minutes |
|---|---:|---:|---:|
| Japanese SFT | 10,090 examples | loss 1.200 (ppl 3.32) | 40 |
| English SFT | 6,647 examples | loss 1.184 (ppl 3.27) | 34 |
| English DPO | 1,596 pairs | DPO loss 0.637 | 14 |

<p class="cap">The released runs. Live training curves for every run are on the site.</p>

---

<p class="section">Evaluation</p>

# How it was tested

<div class="grid grid-cols-2 gap-10">
<div>

### Setup
- 270 held-out prompts × 3 seeds per language
- Same decoding for every model: temperature 0.8, top-p 0.95
- A separate safety suite with names and titles not seen in training
- Japanese comparisons: the base model, Qwen3.5-4B, Qwen3.5-9B and the 35B teacher

</div>
<div>

### Checks
- Automatic checks: length, format, script, repetition, safety and refusals
- An LLM judge (llm-jp-4-32b-a3b-thinking) compares two stories in both orders; a verdict counts only if both agree
- The judge was tested first on known answers: 86.7 % (JP), 93.3 % (EN)
- JGLUE for general Japanese ability, and a check for copied training text

</div>
</div>

---

<p class="section">Evaluation</p>

# Results

| Held-out test | Base model | Kitsune |
|---|---:|---:|
| Stories within the requested length (JP / EN) ↑ | 0.4 % / 22 % | **50 % / 81 %** |
| Outputs with markdown or meta text (JP / EN) ↓ | 76 % / 95 % | **0 % / 0 %** |
| Broken or unfinished outputs (JP / EN) ↓ | 17 % / 19 % | **0.5 % / 0 %** |
| Disallowed requests carried out (JP / EN) ↓ | 85 % / 79 % | **5 % / 0 %** |
| Validation perplexity (JP / EN) ↓ | 6.67 / 7.59 | **3.31 / 3.27** |

<p class="cap">Kitsune is the released model for each language (Japanese SFT, English SFT + DPO). 270 prompts × 3 seeds; safety suite 45 × 3.</p>

<p class="text-sm mt-4">General Japanese ability (four JGLUE tasks) shows no clear loss, and no Japanese test output copies a 32-character span from the training stories.</p>

---
layout: two-cols
---

<p class="section">Evaluation</p>

# The judge and length

- On full outputs, the judge prefers the base model: net −0.44 (JP) and −0.57 (EN)
- But the base model writes 1.7× longer (median 1,397 vs 828 characters), usually past the requested length
- With both stories cut to the same opening length, the gap disappears: +0.12 [−0.07, +0.32] (JP), −0.03 [−0.22, +0.15] (EN)
- So the base model's lead is mostly extra length that nobody asked for

::right::

<div class="pl-6 pt-16">
<img src="../reports/figures/judge_preference.png" class="fig" alt="Judge net preference with confidence intervals" />
<p class="cap">Net preference with 95 % CIs. Grey: full outputs; diamonds: equal-length openings.</p>
</div>

---

<p class="section">Evaluation</p>

# Safety

<img src="../reports/figures/safety.png" class="fig" style="max-height: 170px" alt="Outcomes on disallowed requests" />

<p class="cap">What each model does with held-out disallowed requests (45 prompts × 3 seeds per language).</p>

<div class="text-sm">

- Japanese DPO on quality pairs alone improved length (50 % → 67 %) but cut refusals (75 % → 49 %), so the Japanese release stays SFT
- English DPO with 135 refusal pairs kept 0 % violations and added 10 points of length adherence, so it ships
- The release rule was written before any judge result: ship DPO only if safety holds and the judge does not prefer SFT

</div>

---

<p class="section">Outlook</p>

# Limitations and next steps

<div class="grid grid-cols-2 gap-10">
<div>

### Limitations
- No human evaluation yet; quality rests on automatic checks and one LLM judge
- The models inherit their generators' habits; the Japanese models lose clearly to the 35B teacher
- Safety filters are word lists: they miss paraphrases and flag idioms
- Titles that ask to drop fantasy are followed more often than by the base model

</div>
<div>

### Next steps
- A small human preference study
- Japanese DPO with refusal pairs, the recipe that worked in English
- Adversarial-title examples in training
- A full LoRA rank sweep at the main data size

</div>
</div>

---

<p class="section">Release</p>

# Try it

<div class="grid grid-cols-2 gap-10">
<div>

### Use
- Merged weights for both models on Hugging Face
- GGUF Q4_K_M and Q8_0 files: 5.4 (JP) and 6.0 (EN) tokens/s on 8 CPU cores
- A live demo on Hugging Face and a free Colab notebook
- LoRA adapters and both datasets

</div>
<div>

### Rebuild
- Every number here comes from the `reports/` folder
- `make eval` rebuilds every table and figure on a CPU
- Every GPU step is a `make` target behind a budget check

</div>
</div>

<div class="mt-12 text-sm text-center">

[Site](https://kitsune-tales-qwen.vercel.app) · [Code](https://github.com/whoashish115/kitsune-tales-qwen) · [Models](https://huggingface.co/collections/whoashish115/kitsune-tales-6abd4e61de4896bb86692bc1) · [Demo](https://huggingface.co/spaces/whoashish115/kitsune-tales) · [Report](https://github.com/whoashish115/kitsune-tales-qwen/blob/main/REPORT.md)

</div>
