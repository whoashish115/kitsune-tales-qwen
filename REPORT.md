# Kitsune Tales

*Technical report, v0.1, September 2026: fine-tuning effective-4B models for original fantasy light-novel fiction in Japanese and English. Every number and table is generated from the files in `reports/`.*

## Abstract

We fine-tune Gemma 4 E4B with bf16 LoRA to write original, general-audience fantasy fiction in three formats, in two
languages. `kitsune-tales-e4b-jp` (Japanese; SFT, with a DPO experiment) is the main study. `kitsune-tales-e4b-en` (English with Japanese
anime / light-novel themes; SFT, then DPO, with the identical recipe) tests whether the pipeline transfers to a second language.
Both use only synthetic data from Apache-2.0 models, filtered by rule-based and cross-model LLM checks.
We evaluate the Japanese model on 270 held-out prompts frozen before data generation, against the base model, Qwen3.5-4B
and Qwen3.5-9B, and the 35B data generator; the English model on its own 270 frozen prompts against the base model. We report 95 % bootstrap confidence intervals, an LLM judge that
must first pass a known-answer test, a train/test leakage audit, and a general-capability regression check.

**Results.** Both fine-tunes follow the request far better than the base model. Japanese length adherence rises from
0.4 % to 50 %, and English from 22 % to 81 %. Markdown artifacts and degenerate outputs fall to about 0 %.
Policy violations on held-out disallowed requests fall from 85 % to 5 % (Japanese) and from 79 % to 0 % (English),
and general Japanese ability (JGLUE via lm-eval) is preserved within noise. On full outputs, a validated LLM judge
prefers the base model, which writes about 1.7× longer than requested. When both outputs are cut to the same length,
that preference disappears: net +0.12 for Japanese and −0.03 for English, both with CIs spanning 0. Quality-only
DPO lowered Japanese refusals (violations 5 % → 21 %). Adding 135 refusal-preference pairs removed that side effect
in English (0 % violations, plus 10 points of length adherence). Under a release rule fixed before the judge results,
we ship Japanese SFT and English SFT + DPO. Everything, including data generation, cost $41.34 of cloud GPU credit
(billed).

## 1. Introduction

Open Japanese creative-writing models are either large (costly to run) or general-purpose (weak at a specific
genre's conventions). Light-novel fantasy is a large, well-defined genre with strong conventions: tropes such as
isekai and the villainess story, dialogue-heavy style, and long descriptive titles. That makes it a good target for a
small specialized model. Our question is how close a 4B model can get to its 35B teacher on this task, at a cost an
individual can afford, while measuring the result honestly.

## 2. Related work

Only sources we opened are cited (see `docs/related_work.md`).
LoRA [Hu et al. 2021] freezes the base and learns low-rank updates that merge without inference latency; QLoRA
[Dettmers et al. 2023] adds 4-bit base weights, which we do not use because the Qwen3.5 guidance advises against it.
LoRA preserves out-of-domain ability better than full fine-tuning [Biderman et al. 2024], which motivates our
regression check. Likelihood-maximizing decoding degenerates into repetition [Holtzman et al. 2020]; we measure
repetition directly. Deduplication reduces memorization and train–test overlap [Lee et al. 2022]. LLM judges show
position, verbosity and self-enhancement biases [Zheng et al. 2023; Wang et al. 2023] and favor their own
generations [Panickssery et al. 2024]. For creative writing, even strong judges agree with humans only about 73 % of the
time [Fein et al. 2025, LitBench]. WebNovelBench [Lin et al. 2025] frames synopsis-to-story evaluation for web novels.

## 3. Data

<!-- BEGIN:data -->
| Dataset | Generations | Kept after filters + dedup | Kept gen1 / gen2 | Cross-model labels | Train / val | Policy templates |
|---|---|---|---|---|---|---|
| Japanese | 15,320 | 10,257 (67%) | 5,517 / 4,740 | 100% | 10,090 / 302 | 135 |
| English | 7,640 | 6,705 (88%) | 3,895 / 2,810 | 100% | 6,647 / 193 | 135 |

Filter funnels, length histograms and genre × format grids: `reports/data*/`.
<!-- END:data -->

## 4. Method

- Base: Gemma 4 E4B, text model only (7.52B stored / 4.62B effective parameters), chosen over Qwen3.5-4B by a
  pre-registered zero-shot bake-off (D-001a).
- SFT: bf16 LoRA on every linear layer of the language model, with loss on the assistant turn only (labels built
  explicitly and verified against the tokenizer), cosine schedule, fixed seed, and no packing (D-012).
- DPO: two self-samples on each of 2,400 training prompts from the SFT model. Pairs are rule-decided or
  teacher-judged; a judged pair counts only if both orders agree. The reference is a frozen copy of the SFT adapter,
  and the DPO labeler (the teacher) differs from the eval judge (D-011). β = 0.1, 1 epoch, micro-batch 1 × 16.
  - *Teacher labels (D-027).* The teacher writes one sentence per criterion within 2,048 output tokens, so every
    call ends in a verdict (3,797 of 3,798; median 205 tokens). This gives 1,709 Japanese pairs.
  - *Safety pairs (D-029).* Quality-only DPO lowered refusals on held-out disallowed prompts, so English DPO adds 135
    refusal-preference pairs built from the training disallowed prompts (names disjoint from the eval suite).
- Release rule (D-029, fixed before any judge result was seen): SFT + DPO ships only if its policy-violation rate is
  within 5 points of SFT's and the judge does not prefer SFT. Otherwise the SFT model ships.

## 5. Experimental setup

<!-- BEGIN:setup -->
All runs: Gemma 4 E4B (`google/gemma-4-E4B-it`, pinned revision), bf16 LoRA on every language-model linear layer, loss on the assistant turn only, no packing, cosine schedule, seed 42, 1 epoch, batch 16, H100. Validation loss is on each language's own validation split.

| Run | Language | Stage | Train examples | LoRA r | Trainable params | Val loss (ppl) | GPU minutes |
|---|---|---|---|---|---|---|---|
| `sft-main` | ja | SFT | 10,090 | 32 | 77.8M | 1.2004 (3.32) | 40 |
| `sft-en-main` | en | SFT | 6,647 | 32 | 77.8M | 1.1842 (3.27) | 34 |
| `dpo-main-v2` | ja | DPO (β 0.1) | 1,709 pairs | 32 | same adapter | DPO loss 0.5872 | 22 |
| `dpo-en-main` | en | DPO (β 0.1) | 1,596 pairs | 32 | same adapter | DPO loss 0.6367 | 14 |

Japanese DPO pairs: 448 rule-decided + 1,261 teacher-judged (both orders agree); 638 prompts dropped as ties or order-inconsistent, 53 with both samples failing rules.

English DPO pairs: 293 rule-decided + 1,168 teacher-judged (both orders agree) + 135 safety (refusal-preference) pairs; 905 prompts dropped as ties or order-inconsistent, 34 with both samples failing rules.

**Ablations (Japanese, same validation split).** Data scaling at r = 32, and LoRA rank at a fixed 25 % subset:

| Run | Data | Train examples | LoRA r | Trainable params | Val loss | Val ppl |
|---|---|---|---|---|---|---|
| `abl-data10` | 10 % | 1,009 | 32 | 77.8M | 1.3637 | 3.91 |
| `abl-data30` | 30 % | 3,027 | 32 | 77.8M | 1.2877 | 3.62 |
| `sft-main` | 100 % | 10,090 | 32 | 77.8M | 1.2004 | 3.32 |
| `abl-r16` | 25 % | 2,522 | 16 | 38.9M | 1.3185 | 3.74 |
| `abl-r64` | 25 % | 2,522 | 64 | 155.5M | 1.2827 | 3.61 |
<!-- END:setup -->
