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

## 6. Results

<!-- BEGIN:results -->
Systems (all decoded identically: temperature 0.8, top-p 0.95, 3 seeds):

- `base`: Gemma 4 E4B instruct, zero-shot (the base model)
- `kitsune-sft`: LoRA SFT, **released as `kitsune-tales-e4b-jp`**
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
<!-- END:results -->

### 6.2 English model

Same protocol on the English test set (270 prompts × 3 seeds + a 72-prompt policy suite), `base-en` vs `kitsune-en`.
The metric keys are the Japanese ones with English definitions: the Latin-script ratio replaces the Japanese-script ratio,
and other-script leakage (CJK/kana/Hangul/Cyrillic) replaces Chinese contamination.

<!-- BEGIN:results_en -->
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
<!-- END:results_en -->
