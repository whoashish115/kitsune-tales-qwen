# Design decisions

Each entry records a choice, the evidence behind it and what it rules out. Identifiers (D-001, D-012, …) are stable
because code comments and configs cite them; entries that only concerned infrastructure bookkeeping are not listed.

---
## D-001 / D-001a: Base model: Gemma 4 E4B

Requirements: a license that allows redistributing derivatives, strong Japanese, 3B to 9B parameters, a usable chat
template, and a quantized build that runs on a CPU.

| Model | Params | License | Japanese evidence | Notes |
|---|---|---|---|---|
| Qwen3.5-4B | 4B | Apache-2.0 | Nejumi LB4 0.735 (top of sub-10B) | hybrid Gated DeltaNet / attention; QLoRA discouraged |
| Qwen3.5-9B | 9B | Apache-2.0 | strong sub-10B | about 2× the cost |
| **Gemma 4 E4B-it** | 7.52B stored, 4.62B effective | Apache-2.0 | Nejumi LB4 0.669 | standard transformers, mature llama.cpp support |
| llm-jp-4-8b-thinking | 8.6B | Apache-2.0 | JA MT-Bench 7.54 | thinking-only post-training |
| Qwen3-Swallow-8B-RL | 8.2B | Apache-2.0 | Nejumi LB4 0.655 | reasoning cannot be disabled |
| Nemotron Nano 9B v2 JP | 9B | NVIDIA Open Model License | Nejumi LB4 0.711 | not OSI, Mamba tooling |

Qwen3.5-4B was the provisional choice, with a pre-registered rule: switch to Gemma 4 E4B if Qwen3.5-4B is clearly worse
on script purity or coherence in a zero-shot bake-off (12 prompts drawn with a separate seed, disjoint from the test set).

| Bake-off metric | Qwen3.5-4B | Gemma 4 E4B |
|---|---|---|
| Chinese contamination | 4/12 (whole words: `实际上`, `冒险家`) | 0/12 |
| Repetitive | 1/12 | 0/12 |
| Fantasy-only | 11/12 | 12/12 |
| Genre-cue adherence | 0.86 | 0.93 |

Qwen3.5-4B leaks Chinese words into Japanese prose, so the base model is **Gemma 4 E4B** (revision `ee0ef60`). The
bake-off is a screen (n = 12); the proper comparison is in the main evaluation, where Qwen3.5-4B stays as a baseline.

Gemma 4 E4B stores 7.52B parameters (3.95B in transformer layers, 0.67B tied embeddings, 2.90B per-layer embedding
tables that cost almost no compute) and computes like a 4.62B model, hence "E4B" and the model names
`kitsune-tales-e4b-jp` / `kitsune-tales-e4b-en`. Thinking is off by default; `<turn|>` ends a turn; tokenization gives
1.43 Japanese characters per token.
## D-002: bf16 LoRA, not QLoRA

A bf16 LoRA fits easily on one H100, so 4-bit QLoRA would add quantization error for no saving. LoRA targets every
linear layer of the language model (attention, MLP and the per-layer-embedding projections); the vision and audio
towers are frozen. r = 32, α = 64, dropout 0.05: 77.8M trainable parameters (0.97 % of the checkpoint).
## D-003 / D-006: Data sources and generators

- **No web fiction.** Japanese web-novel corpora have unclear provenance or restrictive terms (Syosetu text belongs to
  its authors). Every training story is synthetic.
- **Generators:** Qwen3.6-35B-A3B (FP8) and Gemma 4 26B-A4B, both Apache-2.0, which places no restriction on training
  other models on their outputs. Two families give stylistic variety and allow cross-labeling (D-011).
- **Seeds:** genre, format and title prompts from hand-written templates and word lists, plus titles brainstormed by
  the generators and then filtered.
## D-004: Judge model

**llm-jp-4-32b-a3b-thinking** (Apache-2.0, NII) judges pairwise. It is from a different family than the generators and
the base model, which limits self-preference, and it reads both Japanese and English. Every comparison runs in both
presentation orders; a win counts only when both orders agree. Judge results are reported only together with the
known-answer validation (D-010).
## D-007: Task format and data plan

A fixed system prompt (general audience, original, fantasy only) and a user turn with genres (from a 9-genre
taxonomy), title and format; continuations add the passage to continue.

| Format | Share | Target (JP chars / EN words) | Training filter bounds (JP) |
|---|---|---|---|
| あらすじ synopsis | 25 % | 200–500 / 150–350 | 150–600 |
| 短編 short story | 50 % | 800–1,500 / 600–1,100 | 750–1,650 |
| 続き continuation | 25 % | 400–800 / 300–600 | 300–1,000 |

Continuation examples come from separate stories (opening as passage, the rest as target, cut at a sentence end).
About 1.3 % of training examples are templated policy responses: a fixed refusal for disallowed requests (sexual
content, real people, existing IP, hate) and a one-line note plus a fantasy rewrite for off-genre requests.

**Filters, in order:** schema; script purity (Japanese-script ratio ≥ 0.90, hiragana ≥ 0.15, no simplified-Chinese or
non-CP932 kanji; for English, no CJK, kana, Hangul, Cyrillic, Devanagari or Thai); English/markup leakage (D-018);
length; repetition (char-8-gram uniqueness, repeated lines, compression ratio); fantasy lexicon; safety lexicon and PII;
real-person and IP blocklist; title and genre-tag consistency; cross-model LLM labels; MinHash deduplication (char
5-grams, Jaccard ≥ 0.7).

**Splits.** The 270 test prompts per language (9 genres × 3 formats × 10) and the 72-prompt policy suite were generated
from their own seeds and SHA-256-frozen before any training data existed. Training titles within near-duplicate distance
of a test title are excluded. Validation is 3 % of kept examples, stratified by genre × format.
## D-008 / D-010: Evaluation design

- **Systems:** base model, SFT, SFT + DPO; for Japanese also Qwen3.5-4B, Qwen3.5-9B and the 35B teacher as references.
- **Decoding:** temperature 0.8, top-p 0.95, top-k 50, repetition penalty 1.05, three seeds, vLLM.
- **Automatic metrics** with 95 % percentile-bootstrap CIs (10,000 resamples): over prompts for the test suite, over
  generations for the policy suite. Length adherence, markdown and meta text, degeneration, repetition, script purity,
  genre cues, fantasy-only, refusals and violations, distinct-n, self-BLEU.
- **Judge validation first:** 60 known-answer pairs per language, an intact story against a corrupted copy (shuffled
  sentences, loops, truncation, a story for another request, mixed-in script). Japanese 86.7 %, English 93.3 %.
- **Leakage audit:** share of each test output's 32-character windows found verbatim in the training stories.
- **General ability:** four JGLUE tasks through lm-eval (ja_leaderboard, 500 items each), base vs released.
- **Samples:** seed-0 outputs chosen by a seeded random draw stratified by genre × format, never for quality.
## D-011: Cross-model labels; DPO labeler differs from the evaluation judge

- Each generator labels the other generator's stories. All kept stories carry a cross-label (`label_sources` in the
  data statistics).
- DPO preferences come from the teacher (Qwen3.6-35B-A3B); the evaluation judge is llm-jp-4, another family, so DPO
  cannot optimize directly for the judge's taste.
- Model selection (bake-off, ablations) never touches test prompts.
## D-012: No sequence packing

Packing two sequences into one row with reset `position_ids` changes the logits of the second sequence by up to 14.5
(repeat-pass noise 0.0) for Gemma 4 E4B, because attention is not masked by position ids in this stack. Packing would
leak context across examples, so training uses `packing: false` with length-grouped batches.
## D-016: Stack checks before spending

- Merges are done in fp32 and rounded once to bf16; the check reports teacher-forced top-1 agreement and mean KL
  against the unmerged adapter (98.2–99.0 % and about 0.0012 for the released models).
- vLLM 0.30 serves Gemma 4 text-only and with LoRA; FlashInfer's sampler is disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`)
  so the slim image needs no CUDA compiler.
## D-017 to D-020: Data findings from the probe and inspection

- Qwen3.6's simplified-Chinese contamination rises with temperature (0.7: 5 %, 0.8: 22 %, 1.0: 60 %), so generation
  runs at ≤ 0.75 with an explicit instruction against simplified Chinese.
- English words and markup appeared in Japanese prose; a `latin_leak` filter drops Latin runs of three or more letters
  (except short all-caps status terms such as HP/MP), markup characters and prompt meta-text. The same check is an
  evaluation metric.
- Manual inspection of 100 records added a title filter and a CP932 kanji check, and removed two lexicon false
  positives (a blocklisted name matching inside longer names, and a meta-text pattern matching ordinary prose).
- Final Japanese data: 10,257 kept of 15,320 generations, 10,090 train / 302 validation.
