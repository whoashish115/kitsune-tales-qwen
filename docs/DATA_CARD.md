# Data card

> Counts, hashes and filter statistics come from `reports/data/stats.json` and `reports/data_en/stats.json`; nothing here is estimated by hand.

## Summary

Instruction-tuning pairs (request → Japanese fantasy text) in three formats (あらすじ synopsis, 短編 short story,
続き continuation), across a fixed 9-genre fantasy taxonomy. Plus a small set of templated policy examples
(refusals and off-genre redirects).

## Sources and licenses

| Source | What | License | Provenance recorded |
|---|---|---|---|
| Seed prompts (ours) | Title templates and word lists, genre combinations, formats | Apache-2.0 | `src/data/seeds.py`, fixed seeds |
| LLM-brainstormed titles | Titles proposed by the generator from our templates | Apache-2.0 generator; see below | per-record `meta.title_source = "llm"` |
| Generator 1 | `Qwen/Qwen3.6-35B-A3B-FP8` @ `95a723d08a9490559dae23d0cff1d9466213d989` | Apache-2.0 | per-record `generator` = model@revision |
| Generator 2 | `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | Apache-2.0 | same |
| Refusal texts | Fixed templates (`kitsune.data.policy.refusal_text`) | Apache-2.0 | `source = "seed"` |

**Output-use terms.** Both generators are released under Apache-2.0, which grants use, reproduction and derivative
rights. Unlike some community model licenses, it does not restrict how model outputs are used, including training
other models. No closed-API model outputs and no scraped web text are used.

**No real web-novel text.** Japanese light-novel and web-novel corpora on the Hub either have unclear provenance or
restrictive terms. For example, Syosetu works are copyrighted by their individual authors, so none are used (D-006).

## Schema

One JSON object per line (`src/schema.py`, validated in tests):
`id, genres (1–3, taxonomy), title, format, prompt, response, language, source (real|synthetic|seed),
generator, license, filters_passed, hash (sha256 of normalized prompt+response), meta`.
`meta` holds the sampling parameters, style knobs (POV, tone, opening, protagonist), label source (cross/self),
LLM labels, and for 続き the passage length.

## Generation

- Style knobs are sampled per request (POV × tone × opening × style × protagonist) to avoid collapse into one voice.
- A sampling grid of temperature 0.7–1.0, top-p 0.9–0.95 and presence penalty 0.5–1.0 is drawn per request.
- `続き` pairs come from separate "source" stories, cut at sentence boundaries: a 200–400-character passage, then
  400–800 characters of continuation.

## Filtering

Filters run in order; every dropped story keeps its first failing reason.

1. Generation finished (no truncation at the token cap).
2. No template artifacts (thinking tags, chat tokens, prompt echoes, code fences).
3. Japanese purity: Japanese-script ratio ≥ 0.90, hiragana ratio ≥ 0.15, and zero simplified-Chinese-only glyphs.
   A glyph counts as simplified-only if it cannot be encoded in CP932, which is verified in tests.
4. Length bounds per format.
5. Repetition: char 8-gram uniqueness ≥ 0.85, no line repeated more than twice, zlib ratio ≥ 0.18, ≤ 10 % repeated 30-char chunks.
6. Fantasy: ≥ 2 distinct fantasy-lexicon terms (≥ 1 for 続き), **and** the LLM label `fantasy`.
7. Safety: explicit-sexual lexicon (hard drop; sexual + minor markers labeled separately), self-harm, heavy gore,
   **and** the LLM label `general_audience`.
8. PII (email, URL, phone, postal code, card-like numbers).
9. Real people and existing IP: blocklist **and** the LLM label.
10. Title and tag consistency: title keywords and per-genre cue words, **and** LLM `genre_match`/`title_match` ≥ 1.
11. LLM quality label ≥ 3 of 5.
12. Deduplication: exact (normalized) and near-duplicate (MinHash LSH over char 5-grams, confirmed by exact Jaccard ≥ 0.7).

LLM labels come preferably from the *other* generator (cross-labeling). Self-labels are used only where no cross-label
exists, and the count of each kind is reported.

## Splits

- **Test:** 270 held-out prompts (9 genres × 3 formats × 10), built before any generation, SHA-256
  `16b6214e3e281c82f7c732f98017e442b71f1a0f42eebbf819c7bbdde688b599` (`data/test_prompts.jsonl`).
  Titles near any test title (char-bigram Jaccard ≥ 0.6) are excluded from training.
- **Eval policy suite:** 72 prompts (disallowed, off-genre, adversarial), using names and titles disjoint from the training policy prompts,
  SHA-256 `713c07479107e84a3ee93d2a4bc7c2d0b40290e9b2a1cbb4e3fd2975850aea75`.
- **Validation:** 3 % of story records, stratified by primary genre × format.

## Statistics

_Filled from `reports/data/stats.json` and the plots in `reports/data/`: filter funnel, length histograms, genre × format grid._

## Limitations

- The style is inherited from two generator models, so the data reflects their habits, clichés and blind spots.
- The taxonomy and title templates emphasize popular web-novel tropes (isekai, villainess, guild).
- Rule-based filters are lexicon-based and can both over-block (for example 淫魔) and miss paraphrased content.
  LLM labels reduce but do not remove this.
- Manual inspection covers a seeded random sample of 100 records (`reports/data/inspection_sample.md`), not the whole set.

## English dataset

`kitsune-tales-en-fantasy-sft` (D-024).

The English model's data comes from the same pipeline, with English-specific prompts, filters and labeler instructions
(`src/en.py`). The genre taxonomy, schema (`language = "en"`) and split procedure are shared. Every number
below comes from `reports/data_en/stats.json`.

| | gen1 (Qwen3.6-35B-A3B-FP8) | gen2 (Gemma 4 26B-A4B) | Total |
|---|---|---|---|
| Generations | 4,520 | 3,120 | 7,640 |
| Kept after filters + dedup | 3,895 | 2,810 | 6,705 (88 %) |
| LLM labels | cross-labeled by gen2 | cross-labeled by gen1 | 100 % cross-labels |

- **Splits:** 6,647 train / 193 validation, which includes 135 templated refusals and off-genre redirects.
  SHA-256: train `de150d8b…c4`, val `48a8e4c7…3e`.
- **Test:** 270 English prompts (9 × 3 × 10) and a 72-prompt policy suite, frozen before any English generation:
  `data/test_prompts_en.jsonl` `f0f098fd…29` and `data/eval_policy_prompts_en.jsonl` `bc7576e9…06`.
  The test set with its 90 frozen continuation passages is `data/test_set_en.jsonl` `8173bf22…9d`.
  Training titles near a test title (char-bigram Jaccard ≥ 0.85, a looser bar than Japanese because English titles
  share more function words) are excluded.
- **Lengths in words:** synopsis 150–350, short story 600–1,100, continuation 300–600 (passage 120–250). Filters allow a
  margin of 110–420, 550–1,250 and 220–700 words respectively.
- **English filters:** script purity (no CJK, kana, Hangul, Cyrillic, Devanagari or Thai characters), meta-text leakage,
  in-story "self-corrections" (see below), markdown, length, repetition, fantasy lexicon, safety and PII, real
  people and existing IP, and title and tag consistency. The safety regexes for story text were narrowed after
  inspection, because "a suicide mission", "lustrous" and "a naked blade" are ordinary fantasy prose. Request
  screening keeps the broad lists.
- **Name debiasing.** Generator 1 used the same few invented names everywhere: "Elara" in 67 % of its stories,
  "Kaelen" in 37 %, "Aethelgard" in 34 %. Generator 2's prompts therefore suggest a protagonist name from varied,
  gender-matched pools and list the overused names to avoid. The pipeline then replaces any remaining overused names
  deterministically per sample (`rebalance_names_en`), consistently across passage and continuation, and never when
  the name is in the title. Afterwards the most frequent invented name appears in 7.7 % of training stories ("Jiro"),
  down from 67 % ("Elara") before.
- **Prompt-leak filter.** The avoid-names instruction occasionally leaked into gen2's prose as an in-story
  self-correction ("the village of Millbrook—no, wait, he corrected himself…"). 46 such records are dropped by the
  `self_correction` filter.
- **Known residual habits:** about 8 % of short stories and 17 % of synopses quote the full title verbatim, and a
  Western-fantasy default register is common despite the Japanese light-novel framing.
