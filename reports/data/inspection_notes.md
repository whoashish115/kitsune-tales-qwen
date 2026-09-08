# Manual inspection gate: 100 random training records (seed 99)

Inspected on 2026-09-30 from `reports/data/inspection_compact.txt` (opening ≈ 260 chars and closing ≈ 110 chars of each
record, from the first full build). The sample covered both generators, all 9 genres and all 3 formats, plus policy examples.

## Overall

- **Quality is good.** Fluent Japanese, light-novel conventions (「」 dialogue, short paragraphs, ⏎-separated beats),
  coherent arcs in 短編, and complete summaries with endings in あらすじ. 続き continuations pick up mid-scene as intended.
- Both generators are usable. gen2 (Gemma 4 26B) writes longer, more polished prose and uses full-width indentation (「　」).
  gen1 (Qwen3.6-35B) is more varied in plot but noisier (see below).
- The off-genre redirect example (#080, 猫と暮らすワンルーム) carries the fixed prefix and a genuine fantasy transposition. Refusals are correct.

## Problems found → action

| # | Problem | Examples | Action |
|---|---|---|---|
| 1 | LLM-brainstormed **titles** were never purity-checked | `耳रक्षक` (#039), `傳說` (#066), `元・银の杖使い` (#069), `つらくirsch` (#073), `冒险者` (#077); corpus-wide 168 titles (1.7 %) | New `title_clean` filter (purity, Latin leak, Cyrillic/Devanagari/Hangul/Thai). Dataset rebuilt (D-020) |
| 2 | Simplified glyphs missing from the curated list | `诊察室` (#041); corpus-wide 150 stories (1.5 %) with non-CP932 kanji (`罢 沉 孽 诅 笺`…) | Purity now rejects **any** kanji outside CP932. Rebuilt |
| 3 | Stray markdown | single `*` italic marker (#042) | Rare; noted. The markdown metric tracks it in eval |
| 4 | Stylistic collapse | many endings of the form 「…物語は、まだ始まったばかりだ」; recurring names (アストラル, エルゼ・フォン・ローゼンタール/ローゼリア); recurring 忘却の森 / 紫色の霧 | Not filterable without harming quality. Reported as a known bias; diversity measured with distinct-n / self-BLEU |
| 5 | Protagonist skew | 老騎士 and 元冒険者の中年 over-represented (2 of 8 protagonist knobs) | Noted as a data bias |
| 6 | Occasional POV inconsistency within a story (僕 ↔ わたくし, #031) | 1 case in 100 | Generator error; not filterable by rules; noted |

No sexual content, real people, existing IP (beyond policy refusals) or PII was seen in the 100 records.
