# Data card: Kitsune synthetic Japanese fantasy fiction

> Counts, hashes and filter statistics come from `reports/data/stats.json` and `reports/data_en/stats.json`; nothing here is estimated by hand.
## Summary

Instruction-tuning pairs (request → Japanese fantasy text) in three formats (あらすじ synopsis, 短編 short story,
続き continuation), across a fixed 9-genre fantasy taxonomy. Plus a small set of templated policy examples
(refusals and off-genre redirects).
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
