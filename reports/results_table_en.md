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
