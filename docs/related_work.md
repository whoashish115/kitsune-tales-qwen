# Literature scan (Phase 1)

Only sources that were actually opened are listed. The **Read** column says how deeply:
`abstract` = abstract/landing page only, `card` = model card or documentation page.
`REPORT.md` cites only from this list.

## Methods

| Source | Read | Takeaway used in this project |
|---|---|---|
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685 (2021) | abstract | Frozen base with trainable low-rank updates. The update merges into the weights, so the merged model adds no inference latency, which we verify (merged vs adapter-on-base check). |
| Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314 (2023) | abstract | NF4 + double quantization + paged optimizers. Not used: Qwen3.5 guidance advises against 4-bit training (D-002). |
| Biderman et al., *LoRA Learns Less and Forgets Less*, TMLR (2024), arXiv:2405.09673 | abstract | LoRA preserves out-of-domain ability better than full FT and keeps generations more diverse. Motivates LoRA plus a general-capability regression check. |
| Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR (2020), arXiv:1904.09751 | abstract | Likelihood-maximizing decoding yields bland, repetitive text; nucleus sampling helps. Motivates sampling-based decoding and the repetition/degenerate-output metrics. |
| Lee et al., *Deduplicating Training Data Makes Language Models Better*, ACL (2022), arXiv:2107.06499 | abstract | Near-duplicate removal reduces memorization about 10× and train–test overlap. Motivates the MinHash dedup and the title-level test-set exclusion. |

## Evaluation

| Source | Read | Takeaway |
|---|---|---|
| Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS D&B (2023), arXiv:2306.05685 | abstract | Judges show position, verbosity and self-enhancement biases. We use position swapping and a length-aware rubric. |
| Wang et al., *Large Language Models are not Fair Evaluators*, arXiv:2305.17926 (2023) | abstract | Ranking can flip just by swapping order. We aggregate over both orders and report the order-disagreement rate. |
| Panickssery, Bowman, Feng, *LLM Evaluators Recognize and Favor Their Own Generations*, arXiv:2404.13076 (2024) | abstract | Self-preference correlates with self-recognition. The judge is from a different family than both the data generator and the base model (D-004). |
| Fein et al., *LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing*, arXiv:2507.00769 (2025) | abstract | The best off-the-shelf judge agreed with human story preferences only ~73 % of the time. Our judge results are reported as indicative, not definitive. |
| Lin, Zheng, Wang, *WebNovelBench: Placing LLM Novelists on the Web Novel Distribution*, arXiv:2505.14818 (2025) | abstract | Synopsis-to-story framing with multi-dimension LLM judging (Chinese web novels). Informs our `あらすじ`/`短編` formats and rubric dimensions. |
| Li et al., *A Diversity-Promoting Objective Function for Neural Conversation Models*, NAACL (2016), arXiv:1510.03055 | abstract | Origin of the distinct-n diversity metrics (definitions are in the paper body, which was not read; we define ours explicitly in `src/eval/metrics.py`). |
| Zhu et al., *Texygen: A Benchmarking Platform for Text Generation Models*, arXiv:1802.01886 (2018) | abstract | Origin of Self-BLEU (definition not on the abstract page; ours is defined explicitly in code). |

## Models and tooling (cards and docs)

| Source | Read | Takeaway |
|---|---|---|
| Qwen/Qwen3.5-4B model card (Hugging Face) | card | Apache-2.0; hybrid Gated DeltaNet/Gated Attention; `enable_thinking=False`; non-thinking sampling defaults; `-Base` variant exists. |
| Unsloth, *Qwen3.5 Fine-tuning Guide* | card | bf16 LoRA ≈ 10 GB for 4B; QLoRA not recommended; transformers v5 required; templates must match at GGUF export. |
| tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2 model card; Swallow project page | card | Apache-2.0; reasoning cannot be toggled off. |
| llm-jp/llm-jp-4-8b-thinking model card; NII press release (2026-04-03) | card | Apache-2.0; 8.6B; 64k context; Harmony-style template. |
| Qualiteg, *Japanese LLM Rankings 2026 (Sep 1 edition)*, Nejumi Leaderboard 4 excerpts | card | Sub-10B Japanese scores used in the D-001 table. |
| lilting.ch, *9 Japanese LLMs in April 2026 compared* | card | Qualitative notes (thinking-budget issues of llm-jp-4-32B-A3B; Nemotron JP license). |
| Artificial Analysis, *Qwen3.5 small models*; Codersera Qwen 3.5–3.8 guide | card | Release dates, sizes, license, hybrid attention layout of Qwen3.5/3.6/3.8. |
