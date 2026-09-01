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
