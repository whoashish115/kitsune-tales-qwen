# Compute

All GPU and CPU work ran on [Modal](https://modal.com) across two accounts with fixed credit. The total billed by Modal
for this project is **$41.34**; neither account exceeded its credit.

| Account (CLI profile) | Credit | Hard stop | Billed | Used for |
|---|---:|---:|---:|---|
| `kitsune30` | $30.00 | $29.60 | **$28.29** | smoke tests, data generation, pilot, main SFT, English DPO, evaluation, merges, GGUF |
| `kitsune12` | $14.28 | $13.90 | **$13.06** | ablations, Japanese DPO labels and training, baselines, judges, lm-eval |
| **Total** | $44.28 | | **$41.34** | |
## Budget guard

Every job goes through `kitsune.cost.guard` before launch. It compares the account's hard stop with

    max(Modal billed total, ledger including running jobs' estimates) + 1.25 × the new job's estimate

and refuses to launch past it. Each job writes a ledger row (`reports/cost_ledger.jsonl`) with its estimate before it
starts and its measured wall-clock cost when it ends. Measured over the project, the ledger was 0.96 to 1.12 × Modal's
billing per account, so recorded spend counts at face value (D-014, D-025, D-028).
## Prices

From `modal billing rates` (September 2026). CPU and memory are billed on top of the GPU rate:
`cost = hours × (gpu_rate + cores × 0.0473 + GiB × 0.008)`, so an H100 job with 8 cores and 64 GiB costs $4.84 per hour.

| Resource | Rate |
|---|---|
| Nvidia L4 | $0.80 / h |
| Nvidia L40S | $1.95 / h |
| Nvidia A100 80 GB | $2.50 / h |
| Nvidia H100 | $3.95 / h |
| CPU | $0.0473 / core / h |
| Memory | $0.008 / GiB / h |
| Volumes | $0.09 / GiB / month |
## Efficiency choices

- **Cost per example, not per hour.** The H100 trains about 4× more examples per hour than an L40S at 1.7× the price,
  so every SFT run uses an H100 (D-022). Generation and judging use vLLM on one H100.
- **No GPU time on downloads.** Base, generator and judge weights are cached on a Modal volume by CPU-only jobs before
  any GPU job starts.
- **One epoch at batch 16**, sized from the pilot's measured throughput (D-021).
- **Explicit timeouts and resumable jobs.** Training resumes from checkpoints only when its data fingerprint matches
  (D-023); generation and judging skip finished shards.
## Release upload

The merged weights, the English LoRA and the GGUF files (about 59 GB) are copied from the Modal volume to the Hugging
Face repos by a CPU-only job (`src/modal_hub.py`: 4 cores, 8 GiB, estimated 0.5 h, about $0.16 with the guard factor).
`upload_folder` skips files the Hub already has, so a rerun is cheap. Measured: about 3 minutes, $0.013.
