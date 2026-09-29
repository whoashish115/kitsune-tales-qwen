# Reproduce

Every GPU step runs on rented cloud GPUs and records an estimate and then the measured cost in `reports/cost_ledger.jsonl`.
`kitsune.cost.guard` refuses to launch a job that would push an account past its kill threshold (`docs/BUDGET.md`).
Estimates below are the *planned* figures. The measured figures are in the ledger and in `docs/BUDGET.md`.

## 0. Setup (free)

```bash
git clone <this repo> && cd kitsune
uv sync                              # Python 3.12, locked dependencies (uv.lock)
uv run pytest -q                     # 125 CPU tests; downloads the pinned tokenizer (~10 MB)
make gpu-login                       # log in to the cloud GPU account (once)
make gpu-secrets                     # stores WANDB_API_KEY, WANDB_ENTITY and HF_TOKEN from your shell; W&B project: kitsune-tales
```

## 1. Test set

Free, on CPU.

```bash
uv run python -m kitsune.data.cli verify-test   # the committed files must match data/test_prompts.sha256
```

## 2. Smoke test

With the base-model bake-off; about $1 on an L4.

```bash
make gpu JOB=smoke   # load, packing, LoRA, merge, vLLM
make gpu JOB=bakeoff   # Qwen3.5-4B vs Gemma 4 E4B, zero-shot
```

## 3. Data

About $6–10 on an H100.

```bash
# 500-prompt probe: measures throughput and filter pass rate before sizing the full run
make gpu JOB="data --mode probe --gen gen1 --n-prompts 500"
# full runs (sizes set from the probe; see docs/DECISIONS.md)
make gpu JOB="data --mode full --gen gen1 --n-prompts 11000 --title-calls 60"
make gpu JOB="data --mode full --gen gen2 --n-prompts 5500 --title-calls 30"
make gpu JOB="data --mode full --gen gen1 --label-only"   # cross-label gen2
uv run python -m kitsune.data.pipeline build --raw data/raw/full        # filters → dedup → split, stats and plots
make gpu JOB=data_push
make gpu JOB="purge --which gen1,gen2"   # stop paying for storage
```

## 4–5. Training

About $1.5 for the pilot, $5 for the main run and $3 for DPO.

```bash
make gpu JOB="train --config configs/train_pilot.yaml --name sft-pilot"
make gpu JOB="train --config configs/train_main.yaml --name sft-main --est-hours 1.2"
make gpu JOB="merge --adapter sft-main"
make gpu JOB="dpo --sft-run sft-main --name dpo-main"
make gpu JOB="merge --adapter dpo-main"
```

Ablations (LoRA rank, data scaling) use the same `train` job with the `configs/ablation_*.yaml` files.

## 6. Evaluation

About $5.

```bash
make gpu JOB="eval_generate --systems base,kitsune-sft,kitsune"
make gpu JOB="eval_generate --systems qwen3.5-9b,gemma-4-e4b,teacher"
make gpu JOB=eval_judge
make gpu JOB=eval_ppl
make gpu JOB=lm_eval ACCOUNT=kitsune12
make eval        # CPU: rebuilds reports/results.json, results_table.md, figures, README/REPORT tables
```

`make eval` needs only the committed `reports/` files, so anyone can re-derive every reported number for free.

## 7. Release

About $0.02 on CPU.

```bash
make upload                                        # merged weights, LoRA and GGUF files from the cloud volume to the Hub
uv run python -m kitsune.hub stage                 # cards, figures, datasets and the Space into local mirrors
uv run python -m kitsune.hub sync                  # upload the mirrors to the Hub
uv run python -m kitsune.hub upload-space          # the ZeroGPU Space
```
