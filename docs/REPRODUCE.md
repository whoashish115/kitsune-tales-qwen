# Reproducing Kitsune from a fresh clone

Every GPU step runs on Modal and records an estimate and then the measured cost in `reports/cost_ledger.jsonl`.
`kitsune.cost.guard` refuses to launch a job that would push an account past its kill threshold (`docs/BUDGET.md`).
Estimates below are the *planned* figures. The measured figures are in the ledger and in `docs/BUDGET.md`.

## 0. Setup (free)

```bash
git clone <this repo> && cd kitsune
uv sync                              # Python 3.12, locked dependencies (uv.lock)
uv run pytest -q                     # 92+ CPU tests; downloads the pinned tokenizer (~10 MB)
modal token new --profile kitsune30  # Modal account for the main pipeline
modal secret create wandb WANDB_API_KEY=... WANDB_ENTITY=... --profile kitsune30   # W&B project: kitsune-tales
```

## 1. Freeze the held-out test set (free, CPU)

```bash
uv run python -m kitsune.data.cli verify-test   # the committed files must match data/test_prompts.sha256
```

## 2. Smoke test and base-model bake-off (≈ $1, L4)

```bash
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::smoke     # load, packing, LoRA, merge, vLLM
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::bakeoff   # Qwen3.5-4B vs Gemma 4 E4B, zero-shot
```

## 3. Data (≈ $6–10, H100)

```bash
# 500-prompt probe: measures throughput and filter pass rate before sizing the full run
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::data --mode probe --gen gen1 --n-prompts 500
# full runs (sizes set from the probe; see docs/DECISIONS.md)
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::data --mode full --gen gen1 --n-prompts 11000 --title-calls 60
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::data --mode full --gen gen2 --n-prompts 5500 --title-calls 30
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::data --mode full --gen gen1 --label-only   # cross-label gen2
uv run python -m kitsune.data.pipeline build --raw data/raw/full        # filters → dedup → split, stats and plots
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::data_push
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::purge --which gen1,gen2   # stop paying for storage
```

## 4–5. Training (≈ $1.5 pilot, ≈ $5 main, ≈ $3 DPO)

```bash
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::train --config configs/train_pilot.yaml --name sft-pilot
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::train --config configs/train_main.yaml --name sft-main --est-hours 1.2
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::merge --adapter sft-main
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::dpo --sft-run sft-main --name dpo-main
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::merge --adapter dpo-main
```

Ablations (LoRA rank, data scaling) run on the second account with the same `train` entrypoint and the
`configs/ablation_*.yaml` files.

## 6. Evaluation (≈ $5)

```bash
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::eval_generate --systems base,kitsune-sft,kitsune
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::eval_generate --systems qwen3.5-9b,gemma-4-e4b,teacher
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::eval_judge
MODAL_PROFILE=kitsune30 uv run modal run src/modal_app.py::eval_ppl
MODAL_PROFILE=kitsune12 uv run modal run src/modal_app.py::lm_eval
make eval        # CPU: rebuilds reports/results.json, results_table.md, figures, README/REPORT tables
```

`make eval` needs only the committed `reports/` files, so anyone can re-derive every reported number for free.
