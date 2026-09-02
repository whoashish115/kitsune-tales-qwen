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
