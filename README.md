# Kitsune Tales

Original fantasy light-novel fiction from a 4.6B-effective-parameter model, in Japanese and English.

## Reproduce

See [docs/REPRODUCE.md](docs/REPRODUCE.md). In short: `uv sync`, log in to the cloud GPU account and add the secrets, then run
the orchestrator tracks; `python -m kitsune.eval.report`, `python -m kitsune.figures` and `python -m kitsune.readme` rebuild every table
and figure from `reports/`.

## Data

15,320 Japanese and 7,640 English generations from Qwen3.6-35B-A3B and
Gemma 4 26B-A4B (both Apache-2.0); each model labels the other's stories. Datasets:
[kitsune-tales-jp-fantasy-sft](https://huggingface.co/datasets/whoashish115/kitsune-tales-jp-fantasy-sft) and [kitsune-tales-en-fantasy-sft](https://huggingface.co/datasets/whoashish115/kitsune-tales-en-fantasy-sft); details in
[docs/DATA_CARD.md](docs/DATA_CARD.md).

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/data_funnel.svg"><img src="reports/figures/data_funnel.png" alt="Data funnel" width="100%"></picture><br><sub>From generations to training examples.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/data_rejections.svg"><img src="reports/figures/data_rejections.png" alt="Rejection reasons" width="100%"></picture><br><sub>The ten most frequent rejection reasons per language.</sub></p>

## Training

Nine logged runs (pilot, four ablations, two SFT, two DPO), all on [W&B](https://wandb.ai/whoashish115-base/kitsune-tales); trainer logs are in
`reports/train_logs/` and interactive curves are on the [site](https://kitsune-tales-qwen.vercel.app#training).

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/sft_loss.svg"><img src="reports/figures/sft_loss.png" alt="SFT loss" width="100%"></picture><br><sub>SFT loss: training (every 10 steps, smoothed) and validation.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/sft_dynamics.svg"><img src="reports/figures/sft_dynamics.png" alt="SFT dynamics" width="100%"></picture><br><sub>Learning rate, gradient norm, token accuracy and entropy during SFT.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/sft_runs.svg"><img src="reports/figures/sft_runs.png" alt="All SFT runs" width="100%"></picture><br><sub>All seven SFT runs on a common axis of training examples seen.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/ablations.svg"><img src="reports/figures/ablations.png" alt="Ablations" width="100%"></picture><br><sub>Ablations: data share (10 / 30 / 100 %) and LoRA rank (16 / 64 at 25 %).</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/dpo_training.svg"><img src="reports/figures/dpo_training.png" alt="DPO" width="100%"></picture><br><sub>DPO loss, held-out preference accuracy and reward margin.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/dpo_rewards.svg"><img src="reports/figures/dpo_rewards.png" alt="DPO rewards" width="100%"></picture><br><sub>DPO implicit rewards and log-probabilities of chosen vs rejected answers.</sub></p>
