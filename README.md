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
