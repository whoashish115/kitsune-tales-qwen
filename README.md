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

## Results

| Measured on held-out prompts | Base model | Kitsune | n |
|---|---:|---:|---:|
| Japanese stories within the requested length | 0.4 % | **50 %** | 270 × 3 |
| English stories within the requested length | 22 % | **81 %** | 270 × 3 |
| Outputs with markdown or meta text (JP / EN) | 76 % / 95 % | **0 % / 0 %** | 270 × 3 |
| Disallowed requests carried out anyway, JP | 85 % | **5 %** | 45 × 3 |
| Disallowed requests carried out anyway, EN | 79 % | **0 %** | 45 × 3 |
| Judge net preference vs base, full outputs (JP / EN) | | -0.44 [-0.54, -0.34] / -0.57 [-0.66, -0.47] | 145 / 150 |
| Judge net preference vs base, equal-length openings (JP / EN) | | +0.12 [-0.07, +0.32] / -0.03 [-0.22, +0.15] | 145 / 150 |
| Validation perplexity (JP / EN) | 6.67 / 7.59 | **3.31 / 3.27** | |

n = held-out prompts × seeds (or judged pairs). Brackets are 95 % bootstrap CIs. The judge prefers the base model on
full outputs because it writes far past the requested length; on equal-length openings the difference is not
significant in either language.

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/eval_metrics.svg"><img src="reports/figures/eval_metrics.png" alt="Automatic metrics" width="100%"></picture><br><sub>Automatic metrics for every system, mean and 95 % CI. Test suite: 270 prompts × 3 seeds; policy suite: 72 prompts × 3 seeds.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/eval_lengths.svg"><img src="reports/figures/eval_lengths.png" alt="Output length" width="100%"></picture><br><sub>Output length against the requested range (shaded), all test generations.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/eval_length_grid.svg"><img src="reports/figures/eval_length_grid.png" alt="Length adherence by genre and format" width="100%"></picture><br><sub>Length adherence by primary genre and format, base vs released model.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/judge_preference.svg"><img src="reports/figures/judge_preference.png" alt="Judge net preference" width="100%"></picture><br><sub>Pairwise judge: full outputs (grey) vs equal-length openings (blue). Most of the base model's lead is length.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/judge_outcomes.svg"><img src="reports/figures/judge_outcomes.png" alt="Judge win, tie and loss" width="100%"></picture><br><sub>The same comparisons as win / tie / loss shares, with order consistency.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/judge_validation.svg"><img src="reports/figures/judge_validation.png" alt="Judge validation" width="100%"></picture><br><sub>The judge picks the intact story over a corrupted copy in 86.7 % (JP) and 93.3 % (EN) of known-answer pairs.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/safety.svg"><img src="reports/figures/safety.png" alt="Safety outcomes" width="100%"></picture><br><sub>Held-out disallowed requests (45 prompts × 3 seeds per language): refusals, safe redirects, violations.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/eval_diversity.svg"><img src="reports/figures/eval_diversity.png" alt="Lexical diversity" width="100%"></picture><br><sub>Lexical diversity: corpus distinct-n and self-BLEU across seeds.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/lmeval.svg"><img src="reports/figures/lmeval.png" alt="JGLUE" width="100%"></picture><br><sub>General Japanese ability on JGLUE (lm-eval, 500 items per task): no clear regression.</sub></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/dark/ppl_leakage.svg"><img src="reports/figures/ppl_leakage.png" alt="Perplexity and leakage" width="100%"></picture><br><sub>Validation perplexity halves; verbatim overlap with the training stories stays near zero.</sub></p>

Full per-system tables with confidence intervals are in [REPORT.md](REPORT.md#6-results).
