# Kitsune Tales

Original fantasy light-novel fiction in Japanese from a small fine-tuned model.

## Reproduce

See [docs/REPRODUCE.md](docs/REPRODUCE.md). In short: `uv sync`, log in to the cloud GPU account and add the secrets, then run
the orchestrator tracks; `python -m kitsune.eval.report`, `python -m kitsune.figures` and `python -m kitsune.readme` rebuild every table
and figure from `reports/`.
