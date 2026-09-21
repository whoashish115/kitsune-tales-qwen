"""Upload the released weights from the Modal volume to the (private) Hugging Face repos.

    modal run --profile kitsune30 src/modal_hub.py
from __future__ import annotations
A separate app from ``modal_app`` so its ``huggingface`` secret is only needed here. CPU only: the files already sit on
the volume, so this is a network copy (about 59 GB: two merged bf16 models, the English LoRA, four GGUF files).
Cards, figures and the Japanese LoRA are uploaded from the workstation by ``python -m kitsune.hub upload-local``.
"""
import modal
from kitsune import versions
# (repo, folder on the volume, files to include)
JOBS: list[tuple[str, str, list[str]]] = [
    (versions.HF_MODEL_REPO, "/data/merged/sft-main", ["*.safetensors", "*.json", "chat_template.jinja"]),
    (
        versions.HF_MODEL_REPO_EN,
        "/data/merged/dpo-en-main",
        ["*.safetensors", "*.json", "chat_template.jinja"],
    ),
    (
        versions.HF_ADAPTER_REPO_EN,
        "/data/runs/dpo-en-main/adapter",
        ["adapter_*", "tokenizer*", "chat_template.jinja"],
    ),
    (versions.HF_GGUF_REPO, "/data/gguf/sft-main", ["*.gguf"]),
    (versions.HF_GGUF_REPO_EN, "/data/gguf/dpo-en-main", ["*.gguf"]),
]
