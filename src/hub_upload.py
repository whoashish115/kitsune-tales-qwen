"""Upload the released weights from the cloud volume to the (private) Hugging Face repos.

    make upload
A separate app from ``gpu_jobs`` so its ``huggingface`` secret is only needed here. CPU only: the files already sit on
the volume, so this is a network copy (about 59 GB: two merged bf16 models, the English LoRA, four GGUF files).
Cards, figures and the Japanese LoRA are uploaded from the workstation by ``python -m kitsune.hub sync``.
"""
import modal
from kitsune import versions
from __future__ import annotations
app = modal.App("kitsune-hub")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.33.0", "hf_xet")
    .add_local_python_source("kitsune")
)
data_vol = modal.Volume.from_name("kitsune-data")
