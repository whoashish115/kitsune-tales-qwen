"""Upload the released weights from the Modal volume to the (private) Hugging Face repos.

    modal run --profile kitsune30 src/modal_hub.py

A separate app from ``modal_app`` so its ``huggingface`` secret is only needed here. CPU only: the files already sit on
the volume, so this is a network copy (about 59 GB: two merged bf16 models, the English LoRA, four GGUF files).
Cards, figures and the Japanese LoRA are uploaded from the workstation by ``python -m kitsune.hub upload-local``.
"""

from __future__ import annotations

import modal
from kitsune import versions

app = modal.App("kitsune-hub")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.33.0", "hf_xet")
    .add_local_python_source("kitsune")
)
data_vol = modal.Volume.from_name("kitsune-data")

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


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3 * 3600,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def upload(jobs: list[tuple[str, str, list[str]]]) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    done = []
    for repo, folder, allow in jobs:
        api.create_repo(repo, private=True, exist_ok=True)  # never flips an existing repo's visibility
        info = api.upload_folder(
            repo_id=repo,
            folder_path=folder,
            allow_patterns=allow,
            commit_message=f"Weights from {folder.removeprefix('/data/')}",
        )
        files = [f for f in api.list_repo_files(repo) if not f.startswith(("figures/", "."))]
        done.append({"repo": repo, "commit": getattr(info, "oid", ""), "files": files})
        print("uploaded", repo, len(files), "files")
    return done


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    from kitsune.modal_app import ledger

    with ledger(
        "8", "hf-upload-weights", "CPU", 0.5, 4, 8, notes="merged JP/EN, EN LoRA, 4 GGUF to private HF repos"
    ):
        res = upload.remote(JOBS)
    Path("reports/hf_upload.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
