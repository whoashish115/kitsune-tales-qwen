"""Container images and functions for every cloud GPU/CPU job in the project.
Run from the repo root with the project venv, choosing the account explicitly:

    make gpu JOB=smoke
    make gpu JOB=bakeoff
    make gpu JOB="data --mode probe"
    ...
Every GPU function has an explicit ``timeout`` and explicit CPU/memory (billed on top of the
GPU, see docs/BUDGET.md). Every local entrypoint writes an estimate to the cost ledger and
passes the per-account kill-threshold guard *before* launching, then records the measured
wall-clock *after* (``kitsune.cost``).
"""
from __future__ import annotations
import gzip
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import modal
from kitsune import cost, versions
APP_NAME = "kitsune"
app = modal.App(APP_NAME)
MODELS_DIR = "/models"  # HF cache for pre-downloaded weights
DATA_DIR = "/data"  # datasets, generations, checkpoints, eval outputs
models_vol = modal.Volume.from_name("kitsune-models", create_if_missing=True)
data_vol = modal.Volume.from_name("kitsune-data", create_if_missing=True)
VOLS = {MODELS_DIR: models_vol, DATA_DIR: data_vol}
wandb_secret = modal.Secret.from_name("wandb")
_ENV = {
    "HF_HOME": MODELS_DIR,
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # Same reason for MoE/FP8 paths: stay on vLLM's prebuilt Triton/CUTLASS kernels (no JIT toolchain needed).
    "VLLM_USE_FLASHINFER_MOE_FP8": "0",
    "VLLM_USE_FLASHINFER_MOE_FP16": "0",
    "VLLM_USE_DEEP_GEMM": "0",
}
cpu_image = (
    modal.Image.debian_slim(python_version=versions.PYTHON_VERSION)
    .uv_pip_install(
        *(
            f"{p}=={versions.GPU_PACKAGES[p]}"
            for p in ("huggingface-hub", "wandb", "pydantic", "pyyaml", "datasketch")
        ),
        "hf-xet==1.6.0",
        "numpy==2.3.5",
    )
    .env(_ENV)
    .add_local_python_source("kitsune")
)
gpu_base = (
    modal.Image.debian_slim(python_version=versions.PYTHON_VERSION)
    .apt_install("git")
    .uv_pip_install(
        *(f"{p}=={v}" for p, v in versions.GPU_PACKAGES.items()),
        "hf-xet==1.6.0",
        "numpy==2.3.5",
        "matplotlib==3.11.2",
    )
    .env(_ENV)
)
gpu_image = gpu_base.add_local_python_source("kitsune")
MODEL_IDS: dict[str, tuple[str, str]] = {
    "base": (versions.BASE_MODEL, versions.BASE_REVISION),
    "alt": (versions.ALT_BASE_MODEL, versions.ALT_BASE_REVISION),
    "gen1": (versions.GENERATOR_MODEL, versions.GENERATOR_REVISION),
    "gen2": (versions.GENERATOR2_MODEL, versions.GENERATOR2_REVISION),
    "judge": (versions.JUDGE_MODEL, versions.JUDGE_REVISION),
}

# =========================================================================== local helpers

def _git_commit() -> str:
    raise NotImplementedError

def _upload(files: dict[str, str]) -> None:
    """Upload local files to the data Volume: {local_path: remote_path}."""
    with data_vol.batch_upload(force=True) as batch:
        for local, remote in files.items():
            batch.put_file(local, remote)

def _download(remote: str, local: str) -> None:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as f:
        for chunk in data_vol.read_file(remote):
            f.write(chunk)

def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def _read_jsonl_any(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        return [json.loads(x) for x in f if x.strip()]

# =========================================================================== Phase 2: hello + weights

@app.function(
    scaledown_window=2, image=cpu_image, cpu=1, memory=1024, timeout=600, volumes={MODELS_DIR: models_vol}
)
def delete_weights(repo_id: str) -> str:
    """Remove a model from the Volume as soon as its phase ends ($0.09/GiB/month)."""
    import shutil

    d = Path(MODELS_DIR) / "hub" / ("models--" + repo_id.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d)
        models_vol.commit()
        return f"deleted {d}"
    return f"not present: {d}"

@app.function(scaledown_window=2, image=gpu_image, gpu="L4", cpu=4, memory=32768, timeout=1500, volumes=VOLS)
def smoke_vllm(use_adapter: bool) -> dict:
    from kitsune.smoke import _guard, check_vllm

    data_vol.reload()
    adapter = f"{DATA_DIR}/smoke/smoke-train/adapter" if use_adapter else None
    return _guard("vllm", check_vllm, versions.BASE_MODEL, versions.BASE_REVISION, adapter)

@app.local_entrypoint()
def smoke() -> None:
    _ensure_weights("base")
    with ledger("2", "smoke-hf", "L4", 0.5, 4, 64, notes="load/packing/train/merge checks"):
        hf = smoke_hf.remote()
    with ledger("2", "smoke-vllm", "L4", 0.25, 4, 32, notes="vLLM text-only + LoRA serving"):
        vl = smoke_vllm.remote(bool(hf.get("train", {}).get("ok")))
    out = {"hf": hf, "vllm": vl}
    _save_json("reports/smoke.json", out)
    print(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "trace"} if isinstance(v, dict) else v
                for k, v in hf.items()
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        )[:6000]
    )
    print(json.dumps({k: v for k, v in vl.items() if k != "trace"}, ensure_ascii=False, indent=1)[:3000])

@app.function(scaledown_window=2, image=gpu_image, gpu="L4", cpu=4, memory=65536, timeout=1800, volumes=VOLS)
def smoke_merge() -> dict:
    """Re-run only the merge check on the smoke adapter (fp32 merge + KL/top-1 metrics)."""
    from kitsune.smoke import _guard
    from kitsune.train.merge import merge_and_verify

    data_vol.reload()
    res = _guard(
        "merge",
        merge_and_verify,
        versions.BASE_MODEL,
        versions.BASE_REVISION,
        f"{DATA_DIR}/smoke/smoke-train/adapter",
        f"{DATA_DIR}/smoke/smoke-merged",
        48,
    )
    data_vol.commit()
    return res

@app.local_entrypoint()
def bakeoff() -> None:
    import random

    from kitsune.data.seeds import build_train_prompts
    from kitsune.fixtures import STORY
    from kitsune.schema import read_jsonl

    test_titles = [p["title"] for p in read_jsonl("data/test_prompts.jsonl")]
    seeds = build_train_prompts(12, test_titles, seed=424242)  # disjoint from test by construction
    prompts = []
    for s in seeds:
        d = s.to_dict()
        if d["format"] == "続き":
            d["passage"] = STORY[:300]
        prompts.append(d)
    random.Random(0).shuffle(prompts)
    _ensure_weights("base,alt")
    rows = []
    for key in ("base", "alt"):
        with ledger("2", f"bakeoff-{key}", "L4", 0.25, 4, 32, notes=MODEL_IDS[key][0]):
            rows += bakeoff_generate.remote(key, prompts)
    _save_json("reports/bakeoff/generations.json", rows)
    from kitsune.eval.bakeoff import summarize

    summary = summarize(rows)
    _save_json("reports/bakeoff/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
