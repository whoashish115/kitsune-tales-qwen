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
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which the slim image lacks; use vLLM's torch sampler.
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

@contextmanager
def ledger(
    phase: str,
    job: str,
    gpu: str,
    est_hours: float,
    cpu: float,
    mem_gib: float,
    planned_remaining: float = 0.0,
    notes: str = "",
) -> Iterator[cost.LedgerEntry]:
    """Guard + ledger row before launch; measured wall-clock after (an upper bound on billed time)."""
    e = cost.open_job(phase, job, gpu, est_hours, cpu, mem_gib, planned_remaining, notes)
    print(f"[budget] {e.account}: launching {job} on {gpu}, estimate ${e.est_usd:.3f} ({est_hours:.2f} h)")
    t0 = time.time()
    try:
        yield e
    finally:
        c = cost.close_job(job, (time.time() - t0) / 3600)
        acct_total = cost.spent(cost.read_ledger(), c.account)
        print(f"[budget] {job}: measured ${c.actual_usd:.3f}; {c.account} total ${acct_total:.2f}")

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

def _save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")



def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _read_jsonl_any(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        return [json.loads(x) for x in f if x.strip()]

# =========================================================================== Phase 2: hello + weights

@app.function(
    scaledown_window=2, image=cpu_image, gpu="T4", cpu=1, memory=1024, timeout=300, secrets=[wandb_secret]
)
def hello_gpu(git_commit: str, cumulative_usd: float) -> dict:
    """Prove the GPU, the W&B secret and run logging work (logs a tiny dummy run)."""
    import subprocess

    import wandb

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    run = wandb.init(
        project=versions.WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY"),
        group="setup",
        job_type="hello",
        name="hello-gpu",
        config={"git_commit": git_commit, "gpu": smi},
    )
    for step in range(5):
        run.log({"dummy/loss": 1.0 / (step + 1), "cost/cumulative_usd": cumulative_usd}, step=step)
    url = run.url
    run.finish()
    return {"nvidia_smi": smi, "wandb_url": url}

@app.local_entrypoint()
def hello() -> None:
    raise NotImplementedError

@app.function(
    scaledown_window=2, image=cpu_image, cpu=4, memory=8192, timeout=3600, volumes={MODELS_DIR: models_vol}
)
def download_weights(repo_id: str, revision: str) -> dict:
    """Download a snapshot into the models Volume on a CPU container (no GPU billed)."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(repo_id, revision=revision, max_workers=16)
    models_vol.commit()
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {"repo": repo_id, "gb": round(size / 1e9, 2), "minutes": round((time.time() - t0) / 60, 1)}

def _ensure_weights(which: str) -> None:
    for w in which.split(","):
        repo, rev = MODEL_IDS[w]
        with ledger("infra", f"download-{w}", "CPU", 0.3, 4, 8, notes=repo):
            print(download_weights.remote(repo, rev))

@app.local_entrypoint()
def download(which: str = "base") -> None:
    raise NotImplementedError

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

@app.local_entrypoint()
def purge(which: str) -> None:
    for w in which.split(","):
        print(delete_weights.remote(MODEL_IDS[w][0]))

@app.function(scaledown_window=2, image=gpu_image, gpu="L4", cpu=4, memory=65536, timeout=2400, volumes=VOLS)
def smoke_hf() -> dict:
    from kitsune.smoke import run_all

    res = run_all(versions.BASE_MODEL, versions.BASE_REVISION, f"{DATA_DIR}/smoke")
    data_vol.commit()
    return res

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
def smoke_rerun(parts: str = "vllm,merge") -> None:
    """Re-run selected smoke checks and update reports/smoke.json."""
    rep = json.loads(Path("reports/smoke.json").read_text(encoding="utf-8"))
    if "vllm" in parts:
        with ledger("2", "smoke-vllm-rerun", "L4", 0.2, 4, 32, notes="after VLLM_USE_FLASHINFER_SAMPLER=0"):
            rep["vllm"] = smoke_vllm.remote(True)
    if "merge" in parts:
        with ledger("2", "smoke-merge-rerun", "L4", 0.15, 4, 64, notes="fp32 merge + KL/top-1"):
            rep["hf"]["merge"] = smoke_merge.remote()
    _save_json("reports/smoke.json", rep)
    for k in ("vllm",):
        print(
            json.dumps({kk: vv for kk, vv in rep[k].items() if kk != "trace"}, ensure_ascii=False, indent=1)[
                :3000
            ]
        )
    print(json.dumps({kk: vv for kk, vv in rep["hf"]["merge"].items() if kk != "trace"}, indent=1)[:2000])

BAKEOFF_GPU = {"base": "L4", "alt": "L4"}

@app.function(scaledown_window=2, image=gpu_image, gpu="L4", cpu=4, memory=32768, timeout=1800, volumes=VOLS)
def bakeoff_generate(model_key: str, prompts: list[dict]) -> list[dict]:
    """Zero-shot generations for the D-001 bake-off (prompts are NOT from the test set)."""
    raise NotImplementedError

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



def _pull(mode: str) -> None:
    raise NotImplementedError

@app.local_entrypoint()
def data_pull(mode: str = "full") -> None:
    """Copy raw generations/labels for ``mode`` from the Volume into data/raw/<mode>/."""
    raise NotImplementedError

@app.local_entrypoint()
def data_push(lang: str = "ja") -> None:
    """Upload the processed dataset (built locally by the pipeline) for training."""
    d = "processed_en" if lang == "en" else "processed"
    _upload({f"data/{d}/train.jsonl": f"/{d}/train.jsonl", f"data/{d}/val.jsonl": f"/{d}/val.jsonl"})
    print(f"uploaded data/{d}/{{train,val}}.jsonl")

# =========================================================================== Phases 4-5: training

@app.function(scaledown_window=2, image=gpu_image, gpu="L4", cpu=4, memory=65536, timeout=3600, volumes=VOLS)
def merge_fn(
    adapter_run: str, base_model: str = versions.BASE_MODEL, base_revision: str = versions.BASE_REVISION
) -> dict:
    from kitsune.train.merge import merge_and_verify

    data_vol.reload()
    rep = merge_and_verify(
        base_model,
        base_revision,
        f"{DATA_DIR}/runs/{adapter_run}/adapter",
        f"{DATA_DIR}/merged/{adapter_run}",
    )
    data_vol.commit()
    return rep

# =========================================================================== Phase 6: judge, perplexity, regression
