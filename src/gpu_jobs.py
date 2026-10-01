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

# =========================================================================== container helpers

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
    with ledger("2", "phase2-hello-gpu", "T4", 0.05, 1, 1, notes="GPU + W&B + secret check"):
        print(hello_gpu.remote(_git_commit(), cost.spent(cost.read_ledger())))

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

# =========================================================================== Phase 2: smoke + bake-off

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
    from kitsune.data.generate import ChatEngine, GenJob
    from kitsune.prompts import SYSTEM_PROMPT, StoryRequest, build_user_prompt
    from kitsune.taxonomy import FORMATS

    model, rev = MODEL_IDS[model_key]
    eng = ChatEngine(model, rev, max_model_len=4096, gpu_memory_utilization=0.90, enforce_eager=True)
    jobs = []
    for p in prompts:
        user = build_user_prompt(StoryRequest(p["genres"], p["title"], p["format"], p.get("passage")))
        jobs.append(
            GenJob(
                p["id"],
                "bakeoff",
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.05},
                FORMATS[p["format"]].max_new_tokens,
            )
        )
    t0 = time.time()
    out = eng.chat(jobs, seed=0)
    dt = time.time() - t0
    for o, p in zip(out, prompts, strict=True):
        o.update({"system": model_key, "prompt": p, "gen_seconds_total": dt, "load_kwargs": eng.load_kwargs})
    return out

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

# =========================================================================== Phase 3: data

@app.function(
    scaledown_window=2,
    image=gpu_image,
    gpu="H100",
    cpu=8,
    memory=40960,
    timeout=6 * 3600,
    volumes=VOLS,
    secrets=[wandb_secret],
)
def data_run(gen_key: str, mode: str, cfg: dict) -> dict:
    """One generator container: [test passages] → title brainstorm → stories → self-labels → cross-labels.

    Resumable: every stage writes shards to the Volume and skips shards that already exist.
    """
    from kitsune.data.dedup import TitleIndex
    from kitsune.data.generate import ChatEngine, GenJob
    from kitsune.data.labels import label_job, structured_output_kwargs
    from kitsune.data.plan import clean_llm_titles, story_plan, test_passage_plan, title_brainstorm_plan

    lang = cfg.get("lang", "ja")
    if lang == "en":  # D-024: same stages, English plans/labeler prompts
        from kitsune.data.plan_en import (
            clean_llm_titles_en,
            story_plan_en,
            test_passage_plan_en,
            title_brainstorm_plan_en,
        )
        from kitsune.en import label_job_en

    data_vol.reload()
    root = Path(DATA_DIR) / "raw" / mode
    root.mkdir(parents=True, exist_ok=True)
    model, rev = MODEL_IDS[gen_key]
    eng = ChatEngine(
        model,
        rev,
        max_model_len=cfg.get("max_model_len", 4096),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.92),
        chat_template_kwargs=cfg.get("chat_template_kwargs"),
        max_num_seqs=cfg.get("max_num_seqs", 256),
    )
    stats: dict[str, Any] = {
        "gen_key": gen_key,
        "model": model,
        "revision": rev,
        "mode": mode,
        "load_kwargs": eng.load_kwargs,
        "stages": {},
    }

    def run_stage(name: str, jobs: list[GenJob], shard: int = 2000, extra: dict | None = None) -> list[dict]:
        outs: list[dict] = []
        t0, ntok, nprompt = time.time(), 0, 0
        for i in range(0, len(jobs), shard):
            path = root / f"{name}_{i // shard:03d}.jsonl.gz"
            if path.exists():
                outs += _read_jsonl_any(str(path))
                continue
            chunk = jobs[i : i + shard]
            res = eng.chat(chunk, seed=cfg.get("seed", 0) + i, extra_sampling=extra)
            rows = []
            for j, r in zip(chunk, res, strict=True):
                ntok += r["n_tokens"]
                nprompt += r["n_prompt_tokens"]
                rows.append(
                    {
                        **r,
                        "kind": j.kind,
                        "generator": f"{model}@{rev[:12]}",
                        "gen_key": gen_key,
                        "sampling": j.sampling,
                        "meta": j.meta,
                    }
                )
            _write_jsonl_gz(str(path), rows)
            data_vol.commit()
            outs += rows
        dt = time.time() - t0
        stats["stages"][name] = {
            "n": len(jobs),
            "seconds": round(dt, 1),
            "gen_tokens": ntok,
            "prompt_tokens": nprompt,
            "gen_tokens_per_s": round(ntok / dt, 1) if dt > 0 else None,
        }
        print(f"[data_run] {name}: {stats['stages'][name]}")
        return outs

    test_prompts = _read_jsonl_any(
        f"{DATA_DIR}/frozen/{'test_prompts_en' if lang == 'en' else 'test_prompts'}.jsonl"
    )
    test_titles = [p["title"] for p in test_prompts]
    gens: list[dict] = []
    if not cfg.get("label_only"):
        if gen_key == "gen1" and mode in ("full", "full_en"):
            tp_plan = test_passage_plan_en if lang == "en" else test_passage_plan
            run_stage("test_passages", tp_plan(test_prompts, seed=9))
        tb_plan = title_brainstorm_plan_en if lang == "en" else title_brainstorm_plan
        titles_out = (
            run_stage(f"titles_{gen_key}", tb_plan(cfg["title_calls_per_genre"], seed=cfg["seed"] + 5))
            if cfg["title_calls_per_genre"]
            else []
        )
        llm_titles = (
            clean_llm_titles_en(titles_out, test_titles)
            if lang == "en"
            else clean_llm_titles(titles_out, TitleIndex(test_titles))
        )
        stats["llm_titles_per_genre"] = {g: len(v) for g, v in llm_titles.items()}
        plan = (story_plan_en if lang == "en" else story_plan)(
            gen_key,
            cfg["n_prompts"],
            llm_titles,
            test_titles,
            cfg["llm_title_share"],
            cfg["offgenre_per_prompt"],
            seed=cfg["seed"],
        )
        gens = run_stage(f"gen_{gen_key}", plan)

    # Labels: this model labels its own samples (self) and every other generator's samples present (cross).
    lab_extra = structured_output_kwargs()
    to_label = [(g, gen_key) for g in gens if g["kind"] in {"story", "synopsis", "source", "offgenre"}]
    for other in sorted(root.glob("gen_gen*_000.jsonl.gz")):
        other_key = other.name.split("_")[1]
        if other_key != gen_key:
            for f in sorted(root.glob(f"gen_{other_key}_*.jsonl.gz")):
                to_label += [
                    (g, other_key)
                    for g in _read_jsonl_any(str(f))
                    if g["kind"] in {"story", "synopsis", "source", "offgenre"}
                ]
    ljobs = []
    for g, _src in to_label:
        seed = g["meta"].get("seed") or {}
        pol = g["meta"].get("policy") or {}
        genres = seed.get("genres") or ["ハイファンタジー"]
        title = seed.get("title") or pol.get("title", "")
        fmt = seed.get("format") or pol.get("format", "短編")
        if lang == "en":  # ~1000-word stories are ~6k characters; keep the whole text
            ljobs.append(label_job_en(g["id"], genres, title, fmt, g["text"][:7000], gen_key))
        else:
            ljobs.append(label_job(g["id"], genres, title, fmt, g["text"][:3000], gen_key))
    stage = f"labels_by_{gen_key}{'_cross' if cfg.get('label_only') else ''}"
    stats["label_structured_outputs"] = sorted(lab_extra)
    try:
        labels = run_stage(stage, ljobs, shard=4000, extra=lab_extra)
    except Exception as e:  # a decoding-backend failure must not waste the finished generations
        stats["label_structured_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        stats["label_structured_outputs"] = []
        labels = run_stage(stage + "_free", ljobs, shard=4000, extra=None)
    n_valid = 0
    from kitsune.data.labels import parse_labels

    for r in labels:
        n_valid += parse_labels(r["text"]) is not None
    stats["n_labels"] = len(labels)
    stats["n_labels_valid_json"] = n_valid
    (root / f"stats_{gen_key}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    data_vol.commit()
    return stats

@app.local_entrypoint()
def data(
    mode: str = "probe",
    gen: str = "gen1",
    n_prompts: int = 500,
    title_calls: int = 3,
    est_hours: float = 0.5,
    planned_remaining: float = 0.0,
    label_only: bool = False,
    lang: str = "ja",
) -> None:
    """Generate (probe or full) with one generator; then pull outputs locally into data/raw/<mode>/.

    ``--lang en`` (D-024) writes to ``/raw/<mode>`` with English plans; use ``--mode full_en``.
    """
    import yaml

    dcfg = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
    gcfg = dcfg["generator1" if gen == "gen1" else "generator2"]
    ecfg = dcfg["english"] if lang == "en" else {}
    cfg = {
        "n_prompts": n_prompts,
        "title_calls_per_genre": title_calls,
        "llm_title_share": ecfg.get("llm_title_share", gcfg.get("llm_title_share", 0.5)),
        "offgenre_per_prompt": ecfg.get("offgenre_samples_per_prompt", dcfg["offgenre_samples_per_prompt"])
        if mode.startswith("full")
        else 1,
        "seed": (1000 if gen == "gen1" else 2000) + (2000 if lang == "en" else 0),
        "max_model_len": gcfg.get("max_model_len", 4096),
        "gpu_memory_utilization": gcfg.get("gpu_memory_utilization", 0.92),
        "label_only": label_only,
        "lang": lang,
    }
    if lang == "en":
        _upload({"data/test_prompts_en.jsonl": "/frozen/test_prompts_en.jsonl"})
    else:
        _upload({"data/test_prompts.jsonl": "/frozen/test_prompts.jsonl"})
    _ensure_weights(gen)
    with ledger(
        "3",
        f"data-{mode}-{gen}{'-label' if label_only else ''}",
        "H100",
        est_hours,
        8,
        40,
        planned_remaining,
        notes=f"{MODEL_IDS[gen][0]} n={n_prompts}",
    ):
        stats = data_run.remote(gen, mode, cfg)
    _save_json(f"reports/data/run_{mode}_{gen}_{'label' if label_only else 'gen'}.json", stats)
    _pull(mode)
    print(json.dumps(stats, ensure_ascii=False, indent=1))

def _pull(mode: str) -> None:
    n = 0
    for e in data_vol.listdir(f"/raw/{mode}"):
        name = Path(e.path).name
        _download(e.path, f"data/raw/{mode}/{name}")
        n += 1
    print(f"pulled {n} files into data/raw/{mode}/")

@app.local_entrypoint()
def data_pull(mode: str = "full") -> None:
    """Copy raw generations/labels for ``mode`` from the Volume into data/raw/<mode>/."""
    _pull(mode)

@app.local_entrypoint()
def data_push(lang: str = "ja") -> None:
    """Upload the processed dataset (built locally by the pipeline) for training."""
    d = "processed_en" if lang == "en" else "processed"
    _upload({f"data/{d}/train.jsonl": f"/{d}/train.jsonl", f"data/{d}/val.jsonl": f"/{d}/val.jsonl"})
    print(f"uploaded data/{d}/{{train,val}}.jsonl")

# =========================================================================== Phases 4-5: training

@app.function(
    scaledown_window=2,
    image=gpu_image,
    gpu="H100",
    cpu=8,
    memory=49152,
    timeout=8 * 3600,
    volumes=VOLS,
    secrets=[wandb_secret],
)
def train_fn(
    cfg: dict, run_name: str, gpu: str, prior_spend: float, git_commit: str, data_subdir: str = "processed"
) -> dict:
    from kitsune.train.sft import train as sft_train

    data_vol.reload()
    return sft_train(
        cfg,
        f"{DATA_DIR}/{data_subdir}",
        f"{DATA_DIR}/runs",
        run_name,
        gpu,
        cfg.get("cpu", 8),
        cfg.get("memory_gib", 48),
        prior_spend,
        git_commit,
        commit_fn=data_vol.commit,
    )

@app.local_entrypoint()
def train(config: str, name: str, est_hours: float = 0.5, planned_remaining: float = 0.0) -> None:
    import yaml

    if Path(f"reports/train/{name}.json").exists():  # idempotent: a restarted orchestrator never re-trains
        print(f"[train] {name}: already trained (reports/train/{name}.json), skipping")
        return

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    _ensure_weights("base")  # on CPU, so the GPU container never waits on a download
    gpu, cpu, mem = cfg.get("gpu", "H100"), cfg.get("cpu", 8), cfg.get("memory_gib", 48)
    fn = train_fn.with_options(
        gpu=gpu, cpu=cpu, memory=int(mem * 1024), timeout=int(cfg.get("timeout_h", 4) * 3600)
    )
    with ledger(
        "5" if "main" in name else "4",
        f"train-{name}",
        gpu,
        est_hours,
        cpu,
        mem,
        planned_remaining,
        notes=config,
    ):
        summary = fn.remote(
            cfg, name, gpu, cost.spent(cost.read_ledger()), _git_commit(), cfg.get("data_subdir", "processed")
        )
    _save_json(f"reports/train/{name}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

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

# =========================================================================== Phase 6: evaluation

@app.function(
    scaledown_window=2, image=gpu_image, gpu="L4", cpu=4, memory=32768, timeout=3 * 3600, volumes=VOLS
)
def eval_generate_fn(
    system: str,
    model: str,
    revision: str | None,
    seeds: list[int],
    decoding: dict,
    max_new: dict,
    engine_kwargs: dict,
    lang: str = "ja",
) -> dict:
    """Generate the full test + policy suites for one system; writes /data/eval/generations/<system>.jsonl.gz.

    ``lang="en"`` uses the English test set, policy suite and system prompt (D-024) and writes to
    /data/eval/generations_en/.
    """
    from kitsune import en
    from kitsune.data.generate import ChatEngine, GenJob
    from kitsune.prompts import SYSTEM_PROMPT, StoryRequest, build_user_prompt

    data_vol.reload()
    sfx = "_en" if lang == "en" else ""
    test = _read_jsonl_any(f"{DATA_DIR}/frozen/test_set{sfx}.jsonl")
    policy = _read_jsonl_any(f"{DATA_DIR}/frozen/eval_policy_prompts{sfx}.jsonl")
    system_prompt = en.SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT
    eng = ChatEngine(model, revision or "main", **engine_kwargs)
    rows: list[dict] = []
    t0 = time.time()
    for s in seeds:
        jobs, meta = [], []
        for p in test:
            user = (
                en.build_user_prompt_en(p["genres"], p["title"], p["format"], p.get("passage"))
                if lang == "en"
                else build_user_prompt(StoryRequest(p["genres"], p["title"], p["format"], p.get("passage")))
            )
            jobs.append(
                GenJob(
                    p["id"],
                    "eval",
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
                    decoding,
                    max_new[p["format"]],
                )
            )
            meta.append(
                {
                    "suite": "test",
                    "prompt_id": p["id"],
                    "format": p["format"],
                    "genres": p["genres"],
                    "title": p["title"],
                    "passage": p.get("passage"),
                }
            )
        for p in policy:
            user = (
                en.policy_user_prompt_en(p)
                if lang == "en"
                else f"ジャンル: {p['genres_text']}\nタイトル: {p['title']}\n形式: {p['format']}"
            )
            jobs.append(
                GenJob(
                    p["id"],
                    "eval",
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
                    decoding,
                    max_new[p["format"]],
                )
            )
            meta.append(
                {
                    "suite": "policy",
                    "prompt_id": p["id"],
                    "kind": p["kind"],
                    "format": p["format"],
                    "title": p["title"],
                }
            )
        for r, m in zip(eng.chat(jobs, seed=10_000 * (s + 1)), meta, strict=True):
            rows.append(
                {
                    "system": system,
                    "seed": s,
                    **m,
                    "text": r["text"],
                    "finish_reason": r["finish_reason"],
                    "n_tokens": r["n_tokens"],
                }
            )
    out = f"{DATA_DIR}/eval/generations{sfx}/{system}.jsonl.gz"
    _write_jsonl_gz(out, rows)
    data_vol.commit()
    return {
        "system": system,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "gen_tokens": sum(r["n_tokens"] for r in rows),
        "load_kwargs": eng.load_kwargs,
    }

@app.local_entrypoint()
def eval_generate(systems: str, gpu: str = "L4", est_hours: float = 0.3) -> None:
    """Generate eval suites for the named systems (comma-separated keys of configs/eval.yaml)."""
    import yaml

    ecfg = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))
    _upload(
        {
            f"data/{f}{sfx}.jsonl": f"/frozen/{f}{sfx}.jsonl"
            for f in ("test_set", "eval_policy_prompts")
            for sfx in ("", "_en")
            if Path(f"data/{f}{sfx}.jsonl").exists()
        }
    )
    for system in systems.split(","):
        sc = ecfg["systems"][system]
        lang = sc.get("lang", "ja")
        sfx = "_en" if lang == "en" else ""
        local = f"reports/generations{sfx}/{system}.jsonl.gz"
        if Path(local).exists():
            print(f"[eval_generate] {system}: generations already exist, skipping (idempotent)")
            continue
        max_new = ecfg["max_new_tokens_en" if lang == "en" else "max_new_tokens"]
        model = f"{DATA_DIR}/merged/{sc['merged']}" if sc.get("merged") else sc["model"]
        engine = {"max_model_len": 4096, "gpu_memory_utilization": 0.90, **sc.get("engine", {})}
        g = sc.get("gpu", gpu)
        fn = eval_generate_fn.with_options(gpu=g)
        with ledger("6", f"evalgen-{system}", g, est_hours, 4, 32, notes=model):
            info = fn.remote(
                system,
                model,
                None if sc.get("merged") else sc.get("revision"),
                ecfg["seeds"],
                ecfg["decoding"],
                max_new,
                engine,
                lang,
            )
        _download(f"/eval/generations{sfx}/{system}.jsonl.gz", local)
        print(info)

# =========================================================================== Phase 6: judge, perplexity, regression
@app.local_entrypoint()
def eval_judge(
    comparisons: str = "kitsune:base,kitsune:kitsune-sft,kitsune:teacher",
    n_pairs: int = 150,
    n_validation: int = 60,
    judge: str = "judge",
    est_hours: float = 0.6,
    excerpt: str = "",
    excerpt_pairs: int = 60,
) -> None:
    """Pairwise comparisons (both orders) + known-answer validation with the eval judge (D-004).

    ``excerpt`` (e.g. "kitsune-sft:base") adds length-matched comparisons of 600-character short-story openings.
    """
    print("[debug] eval_judge", flush=True)
    from kitsune.eval.judge_plan import comparison_jobs, excerpt_jobs, validation_jobs
    from kitsune.schema import read_jsonl, write_jsonl

    gens = {p.name.split(".")[0]: list(read_jsonl(p)) for p in Path("reports/generations").glob("*.jsonl.gz")}
    jobs, files = [], {}
    for comp in comparisons.split(","):
        x, y = comp.split(":")
        js = comparison_jobs(gens[x], gens[y], x, y, n_pairs)
        files.update({j.id: f"reports/judge/{judge}__{x}__vs__{y}.jsonl" for j in js})
        jobs += js
    for comp in filter(None, excerpt.split(",")):
        x, y = comp.split(":")
        js = excerpt_jobs(gens[x], gens[y], x, y, excerpt_pairs, 600)
        files.update({j.id: f"reports/judge/{judge}__excerpt-{x}__vs__excerpt-{y}.jsonl" for j in js})
        jobs += js
    vj = validation_jobs(list(read_jsonl("data/processed/val.jsonl")), n_validation) if n_validation else []
    files.update({j.id: f"reports/judge/{judge}__validation.jsonl" for j in vj})
    jobs += vj
    _ensure_weights(judge)
    with ledger("6", f"judge-{judge}", "H100", est_hours, 8, 64, notes=f"{len(jobs)} calls"):
        res = judge_fn.remote(judge, [j.to_dict() for j in jobs])
    by_file: dict[str, list[dict]] = {}
    for r in res:
        by_file.setdefault(files[r["id"]], []).append(r)
    for f, rows in by_file.items():
        write_jsonl(f, rows)
        print(f, len(rows), "invalid:", sum(r["verdict"] is None for r in rows))

@app.local_entrypoint()
def eval_judge_en(
    comparisons: str = "kitsune-en:base-en,kitsune-en:kitsune-en-sft",
    n_pairs: int = 150,
    n_validation: int = 60,
    judge: str = "judge",
    est_hours: float = 0.6,
    excerpt: str = "kitsune-en:base-en",
    excerpt_pairs: int = 60,
) -> None:
    """English pairwise judge (D-024): same protocol as ``eval_judge`` (both orders + known-answer pairs), plus
    length-matched comparisons of 2,000-character short-story openings for ``excerpt``."""
    from kitsune.eval.judge_plan import comparison_jobs, excerpt_jobs, validation_jobs
    from kitsune.schema import read_jsonl, write_jsonl

    gens = {
        p.name.split(".")[0]: list(read_jsonl(p)) for p in Path("reports/generations_en").glob("*.jsonl.gz")
    }
    jobs, files = [], {}
    for comp in comparisons.split(","):
        x, y = comp.split(":")
        js = comparison_jobs(gens[x], gens[y], x, y, n_pairs, lang="en")
        files.update({j.id: f"reports/judge_en/{judge}__{x}__vs__{y}.jsonl" for j in js})
        jobs += js
    for comp in filter(None, excerpt.split(",")):
        x, y = comp.split(":")
        js = excerpt_jobs(gens[x], gens[y], x, y, excerpt_pairs, 2000, lang="en")
        files.update({j.id: f"reports/judge_en/{judge}__excerpt-{x}__vs__excerpt-{y}.jsonl" for j in js})
        jobs += js
    vj = validation_jobs(list(read_jsonl("data/processed_en/val.jsonl")), n_validation, lang="en")
    files.update({j.id: f"reports/judge_en/{judge}__validation.jsonl" for j in vj})
    jobs += vj
    _ensure_weights(judge)
    with ledger("6", f"judge-en-{judge}", "H100", est_hours, 8, 64, notes=f"{len(jobs)} calls"):
        res = judge_fn.remote(judge, [j.to_dict() for j in jobs], 12288, None, "en")
    by_file: dict[str, list[dict]] = {}
    for r in res:
        by_file.setdefault(files[r["id"]], []).append(r)
    for f, rows in by_file.items():
        write_jsonl(f, rows)
        print(f, len(rows), "invalid:", sum(r["verdict"] is None for r in rows))

@app.function(scaledown_window=2, image=gpu_image, gpu="L4", cpu=4, memory=32768, timeout=3600, volumes=VOLS)
def ppl_fn(models: dict[str, str], max_length: int = 2048, lang: str = "ja") -> dict:
    """Assistant-token perplexity on the validation split, with the same tokenization as training."""
    import math as _m

    import torch
    from kitsune.en import SYSTEM_PROMPT_EN
    from kitsune.prompts import SYSTEM_PROMPT, tokenize_example
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_vol.reload()
    val = _read_jsonl_any(f"{DATA_DIR}/{'processed_en' if lang == 'en' else 'processed'}/val.jsonl")
    system = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT
    out = {}
    for name, path in models.items():
        rev = versions.BASE_REVISION if path == versions.BASE_MODEL else None
        tok = AutoTokenizer.from_pretrained(path, revision=rev)
        m = AutoModelForCausalLM.from_pretrained(
            path, revision=rev, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        nll, n = 0.0, 0
        with torch.no_grad():
            for r in val:
                try:
                    ex = tokenize_example(tok, r["prompt"], r["response"], max_length, system=system)
                except ValueError:
                    continue
                ids = torch.tensor([ex["input_ids"]], device="cuda")
                labels = torch.tensor([ex["labels"]], device="cuda")
                k = int((labels[:, 1:] != -100).sum())
                nll += float(m(input_ids=ids, labels=labels).loss) * k
                n += k
        out[name] = {"nll_per_token": nll / n, "ppl": _m.exp(nll / n), "n_tokens": n, "n_examples": len(val)}
        del m
        torch.cuda.empty_cache()
    return out

def _release(lang: str) -> dict:
    import yaml

    return yaml.safe_load(Path("configs/release.yaml").read_text(encoding="utf-8"))[lang]

def _system_path(ecfg: dict, s: str) -> tuple[str, str | None]:
    sc = ecfg["systems"][s]
    if sc.get("merged"):
        return f"{DATA_DIR}/merged/{sc['merged']}", None
    return sc["model"], sc.get("revision")

@app.function(
    scaledown_window=2, image=lmeval_image, gpu="L40S", cpu=4, memory=32768, timeout=3 * 3600, volumes=VOLS
)
def lm_eval_fn(path: str, revision: str | None, tasks: list[str], limit: int) -> dict:
    import lm_eval

    data_vol.reload()
    args = f"pretrained={path},dtype=bfloat16" + (f",revision={revision}" if revision else "")
    res = lm_eval.simple_evaluate(
        model="hf",
        model_args=args,
        tasks=tasks,
        limit=limit,
        batch_size="auto",
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    keep = {
        "results": res["results"],
        "n-samples": res.get("n-samples"),
        "config": {"model_args": args, "tasks": tasks, "limit": limit},
        "versions": res.get("versions"),
    }
    return json.loads(json.dumps(keep, default=str))

# =========================================================================== Phase 5b: DPO

@app.function(
    scaledown_window=2, image=gpu_image, gpu="L40S", cpu=4, memory=32768, timeout=2 * 3600, volumes=VOLS
)
def dpo_sample_fn(merged_sft: str, n_prompts: int, seed: int = 0, lang: str = "ja") -> dict:
    """Two samples per training prompt from the SFT model, for DPO pair construction."""
    print("[debug] dpo_sample_fn", flush=True)
    import random as _r

    from kitsune.data.generate import ChatEngine, GenJob
    from kitsune.en import FORMATS_EN, SYSTEM_PROMPT_EN
    from kitsune.prompts import SYSTEM_PROMPT, render_prompt
    from kitsune.taxonomy import FORMATS
    from transformers import AutoTokenizer

    en = lang == "en"
    system = SYSTEM_PROMPT_EN if en else SYSTEM_PROMPT
    formats = FORMATS_EN if en else FORMATS
    marker = "Passage:\n" if en else "本文:\n"
    data_vol.reload()
    train = [
        r
        for r in _read_jsonl_any(f"{DATA_DIR}/{'processed_en' if en else 'processed'}/train.jsonl")
        if "policy_kind" not in r["meta"]
    ]
    _r.Random(seed).shuffle(train)
    train = train[:n_prompts]
    tok = AutoTokenizer.from_pretrained(merged_sft)
    eng = ChatEngine(merged_sft, "main", max_model_len=4096, gpu_memory_utilization=0.90)

    def mk(r: dict) -> GenJob:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]
        sampling = {"temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.05}
        return GenJob(r["id"], "dpo", msgs, sampling, formats[r["format"]].max_new_tokens)

    a = eng.chat([mk(r) for r in train], seed=seed + 1)
    b = eng.chat([mk(r) for r in train], seed=seed + 100_001)
    rows = []
    for r, x, y in zip(train, a, b, strict=True):
        passage = r["prompt"].split(marker, 1)[1] if marker in r["prompt"] else None
        rows.append(
            {
                "id": r["id"],
                "prompt_text": render_prompt(tok, r["prompt"], system),
                "user_prompt": r["prompt"],
                "format": r["format"],
                "genres": r["genres"],
                "title": r["title"],
                "passage": passage,
                "a": x["text"],
                "b": y["text"],
            }
        )
    _write_jsonl_gz(f"{DATA_DIR}/{'dpo_en' if en else 'dpo'}/samples.jsonl.gz", rows)
    data_vol.commit()
    return {"n": len(rows)}

@app.local_entrypoint()
def dpo(
    sft_run: str = "sft-main",
    name: str = "dpo-main",
    n_prompts: int = 2400,
    config: str = "configs/train_dpo.yaml",
    stage: str = "all",
    until: str = "train",
    lang: str = "ja",
    outdir: str = "",
    safety_pairs: bool = True,
) -> None:
    """DPO: sample (SFT) → label (teacher, both orders) → build pairs → train.

    ``outdir`` (default data/dpo or data/dpo_en) names the local and Volume directory, so a relabeled rerun
    (dpo_v2, D-027) keeps the first run's files intact.

    ``stage`` is the first stage to run and ``until`` the last, so the stages can run on different cloud accounts
    (the English labels run on kitsune12, D-026). Stages whose outputs already exist locally (samples, teacher
    verdicts) are skipped, so re-running after a failure never pays for sampling or labeling twice. Delete them to
    redo a stage. ``lang="en"`` uses data/dpo_en/, the English training prompts, filters and judge rubric.
    """
    import yaml
    from kitsune import en as en_mod
    from kitsune.eval.judge import judge_job
    from kitsune.eval.judge_plan import _request_text
    from kitsune.schema import read_jsonl, write_jsonl
    from kitsune.train.dpo import build_pairs, rule_ok

    en = lang == "en"
    d = outdir or ("dpo_en" if en else "dpo")
    samples_p, verdicts_p, pairs_p = (
        f"data/{d}/samples.jsonl.gz",
        f"data/{d}/teacher_verdicts.jsonl",
        f"data/{d}/pairs.jsonl",
    )
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    stages = ["sample", "label", "pairs", "train"]
    start = 0 if stage == "all" else stages.index(stage)
    last = stages.index(until)
    if start <= 0 and Path(samples_p).exists():
        print(f"[dpo] samples exist ({samples_p}); skipping the sampling stage")
        start = 1
    if start <= 1 and Path(verdicts_p).exists():
        print(f"[dpo] teacher verdicts exist ({verdicts_p}); skipping the labeling stage")
        start = 2
    if start <= 0 <= last:
        _ensure_weights("base")
        with ledger(
            "5b", f"dpo-sample{'-en' if en else ''}", "L40S", 0.4, 4, 32, notes=f"{n_prompts} prompts x2"
        ):
            print(dpo_sample_fn.remote(f"{DATA_DIR}/merged/{sft_run}", n_prompts, 0, lang))
        _download(f"/{d}/samples.jsonl.gz", samples_p)
    if last < 1:
        return
    samples = list(read_jsonl(samples_p))
    if start <= 1:
        jobs = []
        for s in samples:
            if rule_ok(s["a"], s["format"], s["genres"], s["title"], s["passage"], lang) and rule_ok(
                s["b"], s["format"], s["genres"], s["title"], s["passage"], lang
            ):
                req, make = (
                    (en_mod.request_text_en(s), en_mod.judge_job_en) if en else (_request_text(s), judge_job)
                )
                jobs += [
                    make(s["id"], "xy", req, s["a"], s["b"], max_tokens=2048, brief=True),
                    make(s["id"], "yx", req, s["b"], s["a"], max_tokens=2048, brief=True),
                ]
        _ensure_weights("gen1")
        job = f"dpo-label-teacher{'-en' if en else ''}"
        with ledger("5b", job, "H100", 0.35, 8, 64, notes=f"{len(jobs)} judge calls"):
            res = judge_fn.remote("gen1", [j.to_dict() for j in jobs], 8192, None, lang)
        write_jsonl(verdicts_p, res)
    if last < 2:
        return
    verdicts: dict[str, dict] = {}
    for r in read_jsonl(verdicts_p):
        verdicts.setdefault(r["pair_id"], {})[r["order"]] = r["verdict"]
    pairs, counts = build_pairs(samples, verdicts, lang, safety=safety_pairs and d != "dpo_v2")
    write_jsonl(pairs_p, pairs)
    _save_json(f"reports/{d.replace('dpo', 'dpo_pairs', 1)}.json", counts)
    print("pairs:", counts)
    if last < 3:
        return
    if Path(f"reports/train/{name}.json").exists():
        print(f"[dpo] {name}: already trained, skipping")
        return
    _upload({pairs_p: f"/{d}/pairs.jsonl"})
    gpu = cfg.get("gpu", "H100")
    fn = dpo_train_fn.with_options(gpu=gpu)
    with ledger(
        "5b", f"train-{name}", gpu, cfg.get("est_hours", 0.6), cfg.get("cpu", 8), cfg.get("memory_gib", 48)
    ):
        summary = fn.remote(cfg, sft_run, name, gpu, cost.spent(cost.read_ledger()), _git_commit(), d)
    _save_json(f"reports/train/{name}.json", summary)
    print(summary)

@app.function(scaledown_window=2, image=gguf_image, cpu=8, memory=16384, timeout=1800, volumes=VOLS)
def gguf_smoke_fn(merged_run: str, quant: str, lang: str) -> dict:
    """Re-run the llama.cpp smoke generation on an existing GGUF, capturing stderr (no re-conversion)."""
    raise NotImplementedError

@app.local_entrypoint()
def judge_smoke(judge: str = "judge") -> None:
    """De-risk the eval judge early: bake-off pairs (both orders) + 5 known-answer corruption pairs."""
    import random as _r

    from kitsune.eval.judge import CORRUPTIONS, corrupt, judge_job
    from kitsune.eval.judge_plan import _request_text
    from kitsune.fixtures import GENRES, STORY, SYNOPSIS, TITLE

    rows = json.loads(Path("reports/bakeoff/generations.json").read_text(encoding="utf-8"))
    by = {}
    for r in rows:
        by.setdefault(r["prompt"]["id"], {})[r["system"]] = r
    jobs = []
    for pid, d in sorted(by.items()):
        if {"base", "alt"} <= set(d):
            req = _request_text(d["base"]["prompt"])
            jobs += [
                judge_job(f"bake__{pid}", "xy", req, d["base"]["text"], d["alt"]["text"]),
                judge_job(f"bake__{pid}", "yx", req, d["alt"]["text"], d["base"]["text"]),
            ]
    rng = _r.Random(0)
    req = _request_text({"genres": GENRES, "title": TITLE, "format": "短編"})
    for kind in CORRUPTIONS:
        bad = corrupt(STORY, kind, rng, other_story=SYNOPSIS)
        j1, j2 = (
            judge_job(f"val__{kind}", "xy", req, STORY, bad),
            judge_job(f"val__{kind}", "yx", req, bad, STORY),
        )
        j1.meta["corruption"] = j2.meta["corruption"] = kind
        jobs += [j1, j2]
    _ensure_weights(judge)
    with ledger("6", f"judge-smoke-{judge}", "H100", 0.3, 8, 64, notes=f"{len(jobs)} calls"):
        t0 = time.time()
        res = judge_fn.remote(judge, [j.to_dict() for j in jobs])
        secs = time.time() - t0
    from kitsune.eval.judge import combine

    pairs: dict[str, dict] = {}
    for r in res:
        pairs.setdefault(r["pair_id"], {})[r["order"]] = r["verdict"]
    val = {p: combine(v.get("xy"), v.get("yx")).result for p, v in pairs.items() if p.startswith("val__")}
    bake = {p: combine(v.get("xy"), v.get("yx")) for p, v in pairs.items() if p.startswith("bake__")}
    out = {
        "judge": MODEL_IDS[judge][0],
        "n_calls": len(res),
        "invalid_verdicts": sum(r["verdict"] is None for r in res),
        "known_answer_correct": sum(v == "x" for v in val.values()),
        "known_answer_total": len(val),
        "known_answer": val,
        "bakeoff_qwen_vs_gemma": {
            "qwen_wins": sum(o.result == "x" for o in bake.values()),
            "gemma_wins": sum(o.result == "y" for o in bake.values()),
            "ties_or_inconsistent": sum(o.result == "tie" for o in bake.values()),
            "position_consistent": sum(o.consistent for o in bake.values()),
        },
        "wall_seconds_incl_load": round(secs, 1),
        "mean_output_tokens": sum(r["n_tokens"] for r in res) / max(len(res), 1),
        "raw_examples": [r["raw"][-800:] for r in res[:3]],
    }
    _save_json("reports/judge_smoke.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "raw_examples"}, ensure_ascii=False, indent=1))
