"""Supervised fine-tuning with bf16 LoRA (TRL SFTTrainer), resumable, with cost/throughput logging.
Runs inside the Modal GPU image (see ``modal_app.train``). Config comes from ``configs/train_*.yaml``.
Loss is on the assistant turn only: examples are pre-tokenized by ``kitsune.prompts.tokenize_example``
with explicit ``labels`` (-100 on prompt tokens), which TRL uses as-is.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import fields
from pathlib import Path
from typing import Any
from kitsune import cost, naming, versions
from kitsune.prompts import tokenize_example
from kitsune.schema import file_sha256, read_jsonl
def system_prompt_for(cfg: dict[str, Any]) -> str | None:
    """System prompt for the config's language: None keeps the Japanese default, "en" uses D-024's."""
    raise NotImplementedError

def train(
    cfg: dict[str, Any],
    data_dir: str,
    out_root: str,
    run_name: str,
    gpu: str,
    cpu_cores: float,
    mem_gib: float,
    prior_spend_usd: float,
    git_commit: str,
    commit_fn: Any = None,
) -> dict[str, Any]:
    """Train one LoRA adapter. Resumes from the latest checkpoint in ``out_root/run_name`` if present."""
    print("[debug] train", flush=True)
    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    t_start = time.time()
    rate = cost.hourly_rate(gpu, cpu_cores, mem_gib)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    model_id, revision = cfg["base_model"], cfg["base_revision"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    train_path, val_path = Path(data_dir) / "train.jsonl", Path(data_dir) / "val.jsonl"
    system = system_prompt_for(cfg)
    train_rows, train_stats = load_split(
        train_path, tok, cfg["max_length"], cfg.get("data_fraction", 1.0), seed, system
    )
    val_rows, val_stats = load_split(
        val_path, tok, cfg["max_length"], cfg.get("val_fraction", 1.0), seed, system
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    target_regex, leaves = lora_target_regex(model)
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_regex,
        bias="none",
        task_type="CAUSAL_LM",
    )

    out_dir = Path(out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    targs = _training_args(
        SFTConfig,
        {
            "output_dir": str(out_dir),
            "run_name": run_name,
            "report_to": ["wandb"],
            "bf16": True,
            "seed": seed,
            "data_seed": seed,
            "max_length": cfg["max_length"],
            "packing": cfg.get("packing", False),
            "padding_free": cfg.get("padding_free", False),
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "per_device_train_batch_size": cfg["per_device_batch_size"],
            "per_device_eval_batch_size": cfg["per_device_batch_size"],
            "gradient_accumulation_steps": cfg["grad_accum"],
            "learning_rate": cfg["learning_rate"],
            "lr_scheduler_type": "cosine",
            "warmup_ratio": cfg.get("warmup_ratio", 0.05),
            "weight_decay": cfg.get("weight_decay", 0.0),
            "max_grad_norm": cfg.get("max_grad_norm", 1.0),
            "num_train_epochs": cfg.get("epochs", 1),
            "max_steps": cfg.get("max_steps", -1),
            "logging_steps": cfg.get("logging_steps", 10),
            "eval_strategy": "steps",
            "eval_steps": cfg.get("eval_steps", 100),
            "save_strategy": "steps",
            "save_steps": cfg.get("save_steps", 100),
            "save_total_limit": 2,
            "load_best_model_at_end": False,
            "group_by_length": cfg.get("group_by_length", True),
            "dataloader_num_workers": 2,
            "remove_unused_columns": True,
            "completion_only_loss": None,
        },
    )

    wandb.init(
        project=versions.WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY"),
        group=cfg.get("wandb_group", "pilot"),
        name=naming.run_name("sft", cfg),
        tags=naming.run_tags("sft", cfg),
        # One W&B run per (run name, data, config): re-runs on new data never append to an old run's history.
        id=cfg.get("wandb_run_id")
        or f"{run_name}-{hashlib.sha256((file_sha256(train_path) + json.dumps(cfg, sort_keys=True, default=str)).encode()).hexdigest()[:8]}",
        resume="allow",
        config={
            "train_cfg": cfg,
            "run_dir": run_name,
            "model_name": versions.model_name(cfg.get("lang", "ja")),
            "git_commit": git_commit,
            "base_model": model_id,
            "base_revision": revision,
            "dataset_sha256": {"train": file_sha256(train_path), "val": file_sha256(val_path)},
            "data_stats": {"train": train_stats, "val": val_stats},
            "lora_target_leaves": leaves,
            "hardware": {
                "gpu": gpu,
                "cpu_cores": cpu_cores,
                "mem_gib": mem_gib,
                "torch_gpu": torch.cuda.get_device_name(0),
            },
            "rate_usd_per_h": rate,
        },
    )

    class Telemetry(TrainerCallback):
        """Cost, throughput and memory; commits the Volume after every checkpoint save."""

        def __init__(self) -> None:
            self.t_last = time.time()
            self.tokens_seen = 0

        def on_log(self, args, state, control, logs=None, **kw):  # type: ignore[no-untyped-def]
            if logs is None:
                return
            now = time.time()
            elapsed_h = (now - t_start) / 3600
            extra = {
                "cost/job_usd": elapsed_h * rate,
                "cost/cumulative_usd": prior_spend_usd + elapsed_h * rate,
                "sys/gpu_mem_max_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            ntok = logs.get("num_tokens")
            if ntok is not None:
                extra["perf/tokens_per_s"] = (ntok - self.tokens_seen) / max(now - self.t_last, 1e-6)
                self.tokens_seen = ntok
            self.t_last = now
            wandb.log(extra, step=state.global_step)

        def on_save(self, args, state, control, **kw):  # type: ignore[no-untyped-def]
            if commit_fn is not None:
                commit_fn()

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows),
        processing_class=tok,
        peft_config=lora,
        callbacks=[Telemetry()],
    )
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    wandb.config.update({"params/trainable": n_trainable, "params/total": n_total}, allow_val_change=True)

    # Resume only from checkpoints made on the *same* data and config (an orphaned run once left
    fingerprint = json.dumps(
        {"train": file_sha256(train_path), "val": file_sha256(val_path), "cfg": cfg},
        sort_keys=True,
        default=str,
    )
    fp_file = out_dir / "run_fingerprint.json"
    has_ckpt = any(p.name.startswith("checkpoint-") for p in out_dir.iterdir())
    if has_ckpt and (not fp_file.exists() or fp_file.read_text(encoding="utf-8") != fingerprint):
        raise RuntimeError(
            f"{out_dir} has checkpoints from a different dataset/config; delete them before re-running"
        )
    fp_file.write_text(fingerprint, encoding="utf-8")
    result = trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    final_eval = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    elapsed_h = (time.time() - t_start) / 3600
    summary = {
        "run_name": run_name,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "eval_loss": final_eval.get("eval_loss"),
        "eval_ppl": math.exp(final_eval["eval_loss"]) if final_eval.get("eval_loss") is not None else None,
        "params_trainable": n_trainable,
        "params_total": n_total,
        "data": {"train": train_stats, "val": val_stats},
        "job_hours": elapsed_h,
        "job_usd": elapsed_h * rate,
        "tokens_per_s_mean": train_stats["tokens"]
        * targs.num_train_epochs
        / max(result.metrics.get("train_runtime", 1), 1)
        if targs.max_steps in (-1, None)
        else None,
        "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    art = wandb.Artifact(f"adapter-{run_name}", type="model", metadata=summary)
    art.add_dir(str(adapter_dir))
    wandb.log_artifact(art)
    wandb.summary.update({k: v for k, v in summary.items() if isinstance(v, int | float)})
    wandb.finish()
    if commit_fn is not None:
        commit_fn()
    return summary
