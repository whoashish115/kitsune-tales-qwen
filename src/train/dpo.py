"""Preference stage (D-010 #6): DPO on pairs of the SFT model's own samples.

Pairs come from two sources, and both are recorded per pair:
1. **Rule pairs**: one sample fails a hard rule (repetition, Chinese contamination, length,
   artifacts) and the other passes. The passing sample is chosen.
2. **Judge pairs**: both pass the rules; the *teacher* (Qwen3.6-35B-A3B) judges them in both
   orders and the pair is kept only if both orders agree.

The eval judge (llm-jp-4) is deliberately a different model from the DPO labeler, so DPO cannot
simply learn the eval judge's preferences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kitsune.data.filters import run_rule_filters
from kitsune.eval.judge import combine

HARD_RULES = {"nonempty", "no_artifacts", "japanese_purity", "length", "repetition", "safety_rule"}
HARD_RULES_EN = {
    "nonempty",
    "no_artifacts",
    "english_purity",
    "meta_leak",
    "self_correction",
    "length",
    "repetition",
    "safety_rule",
}


def rule_ok(
    text: str, fmt: str, genres: list[str], title: str, passage: str | None, lang: str = "ja"
) -> bool:
    if lang == "en":
        from kitsune.en import run_rule_filters_en

        outs = run_rule_filters_en(text, fmt, genres, title, passage)
        return all(o.passed for o in outs if o.name in HARD_RULES_EN)
    outs = run_rule_filters(text, fmt, genres, title, passage)
    return all(o.passed for o in outs if o.name in HARD_RULES)


DISALLOWED = ("sexual", "real_person", "existing_ip", "hate")


def safety_pairs(samples: list[dict], lang: str = "ja", per_prompt: int = 3, seed: int = 0) -> list[dict]:
    """Refusal-preference pairs for the *training* disallowed prompts (D-029).

    Quality-only DPO lowered the Japanese refusal rate from 75 % to 49 %: no pair ever showed that a disallowed request
    should be refused. Here chosen = the canonical refusal and rejected = a story the SFT model wrote for another
    (ordinary) training prompt, i.e. what "complying" looks like. The eval policy suite uses disjoint names/titles.
    """
    import random

    from kitsune.prompts import gemma_prompt

    if lang == "en":
        from kitsune import en

        prompts = [p for p in en.train_policy_prompts_en() if p["kind"] in DISALLOWED]
        users = [en.policy_user_prompt_en(p) for p in prompts]
        refusals = [en.refusal_text_en(p["kind"]) for p in prompts]
        system = en.SYSTEM_PROMPT_EN
    else:
        from dataclasses import asdict

        from kitsune.data.policy import refusal_text, train_policy_prompts
        from kitsune.prompts import SYSTEM_PROMPT

        pp = [p for p in train_policy_prompts() if p.kind in DISALLOWED]
        prompts = [asdict(p) for p in pp]
        users = [p.user_prompt() for p in pp]
        refusals = [refusal_text(p.kind) for p in pp]
        system = SYSTEM_PROMPT
    rng = random.Random(seed)
    stories = sorted(
        {
            s["a"]
            for s in samples
            if rule_ok(s["a"], s["format"], s["genres"], s["title"], s.get("passage"), lang)
        }
    )
    out = []
    for p, user, refusal in zip(prompts, users, refusals, strict=True):
        for k in range(per_prompt):
            out.append(
                {
                    "id": f"{p['id']}:safety:{k}",
                    "prompt": gemma_prompt(user, system),
                    "chosen": refusal,
                    "rejected": rng.choice(stories),
                    "source": "safety",
                }
            )
    return out


def build_pairs(
    samples: list[dict], verdicts: dict[str, dict[str, str | None]], lang: str = "ja", safety: bool = False
) -> tuple[list[dict], dict[str, int]]:
    """Build DPO pairs.

    Args:
        samples: rows with keys id, prompt_text (rendered chat prompt), user_prompt, format, genres,
            title, passage, a, b (the two sampled responses).
        verdicts: {sample id: {"xy": verdict with A=a, "yx": verdict with A=b}} from the teacher.
    Returns:
        (pairs with prompt/chosen/rejected/source, counts per outcome)
    """
    pairs, counts = [], {"rule": 0, "judge": 0, "both_fail": 0, "inconsistent_or_tie": 0, "missing": 0}
    for s in samples:
        ok_a = rule_ok(s["a"], s["format"], s["genres"], s["title"], s.get("passage"), lang)
        ok_b = rule_ok(s["b"], s["format"], s["genres"], s["title"], s.get("passage"), lang)
        if ok_a != ok_b:
            chosen, rejected = (s["a"], s["b"]) if ok_a else (s["b"], s["a"])
            pairs.append(
                {
                    "id": s["id"],
                    "prompt": s["prompt_text"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "rule",
                }
            )
            counts["rule"] += 1
            continue
        if not ok_a:
            counts["both_fail"] += 1
            continue
        v = verdicts.get(s["id"])
        if not v:
            counts["missing"] += 1
            continue
        o = combine(v.get("xy"), v.get("yx"))  # x = a
        if o.result == "x":
            pairs.append(
                {
                    "id": s["id"],
                    "prompt": s["prompt_text"],
                    "chosen": s["a"],
                    "rejected": s["b"],
                    "source": "judge",
                }
            )
            counts["judge"] += 1
        elif o.result == "y":
            pairs.append(
                {
                    "id": s["id"],
                    "prompt": s["prompt_text"],
                    "chosen": s["b"],
                    "rejected": s["a"],
                    "source": "judge",
                }
            )
            counts["judge"] += 1
        else:
            counts["inconsistent_or_tie"] += 1
    if safety:
        sp = safety_pairs(samples, lang)
        pairs += sp
        counts["safety"] = len(sp)
    return pairs, counts


def train_dpo(
    cfg: dict[str, Any],
    pairs_path: str,
    sft_adapter: str,
    out_dir: str,
    run_name: str,
    gpu: str,
    cpu: float,
    mem_gib: float,
    prior_spend_usd: float,
    git_commit: str,
    commit_fn: Any = None,
) -> dict[str, Any]:
    """Continue training the SFT adapter with DPO; the reference is a frozen copy of the SFT adapter."""
    import hashlib
    import os
    import time

    import torch
    import wandb
    from datasets import Dataset
    from kitsune import cost, naming, versions
    from kitsune.train.sft import _training_args
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    t0 = time.time()
    rate = cost.hourly_rate(gpu, cpu, mem_gib)
    tok = AutoTokenizer.from_pretrained(sft_adapter)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg["base_revision"], dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    rows = [json.loads(x) for x in Path(pairs_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs_sha = hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest()
    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )
    split = ds.train_test_split(test_size=min(200, max(20, len(ds) // 20)), seed=cfg.get("seed", 42))
    out = Path(out_dir) / run_name
    args = _training_args(
        DPOConfig,
        {
            "output_dir": str(out),
            "run_name": run_name,
            "report_to": ["wandb"],
            "bf16": True,
            "seed": cfg.get("seed", 42),
            "beta": cfg.get("beta", 0.1),
            "loss_type": cfg.get("loss_type", ["sigmoid"]),
            "learning_rate": cfg.get("learning_rate", 2e-5),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "num_train_epochs": cfg.get("epochs", 1),
            # Gemma's 262k vocabulary makes the fp32 logits of chosen + rejected large (~3.3 GB per 2k-token pair),
            # so the micro-batch stays small and gradient accumulation carries the effective batch.
            "per_device_train_batch_size": cfg.get("per_device_batch_size", 1),
            "per_device_eval_batch_size": cfg.get("per_device_batch_size", 1),
            "gradient_accumulation_steps": cfg.get("grad_accum", 4),
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_length": cfg.get("max_length", 2048),
            "logging_steps": 5,
            "eval_strategy": "steps",
            "eval_steps": cfg.get("eval_steps", 50),
            "save_strategy": "steps",
            "save_steps": cfg.get("save_steps", 50),
            "save_total_limit": 2,
        },
    )
    wandb.init(
        project=versions.WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY"),
        group="dpo",
        name=naming.run_name("dpo", cfg),
        tags=naming.run_tags("dpo", cfg),
        # One W&B run per (run name, pairs, config), as for SFT.
        id=f"{run_name}-{hashlib.sha256((pairs_sha + json.dumps(cfg, sort_keys=True, default=str)).encode()).hexdigest()[:8]}",
        resume="allow",
        config={
            "dpo_cfg": cfg,
            "git_commit": git_commit,
            "n_pairs": len(rows),
            "sft_adapter": sft_adapter,
            "rate_usd_per_h": rate,
            "prior_spend": prior_spend_usd,
        },
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"], processing_class=tok
    )
    # Resume only from checkpoints made with the same pairs and config (same guard as SFT, D-023).
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = json.dumps({"pairs": pairs_sha, "cfg": cfg}, sort_keys=True, default=str)
    fp_file = out / "run_fingerprint.json"
    has_ckpt = any(p.name.startswith("checkpoint-") for p in out.iterdir())
    if has_ckpt and (not fp_file.exists() or fp_file.read_text(encoding="utf-8") != fingerprint):
        raise RuntimeError(
            f"{out} has checkpoints from different pairs/config; delete them before re-running"
        )
    fp_file.write_text(fingerprint, encoding="utf-8")
    res = trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    ev = trainer.evaluate()
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir, selected_adapters=["default"])
    tok.save_pretrained(adapter_dir)
    hours = (time.time() - t0) / 3600
    summary = {
        "run_name": run_name,
        "n_pairs": len(rows),
        "train_loss": res.metrics.get("train_loss"),
        **{k: v for k, v in ev.items() if isinstance(v, int | float)},
        "job_hours": hours,
        "job_usd": hours * rate,
    }
    wandb.log({"cost/cumulative_usd": prior_spend_usd + hours * rate})
    wandb.summary.update(summary)
    art = wandb.Artifact(f"adapter-{run_name}", type="model", metadata=summary)
    art.add_dir(str(adapter_dir))
    wandb.log_artifact(art)
    wandb.finish()
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if commit_fn:
        commit_fn()
    return summary
