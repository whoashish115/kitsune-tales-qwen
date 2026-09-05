"""GPU smoke tests for the Qwen3.5 hybrid architecture (Phase 2 gate; runs on one cheap GPU).

Each check is independent and returns a dict, so one failure does not hide the others:
1. ``load``        – AutoModelForCausalLM loads the text model; parameter counts; LoRA target leaves;
                     whether flash-linear-attention kernels are active.
2. ``packing``     – are packed (padding-free, position_ids reset) sequences numerically equivalent to
                     separate ones? Gated DeltaNet layers carry recurrent state, so this decides D-012.
3. ``train``       – a few LoRA steps on fixture examples: finite, decreasing loss; tokens/s; memory.
4. ``merge``       – merged vs adapter-on-base equivalence (the same check used for the release).
5. ``vllm``        – vLLM loads the base text-only, generates Japanese, and (if supported) serves a LoRA.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from kitsune.fixtures import GENRES, STORY, SYNOPSIS, TITLE
from kitsune.prompts import StoryRequest, build_user_prompt, render_prompt, tokenize_example


def _free() -> None:
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _guard(name: str, fn: Any, *a: Any, **k: Any) -> dict[str, Any]:
    t0 = time.time()
    try:
        out = fn(*a, **k)
        _free()
        return {"ok": True, "seconds": round(time.time() - t0, 1), **out}
    except Exception as e:  # report and continue with the other checks
        _free()
        return {
            "ok": False,
            "seconds": round(time.time() - t0, 1),
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-3000:],
        }


def _examples(tok: Any, n: int = 16) -> list[dict]:
    u1 = build_user_prompt(StoryRequest(GENRES, TITLE, "短編"))
    u2 = build_user_prompt(StoryRequest(GENRES, TITLE, "あらすじ"))
    rows = [tokenize_example(tok, u1, STORY, 4096), tokenize_example(tok, u2, SYNOPSIS, 4096)]
    return (rows * n)[:n]


def check_load(model_id: str, revision: str) -> dict[str, Any]:
    import torch
    from kitsune.train.sft import lora_target_regex
    from transformers import AutoModelForCausalLM
    from transformers.utils import import_utils

    m = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    regex, leaves = lora_target_regex(m)
    total = sum(p.numel() for p in m.parameters())
    fla = getattr(import_utils, "is_flash_linear_attention_available", lambda: None)()
    names = [n for n, _ in m.named_modules()][:5]
    return {
        "class": type(m).__name__,
        "params_total": total,
        "lora_leaves": leaves,
        "lora_regex": regex,
        "fla_available": fla,
        "first_modules": names,
    }


def check_packing(model_id: str, revision: str) -> dict[str, Any]:
    """Compare logits of two sequences run separately vs packed into one row with reset position_ids."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    m = (
        AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    a = tok(STORY[:400], add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    b = tok(SYNOPSIS, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        la = m(input_ids=a).logits.float()
        lb = m(input_ids=b).logits.float()
        packed = torch.cat([a, b], dim=1)
        pos = torch.cat([torch.arange(a.shape[1]), torch.arange(b.shape[1])]).unsqueeze(0).cuda()
        lp = m(input_ids=packed, position_ids=pos).logits.float()
    da = (lp[:, : a.shape[1]] - la).abs().max().item()
    db = (lp[:, a.shape[1] :] - lb).abs().max().item()
    # bf16 noise between two separate forward passes of the same input, for scale:
    with torch.no_grad():
        lb2 = m(input_ids=b).logits.float()
    noise = (lb2 - lb).abs().max().item()
    return {
        "max_diff_first_seq": da,
        "max_diff_second_seq": db,
        "bf16_repeat_noise": noise,
        "packing_equivalent": db < max(0.5, 5 * noise),
        "note": "second_seq diff >> noise means state leaks across packed boundaries (packing unsafe)",
    }


def check_train(model_id: str, revision: str, out_dir: str, steps: int = 8) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from kitsune.train.sft import _training_args, lora_target_regex
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    regex, _ = lora_target_regex(m)
    rows = _examples(tok)
    args = _training_args(
        SFTConfig,
        {
            "output_dir": out_dir,
            "report_to": [],
            "bf16": True,
            "max_steps": steps,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-4,
            "logging_steps": 1,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "save_strategy": "no",
            "max_length": 2048,
            "seed": 0,
        },
    )
    tr = SFTTrainer(
        model=m, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=regex, task_type="CAUSAL_LM"),
    )  # fmt: skip
    trainable = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr.train()
    dt = time.time() - t0
    losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
    ntok = sum(len(r["input_ids"]) for r in rows[: steps * 2])
    tr.model.save_pretrained(f"{out_dir}/adapter")
    tok.save_pretrained(f"{out_dir}/adapter")
    return {
        "losses": losses,
        "loss_finite_and_decreasing": all(x == x for x in losses) and losses[-1] < losses[0],
        "trainable_params_r16": trainable,
        "tokens_per_s": ntok / dt,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def check_vllm(model_id: str, revision: str, adapter_dir: str | None) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    prompt = render_prompt(tok, build_user_prompt(StoryRequest(GENRES, TITLE, "あらすじ")))
    base_kwargs: dict[str, Any] = {
        "model": model_id,
        "revision": revision,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
    }
    out: dict[str, Any] = {}
    llm = None
    for extra in ({"language_model_only": True, "enable_lora": True, "max_lora_rank": 64},
                  {"limit_mm_per_prompt": {"image": 0, "video": 0}, "enable_lora": True, "max_lora_rank": 64},
                  {"limit_mm_per_prompt": {"image": 0, "video": 0}}):  # fmt: skip
        try:
            llm = LLM(**base_kwargs, **extra)
            out["llm_kwargs"] = extra
            break
        except Exception as e:
            out.setdefault("load_errors", []).append(f"{extra}: {type(e).__name__}: {str(e)[:300]}")
    if llm is None:
        raise RuntimeError("vLLM could not load the model: " + " | ".join(out.get("load_errors", [])))
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400, seed=0)
    t0 = time.time()
    res = llm.generate([prompt] * 8, sp)
    dt = time.time() - t0
    out["base_sample"] = res[0].outputs[0].text[:600]
    out["base_tokens_per_s_batch8"] = sum(len(r.outputs[0].token_ids) for r in res) / dt
    if adapter_dir and out["llm_kwargs"].get("enable_lora"):
        try:
            from vllm.lora.request import LoRARequest

            r2 = llm.generate([prompt], sp, lora_request=LoRARequest("smoke", 1, adapter_dir))
            out["lora_sample"] = r2[0].outputs[0].text[:300]
            out["lora_supported"] = True
        except Exception as e:
            out["lora_supported"] = False
            out["lora_error"] = f"{type(e).__name__}: {str(e)[:500]}"
    return out


def run_all(model_id: str, revision: str, work: str) -> dict[str, Any]:
    from kitsune.train.merge import merge_and_verify

    res: dict[str, Any] = {"load": _guard("load", check_load, model_id, revision)}
    res["packing"] = _guard("packing", check_packing, model_id, revision)
    res["train"] = _guard("train", check_train, model_id, revision, f"{work}/smoke-train")
    adapter = f"{work}/smoke-train/adapter"
    if res["train"]["ok"]:
        res["merge"] = _guard(
            "merge", merge_and_verify, model_id, revision, adapter, f"{work}/smoke-merged", 48
        )
    return res
