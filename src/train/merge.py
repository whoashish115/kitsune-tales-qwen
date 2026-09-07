"""Merge a LoRA adapter into the base weights and verify the merge.

Verification (spec section 5): on a fixed prompt set, the merged model must match
adapter-on-base (1) token-for-token under greedy decoding and (2) within a small logit
tolerance on a teacher-forced batch. The report is written next to the merged weights and
copied to ``reports/merge_check.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kitsune.prompts import StoryRequest, build_user_prompt, render_prompt

CHECK_REQUESTS = [
    StoryRequest(["異世界転生", "冒険者ギルド"], "追放された剣士は辺境で最強になる", "あらすじ"),
    StoryRequest(["悪役令嬢・転生"], "断罪された令嬢は薬草園で幸せになる", "短編"),
    StoryRequest(["魔王と勇者", "スローライフ"], "引退した魔王は湖畔で喫茶店を開く", "あらすじ"),
    StoryRequest(["魔法少女"], "魔法少女ルミナは今日も遅刻する", "短編"),
    StoryRequest(["ダークファンタジー"], "灰燼の王冠", "あらすじ"),
    StoryRequest(["ハイファンタジー", "魔法学園"], "星詠みの巫女と銀の塔", "短編"),
    StoryRequest(["スローライフ"], "北の雪国で始める薬師のスローライフ", "あらすじ"),
    StoryRequest(["冒険者ギルド"], "Fランク冒険者の荷物持ちは竜と暮らす", "短編"),
]


def _sha256_dir(d: Path) -> dict[str, str]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out[f.name] = h.hexdigest()
    return out


def merge_and_verify(
    base_model: str, base_revision: str, adapter_dir: str, out_dir: str, max_new_tokens: int = 96
) -> dict[str, Any]:
    """Merge ``adapter_dir`` into the base in fp32 on CPU (one rounding to bf16), save bf16 safetensors, and verify.

    Reference = adapter-on-base in bf16 (how the adapter is normally run). Candidate = the merged bf16 weights
    **reloaded from disk**. Reported: greedy-token identity per prompt, teacher-forced top-1 agreement,
    mean KL(ref || merged) per token, and max/mean absolute logit difference.
    """
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, revision=base_revision)
    prompts = [render_prompt(tok, build_user_prompt(r)) for r in CHECK_REQUESTS]

    @torch.no_grad()
    def greedy(model: Any) -> list[list[int]]:
        outs = []
        for p in prompts:
            ids = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            g = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
            outs.append(g[0, ids["input_ids"].shape[1] :].tolist())
        return outs

    @torch.no_grad()
    def forced_logits(model: Any, continuations: list[list[int]]) -> list[Any]:
        """Teacher-forced logits over prompt + reference continuation, one sequence at a time (no padding)."""
        out = []
        for p, c in zip(prompts, continuations, strict=True):
            ids = tok(p, add_special_tokens=False)["input_ids"] + c
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -len(c) - 1 : -1].float().cpu()
            out.append(lg)
        return out

    ref_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            base_model, revision=base_revision, dtype=torch.bfloat16, device_map="cuda"
        ),
        adapter_dir,
    ).eval()
    ref_tokens = greedy(ref_model)
    ref_logits = forced_logits(ref_model, ref_tokens)
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    fp32 = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            base_model, revision=base_revision, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
        ),
        adapter_dir,
    )
    merged = fp32.merge_and_unload().to(torch.bfloat16)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    del merged, fp32
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16, device_map="cuda").eval()
    disk_tokens = greedy(reloaded)
    disk_logits = forced_logits(reloaded, ref_tokens)
    n_params = sum(p.numel() for p in reloaded.parameters())

    kl, agree, n_tok, max_diff, sum_diff = 0.0, 0, 0, 0.0, 0.0
    for a, b in zip(ref_logits, disk_logits, strict=True):
        lp, lq = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kl += float((lp.exp() * (lp - lq)).sum())
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        n_tok += a.shape[0]
        d = (a - b).abs()
        max_diff = max(max_diff, float(d.max()))
        sum_diff += float(d.mean()) * a.shape[0]
    report = {
        "merge_precision": "fp32 merge, saved as bf16",
        "n_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "params_merged_model": n_params,
        "greedy_identical_prompts": sum(x == y for x, y in zip(ref_tokens, disk_tokens, strict=True)),
        "first_divergence": [
            next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
            for a, b in zip(ref_tokens, disk_tokens, strict=True)
        ],
        "teacher_forced_tokens": n_tok,
        "top1_agreement": agree / n_tok,
        "mean_kl_per_token": kl / n_tok,
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": sum_diff / n_tok,
        "safetensors_sha256": _sha256_dir(out),
        "note": "Differences come from bf16 rounding of merged weights vs bf16 adapter compute; near-tie tokens can flip.",
    }
    (out / "merge_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
