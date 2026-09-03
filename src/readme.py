"""Inject generated tables into README.md / REPORT.md between marker comments.

    <!-- BEGIN:results -->     reports/results_table.md (Japanese)
    <!-- BEGIN:results_en -->  reports/results_table_en.md (English)
    <!-- BEGIN:data -->        dataset statistics from reports/data*/stats.json
    <!-- BEGIN:setup -->       training runs and ablations from reports/train/*.json + configs/
    <!-- BEGIN:budget -->      Modal's billed totals (the ledger gives the per-job breakdown)

Only generated content goes between markers, so every number there traces to ``reports/`` or Modal billing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from kitsune import cost

TRAIN = Path("reports/train")


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(.*?)(<!-- END:{name} -->)", re.S)
    if not pat.search(text):
        return text
    return pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(3), text)


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def budget_block() -> str:
    lines = []
    total = 0.0
    for acct, cap in cost.ACCOUNT_CAPS_USD.items():
        if acct not in ("kitsune30", "kitsune12"):
            continue
        billed = cost.billed_usd(acct)
        if billed is None:
            billed = cost.spent(cost.read_ledger(), acct)
            src = "ledger (billing unavailable)"
        else:
            src = "Modal billing"
        total += billed
        lines.append(
            f"- `{acct}`: **${billed:.2f}** of ${cap:.2f} credit ({src}; hard stop ${cost.ACCOUNT_KILL_USD[acct]:.2f})"
        )
    lines.append(f"- **Total: ${total:.2f}**")
    return (
        "Compute spend as billed by Modal (`modal billing report`); the per-job breakdown with estimates and measured\n"
        "wall-clock is in `reports/cost_ledger.jsonl` and `docs/BUDGET.md`.\n\n" + "\n".join(lines)
    )


def data_block() -> str:
    rows = []
    for lang, path in (
        ("Japanese", Path("reports/data/stats.json")),
        ("English", Path("reports/data_en/stats.json")),
    ):
        d = _load(path)
        if not d:
            continue
        kept = d["n_kept_synthetic"]
        gen = d["n_generations"]
        by = d.get("by_generator_kept", {})
        cross = sum(v for k, v in d.get("label_sources", {}).items() if k.startswith("cross"))
        rows.append(
            f"| {lang} | {gen:,} | {kept:,} ({kept / gen:.0%}) | {by.get('gen1', 0):,} / {by.get('gen2', 0):,} | "
            f"{cross / kept:.0%} | {d['n_train']:,} / {d['n_val']:,} | {d.get('n_refusal_templates', 0)} |"
        )
    head = (
        "| Dataset | Generations | Kept after filters + dedup | Kept gen1 / gen2 | Cross-model labels | Train / val | Policy templates |\n"
        "|---|---|---|---|---|---|---|"
    )
    return (
        head
        + "\n"
        + "\n".join(rows)
        + "\n\nFilter funnels, length histograms and genre × format grids: `reports/data*/`."
    )


_CFG = {
    "sft-pilot": "configs/train_pilot.yaml",
    "sft-main": "configs/train_main.yaml",
    "sft-en-main": "configs/train_en_main.yaml",
    "abl-r16": "configs/ablation_r16.yaml",
    "abl-r64": "configs/ablation_r64.yaml",
    "abl-data10": "configs/ablation_data10.yaml",
    "abl-data30": "configs/ablation_data30.yaml",
}


def setup_block() -> str:
    out = [
        "All runs: Gemma 4 E4B (`google/gemma-4-E4B-it`, pinned revision), bf16 LoRA on every language-model linear "
        "layer, loss on the assistant turn only, no packing, cosine schedule, seed 42, 1 epoch, batch 16, H100. "
        "Validation loss is on each language's own validation split.",
        "",
        "| Run | Language | Stage | Train examples | LoRA r | Trainable params | Val loss (ppl) | GPU minutes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run, lang in (("sft-main", "ja"), ("sft-en-main", "en")):
        d = _load(TRAIN / f"{run}.json")
        if d:
            cfg = yaml.safe_load(Path(_CFG[run]).read_text(encoding="utf-8"))
            out.append(
                f"| `{run}` | {lang} | SFT | {d['data']['train']['n']:,} | {cfg['lora_r']} | {d['params_trainable'] / 1e6:.1f}M | "
                f"{d['eval_loss']:.4f} ({d['eval_ppl']:.2f}) | {d['job_hours'] * 60:.0f} |"
            )
    for run, lang in (("dpo-main-v2", "ja"), ("dpo-en-main", "en")):
        d = _load(TRAIN / f"{run}.json")
        if d:
            out.append(
                f"| `{run}` | {lang} | DPO (β 0.1) | {d['n_pairs']:,} pairs | 32 | same adapter | "
                f"DPO loss {d['eval_loss']:.4f} | {d['job_hours'] * 60:.0f} |"
            )
    for p in (Path("reports/dpo_pairs_v2.json"), Path("reports/dpo_pairs_en.json")):
        c = _load(p)
        if c:
            lang = "Japanese" if "v2" in p.name else "English"
            safety = f" + {c['safety']} safety (refusal-preference) pairs" if c.get("safety") else ""
            out.append(
                f"\n{lang} DPO pairs: {c['rule']:,} rule-decided + {c['judge']:,} teacher-judged (both orders agree){safety}; "
                f"{c['inconsistent_or_tie']:,} prompts dropped as ties or order-inconsistent, {c['both_fail']} with both samples failing rules."
            )
    out.append(
        "\nEnglish DPO was paused at step 50 of 95 during a budget reconciliation and resumed from that checkpoint (same "
        "pairs and config, D-031); its GPU minutes cover the resumed run."
    )
    abl = [
        ("abl-data10", "10 %"),
        ("abl-data30", "30 %"),
        ("sft-main", "100 %"),
        ("abl-r16", "25 %"),
        ("abl-r64", "25 %"),
    ]
    rows = []
    for run, frac in abl:
        d = _load(TRAIN / f"{run}.json")
        if not d:
            continue
        cfg = yaml.safe_load(Path(_CFG[run]).read_text(encoding="utf-8"))
        rows.append(
            f"| `{run}` | {frac} | {d['data']['train']['n']:,} | {cfg['lora_r']} | {d['params_trainable'] / 1e6:.1f}M | "
            f"{d['eval_loss']:.4f} | {d['eval_ppl']:.2f} |"
        )
    if rows:
        out += [
            "",
            "**Ablations (Japanese, same validation split).** Data scaling at r = 32, and LoRA rank at a fixed 25 % subset:",
            "",
            "| Run | Data | Train examples | LoRA r | Trainable params | Val loss | Val ppl |",
            "|---|---|---|---|---|---|---|",
            *rows,
        ]
    return "\n".join(out)


def main() -> None:
    tables = {"results": Path("reports/results_table.md"), "results_en": Path("reports/results_table_en.md")}
    blocks = {"data": data_block(), "setup": setup_block(), "budget": budget_block()}
    for doc in (Path("README.md"), Path("REPORT.md")):
        if not doc.exists():
            continue
        t = doc.read_text(encoding="utf-8")
        for block, path in tables.items():
            if path.exists():
                body = path.read_text(encoding="utf-8")
                if doc.name == "README.md":  # the README nests these tables under a level-3 heading
                    body = body.replace("\n## ", "\n#### ")
                t = replace_block(t, block, body)
        for block, body in blocks.items():
            t = replace_block(t, block, body)
        doc.write_text(t, encoding="utf-8", newline="\n")
        print(f"updated {doc}")


if __name__ == "__main__":
    main()
