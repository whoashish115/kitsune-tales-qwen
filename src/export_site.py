"""Export every number the project website shows into one JSON file (``reports/site/kitsune.json``).
The website lives in its own repository (kitsune-tales-qwen-site); ``--site <path>`` copies this file and the
figures into a local checkout of it, so the statistics are only ever produced here.

    python -m kitsune.export_site
The site renders only what this file contains, and this file is built only from ``reports/``, ``configs/`` and
Modal's billed totals, so every figure on the site traces to the same evaluation outputs as REPORT.md.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import yaml
from kitsune import cost, en, versions
from kitsune.eval.report import SYSTEM_NOTES
from kitsune.taxonomy import FORMATS, GENRE_EN, GENRES
R = Path("reports")
OUT = Path("reports/site/kitsune.json")
SITE_URL = "https://kitsune-tales-qwen.vercel.app"

def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def _ci(d: dict | None) -> dict | None:
    if not d:
        return None
    return {"mean": d["mean"], "low": d["low"], "high": d["high"], "n": d.get("n")}

def _systems(res: dict, order: list[str]) -> list[dict]:
    raise NotImplementedError

def _judge(res: dict) -> dict:
    j = res.get("judge", {})
    comps = []
    for k, v in j.items():
        if k.endswith(":validation"):
            continue
        a, b = k.split(":", 1)[1].split("_vs_")
        comps.append(
            {
                "x": a.removeprefix("excerpt-"),
                "y": b.removeprefix("excerpt-"),
                "length_matched": a.startswith("excerpt-"),
                "win": v["win_rate"],
                "tie": v["tie_rate"],
                "loss": v["loss_rate"],
                "net": _ci(v["net_preference"]),
                "pairs": v["n_pairs"],
                "consistency": v["position_consistency"],
            }
        )
    val = j.get("judge:validation")
    return {
        "comparisons": comps,
        "validation": {
            "accuracy": _ci(val["accuracy_all"]),
            "by_corruption": val["accuracy_by_corruption"],
            "n": val["n"],
        }
        if val
        else None,
    }

def _data(p: Path) -> dict | None:
    d = _load(p)
    if not d:
        return None
    drops = sorted(d.get("drop_reasons", {}).items(), key=lambda x: -x[1])[:8]
    return {
        "generations": d["n_generations"],
        "kept": d["n_kept_synthetic"],
        "train": d["n_train"],
        "val": d["n_val"],
        "policy_templates": d.get("n_refusal_templates"),
        "by_generator_total": {k: v for k, v in d.get("by_generator_total", {}).items() if k},
        "by_generator_kept": d.get("by_generator_kept", {}),
        "label_sources": d.get("label_sources", {}),
        "top_drops": [{"reason": k, "n": v} for k, v in drops],
        "dedup": d.get("dedup", {}),
    }

def _budget() -> dict:
    rows = [
        json.loads(x) for x in (R / "cost_ledger.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    names = {"2": "Setup & bake-off", "infra": "Weight downloads", "3": "Synthetic data", "4": "Pilot & ablations",
             "5": "Main SFT", "5b": "DPO", "6": "Evaluation", "7": "Merge & GGUF"}  # fmt: skip
    phase: dict[str, float] = {}
    for r in rows:
        if r["account"] in ("kitsune30", "kitsune12"):
            phase[names.get(r["phase"], r["phase"])] = phase.get(names.get(r["phase"], r["phase"]), 0) + (
                r.get("actual_usd") or 0
            )
    billed = {a: cost.billed_usd(a) for a in ("kitsune30", "kitsune12")}
    return {
        "accounts": [
            {
                "id": a,
                "credit": cost.ACCOUNT_CAPS_USD[a],
                "billed": billed[a],
                "stop": cost.ACCOUNT_KILL_USD[a],
            }
            for a in ("kitsune30", "kitsune12")
        ],
        "total_billed": sum(v for v in billed.values() if v),
        "phases": [{"phase": k, "usd": round(v, 2)} for k, v in phase.items()],
    }
