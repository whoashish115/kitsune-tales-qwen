"""Export every number the project website shows into one JSON file (``reports/site/kitsune.json``).

The website lives in its own repository (kitsune-tales-qwen-site); ``--site <path>`` copies this file and the
figures into a local checkout of it, so the statistics are only ever produced here.

    python -m kitsune.site_export

The site renders only what this file contains, and this file is built only from ``reports/``, ``configs/`` and
the billed cloud GPU totals, so every figure on the site traces to the same evaluation outputs as REPORT.md.
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
    out = []
    for s in order:
        if s not in res["systems"]:
            continue
        v = res["systems"][s]
        out.append(
            {
                "id": s,
                "note": SYSTEM_NOTES.get(s, s),
                "n_generations": v["n_generations"],
                "test": {k: _ci(x) for k, x in v["test"].items()},
                "policy": {k: (_ci(x) if isinstance(x, dict) else x) for k, x in v["policy"].items()},
                "diversity": v.get("diversity", {}),
            }
        )
    return out


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


def _train() -> dict:
    t = {p.stem: _load(p) for p in sorted((R / "train").glob("*.json"))}
    cfg = {
        "sft-main": "configs/train_main.yaml",
        "sft-en-main": "configs/train_en_main.yaml",
        "abl-r16": "configs/ablation_r16.yaml",
        "abl-r64": "configs/ablation_r64.yaml",
        "abl-data10": "configs/ablation_data10.yaml",
        "abl-data30": "configs/ablation_data30.yaml",
    }
    runs = {}
    for name, d in t.items():
        row: dict[str, Any] = {
            k: d.get(k)
            for k in (
                "eval_loss",
                "eval_ppl",
                "train_loss",
                "job_hours",
                "n_pairs",
                "params_trainable",
                "params_total",
            )
        }
        row["train_examples"] = (d.get("data") or {}).get("train", {}).get("n")
        row["reward_accuracy"] = d.get("eval_rewards/accuracies")
        row["reward_margin"] = d.get("eval_rewards/margins")
        if name in cfg:
            c = yaml.safe_load(Path(cfg[name]).read_text(encoding="utf-8"))
            row |= {
                "lora_r": c["lora_r"],
                "lora_alpha": c["lora_alpha"],
                "data_fraction": c.get("data_fraction", 1.0),
                "lr": c["learning_rate"],
                "batch": c["per_device_batch_size"] * c["grad_accum"],
                "epochs": c.get("epochs", 1),
            }
        runs[name] = row
    return {
        "runs": runs,
        "dpo_pairs": {"ja_v2": _load(R / "dpo_pairs_v2.json"), "en": _load(R / "dpo_pairs_en.json")},
        "merge": {
            k: _load(R / f"merge_check_{k}.json") and {
                m: _load(R / f"merge_check_{k}.json").get(m) for m in ("top1_agreement", "mean_kl_per_token", "max_abs_logit_diff")
            }
            for k in ("sft-main", "dpo-main-v2", "sft-en-main", "dpo-en-main")
        },
    }  # fmt: skip


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


def _gguf() -> dict:
    out = {}
    for lang, merged in (("jp", "sft-main"), ("en", "dpo-en-main")):
        g, s = _load(R / f"gguf_{merged}.json"), _load(R / f"gguf_smoke_{merged}.json")
        tps = None
        if s:
            m = re.search(r"Generation:\s*([\d.]+)\s*t/s", json.dumps(s))
            tps = float(m.group(1)) if m else None
        out[lang] = {
            "files": {k: {"gb": round(v["bytes"] / 1e9, 2), "sha256": v["sha256"]} for k, v in (g or {}).get("files", {}).items()},
            "cpu_tokens_per_s": tps,
            "llama_cpp_commit": (g or {}).get("llama_cpp_commit"),
        }  # fmt: skip
    return out


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


def _samples() -> dict:
    tr = _load(R / "translations_jp.json") or {}
    jp = _load(R / "samples_gallery_jp.json") or []
    enx = _load(R / "samples_gallery_en.json") or []

    def pick(rows: list[dict], n: int) -> list[dict]:
        seen, out = set(), []
        for r in rows:  # one per format first, in gallery order (a seeded random draw)
            if r["format"] not in seen:
                seen.add(r["format"])
                out.append(r)
        return (out + [r for r in rows if r not in out])[:n]

    return {
        "jp": [{**{k: r.get(k) for k in ("prompt_id", "genres", "title", "format", "passage", "text")}, "translation_en": tr.get(r["prompt_id"], "")} for r in pick(jp, 4)],
        "en": [{k: r.get(k) for k in ("prompt_id", "genres", "title", "format", "passage", "text")} for r in pick(enx, 4)],
    }  # fmt: skip


TRAIN_KEYS = (
    "loss",
    "grad_norm",
    "learning_rate",
    "mean_token_accuracy",
    "entropy",
    "rewards/accuracies",
    "rewards/margins",
    "rewards/chosen",
    "rewards/rejected",
    "logps/chosen",
    "logps/rejected",
)
EVAL_KEYS = (
    "eval_loss",
    "eval_mean_token_accuracy",
    "eval_entropy",
    "eval_rewards/accuracies",
    "eval_rewards/margins",
)


def _curves() -> dict:
    """Trainer logs of every run (reports/train_logs/*.json, copied from each run's trainer_state.json)."""
    out = {}
    for f in sorted((R / "train_logs").glob("*.json")):
        hist = json.loads(f.read_text(encoding="utf-8"))["log_history"]
        tr = [x for x in hist if "loss" in x and "eval_loss" not in x and "train_runtime" not in x]
        ev = [x for x in hist if "eval_loss" in x]

        def cols(rows: list[dict], keys: tuple[str, ...]) -> dict:
            got = {
                k: [round(float(x[k]), 6) for x in rows] for k in keys if rows and all(k in x for x in rows)
            }
            return {"step": [x["step"] for x in rows], "epoch": [round(x["epoch"], 4) for x in rows], **got}

        out[f.stem] = {"train": cols(tr, TRAIN_KEYS), "eval": cols(ev, EVAL_KEYS)}
    return out


GITHUB_REPO = "https://github.com/whoashish115/kitsune-tales-qwen"
HF_COLLECTION = "https://huggingface.co/collections/whoashish115/kitsune-tales-6abd4e61de4896bb86692bc1"


def _links() -> dict:
    hf = "https://huggingface.co"
    return {
        "site": SITE_URL,
        "site_repo": "https://github.com/whoashish115/kitsune-tales-qwen-site",
        "github": GITHUB_REPO,
        "wandb": f"https://wandb.ai/whoashish115-base/{versions.WANDB_PROJECT}",
        "collection": HF_COLLECTION,
        "space": f"{hf}/spaces/{versions.HF_SPACE_REPO}",
        "author": {"name": "Ashish Kumar", "github": "https://github.com/whoashish115", "hf": f"{hf}/whoashish115"},
        "models": {
            "jp": {"merged": f"{hf}/{versions.HF_MODEL_REPO}", "lora": f"{hf}/{versions.HF_ADAPTER_REPO}", "gguf": f"{hf}/{versions.HF_GGUF_REPO}"},
            "en": {"merged": f"{hf}/{versions.HF_MODEL_REPO_EN}", "lora": f"{hf}/{versions.HF_ADAPTER_REPO_EN}", "gguf": f"{hf}/{versions.HF_GGUF_REPO_EN}"},
        },
        "datasets": {"jp": f"{hf}/datasets/{versions.HF_DATASET_REPO}", "en": f"{hf}/datasets/{versions.HF_DATASET_REPO_EN}"},
    }  # fmt: skip


# The site shows the headline figures only; the full set lives in README.md and REPORT.md.
SITE_FIGURES = ("data_funnel", "judge_preference", "safety")


def sync_site(site: Path) -> None:
    """Copy the exported numbers and the site's figures into a checkout of the website repository."""
    import shutil

    (site / "src" / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT, site / "src" / "data" / "kitsune.json")
    figs = site / "public" / "figures"
    shutil.rmtree(figs, ignore_errors=True)
    (figs / "dark").mkdir(parents=True)
    for name in SITE_FIGURES:
        for ext in (".svg", ".png"):
            shutil.copy(R / "figures" / f"{name}{ext}", figs / f"{name}{ext}")
        shutil.copy(R / "figures" / "dark" / f"{name}.svg", figs / "dark" / f"{name}.svg")
    print(f"synced data and {len(SITE_FIGURES)} figures into {site}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, help="local checkout of kitsune-tales-qwen-site to sync into")
    args = ap.parse_args()
    ja = _load(R / "results.json")
    enr = _load(R / "results_en.json")
    rel = yaml.safe_load(Path("configs/release.yaml").read_text(encoding="utf-8"))
    ecfg = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))
    data = {
        "project": {
            "name": "Kitsune Tales",
            "base_model": versions.BASE_MODEL,
            "base_revision": versions.BASE_REVISION,
            "params": {"stored_b": 7.52, "effective_b": 4.62},
            "generators": [versions.GENERATOR_MODEL, versions.GENERATOR2_MODEL],
            "judge": versions.JUDGE_MODEL,
            "wandb": f"https://wandb.ai/whoashish115-base/{versions.WANDB_PROJECT}",
            "links": _links(),
            "models": [
                {
                    "slug": versions.MODEL_SLUG,
                    "lang": "ja",
                    "release_system": rel["ja"]["system"],
                    "recipe": "SFT",
                },
                {
                    "slug": versions.MODEL_SLUG_EN,
                    "lang": "en",
                    "release_system": rel["en"]["system"],
                    "recipe": "SFT + DPO",
                },
            ],
            "genres": [{"ja": g, "en": GENRE_EN[g], "en_prompt": en.GENRE_NAME_EN[g]} for g in GENRES],
            "formats": [
                {
                    "ja": k,
                    "en": en.FORMAT_NAME_EN[k],
                    "ja_chars": [f.target_min, f.target_max],
                    "en_words": [en.FORMATS_EN[k].target_min, en.FORMATS_EN[k].target_max],
                }
                for k, f in FORMATS.items()
            ],
            "decoding": ecfg["decoding"],
            "seeds": ecfg["seeds"],
        },
        "data": {"ja": _data(R / "data" / "stats.json"), "en": _data(R / "data_en" / "stats.json")},
        "training": _train(),
        "curves": _curves(),
        "eval": {
            "ja": {
                "systems": _systems(
                    ja, ["base", "kitsune-sft", "kitsune", "qwen3.5-4b", "qwen3.5-9b", "teacher"]
                ),
                "judge": _judge(ja),
                "ppl": ja.get("ppl"),
                "lm_eval": ja.get("lm_eval_summary"),
                "leakage": ja.get("leakage"),
            },
            "en": {
                "systems": _systems(enr, ["base-en", "kitsune-en-sft", "kitsune-en"]),
                "judge": _judge(enr),
                "ppl": _load(R / "ppl_en.json"),
                "leakage": _load(R / "leakage_en.json"),
            },
        },
        "gguf": _gguf(),
        "budget": _budget(),
        "samples": _samples(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    if args.site:
        sync_site(args.site)


if __name__ == "__main__":
    main()
