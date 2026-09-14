"""Research-style names for runs and artifacts (D-015).

sft-kitsune-tales-e4b-jp-r32-lr2e-4-ep1-d100-s42
dpo-kitsune-tales-e4b-jp-b0.1-lr2e-5-ep1-s42
sft-kitsune-tales-e4b-en-r32-lr2e-4-ep1-d100-s42
"""

from __future__ import annotations

from typing import Any

from kitsune import versions


def _lr(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e")


def run_name(stage: str, cfg: dict[str, Any]) -> str:
    """Deterministic, human-readable W&B run name from the stage and the key hyperparameters."""
    parts = [stage, versions.model_slug(cfg.get("lang", "ja"))]
    if stage == "dpo":
        parts += [
            f"b{cfg.get('beta', 0.1)}",
            f"lr{_lr(cfg.get('learning_rate', 2e-5))}",
            f"ep{cfg.get('epochs', 1)}",
        ]
    else:
        parts += [f"r{cfg['lora_r']}", f"lr{_lr(cfg['learning_rate'])}", f"ep{cfg.get('epochs', 1)}"]
        frac = cfg.get("data_fraction", 1.0)
        parts.append(f"d{round(frac * 100)}")
    parts.append(f"s{cfg.get('seed', 42)}")
    return "-".join(parts)


def run_tags(stage: str, cfg: dict[str, Any]) -> list[str]:
    lang = cfg.get("lang", "ja")
    return [
        versions.MODEL_FAMILY.lower(),
        stage,
        cfg.get("wandb_group", stage),
        f"lang-{versions.LANG_SUFFIX[lang]}",
        "gemma-4-e4b",
        "lora-bf16",
    ]
