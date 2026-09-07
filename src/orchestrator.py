"""Resumable pipeline runner: each track runs its Modal steps in order and skips the ones already done.

    python -m kitsune.orchestrator --track main   # kitsune30: data, SFT, DPO, evaluation, export
    python -m kitsune.orchestrator --track k12    # kitsune12: ablations, lm-eval, baselines, judges
    python -m kitsune.orchestrator --track en     # English model (D-024): data, SFT, DPO, evaluation, judge, GGUF
A finished step writes ``reports/orchestrator/<track>/<step>.json`` (local run state, not in git). GPU jobs go
through the budget guard in ``modal_app.ledger``. A failed step stops its track, so nothing downstream runs on bad
inputs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from __future__ import annotations
def _note(kind: str, task: str, result: str = "", account: str = "") -> None:
    """One console line per pipeline event (GPU jobs are also recorded in reports/cost_ledger.jsonl)."""
    raise NotImplementedError

def _done(track: str, name: str) -> Path:
    raise NotImplementedError

# ----------------------------------------------------------------------------- local steps

def pipeline_build() -> None:
    from kitsune.data.pipeline import build

    stats = build(ROOT / "data" / "raw" / "full", ROOT / "data" / "processed", ROOT / "reports" / "data")
    if stats["n_train"] < 3000:
        raise RuntimeError(f"too few training examples after filtering: {stats['n_train']}")
    _note(
        kind="cpu",
        task="pipeline build (filters → dedup → split)",
        result=f"train {stats['n_train']}, val {stats['n_val']}, kept synthetic {stats['n_kept_synthetic']} of {stats['n_candidates']}",
    )

def main_est_hours() -> list[str]:
    """Estimate the main run's duration from the pilot's measured throughput (+25 % margin)."""
    p = json.loads((ROOT / "reports" / "train" / "sft-pilot.json").read_text(encoding="utf-8"))
    tps = p.get("tokens_per_s_mean") or 1500.0
    frac = 0.05
    total_tokens = p["data"]["train"]["tokens"] / frac * 1  # 1 epoch of the full data (D-021)
    hours = max(0.3, total_tokens / tps / 3600 * 1.25)
    _note(
        kind="decision",
        task="main SFT duration from pilot throughput",
        result=f"{tps:.0f} tok/s → est {hours:.2f} h",
    )
    return ["--est-hours", f"{hours:.2f}"]
