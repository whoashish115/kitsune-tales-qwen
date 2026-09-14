"""Resumable pipeline runner: each track runs its Modal steps in order and skips the ones already done.

    python -m kitsune.orchestrator --track main   # kitsune30: data, SFT, DPO, evaluation, export
    python -m kitsune.orchestrator --track k12    # kitsune12: ablations, lm-eval, baselines, judges
    python -m kitsune.orchestrator --track en     # English model (D-024): data, SFT, DPO, evaluation, judge, GGUF
A finished step writes ``reports/orchestrator/<track>/<step>.json`` (local run state, not in git). GPU jobs go
through the budget guard in ``modal_app.ledger``. A failed step stops its track, so nothing downstream runs on bad
inputs.
"""
from __future__ import annotations
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
ROOT = Path(__file__).resolve().parents[1]
MARKS = ROOT / "reports" / "orchestrator"
APP = "src/modal_app.py"
PY = sys.executable

@dataclass
class Step:
    name: str
    account: str | None = None  # Modal profile for modal steps; None for local steps
    entry: str | None = None  # modal_app entrypoint
    args: list[str] = field(default_factory=list)
    fn: Callable[[], None] | None = None  # local step
    wait_for: list[str] = field(
        default_factory=list
    )  # marker paths (relative to MARKS) that must exist first
    args_fn: Callable[[], list[str]] | None = None  # compute args at run time (e.g. from the pilot)

def _note(kind: str, task: str, result: str = "", account: str = "") -> None:
    """One console line per pipeline event (GPU jobs are also recorded in reports/cost_ledger.jsonl)."""
    raise NotImplementedError

def _done(track: str, name: str) -> Path:
    return MARKS / track / f"{name}.json"

def _run_modal(step: Step, track: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%MZ")
    log = ROOT / "logs" / "runs" / f"{ts}_{track}_{step.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    args = step.args + (step.args_fn() if step.args_fn else [])
    cmd = [PY, "-m", "modal", "run", f"{APP}::{step.entry}", *args]
    env = {**os.environ, "MODAL_PROFILE": step.account or "", "PYTHONIOENCODING": "utf-8"}
    print(f"[orchestrate] {track}/{step.name}: {' '.join(cmd[3:])} (account {step.account})", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False
        ).returncode
    if rc != 0:
        raise RuntimeError(f"modal step {step.name} exited {rc}; see {log.relative_to(ROOT)}")

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

def gate_pilot() -> None:
    p = json.loads((ROOT / "reports" / "train" / "sft-pilot.json").read_text(encoding="utf-8"))
    if not p.get("eval_loss") or p["eval_loss"] != p["eval_loss"] or p["eval_loss"] > 5:
        raise RuntimeError(f"pilot gate failed: eval_loss={p.get('eval_loss')}")

def build_test_set_en() -> None:
    from kitsune.data.cli import TEST_SET_EN_HASH
    from kitsune.data.cli import build_test_set_en as _b

    if not TEST_SET_EN_HASH.exists():
        print(_b(ROOT / "data" / "raw" / "full_en"))

def pipeline_build_en() -> None:
    """Build the English dataset and gate it: enough examples, and the drop profile must look sane."""
    from kitsune.data.pipeline import build

    stats = build(
        ROOT / "data" / "raw" / "full_en",
        ROOT / "data" / "processed_en",
        ROOT / "reports" / "data_en",
        lang="en",
    )
    if stats["n_train"] < 2500:
        raise RuntimeError(f"too few English training examples after filtering: {stats['n_train']}")
    _note(
        kind="cpu",
        task="English pipeline build (filters → dedup → split)",
        result=f"train {stats['n_train']}, val {stats['n_val']}, kept synthetic {stats['n_kept_synthetic']} of {stats['n_candidates']}",
    )

def copy_adapter(run: str, src_profile: str, dst_profile: str) -> None:
    """Copy runs/<run>/adapter between the two Modal accounts through the local disk (kept as the offline copy)."""
    local = ROOT / "data" / "adapters" / run
    if not (local / "adapter" / "adapter_model.safetensors").exists():
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                PY,
                "-m",
                "modal",
                "volume",
                "get",
                "kitsune-data",
                f"runs/{run}/adapter/",
                f"{local}/",
                "--profile",
                src_profile,
                "--force",
            ],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [
            PY,
            "-m",
            "modal",
            "volume",
            "put",
            "kitsune-data",
            str(local / "adapter"),
            f"runs/{run}/adapter",
            "--profile",
            dst_profile,
            "--force",
        ],
        cwd=ROOT,
        check=True,
    )
    out = subprocess.run(
        [PY, "-m", "modal", "volume", "ls", "kitsune-data", f"runs/{run}/adapter", "--profile", dst_profile],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if f"runs/{run}/adapter/adapter_model.safetensors" not in out.replace("\\", "/"):
        raise RuntimeError(f"adapter copy to {dst_profile} not where expected:\n{out[:500]}")

def make_eval() -> None:
    raise NotImplementedError
