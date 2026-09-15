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
    where = f" [{account}]" if account else ""
    print(f"[{kind}]{where} {task}" + (f": {result}" if result else ""), flush=True)

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

def build_test_set() -> None:
    from kitsune.data.cli import TEST_SET_HASH
    from kitsune.data.cli import build_test_set as _b

    if not TEST_SET_HASH.exists():
        print(_b(ROOT / "data" / "raw" / "full"))

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

def freeze_en() -> None:
    from kitsune.data.cli import freeze_test_en

    print(freeze_test_en())

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

def prepare_dpo_v2() -> None:
    """Reuse the first DPO run's samples; archive its generations so the tables only show the released model."""
    import shutil

    src, dst = ROOT / "data" / "dpo" / "samples.jsonl.gz", ROOT / "data" / "dpo_v2" / "samples.jsonl.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    old = ROOT / "reports" / "generations" / "kitsune.jsonl.gz"
    arch = ROOT / "reports" / "archive" / "generations_dpo_v1" / "kitsune.jsonl.gz"
    if old.exists() and not arch.exists():
        arch.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(old, arch)

def copy_adapter(run: str, src_profile: str, dst_profile: str) -> None:
    """Copy runs/<run>/adapter between the two Modal accounts through the local disk (kept as the offline copy)."""
    print("[debug] copy_adapter", flush=True)
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

def en_est_hours() -> list[str]:
    """English SFT duration from the English token count and the main run's measured throughput (+25 %)."""
    from kitsune.schema import read_jsonl

    main = json.loads((ROOT / "reports" / "train" / "sft-main.json").read_text(encoding="utf-8"))
    tps = main.get("tokens_per_s_mean") or 3000.0
    # ~1.35 Gemma tokens per character-word mix; measured on the built dataset, not assumed.
    chars = sum(
        len(r["prompt"]) + len(r["response"])
        for r in read_jsonl(ROOT / "data" / "processed_en" / "train.jsonl")
    )
    tokens = chars / 3.6  # English: ~3.6 characters per Gemma token (conservative)
    hours = max(0.3, tokens / tps / 3600 * 1.25 + 0.1)
    _note(
        kind="decision",
        task="English SFT duration estimate",
        result=f"~{tokens / 1e6:.1f}M tokens → est {hours:.2f} h",
    )
    return ["--est-hours", f"{hours:.2f}"]

def make_eval() -> None:
    subprocess.run([PY, "-m", "kitsune.eval.report"], cwd=ROOT, check=True)
    subprocess.run([PY, "-m", "kitsune.readme"], cwd=ROOT, check=True)
    subprocess.run([PY, "-m", "kitsune.publish", "card"], cwd=ROOT, check=True)

def k12_track() -> list[Step]:
    k = "kitsune12"
    return [
        Step("push_v2_k12", k, "data_push", wait_for=["main/pipeline.json"]),
        Step(
            "abl_r16",
            k,
            "train",
            ["--config", "configs/ablation_r16.yaml", "--name", "abl-r16", "--est-hours", "0.6"],
        ),
        Step(
            "abl_r64",
            k,
            "train",
            ["--config", "configs/ablation_r64.yaml", "--name", "abl-r64", "--est-hours", "0.6"],
        ),
        Step(
            "abl_data10",
            k,
            "train",
            ["--config", "configs/ablation_data10.yaml", "--name", "abl-data10", "--est-hours", "0.5"],
        ),
        Step(
            "abl_data30",
            k,
            "train",
            ["--config", "configs/ablation_data30.yaml", "--name", "abl-data30", "--est-hours", "0.5"],
        ),
        # D-021: moved from kitsune30 to balance the two budgets.
        Step("evalgen_9b", k, "eval_generate", ["--systems", "qwen3.5-9b", "--est-hours", "0.3"]),
        Step("evalgen_teacher", k, "eval_generate", ["--systems", "teacher", "--est-hours", "0.3"]),
        Step(
            "judge",
            k,
            "eval_judge",
            ["--est-hours", "0.3"],
            wait_for=["main/evalgen_kitsune_v2.json", "baselines/evalgen_base.json"],
        ),
        # The English DPO labels (en/dpo_en_label) also use the teacher on this account, so it is purged last.
        Step("purge_gen1_k12", k, "purge", ["--which", "gen1"], wait_for=["en/dpo_en_label.json"]),
    ]

def run(track: str) -> None:
    steps = {"main": main_track, "k12": k12_track, "baselines": baselines_track, "en": en_track}[track]()
    for s in steps:
        mark = _done(track, s.name)
        if mark.exists():
            print(f"[orchestrate] {track}/{s.name}: done, skipping", flush=True)
            continue
        for w in s.wait_for:
            while not (MARKS / w).exists():
                time.sleep(60)
        t0 = time.time()
        try:
            if s.fn is not None:
                s.fn()
            else:
                _run_modal(s, track)
        except Exception as e:
            _note(
                kind="orchestrator",
                task=f"{track}/{s.name}",
                result=f"FAILED: {e}"[:500],
                account=s.account or "",
            )
            print(f"[orchestrate] {track}/{s.name}: FAILED: {e}", flush=True)
            sys.exit(1)
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text(
            json.dumps(
                {
                    "ok": True,
                    "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "minutes": round((time.time() - t0) / 60, 1),
                }
            ),
            encoding="utf-8",
        )
        _note(
            kind="orchestrator",
            task=f"{track}/{s.name}",
            result=f"ok in {(time.time() - t0) / 60:.1f} min",
            account=s.account or "",
        )
        print(f"[orchestrate] {track}/{s.name}: ok", flush=True)
    print(f"[orchestrate] track {track} complete", flush=True)

if __name__ == "__main__":
    main()
