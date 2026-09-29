"""Resumable pipeline runner: each track runs its cloud GPU steps in order and skips the ones already done.

    python -m kitsune.orchestrate --track main   # kitsune30: data, SFT, DPO, evaluation, export
    python -m kitsune.orchestrate --track k12    # kitsune12: ablations, lm-eval, baselines, judges
    python -m kitsune.orchestrate --track en     # English model (D-024): data, SFT, DPO, evaluation, judge, GGUF

A finished step writes ``reports/orchestrator/<track>/<step>.json`` (local run state, not in git). GPU jobs go
through the budget guard in ``gpu_jobs.ledger``. A failed step stops its track, so nothing downstream runs on bad
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
APP = "src/gpu_jobs.py"
PY = sys.executable


@dataclass
class Step:
    name: str
    account: str | None = None  # cloud profile for GPU steps; None for local steps
    entry: str | None = None  # gpu_jobs entrypoint
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


def _run_cloud(step: Step, track: str) -> None:
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
        raise RuntimeError(f"cloud step {step.name} exited {rc}; see {log.relative_to(ROOT)}")


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
    """Copy runs/<run>/adapter between the two cloud accounts through the local disk (kept as the offline copy)."""
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


def make_eval_en() -> None:
    subprocess.run([PY, "-m", "kitsune.eval.report", "--lang", "en"], cwd=ROOT, check=True)


def make_eval() -> None:
    subprocess.run([PY, "-m", "kitsune.eval.report"], cwd=ROOT, check=True)
    subprocess.run([PY, "-m", "kitsune.readme"], cwd=ROOT, check=True)
    subprocess.run([PY, "-m", "kitsune.publish", "card"], cwd=ROOT, check=True)


# ----------------------------------------------------------------------------- tracks


def main_track() -> list[Step]:
    k, k12 = "kitsune30", "kitsune12"
    dpo_v2 = ["--sft-run", "sft-main", "--name", "dpo-main-v2", "--outdir", "dpo_v2"]
    return [
        Step(
            "data_gen1",
            k,
            "data",
            [
                "--mode",
                "full",
                "--gen",
                "gen1",
                "--n-prompts",
                "9000",
                "--title-calls",
                "60",
                "--est-hours",
                "0.6",
                "--planned-remaining",
                "18",
            ],
        ),
        Step(
            "data_gen2",
            k,
            "data",
            [
                "--mode",
                "full",
                "--gen",
                "gen2",
                "--n-prompts",
                "6000",
                "--title-calls",
                "30",
                "--est-hours",
                "0.8",
                "--planned-remaining",
                "16",
            ],
        ),
        Step(
            "data_gen1_crosslabel",
            k,
            "data",
            [
                "--mode",
                "full",
                "--gen",
                "gen1",
                "--label-only",
                "--est-hours",
                "0.25",
                "--planned-remaining",
                "15",
            ],
        ),
        Step("purge_gen2", k, "purge", ["--which", "gen2"]),
        Step("test_set", fn=build_test_set),
        Step("pipeline", fn=pipeline_build),
        Step("push_k30", k, "data_push"),
        Step(
            "pilot",
            k,
            "train",
            [
                "--config",
                "configs/train_pilot.yaml",
                "--name",
                "sft-pilot",
                "--est-hours",
                "0.4",
                "--planned-remaining",
                "14",
            ],
        ),
        Step("pilot_gate", fn=gate_pilot),
        Step("push_v2_k30", k, "data_push"),  # D-020: dataset rebuilt after the inspection gate
        Step(
            "main",
            k,
            "train",
            ["--config", "configs/train_main.yaml", "--name", "sft-main", "--planned-remaining", "8"],
            args_fn=main_est_hours,
        ),
        Step("merge_sft", k, "merge", ["--adapter", "sft-main"]),
        Step("dpo", k, "dpo", ["--sft-run", "sft-main", "--name", "dpo-main"]),
        Step("merge_dpo", k, "merge", ["--adapter", "dpo-main"]),
        Step(
            "evalgen_kitsune", k, "eval_generate", ["--systems", "kitsune-sft,kitsune", "--est-hours", "0.35"]
        ),
        # D-027: the first DPO's teacher labels were 87 % truncated. Relabel the same samples (kitsune12, where the
        # teacher is cached), retrain on kitsune12 (budget), then merge + evaluate the v2 adapter on kitsune30.
        Step("dpo_v2_prepare", fn=prepare_dpo_v2),
        Step("dpo_v2_label", k12, "dpo", [*dpo_v2, "--stage", "label", "--until", "label"]),
        Step("dpo_v2_adapter_to_k12", fn=lambda: copy_adapter("sft-main", "kitsune30", "kitsune12")),
        Step("dpo_v2_train", k12, "dpo", [*dpo_v2, "--stage", "pairs", "--until", "train"]),
        Step("dpo_v2_adapter_to_k30", fn=lambda: copy_adapter("dpo-main-v2", "kitsune12", "kitsune30")),
        Step("merge_dpo_v2", k, "merge", ["--adapter", "dpo-main-v2"]),
        Step("evalgen_kitsune_v2", k, "eval_generate", ["--systems", "kitsune", "--est-hours", "0.25"]),
        Step("purge_gen1", k, "purge", ["--which", "gen1"]),
        Step("ppl", k, "eval_ppl"),
        # Regression check on base and the released Japanese model (configs/release.yaml). 200 items per task: the
        # remaining kitsune30 budget, measured, does not cover 500 (D-030).
        Step(
            "lmeval", k, "lm_eval", ["--systems", "base,kitsune-sft", "--limit", "200", "--est-hours", "0.2"]
        ),
        # The English track exports the Japanese GGUF too (gguf_jp); this step then finds it done and skips.
        Step("gguf", k, "gguf", ["--merged", "dpo-main-v2"], wait_for=["en/gguf_jp.json"]),
        Step("report", fn=make_eval, wait_for=["k12/judge.json"]),
    ]


def baselines_track() -> list[Step]:
    """Everything that does not need the trained model runs in parallel with training (saves evening time)."""
    k = "kitsune30"
    return [
        Step(
            "evalgen_base",
            k,
            "eval_generate",
            ["--systems", "base,qwen3.5-4b", "--gpu", "L4", "--est-hours", "0.35"],
            wait_for=["main/test_set.json"],
        ),
    ]


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


def en_track() -> list[Step]:
    """English variant (D-024) on kitsune30: both generators + cross-labels, SFT, eval, judge (kitsune12), GGUF.

    Order keeps each generator's weights needed once: gen1 (stories + self-labels + test passages) →
    gen2 (stories + labels of both) → gen1 label-only (cross-labels gen2's stories).
    """
    k, k12 = "kitsune30", "kitsune12"
    data = ["data", "--lang", "en", "--mode", "full_en"]
    dpo_en = [
        "--lang",
        "en",
        "--sft-run",
        "sft-en-main",
        "--name",
        "dpo-en-main",
        "--config",
        "configs/train_dpo_en.yaml",
    ]
    return [
        Step("freeze_en", fn=freeze_en),
        Step(
            "data_en_gen1",
            k,
            data[0],
            [*data[1:], "--gen", "gen1", "--n-prompts", "4400", "--title-calls", "30", "--est-hours", "0.45"],
        ),
        Step(
            "data_en_gen2",
            k,
            data[0],
            [*data[1:], "--gen", "gen2", "--n-prompts", "3000", "--title-calls", "25", "--est-hours", "0.4"],
        ),
        Step(
            "data_en_gen1_crosslabel",
            k,
            data[0],
            [*data[1:], "--gen", "gen1", "--label-only", "--est-hours", "0.15"],
        ),
        Step("purge_gen2_en", k, "purge", ["--which", "gen2"]),
        Step("test_set_en", fn=build_test_set_en),
        Step("pipeline_en", fn=pipeline_build_en),
        Step("push_en", k, "data_push", ["--lang", "en"]),
        Step(
            "sft_en",
            k,
            "train",
            ["--config", "configs/train_en_main.yaml", "--name", "sft-en-main"],
            args_fn=en_est_hours,
        ),
        Step("merge_en", k, "merge", ["--adapter", "sft-en-main"]),
        Step(
            "evalgen_en_sft",
            k,
            "eval_generate",
            ["--systems", "base-en,kitsune-en-sft", "--est-hours", "0.3"],
        ),
        # D-026: English DPO with the Japanese recipe; the teacher labels run on kitsune12 (budget balance).
        Step("dpo_en_sample", k, "dpo", [*dpo_en, "--stage", "sample", "--until", "sample"]),
        Step("dpo_en_label", k12, "dpo", [*dpo_en, "--stage", "label", "--until", "label"]),
        Step("dpo_en_train", k, "dpo", [*dpo_en, "--stage", "pairs", "--until", "train"]),
        Step("merge_dpo_en", k, "merge", ["--adapter", "dpo-en-main"]),
        Step("evalgen_en", k, "eval_generate", ["--systems", "kitsune-en", "--est-hours", "0.2"]),
        Step("ppl_en", k, "eval_ppl", ["--systems", "base-en,kitsune-en-sft,kitsune-en"]),
        Step("gguf_en", k, "gguf", ["--merged", "dpo-en-main", "--lang", "en"]),
        # The Japanese model's GGUF export.
        Step(
            "gguf_jp",
            k,
            "gguf",
            ["--merged", "dpo-main-v2", "--lang", "ja"],
            wait_for=["main/merge_dpo_v2.json"],
        ),
        Step("purge_gen1_en", k, "purge", ["--which", "gen1"], wait_for=["main/dpo.json"]),
        # On kitsune12, where the judge weights are cached (D-031).
        Step("judge_en", k12, "eval_judge_en", ["--est-hours", "0.25"]),
        Step("report_en", fn=make_eval_en),
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
                _run_cloud(s, track)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["main", "k12", "baselines", "en"], required=True)
    run(ap.parse_args().track)


if __name__ == "__main__":
    main()
