"""Compute-cost estimation, the spend ledger, and the budget guard.
Rates are a snapshot of ``modal billing rates`` taken on 2026-09-29 (see docs/BUDGET.md).
The ledger is ``reports/cost_ledger.jsonl`` (machine-readable, committed); the table in
``docs/BUDGET.md`` is regenerated from it with ``python -m kitsune.cost render``.
"""
from __future__ import annotations
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
RATES_SOURCE = "modal billing rates, 2026-09-29"
GPU_USD_PER_H: dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "CPU": 0.0,
}
CPU_USD_PER_CORE_H = 0.0473
MEM_USD_PER_GIB_H = 0.008
VOLUME_USD_PER_GIB_MONTH = 0.09
# Per-account credit and hard stops (D-014): spend never exceeds an account's credit, and each stop keeps about $0.40
ACCOUNT_CAPS_USD: dict[str, float] = {"kitsune30": 30.00, "kitsune12": 14.28}
ACCOUNT_KILL_USD: dict[str, float] = {"kitsune30": 29.60, "kitsune12": 13.90}
HARD_CAP_USD = sum(ACCOUNT_CAPS_USD.values())
KILL_THRESHOLD_USD = sum(ACCOUNT_KILL_USD.values())
PROJECT_START = datetime(2026, 9, 29, tzinfo=UTC)
APP_NAME = "kitsune"
REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "reports" / "cost_ledger.jsonl"

class BudgetExceededError(RuntimeError):
    """Raised by :func:`guard` when a job would push projected spend past the kill threshold."""

def hourly_rate(gpu: str, cpu_cores: float, mem_gib: float, gpu_count: int = 1) -> float:
    """All-in $/hour for one container: GPU(s) + CPU cores + memory."""
    if gpu not in GPU_USD_PER_H:
        raise KeyError(f"unknown GPU {gpu!r}; known: {sorted(GPU_USD_PER_H)}")
    return GPU_USD_PER_H[gpu] * gpu_count + cpu_cores * CPU_USD_PER_CORE_H + mem_gib * MEM_USD_PER_GIB_H

def estimate(gpu: str, hours: float, cpu_cores: float, mem_gib: float, gpu_count: int = 1) -> float:
    """Estimated $ for a job of ``hours`` wall-clock."""
    return hours * hourly_rate(gpu, cpu_cores, mem_gib, gpu_count)

def active_account() -> str:
    """The Modal account a job will bill to: ``MODAL_PROFILE`` if set, else the active CLI profile."""
    import os

    if os.environ.get("MODAL_PROFILE"):
        return os.environ["MODAL_PROFILE"]
    try:
        import modal.config

        return str(modal.config._profile)
    except Exception:
        return "unknown"

@dataclass
class LedgerEntry:
    phase: str
    job: str
    gpu: str
    cpu_cores: float
    mem_gib: float
    est_hours: float
    est_usd: float
    actual_hours: float | None = None
    actual_usd: float | None = None
    notes: str = ""
    started_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    account: str = ""

def write_ledger(entries: list[LedgerEntry], path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

def billed_usd(account: str | None = None) -> float | None:
    """This project's spend on ``account`` according to Modal billing (``modal billing report --csv``).

    ``--for "this month"`` includes the current partial day (``--start`` reports complete days only); once
    in October, the project's September days (from PROJECT_START) are added. Only rows whose description is
    one of this project's app/volume names are summed, so other projects in the same workspace are not counted.
    Returns None if the CLI call fails.
    """
    raise NotImplementedError
