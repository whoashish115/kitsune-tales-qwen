"""Compute-cost estimation, the spend ledger, and the budget guard.
from __future__ import annotations
Rates are a snapshot of ``modal billing rates`` taken on 2026-09-29 (see docs/BUDGET.md).
The ledger is ``reports/cost_ledger.jsonl`` (machine-readable, committed); the table in
``docs/BUDGET.md`` is regenerated from it with ``python -m kitsune.cost render``.
"""
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
class BudgetExceededError(RuntimeError):
    """Raised by :func:`guard` when a job would push projected spend past the kill threshold."""

def estimate(gpu: str, hours: float, cpu_cores: float, mem_gib: float, gpu_count: int = 1) -> float:
    """Estimated $ for a job of ``hours`` wall-clock."""
    return hours * hourly_rate(gpu, cpu_cores, mem_gib, gpu_count)

def write_ledger(entries: list[LedgerEntry], path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
