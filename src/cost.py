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
# as a buffer for billing lag. The active account is the Modal CLI profile.
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


def read_ledger(path: Path = LEDGER_PATH) -> list[LedgerEntry]:
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [LedgerEntry(**r) for r in rows]


def write_ledger(entries: list[LedgerEntry], path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def spent(entries: list[LedgerEntry], account: str | None = None) -> float:
    """Committed spend (optionally for one account): actual where known, otherwise the estimate."""
    return sum(
        e.actual_usd if e.actual_usd is not None else e.est_usd
        for e in entries
        if account is None or e.account == account
    )


# The local ledger times .remote() calls; Modal also bills container startup, the idle window after a
# job, and image builds (as CPU). Early on 2026-09-29 (many image builds) billing was 1.17-1.3 × ledger (D-014);
# over the full day ($15, 2026-09-29 21:16 UTC) Modal's live hourly billing was 1.03 × ledger (D-025).
# Recorded spend is inflated by LEDGER_SAFETY, and a new job's *estimate* by ESTIMATE_SAFETY (estimates are
# the less certain number). Modal's own billed total is always a floor (see ``guard``).
# D-028: at face value. Modal's live billed total (floor) and the ledger including running jobs' estimates are
# compared directly; the ledger measured 0.96-1.12 x billing per account, so an extra factor double-counted margin.
LEDGER_SAFETY = 1.0
ESTIMATE_SAFETY = 1.25
LAST_BILLED: float | None = None  # Modal-billed spend seen by the most recent open_job (for the work log)
PROJECT_APP_NAMES = ("kitsune", "kitsune-models", "kitsune-data")


def billed_usd(account: str | None = None) -> float | None:
    """This project's spend on ``account`` according to Modal billing (``modal billing report --csv``).

    ``--for "this month"`` includes the current partial day (``--start`` reports complete days only); once
    in October, the project's September days (from PROJECT_START) are added. Only rows whose description is
    one of this project's app/volume names are summed, so other projects in the same workspace are not counted.
    Returns None if the CLI call fails.
    """
    import csv
    import io
    import subprocess
    import sys

    account = account or active_account()
    base = [
        sys.executable,
        "-m",
        "modal",
        "billing",
        "report",
        "--resolution",
        "d",
        "--csv",
        "--profile",
        account,
    ]
    queries = [["--for", "this month"]]
    now = datetime.now(UTC)
    if (now.year, now.month) != (PROJECT_START.year, PROJECT_START.month):
        queries.append(["--start", PROJECT_START.date().isoformat(), "--end", "2026-10-01"])
    total = 0.0
    for q in queries:
        try:
            out = subprocess.run(base + q, capture_output=True, text=True, timeout=120, check=True).stdout
        except Exception:
            return None
        total += sum(
            float(r["Cost"])
            for r in csv.DictReader(io.StringIO(out))
            if r.get("Description") in PROJECT_APP_NAMES
        )
    return total


def guard(
    new_estimate_usd: float,
    planned_remaining_usd: float = 0.0,
    path: Path = LEDGER_PATH,
    account: str | None = None,
    billed: float | None = None,
) -> float:
    """Raise if spent + this job + the rest of the plan would exceed a hard stop.

    ``spent`` is the larger of the ledger and Modal's own billing (``billed``), so an underestimating
    ledger can never let a job through. Checks the account's hard stop (D-014) and the project total.
    Returns the account's projected total. Call before every GPU launch.
    """
    account = account or active_account()
    if account not in ACCOUNT_KILL_USD:
        raise BudgetExceededError(
            f"unknown Modal account {account!r}; expected one of {sorted(ACCOUNT_KILL_USD)}"
        )
    entries = read_ledger(path)
    already = max(spent(entries, account) * LEDGER_SAFETY, billed or 0.0)
    projected = already + new_estimate_usd * ESTIMATE_SAFETY + planned_remaining_usd
    if projected > ACCOUNT_KILL_USD[account]:
        raise BudgetExceededError(
            f"{account}: projected ${projected:.2f} > kill threshold ${ACCOUNT_KILL_USD[account]:.2f}; "
            "shrink the plan (see BUDGET.md)"
        )
    others = (spent(entries) - spent(entries, account)) * LEDGER_SAFETY
    total = others + already + new_estimate_usd * ESTIMATE_SAFETY + planned_remaining_usd
    if total > KILL_THRESHOLD_USD:
        raise BudgetExceededError(f"project: projected ${total:.2f} > ${KILL_THRESHOLD_USD:.2f}")
    return projected


def open_job(
    phase: str,
    job: str,
    gpu: str,
    est_hours: float,
    cpu_cores: float,
    mem_gib: float,
    planned_remaining_usd: float = 0.0,
    notes: str = "",
    path: Path = LEDGER_PATH,
    account: str | None = None,
) -> LedgerEntry:
    """Guard, then append an estimate row to the ledger (before launch).

    An older open row for the same job belongs to a launcher that was stopped before it could record
    its wall-clock; it is closed at $0 first (its billed time still counts via ``billed_usd``), so a
    restart never counts one job's estimate twice.
    """
    account = account or active_account()
    entries = read_ledger(path)
    stale = [e for e in entries if e.job == job and e.account == account and e.actual_usd is None]
    for e in stale:
        e.actual_hours, e.actual_usd = 0.0, 0.0
        e.notes = (e.notes + "; superseded by a later row for the same job (launcher restarted)").strip("; ")
    if stale:
        write_ledger(entries, path)
    est = estimate(gpu, est_hours, cpu_cores, mem_gib)
    billed = billed_usd(account) if path == LEDGER_PATH else None
    global LAST_BILLED
    LAST_BILLED = billed
    projected = guard(est, planned_remaining_usd, path, account, billed=billed)
    print(
        f"[budget] {account}: billed so far ${billed if billed is not None else float('nan'):.3f}; projected after this job ${projected:.2f} (stop ${ACCOUNT_KILL_USD[account]:.2f})"
    )
    entries = read_ledger(path)
    e = LedgerEntry(
        phase, job, gpu, cpu_cores, mem_gib, est_hours, round(est, 4), notes=notes, account=account
    )
    entries.append(e)
    write_ledger(entries, path)
    return e


def close_job(job: str, actual_hours: float, notes: str = "", path: Path = LEDGER_PATH) -> LedgerEntry:
    """Fill in actual hours/$ for the most recent open row named ``job``."""
    entries = read_ledger(path)
    for e in reversed(entries):
        if e.job == job and e.actual_hours is None:
            e.actual_hours = round(actual_hours, 4)
            e.actual_usd = round(estimate(e.gpu, actual_hours, e.cpu_cores, e.mem_gib), 4)
            if notes:
                e.notes = (e.notes + "; " + notes).strip("; ")
            write_ledger(entries, path)
            return e
    raise KeyError(f"no open ledger row for job {job!r}")


def render_table(entries: list[LedgerEntry]) -> str:
    """Markdown ledger table for docs/BUDGET.md."""
    head = (
        "| # | Date (UTC) | Account | Phase | Job | GPU | Est. h | Est. $ | Actual h | Actual $ | Cumulative $ | Notes |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    cum = 0.0
    for i, e in enumerate(entries, 1):
        cum += e.actual_usd if e.actual_usd is not None else e.est_usd
        ah = f"{e.actual_hours:.4f}" if e.actual_hours is not None else "n/a"
        au = f"{e.actual_usd:.4f}" if e.actual_usd is not None else "n/a"
        lines.append(
            f"| {i} | {e.started_utc[:16]} | {e.account} | {e.phase} | `{e.job}` | {e.gpu} | {e.est_hours:.3f} | "
            f"{e.est_usd:.4f} | {ah} | {au} | {cum:.4f} | {e.notes} |"
        )
    return "\n".join(lines)


def summary_lines(entries: list[LedgerEntry]) -> list[str]:
    """Per-account and project spend vs caps."""
    out = [f"{acct}: ${spent(entries, acct):.4f} / ${cap:.2f}" for acct, cap in ACCOUNT_CAPS_USD.items()]
    out.append(f"Project: ${spent(entries):.4f} / ${HARD_CAP_USD:.2f}")
    return out


def main(argv: list[str]) -> int:
    if argv[:1] == ["render"]:
        entries = read_ledger()
        print(render_table(entries))
        print()
        print("\n".join(summary_lines(entries)))
        return 0
    print("usage: python -m kitsune.cost render")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
