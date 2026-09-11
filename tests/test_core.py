from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pydantic
import pytest
from kitsune import cost
from kitsune.data.dedup import dedup, jaccard, shingles, title_is_near
from kitsune.eval.metrics import (
    bootstrap_ci,
    char_bleu,
    distinct_n,
    output_metrics,
    paired_bootstrap_diff,
    self_bleu,
)
from kitsune.schema import Record, content_hash, make_record, read_jsonl, write_jsonl

A = "王都の冒険者ギルドは、今日も朝から騒がしかった。追放された剣士は依頼書を差し出した。"
B = "辺境の村では、元魔王が静かに畑を耕していた。勇者はその姿を見て、剣を下ろした。"


# ----------------------------------------------------------------------------- dedup


def test_dedup_exact_and_near() -> None:
    near_a = A.replace("今日も", "今朝も")
    items = [A, B, A, " " + A + "\n", near_a]
    kept, rep, dup_of = dedup(items, key=lambda x: x, threshold=0.7)
    assert kept == [A, B]
    assert rep.exact_dupes == 2 and rep.near_dupes == 1
    assert dup_of[2] == 0 and dup_of[4] == 0


def test_dedup_keeps_distinct() -> None:
    kept, rep, _ = dedup([A, B], key=lambda x: x)
    assert len(kept) == 2 and rep.exact_dupes == rep.near_dupes == 0


def test_jaccard_and_titles() -> None:
    assert jaccard(shingles(A), shingles(A)) == 1.0
    assert jaccard(shingles(A), shingles(B)) < 0.1
    assert title_is_near("追放された剣士は最強になる", ["追放された剣士は最強になる！"])
    assert not title_is_near("星降る湖の歌姫", ["追放された剣士は最強になる"])


# ----------------------------------------------------------------------------- metrics


def test_distinct_n() -> None:
    assert distinct_n(["ああああ"], 1) == pytest.approx(0.25)
    assert distinct_n(["あいうえ"], 2) == 1.0
    assert distinct_n([], 2) == 0.0


def test_bleu_and_self_bleu() -> None:
    assert char_bleu(A, [A]) == pytest.approx(1.0)
    assert char_bleu(A, [B]) < 0.2
    assert self_bleu([A, A, A]) == pytest.approx(1.0)
    assert self_bleu([A, B]) < 0.2
    assert math.isnan(self_bleu([A]))


def test_output_metrics_flags() -> None:
    m = output_metrics(A * 20, "短編", ["冒険者ギルド"], "追放された剣士")
    assert m.repetitive == 1.0 and m.degenerate == 1.0
    m2 = output_metrics(A + B, "あらすじ", ["冒険者ギルド"], "追放された剣士")
    assert m2.repetitive == 0.0 and m2.fantasy == 1.0 and m2.title_reflected == 1.0
    assert m2.zh_contaminated == 0.0 and m2.unsafe == 0.0
    assert math.isnan(output_metrics(A, "続き", ["冒険者ギルド"], "題", passage=B).title_reflected)


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, 300)
    ci = bootstrap_ci(x, n_boot=2000)
    assert ci.low < ci.mean < ci.high
    assert ci.high - ci.low < 0.05
    d = paired_bootstrap_diff(list(x), list(x + 0.1), n_boot=2000)
    assert d.mean == pytest.approx(0.1) and d.low > 0
    assert bootstrap_ci([float("nan")]).n == 0


# ----------------------------------------------------------------------------- schema


def _rec(**over: object) -> Record:
    base = dict(
        id="x1",
        genres=["冒険者ギルド"],
        title="追放された剣士",
        format="短編",
        prompt="ジャンル: 冒険者ギルド\nタイトル: 追放された剣士\n形式: 短編",
        response=A,
        source="synthetic",
        generator="Qwen/Qwen3.6-35B-A3B-FP8@95a723d",
        license="apache-2.0",
    )
    base.update(over)
    return make_record(**base)


def test_record_valid_and_jsonl_round_trip(tmp_path: Path) -> None:
    r = _rec()
    assert r.hash == content_hash(r.prompt, r.response)
    p = tmp_path / "d.jsonl"
    assert write_jsonl(p, [r, r.model_dump()]) == 2
    rows = list(read_jsonl(p))
    assert Record(**rows[0]) == r
    assert "冒険者" in p.read_text(encoding="utf-8")  # non-ASCII kept


@pytest.mark.parametrize(
    "over",
    [
        {"genres": ["恋愛"]},
        {"genres": []},
        {"genres": ["冒険者ギルド", "冒険者ギルド"]},
        {"format": "長編"},
        {"source": "scraped"},
    ],
)
def test_record_rejects_bad_fields(over: dict) -> None:
    with pytest.raises(pydantic.ValidationError):
        _rec(**over)


def test_record_rejects_wrong_hash() -> None:
    d = _rec().model_dump()
    d["response"] = B
    with pytest.raises(pydantic.ValidationError):
        Record(**d)


def test_record_rejects_unknown_keys() -> None:
    d = _rec().model_dump()
    d["surprise"] = 1
    with pytest.raises(pydantic.ValidationError):
        Record(**d)


# ----------------------------------------------------------------------------- cost


def test_hourly_rate_includes_cpu_and_memory() -> None:
    r = cost.hourly_rate("H100", cpu_cores=8, mem_gib=64)
    assert r == pytest.approx(3.95 + 8 * 0.0473 + 64 * 0.008)
    with pytest.raises(KeyError):
        cost.hourly_rate("TPU", 1, 1)


def test_ledger_guard_and_close(tmp_path: Path) -> None:
    p = tmp_path / "ledger.jsonl"
    cost.open_job("2", "hello", "T4", 0.1, 1, 1, path=p, account="kitsune30")
    assert cost.spent(cost.read_ledger(p)) == pytest.approx(cost.estimate("T4", 0.1, 1, 1), abs=1e-4)
    e = cost.close_job("hello", 0.05, path=p)
    assert e.actual_usd == pytest.approx(cost.estimate("T4", 0.05, 1, 1), abs=1e-4)
    with pytest.raises(cost.BudgetExceededError):
        cost.open_job("5", "too-big", "H100", 10, 8, 64, path=p, account="kitsune30")
    with pytest.raises(cost.BudgetExceededError):  # planned remainder counts too
        cost.guard(1.0, planned_remaining_usd=28.5, path=p, account="kitsune30")
    # Accounts are budgeted separately, each with a $1 gap below its credit ($30 / $14.28);
    # new estimates are inflated by ESTIMATE_SAFETY (1.25); recorded spend counts at face value (D-028).
    assert cost.guard(10.0, path=p, account="kitsune12") == pytest.approx(12.5)
    with pytest.raises(cost.BudgetExceededError):
        cost.guard(11.2, path=p, account="kitsune12")
    # Recorded spend is inflated by LEDGER_SAFETY.
    q = tmp_path / "ledger2.jsonl"
    cost.open_job("5", "big", "H100", 2.0, 8, 64, path=q, account="kitsune30")
    rec = cost.spent(cost.read_ledger(q), "kitsune30")
    assert cost.guard(0.0, path=q, account="kitsune30") == pytest.approx(rec * cost.LEDGER_SAFETY)
    # Modal's own billing wins when it is higher than the ledger.
    with pytest.raises(cost.BudgetExceededError):
        cost.guard(1.0, path=p, account="kitsune30", billed=29.2)
    assert cost.ACCOUNT_KILL_USD == {"kitsune30": 29.6, "kitsune12": 13.9}
    with pytest.raises(cost.BudgetExceededError):  # unknown account = refuse to launch
        cost.guard(0.01, path=p, account="default")
    assert "hello" in cost.render_table(cost.read_ledger(p))
    assert cost.read_ledger(p)[0].account == "kitsune30"
