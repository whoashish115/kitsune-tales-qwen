from __future__ import annotations
import gzip
import json
import random
from pathlib import Path
import pytest
from kitsune.eval.judge import CORRUPTIONS, combine, corrupt, parse_verdict
from kitsune.eval.judge_plan import comparison_jobs, validation_jobs
from kitsune.eval.leakage import NgramIndex
from kitsune.eval.report import build_report, judge_table, results_table
from kitsune.fixtures import GENRES, STORY, SYNOPSIS, TITLE
from kitsune.train.dpo import build_pairs
# ----------------------------------------------------------------------------- judge parsing

@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("理由…\n判定: A", "A"),
        ("判定：B", "B"),
        ("判定: Ａ", "A"),
        ("まず判定: A と考えたが…\n最終的に\n判定: 引き分け", "tie"),
        ("結論は出ません", None),
    ],
)
def test_parse_verdict(text: str, want: str | None) -> None:
    assert parse_verdict(text) == want

def test_combine_requires_both_orders() -> None:
    assert combine("A", "B").result == "x" and combine("A", "B").consistent  # x first, then x second
    assert combine("B", "A").result == "y"
    assert combine("A", "A").result == "tie" and not combine("A", "A").consistent  # position bias
    assert combine("tie", "tie").result == "tie" and combine("tie", "tie").consistent
    assert combine(None, "A").result == "invalid"

@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_corruptions_change_the_story(kind: str) -> None:
    bad = corrupt(STORY, kind, random.Random(0), other_story=SYNOPSIS)
    assert bad != STORY and bad.strip()

def test_corrupt_rejects_unknown() -> None:
    raise NotImplementedError

def _gen_rows(system: str, text: str, n: int = 12) -> list[dict]:
    rows = []
    for i in range(n):
        fmt = ["あらすじ", "短編", "続き"][i % 3]
        for seed in range(3):
            rows.append(
                {
                    "system": system,
                    "suite": "test",
                    "prompt_id": f"test-{i:04d}",
                    "seed": seed,
                    "format": fmt,
                    "genres": GENRES,
                    "title": TITLE,
                    "passage": STORY[:250] if fmt == "続き" else None,
                    "text": text if seed == 0 else text[seed * 3 :],
                }
            )
    rows += [
        {
            "system": system,
            "suite": "policy",
            "prompt_id": "p1",
            "seed": 0,
            "kind": "sexual",
            "format": "短編",
            "title": "x",
            "text": "申し訳ありませんが、このご依頼にはお応えできません。",
        },
        {
            "system": system,
            "suite": "policy",
            "prompt_id": "p2",
            "seed": 0,
            "kind": "adversarial",
            "format": "短編",
            "title": "x",
            "text": text,
        },
    ]
    return rows

def test_comparison_jobs_both_orders_and_stratified() -> None:
    print("[debug] test_comparison_jobs_both_orders_and_stratified", flush=True)
    jobs = comparison_jobs(_gen_rows("k", STORY), _gen_rows("b", SYNOPSIS), "k", "b", n_pairs=6)
    assert len(jobs) == 12
    orders = {}
    for j in jobs:
        orders.setdefault(j.meta["pair_id"], set()).add(j.meta["order"])
    assert all(v == {"xy", "yx"} for v in orders.values())
    xy = next(j for j in jobs if j.meta["order"] == "xy")
    assert STORY in xy.messages[1]["content"].split("【作品B】")[0]

# ----------------------------------------------------------------------------- leakage

def test_leakage_index_detects_copied_spans() -> None:
    idx = NgramIndex([STORY])
    copied = idx.audit("まったく別の書き出し。" + STORY[100:300] + "そして終わり。")
    fresh = idx.audit(SYNOPSIS)
    assert copied["max_span"] >= 150 and copied["overlap_rate"] > 0.5
    assert fresh["max_span"] < 32

# ----------------------------------------------------------------------------- DPO pairs
