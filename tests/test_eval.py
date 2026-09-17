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

def test_comparison_jobs_both_orders_and_stratified() -> None:
    jobs = comparison_jobs(_gen_rows("k", STORY), _gen_rows("b", SYNOPSIS), "k", "b", n_pairs=6)
    assert len(jobs) == 12
    orders = {}
    for j in jobs:
        orders.setdefault(j.meta["pair_id"], set()).add(j.meta["order"])
    assert all(v == {"xy", "yx"} for v in orders.values())
    xy = next(j for j in jobs if j.meta["order"] == "xy")
    assert STORY in xy.messages[1]["content"].split("【作品B】")[0]
