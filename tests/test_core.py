from __future__ import annotations
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
import math
from pathlib import Path
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

def test_distinct_n() -> None:
    raise NotImplementedError
