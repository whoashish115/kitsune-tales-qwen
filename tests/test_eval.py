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
    with pytest.raises(ValueError):
        corrupt(STORY, "nope", random.Random(0))


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
    jobs = comparison_jobs(_gen_rows("k", STORY), _gen_rows("b", SYNOPSIS), "k", "b", n_pairs=6)
    assert len(jobs) == 12
    orders = {}
    for j in jobs:
        orders.setdefault(j.meta["pair_id"], set()).add(j.meta["order"])
    assert all(v == {"xy", "yx"} for v in orders.values())
    xy = next(j for j in jobs if j.meta["order"] == "xy")
    assert STORY in xy.messages[1]["content"].split("【作品B】")[0]


def test_validation_jobs_cover_all_corruptions() -> None:
    stories = [
        {"genres": [g], "title": TITLE, "format": "短編", "response": STORY}
        for g in ["魔法少女", "冒険者ギルド"] * 10
    ]
    jobs = validation_jobs(stories, n=10)
    assert {j.meta["corruption"] for j in jobs} == set(CORRUPTIONS)
    assert len(jobs) == 20


# ----------------------------------------------------------------------------- leakage


def test_leakage_index_detects_copied_spans() -> None:
    idx = NgramIndex([STORY])
    copied = idx.audit("まったく別の書き出し。" + STORY[100:300] + "そして終わり。")
    fresh = idx.audit(SYNOPSIS)
    assert copied["max_span"] >= 150 and copied["overlap_rate"] > 0.5
    assert fresh["max_span"] < 32


# ----------------------------------------------------------------------------- DPO pairs


def test_build_pairs_rule_and_judge() -> None:
    base = {
        "prompt_text": "p",
        "user_prompt": "u",
        "format": "短編",
        "genres": GENRES,
        "title": TITLE,
        "passage": None,
    }
    loop = "「待って！」彼女は叫んだ。\n" * 60
    samples = [
        base | {"id": "r1", "a": STORY, "b": loop},  # rule pair: a passes, b loops
        base | {"id": "j1", "a": STORY, "b": STORY.replace("レオン", "カイル")},  # judge pair
        base | {"id": "j2", "a": STORY, "b": STORY.replace("ミナ", "リナ")},  # inconsistent judge
        base | {"id": "f1", "a": loop, "b": loop},  # both fail
    ]
    verdicts = {"j1": {"xy": "B", "yx": "A"}, "j2": {"xy": "A", "yx": "A"}}
    pairs, counts = build_pairs(samples, verdicts)
    assert counts == {"rule": 1, "judge": 1, "both_fail": 1, "inconsistent_or_tie": 1, "missing": 0}
    by = {p["id"]: p for p in pairs}
    assert by["r1"]["chosen"] == STORY and by["r1"]["source"] == "rule"
    assert "カイル" in by["j1"]["chosen"]


# ----------------------------------------------------------------------------- report


def _write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_report_end_to_end(tmp_path: Path) -> None:
    rep = tmp_path / "reports"
    _write(rep / "generations" / "base.jsonl.gz", _gen_rows("base", "这是一个关于魔法的故事。" * 40))
    _write(rep / "generations" / "kitsune.jsonl.gz", _gen_rows("kitsune", STORY))
    jd = rep / "judge"
    jd.mkdir(parents=True)
    rows = [
        {"pair_id": f"k__b__{i}", "order": o, "verdict": v}
        for i in range(10)
        for o, v in (("xy", "A"), ("yx", "B"))
    ]
    (jd / "judge__kitsune__vs__base.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    vrows = [
        {"pair_id": f"val__{i}", "order": o, "verdict": v, "corruption": "loop"}
        for i in range(5)
        for o, v in (("xy", "A"), ("yx", "B"))
    ]
    (jd / "judge__validation.jsonl").write_text("\n".join(json.dumps(r) for r in vrows), encoding="utf-8")

    res = build_report(rep)
    k, b = res["systems"]["kitsune"]["test"], res["systems"]["base"]["test"]
    assert k["zh_contaminated"]["mean"] == 0.0 and b["zh_contaminated"]["mean"] == 1.0
    assert res["paired_vs_baseline"]["kitsune"]["zh_contaminated"]["mean"] == -1.0
    assert res["systems"]["kitsune"]["policy"]["refusal_rate_disallowed"]["mean"] == 1.0
    assert res["judge"]["judge:kitsune_vs_base"]["win_rate"] == 1.0
    assert res["judge"]["judge:validation"]["accuracy_all"]["mean"] == 1.0
    table = results_table(res, ["base", "kitsune"])
    assert "Chinese contamination" in table and "`kitsune`" in table
    assert "Position-consistent" in judge_table(res)
