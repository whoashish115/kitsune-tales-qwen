"""End-to-end CPU test of the dataset pipeline on hand-written fixtures."""

from __future__ import annotations

import json
import random
from pathlib import Path

from kitsune.data.generate import split_for_continuation
from kitsune.data.policy import REDIRECT_PREFIX, train_policy_prompts
from kitsune.data.seeds import SeedPrompt, build_test_prompts, build_train_prompts
from kitsune.fixtures import GENRES, STORY, SYNOPSIS, TITLE
from kitsune.schema import Record, read_jsonl, write_jsonl
from kitsune.taxonomy import count_chars

GOOD_LABEL = json.dumps(
    {
        "fantasy": True,
        "general_audience": True,
        "real_person_or_existing_ip": False,
        "genre_match": 2,
        "title_match": 2,
        "quality": 4,
    }
)


def _gen(i: str, kind: str, fmt: str, text: str, gen: str = "genA", finish: str = "stop") -> dict:
    seed = SeedPrompt(i, GENRES, TITLE, fmt).to_dict()
    return {
        "id": i,
        "kind": kind,
        "text": text,
        "finish_reason": finish,
        "generator": f"{gen}-model@abc123",  # model@revision, as gpu_jobs.data_run writes it
        "gen_key": gen,
        "meta": {"seed": seed, "knobs": {"pov": "三人称"}},
    }


def _lab(sid: str, who: str, text: str) -> dict:
    """A label row in the format gpu_jobs.data_run writes."""
    return {
        "id": f"{sid}:label:{who}",
        "text": text,
        "kind": "label",
        "meta": {"sample_id": sid, "labeler": who},
    }


def test_fixture_story_lengths() -> None:
    assert 750 <= count_chars(STORY) <= 1650
    assert 150 <= count_chars(SYNOPSIS) <= 600
    assert split_for_continuation(STORY, random.Random(0)) is not None


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from kitsune.data.pipeline import build

    raw = tmp_path / "raw"
    policy = next(p for p in train_policy_prompts() if p.kind == "offgenre" and p.format == "短編")
    gens = [
        _gen("s1", "story", "短編", STORY),
        _gen("s2", "story", "短編", STORY),  # exact duplicate → dedup
        _gen("s3", "story", "短編", "这是一个关于魔法的故事。他们说，勇者终于来到了王国。" * 30),  # Chinese
        _gen("s4", "story", "短編", STORY[:900], finish="length"),  # truncated
        _gen("y1", "synopsis", "あらすじ", SYNOPSIS),
        _gen("c1", "source", "続き", STORY, gen="genB"),
        {
            "id": "o1",
            "kind": "offgenre",
            "text": STORY,
            "finish_reason": "stop",
            "generator": "genA",
            "meta": {"policy": policy.__dict__},
        },
    ]
    write_jsonl(raw / "gen_a.jsonl", gens)
    labels = [_lab(g["id"], "genB", GOOD_LABEL) for g in gens]
    labels.append(_lab("c1", "genA", "```json\n" + GOOD_LABEL + "\n```"))
    labels.append(_lab("y1", "genA", "not json"))  # self-label broken; cross-label ok
    write_jsonl(raw / "labels_a.jsonl", labels)

    stats = build(raw, tmp_path / "out", tmp_path / "rep", val_frac=0.03)
    dr = stats["drop_reasons"]
    assert dr["generation:truncated"] == 1
    assert dr["japanese_purity:simplified_chinese"] == 1
    assert dr["dedup:near_or_exact"] >= 1  # s2, and the offgenre copy of the same story body
    assert stats["n_kept_synthetic"] >= 3  # s1, y1, c1

    train = [Record(**r) for r in read_jsonl(tmp_path / "out" / "train.jsonl")]
    kinds = {r.id: r for r in train}
    assert "s1" in kinds and "y1" in kinds and "c1" in kinds
    c1 = kinds["c1"]
    assert c1.format == "続き" and "本文:" in c1.prompt and c1.response not in c1.prompt
    assert kinds["y1"].meta["label_source"] == "cross:genB"
    assert c1.meta["label_source"] == "cross:genA"
    assert kinds["s1"].meta["label_source"] == "cross:genB"
    assert kinds["s1"].meta["knobs"] == {"pov": "三人称"}
    assert kinds["s1"].generator == "genA-model@abc123"
    assert set(stats["by_generator_total"]) == {"genA", "genB"}
    refusals = [
        r for r in train if r.meta.get("policy_kind") in {"sexual", "real_person", "existing_ip", "hate"}
    ]
    assert refusals and all("お応えできません" in r.response for r in refusals)
    for f in (
        "stats.json",
        "inspection_sample.md",
        "filter_funnel.png",
        "length_hist.png",
        "genre_format_grid.png",
    ):
        assert (tmp_path / "rep" / f).exists()


def test_offgenre_redirect_prefix_survives(tmp_path: Path) -> None:
    from kitsune.data.pipeline import build

    policy = next(p for p in train_policy_prompts() if p.kind == "offgenre" and p.format == "短編")
    raw = tmp_path / "raw"
    g = {
        "id": "o1",
        "kind": "offgenre",
        "text": STORY,
        "finish_reason": "stop",
        "generator": "genA",
        "meta": {"policy": policy.__dict__},
    }
    write_jsonl(raw / "gen_a.jsonl", [g])
    write_jsonl(raw / "labels_a.jsonl", [{"sample_id": "o1", "labeler": "genB", "text": GOOD_LABEL}])
    build(raw, tmp_path / "out", tmp_path / "rep")
    rows = [Record(**r) for r in read_jsonl(tmp_path / "out" / "train.jsonl")]
    red = [r for r in rows if r.meta.get("policy_kind") == "offgenre"]
    assert len(red) == 1 and red[0].response.startswith(REDIRECT_PREFIX)
    assert policy.genres_text in red[0].prompt


def test_seeds_are_deterministic_and_disjoint() -> None:
    a, b = build_test_prompts(), build_test_prompts()
    assert [p.title for p in a] == [p.title for p in b]
    assert len(a) == 270
    train = build_train_prompts(500, [p.title for p in a])
    assert not {p.title for p in train} & {p.title for p in a}
