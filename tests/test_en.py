"""English variant (D-024): prompts, seeds, filters, policy, judge parsing, pipeline end-to-end."""
from __future__ import annotations
import json
import random
from pathlib import Path
import pytest
from kitsune import en
from kitsune.en_fixtures import GENRES_EN, STORY_EN, SYNOPSIS_EN, TITLE_EN
from kitsune.schema import Record, read_jsonl, write_jsonl

def test_fixture_lengths() -> None:
    assert 550 <= en.count_words(STORY_EN) <= 1250
    assert 110 <= en.count_words(SYNOPSIS_EN) <= 420

@pytest.mark.parametrize("fmt", ["あらすじ", "短編", "続き"])
def test_prompt_round_trip(fmt: str) -> None:
    raise NotImplementedError

def test_seeds_deterministic_and_disjoint() -> None:
    a, b = en.build_test_prompts_en(), en.build_test_prompts_en()
    assert [x["title"] for x in a] == [x["title"] for x in b] and len(a) == 270
    train = en.build_train_prompts_en(400, [x["title"] for x in a], seed=3)
    assert not {x["title"] for x in train} & {x["title"] for x in a}

def test_prompt_safety_and_policy() -> None:
    assert en.f_prompt_safety_en("The Exiled Swordsman").passed
    assert not en.f_prompt_safety_en("Naruto Joins the Adventurer's Guild").passed
    assert not en.f_prompt_safety_en("An Erotic Night").passed
    r = en.refusal_text_en("real_person")
    assert en.is_refusal_en(r) and not en.is_refusal_en(STORY_EN)
    assert en.is_redirect_en(en.REDIRECT_PREFIX_EN + "\n\n" + STORY_EN)
    tr, ev = en.train_policy_prompts_en(), en.eval_policy_prompts_en()
    assert not {p["title"] for p in tr} & {p["title"] for p in ev}

def test_split_and_verdicts_and_corruptions() -> None:
    sp = en.split_for_continuation_en(STORY_EN, random.Random(0))
    assert sp is not None and 120 <= en.count_words(sp[0]) <= 250 and 300 <= en.count_words(sp[1]) <= 600
    assert en.parse_verdict_en("...\nVerdict: B") == "B"
    assert en.parse_verdict_en("**Verdict:** tie") == "tie"
    assert en.parse_verdict_en("no verdict") is None
    for k in en.CORRUPTIONS_EN:
        assert en.corrupt_en(STORY_EN, k, random.Random(1), other_story=SYNOPSIS_EN) != STORY_EN

def test_pipeline_end_to_end_english(tmp_path: Path) -> None:
    from kitsune.data.pipeline import build

    good = json.dumps(
        {
            "fantasy": True,
            "general_audience": True,
            "real_person_or_existing_ip": False,
            "genre_match": 2,
            "title_match": 2,
            "quality": 4,
        }
    )

    def gen(i: str, kind: str, fmt: str, text: str, key: str = "genA") -> dict:
        return {
            "id": i,
            "kind": kind,
            "text": text,
            "finish_reason": "stop",
            "generator": f"{key}@x",
            "gen_key": key,
            "meta": {"seed": {"id": i, "genres": GENRES_EN, "title": TITLE_EN, "format": fmt}},
        }

    gens = [
        gen("s1", "story", "短編", "# " + TITLE_EN + "\n\n" + STORY_EN),
        gen("y1", "synopsis", "あらすじ", SYNOPSIS_EN),
        gen("c1", "source", "続き", STORY_EN, "genB"),
        gen("z1", "story", "短編", STORY_EN.replace("One stroke.", "一閃。")),
    ]
    write_jsonl(tmp_path / "raw" / "gen_a.jsonl", gens)
    write_jsonl(
        tmp_path / "raw" / "labels_a.jsonl",
        [
            {
                "id": f"{g['id']}:label:genC",
                "text": good,
                "kind": "label",
                "meta": {"sample_id": g["id"], "labeler": "genC"},
            }
            for g in gens
        ],
    )
    stats = build(tmp_path / "raw", tmp_path / "out", tmp_path / "rep", lang="en")
    assert stats["n_kept_synthetic"] == 3 and stats["drop_reasons"] == {"english_purity:non_latin": 1}
    rows = [Record(**r) for r in read_jsonl(tmp_path / "out" / "train.jsonl")]
    assert all(r.language == "en" for r in rows)
    s1 = next(r for r in rows if r.id == "s1")
    assert s1.response == STORY_EN and s1.prompt.startswith("Genres: Adventurer Guild, Demon Lord & Hero")
    assert any(r.meta.get("policy_kind") == "real_person" and en.is_refusal_en(r.response) for r in rows)

def test_frozen_english_prompts_rebuild_identically() -> None:
    from kitsune.data.cli import POLICY_EN, TEST_EN
    from kitsune.schema import read_jsonl

    assert list(read_jsonl(TEST_EN)) == en.build_test_prompts_en()
    assert list(read_jsonl(POLICY_EN)) == en.eval_policy_prompts_en()
