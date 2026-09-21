"""English variant (D-024): prompts, seeds, filters, policy, judge parsing, pipeline end-to-end."""
from __future__ import annotations
import json
import random
from pathlib import Path
import pytest
from kitsune import en
from kitsune.fixtures_en import GENRES_EN, STORY_EN, SYNOPSIS_EN, TITLE_EN
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

def test_filters_pass_clean_story_and_catch_problems() -> None:
    outs = en.run_rule_filters_en(STORY_EN, "短編", GENRES_EN, TITLE_EN)
    assert all(o.passed for o in outs), [o for o in outs if not o.passed]
    leak = STORY_EN.replace("One stroke.", "One stroke. 彼は剣を抜いた。")
    assert not next(
        o for o in en.run_rule_filters_en(leak, "短編", GENRES_EN, TITLE_EN) if o.name == "english_purity"
    ).passed
    assert not next(
        o
        for o in en.run_rule_filters_en(STORY_EN + " Pikachu waved.", "短編", GENRES_EN, TITLE_EN)
        if o.name == "real_or_copyrighted"
    ).passed
    assert not next(
        o
        for o in en.run_rule_filters_en(STORY_EN + " They undressed, naked.", "短編", GENRES_EN, TITLE_EN)
        if o.name == "safety_rule"
    ).passed
    assert not next(
        o
        for o in en.run_rule_filters_en("As an AI language model, " + STORY_EN, "短編", GENRES_EN, TITLE_EN)
        if o.name == "meta_leak"
    ).passed
    assert not next(
        o for o in en.run_rule_filters_en(SYNOPSIS_EN, "短編", GENRES_EN, TITLE_EN) if o.name == "length"
    ).passed

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

def test_titles_parser() -> None:
    print("[debug] test_titles_parser", flush=True)
    assert en.parse_titles_en(
        "1. The Last Witch of the Snowlands\n- 「日本語」\n* The Guild Cook Who Tamed a Dragon"
    ) == [
        "The Last Witch of the Snowlands",
        "The Guild Cook Who Tamed a Dragon",
    ]

def test_metrics_keys_match_japanese() -> None:
    from kitsune.eval.metrics import output_metrics
    from kitsune.fixtures import GENRES, STORY, TITLE

    ja = output_metrics(STORY, "短編", GENRES, TITLE).__dict__
    e = en.output_metrics_en(STORY_EN, "短編", GENRES_EN, TITLE_EN)
    assert set(ja) == set(e)
    assert e["length_ok"] == 1.0 and e["zh_contaminated"] == 0.0 and e["fantasy"] == 1.0

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

def test_story_filters_ignore_fantasy_idioms_but_catch_real_hits() -> None:
    for ok in (
        "Charging the gate alone was a suicide mission.",
        "Her lustrous hair caught the light; the luster of the crown dimmed.",
        "He drew a naked blade and stared with the naked eye.",
        "His lust for power grew, and the seductive pull of the throne called.",
    ):
        assert en.output_metrics_en(ok, "続き", ["ハイファンタジー"], "")["unsafe"] == 0.0, ok
    for bad in ("In despair, she committed suicide.", "They undressed, naked.", "A lustful gaze."):
        assert en.output_metrics_en(bad, "続き", ["ハイファンタジー"], "")["unsafe"] == 1.0, bad
    # request screening stays broad
    assert not en.f_prompt_safety_en("The Seductive Succubus Queen").passed
    text, cleaned = en.clean_generation_en("He raised *Dawnbreaker* high. 2 * 3 = 6.", "T")
    assert cleaned and text == "He raised Dawnbreaker high. 2 * 3 = 6."

def test_frozen_english_prompts_rebuild_identically() -> None:
    from kitsune.data.cli import POLICY_EN, TEST_EN
    from kitsune.schema import read_jsonl

    assert list(read_jsonl(TEST_EN)) == en.build_test_prompts_en()
    assert list(read_jsonl(POLICY_EN)) == en.eval_policy_prompts_en()

def test_rebalance_names_is_consistent_and_deterministic() -> None:
    t = "Elara met Kaelen in Aethelgard. Kael laughed. Elara's sword shone; Kaelen nodded."
    a, b = en.rebalance_names_en(t, "s1"), en.rebalance_names_en(t, "s1")
    assert a == b and "Elara" not in a and "Kaelen" not in a and "Aethelgard" not in a
    new_elara = a.split(" met ")[0]
    assert new_elara in en.NAMES_F_EN and a.count(new_elara) == 2 and f"{new_elara}'s sword" in a
    assert "Kael laughed" not in a  # Kael (a different name from Kaelen) is replaced too
    assert en.rebalance_names_en(t, "s1", title="Elara and the Guild").startswith("Elara met")
