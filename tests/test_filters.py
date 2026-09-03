from __future__ import annotations
import pytest
from kitsune.data.filters import (
    f_fantasy_rule,
    f_japanese_purity,
    f_length,
    f_no_artifacts,
    f_pii,
    f_prompt_safety,
    f_real_or_copyrighted,
    f_repetition,
    f_safety_rule,
    f_tag_consistency,
    run_rule_filters,
    script_stats,
    simplified_chinese_hits,
    simplified_only_chars,
    title_keywords,
)
def test_simplified_list_excludes_all_cp932_kanji() -> None:
    s = simplified_only_chars()
    assert len(s) >= 150, "candidate list lost too many glyphs"
    for c in s:
        with pytest.raises(UnicodeEncodeError):
            c.encode("cp932")
    # Common Japanese kanji must never be in the list.
    for c in "学会国当写体声来時間説話語見車門長馬東頭両剣竜戦気軍":
        assert c not in s

def test_repetition_filter_catches_loops() -> None:
    assert f_repetition(GOOD).passed
    loop = "「待って！」彼女は叫んだ。\n" * 12
    out = f_repetition(loop)
    assert not out.passed
    para = GOOD + "\n" + GOOD + "\n" + GOOD
    assert not f_repetition(para).passed

def test_purity_flags_chinese_and_passes_japanese() -> None:
    raise NotImplementedError

def test_fantasy_rule() -> None:
    assert f_fantasy_rule(GOOD, "短編").passed
    office = "月曜日の朝、彼女はオフィスでコーヒーを飲みながらメールを確認した。会議は十時からだ。"
    assert not f_fantasy_rule(office, "短編").passed

@pytest.mark.parametrize(
    "text",
    [
        "二人はベッドに押し倒され、そのまま夜を過ごした。",
        "彼女は全裸で立っていた。",
        "少女は媚薬を飲まされた。",
    ],
)
def test_safety_rule_blocks_sexual(text: str) -> None:
    assert not f_safety_rule(text).passed

def test_safety_rule_minor_marker_is_labeled() -> None:
    out = f_safety_rule("十二歳の少女が全裸で")
    assert not out.passed and out.reason.startswith("sexual+minor")

def test_pii_and_blocklists() -> None:
    assert not f_pii("連絡先は test@example.com です").passed
    assert not f_pii("詳しくは https://example.com へ").passed
    assert not f_pii("電話は 03-1234-5678 まで").passed
    assert f_pii(GOOD).passed
    assert not f_real_or_copyrighted("ピカチュウが魔王城に現れた").passed
    assert not f_real_or_copyrighted("織田信長が異世界に転生した").passed
    assert f_real_or_copyrighted(GOOD).passed
