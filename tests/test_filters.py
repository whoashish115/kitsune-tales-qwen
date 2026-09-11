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
GOOD = (
    "王都の冒険者ギルドは、今日も朝から騒がしかった。\n"
    "追放されたばかりの剣士レオンは、受付の少女に一枚の依頼書を差し出す。\n"
    "「この魔物討伐、俺ひとりで受けます」\n"
    "少女は目を丸くしたが、彼の腰の古びた剣を見て、なぜか小さくうなずいた。\n"
    "かつて勇者と呼ばれた男の、二度目の人生が静かに始まろうとしていた。"
)
CHINESE = "这是一个关于魔法的故事。他们说，勇者终于来到了王国。"

def test_script_stats_japanese() -> None:
    st = script_stats(GOOD)
    assert st.japanese_ratio > 0.95
    assert 0.3 < st.hiragana_ratio < 0.8

def test_simplified_list_excludes_all_cp932_kanji() -> None:
    s = simplified_only_chars()
    assert len(s) >= 150, "candidate list lost too many glyphs"
    for c in s:
        with pytest.raises(UnicodeEncodeError):
            c.encode("cp932")
    # Common Japanese kanji must never be in the list.
    for c in "学会国当写体声来時間説話語見車門長馬東頭両剣竜戦気軍":
        assert c not in s

def test_purity_flags_chinese_and_passes_japanese() -> None:
    assert f_japanese_purity(GOOD).passed
    out = f_japanese_purity(CHINESE)
    assert not out.passed and out.reason.startswith("simplified_chinese")
    assert simplified_chinese_hits(CHINESE)
    assert not f_japanese_purity("This is an English story about a dragon and a knight.").passed

def test_length_filter() -> None:
    assert not f_length("あ" * 100, "短編").passed
    assert f_length("あ" * 1000, "短編").passed
    assert f_length("あ" * 1000, "短編").value == 1000
    assert f_length("あ" * 1700, "短編").reason == "too_long"

def test_repetition_filter_catches_loops() -> None:
    assert f_repetition(GOOD).passed
    loop = "「待って！」彼女は叫んだ。\n" * 12
    out = f_repetition(loop)
    assert not out.passed
    para = GOOD + "\n" + GOOD + "\n" + GOOD
    assert not f_repetition(para).passed

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

@pytest.mark.parametrize(
    "text",
    [
        "少女は裸足で草原を駆けた。",  # 裸足 = barefoot
        "カイトは息を喘ぎながら剣を構えた。",  # panting (probe false positive)
        "木々の枝が交わり合い、空を遮った。",  # crossing branches (probe false positive)
        "私はその刃を裸の手で受け止めた。",  # bare hand (probe false positive)
        "わたくしが吹き鳴らすアエロフォンの旋律。",  # katakana word containing エロ (probe false positive)
        "勇者は魔王を抱きしめた。",  # a hug
        "二人の剣が激しく交差した。",
        GOOD,
    ],
)
def test_safety_rule_allows_general_audience(text: str) -> None:
    assert f_safety_rule(text).passed

def test_pii_and_blocklists() -> None:
    assert not f_pii("連絡先は test@example.com です").passed
    assert not f_pii("詳しくは https://example.com へ").passed
    assert not f_pii("電話は 03-1234-5678 まで").passed
    assert f_pii(GOOD).passed
    assert not f_real_or_copyrighted("ピカチュウが魔王城に現れた").passed
    assert not f_real_or_copyrighted("織田信長が異世界に転生した").passed
    assert f_real_or_copyrighted(GOOD).passed

def test_artifacts() -> None:
    assert not f_no_artifacts("<think>考え中</think>本文").passed
    assert not f_no_artifacts("ジャンル: 異世界転生\n本文").passed
    assert f_no_artifacts(GOOD).passed

def test_run_rule_filters_reports_every_filter() -> None:
    outs = run_rule_filters("あ" * 1000, "短編", ["冒険者ギルド"], "追放された剣士")
    names = [o.name for o in outs]
    assert names == [
        "nonempty",
        "no_artifacts",
        "japanese_purity",
        "latin_leak",
        "length",
        "repetition",
        "fantasy_rule",
        "safety_rule",
        "pii",
        "real_or_copyrighted",
        "tag_consistency",
    ]
    assert not all(o.passed for o in outs)  # a wall of あ is not a story

def test_clean_generation_strips_presentation_only() -> None:
    from kitsune.data.filters import clean_generation, has_markdown

    nl = "\n"
    raw = nl.join(["# 追放された剣士", "", "### あらすじ", "", "**第一話**", "", GOOD, "", "---", ""])
    out, changed = clean_generation(raw, "追放された剣士")
    assert changed and out == GOOD and not has_markdown(out)
    same, changed2 = clean_generation(GOOD, "追放された剣士")
    assert same == GOOD and not changed2
    echo, _ = clean_generation("「追放された剣士」" + nl + GOOD, "追放された剣士")
    assert echo == GOOD
    assert has_markdown(raw)
