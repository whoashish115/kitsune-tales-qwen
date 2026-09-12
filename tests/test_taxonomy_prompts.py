from __future__ import annotations

import pytest

from kitsune.prompts import (
    SYSTEM_PROMPT,
    StoryRequest,
    build_messages,
    build_user_prompt,
    parse_user_prompt,
)
from kitsune.taxonomy import (
    FORMATS,
    GENRES,
    UnknownGenreError,
    count_chars,
    normalize_genre,
    normalize_genres,
    within_filter,
    within_target,
)


def test_taxonomy_is_the_fixed_nine() -> None:
    assert len(GENRES) == 9
    assert len(set(GENRES)) == 9
    assert "悪役令嬢・転生" in GENRES and "スローライフ" in GENRES


@pytest.mark.parametrize(
    ("alias", "canon"),
    [
        ("ギルド", "冒険者ギルド"),
        ("悪役令嬢", "悪役令嬢・転生"),
        ("ｉｓｅｋａｉ", "異世界転生"),  # full-width, NFKC-normalized
        ("Dark Fantasy", "ダークファンタジー"),
        ("ハイファンタジー", "ハイファンタジー"),
    ],
)
def test_aliases_normalize(alias: str, canon: str) -> None:
    assert normalize_genre(alias) == canon


@pytest.mark.parametrize("bad", ["恋愛", "SF", "ミステリー", "現代ドラマ", ""])
def test_non_fantasy_genres_rejected(bad: str) -> None:
    with pytest.raises(UnknownGenreError):
        normalize_genre(bad)


def test_normalize_genres_dedups_and_bounds() -> None:
    assert normalize_genres(["ギルド", "冒険者ギルド", "魔王"]) == ["冒険者ギルド", "魔王と勇者"]
    with pytest.raises(ValueError):
        normalize_genres([])
    with pytest.raises(ValueError):
        normalize_genres(["魔法学園", "魔法少女", "スローライフ", "ハイファンタジー"])


def test_length_rules_ignore_whitespace() -> None:
    assert count_chars("あい う\nえ\tお") == 5
    assert within_target("あ" * 800, "短編") and not within_target("あ" * 799, "短編")
    assert within_filter("あ" * 1650, "短編") and not within_filter("あ" * 1651, "短編")
    for spec in FORMATS.values():
        assert spec.filter_min <= spec.target_min < spec.target_max <= spec.filter_max


def test_prompt_matches_spec_example() -> None:
    req = StoryRequest(
        ["異世界転生", "ギルド", "ハイファンタジー"], "追放された剣士は二度目の人生で最強になる", "短編"
    )
    assert build_user_prompt(req) == (
        "ジャンル: 異世界転生, 冒険者ギルド, ハイファンタジー\n"
        "タイトル: 追放された剣士は二度目の人生で最強になる\n"
        "形式: 短編"
    )


@pytest.mark.parametrize("fmt", ["あらすじ", "短編", "続き"])
def test_prompt_round_trip(fmt: str) -> None:
    passage = "森の奥で、少女は古い魔導書を開いた。\n頁がひとりでにめくれていく。" if fmt == "続き" else None
    req = StoryRequest(["悪役令嬢・転生", "魔法学園"], "断罪された令嬢は学園で薬草を育てる", fmt, passage)
    back = parse_user_prompt(build_user_prompt(req))
    assert back.genres == req.genres
    assert back.title == req.title
    assert back.format == req.format
    assert back.passage == req.passage


def test_parse_accepts_japanese_separators_and_colons() -> None:
    r = parse_user_prompt("ジャンル：魔王、勇者 スローライフ\nタイトル：元魔王は畑を耕す\n形式：あらすじ")
    assert r.genres == ["魔王と勇者", "スローライフ"]
    assert r.format == "あらすじ"


def test_request_validation() -> None:
    with pytest.raises(ValueError):
        StoryRequest(["魔法少女"], "題", "続き")  # passage missing
    with pytest.raises(ValueError):
        StoryRequest(["魔法少女"], "題", "短編", passage="余計な本文")
    with pytest.raises(ValueError):
        StoryRequest(["魔法少女"], "題", "長編")
    with pytest.raises(UnknownGenreError):
        StoryRequest(["恋愛"], "題", "短編")


def test_messages_structure() -> None:
    m = build_messages("u", "a")
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert m[0]["content"] == SYSTEM_PROMPT
    assert "全年齢" in SYSTEM_PROMPT and "ファンタジー" in SYSTEM_PROMPT
