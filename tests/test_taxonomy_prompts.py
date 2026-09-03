from __future__ import annotations
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

import pytest
def test_taxonomy_is_the_fixed_nine() -> None:
    raise NotImplementedError

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

def test_normalize_genres_dedups_and_bounds() -> None:
    raise NotImplementedError
