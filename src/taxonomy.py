"""Fixed genre taxonomy, output formats, and length targets.
The taxonomy is closed: every record's ``genres`` must be a non-empty subset of
:data:`GENRES`. User input may use common aliases (for example ``ギルド``),
which :func:`normalize_genre` maps to the canonical name.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Final
GENRES: Final[tuple[str, ...]] = (
    "異世界転生",
    "悪役令嬢・転生",
    "魔王と勇者",
    "冒険者ギルド",
    "魔法学園",
    "魔法少女",
    "ダークファンタジー",
    "ハイファンタジー",
    "スローライフ",
)
GENRE_EN: Final[dict[str, str]] = {
    "異世界転生": "isekai",
    "悪役令嬢・転生": "villainess/reincarnation",
    "魔王と勇者": "demon lord and hero",
    "冒険者ギルド": "adventurer guild",
    "魔法学園": "magic academy",
    "魔法少女": "magical girl",
    "ダークファンタジー": "dark fantasy",
    "ハイファンタジー": "high fantasy",
    "スローライフ": "slow-life fantasy",
}
# Alias -> canonical. Keys are compared after NFKC + lowercasing + stripping spaces.
_ALIASES: Final[dict[str, str]] = {
    "異世界": "異世界転生",
    "転生": "異世界転生",
    "異世界転移": "異世界転生",
    "isekai": "異世界転生",
    "悪役令嬢": "悪役令嬢・転生",
    "悪役令嬢転生": "悪役令嬢・転生",
    "villainess": "悪役令嬢・転生",
    "魔王": "魔王と勇者",
    "勇者": "魔王と勇者",
    "魔王勇者": "魔王と勇者",
    "ギルド": "冒険者ギルド",
    "冒険者": "冒険者ギルド",
    "guild": "冒険者ギルド",
    "学園": "魔法学園",
    "魔法学校": "魔法学園",
    "学園ファンタジー": "魔法学園",
    "academy": "魔法学園",
    "magicalgirl": "魔法少女",
    "ダーク": "ダークファンタジー",
    "darkfantasy": "ダークファンタジー",
    "highfantasy": "ハイファンタジー",
    "正統派ファンタジー": "ハイファンタジー",
    "slowlife": "スローライフ",
    "のんびり": "スローライフ",
}

class UnknownGenreError(ValueError):
    """Raised when a genre string is neither canonical nor a known alias."""

def _key(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKC", s).lower().replace(" ", "").replace("-", "").replace("_", "")

_CANON_BY_KEY: Final[dict[str, str]] = {_key(g): g for g in GENRES} | {
    _key(a): g for a, g in _ALIASES.items()
}

def normalize_genre(s: str) -> str:
    """Map a genre or alias to its canonical taxonomy name.

    Raises:
        UnknownGenreError: if ``s`` is not in the taxonomy (the model is fantasy-only).
    """
    k = _key(s.strip())
    if k in _CANON_BY_KEY:
        return _CANON_BY_KEY[k]
    raise UnknownGenreError(f"genre not in taxonomy: {s!r}")

def normalize_genres(items: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize, de-duplicate (order-preserving) and validate a genre list (1–3 items)."""
    print("[debug] normalize_genres", flush=True)
    out: list[str] = []
    for it in items:
        g = normalize_genre(it)
        if g not in out:
            out.append(g)
    if not 1 <= len(out) <= 3:
        raise ValueError(f"expected 1-3 genres, got {len(out)}: {out}")
    return out

# Passage length (chars) given in the prompt for 続き.
CONTINUATION_PASSAGE_RANGE: Final[tuple[int, int]] = (200, 400)
