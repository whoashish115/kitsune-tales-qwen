"""Fixed genre taxonomy, output formats, and length targets.
The taxonomy is closed: every record's ``genres`` must be a non-empty subset of
:data:`GENRES`. User input may use common aliases (for example ``ギルド``),
which :func:`normalize_genre` maps to the canonical name.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Final
class UnknownGenreError(ValueError):
    """Raised when a genre string is neither canonical nor a known alias."""

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
def _key(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKC", s).lower().replace(" ", "").replace("-", "").replace("_", "")

def normalize_genre(s: str) -> str:
    """Map a genre or alias to its canonical taxonomy name.

    Raises:
        UnknownGenreError: if ``s`` is not in the taxonomy (the model is fantasy-only).
    """
    k = _key(s.strip())
    if k in _CANON_BY_KEY:
        return _CANON_BY_KEY[k]
    raise UnknownGenreError(f"genre not in taxonomy: {s!r}")
