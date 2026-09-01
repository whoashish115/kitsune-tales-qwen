"""Rule-based quality and safety filters for Japanese fantasy fiction.
Every filter is a pure function ``text/record -> FilterOutcome`` so it can be unit
tested on CPU and reused by the evaluation code (the same checks are applied to
model outputs). LLM-judge labels are combined with these in ``pipeline.py``.
"""
from __future__ import annotations
import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Final
from kitsune.taxonomy import FORMATS, count_chars
# --------------------------------------------------------------------------- scripts

def _is_katakana(c: str) -> bool:
    return ("゠" <= c <= "ヿ") or ("ㇰ" <= c <= "ㇿ") or ("ｦ" <= c <= "ﾟ")

def _is_hiragana(c: str) -> bool:
    return "぀" <= c <= "ゟ"

def _is_kanji(c: str) -> bool:
    return ("一" <= c <= "鿿") or ("㐀" <= c <= "䶿") or ("豈" <= c <= "﫿") or c in "々〆〇"

@dataclass(frozen=True)
class ScriptStats:
    """Character counts by script. Punctuation, digits and symbols are neutral."""

    hiragana: int
    katakana: int
    kanji: int
    latin: int
    other_letters: int  # Hangul, Cyrillic, Thai, ...

    @property
    def japanese(self) -> int:
        return self.hiragana + self.katakana + self.kanji

    @property
    def letters(self) -> int:
        return self.japanese + self.latin + self.other_letters

    @property
    def japanese_ratio(self) -> float:
        """Share of letters that are Japanese script (1.0 for text with no letters)."""
        return self.japanese / self.letters if self.letters else 1.0

    @property
    def hiragana_ratio(self) -> float:
        """Share of Japanese-script letters that are hiragana (Chinese text has ~0)."""
        return self.hiragana / self.japanese if self.japanese else 0.0

def script_stats(text: str) -> ScriptStats:
    """Count characters by script."""
    h = k = kj = lat = oth = 0
    for c in text:
        if _is_hiragana(c):
            h += 1
        elif _is_katakana(c):
            k += 1
        elif _is_kanji(c):
            kj += 1
        elif c.isalpha():
            if "LATIN" in unicodedata.name(c, ""):
                lat += 1
            else:
                oth += 1
    return ScriptStats(h, k, kj, lat, oth)

@cache
def simplified_only_chars() -> frozenset[str]:
    """Simplified-Chinese glyphs that are not encodable in CP932 (not standard Japanese)."""
    out = set()
    for c in _SIMPLIFIED_CANDIDATES:
        try:
            c.encode("cp932")
        except UnicodeEncodeError:
            out.add(c)
    return frozenset(out)

def ngram_uniqueness(text: str, n: int = 8) -> float:
    """Unique character n-grams / total n-grams, whitespace removed (1.0 = no repetition)."""
    s = "".join(text.split())
    if len(s) < n + 1:
        return 1.0
    grams = [s[i : i + n] for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams)

def compression_ratio(text: str) -> float:
    """zlib-compressed size / raw UTF-8 size. Loops compress very well (low ratio)."""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 9)) / len(raw)

_PII_PATTERNS: Final[dict[str, str]] = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "url": r"https?://|www\.",
    "phone": r"(?<!\d)0\d{1,4}[-‐ー−]\d{1,4}[-‐ー−]\d{3,4}(?!\d)",
    "postal": r"〒\s?\d{3}-\d{4}",
    "card": r"(?<!\d)\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}(?!\d)",
}
def _compile(pats: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in pats))

def title_keywords(title: str) -> list[str]:
    """Content words of a title: runs of ≥2 kanji or ≥2 katakana (a cheap proxy for nouns)."""
    kws = re.findall(r"[一-鿿々]{2,}|[゠-ヿー]{2,}", title)
    return sorted(set(kws), key=len, reverse=True)

# --------------------------------------------------------------------------- outcomes
