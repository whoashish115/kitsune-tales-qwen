"""Automatic metrics for generated stories, plus bootstrap confidence intervals.
Diversity metrics work on characters for Japanese (no whitespace word boundaries) and on lowercased word
tokens for English (``unit="word"``, the standard for Li et al. 2016's distinct-n). They need no model tokenizer
and are reproducible on CPU.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import numpy as np
from kitsune.data.filters import (
    f_fantasy_rule,
    f_repetition,
    f_safety_rule,
    genre_cue_hits,
    has_markdown,
    latin_leaks,
    non_jis_kanji_rate,
    script_stats,
    simplified_chinese_hits,
    title_keywords,
)
from kitsune.taxonomy import count_chars, within_target

def _chars(text: str) -> str:
    return "".join(text.split())

_WORD = re.compile(r"[a-z0-9']+")

def _units(text: str, unit: str = "char") -> str | tuple[str, ...]:
    """Characters without whitespace (Japanese) or lowercased word tokens (English)."""
    return tuple(_WORD.findall(text.lower())) if unit == "word" else _chars(text)

def distinct_n(texts: Sequence[str], n: int, unit: str = "char") -> float:
    """Distinct-n: unique n-grams / total n-grams, pooled over ``texts``.

    (Li et al. 2016. Japanese uses characters because it has no spaces; English uses words.)
    Returns 0.0 if there are no n-grams.
    """
    total = 0
    uniq: set = set()
    for t in texts:
        s = _units(t, unit)
        grams = [s[i : i + n] for i in range(len(s) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0

def self_bleu(texts: Sequence[str], max_n: int = 4, unit: str = "char") -> float:
    """Mean BLEU of each text against all others (lower = more diverse). NaN if < 2 texts.

    Computed within one prompt's generations (e.g. the 3 seeds), then averaged over prompts.
    """
    if len(texts) < 2:
        return float("nan")
    scores = [
        char_bleu(t, [o for j, o in enumerate(texts) if j != i], max_n, unit) for i, t in enumerate(texts)
    ]
    return float(np.mean(scores))

@dataclass(frozen=True)
class OutputMetrics:
    """Per-output metrics. Rates are 0/1 per output, averaged later."""

    chars: int
    length_ok: float
    japanese_ratio: float
    hiragana_ratio: float
    zh_contaminated: float
    non_jis_kanji_rate: float
    repetitive: float
    degenerate: float
    fantasy: float
    genre_cue_rate: float
    title_reflected: float
    unsafe: float
    markdown: float
    latin_leak: float

def output_metrics(
    text: str, fmt: str, genres: list[str], title: str, passage: str | None = None
) -> OutputMetrics:
    """Compute all rule-based metrics for one generation."""
    raise NotImplementedError
