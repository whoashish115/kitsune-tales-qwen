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
def _chars(text: str) -> str:
    return "".join(text.split())

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

_WORD = re.compile(r"[a-z0-9']+")
