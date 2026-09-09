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


def _ngram_counts(s: str | tuple[str, ...], n: int) -> Counter:
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def char_bleu(hyp: str, refs: Sequence[str], max_n: int = 4, unit: str = "char") -> float:
    """Sentence-level BLEU (characters, or words with ``unit="word"``), add-one smoothing on higher orders, brevity penalty."""
    h = _units(hyp, unit)
    rs = [_units(r, unit) for r in refs]
    if not h or not rs:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        hc = _ngram_counts(h, n)
        if not hc:
            return 0.0
        max_ref: Counter[str] = Counter()
        for r in rs:
            for g, c in _ngram_counts(r, n).items():
                max_ref[g] = max(max_ref[g], c)
        clipped = sum(min(c, max_ref[g]) for g, c in hc.items())
        total = sum(hc.values())
        p = (clipped + (1 if n > 1 else 0)) / (total + (1 if n > 1 else 0))
        if p == 0:
            return 0.0
        log_p += math.log(p) / max_n
    ref_len = min((len(r) for r in rs), key=lambda rl: (abs(rl - len(h)), rl))
    bp = 1.0 if len(h) > ref_len else math.exp(1 - ref_len / max(len(h), 1))
    return bp * math.exp(log_p)


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
    st = script_stats(text)
    rep = not f_repetition(text).passed
    stripped = text.strip()
    degenerate = (
        count_chars(text) < 50
        or "<think>" in text
        or "<|im_" in text
        or rep
        or (bool(stripped) and stripped[-1] not in "。」』！？!?…―）)》】〉」")
    )
    ctx = (passage + "\n" + text) if passage else text
    cues = [bool(genre_cue_hits(ctx, g)) for g in genres]
    kws = title_keywords(title)
    return OutputMetrics(
        chars=count_chars(text),
        length_ok=float(within_target(text, fmt)),
        japanese_ratio=st.japanese_ratio,
        hiragana_ratio=st.hiragana_ratio,
        zh_contaminated=float(bool(simplified_chinese_hits(text))),
        non_jis_kanji_rate=non_jis_kanji_rate(text),
        repetitive=float(rep),
        degenerate=float(degenerate),
        fantasy=float(f_fantasy_rule(ctx, fmt).passed),
        genre_cue_rate=float(np.mean(cues)) if cues else 0.0,
        title_reflected=float(any(k in text for k in kws)) if (kws and fmt != "続き") else float("nan"),
        unsafe=float(not f_safety_rule(text).passed),
        markdown=float(has_markdown(text)),
        latin_leak=float(bool(latin_leaks(text))),
    )


@dataclass(frozen=True)
class CI:
    mean: float
    low: float
    high: float
    n: int

    def fmt(self, digits: int = 3, pct: bool = False) -> str:
        k = 100.0 if pct else 1.0
        return f"{self.mean * k:.{digits}f} [{self.low * k:.{digits}f}, {self.high * k:.{digits}f}]"


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI of ``stat`` over ``values`` (NaNs dropped).

    ``values`` should be one number per *prompt* (already averaged over seeds), so the
    resampling unit is the prompt, not the individual generation.
    """
    x = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(float(stat(x)), float(lo), float(hi), len(x))


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> CI:
    """CI of mean(b - a) over paired per-prompt values (e.g. Kitsune minus base)."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if not (math.isnan(x) or math.isnan(y))]
    d = np.asarray([y - x for x, y in pairs], dtype=float)
    return bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
