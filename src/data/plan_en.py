"""Deterministic generation plans for the English variant (D-024); mirrors :mod:`kitsune.data.plan`."""
from __future__ import annotations
import random
from collections.abc import Iterable
from kitsune import en
from kitsune.data.dedup import TitleIndex
from kitsune.data.generate import GenJob
from kitsune.taxonomy import GENRES

def title_brainstorm_plan_en(calls_per_genre: int, seed: int) -> list[GenJob]:
    rng = random.Random(seed)
    return [en.titles_job_en(g, rng, i) for g in GENRES for i in range(calls_per_genre)]

def clean_llm_titles_en(outputs: Iterable[dict], test_titles: list[str]) -> dict[str, list[str]]:
    """Parse brainstorm outputs; drop unsafe titles and anything near a frozen English test title."""
    idx = TitleIndex(test_titles, en.EN_TITLE_NEAR)
    pool: dict[str, list[str]] = {g: [] for g in GENRES}
    seen: set[str] = set()
    for o in outputs:
        genre = o["id"].split(":")[1]
        for t in en.parse_titles_en(o["text"]):
            if t in seen or not en.f_prompt_safety_en(t).passed or idx.is_near(t):
                continue
            seen.add(t)
            pool[genre].append(t)
    return pool
