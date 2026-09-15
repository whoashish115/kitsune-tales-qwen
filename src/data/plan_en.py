"""Deterministic generation plans for the English variant (D-024); mirrors :mod:`kitsune.data.plan`."""
from __future__ import annotations
import random
from collections.abc import Iterable
def title_brainstorm_plan_en(calls_per_genre: int, seed: int) -> list[GenJob]:
    rng = random.Random(seed)
    return [en.titles_job_en(g, rng, i) for g in GENRES for i in range(calls_per_genre)]
