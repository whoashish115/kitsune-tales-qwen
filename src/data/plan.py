"""Deterministic generation plans (pure Python; runs locally and inside the GPU container).
A plan is a list of :class:`GenJob`. Given the same config, frozen test titles and brainstormed
titles, the same plan is produced, so a crashed run resumes on identical jobs.
"""
from __future__ import annotations
import random
from collections.abc import Iterable
from kitsune.data.dedup import TitleIndex
from kitsune.data.filters import f_prompt_safety
from kitsune.data.generate import GenJob, offgenre_job, parse_titles, story_job, titles_job
from kitsune.data.policy import train_policy_prompts
from kitsune.data.seeds import FORMAT_WEIGHTS, SeedPrompt, build_train_prompts, make_genres, make_title
from kitsune.taxonomy import GENRES

def title_brainstorm_plan(calls_per_genre: int, seed: int) -> list[GenJob]:
    rng = random.Random(seed)
    jobs = []
    for g in GENRES:
        for i in range(calls_per_genre):
            examples = [make_title(g, rng) for _ in range(3)]
            jobs.append(titles_job(g, examples, rng, i))
    return jobs

def clean_llm_titles(outputs: Iterable[dict], test_index: TitleIndex) -> dict[str, list[str]]:
    """Parse brainstorm outputs; drop unsafe titles and anything near a frozen test title."""
    pool: dict[str, list[str]] = {g: [] for g in GENRES}
    seen: set[str] = set()
    for o in outputs:
        genre = o["id"].split(":")[1]
        for t in parse_titles(o["text"]):
            if t in seen or not f_prompt_safety(t).passed or test_index.is_near(t):
                continue
            seen.add(t)
            pool[genre].append(t)
    return pool
