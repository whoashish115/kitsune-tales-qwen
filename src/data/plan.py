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


def test_passage_plan(test_prompts: list[dict], seed: int, candidates: int = 2) -> list[GenJob]:
    """``candidates`` source stories per held-out 続き prompt; the first valid one becomes the frozen passage."""
    rng = random.Random(seed)
    jobs = []
    for p in test_prompts:
        if p["format"] != "続き":
            continue
        for k in range(candidates):
            j = story_job(SeedPrompt(p["id"], p["genres"], p["title"], "続き"), rng, kind="test_passage")
            j.id = f"{p['id']}:test_passage:{k}"
            jobs.append(j)
    return jobs


def story_plan(
    gen_key: str,
    n_prompts: int,
    llm_titles: dict[str, list[str]],
    test_titles: list[str],
    llm_title_share: float,
    offgenre_per_prompt: int,
    seed: int,
) -> list[GenJob]:
    """Training generation jobs for one generator.

    ``n_prompts`` story/synopsis/source prompts (a ``llm_title_share`` of them with LLM-brainstormed
    titles, the rest templated), plus ``offgenre_per_prompt`` redirect stories per off-genre prompt.
    """
    rng = random.Random(seed)
    n_llm = int(n_prompts * llm_title_share)
    fmts, weights = list(FORMAT_WEIGHTS), list(FORMAT_WEIGHTS.values())
    per_genre = {g: rng.sample(ts, len(ts)) for g, ts in llm_titles.items()}
    llm_seeds: list[SeedPrompt] = []
    i = 0
    while len(llm_seeds) < n_llm and any(per_genre.values()):
        g = GENRES[i % len(GENRES)]
        i += 1
        if not per_genre.get(g):
            continue
        t = per_genre[g].pop()
        fmt = rng.choices(fmts, weights)[0]
        llm_seeds.append(
            SeedPrompt(f"{gen_key}-llm-{len(llm_seeds):06d}", make_genres(g, rng), t, fmt, "llm")
        )
    # Whatever the brainstorm could not supply is topped up with templated titles.
    templated = build_train_prompts(n_prompts - len(llm_seeds), test_titles, seed=seed)
    seeds = [
        SeedPrompt(f"{gen_key}-{s.id}", s.genres, s.title, s.format, "template") for s in templated
    ] + llm_seeds
    jobs = [story_job(s, rng) for s in seeds]
    for p in train_policy_prompts():
        if p.kind == "offgenre":
            jobs += [offgenre_job(p, rng, n) for n in range(offgenre_per_prompt)]
    for j in jobs:
        j.id = j.id if j.id.startswith(gen_key) else f"{gen_key}-{j.id}"
    return jobs
