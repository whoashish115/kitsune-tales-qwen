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


def test_passage_plan_en(test_prompts: list[dict], seed: int, candidates: int = 2) -> list[GenJob]:
    """``candidates`` source stories per held-out continuation prompt (first valid one is frozen)."""
    rng = random.Random(seed)
    jobs = []
    for p in test_prompts:
        if p["format"] != "続き":
            continue
        for k in range(candidates):
            j = en.story_job_en(p, rng, kind="test_passage")
            j.id = f"{p['id']}:test_passage:{k}"
            jobs.append(j)
    return jobs


def story_plan_en(
    gen_key: str,
    n_prompts: int,
    llm_titles: dict[str, list[str]],
    test_titles: list[str],
    llm_title_share: float,
    offgenre_per_prompt: int,
    seed: int,
) -> list[GenJob]:
    """English training jobs for one generator (same shape as :func:`kitsune.data.plan.story_plan`)."""
    rng = random.Random(seed)
    n_llm = int(n_prompts * llm_title_share)
    fmts, weights = ["あらすじ", "短編", "続き"], [0.25, 0.5, 0.25]
    per_genre = {g: rng.sample(ts, len(ts)) for g, ts in llm_titles.items()}
    seeds: list[dict] = []
    i = 0
    while len(seeds) < n_llm and any(per_genre.values()):
        g = GENRES[i % len(GENRES)]
        i += 1
        if not per_genre.get(g):
            continue
        seeds.append(
            {
                "id": f"{gen_key}-en-llm-{len(seeds):06d}",
                "genres": en._genres(g, rng),
                "title": per_genre[g].pop(),
                "format": rng.choices(fmts, weights)[0],
                "title_source": "llm",
            }
        )
    for s in en.build_train_prompts_en(n_prompts - len(seeds), test_titles, seed=seed, strict=False):
        seeds.append({**s, "id": f"{gen_key}-{s['id']}"})
    jobs = [en.story_job_en(s, rng) for s in seeds]
    for p in en.train_policy_prompts_en():
        if p["kind"] == "offgenre":
            jobs += [en.offgenre_job_en(p, rng, n) for n in range(offgenre_per_prompt)]
    for j in jobs:
        j.id = j.id if j.id.startswith(gen_key) else f"{gen_key}-{j.id}"
    return jobs
