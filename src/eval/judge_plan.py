"""Build judge jobs: system-vs-system comparisons and the known-answer validation set."""

from __future__ import annotations

import random
from collections import defaultdict

from kitsune.data.generate import GenJob
from kitsune.eval.judge import CORRUPTIONS, corrupt, judge_job


def _request_text(row: dict) -> str:
    lines = [f"ジャンル: {', '.join(row['genres'])}", f"タイトル: {row['title']}", f"形式: {row['format']}"]
    if row.get("passage"):
        lines.append("本文（この続きを書く依頼）:\n" + row["passage"])
    return "\n".join(lines)


def _lang_fns(lang: str):
    """(request text, judge job, corruption kinds, corrupt) for the language (English: D-024)."""
    if lang == "en":
        from kitsune import en

        return en.request_text_en, en.judge_job_en, en.CORRUPTIONS_EN, en.corrupt_en
    return _request_text, judge_job, CORRUPTIONS, corrupt


def comparison_jobs(
    rows_x: list[dict], rows_y: list[dict], x: str, y: str, n_pairs: int, seed: int = 0, lang: str = "ja"
) -> list[GenJob]:
    """``n_pairs`` test prompts (stratified by format, seed-0 outputs), each judged in both orders."""
    request_text, make_job, _, _ = _lang_fns(lang)
    rng = random.Random(seed)
    bx = {r["prompt_id"]: r for r in rows_x if r["suite"] == "test" and r["seed"] == 0}
    by = {r["prompt_id"]: r for r in rows_y if r["suite"] == "test" and r["seed"] == 0}
    common = sorted(set(bx) & set(by))
    by_fmt: dict[str, list[str]] = defaultdict(list)
    for pid in common:
        by_fmt[bx[pid]["format"]].append(pid)
    chosen: list[str] = []
    fmts = sorted(by_fmt)
    for f in fmts:
        rng.shuffle(by_fmt[f])
    i = 0
    while len(chosen) < min(n_pairs, len(common)):
        f = fmts[i % len(fmts)]
        if by_fmt[f]:
            chosen.append(by_fmt[f].pop())
        i += 1
    jobs = []
    for pid in chosen:
        req = request_text(bx[pid])
        pair = f"{x}__{y}__{pid}"
        jobs.append(make_job(pair, "xy", req, bx[pid]["text"], by[pid]["text"]))
        jobs.append(make_job(pair, "yx", req, by[pid]["text"], bx[pid]["text"]))
    return jobs


EXCERPT_NOTE = {
    "ja": "\n（注: 両作品とも、同じ長さに切りそろえた冒頭部分だけを示しています。長さや結末の有無ではなく、文章と物語の質だけで比べてください。）",
    "en": "\n(Note: both texts are openings cut to the same length. Judge only the prose and storytelling quality, not length or whether the story ends.)",
}


def cut_excerpt(text: str, chars: int, lang: str) -> str | None:
    """The opening of ``text`` cut at the last sentence end at or before ``chars`` characters (None if too short)."""
    if len(text) < chars:
        return None
    head = text[:chars]
    ends = [
        head.rfind(c)
        for c in (("。", "」", "！", "？") if lang == "ja" else (". ", "! ", "? ", '." ', '!" ', '?" '))
    ]
    cut = max(ends)
    if cut < chars // 2:
        return None
    return head[: cut + (1 if lang == "ja" else 2)].rstrip()


def excerpt_jobs(
    rows_x: list[dict],
    rows_y: list[dict],
    x: str,
    y: str,
    n_pairs: int,
    chars: int,
    lang: str = "ja",
    seed: int = 0,
) -> list[GenJob]:
    """Length-matched comparison: seed-0 short-story openings of both systems cut to the same length, both orders.

    Controls the judge's length preference (the base model writes ~1.7x longer than requested).
    """
    request_text, make_job, _, _ = _lang_fns(lang)
    rng = random.Random(seed)
    bx = {
        r["prompt_id"]: r for r in rows_x if r["suite"] == "test" and r["seed"] == 0 and r["format"] == "短編"
    }
    by = {
        r["prompt_id"]: r for r in rows_y if r["suite"] == "test" and r["seed"] == 0 and r["format"] == "短編"
    }
    common = sorted(set(bx) & set(by))
    rng.shuffle(common)
    jobs = []
    for pid in common:
        a, b = cut_excerpt(bx[pid]["text"], chars, lang), cut_excerpt(by[pid]["text"], chars, lang)
        if not a or not b:
            continue
        req = request_text(bx[pid]) + EXCERPT_NOTE[lang]
        pair = f"excerpt-{x}__excerpt-{y}__{pid}"
        jobs.append(make_job(pair, "xy", req, a, b))
        jobs.append(make_job(pair, "yx", req, b, a))
        if len(jobs) >= 2 * n_pairs:
            break
    return jobs


def validation_jobs(stories: list[dict], n: int, seed: int = 0, lang: str = "ja") -> list[GenJob]:
    """Known-answer pairs: intact story (x) vs a corrupted copy (y), one corruption type per story, both orders.

    ``stories`` rows need prompt fields (genres, title, format, passage?) and ``response``.
    """
    request_text, make_job, corruptions, corrupt_fn = _lang_fns(lang)
    rng = random.Random(seed)
    pool = [s for s in stories if s["format"] == "短編"]
    rng.shuffle(pool)
    pool = pool[:n]
    jobs = []
    for i, s in enumerate(pool):
        kind = corruptions[i % len(corruptions)]
        other = next((o for o in pool if o["genres"][0] != s["genres"][0]), pool[(i + 1) % len(pool)])
        bad = corrupt_fn(s["response"], kind, rng, other_story=other["response"])
        req = request_text(s)
        pid = f"val__{i:03d}__{kind}"
        j1, j2 = make_job(pid, "xy", req, s["response"], bad), make_job(pid, "yx", req, bad, s["response"])
        j1.meta["corruption"] = j2.meta["corruption"] = kind
        jobs += [j1, j2]
    return jobs
