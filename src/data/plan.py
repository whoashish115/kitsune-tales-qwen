"""Deterministic generation plans (pure Python; runs locally and inside the GPU container).
A plan is a list of :class:`GenJob`. Given the same config, frozen test titles and brainstormed
titles, the same plan is produced, so a crashed run resumes on identical jobs.
"""
import random
from collections.abc import Iterable
from kitsune.data.dedup import TitleIndex
from kitsune.data.filters import f_prompt_safety
from kitsune.data.generate import GenJob, offgenre_job, parse_titles, story_job, titles_job
from kitsune.data.policy import train_policy_prompts
from kitsune.data.seeds import FORMAT_WEIGHTS, SeedPrompt, build_train_prompts, make_genres, make_title
from kitsune.taxonomy import GENRES
