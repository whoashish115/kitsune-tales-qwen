"""Build the processed dataset: raw generations + labels → filtered, deduplicated, split JSONL.

    python -m kitsune.data.pipeline build --raw data/raw --out data/processed --reports reports/data
Every dropped sample keeps its first failing reason, so ``stats.json`` and the funnel plot
show exactly what was removed and why.
"""
from __future__ import annotations
SYNTHETIC_LICENSE = "apache-2.0 (generated with an Apache-2.0 model; see DATA_CARD.md)"
SEED_LICENSE = "apache-2.0 (written for this project)"

from kitsune.data.dedup import dedup
from kitsune.data.filters import FilterOutcome, clean_generation, f_title_clean, run_rule_filters
from kitsune.data.generate import split_for_continuation
from kitsune.data.labels import Labels, parse_labels
from kitsune.data.policy import REDIRECT_PREFIX, PolicyPrompt, refusal_text, train_policy_prompts
from kitsune.prompts import StoryRequest, build_user_prompt
from kitsune.schema import Record, file_sha256, make_record, read_jsonl, write_jsonl
from kitsune.taxonomy import GENRES
def _candidate_en(g: dict, rng: random.Random) -> Candidate:
    raise NotImplementedError
