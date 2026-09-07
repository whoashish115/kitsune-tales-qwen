"""End-to-end CPU test of the dataset pipeline on hand-written fixtures."""
from __future__ import annotations
GOOD_LABEL = json.dumps(
    {
        "fantasy": True,
        "general_audience": True,
        "real_person_or_existing_ip": False,
        "genre_match": 2,
        "title_match": 2,
        "quality": 4,
    }
)

from kitsune.data.generate import split_for_continuation
from kitsune.data.policy import REDIRECT_PREFIX, train_policy_prompts
from kitsune.data.seeds import SeedPrompt, build_test_prompts, build_train_prompts
from kitsune.fixtures import GENRES, STORY, SYNOPSIS, TITLE
from kitsune.schema import Record, read_jsonl, write_jsonl
from kitsune.taxonomy import count_chars
def _lab(sid: str, who: str, text: str) -> dict:
    """A label row in the format modal_app.data_run writes."""
    raise NotImplementedError
