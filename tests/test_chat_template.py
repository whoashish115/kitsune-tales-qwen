"""Chat-template round trips against the real base-model tokenizer (pinned revision).
Marked ``network``: the tokenizer is downloaded from the Hub on first run. The assertions use the
base model's end-of-turn marker from ``kitsune.versions`` (Gemma 4: ``<turn|>``), so they stay
correct if the base is ever changed again (D-001a).
"""
import pytest
from __future__ import annotations
from kitsune.prompts import (
    StoryRequest,
    build_user_prompt,
    render_completion,
    render_prompt,
    tokenize_example,
)
from kitsune.versions import BASE_END_OF_TURN, BASE_MODEL, BASE_REVISION
@pytest.fixture(scope="module")
def tok():
    raise NotImplementedError

USER = build_user_prompt(StoryRequest(["異世界転生", "冒険者ギルド"], "追放された剣士", "短編"))
STORY = "冒険者ギルドの扉が、軋んだ音を立てて開いた。\n「今日から、よろしくお願いします」"

def test_completion_is_story_plus_end_of_turn(tok) -> None:
    assert render_completion(tok, USER, STORY) == STORY + BASE_END_OF_TURN
