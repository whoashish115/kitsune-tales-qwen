"""Chat-template round trips against the real base-model tokenizer (pinned revision).

Marked ``network``: the tokenizer is downloaded from the Hub on first run. The assertions use the
base model's end-of-turn marker from ``kitsune.versions`` (Gemma 4: ``<turn|>``), so they stay
correct if the base is ever changed again (D-001a).
"""

from __future__ import annotations

import pytest

from kitsune.prompts import (
    StoryRequest,
    build_user_prompt,
    render_completion,
    render_prompt,
    tokenize_example,
)
from kitsune.versions import BASE_END_OF_TURN, BASE_MODEL, BASE_REVISION

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    except OSError as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


USER = build_user_prompt(StoryRequest(["異世界転生", "冒険者ギルド"], "追放された剣士", "短編"))
STORY = "冒険者ギルドの扉が、軋んだ音を立てて開いた。\n「今日から、よろしくお願いします」"


def test_generation_prompt_is_non_thinking(tok) -> None:
    p = render_prompt(tok, USER)
    assert p.endswith("<|turn>model\n"), repr(p[-40:])  # no thinking channel is opened
    assert "<|think|>" not in p
    assert "<|turn>system\n" in p and USER in p


def test_end_of_turn_is_a_stop_token(tok) -> None:
    tid = tok.convert_tokens_to_ids(BASE_END_OF_TURN)
    assert isinstance(tid, int) and tid != tok.unk_token_id


def test_completion_is_story_plus_end_of_turn(tok) -> None:
    assert render_completion(tok, USER, STORY) == STORY + BASE_END_OF_TURN


def test_labels_mask_prompt_and_decode_to_story(tok) -> None:
    ex = tokenize_example(tok, USER, STORY, max_length=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labels)
    n_prompt = len(tok(render_prompt(tok, USER), add_special_tokens=False)["input_ids"])
    assert all(x == -100 for x in labels[:n_prompt])
    assert all(x != -100 for x in labels[n_prompt:])
    trained = tok.decode([t for t, lab in zip(ids, labels, strict=True) if lab != -100])
    assert trained == STORY + BASE_END_OF_TURN
    # The concatenation equals what the model sees at inference: prompt tokens, then generated tokens.
    assert tok.decode(ids) == render_prompt(tok, USER) + STORY + BASE_END_OF_TURN
    # Exactly one BOS, at the start (no double BOS from add_special_tokens).
    assert ids.count(tok.bos_token_id) == 1 and ids[0] == tok.bos_token_id


def test_too_long_raises(tok) -> None:
    with pytest.raises(ValueError):
        tokenize_example(tok, USER, STORY * 50, max_length=64)
