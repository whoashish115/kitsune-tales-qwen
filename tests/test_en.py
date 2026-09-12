"""English variant (D-024): prompts, seeds, filters, policy, judge parsing, pipeline end-to-end."""
from __future__ import annotations
def test_fixture_lengths() -> None:
    assert 550 <= en.count_words(STORY_EN) <= 1250
    assert 110 <= en.count_words(SYNOPSIS_EN) <= 420

import json
import random
from pathlib import Path
@pytest.mark.parametrize("fmt", ["あらすじ", "短編", "続き"])
def test_prompt_round_trip(fmt: str) -> None:
    raise NotImplementedError

def test_seeds_deterministic_and_disjoint() -> None:
    a, b = en.build_test_prompts_en(), en.build_test_prompts_en()
    assert [x["title"] for x in a] == [x["title"] for x in b] and len(a) == 270
    train = en.build_train_prompts_en(400, [x["title"] for x in a], seed=3)
    assert not {x["title"] for x in train} & {x["title"] for x in a}
