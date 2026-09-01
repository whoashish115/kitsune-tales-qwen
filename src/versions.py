"""Single source of truth for pinned versions and model revisions.
from __future__ import annotations
The Modal images in ``modal_app.py`` install exactly these versions, and
``tests/test_versions.py`` checks that ``pyproject.toml`` agrees with them.
Model revisions are full Hugging Face commit hashes, resolved on 2026-09-29.
"""
PYTHON_VERSION = "3.12"
# D-001a (bake-off, 2026-09-29): the base switched from Qwen3.5-4B to Gemma 4 E4B under the pre-registered rule.
BASE_MODEL = "google/gemma-4-E4B-it"
BASE_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
BASE_END_OF_TURN = "<turn|>"  # the chat template's end-of-turn marker; a generation stop token (id 106)
ALT_BASE_MODEL = "Qwen/Qwen3.5-4B"  # former choice; kept as an extra baseline
ALT_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GENERATOR_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"  # D-003 primary synthetic-data generator
GENERATOR_REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
