"""Data CLI: freeze the held-out prompt sets.
from __future__ import annotations
python -m kitsune.data.cli freeze-test        # writes + hashes data/test_prompts.jsonl, data/eval_policy_prompts.jsonl
python -m kitsune.data.cli verify-test        # fails if the frozen files changed
python -m kitsune.data.cli freeze-test-en     # English variant (D-024): data/test_prompts_en.jsonl + policy suite
python -m kitsune.data.cli build-test-set-en --raw data/raw/full_en
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from kitsune.data.policy import eval_policy_prompts
from kitsune.data.seeds import build_test_prompts
from kitsune.schema import file_sha256, write_jsonl
