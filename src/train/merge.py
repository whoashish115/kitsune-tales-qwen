"""Merge a LoRA adapter into the base weights and verify the merge.
Verification (spec section 5): on a fixed prompt set, the merged model must match
adapter-on-base (1) token-for-token under greedy decoding and (2) within a small logit
tolerance on a teacher-forced batch. The report is written next to the merged weights and
copied to ``reports/merge_check.json``.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
