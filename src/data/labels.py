"""LLM labels for data filtering (the "small judge" half of the fantasy/safety/consistency filters).
A labeler model reads (request, story) and returns strict JSON. Labels from a model that did
not write the sample ("cross" labels) are preferred over self-labels; see D-011.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Final
from __future__ import annotations
from kitsune.data.generate import GenJob
LABEL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "fantasy": {"type": "boolean"},
        "general_audience": {"type": "boolean"},
        "real_person_or_existing_ip": {"type": "boolean"},
        "genre_match": {"type": "integer", "minimum": 0, "maximum": 2},
        "title_match": {"type": "integer", "minimum": 0, "maximum": 2},
        "quality": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": [
        "fantasy",
        "general_audience",
        "real_person_or_existing_ip",
        "genre_match",
        "title_match",
        "quality",
    ],
    "additionalProperties": False,
}
