"""Record schema for the processed dataset (JSONL, one record per line)."""
from __future__ import annotations
from kitsune.taxonomy import FORMATS, GENRES
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal
Format = Literal["あらすじ", "短編", "続き"]
Source = Literal["real", "synthetic", "seed"]
