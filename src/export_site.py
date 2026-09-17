"""Export every number the project website shows into one JSON file (``reports/site/kitsune.json``).
The website lives in its own repository (kitsune-tales-qwen-site); ``--site <path>`` copies this file and the
figures into a local checkout of it, so the statistics are only ever produced here.

    python -m kitsune.export_site
from __future__ import annotations
The site renders only what this file contains, and this file is built only from ``reports/``, ``configs/`` and
Modal's billed totals, so every figure on the site traces to the same evaluation outputs as REPORT.md.
"""
import json
import re
from pathlib import Path
from typing import Any
import yaml
from kitsune import cost, en, versions
from kitsune.eval.report import SYSTEM_NOTES
from kitsune.taxonomy import FORMATS, GENRE_EN, GENRES
def _systems(res: dict, order: list[str]) -> list[dict]:
    raise NotImplementedError
