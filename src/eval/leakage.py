"""Train/test leakage and memorization audit.
- ``overlap_rate``: share of an output's 32-char windows found in the training set.
- ``max_span``: the longest run of consecutive overlapping windows, converted to characters.
For each generated output we find the longest span (in characters) that also occurs verbatim
somewhere in the training responses, using hashed character n-grams:
The base model never saw our training data, so its numbers are the *chance* baseline (common
phrases); Kitsune's excess over the base is the memorization signal we report.
"""
from __future__ import annotations
import numpy as np
