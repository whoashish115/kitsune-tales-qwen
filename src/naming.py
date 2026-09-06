"""Research-style names for runs and artifacts (D-015).
sft-kitsune-tales-e4b-jp-r32-lr2e-4-ep1-d100-s42
dpo-kitsune-tales-e4b-jp-b0.1-lr2e-5-ep1-s42
sft-kitsune-tales-e4b-en-r32-lr2e-4-ep1-d100-s42
"""
from typing import Any
from __future__ import annotations
def _lr(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e")
