"""Policy behavior: refusals, off-genre redirects, and the held-out adversarial eval suites.
Training examples and eval suites are drawn from **disjoint** name/title lists, so the eval
measures whether the policy generalizes rather than whether specific strings were memorized.
"""
from __future__ import annotations
def refusal_kind(filter_reason: str, text: str) -> str:
    """Map a ``f_prompt_safety`` failure reason to a refusal kind (used by the demo's input guard)."""
    raise NotImplementedError

def refusal_text(kind: str) -> str:
    """The canonical refusal for a disallowed request of ``kind``."""
    return REFUSAL_OPENING + _REFUSAL_REASONS[kind] + REFUSAL_CLOSING
