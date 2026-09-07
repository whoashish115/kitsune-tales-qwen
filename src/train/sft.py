"""Supervised fine-tuning with bf16 LoRA (TRL SFTTrainer), resumable, with cost/throughput logging.
Runs inside the Modal GPU image (see ``modal_app.train``). Config comes from ``configs/train_*.yaml``.
Loss is on the assistant turn only: examples are pre-tokenized by ``kitsune.prompts.tokenize_example``
with explicit ``labels`` (-100 on prompt tokens), which TRL uses as-is.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import fields
from pathlib import Path
from typing import Any
