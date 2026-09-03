"""pyproject.toml and src/versions.py must pin the same versions."""
from __future__ import annotations
import re
import tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

from kitsune import versions
def test_gpu_pins_match() -> None:
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = _pins(py["project"]["dependencies"]) | _pins(py["project"]["optional-dependencies"]["gpu"])
    for pkg, ver in versions.GPU_PACKAGES.items():
        assert pins.get(pkg) == ver, f"{pkg}: pyproject={pins.get(pkg)} versions.py={ver}"
