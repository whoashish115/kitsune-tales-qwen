"""pyproject.toml and src/versions.py must pin the same versions."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from kitsune import versions

ROOT = Path(__file__).resolve().parents[1]


def _pins(reqs: list[str]) -> dict[str, str]:
    out = {}
    for r in reqs:
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^;\s]+)", r)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def test_gpu_pins_match() -> None:
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = _pins(py["project"]["dependencies"]) | _pins(py["project"]["optional-dependencies"]["gpu"])
    for pkg, ver in versions.GPU_PACKAGES.items():
        assert pins.get(pkg) == ver, f"{pkg}: pyproject={pins.get(pkg)} versions.py={ver}"


def test_revisions_are_full_hashes() -> None:
    for rev in (
        versions.BASE_REVISION,
        versions.ALT_BASE_REVISION,
        versions.GENERATOR_REVISION,
        versions.GENERATOR2_REVISION,
        versions.JUDGE_REVISION,
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", rev)


def test_all_models_permissively_licensed() -> None:
    assert set(versions.MODEL_LICENSES.values()) == {"apache-2.0"}
