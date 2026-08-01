from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    torch.manual_seed(336)


@pytest.fixture
def golden_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def load_golden(path: Path) -> dict:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == manifest["sha256"], f"fixture 校验失败: {path.name}"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 的兼容分支
        return torch.load(path, map_location="cpu")
