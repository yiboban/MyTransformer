from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_all_golden_fixtures_have_version_dtype_config_weights_output_and_valid_sha():
    root = Path(__file__).parents[1] / "tests" / "fixtures"
    expected = {"rope_v1", "mha_rope_v1", "block_v1", "lm_v1"}
    assert {path.stem for path in root.glob("*.pt")} == expected
    for path in root.glob("*.pt"):
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert manifest["fixture_version"] == 1
        assert manifest["dtype"] == "float32"
        assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        payload = safe_load(path)
        assert payload["fixture_version"] == 1
        assert payload["dtype"] == "float32"
        assert isinstance(payload["config"], dict)
        assert "expected" in payload
        assert "state_dict" in payload
        assert all(torch.is_tensor(v) for v in payload["state_dict"].values())


def test_fixtures_are_small_and_cpu_only():
    root = Path(__file__).parents[1] / "tests" / "fixtures"
    for path in root.glob("*.pt"):
        assert path.stat().st_size < 250_000
        payload = safe_load(path)
        tensors = []
        stack = [payload]
        while stack:
            value = stack.pop()
            if torch.is_tensor(value):
                tensors.append(value)
            elif isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        assert tensors and all(t.device.type == "cpu" for t in tensors)
