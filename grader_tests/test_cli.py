from __future__ import annotations

import json
import subprocess
import sys


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "grader", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_list_succeeds_and_totals_100_points():
    result = run_cli("--list")
    assert result.returncode == 0
    assert "bpe" in result.stdout and "attention" in result.stdout and "lm" in result.stdout
    assert "100 分" in result.stdout


def test_cli_stub_stage_fails_cleanly_with_hint_and_original_error():
    result = run_cli("--stage", "attention")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "NotImplementedError" in combined
    assert "可能原因" in combined and "sqrt(d_head)" in combined
    assert "INTERNALERROR" not in combined


def test_cli_all_writes_structured_report(tmp_path):
    report = tmp_path / "latest.json"
    result = run_cli("--all", "--json", str(report))
    assert result.returncode != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["max_score"] == 100
    assert [item["stage"] for item in payload["results"]] == [
        "bpe",
        "primitives",
        "rope",
        "attention",
        "ffn",
        "block",
        "lm",
    ]
    assert all(item["failed"] > 0 for item in payload["results"])


def test_cli_verbose_keeps_pytest_traceback():
    result = run_cli("--stage", "ffn", "--verbose")
    assert result.returncode != 0
    assert "short test summary info" in result.stdout
    assert "NotImplementedError" in result.stdout
