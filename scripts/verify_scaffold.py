"""验证“作业脚手架完整且学生核心 TODO 仍未完成”。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
REQUIRED = {
    "README.md",
    "ASSIGNMENT.md",
    "AGENTS.md",
    "pyproject.toml",
    "uv.lock",
    "grader/adapters.py",
    "grader/__main__.py",
    "tests/test_attention.py",
    "grader_tests/test_mutations.py",
    "docs/00_学习路线与接口总览.md",
    ".github/workflows/scaffold.yml",
}
STUB_FILES = {
    "src/mytransformer/tokenizer.py": "TODO(A01)",
    "src/mytransformer/layers.py": "TODO(A02)",
    "src/mytransformer/rope.py": "TODO(A03)",
    "src/mytransformer/attention.py": "TODO(A04)",
    "src/mytransformer/ffn.py": "TODO(A05)",
    "src/mytransformer/block.py": "TODO(A06)",
    "src/mytransformer/model.py": "TODO(A07)",
}


def run(command: list[str], *, expect_success: bool) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    if expect_success != (result.returncode == 0):
        raise RuntimeError(f"命令退出码不符合预期: {' '.join(command)}\n{output}")
    return output


def main() -> int:
    missing = sorted(path for path in REQUIRED if not (ROOT / path).is_file())
    if missing:
        raise RuntimeError(f"缺少要求文件: {missing}")
    for relative, marker in STUB_FILES.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        if marker not in text or "NotImplementedError" not in text:
            raise RuntimeError(f"学生骨架不再是未完成状态: {relative}")

    listed = run([sys.executable, "-m", "grader", "--list"], expect_success=True)
    assert "100 分" in listed
    failed = run([sys.executable, "-m", "grader", "--stage", "attention"], expect_success=False)
    assert "NotImplementedError" in failed and "可能原因" in failed and "INTERNALERROR" not in failed
    self_tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "grader_tests",
            "-q",
            "--basetemp=reports/.pytest_tmp_verify",
        ],
        expect_success=True,
    )
    assert "passed" in self_tests
    print("Scaffold verification passed: 文件、TODO、grader 失败路径和 grader 自测均正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
