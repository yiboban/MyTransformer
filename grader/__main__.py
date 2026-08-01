"""`python -m grader` 本地评分入口。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .hints import hint_for
from .rubric import BY_KEY, STAGES, Stage


@dataclass
class StageResult:
    stage: str
    title: str
    passed: int
    failed: int
    score: float
    max_score: int
    duration_seconds: float
    first_error: str
    likely_cause: str
    docs: str
    command: str


def _counts(output: str, returncode: int) -> tuple[int, int]:
    passed = sum(int(x) for x in re.findall(r"(\d+) passed", output))
    failed = sum(int(x) for x in re.findall(r"(\d+) failed", output))
    errors = sum(int(x) for x in re.findall(r"(\d+) error", output))
    if returncode and failed + errors == 0:
        errors = 1
    return passed, failed + errors


def _first_error(output: str) -> str:
    for line in output.splitlines():
        if "NotImplementedError" in line or line.startswith("E   "):
            return line.removeprefix("E   ").strip()[:500]
    return "无关键错误" if "failed" not in output else "请使用 --verbose 查看首个 traceback"


def run_stage(stage: Stage, verbose: bool) -> StageResult:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-m",
        stage.marker,
        "--tb=short",
        f"--basetemp=reports/.pytest_tmp_{stage.key}",
    ]
    if not verbose:
        command.extend(["-q", "--disable-warnings"])
    start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    duration = time.perf_counter() - start
    output = proc.stdout + proc.stderr
    passed, failed = _counts(output, proc.returncode)
    total = passed + failed
    score = stage.points * passed / total if total else 0.0
    if "NotImplementedError" in output:
        # 核心入口仍是 stub 时，不因 AST/API 等框架检查通过而给完成分。
        score = 0.0
    first = _first_error(output)
    if verbose:
        print(output.rstrip())
    return StageResult(
        stage=stage.key,
        title=stage.title,
        passed=passed,
        failed=failed,
        score=round(score, 2),
        max_score=stage.points,
        duration_seconds=round(duration, 3),
        first_error=first,
        likely_cause=hint_for(stage, first),
        docs=stage.docs,
        command=f"python -m pytest tests -m {stage.marker} -q",
    )


def _print_result(result: StageResult) -> None:
    print(f"\n[{result.stage}] {result.title}")
    print(f"通过 {result.passed} / 失败 {result.failed}")
    print(f"得分 {result.score:.2f}/{result.max_score}，耗时 {result.duration_seconds:.3f}s")
    if result.failed:
        print(f"第一个关键错误：{result.first_error}")
        print(f"可能原因：{result.likely_cause}")
    print(f"推荐阅读：{result.docs}")
    print(f"对应测试：{result.command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MyTransformer 本地评分器")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--list", action="store_true", help="列出阶段与分值")
    target.add_argument("--stage", choices=sorted(BY_KEY), help="运行单个阶段")
    target.add_argument("--all", action="store_true", help="运行全部阶段")
    parser.add_argument("--json", type=Path, help="写入结构化 JSON 报告")
    parser.add_argument("--verbose", action="store_true", help="显示完整 pytest traceback")
    args = parser.parse_args(argv)

    if args.list:
        for stage in STAGES:
            print(f"{stage.key:12} {stage.points:>3} 分  {stage.title}  ({stage.docs})")
        print(f"{'total':12} {sum(s.points for s in STAGES):>3} 分")
        return 0

    selected = STAGES if args.all else (BY_KEY[args.stage],)
    results = [run_stage(stage, args.verbose) for stage in selected]
    for result in results:
        _print_result(result)
    total_score = sum(result.score for result in results)
    total_max = sum(result.max_score for result in results)
    print(f"\n总分：{total_score:.2f}/{total_max}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "score": round(total_score, 2),
            "max_score": total_max,
            "results": [asdict(result) for result in results],
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入：{args.json}")
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
