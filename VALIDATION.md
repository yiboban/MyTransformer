# 框架验证记录

验证日期：2026-08-01。最终提交前会按本文件更新结果。

## 临时参考实现验证

开发时在 `.gitignore` 排除的临时目录放置了一份独立正确实现，仅用于审计测试 oracle 和生成 tiny golden fixture。该实现已删除，不会提交。

```text
python -m pytest tests -m "not slow" -q  → 60 passed, 1 deselected
python -m pytest tests -m slow -q        → 1 passed, 60 deselected
python -m pytest tests -q                → 61 passed
```

这证明全部 student tests 在符合公开契约的实现下可通过，包括 optional tiny repeated-corpus loss decrease。

## 学生 stub 状态的预期

学生骨架恢复后，`grader_tests`、fixture integrity 和 `verify_scaffold.py` 应通过；具体 assignment stage 应返回非零，并显示 `NotImplementedError`、中文可能原因、推荐文档和测试命令。绿色 Scaffold CI 只代表框架健康。

## 学生 stub 状态的实际最终回归

```text
uv sync                                      → 成功，uv.lock 锁定 46 packages
uv run ruff check .                          → All checks passed
uv run pytest grader_tests -q                → 18 passed
uv run python scripts/verify_scaffold.py     → Scaffold verification passed
uv run python -m grader --list               → 成功，7 stages / 100 分
uv run python -m grader --stage attention    → 预期非零；0/20，清晰 TODO 提示
uv run python -m grader --all                → 预期非零；0/100，7 stages 报告完整
uv run python -m grader --all --json ...     → 预期非零；JSON schema v1 已生成
```

Windows 上额外验证了 pytest 临时目录被固定在仓库内的已忽略目录，不依赖系统 temp symlink 权限。
