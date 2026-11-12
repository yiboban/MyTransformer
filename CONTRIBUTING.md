# 贡献指南

感谢你愿意改进 MyTransformer！这是一个让学习者自己完成 Transformer 的**编程作业仓库**，不是完整参考实现。贡献时请保留这一教学目标。

## 可以贡献什么

- 修复文档中的错别字、失效链接，或补充易于理解的学习指引。
- 报告可复现的安装、跨平台或评分器问题；建议附上 Python 版本、操作系统、命令、预期与实际输出。
- 改进脚手架、CI 或测试的稳定性，但**不要**删除、跳过或弱化正确性测试。

请勿提交完整学生答案、直接填完核心 TODO，或在 `grader/adapters.py` 中加入实际算法。不要提交模型权重、缓存、个人数据或密钥。更多边界请参阅 [AGENTS.md](AGENTS.md)。

## 本地准备

建议使用 Python 3.11 和 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/<你的用户名>/MyTransformer.git
cd MyTransformer
uv sync --extra dev
```

若不使用 uv，可按照 [README 的 pip 安装说明](README.md#环境安装)配置环境。

## 修改后检查

先运行与脚手架相关的检查，均不需要 GPU 或下载数据集：

```bash
uv run ruff check .
uv run pytest grader_tests -q
uv run python scripts/verify_scaffold.py
```

如果修改涉及某个作业阶段，可额外运行 `uv run python -m grader --stage <阶段名>`，或对应的 `tests/` 测试文件。**注意：** 原始学生骨架刻意保留 `NotImplementedError`，因此学生阶段测试失败不代表你的文档或脚手架修改有问题；不要为了让它们变绿而改动评分标准。

## 提交 Pull Request

1. Fork 仓库，并从最新 `main` 创建一个描述性的分支（如 `docs/fix-typo`）。
2. 一次 PR 只解决一个小问题；提交信息尽量清楚，例如 `docs: clarify setup steps`。
3. 在 PR 描述中说明改动动机、修改内容和实际运行的检查；如果无法运行某项检查，请写明原因。
4. 提交到上游仓库的 `main`，并根据维护者的 review 意见迭代。

如果还不确定问题属于文档还是实现，可以先提交 Issue 讨论。
