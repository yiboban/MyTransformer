# MyTransformer：从零实现现代 Decoder-only Transformer

这是一套本地编程作业，不是“已经写好的 Transformer 教程”。你将依次完成 byte-level BPE、数值基础层、RoPE、因果自注意力、SwiGLU、pre-norm block，最后组合成一个可训练、可生成文本的 Decoder-only Transformer Language Model。

本项目受 [Stanford CS336](https://cs336.stanford.edu/) 的教学方式（小接口、adapter、pytest 驱动）启发，但讲义、代码、测试和 fixture 均为独立原创；本项目与 Stanford University 或官方课程没有隶属、认可或维护关系。仓库不含可直接复制的学生答案。

> GitHub 上的绿色 Scaffold CI 表示作业框架正常，不代表学生已经完成 Transformer。

## 学习路线

`A00 张量约定 → A01 BPE → A02 基础层 → A03 RoPE/Mask → A04 Attention → A05 SwiGLU → A06 Block → A07 LM`

完整模型固定采用：`RoPE + RMSNorm + SwiGLU + causal self-attention`，是 Decoder-only Language Model，不是机器翻译的 Encoder-Decoder Transformer。默认测试模型只有约数万参数，CPU 即可完成；8GB 显存的 RTX 4060 Laptop 也足够。

## 环境安装

推荐 Python 3.11 与 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --extra dev
```

传统 pip（Windows PowerShell、Linux/macOS 均可）：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

依赖安装完成后，所有测试都离线执行：不下载模型，不访问数据集，不依赖 GPU 或本机绝对路径。

## 第一次运行

```bash
uv run python -m grader --list
uv run python -m grader --stage bpe
```

初始学生代码在核心位置含 `TODO` 和 `NotImplementedError`，所以 stage 测试失败是正常且有意的。先读错误提示和推荐文档；不要一开始运行整个测试套件并被大量红色输出淹没。

遇到张量形状、数值稳定性或梯度相关问题时，可先查阅 [常见错误排查](docs/08_常见错误排查.md)。

常用命令：

```bash
uv run pytest tests/test_mask.py -q        # 单文件
uv run pytest tests -m attention -q        # 单阶段 pytest
uv run python -m grader --stage attention  # 带分数和中文提示
uv run python -m grader --all
uv run python -m grader --all --json reports/latest.json
uv run python -m grader --stage lm --verbose
```

推荐从 [ASSIGNMENT.md](ASSIGNMENT.md) 和 [docs/00_学习路线与接口总览.md](docs/00_学习路线与接口总览.md) 开始，然后打开 [src/mytransformer/tokenizer.py](src/mytransformer/tokenizer.py) 完成 Assignment 01。

## 测试与评分

评分共 100 分：BPE 20、基础层 10、RoPE/Mask 15、Attention 20、SwiGLU 10、Block 10、LM 15。测试组合手工小例子、独立数学公式、PyTorch differential、golden fixture、梯度与 metamorphic invariant；浮点数使用明确 tolerance，离散对象精确比较。

`grader/adapters.py` 是测试与实现之间的薄连接层。如果你采用不同文件名或类名，只改 adapter 的导入/参数名映射；不要在 adapter 里补算法。

## 参与贡献

欢迎提交文档勘误、可复现的问题报告和脚手架改进。首次贡献请先阅读 [贡献指南](CONTRIBUTING.md)，并保留学生核心 TODO 的练习性质。

## CI 的含义

默认 `Scaffold CI` 只验证 ruff、grader 自测、fixture/adapter/CLI 和脚手架状态。学生实现故意未完成，因此默认 CI 不把 student tests 的失败当作框架损坏。Actions 页面还提供手动运行的 `Student tests` workflow。

## 许可与边界

代码以 MIT License 发布。CS336 名称仅作来源说明。请在学习时遵守你所在课程的 academic integrity（学术诚信）政策；AI 最适合解释形状、数学和失败原因，而不是替你填完所有 TODO。
