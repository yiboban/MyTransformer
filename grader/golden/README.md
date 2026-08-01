# Golden fixture 位置

可执行的 tiny golden fixture 与其 SHA-256 manifest 统一保存在 `tests/fixtures/`，因为它们既是学生测试输入，也是 `grader_tests/test_fixture_integrity.py` 的校验对象。

本目录只保留位置说明，避免在 grader 与 tests 中复制两套二进制数据并产生版本漂移。fixture v1 由开发阶段已删除的独立临时实现生成；仓库中不含生成器或参考答案。
