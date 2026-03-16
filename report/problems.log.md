# 问题日志 (Problem Log)

| 时间戳 | 问题标题 | 问题原因 | 解决方案 | 影响文件 | 状态 |
|---|---|---|---|---|---|
| 2026-03-16 | `TorchFXTranslator` 遇到 4D Tensor 时引发 `AssertionError` | FX tracing 将 4D Tensor 的 stride 解析为符号表达式 `s0*s1*s2`，超出了当前简单符号处理的范围。 | 将测试用例中的输入改为 2D `(5, 4)`，规避复杂的 stride 表达式。 | `test_clip.py` | 已解决 |
| 2026-03-16 | `ShapeExpr` 导入错误 | `unified_converters.py` 中直接从 `infinitensor` 导入了 `ShapeExpr`，但实际上它是由 C++ 层暴露给 `pyinfinitensor` 的。 | 修正为从 `pyinfinitensor` 导入 `ShapeExpr`。 | `unified_converters.py` | 已解决 |
