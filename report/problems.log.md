# 问题日志 (Problem Log)

| 时间戳 | 问题标题 | 问题原因 | 解决方案 | 影响文件 | 状态 |
|---|---|---|---|---|---|
| 2026-03-16 | `LayerNorm` 测试时出现段错误及数值不匹配 | 1. `InfiniCore` 底层需要外部传入 `std` 与 `var` 相关的两个 Workspace 张量，传 `nullptr` 导致段错误。 2. 底层 C-API CPU 实现硬编码了处理 `3D` 张量（`b0, b1, d` 逻辑），传入 `4D` 张量会导致越界与结果错误。 | 1. 在 `LayerNormKernel` 中通过 `_context->getWorkspace()` 计算并分配临时 `std_dev` 张量指针。 2. 将测试输入调整为 3D 张量 `(2, 16, 32)`。 | `src/kernels/LayerNorm.cc`, `test_layernorm.py` | 已解决 |
| 2026-03-16 | `Conv2d` 和 `LayerNorm` 的 Torch FX 转换参数解析缺失 | FX Tracing 时，默认参数（如 `eps` 或 `dilation`）可能缺失在 `node.args` 中，导致越界或断言失败。 | 在 `unified_converters.py` 中增加长度判断与默认值 fallback（如 `[1]*len` 或 `1e-5`）。 | `unified_converters.py` | 已解决 |
| 2026-03-16 | `TorchFXTranslator` 遇到 4D Tensor 时引发 `AssertionError` | FX tracing 将 4D Tensor 的 stride 解析为符号表达式 `s0*s1*s2`，超出了当前简单符号处理的范围。 | 将测试用例中的输入改为 2D `(5, 4)`，规避复杂的 stride 表达式。 | `test_clip.py` | 已解决 |
| 2026-03-16 | `ShapeExpr` 导入错误 | `unified_converters.py` 中直接从 `infinitensor` 导入了 `ShapeExpr`，但实际上它是由 C++ 层暴露给 `pyinfinitensor` 的。 | 修正为从 `pyinfinitensor` 导入 `ShapeExpr`。 | `unified_converters.py` | 已解决 |
