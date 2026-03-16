# 变更影响范围分析 (Impact Scope Analysis)

基于 `约定式提交 (Conventional Commits)` 规范，本次核心算子集成（包括但不限于 Clip, Conv, LayerNorm, UnaryOps 等）到 InfiniTensor 框架所涉及的影响范围分析如下：

- `feat(frontend)`: 新增前端 Python 层算子注册机制与 PyTorch `ATen` 算子到框架内部算子的转换映射；新增 `GraphBuilder` 构建接口以及 `PyBind11` 的 Python 层接口暴露。
- `feat(ir)`: 新增中间表示（IR）层算子图结构定义对象（`OperatorObj`），包括形状推导（Shape Inference）、数据类型推导（DataType Inference）及算子描述符创建功能。
- `feat(kernel)`: 新增设备端算子的 Kernel 层计算实现，在 `compute` 接口中建立起对 InfiniCore 库中对应 C-API 算子函数的调用和执行逻辑。
- `test(frontend)`: 新增前端 API 单元测试与 PyTorch 转换映射正确性验证。
- `test(backend)`: 新增后端图算子级别及 Kernel 层面的 C++ 测试用例，包括数值精度比对验证与多硬件平台算子调用验证。

---

# User Stories

## Story 1: 前端与 PyTorch API 层面对接 (Frontend & PyTorch Integration)

**1. 明确的业务价值与验收标准 (Definition of Done)**
- **业务价值**：打通框架用户交互层，确保用户能够使用类似 PyTorch 的编程接口自然地调用该算子，从而降低模型迁移到 InfiniTensor 框架的使用成本。
- **验收标准 (DoD)**：
  - 在 `ConverterRegistry` 注册表中完成目标算子（及其所有支持的 Overload）的注册。
  - 完成算子转换映射函数的编写，将 PyTorch FX Node 参数成功解析为 InfiniTensor 内部算子所需的属性。
  - 完成 `GraphBuilder` 中的 C++ 算子构建接口声明与实现，处理好可选输出等不同分支情况。
  - 完成 PyBind11 的接口绑定逻辑，向 Python 层暴露新增的 `GraphBuilder` 接口。
  - 前端静态代码检查无报错，对应算子的 Python 端单元测试通过。

**2. 前置依赖、输入输出及接口变动清单**
- **前置依赖**：需明确目标算子在 PyTorch 中的 ATen 名称及其重载签名（如 `aten.mul.Tensor`）。
- **输入输出**：输入为上游传入的 PyTorch FX Graph 节点（包含输入张量引用及字典属性），输出为映射转换后构建的 InfiniTensor `Tensor` 对象。
- **接口变动清单**：
  - `python/src/infinitensor/converter/unified_converters.py`（新增对应的转换器及注册逻辑）
  - `include/core/graph_builder.h` 与 `src/core/graph_builder.cc`（新增对应的算子构建方法）
  - `python/bindings/graph.hpp`（新增 `GraphBuilderObj` 的 `def` 绑定代码）

**3. 测试策略与回归范围**
- **测试策略**：在 `python/tests/` 目录下编写基于 `pytest` 的前端测试。传入模拟的 `torch.fx` 图节点，断言转换输出的对象是否符合预期的算子类型和属性值。
- **回归范围**：现有算子的 PyTorch 到 FX Graph 转换逻辑。需确保新增的注册未污染 `ConverterRegistry` 映射机制或覆盖已有的同名算子行为。

**4. 预计工作量与优先级**
- **预计工作量**：2 人日
- **优先级**：高 (P0)

---

## Story 2: 后端算子图层面定义与实现 (Backend Graph IR Definition)

**1. 明确的业务价值与验收标准 (Definition of Done)**
- **业务价值**：为框架的中间表示层（IR）提供算子的完整语义定义，使图优化阶段与内存规划调度器能准确获取算子的输入输出依赖、形状与类型信息。
- **验收标准 (DoD)**：
  - 继承自 `OperatorObj` 的算子类实现完毕，完成必要的类构造及虚函数（如 `toString`）的重写。
  - 算子形态推导函数 `inferShape` 可正确处理各类合法输入及特殊情况（如广播机制、标量计算）。
  - 数据类型推导函数 `inferDataType` 逻辑正确。
  - `createOpDesc` 能够根据前端属性正确组装出对接底层的算子描述符对象。
  - 图层面 C++ 单元测试验证通过。

**2. 前置依赖、输入输出及接口变动清单**
- **前置依赖**：Story 1 中约定的 `GraphBuilder` 接口输入参数；InfiniCore 底层对应算子的描述符参数结构。
- **输入输出**：输入为前置节点的输出张量维度、类型以及具体算子属性；输出为推导确定的新张量维数/形态、数据类型以及供底层使用的 `OpDescriptor`。
- **接口变动清单**：
  - `include/operators/xxxop.h`（新增具体算子类声明与方法签名）
  - `src/operators/xxxop.cc`（新增算子属性装配、形状类型推断与描述符生成逻辑实现）

**3. 测试策略与回归范围**
- **测试策略**：在 `test/operators/` 目录下编写 C++ 测试用例，人为构造多组输入边界（不同形状大小、动态维度占位符等），执行推导函数并验证返回形状或抛出的维度不匹配错误是否符合预期。
- **回归范围**：框架统一的计算图拓扑遍历及内存规划逻辑，需确保新增算子的 IR 节点不会导致图调度引擎发生循环依赖或内存分配死锁。

**4. 预计工作量与优先级**
- **预计工作量**：3 人日
- **优先级**：高 (P0)

---

## Story 3: 后端算子 Kernel 层面实现与多平台验证 (Backend Kernel Implementation)

**1. 明确的业务价值与验收标准 (Definition of Done)**
- **业务价值**：实现框架逻辑与底层硬件的高效互通，通过调用 InfiniCore 算子库实现计算的物理下沉，达成“全链路调通”的核心目标。
- **验收标准 (DoD)**：
  - Kernel 的 `compute` 方法能够成功获取并分配该算子底层执行所需的 `Workspace` 工作区大小。
  - 成功调用对应的 `infiniopXXX` 执行函数处理设备内存数据的计算。
  - 计算逻辑绑定到具体的硬件上下文 Stream，并通过宏 `REGISTER_KERNEL_ALL_DEVICES` 正确完成算子与后端的注册绑定。
  - 针对该算子的后端 Kernel 计算精度测试通过（相对 CPU 纯量运算实现验证）。
  - （结合 Story 4）确保在非 CPU 的异构加速卡平台上顺利执行不报错。

**2. 前置依赖、输入输出及接口变动清单**
- **前置依赖**：Story 2 完成算子 IR 节点定义；InfiniCore 库中已提供该算子稳定可用的执行 API 及空间大小查询接口。
- **输入输出**：输入为运行时对象（`RuntimeObj`）、算子描述符以及 InfiniCore 库分配的内存工作区指针；输出为通过硬件流（Stream）异步或同步写入到显存目标地址的运算结果。
- **接口变动清单**：
  - `src/kernels/xxxop.cc`（新增特定 Kernel 类的 `compute` 方法实现及底层 API 桥接）

**3. 测试策略与回归范围**
- **测试策略**：
  - 编写 `test/kernels/test_xxx_kernel.cc` 测试文件。
  - 构造具有随机初始化数据的张量作为输入，对比本算子调用 InfiniCore 计算输出与参考 CPU 裸函数计算输出，检查误差是否在特定数据类型（如 Float32、Float16）的阈值以内。
  - 配合不同 Xmake/CMake profile，在多平台（英伟达、沐曦等）环境中验证算子正确性。
- **回归范围**：设备的 Runtime 内存池申请释放行为，确保算子调用期间的临时空间借用未引起 OOM 或其他算子的显存读写越界。

**4. 预计工作量与优先级**
- **预计工作量**：4 人日
- **优先级**：最高 (P0)