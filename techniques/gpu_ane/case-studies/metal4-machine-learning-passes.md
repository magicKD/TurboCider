# Metal 4 machine-learning pass：公共模型入口与 GPU/ANE 自动选择

<!-- knowledge
id: technique-gpu-ane-case-studies-metal4-machine-learning-passes
type: case-study
operator: scheduling
backend: metal, coreml, private-ane
device: cpu, gpu, ane
symptom: straggler, memory-bound
semantics: contract-only
boundary: request
api: public, private
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [硬件/API](../hardware-and-tensorops.md) · [来源快照](../sources.md#metal4-public)

核查 2026-10-16。这是与普通 MSL TensorOps、`MLModel.predict`、私有 ANE bridge **分别建模**的公共入口。本轮阅读 Apple 四份官方正文、示例包的 11 个选定文本文件及 13 个公共 SDK header 的指定范围；运行了独立小客户端，不运行或分发官方示例实现。

本页数值与源码计数是第四轮快照；第五轮已继续验证[GPU→buffer-backed模型pass→GPU消费者](metal4-consumer-layout-and-handoff.md)，新增ML usage的offset0限制及实际padded stride，不改写这里两shape/168输出的历史范围。

## 1. “GPU timeline”不是物理设备的充分判据

Apple [Machine learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes) 明确说明：系统可以为每个模型自动选择 GPU 或 Apple Neural Engine；若选择 ANE，GPU 还可执行独立的 render/compute 工作。**因此不能把 Metal 4 ML pass 一概说成 GPU-only，也不能把它直接称为已实现 GPU+ANE 并行。**

这与 M5 GPU core 内的 NAX 不同。普通 `matmul2d`/cooperative tensor kernel 是 GPU 执行表达；`MTL4MachineLearningCommandEncoder` 则以预转换网络为输入，并由系统决定相应推理引擎。API 在同一 timeline 组织工作，不等于每个 pass 都运行在同一个物理单元。

所读 pipeline descriptor 提供函数和输入尺寸配置，并没有本轮可用的“强制 ANE 且报告实际 placement”选项。不要直接把 `MLModelConfiguration.computeUnits` 套到另一个 API，也不要把 header 可见或 API 成功当真实 ANE 计数。

## 2. 构建与运行链

官方 [详细示例](https://developer.apple.com/documentation/metal/running-a-machine-learning-model-on-the-gpu-timeline) 和包内代码对应如下：

```text
Core ML .mlpackage
  → metal-package-builder（开发期转换）
  → Metal ML .mtlpackage / library
  → main 函数 reflection + 指定输入尺寸
  → MTL4Compiler → MTL4MachineLearningPipelineState
  → tensor bindings + argument table + placement intermediates heap
  → MTL4MachineLearningCommandEncoder / dispatchNetwork
  → stage barrier 或 shared event / feedback
  → 合法的下游 tensor consumer
```

转换命令由官方 README 给出：

```sh
xcrun metal-package-builder -ml example.mlpackage -o example.mtlpackage
```

官方说明该工具随 Xcode 26+ 提供。当前激活的 Command Line Tools/SDK26.5 中 `xcrun --find metal-package-builder` 返回 72；这不证明机器上绝无其他 Xcode，也不表示所有 ML pass 都无法执行。

下载的官方 70,065-byte ZIP 已包含一个预生成小包，manifest 的类型为 `MLLibrary`，内容指向 `library.mpsgraphpackage`。本轮只提取其四个 data/manifest 文件，共 1,671 bytes，**不需要 builder**便可尝试加载。没有提取/执行 ZIP 内 `.git`、hooks 或源码工程。该包的具体表示同样不是 ANE 证据。

## 3. 动态形状要变成具体 pipeline 身份

`MLMatrixMultiplier+PipelineCompilation.m:26–97` 先取 `main` function reflection，找出两个输入并按名字映射，设置尺寸再编译；`TensorSetup.m:49–64` 拒绝还带动态维度的结果绑定。

官方正文要求：动态输入的每种唯一尺寸组合需要相应 pipeline state，不是一个编译物自动覆盖所有 token bucket。因此模型内容、入口、shape、dtype、编译选项与环境应进入 cache 身份；多 bucket 预编译要计加载/驻留成本。这是迁移建议，不是本轮测到了的 cache 效果。

binding 顺序没有保证，须根据 name/index/type/dimensions 配对。示例按包含 `input` 的名字筛选、假定 rank2/两个输入，是它的 toy ABI，不能移植成任意 Transformer/DiT 的通用反射器。

`MTLTensorExtents` 的第 0 维是**最内层**：row-major A[M,K] 用 `[K,M]`，B[K,N] 用 `[N,K]`，output[M,N] 用 `[N,M]`。不要将常见外到内 shape 数组原封不动传入。

## 4. 两种 stride 合同，不要混成“零拷贝”

SDK `MTLTensor.h` 区分：

| 用途 | stride/存储要求 | 学习要点 |
| --- | --- | --- |
| CPU `replaceSlice` / `getBytes` | stride 以元素计，合法单调/不交叠的 host layout | 可以传 dense host 数组；这些方法本身就是数据传输 |
| 从 MTLBuffer 创建 tensor 的 descriptor | 第一个 stride=1；ML usage 的第二 stride 须满足 64-byte 对齐，后续维度按前一 stride×extent；ML buffer offset必须0；sub-byte 的额外对齐为 128 bytes | 元素数换成字节/bit 后检查，不能拿字节值直接作 element stride或沿用任意subbuffer offset |
| 独立创建的 tensor | `buffer` 可为 nil，实际 `.strides` 仅适用于 buffer-backed tensor | 不把 host dense layout 当内部 physical layout |

例如 FP32、逻辑行宽17的 buffer-backed ML tensor，dense stride17对应68 bytes，不满足该64-byte条件；至少可选 stride32，对应128 bytes。还须满足具体 buffer 大小、offset、usage 和消费者约束。本轮未运行 buffer-backed 或 sub-byte 变体。

SDK26.5 枚举中 INT4/UINT4 标注 macOS26.4；它只是数据格式/API 可用性，不能推出目标算子会用整数 MAC、支持任意 GGUF packing 或比 FP16 快。

## 5. Residency、scratch、同步与失败

示例 `MLMatrixMultiplier.m:221–248` 把 tensor 的 GPU resource ID 绑定到反射 index，并把 tensor 与 intermediates heap 入 residency set。ML encoder 的 scratch 使用 placement heap，至少满足 `pipeline.intermediatesHeapSize`；并发推理不能未经别名/生命周期检查就共享这块 scratch。

生产者 compute→ML→消费者 compute 的依赖需围绕 `machineLearning` 与对应 stage 编排 barrier/fence；同一 command buffer 不消除依赖，residency 不替代内存可见性。CPU 读取则等 shared event 或 terminal feedback。没有依赖的 GPU 工作才是 ANE 自动放置后潜在的并行候选。

官方样例是入门教材，不是完整生产失败合同：

- `MLMatrixMultiplier.m:281–285` 超时只 log，方法为 void；`main.m:56–57` 随后仍取 tensor 读出。生产接入应传播失败并禁止输出发布/复用，不应照搬这个简化路径。
- `Matrix.m:189–191` 以 `fabs(diff)>tolerance` 拒绝误差；NaN 会使这个比较为 false。CPU 单元测试保留这个数学负例，实际 validator 先检查 finite。

以上是所读代码的静态控制流/数值语义；**没有注入官方 sample 超时或 NaN，也不声称复现了它的实机故障**。

## 6. 本机小包实测：两种 shape 可执行，物理 placement 未知

独立客户端 和 显式驱动 使用已固定 SHA 的官方模型数据、公共 API 与独立 FP64 oracle。host 端先 `-Wall -Wextra -Werror` 编译；首次运行准入读到36%即拒绝，没有启动设备 child。期间继续源码/文档工作；后来新目录重试，五次采样均为0才运行。

| M / K / N | 全输出检查 | max abs | pipeline build ms | submit→event+feedback ms |
| --- | ---: | ---: | ---: | ---: |
| 3 / 7 / 5 | 15 | 0 | 141.113 | 2.780 |
| 17 / 33 / 9 | 153 | 0 | 55.590 | 2.755 |

两种专门化各运行一次；rank、dtype、反射绑定、event、feedback/error、全部168个输出均检查。输入是小型可精确表示的 dyadic 值，故零误差不证明任意 FP32 fidelity、实际 accumulator 位宽或模型质量。两图要求的 intermediate heap 都为64 bytes，客户端保守分配4096 bytes；这不是总驻留/峰值内存。

输入/输出使用 host tensor copy。计时未含全部加载、copy、分配与 CPU 验证，没有对照优化 GEMM，也没有做 warm 统计。**这些数字只描述本次执行边界，不能与普通 compute kernel、Core ML predictor 或 MLX 做速度排名。** 官方也提醒玩具 matmul 不是推荐用模型 pass 加速单个矩阵乘的性能示范。

此次实际推理单元记为 `not-observed`，没有 ANE trace、能耗或设备 overlap；不借官方自动放置说明把结果标成 ANE。SDK 支持、包加载、图执行、数值合格、物理放置和完整模型提速仍是不同证据。

## 7. 重放与下一步

下载地址和 ZIP SHA 见来源清单；只接受本轮固定归档，活跃 URL 内容改变时应先重审，而不是放宽校验。模型数据和官方源码不随 Wiki 分发。

下一步是实际 compute→ML→compute 的无 host round-trip 消费者、buffer-backed stride、bucket cache、真正模型图与独立 GPU 工作；同时补物理 ANE 放置/overlap 证据。不能从本轮两种 matmul 专门化推断所有 Core ML operator 都能转换、所有图都自动使用 ANE或已有端到端加速。
