# Runtime-weight Core ML：组件原型与收尾记录（2026-09-28）

后续进展：当日已完成 base 模型显式路由接入，见
[两模型集成与整请求筛选](runtime-ane-integration-2026-09-28.md)。
本页保留**集成前**的组件证据与边界；下方“尚未接入”是当时状态。

状态：**optional 单层研究原型，不是已交付的整模型后端**。
保留公开 Core ML 导出器、执行器、独立探针和测试；不加入 shipping
构建、App、生成 CLI、`execution=auto` 或默认 `make test`。
既有 GPU / 冻结权重 Core ML 最快路线保持不变。

## 三种路线不要混淆

| 路线 | 图中权重 | LoRA 与切分 | 验证范围 |
| --- | --- | --- | --- |
| 现有 base GPU/Core ML | 每层冻结 base 权重 | 主要按 FFN intermediate channels 切分；长序列可循环 tile | 已有整图证据，保持原选择策略 |
| 现有 `lora_fused` | 每层冻结 base 权重，**不含 LoRA** | GPU 动态 gate/up 修正在 SiLU 前输入图；hidden 返回 GPU 计算 down-LoRA | 同一 base 图可供 base 和不同 LoRA 复用；Z 有收益，Qwen 无稳固优势 |
| 新 runtime-weight 原型 | 无 checkpoint 矩阵常量；权重是运行时输入 | 全宽 FFN 按 token rows 分给 GPU/Core ML；目前**无 LoRA 支持** | 单层探针、换权重及小图验证；尚未接入模型 |

用户要求的“不融合 LoRA 到 base checkpoint / Core ML artifact”不变。
新原型只 staging base 矩阵，未实现临时或持久 LoRA 合并，也不能用它替代
现有完整 runtime LoRA 路线。这里的“共图”是**相同几何**共用图，
不是 Z 与 Qwen 不同 H/F 形状强行共用一个产物。
既有整图结果与 CLI 示例见[维护总览](runtime-lora-acceleration-2026-09-28.md)。

## 保留的实现

| 文件 | 职责 |
| --- | --- |
| `tools/coreml/export_runtime_ane.py` | 离线公开 coremltools 导出 Matmul / SwiGLU / GELU；`[out,in]` 权重输入；K/N 分块在图内；保存源 package、编译图和 SHA256 receipt |
| `native/backends/ane_runtime.hpp` / `.mm` | 一个图、单层 IOSurface 权重槽、固定 chunk I/O、持久调度线程、独立转换池；图接口/receipt 校验及失败状态 |
| `native/backends/ane_runtime_convert.hpp` | BF16/FP16/FP32 → FP16 的 NEON staging；保留 scalar 对照；BF16 输出舍入 |
| `tools/native/ane_runtime_probe.cpp` | 编译 MLX 全 GPU FFN 对照、GPU head/Core ML tail 并行；真权重/捕获输入可选；两次 warmup 后交错计时 |
| `tools/native/build_ane_runtime_probe.sh` | 仓库依赖解析，C++20、ARC、严格 warnings；不依赖兄弟项目或本机硬编码路径 |
| `tests/native/test_ane_runtime.py` / `ane_runtime_convert_test.cpp` | 几何、无常量矩阵、转换、自检、真实小图调用、多 chunk、权重切换、无效存储、篡改拒绝和溢出恢复 |

显式入口为 `make build-runtime-ane-probe` 和 `make test-runtime-ane`。
`tests/repository/test_layout.py` 防止原型误入默认构建/打包/测试，并检查
新工具不引入用户目录或兄弟仓库路径。运行时用公开 Core ML API；
没有复制参考项目源码或引入未公开的 compiled-model emitter。

## 性能：只证明组件潜力

M4 Max 64 GB、macOS 26.6.2、coremltools 8.3.0。
真实 BF16 权重 + 既有捕获输入，1024 行，其中 GPU 768 行、Core ML 256 行；
chunk=256、内部 K/N tile=1024、SiLU lowering=`exp`、SIMD staging。
每个模型一次进程，2 次 warmup 排除，8 次交错测量取中位。
两模型串行测试；事前进程检查未发现活跃的本地推理任务，仍有 Codex/opencode
与桌面进程，**并非独占 GPU/ANE 的证明**。

| 最终版单层复测 | 全 GPU | 并行计算/收集 | 权重 staging | 并行含 staging | 含 staging 加速比 | 相对 L2 / cosine |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen block 0，teapot step 1 | 21.098 ms | 16.533 ms | 2.492 ms | 19.022 ms | **1.109×** | 0.001682 / 0.999999 |
| Z `noise_refiner.0`，lighthouse 捕获 | 16.565 ms | 13.086 ms | 2.084 ms | 15.179 ms | **1.091×** | 0.001269 / 0.999999 |

各列分别取中位，不要求相加精确相等。GPU 基准是探针的编译 MLX FFN，
不是 shipping 整模型最优 GPU 路线；权重加载、自检、graph load 不计入热态。
`split` 包含 Core ML 输入转换、输出复制、GPU tail 张量组装/拼接与等待；
`with_stage` 再包含每次权重 staging。**没有 attention，因此未隐藏 staging**。
不含 text encoder、attention、采样、VAE、PNG 或模型调度成本。
排除 staging 的 1.276× / 1.266× 不能报告为真实整图收益。

Z 首次 warmup 溢出后重试两次，headroom=16；计时重复沿用此缩放，
故不是“每层首次遇到溢出”的性能。Qwen headroom=1、无重试。
Z 最大绝对误差 1024 来自大幅度 BF16 输出；相对误差较小不等于视觉通过。
两组均未重跑整图生成或肉眼质量验收，也没有 LoRA。
显式 IOSurface 槽约 Qwen 292 MiB / Z 229 MiB，保守 admission 估计
约 666 MiB / 537 MiB；这些不是实测进程峰值或无 swap 证据。

本地复测证据来源（均为相对路径，不随源码分发）：

- Qwen checkpoint：`models/Comfy-Org-Qwen-Image-2.1/diffusion_models/qwen_image_2.1_bf16.safetensors`；
  输入：`results/qwen21/qwen21-w8a8-calibration-512-teapot/block0/step1.npy`；
  图：`outputs/runtime-ane/qwen21-c256-t1024-exp/manifest.json`。
- Z checkpoint：`models/Comfy-Org-z_image_turbo/split_files/diffusion_models/z_image_turbo_bf16.safetensors`；
  输入：`outputs/z-w8a8-heldout-8step-lighthouse/capture/block0/sample-21100-126519308805166-160.npy`；
  图：`outputs/runtime-ane/z-image-c256-t1024-exp/manifest.json`。

### 之前筛选的有效经验（非最终版复测）

- 512 行/一半 token 分配给 Core ML 偏慢：合成 Qwen GPU 21.118 ms，
  split 23.728 ms；Z GPU 17.647 ms，split 19.031 ms，均尚未计 staging。
- 256 行改善平衡，但 scalar staging 抵消收益：Qwen 约 6.2 ms、
  Z 约 5.0 ms。SIMD 降至约 2.3–2.5 ms / 1.9 ms；早期同二进制
  scalar → SIMD → SIMD → scalar 筛选中，两模型含 staging 约 1.11×。
- 真实 Qwen block 3/15/31 早期含 staging 为 1.117× / 1.140× / 1.115×。
  在最后 BF16/headroom 输出改动之前测得，不能视作最终版全层验收。
- `builtin` 和 `sigmoid` SiLU lowering 在大尺寸 CPU+NE 自检失败，
  CPU-only 通过；保留为显式对照，默认 `exp`，**没有放宽自检阈值**。
  `exp` 使用 `x / (1 + exp(-x))`，保留非线性；FP16 有限动态范围
  与舍入仍会产生近似，不能称位级等价。

单层收益尚不能与现有 Z base 约 1.31× 整请求成绩比较，更不能据此声称
runtime-weight 胜过冻结图。性能损失已经观察到 staging 与切分失衡；
attention 重叠、模型级同步/共享带宽的实际影响还需要集成后测量。
`CPUAndNeuralEngine` 只是调度策略，物理 ANE residency **未知**；
本原型是 FP16 runtime-weight，也不是 W8A8/INT8 吞吐实验。

## 正确性约束与已修问题

- 必须先在实际形状跑两个稀疏权重集的自检，再 staging；负值、转置、
  激活和换权重都检查。失败 staging 不能复用旧权重；短 input/output 被拒绝。
- 输出使用 `getBytesWithHandler` 有限期读取，避免 `dataPointer`
  长期锁住 IOSurface 导致下一次 prediction 失败。输出是有计时的复制，非零复制。
- 持久 worker 用 `std::exchange(job_, {})` 取任务，显式清队列；
  不能依赖 moved-from `std::function` 必定为空，避免重复执行/提前 join。
- FP16 非有限输出对 SwiGLU 的 up / GELU 的 down 权重作 4 倍 headroom
  重试，上限 4096，再恢复到 BF16；不缩放 GELU 非线性输入。
  数学齐次性不消除有限精度误差，缩放可能增加下溢；gate 或 GELU 中间
  溢出等情况不保证可修复，最终仍必须返回失败，不能接受 NaN/Inf。
- 任意失败可能已经写出部分 chunk；未来模型调用者必须对**整个请求区间**
  在 GPU 重算。当前低层只返回错误，探针中止，未提供模型级 fallback。
- 公共调用由单一 host owner 串行驱动；借用权重、输入和输出需存活到
  `wait_stage()` / `finish()`。不是多调用者队列，不支持运行中取消。

## 复现入口

在仓库根目录、Apple Silicon macOS 和已配置的项目 MLX/coremltools
环境下运行；默认 Makefile 优先使用 `.venv` / `.deps` 的 Python。
导出目标必须不存在，避免覆盖已有实验。生成物放已忽略的 `outputs/`。

```sh
make build-runtime-ane-probe
make test-runtime-ane

# 先验证小图，不需要下载真实 checkpoint
.venv/bin/python3 tools/coreml/export_runtime_ane.py \
  --kind swiglu --rows 32 --hidden 64 --width 96 \
  --tile-k 33 --tile-n 47 --output outputs/runtime-ane/example-small
build/native/ane-runtime-probe \
  outputs/runtime-ane/example-small/manifest.json 8 97 ne - 2 simd

# Qwen 几何；导出不需要 checkpoint，也不包含 LoRA
.venv/bin/python3 tools/coreml/export_runtime_ane.py \
  --kind swiglu --rows 256 --hidden 4096 --width 12288 \
  --tile-k 1024 --tile-n 1024 --output outputs/runtime-ane/example-qwen
build/native/ane-runtime-probe \
  outputs/runtime-ane/example-qwen/manifest.json 8 1024 ne - 1 simd
```

若环境不在 `.venv`，替换为安装项目依赖的 Python；不要硬编码开发机路径。
Z 形状改为 `--hidden 3840 --width 10240` 并选择新导出目录。
`-` 表示合成权重；真实实验完整参数为：

```text
ane-runtime-probe MANIFEST REPEATS ROWS cpu|ne BUNDLE_OR_CHECKPOINT ANE_CHUNKS simd|scalar INPUT_NPY BLOCK
```

Qwen block 为 0–31；Z 0/1 为 noise refiners，2–31 为 layers.0–29。
最后两个参数可省略；提供捕获 input 时其 reshape 行数必须与 ROWS 一致。
`real_weights` 字段仅表示传入了 safetensors 文件（也可能是合成测试 bundle），
不是“已测试真实模型”证明；必须同时记录 checkpoint 与输入来源。
此处 manifest 与 shipping 冻结图 ABI 不同，**不要传给生成 CLI 的
`--ane-manifest`**。

## 尚未完成，不作为本轮收尾的隐含交付

1. 两模型 pipeline 的 pre-attention staging、tensor lifetime 管理与
   GPU failed-range 重算；保留现有 GPU kernel 和冻结图选择。
2. adaptive token-row 调度、GPU-only 退路、request/plan/result 状态及取消。
3. runtime LoRA 动态修正接口；当前原型不支持 adapter 或 Q4/Q8 staging。
4. 与统一内存 planner/实际压力联动、实测峰值/无 swap；receipt 目前是
   hash-then-load，不是防并发修改的不可变 artifact lease。
5. base/LoRA、512/1024、1–3-reference 编辑的配对整请求测试与肉眼验收；
   先确认其他 AI 推理负载，繁忙时等待，不终止他人进程。

本轮收尾已确认 6 项 runtime 组件测试通过（含 CPU-only 和 CPU+NE 小图、
全部 BF16/FP16 位型 SIMD/scalar 对照及 headroom 回归），并用最终代码
复测上表两个真实单层。历史原始图、输入和编译缓存本地保留；不把生成物
加入源码，不因文档整理清空模型或缓存。其他收尾回归见维护总览。
