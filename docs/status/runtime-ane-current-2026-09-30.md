# Runtime ANE：当前代码、验证进度与提交边界

更新于 2026-09-30。本页是代码整理后的**当前状态**，逐轮参数、原始结果与
负结果仍保留在[加速决策入口](acceleration.md)及其链接的专项报告。
本次源代码整理不改变 FFN/QKV 数学、图协议或默认选路；以下性能仍属于
各自的历史构建，不重标为本次重新构建的成绩。

## 代码职责和已实现能力

| 层次 | 代码 | 当前行为 |
| --- | --- | --- |
| 请求与计划 | `native/runtime/plan.cpp`、`native/runtime/session.hpp`、`apps/cli/main.mm` | 显式 `gpu_ane`、manifest、`hybrid_mlp_mode=runtime` 或研究用 `runtime_qkv`；近似授权与资格门禁保留，普通请求仍走 GPU |
| 通用 runtime | `native/backends/ane_runtime.{hpp,mm}`、`ane_ffn.{hpp,cpp}`、`ane_scheduler.hpp` | checkpoint-independent 固定形状 Core ML runtime-weight 图，层间更换权重；GPU/ANE token-row 分割、多 chunk、异步 staging/worker、自适应整块探测与失败后完整 GPU 重算 |
| 内存和量化 | `ane_memory.{hpp,cpp}`、`ane_runtime_{convert,quant}.hpp` | BF16/FP16 与 Q4/Q8 权重转换、可选图加载/请求/scratch 内存准入、FP16 headroom 恢复；已校验图使用私有快照，仍非完整设备内存认证 |
| 模型 | `native/models/z_image/`、`native/models/qwen21/` 及对应 module | Z BF16 和原生 Q8 GGUF base；Qwen BF16 base、v2 共图 runtime LoRA、1–3 图编辑的受限路径；模型自己的 GPU attention/残差与 LoRA 修正不复制到通用后端 |
| QKV 研究 | `ane_qkv.{hpp,cpp}`、`ane_qkv_scheduler.hpp`、Qwen pipeline | 1024² resident base 三份独立 Q/K/V 权重直填一张 MatMul 图；保留 GPU norm/RoPE/attention/FFN，固定 chunk 或显式 auto 一 chunk 开/关，不与 FFN ANE 同时启用 |
| 离线工具和验收 | `tools/coreml/export_runtime_ane.py`、`tools/native/ane_runtime_probe.cpp`、`tools/validation/runtime_ane_*.py`、`tests/native/` | 图导出、真实权重组件探针、GPU/runtime/冻结图整请求对照、LoRA 状态切换、独立进程树采样、host 与 Core ML 小图回归；不在产品请求里运行 Python SDK |

LoRA 不合入 base checkpoint、Core ML 图或 runtime base slots；v1 图只供
无 adapter 的 base 请求，完整 gate/up 非线性修正及 down-LoRA 必须使用
v2 图。冻结 base 仍是独立、受 checkpoint/shape 限制的路径，不把图类型
混用。`cpuAndNeuralEngine` 仅是 Core ML 请求策略，现有遥测不能证明
实际 ANE 驻留或 GPU/ANE 物理重叠。具体生命周期/所有权约束见
[后端维护说明](../../native/backends/README.md)，复现入口见
[验证工具说明](../../tools/validation/README.md)。

## 已完成的同库整请求对照

M4 Max 64 GB、resident、完整 `request_wall` 热请求（包含 VAE/PNG，
排除冷请求）；每行只与**本行的同一构建、同一输入、相同 GPU 开关**比较：

| 工作负载 | 库 SHA 前缀 | GPU | runtime FFN | 冻结 base | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen BF16 1024²/40 步，狐狸，Q/K 融合开 | `875b0d3d` | 183.086 s | v1 144.564 s | 160.285 s | runtime/GPU 1.266×；三路正反序各一热，仍显式 optional |
| Z BF16 1024²/8 步，灯塔/肖像 | `875b0d3d` | 31.429 / 31.401 s | v1 28.400 / 28.388 s | 无匹配分母 | 1.107×/1.106×；各四热正反序 |
| Z Q8 GGUF 1024²/8 步，灯塔/肖像 | `875b0d3d` | 40.891 / 40.894 s | v1 35.198 / 35.227 s | 未测 | 1.162×/1.161×；各四热正反序 |
| Z BF16 512²/8 步 | `28ff6aaf` | 6.997 s | 6.399 s | 5.347 s | 同组三路冻结图最快 |
| Qwen BF16 512²/40 步，Q/K 融合开 | `28ff6aaf` | 41.744 s | 35.592 s | 29.062 s | 同组三路冻结图最快 |

证据分别见 [Qwen v1](runtime-ane-v1-base-2026-09-30.md)、
[Z BF16](runtime-ane-z-1024-v1-2026-09-30.md)、
[Z Q8](runtime-ane-z-gguf-q8-1024-2026-09-30.md)和
[图快照/512² 对照](runtime-ane-artifact-lease.md)。不可跨构建或
checkpoint 拼接绝对时间/冻结分母；Qwen LoRA/编辑也不能套用 base 加速。
当前 Qwen `runtime_qkv` 在 875b 库的同融合 1024²/40 步正反向四路线
两热均值为 GPU/QKV auto/FFN runtime/冻结图
183.077/182.961/147.001/151.766 s。QKV auto 仅 1.001× GPU，
不足以替换 FFN runtime 或默认 GPU；详见[独立记录](runtime-ane-qwen-qkv-auto.md)。

近期 Z Q8 肖像 fixed-3/fixed-4 单 batch pilot 热中位分别 37.190/35.315 s，
与 auto 35.227 s 不属于交错 A/B；固定 3 较慢，固定 4 不证明更快，
因此没有改变调度。Q8 肖像 auto 两个 runtime trial 的 PNG 字节不同，
且一个 GPU 采样窗口有内存压缩；目视没有明显崩坏，不是广泛画质资格。

## 本次代码整理与尚未完成

- Qwen FFN/QKV 共用一个内部可选内存预算计算，先扣 4 GiB 系统余量，
  限制在 2 GiB 内，避免异常大活动内存计数加法溢出伪造余量。
- 删除 Makefile 中重复的 `package` 帮助行；维护文档改正 Qwen QKV
  已接入显式研究路由、auto on/off 已测试的状态，区分 Z QKV 负结果。
- 保留独立输入、GPU/冻结/FFN/QKV 的准确回执、失败回退和私有图快照；
  不恢复此前整请求无益的 GPU-first、并行输出恢复或 c704 默认替换。

尚缺真实物理 ANE 放置/重叠追踪、更多设备和低内存压力验收、
多提示词与多个训练 LoRA 的视觉资格、Qwen 一/两图与高分辨率编辑覆盖、
Q4 更广覆盖，以及值得启用的 QKV 分区/FFN 联合调度。
本次只提交源码、测试、工具、便携模板与文字记录；被忽略的
`outputs/`、模型、编译图和生成图片不提交。工作树中原有
`results/` 暂存删除另属待处理产物，不混入本次代码提交，也不撤销。

## 本轮验证口径

本轮代码清理后 `TURBOCIDER_NATIVE_ONLY=1 make build` 退出 0；新库
SHA-256 为 `201a4926417ca3448d77638171f2d5ba5e84fa1bd2ea141aa9feb964ac524958`。
`make test-acceleration-contract` 退出 0：6 项 runtime host、10 项离线
设备计划、44 项整请求 screen、6 项内存 screen、17 项共图切换及 12 项
CLI 门禁。`make test-runtime-ane` 的 6 项 host 与 12 项 Core ML/MLX
集成退出 0；`make test-qwen21` 的 38+7+31 项 Python 检查及三个
native/workflow 程序退出 0。完整 `make test` 亦退出 0；其中没有
fixture/测试专用构建的项目按原套件标记 skip，不算新增通过覆盖。
最初在受限沙箱内的 `ps`/Metal 权限失败，分别在允许访问硬件的
沙箱外重跑上述套件后通过，并非产品代码失败。

五份关键维护文档的本地链接检查未发现缺失，`git diff --check` 通过。
这些构建/回归不是当前新库的真实整模型性能或跨模型视觉复测；上表
的 875b/28ff 原始库身份不可因本次代码整理而改写。
