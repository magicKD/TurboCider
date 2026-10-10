# Native Metal Graph：实施源码映射与未完成边界

日期：2026-10-07。本记录来自只读源码、现有报告及本地 safetensors **header** 检查；没有重新计算模型 hash，没有下载、导出模型，没有新增推理实验。源码正在并行实现，行号仅为本次定位辅助；以符号名为准。

这是首轮 Foundation 的历史映射；随后新增的 attention/QwenOps/完整 block/W0 桥及当前源码状态见 [2026-10-08 组件记录](2026-10-08-native-block-and-ane-handoff.md) 和 [11 的接线合同](../design/qwen-image-gpu-ane-command-graph/11-native-block-and-handoff-integration.md)。以下未完成项描述当轮时点，后续状态以新记录为准。

## 1. 当前状态与最小接线顺序

已实现 `native/backends/metal_graph/` 中 `Backend` / `CommandGraph` / `Ticket`、原生 BF16 `NativeFfn`、`mlx_bridge` 和 W0 `MlxFfnIsland` 的首个纵切。已有 host 编译和独立 Metal 合同测试通过记录；主任务另已完成少量真实 base FFN 组件回放，**不能标记完整 native block/DiT、异步 ANE 图或编辑 encoder 已完成**，也没有默认集成或新 E2E 资格。

本次组件输入为 layer 0 的 step 1/19，full 和 C=6144 分别测量；相对原 GPU 输出 RRMS 约 0.11–0.19%。full 的 GPU 对照约 41.95–42.02 ms、native wall 约 41.07–41.63 ms；half 的 GPU 对照约 21.08–21.73 ms、native wall 约 20.73–21.03 ms。收益小、覆盖有限且尚不稳定，不能推导整图收益或 ANE 重叠效果。原始记录为 `outputs/qwen-real-ffn-20261007/native-ffn-base-full.json` 与 `native-ffn-base-half.json`；这是主任务提供的实测结果，本源码审查未自行运行设备实验。

最小顺序：复用已有真实 BF16 输入验证原生 FFN → base 的 FFN island 接线 → 原生 attention/完整 block → 完整 native DiT → 同一原生图的异步 ANE 分支 → LoRA → 编辑 encoder。第一步不要求重建模型或捕获新轨迹。island 可以验证局部 kernel，但每层回 MLX attention 的执行仍属桥接路线；最终目标由 [设计 10](../design/qwen-image-gpu-ane-command-graph/10-native-metal-graph-backend.md) 定义。

| 现有源码 | 最小注入位置与约束 |
|---|---|
| [Qwen pipeline](../../native/models/qwen21/pipeline.cpp)，`run_ffn` / `runtime_ffn_->run`，约 1173 行 | 替换 base 的 full-GPU / channel-GPU FFN 回调；输入已是 FFN norm/modulation 后的真实张量。准备 PSO/权重视图在热循环外，不能在回调里重读权重 |
| 同文件，约 1091 行 `gate_up.weight` 拆分 | 原 checkpoint 前 F 行是 gate、后 F 行是 up；导入两个 offset view，down 列切片保留完整物理行距 |
| [Transformer](../../native/models/qwen21/transformer.hpp)，`set_plan_mlp` / `set_decode_mlp` | 可用 `SplitUntimed` 纵切接入原生 GPU FFN；不安装 prefill 回调则首步维持已有 GPU 路线。输出必须已完成且拥有合法 backing；仅创建 lazy array 不满足 W0 |
| Qwen pipeline，`dit.forward`，约 1290 行 | 完整 DiT delegate 的后续入口；保留 sigma/schedule、prefix/reference 条件、Euler latent 更新、finite check、取消和进度语义。仅替换内部 FFN 不算完整 delegate |
| [Z-Image](../../native/models/z_image/z_image.cpp)，`z_runtime_block`，约 1331 行 | dense base 分支的 runtime full/channel FFN 回调可接入相同 `NativeFfn`；第一版拒绝 LoRA/GGUF/ConvRot 等不支持 recipe，不能静默解释为 BF16 |
| 同文件，`z_transformer` 约 1880 行、`generate` denoise loop 约 4309 行 | 后续原生完整 Z DiT 的边界；有 noise/context refiners 和 main layers，不是把 Qwen block 参数改宽度即可 |
| [Qwen Session](../../native/models/qwen21/pipeline.hpp)、[module](../../native/models/qwen21_module.cpp) | Session 当前直接持有 MLX weights/tensors 和 hybrid 对象。需要明确 delegate/receipt，但不再建模型库；FFN island 回执不得伪称全 native DiT |

## 2. 模型尺寸：已由本地 header 核对

矩阵均为 checkpoint row-major `[out,in]`，输入 `[M,H]`。下表是本机文件实值，不根据模型昵称猜测。

| 模型 | H / F | gate / up | down | attention |
|---|---|---|---|---|
| Qwen Image 2.1 DiT | 4096 / 12288 | `transformer_blocks.0.img_mlp.gate_up.weight` BF16 `[24576,4096]`，按行平分 | `img_mlp.out.weight` BF16 `[4096,12288]` | 32 heads ×128；32 blocks |
| Z-Image Turbo DiT | 3840 / **10240** | `layers.0.feed_forward.w1.weight` / `w3.weight` BF16 `[10240,3840]` | `w2.weight` BF16 `[3840,10240]` | 30 heads ×128；30 main blocks，另有 refiners |
| Qwen 编辑视觉塔 | 1152 / **4304** | `model.visual.blocks.0.mlp.linear_fc1.weight` BF16 `[4304,1152]`，有 bias，GELU | `linear_fc2.weight` BF16 `[1152,4304]` | 16 heads ×72；27 blocks |

原生 FFN 接口可共享 `MatrixView` / `FfnBindings`，但不能硬编码 Qwen 的 F=12288。视图须包含 logical shape、真实 stride、byte offset/range 和 retained owner。尤其 down 的 channel 子区间不是紧凑 `[H,C]` checkpoint：原物理行距仍是 F。视觉 GELU/bias 路线独立，不能调用 SwiGLU FFN 并称等价。

Qwen 的 512² target 是 **1024 行**。首步包含 text/reference/target，稳态 query 是 target，但 K/V 仍含 prefix；[sequence.cpp](../../native/models/qwen21/sequence.cpp) 的 segment 可见性、文本因果关系、图像范围和最后 target-only 规则必须保留。DiT 使用 LayerNorm、Q/K RMSNorm，RoPE axes `[16,56,56]`。

Z 在 512² 时 image 也是 1024 行；caption 按 32 行补齐并使用 learned padding，main blocks 每步仍处理 image + padded caption，不能套用 Qwen 稳态 1024 行假设。noise refiners 2 层对应 ANE ordinal 0/1，main 30 层对应 2…31；context refiners 2 层不在这 32 个 ordinal 内。Z 的 RMSNorm、gated residual、RoPE axes `[32,48,48]` / theta 256 均须沿用其本身语义。context refinement 的精确缓存可复用现有实现。

## 3. Native attention 是完整 DiT 的前置条件

[mlx.cpp 的 `attend`](../../native/backends/mlx.cpp) 当前调用 MLX SDPA；只迁移 GEMM/FFN 仍无法自主控制所有去噪 GPU commands。最低原生 attention 要求：

- BF16 Q/K/V、FP32 accumulation/online softmax，支持 D=128 和真实 query/key strides；不创建完整 N² score 矩阵作为生产路径。
- Qwen prefix-build segment mask、稳态 target queries + prefix/target K/V，正确处理 query 的逻辑 offset；可先保留受预算管理的 native concat，再优化双区间读取。
- Z 的 padding/visible rows 和非 Qwen mask；QK norm、RoPE 和 transpose 排序独立验证。
- 数值模式独立标识。text encoder 当前请求 FP32 attention；不能因 DiT BF16 kernel 可运行就改变 encoder 合同。
- native output projection、norm/residual、最终 projection、Euler 更新及 finite check 一并纳入原生图。MLX 仅保留在声明的 encoder/VAE 阶段边界。

本地 vpipe 可读候选为 `gpu-kernels/metal/attention/attn_steel.metal`、`sdpa.metal` 和 vendored MLX steel attention headers。D=128 BF16 specialization 可作适配起点，host 参数包括 qL/kL、qL offset、strides 和 mask/causal constants；不能把 LLM 单 token decode 语义套入图像多 query。现有 steel 实例不覆盖视觉塔 D=72；通用朴素 SDPA 可作小张量 oracle，不能据此声称高效原生 encoder 已有实现。移植时保留 Apple/MLX 等对应许可声明，不把外部仓库路径设为运行时依赖。

## 4. 来源、内存和设备寿命

[ane_ffn.cpp 的 `device_view`](../../native/backends/ane_ffn.cpp) 已有重要范式：导出底层 buffer、byte offset、row pitch、dtype，同时保留 `Tensor` 与 `data_shared_ptr()`。新 [mlx_bridge](../../native/backends/metal_graph/mlx_bridge.hpp) 仅导入/保留 backing，不替调用者执行 `eval`，也不证明先前 MLX queue 已完成；跨队列 readiness 必须显式建立。输出再次交给 MLX 前，W0 路线须等 native GPU 物理完成。

`Backend::AllocationHooks` 是现有内存账本的接线点，**不是已经接通的账本**。[ModelSession::bind_memory_context](../../native/runtime/session.hpp) 默认空实现，当前 Qwen/Z Session 尚无 override；必须在 Session 接入 [MemoryExecutionContext](../../native/runtime/memory_execution.hpp)，以 reserve/commit site、storage identity 和 completion mailbox 管理新 allocation 的准入及 owner-thread 回收。异步 completion 线程不能直接销毁只允许 owner-thread 使用的 reservation。导入现有 MLX allocation 不应把同一物理 backing 重复计费，但它的延长使用期仍要计入峰值。

普通 Qwen/Z dense loading 目前物化 MLX 权重；Qwen layer-staged 路线另有 [SourceLease](../../native/runtime/streaming/source_lease.hpp) 与懒加载/释放机制。最小实现复用现有 loader 和 retained view，不制造第二份全 DiT 常驻副本。packed QKV、转置或量化 bank 必须另行预算，不因“只有 metadata bridge”而遗漏真实新增存储。

现有 `tc::metal_graph::Ticket` 的 `physically_complete` 仅证明本次 Metal CB 完成。`EventSource::External` 和 CPU `signal_external` 不证明 ANE 已停止访问；新增 ANE 请求必须进入共同完成域，保持 binding/program/surface/bank owner，按 [03 的 `ResultReadPermit`](../design/qwen-image-gpu-ane-command-graph/03-command-graph-and-handoff.md) 与窗口证书授权 restore。发生失败时不能人工放行 event 后读取仍被 ANE 写入的 output，或覆盖未退役的权重 bank。

## 5. LoRA 与编辑 encoder 的后续接入

LoRA factor 已由 [Weights](../../native/backends/mlx.hpp) 统一加载，runtime map 当前为私有实现。后续提供只读 prepared factor view，复用现有 target 映射、scale 和顺序，不再单独解析 adapter 文件。BaseOnly 不分配 LoRA delta/hidden-export 超集资源。之后才加入共享 FP32 low、pre-SiLU gate/up 修正、分片 down rank 与独立数值 recipe。

[设计 04](../design/qwen-image-gpu-ane-command-graph/04-qwen-lora-and-quantization.md) 已区分正常 GPU 的逐 adapter 舍入和兼容 hybrid 的聚合 delta 舍入。必须同时对正常 GPU oracle 和声明的兼容 oracle 检查；不能仅 hybrid 自测闭环。当前已有 LoRA 样本属于单 r128、strength=1 的已测域，其他 rank/strength 保留功能并不等于取得自动加速资格。

编辑入口在 [pipeline.cpp](../../native/models/qwen21/pipeline.cpp) 的 `VisionEncoder` → `assemble_prompt` → `TextEncoder::encode_embeddings`，参考图 VAE 编码是另一阶段。原生化所需语义包括：

- [视觉塔](../../native/models/qwen21/vision.hpp)：patch16/temporal2/merge2、27 层、D=72；4304 FFN 需 tail 支持。GELU 有原实现的 FP32 promotion，不同于 DiT SwiGLU；视觉 RoPE 为 rotate-half 路线。
- [文本塔](../../native/models/qwen21/text_encoder.hpp)：36 层、H=4096、32 Q heads/8 KV heads、D=128、theta=5e6、mRoPE `[24,20,20]`，因果 GQA，FFN F=12288。
- 视觉 DeepStack 来自层 8/16/24，但注入语言塔前 3 层，不能误插语言层 8/16/24；edit 设置 `final_norm=false`，文本输出保留/丢弃 image token 的规则一致。
- 视觉输入预处理与参考 VAE 的 resize/alpha 路线不同；不能因为同一图像而共享错误缓存。现有图像/encoder/geometry 身份与有界缓存先复用，原生 layout/rounding 改变则使相应缓存失效。

以上 full native DiT、LoRA 原生执行和 encoder 仍属待实现事项。

## 6. 本地资产、构建与历史报告

仅使用现有本地文件：

| 资产 | 路径 |
|---|---|
| Qwen BF16 DiT | `/Users/kd/Documents/project/mac_image_generation/models/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_bf16.safetensors`（14,230,280,616 bytes） |
| Z BF16 DiT | `/Users/kd/Documents/project/mac_image_generation/models/comfy_z_image_turbo/models/diffusion_models/z_image_turbo_bf16.safetensors`（12,309,866,400 bytes） |
| Qwen VL encoder | `/Users/kd/Documents/project/mac_image_generation/models/Qwen-Image-2.1/text_encoders/qwen3vl_8b_bf16.safetensors`（17,534,334,616 bytes） |
| 旧 probe/runtime | `build/qwen-real-ffn-20261007/{turbocider,qwen21-real-ffn-probe,libturbocider.dylib}`；它们早于本次原生后端，不能证明新代码已运行 |
| base 完整输入回放 | `outputs/qwen-real-ffn-20261007/base-captures/75956-1226003966809291-replay/` |
| LoRA 完整输入回放 | `outputs/qwen-real-ffn-20261007/lora-captures/75991-1226054446574458-replay/` |

每组回放有 9 个 BF16 `[1,1024,4096]` input，约 72 MiB；层 0/15/31，base 步 1/10/19，LoRA 步 1/3/5。**目录里的 `complete.json` 继承了早期 sparse FP16 capture 的 `input-N.f16`/8 rows 信息，不能作为完整回放文件 shape/provenance 的替代。** 以实际 `.safetensors` header 的 `capture_recipe=qwen21-real-ffn-bf16-v1`、`trajectory=forced-gpu`、model/adapter identity、step/layer 为准；每个选用文件都校验，不只检查目录名。既有 probe 已使用这些 full metadata；本轮不需要新捕获。

[tools/native/build.sh](../../tools/native/build.sh) 使用显式 `SOURCES`；新增 `.cpp/.mm` 必须加入，不能因文件在目录内便假定已链接。`.mm` 现有工具链带 ARC，Metal 框架已有链接。build identity 会覆盖 native 下相关 host/shader 后缀；仍须确认外置 shader/metallib 的打包和 digest，不能在运行时访问开发机的 Splash/vpipe 源目录。源代码并行变更时先协调冻结再验证 identity。本文检查不执行构建。

Splash 本地 HEAD 是 `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`；文档原 pin 是 `80827e1c3593f7d10c266394e4d4381321b73ad0`。只读 diff 确认两者的 `CommandGraph.hpp`、`MetalBackend.hpp/.mm`、`AneFfn.cpp`、`Program.mm`、`Handoff.mm` 六个核心文件一致，未推导全仓一致。本地 vpipe HEAD 为 `40706e01583b8087cfbb07c148974b35aa4fa9c1`，与文档固定 `3e2682a0847c9e76dd37978f2be01f53195ae5e3` 不同；源码引用与真正移植版本要分别记录。

[既有真实 FFN 报告](2026-10-07-qwen-real-ffn-screen.md) 的数字属于旧 hybrid 路线：base 整体 GPU 48.400 s、候选 43.044 s；LoRA GPU 20.664 s、候选 21.292 s。base **整个请求累计** staging 13.10 s、暴露等待 0.232 s，二者不是每层耗时；已经重叠的时间不能相加或全额计作可消除开销。这些结果用于选择候选，不是新原生后端收益证明。

用户允许的 5–8% 数值差异仅作为明确指标下的候选筛选参考，最终看同 seed 图像并保留条件/LoRA效果检查；不允许据此跳过算子、生命周期、mask 或内存安全。先用现有少量 FFN 回放，组件证明完整路径有收益后再安排最多 1–2 组 E2E，不提前导出全模型或生成额外大缓存。
