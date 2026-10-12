# 独立原生 DiT：实际实现与接入边界

更新：2026-10-08。本章记录路线 N 的实现。`NativeDenoiser` 已编码完整 base DiT/Euler，并已接入 private W1S 稳态 FFN。上一 register 整图验证构建（cache off）完成一组 512² 整图 GPU/ANE 对照，去噪省时 13.60%、完整请求省时 9.97%，见[最新 Base 状态](../../status/2026-10-08-native-register-w1-base-e2e.md)。这是成功路径和单组收益证据，**不代表泛化性能、编辑、完整请求内存或 App 已验收**。历史 native GPU 配对与 W1 tiny 组件保留各自证据域。LoRA 仍待独立实现和验证。

当前源码新增默认关闭的 compact row-scale cache；生成数据及真实 FFN 的身份／精确复用已通过，完整组件耗时随顺序反转，没有稳定净收益，未追加整图，见[真实缓存筛选](../../status/2026-10-08-native-scale-cache-real-screen.md)。

## 1. 单独掌握 command graph

新后端的去噪热路径不调用 MLX GEMM、SDPA、compile 或 eval。它使用 `metal_graph::Backend` 的 queue、PSO、BufferView、CommandGraph 和 Ticket，自主记录输入投影、调制、所有 block、输出投影与采样更新。MLX 保留在源权重/encoder/VAE 的边界；`mlx_native_denoiser` 只导入已有 produced BF16 buffer 及真正的 Data owner，不复制 checkpoint，不隐式求值或解量化。

这实现 Splash 管理思路中最重要的前提：GPU 的具体提交范围属于应用。当前 W1S 已把 staging、GPU FFN 通道、ANE handoff、restore/join 纳入完整完成域，仍逐层主机准入；多层设备图目标见 [16](16-splash-native-backend-spec.md)。当前 Backend 的保守 inter-dispatch buffer barrier 仍保留，不能为提高跑分先删除安全依赖。

实际入口：

| 源文件 | 责任 |
|---|---|
| `native/models/qwen21/native_denoiser.hpp/.cpp` | base 权重合同、request arena、完整 DiT、prefix 事务、精确 schedule 驱动 |
| `native/models/qwen21/native_denoiser_math.cpp` | 小型 time/RoPE 常量，独立 CPU oracle 可检验 |
| `native/models/qwen21/mlx_native_denoiser.hpp/.cpp` | 保留原 MLX allocator owner 的 BF16 来源导入 |
| `native/models/qwen21/native_block.hpp/.cpp` | 各层共用 projection/pointwise PSO 与 phase arena |
| `native/backends/metal_graph/native_step_ops.hpp/.mm` | stride copy、BF16 Euler、reset/finite 验收 |

接口示例是当前 API。外部读取 readiness、source lease 与生产内存上下文仍由调用者证明：

```cpp
auto bindings = import_base_denoiser_weights(backend, produced_base_weights);
auto model = NativeDenoiser::prepare(backend, std::move(bindings), source_revision);
DenoiserInput input;
input.latents = produced_initial_latents;
input.text = produced_conditioning;
input.height = packed_latent_height;
input.width = packed_latent_width;
input.identity = {source_revision, conditioning_revision, "", generation};
input.sigmas = exact_pipeline_sigmas; // Includes the final zero.
auto run = model.begin(std::move(input));
while (run->state() != DenoiserState::Complete) run->step();
auto result = run->final_latents();
```

该 API 消费现有 pipeline 生成的 schedule，不重新推导或改变 base/adapter 步数。`height/width` 是 packed latent token 几何，512² 对应 32×32；参考输入已经是 normalized VAE latents。

## 2. 数学和 phase

时间输入是两行 `[cos128,sin128]`，分别对应当前 sigma 和零时间。频率先 double exp/log 再 float，角度保留 `(sigma*1000.f)*frequency` 两步 FP32，最终 cast BF16。经 linear1→SiLU→linear2 得到 temb，再共用一次 SiLU(temb) 给 modulation 和 norm_out。

文本 conditioning 使用 FP32 RMS、FP32 `weight+1` 和乘法，最后一次 BF16 cast；text-in GELU 则使用 BF16 scalar 和逐 primitive 舍入。它们与 Q/K fast RMS、vision GELU 不共用近似 recipe。

SiLU 明确保留 exp、`1+exp`、reciprocal、正值 `1-y` 和乘法各自的 BF16 边界；不能只在 sigmoid 末尾 cast。实际 MLX 编译后的 `tc::silu` 与分开物化的 sigmoid 再乘输入，甚至会在极少数 exp 舍入分界处不同。原生 SiLU 使用显式 `metal::fast::exp` 对齐编译后 SiLU 的函数策略；Backend 的 Safe math／Precise 默认保持不变，不能据此放宽所有算子的数学策略。recipe 与逐阶段设备诊断单独记入本轮状态。

首步按 `make_sequence_geometry` 交错 text/ref segments，目标段始终最后。text/ref 行使用零时间 modulation，target 使用当前时间 modulation。RoPE 保持 16/56/56 三轴 float pow；attention causal offset 是真实 sequence 行，不能用 RoPE 坐标或 reference token 面积代替。

稳态只投影 target latents，并绑定每层自己的完整 prefix K/V。modulation 直接使用当前时间 row 0 的四个 broadcast 子视图，避免稳态展开 `[T,H]`。最后先裁 target，再做 norm_out scale、LayerNorm、projection。

Euler 保留 `BF16(dt)`、`BF16(noise*dt)`、`BF16(latents+product)` 三个边界，没有 FP32 fused 更新。当前输入/来源必须 BF16；GGUF、ConvRot、runtime adapter 和 FFN bias 等未支持语义显式拒绝。

## 3. PSO 和内存复用

H→H Linear 和 QwenOps 由模型只准备一次；NativeBlock 接受 shared ops，每层仅换 retained 权重 views，不重新 prepare。GPU-only 默认 geometry 共 42 个 PSO（hybrid 的 stage/controller PSO 另计）：五种 Linear geometry 各 4、QwenOps 11、StepOps 6、attention 2、FFN 3。不能创建 32 套 NativeBlock prepare。

request 只分配一个 block arena，容量取 prefix-build 与 steady-decode 需求的最大值；两种 workspace 是该 backing 的不同视图。它在同一 queue 的有序 block 之间复用；phase 改变前上一步必须成功物理完成。hidden 与 latents 分别 ping-pong；32 层 prefix K/V 各有独立 backing，不借用可重写 scratch。

当前 arena 仍使用 11 的保守 scratch 布局：13 个 H 矩阵加一个 F 矩阵，稳态另有 joined K/V。允许后续做 liveness 复用，但必须先增加明确 read/write sets 和跨 dispatch 生命周期证明。长文本/1–2 个参考图的总行数在创建 sequence、RoPE 或 cache 之前有界检查，默认总行上限 8192；它是此原型的容量合同，不是模型 prompt 限制，也不能静默截断输入。

`DenoiserMemory` 是 requested payload 统计，**不是**实际 allocator capacity 或完整进程峰值。Backend AllocationHooks 能持有账本 reservation/lease；现有 MLX Data 可能自带 lease，但普通 MLX allocation 和 IOSurface 不能据此默认视为已计费。新 [13 的 Session 接线](13-native-session-and-memory-phase.md) 已将 native 所有分配按 site/class、actual capacity 接入现有 Context；MLX source/import 明确覆盖在 retained baseline，报告仅属于 native phase。完整请求认证仍未成立；W1S 的 IOSurface／consumer drain 已在同一 phase 接线，验证范围见 13/15。

## 4. 缓存事务和失败

source revision、conditioning revision、generation 与 request 几何绑定在不可替换的 request 中。每层 prefix 写入只属于 tentative bank；只有整步 Ticket 成功、GPU 物理完成且 noise、latents、全部新 prefix K/V finite，才发布 bank 和下一步 latent。

任何 encode、submit、timeout、非 finite 错误都会把 run 标记 Retired，后续不得重试消费半成品 bank。超时不 signal 外部 event，不允许重写其时间常量、scratch 或 cache。已提交 graph 的 retained views/owners 留到设备实际完成；逻辑失败不作为预算或 IOSurface 复用许可。

run 保留最近直接 GPU Ticket，并聚合当前 hybrid controller 的 producer／ANE／GPU 补集／restore consumer 完成域。`physical_status()` 查询全部实际提交状态；`drain(0)` 仅查询，`drain(positive_timeout)` 有界等待物理完成，负 timeout 拒绝。未提交过时返回 `submitted_buffers=0`、`physically_complete=true`，含义是没有设备工作待回收。这些接口不能证明 finite 验收、释放所有 request owner 或成功结束内存账本；已退休的 run 即使晚到完成也不会恢复或发布结果。同步 step 与查询共用 mutex；获得控制权后设定整步绝对截止时间，全部层、encode/submit 返回后检查和最终 graph 共用此期限，不每层重置预算。它不能抢占同步 driver 调用，也不是跨线程抢占取消 deadline。

`final_latents()` 仅在整个 schedule 完成后可读；Complete 后禁止再 step，所以返回 view 不会被后续 ping-pong 覆盖。当前 API 不提供跨 request prefix import/export：来源完整性、借出后不可变性、consumer ticket 和并发 ledger 接线前，不提前开放该功能。

## 5. 验收和后续

CPU tests 分别验证 time/RoPE/sequence、BF16 更新、shape/stride/alias 和 backing-before-accounting 销毁顺序。tiny DiT 对照必须使用实际 MLX Transformer 与同一 BF16 source，而非自写近似参考。真实 block 回放使用原 compiled callable 的实际 args/output；一步完整 GPU 轨迹只用于产生下一步真实输入，第二步 block 0 后停止，张量不落盘、无 VAE/PNG。

这些证据分别属于基础数学、完整算子 recipe、小几何设备正确性和真实组件筛选。只有匹配 build/source/conditioning、无近似 cache 的配对完整 512² 请求及图像比较，才能给独立 native backend 整图性能资格。

本轮实际进展见 [完整图与真实 block 状态](../../status/2026-10-08-native-denoiser-and-real-block.md)：tiny 生图／单参考图／双参考图两步 latent 对照通过；真实 H4096 block 有少量组件收益。真实 probe 为计时外的小型 prefix 布局转换单独计费，不将组件耗时下降当作完整 20 步、ANE 并行或 App 性能。

后续顺序保持：

1. [13 的 native Session/source/memory 接线](13-native-session-and-memory-phase.md) 已实现；按当前构建验证 base GPU-only 完整请求，不将 phase guard 写成完整请求认证。
2. W1S 的稳态完整分支已源码接线；在当前集成构建验证全部完成域与已有真实 FFN 完整成本，有收益才做有限 Base E2E，再独立推进无逐层 host admission 的 W2。
3. LoRA 的 gate/up 修正在 SiLU 前、down 修正只加入一次；复用 operator/plan，独立验证 rank scratch 和准备开销。
4. 编辑输入的 1–2 reference geometry 与 VAE/encoder 边界、Z-image 模型 adapter 分别验收。
5. App 显示实际 route/qualification 与 fallback；未验收的 native/private 分支不作为默认发布配置。
