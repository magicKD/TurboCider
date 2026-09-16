# 04 · 模型接入与统一管理

[目录](README.md) · [协议](03-runtime-protocol.md) · [代码迁移](06-migration.md)

统一的是资源和生命周期，不是强行让四个模型使用同样的 block 内存布局。

## 1. ModelResourceDescriptor（拟议）

```text
identity:
  model/checkpoint/backend/adapter revision/format/layout revision
stages[]:
  stable ID + dependencies + passes + fixed baseline policy
  block IDs in execution order + layout classes
  tensor file ranges + backing size/alignment + shared storage aliases
  streamable group constraints + allowed K/G/P/D/Q
  mandatory resident objects + workspace/conditioning/output resource specs
  last-use contract + supported fence mechanism
  metadata_exact | validated_envelope | heuristic provenance per resource
```

描述查询分两级：静态 capabilities 不访问 checkpoint；exact describe 可打开 header/index，但不加载权重或创建 GPU buffers。
checkpoint 内容 hash 较大时可利用受验证的离线 identity；不能仅凭文件路径/mtime 宣称内容一致。

descriptor 返回 supported configurations，不返回“自己已认证”。registry/evidence 在框架层审核。

## 2. 各模型接入矩阵

| 模型 | 首轮边界 | 当前约束 | 后续扩展 |
|---|---|---|---|
| LTX C/Metal | denoiser 同一个 weight pool 跨两 stage 的 pass | 48 blocks、最多三槽、当前 P≥1、先 G=1 | 抽离 weights[0] metadata 后评估 P=0/G>1 |
| H3 C/Metal | text/DiT/VAE 外层阶段 + 固定双槽 DiT | stream_slots[2]、xor 轮换、首发不走 quantized cache | 增加单槽 serial、再评估更多槽 |
| Flux MLX | text→transformer→VAE 的 component staged | lazy graph、compiled-step/cache 引用 | 双流/单流 block 分不同 layout class，独立 group adapter |
| Z-Image / GGUF | text/transformer/VAE component + 格式分离 | Weights 整组件加载，GGUF 解包/packed storage | 先按 tensor ranges/group reader，不能把任意 shard 当 block |

“video-only”只是不输出音频，不等于 H3/LTX 内部 audio tensor 消失。仍执行的 AV computation、workspace 和 latent 必须保留并记账。

## 3. LTX：第一个统一 adapter

源码锚点：`native/models/ltx_runtime/ltx_blocks.c` 中的 `block_stream_source`、`streamed_block_slot`、
`start/finish_streamed_block_load()`、`run_streamed_block_stack()`、`ltx_native_create()`、`run_denoise_schedule()`。

### 3.1 Describe 和 layout

- 从 checkpoint header 描述 48 个 block；不能先真实加载 block0 再估整个模型内存。
- 每个 block 包括 video/audio attention、MLP、norm、scale、table 的所有 backing；总 bytes 相等也要比 layout。
- G=1 首发；metadata 证明 shapes 后可创建固定 slots。block0 workspace metadata 与常驻 weight 分离前拒绝 P=0。
- 新 C options 接收 exact resolved plan/版本化 handle；旧 `memory_budget_bytes/max_refill_slots` 只在 legacy 分支使用。
- 不允许 `tc_block_residency_plan_build()` 二次把三槽改成两槽或 resident；create 返回 actual layout，框架必须核对。

### 3.2 Execute

- owner 预先 reserve/allocate slot，worker 仅 fill；不再由后台第一次 `load_block_weights()` 决定 allocation instance。
- numerical `run_block()`/conditioning kernel 复用，不把每个 GEMM 都封进 virtual API。
- non-batch 的 GPU helper 有同步等待，但 batch 会延期；统一 adapter 必须返回真实最后 reader fence，不假设 C 函数返回=GPU 完成。
- 为每个 pass/step/group 建 semantic 映射；Stage 2 的 step 坐标延续 Stage 1，不混淆局部 sigma index。
- upsample boundary 的两种 latent、MLX upsampler、denoiser pool 可能同时存活；whole-request plan 必须包含此峰值。
- Transformer pool 释放并 drain 后才进入首发 Video VAE；不要隐式预取 VAE。

### 3.3 最小合入门

精确三槽/单 worker 合成测试先过；真实设备另选择与旧路线实际语义对齐的配置对照。要求质量不变、pool backing 一次创建、多次 refill、无 worker-side context 操作。
不能用“新的手动 P=1”与“旧自动 P=19”测量框架开销，必须先对齐 P/K/G/D/Q。
当前旧三槽会在 startup 同时启动多条 loader 线程，不等于 Q=1。三槽单 worker 只能作为合成/策略候选；
只有实际并发、retention和预取语义均对齐才能称同布局性能对照，详见 [10](10-executor-implementation.md) 第7节。

## 4. H3：第二个 adapter

源码锚点：`h3_dit.c` 的 `allocate_stream_slot()`、`read_stream_layer()`、`stream_slots[2]`、
`stream_ready_slot ^ 1u`；外层 `h3.c/h3_generate()`；`h3_session.mm` 的 memory context binding。

- 首先映射现有 K=2、G=1 和已支持 prefix 的布局；证明不改变 weight format、active block 集及 sampler。
- 当前双槽预加载/readiness 跨 forward 的规则需编译成显式 pass initialization；不能绕过 owner event。
- norms、AdaLN precompute、text refinement、latent workspace 属于额外资源，不在 slot_count 中。
- 当前首发 memory policy 排除 quantized_cache；不能把默认 H3 双槽 slot 描述成“所有量化格式通用”。
- H3 coarse TEXT/LOAD/DiT/VAE/EXPORT semantic events 尚缺；先接外层，再接 block/refill，terminal 不放宽。
- 单槽要单独实现同步 refill 分支，覆盖 init/first-block/最后一次/取消；不是把数组长度改成 1。
- 第三槽需要去掉 xor 假设、增加 mapping 并重新校验 upper。已有双槽路径不顺手重写。

## 5. Flux：保护已有 compiled-step 性能

源码锚点：`native/models/flux2/flux_transformer.cpp`、`flux_text.cpp`、`flux_vae.cpp`、
`native/backends/mlx.hpp` 中的 `Weights`。

当前 transformer 双流和单流层不同；4B compiled single-block 路径刻意避免每 block eval，最终 step 才同步。
框架集成不得在 default 路径插入逐层等待，否则即使内存充足也会退化。

第一阶段：只描述 component 生命周期，测 framework envelope，保持 plan-only；不以 memory limit API 代替 hard cap。
第二阶段：独立 ranged weight reader + group materialization；编译函数不能闭包捕获旧全模型 weights。
slot handle 作为明确输入，或每个候选静态 shape/layout cache key，证明跨 group 不保留旧 backing。
第三阶段：双流/单流不同 pool layout class；stage barrier 释放或按 max padding 预分配，二者作为不同 layout revision。

新 streamed candidate 可以在 group 边界 eval/synchronize，但该成本和丢失 fusion 必须进入性能报告；不牺牲默认路线。

## 6. Z-Image：文件 shard 与执行 group 不相等

源码锚点：`native/models/z_image/z_image.cpp` 的 `load_z_component()`、`load()`、`encode_text()`、
`unload()`，及 `Weights::load_gguf_file()/erase_prefix()/materialize()`。

- 一个 shard 可跨多个 block，一个 block 也可跨多个 shard；descriptor 以 tensor range 列表表达 group。
- GGUF、BF16、ConvRot 等各有 adapter format/layout revision；解包 scratch 与 packed scale 不能漏算。
- `erase_prefix()` 仅删除 map 引用，不能证明 lazy graph/GPU 已释放；需要完成边界和 cache 观测。
- text conditioning materialize 后才能卸载 encoder；保留的 conditioning 仍是 request allocation。
- 第一阶段先 component staged、质量与 envelope；第二阶段再 group reader，不对当前 resident `Weights` 做全局 invasive 改造。

## 7. 新模型 onboarding 清单

1. 实现静态支持集和 metadata-only descriptor；给出至少一条明确不支持分支。
2. 定义逻辑 block、layout class、pass、group 与 checkpoint ranges。
3. 写一个 tiny fake descriptor + fake backend，不依赖真实大 checkpoint 验证 K/G/P/D/Q。
4. 提供 pool create/fill/encode/fence/drain/destroy；证明每个 backing 的 owner。
5. 接外层 stage boundary、结果 actual layout、quality parity。
6. 先获得 layout_validated 的真实 GPU证据，再补 required-site closure/guard 获得 bounded_certified。
7. 记录关闭新框架时的 default ABBA，包含已有 legacy streamed/resident 两条路径。

模型接入不应新增一套自定义线程池、预算算法、swap 检测和结果字段；这些由框架统一。
