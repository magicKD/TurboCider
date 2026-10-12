# 512²：共享 LoRA ranks、GPU tile 和实际解码复用窗口

2026-10-08，Asia/Singapore，M4 Max 64GB / macOS26.6.2。接续
[encoder 来源复用](local512-encoder-source-2026-10-07.md)和
[有界解码 consumer](convrot-register-dense-window-2026-10-06.md)。
只读本地模型/adapter；没有下载、模型改写或 dense sidecar。整体
Z/Qwen base、真实 LoRA、1–2参考图、快于优化 GPU 的目标仍未完成。

本轮结论：共享 ranks 对现有 Qwen 混合路线有小幅诊断收益且 PNG 不变；
GPU tile 没有稳定收益；GGUF 的实际八次复用仍不足以摊薄解码成本。
这些实验均不提升默认配置，`qualification_passed=false`。

## 共享的是 LoRA A 投影，不是 adapter 权重或条件缓存

显式开关 `TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS=1` 默认关闭。
要求 resident512²、显式 runtime、真实 inference-time LoRA、Private
固定非零 channel share；普通 GPU/base 请求忽略有效开关，非法值拒绝。

新的 `Weights::lora_input_ranks` 每个 operation 生成 `X Aᵀ`，GPU
channel head 和 ANE gate/up correction 使用同一组结果。保留原
FP32/可选 FP16 rank dtype、adapter alpha/strength、每个叠加 adapter 的
GPU 舍入顺序、pre-SiLU correction 和 joined-hidden 的一次 down-LoRA。
不把 LoRA 合并进 BF16/W8 权重，不改 checkpoint 或 ANE basis。

图 factory 与 ranks 都是 request/operation-local；没有 FFN adapter 的
block 使用原路径，不编译零输出 rank 图。原通用 projection/delta API
委托给同一实现，新的 rank 输入检查 adapter 数量及实际使用 rank 的
shape/dtype。裸 `vector<Tensor>` 不自动检测同形状的跨 adapter generation
误用；调用者必须绑定同一输入/列范围/adapter generation，不能缓存到
下一请求。当前 Qwen 调用点遵守此约束。

回执区分 `prepared_sets_this_request`、
`completed_hybrid_blocks_this_request`、
`completed_adapter_rank_arrays_this_request`。只有两个 callback 均消费
共享输入、runtime 成功返回且没有 complete-GPU fallback 时才记 completed。
这是 graph-output/成功 operation 证据，**不是物理 kernel 数量**。关闭
channel-range correction 的旧调试配置不会产生双 consumer completed
计数；本轮 screen 使用原默认 narrow correction。

## 同库、同来源的两参考图对照

Viggle r256、seed29、六步、512²，两张原 teapot reference，三个不同
prompt 均 conditioning miss。两臂都是 Private DiT5120 + GPU encoder，
使用相同 encoder source retention；不把保留来源的收益算给共享 ranks。
新增 screen 支持 `--lora-ranks-screen`、反向 order 和实际 completed 校验。

| 顺序 | off 冷 / warm median s | on 冷 / warm median s | warm wall 减少 |
| --- | --- | --- | ---: |
| off→on | 18.446991 / 14.795037 | 18.360291 / 14.390060 | 2.74% |
| on→off | 18.556607 / 14.584291 | 18.495289 / 14.385608 | 1.36% |

两个方向对应三张 PNG 的字节 hash 全部相同；227个实际 adapter
projections，source 只 load1次。on 每请求 prepared/completed192组，
384个 rank arrays；off 三计数均0。不是 selection label 冒充执行。
两个方向的每个 cold/warm 样本保留，未合并成更有利的分母。

首先尝试的 continuous-load 窗口在 off 臂完成后，被外部 ComfyUI CPU
活动拒绝，on 未开始，summary incomplete。保留原窗口/PNG/日志，不
删 busy sample、不放宽 gate、不停止外部进程。上表来自另起的、明确
不申请 load 资格的诊断，不是正式加速证明。四个 process-tree memory
报告 complete；reverse on 窗口观测到65,536 bytes系统 swap-in，其余
三个窗口为0，swap-out全部0，不能把系统 swap 归因给一个 backend。
phys-footprint peak约38.9–39.2GB，scope 不含外部服务/driver归因。

目录 `outputs/local512-qwen-edit2-lora-ranks-v3-diagnostic-{forward,reverse}-20261008/`。
被拒绝窗口为 `outputs/local512-qwen-edit2-lora-ranks-v3-forward-20261008/`。

## 单参考图：加入完整 GPU 控制后仍未盈利

同一 v3 库、相同 retention、真实 adapter/六步/seed29、三个 fresh
prompt，`--lora-ranks-gpu-control` 保留完整原 GPU，不加 bridge。
顺序 on→off→GPU；同样只作有 memory 采样的诊断，无 load 资格。

| route | cold request s | fresh warm median s |
| --- | ---: | ---: |
| Private5120 / ranks on | 16.149158 | 11.962596 |
| Private5120 / ranks off | 16.306966 | 12.012774 |
| complete GPU | 15.838547 | 10.963575 |

on/off 三张 PNG exact；两臂每请求192个成功共享/未共享 hybrid blocks，
每请求224次实际 ANE calls，累计224/448/672。GPU 没有共享/ANE执行。
三个 memory 报告 complete、swap-in/out0；峰值约38.3/38.6/40.2GB。
共享 ranks 只减少很小一部分混合开销，没有弥补相对纯 GPU 的差距。

目视检查三个原尺寸 whole pair，以及第一个 case 的全部同坐标三裁剪：
壶形、盖/钮、把手、阴影与构图非常接近，釉面/高光/纹理有细小变化；
未看到新增断裂或色块。只是这些 case 的有限 agent 观察，不是用户
批准或多 seed 质量保证。另两 case 的裁剪虽已生成但未作本轮目视
验收；自动 visual manifest 保持 pending。原历史 N1失败不改写。

目录 `outputs/local512-qwen-edit1-lora-ranks-v3-gpu-control-20261008/`。

## GPU kernel：保留物理 pitch，拒绝无稳定收益的默认 tile

`dense_gpu_projection.hpp` 开放 BM16/32/64、BN64/128 及显式 static
slice 候选；只有 M/N 完整对齐时使用 static，否则沿用 checked dynamic
路径。默认仍 BM32/BN128、dynamic、原 FP32 accumulation 与输出边界。

真实 layer0 BF16 down 权重，physical pitch12288、不紧凑复制权重；
M1024/1056/3137 × K3072/7168/10752、N4096，合成有限 hidden，八个
recipe 各3次 warmup、9个循环换序样本。这是 operator host span，
不是全 FFN/request 或 GPU timestamp。全部72个 recipe 的 relL2 为0。

| M / K7168 | 原32×128 ms | 32×64 ms | static32×128 ms | 64×128 ms |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 4.281833 | 4.297250 | 4.277042 | 4.772625 |
| 1056 | 4.434208 | 4.382083 | 4.436125 | 5.257667 |
| 3137 | 12.577917 | 12.677209 | 12.582917 | 14.171250 |

小差异没有证明稳定盈利，大 tile 较慢；没有硬件 counters，不能仅凭
跨度断言 register pressure/occupancy 是唯一原因。新实际 Metal 回归
另覆盖144个 FP16/BF16、F32 partial/narrow output、尾块、非零物理
row/column offset、strided input case，以及16个非法配置拒绝。
与独立 FP32 matmul 和原 tile 分别比较，门槛不放宽。

证据 `outputs/local512-qwen-down-kernel-v3-component-20261008.json`。
独立 probe 无 adjacent dylib；absolute rpath 指向已保留 source-v4
Private库，不能冒称 wrapper 捕获到相邻 v3 库或实际 loaded-image trace。
v1/v2 Metal template 编译失败的记录保留。

## GGUF/ConvRot：一次解码 + R 个真实 consumer

`gpu_weight_consumer_probe` 新增可选 `actual-reuses`，1..16，prepared
输入总量 upper256MiB。每次 trial 清空 window、新 content generation，
使用 R 个不同已准备的 activation。计时包括第一次 typed decode、finite
验证、完成 fence，以及每次 cache lookup/GEMM/eval；与 R 次 packed QMM
串行交替，3次 warmup、9个 hot，保留全部样本。原分离 decode/GEMM
screen 与12种 window lifecycle self-test仍保留。

本地 Z layer0 gate，M1056/N10240/K3840，dense backing75MiB，一个
矩阵槽；R8 每个 trial 的实际 miss1/hit7、decode75MiB。GGUF Q4_0/
Q8_0 使用 FP16 affine metadata；ConvRot 使用原 legacy BF16 scale、
保持 Comfy rotated basis，不 inverse H256、不重新量化原权重。

| 来源 | R | R次 packed QMM ms | decode + R次 GEMM ms |
| --- | ---: | ---: | ---: |
| GGUF Q4 | 1 | 6.046208 | 11.113708 |
| GGUF Q8 | 1 | 6.104958 | 10.508416 |
| ConvRot | 1 | 6.558708 | 10.258709 |
| GGUF Q4 | 8 | 48.510042 | 52.405292 |
| GGUF Q8 | 8 | 48.856292 | 52.458666 |
| ConvRot | 8 | 52.589292 | 52.187208 |

六格所有 tested input 的输出按 bytes 相同，结束 dense claim0；observer
完整、最大 gap<149ms、binary unchanged。不是 source read/packing、输入
旋转、全模型/LoRA/视觉、异步 prefetch 或 whole-process memory 资格。
GGUF 即使实际 R8仍较慢，ConvRot约0.76%小差异不足以推荐新路由。

更重要的是：full DiT 有多层轮转，单/双矩阵槽不等于能够跨八个 step
保留同一矩阵。此前 raw window 的真实 hits0/连续 eviction 仍是反例。
没有把本工具的连续 R8冒称真实模型的跨步复用，未自动接入 predecode。

## 接续方向：格式/执行布局和混合 ANE

不修改 GGUF container/header 或生成全模型 dense 副本。优先优化
实际消费布局与来源：保留 compressed masters、消除重复 metadata
准备、按 encoding 选择 packed GPU 或 bounded raw→ANE staging。只有
实际 hit/eviction/read/wait 和完整窗口盈利证明，才接通提前 decode。
大 M 与 M1/短序列必须单独测；不要用 layer0 R8代替全层 cache审计。

ConvRot GPU 下一步应拆分 rotate/QMM/down/join 后再试融合，保留 BF16
边界与 Comfy ordering；本轮 dense down tile 的结果不能当 ConvRot
rotate/QMM融合收益。Private 的 direct Q8/Comfy W8A8 和 Public 的共享
W8接口已经有独立实现/证据，本轮没有再引入大型 ANE 软件 BF16舍入图。

Private/Public 杂糅优先按 operation/layer/source/shape 选**一个** ANE
executor，与原 GPU complement 配合：敏感/短层留 GPU，兼容性或精度
需要的层考虑 Public，已实测盈利的 W8长层才考虑 Private。后续 backend
override应作为不可变配置传入 factory，不在并发请求间改 process env；
cache key隔离 backend/basis/recipe/ABI/adapter generation，避免同一片
权重同时常驻两种 ANE格式。当前尚未自动接入这套三选一调度，构建期
capability fallback不能冒充每层盈利选择。晚期失败继续完整 GPU重算，
不发布 partial scratch；Core ML/Private调用计数不证明物理 INT8 MAC
或 GPU/ANE overlap。

## 回归、构建与保留

最终 Public18项、Private21项选定回归通过，无 skip；包括新的 LoRA
数值/receipt/plan gates、Metal tile、原 projection、实际 encoder、source
generation/admission、screen contract、dense window，以及 Private channel/
prepared calibration/失败恢复。不是全仓或完整模型矩阵验收。
Public actual release-binary guard通过，两库各489个source inputs匹配。

```text
Private 915a2abdd852783168e6825f69be7e25487c9bff4740ded96302d3fb736b746c
Public  3c740cb2f1bf35726dcf858cdeffcf0796d1b323a7e27030b4837716db91f086
```

v1 brace编译失败、v2夹具名称/编译失败、首次tail拒绝夹具未materialize、
request-gate夹具保留ANE manifest等失败日志都保留；修复的是各自夹具，
没有把非法请求放宽或修改数值门槛。构建包含原有 ConvRot草稿，是
working-tree snapshot；本次 selective commit不夹带这些草稿。

完成所有 owned jobs后清理四个 ranks v1/v2/v3 Private/Public build的
773个 `.o`、61,249,000 logical bytes（约58.4MiB）及四个空 module-cache
目录，可重新构建；保留 dylib/CLI/probe、manifest、日志、PNG及模型。
完整路径/样本在 outputs，提交的汇总使用相对路径，见
[机器证据](../design/validation/local512-shared-lora-and-decode-20261008.json)。
