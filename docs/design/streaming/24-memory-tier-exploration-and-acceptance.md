# 24 · 四模型内存档位探索、测量、实施与验收

[目录](README.md) · [Public preset 产品/API 方案](23-public-memory-tier-presets.md) · [现有证据](13-implementation-progress.md)

实施细节：[25 代码/API/模型施工](25-public-preset-implementation-spec.md) · [26 采样协议/测试/发布](26-public-preset-acceptance-and-release.md)。
本文保留候选策略和 MT 批次；25/26进一步展开接口与验收，不代表新增测试已经通过。

日期：2026-09-17。状态：**待实施实验与工程方案**。候选集合不是已通过测试的参数推荐，目标档位不是实测峰值。
本轮未运行新的模型推理、sweep、pressure 或 swap 实验。P0–P4 与统计门槛继续使用 [12](12-acceptance-playbook.md)。

## 1. 实验要回答什么

对每个模型、artifact、工作负载、设备与部署方式，回答：

1. 哪些候选 layout 确实可以执行且输出不变？
2. 整请求峰值发生在构造、文本编码、denoiser、upsample、VAE 还是 export/cleanup？
3. P/G/K/D/Q 和组件释放策略各能节省多少内存，代价是什么？
4. 8/10/12/16/20 GiB 各目标下，哪个已验证候选最快且足够稳定？
5. 当前设备上的结果能覆盖哪些 workload？哪些硬件还没有证据？
6. 默认关闭的 App/CLI/service 路线是否完全保持原表现？

不要求每个模型有五个不同 layout，不要求 8 GiB 一定能运行，不把 quantization、跳 block、降尺寸作为隐藏 streaming 策略。

## 2. 当前起点：能力、证据、限制分开

| 模型 | 现有 exact candidate 能力 | 已有性能锚点 | 本轮应探索什么 |
|---|---|---|---|
| LTX-2.5 distilled C/Metal | G1、P≥1、K1..3，受 group/capacity 限制 | P8/K3/D2/Q3，64×64×9，11 passes | 优先 prefix，再 K2/K3，最后 K1 最低占用 |
| Z-Image Turbo Comfy BF16 | G1/K2/D0/Q1，P 可变；30 blocks | P14，64×64，1 step | 保持双槽/claim-overlap，扫描 prefix |
| MiniMax H3 Turbo original BF16 | G1/K2/D1/Q1，P0..48，carry | P0，50 blocks×4 passes，512×512×22 denoiser probe | prefix + 完整 text/DiT/VAE/export；不扩普通 H3 |
| Flux.2 Klein 9B Diffusers BF16 | P0/G1/K2/D0..1/Q1..2，双 retained pool | D1/Q2，64×64，2 steps | 先测组件峰值；prefix 调整需新增实现 |

代码能接受某个 P 不等于该组合已获 public 资格。所有发布项仍需 exact workload 与 lifecycle/质量确认。
首版统一 G1，避免把任意 grouping/fusion 安全性混入内存档位发布。

### 2.1 现有性能只能作为锚点

| 模型 | 同布局 wall median ratio | denoise median ratio | 当前证据结论 |
|---|---:|---:|---|
| LTX | 0.98810 | 0.97249 | 冻结 tiny P1 PASS |
| Z-Image Turbo | 0.99986 | 1.00313 | 冻结 tiny P1 PASS |
| H3 Turbo | 1.01066 | 1.01185 | 工程验收完成，strict bootstrap INCONCLUSIVE |
| Flux.2 Klein 9B | 0.99971 | 0.99816 | 冻结 tiny same-layout P1 PASS |

来源：[13 第 13.11–13.15 节](13-implementation-progress.md)。Flux raw summary 位于
[summary.json](../../../results/streaming/flux/2026-09-17/p1-same-layout/summary.json)。
已保存 Flux/H3 环境为 M4 Max、64 GiB、Apple SSD、无外加压力；不是 8/16 GiB 物理机器测试。

Flux 当前 tiny 两步 anchor 的 MLX peak 为 11,693,804,356 bytes，约 10.89 GiB；resident 为
18,303,578,036 bytes，差约 6.15 GiB。该值不能说明 Flux 已有“12 GiB 整请求档”：

- 8/10 GiB 已低于这个 allocator anchor，不能复用同一记录宣称可行。
- 12 GiB 若使用 `H(T)=10%T`，10.89+1.2 已超过 12，尚未计非 MLX 覆盖问题。
- 16/20 GiB 只能列为优先确认目标，仍需真实 normal-target 整请求证据。
- 若后续分析发现峰值来自 text encoder，增加 denoiser prefix 不会降低这个下界。

## 3. 通用候选生成方法

### 3.1 先 metadata，再短测，不做全笛卡尔积

先读取 descriptor，输出 block/source/destination bytes、resident fields、pool classes、shape 和 adapter 能力。
同构 G1 下仅用于解释的权重估计：

```text
weights(P,K) ≈ fixed + sum(prefix destination bytes) + sum(slot field capacities)
logical_read_per_request = resident_initial_reads + prefix_initial_reads
                          + sum(pass suffix source reads)
```

真实容量必须逐 field 取兼容最大值、加 alignment/derived scratch；source bytes 与 destination bytes 不等同。
K 不是所有模型共同单位：Flux K2 有 dual/single 两池四槽，LTX K3 是一池三槽。

完整峰值要按 live timeline：

```text
candidate floor = max(
    construction/text live set,
    denoiser weights + activation/workspace live set,
    upsample/stage transition live set,
    VAE live set,
    export/drain live set)
```

这里各项都包含对应同时存活的 baseline/cache/conditioning；不能把各阶段独立峰值求和，也不能漏掉过渡期的两套权重。
有一项未知时只做探索预测，不生成可发布的内存档。

### 3.2 目标附近细化 prefix

给定目标 T 和一个 K/D/Q，使用 metadata 与已校准 activation/组件数据筛掉明显不可能的 P。
在每个目标边界附近保留 `P_est-1 / P_est / P_est+1`（在能力范围内），而非穷举所有 P。
P_est 是离线候选，不是 runtime 隐式修改用户请求；真实读写、cache、lazy materialization 可使峰值非单调，不能只二分测一个点。

候选首次 coarse pass 每模型每 workload 最多选 8 个，确定约 2–3 个非支配候选后才增加边界样本。
下文列出的集合是枚举上限，**不是一次必须全跑**。

### 3.3 Pareto 筛选

主要维度是完整请求耗时、校准峰值和失败/质量；I/O bytes、P95、cleanup、线程/分配作为约束与诊断。
若 A 的内存和 wall 均不优于 B 且无显著 tail/lifecycle 优势，A 不进入 UI。
同档不预设“P 最大就是最快”，也不预设 K3/Q3 一定优于 K2/Q1。

## 4. LTX：prefix 为主，双槽/三槽为辅

### 4.1 保持的语义

范围固定为已有 LTX-2.5 distilled ConvRot INT8 checkpoint、GPU C/Metal dense video.generate、11 passes。
不新增量化，不改变 kernel、Fast A/V/attention 配置，不启用 Sol、audio、I2V、LoRA 或 ANE。
这些参数要写入 workload identity，而不是为了让小档成功而静默关闭/改变。

### 4.2 候选族

| ID 前缀（研究用） | 布局 | 目的 | 当前实施状态 |
|---|---|---|---|
| `ltx.anchor` | P8/G1/K3/D2/Q3 | 重放现有对照 | 已有 frozen tiny 证据 |
| `ltx.low2` | P∈{1,4,8}/G1/K2/D1/Q1或2 | 较低 slot 占用且保留 overlap | 需逐布局 real request 验证 |
| `ltx.balanced3` | P∈{4,8,12,16}/G1/K3/D2/Q1或3 | 深流水对比读取并发 | 需验证 |
| `ltx.prefix3` | P∈{20,24,32,40}/G1/K3/D2/Q3 | 高档减少 11 次反复读取 | 先容量筛选再运行 |
| `ltx.minimum1` | P1/G1/K1/D0/Q1 | K2 仍放不下时的最低占用候选 | 已有 tiny K1 功能历史，速度/normal-target 未认证 |

每 pool 必须至少 K 个 suffix group：48 blocks 时 P≤48−K。不能 P=N、K>0 冒充 resident。
优先选 P1/K2、P4/K2、P8/K3、P12/K3 等少量组合；P20+ 只有内存估计表明目标仍可容纳才运行。

### 4.3 按档位的探索方向，不是固定映射

| 目标 | 首先比较 | 若无候选符合 |
|---|---|---|
| 8/10 GiB | P1/K2 与 P1/K1；确认 text/VAE 下界 | 报 component floor 或 unsupported，不宣称靠单槽必能运行 |
| 12 GiB | P1/4/8、K2；接近边界与 K3 同目标对比 | 保留较低目标候选或拒绝 |
| 16 GiB | P8/12/16 的 K2/K3 | 选择实际 wall/P95 更稳的一个 |
| 20 GiB | P16+ 对照 P8/K3；同时测组件峰值 | 若高 P 无收益则复用较低档 preset |

现有每槽 GPU backing metadata 约 387,981,696 bytes，来自历史固定 checkpoint；只能估权重池，不能乘 K 后
认定整请求占用。CPU table、scratch、Gemma、connector、两 stage/upsample、VAE 和媒体全部单列。

### 4.4 需要补的实现与验收

- 复用 `ltx_streaming_descriptor.*`、`ltx_streaming_plan.*`、`ltx_streaming_adapter.inc`；不同 K/P 仍走同一 executor。
- 确认 generic plan、构造 snapshot 和实际 backing 同源，prefix 无隐藏全模型 materialization。
- full-pipeline 测 Gemma/connector、Stage1、upsample、Stage2、VAE/export、销毁。
- K1 首槽加载可与 prefix 重叠；比较 baseline 时记录 startup，不能算成纯 framework overhead。
- 若 8/10 档被 VAE/text 主导，另建带明确生命周期的 component-plan revision；不直接改 block schedule 掩盖。
- 延续 LTX 已有 session fault 测试：cancel/retry、shape A→B→A、unsafe drain、engine free，扩到准备发布的 K/P。

## 5. Z-Image Turbo：维持 K2，按 prefix 形成平滑档位

### 5.1 首轮只改 P

现有 `StreamingPlanView` 限定 K2/G1/D0/Q1/reload；adapter 的 claim-overlap 保证 D0 仍能两槽交替读/算。
首轮保持这组调度，不为每个档位重写算法。30 blocks、每块 13 BF16 fields；公开范围为 Comfy 单文件 BF16。

候选 P 集合：`{0,4,8,12,14,18,22,26,28}`，P28 留两组，与 K2 匹配。先 coarse 选 `{0,8,14,22}`，
再根据峰值在目标附近增补。P29 不满足两槽/两组要求，不应作为“几乎常驻”强行接受。

| 目标 | 探索优先级 | 判断重点 |
|---|---|---|
| 8 GiB | P0/4 | Qwen/refiner/VAE 是否已超过目标 |
| 10 GiB | P4/8/12 | 累计 prefix + 文本长度、分辨率影响 |
| 12 GiB | P8/12/14 | 与现有 P14 anchor 相邻，不能直接把旧 10 GiB budget 平移 |
| 16 GiB | P14/18/22 | 减少每 step 读取，是否有实际收益 |
| 20 GiB | P22/26/28 或复用低档 | 若默认 resident 更快，提示用户可关闭 streaming |

以上全部为筛选顺序；实际 prefix 以完整请求数据决定。

### 5.2 最小改造面

- `native/platform/apple/z_image_streaming_descriptor.mm` 已允许 P 变化，优先补验证而非扩 K/G。
- `native/models/z_image/z_image.cpp` 与 `native/platform/apple/z_image_weight_stream.mm` 审计 fixed/prefix 加载与 lazy arrays。
- 多 step 请求必须按真实 pass 数统计 fills，不能把现有 1-step P1 直接覆盖 9-step App 默认。
- 目前 policy 记录 pool 保留到 VAE；测清 denoiser backing 与 VAE 的重叠。必要时新增明确的
  `release-before-vae` component revision，完成 `mx::eval`/drain、解除 alias 后释放并做数值/故障回归。
- 该组件释放变更算策略实验 P4；对同组件策略再做 P1，不能与旧 pool retention 不同的路径比较却称“纯框架”。
- K1、K3、D1/Q2、G2 暂不进入第一批 App 档位；后续只有数据证明现有 prefix 不够才扩 adapter。

## 6. H3：只做 Turbo original BF16，双槽 carry 不动

### 6.1 候选族

固定全部 50 个有效 block、4-step Turbo、K2/G1/D1/Q1、carry_first_group。
不得通过 active-block pruning、token reduction、first-block cache 或量化来实现小档位。
P 候选：`{0,2,4,8,12,16,24}`，依据真实 metadata 与预算筛掉过大值；P 的代码可接受上界不是待发布承诺。

| 目标 | 候选方向 | 必须说明 |
|---|---|---|
| 8/10 GiB | P0，先验证完整管线下界 | 若 text/workspace/VAE 已不适合，直接不提供此档 |
| 12 GiB | P0/2/4 | 不因 denoiser probe 成功而认为 full video 成功 |
| 16 GiB | P4/8/12 附近 | metadata 筛选；prefix 内存换取四次读取减少 |
| 20 GiB | P8/12/16/24 附近 | 保持相同 kernel/quality；按真实瓶颈筛选 |

P 变化会改变 suffix group 数，需要测试奇数/偶数组合下 `(pass*groups+ordinal)%K` carry rotation。
至少增加一个奇数 suffix 的边界样例，不能仅测 P 为偶数的“恰好容易”情况。

### 6.2 App 档位证据不能只用当前 probe

当前性能证据是合成 conditioning 的 denoiser probe，wall 范围为 probe 启动到模型释放，artifact 是 joint latents。
Public App 使用 `video.generate`、真实文本、VAE、MP4；必须另跑完整请求。
推荐第一张 full-pipeline 卡沿用 module 默认 512×512×22、4 steps、无 audio/参考输入，再扩 39/73 帧等合法卡。
H3 module 要求 frames=5+17n，范围 22..362；不是用户任意输入帧数都被 streaming 自动接受。

### 6.3 性能和实施边界

- 每大矩阵大块 `pread` 保持已排查路径；不要为统一 chunk policy 回退到已知较慢的细粒度读取。
- prefix 小 tensor/workspace 真实 residency、legacy cache 退出、reader callback 与取消都需进入 full session 审计。
- 保留当前 H3 工程验收与 bootstrap INCONCLUSIVE；接受合理 I/O 波动不等于取消质量、内存与生命周期门。
- H3 不为筛选“好看的 PASS”无限重复 TB 级读取；预先冻结 read budget 和停止条件。
- 不实施普通 H3、其他 checkpoint 或量化 H3；扩大本轮范围必须另行决定。

## 7. Flux.2 Klein 9B：先确定峰值，再分两阶段扩展

### 7.1 F-A：现有实现可直接探索的范围

固定 P0/G1/K2、dual/single 两个 retained pools、reload；探索 `(D,Q)={(1,1),(1,2)}`，D0/Q1 只作诊断。
改变 Q 对池容量几乎不构成新档位，不能把 Q1 标成“8 GiB”、Q2 标成“16 GiB”。
先测 text/denoise/VAE 各阶段；现有 Qwen 阶段化释放和 denoiser→VAE 的 release-before-vae 保持不变。

### 7.2 F-B：新增 resident prefix，支持较高目标换更少 I/O

当前 `flux_streaming_descriptor.mm::StreamingPlanView` 显式要求 `prefix==0`，执行 adapter 也按全部 dual/single group 运行。
所以 Flux 的 P2/P4/P6 **不是配置已经支持的能力**。建议先实施单一全局连续 prefix，沿用现有 P 语义：

```text
block order = dual[0..7] + single[0..23]
P2/4/6: 常驻前 P 个 dual blocks，剩下 dual 与全部 single 仍 streaming
P8:     dual 全部常驻，只建立 single pool
P12:    全部 dual + 前 4 个 single 常驻，只建立 suffix single pool
```

候选第一批 `{P2,P4,P6}`；P8/P12 等跨 class prefix 单独做一轮。P7 会只留下一个 dual group，与 K2 不符，拒绝；
P8 则该 class 已全部 resident，compiler 不应创建空 dual pool。保留“每个实际 streamed pool 至少 K 组”的规则。
其余 P 类似检查，不随意为最后一个 group 降成 K1。

实施项：

1. 描述常驻 prefix fields/source，分配 owner 持有的 resident MLX arrays，并核算其 full lifetime。
2. `FluxExactStream` 的 `encode_prefix` 每 pass 正确执行 prefix，不把常驻误写成只算一次。
3. dual→single 的 context/image concatenate 只执行一次，不因 prefix 跨界重算/漏算。
4. `prepare_group`/binding 以真实 block ID 映射，resident 和 streamed arrays 不交叉覆盖。
5. compiler 输出可为一个或两个 suffix pool；修改当前“必须两个池/全部32组”的断言为基于 descriptor 的验证。
6. 继续 retain_all（仅 retain 仍有 streamed groups 的 pool）；不同 P 的 source/layout digest 变化进入资格。
7. 修改 direct same-layout baseline 或建立独立参考，确保新 P 的 P1 仍比较相同生命周期/池策略。
8. 测 P0/P2/P6/P7拒绝/P8/P12，至少两 pass、取消和 resident-prefix reader drain。

### 7.3 低于当前 anchor 的档位

增加 P 只会增加 denoiser backing，不是 8/10 GiB 的解法。按峰值定位处理：

- 若 text encoder floor 主导：需要独立 text-stage streaming/受验证组件计划；此扩展另做 adapter/质量验收，不伪装成改 K。
- 若双 pool 同时存活主导：可以研究 class-serial 释放，但当前 generic serial 在 class/pass 边界重建 pool，
  steady allocations 非零，不能冒充当前零分配 P1 通过。先作为离线策略实验，不纳入第一批默认 public preset。
- 若单一 union arena 值得做：需要证明 field/alignment/binding 可复用、class generation 与 fence 正确，属于新的 backing revision，
  不应在最初档位发布中顺手实现。
- 若 activation/VAE 主导：显式 tiling/组件释放是新 execution policy；只有独立校验后才能成为另一个 preset。

### 7.4 档位探索顺序

| 目标 | Flux 9B 计划 |
|---|---|
| 8/10 GiB | 暂不映射，先查 full-pipeline floor；可能需要组件级后续工作 |
| 12 GiB | 现有 tiny MLX anchor 加默认档位余量已不满足；验证新策略后再决定 |
| 16 GiB | 先确认 P0，再研究 P2/4 的可行与速度，不提前公开 |
| 20 GiB | 确认 P0 可复用；若实测值得则发布 P4/6/8 等新 prefix 记录 |

Flux 4B compiled graph 不在该范围。当前 App 默认可能是 Flux 4B，新开关应显示不支持，不能按 `flux*` 通配套用 9B。

## 8. Workload 卡与内存测量

### 8.1 最小 workload 集

| 模型 | 功能锚点 | 首轮产品卡（候选，运行前经 module 校验并冻结） |
|---|---|---|
| LTX | 64×64×9，11 steps | 512×320×33、512×320×97，11 steps；常用更大卡另行签核 |
| Z-Image | 64×64，1 step | 512×512、1024×1024，9 steps |
| H3 Turbo | probe 512×512×22，4 steps | full-video 512×512×22，再选 39/73 帧，4 steps |
| Flux 9B | 64×64，2 steps | 512×512、1024×1024，4 steps（当前 module 默认） |

图像至少增加一张非正方形卡；text token rows 设短、典型、接近支持上限三类。先 tiny 排错，随后 normal-target 才能生成 App 档。
固定 checkpoint、seed、prompt/token count、steps、backend、compile mode、batch、dtype、inputs、audio、approximation、tile 和 retention。
首版只发行实际确认过的 shape/steps；不能仅对总像素数或总 video tokens 做线性外推。

### 8.2 观测字段

| 字段 | 采集方式/含义 | 禁止误用 |
|---|---|---|
| `mlx_peak_bytes` | 当前请求 MLX allocator peak，记录 reset 范围 | 不代表 native Metal、OS cache 或完整 RSS |
| `mlx_active_bytes` / cache | 边界快照与峰值标记 | 不把独立峰值直接相加 |
| `native_metal_live_backing_peak_bytes` | 已覆盖 allocator hooks，含 reader pending release | 未覆盖 site 必须标 incomplete |
| `metal_allocated_sampled_peak_bytes` | 同一 device 观察，带 interval/scope | 不是精确硬件 VRAM used，也不能再加到 MLX |
| `process_footprint_sampled_peak_bytes` | `task_vm_info.phys_footprint` 时间序列 | 采样不能证明连续时间绝不越界 |
| `process_peak_rss_bytes` | `ru_maxrss`，进程生命期高水位 | 不能 reset，不是多请求的独立峰值 |
| `execution_tree_footprint_conservative_peak_bytes` | 对齐时间采样 parent/helper，记录保守重复计算 | 不求各进程历史峰值之和，不漏 worker |
| `managed_live_bytes` | 按唯一 backing/alias 的资源计数 | 不把容量、pending release 和 live storage 重复计 |
| `swapin/out_delta` / compressed / pressure | 系统观察、时间范围明确 | 不能直接归因于本任务；不存在可靠观察则 unknown |
| `logical_source_read_bytes` | pager/source range 累计 | 不等于 SSD 物理读量 |
| `materialized_fill_bytes` | slot completion/backing 内容累计 | LTX conversion 下不能当 source bytes |
| `physical_disk_bytes` | 可获得的受信 OS 统计，注明 device/system scope | 文件 cache 命中不应该报成 SSD 读取 |

当前 `StageExecutor.bytes_loaded` 增加的是 completion record 的 bytes，必须核对 adapter 定义；新测量工具不一律重命名为 source traffic。
源码现有 `results.mm` 的 memory.scope 主要指 MLX；LTX/H3 App 档位需要额外 scope，不能改旧字段含义而不升级报告。

### 8.3 当前 runner 的两处缺口

`tools/native/run_streaming_campaign.py` 已有长期 worker/per-request engine lifecycle、ABBA 与 `ru_maxrss`，但：

1. `per_request engine` 不是 `fresh process`。同 worker 先跑大 preset 后跑小 preset，后者 `ru_maxrss` 会带上前者高水位。
2. 当前成功样本摘要保留 block/timing 等，不完整转存分阶段 MLX/原生 Metal/footprint。Probe 还需确认统计的是子进程而非 wrapper。

因此新增独立 `memory_calibration` 采样模式：每个 cold-memory 样本使用新进程，完整保存 native result，
记录 PID/tree、baseline、first allocation、阶段切换与 cleanup。不要修改原 P1 runner 的长期 worker 语义，避免历史统计不可复现。

### 8.4 冷、热和模式切换

三类条件分开：

- `fresh_process`：单请求，含构造/所有首次 materialization；page cache 状态 separately unknown/warm，不叫冷 SSD。
- `warm_request`：同 engine 至少 10 次观测增长/泄漏与 warm performance；`ru_maxrss` 只报 aggregate，单请求用独立时间窗采样。
- `mode_transition`：已有 resident → streaming → off/default，再切回；包含旧 cache 清理、峰值重叠和重载成本。

模型内部 `reset_peak_memory()` 通常在 generate，不自动覆盖 engine constructor，也不能测结束后又 reset 以丢掉最高值。
Sampler 从 process/request setup 前开始；各 reset epoch 与 baseline 记录到报告。采集 baseline 必须已进入实验计时/范围，不能把 setup 内存隐藏。

### 8.5 采样与 instrumentation 预算

建议 calibration 初值：进程/系统采样 20 ms、关键阶段必采；另选一张短卡用 5 ms 复查遗漏。
这是离线诊断，不进入默认 App 热路径。测量请求结束后等待安全 reader drain 并取终态；不在每个 block 插入全设备 synchronize。
原生 allocator/slot hooks 给精确可覆盖的 live count，采样给 process scope，两者互校验但不相加。
release timing 和 audit/trace/密集 memory sampling 分开，记录 instrumented/uninstrumented overhead。

## 9. 如何从 raw 数据生成目标档位

### 9.1 生成校准 envelope

对同一个 workload/deployment bucket：

1. 保存 fresh-process、warm、mode-transition 全部峰值和覆盖信息，失败不删。
2. 以独立确认样本的最大完整 request footprint、阶段纳管峰值/残差估计形成 M；记录具体取值法和版本。
3. 保留 MLX/Metal 辅助域的冲突检查；主域比已知 live backing 还小需调查覆盖，不能取更小者发布。
4. 任何关键组件、helper 或 construction 未覆盖则 `memory_calibration=incomplete`，不发布目标。
5. `M+H(T)<=T` 才能宣称此 workload/部署适配该目标；H 规则见文档 23。

M 是带测量限制的校准值，不是概率保证的绝对上界。对真正硬 cap 仍需独立 required-site closure 和 bounded registry。
实际大内存机器上测得的 M 也不能证明同目标的低物理内存设备不会压缩/swap。

第一版估计器固定为 `observed_tree_max_v1`，避免实施者自行挑低值：

```text
F_i = 第 i 个确认样本中，同时间轴 execution-tree footprint 的最大观测值
M_observed = max(F_i for 全部确认 cold/warm/transition 样本)
M = max(M_observed, 可与同一 scope 对齐的完整阶段预测值)
```

后一项尚不能可靠生成时保留 null，仅在观测覆盖足够且有独立审核时发 experimental empirical-target，
并显示 `estimate_kind=observed_calibration_not_bound`。不能把不同 scope 的 MLX/Metal/RSS 峰值加进预测值。
存在请求失败、关键阶段缺观测、异常峰值或低内存压力污染时停止自动分档，先诊断，不删除样本取剩余成功最大值。
未来换 envelope estimator 要 bump revision、重做确认，不静默沿用旧档资格。

额外准入用的 `expected_additional_bytes` 应来自同 scope、同缓存状态的 baseline-to-peak 记录，并在当前 baseline
超出校准范围时拒绝/重查。不能以 `max(historical peak)-max(historical baseline)` 推导请求增量，也不能从不同 PID 的值相减。

### 9.2 档位算法

```text
for target in [8,10,12,16,20] GiB:
    candidates = layout_confirmation_passed && workload_match && artifact_match
                 && deployment_match && calibration_complete
    fitting = [p for p in candidates if M(p) + H(target) <= target]
    if fitting.empty:
        publish unavailable reason, not a fabricated layout
    else:
        choose frozen independent-confirmation rank
        emit target -> preset_revision
```

这是离线生成 proposed mappings 的算法，不能自行开启 public。审阅签核后，runtime resolver 还必须过滤
release channel/revocation；H3 的工程决策使用单独状态，不把 INCONCLUSIVE 改成 `statistics_passed=true`。

对数值接近的候选优先低内存/低 I/O/稳定参数；不要为了声称“12 比 10 更快”挑选噪声赢家。
理论上目标越大候选越多；实际 rank 未显示提升就共用已有 preset。

### 9.3 纯示意算例（不是任何模型实测）

假定某 workload 确认后有 A：M=7 GiB、wall=10 s；B：M=9 GiB、wall=8 s；C：M=11 GiB、wall=7.9 s。
按 H=10%T：8 档可选 A；10 档可选 B；12 档仍选 B，因为 C+1.2>12；16 档 C 可能可行，
但 0.1 s 若在噪声内，仍可选择更省内存的 B。不能以 C 的 denoiser peak 11 GiB 直接命名“12 GiB 档”。

## 10. 实验执行协议与资源限额

### 10.1 E0：Metadata/plan-only

读取 header/index 与已验证 artifact identity，列候选、pool bytes、prefix bytes、逻辑读取量、未知组件。
输出 `candidates.json` 和 rejected reasons，不加载 GPU、不自动改 public registry。
同一配置去重依据 layout+component identity，不因分到不同目标重复跑。

### 10.2 E1：功能 smoke

先执行现有 anchor，再最多 8 个经筛选候选，每个一轮 tiny parity；不通过输出/reader/cleanup 就停止该族。
H3 先 probe 定位，再 full video；不能以 probe 替代终态 MP4 的正确性。
新 Flux prefix、奇数 suffix carry、单池/双池边界都必须先独立 host/Metal test。

### 10.3 E2：内存探索

每候选/workload 初值：3 次 fresh-process calibration + 1 组 10 次 warm reuse；无需对所有组合重复。
先测 text/VAE floor，如果已超过最小 T，停止该目标进一步 denoiser sweep。
这一步只能筛选和估计噪声，不能用三次最大值声称硬内存上限。

### 10.4 E3：独立确认

每个待发布 workload 保留最多 2–3 个候选，冻结新 policy 后用未参与探索的 prompt/seed 和运行批次确认：

- memory：至少 10 个 fresh-process 完整样本及 repeated-request/mode-switch 集；最终数量与停止规则运行前固定。
- time：P1/P4 用 [12](12-acceptance-playbook.md) 的 ABBA/BAAB 与 bootstrap；20 pairs 是起点，不是通用充分样本数。
- quality：确定性路径 latent/PNG byte-exact；视频比较 decoded frames/latents，不仅比较可能含时间 metadata 的 MP4 文件 hash。
- tail：P95 置信不足就 INCONCLUSIVE，不以某次最快值签核。

不同 preset 改 P/K/retention 的收益是 P4，不能把所有 sweep 都叫 P1。没有同布局 baseline 时先报告策略比较，
为新实现补 direct replay/reference 后再宣称 framework overhead。

### 10.5 E4：App/低物理内存资格

App 普通用户路径（public constructor）重放已签候选，验证 Swift v2、完整输出、UI 状态、取消/队列与真实部署峰值。
如果要标注支持某个低 RAM 设备，需该设备或明确限制的兼容硬件类上完整测试；64 GiB 设置 target=8 GiB 不是物理 8 GiB。
受控 pressure 单独操作并预授权；不给 sweep 默认附带压力进程。P3 对照不关闭 swap、不 purge 未知缓存、不运行无界 allocator。

### 10.6 每次 campaign 必填资源限制

`max_candidates`、`max_requests`、`max_wall_seconds`、`max_logical_read_bytes`、输出空间上限、free-disk floor、
最低系统余量、pressure/thermal stop、每请求 timeout、cleanup timeout 与最大停止等待。
超限保存 incomplete/reason 后停止，不自动增加预算；GPU 实验一次只跑一个，以免互相污染 I/O/内存。

H3 示例：按 descriptor 的每请求 source bytes 乘请求数预估读取量；超过约定 disk-read budget 就先减少 coarse candidates，
不靠反复全矩阵重跑寻求一个统计 PASS。逻辑读取限额也不等于物理 SSD 写入磨损，二者分开记。

## 11. 工具链与证据工件

### 11.1 复用现有工具

现有文件：

- `tools/native/run_streaming_campaign.py`：正式 policy/ABBA runner。
- `tools/native/verify_streaming_campaign.py`：独立 verifier。
- `tools/native/run_streaming_audit.py`：独立 audit 收集。
- `tools/native/capture_streaming_source_identity.py`：构建/源码封存。
- `tests/native/test_streaming_campaign_verifier.py`：身份、协议和缺失证据拒绝回归。

可复现已有 Flux P1 verifier（读取已有 bundle，不跑 GPU）：

```sh
python3 -B tools/native/verify_streaming_campaign.py \
  --bundle results/streaming/flux/2026-09-17/p1-same-layout
```

文档任务不需要实际重跑该性能 campaign；旧 summary 只作 anchor。

### 11.2 新增工具（尚不存在，不可当现成命令）

| 拟新增文件 | 输入/输出与约束 |
|---|---|
| `tools/native/inspect_streaming_candidates.py` | 调 native metadata 接口枚举受限配置，输出计划与拒绝原因；默认无 GPU |
| `tools/native/calibrate_streaming_memory.py` | fresh-process 观测、完整结果、同时间轴 helper、coverage 检查 |
| `tools/native/explore_streaming_presets.py` | 读取冻结候选和资源上限，串行执行 E1/E2，输出 Pareto 候选；GPU 需显式许可 |
| `tools/native/build_streaming_catalog.py` | 读取已确认 evidence，输出 proposed record/patch；不自动 release-enabled |
| `tools/native/verify_streaming_calibration.py` | 独立校验 raw PID/time/scope/coverage，重新计算M和档位；不信任summary自报PASS |
| `tests/native/test_streaming_memory_calibration.py` | high-water 污染、null、helper、阶段与覆盖反例 |
| `tests/native/test_streaming_preset_resolver.py` | scope、档位边界、排序、身份、撤回与 off fast path |

各工具只生成人工可审阅的文件；catalog 发布须独立审阅。尽量扩现有格式/runner共享逻辑，不复制一套 kernel/runtime。

### 11.3 最小 bundle

```text
policy.json                 # 探索集合/预算/停止条件，事先冻结
source-identity.json
build-identity.json
artifact-manifest.json
environment.json            # GPU/RAM/SSD/OS/热状态/cache/container
candidates.json             # raw user intent + resolved layouts
rejections.jsonl
raw-native-results.jsonl
memory-samples.jsonl        # request/stage/PID/time/scope/availability
stage-memory-summary.json
raw-samples.jsonl           # wall/denoise/quality 对应 request identity
quality.json
lifecycle.json
audit.json
pareto.json
confirmation-policy.json
confirmation/               # 独立于探索的 raw evidence
proposed-catalog-records.json
manifest.json               # 文件 digest，不包含机器私有路径/凭证
```

raw 运行输出默认留指定实验目录，不自动提交大媒体、权重或用户 prompt。签核记录保留可重放必要 identity，隐私内容可用固定测试 prompt。

## 12. 实施顺序与每批完成门

| 批次 | 内容 | 完成门 |
|---|---|---|
| MT-01 | 数据口径和校准工具：scope/phase、fresh-process、保存原生 memory result | fake & host 覆盖 high-water/unknown/helper；默认路径零调用 |
| MT-02 | 四模型 metadata 候选输出 + anchor 重放 | bytes/能力验证、unsupported 明确；不扩大 public |
| MT-03 | LTX/Z prefix、H3 Turbo full-pipeline 的受限探索 | full-request 峰值、质量、生命周期、Pareto；不预写档位 |
| MT-04 | Flux P0 全流程分析；需要时实现 P2/4/6 resident prefix | host/Metal/direct parity、concat/class 边界、默认 P0 |
| MT-05 | selector v2 + 只读 catalog + native resolver/authority | exact match/撤回/冲突/fail-closed；public 仅测试 fixture 可执行 |
| MT-06 | Swift v2 envelope、高级开关、档位/报告、旧 draft/legacy 共存 | App integration、public constructor、off request 不变、异步 stale query |
| MT-07 | 独立确认、产品 workload、目标硬件、P0/P1/P4 | release evidence 足够才签核首批 catalog |
| MT-08 | 公开 experimental presets，逐模型/档位开启 | 普通 App/CLI/service 成功、未知组合拒绝、回滚可用 |
| MT-09（独立后续） | hard-memory guard、完整资源上界、低内存 L3/P2、必要 P3 | 与 layout-only 资格分开发布，不改变 preset 目标语义 |

不需要等 Flux 的低档位或全部模型完成才发布先完成的记录；不需要新增第三/第四套模型 scheduler。
MT-05/06 可先用测试 catalog 做 UI 开发，测试记录绝不能进入 release registry。
MT各批的具体代码子任务、依赖、测试与合并门见[26 第13节](26-public-preset-acceptance-and-release.md)。
raw采样格式和预算计算见26第3–4节；high-water/PID/缺样本/失败等反例验收见26第10节。

## 13. 验收矩阵

| ID | 测试 | 必须得到的结果 |
|---|---|---|
| MT-CFG-01 | 原 v1/manual、旧 profile、无新字段的 draft | 原序列化/默认路由不变 |
| MT-CFG-02 | 新 schema v2 memory_tier/preset；bool/负数/溢出/重复键 | 有效配置通过，其余早拒绝 |
| MT-CFG-03 | profile on/request off、selection 切换、旧 residency/budget 冲突 | 单一 authority，off 不创建新对象 |
| MT-UI-01 | 模型切换、shape/prompt 编辑，异步响应逆序 | 旧结果不覆盖新 draft，不复用别的模型 layout |
| MT-UI-02 | Z-Image legacy 6/8/10/12GiB 保存任务迁移 | 不改变旧采样预算口径；关闭新版恢复原设置 |
| MT-UI-03 | Flux 4B、GGUF、ANE、LoRA、audio 请求 | 说明不支持，不悄悄改选项 |
| MT-SEL-01 | T 边界正好够/差1 byte、same target 多候选 | 确定匹配、不自动超目标；tie-break 稳定 |
| MT-SEL-02 | 8/10/12/16/20 单调候选集、多档同 preset | 正确去重、只显示已验证档 |
| MT-SEL-03 | 用户伪造 record/hash/release、未知 revision | native 不接受外部自授权 |
| MT-SEL-04 | queue 后撤回、权重更换、runtime 升级 | 重新核验并拒绝，不复用过期许可 |
| MT-MEM-01 | MLX 缺失/原生 Metal、helper 未观测 | unknown/incomplete，不报0，不分档 |
| MT-MEM-02 | 同 worker 大配置后小配置 | ru_maxrss 保留进程 scope，不当独立小配置峰值 |
| MT-MEM-03 | MLX/Metal/footprint 重叠、多个 PID 不同时峰值 | 无重复相加/各历史峰值求和错误 |
| MT-MEM-04 | VAE/text 高于 denoise、construction 双重 materialization | 峰值阶段准确；不能只按 slot pool 发布 |
| MT-MEM-05 | target 仅按 denoiser 够，full-request 不够 | 目标不可用，返回 component floor |
| MT-LAY-01 | LTX K1/2/3，不同 P，suffix 少于 K | 安全执行或确定拒绝；无 silent K shrink |
| MT-LAY-02 | Z P0/P14/P28 与 P29、9-step | fills/pass 正确，claim-overlap 保留，P29 拒绝 |
| MT-LAY-03 | H3 奇/偶 suffix 跨4pass carry，P变化 | 200或实际suffix×4 fill、ticket/generation/输出正确 |
| MT-LAY-04 | Flux P0/P2/P6/P7/P8/P12 | P7拒绝；pool数量正确；dual-single concatenate 精确一次 |
| MT-LIFE-01 | cancel、short read、stale completion、reader未到、unsafe destroy | 停止派发，安全 drain 或隔离，不提前 free |
| MT-LIFE-02 | success→success、A→B→A、模式切换、engine free | 无增长泄漏，实际 plan 与报告一致 |
| MT-QUAL-01 | 与同质量 baseline 的 latents/PNG/decoded video | 按冻结规则一致，无隐式近似 |
| MT-P0-01 | 新功能关闭，resident/legacy-streamed，App+CLI/service | 新 hook/probe/pool/thread/clear 为0；时间阈值按12 |
| MT-P1-01 | 相同 layout/component/lifecycle 的 direct/legacy 对照 | framework overhead按12；新增 serial 重分配不得冒充零分配 |
| MT-P4-01 | 不同 prefix/K/policy 独立确认 | Pareto/rank有raw证据，不把探索赢家当确认结论 |
| MT-PUB-01 | App public constructor + 已签 selector | 成功；无需candidate符号或测试环境变量 |
| MT-PUB-02 | 未注册tuple、prepare不支持、guard未认证 | 分别明确拒绝，不调用旧 heuristic 隐式回退 |
| MT-LOW-01 | 真低RAM或明确授权压力条件 | 报成功/失败率、swap/pressure，不能用64GiB软件target替代 |

Fixture 只能证明 parser/algorithm/failure handling；不能生成 `public_stable` 记录。
实机缺权重或无设备访问时明确 SKIP/BLOCKED，不能与 host PASS 合并成实机认证。

## 14. 发布卡片模板

每个 model/workload/target 的 release review 至少填以下表；未知值保持 pending，不填猜测数字：

| 字段 | 发布要求 |
|---|---|
| Model/artifact/backend | 准确 ID + digest/revision |
| Workload | shape/frames/steps/token rows/operation/quality flags |
| Device/deployment | GPU/RAM/SSD class、OS/MLX、embedded/worker |
| Target | T、H(T)、calibrated M、scope/coverage |
| Execution | P/G/K/D/Q、pool count、pass transition、component policy |
| Memory phases | construct/text/denoise/upsample/VAE/export/drain 各证据 |
| Performance | P0/P1/必要P4，冷/热/切换范围、median/P95 与区间 |
| Quality/lifecycle | 完整输出一致、取消/失败/恢复/teardown |
| Evidence | exploration 与 confirmation raw/manifest/audit |
| Availability | public-experimental/stable，适用目标集合，明确不支持的分支 |
| Rollback | catalog revision、revocation 操作、错误展示 |

H3 的合理波动可以用明确实验等级和工程决策处理，不修改历史 INCONCLUSIVE，也不要求普通 H3 扩展。
如果全请求某阶段超过目标，必须撤下/修订该档记录，不能只删掉那个样本。

## 15. 完成定义

完成不是“App 有一个 8/10/12/16/20 下拉框”，而是：

1. 开关默认 off，旧功能和性能保持原样。
2. App 仅显示符合当前模型/任务/设备的已审核档位。
3. 用户选择内存目标后，通过 public route 确定性选择后台 exact preset。
4. resolved/actual、backing生命周期、完整输出质量可核对。
5. 档位来自 full-request、多 scope、独立确认的实测，不是凭 slot 数或 MLX 单指标猜测。
6. 无匹配项、内存压力、撤回和失败都有清楚反馈，不静默换算法。
7. 8/10 档可对部分模型保持不可用；16/20可共用preset，不制造虚假分层。
8. 对硬 cap/zero swap/快于swap的承诺继续留在独立 bounded/P3 资格中。

本轮交付是该探索与实施规格。执行这些实验、填写真实内存表并启用 public catalog，是后续代码与实机验收任务。
