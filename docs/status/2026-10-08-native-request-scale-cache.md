# Native W1：请求内紧凑权重尺度复用

2026-10-08。请求内 compact row-scale cache 已实现，完整 native/private 构建及生成数据的 GPU／GPU+ANE 路径通过。**本轮没有加载真实模型或运行整图，尚未测量缓存的模型加速收益**。机器记录见 [JSON](2026-10-08-native-request-scale-cache.json)，设计接线见 [13](../design/qwen-image-gpu-ane-command-graph/13-native-session-and-memory-phase.md)、[15](../design/qwen-image-gpu-ane-command-graph/15-native-w1-implementation.md)、[16](../design/qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)。

后续真实 FFN 记录见[OFF／ON 筛选](2026-10-08-native-scale-cache-real-screen.md)：实际 MLX 身份与精确复用通过，但没有稳定完整路径净收益，未追加整图；以下内容保持上一轮生成数据验证的记录范围。

## 实现

新增 `qwen21_native_scale_cache` 严格布尔字段，默认关闭，仅允许显式 native/private Qwen Base W1。schema 1 顶层／schema 2 execution、类型、字段存在性和执行域分别校验；GPU、MLX、其他模型、LoRA／编辑及 public 构建不接受这条实验路线。没有增加磁盘模型或持久化尺度文件。

每层 cache entry 只保存 gate／up／down 的原始 FP16 row-scale bits。cold 保持原 scales→codes，并 gather 到紧凑 buffer；hit scatter 回当前 pitched Surface，再照旧生成 codes。A8／tx 每次重算，W8 codes、restore 及舍入算术不变。复用已有 `tc_ane_w8_scale_copy`，开启时 prepare 增加一个 PSO；W cold 为九个 dispatch，hit 为六个，其中三个是小规模复制。

来源 key 使用真实 allocation ownership control block：原生分配使用实际 owning `Buffer::Impl`；MLX importer 单独持有 `data_shared_ptr()`，继续保留 array owner。缺少真实 allocation identity 的通用 import 不能进入缓存。key 同时覆盖 native backing、完整 view 的 offset／extent／shape／pitch／dtype、不可变 source revision／generation、通道区域、basis／seed／norm／rotation block／staging recipe。融合 gate／up 共用 backing，但 offset 不同，不能互相命中。身份只证明分配生命周期；不可变性仍来自已物化、已同步、由 Session 验证的 request source 合同。

缓存归单个 `DenoiserRun`／controller，保留强 source 和 cache buffer owners，没有跨请求命中。状态为 Empty→Pending→Valid：**只有 sealed producer flags、ANE／GPU 两分支、restore／finite 和 aggregate 物理完成全部成功后才发布 Valid**。encode、event signal 或单独 callback 不发布缓存。执行失败退休整个 controller 的所有 entries；未知或晚到的设备访问仍保留 owner 到物理 drain，不能靠清空 metadata 提前释放存储。

生产配置为 32 个按层合并 backing，有效载荷 **1 MiB**；计划另留每 backing 16 KiB 容量 margin，上界增加 **1.5 MiB**。实际 buffer 只申请 payload，capacity、driver `allocatedSize` 和进程观测分别记录。所有分配使用原 Backend hooks 和同一 phase ledger：开启时 39 sites／137 backings，关闭仍为 38／105。源权重已经属于 baseline，不重复生成账本 claim。不会缓存 32 层完整 W8 codes 的约 2.25 GiB 数据。

按当前 20 步、32 层、首步 GPU 的规则，预计完整成功请求为 **32 次 fill／576 次 hit**，每个 lookup 的单位是一个 gate/up/down layer；这是预计值，本轮没有实测这组整图计数。CLI 监督器已能显式选择缓存，并验收实际启用、容量、payload、fill/hit 及阶段释放，而不是只看请求值。

## 验证

当前完整构建身份为 `tc-runtime-build-v1-16fecb41f7b0243a518481d5f01d5dac5da5e93ac78c3712033f4bb2c7a13dbf`，563 项构建源与 manifest 一致，使用进程内 CLT、native/package-only。public/private 严格 host 检查通过；完整构建仅有三个既有 Metal deprecation warnings，没有构建 App。

CPU 检查覆盖真实 shared ownership control-block、同地址不同 owner、aliasing pointer、过期 owner、融合 offset、view／recipe／source 代次失效、Pending／Retired／旧 invocation 禁止发布，以及 phase 预算、label/hooks、late owner 和零账本。stage source 428 项、layout 82 项、生成探针 CPU 649 项通过；9 项 ABI 请求路由、129 项通用合同（3 项跳过）、7 项实验监督检查通过。

通用合同初次在 sandbox 内因 Metal 设备不可见导致一个已有 profile 测试返回“GPU 不匹配”而失败；同一源码和 binary 在可读取真实设备信息的环境通过。两份原始日志均保留，没有为测试修改匹配规则或模型代码。

GPU staging 的一次有界验证为 A→B→hit A→cold A，generic／register 各一遍，共 **8 submissions／82 tiny dispatch**，无 ANE request、无模型加载、无测速循环。B 的尺度明确不同，确保 bank 确实被覆盖。恢复 A 后，三个 scale 集、codes、A8、flags、cache guards 和 bank padding 与 cold A **逐字节一致**；source 保持不变，16 个录制前拒绝检查通过。见 [staging 回执](../../outputs/native-w1-scale-cache-20261008/stage-byte-reuse.log)。

完整 controller 验证为 M32/H128/F1024/A512、两组独立且不可变的生成权重，同一个 register controller：A slot 0→B slot 1→A slot 0，只执行一次 dense A、三次 hybrid。

| 实际 invocation | 缓存结果 | 物理完成／finite |
|---|---|---|
| A／slot 0 | miss | 通过 |
| B／slot 1 | miss | 通过 |
| A／slot 0 | hit | 通过 |

实际 **2 entries／4608 B payload／2 misses／1 hit**。A 被 B 覆盖后，cold A 与 hit A 最终 BF16 输出**逐位相同**；两者对 dense A 的 RRMS 均 **2.4757%**，在允许的 8% screening 范围内，最大差只作诊断。source／padding、实际 producer／ANE／complement／consumer 完成证据和唯一 ledger 归零通过。owned ledger 峰值 2,326,924 B，进程 peak footprint 103,826,008 B；这不是生产模型或完整请求内存认证。见 [controller 回执](../../outputs/native-w1-scale-cache-20261008/controller-cache-reuse.log)。

tiny 调用的 host/device 时间只保留诊断，不由三次调用推算模型加速比。MLX Data identity 的 importer 已编译并接线，当前设备 fixture 使用原生 generated allocation，尚未单独验证实际 MLX import 的 cache hit；下一次已有真实 FFN 回放会覆盖该链。

## 证据范围与下一步

[上一 register Base 整图对照](2026-10-08-native-register-w1-base-e2e.md) 绑定 `4fd1e980…` 构建，未开启本缓存；当时去噪省时 13.60%、请求省时 9.97%、图像接近。那些结果不转授当前源码或缓存单项收益，也不与 tiny 数据混算。

下一次性能实验先复用已捕获的同一个真实 FFN 输入，比较完整 cache-off／cache-on 路径，检查实际 MLX allocation identity、cold fill／hit、完整误差／物理完成／账本与准备成本。净收益成立后才增加一组匹配 Base E2E。本轮不再跑模型。

当前仍是 W1S 逐层主机准入。LoRA 全目标及 pre-SiLU／down ABI、编辑／encoder、Z-image、W1R/W2 设备图和 App 高级配置仍需继续实现和分别验收，不因缓存完成而缩减目标。

## 清理

已删除本轮明确归属的 **240 个文件／20,894,148 B（19.93 MiB）**：object/static archives、module cache 和一个 owned ANE program cache。测量二进制、library/CLI、manifest、日志和必要 CPU 回执保留，摘要未变。清理明细见 [owned cleanup](../../outputs/native-w1-scale-cache-20261008/owned-cleanup.json)。没有下载／加载模型，没有复制完整 checkpoint，历史证据及系统 driver cache 保留。
