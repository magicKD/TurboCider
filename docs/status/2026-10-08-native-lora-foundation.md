# Native LoRA hybrid 基础实现与本轮结论

日期：2026-10-08，M4 Pro／48 GiB。**本轮完成了独立 Metal Graph 后端的 LoRA hybrid 基础接口；尚未接通 LoRA GPU／ANE 并行请求，也没有新的性能资格。** 默认路线保持原有行为。没有下载模型、复制 checkpoint 或导出全模型；只做两组轻量设备验证。

完整设计见 [16 · Splash 风格后端规格](../design/qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)和 [17 · Native LoRA](../design/qwen-image-gpu-ane-command-graph/17-native-lora-integration.md)。机器可读身份／证据索引见[本轮 JSON](2026-10-08-native-lora-foundation.json)。

## 代码实现

1. [NativeLoRA](../../native/backends/metal_graph/native_lora.hpp) 增加 `validate_projection`、`encode_low`、`encode_apply_inplace`。descriptor 保留实际 views／owners 和完整项选择的 Scalar／SG 数学，复用原 PSO／shader；旧 ordered correction／SiLU 行为和类布局不变，默认零额外 PSO。low 内容身份及跨图就绪仍由调用方证明。
2. [native_ane_lora_contract.hpp](../../native/backends/metal_graph/native_ane_lora_contract.hpp) 增加不可变 Base／HiddenReference／Rank128Candidate Spec，统一输入、packed 输出行段、pitch／capacity、writer masks、source／adapter／invocation 元数据。它不创建 Surface、验证 held FD 或生成物理 permit。
3. [ane_mil.cpp](../../native/backends/private/ane_mil.cpp) 增加 `native_lora_program(Spec)`。Hidden 在 Base 七输入上增加 `dg/du`，输出 `H+1+C`；Rank 再增加 `ad[128,C]`，旋转／A8 前用有序 FP16 分块求 low，输出 `H+1+128`。临界有限值 guard 聚合到无效 hscale 哨兵。动态 factors 不嵌入常量，旧 emitter 的 MIL／rotation bytes 保留。
4. 新增 [shared-low 微型工具](../../tools/native/native_lora_shared_low_component.cpp)和 [MLX 完整 FFN 暖机对照](../../tools/native/native_qwen_lora_mlx_ffn_replay.cpp)。前者验证共享一次 low 的两个 B 行窗；后者执行实际 MLX 编译 closure，避免把首次 oracle 执行当作热性能基线。

真实 Spec 域固定 M1024／H4096／F12288／C6144、一个 r128／strength1 adapter、headroom1／无 bias。Hidden 增量 surface payload 36 MiB；Rank 增量 25.75 MiB；两者后续另需 0.5 MiB low。Base 九 surfaces 的 payload 为 85.00390625 MiB、capacity upper 为 85.03125 MiB。它们只是精确组件预算，不能当作整个请求或进程上界。

## 构建与 CPU 验证

private package 构建通过，runtime identity 为 `tc-runtime-build-v1-47d3ed80c5de774df4ffa3ed634f9d362a0074f90c8b426a13df6f10abd4904a`；571 个 identity sources 摘要一致，前后未改动。只有既有 `ane_gpu.mm` 的三条 deprecated `fastMathEnabled` 警告。package-only 没有自动运行全测试集。

| 检查 | 结果／边界 |
|---|---|
| split API | 新 172 项；连既有 contract／source／math 共 1646 项，public/private host 通过；旧两个 shader body 字节不变 |
| Spec metadata | 244 项，含 209 个拒绝案例；CPU-only，没有设备／MLX 依赖 |
| MIL emitter | 685 项，4 组 legacy byte fixture；逐输入 dtype／shape／pitch、所有 guard token 的最终可达性和负向修改通过 |
| 两个 probe | strict host／CPU 合同和新库链接通过；CPU 模式不加载模型或运行设备 |

[构建日志](../../outputs/native-lora-foundation-20261008/full-build.log)、[split 检查](../../outputs/native-lora-foundation-20261008/split-host.log)、[MIL 检查](../../outputs/native-lora-foundation-20261008/mil-cpu-v3.log)保留。MIL 只是成功生成源码，**没有新 ANE program 的编译／运行、placement 或硬件 INT8 证据**。

## 两组有界设备工作

**第一组：共享 low 的 tiny 正确性。** M8／K128／N128／r128，BF16 dyadic fixture，实际 held-fd Weights loader／supplied-base oracle。旧路径 low＋apply 为 2 dispatch；新路径一个 low＋两个 64 行 B slice apply 为 3 dispatch。共两次 native Ticket、五个 dispatch；两个 low、两份 output 与实际 MLX／Weights oracle逐位一致，finite、源 owner／hash、padding 和新增 ledger 归零通过。[原始回执](../../outputs/native-lora-foundation-20261008/shared-low-device/receipt.jsonl)与[监督记录](../../outputs/native-lora-foundation-20261008/shared-low-supervisor.json)保留。

四个 scratch capacity upper 共 64 KiB，生成 adapter 文件 65,783 B。监督器墙钟约 0.697 s；它不是性能基准。共享 low 的就绪只由同图执行顺序和成功物理 Ticket 证明，不是跨图或 GPU／ANE readiness 资格。

**第二组：真实 MLX／Native 完整 FFN 对照未完成。** 使用已有 layer0／step5 r128 输入，同一份 held Weights，仅物化该层三个 base 权重、六个 factors 和输入，payload 308 MiB；没有 materialize 整个模型。计划 warm AB、正式 AB／BA，共 3 MLX＋3 native，无额外 oracle。实际只完成 warm AB 和正式 AB，共 **2 MLX＋2 native**，第二对 checkpoint 报：

```text
memory_pressure_abort: unattributed process footprint exceeds framework upper
```

512 MiB framework allowance 是未认证的初始估算。护栏正常中止此候选；未关闭护栏、未扩大预算重跑，未执行 BA 或 E2E。[failure receipt](../../outputs/native-lora-foundation-20261008/mlx-ffn.json)保留已完成的四个 arm 和错误，[监督记录](../../outputs/native-lora-foundation-20261008/mlx-ffn-supervisor.json)记录退出 1、约 1.942 s、无超时／信号／重试。

MLX 完成证据是实际 `eval/synchronize`，native 是每次成功物理 Ticket；不是进程退出。failure receipt 确认所有已开始工作到达完成边界、无 uncertain owner quarantine，但没有保存完整 pair 数值／来源末尾复核、memory report 和 phase release 指标。因此本轮不声明完整数值资格、真实对照 ledger 归零、稳定快慢或主瓶颈。不能拿暖机的首次编译差异当作提速。

### 框架预算为何需要重做

新 probe 在首次 MLX 执行之前取 baseline；旧 paired probe 则在 oracle 执行并清理 cache 之后取 baseline。复用同一个 512 MiB allowance 没有覆盖这种边界变化。正常 [Weights::project](../../native/backends/mlx.cpp) 会产生 FP32 input／factor casts、low、完整 delta 和校正后的 concat。MLX 首次 `compile` 调用才建立 trace／fusion；输出 Data owner 退休也不意味着 allocator 的 cached Metal buffer 已归还系统。这里更支持“初始预算覆盖不足”的解释，**不是已经证实泄漏或系统 OOM**；本次缺少失败峰值，无法进行精确归因。

对 M1024／H4096／F12288／r128，下面是按源码得到的**单次未融合逻辑分配总量**：不假定 CSE／fusion／donation，**不等于同时存活峰值，更不是物理 footprint 上界**。

| 类别 | MiB |
|---|---:|
| base gate/up 与两次 corrected concat | 144 |
| 两个 BF16 corrected half | 48 |
| sigmoid／SiLU／hidden | 72 |
| down base 与最终 BF16 输出 | 16 |
| FP32 X／hidden casts，不假定两次 X cast 共享 | 80 |
| 六个 FP32 factors | 24 |
| 三个 FP32 low | 1.5 |
| 三路 FP32 delta matmul 输出 | 112 |
| scale／base-cast／add 的未融合输出 | 336 |
| 合计 | **833.5** |

临时 split-K、缓存保留、页对齐、编译 tape／library／PSO 和 driver 另计；实际选择、liveness 与共享情况必须在下一轮回执中确认。不能直接拿上表总量提高 guard 后宣称内存可控。预算应拆为 active 增量、cache policy、backend 临时区、compiler／driver allowance，保留 source baseline 与 native scratch 的独立口径。显式 cache cap 若被采用，会改变 MLX allocator 行为，必须报告，不冒充默认 MLX 配置。

[MemoryAdmission::checkpoint](../../native/runtime/memory_accounting.cpp) 已在抛错前保存 failure footprint／unattributed／phase，当前 probe catch 没把它们写出。下一版必须先保存失败边界 `metrics()+admission.snapshot()`，再 release／finalize／`take_report()` 保存清理终态；只在释放后取 metrics，会用旧 footprint 减少后的 known bytes，混淆失败时的 unattributed。pair 诊断也应先保存再执行 guard；run 整体仍标 failed，不赋性能资格。上述是代码阅读结论，本轮没有重跑或修改 probe。

## 设计修正与下一步

- 新后端 N 独立编码并提交完整 DiT／Euler，按 Splash 的 graph／program／binding／handoff／资源组织推进；A 继续保留 MLX。两路线共用模型来源、内存合同和资格管理。
- 同 backing、F-pitch hidden 的左右列窗需要带步长的逐行访问区域。两个包围 ByteRange 会重叠；设计增加 StridedRows／跨行 hazard 合同，不能用不同 wrapper 或 storage 假身份绕过 alias。
- 精确区分已实现 Spec／emitter／split 与待实现 delta／Adown writers、hidden／rank restore、recipe-specific sealed status、controller 和模型／Session 路由。Base masks 保持 15／16，新 masks 不能直接传给旧 AneStageStatus。
- 显式 native 超域／缺 capability 要拒绝；自动策略才可选择兼容路线。ANE request／event wrapper 一次性，底层 SharedEvent timeline 可复用但值不能重用。

下一轮先补齐 probe 失败时的 process baseline／current／peak、MLX active／cache／peak 与 native ledger 诊断，按 source、compiled constants、FP32 rank 分支与 allocator liveness 推导预算，再做有限公平对照。性能改造仍需可隔离的完整路径证据；不简单提高或删除 guard。

hybrid 首包继续先做 HiddenReference 的 delta writer／status／restore 和单 bank W1S 控制器，生成数据闭环后再用本地真实 FFN；Rank 是独立候选。组件有净收益后才接模型和有限整图。共同 IR、跨请求准备资产、W1R／W2、编辑 encoder、Z-image 和 App 仍需独立实施／验收，旧 Base 一组收益不转授它们或本轮 LoRA。

## 清理

[owned 清理回执](../../outputs/native-lora-foundation-20261008/owned-cleanup.json)记录删除 250 个文件、34,613,684 B（约 33.01 MiB）：只涉及本轮 `.o`／`.a`／module caches、八个 rotation fixture 与微型生成 adapter。原模型／capture、库／CLI／probe、原始日志与 receipts 都保留；没有删除系统／Apple driver cache。

清理摘要：`a8012b47fc0e3343fd6f7c5087645059a38f522769fb4ec87f84f1856155e4c2`。完整功能目标仍在推进，本轮属于基础实现进展，不能标为正式 LoRA GPU／ANE 加速交付。
