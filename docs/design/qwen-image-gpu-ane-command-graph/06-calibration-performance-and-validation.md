# 06 · 校准、性能归因与验证方案

状态：验证设计，未运行本章提出的新实验。既有数据见 [01](01-evidence-and-gap-analysis.md)。本机为笔记本，采用“静态审查 → 无模型合同测试 → 少量真实组件 → 最多两组 E2E”的推进方式。

## 1. 四个独立问题

必须分别回答：

1. 数学和状态是否正确，LoRA、文本、参考图是否真的参与？
2. 外部设备访问是否安全，失败/取消/迟到回调是否能恢复？
3. 在相同算子、数值合同和生命周期下，新调度是否更快？
4. 加上初始化、encoder、VAE 和 App 开销后，用户请求是否更快？

不能用第四项的 worker/cache 变化证明第三项，也不能用组件第 3 项通过代替第 1、2、4 项。

## 2. 对照组和可归因改动

| 对照 | 保持一致 | 只改变 | 能回答 |
|---|---|---|---|
| A：当前最佳 GPU vs 当前最佳 hybrid | source、prompt、seed、steps、缓存、构建 | 既有完整路线 | 复现历史方向；非新图收益 |
| B：legacy hybrid vs event-graph hybrid | kernel、份额、rows、recipe、LoRA 算法、worker | 提交/同步方式 | command graph 净收益 |
| C：event graph 旧 kernel vs 新 kernel | 图、请求、数值合同 | 一个 fusion / layout 优化 | kernel 净收益 |
| D：GPU legacy worker vs GPU retained worker | 请求、GPU 算子、精确缓存策略 | 生命周期 | worker 复用收益 |
| E：GPU retained vs hybrid retained | 同 worker/cache/验证冷热状态 | ANE 路线 | 用户连续请求的 ANE 净收益 |
| F：native GPU vs MLX GPU | 请求、模型语义、允许误差、冷热条件 | GPU executor | 新后端 GPU 路线是否退化 |
| G：N-GPU vs N-GPU+ANE | 同一原生后端、模型/条件、相同 GPU kernel 家族 | ANE 通道分支与其准备/同步 | ANE 在新后端内的净收益 |
| H：当前最佳 hybrid vs N-GPU+ANE | 请求、source、质量要求、冷热条件 | 完整后端 | 是否超过现有可用方案 |

这里的 A–H 是对照编号，独立后端路线 A/N 的定义见 [10](10-native-metal-graph-backend.md)。N 后端至少回答 F/G/H 三个问题，分组件和后续会话验证，不增加单次实验预算。

不要求一次全部运行。这是可归因矩阵；每个阶段只选择要验证的一个问题。当前最佳 base 与 LoRA 使用不同份额/提交策略，不能强行用一个全局开关作为二者 baseline。

对质量使用 04 的双 oracle：

- **整请求纯 GPU oracle**：同初始 noise、全部步骤、相同模型/LoRA/条件；
- **兼容 hybrid oracle**：复现已接受的分片与舍入合同，帮助定位新调度引入的误差。

事务失败后的 GPU replay 从已提交检查点继续，之前可能已有合法 hybrid 结果。它是恢复路径，不能标为整请求纯 GPU oracle。

## 3. 测量合同

### 3.1 端到端边界

`request_wall` 从收到已序列化的请求后进入 native 请求执行开始，直到最终产物和校验结果可返回；排队/UI 用户思考时间另列。包含实际发生的源验证、模型加载、程序准备、推理、转换和产物写出。若现有工具的时间边界不同，保留其字段并新增显式边界，不静默更改含义。

同时记录：

- App queue / process spawn / IPC / request validation；
- source verification、weight load、program compile/load、binding、PSO compile；
- text/vision encode、prefill、steady denoise、VAE；
- graph build、host submission、device execution、drain；
- output serialization / artifact validation。

具有嵌套或重叠关系的字段标注 span，不把它们相加作 total。

### 3.2 设备时间和 host 时间

| 指标 | 定义 | 不能解释成 |
|---|---|---|
| `gpu_cb_start/end` | 支持时记录每个相关 CB 的 GPU 时间 | 最后一个 CB 的时间代表整个 FFN |
| `ready_observed_host` | host 收到 ready 通知的时间 | 精确 GPU signal 时间 |
| `ane_callback_host` | driver completion callback 到达 | 独立硬件 ANE kernel 时间 |
| `ane_api_wait` | host 在 API/condition 上等待 | 可与并行 GPU 时间相加的耗时 |
| `stage_wait_exposed` | 主执行链等待 staging 的时间 | staging 的总 GPU/带宽成本 |
| `graph_build_cpu` | CPU 构建/绑定/编码时间 | 一定暴露在关键路径上的开销 |
| `physical_release_latency` | 最后物理用户完成到资源可复用 | 结果提交延迟 |
| `fallback_wall` | 重算与恢复成本 | 正常 hybrid 已经完成的性能样本 |

为每次请求/step/window/layer/job 分配关联 ID。不得为了计时在每个普通 tensor 后增加 `mx::eval`，否则测到的是被 instrumentation 改坏的路径。

设备支持时用完整 CB timestamps / counter 采样辅助定位；ANE 缺乏可用硬件 trace 时明确写“host/API 观测”，不伪造物理 overlap 比例。现有 `z_image_gpu_time_probe.cpp` 是最后 CB 的诊断示例，不能直接当作整图计时器。

### 3.3 内存和磁盘

记录物理 footprint、owned resident、pending-release、quarantined bytes、framework allocator/cache、系统可用内存与 swap delta。共享 buffer 不因 GPU/ANE 两个视图重复计数。

记录 source bytes 与 derived bytes，编译缓存、诊断输入、图像、日志分开。清理只按 owned manifest 执行，见 05；无法精确统计系统 ANE/Core ML 缓存时明确未知，不声称全部已删除。

## 4. 校准设计

### 4.1 状态

```text
Unknown
  -> CapabilityChecked
  -> CandidateGenerated
  -> NumericalScreened
  -> FullPathMeasured
  -> LocalExperimentalAccepted
  -> Qualified(scope)

任何阶段 -> Rejected(reason) / Invalidated(reason)
```

`LocalExperimentalAccepted` 是本机小样本可继续探索，不等于自动选择资格。持久校准文件只有通过构建/source/recipe 校验及必要的轻量 recheck 后才产生有效对象，读取 JSON 本身不授权执行。

### 4.2 搜索空间顺序

初始固定当前已接受的 Hadamard W8A8 recipe 与数值边界：

1. Base 阶段先验证既有候选 6144 通道的提交方式；后续 LoRA 阶段另以其既有 4096 候选起步，不同时搜索。
2. 优化稳定后，只测该份额相邻的 512-channel unit 或已有第二候选，不重新扫描全部份额。
3. 单独比较首步 GPU 与一个邻近较大 row function；steady 固定 1024。
4. Base 的 gate/up fusion 独立验证；LoRA scratch 共享等到后续 LoRA 阶段。
5. 最后才引入新 SmoothQuant、rotation/group size、FP32 join 或 decoded-weight cache。

每增加一个改变 rounding/误差分布的选项，生成新 QuantRecipe/RoundingContract 身份并重做数值门，不把它当作同一调度 profile 的开关。

### 4.3 实际输入与形状

优先复用既有 144 MiB 的真实 FFN 样本：层 0/15/31、早/中/末步骤、base 和 LoRA 分开。校准用早期样本，保留中/末样本做独立检查；不得用全部样本搜索阈值后仍称“未参与校准”。

增加完整 block 测量需要 attention 输入、prefix/KV 和 modulation 等边界；**现有 FFN 样本不足以重建完整 block**。若缺少相应样本，先定义小范围、可清理的采集 schema，再在一轮短运行中采集所需边界，不下载额外模型。

该采集每个待测工作负载至多执行一个受限 step、只保留少数 block 边界，不执行 VAE 或最终产图，计入组件 GPU 时间与磁盘预算。优先使用已有可匹配样本；不得为凑齐中/晚步 block 样本另跑完整 base 20 步或 LoRA 六步。缺少覆盖如实记录，带 capture 的运行不能作为 E2E baseline，也不自动增加第 7.4 节每会话最多两组、四张之外的完整生成请求。

样本身份包括 source、adapter/strength/order、step/sigma、layer、text/reference geometry、precision、capture build。不能拿 base activation 加一个 LoRA 名称作为 LoRA 校准。

### 4.4 成本模型

用于候选排序的近似式：

```text
T_FFN ≈ T_prepare_exposed
      + max(T_GPU_complement_with_contention, T_ANE_path)
      + T_join_and_remaining_LoRA
      + T_unhidden_submission
```

GPU staging 与 GPU head 共用设备，staging 不能既当作完全并行的第三设备又从 GPU 成本中删掉。模型拟合必须对应同一编译、row、LoRA 和窗口策略。

借鉴 Splash 的两份额、邻居稳健选择、最低收益和逐 row 检查；保留 TurboCider 对带宽耦合不确定性的保守 envelope。不要因为上游假设高份额 ANE-bound，就把 Qwen 所有形状强制按该假设拟合。

选择建议：

- 对测量范围外不外推；
- 平均值不错但某深度/步骤明显退化时，不直接全层启用；
- 近似最优范围内优先较小内存与较少 ANE 通道；
- 是否需要 per-layer exception 由独立验证决定，避免九个样本拟合 32 个自由份额；
- base、LoRA、prefill、steady、reference edit 分别选择；
- 预测值之后必须测量完整候选，不能直接成为产品配置。

## 5. 正确性与数值门

### 5.1 指标

参考输出为 `y_ref`：

```text
RRMS = sqrt(mean((y-y_ref)^2)) / max(sqrt(mean(y_ref^2)), epsilon)
```

同时报告最大绝对误差、非有限数、cosine、输出能量、每样本指标及最差样本；低能量样本不能仅依赖相对误差。保留既有工具定义并记录 epsilon，避免同名指标计算方式不同。

现有量化候选曾用 RRMS 0.03 做组件拒绝线；保留这一历史门的身份，不能与新的探索门混记。用户后续允许的 **5–8% 数值差异可作为候选筛选参考**，但必须明确具体指标、张量与归一化方式，独立记录放宽后的候选 recipe，并以同 seed 图像的肉眼对照作最终验收的一部分。它不是所有层或整图的统一 8% 质量许可；也不能由组件 RRMS 推导图片“质量下降百分比”。

筛选容差不能代替下列合同：

- 调度、缓存命中、布局视图重构：应保持已声明 rounding contract，先比较逐张量与重复稳定性；不能用量化阈值掩盖 race。
- kernel reduction 顺序改变：先建立更严格的浮点容差和与 GPU oracle 的基线，再决定是否接受。
- 量化 recipe：校准外样本必须过门；平均通过而单层/晚期严重失真则拒绝。
- LoRA：检查 base→adapter 的实际效果与 low-rank 中间结果，防止 adapter 被漏算却因总 FFN 误差小而过门。

### 5.2 必要覆盖

无模型/小 tensor 测试覆盖：

- gate/up 的 LoRA 插入位置、down 分片 FP32 累加与舍入；
- strength=0、正负 strength、rank 和 adapter 顺序；未资格组合可正确回退；
- rows 为 bucket−1、bucket、bucket+1，padding 必须清零且输出 trim；
- 同 shape 不同权重/LoRA generation 不错误命中；
- prefix hit/miss、末层 target-only、参考图顺序与几何；
- dense/affine/GGUF/ConvRot recipe 不兼容时明确拒绝；
- 空的/非有限/overflow 输入及状态校验。

E2E 还要检查 latent、decoded pixels、构图/文字/编辑条件，而非只看 PSNR。已有单图 PSNR 37.73/42.13 dB 不能形成全局画质标准。

## 6. 同步和资源门

03 的故障注入矩阵必须先通过，至少包含：

1. ANE 部分 enqueue 成功后异常；
2. GPU CB commit 前后异常、callback 比登记更快；
3. ready 未发生、ANE callback 迟到/失败/永久缺失；
4. done 被强行推进但 ANE 仍读写 surface；
5. 同 bank 下一层写入、同 output slot 下一请求写入；
6. epoch 退役后 callback 到达，不得修改新实例；
7. abort 时已排队 restore 的读取权限；
8. 取消 during prepare / running / drain，不能变成成功；
9. GPU replay 自身失败，不递归重试或提交旧结果；
10. worker 隔离后的 pending/quarantine 内存不偷偷归零。

先用 fake executor、CPU 可控制 barrier 和记录访问区间的 allocator 测试时序；真实 driver 层最后做极少量受控验证。不在笔记本上通过反复压满内存或触发长时间 GPU hang 来测试异常。

## 7. 轻量实验预算

### 7.1 第 0 层：只读和 CPU 合同

检查 code/build identity、引用、schema、计划 DAG、资源冲突、cache key、状态机与故障注入。此阶段不启动模型，也不为验证不存在的代码编造通过记录。

既有工具可参考：

- `tools/native/qwen21_real_ffn_probe.cpp`：生产 HybridFfn 回放入口；
- `tools/validation/qwen21_real_ffn_screen.py`：读取 replay reports 并汇总；
- `tests/native/test_real_ffn_screen.py`、`test_ane_replay_capture.py`、`test_qwen21_lora_calibration.py`：采集和数据合同回归；
- `native/backends/ane_cost_model.hpp`：成本模型纯 CPU 核心。

改造后应扩展现有 probe 的 executor 选择，而不是另写一个只测孤立 ANE 的“更快”工具。

### 7.2 第 1 层：真实 FFN

Base 阶段只选择一个 base 候选；后续 LoRA 阶段另选一个 LoRA 候选，不要求一起完成。使用预定 warmup 和两轮相反顺序测量，各自九个样本按固定次序或记录的交错次序处理。测量计算预算建议限制在数十秒量级；编译/首次加载单列、有界且可取消。

如果数值失败、出现 fallback/retry、某轮环境污染或设备压力，记录原因并停止该候选；不得丢掉慢样本后自动追加很多轮。

### 7.3 第 2 层：完整 block

在已验证的 FFN 上加入真实 attention、residual 和 next-weight staging；LoRA 阶段再加入真实 LoRA 分支。测当前最佳 GPU、legacy hybrid、新 graph。重点检查“FFN 快了，但 attention 因 staging 或新 kernel 变慢”。

示例推进标准：新图在所测完整 block 中稳定改善，代表点平均至少约 5%，最差点没有超过预设噪声带的系统退化，且零正确性/访问生命周期错误。5% 是待固定的工程推进线，不是统计显著性声明。

### 7.4 第 3 层：最多两组 E2E

每次笔记本会话最多两组配对、四张。Base 阶段优先只做一组 base 20 步，GPU / 候选各一张；若需第二组，只复核一个明确问题或反向测量顺序，不自动扩展 LoRA。LoRA 在 Base 独立通过后，于后续轻量会话做自己的组件和配对 E2E。首组 GPU→candidate，必要的第二组 candidate→GPU；这不能消除所有热状态偏差，只是有限预算下的方向筛选。

新图与 legacy 的调度对照固定相同 first-step 策略；首步改用 GPU 是另一项变化，不可一起算成 graph 的独立收益。两组不足以回答所有冷热、份额和 phase 问题时，保留未测项，不追加矩阵。

- 固定相同构建、prompt、seed、步数、尺寸、源、adapter；
- 独立记录首次进程、已有磁盘程序、保留 worker 三种冷热状态；不要一轮混合比较；
- 关闭性能无关的 debug/capture、PE 和近似 cache；
- 不同时编译、导出或运行另一个模型；
- 使用系统热状态/功耗模式记录；受热限制时结束该轮，不连续高频补跑；
- 候选失败即停止，保留 GPU 功能；
- 不在此轮扩展到所有长文本/参考图；后续按功能域逐步做小组验证。

通过只标本机实验资格。要转为自动默认，还需在后续分散会话补充不同 prompt/seed、长文本、缓存命中/失效和编辑场景；保持每次笔记本预算，不把发布覆盖压成一次长压测。

## 8. 性能接受与自动回退

建议分别保存：

- `request_gain_first`：匹配首次/缓存条件；
- `request_gain_reused`：匹配连续请求条件；
- `denoise_gain`：辅助定位；
- `quality_scope`、`memory_scope`、`failure_scope`；
- GPU baseline 与 candidate 的完整 implementation/build identity。

最小自动收益阈值建议从约 5% 起作为配置项，同时检查绝对节省和噪声。收益不足时保留 GPU 自动选择，允许诊断模式显式探索，但不要显示“已加速”。

运行中少量代表窗口采样，不要每层都做 GPU 双算。出现持续亏损、内存压力或 source/OS/kernel 变化时降级/失效；数值和设备失败可立即停止当前 hybrid。降级的请求仍记录已发生 ANE 工作和 replay，不伪装为一次纯 GPU 请求。

## 9. 收益敏感性与投入边界

基于现有候选，假设仅去噪继续缩短，其他阶段不变：

| 假设 | base wall | 比原 GPU 少 | LoRA wall | 比原 GPU 少 |
|---|---:|---:|---:|---:|
| 去噪再降 5% | 41.28 s | 14.7% | 20.61 s | 0.3% |
| 去噪再降 10% | 39.52 s | 18.3% | 19.93 s | 3.6% |
| 去噪再降 20% | 35.99 s | 25.6% | 18.56 s | 10.2% |

这是算术场景，不是预测。新图必须测出可对应的实际变化。LoRA 在 setup 不变时需要当前去噪再缩短约 4.61% 才能跨过原 GPU 总时间；仅删除全部 1.187 s setup 的乐观算术结果约 20.105 s，也只有约 2.7% 净优势。

32 层×20 步每层减少 1 ms 仅约 0.64 s；六步约 0.192 s。因此优先消除**实际暴露的**同步、重复计算和内存流量，不能把 CPU API call 数量降低等同于大幅端到端加速。

## 10. 验证回执草案

以下是拟议 JSON，不是现行 schema；省略项需要由实现定义类型，示例不会被现有 loader 接受。

```json
{
  "schema": "tc-hetero-evidence-proposal-v1",
  "status": "local_experimental",
  "scope": {
    "model": "qwen-image-2.1",
    "operation": "image.generate",
    "width": 512,
    "height": 512,
    "references": 0,
    "batch": 1
  },
  "identities": {
    "source_digest": "<verified>",
    "adapter_set_digest": "<canonical-empty-set-for-BaseOnly>",
    "rounding_contract": "<version>",
    "quant_recipe": "<digest>",
    "executor_build": "<digest>",
    "gpu_baseline_build": "<digest>"
  },
  "control": {
    "worker_state": "fresh-process-existing-program-cache",
    "approximate_cache": "off",
    "instrumentation": "boundary-only",
    "sample_order": ["gpu", "candidate"]
  },
  "result": {
    "numerical_pass": false,
    "resource_pass": false,
    "performance_pass": false,
    "actual_ane_calls": 0,
    "replay_count": 0,
    "claim": "unmeasured-placeholder"
  }
}
```

真实回执不得只保留最后通过的一份；记录所有候选与拒绝原因、原始计时/误差文件、源哈希、清理清单。可删除 owned 大张量与无效程序，保留小型证据索引和必要有效样本。
