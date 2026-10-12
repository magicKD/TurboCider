# Native W1：真实 FFN 尺度缓存 OFF／ON 筛选

2026-10-08。仅进行一组真实 layer 0、step 19 输入的 OFF／ON 完整 FFN 对照。**实际 MLX Data 身份、冷填充后命中、BF16 输出逐位一致和物理完成／释放均通过；两次正式记录的收益随执行顺序反转，未证明稳定净收益。** 因此保持 `qwen21_native_scale_cache` 默认关闭，不追加整图 E2E，不授予性能或产品资格。

原始证据：[组件回执](../../outputs/native-w1-scale-cache-real-20261008/component-receipt.json)、[进程计时与内存日志](../../outputs/native-w1-scale-cache-real-20261008/component-stderr-time.log)；汇总与 artifact SHA256 见[机器记录](2026-10-08-native-scale-cache-real-screen.json)。此前的实现和生成数据验证继续保留在[请求内尺度缓存状态](2026-10-08-native-request-scale-cache.md)，本记录补充其尚未覆盖的真实 MLX import／FFN 路径，不改写历史回执。

## 配置与执行范围

实际和工具预期的 library 身份完全一致：

```text
tc-runtime-build-v1-16fecb41f7b0243a518481d5f01d5dac5da5e93ac78c3712033f4bb2c7a13dbf
```

最终核对 563 项 production source 与 manifest 匹配；机器记录分别保存 11 个本次测量 artifact 和 7 个后续工具 CPU artifact，18 个哈希均通过。测量工具最终 strict host 编译／链接成功最初记录于工具调用输出，未单独保存为编译日志，不能用旧日志替代；最终测量 binary 的 CPU-only 输出另存为 [component-cpu.log](../../outputs/native-w1-scale-cache-real-20261008/component-cpu.log)。

设备为 Apple M4 Pro；本地 Qwen Image 2.1 BF16 Base，M=1024、H=4096、F=12288、ANE channels=6144，register W staging，两臂均为 private native W1S。固定同一已捕获的 layer 0／step 19 输入：一次 warmup OFF→ON，随后各一次正式 OFF→ON、ON→OFF，**共六次完整 hybrid FFN 调用**。没有 dense GPU 对照、完整 DiT、图像生成或新 E2E，也没有扩大份额／量化／输入扫描。

没有下载模型或复制完整 checkpoint；只物化已有模型的 layer 0 三组 FFN 权重及已有捕获输入。该回放不覆盖首步 prefix-build、32 层循环、LoRA、编辑、Z-image 或 App。

## 正确性、身份与生命周期

ON 实际为 **1 entry、32768 B payload、1 miss、2 hits**。每次正式记录重新 import gate／up／down，新的 array／view wrapper 均对应真实 MLX `data_shared_ptr()` 控制块和同一 native backing。融合 gate／up 共用实际 Data，但 offset 分别为 0 和 100663296 B；换 wrapper 后仍正确命中。这个结果补足真实 importer 的设备路径，分配身份本身不替代来源不可变合同。

三组 OFF／ON 的最终 BF16 输出均**逐位相同**：4,194,304 个元素有限，RRMS=0、max abs=0、cosine=1。此模式以 finite 和逐位一致作为验收条件，8% RRMS 仅保留诊断，不能用小幅误差掩盖缓存改变算术。它证明相同 W1S 算术下缓存复用精确，不是与原始 dense 模型的质量对照。

held source 与捕获文件前后校验通过，三组权重及输入 tensor 的前后 SHA256 一致；来源证明范围为持有文件描述符、stat generation 和实际参与计算的 tensor，不是整份 checkpoint 的内容认证。六次调用均物理完成，所有 controller／graph／view owner 释放后同一 ledger 归零。没有以超时解锁或清空 claim 代替物理释放。

## 完整 FFN 成本

外层 wall 包含 plan／encode、key／lookup、提交和物理完成；allocation、program／PSO prepare 另列。正式均值排除 warmup：

| 样本 | 实际顺序 | OFF wall | ON wall | ON 相对 OFF |
|---|---|---:|---:|---:|
| Warmup | OFF→ON | 48.428 ms | 28.232 ms | 仅保留冷路径诊断 |
| Record 1 | OFF→ON | 35.269 ms | 24.406 ms | 快 30.80% |
| Record 2 | ON→OFF | 25.934 ms | 37.190 ms | 慢 43.40% |
| 两次正式均值 | 各一种顺序 | 30.601771 ms | 30.797938 ms | 慢 0.641% |

样本太少，0.641% 不能证明稳定退化；同样也没有稳定加速证据，不能只选第一组的 30.80% 作为结论。ON 两次正式记录均实际命中，顺序反转并非缓存未启用。

producer GPU Ticket 跨度在 Record 1 为 OFF 11.668／ON 2.651 ms，Record 2 为 OFF 4.042／ON 11.671 ms；complement GPU 跨度也从 21.909／20.492 ms 变为 20.488／23.311 ms。变化出现在设备跨度中，但这些跨度包含 command buffer 间隙，不是 kernel busy time。现有证据无法归因为频率、系统调度、driver cache、带宽争用或某个 shader；不据此宣称已证明 GPU／ANE 物理重叠或硬件 INT8 算术。

OFF controller prepare 为 1.314224 s，ON 为 1.002958 s；共同 source materialization 为 0.048350 s。两臂使用不同新建 owned MIL 目录，但同一 Backend 上先准备 OFF、后准备 ON，系统 compiler／driver cache 共享且冷状态未知，不能据此认定缓存缩短了冷启动。ON 增加一个 scale-copy PSO 和紧凑尺度 buffer，其成本必须在后续完整请求比较中保留。

## 内存与证据边界

两套 OFF／ON controller 同时常驻并共享工作区：18 个 Surface 共 178,323,456 B，native owned ledger 峰值 **253,874,204 B（242.113 MiB）**。这些是整组组合用量，**不能作为 scale cache 的增量内存**。本组 ON 只有一层 32 KiB payload，未测生产 32 entries／1 MiB 的请求循环。

进程观测 baseline 为 321,454,920 B、结束采样为 591,692,736 B；`time -l` 记录整个探针进程 peak memory footprint **678,937,512 B（647.485 MiB）**，maximum RSS 为 366,346,240 B。ledger、瞬时 footprint、whole-process peak 与 RSS 是不同口径，不能互相替代；这仍不认证 compiler／driver 或完整图像请求的内存容量。日志中的 process swaps=0 也不等于全系统没有内存压力。整个探针进程 real 5.64 s 包含准备和校验，不是完整模型推理时间。

## 决策

本次只授予该构建、该真实输入下的精确缓存复用与成功路径生命周期证据。**没有稳定完整 FFN 净收益，因此不追加 E2E，缓存继续显式实验、默认关闭。** 不从六次调用推算 608 次 FFN、32 misses／576 hits 或完整请求收益，也不继承旧 `4fd1e980…` 缓存关闭时的整图加速结果。

后续只读复核发现，每个 pair 的先运行者 producer GPU 跨度较长；且原探针在 pair 间对每个元素重复调用 contents/type/require，未记录这段 CPU 比较耗时。重复检查是可移除的工具开销，但它导致 GPU 空闲／降频仍是假设，不据此改写本次结果。后续先优化这段比较并做纯 CPU 合同验证，增加独立的 comparison_cpu_seconds，之后再决定是否值得一次有明确假设的轻量对照；本轮不再运行模型。

从 staging 源码看，OFF 的三组 scales＋codes 为 6 个 W dispatch，ON cold fill 为 9 个、hit 为 6 个。命中只是替换 scale reduction，仍生成约 72 MiB W8 bank；不能用 32 KiB cache payload 或 cold→hit dispatch 减少量冒充相对 OFF 的节省。没有新增图像质量、硬件 placement／INT8／overlap 或通用性能资格，也不把 W1R／W2、LoRA 等候选收益归给本缓存。

## 后续工具 CPU 改进

已将 dtype／连续布局／extent／alignment／pointer 检查移出逐元素循环，直接展开 BF16 bits 计算原有统计和逐位比较；保留 signed zero 的位区别、非有限值和全零退化拒绝。新的 `comparison_cpu_seconds` 单独报告 pair 后的比较开销，排除在两臂计时之外。旧六次设备调用没有该字段，不补写或倒推其值。

新工具仅完成严格 [host 编译](../../outputs/native-w1-scale-cache-real-20261008/compare-cpu-host.log)、[链接](../../outputs/native-w1-scale-cache-real-20261008/compare-cpu-link.log)及一次 [CPU 合同](../../outputs/native-w1-scale-cache-real-20261008/compare-cpu-contract.log)：17 cases，包括 14 个无效输入拒绝，未创建设备或加载模型。新 binary 位于 `build/native-replay-compare-cpu-20261008`；原实测 binary 与对应源码快照仍保留。生产 `16fecb41…` 构建没有变化；尚未测新工具对 GPU 启动状态、设备耗时或模型吞吐的影响。

## 清理

在两个 controller 物理完成且 ledger 归零后，删除本轮 owned 的 tool object 和 OFF／ON 两个 MIL cache，共 **7 个文件、12,790,918 B（约 12.20 MiB）**，见 [owned cleanup](../../outputs/native-w1-scale-cache-real-20261008/owned-cleanup.json)。测量 binary、library／CLI、manifest、原始回执、时间日志和 CPU log 保留；未删除原模型或系统 compiler／driver cache。

后续纯 CPU 构建另清理一个 197,800 B object，见 [CPU 工具清理](../../outputs/native-w1-scale-cache-real-20261008/compare-cpu-owned-cleanup.json)。本轮合计清理 12,988,718 B（约 12.39 MiB），保留所有测量及 CPU binary／日志／源码快照。
