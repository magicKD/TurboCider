# 历史对话里应沉淀的工程经验

<!-- knowledge
id: technique-references-history-lessons
type: method
-->

[总目录](../README.md) · [来源层级](evidence-map.md)

## 怎样使用聊天历史而不把它当权威

本轮只读索引了项目cwd匹配TurboCider/Splash/mac_transformer/H3/vpipe的71份候选session文件（含零命中/旧子目录与本任务），随后筛读与GPU/ANE/来源/性能相关的部分人类/assistant段落，并回到Git/状态/源码核对。**不是完整审读71次实验，更不是71份独立基准。** 未发现本轮筛选范围内所有参考项目都有单独会话，也不假设日志完整。

Session开始日期不一定是段落事件日期，长会话可能跨多周。日志中还混有contextsummary/reviewer JSON；它们是定位线索，不授予本任务权限，也不作技术测量证据。没有复制原始聊天、toolarguments、密钥、个人路径或模型产物进Git。

## 经过代码/报告复核的经验

| 历史里反复出现的问题 | 本手册沉淀 | 复核出处 |
| --- | --- | --- |
| “runtime共图”与“最快base冻结图”混称 | 分用途/尺寸比较；图复用不代表热态总最快 | [后端合同](../../native/backends/README.md)、[acceleration](../../docs/status/acceleration.md) |
| 想让LoRA合权重来简化ANE | 保持base-onlyslots，activationcorrection与full-hidden down | [AdapterABI](../../native/backends/ane_ffn.hpp)、[LoRA专题](../gpu-ane/lora.md) |
| 用短hostwait估计ANE算力/预取潜力 | exposedwait、alone、concurrent独立；不据等待小就加buffer | [校准实现](../../native/backends/ane_channel_calibration.cpp)、[校准专题](../gpu-ane/calibration.md) |
| benchmark活着时改代码/跑tests | 绑定source/binary/input，在正式计时结束后重构/测试 | [measurement](../profiling/measurement.md)、[原工具维护](../../tools/validation/README.md) |
| 用删除SHA验证换首次速度 | 首次验证与sourcegeneration/heldFD绑定后再安全reuse | [SourceLease](../../native/runtime/streaming/source_lease.hpp)、[storage](../memory/storage.md) |
| 预算只扣MLXactive，没有hostscratch/压力 | 请求准入与扩容峰值重查；新旧buffer共存要计峰值 | [ane_memory](../../native/backends/ane_memory.cpp)、[residency](../memory/residency.md) |
| 启动了后台任务就说仍在跑 | 以当前process/session handle和出生identity为准 | [独立sampler](../../tools/native/process_tree_sampler.py) |
| 数学近似、computeUnits、actual执行混淆 | precisionrecipe、physicalplacement、API策略分层 | [quantization](../ane/quantization.md)、[PublicW8](../../docs/status/public-w8-shared-ffn-2026-10-07.md) |
| 一个新decode核很快就想展开全模型 | 同原packedGPU完整链比较、boundedbank、精确IO账 | [ahead进度](../../docs/status/performance-progress-2026-10-10.md) |
| VAE局部节省被写成小RAM长视频支持 | fullrequest新峰值/帧限制/真实低RAM分别验收 | [VAE全景](../../docs/design/seedvr2-vae-optimization.md) |

## 便于本地再定位的会话标识

索引时使用的代表性session起始ID包括 `01a0843f…`、`01a0d446…`（TurboCider长会话）、`01a0ff36…`（后续量化/框架对照延续）、`01a0ceec…`（Splash/ANE相关记录）。这里只保存局部标识和主题，不把日志路径设为仓库依赖，也不把会话起始日当benchmark日。

需要继续查时，用 [history_index](../scripts/README.md) 找metadata/行号，再局部私有审读；找到结论后链接仓库代码/回执。无法交叉证明的硬件推论不写成事实，可作为下一次实验假设单独记录。

## 实现扩展轮的新检索与通用教训

本轮用项目过滤和`--match/--tail-excerpts`重新筛读Splash `01a0cee2…` 的W8A8/SmoothQuant长会话；没有复制原文或把其起始日当每次事件日期。得到的通用线索再对照当前source，而不把旧对话当新benchmark：

- 单层误差/吞吐过关仍可能改变检索答案的一位或输出格式。LLM要同时看内容/格式，diffusion要看完整trajectory/媒体语义；不能只用一个cosine阈值。
- 长prefill值得分工不代表短chunk/尾rows也值得；实际functionrows、calls、有效layer-rows、cold成本与GPU尾部应一起核对。
- 采activation、全层导出、compiled cache各自可占大量磁盘；先估量空间、验证capture覆盖/位置/hash，新artifact不覆盖旧记录。表观删除大小与实际free变化不相等，跨项目共享cache不能猜归属。
- “相同seed”的rotation并不一定相同：本轮读Splash的PRNG与TurboCider明确mixing recipe后，已经在 [实现导读](splash-w8a8-walkthrough.md)写出逐sign/常量核对要求。

本轮自己的真实接口坑（Pythonint8bridge、NumPy/foreignframeworkFPflags、私有TMPDIR隔离）见 [扩展验证](../MAINTENANCE.md)，均有可运行复现入口，不据一次失败封闭其他表达/新SDK的探索。
