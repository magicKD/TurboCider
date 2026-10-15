# 实现实验扩展：验证范围与边界

[总目录](../README.md) · [实验室](../labs/README.md) · [逐章映射](../labs/catalog.json) · [覆盖审计](coverage.md)

这是上一轮的历史设备记录，源码快照对应`193b356`；后来`bench.py`与`shared_w8.py`已为自包含重构，不能把下列历史SHA/原生FFN夹具结果当作新源码验证。当前结果见[自包含验收](self-contained-validation.md)；原完整生产夹具入口保留在[integrations](../integrations/README.md)。

2026-10-15，基于同一TurboCider `ad8100f`。这是第二轮实现层扩展，和 [首版历史验收](validation.md)分开；不将旧GPU/ANE模型成绩重标为本轮测量。

## 已完成的实测类型

新增标准库数学/状态/模拟、独立Metal kernels/驱动、公共CoreML lifecycle/MLX分工和原生W8A8 sharedIO wrapper。真实设备为本机M4Max、64GiB、macOS26.6.2；输入都是小型合成值，没有加载用户模型。

Metal7类已实际运行：vector、fusion、SG8GEMM、norm/RoPE、onlineattention、commandbatching、双GPUqueue；每类输出全部校验finite/CPUwitness/tailguards。双GPUqueue不代表GPU/ANE physicaloverlap。

Public W8通过现有原生两个fixture实际执行GPUstaging→CoreML→FP32restore，包含3chunks、修正输入/base-A-base、双bank/futureidentity、headroom、latefailure/refill。它不是完整模型LoRA或设备placement/MACtrace证明。

## 最终验收

| 项目 | 本轮实际结果 |
| --- | --- |
| 逐技术子项实现覆盖 | 28个技术章节全部有代码/cases/至少两项controls/revisit_when；catalogcheck无缺项 |
| 轻量机制实验 | 23个CPU/reference/simulation case实际运行，含M/F/H与KV序列轴、带宽字节模型、W8A8/W4A8、SmoothQuant、索引、状态与真实小文件/队列 |
| 离线测试 | 57个testmethods通过、0skip，Python3.9.6和项目3.11.16分别运行；不重复计成114项 |
| 实际Metal | 7类各8hot/4warmup；额外M33/N41/K29三轴尾部复核；全部CPUwitness、finite和16项outputguard通过 |
| PublicCoreML实际预测 | frozen/runtimeFP16/CPU-only各8calls（含warmup），MLXhybrid6calls；相同exp-SiLU/full-K recipe、同一合成weightSHA、独立FP64oracle |
| PublicW8A8真实共享IO | 两个原生fixture通过，privateclassstrings为空、隔离TMPDIR后stderr为空，5个native输入文件前后SHA一致 |
| PythonINT8接口重测 | 最小cast图成功构建；当前Pythonadapter报value-type限制，按定义返回3并输出明确negativecontrol，未当作设备失败/成功预测 |
| 来源/导航 | 当前Markdown/source/snapshot检查无error/warning；数量由check_wiki实时列出，HTTP/anchor不在该检查范围 |
| 工作区 | techniques共89个文本文件；无模型/binary/媒体/rawsessions；原先2个根README导航修改保留，没有新增runtime/default改动 |

可审阅的 [机器摘要](lab-evidence-20261015.json)保存15份原始记录ID/SHA、actualflags、数值、calls和sourcepins；原始大段日志/工具包在ignored outputs，不随Git发布。当前Metal核心与CoreMLrunner/exporter摘要已与文件内容核对，不把旧构建重标新源码。

## 对照观察，而非产品速度承诺

以下各行只在自己的匹配小算子边界比较，每arm4个hot样本；没有全模型/其他机器的外推，也没有strict competing-load/占用率资格。

| 实验 | baseline host median | candidate host median | 如何解释 |
| --- | ---: | ---: | --- |
| SG8GEMM M33/N41/K29 | 0.167667ms | 0.154250ms | 尾部数值maxabs约1.11e-7；下一步应和强GEMM consumer比，而非长期用scalar作分母 |
| 16个依赖dispatch分CB vs batch | 2.395083ms | 0.238125ms | 显示小launch/sync开销，不能当整个模型10倍加速 |
| M33/H128/F512完整MLX vs Python公共hybrid | 0.197395ms | 0.284709ms | 这个小边界hybrid较慢；是探索hoststage/restore/sharedIO/更大shape的依据，不否定ANE方向 |

同窗口frozenFFN另记录convert39.44ms、compile27.29ms、load28.89ms、firstpredict2.18ms，OS/drivercache未清。它说明setup/first/hot要分开，不是各阶段稳定下界或实际ANEkernel时间。nativeW8整套fixture进程wall不作operatorlatency排名。

## 本轮用户要求逐项复核

- 每个技术子项的实现参考：catalog从实际技术目录反查，章节中有直接代码/实验入口；人工检查过halo/upscaler、Public/Private、CPU/设备的语义匹配。
- 更丰富的技术细节：真实MSL/驱动、normalizedW8A8比例因子、W4A8LUT、SmoothQuant holdout、完整LoRA已有oracle、phase/capability组合与IOowner均落到代码。
- KernelWiki/Sana：直接读schema/源码并记录来源、SHA、迁移边界，不声称兼容数据库或已移植CUDA后端。
- Splash/vpipe/历史：补W8segmentation、Measurement/Handoff导读、typedpread/phase实验，重新筛读相关session并交叉核对；未复制私有原文。
- 工程师/AI接新模型：完整SOP→事实表→技术卡→case/负控制→真实source→block/request→组合/交付；实验库不取代已有模型库、sourcelease或memoryledger。
- 维度/争抢：新增H输入在非线性前归约、H输出concat后的norm统计、M-row FFN和KV分片全局softmax实算；错误merge被negativecontrols抓住，UMA模型明确为合成下界而非实测带宽。
- 开放探索：28项都有重开条件，显式区分历史失利和普遍不可能；数值/source/完成等安全合同仍严格。
- 实用与干净：临时构建/图自动回收，工具包隔离，所有真实/合成/模拟/未资格字段准确，未替换App或用户服务。

## 本轮遇到并修订的真实问题

- Python3.9对大integer转FP16可能抛`struct.error`而非`OverflowError`：参考转换显式float并处理两种错误，保持overflow拒绝。
- 沙箱看不到Metal device / CoreML编译临时目录无权限：保留失败事实，获授权后实际执行；不伪装CPU路径通过。
- CoreML9 Python prediction对NumPy int8 features报`value type not convertible`：改用真实公共native IOSurface/MLMultiArray bridge，不浮点cast冒充W8A8。
- 初次NumPy/Accelerate matmul在foreignframework调用后产生FP-status warnings：换独立FP64 einsum oracle并保持finitegate，不简单屏蔽告警；确切根因未独立归因，源码注释中的stale-flags只是可能解释，不将其写成所有环境事实。
- 系统tmp目录过多条目触发nativeartifactlease的安全枚举上限：child使用本任务privateTMPDIR；未删除其他项目cache或放宽安全guard，重跑两个fixture stderr均为空。

## 来源与经验沉淀

新读KernelWiki/schema、Sana capability/phase/schedule/compose、Splash W8 segmentation/Measurement/Calibration/Handoff、vpipe现有内存合同与当前upscaler source；按代码和旧会话交叉定位。没有复制第三方大段实现或原始聊天。

固定ct9.0只下载到本任务ignored overlay，不修改已有ct8.3环境；无模型下载。所有Private相关内容仍只引用源码与研究合同，新增工具没有启用PrivateAPI或替换App/runtime。

Metal教学驱动保留兼容用的`fastMathEnabled=NO`，SDK会给deprecationwarning，原始报告保留它；不因此声称数值失败或悄悄改fastmath。将来改mathMode是另一个可复核源码版本。

## 复现检查命令

```sh
python3 -B -m unittest discover -s techniques/tests -v
python3 -B techniques/labs/check_catalog.py
python3 -B techniques/scripts/check_wiki.py
python3 -B techniques/labs/run.py all --output outputs/new-cpu-labs.json
python3 -B techniques/labs/metal/run.py gemm --m 33 --n 41 --k 29 --output outputs/new-metal-lab.json
.venv/bin/python -B techniques/labs/ane/bench.py --compare-gpu --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/new-hybrid-lab.json
.venv/bin/python -B techniques/labs/ane/shared_w8.py --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/new-public-w8.json
```

硬件项按需显式运行，不能与本任务其他tests/build重叠；不同环境先读依赖/权限/SDK范围。整体没有运行完整make test、新模型/真实adapter/多设备/低RAM媒体矩阵或连续ANEtrace；这些不是文档/教学实现已完成的硬件资格。
