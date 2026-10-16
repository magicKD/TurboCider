# GPU / Metal 调研覆盖与验收记录

[GPU 导航](README.md) · [来源登记](research-sources.json) · [本机实验](experiments-20261016.md)

日期：2026-10-16。按原始需求与 `TurboCider_Apple_Silicon_Wiki_Learning_Roadmap.md` 的 GPU G01–G08、profiling P01/P05/P06/P07 核对，不把整个路线图的 ANE/知识工程待办改写为本次交付要求。

## 需求 → 当前证据

| 要求 | 已交付的证据 | 不扩大的范围 |
| --- | --- | --- |
| 阅读路线图、补充 GPU | 本导航与各技术专题；复用既有 Metal/真实 Qwen 报告 | 没有另造重复 Wiki 或修改生产实现 |
| 充分研究参考项目 | [引擎地图](inference-engines.md)，16 个项目、27 个逐文件 hash/范围 | 关键区段选读，不声称审计每仓全部源码 |
| 研究其他开源引擎 | MNN、Candle、MLC-LLM、MLX-LM、whisper.cpp、ncnn 等 | 未运行的引擎不参加性能排名；ncnn 明确是 Vulkan |
| 解释不同实现/加速方式 | [GEMM/量化](gemm-and-quantization.md)、[attention](attention-implementations.md)、[扩散/媒体](diffusion-and-media.md) | 数学不变优化与稀疏/量化质量变化分账 |
| GPU/Metal profiling 工具 | [profiling 手册](profiling-playbook.md)，链接已有 capture/counter/时钟校准/trace 实测 | 缺完整 Xcode，不声称新跑 Instruments/shader counters |
| M5 以后 GPU feature/TensorOps | [专章](tensorops-and-m5.md)，API 增量、cooperative 输入、量化、NAX guard | 本机 M4 Max，不编造 M5/未来型号实测 |
| 优化 PR | [8 个 PR 阅读账本](optimization-prs.md)，diff SHA、状态与 key code | open/closed-unmerged/merged 分开；作者数字不当本机成绩 |
| 官方文档、博客、可追溯链接 | 来源登记中的 11 个官方/项目文档，Apple ML Research/Tech Talks、社区硬件研究 | 活动文档有日期而非虚构 commit；外链不是离线源码镜像 |
| 简单实验比较、可靠性 | 两进程/seed、28 组、896 输出，完整样本、FP64 oracle、错误打包/NaN 负控 | 组件不是完整模型请求；额外 cache 和首调用单列 |
| 分主题 Markdown 与 README | 本目录的综合报告和[专题 README](README.md)，总百科已加入口 | 按需求最终指定放在 `metal/`；ANE 与协同报告交叉链接 |

路线图 G05 的 metal-benchmarks 另有 [微基准学习页](empirical-hardware.md)：明确历史 M1/M2 测量、次优测法说明与失效的未来推断，不复制为 M5 规格。P02–P04 的独立 ANE profiling 由既有 ANE 专题承接，不扩大为本轮新跑私有工具。

## 验证命令

```sh
.venv/bin/python -B -m unittest discover -s techniques/tests -p test_metal_survey.py -v
python3 -B -m unittest discover -s techniques/tests
python3 -B techniques/scripts/check_wiki.py
python3 -B techniques/scripts/check_knowledge.py --strict
python3 -B techniques/scripts/check_standalone.py --isolate
```

标准库环境中两项可选 NumPy packing 测试会 skip；有 NumPy 的环境才验证其通过。设备 probe 不在普通 unittest 中静默启动，另按[复现说明](experiments-20261016.md)显式运行。

本次共享工作区有其他会话正在新增 ANE、algorithms、`gpu_ane` 草稿。全树检查可能包含其尚未落盘的链接或依赖声明；本次只修 Metal 负责范围，保留其他改动。为分开验证，可将 Git 基线导出到临时目录，只叠加本轮 Metal 文件、probe、测试和总入口，再跑相同离线检查。这个隔离结果只证明本次变更与基线相容，不能宣称整个并行编辑后的工作区已全部通过。

## 最终检查结果

- **隔离范围**：Git 基线 + 本轮 11 篇 Metal Markdown、来源/测量 capsule、probe、测试、catalog/source/query 登记和总入口；不包含其他会话新增草稿。没有为让检查通过修改验证器的覆盖规则；新增设备 selector 仅登记显式 probe。
- **标准库测试**：共 357 项，355 项通过、2 项可选 NumPy packing 测试明确 skip；同一环境再次经 `--isolate` 运行也通过。48 个 CPU 学习 case 通过，64 张技术卡与 64 篇章节对应。
- **NumPy 环境**：本轮新增 6 项 unittest 全部通过，包括错误 nibble/group、NaN/shape 与 ABBA 负控。28 组真实 MLX 组件比较另外显式执行，不由 unittest 冒充设备执行。
- **导航/依赖**：隔离副本 134 篇 Markdown 的本地链接检查无错误；standalone/isolation、knowledge strict、query/source 生成索引检查通过。新 `TensorOps` 查询能返回专章和来源登记。
- **外部来源**：重新下载 27 个 immutable commit 文件，27 个 SHA256 一致；重新下载 8 份 PR diff，8 个 SHA256 一致。规范 MPP PDF 与主要文档可达；PyTorch/ExecuTorch 的普通 HTTP 客户端遇到 403，正文通过 web 阅读，不将 403 当缺少实现或硬件不支持。
- **完整工作区**：采样时全库测试有 3 项失败，涉及并行草稿未登记 card、未完成链接和未声明 import；完整输出已据实检查，本次 Metal 条目在 catalog/链接检查中无遗留错误。不把隔离通过宣传为整个并行工作区全部通过。

没有未运行的 Instruments/M5/跨引擎完整模型测试被计为通过。上述覆盖是本次文档/学习代码验收，不是产品加速资格。来源与运行的机器可读数据分别见 [source registry](research-sources.json)、[实验 capsule](experiment-evidence-20261016.json)，验证摘要见 [validation](research-validation-20261016.json)。

## 有意保留的研究边界

没有“全网所有项目都已遍历”的不可验证承诺，也没有全局最优引擎结论。源码可迁移原理、已读范围、作者测量、本机组件观测和产品资格分开。M5 对照、更多完整模型/LoRA/视频、跨引擎质量与能耗是明确后续实验，不冒充本轮结果；用户允许的简单验证已用有界组件完成。

本轮未下载模型、未清缓存/模型、未改变系统策略、未修改生产默认、未向上游写入/提交。磁盘约 45 GiB 可用，不需要采取破坏性清理。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/search_contracts.py](../labs/search_contracts.py)、[labs/metal/engine_probe.py](../labs/metal/engine_probe.py)；case：`search-qualification`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
