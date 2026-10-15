# Apple Silicon 源码研究与实测：工作台

[总目录](../README.md) · [既有自包含验收](self-contained-validation.md) · [阶段地图](../models/stage-map.md)

状态：进行中。基线为TurboCider `7e96f51`；已完整读取用户提供的`TurboCider_Apple_Silicon_Wiki_Learning_Roadmap.md`（1350行）。该roadmap是研究计划，不是性能证据。此工作台保留全目标与尚未完成的工作，不以文档数量或微基准通过冒充研究完成。

## 要完成的工作流

1. 知识工程：稳定ID、来源/综合机制/实验分离；受控词表与别名；只读查询、来源展开；candidate/version/revisit账本；人和Agent共用入口。
2. 源码研究：先贯通KernelWiki/KDA、MLX/GGML/MFA/ccv、CoreML/ANE图表达、vpipe/现有hybrid与profiling的纵向链；再扩展图搜索、私有能力/PMU边界及内存媒体。
3. 实机实验：相同shape/source/precision与受保护oracle；GPU、ANE、串行、并发完整consumer边界；搜索和独立确认分开，负结果保留。
4. 模型应用：以可用的Qwen Image 2.1真实shape/activation/request为纵切；候选与默认隔离，数值/质量、时间、内存与实际fallback一起判断。
5. 交付：将可复用know-how回写既有techniques分类、生成查询索引与实验任务；保留来源许可、版本、完整样本与适用范围。

## 当前事实与执行边界

- 工作树开始时干净，未改产品默认或已安装App；现有模型、工具环境与原始报告均保留。
- 本机是M4 Max / 64GB；当前选择的developer directory是CLT，尚未发现完整Xcode。因此Xcode/Instruments能力须实际探测，不伪造trace/counter结果。
- 实验按一个SoC实验槽串行执行；下载/源码阅读不当作设备实测。查询工具不自动下载、安装、运行benchmark或修改安全设置。
- 新参考checkout仅在用户指定的references目录新建；已有参考项目只读，HEAD和dirty文件SHA分别记录。
- 编译/性能/数值资格key分开；API策略、计划估计、实际wall、派生指标与物理设备归因分开。

## 进度

| 项目 | 当前状态 |
| --- | --- |
| roadmap与当前仓库核查 | 已读，基线已确认 |
| 来源项目固定版本与关键源码阅读 | 进行中 |
| 来源卡/词表/查询/候选/版本索引 | 基于现有catalog的关键词查询/本地来源展开已实现；44项候选清单保留。细分元数据、别名、独立来源卡与版本索引仍待实现 |
| GPU量化FFN与attention候选 | 待源码定位、真实shape基线和独立确认 |
| 公共ANE图变体/tiling/handoff | 待匹配数值与完整consumer实测 |
| profiling工具代码审核与实际探针 | 进行中；完整Xcode缺失边界已发现 |
| Qwen Image 2.1实际应用与对比 | 待基线、代表激活与候选筛选 |
| 媒体/内存/算法组合扩展与维护检查 | 待分层推进 |

后续记录必须将upstream/source-reported与本机measured分开。没有得到通过证据的项目保持未完成/待确认，不自动晋升默认或硬件资格。

## 2026-10-15 代码整理验收

本次仅整理已有研究源码、补齐可用的catalog检索与忽略规则，不继续性能研究、不修改产品默认，也不把未完成的多维知识库或MFA实验标为通过。

- `python3 -B techniques/scripts/check_standalone.py --isolate`：128项工具/CPU单元测试、40个CPU示例、80页链接检查、45个章节映射与自包含依赖检查全部通过；隔离副本不包含产品或兄弟仓库。
- `python3 -B -m unittest discover -s tests/repository`：20项仓库测试通过，包含新增的忽略规则正/反例。
- `swiftc -frontend -parse techniques/integrations/mfa_gemm_bench.swift`：语法解析通过，不代表上游package构建、GPU数值或性能验证。
- `git diff --check`通过；新增规则没有屏蔽已跟踪文件。模型、第三方源码、原始测量与机器本地状态仍不纳入提交。
