# KernelWiki与Sana：借组织方式与通用接缝，不照搬后端

<!-- knowledge
id: technique-references-kernelwiki-sana
type: method
-->

已吸收到本目录的实现方案：[技术组合与 schedule](../designs/technique-composition.md)、[GPU 数据流](../designs/gpu-dataflow.md)、[DiT 分阶段方案](../models/dit.md)，配套[本地 policies](../examples/reference/policies.py)与[量化实现](../examples/reference/quantization.py)。下面的源码路径和网页用于出处核查，不是核心学习依赖。

[总目录](../README.md) · [实验室](../examples/README.md) · [技术卡](../templates/technique-card.md)

## KernelWiki的启发

本库借鉴“方法、实现、案例、来源”分层，页面用短 Markdown 元数据直接检索，不维护 JSON catalog 或复算证据链；不声称兼容 KernelWiki 数据库/schema，也没有复制其 skill 或 CUDA 实现。

## Sana sol-engine的本地真实参考

用户指定 [NVlabs/Sana sol-engine](https://github.com/NVlabs/Sana/tree/sol-engine)。本轮同时读本地`../Sana`，HEAD `d0c0a4685ab5dc2336d18b7213d85f13def92418`，trackedworktree干净；不声称与远端分支当前HEAD一致。

| 源文件（相对Sana根目录） | 值得迁移的机制 |
| --- | --- |
| `techniques/technique.py` | capability、phase、reads/writes seams，runtime hooks |
| `techniques/compose.py` | 结构冲突与确定性顺序；exclusiveowner/sameschedule冲突 |
| `techniques/schedule.py` | 参数是(step,stage)函数，不只有一个全局bool |
| `techniques/README.md` | 通用策略与modeladapter接缝分开，build/load transform与runtime technique区分 |
| `site_docs/techniques/kernel/qknorm_rope.md` | fusion作为数据流边界和precision/layout实验 |
| `site_docs/techniques/quant/diffusion_ptq.md` | layer/timestep/bitwidth/granularity与fallback层的搜索空间 |

这里迁移的是设计方法；其CUDA/Blackwell/NVFP4/TransformerEngine路径不是AppleGPU/ANE现成实现。Apple可从packed QMM、FP16/W8A8、CoreML/MPS/Metal边界建立对应实验，不把NVFP4后端名当成支持证据。

## 已经落地的对应实现

[policies.py](../examples/reference/policies.py)是独立的小型effect/capability checker；[caches.py](../examples/reference/caches.py)实现可读residual replay与tokenmass merge；[quantization.py](../examples/reference/quantization.py)实现Q4/Q8物理格式、W8A8与SmoothQuant；[Metal](../examples/metal/kernels.metal)提供真实kernel对照。

这些不是声称完整复现TeaCache/PAB/TaylorSeer/PISA/ToMeSD各论文。它们暴露核心接缝，便于再扩展完整策略、frame/stage布局、补偿项、离线校准和模型质量测试；给定搜索horizon的结构无冲突也不代表所有动态schedule都已证明。

## 面向工程师/AI的使用方式

固定版本的[KernelWiki 阅读笔记](../sources/repos/kernelwiki-retrieval.md)说明检索、schema 和别名的设计启发。当前索引直接来自 Markdown，保留少量别名和按阅读目的排序，不登记实验/验收状态；[使用指南](../guides/knowledge-base.md)解释其边界。

从模型事实生成capability/spec → 选择技术卡 → 结构检查/配置 → 运行smallreference/negativecontrols → 对真实source做性能/质量纵切 → 添加状态与迁移说明。不要让agent只凭某个backend推荐表堆配置，也不要因旧试验失利就剪掉整个研究分支。

固定读本见[来源索引](../sources/README.md)。现有 TurboCider 模型库、source lease、memory ledger 和产品路由仍决定实际行为；知识库不接管它们，也不提供历史实验环境。
