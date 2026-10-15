# KernelWiki与Sana：借组织方式与通用接缝，不照搬后端

[总目录](../README.md) · [实验室](../labs/README.md) · [技术卡](../templates/technique-card.md)

## KernelWiki的启发

2026-10-15直接检查 [KernelWiki](https://github.com/mit-han-lab/KernelWiki) 和 [schema](https://github.com/mit-han-lab/KernelWiki/blob/master/references/schema.md)。其组织把可搜索的优化笔记、原始实现出处、适用硬件/shape、验证证据、复现状态和失败/尝试分开；不是只存一个长README。

本库采用同样有用的区分：章节→技术卡→具体代码/case→controls→证据等级→重开条件。`labs/catalog.json`是轻量映射，不声称兼容KernelWiki数据库/schema，也没有复制其skill或CUDA实现。

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

[policies.py](../labs/policies.py)是独立的小型effect/capability checker；[caches.py](../labs/caches.py)实现可读residual replay与tokenmass merge；[quantization.py](../labs/quantization.py)实现Q4/Q8物理格式、W8A8与SmoothQuant；[Metal](../labs/metal/kernels.metal)提供真实kernel对照。

这些不是声称完整复现TeaCache/PAB/TaylorSeer/PISA/ToMeSD各论文。它们暴露核心接缝，便于再扩展完整策略、frame/stage布局、补偿项、离线校准和模型质量测试；给定搜索horizon的结构无冲突也不代表所有动态schedule都已证明。

## 面向工程师/AI的使用方式

从模型事实生成capability/spec → 选择技术卡 → 结构检查/配置 → 运行smallreference/negativecontrols → 对真实source做性能/质量纵切 → 添加状态与迁移说明。不要让agent只凭某个backend推荐表堆配置，也不要因旧试验失利就剪掉整个研究分支。

来源与文件SHA见 [sources.json](sources.json)；现有TurboCider模型库、sourcelease、memoryledger和产品路由仍是唯一生产合同，这个实验室不接管它们。
