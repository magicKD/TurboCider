# Sana sol-engine：把算法、模型 adapter 与后端能力分开读

[算法入口](README.md) · [现有组合设计](../designs/technique-composition.md) · [来源账本](sources.md)

用户指定项目：[NVlabs/Sana sol-engine](https://github.com/NVlabs/Sana/tree/sol-engine)。本轮深读本地固定 HEAD `d0c0a4685ab5dc2336d18b7213d85f13def92418`，相关 tracked worktree 干净；**不声称它是远端分支当前最新 commit**。更多逐文件信息见 [S01](sources.md#s01)。

## 从真实执行接缝向外读

| 层级 | 入口 | 本次吸收重点 |
| --- | --- | --- |
| effect / capability | `technique.py`, `spec.py`, `compose.py` | 哪个阶段能读/写哪个中间物；互斥写入、排序、模型能力 |
| schedule | `schedule.py` | 参数依赖 step/stage，非一个全局开关 |
| cache controller | `methods/teacache.py`, `step_cache.py`, `payload_cache.py` | whole output 与 residual 的不同边界；状态、重算、缓存身份 |
| token reduction | `methods/token_prune.py` | score/select/gather/restore；ToMe、shape-stable 与 stale compensation 区别 |
| sparse policy | `sparse_attention_policies.py` | route family、block size、density、mask reuse、SAP五步链 |
| sparse kernel | `sparse_backends/sol_attn/interface.py`, `mps/metal.py` | 真正后端分派、shape守卫、Metal summary/routing/softmax |
| 模型接入 / 报告 | `site_docs/techniques/`, model runtime | controller 是否真的跳过目标工作，实际 sparse/cache counters |

文件存在不意味着所有模型都启用了它。文档中的“设计空间候选”，配置开关、adapter 绑定、kernel 成功调用、整模型质量通过，是四个不同证据。

## 模型组合为什么不同

文档将 SANA-Video 的 EasyCache、Cosmos3 的 TeaCache/step-selective low precision、LTX stage-2 的 PISA/token prune/NVFP4 等作为特定 fullopt recipe；Wan/LingBot 又按 base/refiner 或专家/阶段选 cache 与 sparse。这证明优化应按消费者定制，不是给所有模型套同一菜单。

这些 fullopt 数字包含 kernel fusion、compile、cache、低精度、sparse、甚至步选择等多个因素；不作为单项算法倍率。本文不重新发布其跨 GPU 表格，更不迁移成 Apple 预测。

## 通用层本身也有边界

`TeaCache` 的 whole-step controller 可返回旧 step output；`TeaCacheResidual` 在 PRE_BLOCKS 加 residual，模型 adapter 再跳 block loop。不能看到同名就假设与作者 FLUX residual 公式完全一致。

`token_prune.py` 的普通 top-k、ToMeSD-like、CAT-style controller 共用接口，但并不完整复现每篇论文的全部训练/池化/attention 状态。`sparse_attention_policies.py` 的 pure Torch map 也是算法参考，直到 runtime consumer 真正执行，才算 GPU 证据。

## Apple 实施顺序（建议，不是新增生产后端）

1. 提取设备无关 policy 与 schema：request/CFG/stage 身份、连续 step、有限质量预算、full/reuse receipts。
2. 用 MLX 自带 attention/QMM 建立优化 dense 基线，确认 full control 不受 hook 本身破坏。
3. 先做整段 residual reuse 与逐层混合 precision；无需先重写所有 CUDA kernel。
4. 真需要 sparse 时复用算法、重新实现 Metal tile/routing，而非导入 CuTe/Triton CUDA 包；优先核查已有 MPS backend。
5. 每项暴露 actual counters/fallback，结构组合 checker 通过后再做完整媒体质量与性能门槛。

动态 shape、cache lifetime、private ANE 接缝仍由本库相应合同约束；不为研究算法绕过取消/drain/ownership。

相关论文入口：[Sol-Engine 2606.23743](https://arxiv.org/abs/2606.23743)；本轮结论以固定源代码和文档为依据，没有运行其 NVIDIA workflow。
