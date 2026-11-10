# Qwen Q4_K_M 与 BF16 streaming 历史实验

本目录集中保存 2026-10-10、M4 Max 64GB / macOS 26.6.2 的七篇接续记录：
混合 GGUF 导入、GPU/Private ANE 分工、encoder 导入与原始 BF16 双阶段
streaming，以及 GPU prefill、源注册与后续同内存窗口。原文中的“当前”“本轮”、默认行为和未完成项均属于各自记录时点，
不能据此判断当前版本的支持范围或性能资格。

## 记录顺序

| 记录 | 已观察内容与限制 |
| --- | --- |
| [混合格式导入与首个完整 GPU 生成](qwen21-q4km-support-2026-10-10.md) | 原文件身份、Q4/Q5/Q6 混合格式、实 Metal 投影与生成；四步烟测、后续四十步 GPU 对照和失败的参考采样分别保留，量化 GPU 当时未胜匹配 BF16 |
| [GPU/ANE 通道分工与 down-kernel 筛选](qwen21-q4km-hybrid-2026-10-10.md) | 四十步正反序、实际预测计数与完整 packed GPU 恢复；各窗口使用各自 GPU 分母，部分 kernel shape 无收益，未证明物理并行或完整产品资格 |
| [有界并行导入与局部 embedding](qwen21-q4km-encoder-import-2026-10-10.md) | 同库导入组件正反序、实际整图不变与冷 encoder 对照；八个 workers 是该设备上的实验选择，更强 Unsloth 冷端反例仍保留 |
| [原始 BF16 双阶段 streaming 与同内存对照](qwen21-bf16-streaming-2026-10-10.md) | encoder 和 DiT 均有界读取原 BF16；同库四路完整四十步、实际进程树峰值与图像检查，保留早期失败和不同驻留生命周期的成本 |
| [GPU encoder/prefill 融合](qwen21-gpu-encoder-prefill-2026-10-10.md) | 区分矩阵组件与 verified tokenizer 复用，保留冷首请求变慢、实际张量/图像差异和原 Unsloth CPU/GPU 持久会话对照；warm encoder 倍率不等于完整请求倍率 |
| [首次 SHA 注册与跨进程复用](qwen21-source-registration-2026-10-10.md) | 本机本用户 HMAC/generation 缓存合同、并发失败修复、首次与已注册新进程分别计时；零 SHA bytes 不等于零模型 I/O，不把本地缓存当第三方签名 |
| [新的同库四路对照与导入 buffer](qwen21-matched-gguf-bf16-2026-10-10.md) | 独立新库/优化 profile 的相近内存比较；保留高 I/O 状态 cold 反例、direct-U4 较慢结果和 1/4/8 MiB 筛选，默认 1 MiB，不把组件收益推广成稳定完整请求收益 |

各记录按接续关系排列，保留各自库、profile 和测量窗口。后续结果不能倒写成
早期窗口已经达到的能力；较晚记录中的高 I/O cold 反例也不能被较早低耗时覆盖。

## 阅读边界

- CPU 导入、encoder、单投影、FFN、DiT 和完整请求分别解释；四步预检不替代
  四十步比较，组件倍率不等于整请求倍率，不同库的最快阶段不能拼成一轮结果。
- Q4_K_M 是混合格式文件配方。Q6_K 重编码、系数舍入、FP16/BF16、原始
  checkpoint 与参考引擎的 RNG/noise 差异沿用原记录，不宣称跨引擎图像逐位一致。
- 同内存比较使用该窗口实际 process-tree peak；managed weight 预算不是整请求
  RAM 硬上限。系统 swap/compression 计数不归因单个模型或 kernel。
- cold/warm、首次 SHA 注册、源验证、source retention 和每请求重新加载的范围
  不混用；首次完整 SHA、后续 generation/认证复验以及取消、所有权和内存准入
  各自遵循原合同，不将注册命中描述为零模型读取。
- 已通过的局部检查不替代更多场景、quiet-load、Public、编辑或 LoRA 的完整验收。
  当前产品能力从[用户文档](../../public/README.md)和[加速决策](../../status/acceleration.md)进入。

## 存储与复现

七篇完整正文只维护在本目录；原 `docs/status/` 页面保留原章节标题和跳转。
机器可读 JSON 仍在原 `docs/design/validation/` 路径，本次不复制或改写。
原文中的命令及普通源码/产物路径仍从仓库根目录解释；`outputs/`、`build/`、
模型与本机回执不保证随 checkout 提供，先核对记录要求的库、源和工作负载。
本次归档没有重跑这些模型实验；新结果遵循[实验归档约定](../README.md)。
