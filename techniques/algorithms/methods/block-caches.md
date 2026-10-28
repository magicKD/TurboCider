# FBCache / DBCache：以前段变化决定是否执行中后段

<!-- knowledge
id: technique-algorithms-methods-block-caches
type: method
stage: dit
operator: sampling, attention
backend: mlx, metal
device: gpu
symptom: stale-cache, precision-loss
semantics: approximate
representation: bf16
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [缓存生命周期](caches.md) · [TaylorSeer](taylorseer.md)

来源：ParaAttention [S05](../sources.md#s05)、Cache-DiT [S06](../sources.md#s06)、C++ 移植 [S08](../sources.md#s08)、本库 Qwen 历史报告 [S29](../sources.md#s29)。分类：`approximate-train-free`。Cache-DiT 是框架，DBCache 是其中机制，不能当同义词。

## 三段边界比“缓存几层”更重要

令模型 block 栈分为 front F、中段 M、back B。完整路径为 `u=F(h); v=M(u); y=B(v)`。保存 `R_M=v-u`；命中时执行 `u_current=F(h_current)`，用 `v_hat=u_current+R_M_old`，再执行 `B(v_hat)`。命中仍要付 F/B、norm、projection 与缓存读写成本。

FBCache 的典型特例是 F=首块、B=空，其首块 residual 相对差较小便复用 remaining residual。ParaAttention `utils.py::CachedTransformerBlocks` 明确先跑首块再判断，full 时保存从首块后到最终的 residual；图像与文本流可有各自缓存。其比较函数的分母/anchor 方向也应按代码记录，不能只写“相对 L1”便认为各实现一致。

DBCache 将 F/B 数量、warmup、最多总 hit、最多连续 hit、累计差与 SCM 步 mask 变成可组合约束；model forward pattern 决定单流/双流 residual 怎么拆。SCM 的“这个 step 可以缓存”与“当前 residual 已满足动态阈值”是两个门，静态 mask 不是免费质量证明。

## 可迁移到 MLX / MPS / Core ML 的部分

固定 F/M/B 分图最容易解释：GPU 在 device 上计算 front residual、reduce 指标，host 选 full-M 或 add-residual，再接 back。Core ML 可用固定输入输出 shape 的两个图；ANE 在 hit 时不被调用是可能的，但本轮未实测此异构路径，不能声称与 GPU 重叠或快于纯 GPU。

大 hidden cache 在长视频下可能是数百 MiB 甚至 GiB；同时缓存多个 CFG / stage / layer stream 会倍增。采用 request-local ownership、有效 step 标记和 evaluated leaf；不沿用旧 input 的 lazy graph。prefix prefill 与 target-only decode 的 token 集合不一样，shape 相同也不代表 cache 身份相同。

## Apple 已有的具体证据

本库 2026-09-30 Qwen-Image-2.1 报告在 M4 Pro 48 GiB、512²、BF16、F=8/B=0、8 full warmup、末步 full 的配方上给出了完整请求结果：Base 25 步 generate 的 balanced 为 55.47→38.27 秒（1.45×，RGB SSIM 0.96555）；25 步 edit 为 82.89→62.47 秒（1.33×，SSIM 0.99331）。这些是**既有历史样本，不是本轮重跑**。[完整报告（可选产品来源）](../../../docs/status/2026-09-30-qwen21-dit-cache.md)

同报告 repeated-edit 的 warm-prefix 分母下只有 1.16×；不能把 prefix 与 DBCache 倍率相乘。六步 student 未获该策略资格，普通 LoRA 也必须与该 LoRA 自己的 uncached oracle 比较。

2026-10-17新增[两prompt原生完整请求对照](../case-studies/native-cache-request-study.md)：M4 Max/64GiB、20步、相同BF16纯GPU计划和近似permission，18次完整PNG生成、360步profile，conservative/balanced/fast实际hit5/8/9步，完整调用约1.21/1.38–1.39/1.45–1.46×。自行车balanced/fast的RGB SSIM只有0.8172/0.8032，fast alpha最大差49/255；相同日程不保证跨prompt质量。它是新本机证据，不覆盖上面的历史M4 Pro测量、edit/LoRA/student或视频资格；软件计数结合源码loop跳过不等于硬件shader trace。

## 调参顺序与失败对照

先 threshold=0 / 强制 full 控制；再固定 F/B 探索 threshold 与 max-hit；最后才探索 F/B。F 增大通常提高 indicator 成本，并不保证质量单调改善；B 保留修正能力，也使 hit 节省减少。

必须测：最后一步 full、首个 prefix 不成为 target residual、跳 step/重试/取消/换 LoRA 清状态、正负分支隔离、双流文本 residual、cached add 的真实存储与有限值。使用 `executed_blocks` 验证实际跳过，不用开关选中或 estimated saved FLOPs 代替执行证据。

Cache-DiT 设计原文：[DBCache design](https://cache-dit.readthedocs.io/en/latest/user_guide/DBCACHE_DESIGN/)；固定源码入口见 [S06](../sources.md#s06)。
