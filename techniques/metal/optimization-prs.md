# 优化 PR 阅读账本：状态、patch、收益条件与反例

[GPU 导航](README.md) · [diff 哈希清单](research-sources.json) · [TensorOps](tensorops-and-m5.md)

核查时点：2026-10-16。本轮读了下列 8 个 PR 的 patch/关键区段，并核对页面或 GitHub search 状态。GitHub REST pull 详情接口本轮返回 rate-limit 403，改用公开 PR 页面与 `.diff`；不虚构 API 取得的 merge SHA。diff SHA256 固定本次阅读内容，PR 活动分支可以继续修改。

`closed` 不等于 `merged`；`open` patch 不等于主线、wheel 或产品已有。性能数字若来自作者，只能称“作者报告”；本轮未在 M5 复现这些 patch。

| PR | 本轮状态 | patch 的实质 | 可迁移学习点 |
| --- | --- | --- | --- |
| [MLX #4682](https://github.com/ml-explore/mlx/pull/4682) | open | transposed affine QMM 的 16-row 调度、分离 fetch/store、双 buffer、split-K | 小/中 M 的并行不足与 packed metadata 预取；物理 K pitch 必须保持 |
| [MLX #4684](https://github.com/ml-explore/mlx/pull/4684) | merged，2026-10-14 | 放开 non-transposed NAX `gather_qmm_rhs` guard | 小改动也需检查 dtype/TF32 与实际 dispatch；不能只读 shader |
| [MLX #4476](https://github.com/ml-explore/mlx/pull/4476) | closed，未合并，2026-10-05 | 提议扩大 causal D256 NAX head-dim-split 的一段 query 范围，增加测试 | 修改 shape dispatch 而非重写 attention；未合并不作为主线已有能力 |
| [MLX #4594](https://github.com/ml-explore/mlx/pull/4594) | merged，2026-09-30 | gated-delta NAX cooperative operand 类型改由 operation helpers 推导 | SDK 类型接口演进；兼容性 patch 不是无条件性能提升 |
| [MLX #4689](https://github.com/ml-explore/mlx/pull/4689) | merged，2026-10-14 | fence wait 周围 encoder 切分，并控制 in-flight 等待 | 同步正确性是优化前提；少 barrier 不应破坏前进性 |
| [llama.cpp #28439](https://github.com/ggml-org/llama.cpp/pull/28439) | open | non-vector Flash Attention 宽 Q tile 与 SKU tuning/测试 | KV 复用收益受 occupancy、尾部 padding、TG memory、SKU 限制 |
| [PyTorch #198564](https://github.com/pytorch/pytorch/pull/198564) | open | 将已有 MPS prefill 核暴露到 flash-attention backend 选择 | API/backend 标签与实际 kernel 分开；新增 LSE/backward 合同需审查 |
| [PyTorch #189019](https://github.com/pytorch/pytorch/pull/189019) | open | MetalGraph capture/replay、buffer pin 与 unsafe-operation guards | 重放不是仅保存 command；地址、RNG、CPU fallback、owner 都有语义 |

## 1. MLX #4682：如何看懂一份“更快”的 packed GEMM patch

先读 `quantized.cpp` 的新候选条件，再读 `QuantizedBlockLoaderStaged` 和 `qmm_t_nax_rows16_impl`。fetch 将 packed words、scale/bias 先放寄存器，store 再解码进 TG；用下一 BK 的读取与当前矩阵计算交叠，并为 W tile 双 buffer。新调度限定 BK/BN、16-bit operand 和地址计算；部分 K 路线把 partial 累加到独立 buffer。

作者提供 M5 设备/shape 的 benchmark，但不能直接变成 TurboCider 的加速率。我们应重放 M1/16/64/256、整除与尾部、不同 group/bit、stride/偏移；同时测临时内存、reduction 和完整 FFN。读取更多字节、寄存器增长、极小 M 或本来就占满 GPU 都可能使候选退化。

## 2. #4684 与 #4476：dispatch 是独立研究对象

#4684 仅改变一个 guard，已合并为 `588553ca95515420386990c8fc36d585a2288d65`；仍需要本机性能资格。#4476 附带 causal D256 短 prefill 窗口测试，但已关闭未合并；还需保留非 causal、FP32/TF32、mask、GQA 和窗口两侧控制。不能把一个区间的门槛迁给所有 GPU、head_dim 或旧 wheel。

本轮特别以页面的 merged/closed 事件再次核查，纠正了仅凭 search API 的 `state: closed` 无法区分两者的问题。#4476 的关闭原因没有足够一手解释，不推测为“收益不足”或“数值失败”。后续相关 PR 也不能在未阅读时当作它的等价合并。

## 3. GGML #28439：把 tail 和设备 SKU 放进选择器

patch 将 query tile 加宽，增加 KV 重用，但只实例化适合的 F16 KV 和 head/tile 配置。host 估算 TG memory、tile 数和尾部额外 padding；tuner 按 SKU 而不只按 family 登记。patch 的 README 还要求重复测接近阈值的 cell，并在足够大 launch 上满足一致收益才生成表。

学习目标是“候选→正确性→有界 tuning table→fallback”的闭环，而非直接抄一个 tile。PR 附带强制候选的 numerical slice 很重要：自动 selector 在某台机器永远回退时，普通测试通过并不能证明新核已被覆盖。

## 4. #4689 与 #189019：正确性会决定性能优化能否存在

MLX fence patch 改变 encoder/等待边界，解决前进性问题。PR 讨论中对底层缓存/硬件解释存在推测；本报告只记录代码变动和修复目标，不把讨论当 Apple 的微架构规范。

PyTorch MetalGraph patch 对 capture 绑定的 buffer 做 pin，避免地址被 allocator 复用；对 CPU fallback 和捕获 RNG 加拒绝路径，防止 replay 静默跳计算或重复同一随机数。它仍是待审候选，不把 `torch.mps.MetalGraph` 写成已发布稳定 API。借鉴到 TurboCider 时同样需要输入变更、取消、stale output、owner 回收和 graph-cache 失效测试。

## 维护规则

重访 PR 时比较 diff hash、head/base、状态、merge 内容和后续 revert/fix；不要只改表中状态。重跑合法输入、负控和完整消费者后才改变采用结论。上游数字、主线存在、本机验证、产品资格分成四个字段；这份阅读账本不向产品下发任何开关。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/search_contracts.py](../labs/search_contracts.py)、[labs/contracts.py](../labs/contracts.py)；case：`search-qualification`、`owned-command-graph`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
