# 优化 PR 读本：收益、正确性与兼容性分开

[专题导航](README.md) · [GEMM](gpu-gemm-quantization.md) · [Attention](gpu-attention-runtime.md) · [来源清单](sources.json)

状态核查时间：2026-10-16。PR 状态会变化；以下是当日网页观察，不是持续更新服务。patch 下载只用于阅读，没有执行或合并到产品。完整 snapshot SHA-256 记录在来源清单；短 commit 只是阅读定位，不能替代 full source/binary 身份。

## 六项值得精读的改动

| PR / 当日状态 | 机制与已读范围 | 迁移前的验证 |
| --- | --- | --- |
| [MLX #4516](https://github.com/ml-explore/mlx/pull/4516)，已于 2026-10-04 合并，merge `1eb18d7` | PR 描述+patch 开头：先拓宽再求输入和，减少 affine bias contribution 的 half 舍入；分批 QMM decode 复用等改动 | 独立 decoder/FP64 oracle、bias 贡献、不同 bit/group/M、split/tail。它既是性能改进也是数值修正 |
| [MLX #4682](https://github.com/ml-explore/mlx/pull/4682)，Open | 两个 patch 的说明和关键 hunks：mid-M NAX split-K、FP32 partial、16-row simdgroups、packed words/activation prefetch、双 buffer | 特定 N/K/group 对齐、输入 offset、batch fallback、FP32/TF32、寄存器压力与 reduction 成本 |
| [MLX #4596](https://github.com/ml-explore/mlx/pull/4596)，Open | 描述+patch 1–60：generic vector SDPA 以四个 key unroll/prefetch，提高 memory-level parallelism，保留 scalar tail | 实际 generic/GQA dispatch、D/context/tail、online-softmax 数值；不是所有 attention 核都变更 |
| [MLX #4594](https://github.com/ml-explore/mlx/pull/4594)，Open | 描述+patch 1–60：gated-delta NAX cooperative 类型通过 operation helper 推导 | 新 SDK 编译/旧 SDK 回退与数值；这是兼容性修复，没有据此声称加速倍率 |
| [llama.cpp #20962](https://github.com/ggml-org/llama.cpp/pull/20962)，已于 2026-03-25 合并，merge `adbadfaf` | 描述+patch 1–60：分离 tensor matmul2d 与 legacy SIMD-group 核，独立 layout/threadgroup/dispatch | Metal family/SDK、quant decode、非整除输出；保留 legacy 路线。不能只替换 MMA 而保留旧核全部数据流 |
| [llama.cpp #27461](https://github.com/ggml-org/llama.cpp/pull/27461)，已于 2026-09-02 合并，merge `30e53b91` | PR 讨论+patch 开头：Tensor API 需要正确 Metal language、库编译和 shader 路径；讨论后采用分开 tensor kernel library 的解决方向 | embedded/source/external library 各配置、MPP header/compiler 可用性。硬件支持不等于 build artifact 包含功能 |

上述上游 benchmark 均为 **source-reported**。本机安装 MLX 0.32.0，不把它冒充这些 PR 的 parent/head A/B；M4 Max 也不能验证 M5 NAX kernel 的性能。主线源码含某项修复不证明某个发布 wheel 是该 commit 构建。

## #4682 的两个不同实验变量

第一 patch 增加 split-K 路由，让 mid-M、输出 tile 太少的场景有更多 threadgroups；改变了归约顺序和 scratch。第二 patch 改 tile/加载 schedule，在当前矩阵计算前先取下一 K step 的 packed 数据，再解码；它尝试保持**同一个 split 配置下**的累加顺序。

不能把作者“同 split bit-identical”的说明扩大为与原 unsplit 路线逐位相同，也不能把最初 patch 的 split 上限当成最终 patch 的参数。下载的系列包含 `6b2af68…`、`6e7ab39…`；第二说明已把 split 限制改为最多 4、每段至少 256 列。移植时读最终组合后的文件，而非只拼第一版文字。

## 一个优化 PR 的合格学习记录

必须保存 parent/head、当日状态、changed paths、patch hash、原问题和适用条件；然后把“实现了什么”“作者测了什么”“我们测了什么”分开。合并只是代码治理状态，不是本机加速证据。

复现实验固定 GPU、OS/SDK、模型/quant source，构建 parent/head 两份，先过相同 oracle，再交替计时。报告所有合法 shape 与失败，不只选表里最大的倍率。吞吐提升同时记录额外 resident/scratch；新 GPU 优化上线后重新校准 ANE share。

本专题没有导入上游 shader/patch；代码许可与 third-party notices 如将来移植需逐文件核对。当前只有原创机制解释、链接和内容身份。
