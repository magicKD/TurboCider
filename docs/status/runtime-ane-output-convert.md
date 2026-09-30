# 输出恢复转换候选：暂不纳入运行路径

## 收尾决策

候选只完成 host 数值与微基准检查，没有完成候选库的整模型 before/after。
本轮按代码整理要求撤出整数转换分支，恢复已验证的浮点 SIMD 路径；不是
“已证明无收益”，而是**端到端收益尚未验证**。不新增 optional 开关。
保留独立有用的穷举回归，当时产品库仍为 [readiness 优化](runtime-ane-lora-ready.md)
的 `4d864b4439adb23cd53aa90dfe2884a3ced46027a4d24b4ac3e5ba2126ee6eac`。

## 已完成的实验

目标是 LoRA 路径的 FP16 hidden → BF16 恢复转换，不改 base/LoRA 权重、
SiLU 顺序、调度或溢出检查。对于正的 2 的整数次幂 headroom（1…4096），
尝试用整数指数调整与 round-to-nearest-even 替代浮点转换。

- 第一版遇到 subnormal 的 SIMD 组回退浮点路径。在 10% subnormal 的
  合成数据中约 2.30–2.36 ms，原版约 0.96 ms，已淘汰。
- 第二版用 CLZ 无分支规范化 subnormal。320×12288 元素、scale=1/16、
  subnormal 比例 0/1/10%、两次预热及 20 次测量，ABBA 中稳定区间
  约 0.793–0.810 ms，原版约 0.960 ms，各 checksum 一致。
  候选启动样本可达约 1.70 ms，存在明显预热/频率影响；这些时间不能
  当作模型加速比，也不足以证明整请求收益。

微基准记录保留在被忽略的
`outputs/runtime-ane/output-convert-{host,normalized-host}-abba.jsonl`；
一次性驱动和探针不接入产品构建或默认测试。

先前已启动的**原版**整模型 benchmark 正常结束，两份 summary 均 complete。
M4 Max 64 GB、512²、resident、固定 chunks=1、profile 关闭，每模型两次
trial，各一冷两热；整请求含 VAE/PNG，四个热请求的中位数如下：

| 工作负载 | 原版热请求（s） | 中位数 |
| --- | --- | ---: |
| Qwen Viggle v0.2.1 r256，6 步，三张 ref512，c288/K=N1024 v2 | 15.516411 / 15.509858 / 15.513039 / 15.570392 | 15.514725 s |
| Z distill patch LoRA，8 步，c352/K1024/N512 v2 | 7.915585 / 7.795921 / 7.840114 / 7.772844 | 7.818017 s |

每个 Qwen trial 累计 runtime predictions 576，Z 为 771；均无 runtime
failure。这只是原版补充记录，没有对应 after，不更新 fastest 推荐或
对 GPU 的加速比。工作负载、身份、请求及系统快照见：

- `outputs/runtime-ane/qwen-edit3-output-convert-before-fixed1/summary.json`
- `outputs/runtime-ane/z-output-convert-before-fixed1/summary.json`

## 保留的代码与重新探索条件

`tests/native/ane_runtime_convert_test.cpp` 保留全部 65,536 个 FP16 位模式
乘以 13 个支持的 headroom、SIMD 八路加 scalar tail 的穷举测试：有限值
逐位相同、正负零、subnormal、NaN/Inf 分类及两种有限性状态均校验；
另测试异常 lane 不影响邻居。撤回候选后，4 项 host 测试与加速契约通过。

未来重新探索必须在无竞争推理时完成同工作负载的真实 before/after，
同时检查整请求和 output-handling 窗口，再验证 base/adapter 切换与回退。
不能因为局部快约 0.16 ms 就自动进入保留路线，也不能弱化检查换取提速。
