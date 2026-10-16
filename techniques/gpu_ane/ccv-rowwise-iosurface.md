# ccv / MFA：rowwise INT8、图内 K-split 与 IOSurface 交接

[专题导航](README.md) · [内存边界](memory-and-handoff.md) · [量化](ane-graphs-and-quantization.md)

固定读本：[`liuliu/ccv@66dc16a3a3b688adfa6dc1d19eeeb3071beeb701`](https://github.com/liuliu/ccv/tree/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701)。重点文件 [`lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm`](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm) 的 1–130、1130–1220、1290–1345；函数定位和完整来源在 [来源页](sources-and-coverage.md)。本轮为源码核查，未编译/执行上游。

## 三个执行层不要混淆

Draw Things 是模型与应用层；ccv 的 graph/runtime 管理张量与设备；MFA/NA shader 是 GPU kernel 层，另有 Core ML/ANE bridge。`NA*MatMul` 名字不表示独立 ANE。阅读顺序应是模型 block→ccv op/backend→descriptor/kernel 或 ANE bridge→实际 consumer，而不是凭目录名推断请求放置。

## Rowwise 低比特路线

该 bridge 的 scratch 分 activation INT8 surface、weight INT8 surface、output FP16 surface 和 scale buffer；选择 `CPUAndNeuralEngine`。这证明图和输入表达的设计，不是物理 INT8 MAC 或 ANE-only 驻留的独立证据。

rowwise scale 允许较小的量化载荷；输出要按原 source 语义恢复。若加入 Hadamard/旋转或重新量化，必须核对其 source 数学和误差，不能将 GGUF/MLX affine 权重直接等同于这个 INT8 编码。GPU pack/quantize、scale restore 和 join 都属于完整边界。

## K-split 的具体价值

`select_matmul_k_split(M,K)` 对 M<512 保持 1；较大 M 按不同 K threshold 选择 1/2/4。注释明确是避开测到的 matmul performance cliff，而不是 ANE 矩阵大小的官方上限。

这种 graph 内的 K 分片仍需精确求和：`C = Σ A[:,Ki] B[Ki,:]`。它不是 host 连续调用多个 Core ML 图，也不是 GPU/ANE 切 intermediate F；三个方案的 launch、scratch 和数值归约成本不同。应对同函数的 monolithic、graph-tiled、host-tiled 分别测 compile、warm、完整 consumer。

## IOSurface 与 output backing

输入/权重/输出用 CoreVideo pixel buffer 的 IOSurface 作存储，再映射成 shared Metal buffer。activation、weight、output surface 各有独立有界 cache；读本额度为 512/128/1024 MiB。这些是项目策略而非硬件要求，复制进低 RAM 产品前必须重测。

`ccv_nnc_mfa_ane_rowwise_coreml_evaluate` 构造 multiarray wrappers 和 feature provider，尝试 `setOutputBackings` 后 prediction，并读取返回 multiarray、检查其计数字节数。

**重要的源码边界：已读函数并未通过 dataPointer/实际 surface 身份验证返回输出确实采用给定 backing。** 大小匹配不等于 alias 成立；调用 selector 成功也不保证后续 GPU 读的是模型写入的存储。迁移时应独立验证真实 returned backing；不一致则显式 copy 或拒绝，不用“传入 outputBackings”当零拷贝证明。此观察限定于该函数/commit，不声称审计了整个上层调用链。

## Fast fence 不只是计数器

`...fast_fence_encode_update` 结束当前 encoder，新建 compute encoder，先 touch 跨 Metal→Core ML 的 activation/weight buffer，执行 buffer memory barrier，再更新 shared timestamp lane。源码注释把执行排序与跨域可见性分开。

这是特定实现的低延迟交接技巧，不能推导成“CPU 看见原子 flag 就能安全读任意 GPU 数据”的公共普遍规则。迁移须同时验证 GPU 生产完成、cache visibility、ANE 完成和 consumer 依赖；对版本不支持的路径保留正式 command completion 的保守回退。忙等也有 CPU/功耗成本。

## 值得复现的消融

| 控制变量 | 对照 | 要防止的错觉 |
| --- | --- | --- |
| 图表达 | monolithic / graph K-split | 单图快却因 staging 或 restore 失利 |
| I/O | copy / requested backing / verified actual backing | 把 API 意图当成存储事实 |
| 同步 | formal completion / fast fence | 仅变短 host wait，却读取旧输出 |
| cache | 冷/热、不同额度、淘汰 | 将无限驻留收益用于低 RAM 声明 |
| 精度 | FP16 / 同 source W8 / W8A8 | 换了权重/activation recipe 后称无损实现提速 |

每个输出 slot 需要 owner、shape、stride、generation 与 last-reader completion；失败后 drain 再复用。可运行的本库共享边界与错误 stride/stale 负控见 [实验页](experiments.md)，其结果不直接授予 ccv 上游 qualification。
