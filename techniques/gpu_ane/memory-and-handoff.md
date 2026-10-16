# 共享内存与消费者 ABI：把“零拷贝”拆成可验证的事实

[专题导航](README.md) · [ccv 源码](ccv-rowwise-iosurface.md) · [实验记录](experiments.md)

官方入口：[Metal shared resource](https://developer.apple.com/documentation/metal/mtlstoragemode/shared)、[IOSurface](https://developer.apple.com/documentation/iosurface)、[MLPredictionOptions](https://developer.apple.com/documentation/coreml/mlpredictionoptions)。本库既有 [公共共享内存取证](../sources/official/shared-memory.md)、[实际 request surface](../gpu-ane/shared-surface.md) 提供更详细版本/源文件记录。

## 一条完整交接链

```text
GPU 原始输入
  → GPU pack/cast/quantize 到实际 ANE input backing
  → producer 完成 + 跨框架可见
  → ANE graph 完成并写 actual output backing
  → GPU view/restore + complement partial
  → runtime join → 完整输出 ready
```

统一物理 DRAM 只消除某类设备间独立显存复制的必要性，不能自动消除 pack、padding、layout conversion、框架内部 staging、cache flush、ownership 和 completion。每个箭头分别计时，外部 copy=0 不等于 runtime 内部 copy=0。

## Tensor ABI 至少包含什么

| 类别 | 必须绑定的内容 | 典型错误 |
| --- | --- | --- |
| 数学 | checkpoint、LoRA、bias/residual、precision recipe | 两支计算了不同模型 |
| 几何 | shape、dtype、element stride、byte pitch、offset、padding | 逻辑 shape 对，物理读错仍 finite |
| 分片 | row/F 区间、gate/up 配对、packed group | gate/up 对不齐、down 漏片 |
| 存储 | actual surface/buffer、owner、allocation generation | requested backing 未采用、地址复用 |
| 时序 | producer ready、ANE ready、last-reader done | 未提交/未完成、旧输出、提前释放 |

例如 `[1,C,1,S]` 的 channel-first surface 与 GPU `[S,C]` row-major 不因元素数量相同就兼容；CCV 读本、Core ML 图与真实 private matmul port 还可能不同。本库 Qwen 的实际端口为 `[1,1,1024,4096]`、row pitch 8192 bytes，不能套用之前 toy 的 BC1S stride。

## 如何证明 backing 被真正采用

请求 wrapper、返回 multiarray、底层 surface、Metal mapping 与 GPU view 分层核对。若 returned pointer/surface 与期望不一致，按实际 stride copy 到 owned buffer，或拒绝当前优化。只核对 shape/count、调用 `setOutputBackings` 成功或 callback 返回，都不充分。

直接映射也要满足 API 的 page/alignment/length/resource 合同；owner 要覆盖 GPU/ANE 最后一次访问。noop deallocator 只表示借用，不提供外部对象的 lifetime。tracked/untracked Metal resource 的 hazard 规则不同，不能为了去等待盲目改为 untracked。

## MLX 的特殊接缝

先确认 actual external-memory adoption，失败回退是否发生 copy；view 是否共享 owner，非连续 stride 是否触发 contiguous temporary。见 [MLX 外部存储导读](../sources/repos/mlx-external-memory.md) 和 [实测](../review/mlx-adoption-20261016.md)。

输出 buffer 内容会改变时，把每次输入 view 作为 compiled join 的**运行时参数**，而不是只在闭包里捕获一个已 eval 的 array。same-dtype copy 在某路径可 alias；“我调用过 copy”不证明 snapshot。generation、poison、先前 point 的真实输出是很有效的 stale 负控。

## 所有权与失败

单 slot 串行复用，或者用有限 ring/bank 明确每个 slot 的 last-reader completion。取消不能立即 free/recycle；先停止后续提交、等本请求已提交工作完成，再丢弃输出并释放 owned resources。只释放自己创建的对象，不清全局系统缓存或重启服务。

普通 command completion 是保守边界；fast fence/atomic flag 只有在具体 runtime 的执行顺序与可见性都经验证时才可替代一部分等待。CPU 观察窗口相交不是物理设备并行证明。

## 负控与测试

```sh
python3 -B techniques/labs/run.py consumer-strides
python3 -B techniques/labs/run.py resource-hazards
python3 -B techniques/labs/run.py segmented-ffn
```

以上是 CPU 合同演示。设备实验需额外故意错 stride/padding、改 generation、拿前一输入输出、alias 写冲突，并核对完整输出而不是几个抽样点。本库 [shared FFN](../review/qwen-shared-ffn-20261016.md) 的 exact runtime join 与 [shared surface](../review/private-shared-20261016.md) 的 finite-but-wrong 负控是独立证据，不应拼成所有引擎都具备的性质。
