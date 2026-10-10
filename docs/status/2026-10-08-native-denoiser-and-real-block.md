# 原生 Qwen 去噪图与真实 block 回放（2026-10-08）

本轮把独立 Native Metal Graph 从单个 block 扩展到完整 base DiT／Euler 算子图。原生图自主编码输入、条件、时间调制、全部层、输出和采样更新；去噪热路径不调用 MLX GEMM、SDPA、compile 或 eval。权重和 encoder／VAE 仍沿用既有边界。

**当前证据支持小几何正确性和真实 layer 0 组件收益；没有新增完整生图性能资格、ANE 并行资格或 App 发布资格。** 未下载模型、未复制 checkpoint，没有新增 E2E 图片组。

## 实现与资源边界

- `NativeDenoiser` 使用所有层共用的 42 个 PSO、一份 max(prefill, steady) block arena、hidden／latent ping-pong 与每层独立 prefix bank。BF16 输入／源权重，base only；runtime LoRA、GGUF、ConvRot 等未支持语义显式拒绝。
- 首步按既有文本／参考图 sequence 几何和 mask 组装，后续仅计算 target；时间 modulation 保留 prefix 零时间、target 当前时间。Euler 保留三次 BF16 舍入。
- prefix 和 latent 只在整步成功、物理完成且 finite 验收后发布；失败退休，不能重用半成品。新增保留 Ticket 的 `physical_status()`／`drain()`，允许失败后追踪物理完成，不能恢复失败结果。
- `DenoiserMemory` 仍是 requested payload 统计。生产 MemoryExecutionContext／普通 MLX／IOSurface 统一账本、source lease 与 Session delegate 接线未完成。
- Buffer backing 先释放、accounting claim 后释放的实际析构顺序已修正；严格 CPU lifetime 测试通过。静态细节见[资源审查](2026-10-08-native-denoiser-review.md)，实际 API 见[设计第 12 章](../design/qwen-image-gpu-ane-command-graph/12-native-denoiser-runtime.md)。

## 小几何完整图

使用真正的现有 MLX Transformer 与同一生成的 BF16 权重；H128、1 head、D128、FFN256、两层、两步。包括生图、一个参考图、两个参考图；检查源／padding hash、非零 offset、行 pitch、finite、prefix 与完成状态。参考使用实际 BF16 Euler，没有自写近似 Transformer 替代 oracle。

v3 及最终 v4（runtime ID 后缀 `03830b95639c…`）三个用例的最终 latent 全部 0 ULP、RRMS=0、最大绝对误差=0；原门槛 RRMS≤1% 且 max_abs≤0.01 保持。v4 还通过未提交／成功提交后的 physical status、零／正 drain、负时限拒绝和状态不变检查。它不验证真实模型所有层／时间点、设备故障注入或最终图像质量。最后构建的复核及 artifact 身份记入[机器回执](2026-10-08-native-denoiser-and-real-block.json)。

发现并修正两层数值问题：

1. 早期原生 Sigmoid 保留了过多 FP32 中间值，与 MLX BF16 exp／加法／除法舍入不符，tiny base／双参考图达到 max_abs=0.015625 并失败。修正 BF16 primitive 边界后上述 tiny 全通过，未放宽门槛。
2. v3 的 1024 个 SiLU 输入仍有一个 2 ULP 差异，x=-6.84375。8 个输入的逐阶段设备诊断确认，编译后的 MLX SiLU 使用 Fast exp，在此 BF16 midpoint 取 936；原生 Precise exp 取 940。分开物化的 MLX sigmoid 再乘输入也取后一结果，不能替代编译后的 oracle。仅 SiLU 显式对齐 Fast exp，其他算子的 Safe／Precise 策略保留。最终 v4 单算子 1024 个输入全部 0 ULP，未放宽门槛；这仍不是穷举所有 BF16 值的证明。

主机 `std::exp` oracle 不能证明设备 Fast intrinsic 在舍入分界处位级一致；历史 CPU 组件记录保留原始范围，不重新标成 MLX 设备证明。

## 真实 Qwen 组件结果

M4 Pro／48 GiB，既有本地 Qwen Image 2.1 BF16，512²、seed 42、普通 encoder、29 个文本 token、base 20 步 schedule、近似 DiT cache 关闭。捕获同一 Session 原 compiled callable 的真实输入／输出：GPU 完成第一步和 Euler，只为产生第二步真实输入；第二步 layer 0 后停止，无 VAE 解码／图像导出。

每 phase：双方各一次 warmup，随后两次 AB／BA 配对；CPU hash／error 扫描、prepare、权重导入、预分配输出／workspace 在组件计时外。Native wall 包含 encode／submit／物理完成及其中的容器／command 分配，GPU wall 包含原 compiled callable／eval 及其内部 MLX allocation。以下是 **v3 测量构建** 的两样本均值，不能当作最终构建新增跑分或统计稳定性证明。

诊断 observer 在 block 边界显式 `mx::eval` 并固定输入 Data，保证实际输入可重复回放、禁止 input donation。原 callable 的数学保持，但这种分界／owner 固定可能影响 allocation 和跨 block lazy 调度；该基线不是无 hook 的生产整条 GPU 路径。

| Phase | 实际行数 | GPU wall | Native wall | 耗时下降 | 整体 RRMS | max_abs |
|---|---:|---:|---:|---:|---:|---:|
| 首步 prefill | 1053 | 69.530 ms | 65.059 ms | 6.43% | 0.01150% | 1.0 |
| 第二步 steady | 1024 | 65.860 ms | 63.304 ms | 3.88% | 0.00752% | 0.5 |

真实 activation 的最大绝对误差与 tiny latent 门槛不是同一个指标。这里使用用户允许的轻量组件 RRMS≤8% 筛选，并要求 warmup 后被筛查输出的全部元素 finite。v3 只在 warmup 后计算 native error，未逐次校验两个 timed native 输出是否与筛查输出相同；GPU replay 逐次检查 reference SHA。后续 probe 已补计时外的 native 重复 SHA 验证，但本轮没有另跑真实模型补授该资格。不据此证明逐元素、完整轨迹或图像质量。

原生 standalone block prepare 为 228.419 ms；完整模型设计复用 PSO，不逐层 prepare。原 GPU K 为 head-major 但保留完整 sequence 的 head pitch，V 为 head-interleaved layout；两者均需转换成原生 compact prefix 合同。probe 在计时外做每 phase 475,136 B（0.453125 MiB）的强制 BF16 compact copy，并以 logical SHA 确认值不变；分别花 2.919／0.945 ms。稳态原生 block 消费的是实际 GPU prefix 的 compact copy，**不是原生首步 cache**。原 compiled callable 仍使用原 cache；不能隐藏此转换成本，跨框架桥接路线必须计入它。直接 native cache 不需要相同转换，但其完整轨迹收益尚未测量。

成功 probe 总墙钟 9.666 s；swap 使用量前后均为 3305.75 MiB，未增长。完整 GPU step、encoder／DiT／既有 VAE 权重预加载是该有界轨迹的一部分；只计时两处 block，不把整个 probe 时间当作生图耗时。

前两次尝试分别在错误的 M5 layer-staged preflight、原 cache stride 适配处失败。前者未加载模型；后者只完成 prefill native warmup，没有性能资格。修复沿用 M4 普通 `component_staged` admission，保留严格原生 HeadView 合同，以单独小型转换适配 probe，没有设备策略 force override。

## 可追溯性、清理与下一步

原始日志、输入／output SHA、prefix 物理 stride、两组配对时间和 build 身份保存在 `outputs/qwen-native-denoiser-20261008`；机器回执另记录最终 artifact、source、清理清单。v3 真实回放的 library／probe 和 manifest 单独保留，后续 drain／SiLU 修订不继承该构建的性能资格。已清理的对象和本轮构建缓存按所有权列出；不删除模型、既有 dist App 或历史基线。

本轮清理 301 个可重建对象／静态库／临时 CPU／shader 诊断文件，共 30,727,645 B（29.30 MiB），逐项记录 SHA 并确认不存在。保留当前已链接 runtime／probe 和约 9.66 MiB 的 v3 library／probe 测量证据；不清理无所有权记录的系统驱动／framework cache。

接着完成 Session／source／统一内存账本接线，进行一组完整 base GPU-only 对照，再接 native GPU／ANE 通道切分与 slot 生命周期。LoRA、Z-image、编辑 encoder 和 App 配置继续按独立里程碑推进。当前几毫秒的 block 收益不支持预测 Splash 等比例加速；下一轮重点衡量 FFN staging、同步、精度 restore／join 与实际 GPU 补集。
