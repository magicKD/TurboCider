# ANE 图搜索：缓存身份、严格门槛与失败缩减

[总目录](../README.md) · [ANEForge 源码卡](../sources/repos/aneforge-rewrite-search.md) · [本机全部记录](../review/aneforge-20261016.md) · [本地合同代码](../labs/search_contracts.py)

适用：研究同一算子的不同图表达、常量精度或合法切分；不把自动搜索直接接入产品路由。语义：重写通常只是实数代数等价，有限精度、函数近似与编译 lowering 仍须独立验证。

## 1. 搜索提示、编译资产和数值资格分开

| 身份 | 至少绑定什么 | 命中后能做什么 |
| --- | --- | --- |
| 结构候选提示 | 拓扑、shape、dtype、stride、mask/segments | 提高某候选的试验优先级 |
| 编译资产 | 图/MIL/实现、常量内容、backend、OS/SDK/工具链 | 在其有效范围内复用编译产物 |
| 数值资格 | checkpoint/LoRA、输入集、变换/精度配方、独立 oracle 与门槛 | 仅复用已经覆盖的验证范围 |
| 性能资格 | 设备/软件、完整 workload、驻留/并发、边界与独立确认 | 只在匹配消费者和环境中采用 |

本地 `identities` 是原创应用合同示范，不模拟上游 hash。实际调用上游后发现：相同结构/不同 W 的图 key 相同，但 MIL+weights 编译 key 不同；cache hit 直接 build，不重新 measure。两件事并不矛盾。即便新权重得到正确的新 binary，旧 winner 的精度/速度也没有自动重验。

```sh
python3 -B techniques/labs/run.py search-qualification
```

此 CPU case 不导入 ANEForge、NumPy、MLX，不联网或执行设备。

## 2. 接受逻辑应 fail closed，而不是“总能返回一个 model”

冻结原图与[独立完整 FP64 oracle](../labs/ane/precision_oracle.py)，先确认 baseline 的数学误差，再评价候选。所有 first/warmup/timed 输出都须 finite、ABI 一致并过固定门槛；失败 arm 不当作任何 speedup 的分子或分母。

上游 `tune_precision(target_error=...)` 的 mock 负控中，所有候选 error 都是0.2、预算0.001，仍返回 model，并在 reason 写明没有满足预算。调用方必须解析与复核数值结果，不能仅判断 API 没抛异常。没有独立 reference 时也不能偷偷沿用“FP32 通过”的标签。

本地 `paired_result` 要求完整 first、2轮warmup、6轮测量，固定 host FP16→新 host FP32 边界、同 pair 输入 SHA、typed 正时间；data verifier 另查三 arm 的六种平衡顺序。bootstrap 只是6对探索性样本的不确定性，不授予请求或芯片资格。

## 3. 失败缩减要保持 oracle，而不是降低门槛

本轮实际缩减链：完整 SwiGLU → projection+SiLU → 单个 SiLU 程序 → 常量点。完整图与单算子都出现约4.2–4.4%的 relative L2；单独原生 matmul 数值通过。`SiLU(0)` 的观察值约为−0.001704，而数学值为0；相同私有 runtime 下改写为 `x / (1 + exp(-x))` 后该点为0，完整 FFN 误差也降到约0.43%。

这把可复现范围缩到**当前系统/入口的单算子程序**，不定位 Apple 编译器内部阶段、物理 kernel 或硬件缺陷。不能据此声称所有 ANE/Core ML SiLU 都错误。

exp 形式只在本轮输入分布和 shape 通过；较大负值可能令 FP16 exp 溢出。迁移到真实激活、不同精度、量化、LoRA 或其他层前必须重验范围/极值，不能把这条局部变换列为全局无损替代。reduce-sum→ones-matmul 在本轮 cancellation 图两端均 exact，并未证实普遍提高精度或稳定提速。

后续[真实Qwen分担](../gpu-ane/private-ffn-boundary.md)使用`exp(-abs(g))`和select的稳定形式，并保留所有full/prefix数值门槛；[本机确认](../review/qwen-private-ffn-20261016.md)只给指定warm完整FFN消费者的观察，不将它改挂为上文原toy图已获得全局domain或产品资格。

## 4. 研究 probe 的最小能力地图

| 实际图/控制 | 当前观察 | 仍未证明 |
| --- | --- | --- |
| frozen matmul、同图 identity | 同步执行、FP64 gate 与活 view/copy/poison 控制通过 | request、GPU→ANE 零拷贝或物理放置 |
| INT8 weight matmul | 局部数值通过；相对 FP16 无稳定优势 | 物理 INT8 MAC、完整 INT8 FFN 资格 |
| 原生 SiLU/FFN | 本轮门槛拒绝，不给性能倍率 | 所有芯片/API 的普遍限制 |
| exp-SiLU FFN | 两 seed、小型完整 host-fed FFN 局部通过 | 真实 Qwen 激活、handoff、LoRA 与全请求质量 |

不要把 CPU-origin boundary 的优势借给 GPU-resident 模型。prepared GPU 诊断不包含 host readback；不能与完整 Model call 混成同一倍率。compile/load、首次、热调用分别记；没有清系统 cache，因此不称真正 cold start。

## 实现参考与实验

核心是[CPU 身份与严格 gate](../labs/search_contracts.py)、[独立 oracle](../labs/ane/precision_oracle.py)、[离线反例测试](../tests/test_aneforge_research.py)、[只读 data verifier](../scripts/curate_aneforge_campaign.py)。真实 upstream 执行是明确可选的[integration](../integrations/README.md)，不属于核心依赖，也不由查询/CPU all 自动触发。

重开条件：新 OS/SDK/ANEForge ref、编译器/权重内容、输入范围/shape/布局、mask、LoRA、actual backing 或消费者边界改变。失败/取消先停止该候选，保留 receipt；不能为继续搜索关闭 breaker、安全保护或清理系统缓存。
