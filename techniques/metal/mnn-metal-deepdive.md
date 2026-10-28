# MNN Metal 深读：卷积选路、低比特融合和 encode replay

<!-- knowledge
id: technique-metal-mnn-metal-deepdive
type: implementation
stage: decode, vae, ffn
operator: conv, gemv, quantization, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, memory-bound, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [第二轮来源与范围](sources.md#deepdives-sources-20261016) · [持续研究](../MAINTENANCE.md)

2026-10-16，固定 `alibaba/MNN@dbf9ba61b08106a3b43426e295d1a29cb41cc8fe`。本页记录第二轮源码深读：沿卷积、量化 1×1、tuning、replay 与 attention 的动态更新继续读；当时没有构建/运行完整 MNN 模型。后续[第九轮真实引擎实验](case-studies/mnn-engine-convolution-and-resize.md)已运行合成卷积消费者，并补充热resize、调优时钟单位、实际architecture及CPU数值失败边界；仍未验证完整导出模型或本页量化LLM融合。

## 1. 算法选择在 kernel 之前

[MetalConvolution.mm](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MetalConvolution.mm)的 creator 先检查动态权重、group 和量化表达等边界，再依次考虑 Winograd、1×1 与普通卷积。普通卷积还按 stride/dilation/padding 划分空间/输出 channel 打包核，不是所有 Conv 都走同一个 GEMM。

`onResize` 准备常量、pack/线程组和 pipeline；`onEncode` 绑定实际 tensor 的 buffer/offset 后提交。C4 packing 使 channel 数以 `ceil(C/4)` 进入物理布局。逻辑 NCHW shape 正确，不代表 buffer 可以被另一个后端按同一顺序解释。

Winograd 的 `isValid` 限制 square 3×3/5×5、stride/dilation=1，并受 runtime 内存等级约束；变换会增加权重/临时工作集。该读本的门槛计算用变换后**元素数**，不能把注释中的 M 直接解释为实测 MiB。它是数学分解+布局+精度的联合方案，应用到 VAE 还需验证 fp16 范围、误差、tile 与完整消费者。

## 2. 调参是有成本、有 cache 的程序

`MNNMetalContext.mm::getGridAndThreadgroup` 的入口 key 包含 kernel 名和 GWS；先查 exact，再尝试 close-shape 结果，否则以启发式为起点，按 TuneLevel 扫线程组。测量是在一个 command buffer 中重复 dispatch，Heavy 还调整循环数。它测的是该重复、驻留工作负载，不是完整模型时延。

值得迁移的不是某组线程维度，而是：合法空间先有界、先查命中、实际 buffer/offset 参与调参、保存结果、冷编译/tune 与 steady state 分账。近似 shape 命中仍需完整数值和性能确认；不能把 cache 命中等同于新 shape 获得资格。跨 OS/SDK/GPU/dtype/pipeline source 的 cache identity 也需要在更上层核对。

## 3. 量化 decode：算“有多少块在飞”，不只算 load 次数

[MetalConvolution1x1.mm](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MetalConvolution1x1.mm)前 300 行提供清楚的 lane/quant-block 设计案例：`L` 个 lane 合作解一块，约 `32/L` 块可同时在飞。改变 L 同时改变每块 load/reduction、每轮 metadata、迭代次数与延迟隐藏；单看每 lane 的 uint4 load 数会出现多个候选打平。

读本将可推导的 load 模型与实测 tie-break 分开，并区分 kernel family、split-K 与 tensor-capable device 的校准条件。它明确指出一个简单“每轮惩罚”无法解释所有 tie 反转。这种设计比伪装成精确硬件公式更可复用：成本模型用于缩小候选，测量用于打破同成本候选之间的平局。

注意：一个普通 GEMV 的 lane 规则受 `tensorApiDevice` 标记影响，不证明该 GEMV 本身执行 TensorOps/NAX 指令。需要继续读生成 shader 和实际 dispatch。

## 4. Gate/up、QKV 融合：保留每个 tensor 的语义

`setupGateUpFusion` 让两条投影共享读入与调度，部分路径还能融合 SiLU×up epilogue。源码专门保留 up 投影自己的 `scaleCoef`；错误复用 gate 的系数，会使“同样 shape、同样 bit 数”变成错误权重。scale/zero/布局/激活顺序是每份参数的合同，不可因融合而合并。

4-SIMD-group、dual-stream、split-K、W16 load 等是不同候选，不是一个总开关。加大并行也可能增加 reduction、TG memory 或寄存器压力。`setupQKVFusion` 还检查 quant layout 与投影组合，不能只凭 Q/K/V 相邻就无条件融合。上述特定阈值/注释收益没有在本机 MNN 重测。

可迁移到 TurboCider 的实验：先验证 fused gate/up 的独立 metadata 和 source identity，再比较读权重复用、launch 数与完整 SwiGLU→down。不得以 gate-only 或 ready FP16 精度岛代替原始 FFN。

## 5. Encode replay 不是 ICB，也不是免编码 GPU graph

`MetalReplay.hpp/.mm` 使用 `NSProxy` 录下 host encoder 调用与参数，之后重新发出 setPipeline/setBuffer/setBytes/dispatch。它跳过算子 `onEncode` 中的分支、pipeline 选择与 ARC 工作，仍在 host 编码 Metal；不能称为 GPU 自行 replay 一整个模型，也不同于 MLX 的图编译。

`MetalExecution::onExecute` 观察连续 I/O 地址指纹后录制；命中时重新验证 current buffer+offset。重复失效会禁用该 op，未建模的 selector 会使录制不适用。profiling 的 per-op encoder 变化不能由录制表示，所以该构建模式明确禁用 replay；诊断模式性能不是正式 replay 性能。

### 最容易误解的 lifetime 条件

录制里保存 raw `Tensor*`，validator 要解引用它；**已释放 Tensor 不是“验证时返回 false”就能安全处理的对象**。内部临时 tensor 必须长期存活或被 generation 守卫；父 fused op 还要对 child tensor 重建负责。仅相同 MTLBuffer 地址不够，pool 可复用同一 backing 的不同子分配。

`MetalAttention.mm::onReplayUpdate` 因此会重新计算 path flags、检查 KV Tensor/scale-buffer 身份与事件结构；先验证，再修改 KV length/参数/grid，避免失败回退后执行两次状态推进。prefill、disk KV、共享 KV 等不满足合同的路径退回正常 encode。

一个仍需实机核查的源码问题：录制器对 resource/heap hint 有专门不录制分支，首个录制 token 的实际资源声明与正常路径是否完全等价，应结合对应 op/资源模式测试。本轮没有复现错误，不据此宣称已发现运行时 bug，也没有修改上游。

## 6. Profiling 的具体落点

`MNN_SESSION_CPU_TRACE` 分开累计 encode、commit、wait 和 wait-site；GPU busy/gap 又是另一类跨度。`MNN_METAL_OP_PROFILE` 的 encoder 边界和 replay exclusion 是工具扰动的明确源码证据。先比未采集完整工作负载，再用这些模式解释 host/queue/kernel；不要把切分后的每层时间加起来当原请求。

本机 [MPS Matrix / MLX 对照](case-studies/mps-vs-mlx-20261016.md)不是 MNN benchmark，不能据此给 MNN 排名。[后续实际MNN小型Conv消费者](case-studies/mnn-engine-convolution-and-resize.md)已补上引擎执行、部分resize/replay证据；量化融合、任意lifetime invalidation和图像/视频质量仍是待运行研究。

本地可执行学习控制：[owner/graph 合同](../examples/reference/contracts.py)、[stride/资源](../examples/reference/heterogeneous.py)、测试；case `owned-command-graph` / `resource-hazards`。这些原创模型验证一般合同，不冒充上游 MNN 实现执行。
