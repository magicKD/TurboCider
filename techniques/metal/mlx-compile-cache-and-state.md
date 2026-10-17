# MLX 编译深读：cache key、融合边界、shape 与状态

<!-- knowledge
id: technique-metal-mlx-compile-cache-and-state
type: implementation
stage: pipeline, ffn
operator: norm, layout, scheduling
backend: mlx, metal
device: cpu, gpu
symptom: compile-bound, launch-bound, wrong-layout, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [源码登记](sources.md#compiler-sources-20261016) · [设备实验](../MAINTENANCE.md)

2026-10-16。主线读本 `36969c094354086e15d2a327a9c50feb20b9cf11`；另读 `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` 的旧版 compile cache，对照已安装 MLX 0.32.0 的实测。**相同版本字符串不证明 wheel 从这个 commit/编译选项构建**；源码与 runtime 各自记录身份，不能把主线新修复直接授予旧 wheel。

## 1. compile 不等于单 kernel 或永久 GPU replay

`compile.cpp` 先 trace 构图、简化，再抽取可融合子图，构造 Compiled primitive；其 Metal backend 为这些 primitive 生成/缓存 MSL。GEMM、reduction、SDPA 等可继续使用自身成熟 primitive，而不必全部变成一个巨大的生成核。

所读 `is_fusable` 包括 unary/binary/select/broadcast，并不直接包括通用 reduction/GEMM。递归还受 stream、depth、输入数、global output shape、多 consumer 等条件约束；broadcast 可被特别处理，避免一次不恰当 materialize 扩大内存。源文件中的 depth=11、arrays=24 是该实现限制，不是硬件常数或建议随便增大。

因此 RMS-chain 中 compile 的收益不等于“RMS reduce 与全部后续操作一定融合为一条 kernel”；单 GEMM 包上 compile 也不会自动获得相同收益。要看实际 primitive/tape、source、dispatch 与完整消费者，不能只看 Python 装饰器。

## 2. 一份 cache 至少包含几种身份

`CompileCache::find` 外层按 function id 查找，内部比较 default stream/device、输入数、rank、dtype、shapeless flag、constants，普通模式还比较每个 shape。input strides 不属于这里所有层的统一 key；Metal compiled path 另外选择 contiguous/strided、large-index 与动态维度变体。

Python trace 缓存命中、生成 shader/library 命中、pipeline 命中、输出值的数学正确性，是不同层。记录 Python 函数体执行次数可以验证 retrace，但不能据此统计 GPU 编译次数或 infer 每个 dispatch 名称。

本机调用序列为 2×3 重复、4×3、4×5、rank1、scalar、FP16 scalar、回到 FP32 scalar：fixed 的累计 trace 数是 `1,1,2,3,4,5,6,6`；shapeless 是 `1,1,1,1,2,3,4,4`。正 stride 的 transpose view 又通过全量参考，但未增加这些 Python trace 数。它不证明后端没有首次创建另一 pipeline。

## 3. Shapeless 不是忽略全部形状语义

读本 cache 比较仍检查 rank/dtype。官方文档里的简短 variable-shape 示例不能替代实际版本代码和测试，尤其不能推导为跨 rank 永不 retrace。shape 同 rank 可复用 graph，也不表示 Python 构图时读取的 shape 会自动成为动态控制流。

本机负控在 trace 时根据 `x.shape[0]` 选 `x+2` 或 `x-2`。shapeless 第一次见 2×3 后，4×3 仍用旧分支，输出 finite 但对原 Python 函数错误（max abs=4）；这符合 shape-dependent Python 控制流不应盲用 shapeless 的合同，不是 GPU 故障。需要用 tensor/dynamic 表达、显式 bucket 或普通 compile，而不是扩大误差门槛掩盖旧分支。

reshape/flatten、broadcast、mask、分段 attention、分辨率与 prompt 长度，都可能把静态 shape 读入 Python 路径。先证明 graph 的 shape-polymorphic 性质，再考虑 shapeless；新 shape 通过一个简单 unary 不等于整个 DiT 合法。

## 4. 隐式参数是 constants，不是免费可变模型状态

若闭包读取 `state[0]` 而它不在参数/捕获 inputs 中，trace 后替换 list 元素不会自动改变 cached graph。实测 gain 2→3 后，隐式版本仍返回旧结果，对新期望 relative L2=1/3；显式传 array gain 或 `inputs=state` 的版本则正确跟随更新。

这直接关系到 LoRA、模型参数替换、scheduler 状态和随机数：形状没变，不等于所有 source generation 没变；cache key 包含某个常量类别，也不自动涵盖任意外部对象的内容身份。需要定义参数更新方式、显式输入、失效/重编译与 owner lifetime，不能靠函数名或 checkpoint 路径判断。

本轮未验证训练 optimizer、复杂嵌套对象、跨线程 RNG 或动态 LoRA 请求；不要从这组三元素增益测试扩大资格。

## 5. Stride 与整数索引是独立的正确性问题

`backend/metal/compiled.cpp` 为 contiguous、固定 rank strided、dynamic-rank、large-index 等生成 kernel，按实际 strides/shape 提取元素位置。large 不只与 tensor 元素数有关：负 stride 需要有符号索引；把 uint 偏移用于反向 view 可产生错误。

[编译相关 PR](compiler-optimization-prs.md)中的 #3720 修复这一层，不是改数学公式；它与输入是否被复制、预先 contiguous 或 matrix kernel 的 row pitch 也不同。本轮只运行了正 stride transpose，未把 PR 的回归矩阵冒充全部本机覆盖。

## 6. Cache 销毁也是生命周期合同

主线源码与旧版在 cache 句柄、共享引用与同步上有差异。[#4248](compiler-optimization-prs.md)针对 Python function 在不同线程销毁时擦错 cache/竞态的问题，引入原 cache 的引用和保护；作者也明确保留并发假设。它不是允许任意多线程并发修改同一 compiled function 的保证。

本轮没有做破坏性并发/解释器关闭实验。版本变化后应重核 thread、stream、function lifetime 与 allocator 的关系，避免把旧问题直接宣布为当前必现，或看到新锁就授予无条件线程安全。

可运行代码：compile_probe.py，显式 `mlx:compile-contracts` selector。普通离线控制见 compiler_contracts.py；源码阅读、CPU 模型、GPU 数值、性能与模型采用继续分层。
