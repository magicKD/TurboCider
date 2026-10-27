# tinygrad Metal 深读：搜索、TinyJit、ICB 与主机提交编译

<!-- knowledge
id: technique-metal-tinygrad-metal-jit
type: implementation
stage: prefill, decode, ffn
operator: gemm, norm, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: compile-bound, launch-bound, wrong-layout, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [来源](sources.md#compiler-sources-20261016) · [实测与失败记录](../MAINTENANCE.md)

2026-10-16，固定 `tinygrad/tinygrad@3d057f05df532deca7661bfb2400c6564f96a04a`。本页读 UOp/renderer/search/Jit/Metal runtime，也实际运行了小型 Metal 组件。它不是 tinygrad 全模型、训练或上游全部 tests 的验收。

## 1. 一条完整纵向链

```text
Tensor graph / views
  → schedule + memory planning + legal optimization choices
  → optional BEAM search / tensor-core optimization
  → UOp PROGRAM / generated Metal source / compiled binary
  → HCQ lowering：主机 submit 程序 + Metal indirect commands
  → queue/fence/event/residency + device synchronize
```

`MetalRenderer` 在 Apple family guard 后登记 `tc.metal`，生成 `simdgroup_*8x8` 与 `simdgroup_multiply_accumulate`。这种内部 “tensor_cores” 名称不是 NAX/ANE 证据；当前读本的这条路径仍是 SIMD-group matrix。BF16/FP8/FP64 支持有独立条件，不能仅按设备支持新 Metal 语言版本推断 dtype 或指令路线。

本机 tinygrad 记录的 `Apple9` 来自 `MTLGPUFamily`，不是 MLX `applegpu_g16s` 字符串里的 architecture generation；两个编号体系不可直接比较，更不能用数值大小决定 NAX 能力。

## 2. Metal kernel 之外还有一个主机编译器

`ops_metal.py::MetalCompiler` 直接调用未文档化 MTLCompiler 服务，组装编译请求并得到 MTLB。编译模块还导入 LLVM bindings；更关键的是 HCQ 会生成 CPU 提交程序，其默认 `HCQ_RUNTIME_DEV=CPU:LLVM`。

本机第一次 smoke 在加载 host LLVM 时失败，尚未证明 GPU 算术有问题。根据源码，子进程显式使用 `HCQ_RUNTIME_DEV=CPU:CLANG` 与 `DEV=METAL;CPU:CLANG` 后，CLT clang 编译提交程序，Metal 计算成功。没有安装 LLVM、修改上游或关闭系统保护。这个版本绑定的正常后端选择不能推广成所有 tinygrad 版本的通用开关。

主机代码生成、MSL 编译、library/pipeline 创建、实际 kernel 运行与输出同步要分别归因。一个前端/依赖错误不能写成“芯片不支持 Metal”；一次跑通也不证明 private compiler 接口长期兼容。

## 3. TinyJit 的三个阶段与失效条件

`engine/jit.py` 的第一次调用正常执行；第二次收集 linears、合并、计划 scratch、编译/link；以后按 expected input view/变量/dtype/device 和参数名验证后 replay。它保留的 output Tensor 对象可以复用，后一次结果可能覆盖此前引用看到的内容；需要旧值的消费者应在完成边界后取得独立 snapshot，而不是把旧 Python 引用当不可变记录。

实际测试覆盖新输入值、同 shape 的新 weight、shape-change 拒绝和 duplicate-buffer 拒绝。相同 shape 不表示权重值恒定；明确作为参数传入的 W 应跟随新值。fixed JIT 输入合同不满足时，本读本会拒绝，不是像 MLX 普通 compile 那样为新 shape 自动新建 cache entry。

memory planner 要保留 live Tensor/constant buffers；捕获后的中间 graph 及时丢弃，避免保留两套 working set。输入恰好落在 JIT 内部 buffer 时还可能需要独立 copy 以保护读写语义。取消、重入、异步 consumer 和真实模型状态没有在本轮一并验证。

## 4. ICB 不只属于 TinyJit

`ops_metal.py::MetalQueue` 本身就把 kernel pipeline、argument buffer 和 dispatch 写进 indirect command buffer，发出 executeCommands/fence/event。TinyJit 在此之上合并已捕获的执行序列、复用内存/编译与主机提交工作。**不能说 eager 不用 ICB、TinyJit 一开才使用 ICB。**

参数结构中存 GPU 地址与 scalars；这与只存 32-bit buffer offset 不同。residency set 不可用时走 resource 列表；`MetalAllocator._free` 会先同步，因为 Metal 不会替此调用图追踪所有可达 GPU 地址。ICB 更新了指针不等于原 owner 可以立刻释放。

图中声明 kernel 仍不能替代实际执行证据。HCQ lowering 后原 `Ops.PROGRAM` 已被提交 IR 消费，直接扫描最终图找 source 会误报“没有 Metal program”。本轮 collector 最初就出现此客户端错误；修复为用 HCQ 保存的 program key 匹配 compiler cache，记录对应 source/binary SHA。它是编译/提交元数据，不是硬件 counter trace。

## 5. BEAM：有界合法变换搜索，不是自动性能证明

`codegen/opt/search.py` 枚举 split/upcast/local/swap/tensor-core 等动作，过滤非法或过大的资源配置；按编译 binary 去重，给编译/设备计时设置预算，保留较快候选。它将测量结果与已用 options 缓存，减少后续探索。

几个值得保留的边界：search key 含 AST/device/renderer suffix 和搜索设置；probe 可能缩小 global grid 再按 factor 放大时间；用少量计时的最小值筛选，而非完整消费者的 ABBA。缩小 workload 会改变 launch/cache/occupancy 占比，线性外推只能作搜索启发式。合法变换仍需独立全输出验证、tail/stride/极值和最终模型确认。

本轮设备扫描固定 BEAM=0、PARALLEL=0、CACHELEVEL=0，cache 重定向到 ignored 实验目录；**没有运行 BEAM 性能搜索**，不将普通 JIT 收益说成 BEAM 贡献。后续若启用，要单列搜索成本、候选空间、数值门禁和独立确认。

后续[第八轮](case-studies/tinygrad-beam-search-and-amortization.md)已运行同一固定checkout的真正BEAM/Metal搜索、cache重建和同框架TinyJit对照；本页保留第三轮历史配置，不把其Jit/eager倍率改写成BEAM贡献。新的局部收益、搜索成本、关闭PR和失败语义见后续页。

## 6. Profiling：软件估算与设备时间分开

`renderer/Estimates` 的 FLOPs、loads/stores、去重逻辑 bytes 是从 UOp 推导的 estimates，不是 DRAM/ALU PMU。VIZ 支持 rewrite/source/kernel 的关联，CLI 的 JSON 便于自动分析；AMD SQTT/PMC 的说明不能继承为 Apple GPU 功能。

Metal runtime 的 profile slot 最初保存 command-buffer handle，synchronize 后读取 GPUStartTime/GPUEndTime，再换算 ns。那是已完成 buffer 时间，不是任意 per-op counter。本轮正式比较只报告 host call→device synchronize wall，没有拿这些 estimates 伪装硬件吞吐。

数值/性能与 source/binary hash 见[本机报告](../MAINTENANCE.md)。约 6× 的某些 TinyJit/eager 组件比值只属于本框架相同输入和边界；与 MLX、MPS、真实推理请求的速度排名无关。

本地 CPU 控制：compiler_contracts.py、search qualification。真实 probe 是可选 tinygrad integration，必须显式给 source checkout；核心离线学习不依赖兄弟仓库。
