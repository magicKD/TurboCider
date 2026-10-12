# 05 · M5 GPU W8A8：接口、kernel 与兼容性设计

[目录](README.md) · 本轮没有 M5 实机；此页不是已完成的 M5 backend。

## 1. 目标和能力分层

以 Metal Performance Primitives 的 tensor / `matmul2d` integer operands 与
INT32 累加为候选，服务 ConvRot 已量化权重，不将 MLX `quantized_matmul`
误称原生 W8A8。M4 继续已验证 packed/浮点 fallback。

不能仅判断 `Metal4`、芯片字符串含 M5 或 kernel symbol 存在。vpipe 的参考实现
区分 Apple10 matrix-core capability，还为不支持平台提供某些占位 entry；这提醒
我们“函数能找到”远远不足以授权执行。

拟议 capability 状态：

```text
unavailable → api_available → shader_compiled → numerical_probe_passed
            → device_shape_qualified → performance_qualified
```

每个跃迁需要独立证据。老系统/SDK 编译基础引擎不引用不存在的枚举/头文件；
MPP 代码置于受 build/runtime guards 保护的独立编译单元。运行时检测 SDK/API
可用性、device family、pipeline 创建和自检。测试 mock 只能证明分支逻辑，
不能把 M4 软件 fallback 结果写为 M5 硬件验证。

vpipe该快照还把matrix-core路径限制在macOS 26.2及以上，并说明较早26.x上曾有
codegen/AIR兼容问题。本项目应将最低OS、SDK编译目标和metallib target纳入
capability/manifest，而非只检查芯片；最终支持范围仍需本项目实机确认。

## 2. 拟议原生算子接口

```text
convrot_w8a8(x, signed_w, row_scales, rotation_recipe,
             activation_quant_spec, bias, output_dtype, scratch_lease)
```

- 对 MLX 暴露一个明确 primitive/后端边界，保持 lazy dependency、stream 和 output
  lifetime；不得借 host `data()` 隐式同步每次计算。
- `scratch_lease` 来自请求内存计划，不能 kernel wrapper 自动扩出无界缓存。
- 代码与 `Weights` 生命周期分开；shader cache 按 device/OS/shader revision，
  weight repack 按 tensor/source/slice/representation generation。
- fallback 保持相同输入/输出和 bias/LoRA 语义；记录 reason/backend/precision。
- 未经资格的设备不能由环境变量强行标为 `performance_qualified`。

## 3. Kernel pipeline

### 3.1 Gate/up 输入准备

第一版先可靠实现 H256 与 per-row A8，后融合：

1. 旋转输入，产生临时 R；
2. 归约每 row 最大绝对值，得到 sx；
3. R→Qx；
4. gate/up 共享 Qx/sx，分别执行整数 GEMM 或共享加载的双输出 kernel。

每 row scale 横跨 K；单个 256 元素 butterfly workgroup 不能在没有全行归约的
条件下得到正确全行 scale。因此“旋转+A8一个 kernel”不是天然成立：可采用
分阶段 reduction、较大 row workgroup，或明确改为 per-group A8，并重做数学/质量资格。
融合必须计入额外通信和 register/threadgroup 压力，不能只看 launch 数。

### 3.2 整数 GEMM

- X: `[M,K]` signed I8，W: `[N,K]` signed I8；逻辑 transpose_y，不每调用重排 W。
- 分别处理完整 tile 与 M/N/K tail；padding codes=0，不能影响有效 row scale。
- INT32 accumulator，host 以 `K*max_abs_qx*max_abs_qw` 检查最坏上界；更长 K
  使用安全部分和，不依赖 wraparound。
- epilogue 的 scale/bias 浮点部分使用 FP32，然后一次输出 BF16/FP16；点积仍为 INT32。
- per-group scale 方案在组边界 rescale 部分和，cannot scale once at end。
- 若硬件/MPP 指令提供 relaxed precision，第一版关闭，单独测后再资格化。

建议筛选 M tile 32/64/128，N tile 32/64/128，K tile 32/64/128/256；这是搜索
起点，不是 M5 最优参数或指令支持保证。以实际 SDK 能编译的合法组合过滤。

### 3.3 SwiGLU 与 down

gate/up 恢复各自 scale 后才能 SiLU/相乘，不能将这些不同尺度移到整个 FFN 尾部。
hidden 用模型允许的浮点精度；down 再做 H256/A8。异常 row 可局部浮点路由，但
gather/scatter、dual storage 和 fallback 成本必须进入预算。

## 4. Split-K 不是无代价优化

参考 vpipe 的 group-aligned split-K 可提高小 M 下 occupancy，但全局 partial plane
通常是 `splits*M*N*4` bytes。内存受限时它可能抵消量化省下的 RAM。

- 默认 unsplit；显式搜索 2/4/... 并在运行前 reservation。
- 输出 N/M 再分片可缩小 partial planes，tradeoff 是更多 dispatch/fold。
- K 分片 balanced、group-aligned；尾块不可只为整除而丢数据。
- fold 使用声明好的浮点/整数归约，bias 仅一次；比较含 fold 的整体耗时。
- 不继承 vpipe “未调优默认 split=2”的性能结论；没有 M5 实机时全部记未资格。

## 5. GGUF 与 M5 INT8 的边界

ConvRot 是优先对象，因为 weight scale 在每输出行一致。Q8_0/K/IQ 不自动等价于
per-channel W8。三种后续方案明确区分：

1. 保持 block scales，integer partial dot 分组后浮点累加：无新权重量化，但 kernel复杂。
2. 有界反量化为 BF16后 dense：首发通用方案，不增加 A8。
3. 有界反量化再 per-channel W8：新增权重误差和 quantization成本，必须另一个
   `allow_requantization=true` recipe；不能作为格式兼容的默认捷径。

## 6. 没有 M5 时能交付什么

- 独立的数学 oracle、signed/scale/tail/overflow/stride测试。
- 能力接口与 false→fallback 测试、不同 SDK 的编译隔离。
- 内存描述与 shader/packing 版本合同，性能计数和 benchmark 场景。
- 如果新增 shader，可报告 SDK compile-only；不能报告 GPU 执行或加速。

本轮交付以文档设计和已有 GPU共享旋转实验为限，没有写一个未运行的 M5 shader
然后把功能标为实现完成。后续硬件到位先测 MatMul，再 FFN，最后完整 ConvRot
生成；验收前不修改 M4/M5 默认策略或现有历史认证。

## 7. 实机验收清单

记录芯片完整型号/内存、OS/SDK、shader hash、MLX/C++版本；覆盖 M=1/33/1056/4128
和尾部 shape。对比同预算 packed、bounded BF16、候选 W8A8，包含 H/A8/GEMM/
epilogue/alloc/fence。测试 warm/cold、前瞻0/1/2、压力下 fallback。
硬件算术/placement证据、A8 quality与完整请求wall都通过才有资格开启 auto。
证据可支持什么、不可支持什么，以及无实机时 not_run 的规则，见
[11 第 7 节](11-acceptance-profiles-and-feasibility.md)。shader 编译和数值 probe 不能
单独替代真实 integer lowering/pipeline 证据。
