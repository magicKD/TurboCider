# ANEForge：结构搜索、精度预算和 e5rt 边界

<!-- knowledge
id: source-upstream-aneforge-search
type: source
-->

[来源目录](../README.md) · [机制与负控](../../ane/graph-rewrite-search.md) · [本机实验](../../ane/case-studies/graph-rewrite-search-cost.md)

固定读本：`sbryngelson/ANEForge@896c4943c5ebf811c0e815f8d1381fd1e157fa7d`，2026-10-16，tracked worktree 干净，不声称最新版本。逐文件 SHA 与阅读范围在[唯一来源登记](../README.md)。上游 README、AGENTS、SECURITY、LICENSE、NOTICE 已读；核心重点是 `_optimize.py` 全文725行、`_runtime.py` 全文280行，以及 graph、MIL emitter、rewrite、target 与 native shim 的选定范围。没有把未读的全部 bridge/async/模型加载器当作已审计。

## 三条实际调用链

`graph.Tensor` → `_compile.compile(opt=0)` → `_Emitter` / `_blob.BlobWriter` → MIL + weights → `E5RT.compile` → `_lib/ane_e5rt_dispatch.mm`。

`tune` → `_graph_key` / `_load_cache` → cache hit 时直接 `build_variant`；miss 时枚举、cost 排序/剪枝、`measure`、缓存 winner。

`Model.__call__` → 每次输入转 FP16/copy → `Program.eval` feed/同步 execute/read → 输出转新的 FP32 数组。`input_view/output_view` 是另一条借用内存接口，不应把它的 execute-only 时间借给完整 Model 边界。

## 值得吸收但必须加合同的机制

| 源码机制 | 能借鉴什么 | 不自动成立的结论 |
| --- | --- | --- |
| `_graph_key` 把 ndarray 编为 shape/dtype，忽略权重值 | 结构相同可复用候选优先级 | 新 checkpoint/LoRA 的精度与性能资格 |
| `_emit_program_dir` hash MIL bytes + weight blob | 常量内容改变会得到不同编译目录 | 结构搜索 cache hit 已经重新验数值 |
| `measure` 返回最小 latency；warmup 输出过 gate | 有界候选快速筛选 | 所有 timed 调用正确、稳定中位收益或独立确认 |
| lossless variants 按 cost 排序，首个成功结果成为 reference | 无 lossy 自我作证 | reference 一定就是原 opt=0 图 |
| `tune_precision` 没有候选满足预算时仍返回 most-accurate | 报告保留失败理由 | 返回 model 就意味着满足 `target_error` |
| unsupported reference 可退到 FP16 baseline | 仍可做实现间比较 | 获得了独立 FP32 数学 oracle |
| `_route_ids` 排除 causal/masked SDPA | 合法变换空间应先于搜索 | 直接调用无 mask 的 decomposition helper 也有同样保护 |

上游精度 gate 是 max-absolute-error / max-absolute-reference，与本轮独立 full-output relative L2 不同；不能直接搬 `5e-3`、1.10× 或固定 cost 常数作为 TurboCider 规则。`input(max_abs=...)` 是调用者承诺，不是 runtime clamp；target detection 的 brand/family 映射也不是物理能力实测。

## Private runtime 与生命周期

shim 动态载入系统 Espresso 私有符号；默认请求 device mask `0x4`，segmenter 字符串是 `graph`，不是“关闭 segmenter”。本轮明确指定预构建 dylib、禁止 auto-build、使用任务 cache；保留编译失败 breaker，不开 profiling、fork-safety 例外、experimental async 或多 op alias。

上游 import 会设置 `KMP_DUPLICATE_LIB_OK=TRUE`；本轮只在独立子进程导入，不改变调用方长期环境。`ANEFORGE_NO_AUTOBUILD`本身不禁止搜索其他已存在dylib，所以客户端先检查明确dylib存在再指定override，不能只靠这个开关证明binary身份。

同步成功后才读输出；Model 返回独立副本，Program 内部输出数组和裸 ctypes view 则会复用。view 不持有 Program owner，绝不能在 release 后读取；本轮只测试活 Program 内的更新/poison/copy 控制，不真正触发 UAF。

`int8=True` 的已读 emitter 用 per-output-channel symmetric INT8、FP16 scale 和 `constexpr_affine_dequantize`；文件更小不证明物理 INT8 MAC。MIL 中的 buildInfo 字符串也是上游 emitter 写入的 metadata，不作为本机真实工具链版本。

## 实际执行与边界

原 checkout 保持只读，复制到 ignored 任务目录后构建小 shim；58个上游文件在运行后与原树逐项比较，关键 per-run 文件还做前后 hash。没有安装包/下载模型/修改系统安全设置。13组真实同步设备实验、1个成功 CPU contract 与1个 frontend 参数错误全部见。CPU cache/precision 的控制流测试使用显式 mock，不伪装成硬件结果。

后续已将stable exp(-abs)与配对channel分担用于[真实Qwen layer0完整FFN消费者](../../gpu_ane/case-studies/qwen-private-ffn-copy-bridge.md)，并与完整原BF16 GPU及同图serial比较，冻结share后做step19/10和24-group确认。这是copy-based API桥接的warm组件观察；不把小图约3×迁移到Qwen，不授予完整请求、LoRA、shared backing、物理overlap或全上游engine资格。

许可：MIT，Copyright (c) 2026 Spencer H. Bryngelson；上游 NOTICE 明确不授予 Apple 私有接口权利。这里只分发原创 API 客户端、独立合同/oracle 与脱敏报告；上游源码/shim binary/系统 framework 均不打包进核心 Wiki。

[固定 optimizer](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_optimize.py) · [固定 runtime](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_runtime.py) · [固定 shim](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_lib/ane_e5rt_dispatch.mm)。
