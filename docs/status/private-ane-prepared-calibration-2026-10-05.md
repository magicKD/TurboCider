# Private ANE：预绑定 calibration batch

接续 [deferred join 初筛](private-ane-deferred-join-2026-10-05.md)。实现提交
为 `e9105a8`。本轮补齐独立测量的底层接口，不把它称作已完成的模型
bandwidth-aware calibration；四格 base ≥1.2×、LoRA加速和完整质量/
内存/设备 trace 目标仍 active。

## 计时与所有权

现有 `prediction_seconds` 是 `request.finish()` 的 exposed host wait，
可能包含 GPU producer和交接，不能代替 ANE-alone。新增 move-only
`Program::prepare()`：先绑定并保留输入/output/model/event，不提交
evaluation；`PreparedRequest::submit()` 消费一次。未提交对象销毁时
解除 request/completion callback 的引用环；提交后的 ticket仍保留原
异步资源与timeout quarantine。原 `enqueue()` 用两阶段实现，正常
推理仍使用 GPU producer 的 ready signal，不将 calibration CPU release
带入模型的 shared-event 路径。

新 `CalibrationBatch` 接收1–4组预完成、只读输入与各自独立output。
在计时前满足依赖；计时覆盖实际 client submissions、GPU callback
及两侧 completion joins，不包含绑定、前置 producer或signal CB。
GPU-alone不提交Private evaluation；ANE-alone没有GPU callback；
concurrent arm不以 ANE output作为GPU callback的依赖。测到的是
host batch span，不是物理ANE kernel时间，也不是physical overlap证明。

重复消费、空/超过4组、output别名、输入/output别名、重用timeline和
不完整GPU callbacks拒绝。GPU submit或finish异常仍drain全部已提交
Private tickets并保留原异常；失败不发布 inference scratch或successful
sample。调用方仍须在构建前完成 producer、保证GPU mutable scratch
独立、做calibration arena memory admission，并校验完整pipeline数据。
Public构建不包含该Private实现；Source/MIL/精度/原始weights不变。

## 验证范围

- Private native-only新库：
  `02f5d0184dea7bf40c0ba3290dd7fac72ae403db3b31b9811343abd55e71c757`。
- Public native-only新库：
  `a618a736574e362734030978ff9e8feb7d1fd59d44940a0705e5d687cefbc045`。
- 最终14 Private host/hardware/MLX、13 Public Core ML/MLX/receipt通过；
  两库各32项Qwen原生C API contract、60 screen host通过。新Public
  实际flags/class strings/PrivateFrameworks链接发行检查通过。
- 新实机fixture是H128/F2560/33-row的W8A8 SwiGLU，share0.4/0.8，
  五组不同合成权重以深度ordinal0/7/15/23/31标记；独立1次/4次
  evaluation，GPU-alone、ANE-alone与concurrent三类实际提交通过。
  有限非零输出、重复逐位一致、move/one-shot、未提交销毁、调用方
  Device/Program/input wrappers先销毁、别名/partial binding拒绝、
  GPU submit/finish异常后的drain/reuse均通过。
- **Fixture GPU callback只含FFN head，不含staging/restore；第五组不是
  实模型的未来pipeline采样。** 不能用这些toy host spans选择模型share、
  拟合已资格带宽模型或报告产品加速。
- 验证库含独立ConvRot working-tree修改，没有clean staged tree重建。
  本提交只含本专项代码；ConvRot仍留在工作区。未运行完整`make test`，
  既有Qwen3 source字符串断言未在此修改。不声称native INT8 MAC placement。

## 最新库实模型检查与无效性能窗口

同一新库的Z512 base、fox/seed42、8步、c1056/v1/Fa4096、Private
W8A8 GPU I/O、固定async1、deferred0；scale-cache/launch-fence/
stage-specialize1，W-prefetch/A8-lookahead0。先后两个GPU→Private
对照均在GPU arm被连续load verifier否决：分别57个samples中2个、
54个samples中2个观察到短时高CPU ComfyUI Python进程。两个原始
summary均保持incomplete、0个已认可trials，没有继续Private arm，
没有借用其GPU时间或放宽门槛。

独立Private generation检查一冷一热实际生成完成；两个原生结果经
shared严格route/IO/data-path/fixed-async/deferred校验，每请求256次
Private calls，0 retry/fallback。热PNG对重构前Private的byte hash同为
`6d3f01c35392246eedc75fddc76c8dc6d0048ad9ec42e339fbaa7bdc20aaf7cf`。
但其26个load samples中仍有1个竞争进程；screen同样incomplete。
仅独立生成回执与PNG一致性通过，不补写成有效性能/媒体资格。
CPU verification保存为该目录的`verification.json`；原始stdout/summary
不重写，读取的是`.stdout.jsonl`，不是旁边的request JSON。

目录、build/test/原始stdout/load/PNG的哈希见
[机器记录](../design/validation/private-ane-prepared-calibration-pilot-20261005.json)。
本轮未暂停或终止任何外部进程；上一轮已恢复的旧download PID已不在
运行，不向复用PID发信号。所有本轮owned build/test/inference handles
已terminal，不遗留measurement任务。

## 接续要求

当前模型channel auto仍为固定share开/关，尚未接入此测量接口。
还需将**实际模型**的多个深度、正确GPU family优化、W/A staging、
restore/join、LoRA成本及未来weight/prefetch纳入三类独立采样，以
1-vs-4扣除固定成本；按§59–68拟合共享带宽、实施memory约束、
near-optimal最小share与minimum gain，再接入按model/geometry/encoding/
backend/SoC/OS/build/kernel和graph ABI隔离的cache。现有untimed host
wait不能充当alone sample，也不凭本fixture或旧库倍率满足该要求。
最新库四格反序/多提示词及latent/感知/语义/memory/device trace验收
仍待完成；Qwen512 LoRA的负收益情况不能被base倍率覆盖。
