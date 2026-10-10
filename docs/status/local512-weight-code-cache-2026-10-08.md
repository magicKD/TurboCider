# 有界 W8 codes 缓存：真实命中，但复制型路线没有整请求收益

2026-10-08，Asia/Singapore；M4 Max 64GB / macOS26.6.2。接续
[共享 LoRA ranks](local512-shared-lora-and-decode-2026-10-08.md)。本轮
实现 Private/Public 共用的转换结果缓存并测真实一/两图编辑；没有
下载或修改模型、增加磁盘权重副本。完整 Z/Qwen base/LoRA、encoder、
GGUF/ConvRot、快于优化 GPU 的目标仍 active。

结论：cache 确实复用 W8 codes，而非仅 scales 或条件缓存；但当前
命中后还要复制到固定 W bank，两个有限诊断都没有整请求收益。
保持默认关闭，不能把实现兼容或 staging 局部减少当作加速验收。

## 实现与准入

显式 `TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES`，默认0，严格
decimal0..2147483648，进入 executor configuration identity。Shared
`gpu::Device` 管理原有 GPU 转换后的 signed I8 codes 和 normalized
FP16 row scales；Private/Public W8 都消费同一实现，无 private selector
进入 Public。原 source、basis、H ordering、W/A量化、LoRA和 restore
舍入不变。不是 full dense cache、GGUF磁盘格式修改或 activation cache。

key 沿用完整 source/metadata weak allocation generation、物理 offset/
pitch/dtype/geometry 与 selected rows/columns/basis/block/seed。Raw GGUF
只有已验证 importer 的完整逻辑内容 tag 才能跨 physical refill命中。
不可变 promise 缺失、generation过期或 transpose A8均不缓存。没有
full payload rehash：同一 generation内改数据违反调用合同，调用者必须
换 generation。模型现有 source bridge提供不可变来源；缓存不强持模型
或source lease，旧来源到期后在下一次 staging purge。

使用 first-admitted live-generation policy，至多128 entries，不为一遍
32层遍历做反复 LRU eviction。容量不足的新来源正常执行原 stager；
不失败伪装命中、不扩容、不降低内存门槛。过期或失败 entry可回收。
每个 graph 的原 memory estimate增加**完整配置 upper**，使用原系统
4GiB reserve与 family optional tier准入；没有把2GiB optional上限提高。

独立 capacity ledger包含 pending和被 producer ticket持有的旧 entry。
erase/clear不能假装释放这些容量。先 reserve page-rounded upper，再
校验 MTL allocation；释放未使用 upper。GPU完成、status flags0后才
发布 entry并计 completed hit/fill；准备/编码异常撤销未提交 entry，
防止永远 pending。失败仍走原完整 operation GPU recompute，不发布
partial scratch。这个 ledger只约束缓存 backings，不是进程/driver RAM cap。

codes cache存连续逻辑 byte span，scales cache存连续 half，不复制原
target的 row/page padding；hit通过blit + stride-aware scale kernel回写
原 bank。`weight_code_cache` nested receipt单列 completed hits、misses/
fills/failed fills、ready/retained、live/peak capacity、declines/ineligible。
它不同于 scale metadata cache，也不证明物理ANE INT8 MAC或overlap。

## 实际 Qwen512²：同库、相同优化 GPU 控制

本地 Viggle r256、六步、seed29、三个 fresh prompts，一/两张原 reference
按原序保持。Private DiT5120、GPU encoder、两臂共享 gate/up ranks，
GPU/两条混合路线使用同一 encoder source retention。开关预算1280MiB；
off=0。每臂独立resident进程，一冷两热，100ms process-tree采样。

| workload / order | GPU warm median s | cache off s | cache on s |
| --- | ---: | ---: | ---: |
| 一图：on→off→GPU | 10.961232 | 11.687484 | 11.975583 |
| 两图：GPU→off→on | 13.141642 | 14.508511 | 14.600090 |

对应 on/off三个 PNG在两种 workload中均按 SHA256一致，且与前轮同
Private recipe的 PNG一致。每请求227个实际 adapter projections、192个
hybrid blocks、384个共享 rank arrays；一/两图实际calls分别224/256。
failure、fallback、overflow retry均0。source只load1次，后两次reuse。

on两种workload最终63个ready entries、fills63、hits累计315/693/1071，
misses累计261/459/657、declines198/396/594、eviction/failed-fill0。
实际retained/live1,321,807,872 bytes，peak1,321,816,064，小于配置
1,342,177,280。不是配置标记假称cache执行；未接纳矩阵继续转换。

这两个诊断没有请求 continuous-load资格，`qualification_passed=false`。
两个workload的route顺序相反，但不是同一workload的ON/OFF双向重复，
不能合并为统计因果或借前轮库的GPU分母。外部推理进程未被停止，
原strict gate未修改。所有6个memory报告complete、系统swap-in/out0。
按表route顺序：一图GPU/off/on峰值约40.5/38.3/39.6GB；两图约
41.0/38.9/40.2GB。scope是load/cold/warm/exit的process tree，不归因
外部服务/driver，新增缓存实际增加约1.3GB进程footprint。

目录：`outputs/local512-qwen-edit{1,2}-weight-code-cache-v1-diagnostic-20261008/`。

## 成本结论与接续

一图两次热请求的 stage host span，on为1.353238/1.470267s，off为
1.628206/2.134728s；post-join span反而较长。stage wait仍很短。这些
窗口含交叠工作，不是独立GPU kernel时间，不能直接相加或据此断定
缓存内存、复制或其他进程是唯一退速原因。

当前复制型缓存不能推荐为新性能默认。下一步优先验证直接绑定已完成
cached IOSurface bank的零复制 consumer；这需要将capacity claim随
surface/bank别名存活，隔离current/future readers并保持原完整失败回退。
不能复用一个已被下层consumer引用的cached bank作新来源写目标。
本轮**没有**实现或宣称这个零复制方案。

之后还需 Z base/真实LoRA、Qwen base generation、encoder与Public
实模型盈利、GGUF raw读取/转换成本、多prompt/seed视觉和有效同库
性能窗口。当前格式组件覆盖不替代这些完整要求；早先dense预解码
负结果仍有效，不通过增加全模型dense副本解决。

## 验证与磁盘

- Private选定19项通过；另在1MiB cache授权环境下原actual channel
  base/LoRA/late-failure/deferred lifetime test1项通过。
- Public选定18项通过；隔离已有coremltools9环境另5项通过，包含真实
  Public W8完整executor的缓存hits、base/动态correction、headroom与
  failure recovery；不是只有shared Metal test。
- 11个actual GPU格式case覆盖dense三dtype、affine Q4/Q8、raw Q4_0/
  Q4_K/Q8_0/Q6_K、direct raw/packed ConvRot，对cache off逐字节比较。
  另有不同B数据/A返回、物理refill、padding、mutable拒绝、generation
  到期、escaped ticket预算、nonfinite失败及healthy refill。GPU首轮
  编译失败因ObjC block不能捕获structured binding，之后一个夹具错误
  假定actual allocation等于page upper；各失败与修正retry日志保留。
- 两库491个source inputs匹配，Public actual release-binary guard通过；
  是含原ConvRot草稿的working-tree snapshot，本次只selective stage
  owned hunks，未借此声明clean staged-tree或全仓测试通过。

```text
Private 03975198ab74865756374d7e1f0040904f83775181523fcc28188e74803387d3
Public  0775a86959a970affe46c03ef12196defc8d70e2be945aa1004f50ad3dd126f6
```

所有owned jobs已terminal后，仅清理上述两build的423个`.o`，32,939,424
logical bytes（约31.4MiB）及两空module-cache，可重建。保留库/CLI、
manifest/log/PNG、模型、adapter。相对路径机器记录见
[本轮证据](../design/validation/local512-weight-code-cache-20261008.json)。
