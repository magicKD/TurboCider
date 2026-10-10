# Qwen GGUF：已测快速路径成为默认

2026-10-11，M4 Max64 GiB / macOS26.6.2。新匹配库为
`build/qwen21-fast-defaults-v1-private/libturbocider.dylib`，native-only、
Private enabled、experimental probes0，513 个 native inputs 与源码匹配。
这次不新增 kernel 算术，而是让符合条件的普通请求直接得到此前已测
的快速组合，不再依赖手填六个优化环境变量。

## 默认与关闭

Qwen GGUF component import 默认 8 个 CPU decoders、4 MiB×2 有界缓冲、
一个 pread worker、verification/import overlap。encoder 与 DiT 都是
GGUF、显式 `allow_approximation=true`、base `image.generate`、512²、
unconstrained resident、无 LoRA/input/prompt enhancement/encoder ANE
manifest/冲突 W8 选项时，默认保留 encoder 权重并开启 GPU prefill。
纯 GPU 与已显式选择的 fixed Private runtime 请求均可使用 GPU encoder。

源格式在 Session 已解析后再决定，不根据目录名字猜模型。BF16、编辑、
其它尺寸、streaming、内存预算和 adapter 请求不自动套用 encoder 快速
默认。`allow_approximation` 合同仍默认 false，未绕过准入或静默放宽精度。
Private ANE 的 backend、许可、manifest 和 channel share 不会自动开启。

`TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS=0` 恢复原 unset-option 行为：
1 decoder、单 1 MiB buffer、无 read-ahead/overlap、无自动 prefill/retention。
各原参数的显式 0/1 或合法 workers/read-size override 优先；例如
`TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU=0` 可只关闭 prefill。已注册的 SHA
策略独立保留：模型 generation 未变时跨进程 SHA bytes=0，不因关闭此
性能 profile 又强制每请求 SHA。缓存缺失/损坏/文件变化时仍重新校验。

## 同库显式组合与真正未设置 override 的对照

原 Q4_K_M encoder/DiT + 原 BF16 VAE，512²/seed29/三个相同 fresh
fox-light prompts，每 route 独立新进程，一 cold 两 warm，全部 condition
miss。4 步含完整 dumps，40 步无 dumps；GPU、Private ANE5120/GPU7168
两路，40 步显式顺序 GPU→ANE，default 反序 ANE→GPU。

screen 的 `--fast-defaults` 真正移除八个 override，并验证实际 import
和 prefill/retention receipts，不是将相同显式 flags 再运行一次。
来源认证仍使用此前已完成的注册；各新进程 encoder/DiT model SHA bytes0，
persistent hit1、record write0。import 仍实际读原权重，零 SHA 不是零 I/O。
两个实际 4 MiB buffers 共8 MiB，8workers、读 worker/drain/jobs 逐项检查。

| 40 步配置 | warm encoder s | warm denoise s | warm request s |
| --- | ---: | ---: | ---: |
| GPU 显式快速组合 | 0.090295 | 43.739972 | 44.430200 |
| GPU 默认组合 | 0.092325 | 43.744789 | 44.449221 |
| Private 混合显式快速组合 | 0.090769 | 36.171430 | 36.872810 |
| Private 混合默认组合 | 0.092333 | 36.164966 | 36.861361 |

默认组合的完整输出与显式组合一致，耗时差约 ±0.04%，不解释为新 kernel
收益。混合对自身 GPU 完整请求约少17.1%，属于现有通道分工的收益。
4 步 24 对完整 text/initial/latents/pixels tensor 的 shape/dtype/payload
逐位一致；4/40 步总计12对完整 PNG byte-exact。跨 GPU/ANE 不要求
bit-exact，不将这里的 default-vs-explicit 相同扩展成量化质量资格。

八份 process-tree memory evidence 均 complete，40 步默认 GPU/mixed
peak physical 为15,268,945,504 /15,417,466,584 bytes。所有八窗口系统
compression/swap-in/out 增量为0，少量 decompression 单列机器记录；
不是 hard-RAM、外部服务/driver 归因或设备独占/thermal 资格。正式实测
期间未并行运行其它模型；这是功能/既有速度回归，不新增跨框架排名。

## 生命周期与回归

真实 C API prepare 测试通过：prompt hit/fresh、显式 prefill切换、
processor mutation拒绝与恢复、unload、未设flags命中同effective profile、
master0真实释放retained state并走 legacy、恢复defaults重新导入/融合。
master0的实际receipt为 workers1、single1MiB、read-ahead/overlap false、
retained bytes0；恢复后 workers8/two4MiB、source loads3，不借旧conditioning。
这不是仅靠 host predicate 声称关闭开关有效。

host options1项、screen contracts9项、真实 session1项通过；前轮 source
registration24项和 SourceLease回归也已复核通过。未重跑所有模型的全量
make test，未新增 public/hard-memory/编辑/LoRA/所有尺寸资格。

[完整逐请求/精确输出/实际配置与生命周期记录](../design/validation/qwen21-fast-defaults-20261011.json)
由 `tools/validation/qwen21_fast_defaults_evidence.py` 独立读取原screens
和成功session日志导出，不加载或重哈希模型。build binaries、模型、
缓存key/receipts、raw PNG/dumps/logs 均不进Git。

整个优化与多模型benchmark目标继续active；
[base报告](../benchmark/README.md)正在补齐当前外部框架对照，未测项不会
用历史不同模型或机器的数字代替。
