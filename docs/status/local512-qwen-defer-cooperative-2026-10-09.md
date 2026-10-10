# Qwen 首步 FFN 并行：延后合并有局部改善，尚未超过 GPU

2026-10-09，Asia/Singapore，M4 Max64GB / macOS26.6.2。接续
[首步分阶段](local512-qwen-prefill-parallel-2026-10-09.md)与
[融合/精度筛选](local512-qwen-fusion-precision-2026-10-09.md)。
本轮继续验证“首步 GPU/ANE 分担 FFN，命中 prefix KV 后独立选路”，
不是把整个首步交给 ANE。没有下载、模型/adapter 改写或 dense sidecar。
完整 Z/Qwen base/LoRA、encoder、GGUF/ConvRot 和整请求目标仍未完成。

## 可以分担什么，以及依赖不能省掉什么

当前 Qwen 图像 SwiGLU 的 intermediate width 是12288。将同一组
gate/up 中间通道划为 G/A 两个不相交集合，每条支路计算自己的
corrected gate/up、SiLU×up 和 down 的对应列，最后加两个4096维
base partial。按通道拆分在实数算术下利用 down 的线性性，不是
GPU 算 gate、ANE 算 up，也不是两设备重复完整 FFN。当前 ANE
支路使用近似 W8A8，所以整个数值配方并非逐位等价。

```text
当前层 attention → FFN input / shared LoRA input ranks
                          ├─ GPU: G gate/up → hidden_G → partial down_G
                          └─ ANE: A gate/up → hidden_A → partial down_A
                                ↓
                partials 相加 + 完整 down-LoRA → 下一层
```

gate/up LoRA 必须在激活之前施加；down-LoRA 需要完整 hidden，或
分别计算 A-rank partial 后加 rank 再做一次 B，不能漏掉 ANE 通道的
adapter。现有 split-down-rank 实验尚未盈利，本轮保持原完整 down-LoRA。
attention 与同层 FFN 仍有输入依赖，下一层也必须等合并，不能把两
设备峰值吞吐简单相加。这里的 decode 指后续 diffusion 步骤，不是
自回归生成：prefix KV 复用减少参考/文本工作，目标图 FFN 仍要计算。

`HybridFfn::run_channels` 已有 ANE worker launch 和 MLX async GPU
head。真正限制前沿的是 launch 前 `eval(packed,gate,up)`、weight
stage ready、A8 staging，以及 finish 中的 GPU restore。LoRA readiness
host span还包含 lazy 上游输入的依赖，不等于独占 LoRA kernel 时间；
stage/worker/host wall spans互有嵌套，不相加当 device profile。异步
调用计数证明分工路径执行，不证明硬件上物理 overlap 或原生 INT8 MAC。

Public wrapper 使用 CPUAndNeuralEngine 计算单元集合；Private 使用
显式 client。前者排除 GPU 作为 Core ML 计算单元，但不保证每个 op
实际落在 ANE，也不承诺与 Metal 的重叠量。本轮实模型只测 Private。

## 延后合并：真实原始 LoRA 单/双参考图

新 `qwen_encoder_residency_screen.py --defer-prefill-screen` 只切换
**已有** native `TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN`，没有新
native 实现或默认变更。三个匹配臂：joint GPU、prefill-only eager
join、prefill-only deferred join；两种 hybrid 的后五步均完整 GPU。
deferred 只有在 ANE 与 restore 完成后才返回独立 MLX-owned 的 lazy
join/down-LoRA，不让下一次 stage 覆盖未消费的 ANE surface。

原 BF16 Qwen、原 Viggle v0.2.1 r256/strength1、512²六步、seed29，
三个 fresh prompts，conditioning 全 miss；每臂独立进程一冷两热。
全部 joint BF16 A/B、F32 ranks、GPU encoder 与相同 encoder source
retention；hybrid 共享 gate/up input ranks。关闭 time reuse、down-rank
split、W-code cache、weight prefetch 与 A8 lookahead。
100ms process-tree memory sampling；没有 strict competing-load 资格。

| route | 单图 warm request s | 单图首步 / 后五步 s | 双图 warm request s | 双图首步 / 后五步 s |
| --- | ---: | --- | ---: | --- |
| matched joint GPU | 10.528775 | 2.431817 / 6.114218 | 12.817958 | 3.759355 / 6.427972 |
| prefill parallel / eager join | 11.324532 | 2.711286 / 6.145284 | 13.800560 | 4.339393 / 6.359388 |
| prefill parallel / deferred join | 11.236662 | 2.587204 / 6.131164 | 13.536478 | 4.046646 / 6.358801 |

单图 Fa7168/Fg5120/bucket2112，首步2096行；双图
Fa5120/Fg7168/bucket1056，首步3144行。后续均1024行。
单图顺序 GPU→off→on，双图 on→off→GPU；这是不同 workload 的
相反顺序，不是同 workload 双向 ABBA 或稳定统计资格。只用各自
同窗口 GPU 作分母，不将各臂 phase 时间拼成不存在的最快请求。

deferred 相对 eager 首步下降4.58%/6.75%，整请求下降0.78%/1.91%；
相对 matched GPU 首步仍慢6.39%/7.64%，整请求仍慢6.72%/5.61%。
这是可测试的改善候选，不是新的默认最快路线。此前 base 首步
2.196→2.007s 的8.6%下降成立于不同的40步无LoRA实验，整请求只有
约0.33%下降，不能套给本轮六步 LoRA。

每 hybrid 请求首步32 successful channel/shared-rank blocks，单/双图
32/96实际 driver calls，后续0；227 adapter bindings，failures、
fallback、overflow retries均0，headroom1。on累计 deferred 与 channel /
async blocks均32/64/96，off deferred均0。validator要求真实累计进度
与 `host_graph_construction_deferred_gpu_consumption` scope，不把缩短
的 post-join host construction 时间当全部算术已消失。

六份 memory reports complete、system swap-in/out0，process-tree
phys-footprint peak约38.36–40.54GB；scope包含load+cold+warm+exit，
不归因外部service/driver。零swap不表示没有compression/共享带宽竞争。

两个 workload 各三张 off/on PNG 全部 exact，并与上一轮对应 joint
PNG exact；GPU controls也与上一轮 exact。已检查 case1单/双图的
GPU/deferred whole和center：主体壶形、把手、双壶位置/颜色/光照
接近，纹理/高光有小变化，未见明显新块纹、断裂或色块。未声称
其他case/seed/detail已经验收；automatic visual manifests保持pending，
有限agent观察不是用户接受或完整画质资格。

证据目录 `outputs/local512-qwen-edit{1,2}-prefill-defer-v1-diagnostic-20261009/`。

## Dense cooperative GPU 核：研究候选，不接入模型

新 `dense_gpu_cooperative.hpp` 比较每 threadgroup 四个独立 SIMD
tiles、显式 K-blocked F32 accumulation。原 BF16/FP16 physical W pitch
保留，right-input 两种 recipe：手动 cooperative loads / tensor view。
没有 widened/compact W，不改变模型默认 consumer。

早期日志完整保留：K480/SM16/BK64出现FP16 NaN；最初 masked-A
实现触发 SDK 的双 cooperative operand K限制。随后240 cases通过。
本轮扩展任意K尾和小M，再发现小输入 MLX constant address-space
与 SDK 双 cooperative M/N限制，分别修复；尾部安全路径使用K32、
M<=32/N32 elementary tiles，不假称请求的大tile始终用于tail。
materialized non-row-major W拒绝测试在明确eval后的源上进行。
最终测试及当前 selector component结果见机器证据，不删除早期失败。

最终800个实际Metal numeric cases全部通过：FP16/BF16、
M1/33/67/128、K1/95/96/449/480、五种请求tile、两输出dtype boundaries、
两种right recipe，覆盖offset、strided input和任意K尾；另20个
malformed contracts拒绝。最大relL2为2.51837e-5，保留原F32 oracle
1e-5/abs2e-5与narrow oracle .003/abs.0005预算，未为修尾放松。

先前 real layer0 original BF16 down `[4096,12288]`、合成 hidden 的
component screen：M1024/2096/3144、K3072/5120/7168/10752，六recipes，
3warmup/15cyclic-order hot。register-right明显慢，tensor-right接近
原multi-SIMD consumer但尚无明确收益；例如M2096/K7168，原核
8.447ms、tensor SM16/BK32/SN32为8.501ms、SM64为9.932ms。
这些是旧v4/v5源码/binaries的量测，不将之冒称最终tail修正源封印。

最终v9 selector probe明确接收 `register|tensor` 并记录right recipe；
同一binary串行复测两种recipe，3warmup/15cyclic hot，每种12组geometry。
以下只比较同一个component窗口的原核，不混用两窗口分母：

| right recipe / M / K | original ms | candidate ms | candidate tile SM/BK/SN |
| --- | ---: | ---: | --- |
| register / 2096 / 7168 | 8.451750 | 15.109958 | 32/32/32 |
| tensor / 2096 / 7168 | 8.448625 | 8.506833 | 16/32/32 |
| register / 3144 / 7168 | 12.569125 | 22.479750 | 32/32/32 |
| tensor / 3144 / 7168 | 12.566833 | 12.677792 | 16/64/32 |

register所有候选约为原核1.65–3.78倍耗时，tensor所有候选仍约慢
0.42–18.30%；全部component relL2为0。两receipt保持
`qualification_passed=false`；不是完整FFN/模型或device trace。
probes使用absolute rpath保留joint-v1 Private，没有相邻dylib或
loaded-image/physical overlap证据。不接入模型。

## 结论与接续

方案在依赖与实现层面可行，base首步已有局部收益，当前真实LoRA
仍未盈利。保持全部新选择显式实验，不修改default。下一步优先
实际 correction/staging/readiness 与 GPU partial-down成本，并以
真实 rows、layer、adapter、phase匹配全FFN成本选 share/route；后续
保留最快 compiled GPU decode。不以更大offload、更松误差或更少
host fences单独当加速证明。允许5–8%近似预算仍须保留finite、
shape、source identity、memory admission和晚期完整GPU重算。

构建、测试、最终量测/身份/清理记录见
[机器证据](../design/validation/local512-qwen-defer-cooperative-20261009.json)。
实模型使用保留joint-v1 Private库；新研究header不在那495-input
snapshot里，不声称当前tree native rebuild、clean staged-only或App发行。
最终host21项、retained Private/Public各7项selected regression通过，
无skip；包括原projection144、joint24、实际compiled phase24及shared
ranks/math/receipt/gates。Public actual release guard通过，首次误指
不存在的manifest导致的工具错误log也保留。没有新native库build，
直接probe编译没有留下`.o`/module-cache；temporary test目录自动
清理，无需删除模型、用户缓存或外部process，保留binary/日志/PNG。
