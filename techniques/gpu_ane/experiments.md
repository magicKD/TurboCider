# 实验：GPU 机制、公共 Core ML 压缩与已有真实异构证据

[专题导航](README.md) · [全部整理数据](experiment-evidence.json) · [只读重算器](curate_experiments.py) · [空闲准入工具](guarded_run.py)

2026-10-16 新运行 4 个成功进程：两组 MLX 机制实验、两组公共 Core ML 压缩实验。**没有重新运行上游引擎、私有 ANE 或完整生成请求。** 下面的真实 Qwen GPU+ANE 是本库此前独立记录，明确另列，不能把两批结果拼成一个新端到端方案。

## 环境与无干扰检查

本机 M4 Max、40 GPU cores、64 GiB，macOS26.6.2/build25G83；安装 MLX0.32.0、NumPy2.0.2、coremltools8.3.0。运行记录保留实际 source/core binary hash，不把所读 MLX 主线 commit 当 wheel 构建身份。没有清系统缓存、锁频、更改电源/安全设置或终止其他应用。

先只读查看进程：未观察到明显活跃的模型 benchmark；常驻 App 不等于正在推理。每次实验由 guard 在启动前连续 5 次、间隔约 1 秒检查 IORegistry `Device Utilization % < 5`，四次均为 `[0,0,0,0,0]`，再串行启动。独占锁只排除本工具的并发调用，不锁其他程序。

**限制：**这是粗粒度系统上下文，不能归因到进程，也可能漏掉短 GPU burst；during 采样也曾为 0，即使 MLX 已实际执行。故不能据此声称硬件绝对空闲或物理 overlap。没有连续 thermal/功率记录。首轮 sandbox 拒绝 archived IORegistry 读取，guard 在任何设备命令前失败；获准后的新目录重试成功，失败未抹去。

## A. GPU：两个新进程、两个 seed、896 个计时输出

执行 [engine_probe.py](../labs/metal/engine_probe.py)，seed53/61，每 case 8 个 ABBA block、32 次测量；每次 measured output 对独立 CPU FP64 reference 检查。14 cases × 32 × 2 = 896 个计时输出，另有 first-call 检查；预热耗时不计入下表。固定数值门槛不在结果出来后放宽。

计时含 Python graph 构造/dispatch/`mx.eval` 完成；输入/权重预先在 GPU，CPU 输出转换/验证在计时外并在下一 arm 前完成。不是纯 kernel time，也不是模型请求。Q4/Q8 的预展开 candidate 排除 decode 准备，并额外驻留 2 MiB FP16 matrix；两支比较**同一份 packed 编码的解码值**，不是原 FP16 与量化模型互比。

以下倍率为每 ABBA 内 `mean(baseline)/mean(candidate)` 的中位，非两个全局中位数之比；>1 代表 candidate 更快。两次方向不同也保留，不取最大值。

| Case / candidate | seed53 | seed61 | 解释 |
| --- | ---: | ---: | --- |
| compile RMS-chain，M1 | 1.127× | 1.147× | 小图减少构图/dispatch 的候选 |
| compile RMS-chain，M128 | 1.266× | 1.253× | 同表达式，可能融合；未 capture 核 |
| compile RMS-chain，M1024 | 1.321× | 1.295× | FP32 输入，独立 reference |
| Q4 预展开，M1 | 0.957× | 0.956× | 未胜过 packed |
| Q4 预展开，M16 | 0.911× | 0.919× | 未胜过 packed |
| Q4 预展开，M256 | 1.001× | 0.994× | 约持平，不能晋升噪声 |
| Q4 预展开，M1024 | 0.938× | 0.932× | 更大 M 也非必胜 |
| Q8 预展开，M1 | 0.973× | 1.012× | 方向不同，不声称稳定收益 |
| Q8 预展开，M16 | 0.946× | 0.961× | 未胜过 packed |
| Q8 预展开，M256 | 1.008× | 0.973× | 方向不同 |
| Q8 预展开，M1024 | 0.952× | 0.956× | 额外驻留仍无收益 |
| fast SDPA，Q1/K4096/D64 | 1.324× | 1.332× | 对 materialized QK/softmax/PV |
| fast SDPA，Q512/K512/D64 | 1.384× | 1.392× | no mask，4 heads |
| fast SDPA，Q192/K257/D80 | 1.221× | 1.201× | 非整齐 K/headD；不覆盖全部 tail |

SDPA 只验证 no-mask、GQA=1 三种 shape，没有 sparse、causal、all-masked-row、长视频质量；也未取得实际 kernel trace，不将 D80 结果称 M5 padding/NAX 提速。量化实验 N=K=1024、group64，不覆盖所有 GGUF 格式。

## B. Core ML：FP16、W8、LUT4、W8A8 与 compiled MLX

执行 [compression_survey.py](../labs/ane/compression_survey.py)。每个图是一串相同宽度的 1×1 Conv+ReLU：C256/R256/L4/seed59 与 C512/R512/L8/seed67。输入/权重 FP16；W8 使用 weight quantization，LUT4 是确定性 scalar Lloyd 码本，W8A8 另有 4 份独立校准输入。

每个 case 5 arms，轮换并反序顺序，3 次 first/warmup+10 次 timed，65 次全输出检查；两 case 共130次。各 trial 所有 arm 使用同一输入，对原始 FP16 参数的 CPU FP64 全图 reference。计时是**host 输入→runtime 完成→新的 owned host 输出**，含必需转换/同步/回读；不是 ready-GPU-only。编译、压缩、模型加载和 oracle 在 warm timing 外。

| Arm | 小图 median ms | 大图 median ms | 小图最大 relative L2 | 大图最大 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Core ML FP16 | 0.1813 | 0.9136 | 0.00405 | 0.00316 |
| W8 weight-only | 0.1850 | 0.8396 | 0.01582 | 0.01780 |
| LUT4 | 0.1832 | 0.8427 | 0.20421 | 0.24880 |
| W8A8 | 0.2315 | 0.7688 | 0.03479 | 0.04076 |
| compiled MLX FP16 | 0.2507 | 1.1752 | 0.00044 | 0.00052 |

两图的 Core ML 配置均为 `CPU_AND_NE`，但**小图计划中非 const 的 preferred device 全为 CPU，大图全为 ANE**。这是 plan/predicted，不是物理执行计数；不能把小图称“ANE 比 GPU 快”。两组 C/R/L 和 seed 同时不同，不足以隔离是哪一维引起切换或确定某个硬件 threshold。

W8A8 相对同图 FP16：小图更慢，大图 warm 中位约1.189×（ratio of medians，非 ABBA 统计），而且是近似配方。不是整模型或物理整数 MAC 证明。LUT4 的约20%–25% L2 即使通过脚本预设0.35的**演示门槛**，本专题也不视为可接受质量；W8/W8A8 同样需要真实模型质量确认。

package 逻辑字节小图 FP16/W8/LUT4/W8A8 分别 530196/270782/137910/274257，大图 4204960/2117523/1061282/2123783；这不等于 compiled/resident/peak memory。大图 Core ML load 约43–78ms，warm 约0.77–0.91ms，不能忽略 setup；MLX first call 约6.4ms，亦非严格冷进程比较。所有 first/compile/load 数字保留在 JSON。

## C. 已有真实 GPU+ANE 证据：正负结果并列

| 既有实验 | 可得结论 | 不能得出的结论 |
| --- | --- | --- |
| [真实 Qwen 公共图/tiling](../review/qwen-ffn-tiling-20261015.md) | serial→hybrid 约1.16–1.17×，但完整 GPU→hybrid 约0.18×，负收益 | 并发相对串行快，不等于优于 GPU |
| [ANEForge 小 FFN](../review/aneforge-20261016.md) | 不同表达有数值正负结果；exp 图在其 host-fed 边界较快 | toy 正结果不能覆盖真实 Qwen/handoff 负结果 |
| [private copy-bridge FFN](../review/qwen-private-ffn-20261016.md) | 固定 layer0 warm 完整组件约1.08–1.12× | 不是全请求、零拷贝或物理并行证明 |
| [真实 shared FFN 五臂](../review/qwen-shared-ffn-20261016.md) | M1024/H4096/F12288、ANE F2048，长确认 GPU/shared约1.146–1.149×；同client copy→shared仅约1.1% | 不是“零拷贝使整请求快15%”；host overlap 不等于物理 overlap |

shared FFN 记录覆盖560次完整调用与264次 exact runtime join，并保留首轮 copy/shared 方向相反、load 时间大幅变化和存储准入失败。它是 BF16/FP16 混合精度 layer0 组件，没有完整请求/LoRA/低内存/取消资格。本轮只是重新阅读其记录，未重新跑560次。

## 可复现命令与边界

设备命令会占用本机计算，先检查没有其他 GPU 工作；guard 缺计数或忙时拒绝启动。下面路径须换为**尚不存在**的新目录/JSON。不要同时运行两项。

```sh
.venv/bin/python techniques/gpu_ane/guarded_run.py \
  --output-dir outputs/gpu-ane-replay-admission --timeout 120 -- \
  .venv/bin/python techniques/labs/metal/engine_probe.py \
  --seed 53 --blocks 8 --output outputs/gpu-ane-replay.json

.venv/bin/python techniques/gpu_ane/guarded_run.py \
  --output-dir outputs/ane-compression-replay-admission --timeout 180 -- \
  .venv/bin/python techniques/labs/ane/compression_survey.py --device \
  --channels 512 --rows 512 --layers 8 --seed 67 --rounds 10 \
  --output-dir outputs/ane-compression-replay
```

只读整理本轮原始记录，不重跑设备：

```sh
python3 techniques/gpu_ane/curate_experiments.py \
  --input-dir outputs/gpu-ane-survey-20261016 --output outputs/gpu-ane-summary-replay.json
python3 -m unittest discover -s techniques/tests -p test_gpu_ane_survey.py -v
```

原始源码快照、patch、模型编译资产和 logs 只在 ignored outputs；可分发 JSON 保存全部测量/误差、来源 hash 与安全准入摘要，不含用户模型、激活、私有地址或序列号。curator 是数据验证，不是重放原输出或硬件 attestation；未采到的 thermal、全部动态依赖、内部 copy/placement/counter 保持未知。
