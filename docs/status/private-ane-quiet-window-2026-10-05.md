# Private ANE：安静窗口的同库对照

2026-10-05 接续 [fixed async](private-ane-fixed-async.md) 与
[负载来源诊断](private-ane-load-origin.md)。四格 base ≥1.2×、LoRA、
完整质量/内存与设备并发目标仍未完成；本页区分有效初筛与正式资格。

## 测量边界

用户授权后，仅对经启动时间、可执行文件及 kernel argv 核对的
`download_ltx25.py` 进程使用可恢复的 SIGSTOP。未修改或删除下载文件，
未暂停其它外部进程。恢复前须重新核验身份，避免对复用 PID 发送信号。
本地恢复记录保留在 `outputs/private-ane-download-pause-ledger.json`。
本轮全部 benchmark handles 终止后，已重新核验原进程的启动时间、
可执行文件、kernel argv 与 stopped state，发送 SIGCONT，并观察到其
不再处于停止状态。测量窗口已结束，下载已恢复；未修改下载文件。

所有下表对照使用同一 Private-enabled 库：
`da79328f140eb65312b1a5e48b9b03911141d2c54e8fbc130c2c1f5c1401638c`，
目录 `build/private-ane-fixed-async`。GPU/Private 串行、相同请求与精度；
fox/seed42、resident，排除各路线第一个冷请求，统计全部预声明热样本。
计时为含 VAE/PNG 的 native request wall，不含 profile 或张量 dump。

Private 策略为 W8A8/Hadamard、GPU IOSurface、chunks=1、W-prefetch=0、
scale-cache=1、launch-fence=1、stage-specialize=1、fixed-async=1；
512² A8-lookahead=0，1024²为1。Qwen base 的两路线均开启 Q/K norm-RoPE。
所有有效 arm 都通过连续 `--observe-load`，而非只有启动时快照。
该检查仍是 CPU heuristic，不证明 GPU/ANE 独占或物理设备重叠。

## 已完成的 base 初筛

| 模型/尺寸/Fa | 顺序/热样本数 | GPU median (s) | Private median (s) | 中位倍率 | min(GPU)/max(Private) |
| --- | --- | ---: | ---: | ---: | ---: |
| Z512/4096 | Private→GPU / 3 | 7.008714 | 5.818902 | 1.20447× | 1.20328× |
| Z512/4096 | GPU→Private / 5 | 6.992895 | 5.820822 | 1.20136× | 1.19841× |
| Z512/5120 | GPU→Private / 3 | 6.992906 | 6.196786 | 1.12847× | 1.12394× |
| Z1024/5120 | GPU→Private / 2 | 31.261005 | 28.310134 | 1.10423× | 1.10414× |
| Qwen512/5120 | Private→GPU / 2 | 41.723789 | 33.037818 | 1.26291× | 1.26253× |
| Qwen512/5120 | GPU→Private / 3 | 41.723974 | 32.994037 | 1.26459× | 1.26391× |
| Qwen1024/5120 | Private→GPU / 2 | 183.183326 | 154.379176 | 1.18658× | 1.18550× |

Z 为8步，Qwen base 为40步。以上 summary 均 complete，实际 Private
调用存在，0 failed fallback、0 overflow retry。Z512/4096 两方向的中位数
都越过1.2，但正向样本的保守边界低于1.2：它是窄裕量初筛，不是稳健资格。
增大 Fa 并非单调加速；Z512/5120 的负结果保留，不作为推荐配置。
Z1024 未达标，不能以历史其它库的 GPU 分母或局部算子倍率补齐。
Qwen1024 同样未达标；每请求2560次实际 Private 调用、累计3840个
untimed/async block，无失败回退。不能把接近门槛表述成已通过1.2×。

Qwen512 使用真正 base-only v1 模板：
`outputs/runtime-ane/private-channel-qwen-c1056-k1024-n512-v1-base/manifest.json`。
它没有 v2 的 gate/up LoRA correction 输入及 hidden 输出；LoRA 仍使用 v2。
该路径是本轮新导出的模板，不是不存在的无 `-base` 后缀目录。
不能将跨构建的 v1/v2 时间差全部归因于此图版本，仍需同库消融。
反序新增三个热样本，中位及保守边界均超过1.26×，支持该固定 workload
的收益可复现；不把它外推到其它提示词、LoRA 或1024²。

## 数值证据的限度

同一 fox/seed42 的首个热输出，GPU vs Private uint8 RGB 比较：

| case | RMSE | PSNR (dB) | correlation | max abs |
| --- | ---: | ---: | ---: | ---: |
| Z512/4096 正向 | 5.279503 | 33.678942 | 0.996773 | 185 |
| Z1024/5120 | 8.271751 | 29.778854 | 0.992795 | 209 |
| Qwen512/5120 v1 | 4.706274 | 34.677259 | 0.998477 | 142 |

Qwen512 的两幅图已查看，未见该固定样本的明显结构破坏。这不等于多提示词、
latent、感知质量或 LoRA 资格；尤其 Z1024 的像素差异不得藏在相关系数后。
前轮优化前后像素逐位相同，只证明优化本身未改变对应 Private 样本，
不是 W8A8 相对 GPU 的精确性证明。没有声称观察到硬件 INT8 MAC。

## 512² LoRA：兼容通过，不推广负收益配置

两组均为同库一冷两热、v2、range=1、fixed-async=1，并在每条路线上
独立采样 process tree memory；CPU load 与 memory evidence verifier 均通过。
完整 gate/up 子范围修正及一次 full-hidden down-LoRA 实际执行；adapter
未合入 base 或 ANE 权重槽。Qwen 使用原 FP32 rank matmuls，未开启仅适用于
base 的 Q/K norm-RoPE，也没有使用 FP16 LoRA 近似改变分母。

| 现有 adapter / 步数 / Fa | 顺序 | GPU median (s) | Private median (s) | 倍率 |
| --- | --- | ---: | ---: | ---: |
| Z distill patch / 8 / 4096 | Private→GPU | 8.590888 | 7.224057 | 1.18921× |
| Qwen Viggle v0.2.1 r256 / 6 / 5120 | GPU→Private | 8.100264 | 8.208955 | 0.98676× |

Z 的冷请求有两次 headroom retry，最后 scale=16；两个热请求各256次
调用，无新 retry 或 failed fallback。Qwen 各请求192次调用，0 retry/fallback。
累计实际 narrow callbacks 分别768/576，full callbacks=0；不是仅验证配置。
Z 是正收益初筛，Qwen 是保留的负结果：当前该 adapter 仍推荐 GPU，不能把
base 的1.263×套用到 LoRA，也不能将可执行兼容等同于更快或画质通过。

独立 process-tree peak footprint（含冷请求）GPU/Private 分别为：
Z 25,455,781,656 / 24,204,732,304 bytes，Qwen 22,133,094,512 /
20,108,374,856 bytes。两组 swap in/out=0；Qwen 仍观察到 compression/
decompression 活动，因此不是零内存压力或完整 driver/wired/低内存资格。
单图 GPU/Private RMSE/PSNR/correlation/max abs：Z
10.736828 / 27.513284dB / 0.987757 / 229，Qwen
3.690081 / 36.790085dB / 0.999059 / 107。尚未完成正式 LoRA 质量验收。

原始目录：`outputs/private-ane-quiet-z512-lora-a4096-fixed/` 与
`outputs/private-ane-quiet-qwen512-lora-a5120-fixed/`。

## K tile 候选：保留控制组

新导出 Z1024/c2112/k2048/n512/v1 模板，仅使用现有 emitter 接口，
未修改 native 库、权重、recipe 或默认 tile。Fa4096、Private→GPU、一冷
两热：GPU=31.290369s，Private=28.280995s，1.10641×，load gate 通过。
目录 `outputs/private-ane-quiet-z1024-a4096-k2048-fixed/`。它仍未达到1.2×；
同库、同 Fa 的 k1024 控制组（GPU→Private、两热）也已完成：
GPU=31.300109s，Private=28.480655s，1.09900×。k2048 的 Private
中位耗时仅改善约0.7%，不是四格达标的解决方案，也不足以推广默认。
控制组目录 `outputs/private-ane-quiet-z1024-a4096-k1024-fixed/`，同样通过
连续 load gate。两组改变了 FP16 partial accumulation 的分块，不能根据
同 recipe 宣称新旧输出逐位相同；正式质量资格仍需独立检查。

## 原始证据与后续

实现已单独提交为 `393c258`（aligned W8 reads、LoRA range 与 fixed async）；
验证记录与实现分开归档。Public 默认、固定 async 默认0以及未资格候选边界不变。

本轮提交前重新执行：13 Private host/hardware/MLX、13 Public
Core ML/MLX/receipt、54 screen host、7 runtime host、6 memory-runner、
4 load-observer、8 repository layout、3 release guard，均通过；实际 Public
库的 flags/class strings/direct links 发行 guard 通过。`git diff --check`
和 staged diff 检查通过。未重跑完整 `make test`，既有 Qwen3 stale
source-string assertion 未在本专项修改，不能声称全量 suite 已通过。

这些验证使用上述带哈希的工作树构建快照，包含独立 ConvRot 用户改动；
不是对 clean staged tree 重新构建的声明。暂存时 Z cpp、results 与 session
仅选 ANE callback/counters hunks，ConvRot 片段和其独立文件继续留在工作树。

原始请求、PNG、stdout/stderr、连续负载及 summary 保留在：

- `outputs/private-ane-quiet-z512-a4096-fixed-reverse/`
- `outputs/private-ane-quiet-z512-a4096-fixed-forward5/`
- `outputs/private-ane-quiet-z512-a5120-fixed/`
- `outputs/private-ane-quiet-z1024-a5120-fixed/`
- `outputs/private-ane-quiet-qwen512-a5120-base-v1-fixed/`
- `outputs/private-ane-quiet-qwen512-a5120-base-v1-fixed-forward3/`
- `outputs/private-ane-quiet-qwen1024-a5120-base-v1-fixed/`

此前被 load gate 拒绝的报告仍 incomplete；没有放宽 gate 或追认其倍率。
逐 case summary byte hash、原始热样本与连续负载证据哈希见
[机器记录](../design/validation/private-ane-quiet-window-20261005.json)。
Public 默认和私有 build/runtime 双 gate 保留。现有 ConvRot 用户改动不混入
本专项提交，模型/adapter/reference/原设计稿不改。

待完成：Qwen1024 ≥1.2×、Z1024 share/bucket 优化与 Z512 稳健裕量，
1024² LoRA 实际模型兼容与性能、512² LoRA 多提示词/反序与 Qwen 加速，
以及多提示词/latent/媒体/内存验收、
bandwidth-aware calibration 和实际 device trace。本轮下载已核验并恢复，
后续性能测量须重新满足安静窗口，不能沿用本轮的独占或暂停假设。

本轮只读检查发现 developer directory 为 Command Line Tools，且
`xcrun xctrace list templates` 报工具不存在；`/Applications` 下也未找到
Instruments/xctrace。未安装或改变开发工具设置。该限制只解释本轮未采集
device trace，不能将缺失证据改写成并发验收通过。
