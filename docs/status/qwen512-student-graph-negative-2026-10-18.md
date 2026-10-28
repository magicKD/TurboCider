# Qwen512 跨请求保留编译图：净收益不足，停止运行时复用

2026-10-18，Asia/Singapore，M4 Max/64GiB、macOS26.6.2/build25G83。
只涉及 Qwen Image 2.1 文生图和图生图；
不修改 Z Image、已安装 App 或默认配置。完整优化目标仍 active。

## 结论

跨请求保留 Transformer block 编译函数确实能消除本轮观察到的 C++ wrapper
构建和 trace，但匹配整请求只减少约0.3–0.9%，且额外保留1.25–2.5 GiB
MLX active memory。未达到事前3%净收益门槛，已经从 Session 执行路径撤下。
不因 trace 数下降而宣称生成显著加速。

成功的不可变编辑 prefix snapshot、保守 residual cache/GPU+ANE 组合，以及
原 r128/r256 六步 GPU/joint 路径独立保留。本次没有改采样步数或模型权重。

## 同来源、同算术的19请求

使用原 BF16 encoder/DiT/VAE、已 pin 的 Viggle v0.2.1 r256、strength1、
inference-time/227 bindings、GPU/joint BF16 A/B operands/F32 ranks，输出512²，
实际六步。没有 ANE、DiT residual cache、末步 FFN 复用或权重合并。
图生图使用同一参考、ref512；两臂均保留不可变 prefix hit 作为强控制。

实验仅保留编译函数和只读权重常量；每次请求清除目标/文本/参考 K/V 等可变值，
同 shape 的新提示词必须重新计算 conditioning。seed73/74、同长度新提示词、
改变 token geometry 均有独立控制，不把 conditioning 热请求叫冷启动。

| 完整请求（含导出） | 控制 / s | 保留图 / s | 名义减少 |
| --- | ---: | ---: | ---: |
| 文生图重复条件 | 7.749488 | 7.691029 | 0.754% |
| 文生图新 seed | 7.769637 | 7.697684 | 0.926% |
| 文生图同长度新提示词 | 8.498051 | 9.229058 | −8.602% |
| 图生图重复条件、两臂 prefix hit | 7.919764 | 7.893114 | 0.337% |
| 图生图新 seed、两臂 prefix hit | 7.946593 | 7.872403 | 0.934% |

重复条件控制取前后控制均值，其他为对应单次控制；小样本不是置信区间。
新提示词的 slowdown 主要伴随 text encoding 从约0.7613升至1.4778秒，
未证明是保留图导致，不以该单例推导因果。

所有表内匹配 PNG、改变 geometry 的匹配 PNG 及返回关闭态控制均 RGBA
逐像素相同。命中时新 block wrappers=0、C++ traces=0；新 generation geometry
各64，新 edit prefix-hit 图各32。它们不是 GPU shader 编译数、提交数或物理
GPU/ANE overlap 证据。

## 内存与测量边界

generation 请求结束后的 MLX active 从14,923,173,106到17,608,085,234 bytes，
增加约2.5 GiB；编辑从15,496,121,586到16,839,380,466，增加约1.25 GiB。
关闭保留后回到原水平。该统计不与 process-tree footprint 相加。

100ms进程树 memory evidence complete，最大gap约110ms、允许500ms；峰值
physical footprint33,438,146,392 bytes，系统swap-out0、swap-in65,536 bytes。
持续 CPU-load 检查因竞争推理负载失败，`qualification_passed=false`。
未停止用户 ComfyUI 进程，不授予安静窗口、设备独占或正式性能资格。

## 已停用的代码与保留的证据

Session 恢复每请求 Transformer 所有权，不跨请求保留整图或可变状态。
`TURBOCIDER_QWEN21_RETAIN_STUDENT_GRAPH` 未设置/0保持普通路径，1在 planning
阶段明确拒绝并说明实验已退休，非法值亦拒绝；不是静默忽略旧命令。

只保留默认关闭的 wrapper/trace 诊断计数器、geometry 检查，以及真实 MLX GPU
生命周期回归：新条件同 shape 不借旧值、geometry 改变重新 trace、与新建
Transformer oracle 对齐、销毁后检查 active memory。它们不启用产品缓存。

研究库（不覆盖）：

```text
build/qwen512-student-graph-20261017/libturbocider.dylib
SHA256 16c5c6ed2fa617c809441564bf268757f1ffc2b1994a1210ea4b1f2ec7d61a7d
```

`tools/validation/qwen_student_graph_screen.py` 已标记为退休实验，并在启动前
要求上述研究库 SHA；不能误用新库启动昂贵且必然被拒绝的19请求。
解析/负对照 host 测试继续保留。

原始目录 `outputs/qwen512-student-graph-pilot-20261018/` 保留全部19 PNG、
请求、`source.json`、`summary.json`、`report.json`、stdout/stderr、load/memory。
`native-source-before-retirement.tar.gz` 保存撤下前相关 native/test/tool 源码；
不改写研究库或原有实验回执。

## 回退后的构建与真实出图

新建库，不覆盖研究库或先前稳定库：

```text
build/qwen512-graph-diagnostics-20261018/libturbocider.dylib
SHA256 b4ad5233ed30b6f065c2494049a84ec204dd7c7e1c335f6d5fe65a782173dd79
```

`TURBOCIDER_ENABLE_PRIVATE_ANE=1 TURBOCIDER_BUILD_LIB_ONLY=1` 构建成功；
并未因此启用实际请求的 ANE。独立核对该 manifest 的全部648项 native source
hash 与当前树一致。接口 contract131项：130通过、1跳过；真实 MLX GPU
lifecycle1项通过，包括默认计数器保持0、诊断显式启用、同 geometry 新值、
改变 geometry 的 oracle parity，以及第二次运行诊断后销毁内存回归。
相关 options/cache-hybrid/prefix/student/rank/archived-runner 的29项 selected
host tests 通过；不是全仓或完整模型/设备资格。

同 r256/joint/ref512、seed73/74 的11个真实请求覆盖文生图、编辑、snapshot
填充/命中/新seed/关闭恢复。新 PNG 与先前稳定库
`b876fc7190cd63259601639ebf1c81c544f6bdc5c17e7f50d1e5bf9db106acff`
的全部11张匹配 RGBA 逐像素一致，所有除输出路径外的请求字段一致。
这证明本批回退输出未改变，不把跨时段历史库当新性能分母。

新库 conditioning-hot 文生图中位7.775729秒；两项强 prefix-hit 编辑为
7.958958/8.070644秒。仅供回退 smoke 观察，没有新的提速声明。新进程树
memory evidence complete，peak31,659,842,312 bytes、系统swap-in/out0；
持续竞争负载仍失败，保持 qualification=false。原始证据在
`outputs/qwen512-graph-retired-smoke-20261018/`；独立像素/输入/source核对在
`outputs/qwen512-graph-retirement-audit-20261018.{py,json}`。

```sh
.venv/bin/python -B tools/validation/qwen_student_prefix_screen.py \
  --library build/qwen512-graph-diagnostics-20261018/libturbocider.dylib \
  --precisions joint --reference-size 512 --seed 73 \
  --output outputs/new-qwen-stable-smoke
```

没有改已安装 App/默认配置，没有下载新模型或清除用户缓存，磁盘约18GiB
可用；owned build/inference/sampler 均已退出。

## 后续

本次减少的 host graph 工作仅几十毫秒，GPU denoising 仍占主要请求时间。
下一步按真实 GEMM/attention 或 GPU/ANE 交接成本选择可证伪实验，优先寻找
省掉大算子工作/流量的机会；不重复已经失败的六步 FFN 复用或 phase 切分，
不把 base40 的收益移植为六步 student 的收益。

本轮完成负结果收尾与真实回退验证，不把它包装为新的性能突破。
